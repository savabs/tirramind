"""Tests for Idea 6 — Neural Hawkes Process Encoder.

Covers:
    1.  NeuralHawkesEncoder instantiates with defaults
    2.  _NHPModel instantiates with correct dimensions
    3.  _NHPModel.forward returns (T, n_types) shape
    4.  _NHPModel.predict_at returns (n_types,) shape
    5.  _NHPModel: hidden state decays with positive dt
    6.  _build_vocab: maps observation_type strings to 1-based indices
    7.  _build_vocab: pad index 0 is not used by any type
    8.  _build_sequences: returns empty when no obs
    9.  _build_sequences: skips sessions with fewer than MIN_EVENTS events
    10. _build_sequences: delta_ts[0] == 0 (first event has no predecessor)
    11. _build_sequences: delta_ts are non-negative
    12. run(): returns empty dict when store has no observations
    13. run(): returns empty when only one event type (< MIN_VOCAB_SIZE)
    14. run(): returns HawkesResult for each event type when data sufficient
    15. run(): prob_72h in [0, 1] for all results
    16. run(): intensity is non-negative for all results
    17. run(): result.event_type matches the key
    18. run(): result.forecast_hours matches encoder setting
    19. store_results(): persists signals with correct naming convention
    20. store_results(): stored value matches prob_72h
    21. TrainerConfig.use_hawkes defaults False
    22. TrainerConfig.hawkes_hidden_dim defaults 64
    23. TrainerConfig.hawkes_n_iters defaults 200
    24. TrainerConfig.hawkes_forecast_hours defaults 72.0
    25. build_model() with use_hawkes=True runs without error
    26. build_model() with use_hawkes=True stores intensity signals
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import torch

from agent.convergence.neural_hawkes import (
    HawkesResult,
    NeuralHawkesEncoder,
    _NHPModel,
    _MIN_VOCAB_SIZE,
)
from agent.models.gnn.trainer import Trainer, TrainerConfig, SyntheticGraphGenerator
from agent.pipeline.store import PipelineStore

# ── Helpers ──────────────────────────────────────────────────────────────────

_DAY = 86_400.0


def _make_store(tmp_path: Path, name: str = "hawkes.db") -> PipelineStore:
    return PipelineStore(str(tmp_path / name))


def _add_obs(
    store: PipelineStore,
    obs_types: list[str],
    n_per_type: int,
    t_start: float,
    t_end: float,
    seed: int = 0,
) -> None:
    """Add observations spread across [t_start, t_end] for each obs_type."""
    rng = np.random.default_rng(seed)
    for obs_type in obs_types:
        entity_id = f"e_{obs_type}"
        store.register_entity(
            entity_type="company", canonical_name=entity_id, entity_id=entity_id
        )
        times = sorted(rng.uniform(t_start, t_end, n_per_type))
        for t in times:
            store.store_entity_observation(
                entity_id=entity_id,
                source_tool="test",
                observation_type=obs_type,
                observed_at=float(t),
                value={"value": float(rng.normal())},
            )


# ═══════════════════════════════════════════════════════════════
# 1. Construction
# ═══════════════════════════════════════════════════════════════


class TestConstruction:

    def test_instantiates_defaults(self):
        enc = NeuralHawkesEncoder()
        assert enc.hidden_dim == 64
        assert enc.n_iters == 200
        assert enc.forecast_hours == 72.0

    def test_instantiates_custom(self):
        enc = NeuralHawkesEncoder(hidden_dim=32, n_iters=10, forecast_hours=24.0)
        assert enc.hidden_dim == 32
        assert enc.n_iters == 10
        assert enc.forecast_hours == 24.0


# ═══════════════════════════════════════════════════════════════
# 2–5. _NHPModel
# ═══════════════════════════════════════════════════════════════


class TestNHPModel:

    def _make_model(self, n_types=4, hidden_dim=16):
        return _NHPModel(n_types=n_types, hidden_dim=hidden_dim, emb_dim=8)

    def test_instantiates_correct_dims(self):
        m = self._make_model(n_types=4, hidden_dim=16)
        assert m.n_types == 4
        assert m.hidden_dim == 16

    def test_forward_output_shape(self):
        m = self._make_model(n_types=4)
        T = 6
        types = torch.randint(1, 5, (T,))
        delta_ts = torch.abs(torch.randn(T)) * 100
        logits = m(types, delta_ts)
        assert logits.shape == (T, 4)

    def test_predict_at_output_shape(self):
        m = self._make_model(n_types=3)
        types = torch.randint(1, 4, (5,))
        delta_ts = torch.abs(torch.randn(5)) * 100
        intensity = m.predict_at(types, delta_ts, forecast_dt=3600.0)
        assert intensity.shape == (3,)

    def test_hidden_state_decays_with_positive_dt(self):
        m = self._make_model(n_types=2)
        h = torch.ones(1, m.hidden_dim)
        h_decayed = m._decay_hidden(h, dt=1000.0)
        # After large dt, hidden state should be smaller
        assert float(h_decayed.abs().mean()) < float(h.abs().mean())

    def test_intensity_non_negative(self):
        m = self._make_model(n_types=3)
        types = torch.randint(1, 4, (4,))
        delta_ts = torch.abs(torch.randn(4)) * 100
        intensity = m.predict_at(types, delta_ts, forecast_dt=3600.0)
        assert (intensity >= 0).all()


# ═══════════════════════════════════════════════════════════════
# 6–11. Vocabulary + Sequence building
# ═══════════════════════════════════════════════════════════════


class TestVocabAndSequences:

    def _make_obs(self, types_counts: dict[str, int], t0: float = 0.0) -> list[dict]:
        obs = []
        t = t0
        for obs_type, count in types_counts.items():
            for _ in range(count):
                obs.append(
                    {"observation_type": obs_type, "observed_at": t, "value": {}}
                )
                t += 3600.0
        return obs

    def test_vocab_maps_to_1based(self):
        enc = NeuralHawkesEncoder()
        obs = self._make_obs({"price": 3, "ais_position": 3})
        enc._build_vocab(obs)
        assert all(v >= 1 for v in enc._vocab.values())
        assert 0 not in enc._vocab.values()

    def test_vocab_pad_idx_not_used(self):
        enc = NeuralHawkesEncoder()
        obs = self._make_obs({"price": 5, "volume": 5, "rate": 5})
        enc._build_vocab(obs)
        assert 0 not in enc._vocab.values()

    def test_build_sequences_empty_obs(self):
        enc = NeuralHawkesEncoder()
        enc._build_vocab([])
        seqs = enc._build_sequences([])
        assert seqs == []

    def test_build_sequences_skips_short_sessions(self):
        enc = NeuralHawkesEncoder(session_days=7)
        # Only 2 observations per 7-day window — below _MIN_EVENTS=4
        obs = self._make_obs({"price": 2}, t0=0.0)
        enc._build_vocab(obs)
        seqs = enc._build_sequences(obs)
        assert seqs == []

    def test_delta_ts_first_is_zero(self):
        enc = NeuralHawkesEncoder(session_days=30)
        obs = self._make_obs({"price": 10, "ais": 10}, t0=1000.0)
        enc._build_vocab(obs)
        seqs = enc._build_sequences(obs)
        if seqs:
            types_list, delta_ts_list = seqs[0]
            assert delta_ts_list[0] == pytest.approx(0.0)

    def test_delta_ts_non_negative(self):
        enc = NeuralHawkesEncoder(session_days=30)
        obs = self._make_obs({"price": 20}, t0=0.0)
        enc._build_vocab(obs)
        seqs = enc._build_sequences(obs)
        for _, delta_ts in seqs:
            assert all(dt >= 0.0 for dt in delta_ts)


# ═══════════════════════════════════════════════════════════════
# 12–18. run()
# ═══════════════════════════════════════════════════════════════


class TestRun:

    def test_empty_store_returns_empty_dict(self, tmp_path):
        store = _make_store(tmp_path)
        enc = NeuralHawkesEncoder(n_iters=5)
        result = enc.run(store)
        assert result == {}

    def test_single_event_type_returns_empty(self, tmp_path):
        store = _make_store(tmp_path)
        as_of = time.time()
        _add_obs(store, ["price"], 30, as_of - 30 * _DAY, as_of)
        enc = NeuralHawkesEncoder(n_iters=2)
        result = enc.run(store, as_of=as_of)
        assert result == {}

    def test_returns_result_for_each_event_type(self, tmp_path):
        store = _make_store(tmp_path)
        as_of = time.time()
        _add_obs(
            store, ["price", "ais_position", "sanctions"], 40, as_of - 60 * _DAY, as_of
        )
        enc = NeuralHawkesEncoder(n_iters=5, hidden_dim=16, emb_dim=8)
        result = enc.run(store, as_of=as_of)
        assert len(result) == 3
        assert set(result.keys()) == {"price", "ais_position", "sanctions"}

    def test_prob_72h_in_unit_interval(self, tmp_path):
        store = _make_store(tmp_path)
        as_of = time.time()
        _add_obs(store, ["price", "volume"], 50, as_of - 60 * _DAY, as_of)
        enc = NeuralHawkesEncoder(n_iters=5, hidden_dim=16, emb_dim=8)
        result = enc.run(store, as_of=as_of)
        for r in result.values():
            assert 0.0 <= r.prob_72h <= 1.0

    def test_intensity_non_negative(self, tmp_path):
        store = _make_store(tmp_path)
        as_of = time.time()
        _add_obs(store, ["price", "volume"], 50, as_of - 60 * _DAY, as_of)
        enc = NeuralHawkesEncoder(n_iters=5, hidden_dim=16, emb_dim=8)
        result = enc.run(store, as_of=as_of)
        for r in result.values():
            assert r.intensity >= 0.0

    def test_result_event_type_matches_key(self, tmp_path):
        store = _make_store(tmp_path)
        as_of = time.time()
        _add_obs(store, ["price", "ais"], 40, as_of - 60 * _DAY, as_of)
        enc = NeuralHawkesEncoder(n_iters=5, hidden_dim=16, emb_dim=8)
        result = enc.run(store, as_of=as_of)
        for key, r in result.items():
            assert r.event_type == key

    def test_forecast_hours_matches_encoder(self, tmp_path):
        store = _make_store(tmp_path)
        as_of = time.time()
        _add_obs(store, ["price", "ais"], 40, as_of - 60 * _DAY, as_of)
        enc = NeuralHawkesEncoder(
            n_iters=5, hidden_dim=16, emb_dim=8, forecast_hours=48.0
        )
        result = enc.run(store, as_of=as_of)
        for r in result.values():
            assert r.forecast_hours == pytest.approx(48.0)


# ═══════════════════════════════════════════════════════════════
# 19–20. store_results()
# ═══════════════════════════════════════════════════════════════


class TestStoreResults:

    def test_persists_signals_correct_naming(self, tmp_path):
        store = _make_store(tmp_path)
        as_of = time.time()
        _add_obs(store, ["price", "rate"], 40, as_of - 60 * _DAY, as_of)
        enc = NeuralHawkesEncoder(
            n_iters=5, hidden_dim=16, emb_dim=8, forecast_hours=72.0
        )
        results = enc.run(store, as_of=as_of)
        n = enc.store_results(results, store)
        assert n == len(results)
        for event_type in results:
            sigs = store.query_signals(f"hawkes.{event_type}.intensity_72h")
            assert len(sigs) >= 1

    def test_stored_value_matches_prob_72h(self, tmp_path):
        store = _make_store(tmp_path)
        as_of = time.time()
        _add_obs(store, ["price", "rate"], 40, as_of - 60 * _DAY, as_of)
        enc = NeuralHawkesEncoder(
            n_iters=5, hidden_dim=16, emb_dim=8, forecast_hours=72.0
        )
        results = enc.run(store, as_of=as_of)
        enc.store_results(results, store)
        for event_type, r in results.items():
            sigs = store.query_signals(f"hawkes.{event_type}.intensity_72h")
            assert len(sigs) >= 1
            assert sigs[-1]["value"] == pytest.approx(r.prob_72h, rel=1e-5)


# ═══════════════════════════════════════════════════════════════
# 21–24. TrainerConfig
# ═══════════════════════════════════════════════════════════════


class TestTrainerConfig:

    def test_use_hawkes_defaults_false(self):
        from agent.models.gnn.trainer import TrainerConfig

        assert TrainerConfig().use_hawkes is False

    def test_hawkes_hidden_dim_defaults_64(self):
        from agent.models.gnn.trainer import TrainerConfig

        assert TrainerConfig().hawkes_hidden_dim == 64

    def test_hawkes_n_iters_defaults_200(self):
        from agent.models.gnn.trainer import TrainerConfig

        assert TrainerConfig().hawkes_n_iters == 200

    def test_hawkes_forecast_hours_defaults_72(self):
        from agent.models.gnn.trainer import TrainerConfig

        assert TrainerConfig().hawkes_forecast_hours == pytest.approx(72.0)


# ═══════════════════════════════════════════════════════════════
# 25–26. build_model() integration
# ═══════════════════════════════════════════════════════════════


class TestBuildModelIntegration:

    def _make_trainer(self, tmp_path: Path, use_hawkes: bool, tag: str) -> Trainer:
        store = _make_store(tmp_path, f"{tag}.db")
        gen = SyntheticGraphGenerator(
            num_companies=4,
            num_countries=2,
            num_vessels=2,
            time_span=3600.0 * 4,
            base_event_rate=0.001,
            seed=77,
        )
        gen.generate(store)
        cfg = TrainerConfig(
            hidden_dim=16,
            memory_dim=16,
            message_dim=16,
            time_dim=8,
            num_heads=1,
            num_layers=1,
            use_hawkes=use_hawkes,
            hawkes_hidden_dim=16,
            hawkes_n_iters=10,
            hawkes_forecast_hours=24.0,
        )
        return Trainer(store, cfg)

    def test_build_model_with_hawkes_no_error(self, tmp_path):
        t = self._make_trainer(tmp_path, use_hawkes=True, tag="hbm")
        model = t.build_model()
        assert model is not None

    def test_build_model_stores_hawkes_signals(self, tmp_path):
        t = self._make_trainer(tmp_path, use_hawkes=True, tag="hsig")
        # Add current-time observations (synthetic uses epoch-relative timestamps)
        as_of = time.time()
        _add_obs(
            t.store, ["price", "ais_position", "rate"], 50, as_of - 60 * _DAY, as_of
        )
        t.build_model()
        # At least one hawkes.*.intensity_24h signal should be stored
        has_signal = False
        for event_type in ["price", "ais_position", "rate"]:
            sigs = t.store.query_signals(f"hawkes.{event_type}.intensity_24h")
            if sigs:
                has_signal = True
                break
        assert has_signal, "Expected at least one hawkes intensity signal stored"
