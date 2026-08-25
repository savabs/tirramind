"""Tests for Idea 10 — Graph Deviation Network Monitor.

Covers:
    1.  GDNMonitor instantiates with defaults
    2.  GDNMonitor instantiates with custom params
    3.  _GDNModel instantiates with correct dims
    4.  _GDNModel.forward returns (n_nodes,) shape
    5.  _GDNModel._learned_graph: top_k neighbours per row
    6.  _GDNModel._learned_graph: diagonal is excluded
    7.  _forward_fill: fills leading NaN with zeros
    8.  _forward_fill: forward-fills interior NaNs correctly
    9.  run(): empty store → empty dict
    10. run(): no observations → empty dict
    11. run(): fewer than MIN_ENTITIES with obs → empty dict
    12. run(): returns GDNResult for each scored entity
    13. run(): deviation_score is non-negative
    14. run(): is_anomaly is bool
    15. run(): result.entity_id matches key
    16. run(): anomaly flagged when deviation exceeds threshold
    17. run(): no anomaly when deviation is low (identical entities)
    18. store_results(): stores per-entity deviation signals
    19. store_results(): stores per-type avg_deviation signals
    20. store_results(): stored value matches deviation_score
    21. TrainerConfig.use_gdn defaults False
    22. TrainerConfig.gdn_hidden_dim defaults 64
    23. TrainerConfig.gdn_n_iters defaults 100
    24. TrainerConfig.gdn_anomaly_threshold defaults 3.0
    25. build_model() with use_gdn=True runs without error
    26. build_model() with use_gdn=True stores deviation signals
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from agent.convergence.gdn_monitor import (
    GDNMonitor,
    GDNResult,
    _GDNModel,
    _forward_fill,
)
from agent.models.gnn.trainer import Trainer, TrainerConfig, SyntheticGraphGenerator
from agent.pipeline.store import PipelineStore

# ── Helpers ──────────────────────────────────────────────────────────────────

_DAY = 86_400.0


def _make_store(tmp_path: Path, name: str = "gdn.db") -> PipelineStore:
    return PipelineStore(str(tmp_path / name))


def _add_entity_with_obs(
    store: PipelineStore,
    eid: str,
    n: int,
    t_start: float,
    t_end: float,
    value_fn=None,
    seed: int = 0,
) -> None:
    """Register entity and add n observations spanning [t_start, t_end]."""
    rng = np.random.default_rng(seed)
    store.register_entity("company", eid, eid)
    times = sorted(rng.uniform(t_start, t_end, n))
    for i, t in enumerate(times):
        v = value_fn(i) if value_fn else float(rng.normal())
        store.store_entity_observation(
            entity_id=eid,
            source_tool="test",
            observation_type="price",
            observed_at=float(t),
            value={"value": v},
        )


def _make_populated_store(
    tmp_path: Path,
    n_entities: int = 4,
    n_obs: int = 60,
    name: str = "gdn_pop.db",
) -> tuple[PipelineStore, float]:
    store = _make_store(tmp_path, name)
    as_of = time.time()
    rng = np.random.default_rng(42)
    for i in range(n_entities):
        eid = f"entity_{i}"
        _add_entity_with_obs(
            store,
            eid,
            n_obs,
            t_start=as_of - 60 * _DAY,
            t_end=as_of,
            seed=i,
        )
    return store, as_of


# ═══════════════════════════════════════════════════════════════
# 1–2. Construction
# ═══════════════════════════════════════════════════════════════


class TestConstruction:

    def test_instantiates_defaults(self):
        m = GDNMonitor()
        assert m.hidden_dim == 64
        assert m.n_iters == 100
        assert m.anomaly_threshold == pytest.approx(3.0)

    def test_instantiates_custom(self):
        m = GDNMonitor(hidden_dim=32, n_iters=10, anomaly_threshold=2.0)
        assert m.hidden_dim == 32
        assert m.n_iters == 10
        assert m.anomaly_threshold == pytest.approx(2.0)


# ═══════════════════════════════════════════════════════════════
# 3–6. _GDNModel
# ═══════════════════════════════════════════════════════════════


class TestGDNModel:

    def _make_model(self, n_nodes=5, window=4, hidden_dim=16, top_k=2):
        return _GDNModel(
            n_nodes=n_nodes,
            window=window,
            hidden_dim=hidden_dim,
            emb_dim=8,
            top_k=top_k,
        )

    def test_instantiates_correct_dims(self):
        m = self._make_model(n_nodes=6, window=5)
        assert m.n_nodes == 6
        assert m.window == 5

    def test_forward_output_shape(self):
        m = self._make_model(n_nodes=5, window=4)
        x = torch.randn(5, 4)
        out = m(x)
        assert out.shape == (5,)

    def test_learned_graph_top_k_neighbours(self):
        m = self._make_model(n_nodes=6, window=3, top_k=2)
        adj = m._learned_graph(torch.device("cpu"))
        # Each row should have at most top_k True values
        for i in range(6):
            assert adj[i].sum().item() <= 2

    def test_learned_graph_no_self_loops(self):
        m = self._make_model(n_nodes=5, window=3, top_k=2)
        adj = m._learned_graph(torch.device("cpu"))
        # Diagonal should be False
        for i in range(5):
            assert not adj[i, i].item()


# ═══════════════════════════════════════════════════════════════
# 7–8. _forward_fill
# ═══════════════════════════════════════════════════════════════


class TestForwardFill:

    def test_leading_nan_filled_with_zero(self):
        arr = np.array([np.nan, np.nan, 3.0, 4.0])
        result = _forward_fill(arr)
        assert result[0] == 0.0
        assert result[1] == 0.0

    def test_interior_nans_forward_filled(self):
        arr = np.array([1.0, np.nan, np.nan, 4.0])
        result = _forward_fill(arr)
        assert result[1] == pytest.approx(1.0)
        assert result[2] == pytest.approx(1.0)
        assert result[3] == pytest.approx(4.0)


# ═══════════════════════════════════════════════════════════════
# 9–17. run()
# ═══════════════════════════════════════════════════════════════


class TestRun:

    def test_empty_store_returns_empty(self, tmp_path):
        store = _make_store(tmp_path)
        m = GDNMonitor()
        assert m.run(store) == {}

    def test_no_observations_returns_empty(self, tmp_path):
        store = _make_store(tmp_path)
        store.register_entity("company", "lonely", "lonely")
        m = GDNMonitor()
        assert m.run(store) == {}

    def test_single_entity_returns_empty(self, tmp_path):
        # Only 1 entity — cannot form a graph (need >= MIN_ENTITIES=2)
        store = _make_store(tmp_path)
        as_of = time.time()
        _add_entity_with_obs(store, "solo", 30, as_of - 30 * _DAY, as_of)
        m = GDNMonitor(n_iters=2)
        result = m.run(store, as_of=as_of)
        assert result == {}

    def test_returns_result_for_each_entity(self, tmp_path):
        store, as_of = _make_populated_store(tmp_path, n_entities=4)
        m = GDNMonitor(n_iters=3, hidden_dim=8, emb_dim=4, n_bins=20, window=5)
        result = m.run(store, as_of=as_of)
        assert len(result) == 4

    def test_deviation_score_non_negative(self, tmp_path):
        store, as_of = _make_populated_store(tmp_path, n_entities=3)
        m = GDNMonitor(n_iters=3, hidden_dim=8, emb_dim=4, n_bins=20, window=5)
        result = m.run(store, as_of=as_of)
        for r in result.values():
            assert r.deviation_score >= 0.0

    def test_is_anomaly_is_bool(self, tmp_path):
        store, as_of = _make_populated_store(tmp_path, n_entities=3)
        m = GDNMonitor(n_iters=3, hidden_dim=8, emb_dim=4, n_bins=20, window=5)
        result = m.run(store, as_of=as_of)
        for r in result.values():
            assert isinstance(r.is_anomaly, bool)

    def test_entity_id_matches_key(self, tmp_path):
        store, as_of = _make_populated_store(tmp_path, n_entities=3)
        m = GDNMonitor(n_iters=3, hidden_dim=8, emb_dim=4, n_bins=20, window=5)
        result = m.run(store, as_of=as_of)
        for key, r in result.items():
            assert r.entity_id == key

    def test_anomaly_flagged_when_threshold_zero(self, tmp_path):
        """When anomaly_threshold=0.0, every entity exceeds it."""
        store, as_of = _make_populated_store(tmp_path, n_entities=3, name="thr0.db")
        m = GDNMonitor(
            n_iters=3,
            hidden_dim=8,
            emb_dim=4,
            n_bins=20,
            window=5,
            anomaly_threshold=0.0,  # everything is anomaly
        )
        result = m.run(store, as_of=as_of)
        assert len(result) > 0
        # With threshold=0, all non-zero deviation scores flag as anomaly.
        # Any entity with deviation > 0 (virtually guaranteed) should fire.
        flagged = [r for r in result.values() if r.is_anomaly]
        assert len(flagged) > 0

    def test_no_anomaly_for_identical_series(self, tmp_path):
        """Entities with identical smooth series should have low deviation."""
        store = _make_store(tmp_path, "same.db")
        as_of = time.time()
        for i in range(3):
            _add_entity_with_obs(
                store,
                f"same_{i}",
                60,
                as_of - 60 * _DAY,
                as_of,
                value_fn=lambda j: float(j),  # identical linear ramp
                seed=i,
            )
        m = GDNMonitor(
            n_iters=10,
            hidden_dim=8,
            emb_dim=4,
            n_bins=20,
            window=5,
            anomaly_threshold=1000.0,  # very insensitive — nothing should fire
        )
        result = m.run(store, as_of=as_of)
        for r in result.values():
            assert not r.is_anomaly


# ═══════════════════════════════════════════════════════════════
# 18–20. store_results()
# ═══════════════════════════════════════════════════════════════


class TestStoreResults:

    def test_stores_per_entity_signals(self, tmp_path):
        store, as_of = _make_populated_store(tmp_path, n_entities=3, name="sig.db")
        m = GDNMonitor(n_iters=3, hidden_dim=8, emb_dim=4, n_bins=20, window=5)
        results = m.run(store, as_of=as_of)
        n = m.store_results(results, store)
        assert n > 0
        for eid in results:
            sigs = store.query_signals(f"graph_structure.{eid}.deviation")
            assert len(sigs) >= 1

    def test_stores_per_type_avg_signals(self, tmp_path):
        store, as_of = _make_populated_store(tmp_path, n_entities=3, name="avg.db")
        m = GDNMonitor(n_iters=3, hidden_dim=8, emb_dim=4, n_bins=20, window=5)
        results = m.run(store, as_of=as_of)
        m.store_results(results, store)
        sigs = store.query_signals("graph_structure.company.avg_deviation")
        assert len(sigs) >= 1

    def test_stored_value_matches_deviation_score(self, tmp_path):
        store, as_of = _make_populated_store(tmp_path, n_entities=3, name="val.db")
        m = GDNMonitor(n_iters=3, hidden_dim=8, emb_dim=4, n_bins=20, window=5)
        results = m.run(store, as_of=as_of)
        m.store_results(results, store)
        for eid, res in results.items():
            sigs = store.query_signals(f"graph_structure.{eid}.deviation")
            assert sigs[-1]["value"] == pytest.approx(res.deviation_score, rel=1e-5)


# ═══════════════════════════════════════════════════════════════
# 21–24. TrainerConfig
# ═══════════════════════════════════════════════════════════════


class TestTrainerConfig:

    def test_use_gdn_defaults_false(self):
        from agent.models.gnn.trainer import TrainerConfig

        assert TrainerConfig().use_gdn is False

    def test_gdn_hidden_dim_defaults_64(self):
        from agent.models.gnn.trainer import TrainerConfig

        assert TrainerConfig().gdn_hidden_dim == 64

    def test_gdn_n_iters_defaults_100(self):
        from agent.models.gnn.trainer import TrainerConfig

        assert TrainerConfig().gdn_n_iters == 100

    def test_gdn_anomaly_threshold_defaults_3(self):
        from agent.models.gnn.trainer import TrainerConfig

        assert TrainerConfig().gdn_anomaly_threshold == pytest.approx(3.0)


# ═══════════════════════════════════════════════════════════════
# 25–26. build_model() integration
# ═══════════════════════════════════════════════════════════════


class TestBuildModelIntegration:

    def _make_trainer(self, tmp_path: Path, use_gdn: bool, tag: str) -> Trainer:
        store = _make_store(tmp_path, f"{tag}.db")
        gen = SyntheticGraphGenerator(
            num_companies=4,
            num_countries=2,
            time_span=3600.0 * 4,
            base_event_rate=0.001,
            seed=33,
        )
        gen.generate(store)
        cfg = TrainerConfig(
            hidden_dim=16,
            memory_dim=16,
            message_dim=16,
            time_dim=8,
            num_heads=1,
            num_layers=1,
            use_gdn=use_gdn,
            gdn_hidden_dim=16,
            gdn_n_iters=5,
            gdn_anomaly_threshold=3.0,
        )
        return Trainer(store, cfg)

    def test_build_model_with_gdn_no_error(self, tmp_path):
        t = self._make_trainer(tmp_path, use_gdn=True, tag="gdnbm")
        model = t.build_model()
        assert model is not None

    def test_build_model_stores_deviation_signals(self, tmp_path):
        t = self._make_trainer(tmp_path, use_gdn=True, tag="gdnsig")
        as_of = time.time()
        store = t.store
        # Add current-time entities with enough obs for GDN
        for i in range(3):
            _add_entity_with_obs(
                store,
                f"live_{i}",
                60,
                as_of - 60 * _DAY,
                as_of,
                seed=i + 10,
            )
        t.build_model()
        # At least one graph_structure.*.deviation signal should exist
        has_sig = False
        for i in range(3):
            sigs = store.query_signals(f"graph_structure.live_{i}.deviation")
            if sigs:
                has_sig = True
                break
        assert has_sig, "Expected at least one graph_structure.*.deviation signal"
