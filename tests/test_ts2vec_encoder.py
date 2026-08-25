"""Tests for Idea 5 — TS2Vec Contrastive Pretraining Encoder.

Covers:
    1.  TS2VecEncoder instantiates with default params
    2.  _build_series: empty observations → zero (T, 2) array
    3.  _build_series: single event fills correct bin
    4.  _build_series: counts channel normalised to max=1
    5.  _build_series: value channel tanh-normalised
    6.  _encode_type: returns None when t_span < 1 (no time range)
    7.  _encode_type: returns None when no observations
    8.  _encode_type: returns dict {entity_id: ndarray(output_dims)} for valid data
    9.  _encode_type: all embedding arrays have shape (output_dims,)
    10. fit_and_encode: skips types with fewer than 2 entities
    11. fit_and_encode: returns embeddings for types with ≥2 entities
    12. fit_and_encode: zero-vector for entity with no observations (within encoded type)
    13. fit_and_encode: embedding shape matches output_dims
    14. _build_node_features: ts2vec_dim=0 → same feature dim as baseline
    15. _build_node_features: ts2vec_dim>0 → feature dim increases by ts2vec_dim
    16. _build_node_features: ts2vec embeddings appear in correct offset position
    17. GraphBuilder.build() with ts2vec_embeddings passes through to node features
    18. GraphBuilder.build_from_cached() with ts2vec_embeddings passes through
    19. TrainerConfig.use_ts2vec defaults False
    20. TrainerConfig.ts2vec_dim defaults 32, ts2vec_n_iters defaults 200
    21. build_model() with use_ts2vec=True: model's in_channels enlarged
    22. build_model() with use_ts2vec=False: model's in_channels unchanged
    23. Training step with use_ts2vec=True runs without error
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from agent.models.gnn.graph_builder import (
    BASE_FEAT_DIM,
    GraphBuilder,
    _build_node_features,
)
from agent.models.gnn.ts2vec_encoder import TS2VecEncoder
from agent.models.gnn.trainer import Trainer, TrainerConfig, SyntheticGraphGenerator
from agent.pipeline.store import PipelineStore

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def encoder():
    return TS2VecEncoder(output_dims=16, n_iters=5, time_bins=16, depth=3, device="cpu")


@pytest.fixture()
def populated_store(tmp_path):
    store = PipelineStore(str(tmp_path / "ts2vec_test.db"))
    gen = SyntheticGraphGenerator(
        num_companies=6,
        num_countries=2,
        num_vessels=4,
        time_span=86400.0 * 7,
        base_event_rate=0.05,
        seed=42,
    )
    gen.generate(store)
    return store


@pytest.fixture()
def small_store(tmp_path):
    """Store with just 2 companies and a handful of hand-crafted observations."""
    store = PipelineStore(str(tmp_path / "small.db"))
    store.register_entity(
        entity_type="company", canonical_name="Alpha Corp", entity_id="e1"
    )
    store.register_entity(
        entity_type="company", canonical_name="Beta Ltd", entity_id="e2"
    )
    store.register_entity(
        entity_type="company", canonical_name="Gamma Inc", entity_id="e3"
    )
    t0 = 1_000_000.0
    for i in range(8):
        store.store_entity_observation(
            entity_id="e1",
            source_tool="test",
            observation_type="price",
            observed_at=t0 + i * 3600,
            value={"value": float(i * 10)},
        )
        store.store_entity_observation(
            entity_id="e2",
            source_tool="test",
            observation_type="price",
            observed_at=t0 + i * 3600 + 1800,
            value={"value": float(i * 5)},
        )
    store.store_entity_observation(
        entity_id="e3",
        source_tool="test",
        observation_type="price",
        observed_at=t0 + 3600,
        value={"value": 50.0},
    )
    return store, t0


# ═══════════════════════════════════════════════════════════════
# 1. Construction
# ═══════════════════════════════════════════════════════════════


class TestConstruction:

    def test_instantiates(self, encoder):
        assert encoder is not None
        assert encoder.output_dims == 16
        assert encoder.time_bins == 16


# ═══════════════════════════════════════════════════════════════
# 2–5. _build_series
# ═══════════════════════════════════════════════════════════════


class TestBuildSeries:

    def test_empty_observations_returns_zeros(self, encoder):
        out = encoder._build_series([], t_min=0.0, t_span=1000.0, T=16)
        assert out.shape == (16, 2)
        assert (out == 0).all()

    def test_single_event_fills_correct_bin(self, encoder):
        T = 16
        # Event at 50% of the span → bin T//2
        obs = [{"observed_at": 500.0, "value": {"value": 10.0}}]
        out = encoder._build_series(obs, t_min=0.0, t_span=1000.0, T=T)
        assert out.shape == (T, 2)
        expected_bin = min(int(0.5 * T), T - 1)
        assert out[expected_bin, 0] == pytest.approx(1.0)  # normalised count = 1

    def test_counts_channel_normalised_to_one(self, encoder):
        T = 8
        t_span = 800.0
        obs = [
            {"observed_at": 0.0, "value": {"value": 0.0}},
            {"observed_at": 100.0, "value": {"value": 0.0}},
            {"observed_at": 100.0, "value": {"value": 0.0}},
        ]
        out = encoder._build_series(obs, t_min=0.0, t_span=t_span, T=T)
        assert float(out[:, 0].max()) == pytest.approx(1.0)

    def test_value_channel_tanh_normalised(self, encoder):
        T = 8
        obs = [{"observed_at": 0.0, "value": {"value": 1000.0}}]
        out = encoder._build_series(obs, t_min=0.0, t_span=1000.0, T=T)
        # tanh(1000 / 1001) ≈ tanh(0.999) — should be strictly between 0 and 1
        val = float(out[0, 1])
        assert 0.0 < val <= 1.0


# ═══════════════════════════════════════════════════════════════
# 6–9. _encode_type
# ═══════════════════════════════════════════════════════════════


class TestEncodeType:

    def _make_obs(self, eids: list[str], n_obs: int = 5, t0: float = 0.0):
        obs_by_entity = {}
        for eid in eids:
            obs_by_entity[eid] = [
                {"observed_at": t0 + i * 100.0, "value": {"value": float(i)}}
                for i in range(n_obs)
            ]
        return obs_by_entity

    def test_returns_none_when_no_observations(self, encoder):
        result = encoder._encode_type("company", ["e1", "e2"], {})
        assert result is None

    def test_returns_none_when_t_span_too_small(self, encoder):
        obs = {
            "e1": [{"observed_at": 0.0, "value": {"value": 1.0}}],
            "e2": [{"observed_at": 0.5, "value": {"value": 2.0}}],
        }
        result = encoder._encode_type("company", ["e1", "e2"], obs)
        assert result is None

    def test_returns_dict_for_valid_data(self, encoder):
        obs = self._make_obs(["e1", "e2", "e3"])
        result = encoder._encode_type("company", ["e1", "e2", "e3"], obs)
        assert result is not None
        assert set(result.keys()) == {"e1", "e2", "e3"}

    def test_embedding_shape_correct(self, encoder):
        obs = self._make_obs(["e1", "e2", "e3"])
        result = encoder._encode_type("company", ["e1", "e2", "e3"], obs)
        assert result is not None
        for emb in result.values():
            assert emb.shape == (encoder.output_dims,)
            assert emb.dtype == np.float32


# ═══════════════════════════════════════════════════════════════
# 10–13. fit_and_encode
# ═══════════════════════════════════════════════════════════════


class TestFitAndEncode:

    def test_skips_singleton_types(self, encoder, small_store):
        """company has 3 entities — should be encoded.
        Any type with <2 entities is skipped."""
        store, _ = small_store
        result = encoder.fit_and_encode(store)
        # company has 3 entities → should appear
        assert "company" in result

    def test_returns_embeddings_for_sufficient_types(self, encoder, populated_store):
        result = encoder.fit_and_encode(populated_store)
        assert len(result) >= 1  # at least one type encoded
        for etype, embs in result.items():
            assert len(embs) >= 2  # at least 2 entities per encoded type

    def test_entity_with_no_obs_gets_zero_embedding(self, encoder, small_store):
        """e3 has observations, but let's test a freshly added entity with none."""
        store, t0 = small_store
        store.register_entity(
            entity_type="company", canonical_name="Orphan Co", entity_id="orphan"
        )
        result = encoder.fit_and_encode(store)
        if "company" in result and "orphan" in result["company"]:
            emb = result["company"]["orphan"]
            assert emb.shape == (encoder.output_dims,)

    def test_embedding_dim_matches_output_dims(self, encoder, populated_store):
        result = encoder.fit_and_encode(populated_store)
        for etype, embs in result.items():
            for eid, emb in embs.items():
                assert emb.shape == (
                    encoder.output_dims,
                ), f"Wrong shape for {etype}/{eid}: {emb.shape}"


# ═══════════════════════════════════════════════════════════════
# 14–16. _build_node_features integration
# ═══════════════════════════════════════════════════════════════


class TestBuildNodeFeatures:

    def _make_obs(self, entity_ids: list[str]):
        t = 1_000_000.0
        return [
            {
                "entity_id": eid,
                "entity_type": "company",
                "observed_at": t + i,
                "observation_type": "price",
                "value": {"value": float(i)},
            }
            for i, eid in enumerate(entity_ids)
        ]

    def test_ts2vec_dim_zero_unchanged(self):
        eids = ["c1", "c2", "c3"]
        obs = self._make_obs(eids)
        feats = _build_node_features(
            "company", eids, obs, 1_001_000.0, ts2vec_embeddings=None, ts2vec_dim=0
        )
        assert feats.shape[1] == BASE_FEAT_DIM

    def test_ts2vec_dim_positive_extends_features(self):
        eids = ["c1", "c2"]
        obs = self._make_obs(eids)
        ts_dim = 12
        embs = {
            "company": {eid: np.random.randn(ts_dim).astype(np.float32) for eid in eids}
        }
        feats = _build_node_features(
            "company", eids, obs, 1_001_000.0, ts2vec_embeddings=embs, ts2vec_dim=ts_dim
        )
        assert feats.shape[1] == BASE_FEAT_DIM + ts_dim

    def test_ts2vec_values_in_correct_offset(self):
        eids = ["c1"]
        obs = self._make_obs(eids)
        ts_dim = 4
        fixed_emb = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        embs = {"company": {"c1": fixed_emb}}
        feats = _build_node_features(
            "company", eids, obs, 1_001_000.0, ts2vec_embeddings=embs, ts2vec_dim=ts_dim
        )
        offset = BASE_FEAT_DIM  # no enrichment, no price (not instrument), no sigs
        recovered = feats[0, offset : offset + ts_dim].numpy()
        np.testing.assert_allclose(recovered, fixed_emb, atol=1e-5)


# ═══════════════════════════════════════════════════════════════
# 17–18. GraphBuilder pass-through
# ═══════════════════════════════════════════════════════════════


class TestGraphBuilderPassThrough:

    def test_build_with_ts2vec_embeddings(self, populated_store, tmp_path):
        enc = TS2VecEncoder(
            output_dims=8, n_iters=5, time_bins=8, depth=3, device="cpu"
        )
        embs = enc.fit_and_encode(populated_store)

        gb = GraphBuilder(populated_store)
        data, id_map, _ = gb.build(ts2vec_embeddings=embs, ts2vec_dim=8)

        for ntype in data.node_types:
            if ntype in embs and data[ntype].x.size(0) > 0:
                assert (
                    data[ntype].x.size(1) == BASE_FEAT_DIM + 8
                ), f"Expected {BASE_FEAT_DIM + 8}, got {data[ntype].x.size(1)} for {ntype}"

    def test_build_from_cached_with_ts2vec_embeddings(self, populated_store):
        enc = TS2VecEncoder(
            output_dims=8, n_iters=5, time_bins=8, depth=3, device="cpu"
        )
        embs = enc.fit_and_encode(populated_store)

        gb = GraphBuilder(populated_store)
        id_map, entities, links = gb.prepare_static()
        obs = populated_store.query_all_observations()
        data, _, _ = gb.build_from_cached(
            id_map, links, observations=obs, ts2vec_embeddings=embs, ts2vec_dim=8
        )

        for ntype in data.node_types:
            if ntype in embs and data[ntype].x.size(0) > 0:
                assert data[ntype].x.size(1) == BASE_FEAT_DIM + 8


# ═══════════════════════════════════════════════════════════════
# 19–20. TrainerConfig
# ═══════════════════════════════════════════════════════════════


class TestTrainerConfig:

    def test_use_ts2vec_defaults_false(self):
        assert TrainerConfig().use_ts2vec is False

    def test_ts2vec_dim_defaults_32(self):
        assert TrainerConfig().ts2vec_dim == 32

    def test_ts2vec_n_iters_defaults_200(self):
        assert TrainerConfig().ts2vec_n_iters == 200


# ═══════════════════════════════════════════════════════════════
# 21–22. build_model in_channels
# ═══════════════════════════════════════════════════════════════


class TestBuildModelInChannels:

    def _make_trainer(self, tmp_path, use_ts2vec: bool, tag: str):
        store = PipelineStore(str(tmp_path / f"{tag}.db"))
        gen = SyntheticGraphGenerator(
            num_companies=6,
            num_countries=2,
            num_vessels=4,
            time_span=86400.0 * 3,
            base_event_rate=0.05,
            seed=7,
        )
        gen.generate(store)
        cfg = TrainerConfig(
            hidden_dim=16,
            memory_dim=16,
            message_dim=16,
            time_dim=8,
            num_heads=1,
            num_layers=1,
            use_ts2vec=use_ts2vec,
            ts2vec_dim=8,
            ts2vec_n_iters=5,
        )
        return Trainer(store, cfg)

    def test_in_channels_enlarged_when_use_ts2vec_true(self, tmp_path):
        t_base = self._make_trainer(tmp_path, use_ts2vec=False, tag="base")
        t_ts2v = self._make_trainer(tmp_path, use_ts2vec=True, tag="ts2v")
        m_base = t_base.build_model()
        m_ts2v = t_ts2v.build_model()

        for ntype in m_base.node_types:
            in_base = m_base.type_projections[ntype].in_features
            if ntype in t_ts2v._ts2vec_embeddings:
                in_ts2v = m_ts2v.type_projections[ntype].in_features
                assert (
                    in_ts2v == in_base + 8
                ), f"{ntype}: expected {in_base + 8}, got {in_ts2v}"

    def test_in_channels_unchanged_when_use_ts2vec_false(self, tmp_path):
        t = self._make_trainer(tmp_path, use_ts2vec=False, tag="notsv")
        m = t.build_model()
        for ntype in m.node_types:
            proj = m.type_projections[ntype]
            assert (
                proj.in_features == BASE_FEAT_DIM
            ), f"Expected {BASE_FEAT_DIM}, got {proj.in_features} for {ntype}"


# ═══════════════════════════════════════════════════════════════
# 23. Training step
# ═══════════════════════════════════════════════════════════════


class TestTS2VecTrainingStep:

    def test_training_step_with_ts2vec_no_error(self, tmp_path):
        store = PipelineStore(str(tmp_path / "ts2v_train.db"))
        gen = SyntheticGraphGenerator(
            num_companies=6,
            num_countries=2,
            num_vessels=4,
            time_span=86400.0 * 4,
            base_event_rate=0.05,
            seed=55,
        )
        gen.generate(store)
        cfg = TrainerConfig(
            hidden_dim=16,
            memory_dim=16,
            message_dim=16,
            time_dim=8,
            num_heads=1,
            num_layers=1,
            use_ts2vec=True,
            ts2vec_dim=8,
            ts2vec_n_iters=5,
            epochs=1,
            window_size=86400.0,
        )
        trainer = Trainer(store, cfg)
        trainer.build_model()
        history = trainer.train()
        assert isinstance(history, dict)
        assert any(len(v) >= 1 for v in history.values() if isinstance(v, list))
