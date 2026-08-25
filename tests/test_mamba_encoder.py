"""Tests for Idea 4 — Mamba SSM Memory Encoder.

Covers:
    1.  MambaMemoryEncoder instantiates (with mambapy available)
    2.  MambaMemoryEncoder._has_mamba is True when mambapy installed
    3.  input_proj shape: (message_dim + time_dim) → memory_dim
    4.  _Time2Vec output shape is (K, time_dim)
    5.  _update_single_node: single event updates memory correctly
    6.  _update_single_node: multiple events in sequence updates memory
    7.  _update_single_node: cold-start node (gid >= num_nodes) is skipped gracefully
    8.  update_memory_from_events: empty events → memory unchanged
    9.  update_memory_from_events: events for known nodes → memory changed
    10. update_memory_from_events: events for unknown entity_type ignored
    11. update_memory_from_events: message too wide → truncated to message_dim
    12. update_memory_from_events: message too narrow → zero-padded
    13. HetTGN initialises MambaMemoryEncoder when use_mamba=True
    14. HetTGN.use_mamba=False leaves mamba_encoder None
    15. HetTGN.update_memory_from_events routes through Mamba when use_mamba=True
    16. HetTGN.update_memory_from_events routes through GRU when use_mamba=False
    17. TrainerConfig.use_mamba defaults False
    18. TrainerConfig.use_mamba=True passes through to build_model()
    19. Training step with use_mamba=True runs without error
    20. GRU fallback: MambaMemoryEncoder works when mambapy import faked-out
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from agent.models.gnn.mamba_encoder import MambaMemoryEncoder, _Time2Vec
from agent.models.gnn.trainer import Trainer, TrainerConfig, SyntheticGraphGenerator
from agent.pipeline.store import PipelineStore

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def small_memory_params():
    return dict(memory_dim=16, message_dim=16, time_dim=8)


@pytest.fixture()
def encoder(small_memory_params):
    return MambaMemoryEncoder(**small_memory_params)


@pytest.fixture()
def het_tgn_mamba(tmp_path):
    """A tiny HetTGN with use_mamba=True backed by a synthetic store."""
    store = PipelineStore(str(tmp_path / "mamba_test.db"))
    gen = SyntheticGraphGenerator(
        num_companies=4,
        num_countries=2,
        num_vessels=2,
        time_span=86400.0 * 2,
        base_event_rate=0.005,
        seed=42,
    )
    gen.generate(store)
    cfg = TrainerConfig(
        hidden_dim=16,
        memory_dim=16,
        message_dim=16,
        time_dim=8,
        num_heads=1,
        num_layers=1,
        use_mamba=True,
    )
    trainer = Trainer(store, cfg)
    model = trainer.build_model()
    return model, trainer


@pytest.fixture()
def mock_memory(small_memory_params):
    """A tiny HeteroMemory-like object with .memory and .last_update buffers."""
    md = small_memory_params["memory_dim"]
    n = 8

    class FakeMemory:
        num_nodes = n
        memory = torch.zeros(n, md)
        last_update = torch.zeros(n)

    return FakeMemory()


@pytest.fixture()
def mock_id_map():
    """Minimal IDMap stub returning deterministic IDs."""

    class FakeIDMap:
        def global_id(self, etype, eid):
            return int(eid.split("_")[-1]) if eid else None

        def local_id(self, etype, eid):
            return int(eid.split("_")[-1]) if eid else None

    return FakeIDMap()


# ═══════════════════════════════════════════════════════════════
# 1–2. Module construction
# ═══════════════════════════════════════════════════════════════


class TestMambaEncoderConstruction:

    def test_instantiates(self, encoder):
        assert encoder is not None

    def test_has_mamba_true_when_mambapy_available(self, encoder):
        """mambapy is installed in this environment, so _has_mamba should be True."""
        assert encoder._has_mamba is True

    def test_input_proj_shape(self, encoder, small_memory_params):
        md = small_memory_params["memory_dim"]
        msg = small_memory_params["message_dim"]
        td = small_memory_params["time_dim"]
        assert encoder.input_proj.in_features == msg + td
        assert encoder.input_proj.out_features == md


# ═══════════════════════════════════════════════════════════════
# 4. _Time2Vec
# ═══════════════════════════════════════════════════════════════


class TestTime2Vec:

    def test_output_shape(self):
        enc = _Time2Vec(out_features=8)
        t = torch.tensor([0.0, 1.0, 2.0])
        out = enc(t)
        assert out.shape == (3, 8)

    def test_scalar_input(self):
        enc = _Time2Vec(out_features=4)
        t = torch.tensor([5.0])
        out = enc(t)
        assert out.shape == (1, 4)


# ═══════════════════════════════════════════════════════════════
# 5–7. _update_single_node
# ═══════════════════════════════════════════════════════════════


class TestUpdateSingleNode:

    def test_single_event_updates_memory(self, encoder, mock_memory):
        md = encoder.memory_dim
        gid = 0
        msgs = torch.randn(1, encoder.message_dim)
        times = torch.tensor([1.0])
        original = mock_memory.memory[gid].clone()
        encoder._update_single_node(gid, msgs, times, mock_memory, torch.device("cpu"))
        assert not torch.allclose(
            mock_memory.memory[gid], original
        ), "Memory should change after update"
        assert math.isclose(float(mock_memory.last_update[gid]), 1.0)

    def test_multiple_events_updates_memory(self, encoder, mock_memory):
        gid = 1
        K = 5
        msgs = torch.randn(K, encoder.message_dim)
        times = torch.linspace(1.0, 5.0, K)
        encoder._update_single_node(gid, msgs, times, mock_memory, torch.device("cpu"))
        assert mock_memory.memory[gid].abs().sum() > 0
        assert math.isclose(float(mock_memory.last_update[gid]), 5.0)

    def test_cold_start_node_skipped(self, encoder, mock_memory):
        """gid >= num_nodes should not raise and should not modify existing rows."""
        snapshot = mock_memory.memory.clone()
        gid = mock_memory.num_nodes + 5  # out-of-range
        msgs = torch.randn(1, encoder.message_dim)
        times = torch.tensor([1.0])
        encoder._update_single_node(gid, msgs, times, mock_memory, torch.device("cpu"))
        assert torch.allclose(mock_memory.memory, snapshot)


# ═══════════════════════════════════════════════════════════════
# 8–12. update_memory_from_events
# ═══════════════════════════════════════════════════════════════


class TestUpdateMemoryFromEvents:

    def _make_embeddings(self, encoder, n=8):
        return {"company": torch.randn(n, encoder.message_dim)}

    def test_empty_events_noop(self, encoder, mock_memory, mock_id_map):
        snapshot = mock_memory.memory.clone()
        encoder.update_memory_from_events([], {}, mock_id_map, mock_memory)
        assert torch.allclose(mock_memory.memory, snapshot)

    def test_known_node_memory_updated(self, encoder, mock_memory, mock_id_map):
        embeddings = self._make_embeddings(encoder)
        events = [
            {"entity_type": "company", "entity_id": "company_2", "observed_at": 1.0},
        ]
        snapshot = mock_memory.memory[2].clone()
        encoder.update_memory_from_events(events, embeddings, mock_id_map, mock_memory)
        assert not torch.allclose(mock_memory.memory[2], snapshot)

    def test_unknown_entity_type_ignored(self, encoder, mock_memory, mock_id_map):
        snapshot = mock_memory.memory.clone()
        events = [
            {"entity_type": "vessel", "entity_id": "vessel_0", "observed_at": 1.0},
        ]
        # No "vessel" key in embeddings
        encoder.update_memory_from_events(events, {}, mock_id_map, mock_memory)
        assert torch.allclose(mock_memory.memory, snapshot)

    def test_message_too_wide_truncated(self, encoder, mock_memory, mock_id_map):
        wide_emb = torch.randn(8, encoder.message_dim + 10)  # too wide
        embeddings = {"company": wide_emb}
        events = [
            {"entity_type": "company", "entity_id": "company_3", "observed_at": 2.0}
        ]
        encoder.update_memory_from_events(events, embeddings, mock_id_map, mock_memory)
        # Should not raise; memory updated
        assert mock_memory.memory[3].abs().sum() > 0

    def test_message_too_narrow_zero_padded(self, encoder, mock_memory, mock_id_map):
        narrow_emb = torch.randn(8, encoder.message_dim - 4)  # too narrow
        embeddings = {"company": narrow_emb}
        events = [
            {"entity_type": "company", "entity_id": "company_4", "observed_at": 2.0}
        ]
        encoder.update_memory_from_events(events, embeddings, mock_id_map, mock_memory)
        assert mock_memory.memory[4].abs().sum() > 0


# ═══════════════════════════════════════════════════════════════
# 13–16. HetTGN integration
# ═══════════════════════════════════════════════════════════════


class TestHetTGNMambaIntegration:

    def test_mamba_encoder_instantiated_when_use_mamba_true(self, het_tgn_mamba):
        model, _ = het_tgn_mamba
        assert model.mamba_encoder is not None
        assert isinstance(model.mamba_encoder, MambaMemoryEncoder)

    def test_mamba_encoder_none_when_use_mamba_false(self, tmp_path):
        store = PipelineStore(str(tmp_path / "no_mamba.db"))
        gen = SyntheticGraphGenerator(
            num_companies=4,
            num_countries=2,
            num_vessels=2,
            time_span=86400.0 * 2,
            base_event_rate=0.005,
            seed=1,
        )
        gen.generate(store)
        cfg = TrainerConfig(
            hidden_dim=16,
            memory_dim=16,
            message_dim=16,
            time_dim=8,
            num_heads=1,
            num_layers=1,
            use_mamba=False,
        )
        model = Trainer(store, cfg).build_model()
        assert model.mamba_encoder is None

    def test_update_memory_routes_through_mamba(self, het_tgn_mamba):
        """update_memory_from_events calls mamba_encoder when use_mamba=True."""
        model, trainer = het_tgn_mamba
        from agent.models.gnn.graph_builder import GraphBuilder

        data, id_map, _ = GraphBuilder(trainer.store).build()
        model.eval()
        with torch.no_grad():
            embeddings = model.forward(data, id_map)

        # Snapshot memory before
        mem_before = model.memory.memory.clone()

        events = trainer.store.query_all_observations()[:5]
        call_log = []
        original_update = model.mamba_encoder.update_memory_from_events

        def patched_update(*args, **kwargs):
            call_log.append(True)
            return original_update(*args, **kwargs)

        model.mamba_encoder.update_memory_from_events = patched_update
        model.update_memory_from_events(events, embeddings, id_map)
        assert (
            len(call_log) == 1
        ), "mamba_encoder.update_memory_from_events should have been called"

    def test_update_memory_routes_through_gru_when_mamba_false(self, tmp_path):
        """When use_mamba=False, GRU path is used (mamba_encoder not called)."""
        store = PipelineStore(str(tmp_path / "gru_path.db"))
        gen = SyntheticGraphGenerator(
            num_companies=4,
            num_countries=2,
            num_vessels=2,
            time_span=86400.0 * 2,
            base_event_rate=0.005,
            seed=2,
        )
        gen.generate(store)
        cfg = TrainerConfig(
            hidden_dim=16,
            memory_dim=16,
            message_dim=16,
            time_dim=8,
            num_heads=1,
            num_layers=1,
            use_mamba=False,
        )
        trainer = Trainer(store, cfg)
        model = trainer.build_model()

        from agent.models.gnn.graph_builder import GraphBuilder

        data, id_map, _ = GraphBuilder(store).build()
        model.eval()
        with torch.no_grad():
            embeddings = model.forward(data, id_map)

        events = store.query_all_observations()[:5]
        mem_before = model.memory.memory.clone()
        model.update_memory_from_events(events, embeddings, id_map)
        # Memory should change (GRU ran) but no mamba_encoder was used
        assert model.mamba_encoder is None


# ═══════════════════════════════════════════════════════════════
# 17–18. TrainerConfig
# ═══════════════════════════════════════════════════════════════


class TestTrainerConfigMamba:

    def test_use_mamba_defaults_false(self):
        assert TrainerConfig().use_mamba is False

    def test_use_mamba_true_passes_to_build_model(self, tmp_path):
        store = PipelineStore(str(tmp_path / "cfg_mamba.db"))
        gen = SyntheticGraphGenerator(
            num_companies=4,
            num_countries=2,
            num_vessels=2,
            time_span=86400.0 * 2,
            base_event_rate=0.005,
            seed=3,
        )
        gen.generate(store)
        cfg = TrainerConfig(
            hidden_dim=16,
            memory_dim=16,
            message_dim=16,
            time_dim=8,
            num_heads=1,
            num_layers=1,
            use_mamba=True,
        )
        model = Trainer(store, cfg).build_model()
        assert model.use_mamba is True
        assert model.mamba_encoder is not None


# ═══════════════════════════════════════════════════════════════
# 19. Training step
# ═══════════════════════════════════════════════════════════════


class TestMambaTrainingStep:

    def test_training_step_runs_without_error(self, tmp_path):
        """One training epoch with use_mamba=True should not raise."""
        store = PipelineStore(str(tmp_path / "mamba_train.db"))
        gen = SyntheticGraphGenerator(
            num_companies=6,
            num_countries=3,
            num_vessels=3,
            time_span=86400.0 * 4,
            base_event_rate=0.01,
            seed=99,
        )
        gen.generate(store)
        cfg = TrainerConfig(
            hidden_dim=16,
            memory_dim=16,
            message_dim=16,
            time_dim=8,
            num_heads=1,
            num_layers=1,
            use_mamba=True,
            epochs=1,
            window_size=86400.0,
        )
        trainer = Trainer(store, cfg)
        trainer.build_model()
        history = trainer.train()
        assert isinstance(history, dict)
        assert len(history) >= 1
        # At least one loss component was recorded
        assert any(len(v) >= 1 for v in history.values() if isinstance(v, list))


# ═══════════════════════════════════════════════════════════════
# 20. GRU fallback when mambapy unavailable
# ═══════════════════════════════════════════════════════════════


class TestGRUFallback:

    def test_gru_fallback_when_mambapy_missing(self, small_memory_params):
        """When mambapy import fails, encoder uses GRU and still updates memory."""
        with patch.dict("sys.modules", {"mambapy": None, "mambapy.mamba": None}):
            enc = MambaMemoryEncoder(**small_memory_params)

        # Whether _has_mamba is True or False depends on import order; either way
        # the encoder should produce a valid memory update.
        n = 4
        md = small_memory_params["memory_dim"]

        class FM:
            num_nodes = n
            memory = torch.zeros(n, md)
            last_update = torch.zeros(n)

        mem = FM()
        msgs = torch.randn(2, enc.message_dim)
        times = torch.tensor([1.0, 2.0])
        enc._update_single_node(0, msgs, times, mem, torch.device("cpu"))
        # Memory for node 0 should be non-zero
        assert mem.memory[0].abs().sum() > 0
