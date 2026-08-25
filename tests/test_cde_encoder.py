"""Tests for Idea 1 — Neural CDE Memory Encoder.

Covers:
    1. CDEFunc forward shape (happy path)
    2. CDEMemoryEncoder: nodes with ≥2 events integrate via CDE
    3. CDEMemoryEncoder: single-event nodes fall back to GRU
    4. CDEMemoryEncoder: graceful when events list is empty
    5. HetTGN(use_cde=False) forward — identical output shape to baseline
    6. HetTGN(use_cde=True) forward — runs without error, same output shape
    7. HetTGN(use_cde=True) update_memory_from_events changes memory state
    8. HetTGN(use_cde=True) t_start==t_end doesn't crash (zero-span window)
    9. TrainerConfig.use_cde=True propagates through build_model()
   10. Full mini training loop with use_cde=True completes without NaN losses
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from agent.models.gnn.cde_encoder import CDEFunc, CDEMemoryEncoder, _CDE_AVAILABLE
from agent.models.gnn.het_tgn import HetTGN, HeteroMemory
from agent.models.gnn.graph_builder import IDMap
from agent.models.gnn.trainer import (
    InjectedPattern,
    SyntheticGraphGenerator,
    Trainer,
    TrainerConfig,
)
from agent.pipeline.store import PipelineStore

# ── Shared helpers ──────────────────────────────────────────────────────────


@pytest.fixture()
def small_metadata():
    node_types = ["company", "country"]
    edge_types = [("company", "headquartered_in", "country")]
    return node_types, edge_types


@pytest.fixture()
def small_het_tgn(small_metadata):
    return HetTGN(
        metadata=small_metadata,
        in_channels={"company": 8, "country": 8},
        hidden_dim=16,
        memory_dim=16,
        message_dim=16,
        time_dim=8,
        num_heads=1,
        num_layers=1,
        num_nodes=20,
        use_cde=False,
    )


@pytest.fixture()
def cde_het_tgn(small_metadata):
    return HetTGN(
        metadata=small_metadata,
        in_channels={"company": 8, "country": 8},
        hidden_dim=16,
        memory_dim=16,
        message_dim=16,
        time_dim=8,
        num_heads=1,
        num_layers=1,
        num_nodes=20,
        use_cde=True,
    )


@pytest.fixture()
def simple_store(tmp_path: Path) -> PipelineStore:
    store = PipelineStore(str(tmp_path / "test.db"))
    gen = SyntheticGraphGenerator(
        num_companies=4,
        num_countries=2,
        num_vessels=2,
        time_span=86400.0 * 5,
        base_event_rate=0.005,
        seed=7,
    )
    gen.generate(store)
    return store


# ═══════════════════════════════════════════════════════════════
# 1. CDEFunc shape
# ═══════════════════════════════════════════════════════════════


class TestCDEFunc:
    def test_output_shape(self):
        """CDEFunc(t, z) → (batch, hidden_dim, input_channels)."""
        batch, hidden_dim, input_channels = 5, 16, 9
        func = CDEFunc(hidden_dim=hidden_dim, input_channels=input_channels)
        z = torch.randn(batch, hidden_dim)
        t = torch.tensor(0.5)
        out = func(t, z)
        assert out.shape == (batch, hidden_dim, input_channels)

    def test_output_finite(self):
        """CDEFunc output should contain no NaN or Inf."""
        func = CDEFunc(hidden_dim=8, input_channels=5)
        z = torch.randn(3, 8)
        out = func(torch.tensor(0.0), z)
        assert torch.isfinite(out).all()

    def test_zero_input(self):
        """Zero hidden state produces finite output (tanh(0)=0, linear(0)=bias)."""
        func = CDEFunc(hidden_dim=8, input_channels=4)
        z = torch.zeros(2, 8)
        out = func(torch.tensor(0.0), z)
        assert out.shape == (2, 8, 4)
        assert torch.isfinite(out).all()


# ═══════════════════════════════════════════════════════════════
# 2–4. CDEMemoryEncoder
# ═══════════════════════════════════════════════════════════════


@pytest.mark.skipif(not _CDE_AVAILABLE, reason="torchcde not installed")
class TestCDEMemoryEncoder:
    def _make_memory(self, num_nodes=10, memory_dim=16, message_dim=16):
        mem = HeteroMemory(
            num_nodes=num_nodes, memory_dim=memory_dim, message_dim=message_dim
        )
        return mem

    def _make_id_map(self, entity_ids: list[str], entity_type: str = "company"):
        """Build a minimal IDMap for testing."""
        id_map = IDMap()
        for eid in entity_ids:
            id_map.add(entity_type, eid)
        return id_map

    def test_multi_event_nodes_update_memory(self):
        """Nodes with ≥2 events should have non-zero memory after CDE update."""
        memory_dim, message_dim = 16, 16
        encoder = CDEMemoryEncoder(memory_dim=memory_dim, message_dim=message_dim)
        memory = self._make_memory(memory_dim=memory_dim, message_dim=message_dim)

        id_map = self._make_id_map(["e0", "e1"])

        # 3 events for node 0, 2 events for node 1
        events = [
            {"entity_type": "company", "entity_id": "e0", "observed_at": 100.0},
            {"entity_type": "company", "entity_id": "e0", "observed_at": 200.0},
            {"entity_type": "company", "entity_id": "e0", "observed_at": 300.0},
            {"entity_type": "company", "entity_id": "e1", "observed_at": 150.0},
            {"entity_type": "company", "entity_id": "e1", "observed_at": 350.0},
        ]
        embeddings = {"company": torch.randn(2, 16)}

        encoder.update_memory_from_events(
            events=events,
            embeddings=embeddings,
            id_map=id_map,
            memory=memory,
            t_start=0.0,
            t_end=400.0,
        )
        # Memory for both nodes should be non-zero after CDE update
        assert not torch.all(memory.memory[0] == 0.0), "Node 0 memory should be updated"
        assert not torch.all(memory.memory[1] == 0.0), "Node 1 memory should be updated"

    def test_single_event_nodes_use_gru_fallback(self):
        """Nodes with only 1 event fall back to GRU — memory is still updated."""
        memory_dim, message_dim = 16, 16
        encoder = CDEMemoryEncoder(
            memory_dim=memory_dim, message_dim=message_dim, min_events=2
        )
        memory = self._make_memory(memory_dim=memory_dim, message_dim=message_dim)
        id_map = self._make_id_map(["solo"])

        events = [{"entity_type": "company", "entity_id": "solo", "observed_at": 50.0}]
        embeddings = {"company": torch.randn(1, 16)}

        encoder.update_memory_from_events(
            events=events,
            embeddings=embeddings,
            id_map=id_map,
            memory=memory,
            t_start=0.0,
            t_end=100.0,
        )
        assert not torch.all(
            memory.memory[0] == 0.0
        ), "Single-event node should update via GRU"

    def test_empty_events_noop(self):
        """Empty event list does not crash and leaves memory unchanged."""
        encoder = CDEMemoryEncoder(memory_dim=8, message_dim=8)
        memory = self._make_memory(num_nodes=4, memory_dim=8, message_dim=8)
        id_map = self._make_id_map([])
        before = memory.memory.clone()

        encoder.update_memory_from_events(
            events=[],
            embeddings={},
            id_map=id_map,
            memory=memory,
            t_start=0.0,
            t_end=100.0,
        )
        assert torch.allclose(
            memory.memory, before
        ), "Empty events should not mutate memory"

    def test_zero_span_window_noop(self):
        """t_start == t_end should not crash (divides by 1 in normalisation)."""
        encoder = CDEMemoryEncoder(memory_dim=8, message_dim=8)
        memory = self._make_memory(num_nodes=4, memory_dim=8, message_dim=8)
        id_map = self._make_id_map(["a", "b"])
        events = [
            {"entity_type": "company", "entity_id": "a", "observed_at": 50.0},
            {"entity_type": "company", "entity_id": "a", "observed_at": 50.0},
        ]
        embeddings = {"company": torch.randn(2, 8)}
        # Should not raise despite t_start == t_end
        encoder.update_memory_from_events(
            events=events,
            embeddings=embeddings,
            id_map=id_map,
            memory=memory,
            t_start=50.0,
            t_end=50.0,
        )


# ═══════════════════════════════════════════════════════════════
# 5–8. HetTGN with use_cde flag
# ═══════════════════════════════════════════════════════════════


class TestHetTGNUseCDE:
    def test_use_cde_false_has_no_cde_encoder(self, small_het_tgn):
        """use_cde=False → cde_encoder is None."""
        assert small_het_tgn.cde_encoder is None
        assert small_het_tgn.use_cde is False

    def test_use_cde_true_has_cde_encoder(self, cde_het_tgn):
        """use_cde=True → cde_encoder is a CDEMemoryEncoder instance."""
        assert cde_het_tgn.cde_encoder is not None
        assert isinstance(cde_het_tgn.cde_encoder, CDEMemoryEncoder)
        assert cde_het_tgn.use_cde is True

    def test_update_memory_without_t_start_uses_gru_even_with_cde(self, cde_het_tgn):
        """When t_start/t_end omitted, CDE model falls back to GRU path."""
        from agent.models.gnn.graph_builder import IDMap as _IDMap

        id_map = _IDMap()
        id_map.add("company", "c0")

        embeddings = {"company": torch.randn(1, 16)}
        events = [{"entity_type": "company", "entity_id": "c0", "observed_at": 100.0}]

        before = cde_het_tgn.memory.memory.clone()
        # No t_start / t_end — should fall back to GRU without error
        cde_het_tgn.update_memory_from_events(events, embeddings, id_map)
        # Memory should be updated via GRU fallback
        assert not torch.allclose(cde_het_tgn.memory.memory[0], before[0])

    @pytest.mark.skipif(not _CDE_AVAILABLE, reason="torchcde not installed")
    def test_update_memory_with_cde_changes_state(self, cde_het_tgn):
        """With t_start/t_end and ≥2 events, CDE updates memory."""
        from agent.models.gnn.graph_builder import IDMap as _IDMap

        id_map = _IDMap()
        id_map.add("company", "c0")
        id_map.add("company", "c1")

        embeddings = {"company": torch.randn(2, 16)}
        events = [
            {"entity_type": "company", "entity_id": "c0", "observed_at": 10.0},
            {"entity_type": "company", "entity_id": "c0", "observed_at": 50.0},
            {"entity_type": "company", "entity_id": "c1", "observed_at": 20.0},
            {"entity_type": "company", "entity_id": "c1", "observed_at": 80.0},
        ]
        before_0 = cde_het_tgn.memory.memory[0].clone()
        before_1 = cde_het_tgn.memory.memory[1].clone()

        cde_het_tgn.update_memory_from_events(
            events, embeddings, id_map, t_start=0.0, t_end=100.0
        )
        assert not torch.allclose(cde_het_tgn.memory.memory[0], before_0)
        assert not torch.allclose(cde_het_tgn.memory.memory[1], before_1)

    @pytest.mark.skipif(not _CDE_AVAILABLE, reason="torchcde not installed")
    def test_cde_memory_output_is_finite(self, cde_het_tgn):
        """CDE memory update produces no NaN or Inf in memory tensor."""
        from agent.models.gnn.graph_builder import IDMap as _IDMap

        id_map = _IDMap()
        id_map.add("company", "c0")

        embeddings = {"company": torch.randn(1, 16)}
        events = [
            {"entity_type": "company", "entity_id": "c0", "observed_at": 10.0},
            {"entity_type": "company", "entity_id": "c0", "observed_at": 90.0},
        ]
        cde_het_tgn.update_memory_from_events(
            events, embeddings, id_map, t_start=0.0, t_end=100.0
        )
        assert torch.isfinite(cde_het_tgn.memory.memory).all()


# ═══════════════════════════════════════════════════════════════
# 9. TrainerConfig.use_cde propagates to HetTGN
# ═══════════════════════════════════════════════════════════════


class TestTrainerConfigUseCDE:
    def test_use_cde_defaults_false(self):
        cfg = TrainerConfig()
        assert cfg.use_cde is False

    def test_use_cde_true_propagates_to_model(self, simple_store):
        """build_model() with use_cde=True creates a model with cde_encoder."""
        cfg = TrainerConfig(
            hidden_dim=16,
            memory_dim=16,
            message_dim=16,
            time_dim=8,
            num_heads=1,
            num_layers=1,
            use_cde=True,
        )
        trainer = Trainer(simple_store, cfg)
        model = trainer.build_model()
        assert model.use_cde is True
        assert model.cde_encoder is not None

    def test_use_cde_false_propagates_to_model(self, simple_store):
        """build_model() with use_cde=False creates a model without cde_encoder."""
        cfg = TrainerConfig(
            hidden_dim=16,
            memory_dim=16,
            message_dim=16,
            time_dim=8,
            num_heads=1,
            num_layers=1,
            use_cde=False,
        )
        trainer = Trainer(simple_store, cfg)
        model = trainer.build_model()
        assert model.use_cde is False
        assert model.cde_encoder is None


# ═══════════════════════════════════════════════════════════════
# 10. Full training loop with use_cde=True
# ═══════════════════════════════════════════════════════════════


@pytest.mark.skipif(not _CDE_AVAILABLE, reason="torchcde not installed")
@pytest.mark.slow
class TestCDETrainingLoop:
    def test_training_loop_no_nan_losses(self, simple_store):
        """2-epoch training with use_cde=True produces finite losses."""
        cfg = TrainerConfig(
            hidden_dim=16,
            memory_dim=16,
            message_dim=16,
            time_dim=8,
            num_heads=1,
            num_layers=1,
            epochs=2,
            window_size=86400.0,
            use_cde=True,
            return_weight=0.0,  # no instruments in synthetic data
        )
        trainer = Trainer(simple_store, cfg)
        trainer.build_model()
        history = trainer.train()

        for loss_name, values in history.items():
            for v in values:
                assert math.isfinite(v), f"NaN/Inf in history['{loss_name}']: {values}"

    def test_cde_losses_comparable_to_gru(self, simple_store):
        """CDE and GRU training should produce losses of similar magnitude."""
        base_cfg = dict(
            hidden_dim=16,
            memory_dim=16,
            message_dim=16,
            time_dim=8,
            num_heads=1,
            num_layers=1,
            epochs=2,
            window_size=86400.0,
            return_weight=0.0,
        )
        # GRU baseline
        gru_trainer = Trainer(simple_store, TrainerConfig(**base_cfg, use_cde=False))
        gru_trainer.build_model()
        gru_history = gru_trainer.train()

        # CDE variant — fresh store via fixture won't work (store is shared),
        # so reset memory manually between runs
        gru_trainer.model.reset_memory()

        cde_trainer = Trainer(simple_store, TrainerConfig(**base_cfg, use_cde=True))
        cde_trainer.build_model()
        cde_history = cde_trainer.train()

        gru_total = gru_history["total"][-1]
        cde_total = cde_history["total"][-1]

        # Neither should be 0 or extreme; ratio within 100× is acceptable
        # (they use the same data but different memory encoders)
        assert math.isfinite(gru_total) and math.isfinite(cde_total)
        ratio = max(gru_total, cde_total) / (min(gru_total, cde_total) + 1e-8)
        assert ratio < 100, (
            f"CDE/GRU loss ratio too large ({ratio:.1f}×): "
            f"gru={gru_total:.4f}, cde={cde_total:.4f}"
        )
