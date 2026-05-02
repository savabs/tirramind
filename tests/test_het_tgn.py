"""Tests for Phase 12c: HeteroMemory + HetTGN model.

Covers:
    HeteroMemory — init, get/update, reset, time encoding, edge cases
    HetTGN — forward shapes, prediction heads, link score, memory update,
             gradient flow, save/load, degenerate graphs
"""

from __future__ import annotations

import io

import torch
from torch_geometric.data import HeteroData

from agent.models.gnn.graph_builder import OBSERVATION_TYPES, IDMap
from agent.models.gnn.het_tgn import HeteroMemory, HetTGN

# ─── Helpers ──────────────────────────────────────────────────


def _make_simple_graph() -> tuple[HeteroData, IDMap, tuple]:
    """Tiny graph: 3 companies, 2 countries, 1 edge type.

    Returns (data, id_map, metadata).
    """
    id_map = IDMap()
    for i in range(3):
        id_map.add("company", f"c{i}")
    for i in range(2):
        id_map.add("country", f"co{i}")

    feat_dim = 9  # match graph_builder output
    data = HeteroData()
    data["company"].x = torch.randn(3, feat_dim)
    data["company"].node_ids = ["c0", "c1", "c2"]
    data["country"].x = torch.randn(2, feat_dim)
    data["country"].node_ids = ["co0", "co1"]

    # c0→co0, c1→co1, c2→co0
    data["company", "headquartered_in", "country"].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 0]],
        dtype=torch.long,
    )
    data["company", "headquartered_in", "country"].edge_attr = torch.randn(3, 2)

    node_types = ["company", "country"]
    edge_types = [("company", "headquartered_in", "country")]
    metadata = (node_types, edge_types)

    return data, id_map, metadata


def _make_model(
    metadata: tuple | None = None,
    in_channels: dict | None = None,
    num_nodes: int = 5,
    **kwargs,
) -> HetTGN:
    """Build a small HetTGN for testing."""
    if metadata is None:
        _, _, metadata = _make_simple_graph()
    if in_channels is None:
        in_channels = {"company": 9, "country": 9}
    return HetTGN(
        metadata=metadata,
        in_channels=in_channels,
        hidden_dim=kwargs.get("hidden_dim", 16),
        time_dim=kwargs.get("time_dim", 8),
        memory_dim=kwargs.get("memory_dim", 16),
        message_dim=kwargs.get("message_dim", 16),
        num_heads=kwargs.get("num_heads", 2),
        num_layers=kwargs.get("num_layers", 1),
        num_nodes=num_nodes,
    )


# ═══════════════════════════════════════════════════════════════
# HeteroMemory
# ═══════════════════════════════════════════════════════════════


class TestHeteroMemoryInit:
    def test_memory_shape(self):
        mem = HeteroMemory(num_nodes=10, memory_dim=32)
        assert mem.memory.shape == (10, 32)

    def test_last_update_shape(self):
        mem = HeteroMemory(num_nodes=10)
        assert mem.last_update.shape == (10,)

    def test_initial_memory_is_zero(self):
        mem = HeteroMemory(num_nodes=5, memory_dim=8)
        assert (mem.memory == 0).all()
        assert (mem.last_update == 0).all()


class TestHeteroMemoryGetUpdate:
    def test_get_memory_shape(self):
        mem = HeteroMemory(num_nodes=10, memory_dim=16)
        ids = torch.tensor([0, 3, 7])
        m, t = mem.get_memory(ids)
        assert m.shape == (3, 16)
        assert t.shape == (3,)

    def test_get_memory_no_grad(self):
        """Retrieved memory should be detached."""
        mem = HeteroMemory(num_nodes=5, memory_dim=8)
        ids = torch.tensor([0])
        m, t = mem.get_memory(ids)
        assert not m.requires_grad
        assert not t.requires_grad

    def test_update_changes_memory(self):
        mem = HeteroMemory(num_nodes=5, memory_dim=8, message_dim=8, time_dim=4)
        ids = torch.tensor([2])
        msg = torch.randn(1, 8)
        ts = torch.tensor([100.0])
        old_mem = mem.memory[2].clone()
        mem.update_memory(ids, msg, ts)
        assert not torch.allclose(mem.memory[2], old_mem)

    def test_update_changes_last_update(self):
        mem = HeteroMemory(num_nodes=5, memory_dim=8, message_dim=8, time_dim=4)
        ids = torch.tensor([1])
        msg = torch.randn(1, 8)
        ts = torch.tensor([42.0])
        mem.update_memory(ids, msg, ts)
        assert mem.last_update[1].item() == 42.0

    def test_update_preserves_other_nodes(self):
        mem = HeteroMemory(num_nodes=5, memory_dim=8, message_dim=8, time_dim=4)
        # Update node 2
        ids = torch.tensor([2])
        msg = torch.randn(1, 8)
        ts = torch.tensor([100.0])
        mem.update_memory(ids, msg, ts)
        # Node 0 should still be zero
        assert (mem.memory[0] == 0).all()
        assert mem.last_update[0].item() == 0.0

    def test_multiple_updates_same_node(self):
        mem = HeteroMemory(num_nodes=3, memory_dim=8, message_dim=8, time_dim=4)
        ids = torch.tensor([1])
        # First update
        mem.update_memory(ids, torch.randn(1, 8), torch.tensor([10.0]))
        mem1 = mem.memory[1].clone()
        # Second update
        mem.update_memory(ids, torch.randn(1, 8), torch.tensor([20.0]))
        mem2 = mem.memory[1].clone()
        # Should change
        assert not torch.allclose(mem1, mem2)
        assert mem.last_update[1].item() == 20.0

    def test_batch_update(self):
        mem = HeteroMemory(num_nodes=5, memory_dim=8, message_dim=8, time_dim=4)
        ids = torch.tensor([0, 2, 4])
        msg = torch.randn(3, 8)
        ts = torch.tensor([100.0, 200.0, 300.0])
        mem.update_memory(ids, msg, ts)
        assert mem.last_update[0].item() == 100.0
        assert mem.last_update[2].item() == 200.0
        assert mem.last_update[4].item() == 300.0
        # Untouched nodes
        assert mem.last_update[1].item() == 0.0
        assert mem.last_update[3].item() == 0.0

    def test_empty_update(self):
        mem = HeteroMemory(num_nodes=5, memory_dim=8, message_dim=8, time_dim=4)
        old = mem.memory.clone()
        mem.update_memory(torch.tensor([], dtype=torch.long), torch.zeros(0, 8), torch.tensor([]))
        assert torch.allclose(mem.memory, old)


class TestHeteroMemoryReset:
    def test_reset_zeros_memory(self):
        mem = HeteroMemory(num_nodes=3, memory_dim=8, message_dim=8, time_dim=4)
        mem.update_memory(
            torch.tensor([0, 1]),
            torch.randn(2, 8),
            torch.tensor([10.0, 20.0]),
        )
        mem.reset()
        assert (mem.memory == 0).all()
        assert (mem.last_update == 0).all()


class TestHeteroMemoryEdgeCases:
    def test_negative_time_delta_clamped(self):
        """If timestamp < last_update, delta is clamped to 0."""
        mem = HeteroMemory(num_nodes=3, memory_dim=8, message_dim=8, time_dim=4)
        ids = torch.tensor([0])
        mem.update_memory(ids, torch.randn(1, 8), torch.tensor([100.0]))
        # Update with earlier timestamp
        mem.update_memory(ids, torch.randn(1, 8), torch.tensor([50.0]))
        # Should not crash, memory updated
        assert mem.last_update[0].item() == 50.0

    def test_very_large_time_delta(self):
        mem = HeteroMemory(num_nodes=3, memory_dim=8, message_dim=8, time_dim=4)
        ids = torch.tensor([0])
        mem.update_memory(ids, torch.randn(1, 8), torch.tensor([1e12]))
        m, _ = mem.get_memory(ids)
        assert torch.isfinite(m).all()


# ═══════════════════════════════════════════════════════════════
# HetTGN model
# ═══════════════════════════════════════════════════════════════


class TestHetTGNForward:
    def test_output_types(self):
        data, id_map, metadata = _make_simple_graph()
        model = _make_model(metadata)
        out = model(data, id_map)
        assert isinstance(out, dict)
        assert "company" in out
        assert "country" in out

    def test_output_shapes(self):
        data, id_map, metadata = _make_simple_graph()
        model = _make_model(metadata, hidden_dim=16)
        out = model(data, id_map)
        assert out["company"].shape == (3, 16)
        assert out["country"].shape == (2, 16)

    def test_output_finite(self):
        data, id_map, metadata = _make_simple_graph()
        model = _make_model(metadata)
        out = model(data, id_map)
        for ntype, emb in out.items():
            assert torch.isfinite(emb).all(), f"NaN/Inf in {ntype} embeddings"

    def test_gradient_flow(self):
        data, id_map, metadata = _make_simple_graph()
        model = _make_model(metadata)
        out = model(data, id_map)
        loss = sum(emb.sum() for emb in out.values())
        loss.backward()
        # Check gradients flow to type projections and HGT
        for ntype, proj in model.type_projections.items():
            assert proj.weight.grad is not None, f"No gradient for {ntype} proj"

    def test_deterministic_eval(self):
        data, id_map, metadata = _make_simple_graph()
        model = _make_model(metadata)
        model.eval()
        model.reset_memory()
        with torch.no_grad():
            out1 = model(data, id_map)
        model.reset_memory()
        with torch.no_grad():
            out2 = model(data, id_map)
        for ntype in out1:
            assert torch.allclose(out1[ntype], out2[ntype])


class TestHetTGNPredictionHeads:
    def test_obs_type_logits_shape(self):
        data, id_map, metadata = _make_simple_graph()
        model = _make_model(metadata)
        out = model(data, id_map)
        logits = model.predict_obs_type(out)
        assert logits["company"].shape == (3, len(OBSERVATION_TYPES))
        assert logits["country"].shape == (2, len(OBSERVATION_TYPES))

    def test_time_delta_shape(self):
        data, id_map, metadata = _make_simple_graph()
        model = _make_model(metadata)
        out = model(data, id_map)
        dt = model.predict_time_delta(out)
        assert dt["company"].shape == (3, 1)
        assert dt["country"].shape == (2, 1)

    def test_time_delta_non_negative(self):
        """Softplus ensures time deltas >= 0."""
        data, id_map, metadata = _make_simple_graph()
        model = _make_model(metadata)
        out = model(data, id_map)
        dt = model.predict_time_delta(out)
        for ntype, d in dt.items():
            assert (d >= 0).all(), f"Negative time delta in {ntype}"


class TestHetTGNLinkScore:
    def test_link_score_shape(self):
        model = _make_model()
        u = torch.randn(5, 16)
        v = torch.randn(5, 16)
        scores = model.link_score(u, v)
        assert scores.shape == (5,)

    def test_link_score_gradient(self):
        model = _make_model()
        u = torch.randn(3, 16, requires_grad=True)
        v = torch.randn(3, 16, requires_grad=True)
        scores = model.link_score(u, v)
        scores.sum().backward()
        assert u.grad is not None
        assert v.grad is not None


class TestHetTGNMemoryUpdate:
    def test_update_from_events(self):
        data, id_map, metadata = _make_simple_graph()
        model = _make_model(metadata)
        model.eval()
        out = model(data, id_map)

        events = [
            {
                "entity_type": "company",
                "entity_id": "c0",
                "observed_at": 1000.0,
                "observation_type": "insider_trade",
            },
        ]
        model.update_memory_from_events(events, out, id_map)
        gid = id_map.global_id("company", "c0")
        assert model.memory.last_update[gid].item() == 1000.0

    def test_update_from_empty_events(self):
        data, id_map, metadata = _make_simple_graph()
        model = _make_model(metadata)
        out = model(data, id_map)
        old_mem = model.memory.memory.clone()
        model.update_memory_from_events([], out, id_map)
        assert torch.allclose(model.memory.memory, old_mem)

    def test_update_unknown_entity_skipped(self):
        data, id_map, metadata = _make_simple_graph()
        model = _make_model(metadata)
        out = model(data, id_map)
        events = [
            {
                "entity_type": "company",
                "entity_id": "unknown_xyz",
                "observed_at": 500.0,
                "observation_type": "insider_trade",
            },
        ]
        old_mem = model.memory.memory.clone()
        model.update_memory_from_events(events, out, id_map)
        # Memory unchanged (unknown entity skipped)
        assert torch.allclose(model.memory.memory, old_mem)

    def test_update_multiple_events(self):
        data, id_map, metadata = _make_simple_graph()
        model = _make_model(metadata)
        out = model(data, id_map)
        events = [
            {
                "entity_type": "company",
                "entity_id": "c0",
                "observed_at": 100.0,
                "observation_type": "insider_trade",
            },
            {
                "entity_type": "country",
                "entity_id": "co1",
                "observed_at": 200.0,
                "observation_type": "geopolitical_event",
            },
        ]
        model.update_memory_from_events(events, out, id_map)
        gid_c0 = id_map.global_id("company", "c0")
        gid_co1 = id_map.global_id("country", "co1")
        assert model.memory.last_update[gid_c0].item() == 100.0
        assert model.memory.last_update[gid_co1].item() == 200.0

    def test_reset_memory(self):
        data, id_map, metadata = _make_simple_graph()
        model = _make_model(metadata)
        out = model(data, id_map)
        events = [
            {
                "entity_type": "company",
                "entity_id": "c0",
                "observed_at": 100.0,
                "observation_type": "insider_trade",
            },
        ]
        model.update_memory_from_events(events, out, id_map)
        model.reset_memory()
        assert (model.memory.memory == 0).all()
        assert (model.memory.last_update == 0).all()


class TestHetTGNEdgeCases:
    def test_single_node_type(self):
        """Graph with only one node type."""
        id_map = IDMap()
        for i in range(4):
            id_map.add("company", f"c{i}")

        data = HeteroData()
        data["company"].x = torch.randn(4, 9)
        data["company"].node_ids = ["c0", "c1", "c2", "c3"]
        # Self-loop edge type
        data["company", "related_to", "company"].edge_index = torch.tensor(
            [[0, 1], [1, 0]],
            dtype=torch.long,
        )
        metadata = (["company"], [("company", "related_to", "company")])
        model = HetTGN(
            metadata=metadata,
            in_channels={"company": 9},
            hidden_dim=16,
            memory_dim=16,
            message_dim=16,
            num_heads=2,
            num_layers=1,
            num_nodes=4,
        )
        out = model(data, id_map)
        assert "company" in out
        assert out["company"].shape == (4, 16)

    def test_no_edges(self):
        """Graph with nodes but no edges — HGT should still work."""
        id_map = IDMap()
        id_map.add("company", "c0")
        id_map.add("country", "co0")

        data = HeteroData()
        data["company"].x = torch.randn(1, 9)
        data["company"].node_ids = ["c0"]
        data["country"].x = torch.randn(1, 9)
        data["country"].node_ids = ["co0"]
        # No edges at all
        metadata = (
            ["company", "country"],
            [("company", "headquartered_in", "country")],
        )

        model = HetTGN(
            metadata=metadata,
            in_channels={"company": 9, "country": 9},
            hidden_dim=16,
            memory_dim=16,
            message_dim=16,
            num_heads=2,
            num_layers=1,
            num_nodes=2,
        )
        out = model(data, id_map)
        assert out["company"].shape == (1, 16)
        assert out["country"].shape == (1, 16)
        assert torch.isfinite(out["company"]).all()
        assert torch.isfinite(out["country"]).all()

    def test_save_load_roundtrip(self):
        data, id_map, metadata = _make_simple_graph()
        model = _make_model(metadata)
        model.eval()
        model.reset_memory()
        with torch.no_grad():
            out_before = model(data, id_map)

        # Save
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)

        # Load into fresh model
        model2 = _make_model(metadata)
        model2.load_state_dict(torch.load(buf, weights_only=True))
        model2.eval()
        model2.reset_memory()
        with torch.no_grad():
            out_after = model2(data, id_map)

        for ntype in out_before:
            assert torch.allclose(out_before[ntype], out_after[ntype], atol=1e-6)

    def test_event_missing_fields_skipped(self):
        """Events with None entity_type or entity_id are skipped."""
        data, id_map, metadata = _make_simple_graph()
        model = _make_model(metadata)
        out = model(data, id_map)
        events = [
            {"entity_type": None, "entity_id": "c0", "observed_at": 100.0},
            {"entity_type": "company", "entity_id": None, "observed_at": 200.0},
            {"observed_at": 300.0},
        ]
        old_mem = model.memory.memory.clone()
        model.update_memory_from_events(events, out, id_map)
        assert torch.allclose(model.memory.memory, old_mem)

    def test_num_layers_2(self):
        """Deeper model doesn't crash."""
        data, id_map, metadata = _make_simple_graph()
        model = _make_model(metadata, num_layers=2)
        out = model(data, id_map)
        assert out["company"].shape == (3, 16)

    def test_different_hidden_dims(self):
        """hidden_dim != memory_dim should work."""
        data, id_map, metadata = _make_simple_graph()
        model = _make_model(metadata, hidden_dim=32, memory_dim=16)
        out = model(data, id_map)
        assert out["company"].shape == (3, 32)
