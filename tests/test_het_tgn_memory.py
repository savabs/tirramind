"""Regression tests for the HeteroMemory update path (audit finding #7/#10).

Background
----------
``HetTGN.update_memory_from_events`` used to read ``ev["entity_type"]`` from
each observation dict and ``continue`` when it was absent.  Store observation
rows (``PipelineStore._entity_obs_row_to_dict``) carry only the
``entity_observations`` columns — id, entity_id, source_tool, observed_at,
ingested_at, observation_type, depth_level, value, metadata — and there is no
``entity_type`` among them.  Every one of the 384,285 rows therefore hit the
``continue``, and all 21 on-disk checkpoints carry an all-zero memory buffer
with ``memory.gru`` / ``memory.time_enc`` still at random init.

These tests pin:
    1. store-shaped events (no ``entity_type`` key) actually write memory;
    2. every event that cannot be applied is counted and logged loudly —
       "memory received zero usable events" must never look like "memory works";
    3. ``HeteroMemory.resize`` still preserves rows and zero-pads new ones.

Tests 1 and 2 fail on the pre-fix code (memory stays exactly zero, and there
is no counter at all).
"""

from __future__ import annotations

import logging

import pytest
import torch
from torch_geometric.data import HeteroData

from agent.models.gnn.graph_builder import IDMap
from agent.models.gnn.het_tgn import HeteroMemory, HetTGN

# ─── Helpers ──────────────────────────────────────────────────

FEAT_DIM = 9


def _make_graph(num_companies: int = 3, num_countries: int = 2) -> tuple[HeteroData, IDMap, tuple]:
    id_map = IDMap()
    for i in range(num_companies):
        id_map.add("company", f"c{i}")
    for i in range(num_countries):
        id_map.add("country", f"co{i}")

    data = HeteroData()
    data["company"].x = torch.randn(num_companies, FEAT_DIM)
    data["company"].node_ids = [f"c{i}" for i in range(num_companies)]
    data["country"].x = torch.randn(num_countries, FEAT_DIM)
    data["country"].node_ids = [f"co{i}" for i in range(num_countries)]
    data["company", "headquartered_in", "country"].edge_index = torch.tensor(
        [[0, 1], [0, 1]],
        dtype=torch.long,
    )

    metadata = (["company", "country"], [("company", "headquartered_in", "country")])
    return data, id_map, metadata


def _make_model(metadata: tuple, num_nodes: int = 8) -> HetTGN:
    return HetTGN(
        metadata=metadata,
        in_channels={"company": FEAT_DIM, "country": FEAT_DIM},
        hidden_dim=16,
        time_dim=8,
        memory_dim=16,
        message_dim=16,
        num_heads=2,
        num_layers=1,
        num_nodes=num_nodes,
    )


def _store_row(entity_id: str, observed_at: float, obs_type: str = "insider_trade") -> dict:
    """An observation dict shaped exactly like PipelineStore returns one.

    Note the absence of any ``entity_type`` key — that absence is the bug.
    """
    return {
        "id": 1,
        "entity_id": entity_id,
        "source_tool": "sec_insider",
        "observed_at": observed_at,
        "ingested_at": observed_at + 60.0,
        "observation_type": obs_type,
        "depth_level": 1,
        "value": {"usd_amount": 1234.0},
        "metadata": None,
    }


def _nonzero_rows(model: HetTGN) -> int:
    return int((model.memory.memory.abs().sum(dim=1) > 0).sum().item())


# ═══════════════════════════════════════════════════════════════
# 1. Resolvable events actually reach memory
# ═══════════════════════════════════════════════════════════════


class TestMemoryReceivesStoreEvents:
    def test_store_shaped_events_write_memory(self):
        """N store rows with no entity_type key → memory is no longer zero.

        Fails on the pre-fix code: every row hit `continue`.
        """
        data, id_map, metadata = _make_graph()
        model = _make_model(metadata)
        model.eval()
        with torch.no_grad():
            embeddings = model(data, id_map)

        assert _nonzero_rows(model) == 0, "memory should start empty"

        events = [
            _store_row("c0", 1000.0),
            _store_row("c1", 1100.0),
            _store_row("co0", 1200.0, "geopolitical_event"),
        ]
        stats = model.update_memory_from_events(events, embeddings, id_map)

        assert stats["events_in"] == 3
        assert stats["resolved"] == 3
        assert stats["applied"] == 3
        assert _nonzero_rows(model) == 3
        assert stats["memory_nonzero_rows"] == 3

        gid = id_map.global_id("company", "c0")
        assert model.memory.last_update[gid].item() == pytest.approx(1000.0)

    def test_repeated_events_on_one_node_applied_sequentially(self):
        """F-11: duplicates for one node are applied in chronological steps."""
        data, id_map, metadata = _make_graph()
        model = _make_model(metadata)
        model.eval()
        with torch.no_grad():
            embeddings = model(data, id_map)

        events = [
            _store_row("c0", 300.0),
            _store_row("c0", 100.0),
            _store_row("c0", 200.0),
        ]
        stats = model.update_memory_from_events(events, embeddings, id_map)

        assert stats["applied"] == 3
        gid = id_map.global_id("company", "c0")
        # Last write wins and is the chronologically latest event.
        assert model.memory.last_update[gid].item() == pytest.approx(300.0)
        assert _nonzero_rows(model) == 1

    def test_stats_are_stored_on_the_model(self):
        data, id_map, metadata = _make_graph()
        model = _make_model(metadata)
        model.eval()
        with torch.no_grad():
            embeddings = model(data, id_map)

        stats = model.update_memory_from_events([_store_row("c0", 10.0)], embeddings, id_map)
        assert model.last_memory_stats is stats
        assert stats["path"] == "gru"

    def test_id_map_wins_over_a_wrong_declared_type(self):
        """A stale entity_type on the event must not route the write elsewhere."""
        data, id_map, metadata = _make_graph()
        model = _make_model(metadata)
        model.eval()
        with torch.no_grad():
            embeddings = model(data, id_map)

        ev = _store_row("c0", 500.0)
        ev["entity_type"] = "country"  # wrong — c0 is a company
        stats = model.update_memory_from_events([ev], embeddings, id_map)

        assert stats["type_disagreement"] == 1
        assert stats["applied"] == 1
        gid = id_map.global_id("company", "c0")
        assert model.memory.last_update[gid].item() == pytest.approx(500.0)

    def test_events_are_not_mutated(self):
        data, id_map, metadata = _make_graph()
        model = _make_model(metadata)
        model.eval()
        with torch.no_grad():
            embeddings = model(data, id_map)

        ev = _store_row("c0", 1.0)
        model.update_memory_from_events([ev], embeddings, id_map)
        assert "entity_type" not in ev, "caller's dict must not be mutated"

    def test_delegated_path_receives_resolved_types(self):
        """Mamba/CDE encoders get events whose entity_type is already resolved."""
        data, id_map, metadata = _make_graph()
        model = _make_model(metadata)
        model.eval()
        with torch.no_grad():
            embeddings = model(data, id_map)

        captured: list[list[dict]] = []

        class _StubEncoder:
            def update_memory_from_events(self, events, embeddings, id_map, memory):  # noqa: ARG002
                captured.append(events)

        model.use_mamba = True
        model.mamba_encoder = _StubEncoder()

        stats = model.update_memory_from_events([_store_row("c0", 7.0)], embeddings, id_map)

        assert stats["path"] == "mamba"
        assert len(captured) == 1
        assert captured[0][0]["entity_type"] == "company"


# ═══════════════════════════════════════════════════════════════
# 2. Unresolvable events are counted, not swallowed
# ═══════════════════════════════════════════════════════════════


class TestUnresolvableEventsAreReported:
    def test_unknown_entity_id_is_counted_and_logged(self, caplog):
        data, id_map, metadata = _make_graph()
        model = _make_model(metadata)
        model.eval()
        with torch.no_grad():
            embeddings = model(data, id_map)

        events = [_store_row("ghost_1", 10.0), _store_row("ghost_2", 20.0)]
        with caplog.at_level(logging.ERROR, logger="agent.models.gnn.het_tgn"):
            stats = model.update_memory_from_events(events, embeddings, id_map)

        assert stats["events_in"] == 2
        assert stats["unknown_entity_id"] == 2
        assert stats["resolved"] == 0
        assert stats["applied"] == 0
        assert _nonzero_rows(model) == 0
        assert any("unknown_entity_id=2" in r.getMessage() for r in caplog.records)

    def test_missing_entity_id_is_counted(self):
        data, id_map, metadata = _make_graph()
        model = _make_model(metadata)
        model.eval()
        with torch.no_grad():
            embeddings = model(data, id_map)

        bad = _store_row("c0", 5.0)
        bad["entity_id"] = None
        stats = model.update_memory_from_events([bad, {}], embeddings, id_map)

        assert stats["missing_entity_id"] == 2
        assert stats["applied"] == 0

    def test_ambiguous_entity_id_is_counted_not_guessed(self):
        """An id registered under two types must not be resolved arbitrarily."""
        data, id_map, metadata = _make_graph()
        model = _make_model(metadata)
        model.eval()
        with torch.no_grad():
            embeddings = model(data, id_map)
        # Register the same entity_id under a second type *after* the forward
        # pass so the graph tensors stay consistent.
        id_map.add("country", "c0")

        stats = model.update_memory_from_events([_store_row("c0", 42.0)], embeddings, id_map)

        assert stats["ambiguous_entity_id"] == 1
        assert stats["resolved"] == 0
        assert stats["applied"] == 0
        assert _nonzero_rows(model) == 0

    def test_partial_drop_is_warned_with_a_breakdown(self, caplog):
        data, id_map, metadata = _make_graph()
        model = _make_model(metadata)
        model.eval()
        with torch.no_grad():
            embeddings = model(data, id_map)

        events = [_store_row("c0", 1.0), _store_row("ghost", 2.0)]
        with caplog.at_level(logging.WARNING, logger="agent.models.gnn.het_tgn"):
            stats = model.update_memory_from_events(events, embeddings, id_map)

        assert stats["applied"] == 1
        assert stats["unknown_entity_id"] == 1
        assert any("dropped 1/2" in r.getMessage() for r in caplog.records)

    def test_memory_row_out_of_range_is_counted_not_crashed(self, caplog):
        """A graph larger than the allocated memory buffer is reported loudly."""
        data, id_map, metadata = _make_graph()
        model = _make_model(metadata, num_nodes=1)  # only node 0 has a memory row
        model.eval()
        with torch.no_grad():
            embeddings = model(data, id_map)

        events = [_store_row("c1", 10.0), _store_row("c2", 20.0)]
        with caplog.at_level(logging.ERROR, logger="agent.models.gnn.het_tgn"):
            stats = model.update_memory_from_events(events, embeddings, id_map)

        assert stats["memory_row_out_of_range"] == 2
        assert stats["applied"] == 0
        assert any("applied NOTHING" in r.getMessage() for r in caplog.records)

    def test_type_not_in_embeddings_is_counted(self):
        data, id_map, metadata = _make_graph()
        model = _make_model(metadata)
        model.eval()
        with torch.no_grad():
            embeddings = model(data, id_map)
        embeddings.pop("country")

        stats = model.update_memory_from_events(
            [_store_row("co0", 9.0, "geopolitical_event")],
            embeddings,
            id_map,
        )
        assert stats["type_not_in_embeddings"] == 1
        assert stats["applied"] == 0

    def test_empty_events_report_a_no_op(self):
        data, id_map, metadata = _make_graph()
        model = _make_model(metadata)
        model.eval()
        with torch.no_grad():
            embeddings = model(data, id_map)

        stats = model.update_memory_from_events([], embeddings, id_map)
        assert stats["events_in"] == 0
        assert stats["applied"] == 0
        assert stats["path"] == "none"

    def test_update_memory_rejects_out_of_range_node_ids(self):
        """Writing past the buffer raises with a readable message."""
        mem = HeteroMemory(num_nodes=4, memory_dim=8, message_dim=8, time_dim=4)
        with pytest.raises(ValueError, match=r"outside \[0, 4\)"):
            mem.update_memory(
                torch.tensor([9], dtype=torch.long),
                torch.randn(1, 8),
                torch.tensor([1.0]),
            )


# ═══════════════════════════════════════════════════════════════
# 3. resize() still zero-pads safely
# ═══════════════════════════════════════════════════════════════


class TestMemoryResize:
    def test_resize_preserves_rows_and_zero_pads(self):
        mem = HeteroMemory(num_nodes=3, memory_dim=8, message_dim=8, time_dim=4)
        mem.memory[0] = 1.5
        mem.memory[2] = -2.0
        mem.last_update[2] = 99.0

        mem.resize(6)

        assert mem.num_nodes == 6
        assert mem.memory.shape == (6, 8)
        assert mem.last_update.shape == (6,)
        assert torch.allclose(mem.memory[0], torch.full((8,), 1.5))
        assert torch.allclose(mem.memory[2], torch.full((8,), -2.0))
        assert mem.last_update[2].item() == pytest.approx(99.0)
        assert torch.all(mem.memory[3:] == 0)
        assert torch.all(mem.last_update[3:] == 0)

    def test_resize_is_a_noop_when_not_growing(self):
        mem = HeteroMemory(num_nodes=4, memory_dim=8, message_dim=8, time_dim=4)
        mem.memory[1] = 3.0
        mem.resize(4)
        mem.resize(2)
        assert mem.num_nodes == 4
        assert mem.memory.shape == (4, 8)
        assert torch.allclose(mem.memory[1], torch.full((8,), 3.0))

    def test_resized_buffers_stay_registered(self):
        mem = HeteroMemory(num_nodes=2, memory_dim=8, message_dim=8, time_dim=4)
        mem.resize(5)
        sd = mem.state_dict()
        assert sd["memory"].shape == (5, 8)
        assert sd["last_update"].shape == (5,)

    def test_write_after_resize_lands_in_the_new_rows(self):
        data, id_map, metadata = _make_graph()
        model = _make_model(metadata, num_nodes=2)
        model.memory.resize(id_map.num_nodes)
        model.eval()
        with torch.no_grad():
            embeddings = model(data, id_map)

        stats = model.update_memory_from_events([_store_row("co1", 77.0)], embeddings, id_map)

        gid = id_map.global_id("country", "co1")
        assert gid >= 2, "this entity must live in the newly added rows"
        assert stats["applied"] == 1
        assert stats["memory_row_out_of_range"] == 0
        assert model.memory.last_update[gid].item() == pytest.approx(77.0)
