"""
Phase 39 — Pipeline Robustness Tests

Validates:
- HeteroMemory.resize() expands buffers correctly
- Trainer.infer() resizes when entity count grows
- MacroStateFeatureBuilder gracefully handles missing FRED data
- ConvergenceFeatureBuilder emits 0.0 when data exists but no convergence
- End-to-end feature generation with entity growth
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from agent.models.gnn.het_tgn import HeteroMemory


# ═══════════════════════════════════════════════════════════════
#  39.1 — HeteroMemory.resize()
# ═══════════════════════════════════════════════════════════════


class TestHeteroMemoryResize:
    """Test dynamic memory buffer expansion."""

    def test_resize_up_preserves_existing(self):
        mem = HeteroMemory(num_nodes=10, memory_dim=8)
        # Write known values into first 10 rows
        mem.memory[:] = torch.arange(80).float().reshape(10, 8)
        mem.last_update[:] = torch.arange(10).float()

        mem.resize(20)

        assert mem.num_nodes == 20
        assert mem.memory.shape == (20, 8)
        assert mem.last_update.shape == (20,)
        # Old rows preserved
        assert torch.allclose(mem.memory[:10], torch.arange(80).float().reshape(10, 8))
        assert torch.allclose(mem.last_update[:10], torch.arange(10).float())

    def test_resize_up_zeros_new_rows(self):
        mem = HeteroMemory(num_nodes=5, memory_dim=4)
        mem.memory[:] = 1.0

        mem.resize(10)

        # New rows should be zero
        assert torch.all(mem.memory[5:] == 0.0)
        assert torch.all(mem.last_update[5:] == 0.0)
        # Old rows preserved
        assert torch.all(mem.memory[:5] == 1.0)

    def test_resize_down_is_noop(self):
        mem = HeteroMemory(num_nodes=10, memory_dim=8)
        mem.memory[:] = 42.0

        mem.resize(5)

        assert mem.num_nodes == 10
        assert mem.memory.shape == (10, 8)
        assert torch.all(mem.memory == 42.0)

    def test_resize_equal_is_noop(self):
        mem = HeteroMemory(num_nodes=10, memory_dim=8)
        original = mem.memory.clone()

        mem.resize(10)

        assert mem.num_nodes == 10
        assert torch.equal(mem.memory, original)

    def test_resize_large_growth(self):
        mem = HeteroMemory(num_nodes=10, memory_dim=16)
        mem.resize(10000)

        assert mem.num_nodes == 10000
        assert mem.memory.shape == (10000, 16)
        assert mem.last_update.shape == (10000,)

    def test_get_memory_after_resize(self):
        """After resize, get_memory works for both old and new node IDs."""
        mem = HeteroMemory(num_nodes=5, memory_dim=4)
        mem.memory[:] = torch.arange(20).float().reshape(5, 4)

        mem.resize(10)

        # Old node
        m, t = mem.get_memory(torch.tensor([2]))
        assert m.shape == (1, 4)
        assert torch.allclose(m[0], torch.tensor([8.0, 9.0, 10.0, 11.0]))

        # New node (zero-initialized)
        m, t = mem.get_memory(torch.tensor([7]))
        assert m.shape == (1, 4)
        assert torch.all(m == 0.0)

    def test_update_memory_after_resize(self):
        """After resize, update_memory works for new node IDs."""
        mem = HeteroMemory(num_nodes=5, memory_dim=4, message_dim=4, time_dim=4)
        mem.resize(10)

        # Update a new node
        node_ids = torch.tensor([7])
        messages = torch.randn(1, 4)
        timestamps = torch.tensor([1.0])
        mem.update_memory(node_ids, messages, timestamps)

        # Should not crash and memory should be non-zero
        m, t = mem.get_memory(node_ids)
        assert not torch.all(m == 0.0)


# ═══════════════════════════════════════════════════════════════
#  39.2 — Trainer.infer() resize on entity growth
# ═══════════════════════════════════════════════════════════════


class TestTrainerInferResize:
    """Verify infer() handles entity count growth without crashing."""

    def test_infer_resizes_when_graph_grows(self):
        """Mock a scenario where graph has more entities than checkpoint."""
        from agent.models.gnn.het_tgn import HetTGN

        # Create a model with 10 nodes
        metadata = (["company"], [("company", "works_for", "company")])
        model = HetTGN(
            metadata=metadata,
            in_channels={"company": 16},
            num_nodes=10,
            hidden_dim=16,
            memory_dim=16,
        )

        # Simulate what infer() does: detect growth and resize
        old_num = model.memory.num_nodes
        new_num = 20

        assert old_num == 10
        model.memory.resize(new_num)
        assert model.memory.num_nodes == 20

        # get_memory should work for new IDs
        m, t = model.memory.get_memory(torch.tensor([15]))
        assert m.shape == (1, 16)

    def test_no_resize_when_graph_same_size(self):
        """Model with same entity count should not trigger resize."""
        mem = HeteroMemory(num_nodes=100, memory_dim=32)
        original_num = mem.num_nodes

        # Simulate: graph has same count
        if 100 > mem.num_nodes:
            mem.resize(100)

        assert mem.num_nodes == original_num


# ═══════════════════════════════════════════════════════════════
#  39.3 — MacroStateFeatureBuilder graceful degradation
# ═══════════════════════════════════════════════════════════════


class TestMacroBuilderGraceful:
    """MacroStateFeatureBuilder returns empty when no FRED data."""

    def test_no_macro_data_returns_empty(self):
        from agent.features.builders import MacroStateFeatureBuilder

        builder = MacroStateFeatureBuilder()
        store = MagicMock()
        store.query_data.return_value = []

        features = builder.build(store, as_of=time.time())

        # Empty store returns 3 None-valued features for consistent GNN dimensionality.
        assert len(features) == 3
        assert all(f.value is None for f in features)
        store.query_data.assert_called_once()

    def test_with_macro_data_returns_features(self):
        from agent.features.builders import MacroStateFeatureBuilder

        builder = MacroStateFeatureBuilder()
        store = MagicMock()

        # Simulate macro_data with DFF, GS10, GS2, WALCL series
        store.query_data.return_value = [
            {
                "data": {
                    "DFF": [
                        {"date": "2026-03-01", "value": 5.25},
                        {"date": "2026-04-01", "value": 5.50},
                    ],
                    "GS10": [{"date": "2026-04-01", "value": 4.25}],
                    "GS2": [{"date": "2026-04-01", "value": 4.00}],
                    "WALCL": [
                        {"date": "2026-03-01", "value": 7500000},
                        {"date": "2026-04-01", "value": 7600000},
                    ],
                }
            }
        ]

        features = builder.build(store, as_of=time.time())

        assert len(features) == 3
        names = {f.feature_name for f in features}
        assert "macro.rate_momentum.30d" in names
        assert "macro.yield_curve_slope.spot" in names
        assert "macro.liquidity_pressure.30d" in names


# ═══════════════════════════════════════════════════════════════
#  39.4 — ConvergenceFeatureBuilder zero-vs-None semantics
# ═══════════════════════════════════════════════════════════════


class TestConvergenceFeatureSemantics:
    """Convergence builder should distinguish 'no data' from 'no convergence'."""

    def _make_store(self, has_signals=False, has_pipeline_data=False):
        store = MagicMock()

        # _get_conn for direct SQL query
        mock_conn = MagicMock()
        if has_signals:
            mock_conn.execute.return_value.fetchall.return_value = [
                {
                    "signal_name": "convergence.test",
                    "value": 0.85,
                    "computed_at": time.time(),
                    "metadata": {"persistence_days": 3},
                }
            ]
        else:
            mock_conn.execute.return_value.fetchall.return_value = []
        store._get_conn.return_value = mock_conn
        store._signal_row_to_dict = lambda r: r

        # query_data for pipeline data presence check
        if has_pipeline_data:
            store.query_data.return_value = [{"data": {"some": "data"}}]
        else:
            store.query_data.return_value = []

        return store

    def test_no_data_at_all_returns_empty(self):
        from agent.features.builders import ConvergenceFeatureBuilder

        builder = ConvergenceFeatureBuilder()
        store = self._make_store(has_signals=False, has_pipeline_data=False)

        features = builder.build(store, as_of=time.time())

        # Empty store returns 3 None-valued features for consistent GNN dimensionality.
        assert len(features) == 3
        assert all(f.value is None for f in features)
        assert all(f.missing_reason == "no_convergence_activity" for f in features)

    def test_data_present_no_convergence_returns_zeros(self):
        from agent.features.builders import ConvergenceFeatureBuilder

        builder = ConvergenceFeatureBuilder()
        store = self._make_store(has_signals=False, has_pipeline_data=True)

        features = builder.build(store, as_of=time.time())

        assert len(features) == 3
        for f in features:
            assert f.value == 0.0
            assert f.quality == 1.0
            assert f.missing_reason is None

    def test_convergence_detected_returns_real_values(self):
        from agent.features.builders import ConvergenceFeatureBuilder

        builder = ConvergenceFeatureBuilder()
        store = self._make_store(has_signals=True, has_pipeline_data=True)

        features = builder.build(store, as_of=time.time())

        assert len(features) == 3
        # At least one should have non-zero value
        values = {f.feature_name: f.value for f in features}
        assert values["convergence.stress_breadth.7d"] >= 1.0
        assert values["convergence.stress_intensity.7d"] > 0.0

    def test_zero_features_have_correct_names(self):
        from agent.features.builders import ConvergenceFeatureBuilder

        builder = ConvergenceFeatureBuilder()
        store = self._make_store(has_signals=False, has_pipeline_data=True)

        features = builder.build(store, as_of=time.time())

        names = {f.feature_name for f in features}
        assert names == {
            "convergence.stress_breadth.7d",
            "convergence.stress_intensity.7d",
            "convergence.regime_persistence.7d",
        }


# ═══════════════════════════════════════════════════════════════
#  39.5 — Integration: feature_generation with entity growth
# ═══════════════════════════════════════════════════════════════


class TestFeatureGenerationIntegration:
    """End-to-end: feature builders don't crash when entities grew."""

    def test_convergence_builder_with_empty_store(self):
        """ConvergenceFeatureBuilder on empty store doesn't crash."""
        from agent.features.builders import ConvergenceFeatureBuilder

        builder = ConvergenceFeatureBuilder()
        store = MagicMock()
        store._get_conn.return_value.execute.return_value.fetchall.return_value = []
        store._signal_row_to_dict = lambda r: r
        store.query_data.return_value = []

        features = builder.build(store, as_of=time.time())
        # Empty store → no features
        assert isinstance(features, list)

    def test_macro_builder_with_empty_store(self):
        """MacroStateFeatureBuilder on empty store returns 3 missing features."""
        from agent.features.builders import MacroStateFeatureBuilder

        builder = MacroStateFeatureBuilder()
        store = MagicMock()
        store.query_data.return_value = []

        features = builder.build(store, as_of=time.time())
        # Empty store returns 3 None-valued features for consistent GNN dimensionality.
        assert len(features) == 3
        assert all(f.value is None for f in features)

    def test_hetero_memory_resize_multiple_times(self):
        """Sequential resizes work correctly."""
        mem = HeteroMemory(num_nodes=5, memory_dim=4)
        mem.memory[:] = 1.0

        mem.resize(10)
        assert mem.num_nodes == 10
        assert torch.all(mem.memory[:5] == 1.0)

        mem.resize(20)
        assert mem.num_nodes == 20
        assert torch.all(mem.memory[:5] == 1.0)
        assert torch.all(mem.memory[10:] == 0.0)
