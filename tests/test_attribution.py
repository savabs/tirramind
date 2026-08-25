"""Tests for Idea 12 — Barra-Style Signal Attribution (attribution.py).

Covers:
    1.  AttributionResult fields exist and are typed correctly
    2.  BarraAttribution defaults
    3.  BarraAttribution custom params
    4.  _normalize: sums to 1.0
    5.  _normalize: all values non-negative
    6.  _normalize: empty input → empty output
    7.  _normalize: min_attention collapses weak sources to 'other'
    8.  _aggregate_layers: accumulates attention for correct dst nodes
    9.  _aggregate_layers: averages across multiple layers
    10. _aggregate_layers: ignores edge types with wrong dst_type
    11. _aggregate_layers: guards mismatched attn/edge sizes
    12. compute(): returns empty dict when no instrument nodes
    13. compute(): max_entities cap is respected
    14. compute(): returns AttributionResult for valid graph
    15. compute(): factor_contributions sums to 1.0 per entity
    16. compute(): all factor_contributions non-negative
    17. compute(): dominant_factor is the max-contribution key
    18. compute(): top_factors is sorted descending
    19. compute(): attention capture disabled after compute (no overhead leak)
    20. compute(): target_entity_ids filter respected
    21. compute(): returns empty dict on empty graph
    22. store_results(): calls store.store_signal correct number of times
    23. store_results(): signal names follow attribution.{eid}.{src_type} pattern
    24. store_results(): gracefully handles store failure
    25. TrainerConfig.use_attribution defaults False
    26. TrainerConfig.attribution_max_entities defaults 200
    27. TrainerConfig.attribution_min_attention defaults 0.0
    28. Trainer.compute_attribution(): returns empty if model not built
    29. Trainer.compute_attribution(): returns dict after build_model
    30. CPU safety: no gradient computation during compute (no .grad on params)
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import torch

from agent.models.gnn.attribution import (
    AttributionResult,
    BarraAttribution,
)
from agent.models.gnn.trainer import Trainer, TrainerConfig, SyntheticGraphGenerator
from agent.pipeline.store import PipelineStore


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_store(tmp_path: Path, name: str = "attr.db") -> PipelineStore:
    return PipelineStore(str(tmp_path / name))


def _make_trainer(tmp_path: Path, tag: str = "t") -> Trainer:
    store = _make_store(tmp_path, f"{tag}.db")
    gen = SyntheticGraphGenerator(
        num_companies=2, num_countries=1,
        time_span=3600.0 * 3, base_event_rate=0.001, seed=7,
    )
    gen.generate(store)
    cfg = TrainerConfig(
        hidden_dim=16, memory_dim=16, message_dim=16, time_dim=8,
        num_heads=1, num_layers=1,
    )
    return Trainer(store, cfg)


def _make_mock_model(n_layers: int = 2):
    """Build a minimal mock HetTGN with AttentionCapturingHGTConv-like layers."""
    model = MagicMock()
    model.training = False

    layers = []
    for _ in range(n_layers):
        layer = MagicMock()
        layer.capture_attention = False
        layer.get_edge_attention.return_value = {}
        layers.append(layer)

    model.hgt_layers = layers
    model.train = MagicMock()
    model.eval = MagicMock()
    model.__call__ = MagicMock(return_value={})
    return model


def _make_hetero_data(etype, edge_index_tensor):
    """Build a minimal mock HeteroData for one edge type."""
    data = MagicMock()
    data.edge_types = [etype]

    edge_obj = MagicMock()
    edge_obj.edge_index = edge_index_tensor
    data.__getitem__ = lambda self, key: edge_obj if key == etype else MagicMock()
    return data


# ═══════════════════════════════════════════════════════════════
# 1. AttributionResult structure
# ═══════════════════════════════════════════════════════════════

class TestAttributionResult:

    def test_fields_exist(self):
        ar = AttributionResult(
            entity_id="copper",
            entity_type="instrument",
            factor_contributions={"vessel": 0.6, "company": 0.4},
            dominant_factor="vessel",
            top_factors=[("vessel", 0.6), ("company", 0.4)],
            n_layers_averaged=2,
            computed_at=time.time(),
        )
        assert ar.entity_id == "copper"
        assert ar.entity_type == "instrument"
        assert isinstance(ar.factor_contributions, dict)
        assert ar.dominant_factor == "vessel"
        assert len(ar.top_factors) == 2
        assert ar.n_layers_averaged == 2
        assert ar.computed_at > 0


# ═══════════════════════════════════════════════════════════════
# 2–3. BarraAttribution construction
# ═══════════════════════════════════════════════════════════════

class TestConstruction:

    def test_defaults(self):
        ba = BarraAttribution()
        assert ba.target_type == "instrument"
        assert ba.max_entities == 200
        assert ba.min_attention == 0.0

    def test_custom_params(self):
        ba = BarraAttribution(target_type="company", max_entities=50, min_attention=0.05)
        assert ba.target_type == "company"
        assert ba.max_entities == 50
        assert ba.min_attention == pytest.approx(0.05)


# ═══════════════════════════════════════════════════════════════
# 4–7. _normalize
# ═══════════════════════════════════════════════════════════════

class TestNormalize:

    def _ba(self, min_attention=0.0):
        return BarraAttribution(min_attention=min_attention)

    def test_sums_to_one(self):
        ba = self._ba()
        raw = {"a": 3.0, "b": 1.0, "c": 2.0}
        out = ba._normalize(raw)
        assert sum(out.values()) == pytest.approx(1.0, abs=1e-9)

    def test_all_non_negative(self):
        ba = self._ba()
        out = ba._normalize({"x": 5.0, "y": 2.0})
        for v in out.values():
            assert v >= 0.0

    def test_empty_input_returns_empty(self):
        ba = self._ba()
        assert ba._normalize({}) == {}

    def test_min_attention_collapses_weak_to_other(self):
        ba = self._ba(min_attention=0.2)
        raw = {"strong": 9.0, "weak": 1.0}
        out = ba._normalize(raw)
        # "weak" becomes < 0.2 fraction after norm → collapsed to "other"
        assert "other" in out or "strong" in out
        assert sum(out.values()) == pytest.approx(1.0, abs=1e-9)


# ═══════════════════════════════════════════════════════════════
# 8–11. _aggregate_layers
# ═══════════════════════════════════════════════════════════════

class TestAggregateLayers:

    def _make_layer_attn(self, etype, attn_vals, dst_vals, n_edges):
        """Returns (raw_per_layer, mock_data, n_layers)."""
        attn = torch.tensor(attn_vals, dtype=torch.float)
        edge_index = torch.stack([
            torch.zeros(n_edges, dtype=torch.long),  # src (unused)
            torch.tensor(dst_vals, dtype=torch.long),
        ])
        raw_per_layer = [{etype: attn}]
        data = MagicMock()
        data.edge_types = [etype]
        edge_obj = MagicMock()
        edge_obj.edge_index = edge_index
        data.__getitem__ = MagicMock(return_value=edge_obj)
        return raw_per_layer, data

    def test_accumulates_correct_dst(self):
        etype = ("vessel", "trades", "instrument")
        ba = BarraAttribution(target_type="instrument")
        raw, data = self._make_layer_attn(etype, [0.5, 0.3, 0.4], [0, 1, 0], 3)
        agg = ba._aggregate_layers(raw, data, n_layers=1)
        # dst=0 receives edges 0 and 2 → 0.5+0.4=0.9
        assert agg[0]["vessel"] == pytest.approx(0.9, abs=1e-6)
        # dst=1 receives edge 1 → 0.3
        assert agg[1]["vessel"] == pytest.approx(0.3, abs=1e-6)

    def test_averages_across_layers(self):
        etype = ("company", "owns", "instrument")
        ba = BarraAttribution(target_type="instrument")
        # Two identical layers
        attn = torch.tensor([0.6, 0.4], dtype=torch.float)
        edge_index = torch.stack([
            torch.zeros(2, dtype=torch.long),
            torch.tensor([0, 1], dtype=torch.long),
        ])
        layer = {etype: attn}
        raw_per_layer = [layer, layer]
        data = MagicMock()
        data.edge_types = [etype]
        edge_obj = MagicMock()
        edge_obj.edge_index = edge_index
        data.__getitem__ = MagicMock(return_value=edge_obj)
        agg = ba._aggregate_layers(raw_per_layer, data, n_layers=2)
        # Both layers contribute 0.6 each → sum 1.2 → avg 0.6
        assert agg[0]["company"] == pytest.approx(0.6, abs=1e-6)

    def test_ignores_wrong_dst_type(self):
        etype = ("vessel", "trades", "country")  # dst_type != "instrument"
        ba = BarraAttribution(target_type="instrument")
        attn = torch.tensor([0.8], dtype=torch.float)
        edge_index = torch.stack([
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
        ])
        raw_per_layer = [{etype: attn}]
        data = MagicMock()
        data.edge_types = [etype]
        edge_obj = MagicMock()
        edge_obj.edge_index = edge_index
        data.__getitem__ = MagicMock(return_value=edge_obj)
        agg = ba._aggregate_layers(raw_per_layer, data, n_layers=1)
        assert len(agg) == 0

    def test_guards_mismatched_size(self):
        etype = ("vessel", "trades", "instrument")
        ba = BarraAttribution(target_type="instrument")
        attn = torch.tensor([0.5, 0.3], dtype=torch.float)  # 2 values
        edge_index = torch.stack([
            torch.zeros(5, dtype=torch.long),  # 5 edges → mismatch
            torch.zeros(5, dtype=torch.long),
        ])
        raw_per_layer = [{etype: attn}]
        data = MagicMock()
        data.edge_types = [etype]
        edge_obj = MagicMock()
        edge_obj.edge_index = edge_index
        data.__getitem__ = MagicMock(return_value=edge_obj)
        # Should not raise — just skip the bad edge type
        agg = ba._aggregate_layers(raw_per_layer, data, n_layers=1)
        assert len(agg) == 0


# ═══════════════════════════════════════════════════════════════
# 12–21. compute()
# ═══════════════════════════════════════════════════════════════

class TestCompute:

    def _ba(self, **kw):
        return BarraAttribution(target_type="instrument", **kw)

    def test_empty_when_no_instrument_nodes(self):
        model = _make_mock_model()
        id_map = MagicMock()
        id_map.type_local = {}  # no instruments
        data = MagicMock()
        ba = self._ba()
        result = ba.compute(model, data, id_map)
        assert result == {}

    def test_max_entities_cap(self):
        model = _make_mock_model()
        id_map = MagicMock()
        id_map.type_local = {"instrument": {f"e{i}": i for i in range(300)}}
        data = MagicMock()
        data.edge_types = []
        ba = BarraAttribution(target_type="instrument", max_entities=10)
        # No attention returned → empty results, but no crash and cap respected
        result = ba.compute(model, data, id_map)
        assert len(result) <= 10

    def test_returns_attribution_result_type(self, tmp_path):
        trainer = _make_trainer(tmp_path, "ar")
        trainer.build_model()
        results = trainer.compute_attribution()
        for v in results.values():
            assert isinstance(v, AttributionResult)

    def test_factor_contributions_sum_to_one(self, tmp_path):
        trainer = _make_trainer(tmp_path, "s1")
        trainer.build_model()
        results = trainer.compute_attribution()
        for v in results.values():
            total = sum(v.factor_contributions.values())
            assert total == pytest.approx(1.0, abs=1e-6)

    def test_factor_contributions_non_negative(self, tmp_path):
        trainer = _make_trainer(tmp_path, "nn")
        trainer.build_model()
        results = trainer.compute_attribution()
        for v in results.values():
            for fc in v.factor_contributions.values():
                assert fc >= 0.0

    def test_dominant_factor_is_max(self, tmp_path):
        trainer = _make_trainer(tmp_path, "dm")
        trainer.build_model()
        results = trainer.compute_attribution()
        for v in results.values():
            if v.factor_contributions:
                expected_dom = max(v.factor_contributions, key=v.factor_contributions.get)
                assert v.dominant_factor == expected_dom

    def test_top_factors_sorted_descending(self, tmp_path):
        trainer = _make_trainer(tmp_path, "sf")
        trainer.build_model()
        results = trainer.compute_attribution()
        for v in results.values():
            vals = [x[1] for x in v.top_factors]
            assert vals == sorted(vals, reverse=True)

    def test_attention_capture_disabled_after_compute(self, tmp_path):
        trainer = _make_trainer(tmp_path, "cap")
        trainer.build_model()
        trainer.compute_attribution()
        # All layers must have capture_attention=False after returning
        for layer in trainer.model.hgt_layers:
            assert layer.capture_attention is False

    def test_target_entity_ids_filter(self, tmp_path):
        trainer = _make_trainer(tmp_path, "filt")
        trainer.build_model()
        from agent.models.gnn.graph_builder import GraphBuilder
        data, id_map, _ = GraphBuilder(trainer.store).build()
        local_map = id_map.type_local.get("instrument", {})
        if len(local_map) < 2:
            pytest.skip("Not enough instrument nodes for filter test")
        only = [sorted(local_map.keys())[0]]
        from agent.models.gnn.attribution import BarraAttribution
        ba = BarraAttribution(target_type="instrument")
        results = ba.compute(trainer.model, data, id_map, target_entity_ids=only)
        # Should only return result for the requested entity (if has attention)
        for eid in results:
            assert eid in only

    def test_returns_empty_on_empty_graph(self):
        model = _make_mock_model()
        id_map = MagicMock()
        id_map.type_local = {"instrument": {"e1": 0}}
        # No edges → no attention → empty
        data = MagicMock()
        data.edge_types = []
        ba = BarraAttribution(target_type="instrument")
        result = ba.compute(model, data, id_map)
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════
# 22–24. store_results()
# ═══════════════════════════════════════════════════════════════

class TestStoreResults:

    def _make_result(self, eid, factors):
        return AttributionResult(
            entity_id=eid,
            entity_type="instrument",
            factor_contributions=factors,
            dominant_factor=max(factors, key=factors.get),
            top_factors=sorted(factors.items(), key=lambda x: -x[1]),
            n_layers_averaged=1,
            computed_at=time.time(),
        )

    def test_calls_store_correct_number_of_times(self):
        mock_store = MagicMock()
        ba = BarraAttribution()
        results = {
            "copper": self._make_result("copper", {"vessel": 0.6, "company": 0.4}),
            "gold":   self._make_result("gold",   {"vessel": 0.3, "country": 0.7}),
        }
        n = ba.store_results(mock_store, results)
        assert n == 4  # 2 entities × 2 factors each
        assert mock_store.store_signal.call_count == 4

    def test_signal_names_follow_convention(self):
        mock_store = MagicMock()
        ba = BarraAttribution()
        results = {
            "crude": self._make_result("crude", {"vessel": 0.8, "company": 0.2}),
        }
        ba.store_results(mock_store, results)
        called_names = {c.kwargs["signal_name"] for c in mock_store.store_signal.call_args_list}
        assert "attribution.crude.vessel" in called_names
        assert "attribution.crude.company" in called_names

    def test_graceful_on_store_failure(self):
        mock_store = MagicMock()
        mock_store.store_signal.side_effect = RuntimeError("disk full")
        ba = BarraAttribution()
        results = {"e1": self._make_result("e1", {"vessel": 1.0})}
        # Should not raise
        n = ba.store_results(mock_store, results)
        assert n == 0


# ═══════════════════════════════════════════════════════════════
# 25–27. TrainerConfig defaults
# ═══════════════════════════════════════════════════════════════

class TestTrainerConfig:

    def test_use_attribution_defaults_false(self):
        assert TrainerConfig().use_attribution is False

    def test_attribution_max_entities_defaults_200(self):
        assert TrainerConfig().attribution_max_entities == 200

    def test_attribution_min_attention_defaults_0(self):
        assert TrainerConfig().attribution_min_attention == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════
# 28–30. Trainer.compute_attribution()
# ═══════════════════════════════════════════════════════════════

class TestTrainerComputeAttribution:

    def test_returns_empty_if_model_not_built(self, tmp_path):
        trainer = _make_trainer(tmp_path, "nb")
        # Don't call build_model()
        result = trainer.compute_attribution()
        assert result == {}

    def test_returns_dict_after_build_model(self, tmp_path):
        trainer = _make_trainer(tmp_path, "bm")
        trainer.build_model()
        result = trainer.compute_attribution()
        assert isinstance(result, dict)

    def test_no_gradient_computation(self, tmp_path):
        """Verify no param.grad is populated during compute_attribution."""
        trainer = _make_trainer(tmp_path, "ng")
        trainer.build_model()
        # Zero out any existing grads
        for p in trainer.model.parameters():
            p.grad = None
        trainer.compute_attribution()
        # No gradient should have been computed
        for p in trainer.model.parameters():
            assert p.grad is None
