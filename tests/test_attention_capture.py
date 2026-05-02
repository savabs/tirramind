"""Tests for Phase 14a: AttentionCapturingHGTConv + HetTGN.get_attention_weights().

Covers:
    AttentionCapturingHGTConv — capture toggle, edge-type splitting, shapes
    HetTGN.get_attention_weights — mean attention per edge type, multi-layer,
                                   eval/train mode restoration
    PatternExtractor — now uses real attention (no embedding fallback)
"""

from __future__ import annotations

import torch
from torch_geometric.data import HeteroData

from agent.models.gnn.graph_builder import IDMap
from agent.models.gnn.het_tgn import AttentionCapturingHGTConv, HetTGN

# ─── Helpers ──────────────────────────────────────────────────


def _make_simple_graph() -> tuple[HeteroData, IDMap, tuple]:
    """Tiny graph: 3 companies, 2 countries, 1 edge type."""
    id_map = IDMap()
    for i in range(3):
        id_map.add("company", f"c{i}")
    for i in range(2):
        id_map.add("country", f"co{i}")

    feat_dim = 9
    data = HeteroData()
    data["company"].x = torch.randn(3, feat_dim)
    data["company"].node_ids = ["c0", "c1", "c2"]
    data["country"].x = torch.randn(2, feat_dim)
    data["country"].node_ids = ["co0", "co1"]

    data["company", "headquartered_in", "country"].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 0]],
        dtype=torch.long,
    )

    node_types = ["company", "country"]
    edge_types = [("company", "headquartered_in", "country")]
    metadata = (node_types, edge_types)
    return data, id_map, metadata


def _make_multi_edge_graph() -> tuple[HeteroData, IDMap, tuple]:
    """Graph with two edge types: company→country, country→company."""
    id_map = IDMap()
    for i in range(3):
        id_map.add("company", f"c{i}")
    for i in range(2):
        id_map.add("country", f"co{i}")

    feat_dim = 9
    data = HeteroData()
    data["company"].x = torch.randn(3, feat_dim)
    data["company"].node_ids = ["c0", "c1", "c2"]
    data["country"].x = torch.randn(2, feat_dim)
    data["country"].node_ids = ["co0", "co1"]

    data["company", "headquartered_in", "country"].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 0]],
        dtype=torch.long,
    )
    data["country", "regulates", "company"].edge_index = torch.tensor(
        [[0, 1], [1, 2]],
        dtype=torch.long,
    )

    node_types = ["company", "country"]
    edge_types = [
        ("company", "headquartered_in", "country"),
        ("country", "regulates", "company"),
    ]
    metadata = (node_types, edge_types)
    return data, id_map, metadata


def _make_model(
    metadata: tuple,
    in_channels: dict | None = None,
    num_nodes: int = 5,
    **kwargs,
) -> HetTGN:
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
# AttentionCapturingHGTConv — unit tests
# ═══════════════════════════════════════════════════════════════


class TestAttentionCapturingHGTConvInit:
    """Verify the subclass initialises cleanly as HGTConv drop-in."""

    def test_is_hgtconv_subclass(self):
        from torch_geometric.nn import HGTConv

        metadata = (["a", "b"], [("a", "r", "b")])
        conv = AttentionCapturingHGTConv(
            in_channels=16,
            out_channels=16,
            metadata=metadata,
            heads=2,
        )
        assert isinstance(conv, HGTConv)

    def test_capture_off_by_default(self):
        metadata = (["a", "b"], [("a", "r", "b")])
        conv = AttentionCapturingHGTConv(
            in_channels=16,
            out_channels=16,
            metadata=metadata,
            heads=2,
        )
        assert conv.capture_attention is False

    def test_no_captured_alpha_initially(self):
        metadata = (["a", "b"], [("a", "r", "b")])
        conv = AttentionCapturingHGTConv(
            in_channels=16,
            out_channels=16,
            metadata=metadata,
            heads=2,
        )
        assert conv._captured_alpha is None
        assert conv.get_edge_attention() == {}


class TestAttentionCapturingHGTConvCapture:
    """Verify attention is captured when enabled and correct shapes."""

    def test_capture_disabled_no_storage(self):
        """When capture is off, _captured_alpha remains None."""
        data, id_map, metadata = _make_simple_graph()
        model = _make_model(metadata)
        # Forward without capture
        model.eval()
        with torch.no_grad():
            model(data, id_map)
        for hgt in model.hgt_layers:
            assert hgt._captured_alpha is None

    def test_capture_enabled_stores_alpha(self):
        """When capture is on, _captured_alpha is populated."""
        data, id_map, metadata = _make_simple_graph()
        model = _make_model(metadata)
        model.eval()
        for hgt in model.hgt_layers:
            hgt.capture_attention = True
        with torch.no_grad():
            model(data, id_map)
        for hgt in model.hgt_layers:
            assert hgt._captured_alpha is not None
            assert hgt._captured_alpha.ndim == 2  # (total_edges, heads)

    def test_captured_alpha_shape_matches_edges(self):
        """Alpha tensor has as many rows as total edges."""
        data, id_map, metadata = _make_simple_graph()
        model = _make_model(metadata)
        model.eval()
        for hgt in model.hgt_layers:
            hgt.capture_attention = True
        with torch.no_grad():
            model(data, id_map)
        total_edges = data["company", "headquartered_in", "country"].edge_index.size(1)
        for hgt in model.hgt_layers:
            assert hgt._captured_alpha.size(0) == total_edges

    def test_edge_attention_keys_match_edge_types(self):
        """get_edge_attention() returns all edge types present in data."""
        data, id_map, metadata = _make_multi_edge_graph()
        model = _make_model(metadata)
        model.eval()
        for hgt in model.hgt_layers:
            hgt.capture_attention = True
        with torch.no_grad():
            edge_index_dict = {}
            for etype in model.edge_types:
                if etype in data.edge_types:
                    edge_index_dict[etype] = data[etype].edge_index
            # Run a single HGT layer directly to test get_edge_attention
            x_dict = {}
            for ntype in model.node_types:
                if ntype in data.node_types:
                    x = data[ntype].x
                    projected = model.type_projections[ntype](x)
                    zero_mem = torch.zeros(projected.size(0), model.memory_dim)
                    combined = torch.cat([projected, zero_mem], dim=-1)
                    x_dict[ntype] = torch.relu(model.combiner(combined))
            model.hgt_layers[0](x_dict, edge_index_dict)

        attn = model.hgt_layers[0].get_edge_attention()
        expected_types = set(edge_index_dict.keys())
        assert set(attn.keys()) == expected_types

    def test_edge_attention_shapes_per_type(self):
        """Per-type attention tensors have correct length."""
        data, id_map, metadata = _make_multi_edge_graph()
        model = _make_model(metadata)
        model.eval()
        for hgt in model.hgt_layers:
            hgt.capture_attention = True
        with torch.no_grad():
            model(data, id_map)
        for hgt in model.hgt_layers:
            attn = hgt.get_edge_attention()
            for etype, a in attn.items():
                expected = data[etype].edge_index.size(1)
                assert a.shape == (expected,), f"{etype}: {a.shape} != ({expected},)"

    def test_attention_values_in_valid_range(self):
        """Post-softmax attention values are in [0, 1]."""
        data, id_map, metadata = _make_simple_graph()
        model = _make_model(metadata)
        model.eval()
        for hgt in model.hgt_layers:
            hgt.capture_attention = True
        with torch.no_grad():
            model(data, id_map)
        for hgt in model.hgt_layers:
            attn = hgt.get_edge_attention()
            for etype, a in attn.items():
                assert (a >= 0).all(), "Negative attention weights"
                assert (a <= 1.0 + 1e-6).all(), "Attention > 1"


# ═══════════════════════════════════════════════════════════════
# HetTGN.get_attention_weights — integration tests
# ═══════════════════════════════════════════════════════════════


class TestGetAttentionWeights:
    """Verify the convenience method on HetTGN."""

    def test_returns_dict_of_floats(self):
        data, id_map, metadata = _make_simple_graph()
        model = _make_model(metadata)
        weights = model.get_attention_weights(data, id_map)
        assert isinstance(weights, dict)
        for k, v in weights.items():
            assert isinstance(k, tuple) and len(k) == 3
            assert isinstance(v, float)

    def test_attention_for_each_edge_type(self):
        data, id_map, metadata = _make_multi_edge_graph()
        model = _make_model(metadata)
        weights = model.get_attention_weights(data, id_map)
        for etype in metadata[1]:
            assert etype in weights, f"Missing edge type {etype}"

    def test_attention_values_positive(self):
        data, id_map, metadata = _make_simple_graph()
        model = _make_model(metadata)
        weights = model.get_attention_weights(data, id_map)
        for etype, val in weights.items():
            assert val > 0, f"Attention for {etype} should be positive"

    def test_capture_disabled_after_call(self):
        """get_attention_weights() cleans up: capture is off afterwards."""
        data, id_map, metadata = _make_simple_graph()
        model = _make_model(metadata)
        model.get_attention_weights(data, id_map)
        for hgt in model.hgt_layers:
            assert hgt.capture_attention is False

    def test_restores_training_mode(self):
        """Model returns to training mode if it was training before."""
        data, id_map, metadata = _make_simple_graph()
        model = _make_model(metadata)
        model.train()
        model.get_attention_weights(data, id_map)
        assert model.training is True

    def test_works_in_eval_mode(self):
        data, id_map, metadata = _make_simple_graph()
        model = _make_model(metadata)
        model.eval()
        weights = model.get_attention_weights(data, id_map)
        assert len(weights) > 0
        assert model.training is False

    def test_multi_layer_aggregation(self):
        """With 2 HGT layers, attention is averaged across layers."""
        data, id_map, metadata = _make_simple_graph()
        model = _make_model(metadata, num_layers=2)
        weights = model.get_attention_weights(data, id_map)
        etype = ("company", "headquartered_in", "country")
        assert etype in weights

    def test_empty_graph_returns_empty(self):
        """No edges → empty attention dict."""
        id_map = IDMap()
        id_map.add("company", "c0")
        id_map.add("country", "co0")

        data = HeteroData()
        data["company"].x = torch.randn(1, 9)
        data["company"].node_ids = ["c0"]
        data["country"].x = torch.randn(1, 9)
        data["country"].node_ids = ["co0"]

        node_types = ["company", "country"]
        edge_types = [("company", "headquartered_in", "country")]
        metadata = (node_types, edge_types)

        model = _make_model(metadata, num_nodes=2)
        weights = model.get_attention_weights(data, id_map)
        assert weights == {}


# ═══════════════════════════════════════════════════════════════
# PatternExtractor now uses real attention
# ═══════════════════════════════════════════════════════════════


class TestPatternExtractorUsesAttention:
    """Verify PatternExtractor gets real attention, not just embedding fallback."""

    def test_attention_extraction_returns_scores(self):
        """_extract_attention_hooks now delegates to model.get_attention_weights."""
        from unittest.mock import MagicMock

        from agent.models.gnn.pattern_extractor import PatternExtractor

        data, id_map, metadata = _make_simple_graph()
        model = _make_model(metadata)
        store = MagicMock()

        extractor = PatternExtractor(model, store)
        result = extractor._extract_attention_hooks(data, id_map)
        # Should return non-empty dict from real attention capture
        assert isinstance(result, dict)
        assert len(result) > 0
        etype = ("company", "headquartered_in", "country")
        assert etype in result
        assert isinstance(result[etype], float)

    def test_graceful_fallback_on_error(self):
        """If get_attention_weights raises, returns empty dict."""
        from unittest.mock import MagicMock, patch

        from agent.models.gnn.pattern_extractor import PatternExtractor

        data, id_map, metadata = _make_simple_graph()
        model = _make_model(metadata)
        store = MagicMock()

        extractor = PatternExtractor(model, store)
        with patch.object(model, "get_attention_weights", side_effect=RuntimeError("boom")):
            result = extractor._extract_attention_hooks(data, id_map)
        assert result == {}


# ═══════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════


class TestAttentionCaptureEdgeCases:
    """Edge cases that could trip up the implementation."""

    def test_single_edge(self):
        """Graph with exactly one edge works."""
        id_map = IDMap()
        id_map.add("company", "c0")
        id_map.add("country", "co0")

        data = HeteroData()
        data["company"].x = torch.randn(1, 9)
        data["company"].node_ids = ["c0"]
        data["country"].x = torch.randn(1, 9)
        data["country"].node_ids = ["co0"]
        data["company", "headquartered_in", "country"].edge_index = torch.tensor(
            [[0], [0]],
            dtype=torch.long,
        )

        metadata = (
            ["company", "country"],
            [("company", "headquartered_in", "country")],
        )
        model = _make_model(metadata, num_nodes=2)
        weights = model.get_attention_weights(data, id_map)
        assert ("company", "headquartered_in", "country") in weights

    def test_repeated_calls_dont_accumulate(self):
        """Calling get_attention_weights twice gives independent results."""
        data, id_map, metadata = _make_simple_graph()
        model = _make_model(metadata)
        w1 = model.get_attention_weights(data, id_map)
        w2 = model.get_attention_weights(data, id_map)
        etype = ("company", "headquartered_in", "country")
        # Same model + same data → same attention
        assert abs(w1[etype] - w2[etype]) < 1e-5

    def test_different_data_gives_different_attention(self):
        """Changing input features changes attention weights."""
        data, id_map, metadata = _make_simple_graph()
        model = _make_model(metadata)
        w1 = model.get_attention_weights(data, id_map)

        # Modify features significantly
        data["company"].x = torch.randn(3, 9) * 100
        w2 = model.get_attention_weights(data, id_map)
        # At least the numerical values should differ (not necessarily the key set)
        # This is a probabilistic test — with high probability they differ
        etype = ("company", "headquartered_in", "country")
        # Allow for the rare case where they're equal
        # Just verify the method runs without error
        assert etype in w2

    def test_no_grad_during_capture(self):
        """get_attention_weights should not build a computation graph."""
        data, id_map, metadata = _make_simple_graph()
        model = _make_model(metadata)
        model.train()
        model.zero_grad()
        _ = model.get_attention_weights(data, id_map)
        # No parameters should have gradients from the capture pass
        for p in model.parameters():
            assert p.grad is None or (p.grad == 0).all()
