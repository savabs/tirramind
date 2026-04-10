"""Tests for WorldModelGraph + NodeSpec — structure, validation, edge cases."""

from __future__ import annotations

import pytest

from pgmpy.factors.discrete import TabularCPD

from agent.models.graph import NodeSpec, WorldModelGraph


# ── Helpers ────────────────────────────────────────────────────


def _regime_node(name="regime.macro"):
    return NodeSpec(
        name=name,
        node_type="regime",
        domain="regime",
        cardinality=3,
        states=("expansion", "contraction", "crisis"),
    )


def _observed_node(name="obs.rate_momentum", feature="macro.rate_momentum.30d"):
    return NodeSpec(
        name=name,
        node_type="observed",
        domain="macro",
        cardinality=3,
        states=("low", "mid", "high"),
        feature_name=feature,
        bin_edges=(-float("inf"), -0.5, 0.5, float("inf")),
    )


def _simple_graph():
    """2 nodes, 1 edge: regime → observed."""
    r = _regime_node()
    o = _observed_node()
    g = WorldModelGraph(nodes=[r, o], edges=[("regime.macro", "obs.rate_momentum")])
    return g


# ── NodeSpec ───────────────────────────────────────────────────


class TestNodeSpec:
    def test_frozen(self):
        n = _regime_node()
        with pytest.raises(AttributeError):
            n.name = "x"  # type: ignore[misc]

    def test_to_dict_round_trip(self):
        n = _observed_node()
        d = n.to_dict()
        n2 = NodeSpec.from_dict(d)
        assert n2 == n

    def test_regime_node_no_feature(self):
        n = _regime_node()
        assert n.feature_name is None
        assert n.node_type == "regime"


# ── Graph construction ─────────────────────────────────────────


class TestGraphConstruction:
    def test_empty_graph(self):
        g = WorldModelGraph()
        assert g.node_names == []
        assert g.edges == []

    def test_add_nodes_and_edges(self):
        g = _simple_graph()
        assert len(g.node_names) == 2
        assert len(g.edges) == 1

    def test_idempotent_add_node(self):
        g = WorldModelGraph()
        n = _regime_node()
        g.add_node(n)
        g.add_node(n)  # same spec = idempotent
        assert len(g.node_names) == 1

    def test_duplicate_node_different_spec_raises(self):
        g = WorldModelGraph()
        g.add_node(_regime_node())
        with pytest.raises(ValueError, match="already exists"):
            g.add_node(
                NodeSpec(
                    name="regime.macro",
                    node_type="latent",  # different type
                    domain="regime",
                    cardinality=2,
                )
            )

    def test_edge_missing_parent_raises(self):
        g = WorldModelGraph(nodes=[_observed_node()])
        with pytest.raises(ValueError, match="Parent node"):
            g.add_edge("nonexistent", "obs.rate_momentum")

    def test_edge_missing_child_raises(self):
        g = WorldModelGraph(nodes=[_regime_node()])
        with pytest.raises(ValueError, match="Child node"):
            g.add_edge("regime.macro", "nonexistent")

    def test_self_loop_raises(self):
        g = WorldModelGraph(nodes=[_regime_node()])
        with pytest.raises(ValueError, match="Self-loop"):
            g.add_edge("regime.macro", "regime.macro")

    def test_get_parents_children(self):
        g = _simple_graph()
        assert g.get_parents("obs.rate_momentum") == ["regime.macro"]
        assert g.get_children("regime.macro") == ["obs.rate_momentum"]
        assert g.get_parents("regime.macro") == []

    def test_filtered_nodes(self):
        g = _simple_graph()
        assert len(g.get_observed_nodes()) == 1
        assert len(g.get_regime_nodes()) == 1
        assert len(g.get_latent_nodes()) == 0


# ── Hashing ────────────────────────────────────────────────────


class TestGraphHashing:
    def test_hash_is_64_hex(self):
        g = _simple_graph()
        h = g.graph_hash()
        assert len(h) == 64
        int(h, 16)  # validates hex

    def test_hash_deterministic(self):
        g1 = _simple_graph()
        g2 = _simple_graph()
        assert g1.graph_hash() == g2.graph_hash()

    def test_hash_changes_with_edge(self):
        r = _regime_node()
        o = _observed_node()
        g1 = WorldModelGraph(nodes=[r, o])
        g2 = WorldModelGraph(
            nodes=[r, o], edges=[("regime.macro", "obs.rate_momentum")]
        )
        assert g1.graph_hash() != g2.graph_hash()

    def test_hash_order_independent(self):
        """Nodes added in different order produce same hash."""
        r = _regime_node()
        o = _observed_node()
        g1 = WorldModelGraph(
            nodes=[r, o], edges=[("regime.macro", "obs.rate_momentum")]
        )
        g2 = WorldModelGraph(
            nodes=[o, r], edges=[("regime.macro", "obs.rate_momentum")]
        )
        assert g1.graph_hash() == g2.graph_hash()


# ── CPD operations ─────────────────────────────────────────────


class TestCPDOperations:
    def test_set_and_get_cpd(self):
        g = WorldModelGraph(nodes=[_regime_node()])
        cpd = TabularCPD(
            variable="regime.macro",
            variable_card=3,
            values=[[0.5], [0.3], [0.2]],
        )
        g.set_cpd("regime.macro", cpd)
        retrieved = g.get_cpd("regime.macro")
        assert retrieved is not None
        assert retrieved.variable == "regime.macro"

    def test_get_cpd_missing(self):
        g = WorldModelGraph(nodes=[_regime_node()])
        assert g.get_cpd("regime.macro") is None

    def test_set_cpd_unknown_node_raises(self):
        g = WorldModelGraph()
        cpd = TabularCPD(variable="x", variable_card=2, values=[[0.5], [0.5]])
        with pytest.raises(ValueError, match="not in graph"):
            g.set_cpd("x", cpd)

    def test_replace_cpd(self):
        g = WorldModelGraph(nodes=[_regime_node()])
        cpd1 = TabularCPD(
            variable="regime.macro",
            variable_card=3,
            values=[[0.5], [0.3], [0.2]],
        )
        cpd2 = TabularCPD(
            variable="regime.macro",
            variable_card=3,
            values=[[0.1], [0.1], [0.8]],
        )
        g.set_cpd("regime.macro", cpd1)
        g.set_cpd("regime.macro", cpd2)
        assert len(g.get_all_cpds()) == 1
        assert g.get_cpd("regime.macro").get_values()[2][0] == pytest.approx(0.8)


# ── Validation ─────────────────────────────────────────────────


class TestGraphValidation:
    def test_valid_graph_with_cpds(self):
        g = _simple_graph()
        cpd_r = TabularCPD(
            variable="regime.macro",
            variable_card=3,
            values=[[0.5], [0.3], [0.2]],
        )
        cpd_o = TabularCPD(
            variable="obs.rate_momentum",
            variable_card=3,
            values=[[0.6, 0.2, 0.1], [0.3, 0.5, 0.3], [0.1, 0.3, 0.6]],
            evidence=["regime.macro"],
            evidence_card=[3],
        )
        g.set_cpd("regime.macro", cpd_r)
        g.set_cpd("obs.rate_momentum", cpd_o)
        errors = g.validate()
        assert errors == []

    def test_missing_cpd_reported(self):
        g = _simple_graph()
        errors = g.validate()
        assert any("missing CPD" in e for e in errors)

    def test_observed_without_feature_name(self):
        bad = NodeSpec(
            name="obs.missing",
            node_type="observed",
            domain="macro",
            cardinality=3,
            states=("a", "b", "c"),
            # feature_name omitted
        )
        g = WorldModelGraph(nodes=[bad])
        errors = g.validate()
        assert any("feature_name" in e for e in errors)

    def test_cardinality_zero(self):
        bad = NodeSpec(
            name="bad.node",
            node_type="latent",
            domain="latent",
            cardinality=0,
        )
        g = WorldModelGraph(nodes=[bad])
        errors = g.validate()
        assert any("cardinality" in e for e in errors)

    def test_states_cardinality_mismatch(self):
        bad = NodeSpec(
            name="bad.node",
            node_type="regime",
            domain="regime",
            cardinality=3,
            states=("a", "b"),  # only 2 states for card=3
        )
        g = WorldModelGraph(nodes=[bad])
        errors = g.validate()
        assert any("states length" in e for e in errors)

    def test_bin_edges_wrong_length(self):
        bad = NodeSpec(
            name="obs.bad",
            node_type="observed",
            domain="macro",
            cardinality=3,
            states=("a", "b", "c"),
            feature_name="x.y.z",
            bin_edges=(-1.0, 0.0, 1.0),  # should be 4 for card=3
        )
        g = WorldModelGraph(nodes=[bad])
        errors = g.validate()
        assert any("bin_edges" in e for e in errors)


# ── Serialization ──────────────────────────────────────────────


class TestGraphSerialization:
    def test_round_trip(self):
        g = _simple_graph()
        d = g.to_dict()
        g2 = WorldModelGraph.from_dict(d)
        assert g2.graph_hash() == g.graph_hash()
        assert set(g2.node_names) == set(g.node_names)
        assert set(g2.edges) == set(g.edges)

    def test_to_dict_contains_hash(self):
        g = _simple_graph()
        d = g.to_dict()
        assert "graph_hash" in d
        assert len(d["graph_hash"]) == 64

    def test_one_node_graph(self):
        g = WorldModelGraph(nodes=[_regime_node()])
        d = g.to_dict()
        g2 = WorldModelGraph.from_dict(d)
        assert g2.node_names == ["regime.macro"]
        assert g2.edges == []
