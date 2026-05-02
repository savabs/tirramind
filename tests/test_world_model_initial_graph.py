"""
Tests for agent/models/initial_graph.py — expert-specified initial DAG.

Validates:
    - 9 nodes, 11 edges, correct structure
    - CPDs are well-formed (columns sum to 1)
    - pgmpy check_model passes
    - Graph hash is deterministic
    - Observed nodes map to correct feature names
    - No orphan / isolated nodes
    - Bin edges are valid
    - Serialization round-trip preserves structure
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from agent.models.graph import WorldModelGraph
from agent.models.initial_graph import (
    ALL_EDGES,
    build_initial_graph,
)


class TestBuildInitialGraph:
    """Smoke tests: graph builds without error and basic structure is correct."""

    def test_builds_without_error(self) -> None:
        graph = build_initial_graph()
        assert isinstance(graph, WorldModelGraph)

    def test_node_count(self) -> None:
        graph = build_initial_graph()
        assert len(graph.node_names) == 20

    def test_edge_count(self) -> None:
        graph = build_initial_graph()
        assert len(graph.edges) == 19

    def test_validates_clean(self) -> None:
        graph = build_initial_graph()
        errors = graph.validate()
        assert errors == [], f"Validation errors: {errors}"


class TestNodeStructure:
    """Verify every expected node exists with correct metadata."""

    @pytest.fixture
    def graph(self) -> WorldModelGraph:
        return build_initial_graph()

    def test_regime_nodes(self, graph: WorldModelGraph) -> None:
        regime_nodes = graph.get_regime_nodes()
        names = {n.name for n in regime_nodes}
        assert names == {"regime.macro", "regime.stress"}

    def test_latent_nodes(self, graph: WorldModelGraph) -> None:
        latent_nodes = graph.get_latent_nodes()
        names = {n.name for n in latent_nodes}
        assert names == {"latent.risk_appetite"}

    def test_observed_nodes(self, graph: WorldModelGraph) -> None:
        obs_nodes = graph.get_observed_nodes()
        names = {n.name for n in obs_nodes}
        expected = {
            "obs.rate_momentum",
            "obs.yield_curve_slope",
            "obs.liquidity_pressure",
            "obs.stress_breadth",
            "obs.stress_intensity",
            "obs.regime_persistence",
            # GNN entity-derived (Phase 19c)
            "obs.person_anomaly",
            "obs.company_anomaly",
            "obs.wallet_anomaly",
            "obs.country_anomaly",
            "obs.vessel_anomaly",
            "obs.person_activity",
            "obs.company_activity",
            "obs.wallet_activity",
            "obs.country_activity",
            "obs.vessel_activity",
            "obs.cross_entity",
        }
        assert names == expected

    def test_all_nodes_have_cardinality_3(self, graph: WorldModelGraph) -> None:
        for spec in graph.node_specs.values():
            assert spec.cardinality == 3, f"{spec.name} has cardinality {spec.cardinality}"

    def test_all_nodes_have_states(self, graph: WorldModelGraph) -> None:
        for spec in graph.node_specs.values():
            assert spec.states is not None, f"{spec.name} missing states"
            assert len(spec.states) == 3, f"{spec.name} has {len(spec.states)} states"


class TestEdgeStructure:
    """Verify all 11 edges are present and semantically correct."""

    @pytest.fixture
    def graph(self) -> WorldModelGraph:
        return build_initial_graph()

    def test_all_expected_edges(self, graph: WorldModelGraph) -> None:
        edge_set = set(graph.edges)
        for parent, child in ALL_EDGES:
            assert (parent, child) in edge_set, f"Missing edge: {parent} → {child}"

    def test_no_extra_edges(self, graph: WorldModelGraph) -> None:
        assert len(graph.edges) == len(ALL_EDGES)

    def test_macro_drives_macro_obs(self, graph: WorldModelGraph) -> None:
        children = set(graph.get_children("regime.macro"))
        assert "obs.rate_momentum" in children
        assert "obs.yield_curve_slope" in children
        assert "obs.liquidity_pressure" in children

    def test_stress_drives_convergence_obs(self, graph: WorldModelGraph) -> None:
        children = set(graph.get_children("regime.stress"))
        assert "obs.stress_breadth" in children
        assert "obs.stress_intensity" in children
        assert "obs.regime_persistence" in children

    def test_macro_to_stress_edge(self, graph: WorldModelGraph) -> None:
        assert ("regime.macro", "regime.stress") in graph.edges

    def test_risk_appetite_parents(self, graph: WorldModelGraph) -> None:
        parents = set(graph.get_parents("latent.risk_appetite"))
        assert parents == {"regime.macro", "regime.stress"}

    def test_risk_appetite_children(self, graph: WorldModelGraph) -> None:
        children = set(graph.get_children("latent.risk_appetite"))
        assert children == {
            "obs.liquidity_pressure",
            "obs.stress_intensity",
            "obs.person_activity",
            "obs.company_activity",
            "obs.wallet_activity",
        }

    def test_no_orphan_nodes(self, graph: WorldModelGraph) -> None:
        """Every node except root observed nodes has at least one parent or child."""
        # Root observed nodes (no causal parent, no children): cross_entity,
        # country_activity, vessel_activity — they provide evidence without
        # being caused by a single regime variable.
        root_observed = {
            "obs.cross_entity",
            "obs.country_activity",
            "obs.vessel_activity",
        }
        for name in graph.node_names:
            if name in root_observed:
                continue
            parents = graph.get_parents(name)
            children = graph.get_children(name)
            assert parents or children, f"{name} is orphan (no edges)"


class TestCPDs:
    """Verify CPDs are valid probability distributions."""

    @pytest.fixture
    def graph(self) -> WorldModelGraph:
        return build_initial_graph()

    def test_every_node_has_cpd(self, graph: WorldModelGraph) -> None:
        for name in graph.node_names:
            cpd = graph.get_cpd(name)
            assert cpd is not None, f"{name} missing CPD"

    def test_cpd_columns_sum_to_one(self, graph: WorldModelGraph) -> None:
        """Each column of each CPD sums to ~1.0."""
        for name in graph.node_names:
            cpd = graph.get_cpd(name)
            values = cpd.get_values()
            col_sums = values.sum(axis=0)
            np.testing.assert_allclose(
                col_sums,
                1.0,
                atol=1e-10,
                err_msg=f"{name} CPD columns don't sum to 1",
            )

    def test_cpd_values_non_negative(self, graph: WorldModelGraph) -> None:
        for name in graph.node_names:
            cpd = graph.get_cpd(name)
            values = cpd.get_values()
            assert np.all(values >= 0), f"{name} has negative CPD values"

    def test_root_node_prior_shape(self, graph: WorldModelGraph) -> None:
        """regime.macro is root — prior should be (3, 1)."""
        cpd = graph.get_cpd("regime.macro")
        assert cpd.get_values().shape == (3, 1)

    def test_regime_stress_cpd_shape(self, graph: WorldModelGraph) -> None:
        """P(stress | macro) → (3, 3)."""
        cpd = graph.get_cpd("regime.stress")
        assert cpd.get_values().shape == (3, 3)

    def test_risk_appetite_cpd_shape(self, graph: WorldModelGraph) -> None:
        """P(risk_appetite | macro, stress) → (3, 9)."""
        cpd = graph.get_cpd("latent.risk_appetite")
        assert cpd.get_values().shape == (3, 9)

    def test_liquidity_pressure_cpd_shape(self, graph: WorldModelGraph) -> None:
        """P(liquidity | macro, risk_appetite) → (3, 9)."""
        cpd = graph.get_cpd("obs.liquidity_pressure")
        assert cpd.get_values().shape == (3, 9)

    def test_stress_intensity_cpd_shape(self, graph: WorldModelGraph) -> None:
        """P(stress_intensity | stress, risk_appetite) → (3, 9)."""
        cpd = graph.get_cpd("obs.stress_intensity")
        assert cpd.get_values().shape == (3, 9)

    def test_single_parent_obs_cpd_shape(self, graph: WorldModelGraph) -> None:
        """Single-parent observed nodes have shape (3, 3)."""
        for name in [
            "obs.rate_momentum",
            "obs.yield_curve_slope",
            "obs.stress_breadth",
            "obs.regime_persistence",
        ]:
            cpd = graph.get_cpd(name)
            assert cpd.get_values().shape == (3, 3), f"{name} shape wrong"

    def test_pgmpy_check_model(self, graph: WorldModelGraph) -> None:
        """pgmpy's own consistency check passes."""
        assert graph.bn.check_model()


class TestDomainLogic:
    """Verify CPDs encode sensible domain knowledge."""

    @pytest.fixture
    def graph(self) -> WorldModelGraph:
        return build_initial_graph()

    def test_expansion_favours_rising_rates(self, graph: WorldModelGraph) -> None:
        cpd = graph.get_cpd("obs.rate_momentum")
        values = cpd.get_values()
        # Column 0 = expansion. Row 2 = rising.
        assert values[2, 0] > values[0, 0], "expansion should favour rising > falling"

    def test_crisis_favours_falling_rates(self, graph: WorldModelGraph) -> None:
        cpd = graph.get_cpd("obs.rate_momentum")
        values = cpd.get_values()
        # Column 2 = crisis. Row 0 = falling.
        assert values[0, 2] > values[2, 2], "crisis should favour falling > rising"

    def test_expansion_favours_steep_curve(self, graph: WorldModelGraph) -> None:
        cpd = graph.get_cpd("obs.yield_curve_slope")
        values = cpd.get_values()
        # Column 0 = expansion. Row 2 = steep.
        assert values[2, 0] > values[0, 0], "expansion should favour steep > inverted"

    def test_crisis_favours_inverted_curve(self, graph: WorldModelGraph) -> None:
        cpd = graph.get_cpd("obs.yield_curve_slope")
        values = cpd.get_values()
        # Column 2 = crisis. Row 0 = inverted.
        assert values[0, 2] > values[2, 2], "crisis should favour inverted > steep"

    def test_calm_favours_narrow_stress(self, graph: WorldModelGraph) -> None:
        cpd = graph.get_cpd("obs.stress_breadth")
        values = cpd.get_values()
        # Column 0 = calm. Row 0 = narrow.
        assert values[0, 0] > values[2, 0], "calm should favour narrow > broad"

    def test_extreme_favours_broad_stress(self, graph: WorldModelGraph) -> None:
        cpd = graph.get_cpd("obs.stress_breadth")
        values = cpd.get_values()
        # Column 2 = extreme. Row 2 = broad.
        assert values[2, 2] > values[0, 2], "extreme should favour broad > narrow"

    def test_calm_favours_persistent(self, graph: WorldModelGraph) -> None:
        cpd = graph.get_cpd("obs.regime_persistence")
        values = cpd.get_values()
        # Column 0 = calm. Row 2 = persistent.
        assert values[2, 0] > values[0, 0], "calm should favour persistent > unstable"

    def test_expansion_plus_calm_favours_risk_on(self, graph: WorldModelGraph) -> None:
        cpd = graph.get_cpd("latent.risk_appetite")
        values = cpd.get_values()
        # Column 0 = (expansion, calm). Row 0 = risk_on.
        assert values[0, 0] > values[2, 0], "expansion+calm → risk_on > risk_off"

    def test_crisis_plus_extreme_favours_risk_off(self, graph: WorldModelGraph) -> None:
        cpd = graph.get_cpd("latent.risk_appetite")
        values = cpd.get_values()
        # Column 8 = (crisis, extreme). Row 2 = risk_off.
        assert values[2, 8] > values[0, 8], "crisis+extreme → risk_off > risk_on"

    def test_expansion_favours_calm(self, graph: WorldModelGraph) -> None:
        cpd = graph.get_cpd("regime.stress")
        values = cpd.get_values()
        # Column 0 = expansion. Row 0 = calm.
        assert values[0, 0] > values[2, 0], "expansion → calm > extreme"

    def test_crisis_favours_extreme(self, graph: WorldModelGraph) -> None:
        cpd = graph.get_cpd("regime.stress")
        values = cpd.get_values()
        # Column 2 = crisis. Row 2 = extreme.
        assert values[2, 2] > values[0, 2], "crisis → extreme > calm"


class TestFeatureMapping:
    """Verify observed nodes map correctly to EngineeredFeature names."""

    @pytest.fixture
    def graph(self) -> WorldModelGraph:
        return build_initial_graph()

    def test_feature_name_mapping(self, graph: WorldModelGraph) -> None:
        expected_mapping = {
            "obs.rate_momentum": "macro.rate_momentum.30d",
            "obs.yield_curve_slope": "macro.yield_curve_slope.spot",
            "obs.liquidity_pressure": "macro.liquidity_pressure.30d",
            "obs.stress_breadth": "convergence.stress_breadth.7d",
            "obs.stress_intensity": "convergence.stress_intensity.7d",
            "obs.regime_persistence": "convergence.regime_persistence.7d",
        }
        for node_name, feat_name in expected_mapping.items():
            spec = graph.get_node(node_name)
            assert spec.feature_name == feat_name, f"{node_name}: expected {feat_name}, got {spec.feature_name}"

    def test_non_observed_nodes_have_no_feature(self, graph: WorldModelGraph) -> None:
        for spec in graph.node_specs.values():
            if spec.node_type != "observed":
                assert spec.feature_name is None, f"{spec.name} is {spec.node_type} but has feature_name"


class TestBinEdges:
    """Verify bin edges are valid for discretization."""

    @pytest.fixture
    def graph(self) -> WorldModelGraph:
        return build_initial_graph()

    def test_observed_nodes_have_bin_edges(self, graph: WorldModelGraph) -> None:
        for spec in graph.get_observed_nodes():
            assert spec.bin_edges is not None, f"{spec.name} missing bin_edges"

    def test_bin_edges_length(self, graph: WorldModelGraph) -> None:
        for spec in graph.get_observed_nodes():
            assert len(spec.bin_edges) == spec.cardinality + 1, (
                f"{spec.name}: expected {spec.cardinality + 1} bin_edges, got {len(spec.bin_edges)}"
            )

    def test_bin_edges_start_neg_inf(self, graph: WorldModelGraph) -> None:
        for spec in graph.get_observed_nodes():
            assert spec.bin_edges[0] == -math.inf, f"{spec.name}: first bin edge should be -inf"

    def test_bin_edges_end_pos_inf(self, graph: WorldModelGraph) -> None:
        for spec in graph.get_observed_nodes():
            assert spec.bin_edges[-1] == math.inf, f"{spec.name}: last bin edge should be +inf"

    def test_bin_edges_monotonic(self, graph: WorldModelGraph) -> None:
        for spec in graph.get_observed_nodes():
            for i in range(len(spec.bin_edges) - 1):
                assert spec.bin_edges[i] < spec.bin_edges[i + 1], f"{spec.name}: bin_edges not monotonic at index {i}"

    def test_non_observed_have_no_bin_edges(self, graph: WorldModelGraph) -> None:
        for spec in graph.node_specs.values():
            if spec.node_type != "observed":
                assert spec.bin_edges is None, f"{spec.name} is {spec.node_type} but has bin_edges"


class TestHashAndSerialization:
    """Verify determinism and round-trip preservation."""

    def test_hash_is_deterministic(self) -> None:
        g1 = build_initial_graph()
        g2 = build_initial_graph()
        assert g1.graph_hash() == g2.graph_hash()

    def test_hash_is_64_hex(self) -> None:
        graph = build_initial_graph()
        h = graph.graph_hash()
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_serialization_round_trip(self) -> None:
        graph = build_initial_graph()
        d = graph.to_dict()
        restored = WorldModelGraph.from_dict(d)
        assert restored.graph_hash() == graph.graph_hash()
        assert set(restored.node_names) == set(graph.node_names)
        assert len(restored.edges) == len(graph.edges)

    def test_serialized_dict_has_expected_keys(self) -> None:
        graph = build_initial_graph()
        d = graph.to_dict()
        assert "nodes" in d
        assert "edges" in d
        assert "graph_hash" in d
        assert len(d["nodes"]) == 20
        assert len(d["edges"]) == 19


class TestWeaklyInformativePrior:
    """Verify the root prior is center-biased."""

    def test_macro_prior_center_biased(self) -> None:
        graph = build_initial_graph()
        cpd = graph.get_cpd("regime.macro")
        values = cpd.get_values().flatten()
        # Middle state (contraction index=1) should have highest prior
        assert values[1] > values[0]
        assert values[1] > values[2]

    def test_macro_prior_sums_to_one(self) -> None:
        graph = build_initial_graph()
        cpd = graph.get_cpd("regime.macro")
        np.testing.assert_allclose(cpd.get_values().sum(), 1.0, atol=1e-10)
