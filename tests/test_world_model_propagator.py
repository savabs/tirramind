"""
Tests for agent/models/propagator.py — belief propagation engine.

Validates:
    - Posterior computation with known CPDs
    - Evidence updates beliefs for children
    - Missing evidence returns priors
    - Discrete state label evidence
    - Continuous value discretization
    - Quality = 0.0 ignores evidence
    - Evidence on non-existent node raises
    - All beliefs are valid categorical distributions
    - value_to_state_index bin edge mapping
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from pgmpy.factors.discrete import TabularCPD

from agent.models.graph import NodeSpec, WorldModelGraph
from agent.models.initial_graph import build_initial_graph
from agent.models.propagator import BeliefPropagator, value_to_state_index

# ── Helpers ────────────────────────────────────────────────────


def _build_simple_graph() -> WorldModelGraph:
    """A → B, both binary, with known CPDs for hand verification."""
    a = NodeSpec(
        name="test.parent",
        node_type="regime",
        domain="test",
        cardinality=2,
        states=("low", "high"),
    )
    b = NodeSpec(
        name="test.child",
        node_type="observed",
        domain="test",
        cardinality=2,
        states=("off", "on"),
        feature_name="test.feature",
        bin_edges=(-math.inf, 0.5, math.inf),
    )
    graph = WorldModelGraph(nodes=[a, b], edges=[("test.parent", "test.child")])

    # P(A): [0.6, 0.4]
    cpd_a = TabularCPD(
        variable="test.parent",
        variable_card=2,
        values=[[0.6], [0.4]],
        state_names={"test.parent": ["low", "high"]},
    )
    # P(B|A):
    #   A=low:  B=off=0.8, B=on=0.2
    #   A=high: B=off=0.3, B=on=0.7
    cpd_b = TabularCPD(
        variable="test.child",
        variable_card=2,
        values=[[0.8, 0.3], [0.2, 0.7]],
        evidence=["test.parent"],
        evidence_card=[2],
        state_names={
            "test.child": ["off", "on"],
            "test.parent": ["low", "high"],
        },
    )
    graph.set_cpd("test.parent", cpd_a)
    graph.set_cpd("test.child", cpd_b)
    return graph


AS_OF = 1_700_000_000.0  # fixed timestamp for tests


# ── value_to_state_index ───────────────────────────────────────


class TestValueToStateIndex:
    def test_three_bins(self) -> None:
        edges = (-math.inf, -0.5, 0.5, math.inf)
        assert value_to_state_index(-1.0, edges) == 0
        assert value_to_state_index(0.0, edges) == 1
        assert value_to_state_index(1.0, edges) == 2

    def test_boundary_goes_right(self) -> None:
        edges = (-math.inf, -0.5, 0.5, math.inf)
        assert value_to_state_index(-0.5, edges) == 1  # right-open: [-0.5, 0.5)
        assert value_to_state_index(0.5, edges) == 2  # last bin closed

    def test_two_bins(self) -> None:
        edges = (-math.inf, 0.0, math.inf)
        assert value_to_state_index(-1.0, edges) == 0
        assert value_to_state_index(1.0, edges) == 1

    def test_extreme_values(self) -> None:
        edges = (-math.inf, 0.0, math.inf)
        assert value_to_state_index(-1e18, edges) == 0
        assert value_to_state_index(1e18, edges) == 1


# ── Propagation with simple graph ──────────────────────────────


class TestSimpleGraphPropagation:
    @pytest.fixture
    def graph(self) -> WorldModelGraph:
        return _build_simple_graph()

    @pytest.fixture
    def propagator(self, graph: WorldModelGraph) -> BeliefPropagator:
        return BeliefPropagator(graph)

    def test_no_evidence_returns_priors(self, propagator: BeliefPropagator) -> None:
        beliefs = propagator.propagate(evidence={}, as_of=AS_OF)
        assert len(beliefs) == 2  # both nodes

        parent_belief = next(b for b in beliefs if b.variable_name == "test.parent")
        assert parent_belief.dist_type == "categorical"
        np.testing.assert_allclose(
            parent_belief.probabilities["low"],
            0.6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            parent_belief.probabilities["high"],
            0.4,
            atol=1e-6,
        )

    def test_evidence_on_child_updates_parent(
        self,
        propagator: BeliefPropagator,
    ) -> None:
        """P(A | B=on) by Bayes:
        P(A=low | B=on) = P(B=on|A=low)*P(A=low) / P(B=on)
            = 0.2*0.6 / (0.2*0.6 + 0.7*0.4) = 0.12 / 0.40 = 0.30
        P(A=high | B=on) = 0.7*0.4 / 0.40 = 0.70
        """
        beliefs = propagator.propagate(
            evidence={"test.child": "on"},
            as_of=AS_OF,
        )
        parent_belief = next(b for b in beliefs if b.variable_name == "test.parent")
        np.testing.assert_allclose(
            parent_belief.probabilities["low"],
            0.30,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            parent_belief.probabilities["high"],
            0.70,
            atol=1e-6,
        )

    def test_evidence_on_parent_updates_child(
        self,
        propagator: BeliefPropagator,
    ) -> None:
        """P(B | A=low):  B=off=0.8, B=on=0.2."""
        beliefs = propagator.propagate(
            evidence={"test.parent": "low"},
            as_of=AS_OF,
        )
        child_belief = next(b for b in beliefs if b.variable_name == "test.child")
        np.testing.assert_allclose(
            child_belief.probabilities["off"],
            0.8,
            atol=1e-6,
        )

    def test_evidence_node_gets_delta(self, propagator: BeliefPropagator) -> None:
        beliefs = propagator.propagate(
            evidence={"test.child": "on"},
            as_of=AS_OF,
        )
        child_belief = next(b for b in beliefs if b.variable_name == "test.child")
        assert child_belief.probabilities["on"] == 1.0
        assert child_belief.probabilities["off"] == 0.0


class TestContinuousEvidence:
    @pytest.fixture
    def graph(self) -> WorldModelGraph:
        return _build_simple_graph()

    @pytest.fixture
    def propagator(self, graph: WorldModelGraph) -> BeliefPropagator:
        return BeliefPropagator(graph)

    def test_float_value_discretized(self, propagator: BeliefPropagator) -> None:
        """test.child has bin_edges (-inf, 0.5, inf).
        value=0.2 → state index 0 → 'off'.
        """
        beliefs = propagator.propagate(
            evidence={"test.child": 0.2},
            as_of=AS_OF,
        )
        child_belief = next(b for b in beliefs if b.variable_name == "test.child")
        assert child_belief.probabilities["off"] == 1.0

    def test_high_float_maps_to_on(self, propagator: BeliefPropagator) -> None:
        """value=0.8 → state index 1 → 'on'."""
        beliefs = propagator.propagate(
            evidence={"test.child": 0.8},
            as_of=AS_OF,
        )
        child_belief = next(b for b in beliefs if b.variable_name == "test.child")
        assert child_belief.probabilities["on"] == 1.0


class TestQuality:
    @pytest.fixture
    def propagator(self) -> BeliefPropagator:
        return BeliefPropagator(_build_simple_graph())

    def test_quality_zero_ignores_evidence(self, propagator: BeliefPropagator) -> None:
        """quality=0 should produce same as no evidence."""
        priors = propagator.propagate(evidence={}, as_of=AS_OF)
        beliefs = propagator.propagate(
            evidence={"test.child": "on"},
            as_of=AS_OF,
            quality={"test.child": 0.0},
        )
        # Non-evidence beliefs should match priors
        prior_parent = next(b for b in priors if b.variable_name == "test.parent")
        parent = next(b for b in beliefs if b.variable_name == "test.parent")
        np.testing.assert_allclose(
            parent.probabilities["low"],
            prior_parent.probabilities["low"],
            atol=1e-6,
        )

    def test_quality_one_is_hard_evidence(self, propagator: BeliefPropagator) -> None:
        """quality=1.0 should produce same as default (hard evidence)."""
        default = propagator.propagate(
            evidence={"test.child": "on"},
            as_of=AS_OF,
        )
        explicit = propagator.propagate(
            evidence={"test.child": "on"},
            as_of=AS_OF,
            quality={"test.child": 1.0},
        )
        dp = next(b for b in default if b.variable_name == "test.parent")
        ep = next(b for b in explicit if b.variable_name == "test.parent")
        np.testing.assert_allclose(
            dp.probabilities["low"],
            ep.probabilities["low"],
            atol=1e-6,
        )


class TestErrorCases:
    @pytest.fixture
    def propagator(self) -> BeliefPropagator:
        return BeliefPropagator(_build_simple_graph())

    def test_nonexistent_node_raises(self, propagator: BeliefPropagator) -> None:
        with pytest.raises(ValueError, match="not in graph"):
            propagator.propagate(
                evidence={"nonexistent.node": "value"},
                as_of=AS_OF,
            )

    def test_invalid_state_label_raises(self, propagator: BeliefPropagator) -> None:
        with pytest.raises(ValueError, match="not in states"):
            propagator.propagate(
                evidence={"test.parent": "invalid_state"},
                as_of=AS_OF,
            )

    def test_invalid_type_raises(self, propagator: BeliefPropagator) -> None:
        with pytest.raises(TypeError, match="must be str, int, or float"):
            propagator.propagate(
                evidence={"test.parent": [1, 2, 3]},
                as_of=AS_OF,
            )

    def test_continuous_value_on_node_without_bins_raises(self) -> None:
        """Parent has no bin_edges — can't discretize a float."""
        graph = _build_simple_graph()
        prop = BeliefPropagator(graph)
        with pytest.raises(ValueError, match="no bin_edges"):
            prop.propagate(
                evidence={"test.parent": 0.5},
                as_of=AS_OF,
            )


class TestBeliefStateProperties:
    """Verify output BeliefState records are well-formed."""

    @pytest.fixture
    def propagator(self) -> BeliefPropagator:
        return BeliefPropagator(_build_simple_graph())

    def test_all_beliefs_categorical(self, propagator: BeliefPropagator) -> None:
        beliefs = propagator.propagate(evidence={}, as_of=AS_OF)
        for b in beliefs:
            assert b.dist_type == "categorical"

    def test_probabilities_sum_to_one(self, propagator: BeliefPropagator) -> None:
        beliefs = propagator.propagate(
            evidence={"test.child": "on"},
            as_of=AS_OF,
        )
        for b in beliefs:
            total = sum(b.probabilities.values())
            np.testing.assert_allclose(total, 1.0, atol=1e-6)

    def test_effective_at_matches_as_of(self, propagator: BeliefPropagator) -> None:
        beliefs = propagator.propagate(evidence={}, as_of=AS_OF)
        for b in beliefs:
            assert b.effective_at == AS_OF

    def test_stale_when_no_evidence(self, propagator: BeliefPropagator) -> None:
        beliefs = propagator.propagate(evidence={}, as_of=AS_OF)
        for b in beliefs:
            assert b.stale is True

    def test_not_stale_with_evidence(self, propagator: BeliefPropagator) -> None:
        beliefs = propagator.propagate(
            evidence={"test.child": "on"},
            as_of=AS_OF,
        )
        # Evidence node should not be stale
        child = next(b for b in beliefs if b.variable_name == "test.child")
        assert child.stale is False

    def test_graph_hash_present(self, propagator: BeliefPropagator) -> None:
        beliefs = propagator.propagate(evidence={}, as_of=AS_OF)
        for b in beliefs:
            assert len(b.model_graph_hash) == 64


class TestWithInitialGraph:
    """Integration tests using the full expert DAG."""

    @pytest.fixture
    def propagator(self) -> BeliefPropagator:
        return BeliefPropagator(build_initial_graph())

    def test_prior_propagation(self, propagator: BeliefPropagator) -> None:
        beliefs = propagator.propagate_priors(as_of=AS_OF)
        assert len(beliefs) == 20  # all 20 nodes

    def test_priors_sum_to_one(self, propagator: BeliefPropagator) -> None:
        beliefs = propagator.propagate_priors(as_of=AS_OF)
        for b in beliefs:
            total = sum(b.probabilities.values())
            np.testing.assert_allclose(total, 1.0, atol=1e-6)

    def test_single_feature_evidence(self, propagator: BeliefPropagator) -> None:
        beliefs = propagator.propagate(
            evidence={"obs.rate_momentum": "rising"},
            as_of=AS_OF,
        )
        assert len(beliefs) == 20
        # regime.macro should shift toward expansion
        macro = next(b for b in beliefs if b.variable_name == "regime.macro")
        assert macro.probabilities["expansion"] > macro.probabilities["crisis"]

    def test_multiple_feature_evidence(self, propagator: BeliefPropagator) -> None:
        beliefs = propagator.propagate(
            evidence={
                "obs.rate_momentum": "falling",
                "obs.yield_curve_slope": "inverted",
                "obs.stress_breadth": "broad",
            },
            as_of=AS_OF,
        )
        # With falling rates + inverted curve + broad stress → crisis-like
        macro = next(b for b in beliefs if b.variable_name == "regime.macro")
        assert macro.probabilities["crisis"] > macro.probabilities["expansion"]

    def test_continuous_value_evidence(self, propagator: BeliefPropagator) -> None:
        """Pass a raw float and verify it gets discretized."""
        beliefs = propagator.propagate(
            evidence={"obs.rate_momentum": -1.0},  # -1.0 → falling
            as_of=AS_OF,
        )
        rm = next(b for b in beliefs if b.variable_name == "obs.rate_momentum")
        assert rm.probabilities["falling"] == 1.0

    def test_all_evidence_produces_beliefs(self, propagator: BeliefPropagator) -> None:
        """Evidence on every observed node → beliefs for all nodes."""
        beliefs = propagator.propagate(
            evidence={
                "obs.rate_momentum": "rising",
                "obs.yield_curve_slope": "steep",
                "obs.liquidity_pressure": "loose",
                "obs.stress_breadth": "narrow",
                "obs.stress_intensity": "low",
                "obs.regime_persistence": "persistent",
            },
            as_of=AS_OF,
        )
        assert len(beliefs) == 20
        macro = next(b for b in beliefs if b.variable_name == "regime.macro")
        # With all expansion-consistent evidence, expansion should dominate
        assert macro.probabilities["expansion"] > 0.5
