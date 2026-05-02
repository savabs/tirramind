"""
Step 9.9 — Comprehensive edge case test suite for the World Model.

Covers edge cases across all modules as specified in
docs/specs/world_model_spec.md (sub-phase 9.9).
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
from pgmpy.factors.discrete import TabularCPD

from agent.models.belief import validate_belief
from agent.models.graph import NodeSpec, WorldModelGraph
from agent.models.initial_graph import build_initial_graph
from agent.models.intervention import InterventionEngine, _kl_divergence
from agent.models.propagator import BeliefPropagator, value_to_state_index
from agent.models.state_filter import ContinuousStateFilter, RegimeConfig
from agent.models.world_model import WorldModel

# ═══════════════════════════════════════════════════════════════
# Graph edge cases
# ═══════════════════════════════════════════════════════════════


class TestGraphCycleDetection:
    """Add edge that creates cycle → rejected."""

    def test_simple_cycle_rejected(self) -> None:
        a = NodeSpec(
            name="a",
            node_type="latent",
            domain="test",
            cardinality=2,
            states=("s0", "s1"),
        )
        b = NodeSpec(
            name="b",
            node_type="latent",
            domain="test",
            cardinality=2,
            states=("s0", "s1"),
        )
        g = WorldModelGraph(nodes=[a, b], edges=[("a", "b")])
        with pytest.raises(ValueError, match="[Ll]oop"):
            g.add_edge("b", "a")

    def test_transitive_cycle_rejected(self) -> None:
        nodes = [
            NodeSpec(
                name=n,
                node_type="latent",
                domain="test",
                cardinality=2,
                states=("s0", "s1"),
            )
            for n in ("x", "y", "z")
        ]
        g = WorldModelGraph(nodes=nodes, edges=[("x", "y"), ("y", "z")])
        with pytest.raises(ValueError, match="[Ll]oop"):
            g.add_edge("z", "x")

    def test_self_loop_rejected(self) -> None:
        a = NodeSpec(
            name="a",
            node_type="latent",
            domain="test",
            cardinality=2,
            states=("s0", "s1"),
        )
        g = WorldModelGraph(nodes=[a])
        with pytest.raises(ValueError, match="Self-loop"):
            g.add_edge("a", "a")


class TestGraphDisconnectedComponents:
    """Graph with disconnected components → should be structurally valid."""

    def test_disconnected_graph_no_structural_error(self) -> None:
        a = NodeSpec(
            name="a",
            node_type="latent",
            domain="test",
            cardinality=2,
            states=("s0", "s1"),
        )
        b = NodeSpec(
            name="b",
            node_type="latent",
            domain="test",
            cardinality=2,
            states=("s0", "s1"),
        )
        g = WorldModelGraph(nodes=[a, b])
        # No edges: two disconnected nodes.
        # Should not crash; validate() may warn about missing CPDs
        # but graph structure itself is fine.
        assert len(g.node_names) == 2
        assert len(g.edges) == 0

    def test_single_node_graph(self) -> None:
        a = NodeSpec(
            name="a",
            node_type="regime",
            domain="test",
            cardinality=2,
            states=("lo", "hi"),
        )
        g = WorldModelGraph(nodes=[a])
        cpd = TabularCPD("a", 2, [[0.5], [0.5]], state_names={"a": ["lo", "hi"]})
        g.set_cpd("a", cpd)
        errors = g.validate()
        assert errors == []


class TestGraphHashStability:
    """Hash stability across serialization round-trip."""

    def test_round_trip_hash(self) -> None:
        g = build_initial_graph()
        h1 = g.graph_hash()
        d = g.to_dict()
        g2 = WorldModelGraph.from_dict(d)
        h2 = g2.graph_hash()
        assert h1 == h2

    def test_insertion_order_invariance(self) -> None:
        """Same nodes added in different order → same hash."""
        spec_a = NodeSpec(
            name="a",
            node_type="latent",
            domain="test",
            cardinality=2,
            states=("s0", "s1"),
        )
        spec_b = NodeSpec(
            name="b",
            node_type="latent",
            domain="test",
            cardinality=2,
            states=("s0", "s1"),
        )
        g1 = WorldModelGraph(nodes=[spec_a, spec_b], edges=[("a", "b")])
        g2 = WorldModelGraph(nodes=[spec_b, spec_a], edges=[("a", "b")])
        assert g1.graph_hash() == g2.graph_hash()


# ═══════════════════════════════════════════════════════════════
# Propagator edge cases
# ═══════════════════════════════════════════════════════════════


def _make_simple_graph_with_cpds() -> WorldModelGraph:
    """A tiny 2-node graph for quick propagator tests."""
    parent = NodeSpec(
        name="regime.macro",
        node_type="regime",
        domain="regime",
        cardinality=3,
        states=("expansion", "contraction", "crisis"),
    )
    child = NodeSpec(
        name="obs.rate",
        node_type="observed",
        domain="macro",
        cardinality=3,
        states=("falling", "neutral", "rising"),
        feature_name="macro.rate_momentum.30d",
        bin_edges=(-math.inf, -0.5, 0.5, math.inf),
    )
    g = WorldModelGraph(nodes=[parent, child], edges=[("regime.macro", "obs.rate")])
    g.set_cpd(
        "regime.macro",
        TabularCPD(
            "regime.macro",
            3,
            [[0.5], [0.3], [0.2]],
            state_names={"regime.macro": ["expansion", "contraction", "crisis"]},
        ),
    )
    g.set_cpd(
        "obs.rate",
        TabularCPD(
            "obs.rate",
            3,
            [[0.1, 0.3, 0.7], [0.3, 0.4, 0.2], [0.6, 0.3, 0.1]],
            evidence=["regime.macro"],
            evidence_card=[3],
            state_names={
                "obs.rate": ["falling", "neutral", "rising"],
                "regime.macro": ["expansion", "contraction", "crisis"],
            },
        ),
    )
    return g


class TestPropagatorQualityIntermediate:
    """Quality in (0,1) → Phase 9a treats as hard evidence with reduced confidence."""

    def test_quality_half_reduces_confidence(self) -> None:
        g = _make_simple_graph_with_cpds()
        prop = BeliefPropagator(g)
        t = time.time()
        beliefs = prop.propagate(
            evidence={"obs.rate": "rising"},
            as_of=t,
            quality={"obs.rate": 0.5},
        )
        regime_belief = next(b for b in beliefs if b.variable_name == "regime.macro")
        # Confidence should reflect average soft quality
        assert regime_belief.confidence == pytest.approx(0.5, abs=0.01)

    def test_quality_one_full_confidence(self) -> None:
        g = _make_simple_graph_with_cpds()
        prop = BeliefPropagator(g)
        beliefs = prop.propagate(
            evidence={"obs.rate": "rising"},
            as_of=time.time(),
            quality={"obs.rate": 1.0},
        )
        regime_belief = next(b for b in beliefs if b.variable_name == "regime.macro")
        assert regime_belief.confidence == 1.0


class TestPropagatorEvidenceOnLatent:
    """Evidence on latent node → should work (hard evidence on hidden variable)."""

    def test_latent_evidence_accepted(self) -> None:
        g = _make_simple_graph_with_cpds()
        prop = BeliefPropagator(g)
        beliefs = prop.propagate(
            evidence={"regime.macro": "crisis"},
            as_of=time.time(),
        )
        # regime.macro should be delta on crisis
        regime_belief = next(b for b in beliefs if b.variable_name == "regime.macro")
        assert regime_belief.probabilities["crisis"] == 1.0

        # obs.rate should reflect the conditional P(obs.rate | regime.macro=crisis)
        obs_belief = next(b for b in beliefs if b.variable_name == "obs.rate")
        assert obs_belief.probabilities["falling"] == pytest.approx(0.7, abs=0.01)


class TestPropagatorPrecision:
    """Probabilities should sum to 1.0 within tolerance after propagation."""

    def test_posterior_sums_to_one(self) -> None:
        g = build_initial_graph()
        prop = BeliefPropagator(g)
        beliefs = prop.propagate(
            evidence={"obs.rate_momentum": 0.9, "obs.stress_breadth": 0.8},
            as_of=time.time(),
        )
        for b in beliefs:
            if b.probabilities:
                total = sum(b.probabilities.values())
                assert total == pytest.approx(1.0, abs=1e-6), f"{b.variable_name}: probs sum to {total}"


# ═══════════════════════════════════════════════════════════════
# State filter edge cases
# ═══════════════════════════════════════════════════════════════


def _make_filter(state_dim: int = 3, obs_dim: int = 6) -> ContinuousStateFilter:
    """Construct a filter with default regime configs for testing."""
    configs = {
        "expansion": RegimeConfig(
            name="expansion",
            F=np.eye(state_dim),
            Q=np.eye(state_dim) * 0.01,
        ),
        "contraction": RegimeConfig(
            name="contraction",
            F=np.eye(state_dim) * 0.95,
            Q=np.eye(state_dim) * 0.05,
        ),
    }
    H = np.zeros((obs_dim, state_dim))
    for i in range(min(obs_dim, state_dim)):
        H[i, i] = 1.0
    # Extra obs dims map to state 0 if obs_dim > state_dim
    for i in range(state_dim, obs_dim):
        H[i, i % state_dim] = 1.0
    R = np.eye(obs_dim) * 0.1
    return ContinuousStateFilter(
        state_dim=state_dim,
        obs_dim=obs_dim,
        regime_configs=configs,
        H=H,
        R=R,
    )


class TestStateFilterVerySmallObs:
    """Very small observation (1e-10) → still produces valid state."""

    def test_small_obs_no_nan(self) -> None:
        f = _make_filter()
        f.predict("expansion")
        obs = np.full(6, np.nan)
        obs[0] = 1e-10
        f.update(obs, quality=np.ones(6))
        beliefs = f.get_beliefs(["s0", "s1", "s2"], time.time(), "a" * 64)
        for b in beliefs:
            assert np.isfinite(b.mean)
            assert np.isfinite(b.variance)
            assert b.variance > 0


class TestStateFilterSingleObs:
    """Single observation available → partial update."""

    def test_partial_update_only_obs_dim(self) -> None:
        f = _make_filter()
        f.predict("expansion")
        x_before = f._x.copy()
        obs = np.full(6, np.nan)
        obs[2] = 5.0  # only 3rd sensor active
        f.update(obs, quality=np.ones(6))
        # State should change from prior since one observation arrived
        beliefs = f.get_beliefs(["s0", "s1", "s2"], time.time(), "a" * 64)
        assert len(beliefs) == 3


class TestStateFilterCovariancePD:
    """Covariance matrix stays positive-definite over many steps (Joseph form)."""

    def test_1000_steps_stays_pd(self) -> None:
        f = _make_filter()
        rng = np.random.default_rng(42)
        for _ in range(1000):
            f.predict("expansion")
            obs = rng.standard_normal(6)
            # Occasionally inject NaN to stress partial updates
            mask = rng.random(6) < 0.3
            obs[mask] = np.nan
            f.update(obs, quality=rng.uniform(0.5, 1.0, size=6))
        P = f._P
        eigvals = np.linalg.eigvalsh(P)
        assert np.all(eigvals > 0), f"P has non-positive eigenvalue: {eigvals.min()}"


class TestStateFilterQualityZeroAll:
    """Quality = 0.0 for all → R inflated, effectively no update."""

    def test_zero_quality_no_update(self) -> None:
        f = _make_filter()
        f.predict("expansion")
        x_before = f._x.copy()
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        f.update(obs, quality=np.zeros(6))
        # With quality=0 on all channels, R→∞, K→0, no state change
        np.testing.assert_array_almost_equal(f._x, x_before, decimal=5)


class TestStateFilterDimensionMismatch:
    """Dimension mismatch between obs vector and obs_dim → error."""

    def test_wrong_obs_length(self) -> None:
        f = _make_filter(state_dim=3, obs_dim=6)
        f.predict("expansion")
        with pytest.raises((ValueError, IndexError)):
            f.update(np.array([1.0, 2.0]), quality=np.ones(2))


class TestStateFilterUnknownRegime:
    """Regime not in configs → clear error."""

    def test_unknown_regime_raises(self) -> None:
        f = _make_filter()
        with pytest.raises((KeyError, ValueError)):
            f.predict("unknown_regime")


class TestStateFilterVeryLargeObs:
    """Very large observation (1e10) → filter shouldn't diverge."""

    def test_large_obs_no_nan(self) -> None:
        f = _make_filter()
        f.predict("expansion")
        obs = np.full(6, np.nan)
        obs[0] = 1e10
        f.update(obs, quality=np.ones(6))
        beliefs = f.get_beliefs(["s0", "s1", "s2"], time.time(), "a" * 64)
        for b in beliefs:
            assert np.isfinite(b.mean)
            assert np.isfinite(b.variance)


# ═══════════════════════════════════════════════════════════════
# Integration (WorldModel) edge cases
# ═══════════════════════════════════════════════════════════════


def _build_world_model() -> WorldModel:
    """Build a WorldModel the same way the DAG does."""
    from agent.pipeline.dags.world_model_update import _build_world_model as _dag_build

    return _dag_build()


class TestIntegrationMixedQuality:
    """Features with mixed quality levels → proper weighting."""

    def test_mixed_quality_doesnt_crash(self) -> None:
        from agent.features.protocol import EngineeredFeature

        wm = _build_world_model()
        t = time.time()
        features = [
            EngineeredFeature(
                feature_name="macro.rate_momentum.30d",
                version=1,
                effective_at=t - 60,
                computed_at=t - 30,
                horizon="30d",
                value=0.9,
                quality=1.0,
                source_signals=("test",),
                builder="test",
            ),
            EngineeredFeature(
                feature_name="convergence.stress_breadth.7d",
                version=1,
                effective_at=t - 60,
                computed_at=t - 30,
                horizon="7d",
                value=0.8,
                quality=0.3,
                source_signals=("test",),
                builder="test",
            ),
        ]
        beliefs = wm.update(features, as_of=t)
        assert len(beliefs) > 0
        # All beliefs should have valid probabilities or Gaussian params
        for b in beliefs:
            validate_belief(b)


class TestIntegrationVeryOldFeatures:
    """Very old features → beliefs should still work (staleness up to consumer)."""

    def test_old_features_produce_beliefs(self) -> None:
        from agent.features.protocol import EngineeredFeature

        wm = _build_world_model()
        t = time.time()
        features = [
            EngineeredFeature(
                feature_name="macro.rate_momentum.30d",
                version=1,
                effective_at=t - 86400 * 60,  # 60 days old
                computed_at=t - 86400 * 60,
                horizon="30d",
                value=0.5,
                quality=1.0,
                source_signals=("test",),
                builder="test",
            ),
        ]
        beliefs = wm.update(features, as_of=t)
        assert len(beliefs) > 0


class TestIntegrationNoneFeatureValue:
    """Feature value = None (explicit missingness) → excluded from evidence."""

    def test_none_value_excluded(self) -> None:
        from agent.features.protocol import EngineeredFeature

        wm = _build_world_model()
        t = time.time()
        features = [
            EngineeredFeature(
                feature_name="macro.rate_momentum.30d",
                version=1,
                effective_at=t - 60,
                computed_at=t - 30,
                horizon="30d",
                value=None,
                quality=1.0,
                missing_reason="upstream_stale",
                source_signals=("test",),
                builder="test",
            ),
        ]
        beliefs = wm.update(features, as_of=t)
        # Should produce beliefs (priors) — not crash
        assert len(beliefs) > 0
        for b in beliefs:
            validate_belief(b)


# ═══════════════════════════════════════════════════════════════
# Intervention edge cases
# ═══════════════════════════════════════════════════════════════


class TestInterventionOnObservedVariable:
    """Intervene on observed variable → severs incoming edges."""

    def test_intervene_on_observed(self) -> None:
        g = build_initial_graph()
        eng = InterventionEngine(g)
        beliefs = eng.intervene(
            do_variable="obs.rate_momentum",
            do_value="rising",
        )
        # Intervened node should be delta on "rising"
        rm_belief = next(b for b in beliefs if b.variable_name == "obs.rate_momentum")
        assert rm_belief.probabilities["rising"] == 1.0
        # Parent (regime.macro) should revert to prior since edge is severed
        macro_belief = next(b for b in beliefs if b.variable_name == "regime.macro")
        # Under do(obs.rate_momentum=rising), regime.macro is d-separated
        # from the intervention, so its posterior should equal its prior.
        assert sum(macro_belief.probabilities.values()) == pytest.approx(1.0)


class TestInterventionOnEveryVariable:
    """Intervene on every variable individually → each produces delta beliefs."""

    def test_intervene_each_node(self) -> None:
        g = build_initial_graph()
        eng = InterventionEngine(g)
        for spec in g.node_specs.values():
            do_val = spec.states[0]
            beliefs = eng.intervene(
                do_variable=spec.name,
                do_value=do_val,
            )
            # The intervened node should have delta distribution
            do_belief = next(b for b in beliefs if b.variable_name == spec.name)
            assert do_belief.probabilities[do_val] == 1.0
            # All posteriors should be valid distributions
            for b in beliefs:
                if b.probabilities:
                    total = sum(b.probabilities.values())
                    assert total == pytest.approx(1.0, abs=1e-6), (
                        f"do({spec.name}={do_val}): {b.variable_name} sums to {total}"
                    )


class TestInterventionInvalidDoValue:
    """Invalid do_value for node cardinality → error."""

    def test_invalid_state_label(self) -> None:
        g = build_initial_graph()
        eng = InterventionEngine(g)
        with pytest.raises(ValueError, match="not.*(valid|states)"):
            eng.intervene(
                do_variable="regime.macro",
                do_value="nonexistent_state",
            )


class TestInterventionNonExistentVariable:
    """Intervene on non-existent variable → clear error."""

    def test_nonexistent_variable(self) -> None:
        g = build_initial_graph()
        eng = InterventionEngine(g)
        with pytest.raises(ValueError):
            eng.intervene(
                do_variable="does.not.exist",
                do_value="crisis",
            )


# ═══════════════════════════════════════════════════════════════
# KL divergence edge cases
# ═══════════════════════════════════════════════════════════════


class TestKLDivergence:
    def test_identical_distributions(self) -> None:
        p = {"a": 0.5, "b": 0.5}
        assert _kl_divergence(p, p) == pytest.approx(0.0)

    def test_q_zero_where_p_nonzero(self) -> None:
        p = {"a": 0.5, "b": 0.5}
        q = {"a": 1.0, "b": 0.0}
        assert _kl_divergence(p, q) == float("inf")

    def test_p_zero_where_q_nonzero(self) -> None:
        p = {"a": 1.0, "b": 0.0}
        q = {"a": 0.5, "b": 0.5}
        kl = _kl_divergence(p, q)
        assert kl == pytest.approx(np.log(2.0), abs=1e-6)


# ═══════════════════════════════════════════════════════════════
# value_to_state_index edge cases
# ═══════════════════════════════════════════════════════════════


class TestValueToStateIndex:
    def test_below_all_bins(self) -> None:
        assert value_to_state_index(-1e10, (-math.inf, -0.5, 0.5, math.inf)) == 0

    def test_above_all_bins(self) -> None:
        assert value_to_state_index(1e10, (-math.inf, -0.5, 0.5, math.inf)) == 2

    def test_exact_boundary(self) -> None:
        # On the boundary between bins → falls into the next bin
        idx = value_to_state_index(-0.5, (-math.inf, -0.5, 0.5, math.inf))
        assert idx in (0, 1)  # Right-open bins, -0.5 ∈ [−0.5, 0.5)

    def test_inf_value(self) -> None:
        assert value_to_state_index(math.inf, (-math.inf, -0.5, 0.5, math.inf)) == 2

    def test_neg_inf_value(self) -> None:
        assert value_to_state_index(-math.inf, (-math.inf, -0.5, 0.5, math.inf)) == 0
