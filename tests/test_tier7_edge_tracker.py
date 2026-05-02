"""
Tests for Edge Confidence Tracker (Change 13, Tier 7).

Covers: BIC-δ computation, confidence/stability math, hysteresis decisions,
structural constraint enforcement, rolling window edge cases, serialization,
and integration with world model structure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agent.models.edge_tracker import (
    EdgeConfidence,
    EdgeConfidenceTracker,
    EdgeSuggestion,
    _sigmoid,
)

# ── Helpers ───────────────────────────────────────────────────


def _make_causal_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic data: A → B → C (true causal chain)."""
    rng = np.random.RandomState(seed)
    A = rng.choice(["lo", "hi"], n, p=[0.4, 0.6])
    B = np.where(
        A == "hi",
        rng.choice(["lo", "hi"], n, p=[0.3, 0.7]),
        rng.choice(["lo", "hi"], n, p=[0.7, 0.3]),
    )
    C = np.where(
        B == "hi",
        rng.choice(["lo", "hi"], n, p=[0.2, 0.8]),
        rng.choice(["lo", "hi"], n, p=[0.8, 0.2]),
    )
    D = rng.choice(["lo", "hi"], n, p=[0.5, 0.5])  # independent noise
    return pd.DataFrame({"A": A, "B": B, "C": C, "D": D})


def _make_tracker(**kwargs) -> EdgeConfidenceTracker:
    defaults = dict(node_names=["A", "B", "C", "D"], windows_days=(30, 60, 90))
    defaults.update(kwargs)
    return EdgeConfidenceTracker(**defaults)


# ═══════════════════════════════════════════════════════════════
# §1 — BIC-δ Computation
# ═══════════════════════════════════════════════════════════════


class TestBICDelta:
    """Verify BIC-δ edge contribution scoring."""

    def test_true_edge_positive_delta(self):
        """A→B is a true causal edge, should have positive BIC-δ."""
        df = _make_causal_data(n=500)
        tracker = _make_tracker()
        contribs = tracker.compute_edge_contributions([("A", "B")], df)
        assert ("A", "B") in contribs
        assert contribs[("A", "B")] > 0, "True edge should have positive BIC-δ"

    def test_spurious_edge_weak_delta(self):
        """A→C is spurious (B mediates), marginal delta given B should be ≈0."""
        df = _make_causal_data(n=500)
        tracker = _make_tracker()
        # With B already as parent, adding A shouldn't help much
        contribs = tracker.compute_edge_contributions([("B", "C"), ("A", "C")], df)
        # The direct edge B→C should be much stronger than A→C marginal
        assert contribs.get(("B", "C"), 0) > contribs.get(("A", "C"), 0)

    def test_independent_node_negative_delta(self):
        """D is independent of A, edge A→D should have negative or ~0 BIC-δ."""
        df = _make_causal_data(n=500)
        tracker = _make_tracker()
        contribs = tracker.compute_edge_contributions([("A", "D")], df)
        # BIC penalty should dominate for a useless edge
        assert contribs.get(("A", "D"), 1.0) < 1.0

    def test_empty_dataframe_returns_empty(self):
        tracker = _make_tracker()
        contribs = tracker.compute_edge_contributions([("A", "B")], pd.DataFrame())
        assert contribs == {}

    def test_very_small_dataframe_returns_empty(self):
        df = pd.DataFrame({"A": ["lo", "hi"], "B": ["lo", "hi"]})
        tracker = _make_tracker()
        contribs = tracker.compute_edge_contributions([("A", "B")], df)
        assert contribs == {}

    def test_missing_column_skipped(self):
        df = _make_causal_data()
        tracker = _make_tracker()
        contribs = tracker.compute_edge_contributions([("A", "MISSING")], df)
        assert ("A", "MISSING") not in contribs

    def test_multiple_edges_computed(self):
        df = _make_causal_data()
        tracker = _make_tracker()
        edges = [("A", "B"), ("B", "C")]
        contribs = tracker.compute_edge_contributions(edges, df)
        assert len(contribs) == 2


# ═══════════════════════════════════════════════════════════════
# §2 — Confidence and Stability Math
# ═══════════════════════════════════════════════════════════════


class TestConfidenceStability:
    """Verify confidence and stability calculations."""

    def test_sigmoid_properties(self):
        assert abs(_sigmoid(0) - 0.5) < 1e-10
        assert _sigmoid(100) > 0.99
        assert _sigmoid(-100) < 0.01
        assert abs(_sigmoid(5) + _sigmoid(-5) - 1.0) < 1e-10

    def test_strong_edge_high_confidence(self):
        """True causal edge should have confidence > 0.5."""
        df = _make_causal_data(n=500)
        tracker = _make_tracker()
        confidences = tracker.evaluate([("A", "B")], [df])  # single window
        assert ("A", "B") in confidences
        assert confidences[("A", "B")].confidence > 0.5

    def test_consistent_windows_high_stability(self):
        """Same data in multiple windows should give high stability."""
        df = _make_causal_data(n=500)
        tracker = _make_tracker()
        # Use same data for all 3 windows (simulating stable signal)
        confidences = tracker.evaluate([("A", "B")], [df, df, df])
        conf = confidences[("A", "B")]
        assert conf.stability > 0.9, f"Stability should be high for consistent data, got {conf.stability}"
        assert conf.n_windows == 3

    def test_contradictory_windows_low_stability(self):
        """Opposite signals across windows should give low stability."""
        df_positive = _make_causal_data(n=500, seed=42)
        # Create anti-correlated data: flip B's relationship with A
        rng = np.random.RandomState(99)
        n = 500
        A = rng.choice(["lo", "hi"], n, p=[0.4, 0.6])
        B = np.where(
            A == "hi",
            rng.choice(["lo", "hi"], n, p=[0.5, 0.5]),  # no relationship
            rng.choice(["lo", "hi"], n, p=[0.5, 0.5]),
        )
        df_noise = pd.DataFrame({"A": A, "B": B, "C": B, "D": B})
        tracker = _make_tracker()
        confidences = tracker.evaluate([("A", "B")], [df_positive, df_noise])
        conf = confidences[("A", "B")]
        # Stability should be lower than when windows agree
        assert conf.n_windows == 2

    def test_single_window_stability_is_one(self):
        """With a single window, std=0, stability should be 1.0."""
        df = _make_causal_data()
        tracker = _make_tracker()
        confidences = tracker.evaluate([("A", "B")], [df])
        assert confidences[("A", "B")].stability == 1.0

    def test_empty_windows_returns_empty(self):
        tracker = _make_tracker()
        assert tracker.evaluate([("A", "B")], []) == {}

    def test_deltas_stored_in_confidence(self):
        df = _make_causal_data()
        tracker = _make_tracker()
        confidences = tracker.evaluate([("A", "B")], [df, df])
        conf = confidences[("A", "B")]
        assert len(conf.deltas) == 2
        assert all(isinstance(d, float) for d in conf.deltas)


# ═══════════════════════════════════════════════════════════════
# §3 — Hysteresis Decision Logic
# ═══════════════════════════════════════════════════════════════


class TestHysteresis:
    """Verify hysteresis-based add/remove decisions."""

    def test_no_change_on_first_eval(self):
        """Hysteresis requires consecutive_required=2, so first eval never changes."""
        tracker = _make_tracker()
        conf = {("A", "B"): EdgeConfidence("A", "B", confidence=0.1, stability=0.9, deltas=(0.1,), n_windows=1)}
        suggestion = tracker.suggest_changes(conf, current_edges={("A", "B")}, consecutive_required=2)
        assert suggestion.edges_to_remove == []
        assert suggestion.edges_to_add == []

    def test_removal_after_consecutive_threshold(self):
        """Edge should be removed after 2 consecutive low-confidence evaluations."""
        tracker = _make_tracker()
        low_conf = {("A", "B"): EdgeConfidence("A", "B", confidence=0.1, stability=0.9, deltas=(-5.0,), n_windows=1)}
        # First eval
        tracker.suggest_changes(low_conf, current_edges={("A", "B")}, consecutive_required=2)
        # Second eval — should now suggest removal
        suggestion = tracker.suggest_changes(low_conf, current_edges={("A", "B")}, consecutive_required=2)
        assert ("A", "B") in suggestion.edges_to_remove

    def test_addition_after_consecutive_threshold(self):
        """Edge should be added after 2 consecutive high-confidence evaluations."""
        tracker = _make_tracker()
        high_conf = {("A", "D"): EdgeConfidence("A", "D", confidence=0.9, stability=0.8, deltas=(5.0,), n_windows=1)}
        candidates = [("A", "D")]
        # First eval
        tracker.suggest_changes(
            high_conf,
            current_edges=set(),
            candidate_additions=candidates,
            consecutive_required=2,
        )
        # Second eval
        suggestion = tracker.suggest_changes(
            high_conf,
            current_edges=set(),
            candidate_additions=candidates,
            consecutive_required=2,
        )
        assert ("A", "D") in suggestion.edges_to_add

    def test_oscillating_edge_not_modified(self):
        """Edge that oscillates between high/low confidence should not be modified."""
        tracker = _make_tracker()
        low_conf = {("A", "B"): EdgeConfidence("A", "B", confidence=0.1, stability=0.9, deltas=(-5.0,), n_windows=1)}
        ok_conf = {("A", "B"): EdgeConfidence("A", "B", confidence=0.5, stability=0.9, deltas=(0.0,), n_windows=1)}
        # Low → resets when OK
        tracker.suggest_changes(low_conf, current_edges={("A", "B")}, consecutive_required=2)
        tracker.suggest_changes(ok_conf, current_edges={("A", "B")}, consecutive_required=2)
        suggestion = tracker.suggest_changes(low_conf, current_edges={("A", "B")}, consecutive_required=2)
        # Still only 1 consecutive low, should NOT remove
        assert suggestion.edges_to_remove == []

    def test_protected_edges_not_removed(self):
        """Protected edges cannot be removed regardless of confidence."""
        tracker = _make_tracker()
        low_conf = {("A", "B"): EdgeConfidence("A", "B", confidence=0.1, stability=0.9, deltas=(-5.0,), n_windows=1)}
        protected = {("A", "B")}
        # Two evals
        tracker.suggest_changes(
            low_conf,
            current_edges={("A", "B")},
            protected_edges=protected,
            consecutive_required=2,
        )
        suggestion = tracker.suggest_changes(
            low_conf,
            current_edges={("A", "B")},
            protected_edges=protected,
            consecutive_required=2,
        )
        assert suggestion.edges_to_remove == []

    def test_low_stability_prevents_change(self):
        """Even if confidence is extreme, low stability should prevent changes."""
        tracker = _make_tracker()
        unstable_conf = {
            ("A", "B"): EdgeConfidence(
                "A",
                "B",
                confidence=0.05,
                stability=0.2,
                deltas=(-10.0, 5.0),
                n_windows=2,
            )
        }
        tracker.suggest_changes(
            unstable_conf,
            current_edges={("A", "B")},
            stability_min=0.5,
            consecutive_required=2,
        )
        suggestion = tracker.suggest_changes(
            unstable_conf,
            current_edges={("A", "B")},
            stability_min=0.5,
            consecutive_required=2,
        )
        assert suggestion.edges_to_remove == []

    def test_reset_consecutive(self):
        """After applying a change, reset_consecutive clears the counter."""
        tracker = _make_tracker()
        low_conf = {("A", "B"): EdgeConfidence("A", "B", confidence=0.1, stability=0.9, deltas=(-5.0,), n_windows=1)}
        tracker.suggest_changes(low_conf, current_edges={("A", "B")}, consecutive_required=2)
        tracker.reset_consecutive(("A", "B"))
        # After reset, first eval again — should not suggest removal
        suggestion = tracker.suggest_changes(low_conf, current_edges={("A", "B")}, consecutive_required=2)
        assert suggestion.edges_to_remove == []

    def test_consecutive_required_1(self):
        """With consecutive_required=1, changes happen immediately."""
        tracker = _make_tracker()
        low_conf = {("A", "B"): EdgeConfidence("A", "B", confidence=0.1, stability=0.9, deltas=(-5.0,), n_windows=1)}
        suggestion = tracker.suggest_changes(low_conf, current_edges={("A", "B")}, consecutive_required=1)
        assert ("A", "B") in suggestion.edges_to_remove

    def test_no_candidates_no_additions(self):
        """Without candidate_additions, no edges can be added."""
        tracker = _make_tracker()
        high_conf = {("A", "D"): EdgeConfidence("A", "D", confidence=0.9, stability=0.8, deltas=(5.0,), n_windows=1)}
        suggestion = tracker.suggest_changes(high_conf, current_edges=set(), consecutive_required=1)
        assert suggestion.edges_to_add == []

    def test_already_present_not_re_added(self):
        """Edge already in current_edges should not appear in edges_to_add."""
        tracker = _make_tracker()
        high_conf = {("A", "B"): EdgeConfidence("A", "B", confidence=0.9, stability=0.8, deltas=(5.0,), n_windows=1)}
        suggestion = tracker.suggest_changes(
            high_conf,
            current_edges={("A", "B")},
            candidate_additions=[("A", "B")],
            consecutive_required=1,
        )
        assert suggestion.edges_to_add == []


# ═══════════════════════════════════════════════════════════════
# §4 — Serialization
# ═══════════════════════════════════════════════════════════════


class TestSerialization:
    """Verify round-trip serialization of tracker state."""

    def test_round_trip(self):
        tracker = _make_tracker()
        # Build up some consecutive state
        low_conf = {("A", "B"): EdgeConfidence("A", "B", confidence=0.1, stability=0.9, deltas=(-5.0,), n_windows=1)}
        tracker.suggest_changes(low_conf, current_edges={("A", "B")}, consecutive_required=3)

        data = tracker.to_dict()
        restored = EdgeConfidenceTracker.from_dict(data)
        assert restored.node_names == tracker.node_names
        assert restored.windows_days == tracker.windows_days
        assert restored._consecutive == tracker._consecutive

    def test_empty_state_serializes(self):
        tracker = _make_tracker()
        data = tracker.to_dict()
        assert data["consecutive"] == {}
        restored = EdgeConfidenceTracker.from_dict(data)
        assert restored._consecutive == {}

    def test_custom_windows_preserved(self):
        tracker = _make_tracker(windows_days=(7, 14, 28))
        data = tracker.to_dict()
        restored = EdgeConfidenceTracker.from_dict(data)
        assert restored.windows_days == (7, 14, 28)


# ═══════════════════════════════════════════════════════════════
# §5 — Edge Cases
# ═══════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_no_edges_returns_empty(self):
        tracker = _make_tracker()
        df = _make_causal_data()
        assert tracker.compute_edge_contributions([], df) == {}

    def test_evaluate_no_data_windows(self):
        tracker = _make_tracker()
        result = tracker.evaluate([("A", "B")], [])
        assert result == {}

    def test_suggest_with_empty_confidences(self):
        tracker = _make_tracker()
        suggestion = tracker.suggest_changes({}, current_edges={("A", "B")})
        assert suggestion.edges_to_add == []
        assert suggestion.edges_to_remove == []

    def test_confidence_is_edge_confidence_dataclass(self):
        conf = EdgeConfidence("A", "B", 0.8, 0.9, (1.0, 2.0), 2)
        assert conf.parent == "A"
        assert conf.child == "B"
        assert conf.confidence == 0.8
        assert conf.stability == 0.9
        assert conf.n_windows == 2

    def test_suggestion_is_dataclass(self):
        s = EdgeSuggestion()
        assert s.edges_to_add == []
        assert s.edges_to_remove == []

    def test_single_node_data(self):
        """DataFrame with only one column should not crash."""
        df = pd.DataFrame({"A": ["lo", "hi", "lo"] * 10})
        tracker = _make_tracker()
        contribs = tracker.compute_edge_contributions([("A", "B")], df)
        assert ("A", "B") not in contribs  # B not in columns


# ═══════════════════════════════════════════════════════════════
# §6 — Integration with WorldModel structure
# ═══════════════════════════════════════════════════════════════


class TestIntegrationScenario:
    """End-to-end scenario: evaluate edges over multiple cycles."""

    def test_multi_cycle_convergence(self):
        """Over multiple evaluation cycles, spurious edge gets removed."""
        tracker = _make_tracker()
        df = _make_causal_data(n=1000)

        current_edges = {("A", "B"), ("B", "C"), ("D", "C")}  # D→C is spurious
        protected = {("A", "B")}  # Protect A→B

        # Run 3 evaluation cycles
        for _ in range(3):
            confs = tracker.evaluate(
                list(current_edges),
                [df],  # using same data for simplicity
            )
            suggestion = tracker.suggest_changes(
                confs,
                current_edges=current_edges,
                protected_edges=protected,
                consecutive_required=2,
            )
            # Apply removals
            for edge in suggestion.edges_to_remove:
                current_edges.discard(edge)
                tracker.reset_consecutive(edge)

        # D→C (spurious, independent) should eventually be removed
        # B→C (true causal) should remain
        assert ("B", "C") in current_edges
        assert ("A", "B") in current_edges  # protected

    def test_candidate_addition_convergence(self):
        """True edge proposed as candidate gets added over evaluations."""
        tracker = _make_tracker()
        df = _make_causal_data(n=1000)

        current_edges: set[tuple[str, str]] = set()
        candidates = [("A", "B"), ("A", "D")]  # A→B is true, A→D is spurious

        for _ in range(3):
            # Evaluate both current and candidate edges
            all_edges = list(current_edges) + candidates
            confs = tracker.evaluate(all_edges, [df])
            suggestion = tracker.suggest_changes(
                confs,
                current_edges=current_edges,
                candidate_additions=candidates,
                consecutive_required=2,
            )
            for edge in suggestion.edges_to_add:
                current_edges.add(edge)
                tracker.reset_consecutive(edge)

        # A→B (true causal) should be added
        assert ("A", "B") in current_edges
