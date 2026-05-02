"""Tests for false discovery rate control (Phase 7c-C.3).

Covers: apply_bh_correction (known rejection sets, all-pass, all-fail,
single p-value, empty), fisher_combined_test (known pairs, single,
empty, p=0 guard, p=1), persistence_filter (first cycle, second cycle,
disappearance, min_periods=1), cross_category_filter (pass, fail,
boundary), apply_all_controls (full synthetic pipeline — 1 real
convergence + spurious → only real survives).
"""

from __future__ import annotations

import math

import pytest

from agent.convergence.coincidence import CoincidenceResult
from agent.convergence.fdr import (
    apply_all_controls,
    apply_bh_correction,
    cross_category_filter,
    fisher_combined_test,
    persistence_filter,
)
from agent.convergence.graph import ConvergenceClique

# ── Helpers ────────────────────────────────────────────────────


def _cr(
    score: float = 2.0,
    p_value: float = 0.01,
    direction: int = 1,
    method: str = "combined",
) -> CoincidenceResult:
    return CoincidenceResult(
        method=method,
        score=score,
        p_value=p_value,
        direction=direction,
    )


def _clique(
    signals: list[str],
    categories: list[str],
    p_values: list[float] | None = None,
    score: float = 0.5,
) -> ConvergenceClique:
    """Build a minimal clique for testing."""
    edges = []
    pvs = p_values or []
    idx = 0
    for i, sa in enumerate(signals):
        for sb in signals[i + 1 :]:
            pv = pvs[idx] if idx < len(pvs) else 0.01
            edges.append((sa, sb, 2.0))
            idx += 1
    return ConvergenceClique(
        signals=sorted(signals),
        categories=sorted(categories),
        edges=edges,
        score=score,
        p_values=pvs if pvs else [0.01] * len(edges),
    )


# ═══════════════════════════════════════════════════════════════
# BH Correction
# ═══════════════════════════════════════════════════════════════


class TestBHCorrection:
    """Benjamini-Hochberg false discovery rate control."""

    def test_known_rejection_set(self):
        """Hand-verified: with q=0.05 and 5 sorted p-values, only the
        smallest should survive BH."""
        pairs = {
            ("a", "b"): 0.001,
            ("a", "c"): 0.01,
            ("a", "d"): 0.04,
            ("b", "c"): 0.20,
            ("b", "d"): 0.80,
        }
        result = apply_bh_correction(pairs, q=0.05)
        # BH: p_{(i)} ≤ (i/m)*q → thresholds 0.01, 0.02, 0.03, 0.04, 0.05
        # 0.001 ≤ 0.01 ✓, 0.01 ≤ 0.02 ✓, 0.04 > 0.03 ✗ → reject first 2
        # statsmodels step-up: largest k where p(k) ≤ k/m*q → k=2
        assert result[("a", "b")] is True
        assert result[("a", "c")] is True
        assert result[("a", "d")] is False
        assert result[("b", "c")] is False
        assert result[("b", "d")] is False

    def test_all_pass(self):
        """All p-values = 0.5 → none rejected at q=0.05."""
        pairs = {("a", "b"): 0.5, ("c", "d"): 0.5, ("e", "f"): 0.5}
        result = apply_bh_correction(pairs, q=0.05)
        assert not any(result.values())

    def test_all_rejected(self):
        """All p-values = 0.001 → all rejected at q=0.05."""
        pairs = {("a", "b"): 0.001, ("c", "d"): 0.001, ("e", "f"): 0.001}
        result = apply_bh_correction(pairs, q=0.05)
        assert all(result.values())

    def test_single_p_highly_significant(self):
        """Single p-value well below q → rejected."""
        pairs = {("x", "y"): 0.001}
        result = apply_bh_correction(pairs, q=0.05)
        assert result[("x", "y")] is True

    def test_single_p_not_significant(self):
        """Single p-value above q → not rejected."""
        pairs = {("x", "y"): 0.10}
        result = apply_bh_correction(pairs, q=0.05)
        assert result[("x", "y")] is False

    def test_empty_input(self):
        result = apply_bh_correction({}, q=0.05)
        assert result == {}

    def test_preserves_pair_keys(self):
        """Output keys must exactly match input keys."""
        pairs = {("sig1", "sig2"): 0.001, ("sig3", "sig4"): 0.9}
        result = apply_bh_correction(pairs, q=0.05)
        assert set(result.keys()) == set(pairs.keys())

    def test_q_one_accepts_everything(self):
        """FDR level q=1.0 means accept all hypotheses."""
        pairs = {("a", "b"): 0.99, ("c", "d"): 0.95}
        result = apply_bh_correction(pairs, q=1.0)
        assert all(result.values())

    def test_p_at_exact_threshold(self):
        """BH threshold check at exact boundary: p_{(1)} vs (1/1)*q."""
        pairs = {("a", "b"): 0.05}
        result = apply_bh_correction(pairs, q=0.05)
        assert result[("a", "b")] is True

    def test_returns_bool_type(self):
        pairs = {("a", "b"): 0.001}
        result = apply_bh_correction(pairs, q=0.05)
        assert isinstance(result[("a", "b")], bool)


# ═══════════════════════════════════════════════════════════════
# Fisher's Combined Test
# ═══════════════════════════════════════════════════════════════


class TestFisherCombinedTest:
    """Fisher's method for combining p-values."""

    def test_two_small_p_values(self):
        """Two p-values of 0.03 → combined p should be < 0.01."""
        combined = fisher_combined_test([0.03, 0.03])
        assert combined < 0.01

    def test_two_nonsignificant(self):
        """Two p-values of 0.5 → combined p near 0.5."""
        combined = fisher_combined_test([0.5, 0.5])
        # χ² = -2*(ln(0.5)+ln(0.5)) ≈ 2.773, df=4
        # sf(2.773, 4) ≈ 0.597
        assert 0.3 < combined < 0.8

    def test_single_p_value(self):
        """With k=1, Fisher's should return the p-value itself."""
        assert fisher_combined_test([0.03]) == pytest.approx(0.03, rel=1e-6)

    def test_empty_returns_one(self):
        assert fisher_combined_test([]) == 1.0

    def test_p_value_zero_clipped(self):
        """p=0.0 must be clipped to _P_FLOOR, not produce -inf."""
        combined = fisher_combined_test([0.0, 0.05])
        assert 0.0 < combined < 1.0
        assert not math.isnan(combined)
        assert not math.isinf(combined)

    def test_p_value_one(self):
        """p=1.0 contributes no evidence: ln(1)=0."""
        combined = fisher_combined_test([1.0, 1.0])
        # χ² = 0, df=4 → sf(0, 4) = 1.0
        assert combined == pytest.approx(1.0)

    def test_many_small_p_values(self):
        """10 p-values all at 0.04 → combined is extremely significant."""
        combined = fisher_combined_test([0.04] * 10)
        assert combined < 1e-5

    def test_mixed_values(self):
        """Mix of strong and weak p-values."""
        combined = fisher_combined_test([0.001, 0.5, 0.8, 0.01])
        # Strong evidence from two p-values should dominate.
        assert combined < 0.01

    def test_result_in_valid_range(self):
        """Combined p-value must be in (0, 1]."""
        for ps in [[0.01, 0.02], [0.5, 0.5], [1e-10, 1e-10]]:
            c = fisher_combined_test(ps)
            assert 0.0 < c <= 1.0


# ═══════════════════════════════════════════════════════════════
# Persistence Filter
# ═══════════════════════════════════════════════════════════════


class TestPersistenceFilter:
    """Convergence must persist for N consecutive cycles."""

    def test_first_cycle_not_emitted(self):
        """First detection: recorded in history but not emitted."""
        history: dict[tuple[str, ...], int] = {}
        c = _clique(["a", "b", "c"], ["cat1", "cat2"])
        survivors = persistence_filter([c], history, min_periods=2)
        assert len(survivors) == 0
        assert history[c.fingerprint()] == 1

    def test_second_cycle_emitted(self):
        """Second consecutive detection: emitted."""
        c = _clique(["a", "b", "c"], ["cat1", "cat2"])
        history: dict[tuple[str, ...], int] = {c.fingerprint(): 1}
        survivors = persistence_filter([c], history, min_periods=2)
        assert len(survivors) == 1
        assert history[c.fingerprint()] == 2

    def test_disappearance_resets_counter(self):
        """If a clique disappears, its counter is removed."""
        c = _clique(["a", "b", "c"], ["cat1", "cat2"])
        history: dict[tuple[str, ...], int] = {c.fingerprint(): 3}
        # Pass empty events (clique disappeared).
        survivors = persistence_filter([], history, min_periods=2)
        assert len(survivors) == 0
        assert c.fingerprint() not in history

    def test_reappearance_requires_full_persistence(self):
        """After disappearance + reset, needs min_periods again."""
        c = _clique(["a", "b", "c"], ["cat1", "cat2"])
        history: dict[tuple[str, ...], int] = {}

        # Cycle 1: appears.
        persistence_filter([c], history, min_periods=2)
        assert history[c.fingerprint()] == 1

        # Cycle 2: disappears.
        persistence_filter([], history, min_periods=2)
        assert c.fingerprint() not in history

        # Cycle 3: reappears — count restarts at 1.
        persistence_filter([c], history, min_periods=2)
        assert history[c.fingerprint()] == 1

    def test_min_periods_one_emits_immediately(self):
        """min_periods=1 → emit on first detection."""
        history: dict[tuple[str, ...], int] = {}
        c = _clique(["a", "b", "c"], ["cat1", "cat2"])
        survivors = persistence_filter([c], history, min_periods=1)
        assert len(survivors) == 1

    def test_empty_events_and_empty_history(self):
        history: dict[tuple[str, ...], int] = {}
        survivors = persistence_filter([], history, min_periods=2)
        assert survivors == []
        assert history == {}

    def test_multiple_cliques_independent_tracking(self):
        """Two cliques tracked independently."""
        c1 = _clique(["a", "b", "c"], ["cat1", "cat2"])
        c2 = _clique(["d", "e", "f"], ["cat3", "cat4"])
        history: dict[tuple[str, ...], int] = {c1.fingerprint(): 1}

        survivors = persistence_filter([c1, c2], history, min_periods=2)
        # c1 at count 2 → emitted, c2 at count 1 → not emitted.
        assert len(survivors) == 1
        assert survivors[0].fingerprint() == c1.fingerprint()

    def test_counter_increments_past_threshold(self):
        """Counter keeps incrementing even after reaching threshold."""
        c = _clique(["a", "b", "c"], ["cat1", "cat2"])
        history: dict[tuple[str, ...], int] = {c.fingerprint(): 5}
        survivors = persistence_filter([c], history, min_periods=2)
        assert len(survivors) == 1
        assert history[c.fingerprint()] == 6


# ═══════════════════════════════════════════════════════════════
# Cross-Category Filter
# ═══════════════════════════════════════════════════════════════


class TestCrossCategoryFilter:
    """Defence-in-depth category check."""

    def test_two_categories_passes(self):
        c = _clique(["a", "b", "c"], ["cat1", "cat2"])
        result = cross_category_filter([c], min_categories=2)
        assert len(result) == 1

    def test_one_category_filtered(self):
        c = _clique(["a", "b", "c"], ["cat1"])
        result = cross_category_filter([c], min_categories=2)
        assert len(result) == 0

    def test_min_categories_one(self):
        """min_categories=1 accepts anything."""
        c = _clique(["a", "b", "c"], ["cat1"])
        result = cross_category_filter([c], min_categories=1)
        assert len(result) == 1

    def test_three_categories_passes(self):
        c = _clique(["a", "b", "c"], ["c1", "c2", "c3"])
        result = cross_category_filter([c], min_categories=2)
        assert len(result) == 1

    def test_empty_input(self):
        assert cross_category_filter([], min_categories=2) == []


# ═══════════════════════════════════════════════════════════════
# Full Pipeline: apply_all_controls
# ═══════════════════════════════════════════════════════════════


class TestApplyAllControls:
    """End-to-end FDR pipeline with synthetic data."""

    @staticmethod
    def _build_scenario():
        """1 real convergence (3 signals, 2 categories, low p-values)
        + 10 spurious pairs (high p-values) → only real survives.
        """
        # Real signals: s1 (positioning), s2 (physical_flow), s3 (macro_momentum)
        categories = {
            "s1": "positioning",
            "s2": "physical_flow",
            "s3": "macro_momentum",
            # Spurious signals
            "n1": "positioning",
            "n2": "positioning",
            "n3": "positioning",
            "n4": "positioning",
            "n5": "financial_stress",
        }

        # Real pairs — highly significant.
        real_scores = {
            ("s1", "s2"): _cr(score=3.5, p_value=0.001),
            ("s1", "s3"): _cr(score=3.0, p_value=0.002),
            ("s2", "s3"): _cr(score=2.8, p_value=0.003),
        }
        real_p = {k: v.p_value for k, v in real_scores.items()}

        # Spurious pairs — high p-values, should not survive BH.
        spurious_scores = {}
        spurious_p = {}
        for i in range(1, 6):
            pair = (f"n{i}", f"n{(i % 5) + 1}")
            spurious_scores[pair] = _cr(score=0.5, p_value=0.3 + i * 0.05)
            spurious_p[pair] = 0.3 + i * 0.05

        all_scores = {**real_scores, **spurious_scores}
        all_p = {**real_p, **spurious_p}

        return all_scores, all_p, categories

    def test_real_convergence_survives_first_cycle(self):
        """Real convergence passes BH + graph, but NOT persistence (cycle 1)."""
        scores, p_vals, cats = self._build_scenario()
        history: dict[tuple[str, ...], int] = {}

        result = apply_all_controls(
            p_vals,
            scores,
            cats,
            history,
            q=0.05,
            min_persist=2,
            min_cats=2,
            min_clique_size=3,
        )
        # First cycle → persistence blocks emission.
        assert len(result) == 0
        # But the clique should now be in history.
        assert len(history) > 0

    def test_real_convergence_survives_second_cycle(self):
        """Second consecutive cycle → real convergence emitted."""
        scores, p_vals, cats = self._build_scenario()
        history: dict[tuple[str, ...], int] = {}

        # Cycle 1.
        apply_all_controls(
            p_vals,
            scores,
            cats,
            history,
            q=0.05,
            min_persist=2,
            min_cats=2,
            min_clique_size=3,
        )
        # Cycle 2.
        result = apply_all_controls(
            p_vals,
            scores,
            cats,
            history,
            q=0.05,
            min_persist=2,
            min_cats=2,
            min_clique_size=3,
        )
        assert len(result) == 1
        clique = result[0]
        assert set(clique.signals) == {"s1", "s2", "s3"}
        # Must span ≥ 2 categories.
        assert len(clique.categories) >= 2

    def test_spurious_only_none_survive(self):
        """Only spurious pairs (high p-values) → nothing emitted."""
        categories = {"n1": "c1", "n2": "c2", "n3": "c3"}
        scores = {
            ("n1", "n2"): _cr(score=0.5, p_value=0.40),
            ("n1", "n3"): _cr(score=0.5, p_value=0.50),
            ("n2", "n3"): _cr(score=0.5, p_value=0.60),
        }
        p_vals = {k: v.p_value for k, v in scores.items()}
        history: dict[tuple[str, ...], int] = {}

        result = apply_all_controls(
            p_vals,
            scores,
            categories,
            history,
            q=0.05,
            min_persist=1,
            min_cats=2,
            min_clique_size=3,
        )
        assert len(result) == 0

    def test_empty_input(self):
        history: dict[tuple[str, ...], int] = {}
        result = apply_all_controls({}, {}, {}, history)
        assert result == []
        assert history == {}

    def test_persistence_one_immediate_emit(self):
        """With min_persist=1, real convergence emits on first cycle."""
        scores, p_vals, cats = self._build_scenario()
        history: dict[tuple[str, ...], int] = {}

        result = apply_all_controls(
            p_vals,
            scores,
            cats,
            history,
            q=0.05,
            min_persist=1,
            min_cats=2,
            min_clique_size=3,
        )
        assert len(result) == 1

    def test_history_cleared_when_nothing_survives_bh(self):
        """When BH rejects everything, history of old cliques is cleaned."""
        dummy_fp = ("old_a", "old_b", "old_c")
        history: dict[tuple[str, ...], int] = {dummy_fp: 5}

        categories = {"x": "c1"}
        result = apply_all_controls(
            {("x", "y"): 0.99},
            {},
            categories,
            history,
            q=0.05,
            min_persist=2,
        )
        assert result == []
        assert dummy_fp not in history

    def test_fisher_combined_attached(self):
        """Surviving cliques should have p_values_combined attribute."""
        scores, p_vals, cats = self._build_scenario()
        history: dict[tuple[str, ...], int] = {}

        # Two cycles to pass persistence.
        apply_all_controls(p_vals, scores, cats, history, q=0.05, min_persist=2)
        result = apply_all_controls(p_vals, scores, cats, history, q=0.05, min_persist=2)

        assert len(result) == 1
        assert hasattr(result[0], "p_values_combined")
        assert 0.0 < result[0].p_values_combined < 1.0  # type: ignore[attr-defined]
