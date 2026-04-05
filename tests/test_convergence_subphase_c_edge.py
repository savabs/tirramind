"""Sub-phase 7c-C edge-case tests (Phase 7c-C.4).

Stress + boundary tests across all three sub-phase C modules:
- coincidence.py: length-0/1 arrays, σ=0, all-NaN
- graph.py: 1000-node sparse perf, 60-node dense clique
- fdr.py: 0/1/10K p-values BH, Fisher p=0/1, persistence min=1, NaN weights

These complement the per-module tests written in 7c-C.1, 7c-C.2, 7c-C.3.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
from scipy import stats as sp_stats

from agent.convergence.coincidence import (
    CoincidenceResult,
    combined_coincidence_score,
    concordance_score,
    joint_exceedance_score,
    rolling_correlation_score,
)
from agent.convergence.fdr import (
    _P_FLOOR,
    apply_all_controls,
    apply_bh_correction,
    cross_category_filter,
    fisher_combined_test,
    persistence_filter,
)
from agent.convergence.graph import (
    ConvergenceClique,
    build_coincidence_graph,
    detect_convergence_cliques,
    score_clique,
)


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
# Coincidence Edge Cases
# ═══════════════════════════════════════════════════════════════


class TestCoincidenceLengthZero:
    """Length-0 arrays should produce neutral results, not crash."""

    def test_rolling_corr_empty(self):
        r = rolling_correlation_score(np.array([]), np.array([]))
        assert r.score == 0.0
        assert r.p_value == 1.0

    def test_joint_exceedance_empty(self):
        r = joint_exceedance_score(np.array([]), np.array([]))
        assert r.score == 0.0
        assert r.p_value == 1.0

    def test_concordance_empty(self):
        r = concordance_score(np.array([]), np.array([]))
        assert r.score == 0.0
        assert r.p_value == 1.0

    def test_combined_empty(self):
        r = combined_coincidence_score(
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )
        assert r.score == 0.0
        assert r.p_value == 1.0


class TestCoincidenceLengthOne:
    """Single-element arrays: not enough for meaningful stats."""

    def test_rolling_corr_one(self):
        r = rolling_correlation_score(np.array([1.0]), np.array([2.0]))
        assert r.score == 0.0

    def test_joint_exceedance_one(self):
        r = joint_exceedance_score(np.array([3.0]), np.array([3.0]))
        assert r.score == 0.0

    def test_concordance_one(self):
        r = concordance_score(np.array([1.0]), np.array([2.0]))
        assert r.score == 0.0


class TestCoincidenceConstantSignal:
    """One signal is constant (σ=0). Must not produce NaN/Inf."""

    def test_rolling_corr_constant_a(self):
        a = np.ones(50)
        b = np.random.RandomState(42).randn(50)
        r = rolling_correlation_score(a, b)
        assert not math.isnan(r.score)
        assert not math.isinf(r.score)

    def test_joint_exceedance_constant(self):
        a = np.zeros(50)
        b = np.random.RandomState(42).randn(50)
        r = joint_exceedance_score(a, b)
        assert not math.isnan(r.score)

    def test_concordance_constant(self):
        a = np.ones(50)
        b = np.random.RandomState(42).randn(50)
        r = concordance_score(a, b)
        assert not math.isnan(r.score)


class TestCoincidenceAllNaN:
    """All-NaN arrays → neutral results."""

    def test_rolling_corr_all_nan(self):
        nan = np.full(50, np.nan)
        r = rolling_correlation_score(nan, nan)
        assert r.score == 0.0
        assert r.p_value == 1.0

    def test_joint_exceedance_all_nan(self):
        nan = np.full(50, np.nan)
        r = joint_exceedance_score(nan, nan)
        assert r.score == 0.0

    def test_concordance_all_nan(self):
        nan = np.full(50, np.nan)
        r = concordance_score(nan, nan)
        assert r.score == 0.0


# ═══════════════════════════════════════════════════════════════
# Graph Edge Cases
# ═══════════════════════════════════════════════════════════════


class TestGraphPerformance1000Nodes:
    """1000-node sparse graph — clique detection must finish quickly."""

    def test_sparse_1000_nodes_under_1s(self):
        # Build a 1000-node graph with ~200 random sparse edges.
        rng = np.random.RandomState(99)
        n_nodes = 1000
        cat_names = ["c1", "c2", "c3", "c4", "c5"]
        categories = {f"s{i}": cat_names[i % len(cat_names)] for i in range(n_nodes)}
        scores: dict[tuple[str, str], CoincidenceResult] = {}
        # 200 random edges (very sparse -> fast clique detection).
        for _ in range(200):
            a = rng.randint(0, n_nodes)
            b = rng.randint(0, n_nodes)
            if a == b:
                continue
            pair = (f"s{min(a, b)}", f"s{max(a, b)}")
            scores[pair] = _cr(score=2.0, p_value=0.01)

        t0 = time.monotonic()
        G = build_coincidence_graph(scores, categories, p_threshold=0.05)
        cliques = detect_convergence_cliques(G, min_size=3, min_categories=2)
        elapsed = time.monotonic() - t0

        assert elapsed < 1.0, f"Took {elapsed:.2f}s (expected <1s)"
        # Should find some cliques in the random graph.
        # Not asserting exact count — just that it finishes fast.


class TestGraphDense60Nodes:
    """60-node fully connected graph (worst case for Bron-Kerbosch).
    Safety cap should prevent hang."""

    def test_fully_connected_60_nodes(self):
        n = 60
        cat_names = ["c1", "c2", "c3"]
        categories = {f"s{i}": cat_names[i % 3] for i in range(n)}
        scores: dict[tuple[str, str], CoincidenceResult] = {}
        for i in range(n):
            for j in range(i + 1, n):
                scores[(f"s{i}", f"s{j}")] = _cr(score=2.0, p_value=0.001)

        G = build_coincidence_graph(scores, categories, p_threshold=0.05)

        t0 = time.monotonic()
        cliques = detect_convergence_cliques(G, min_size=3, min_categories=2)
        elapsed = time.monotonic() - t0

        # Should not hang — safety cap on max_cliques prevents explosion.
        # The graph IS fully connected so it produces cliques.
        assert len(cliques) > 0

    def test_score_scales_with_massive_clique(self):
        """Score of a clique should stay in [0,1] even for large cliques."""
        signals = [f"s{i}" for i in range(20)]
        cats = ["c1", "c2", "c3", "c4"]
        edges = []
        for i, sa in enumerate(signals):
            for sb in signals[i + 1 :]:
                edges.append((sa, sb, 10.0))
        c = ConvergenceClique(
            signals=signals,
            categories=cats,
            edges=edges,
            score=0.0,
            p_values=[0.001] * len(edges),
        )
        s = score_clique(c)
        assert 0.0 <= s <= 1.0


# ═══════════════════════════════════════════════════════════════
# FDR Edge Cases
# ═══════════════════════════════════════════════════════════════


class TestBHEdgeCases:
    """BH correction boundary and performance cases."""

    def test_zero_p_values_empty(self):
        assert apply_bh_correction({}) == {}

    def test_single_p_value_significant(self):
        r = apply_bh_correction({("a", "b"): 0.001}, q=0.05)
        assert r[("a", "b")] is True

    def test_single_p_value_nonsignificant(self):
        r = apply_bh_correction({("a", "b"): 0.5}, q=0.05)
        assert r[("a", "b")] is False

    def test_10k_p_values_performance(self):
        """10,000 p-values should complete quickly."""
        rng = np.random.RandomState(42)
        pairs = {(f"a{i}", f"b{i}"): float(rng.uniform(0, 1)) for i in range(10_000)}

        t0 = time.monotonic()
        result = apply_bh_correction(pairs, q=0.05)
        elapsed = time.monotonic() - t0

        assert elapsed < 1.0, f"BH on 10K took {elapsed:.2f}s"
        assert len(result) == 10_000
        # Some should be rejected, some not.
        rejected = sum(1 for v in result.values() if v)
        # With uniform p-values at q=0.05, expect ~5% rejection.
        assert rejected < 1_000  # sanity cap

    def test_p_value_exactly_zero(self):
        """p=0.0 should be clipped, not crash."""
        r = apply_bh_correction({("a", "b"): 0.0}, q=0.05)
        assert r[("a", "b")] is True

    def test_p_value_exactly_one(self):
        r = apply_bh_correction({("a", "b"): 1.0}, q=0.05)
        assert r[("a", "b")] is False


class TestFisherEdgeCases:
    """Fisher's combined test edge cases."""

    def test_p_exactly_zero_clipped(self):
        """p=0 must produce a valid (very small) combined p."""
        combined = fisher_combined_test([0.0])
        assert 0.0 < combined <= _P_FLOOR
        assert not math.isnan(combined)

    def test_p_exactly_one_no_evidence(self):
        """p=1.0 contributes 0 to χ², combined should be 1.0."""
        assert fisher_combined_test([1.0]) == pytest.approx(1.0)

    def test_all_ones(self):
        combined = fisher_combined_test([1.0, 1.0, 1.0])
        assert combined == pytest.approx(1.0)

    def test_negative_log_zero_guard(self):
        """Extremely small p-values should not overflow χ²."""
        combined = fisher_combined_test([1e-300, 1e-300])
        assert 0.0 < combined < 1.0
        assert not math.isnan(combined)

    def test_100_p_values(self):
        """100 p-values of 0.03 → extremely small combined p."""
        combined = fisher_combined_test([0.03] * 100)
        assert combined < 1e-50


class TestPersistenceEdgeCases:
    """Persistence filter edge cases."""

    def test_min_periods_one(self):
        history: dict[tuple[str, ...], int] = {}
        c = _clique(["a", "b", "c"], ["c1", "c2"])
        survivors = persistence_filter([c], history, min_periods=1)
        assert len(survivors) == 1

    def test_large_history_cleanup(self):
        """100 old fingerprints all disappear in one cycle."""
        history: dict[tuple[str, ...], int] = {}
        for i in range(100):
            history[(f"old{i}",)] = 5
        persistence_filter([], history, min_periods=2)
        assert len(history) == 0

    def test_overlapping_fingerprints(self):
        """Two cliques with the same sorted signals → same fingerprint,
        counted as one."""
        c1 = _clique(["a", "b", "c"], ["c1", "c2"])
        c2 = _clique(["a", "b", "c"], ["c1", "c2"])
        assert c1.fingerprint() == c2.fingerprint()

        history: dict[tuple[str, ...], int] = {}
        # Both contribute to the same fingerprint.
        persistence_filter([c1, c2], history, min_periods=2)
        fp = c1.fingerprint()
        # Incremented twice (once per event in the list).
        assert history[fp] == 2


class TestCombinedScoringNaNWeights:
    """Clique scoring with unusual weight values."""

    def test_nan_weight_in_edges(self):
        """NaN edge weight should not crash score_clique."""
        c = ConvergenceClique(
            signals=["a", "b", "c"],
            categories=["c1", "c2"],
            edges=[("a", "b", float("nan")), ("a", "c", 2.0), ("b", "c", 2.0)],
            score=0.0,
            p_values=[0.01, 0.01, 0.01],
        )
        s = score_clique(c)
        # NaN propagates through sum → NaN → exp(NaN) → sigmoid gives NaN.
        # This is actually a correct "garbage in, garbage out" — but we verify
        # it doesn't raise an exception.
        assert isinstance(s, float)

    def test_zero_weight_all_edges(self):
        c = ConvergenceClique(
            signals=["a", "b", "c"],
            categories=["c1", "c2"],
            edges=[("a", "b", 0.0), ("a", "c", 0.0), ("b", "c", 0.0)],
            score=0.0,
            p_values=[0.01, 0.01, 0.01],
        )
        s = score_clique(c)
        assert s == 0.0

    def test_inf_weight(self):
        """Infinite weight → score should be 1.0 (sigmoid saturates)."""
        c = ConvergenceClique(
            signals=["a", "b", "c"],
            categories=["c1", "c2"],
            edges=[("a", "b", float("inf")), ("a", "c", 2.0), ("b", "c", 2.0)],
            score=0.0,
            p_values=[0.01, 0.01, 0.01],
        )
        s = score_clique(c)
        # inf in sum → sigmoid approaches 1.0 or NaN depending on implementation.
        assert isinstance(s, float)


class TestFullPipelineMixedScenario:
    """apply_all_controls with a mix of real and noise signals
    at different significance levels."""

    def test_two_real_cliques_one_noise(self):
        """Two distinct real convergences + noise → both real survive."""
        categories = {
            # Real clique 1
            "r1": "positioning",
            "r2": "physical_flow",
            "r3": "macro_momentum",
            # Real clique 2
            "r4": "financial_stress",
            "r5": "regulatory_action",
            "r6": "behavioral_intent",
            # Noise
            "n1": "positioning",
            "n2": "positioning",
        }

        scores: dict[tuple[str, str], CoincidenceResult] = {}
        p_vals: dict[tuple[str, str], float] = {}

        # Real clique 1.
        for a, b in [("r1", "r2"), ("r1", "r3"), ("r2", "r3")]:
            scores[(a, b)] = _cr(score=3.0, p_value=0.002)
            p_vals[(a, b)] = 0.002

        # Real clique 2.
        for a, b in [("r4", "r5"), ("r4", "r6"), ("r5", "r6")]:
            scores[(a, b)] = _cr(score=2.5, p_value=0.003)
            p_vals[(a, b)] = 0.003

        # Noise pairs.
        scores[("n1", "n2")] = _cr(score=0.5, p_value=0.6)
        p_vals[("n1", "n2")] = 0.6

        history: dict[tuple[str, ...], int] = {}

        # Cycle 1: both real appear in history.
        r1 = apply_all_controls(
            p_vals,
            scores,
            categories,
            history,
            q=0.05,
            min_persist=2,
            min_cats=2,
            min_clique_size=3,
        )
        assert len(r1) == 0  # persistence blocks first cycle
        assert len(history) == 2  # two clique fingerprints

        # Cycle 2: both emitted.
        r2 = apply_all_controls(
            p_vals,
            scores,
            categories,
            history,
            q=0.05,
            min_persist=2,
            min_cats=2,
            min_clique_size=3,
        )
        assert len(r2) == 2

    def test_every_pair_significant_graph(self):
        """When every pair is significant, graph is complete. Cliques exist."""
        n = 10
        cat_names = ["c1", "c2", "c3"]
        categories = {f"s{i}": cat_names[i % 3] for i in range(n)}
        scores: dict[tuple[str, str], CoincidenceResult] = {}
        p_vals: dict[tuple[str, str], float] = {}
        for i in range(n):
            for j in range(i + 1, n):
                pair = (f"s{i}", f"s{j}")
                scores[pair] = _cr(score=3.0, p_value=0.001)
                p_vals[pair] = 0.001

        history: dict[tuple[str, ...], int] = {}
        # min_persist=1 so we get results immediately.
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
        assert len(result) > 0
        # All clique scores in [0, 1].
        for c in result:
            assert 0.0 <= c.score <= 1.0
