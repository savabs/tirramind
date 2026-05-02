"""Tests for pairwise coincidence scoring (Phase 7c-C.1).

Covers: CoincidenceResult construction, _rolling_pearson helper,
rolling_correlation_score (regime change, constant, independent,
short arrays, NaN), joint_exceedance_score (known exceedances,
independent, edges), concordance_score (perfect, anti, random,
edges), combined_coincidence_score (fusion, weights, Fisher's),
and boundary / degenerate cases.
"""

from __future__ import annotations

import math

import numpy as np

from agent.convergence.coincidence import (
    CoincidenceResult,
    _no_evidence,
    _rolling_pearson,
    _sign,
    _validate_pair,
    combined_coincidence_score,
    concordance_score,
    joint_exceedance_score,
    rolling_correlation_score,
)

# ── Helpers ────────────────────────────────────────────────────

RNG = np.random.RandomState(42)  # reproducible across runs


def _randn(n: int, seed: int = 42) -> np.ndarray:
    return np.random.RandomState(seed).randn(n)


def _correlated_pair(
    n: int,
    rho: float = 0.9,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two series with approximate Pearson ρ = *rho*."""
    rng = np.random.RandomState(seed)
    a = rng.randn(n)
    noise = rng.randn(n)
    b = rho * a + math.sqrt(1 - rho**2) * noise
    return a, b


def _regime_change_pair(
    n_before: int = 100,
    n_after: int = 100,
    rho_before: float = 0.0,
    rho_after: float = 0.9,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a pair whose correlation regime changes mid-stream."""
    rng = np.random.RandomState(seed)
    a = rng.randn(n_before + n_after)
    noise1 = rng.randn(n_before)
    noise2 = rng.randn(n_after)
    b_before = rho_before * a[:n_before] + math.sqrt(1 - rho_before**2) * noise1
    b_after = rho_after * a[n_before:] + math.sqrt(1 - rho_after**2) * noise2
    b = np.concatenate([b_before, b_after])
    return a, b


def _zscore_series(x: np.ndarray) -> np.ndarray:
    """Simple full-series z-scoring for test fixtures."""
    mu = np.nanmean(x)
    s = np.nanstd(x, ddof=1)
    if s < 1e-10:
        return np.zeros_like(x)
    return (x - mu) / s


# ═══════════════════════════════════════════════════════════════
# CoincidenceResult — basic construction
# ═══════════════════════════════════════════════════════════════


class TestCoincidenceResult:
    def test_construction(self):
        r = CoincidenceResult(method="rolling_corr", score=2.5, p_value=0.01, direction=1)
        assert r.method == "rolling_corr"
        assert r.score == 2.5
        assert r.p_value == 0.01
        assert r.direction == 1
        assert r.detail == {}

    def test_default_detail(self):
        r = CoincidenceResult(method="test", score=0, p_value=1, direction=0)
        assert isinstance(r.detail, dict)

    def test_with_detail(self):
        r = CoincidenceResult(
            method="test",
            score=1,
            p_value=0.5,
            direction=-1,
            detail={"key": "val"},
        )
        assert r.detail["key"] == "val"


# ═══════════════════════════════════════════════════════════════
# Helper: _no_evidence
# ═══════════════════════════════════════════════════════════════


class TestNoEvidence:
    def test_neutral_result(self):
        r = _no_evidence("foo", "bar")
        assert r.score == 0.0
        assert r.p_value == 1.0
        assert r.direction == 0
        assert r.detail["reason"] == "bar"


# ═══════════════════════════════════════════════════════════════
# Helper: _sign
# ═══════════════════════════════════════════════════════════════


class TestSign:
    def test_positive(self):
        assert _sign(3.5) == 1

    def test_negative(self):
        assert _sign(-0.001) == -1

    def test_zero(self):
        assert _sign(0.0) == 0


# ═══════════════════════════════════════════════════════════════
# Helper: _validate_pair
# ═══════════════════════════════════════════════════════════════


class TestValidatePair:
    def test_length_mismatch(self):
        r = _validate_pair(np.array([1, 2]), np.array([1]), "m", 1)
        assert r is not None
        assert r.detail["reason"] == "length mismatch"

    def test_too_short(self):
        r = _validate_pair(np.array([1, 2]), np.array([3, 4]), "m", 5)
        assert r is not None
        assert "need >= 5" in r.detail["reason"]

    def test_valid(self):
        r = _validate_pair(np.array([1, 2, 3]), np.array([4, 5, 6]), "m", 3)
        assert r is None


# ═══════════════════════════════════════════════════════════════
# _rolling_pearson
# ═══════════════════════════════════════════════════════════════


class TestRollingPearson:
    def test_perfect_positive_correlation(self):
        a = np.arange(50, dtype=float)
        b = a * 2 + 3
        rho = _rolling_pearson(a, b, window=10)
        # First 9 positions are NaN (insufficient history)
        assert np.all(np.isnan(rho[:9]))
        # From position 9 onward, ρ should be very close to 1
        valid = rho[~np.isnan(rho)]
        assert len(valid) > 0
        np.testing.assert_allclose(valid, 1.0, atol=1e-10)

    def test_perfect_negative_correlation(self):
        a = np.arange(50, dtype=float)
        b = -a * 2 + 100
        rho = _rolling_pearson(a, b, window=10)
        valid = rho[~np.isnan(rho)]
        np.testing.assert_allclose(valid, -1.0, atol=1e-10)

    def test_uncorrelated(self):
        a = _randn(500, seed=1)
        b = _randn(500, seed=2)
        rho = _rolling_pearson(a, b, window=50)
        valid = rho[~np.isnan(rho)]
        # Mean should be near 0 for independent series
        assert abs(np.mean(valid)) < 0.15

    def test_constant_signal_yields_zero(self):
        a = np.ones(30)
        b = np.arange(30, dtype=float)
        rho = _rolling_pearson(a, b, window=10)
        valid = rho[~np.isnan(rho)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-10)

    def test_nan_handling(self):
        a = np.arange(30, dtype=float)
        b = a * 2.0
        b[10] = np.nan
        b[15] = np.nan
        rho = _rolling_pearson(a, b, window=10)
        # Should still produce valid correlations (NaN pairs skipped)
        valid = rho[~np.isnan(rho)]
        assert len(valid) > 0
        # With most of the window clean, ρ should still be ~1
        assert all(r > 0.95 for r in valid)

    def test_too_few_valid_pairs(self):
        """If most of the window is NaN, ρ stays NaN."""
        a = np.arange(10, dtype=float)
        b = np.full(10, np.nan)
        b[0] = 1.0
        b[9] = 9.0
        rho = _rolling_pearson(a, b, window=5)
        # At most 2 valid pairs per window → all NaN (< _MIN_VALID=3)
        assert np.all(np.isnan(rho))

    def test_output_length_matches_input(self):
        a = _randn(100)
        b = _randn(100)
        rho = _rolling_pearson(a, b, window=20)
        assert len(rho) == 100


# ═══════════════════════════════════════════════════════════════
# rolling_correlation_score
# ═══════════════════════════════════════════════════════════════


class TestRollingCorrelationScore:
    def test_regime_change_detected(self):
        """When correlation jumps from ~0 to ~0.9, the z-score should
        be large and the p-value small.  Use a long pre-change window
        and short post-change so the baseline captures the old regime."""
        a, b = _regime_change_pair(
            n_before=180,
            n_after=40,
            rho_before=0.0,
            rho_after=0.9,
        )
        r = rolling_correlation_score(a, b, corr_window=20, baseline_window=150)
        assert r.method == "rolling_corr"
        assert r.score > 1.5  # should be a clear deviation
        assert r.p_value < 0.15
        assert r.direction == 1  # correlation increased

    def test_stable_correlation_not_flagged(self):
        """Two consistently correlated signals should NOT score high."""
        a, b = _correlated_pair(200, rho=0.5, seed=99)
        r = rolling_correlation_score(a, b, corr_window=20, baseline_window=100)
        # Moderate score; p-value should not be extremely small
        assert r.p_value > 0.01

    def test_independent_signals(self):
        """Independent signals should produce a non-significant result."""
        a = _randn(200, seed=10)
        b = _randn(200, seed=20)
        r = rolling_correlation_score(a, b, corr_window=20, baseline_window=100)
        assert r.p_value > 0.01

    def test_short_array_returns_no_evidence(self):
        a = np.arange(5, dtype=float)
        b = np.arange(5, dtype=float)
        r = rolling_correlation_score(a, b, corr_window=20)
        assert r.score == 0.0
        assert r.p_value == 1.0
        assert "reason" in r.detail

    def test_length_mismatch(self):
        a = np.arange(50, dtype=float)
        b = np.arange(30, dtype=float)
        r = rolling_correlation_score(a, b)
        assert r.score == 0.0
        assert r.detail["reason"] == "length mismatch"

    def test_constant_series(self):
        """Both series constant → ρ=0 everywhere → σ(ρ)=0 → score=0."""
        a = np.ones(100)
        b = np.ones(100) * 5
        r = rolling_correlation_score(a, b, corr_window=10, baseline_window=50)
        assert r.score == 0.0
        assert r.p_value == 1.0

    def test_all_nan(self):
        a = np.full(100, np.nan)
        b = np.full(100, np.nan)
        r = rolling_correlation_score(a, b, corr_window=10)
        assert r.score == 0.0
        assert r.p_value == 1.0

    def test_detail_fields_present(self):
        a, b = _correlated_pair(200, rho=0.5)
        r = rolling_correlation_score(a, b, corr_window=20, baseline_window=80)
        assert "rho_current" in r.detail
        assert "rho_baseline_mean" in r.detail
        assert "rho_baseline_std" in r.detail
        assert "n_valid_corr" in r.detail

    def test_divergence_detected(self):
        """Correlation dropping from ~0.9 to ~0 should be flagged as
        direction = -1 (diverging)."""
        a, b = _regime_change_pair(
            n_before=180,
            n_after=40,
            rho_before=0.9,
            rho_after=0.0,
            seed=77,
        )
        r = rolling_correlation_score(a, b, corr_window=20, baseline_window=150)
        assert r.direction == -1


# ═══════════════════════════════════════════════════════════════
# joint_exceedance_score
# ═══════════════════════════════════════════════════════════════


class TestJointExceedanceScore:
    def test_synchronised_tails(self):
        """When both z-scored series have extreme values at the same
        positions, p-value should be small."""
        n = 200
        z_a = _randn(n, seed=1)
        z_b = _randn(n, seed=2)
        # Inject synchronised tail events in the last 20 positions
        for i in range(n - 20, n):
            z_a[i] = 3.0
            z_b[i] = 3.0
        r = joint_exceedance_score(z_a, z_b, z_threshold=2.0, window=20)
        assert r.method == "joint_exceedance"
        assert r.score > 1.0
        assert r.p_value < 0.01
        assert r.direction == 1

    def test_independent_signals(self):
        """Independent z-series should rarely produce joint exceedances."""
        z_a = _randn(500, seed=30)
        z_b = _randn(500, seed=40)
        r = joint_exceedance_score(z_a, z_b, z_threshold=2.0, window=50)
        # Not guaranteed, but overwhelmingly likely p > 0.01
        assert r.p_value > 0.001

    def test_no_exceedances_at_all(self):
        """Series entirely within ±1 σ → marginal rate = 0 → no evidence."""
        z_a = np.ones(100) * 0.5
        z_b = np.ones(100) * -0.5
        r = joint_exceedance_score(z_a, z_b, z_threshold=2.0, window=20)
        assert r.score == 0.0
        assert r.p_value == 1.0
        assert "marginal exceedance rate" in r.detail.get("reason", "")

    def test_short_array(self):
        z_a = np.array([3.0, 3.0])
        z_b = np.array([3.0, 3.0])
        r = joint_exceedance_score(z_a, z_b, z_threshold=2.0, window=5)
        assert r.score == 0.0  # too short

    def test_length_mismatch(self):
        z_a = _randn(50)
        z_b = _randn(30)
        r = joint_exceedance_score(z_a, z_b)
        assert r.detail["reason"] == "length mismatch"

    def test_all_nan(self):
        z_a = np.full(50, np.nan)
        z_b = np.full(50, np.nan)
        r = joint_exceedance_score(z_a, z_b)
        assert r.p_value == 1.0

    def test_detail_fields_present(self):
        z_a = _randn(200, seed=50)
        z_b = _randn(200, seed=60)
        # Ensure some exceedances exist
        z_a[-5:] = 3.0
        z_b[-3:] = 3.0
        r = joint_exceedance_score(z_a, z_b, z_threshold=2.0, window=20)
        for key in (
            "observed",
            "expected",
            "n_trials",
            "p_marginal_a",
            "p_marginal_b",
            "p_joint_null",
        ):
            assert key in r.detail

    def test_nan_in_window_skipped(self):
        """NaN positions inside the window should be excluded, not crash."""
        z_a = np.concatenate([_randn(100, seed=70), np.array([3.0] * 20)])
        z_b = np.concatenate([_randn(100, seed=80), np.array([3.0] * 20)])
        z_a[-5] = np.nan
        z_b[-10] = np.nan
        r = joint_exceedance_score(z_a, z_b, z_threshold=2.0, window=20)
        # Should still compute (fewer trials)
        assert r.detail.get("n_trials", 0) < 20
        assert r.p_value < 1.0


# ═══════════════════════════════════════════════════════════════
# concordance_score
# ═══════════════════════════════════════════════════════════════


class TestConcordanceScore:
    def test_perfect_concordance(self):
        """Two monotonically increasing series → 100% concordance."""
        a = np.arange(30, dtype=float)
        b = np.arange(30, dtype=float) * 3 + 10
        r = concordance_score(a, b, window=20)
        assert r.method == "concordance"
        assert r.score > 2.0  # well above chance
        assert r.p_value < 0.01
        assert r.direction == 1

    def test_perfect_discordance(self):
        """One increasing, one decreasing → 0% concordance."""
        a = np.arange(30, dtype=float)
        b = -np.arange(30, dtype=float)
        r = concordance_score(a, b, window=20)
        assert r.score > 2.0
        assert r.p_value < 0.01
        assert r.direction == -1  # negative coupling

    def test_random_signals_near_half(self):
        """Independent random diffs → concordance ≈ 0.5."""
        a = _randn(500, seed=90)
        b = _randn(500, seed=91)
        r = concordance_score(a, b, window=50)
        hit_rate = r.detail["hit_rate"]
        assert 0.3 < hit_rate < 0.7
        assert r.p_value > 0.01

    def test_short_array(self):
        a = np.arange(5, dtype=float)
        b = np.arange(5, dtype=float)
        # window=20 needs 21 observations minimum
        r = concordance_score(a, b, window=20)
        assert r.score == 0.0
        assert r.p_value == 1.0

    def test_length_mismatch(self):
        a = np.arange(30, dtype=float)
        b = np.arange(20, dtype=float)
        r = concordance_score(a, b)
        assert r.detail["reason"] == "length mismatch"

    def test_constant_series(self):
        """All diffs = 0 → sign(0)==sign(0) → all concordant → high score."""
        a = np.ones(30)
        b = np.ones(30) * 5
        r = concordance_score(a, b, window=20)
        assert r.direction == 1
        assert r.detail["concordant"] == 20

    def test_nan_in_diffs(self):
        a = np.arange(30, dtype=float)
        b = np.arange(30, dtype=float) * 2
        b[25] = np.nan  # creates NaN diffs at positions 24 and 25
        r = concordance_score(a, b, window=20)
        assert r.detail["n_trials"] < 20

    def test_detail_fields_present(self):
        a = np.arange(30, dtype=float)
        b = np.arange(30, dtype=float)
        r = concordance_score(a, b, window=20)
        assert "concordant" in r.detail
        assert "n_trials" in r.detail
        assert "hit_rate" in r.detail

    def test_alternating_concordance(self):
        """Alternating up-down for one, steady up for other → 50%."""
        a = np.arange(50, dtype=float)  # always increasing
        b = np.array([float(i % 2) for i in range(50)])  # alternating
        r = concordance_score(a, b, window=20)
        hit_rate = r.detail["hit_rate"]
        assert 0.3 < hit_rate < 0.7  # near 50%


# ═══════════════════════════════════════════════════════════════
# combined_coincidence_score
# ═══════════════════════════════════════════════════════════════


class TestCombinedCoincidenceScore:
    def test_basic_fusion(self):
        """Combined scorer runs all three methods and produces a result."""
        a, b = _correlated_pair(200, rho=0.5, seed=1)
        z_a = _zscore_series(a)
        z_b = _zscore_series(b)
        r = combined_coincidence_score(a, b, z_a, z_b)
        assert r.method == "combined"
        assert r.score >= 0.0
        assert 0.0 <= r.p_value <= 1.0
        assert r.direction in (-1, 0, 1)
        assert "sub_results" in r.detail
        assert set(r.detail["sub_results"].keys()) == {
            "rolling_corr",
            "joint_exceedance",
            "concordance",
        }

    def test_fisher_chi2_present(self):
        a, b = _correlated_pair(200, rho=0.3)
        z_a, z_b = _zscore_series(a), _zscore_series(b)
        r = combined_coincidence_score(a, b, z_a, z_b)
        assert "fisher_chi2" in r.detail
        assert "fisher_df" in r.detail
        assert r.detail["fisher_df"] == 6  # 2 × 3 methods

    def test_custom_weights(self):
        a, b = _correlated_pair(200, rho=0.5)
        z_a, z_b = _zscore_series(a), _zscore_series(b)
        # Zero out joint_exceedance, boost rolling_corr
        w = {"rolling_corr": 1.0, "joint_exceedance": 0.0, "concordance": 0.0}
        r = combined_coincidence_score(a, b, z_a, z_b, weights=w)
        # Combined score should equal the rolling_corr score
        rc = rolling_correlation_score(a, b)
        assert abs(r.score - rc.score) < 1e-6

    def test_all_weights_zero(self):
        a = np.arange(200, dtype=float)
        z_a = _zscore_series(a)
        w = {"rolling_corr": 0.0, "joint_exceedance": 0.0, "concordance": 0.0}
        r = combined_coincidence_score(a, a, z_a, z_a, weights=w)
        assert r.score == 0.0
        assert r.p_value == 1.0

    def test_regime_change_detected_combined(self):
        """Regime change should be detectable via the combined scorer."""
        a, b = _regime_change_pair(150, 100, rho_before=0.0, rho_after=0.85)
        z_a, z_b = _zscore_series(a), _zscore_series(b)
        r = combined_coincidence_score(a, b, z_a, z_b)
        assert r.p_value < 0.2  # at least one sub-method should fire

    def test_independent_not_significant(self):
        a = _randn(300, seed=100)
        b = _randn(300, seed=200)
        z_a, z_b = _zscore_series(a), _zscore_series(b)
        r = combined_coincidence_score(a, b, z_a, z_b)
        assert r.p_value > 0.001

    def test_fisher_with_perfect_p_values(self):
        """If all sub-methods return p=1.0 (no evidence), combined p
        should also be ~1.0."""
        # Very short array → all methods return no evidence
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        z_a, z_b = _zscore_series(a), _zscore_series(b)
        r = combined_coincidence_score(a, b, z_a, z_b)
        assert r.p_value > 0.9
        assert r.score < 0.01


# ═══════════════════════════════════════════════════════════════
# Edge cases — degenerate inputs
# ═══════════════════════════════════════════════════════════════


class TestEdgeCases:
    # ── Empty arrays ───────────────────────────────────────────

    def test_rolling_corr_empty(self):
        a = np.array([], dtype=float)
        b = np.array([], dtype=float)
        r = rolling_correlation_score(a, b, corr_window=5)
        assert r.score == 0.0

    def test_joint_exceedance_empty(self):
        r = joint_exceedance_score(np.array([]), np.array([]))
        assert r.score == 0.0

    def test_concordance_empty(self):
        r = concordance_score(np.array([]), np.array([]))
        assert r.score == 0.0

    # ── Length-1 arrays ────────────────────────────────────────

    def test_all_methods_length_one(self):
        a = np.array([1.0])
        b = np.array([2.0])
        assert rolling_correlation_score(a, b, corr_window=1).p_value == 1.0
        assert joint_exceedance_score(a, b).p_value == 1.0
        assert concordance_score(a, b, window=1).p_value == 1.0

    # ── All-NaN ────────────────────────────────────────────────

    def test_all_nan_all_methods(self):
        a = np.full(100, np.nan)
        b = np.full(100, np.nan)
        for fn in (rolling_correlation_score, concordance_score):
            assert fn(a, b).p_value == 1.0
        assert joint_exceedance_score(a, b).p_value == 1.0

    # ── σ = 0 (constant input) ────────────────────────────────

    def test_sigma_zero_rolling_corr(self):
        """Constant series → all correlations are 0 → σ(ρ)=0 → score=0."""
        a = np.ones(100)
        b = np.arange(100, dtype=float)
        r = rolling_correlation_score(a, b, corr_window=10)
        assert r.score == 0.0

    def test_sigma_zero_concordance(self):
        """Constant a → diff(a)=0, concordant with diff(b) sign(0)==sign(x)
        depends on b's direction."""
        a = np.ones(30)
        b = np.arange(30, dtype=float)
        r = concordance_score(a, b, window=10)
        # sign(0) ≠ sign(positive) → discordant for most
        assert r.p_value <= 1.0  # should still compute

    # ── Mixed NaN positions ────────────────────────────────────

    def test_sparse_nan_still_computes(self):
        """Scattered NaN positions should be skipped, not crash."""
        rng = np.random.RandomState(999)
        a = rng.randn(200)
        b = rng.randn(200)
        # Sprinkle ~10% NaN
        nan_idx = rng.choice(200, size=20, replace=False)
        a[nan_idx[:10]] = np.nan
        b[nan_idx[10:]] = np.nan
        for fn in (rolling_correlation_score, concordance_score):
            r = fn(a, b)
            assert isinstance(r, CoincidenceResult)
        r2 = joint_exceedance_score(a, b)
        assert isinstance(r2, CoincidenceResult)

    # ── Large arrays (performance sanity) ──────────────────────

    def test_performance_5000_points(self):
        """5000-point series should complete in <5 s (usually <0.5 s)."""
        a = _randn(5000, seed=111)
        b = _randn(5000, seed=222)
        z_a, z_b = _zscore_series(a), _zscore_series(b)
        r = combined_coincidence_score(a, b, z_a, z_b)
        assert isinstance(r, CoincidenceResult)

    # ── p-value bounds ─────────────────────────────────────────

    def test_p_value_bounded_01(self):
        """All p-values should be in [0, 1]."""
        a, b = _regime_change_pair(150, 150, rho_before=0.0, rho_after=0.95)
        z_a, z_b = _zscore_series(a), _zscore_series(b)
        for fn, args in [
            (rolling_correlation_score, (a, b)),
            (joint_exceedance_score, (z_a, z_b)),
            (concordance_score, (a, b)),
            (combined_coincidence_score, (a, b, z_a, z_b)),
        ]:
            r = fn(*args)
            assert 0.0 <= r.p_value <= 1.0, f"{fn.__name__}: p={r.p_value}"

    # ── Score non-negativity ───────────────────────────────────

    def test_scores_nonnegative(self):
        """Scores must be ≥ 0 for all methods."""
        scenarios = [
            _correlated_pair(200, rho=0.5),
            (_randn(200, seed=10), _randn(200, seed=20)),
            _regime_change_pair(100, 100),
        ]
        for a, b in scenarios:
            z_a, z_b = _zscore_series(a), _zscore_series(b)
            assert rolling_correlation_score(a, b).score >= 0.0
            assert joint_exceedance_score(z_a, z_b).score >= 0.0
            assert concordance_score(a, b).score >= 0.0
            assert combined_coincidence_score(a, b, z_a, z_b).score >= 0.0

    # ── Direction values ───────────────────────────────────────

    def test_directions_valid(self):
        """Direction must be in {-1, 0, 1}."""
        a, b = _regime_change_pair(100, 100)
        z_a, z_b = _zscore_series(a), _zscore_series(b)
        for r in (
            rolling_correlation_score(a, b),
            joint_exceedance_score(z_a, z_b),
            concordance_score(a, b),
            combined_coincidence_score(a, b, z_a, z_b),
        ):
            assert r.direction in (-1, 0, 1)


# ═══════════════════════════════════════════════════════════════
# Hand-verified numerical checks
# ═══════════════════════════════════════════════════════════════


class TestHandVerified:
    def test_concordance_all_same_direction(self):
        """20 consecutive matching directions → concordance = 1.0,
        binomial p pprox 2^{-20} ≈ 1e-6."""
        a = np.arange(21, dtype=float)  # 20 diffs, all +1
        b = np.arange(21, dtype=float) * 5  # 20 diffs, all +5
        r = concordance_score(a, b, window=20)
        assert r.detail["concordant"] == 20
        assert r.detail["hit_rate"] == 1.0
        assert r.p_value < 1e-4

    def test_joint_exceedance_known_count(self):
        """Craft a scenario where we know exactly how many joint
        exceedances exist in the window."""
        z_a = np.zeros(100)
        z_b = np.zeros(100)
        # Last 20 positions: inject exactly 5 joint exceedances
        for i in [80, 83, 87, 92, 99]:
            z_a[i] = 3.0
            z_b[i] = 2.5
        # Give marginals some exceedances outside the window too
        z_a[10] = 3.0
        z_b[20] = 3.0
        r = joint_exceedance_score(z_a, z_b, z_threshold=2.0, window=20)
        assert r.detail["observed"] == 5
        assert r.detail["n_trials"] == 20

    def test_rolling_pearson_window3_hand_computed(self):
        """Hand-verify ρ for a tiny 3-element window."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        rho = _rolling_pearson(a, b, window=3)
        # Each 3-element window is perfectly correlated
        assert np.isnan(rho[0])
        assert np.isnan(rho[1])
        np.testing.assert_allclose(rho[2], 1.0, atol=1e-10)
        np.testing.assert_allclose(rho[3], 1.0, atol=1e-10)
        np.testing.assert_allclose(rho[4], 1.0, atol=1e-10)
