"""Tests for Idea 16 — Wasserstein Distributionally Robust Portfolio.

Covers:
    1.  wasserstein_1d: same distribution → 0.0
    2.  wasserstein_1d: non-overlapping distributions → positive
    3.  wasserstein_1d: different-length arrays (interpolation path)
    4.  bootstrap_epsilon: returns float ≥ 0
    5.  bootstrap_epsilon: higher-volatility returns → larger ε
    6.  bootstrap_epsilon: too-few rows → returns 0.0
    7.  robust_covariance: shape (N, N)
    8.  robust_covariance: diagonal inflated by exactly ε
    9.  robust_covariance: PSD (all eigenvalues ≥ 0)
    10. _min_variance_weights: weights sum to 1.0
    11. _min_variance_weights: all weights non-negative
    12. _min_variance_weights: singular matrix → equal weights
    13. _blend_with_views: weights sum to 1.0
    14. _blend_with_views: strong positive view tilts toward that asset
    15. RobustPortfolioWeights is frozen dataclass
    16. build_weights: returns None when too few observations
    17. build_weights: weights sum to 1.0
    18. build_weights: robust_cov > standard_cov when ε > 0
    19. build_weights: epsilon=0 → standard_cov == robust_cov
    20. build_weights: with return_views, highest-view asset gets highest weight
    21. build_weights: caps at max_instruments
    22. build_weights_from_store: returns None on empty store
    23. store_weights: writes N+2 signals (N weights + epsilon + ratio)
    24. store_weights: handles store error gracefully
    25. auto epsilon (None) is calibrated > 0 from noisy returns
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from agent.portfolio.robust_constructor import (
    RobustPortfolioWeights,
    WassersteinRobustPortfolio,
    _blend_with_views,
    _min_variance_weights,
    bootstrap_epsilon,
    robust_covariance,
    wasserstein_1d,
)
from agent.pipeline.store import PipelineStore


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_store(tmp_path: Path, name: str = "rob.db") -> PipelineStore:
    return PipelineStore(str(tmp_path / name))


def _make_returns(T: int = 60, N: int = 5, vol: float = 0.01, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, vol, (T, N))


def _make_eids(N: int) -> list[str]:
    return [f"asset_{i}" for i in range(N)]


# ═══════════════════════════════════════════════════════════════
# 1–3. wasserstein_1d
# ═══════════════════════════════════════════════════════════════

class TestWasserstein1d:

    def test_same_distribution_zero(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert wasserstein_1d(a, a) == pytest.approx(0.0)

    def test_non_overlapping_positive(self):
        a = np.zeros(10)
        b = np.ones(10)
        assert wasserstein_1d(a, b) == pytest.approx(1.0)

    def test_different_length_arrays(self):
        a = np.array([0.0, 1.0, 2.0])
        b = np.array([0.5, 1.5])
        result = wasserstein_1d(a, b)
        assert result >= 0.0
        assert math.isfinite(result)


# ═══════════════════════════════════════════════════════════════
# 4–6. bootstrap_epsilon
# ═══════════════════════════════════════════════════════════════

class TestBootstrapEpsilon:

    def test_returns_non_negative(self):
        r = _make_returns(50, 3)
        eps = bootstrap_epsilon(r, n_bootstrap=20)
        assert eps >= 0.0

    def test_higher_vol_larger_epsilon(self):
        r_low = _make_returns(50, 3, vol=0.001, seed=1)
        r_high = _make_returns(50, 3, vol=0.1, seed=1)
        eps_low = bootstrap_epsilon(r_low, n_bootstrap=50)
        eps_high = bootstrap_epsilon(r_high, n_bootstrap=50)
        assert eps_high > eps_low

    def test_too_few_rows_returns_zero(self):
        r = _make_returns(3, 3)
        eps = bootstrap_epsilon(r, n_bootstrap=10)
        assert eps == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════
# 7–9. robust_covariance
# ═══════════════════════════════════════════════════════════════

class TestRobustCovariance:

    def test_shape(self):
        r = _make_returns(50, 4)
        cov = robust_covariance(r, epsilon=0.01)
        assert cov.shape == (4, 4)

    def test_diagonal_inflated_by_epsilon(self):
        r = _make_returns(50, 3)
        eps = 0.05
        cov_r = robust_covariance(r, epsilon=eps)
        cov_s = np.cov(r.T)
        diag_diff = np.diag(cov_r) - np.diag(cov_s)
        assert np.allclose(diag_diff, eps)

    def test_psd(self):
        r = _make_returns(50, 4)
        cov = robust_covariance(r, epsilon=0.01)
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals >= -1e-10)


# ═══════════════════════════════════════════════════════════════
# 10–12. _min_variance_weights
# ═══════════════════════════════════════════════════════════════

class TestMinVarianceWeights:

    def test_sum_to_one(self):
        cov = np.diag([1.0, 2.0, 3.0])
        w = _min_variance_weights(cov)
        assert sum(w) == pytest.approx(1.0)

    def test_non_negative(self):
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        w = _min_variance_weights(cov)
        assert np.all(w >= -1e-10)

    def test_singular_fallback_equal(self):
        cov = np.zeros((3, 3))
        w = _min_variance_weights(cov)
        assert sum(w) == pytest.approx(1.0)
        assert np.allclose(w, 1.0 / 3)


# ═══════════════════════════════════════════════════════════════
# 13–14. _blend_with_views
# ═══════════════════════════════════════════════════════════════

class TestBlendWithViews:

    def test_sum_to_one(self):
        cov = np.diag([0.01, 0.02, 0.03])
        views = np.array([0.05, 0.01, 0.01])
        w = _blend_with_views(cov, views)
        assert sum(w) == pytest.approx(1.0)

    def test_positive_view_tilts_up(self):
        cov = np.diag([0.01, 0.01])
        views = np.array([0.10, 0.001])
        w = _blend_with_views(cov, views)
        assert w[0] > w[1]


# ═══════════════════════════════════════════════════════════════
# 15–25. WassersteinRobustPortfolio
# ═══════════════════════════════════════════════════════════════

class TestWassersteinRobustPortfolio:

    def test_result_frozen(self):
        r = RobustPortfolioWeights(
            weights={"a": 0.5, "b": 0.5}, epsilon=0.01,
            robust_cov=1.0, standard_cov=0.9, n_assets=2,
            built_at=time.time(),
        )
        with pytest.raises((AttributeError, TypeError)):
            r.epsilon = 0.0  # type: ignore[misc]

    def test_returns_none_too_few_obs(self):
        r = _make_returns(5, 3)
        wp = WassersteinRobustPortfolio(epsilon=0.01, min_history=20)
        result = wp.build_weights(r, _make_eids(3))
        assert result is None

    def test_weights_sum_to_one(self):
        r = _make_returns(60, 4)
        wp = WassersteinRobustPortfolio(epsilon=0.01, min_history=10)
        result = wp.build_weights(r, _make_eids(4))
        assert result is not None
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_robust_cov_greater_than_standard(self):
        r = _make_returns(60, 4)
        wp = WassersteinRobustPortfolio(epsilon=0.1, min_history=10)
        result = wp.build_weights(r, _make_eids(4))
        assert result is not None
        assert result.robust_cov > result.standard_cov

    def test_epsilon_zero_cov_equal(self):
        r = _make_returns(60, 3)
        wp = WassersteinRobustPortfolio(epsilon=0.0, min_history=10)
        result = wp.build_weights(r, _make_eids(3))
        assert result is not None
        assert result.robust_cov == pytest.approx(result.standard_cov, rel=1e-6)

    def test_views_tilt_weights(self):
        r = _make_returns(60, 3)
        wp = WassersteinRobustPortfolio(epsilon=0.01, min_history=10)
        views = {"asset_0": 0.20, "asset_1": 0.001, "asset_2": 0.001}
        result = wp.build_weights(r, _make_eids(3), return_views=views)
        assert result is not None
        assert result.weights["asset_0"] > result.weights["asset_1"]

    def test_caps_max_instruments(self):
        r = _make_returns(60, 20)
        wp = WassersteinRobustPortfolio(epsilon=0.01, min_history=10, max_instruments=5)
        result = wp.build_weights(r, _make_eids(20))
        assert result is not None
        assert result.n_assets == 5

    def test_from_store_empty_returns_none(self, tmp_path):
        store = _make_store(tmp_path, "empty.db")
        wp = WassersteinRobustPortfolio(epsilon=0.01)
        result = wp.build_weights_from_store(store)
        assert result is None

    def test_store_weights_correct_signal_count(self):
        mock_store = MagicMock()
        wp = WassersteinRobustPortfolio(epsilon=0.01)
        result = RobustPortfolioWeights(
            weights={"a": 0.4, "b": 0.6}, epsilon=0.01,
            robust_cov=1.0, standard_cov=0.9, n_assets=2,
            built_at=time.time(),
        )
        n = wp.store_weights(mock_store, result)
        # 2 weights + epsilon + ratio = 4
        assert n == 4

    def test_store_weights_handles_error(self):
        mock_store = MagicMock()
        mock_store.store_signal.side_effect = RuntimeError("fail")
        wp = WassersteinRobustPortfolio(epsilon=0.01)
        result = RobustPortfolioWeights(
            weights={"a": 1.0}, epsilon=0.01,
            robust_cov=0.5, standard_cov=0.4, n_assets=1,
            built_at=time.time(),
        )
        n = wp.store_weights(mock_store, result)
        assert n == 0  # all failed gracefully

    def test_auto_epsilon_positive(self):
        r = _make_returns(60, 4, vol=0.02)
        wp = WassersteinRobustPortfolio(epsilon=None, n_bootstrap=30, min_history=10)
        result = wp.build_weights(r, _make_eids(4))
        assert result is not None
        assert result.epsilon > 0.0
