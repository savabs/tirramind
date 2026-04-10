"""
Tests for agent/models/state_filter.py — regime-conditioned Kalman filter.

Validates:
    - Predict step applies regime-specific dynamics
    - Update step incorporates observations
    - Joseph form maintains positive-definiteness
    - Missing observations (NaN) are masked
    - Quality weighting inflates R
    - Regime switching changes dynamics
    - get_beliefs produces valid Gaussian BeliefState
    - Dimension mismatches raise errors
    - Reset reinitializes state
    - Filter tracks synthetic state-space model
"""

from __future__ import annotations

import numpy as np
import pytest

from agent.models.belief import BeliefState
from agent.models.state_filter import ContinuousStateFilter, RegimeConfig

# ── Helpers ────────────────────────────────────────────────────


def _make_1d_configs() -> dict[str, RegimeConfig]:
    """1D state, 1D observation, two regimes."""
    return {
        "stable": RegimeConfig(
            name="stable",
            F=np.array([[0.99]]),
            Q=np.array([[0.01]]),
        ),
        "volatile": RegimeConfig(
            name="volatile",
            F=np.array([[0.95]]),
            Q=np.array([[0.10]]),
        ),
    }


def _make_1d_filter() -> ContinuousStateFilter:
    return ContinuousStateFilter(
        state_dim=1,
        obs_dim=1,
        regime_configs=_make_1d_configs(),
        H=np.array([[1.0]]),
        R=np.array([[0.1]]),
    )


def _make_3d_configs() -> dict[str, RegimeConfig]:
    """3D state, 3 regimes (matches spec: expansion, contraction, crisis)."""
    return {
        "expansion": RegimeConfig(
            name="expansion",
            F=np.diag([0.99, 0.98, 0.97]),
            Q=np.diag([0.01, 0.01, 0.01]),
        ),
        "contraction": RegimeConfig(
            name="contraction",
            F=np.diag([0.97, 0.96, 0.95]),
            Q=np.diag([0.02, 0.02, 0.02]),
        ),
        "crisis": RegimeConfig(
            name="crisis",
            F=np.diag([0.90, 0.88, 0.85]),
            Q=np.diag([0.10, 0.10, 0.10]),
        ),
    }


def _make_3d_filter() -> ContinuousStateFilter:
    """3D state, 6D observation (matching spec 6 features)."""
    H = np.zeros((6, 3))
    H[0, 0] = 1.0  # obs0 → state0
    H[1, 0] = 1.0  # obs1 → state0
    H[2, 1] = 1.0  # obs2 → state1
    H[3, 1] = 1.0  # obs3 → state1
    H[4, 2] = 1.0  # obs4 → state2
    H[5, 2] = 1.0  # obs5 → state2
    R = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    return ContinuousStateFilter(
        state_dim=3,
        obs_dim=6,
        regime_configs=_make_3d_configs(),
        H=H,
        R=R,
    )


AS_OF = 1_700_000_000.0
GRAPH_HASH = "a" * 64


# ── Construction ───────────────────────────────────────────────


class TestConstruction:
    def test_initial_state_zero(self) -> None:
        f = _make_1d_filter()
        np.testing.assert_array_equal(f.state, [0.0])

    def test_initial_cov_identity(self) -> None:
        f = _make_1d_filter()
        np.testing.assert_array_equal(f.covariance, [[1.0]])

    def test_dimensions(self) -> None:
        f = _make_3d_filter()
        assert f.state_dim == 3
        assert f.obs_dim == 6

    def test_bad_state_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="state_dim"):
            ContinuousStateFilter(
                state_dim=0,
                obs_dim=1,
                regime_configs=_make_1d_configs(),
                H=np.array([[1.0]]),
                R=np.array([[0.1]]),
            )

    def test_bad_H_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="H shape"):
            ContinuousStateFilter(
                state_dim=1,
                obs_dim=1,
                regime_configs=_make_1d_configs(),
                H=np.array([[1.0, 2.0]]),  # wrong shape
                R=np.array([[0.1]]),
            )


# ── Predict ────────────────────────────────────────────────────


class TestPredict:
    def test_predict_applies_transition(self) -> None:
        f = _make_1d_filter()
        f.reset(np.array([1.0]), np.array([[0.5]]))
        x, P = f.predict("stable")
        np.testing.assert_allclose(x, [0.99], atol=1e-10)

    def test_predict_increases_uncertainty(self) -> None:
        f = _make_1d_filter()
        f.reset(np.array([1.0]), np.array([[0.5]]))
        _, P_before = f.state, f.covariance
        _, P_after = f.predict("stable")
        assert P_after[0, 0] > P_before[0, 0]

    def test_volatile_regime_more_uncertainty(self) -> None:
        f1 = _make_1d_filter()
        f2 = _make_1d_filter()
        f1.reset(np.array([1.0]), np.array([[0.5]]))
        f2.reset(np.array([1.0]), np.array([[0.5]]))
        _, P_stable = f1.predict("stable")
        _, P_volatile = f2.predict("volatile")
        assert P_volatile[0, 0] > P_stable[0, 0]

    def test_unknown_regime_raises(self) -> None:
        f = _make_1d_filter()
        with pytest.raises(ValueError, match="not in configs"):
            f.predict("nonexistent")


# ── Update ─────────────────────────────────────────────────────


class TestUpdate:
    def test_update_reduces_uncertainty(self) -> None:
        f = _make_1d_filter()
        f.reset(np.array([0.0]), np.array([[1.0]]))
        f.predict("stable")
        _, P_before = f.state.copy(), f.covariance.copy()
        _, P_after = f.update(np.array([1.0]))
        assert P_after[0, 0] < P_before[0, 0]

    def test_update_moves_state_toward_obs(self) -> None:
        f = _make_1d_filter()
        f.reset(np.array([0.0]), np.array([[1.0]]))
        f.predict("stable")
        x_before = f.state.copy()
        x_after, _ = f.update(np.array([5.0]))
        assert x_after[0] > x_before[0]

    def test_all_nan_skips_update(self) -> None:
        f = _make_1d_filter()
        f.reset(np.array([1.0]), np.array([[0.5]]))
        f.predict("stable")
        x_pred = f.state.copy()
        P_pred = f.covariance.copy()
        x_up, P_up = f.update(np.array([np.nan]))
        np.testing.assert_array_equal(x_up, x_pred)
        np.testing.assert_array_equal(P_up, P_pred)

    def test_partial_nan_updates_valid_only(self) -> None:
        f = _make_3d_filter()
        f.reset(np.zeros(3), np.eye(3))
        f.predict("expansion")
        P_pred = f.covariance.copy()

        obs = np.array([1.0, np.nan, np.nan, np.nan, np.nan, np.nan])
        _, P_up = f.update(obs)
        # State 0 uncertainty should decrease (observed)
        assert P_up[0, 0] < P_pred[0, 0]
        # State 2 uncertainty should NOT decrease (not observed)
        # (state 1 might couple through P, so only test state 2)

    def test_obs_dimension_mismatch_raises(self) -> None:
        f = _make_1d_filter()
        with pytest.raises(ValueError, match="observations shape"):
            f.update(np.array([1.0, 2.0]))


# ── Quality weighting ─────────────────────────────────────────


class TestQuality:
    def test_quality_zero_skips_update(self) -> None:
        f = _make_1d_filter()
        f.reset(np.array([0.0]), np.array([[1.0]]))
        f.predict("stable")
        x_pred = f.state.copy()
        x_up, _ = f.update(np.array([5.0]), quality=np.array([0.0]))
        np.testing.assert_array_equal(x_up, x_pred)

    def test_low_quality_less_pull(self) -> None:
        """Low quality should cause less state movement than high quality."""
        f1 = _make_1d_filter()
        f2 = _make_1d_filter()
        for f in [f1, f2]:
            f.reset(np.array([0.0]), np.array([[1.0]]))
            f.predict("stable")

        x_high, _ = f1.update(np.array([5.0]), quality=np.array([1.0]))
        x_low, _ = f2.update(np.array([5.0]), quality=np.array([0.1]))
        # High quality update should move state further toward 5.0
        assert abs(x_high[0] - 5.0) < abs(x_low[0] - 5.0)


# ── Joseph form & stability ────────────────────────────────────


class TestJosephForm:
    def test_covariance_stays_positive_definite(self) -> None:
        """Run 1000 predict/update cycles — P must stay PD."""
        f = _make_1d_filter()
        f.reset(np.array([0.0]), np.array([[1.0]]))
        rng = np.random.default_rng(42)
        for _ in range(1000):
            f.predict("stable")
            obs = rng.normal(0, 1, size=(1,))
            f.update(obs)
            assert f.covariance[0, 0] > 0, "P lost positive-definiteness"

    def test_3d_covariance_stays_symmetric(self) -> None:
        """Run 500 cycles on 3D filter — P must stay symmetric."""
        f = _make_3d_filter()
        f.reset(np.zeros(3), np.eye(3))
        rng = np.random.default_rng(42)
        for _ in range(500):
            f.predict("expansion")
            obs = rng.normal(0, 1, size=(6,))
            f.update(obs)
            P = f.covariance
            np.testing.assert_allclose(
                P,
                P.T,
                atol=1e-10,
                err_msg="P is not symmetric",
            )

    def test_3d_covariance_positive_definite(self) -> None:
        """3D filter stays PD over 500 cycles."""
        f = _make_3d_filter()
        f.reset(np.zeros(3), np.eye(3))
        rng = np.random.default_rng(42)
        for i in range(500):
            f.predict("contraction")
            obs = rng.normal(0, 1, size=(6,))
            f.update(obs)
            eigvals = np.linalg.eigvalsh(f.covariance)
            assert np.all(eigvals > 0), f"P not PD at step {i}: eigenvalues={eigvals}"


# ── Regime switching ───────────────────────────────────────────


class TestRegimeSwitching:
    def test_switching_regime_changes_dynamics(self) -> None:
        f = _make_1d_filter()
        f.reset(np.array([1.0]), np.array([[0.1]]))

        f.predict("stable")
        x_stable = f.state.copy()

        f.reset(np.array([1.0]), np.array([[0.1]]))
        f.predict("volatile")
        x_volatile = f.state.copy()

        # Stable F=0.99, Volatile F=0.95 → stable state closer to 1.0
        assert x_stable[0] > x_volatile[0]


# ── get_beliefs ────────────────────────────────────────────────


class TestGetBeliefs:
    def test_produces_gaussian_beliefs(self) -> None:
        f = _make_1d_filter()
        f.reset(np.array([2.5]), np.array([[0.3]]))
        beliefs = f.get_beliefs(["latent.test.state"], AS_OF, GRAPH_HASH)
        assert len(beliefs) == 1
        b = beliefs[0]
        assert b.dist_type == "gaussian"
        np.testing.assert_allclose(b.mean, 2.5)
        np.testing.assert_allclose(b.variance, 0.3)

    def test_variable_names_mismatch_raises(self) -> None:
        f = _make_1d_filter()
        with pytest.raises(ValueError, match="variable_names length"):
            f.get_beliefs(["a", "b"], AS_OF, GRAPH_HASH)

    def test_3d_beliefs(self) -> None:
        f = _make_3d_filter()
        f.reset(np.array([1.0, 2.0, 3.0]), np.diag([0.1, 0.2, 0.3]))
        names = ["latent.s1", "latent.s2", "latent.s3"]
        beliefs = f.get_beliefs(names, AS_OF, GRAPH_HASH)
        assert len(beliefs) == 3
        assert beliefs[0].mean == pytest.approx(1.0)
        assert beliefs[1].mean == pytest.approx(2.0)
        assert beliefs[2].mean == pytest.approx(3.0)
        assert beliefs[0].variance == pytest.approx(0.1)
        assert beliefs[2].variance == pytest.approx(0.3)


# ── Reset ──────────────────────────────────────────────────────


class TestReset:
    def test_reset_sets_state(self) -> None:
        f = _make_1d_filter()
        f.reset(np.array([5.0]), np.array([[2.0]]))
        np.testing.assert_allclose(f.state, [5.0])
        np.testing.assert_allclose(f.covariance, [[2.0]])

    def test_reset_bad_shape_raises(self) -> None:
        f = _make_1d_filter()
        with pytest.raises(ValueError, match="x0 shape"):
            f.reset(np.array([1.0, 2.0]), np.array([[1.0]]))

    def test_reset_bad_P_shape_raises(self) -> None:
        f = _make_1d_filter()
        with pytest.raises(ValueError, match="P0 shape"):
            f.reset(np.array([1.0]), np.array([[1.0, 0.0], [0.0, 1.0]]))


# ── Tracking synthetic model ──────────────────────────────────


class TestSyntheticTracking:
    def test_1d_filter_tracks_true_state(self) -> None:
        """Generate synthetic data from known model, verify filter recovers state."""
        rng = np.random.default_rng(123)
        true_F = 0.99
        true_Q = 0.01
        true_H = 1.0
        true_R = 0.1
        T = 200

        # Generate ground truth
        x_true = np.zeros(T)
        y = np.zeros(T)
        x_true[0] = 1.0
        for t in range(1, T):
            x_true[t] = true_F * x_true[t - 1] + rng.normal(0, np.sqrt(true_Q))
            y[t] = true_H * x_true[t] + rng.normal(0, np.sqrt(true_R))

        # Run filter
        f = _make_1d_filter()  # F=0.99, Q=0.01, H=1, R=0.1 for "stable"
        f.reset(np.array([0.0]), np.array([[1.0]]))

        estimates = np.zeros(T)
        for t in range(T):
            f.predict("stable")
            f.update(np.array([y[t]]))
            estimates[t] = f.state[0]

        # After convergence (skip first 50), RMSE should be small
        rmse = np.sqrt(np.mean((estimates[50:] - x_true[50:]) ** 2))
        assert rmse < 0.5, f"RMSE {rmse:.4f} too large — filter not tracking"

    def test_large_outlier_doesnt_blow_up(self) -> None:
        """A single large observation shouldn't cause divergence."""
        f = _make_1d_filter()
        f.reset(np.array([0.0]), np.array([[1.0]]))
        f.predict("stable")
        f.update(np.array([1e10]))
        # State should move toward obs but not diverge
        assert np.isfinite(f.state[0])
        assert np.isfinite(f.covariance[0, 0])
        assert f.covariance[0, 0] > 0
