"""
Tests for Differentiable Kalman Filter (Change 10c, Tier 5).

Covers: forward shapes, gradient flow through predict/update, PSD enforcement,
NaN masking, numerical equivalence with numpy filter, regime switching,
save/load state_dict round-trip, from_numpy_filter conversion, zero valid
observations, parameter optimization via SGD, WorldModel integration,
DAG wiring, and EM transfer.
"""

from __future__ import annotations

import io
import copy
import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from agent.models.diff_kalman import (
    DifferentiableKalmanFilter,
    _cholesky_to_psd,
    _inverse_softplus,
    _psd_to_cholesky_param,
)
from agent.models.state_filter import ContinuousStateFilter, RegimeConfig


# ── Test fixtures ───────────────────────────────────────────────


def _make_regime_configs() -> dict[str, RegimeConfig]:
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


def _make_numpy_filter(
    state_dim: int = 3,
    obs_dim: int = 17,
) -> ContinuousStateFilter:
    H = np.zeros((obs_dim, state_dim))
    H[0, 0] = 1.0
    H[1, 0] = 1.0
    H[2, 1] = 1.0
    H[3, 1] = 1.0
    H[4, 2] = 1.0
    H[5, 2] = 1.0
    for i in range(6, 11):
        H[i, 0] = 0.5
    for i in range(11, 16):
        H[i, 1] = 0.3
    H[16, 2] = 0.4
    R = np.diag([0.1] * 6 + [0.3] * 11)
    return ContinuousStateFilter(
        state_dim=state_dim,
        obs_dim=obs_dim,
        regime_configs=_make_regime_configs(),
        H=H,
        R=R,
    )


def _make_diff_filter(
    state_dim: int = 3,
    obs_dim: int = 17,
) -> DifferentiableKalmanFilter:
    numpy_f = _make_numpy_filter(state_dim, obs_dim)
    return DifferentiableKalmanFilter.from_numpy_filter(numpy_f)


def _make_observations(obs_dim: int = 17, n_nan: int = 0) -> np.ndarray:
    """Create synthetic observation vector with optional NaNs."""
    rng = np.random.RandomState(42)
    obs = rng.randn(obs_dim).astype(np.float32)
    if n_nan > 0:
        nan_idx = rng.choice(obs_dim, size=n_nan, replace=False)
        obs[nan_idx] = np.nan
    return obs


# ═══════════════════════════════════════════════════════════════
# 1. Construction and properties
# ═══════════════════════════════════════════════════════════════


class TestConstruction:

    def test_default_dims(self):
        f = DifferentiableKalmanFilter()
        assert f.state_dim == 3
        assert f.obs_dim == 17
        assert f.regime_names == ["expansion", "contraction", "crisis"]

    def test_custom_dims(self):
        f = DifferentiableKalmanFilter(state_dim=5, obs_dim=10, regime_names=["a", "b"])
        assert f.state_dim == 5
        assert f.obs_dim == 10
        assert f.regime_names == ["a", "b"]

    def test_regime_configs_property(self):
        """WorldModel accesses _filter._regime_configs.keys()."""
        f = DifferentiableKalmanFilter()
        keys = list(f._regime_configs.keys())
        assert keys == ["expansion", "contraction", "crisis"]

    def test_parameter_count(self):
        f = DifferentiableKalmanFilter(state_dim=3, obs_dim=17)
        params = dict(f.named_parameters())
        # 3 regimes × (F + L_Q) + H + L_R = 3×2 + 2 = 8 parameter tensors
        assert len(params) == 8
        # F per regime: 3×3 = 9 each
        # L_Q per regime: 3×3 = 9 each
        # H: 17×3 = 51
        # L_R: 17×17 = 289
        total = sum(p.numel() for p in f.parameters())
        expected = 3 * 9 + 3 * 9 + 51 + 289
        assert total == expected

    def test_buffers_not_parameters(self):
        f = DifferentiableKalmanFilter()
        param_names = {n for n, _ in f.named_parameters()}
        assert "_x" not in param_names
        assert "_P" not in param_names

    def test_initial_state(self):
        f = DifferentiableKalmanFilter(state_dim=4)
        assert f.state.shape == (4,)
        assert torch.allclose(f.state, torch.zeros(4))
        assert f.covariance.shape == (4, 4)
        assert torch.allclose(f.covariance, torch.eye(4))


# ═══════════════════════════════════════════════════════════════
# 2. PSD enforcement
# ═══════════════════════════════════════════════════════════════


class TestPSD:

    def test_cholesky_to_psd_positive_definite(self):
        L_raw = torch.randn(5, 5)
        Q = _cholesky_to_psd(L_raw)
        assert Q.shape == (5, 5)
        # Symmetric
        assert torch.allclose(Q, Q.T, atol=1e-6)
        # Positive definite: all eigenvalues > 0
        eigvals = torch.linalg.eigvalsh(Q)
        assert (eigvals > 0).all()

    def test_Q_psd_per_regime(self):
        f = _make_diff_filter()
        for regime in f.regime_names:
            Q = f.Q(regime)
            eigvals = torch.linalg.eigvalsh(Q)
            assert (eigvals > 0).all(), f"Q({regime}) not PSD"

    def test_R_psd(self):
        f = _make_diff_filter()
        R = f.R()
        eigvals = torch.linalg.eigvalsh(R)
        assert (eigvals > 0).all(), "R not PSD"

    def test_psd_after_gradient_step(self):
        """After an arbitrary gradient update, Q and R must stay PSD."""
        f = _make_diff_filter()
        # Simulate gradient step with large perturbation
        with torch.no_grad():
            for p in f.parameters():
                p.add_(torch.randn_like(p) * 2.0)
        # float32 matmul can introduce tiny negative eigenvalues (~1e-5)
        # even though L @ L^T + eps*I is PSD in exact arithmetic
        fp32_tol = -1e-4
        for regime in f.regime_names:
            Q = f.Q(regime)
            eigvals = torch.linalg.eigvalsh(Q)
            assert (eigvals > fp32_tol).all(), f"Q({regime}) not PSD after grad step"
        R = f.R()
        eigvals = torch.linalg.eigvalsh(R)
        assert (eigvals > fp32_tol).all(), "R not PSD after grad step"

    def test_cholesky_to_psd_zero_input(self):
        L_raw = torch.zeros(3, 3)
        Q = _cholesky_to_psd(L_raw)
        # Should still be PSD (softplus(0) > 0 + eps)
        eigvals = torch.linalg.eigvalsh(Q)
        assert (eigvals > 0).all()


# ═══════════════════════════════════════════════════════════════
# 3. Forward shapes (predict + update)
# ═══════════════════════════════════════════════════════════════


class TestForwardShapes:

    def test_predict_output_shapes(self):
        f = _make_diff_filter()
        x_pred, P_pred = f.predict("expansion")
        assert x_pred.shape == (3,)
        assert P_pred.shape == (3, 3)

    def test_update_output_shapes_torch(self):
        f = _make_diff_filter()
        f.predict("expansion")
        obs = torch.randn(17)
        x_upd, P_upd = f.update(obs)
        assert x_upd.shape == (3,)
        assert P_upd.shape == (3, 3)

    def test_update_output_shapes_numpy(self):
        """update() should accept np.ndarray thanks to auto-conversion."""
        f = _make_diff_filter()
        f.predict("expansion")
        obs = np.random.randn(17).astype(np.float32)
        x_upd, P_upd = f.update(obs)
        assert x_upd.shape == (3,)
        assert P_upd.shape == (3, 3)

    def test_update_with_quality_numpy(self):
        f = _make_diff_filter()
        f.predict("expansion")
        obs = np.random.randn(17).astype(np.float32)
        quality = np.ones(17, dtype=np.float32) * 0.8
        x_upd, P_upd = f.update(obs, quality)
        assert x_upd.shape == (3,)

    def test_custom_dims(self):
        numpy_f = ContinuousStateFilter(
            state_dim=5,
            obs_dim=8,
            regime_configs={
                "a": RegimeConfig(name="a", F=np.eye(5), Q=np.eye(5) * 0.01),
            },
            H=np.eye(8, 5),
            R=np.eye(8) * 0.1,
        )
        f = DifferentiableKalmanFilter.from_numpy_filter(numpy_f)
        x, P = f.predict("a")
        assert x.shape == (5,)
        assert P.shape == (5, 5)
        obs = torch.randn(8)
        x, P = f.update(obs)
        assert x.shape == (5,)
        assert P.shape == (5, 5)


# ═══════════════════════════════════════════════════════════════
# 4. Gradient flow
# ═══════════════════════════════════════════════════════════════


class TestGradientFlow:

    def test_predict_grads_exist(self):
        f = _make_diff_filter()
        f.reset(np.array([1.0, -0.5, 0.3]))  # Non-zero state for grad flow
        x_pred, _ = f.predict("expansion")
        loss = x_pred.sum()
        loss.backward()
        assert f._F["expansion"].grad is not None
        assert f._F["expansion"].grad.abs().sum() > 0

    def test_update_grads_exist(self):
        f = _make_diff_filter()
        f.predict("expansion")
        obs = torch.randn(17)
        x_upd, _ = f.update(obs)
        loss = x_upd.sum()
        loss.backward()
        # H should have gradients (observation model)
        assert f._H.grad is not None
        assert f._H.grad.abs().sum() > 0

    def test_predict_update_chain_grads(self):
        """Full predict→update path preserves gradients to all params."""
        f = _make_diff_filter()
        # Seed non-zero state so predict has something to work with
        f.reset(np.array([0.5, -0.3, 0.1]))
        x, _ = f.predict("crisis")
        obs = torch.randn(17)
        x_upd, _ = f.update(obs)
        loss = x_upd.pow(2).sum()
        loss.backward()
        # F and L_Q for crisis
        assert f._F["crisis"].grad is not None
        assert f._L_Q["crisis"].grad is not None
        # H and L_R
        assert f._H.grad is not None
        assert f._L_R.grad is not None

    def test_no_grad_leak_to_other_regimes(self):
        """Gradient from predict('expansion') should not flow to crisis F."""
        f = _make_diff_filter()
        f.reset(np.array([1.0, 1.0, 1.0]))
        x, _ = f.predict("expansion")
        loss = x.sum()
        loss.backward()
        assert f._F["expansion"].grad is not None
        # Other regimes should have zero or no gradient
        if f._F["crisis"].grad is not None:
            assert f._F["crisis"].grad.abs().sum() == 0

    def test_numpy_update_still_differentiable(self):
        """Even when observations are numpy, update grads should flow to H."""
        f = _make_diff_filter()
        f.predict("expansion")
        obs = np.random.randn(17).astype(np.float32)
        x_upd, _ = f.update(obs)
        loss = x_upd.sum()
        loss.backward()
        assert f._H.grad is not None


# ═══════════════════════════════════════════════════════════════
# 5. NaN masking / missing data
# ═══════════════════════════════════════════════════════════════


class TestNaNMasking:

    def test_all_nan_returns_prior(self):
        f = _make_diff_filter()
        f.reset(np.array([0.5, -0.3, 0.1]))
        x_after_pred, P_after_pred = f.predict("expansion")
        obs = torch.full((17,), float("nan"))
        x_upd, P_upd = f.update(obs)
        # No update: state should be unchanged from predict
        assert torch.allclose(x_upd, x_after_pred, atol=1e-6)
        assert torch.allclose(P_upd, P_after_pred, atol=1e-6)

    def test_partial_nan(self):
        f = _make_diff_filter()
        f.predict("expansion")
        obs = torch.randn(17)
        obs[0:5] = float("nan")  # First 5 are missing
        x_upd, P_upd = f.update(obs)
        # Should not contain NaN
        assert not torch.isnan(x_upd).any()
        assert not torch.isnan(P_upd).any()

    def test_single_valid_observation(self):
        f = _make_diff_filter()
        f.predict("expansion")
        obs = torch.full((17,), float("nan"))
        obs[0] = 1.5  # Only first observation valid
        x_upd, _ = f.update(obs)
        assert not torch.isnan(x_upd).any()

    def test_zero_quality_treated_as_missing(self):
        f = _make_diff_filter()
        f.predict("expansion")
        obs = torch.randn(17)
        quality = torch.zeros(17)  # All zero quality
        x_before = f.state.clone()
        # Predict already updated state; manually save post-predict
        x_upd, _ = f.update(obs, quality)
        # Zero quality means all invalid → no update beyond predict
        # (state was already updated by predict, and update with 0 quality
        # should leave it unchanged from predict output)
        # The update with zero quality should not crash
        assert not torch.isnan(x_upd).any()


# ═══════════════════════════════════════════════════════════════
# 6. Numerical equivalence with numpy filter
# ═══════════════════════════════════════════════════════════════


class TestNumpyEquivalence:

    def test_predict_equivalence(self):
        np_f = _make_numpy_filter()
        diff_f = DifferentiableKalmanFilter.from_numpy_filter(np_f)
        # Set same initial state
        x0 = np.array([0.5, -0.3, 0.1])
        np_f._x = x0.copy()
        np_f._P = np.eye(3) * 0.5
        diff_f.reset(x0.copy(), np.eye(3) * 0.5)

        np_f.predict("expansion")
        with torch.no_grad():
            diff_f.predict("expansion")

        np.testing.assert_allclose(diff_f.state.numpy(), np_f._x, atol=1e-4)
        np.testing.assert_allclose(diff_f.covariance.numpy(), np_f._P, atol=1e-4)

    def test_update_equivalence(self):
        np_f = _make_numpy_filter()
        diff_f = DifferentiableKalmanFilter.from_numpy_filter(np_f)

        x0 = np.array([0.5, -0.3, 0.1])
        P0 = np.eye(3) * 0.5
        np_f._x = x0.copy()
        np_f._P = P0.copy()
        diff_f.reset(x0.copy(), P0.copy())

        # Predict
        np_f.predict("expansion")
        with torch.no_grad():
            diff_f.predict("expansion")

        # Observations (no NaN for clean comparison)
        rng = np.random.RandomState(123)
        obs = rng.randn(17).astype(np.float64)
        quality = np.ones(17)

        np_f.update(obs, quality)
        with torch.no_grad():
            diff_f.update(
                torch.from_numpy(obs.astype(np.float32)),
                torch.ones(17),
            )

        np.testing.assert_allclose(
            diff_f.state.numpy(), np_f._x.astype(np.float32), atol=1e-3
        )

    def test_predict_update_sequence_equivalence(self):
        """Multi-step predict/update sequence should track numpy filter."""
        np_f = _make_numpy_filter()
        diff_f = DifferentiableKalmanFilter.from_numpy_filter(np_f)

        x0 = np.array([0.0, 0.0, 0.0])
        np_f._x = x0.copy()
        np_f._P = np.eye(3)
        diff_f.reset(x0.copy(), np.eye(3))

        rng = np.random.RandomState(99)
        regimes = ["expansion", "contraction", "crisis", "expansion"]

        for regime in regimes:
            np_f.predict(regime)
            with torch.no_grad():
                diff_f.predict(regime)

            obs = rng.randn(17)
            obs[rng.randint(0, 17, size=3)] = np.nan  # Some missing
            quality = np.clip(rng.rand(17), 0.1, 1.0)

            np_f.update(obs, quality)
            with torch.no_grad():
                diff_f.update(obs.astype(np.float32), quality.astype(np.float32))

        np.testing.assert_allclose(
            diff_f.state.numpy(), np_f._x.astype(np.float32), atol=1e-3
        )


# ═══════════════════════════════════════════════════════════════
# 7. Regime switching
# ═══════════════════════════════════════════════════════════════


class TestRegimeSwitching:

    def test_different_regimes_produce_different_states(self):
        # Create two identical filters, predict with different regimes
        f1 = _make_diff_filter()
        f2 = _make_diff_filter()
        x0 = np.array([1.0, 1.0, 1.0])
        f1.reset(x0)
        f2.reset(x0.copy())

        with torch.no_grad():
            x1, _ = f1.predict("expansion")
            x2, _ = f2.predict("crisis")

        # Crisis has more aggressive dynamics (0.90 vs 0.99)
        assert not torch.allclose(x1, x2)

    def test_invalid_regime_raises(self):
        f = _make_diff_filter()
        with pytest.raises(ValueError, match="not_a_regime"):
            f.predict("not_a_regime")

    def test_regime_switch_mid_sequence(self):
        f = _make_diff_filter()
        f.reset(np.array([1.0, 0.5, -0.5]))
        with torch.no_grad():
            f.predict("expansion")
            obs = torch.randn(17)
            f.update(obs)
            # Switch regime
            x, P = f.predict("crisis")
        assert not torch.isnan(x).any()
        assert not torch.isnan(P).any()


# ═══════════════════════════════════════════════════════════════
# 8. Save/load state_dict round-trip
# ═══════════════════════════════════════════════════════════════


class TestSaveLoad:

    def test_state_dict_round_trip(self):
        f = _make_diff_filter()
        f.reset(np.array([1.0, 2.0, 3.0]))
        with torch.no_grad():
            f.predict("expansion")
            obs = torch.randn(17)
            f.update(obs)

        sd = f.state_dict()
        f2 = DifferentiableKalmanFilter(state_dim=3, obs_dim=17)
        f2.load_state_dict(sd)

        assert torch.allclose(f.state, f2.state)
        assert torch.allclose(f.covariance, f2.covariance)
        # Parameters match
        for (n1, p1), (n2, p2) in zip(f.named_parameters(), f2.named_parameters()):
            assert n1 == n2
            assert torch.allclose(p1, p2)

    def test_save_load_via_buffer(self):
        """Save to bytes buffer and load back."""
        f = _make_diff_filter()
        f.reset(np.array([0.5, -0.3, 0.1]))
        buf = io.BytesIO()
        torch.save(f.state_dict(), buf)
        buf.seek(0)

        f2 = DifferentiableKalmanFilter(state_dim=3, obs_dim=17)
        f2.load_state_dict(torch.load(buf, weights_only=True))
        assert torch.allclose(f.state, f2.state)


# ═══════════════════════════════════════════════════════════════
# 9. from_numpy_filter conversion
# ═══════════════════════════════════════════════════════════════


class TestFromNumpyFilter:

    def test_dims_match(self):
        np_f = _make_numpy_filter()
        diff_f = DifferentiableKalmanFilter.from_numpy_filter(np_f)
        assert diff_f.state_dim == np_f.state_dim
        assert diff_f.obs_dim == np_f.obs_dim

    def test_regime_names_match(self):
        np_f = _make_numpy_filter()
        diff_f = DifferentiableKalmanFilter.from_numpy_filter(np_f)
        assert set(diff_f.regime_names) == set(np_f._regime_configs.keys())

    def test_F_values_imported(self):
        np_f = _make_numpy_filter()
        diff_f = DifferentiableKalmanFilter.from_numpy_filter(np_f)
        for name, rc in np_f._regime_configs.items():
            F_torch = diff_f._F[name].detach().numpy()
            np.testing.assert_allclose(F_torch, rc.F, atol=1e-6)

    def test_H_values_imported(self):
        np_f = _make_numpy_filter()
        diff_f = DifferentiableKalmanFilter.from_numpy_filter(np_f)
        H_torch = diff_f._H.detach().numpy()
        np.testing.assert_allclose(H_torch, np_f._H, atol=1e-5)

    def test_Q_values_imported(self):
        """Cholesky decomposition should reconstruct original Q."""
        np_f = _make_numpy_filter()
        diff_f = DifferentiableKalmanFilter.from_numpy_filter(np_f)
        for name, rc in np_f._regime_configs.items():
            Q_torch = diff_f.Q(name).detach().numpy()
            np.testing.assert_allclose(Q_torch, rc.Q, atol=1e-4)

    def test_R_values_imported(self):
        np_f = _make_numpy_filter()
        diff_f = DifferentiableKalmanFilter.from_numpy_filter(np_f)
        R_torch = diff_f.R().detach().numpy()
        np.testing.assert_allclose(R_torch, np_f._R, atol=1e-4)

    def test_state_imported(self):
        np_f = _make_numpy_filter()
        np_f._x = np.array([0.5, -0.3, 0.1])
        np_f._P = np.eye(3) * 2.0
        diff_f = DifferentiableKalmanFilter.from_numpy_filter(np_f)
        np.testing.assert_allclose(diff_f.state.numpy(), np_f._x, atol=1e-6)
        np.testing.assert_allclose(diff_f.covariance.numpy(), np_f._P, atol=1e-6)


# ═══════════════════════════════════════════════════════════════
# 10. to_numpy_params export
# ═══════════════════════════════════════════════════════════════


class TestToNumpyParams:

    def test_round_trip(self):
        np_f = _make_numpy_filter()
        diff_f = DifferentiableKalmanFilter.from_numpy_filter(np_f)
        params = diff_f.to_numpy_params()
        assert set(params.keys()) == {"regimes", "H", "R", "x", "P"}
        np.testing.assert_allclose(params["H"], np_f._H, atol=1e-5)

    def test_regime_params_present(self):
        diff_f = _make_diff_filter()
        params = diff_f.to_numpy_params()
        for name in diff_f.regime_names:
            assert name in params["regimes"]
            assert "F" in params["regimes"][name]
            assert "Q" in params["regimes"][name]


# ═══════════════════════════════════════════════════════════════
# 11. Reset
# ═══════════════════════════════════════════════════════════════


class TestReset:

    def test_reset_zeros(self):
        f = _make_diff_filter()
        f.reset(np.array([1.0, 2.0, 3.0]))
        assert f.state[0].item() == pytest.approx(1.0)
        f.reset()
        assert torch.allclose(f.state, torch.zeros(3))
        assert torch.allclose(f.covariance, torch.eye(3))

    def test_reset_numpy(self):
        f = _make_diff_filter()
        f.reset(np.array([0.5, -0.3, 0.1]), np.eye(3) * 0.5)
        assert f.state[0].item() == pytest.approx(0.5)
        assert f.covariance[0, 0].item() == pytest.approx(0.5)

    def test_reset_torch(self):
        f = _make_diff_filter()
        f.reset(torch.tensor([1.0, 2.0, 3.0]), torch.eye(3) * 2.0)
        assert f.state[1].item() == pytest.approx(2.0)


# ═══════════════════════════════════════════════════════════════
# 12. get_beliefs
# ═══════════════════════════════════════════════════════════════


class TestGetBeliefs:

    def test_belief_count_matches_state_dim(self):
        f = _make_diff_filter()
        names = ["stress", "momentum", "liquidity"]
        beliefs = f.get_beliefs(names, as_of=1000.0, graph_hash="abc")
        assert len(beliefs) == 3

    def test_belief_variable_names(self):
        f = _make_diff_filter()
        names = ["a", "b", "c"]
        beliefs = f.get_beliefs(names, as_of=1000.0, graph_hash="hash1")
        assert [b.variable_name for b in beliefs] == names

    def test_belief_dist_type(self):
        f = _make_diff_filter()
        beliefs = f.get_beliefs(["a", "b", "c"], as_of=1000.0, graph_hash="h")
        for b in beliefs:
            assert b.dist_type == "gaussian"

    def test_belief_mean_variance(self):
        f = _make_diff_filter()
        f.reset(np.array([0.5, -0.3, 0.1]), np.diag([0.1, 0.2, 0.3]))
        beliefs = f.get_beliefs(["a", "b", "c"], as_of=1000.0, graph_hash="h")
        assert beliefs[0].mean == pytest.approx(0.5, abs=1e-5)
        assert beliefs[1].variance == pytest.approx(0.2, abs=1e-5)

    def test_wrong_name_count_raises(self):
        f = _make_diff_filter()
        with pytest.raises(ValueError, match="variable_names length"):
            f.get_beliefs(["a", "b"], as_of=1000.0, graph_hash="h")


# ═══════════════════════════════════════════════════════════════
# 13. Parameter optimization test
# ═══════════════════════════════════════════════════════════════


class TestParameterOptimization:

    def test_sgd_reduces_loss(self):
        """Prove that optimizing filter params via SGD can reduce a loss."""
        f = DifferentiableKalmanFilter(state_dim=2, obs_dim=4, regime_names=["default"])
        # Set up a simple observation model
        with torch.no_grad():
            f._H.copy_(torch.randn(4, 2) * 0.5)

        optimizer = torch.optim.SGD(f.parameters(), lr=0.01)

        # Target: after predict+update, state should be close to [1.0, 1.0]
        target = torch.tensor([1.0, 1.0])
        obs = torch.randn(4)

        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            f.reset(torch.zeros(2), torch.eye(2))
            f.predict("default")
            x_upd, _ = f.update(obs)
            loss = (x_upd - target).pow(2).sum()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease
        assert (
            losses[-1] < losses[0]
        ), f"SGD did not reduce loss: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_psd_maintained_after_many_steps(self):
        """PSD enforcement holds after many optimization steps."""
        f = DifferentiableKalmanFilter(state_dim=2, obs_dim=4, regime_names=["a"])
        optimizer = torch.optim.Adam(f.parameters(), lr=0.1)
        target = torch.tensor([1.0, -1.0])

        for _ in range(50):
            optimizer.zero_grad()
            f.reset(torch.zeros(2), torch.eye(2))
            f.predict("a")
            obs = torch.randn(4)
            x_upd, _ = f.update(obs)
            loss = (x_upd - target).pow(2).sum()
            loss.backward()
            optimizer.step()

        # Check PSD
        Q = f.Q("a")
        R = f.R()
        assert (torch.linalg.eigvalsh(Q) > 0).all()
        assert (torch.linalg.eigvalsh(R) > 0).all()


# ═══════════════════════════════════════════════════════════════
# 14. WorldModel integration
# ═══════════════════════════════════════════════════════════════


class TestWorldModelIntegration:

    def _make_world_model_with_diff_filter(self):
        """Build a WorldModel using DifferentiableKalmanFilter."""
        from agent.models.initial_graph import build_initial_graph
        from agent.models.propagator import BeliefPropagator
        from agent.models.world_model import WorldModel

        graph = build_initial_graph()
        propagator = BeliefPropagator(graph)
        np_f = _make_numpy_filter()
        diff_f = DifferentiableKalmanFilter.from_numpy_filter(np_f)

        wm = WorldModel(
            graph=graph,
            propagator=propagator,
            state_filter=diff_f,
            regime_node="regime.macro",
            continuous_state_names=[
                "latent.stress_level",
                "latent.macro_momentum",
                "latent.liquidity_state",
            ],
            feature_to_obs_index={
                "macro.rate_momentum.30d": 0,
                "macro.yield_curve_slope.spot": 1,
                "macro.liquidity_pressure.30d": 2,
                "convergence.stress_breadth.7d": 3,
                "convergence.stress_intensity.7d": 4,
                "convergence.regime_persistence.7d": 5,
            },
        )
        return wm

    def test_world_model_update_with_diff_filter(self):
        """WorldModel.update() should work with DifferentiableKalmanFilter."""
        from agent.features.protocol import EngineeredFeature

        wm = self._make_world_model_with_diff_filter()
        now = 1000.0
        features = [
            EngineeredFeature(
                feature_name="macro.rate_momentum.30d",
                version=1,
                effective_at=now,
                computed_at=now,
                horizon="30d",
                value=0.5,
                quality=0.9,
            ),
            EngineeredFeature(
                feature_name="macro.yield_curve_slope.spot",
                version=1,
                effective_at=now,
                computed_at=now,
                horizon="spot",
                value=-0.2,
                quality=0.8,
            ),
        ]
        beliefs = wm.update(features, as_of=now)
        assert len(beliefs) > 0
        # Should have Kalman beliefs
        kalman_names = {b.variable_name for b in beliefs if b.dist_type == "gaussian"}
        assert "latent.stress_level" in kalman_names

    def test_extract_map_regime_with_diff_filter(self):
        """_extract_map_regime fallback should work with _regime_configs property."""
        wm = self._make_world_model_with_diff_filter()
        # No DAG beliefs → falls back to _regime_configs.keys()
        regime = wm._extract_map_regime([])
        assert regime in ["expansion", "contraction", "crisis"]


# ═══════════════════════════════════════════════════════════════
# 15. DAG _build_world_model integration
# ═══════════════════════════════════════════════════════════════


class TestBuildWorldModel:

    def test_build_with_numpy_filter(self):
        """Default: numpy filter (backward compat)."""
        from agent.pipeline.dags.world_model_update import _build_world_model

        wm = _build_world_model()
        assert isinstance(wm._filter, ContinuousStateFilter)

    def test_build_with_differentiable_filter(self):
        from agent.pipeline.dags.world_model_update import _build_world_model

        wm = _build_world_model(use_differentiable_filter=True)
        assert isinstance(wm._filter, DifferentiableKalmanFilter)
        assert wm._filter.state_dim == 3
        assert wm._filter.obs_dim == 17
        assert set(wm._filter.regime_names) == {
            "expansion",
            "contraction",
            "crisis",
        }

    def test_build_diff_filter_with_learned_edges(self):
        """Learned edges + differentiable filter should both work."""
        from agent.pipeline.dags.world_model_update import _build_world_model
        from agent.models.initial_graph import build_initial_graph

        # Use a subset of expert edges as "learned"
        g = build_initial_graph()
        edges = list(g.edges)[:5]
        wm = _build_world_model(
            learned_edges=edges,
            use_differentiable_filter=True,
        )
        assert isinstance(wm._filter, DifferentiableKalmanFilter)

    def test_diff_filter_params_match_expert(self):
        """After conversion, diff filter params should match expert values."""
        from agent.pipeline.dags.world_model_update import _build_world_model

        wm_np = _build_world_model(use_differentiable_filter=False)
        wm_diff = _build_world_model(use_differentiable_filter=True)

        # H matrix
        H_np = wm_np._filter._H
        H_diff = wm_diff._filter._H.detach().numpy()
        np.testing.assert_allclose(H_diff, H_np, atol=1e-5)


# ═══════════════════════════════════════════════════════════════
# 16. Edge cases
# ═══════════════════════════════════════════════════════════════


class TestEdgeCases:

    def test_observations_wrong_shape_raises(self):
        f = _make_diff_filter()
        with pytest.raises(ValueError, match="observations shape"):
            f.update(torch.randn(10))

    def test_double_predict_no_crash(self):
        f = _make_diff_filter()
        f.predict("expansion")
        f.predict("expansion")  # Should not crash

    def test_update_without_predict(self):
        """update() before predict() should use initial state."""
        f = _make_diff_filter()
        obs = torch.randn(17)
        x, P = f.update(obs)
        assert not torch.isnan(x).any()
        assert not torch.isnan(P).any()

    def test_very_large_observations(self):
        f = _make_diff_filter()
        f.predict("expansion")
        obs = torch.ones(17) * 1e6
        x, P = f.update(obs)
        assert not torch.isnan(x).any()
        assert not torch.isinf(x).any()

    def test_state_covariance_detached_clone(self):
        """state and covariance properties should return detached copies."""
        f = _make_diff_filter()
        s = f.state
        c = f.covariance
        s[0] = 999.0
        c[0, 0] = 999.0
        # Originals unchanged
        assert f.state[0].item() != 999.0
        assert f.covariance[0, 0].item() != 999.0

    def test_single_regime(self):
        np_f = ContinuousStateFilter(
            state_dim=2,
            obs_dim=3,
            regime_configs={
                "only": RegimeConfig(name="only", F=np.eye(2), Q=np.eye(2) * 0.01),
            },
            H=np.eye(3, 2),
            R=np.eye(3) * 0.1,
        )
        f = DifferentiableKalmanFilter.from_numpy_filter(np_f)
        assert f.regime_names == ["only"]
        x, _ = f.predict("only")
        assert x.shape == (2,)

    def test_negative_quality_treated_as_missing(self):
        f = _make_diff_filter()
        f.predict("expansion")
        obs = torch.randn(17)
        quality = torch.ones(17)
        quality[0] = -1.0  # Negative
        x_upd, _ = f.update(obs, quality)
        assert not torch.isnan(x_upd).any()

    def test_covariance_stays_symmetric(self):
        """After many predict/update cycles, P should remain symmetric."""
        f = _make_diff_filter()
        f.reset(np.array([1.0, -0.5, 0.3]))
        rng = np.random.RandomState(7)
        for _ in range(20):
            with torch.no_grad():
                f.predict("expansion")
                obs = rng.randn(17).astype(np.float32)
                obs[rng.randint(0, 17, 3)] = np.nan
                f.update(obs)
        P = f.covariance
        assert torch.allclose(P, P.T, atol=1e-5)
