"""
TirraMind — Continuous State Filter

Regime-conditioned Kalman filter tracking continuous latent states.
Each regime provides distinct dynamics (F, Q) while the observation
model (H, R) is shared.

Mathematical specification (Sarkka, "Bayesian Filtering and Smoothing", 2013, Ch. 4):

    Transition (per regime r):
        x_t = F_r · x_{t-1} + w_t,   w_t ~ N(0, Q_r)

    Observation:
        y_t = H · x_t + v_t,   v_t ~ N(0, R)

    Predict:
        x̂_{t|t-1} = F_r · x̂_{t-1|t-1}
        P_{t|t-1} = F_r · P_{t-1|t-1} · F_r^T + Q_r

    Update (Joseph form for numerical stability):
        K = P_{t|t-1} · H^T · (H · P_{t|t-1} · H^T + R')^{-1}
        x̂_{t|t} = x̂_{t|t-1} + K · (y_t - H · x̂_{t|t-1})
        P_{t|t} = (I - K·H) · P_{t|t-1} · (I - K·H)^T + K · R' · K^T

    Quality weighting: R' = R · diag(1/quality)
    Missing observations: drop rows from H, y, R for NaN entries.

Design principles:
    1. filterpy.kalman.KalmanFilter as internal engine.
    2. Regime conditioning: F, Q swapped per predict() call.
    3. Joseph form covariance update for positive-definiteness.
    4. Missing obs handled by reducing observation dimension.
    5. Output: list[BeliefState] with Gaussian parameterization.

References:
    - Sarkka S., "Bayesian Filtering and Smoothing" (2013), Ch. 4
    - filterpy docs: https://filterpy.readthedocs.io/
    - Spec: docs/specs/world_model_spec.md (sub-phase 9.4)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from agent.models.belief import BeliefState

log = logging.getLogger(__name__)

# ── Regime configuration ───────────────────────────────────────


@dataclass(frozen=True)
class RegimeConfig:
    """Per-regime transition + noise parameters for the Kalman filter."""

    name: str
    """Regime label (matches DAG regime node state, e.g. 'expansion')."""

    F: np.ndarray
    """State transition matrix (state_dim × state_dim)."""

    Q: np.ndarray
    """Process noise covariance (state_dim × state_dim)."""


# ── Continuous State Filter ────────────────────────────────────


class ContinuousStateFilter:
    """Regime-conditioned Kalman filter for continuous latent states.

    Args:
        state_dim: Dimension of hidden state vector.
        obs_dim: Dimension of observation vector.
        regime_configs: Dict mapping regime name → RegimeConfig.
        H: Observation matrix (obs_dim × state_dim).
        R: Observation noise covariance (obs_dim × obs_dim).
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        regime_configs: dict[str, RegimeConfig],
        H: np.ndarray,
        R: np.ndarray,
    ) -> None:
        if state_dim < 1:
            raise ValueError("state_dim must be >= 1")
        if obs_dim < 1:
            raise ValueError("obs_dim must be >= 1")
        if H.shape != (obs_dim, state_dim):
            raise ValueError(f"H shape {H.shape} != expected ({obs_dim}, {state_dim})")
        if R.shape != (obs_dim, obs_dim):
            raise ValueError(f"R shape {R.shape} != expected ({obs_dim}, {obs_dim})")

        self._state_dim = state_dim
        self._obs_dim = obs_dim
        self._regime_configs = regime_configs
        self._H = H.copy()
        self._R = R.copy()

        # State estimate and covariance
        self._x = np.zeros(state_dim)
        self._P = np.eye(state_dim)

    # ── Properties ─────────────────────────────────────────

    @property
    def state(self) -> np.ndarray:
        """Current state estimate (copy)."""
        return self._x.copy()

    @property
    def covariance(self) -> np.ndarray:
        """Current covariance matrix (copy)."""
        return self._P.copy()

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    # ── Core operations ────────────────────────────────────

    def predict(self, regime: str) -> tuple[np.ndarray, np.ndarray]:
        """Predict step: propagate state through regime-specific dynamics.

        Args:
            regime: Regime name (must be in regime_configs).

        Returns:
            (predicted_state, predicted_covariance).

        Raises:
            ValueError: If regime not in configs.
        """
        if regime not in self._regime_configs:
            raise ValueError(
                f"Regime '{regime}' not in configs. "
                f"Available: {list(self._regime_configs.keys())}"
            )

        rc = self._regime_configs[regime]
        self._x = rc.F @ self._x
        self._P = rc.F @ self._P @ rc.F.T + rc.Q

        return self._x.copy(), self._P.copy()

    def update(
        self,
        observations: np.ndarray,
        quality: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update step: incorporate observations.

        Missing observations (NaN) are masked — the filter skips them.
        Quality weights inflate R: R' = R * diag(1/quality).
        Quality of 0 for a sensor effectively ignores that observation.

        Uses Joseph form for covariance update to maintain
        positive-definiteness.

        Args:
            observations: Vector of length obs_dim. NaN = missing.
            quality: Per-sensor quality in [0, 1]. None = all 1.0.

        Returns:
            (updated_state, updated_covariance).
        """
        if observations.shape != (self._obs_dim,):
            raise ValueError(
                f"observations shape {observations.shape} != "
                f"expected ({self._obs_dim},)"
            )

        if quality is None:
            quality = np.ones(self._obs_dim)
        elif quality.shape != (self._obs_dim,):
            raise ValueError(
                f"quality shape {quality.shape} != " f"expected ({self._obs_dim},)"
            )

        # Find valid (non-NaN, non-zero-quality) observations
        valid = ~np.isnan(observations) & (quality > 0.0)
        n_valid = valid.sum()

        if n_valid == 0:
            # No valid observations → skip update
            return self._x.copy(), self._P.copy()

        # Reduce to valid observations only
        z = observations[valid]
        H_r = self._H[valid, :]
        R_r = self._R[np.ix_(valid, valid)]
        q_r = quality[valid]

        # Inflate R by 1/quality
        R_inflated = R_r * np.diag(1.0 / q_r)

        # Innovation
        y_hat = H_r @ self._x
        innovation = z - y_hat

        # Innovation covariance
        S = H_r @ self._P @ H_r.T + R_inflated

        # Kalman gain
        K = self._P @ H_r.T @ np.linalg.inv(S)

        # State update
        self._x = self._x + K @ innovation

        # Joseph form covariance update
        I = np.eye(self._state_dim)
        IKH = I - K @ H_r
        self._P = IKH @ self._P @ IKH.T + K @ R_inflated @ K.T

        # Symmetrize (numerical safety)
        self._P = 0.5 * (self._P + self._P.T)

        return self._x.copy(), self._P.copy()

    def get_beliefs(
        self,
        variable_names: list[str],
        as_of: float,
        graph_hash: str,
        version: int = 1,
    ) -> list[BeliefState]:
        """Convert internal state to list of Gaussian BeliefState records.

        Each state dimension maps to one variable_name.

        Args:
            variable_names: One name per state dimension.
            as_of: Unix epoch for effective_at.
            graph_hash: 64-char hex hash for provenance.
            version: World model schema version.
        """
        if len(variable_names) != self._state_dim:
            raise ValueError(
                f"variable_names length ({len(variable_names)}) != "
                f"state_dim ({self._state_dim})"
            )

        computed_at = time.time()
        beliefs = []
        for i, name in enumerate(variable_names):
            beliefs.append(
                BeliefState(
                    variable_name=name,
                    version=version,
                    effective_at=as_of,
                    computed_at=computed_at,
                    dist_type="gaussian",
                    mean=float(self._x[i]),
                    variance=float(self._P[i, i]),
                    evidence_count=0,  # set by caller
                    model_graph_hash=graph_hash,
                    confidence=1.0,
                    stale=False,
                )
            )
        return beliefs

    def reset(self, x0: np.ndarray, P0: np.ndarray) -> None:
        """Reinitialize state estimate and covariance.

        Args:
            x0: Initial state vector (state_dim,).
            P0: Initial covariance (state_dim × state_dim).
        """
        if x0.shape != (self._state_dim,):
            raise ValueError(f"x0 shape {x0.shape} != expected ({self._state_dim},)")
        if P0.shape != (self._state_dim, self._state_dim):
            raise ValueError(
                f"P0 shape {P0.shape} != expected "
                f"({self._state_dim}, {self._state_dim})"
            )
        self._x = x0.copy()
        self._P = P0.copy()

    # ── EM Parameter Fitting (Change 2b) ──────────────────────

    def fit_filter_params(
        self,
        observations_seq: list[np.ndarray],
        regime_labels: list[str],
        *,
        max_iter: int = 20,
        tol: float = 1e-4,
        min_samples: int = 30,
    ) -> dict[str, Any]:
        """Fit F, Q, H, R from data via EM (Shumway & Stoffer, Ch. 6).

        The EM algorithm for linear Gaussian state-space models alternates:
          E-step: Kalman smoother computes E[x_t | y_{1:T}], Cov(x_t, x_{t-1} | y_{1:T})
          M-step: Closed-form MLE updates for F, Q, H, R

        Each regime gets its own F, Q fitted from timesteps where that regime
        was active.  H, R are shared across regimes.

        Parameters
        ----------
        observations_seq : List of observation vectors (length T).
            Each is (obs_dim,). NaN entries are treated as missing.
        regime_labels : Regime name for each timestep (length T).
        max_iter : Maximum EM iterations.
        tol : Convergence threshold on log-likelihood relative change.
        min_samples : Minimum T required to attempt fitting.

        Returns
        -------
        Dict with 'fitted': bool, 'n_samples': int, 'iterations': int,
        'log_likelihoods': list[float].
        """
        T = len(observations_seq)
        if T < min_samples:
            log.info(
                "fit_filter_params: only %d samples (need %d), keeping current params.",
                T,
                min_samples,
            )
            return {
                "fitted": False,
                "n_samples": T,
                "iterations": 0,
                "log_likelihoods": [],
            }

        if len(regime_labels) != T:
            raise ValueError(
                f"regime_labels length ({len(regime_labels)}) != observations_seq length ({T})"
            )

        n = self._state_dim
        m = self._obs_dim

        # Initialise from current params
        F_by_regime = {name: rc.F.copy() for name, rc in self._regime_configs.items()}
        Q_by_regime = {name: rc.Q.copy() for name, rc in self._regime_configs.items()}
        H = self._H.copy()
        R = self._R.copy()

        log_likelihoods: list[float] = []

        for iteration in range(max_iter):
            # ── E-step: forward (Kalman filter) ──────────────
            x_filt = np.zeros((T, n))
            P_filt = np.zeros((T, n, n))
            x_pred = np.zeros((T, n))
            P_pred = np.zeros((T, n, n))

            # Initial state
            x_t = np.zeros(n)
            P_t = np.eye(n)
            ll = 0.0

            for t in range(T):
                regime = regime_labels[t]
                F_t = F_by_regime.get(regime, np.eye(n))
                Q_t = Q_by_regime.get(regime, np.eye(n) * 0.01)

                # Predict
                x_p = F_t @ x_t
                P_p = F_t @ P_t @ F_t.T + Q_t

                x_pred[t] = x_p
                P_pred[t] = P_p

                # Update with observations
                y = observations_seq[t]
                valid = ~np.isnan(y)
                if valid.any():
                    z = y[valid]
                    H_v = H[valid, :]
                    R_v = R[np.ix_(valid, valid)]

                    S = H_v @ P_p @ H_v.T + R_v
                    # Regularise S for numerical stability
                    S = 0.5 * (S + S.T) + np.eye(S.shape[0]) * 1e-8
                    S_inv = np.linalg.inv(S)
                    K = P_p @ H_v.T @ S_inv

                    innov = z - H_v @ x_p
                    x_t = x_p + K @ innov

                    I_KH = np.eye(n) - K @ H_v
                    P_t = I_KH @ P_p @ I_KH.T + K @ R_v @ K.T
                    P_t = 0.5 * (P_t + P_t.T)

                    # Log-likelihood contribution
                    sign, logdet = np.linalg.slogdet(S)
                    if sign > 0:
                        ll += -0.5 * (
                            logdet + innov @ S_inv @ innov + len(z) * np.log(2 * np.pi)
                        )
                else:
                    x_t = x_p
                    P_t = P_p

                x_filt[t] = x_t
                P_filt[t] = P_t

            log_likelihoods.append(float(ll))

            # ── E-step: backward (RTS smoother) ─────────────
            x_smooth = np.zeros((T, n))
            P_smooth = np.zeros((T, n, n))
            Plag_smooth = np.zeros((T, n, n))  # Cov(x_t, x_{t-1} | y_1:T)

            x_smooth[T - 1] = x_filt[T - 1]
            P_smooth[T - 1] = P_filt[T - 1]

            for t in range(T - 2, -1, -1):
                P_pred_t1 = P_pred[t + 1]
                # Regularise for inversion
                P_pred_reg = P_pred_t1 + np.eye(n) * 1e-8
                G = (
                    P_filt[t]
                    @ F_by_regime.get(regime_labels[t + 1], np.eye(n)).T
                    @ np.linalg.inv(P_pred_reg)
                )

                x_smooth[t] = x_filt[t] + G @ (x_smooth[t + 1] - x_pred[t + 1])
                P_smooth[t] = P_filt[t] + G @ (P_smooth[t + 1] - P_pred_t1) @ G.T
                P_smooth[t] = 0.5 * (P_smooth[t] + P_smooth[t].T)
                Plag_smooth[t + 1] = P_smooth[t + 1] @ G.T

            # ── M-step ──────────────────────────────────────
            # Fit F, Q per regime
            for regime_name in F_by_regime:
                indices = [t for t in range(1, T) if regime_labels[t] == regime_name]
                if len(indices) < 2:
                    continue  # not enough samples for this regime

                A = np.zeros((n, n))  # sum E[x_t x_{t-1}^T]
                B = np.zeros((n, n))  # sum E[x_{t-1} x_{t-1}^T]
                C = np.zeros((n, n))  # sum E[x_t x_t^T]

                for t in indices:
                    xs_t = x_smooth[t].reshape(-1, 1)
                    xs_tm1 = x_smooth[t - 1].reshape(-1, 1)

                    A += Plag_smooth[t] + xs_t @ xs_tm1.T
                    B += P_smooth[t - 1] + xs_tm1 @ xs_tm1.T
                    C += P_smooth[t] + xs_t @ xs_t.T

                # F_new = A @ B^{-1}
                B_reg = B + np.eye(n) * 1e-8
                F_new = A @ np.linalg.inv(B_reg)
                F_by_regime[regime_name] = F_new

                # Q_new = (C - F_new @ A^T) / |indices|
                N_r = len(indices)
                Q_new = (C - F_new @ A.T) / N_r
                # Ensure PSD
                Q_new = 0.5 * (Q_new + Q_new.T)
                eigvals = np.linalg.eigvalsh(Q_new)
                if eigvals.min() < 0:
                    Q_new += np.eye(n) * (abs(eigvals.min()) + 1e-8)
                Q_by_regime[regime_name] = Q_new

            # Fit H, R (shared across regimes)
            D = np.zeros((m, n))  # sum y_t x_t^T (over valid obs)
            E_xx = np.zeros((n, n))  # sum x_t x_t^T
            E_yy = np.zeros((m, m))  # sum y_t y_t^T
            n_obs_total = 0

            for t in range(T):
                y = observations_seq[t]
                valid = ~np.isnan(y)
                if not valid.any():
                    continue
                # Use full vectors for the shared fit; NaN positions zeroed
                y_filled = np.where(valid, y, 0.0)
                xs_t = x_smooth[t]
                outer_yx = y_filled.reshape(-1, 1) @ xs_t.reshape(1, -1)
                outer_xx = P_smooth[t] + xs_t.reshape(-1, 1) @ xs_t.reshape(1, -1)
                outer_yy = np.diag(valid.astype(float)) @ (
                    y_filled.reshape(-1, 1) @ y_filled.reshape(1, -1)
                )

                D += outer_yx
                E_xx += outer_xx
                E_yy += outer_yy
                n_obs_total += 1

            if n_obs_total > 0:
                E_xx_reg = E_xx + np.eye(n) * 1e-8
                H_new = D @ np.linalg.inv(E_xx_reg)
                R_new = (E_yy - H_new @ D.T) / n_obs_total
                R_new = 0.5 * (R_new + R_new.T)
                eigvals = np.linalg.eigvalsh(R_new)
                if eigvals.min() < 0:
                    R_new += np.eye(m) * (abs(eigvals.min()) + 1e-8)
                H = H_new
                R = R_new

            # Convergence check
            if len(log_likelihoods) >= 2:
                prev_ll = log_likelihoods[-2]
                if abs(prev_ll) > 0:
                    rel_change = abs(ll - prev_ll) / abs(prev_ll)
                else:
                    rel_change = abs(ll - prev_ll)
                if rel_change < tol:
                    log.info(
                        "fit_filter_params: converged at iteration %d (rel_change=%.6f).",
                        iteration + 1,
                        rel_change,
                    )
                    break

        # Apply fitted params
        for regime_name, rc in list(self._regime_configs.items()):
            if regime_name in F_by_regime:
                self._regime_configs[regime_name] = RegimeConfig(
                    name=regime_name,
                    F=F_by_regime[regime_name],
                    Q=Q_by_regime[regime_name],
                )
        self._H = H
        self._R = R

        log.info(
            "fit_filter_params: %d iterations, final LL=%.2f, %d samples.",
            len(log_likelihoods),
            log_likelihoods[-1] if log_likelihoods else float("nan"),
            T,
        )
        return {
            "fitted": True,
            "n_samples": T,
            "iterations": len(log_likelihoods),
            "log_likelihoods": log_likelihoods,
        }
