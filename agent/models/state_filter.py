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

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from agent.models.belief import BeliefState

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
