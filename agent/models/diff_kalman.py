"""TirraMind — Differentiable Kalman Filter (Change 10c)

PyTorch reimplementation of the regime-conditioned Kalman filter from
``ContinuousStateFilter``, with all parameters as ``nn.Parameter`` so
gradients from downstream losses flow through predict/update ops.

Mathematical specification (Sarkka, "Bayesian Filtering and Smoothing",
2013, Ch. 4):

    Transition (per regime r):
        x_t = F_r · x_{t-1} + w_t,   w_t ~ N(0, Q_r)

    Observation:
        y_t = H · x_t + v_t,   v_t ~ N(0, R)

    Predict:
        x̂_{t|t-1} = F_r · x̂_{t-1|t-1}
        P_{t|t-1} = F_r · P_{t-1|t-1} · F_r^T + Q_r

    Update (Joseph form):
        K = P · H^T · (H P H^T + R')^{-1}
        x̂ = x̂ + K · (y − H x̂)
        P̂ = (I − KH) P (I − KH)^T + K R' K^T

PSD enforcement for Q, R via Cholesky parameterisation:
    Q = L_Q @ L_Q^T + ε·I   where diag(L_Q) enforced positive via softplus.
    R = L_R @ L_R^T + ε·I   same.

This module does NOT (yet) wire gradients through the full pipeline to
SAC — the SQLite belief round-trip is still a break.  The value is:
    1. Filter params are nn.Parameter → checkpointable, optimisable
    2. predict/update preserve autograd → ready for future end-to-end
    3. from_numpy_filter() imports expert / EM-fitted values
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from agent.models.belief import BeliefState

log = logging.getLogger(__name__)

_EPS = 1e-5  # PSD floor (needs headroom for float32 rounding on larger matrices)


def _inverse_softplus(x: Tensor) -> Tensor:
    """Numerically stable inverse of softplus.

    softplus(y) = log(1 + exp(y))  →  y = log(exp(x) - 1) = x + log(1 - exp(-x))
    For large x (>20), y ≈ x (avoids exp overflow).
    """
    return torch.where(
        x > 20.0,
        x,
        x + torch.log(-torch.expm1(-x)),
    )


def _cholesky_to_psd(L_raw: Tensor) -> Tensor:
    """Convert unconstrained lower-triangular L to PSD matrix.

    Enforces positive diagonal via softplus, then Q = L @ L^T + ε·I.
    """
    L = torch.tril(L_raw)
    # Enforce positive diagonal
    diag_positive = torch.nn.functional.softplus(torch.diagonal(L))
    L = L - torch.diag(torch.diagonal(L)) + torch.diag(diag_positive)
    return L @ L.T + _EPS * torch.eye(L.shape[0], device=L.device, dtype=L.dtype)


def _psd_to_cholesky_param(M: Tensor) -> Tensor:
    """Convert a PSD matrix to the unconstrained Cholesky parameter.

    Computes Cholesky factor L of M, then applies inverse softplus to the
    diagonal so that ``_cholesky_to_psd`` recovers the original matrix.
    """
    L = torch.linalg.cholesky(M)
    diag = torch.diagonal(L)
    inv_sp_diag = _inverse_softplus(diag)
    return L - torch.diag(diag) + torch.diag(inv_sp_diag)


class DifferentiableKalmanFilter(nn.Module):
    """Regime-conditioned Kalman filter with learnable parameters.

    All filter matrices (F, Q, H, R) are ``nn.Parameter``.  ``predict()``
    and ``update()`` use pure torch ops so autograd can back-prop through
    them.

    Parameters
    ----------
    state_dim : int
        Hidden state dimension (default 3).
    obs_dim : int
        Observation vector dimension (default 17).
    regime_names : list[str]
        Regime identifiers.  One (F, L_Q) pair per regime.
    """

    def __init__(
        self,
        state_dim: int = 3,
        obs_dim: int = 17,
        regime_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        if regime_names is None:
            regime_names = ["expansion", "contraction", "crisis"]

        self._state_dim = state_dim
        self._obs_dim = obs_dim
        self._regime_names = list(regime_names)

        # Per-regime transition + process noise (Cholesky)
        self._F = nn.ParameterDict()
        self._L_Q = nn.ParameterDict()
        for name in regime_names:
            self._F[name] = nn.Parameter(torch.eye(state_dim))
            self._L_Q[name] = nn.Parameter(
                torch.eye(state_dim) * 0.1  # small initial process noise
            )

        # Shared observation model
        self._H = nn.Parameter(torch.zeros(obs_dim, state_dim))
        self._L_R = nn.Parameter(torch.eye(obs_dim) * 0.3)  # initial observation noise

        # State (not parameters — running buffers)
        self.register_buffer("_x", torch.zeros(state_dim))
        self.register_buffer("_P", torch.eye(state_dim))

    # ── Properties ─────────────────────────────────────────

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def regime_names(self) -> list[str]:
        return list(self._regime_names)

    @property
    def state(self) -> Tensor:
        """Current state estimate (detached clone)."""
        return self._x.detach().clone()

    @property
    def covariance(self) -> Tensor:
        """Current covariance (detached clone)."""
        return self._P.detach().clone()

    def Q(self, regime: str) -> Tensor:
        """Compute PSD process noise for a regime."""
        return _cholesky_to_psd(self._L_Q[regime])

    def R(self) -> Tensor:
        """Compute PSD observation noise."""
        return _cholesky_to_psd(self._L_R)

    # ── Core operations ────────────────────────────────────

    @property
    def _regime_configs(self) -> dict[str, Any]:
        """Expose regime names with a dict-like interface.

        WorldModel._extract_map_regime() accesses
        ``self._filter._regime_configs.keys()``.  This property provides
        compatibility so the diff filter can be used as a drop-in.
        """
        return {name: None for name in self._regime_names}

    def predict(self, regime: str) -> tuple[Tensor, Tensor]:
        """Predict step with regime-specific dynamics.

        Returns (predicted_state, predicted_covariance).
        """
        if regime not in self._F:
            raise ValueError(f"Regime '{regime}' not in {list(self._F.keys())}")

        F_r = self._F[regime]
        Q_r = self.Q(regime)

        x_pred = F_r @ self._x
        P_pred = F_r @ self._P @ F_r.T + Q_r

        # Update internal state
        self._x = x_pred
        self._P = P_pred

        return x_pred.clone(), P_pred.clone()

    def update(
        self,
        observations: Tensor | np.ndarray,
        quality: Tensor | np.ndarray | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Update step: incorporate observations.

        Accepts both torch Tensors and numpy arrays (auto-converted).
        NaN observations are masked.  Quality weights inflate R:
        R' = R_valid * diag(1/quality_valid).

        Uses Joseph form for numerical stability.

        Returns (updated_state, updated_covariance).
        """
        # Auto-convert numpy → torch
        if isinstance(observations, np.ndarray):
            observations = torch.from_numpy(observations.astype(np.float32)).to(
                self._x.device
            )
        if isinstance(quality, np.ndarray):
            quality = torch.from_numpy(quality.astype(np.float32)).to(self._x.device)
        if observations.shape != (self._obs_dim,):
            raise ValueError(
                f"observations shape {observations.shape} != ({self._obs_dim},)"
            )

        if quality is None:
            quality = torch.ones(self._obs_dim, device=observations.device)

        # Valid = non-NaN and positive quality
        valid = ~torch.isnan(observations) & (quality > 0.0)
        n_valid = int(valid.sum().item())

        if n_valid == 0:
            return self._x.clone(), self._P.clone()

        # Reduce to valid observations
        z = observations[valid]
        H_r = self._H[valid, :]
        R_full = self.R()
        # Advanced indexing for submatrix
        valid_idx = torch.where(valid)[0]
        R_r = R_full[valid_idx][:, valid_idx]
        q_r = quality[valid]

        # Inflate R by 1/quality
        R_inflated = R_r * torch.diag(1.0 / q_r)

        # Innovation
        y_hat = H_r @ self._x
        innovation = z - y_hat

        # Innovation covariance S = H P H^T + R'
        S = H_r @ self._P @ H_r.T + R_inflated

        # Kalman gain: K = P H^T S^{-1}  (via solve for stability)
        # K = P @ H_r^T @ inv(S)  equiv to  S^T @ K^T = H_r @ P
        K = self._P @ H_r.T @ torch.linalg.solve(S, torch.eye(n_valid, device=S.device))

        # State update
        x_new = self._x + K @ innovation

        # Joseph form covariance update
        I = torch.eye(self._state_dim, device=self._P.device)
        IKH = I - K @ H_r
        P_new = IKH @ self._P @ IKH.T + K @ R_inflated @ K.T

        # Symmetrise
        P_new = 0.5 * (P_new + P_new.T)

        self._x = x_new
        self._P = P_new

        return x_new.clone(), P_new.clone()

    def get_beliefs_differentiable(self) -> tuple[Tensor, Tensor]:
        """Return belief means and variances WITHOUT detaching from autograd.

        Returns
        -------
        (means, variances) where:
            means : Tensor of shape (state_dim,) — current state estimate
            variances : Tensor of shape (state_dim,) — diagonal of covariance

        Used ONLY during training for model-based gradient augmentation.
        The existing ``get_beliefs()`` remains the interface for inference-time
        SQLite persistence (which requires Python floats, not tensors).
        """
        return self._x, torch.diagonal(self._P)

    def get_beliefs(
        self,
        variable_names: list[str],
        as_of: float,
        graph_hash: str,
        version: int = 1,
    ) -> list[BeliefState]:
        """Convert current state to BeliefState records.

        Detaches from autograd and converts to Python floats.
        """
        if len(variable_names) != self._state_dim:
            raise ValueError(
                f"variable_names length ({len(variable_names)}) != "
                f"state_dim ({self._state_dim})"
            )

        computed_at = time.time()
        x = self._x.detach()
        P = self._P.detach()

        beliefs = []
        for i, name in enumerate(variable_names):
            beliefs.append(
                BeliefState(
                    variable_name=name,
                    version=version,
                    effective_at=as_of,
                    computed_at=computed_at,
                    dist_type="gaussian",
                    mean=float(x[i].item()),
                    variance=float(P[i, i].item()),
                    evidence_count=0,
                    model_graph_hash=graph_hash,
                    confidence=1.0,
                    stale=False,
                )
            )
        return beliefs

    def reset(
        self,
        x0: Tensor | np.ndarray | None = None,
        P0: Tensor | np.ndarray | None = None,
    ) -> None:
        """Reinitialise state and covariance.

        Always creates fresh detached tensors to break any autograd graph
        from previous predict/update calls.
        """
        device = self._P.device

        if x0 is not None:
            if isinstance(x0, np.ndarray):
                x0 = torch.from_numpy(x0).float()
            self._x = x0.detach().clone().to(device)
        else:
            self._x = torch.zeros(self._state_dim, device=device)

        if P0 is not None:
            if isinstance(P0, np.ndarray):
                P0 = torch.from_numpy(P0).float()
            self._P = P0.detach().clone().to(device)
        else:
            self._P = torch.eye(self._state_dim, device=device)

    # ── Conversion to/from numpy ContinuousStateFilter ─────

    @classmethod
    def from_numpy_filter(
        cls,
        numpy_filter: Any,
    ) -> DifferentiableKalmanFilter:
        """Create from an existing ContinuousStateFilter.

        Imports F, Q, H, R values.  Q and R are decomposed into Cholesky
        factors via ``torch.linalg.cholesky``.

        Parameters
        ----------
        numpy_filter : ContinuousStateFilter
            Source filter with populated regime_configs, H, R.
        """
        state_dim = numpy_filter.state_dim
        obs_dim = numpy_filter.obs_dim
        regime_names = list(numpy_filter._regime_configs.keys())

        diff_filter = cls(
            state_dim=state_dim,
            obs_dim=obs_dim,
            regime_names=regime_names,
        )

        # Import per-regime F and Q
        for name, rc in numpy_filter._regime_configs.items():
            with torch.no_grad():
                diff_filter._F[name].copy_(torch.from_numpy(rc.F.astype(np.float32)))
                # Decompose Q into unconstrained Cholesky parameter
                Q_t = torch.from_numpy(rc.Q.astype(np.float32))
                Q_t = Q_t + _EPS * torch.eye(state_dim)  # ensure PSD
                L_Q = _psd_to_cholesky_param(Q_t)
                diff_filter._L_Q[name].copy_(L_Q)

        # Import shared H and R
        with torch.no_grad():
            diff_filter._H.copy_(torch.from_numpy(numpy_filter._H.astype(np.float32)))
            R_t = torch.from_numpy(numpy_filter._R.astype(np.float32))
            R_t = R_t + _EPS * torch.eye(obs_dim)
            L_R = _psd_to_cholesky_param(R_t)
            diff_filter._L_R.copy_(L_R)

        # Import state
        diff_filter.reset(numpy_filter._x, numpy_filter._P)

        return diff_filter

    def to_numpy_params(self) -> dict[str, Any]:
        """Export current parameters as numpy arrays.

        Returns dict with:
            regimes: {name: {"F": ndarray, "Q": ndarray}}
            H: ndarray
            R: ndarray
            x: ndarray
            P: ndarray
        """
        regimes = {}
        for name in self._regime_names:
            regimes[name] = {
                "F": self._F[name].detach().cpu().numpy(),
                "Q": self.Q(name).detach().cpu().numpy(),
            }

        return {
            "regimes": regimes,
            "H": self._H.detach().cpu().numpy(),
            "R": self.R().detach().cpu().numpy(),
            "x": self._x.detach().cpu().numpy(),
            "P": self._P.detach().cpu().numpy(),
        }
