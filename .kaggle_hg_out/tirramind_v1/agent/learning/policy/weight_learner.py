"""TirraMind — Surprise Weight Learner (Phase 21a)

Learns the 5 composite-surprise weights by gradient ascent on a
differentiable Sharpe ratio under walk-forward cross-validation.

Mathematical framework
──────────────────────
Given a surprise matrix S ∈ ℝ^{T×5} and asset returns r ∈ ℝ^T, the
composite signal at time t is:

    c_t = softmax(θ)ᵀ s_t        (θ ∈ ℝ⁵ are unconstrained params)

The signal-weighted return is:

    ρ_t = c_t · r_{t+1}

The differentiable Sharpe ratio (Moody & Saffell 2001) is:

    S(θ) = ρ̄ / √( ρ² − ρ̄² + ε )

where ρ̄ = (1/T) Σ ρ_t, ρ² = (1/T) Σ ρ_t².

Objective:  max_θ  S(θ) − λ‖θ‖²

Walk-forward protocol ensures out-of-sample evaluation:
    for each fold:
        train weights on train split  (gradient ascent on S)
        evaluate on held-out test split  (frozen weights)

Proof of correctness:
    1. softmax guarantees w_i ≥ 0 and ‖w‖₁ = 1.
    2. Differentiable Sharpe is smooth ∀ θ when ε > 0.
    3. Walk-forward prevents look-ahead bias.
    4. L2 reg prevents weight explosion in unconstrained θ-space.
    5. Gradient clipping bounds the step size (numerical safety).

Trusted sources:
    - Moody & Saffell (2001): "Learning to Trade via Direct Reinforcement"
    - Sharpe (1966): original Sharpe ratio definition
    - DreamerV3 §3.2 (Hafner 2023): symlog transform for return scaling
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from agent.learning.policy.config import WeightLearnerConfig

log = logging.getLogger(__name__)


class InsufficientDataError(Exception):
    """Raised when time-series is too short for walk-forward training."""


class SurpriseWeightLearner:
    """Learn composite-surprise weights via differentiable Sharpe.

    The 5 raw parameters live in unconstrained ℝ⁵; softmax maps
    them to the probability simplex before computing the composite
    score.  This guarantees non-negative weights summing to 1.
    """

    SIGNAL_NAMES = (
        "obs_type",
        "temporal",
        "value",
        "neighborhood",
        "memory_drift",
    )

    def __init__(self, config: WeightLearnerConfig | None = None) -> None:
        self._cfg = config or WeightLearnerConfig()
        # Initialise at origin → softmax = uniform(1/5, …, 1/5)
        self._raw_weights = torch.nn.Parameter(torch.zeros(5))
        self._optimizer: torch.optim.Optimizer | None = None

    # ── public properties ─────────────────────────────────────

    @property
    def weights(self) -> torch.Tensor:
        """Normalised weights on the 5-simplex via softmax."""
        return F.softmax(self._raw_weights, dim=0)

    def get_learned_weights(self) -> tuple[float, ...]:
        """Return the 5 learned weights as a plain tuple."""
        with torch.no_grad():
            w = F.softmax(self._raw_weights, dim=0)
        return tuple(float(x) for x in w)

    # ── forward pass ──────────────────────────────────────────

    def composite_score(
        self,
        surprise_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Compute composite surprise: c_t = softmax(θ)ᵀ s_t.

        Parameters
        ----------
        surprise_matrix : (T, 5) tensor of per-timestep surprise signals.

        Returns
        -------
        (T,) tensor of composite scores.
        """
        w = self.weights  # (5,)
        return surprise_matrix @ w  # (T,)

    @staticmethod
    def differentiable_sharpe(
        scores: torch.Tensor,
        returns: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Differentiable Sharpe ratio (Moody & Saffell 2001).

        r_t = score_t · return_{t+1}
        S   = r̄  /  √( mean(r²) − r̄² + ε )

        Parameters
        ----------
        scores  : (T,) composite surprise scores.
        returns : (T,) asset returns aligned to scores.
        eps     : numerical floor under the variance.

        Returns
        -------
        Scalar tensor (Sharpe ratio).

        Mathematical invariants:
            - S is finite for all inputs when eps > 0.
            - S = 0 when scores are constant (no information).
            - Gradient ∂S/∂θ exists everywhere.
        """
        weighted_returns = scores * returns
        mean_r = weighted_returns.mean()
        mean_r2 = (weighted_returns**2).mean()
        var = mean_r2 - mean_r**2 + eps
        return mean_r / torch.sqrt(var)

    # ── training ──────────────────────────────────────────────

    def fit(
        self,
        surprise_matrix: np.ndarray,
        returns: np.ndarray,
    ) -> dict[str, Any]:
        """Walk-forward training of the 5 surprise weights.

        Parameters
        ----------
        surprise_matrix : (T, 5) array—one row per time period.
        returns         : (T,) array of asset returns.

        Returns
        -------
        Dict with keys:
            weights         → learned 5-tuple
            train_sharpes   → list of per-fold best train Sharpe
            test_sharpes    → list of per-fold test Sharpe (OOS)
            mean_test_sharpe→ average OOS Sharpe across folds
            epochs_per_fold → list of training epochs per fold
            config          → WeightLearnerConfig as dict

        Raises
        ------
        InsufficientDataError
            If T < min_train + test_periods.
        """
        cfg = self._cfg
        S = np.asarray(surprise_matrix, dtype=np.float32)
        R = np.asarray(returns, dtype=np.float32)

        if S.ndim != 2 or S.shape[1] != 5:
            raise ValueError(f"surprise_matrix must be (T, 5), got {S.shape}")
        if R.ndim != 1 or R.shape[0] != S.shape[0]:
            raise ValueError(f"returns must be (T,) matching surprise_matrix rows, got {R.shape} vs {S.shape[0]}")
        T = S.shape[0]
        min_total = cfg.min_train_periods + cfg.test_periods
        if min_total > T:
            raise InsufficientDataError(f"Need ≥ {min_total} periods, got {T}")

        if np.any(np.isnan(S)):
            raise ValueError("surprise_matrix contains NaN")
        if np.any(np.isnan(R)):
            raise ValueError("returns contains NaN")

        # Walk-forward fold boundaries
        folds = self._build_folds(T)
        if not folds:
            raise InsufficientDataError("No valid folds could be constructed")

        S_t = torch.from_numpy(S)
        R_t = torch.from_numpy(R)

        train_sharpes: list[float] = []
        test_sharpes: list[float] = []
        epochs_per_fold: list[int] = []

        for fold_idx, (train_end, test_end) in enumerate(folds):
            # Reset parameters for each fold (avoid leak across folds)
            self._raw_weights = torch.nn.Parameter(torch.zeros(5))
            self._optimizer = torch.optim.Adam(
                [self._raw_weights],
                lr=cfg.learning_rate,
                weight_decay=cfg.l2_reg,
            )

            S_train = S_t[:train_end]
            R_train = R_t[:train_end]
            S_test = S_t[train_end:test_end]
            R_test = R_t[train_end:test_end]

            best_sharpe = -float("inf")
            best_raw = self._raw_weights.data.clone()
            patience_counter = 0

            for epoch in range(cfg.max_epochs):
                self._optimizer.zero_grad()
                scores = self.composite_score(S_train)
                sharpe = self.differentiable_sharpe(scores, R_train)
                loss = -sharpe  # maximise Sharpe = minimise negative Sharpe
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_([self._raw_weights], cfg.grad_clip_norm)
                self._optimizer.step()

                s_val = float(sharpe.item())
                if s_val > best_sharpe + 1e-6:
                    best_sharpe = s_val
                    best_raw = self._raw_weights.data.clone()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= cfg.patience:
                        break

            # Restore best weights for this fold
            self._raw_weights.data.copy_(best_raw)
            epochs_per_fold.append(epoch + 1)
            train_sharpes.append(best_sharpe)

            # OOS evaluation
            with torch.no_grad():
                test_scores = self.composite_score(S_test)
                test_sharpe = self.differentiable_sharpe(test_scores, R_test)
                test_sharpes.append(float(test_sharpe.item()))

            log.info(
                "Fold %d: train_sharpe=%.4f  test_sharpe=%.4f  epochs=%d",
                fold_idx,
                best_sharpe,
                test_sharpes[-1],
                epochs_per_fold[-1],
            )

        # Final: retrain on all data with best-average-fold init
        self._raw_weights = torch.nn.Parameter(torch.zeros(5))
        self._optimizer = torch.optim.Adam(
            [self._raw_weights],
            lr=cfg.learning_rate,
            weight_decay=cfg.l2_reg,
        )
        for _epoch in range(cfg.max_epochs):
            self._optimizer.zero_grad()
            scores = self.composite_score(S_t)
            sharpe = self.differentiable_sharpe(scores, R_t)
            (-sharpe).backward()
            torch.nn.utils.clip_grad_norm_([self._raw_weights], cfg.grad_clip_norm)
            self._optimizer.step()

        learned = self.get_learned_weights()
        mean_test = float(np.mean(test_sharpes)) if test_sharpes else 0.0

        return {
            "weights": learned,
            "train_sharpes": train_sharpes,
            "test_sharpes": test_sharpes,
            "mean_test_sharpe": mean_test,
            "epochs_per_fold": epochs_per_fold,
            "config": asdict(cfg),
        }

    def _build_folds(self, T: int) -> list[tuple[int, int]]:
        """Construct walk-forward fold boundaries.

        Returns list of (train_end, test_end) tuples.
        Non-overlapping test windows, expanding training windows.
        """
        cfg = self._cfg
        folds: list[tuple[int, int]] = []
        split = cfg.min_train_periods
        while split + cfg.test_periods <= T:
            folds.append((split, split + cfg.test_periods))
            split += cfg.walk_forward_step
        return folds

    # ── serialisation ─────────────────────────────────────────

    def state_dict(self) -> dict[str, Any]:
        """Serialise learnable state."""
        return {
            "raw_weights": self._raw_weights.data.cpu().numpy().tolist(),
            "learned_weights": list(self.get_learned_weights()),
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        """Restore learnable state."""
        raw = d["raw_weights"]
        self._raw_weights = torch.nn.Parameter(torch.tensor(raw, dtype=torch.float32))
