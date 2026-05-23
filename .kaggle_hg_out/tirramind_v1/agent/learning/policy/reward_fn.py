"""TirraMind — RL Reward Function

Stateless reward computation with three components:

Extrinsic (P&L-based):
    r_ext = portfolio_return / max(σ_W, ε)  −  κ · max(0, −CVaR₀.₀₅)

    First term: Sharpe-normalised return (scale-invariant via rolling vol).
    Second term: CVaR penalty activates only on tail loss, encouraging
    the policy to avoid catastrophic drawdowns.

Intrinsic (surprise-based, per ICM — Pathak 2017):
    r_int = mean(composite_surprise_i)  for alerted entities

    Decays linearly: λ(t) = λ₀ · (1 − t/T), shifting the policy from
    exploration to exploitation over training.

Adversarial penalty (Phase 22):
    r_adv = −β · Σ_f  sev_f · conf_f

    Penalises holding positions flagged by the adversarial layer
    (edge decay, VPIN spikes, crowding risk).

Combined:
    r = r_ext + λ(t) · r_int − β · Σ sev · conf

Trusted sources:
    - Sharpe ratio normalisation: Sharpe (1966)
    - CVaR / Expected Shortfall: Acerbi & Tasche (2002)
    - ICM intrinsic motivation: Pathak et al. (2017) arXiv:1705.05363
    - Symlog for stability: Hafner et al. (2023) DreamerV3
    - Adversarial penalty: spec step 22b.2
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from agent.learning.policy.config import RewardConfig

if TYPE_CHECKING:
    from agent.adversarial.flags import AdversarialFlag


class RewardFunction:
    """Compute extrinsic + intrinsic reward for the RL policy."""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self._cfg = config or RewardConfig()

    def extrinsic(
        self,
        portfolio_return: float,
        rolling_returns: np.ndarray,
    ) -> float:
        """Sharpe-normalised return with CVaR penalty.

        Parameters
        ----------
        portfolio_return : single-period portfolio return.
        rolling_returns : recent return history for vol/CVaR estimation.
                          Must have ≥ 1 element.

        Returns
        -------
        Scalar reward.

        Mathematical guarantee:
            - When vol → 0, the vol_floor ε prevents ±∞.
            - CVaR penalty ≥ 0 (only penalises tail loss).
        """
        rolling = np.asarray(rolling_returns, dtype=np.float64)
        if len(rolling) == 0:
            return 0.0

        # Rolling volatility
        vol = float(np.std(rolling))
        vol = max(vol, self._cfg.vol_floor)

        # Sharpe-normalised return
        sharpe_norm = portfolio_return / vol

        # CVaR penalty  (CVaR is mean of worst (1-conf) tail, a negative number for losses)
        if len(rolling) >= 2:
            alpha = 1.0 - self._cfg.cvar_confidence
            cutoff = int(max(1, np.ceil(len(rolling) * alpha)))
            sorted_ret = np.sort(rolling)[:cutoff]
            cvar_val = float(sorted_ret.mean())
        else:
            cvar_val = float(rolling[0])

        # Penalty only activates when CVaR is negative (tail loss)
        cvar_pen = self._cfg.cvar_penalty * max(0.0, -cvar_val)

        return sharpe_norm - cvar_pen

    def intrinsic(
        self,
        surprise_scores: np.ndarray,
    ) -> float:
        """Mean composite surprise of alerted entities.

        Returns 0.0 if no alerts (empty array).
        """
        scores = np.asarray(surprise_scores, dtype=np.float64)
        if len(scores) == 0:
            return 0.0
        return float(np.mean(scores))

    def adversarial_penalty(
        self,
        adversarial_flags: list[AdversarialFlag] | None = None,
    ) -> float:
        """Compute adversarial penalty from active flags.

        Penalty = β · Σ_f (severity_f × confidence_f)

        Returns 0.0 if no flags are provided (backward compatible).
        """
        if not adversarial_flags:
            return 0.0
        total = sum(f.severity * f.confidence for f in adversarial_flags)
        return self._cfg.adversarial_penalty * total

    def combined(
        self,
        portfolio_return: float,
        rolling_returns: np.ndarray,
        surprise_scores: np.ndarray,
        step: int,
        total_steps: int,
        adversarial_flags: list[AdversarialFlag] | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Compute total reward with diagnostics.

        Parameters
        ----------
        step : current training step (for intrinsic decay).
        total_steps : total expected training steps.

        Returns
        -------
        (total_reward, breakdown) where breakdown has keys:
            extrinsic, intrinsic, lambda_t, raw_return, vol, cvar
        """
        rolling = np.asarray(rolling_returns, dtype=np.float64)

        ext = self.extrinsic(portfolio_return, rolling)
        intr = self.intrinsic(surprise_scores)

        # Decay schedule: λ(t) = λ₀ · (1 − t/T)
        if self._cfg.intrinsic_decay and total_steps > 0:
            lambda_t = self._cfg.intrinsic_weight_initial * (1.0 - step / total_steps)
        else:
            lambda_t = self._cfg.intrinsic_weight_initial

        total = ext + lambda_t * intr

        # Adversarial penalty (Phase 22)
        adv_pen = self.adversarial_penalty(adversarial_flags)
        total -= adv_pen

        # Diagnostics
        vol = max(float(np.std(rolling)) if len(rolling) > 0 else 0.0, self._cfg.vol_floor)

        if len(rolling) >= 2:
            alpha = 1.0 - self._cfg.cvar_confidence
            cutoff = int(max(1, np.ceil(len(rolling) * alpha)))
            cvar_val = float(np.sort(rolling)[:cutoff].mean())
        elif len(rolling) == 1:
            cvar_val = float(rolling[0])
        else:
            cvar_val = 0.0

        breakdown = {
            "extrinsic": ext,
            "intrinsic": intr,
            "lambda_t": lambda_t,
            "raw_return": portfolio_return,
            "vol": vol,
            "cvar": cvar_val,
            "adversarial_penalty": adv_pen,
        }
        return total, breakdown
