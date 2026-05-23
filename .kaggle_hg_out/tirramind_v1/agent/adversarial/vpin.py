"""TirraMind — Volume-synchronized PIN (VPIN) Estimator

Estimates the probability of informed trading at daily resolution using
Bulk Volume Classification (BVC).

Mathematical formulation
------------------------
Easley, López de Prado, O'Hara (2012) "Flow Toxicity and Liquidity in a
High-Frequency World".

Given daily data (r_t, V_t) and rolling volatility σ_t:

    V_buy,t  = V_t · Φ(r_t / σ_t)
    V_sell,t = V_t − V_buy,t
    OI_t     = |V_sell,t − V_buy,t|

    VPIN_t = (1/n) Σ_{τ=t-n+1}^{t}  OI_τ / V̄

where V̄ is the mean daily volume over the rolling window and Φ is the
standard normal CDF.

Properties:
    - VPIN ∈ [0, ∞) in theory, but ≈ [0, 1] for well-behaved data.
    - Symmetric flow (buy ≈ sell) → VPIN low.
    - One-sided flow → VPIN high.
    - When r_t / σ_t → +∞, Φ → 1, V_buy = V_t, V_sell = 0.

Trusted source: Easley, López de Prado & O'Hara (2012), JFM.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from agent.adversarial.config import VPINConfig
from agent.adversarial.flags import AdversarialFlag


class VPINEstimator:
    """Daily-resolution VPIN via Bulk Volume Classification."""

    def __init__(self, config: VPINConfig | None = None) -> None:
        self._cfg = config or VPINConfig()

    def compute(
        self,
        returns: np.ndarray,
        volumes: np.ndarray,
    ) -> np.ndarray:
        """Compute VPIN time series from daily returns and volumes.

        Parameters
        ----------
        returns : 1-D array of daily log returns.
        volumes : 1-D array of daily trading volumes (same length).

        Returns
        -------
        1-D array of VPIN values, length = len(returns) - n + 1
        where n = ``VPINConfig.n_buckets``.

        Raises
        ------
        ValueError
            If inputs differ in length, contain NaN, or volumes are
            all zero.
        """
        returns = np.asarray(returns, dtype=np.float64).ravel()
        volumes = np.asarray(volumes, dtype=np.float64).ravel()

        if len(returns) != len(volumes):
            raise ValueError(f"returns and volumes must have equal length, got {len(returns)} and {len(volumes)}")
        if np.any(np.isnan(returns)) or np.any(np.isnan(volumes)):
            raise ValueError("NaN values in returns or volumes")
        if np.all(volumes == 0):
            raise ValueError("All volumes are zero — cannot compute VPIN")

        # Rolling volatility for BVC denominator
        sigma = self._rolling_std(returns, self._cfg.sigma_window)

        # BVC: classify buy/sell volume
        eps = 1e-10
        z = returns / np.maximum(sigma, eps)
        buy_frac = norm.cdf(z)
        v_buy = volumes * buy_frac
        v_sell = volumes * (1.0 - buy_frac)
        oi = np.abs(v_sell - v_buy)  # order imbalance

        # Sliding-window VPIN
        n = self._cfg.n_buckets
        if len(oi) < n:
            return np.array([], dtype=np.float64)

        # Use cumsum for efficient rolling mean of OI and V
        cum_oi = np.concatenate([[0.0], np.cumsum(oi)])
        cum_v = np.concatenate([[0.0], np.cumsum(volumes)])
        window_oi = cum_oi[n:] - cum_oi[:-n]
        window_v = cum_v[n:] - cum_v[:-n]

        # VPIN = mean(OI) / mean(V) = sum(OI) / sum(V) over window
        vpin = window_oi / np.maximum(window_v, eps)

        return vpin

    def flag_spikes(
        self,
        vpin_series: np.ndarray,
        *,
        entity_id: str | None = None,
        timestamp: float | None = None,
    ) -> list[AdversarialFlag]:
        """Return flags for VPIN values exceeding the spike threshold.

        Only the last (most recent) value is checked.
        """
        vpin_series = np.asarray(vpin_series, dtype=np.float64).ravel()
        if len(vpin_series) == 0:
            return []

        latest = float(vpin_series[-1])
        if latest > self._cfg.spike_threshold:
            severity = float(min(latest, 1.0))
            return [
                AdversarialFlag(
                    flag_type="vpin_spike",
                    severity=severity,
                    confidence=severity,
                    entity_id=entity_id,
                    evidence={
                        "vpin_latest": latest,
                        "threshold": self._cfg.spike_threshold,
                        "series_len": len(vpin_series),
                    },
                    timestamp=timestamp if timestamp is not None else 0.0,
                )
            ]
        return []

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
        """Rolling standard deviation with minimum-period fallback.

        For positions before the full window, uses all available data.
        Minimum std is floored at 1e-10 to avoid div-by-zero in BVC.
        """
        out = np.empty_like(arr)
        for i in range(len(arr)):
            start = max(0, i - window + 1)
            out[i] = max(arr[start : i + 1].std(), 1e-10)
        return out
