"""TirraMind — Edge Decay Monitor

Detects structural deterioration of individual signal quality using BOCPD
(Adams & MacKay 2007) applied to rolling annualized Sharpe ratios.

Mathematical formulation
------------------------
Given signal *i*'s per-period return contribution r_{i,t}, compute the
rolling Sharpe:

    S_{i,t} = (mean(r_{i,[t-w,t]}) / max(std(r_{i,[t-w,t]}), ε)) * √P

where P = periods_per_year.  Apply BOCPD to {S_{i,t}}.

The decay score for signal *i* at time *t* is:

    d_i(t) = max(changepoint_probs[-k:])

i.e. the maximum posterior probability of a changepoint in the last k periods
of the Sharpe series.  When d_i(t) exceeds ``EdgeDecayConfig.decay_threshold``,
an ``AdversarialFlag`` with ``flag_type="edge_decay"`` is emitted.

Trusted source : Adams & MacKay (2007) "Bayesian Online Changepoint Detection"
Sharpe ratio   : Sharpe (1966), annualized as in ``agent.quant.scoring``
"""

from __future__ import annotations

import numpy as np

from agent.adversarial.config import EdgeDecayConfig
from agent.adversarial.flags import AdversarialFlag
from agent.quant.changepoint import BOCPD


class EdgeDecayMonitor:
    """Monitor per-signal Sharpe for structural breaks via BOCPD."""

    def __init__(self, config: EdgeDecayConfig | None = None) -> None:
        self._cfg = config or EdgeDecayConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        signal_name: str,
        returns: np.ndarray,
        *,
        timestamp: float | None = None,
    ) -> list[AdversarialFlag]:
        """Evaluate a signal's return series for edge decay.

        Parameters
        ----------
        signal_name : identifier of the signal being monitored.
        returns : 1-D array of per-period returns (e.g. weekly log returns).
        timestamp : optional flag timestamp; defaults to ``None``.

        Returns
        -------
        Empty list if the signal is healthy, or a single-element list
        containing an ``AdversarialFlag`` if decay is detected.
        """
        returns = np.asarray(returns, dtype=np.float64).ravel()

        if len(returns) < self._cfg.min_history:
            return []

        sharpe_series = self._rolling_sharpe(returns)
        if len(sharpe_series) < 2:
            return []

        decay_score = self._bocpd_decay_score(sharpe_series)

        if decay_score > self._cfg.decay_threshold:
            flag = AdversarialFlag(
                flag_type="edge_decay",
                severity=float(min(decay_score, 1.0)),
                confidence=float(min(decay_score, 1.0)),
                signal_name=signal_name,
                evidence={
                    "decay_score": float(decay_score),
                    "latest_sharpe": float(sharpe_series[-1]),
                    "sharpe_series_len": len(sharpe_series),
                },
                timestamp=timestamp if timestamp is not None else 0.0,
            )
            return [flag]

        return []

    def get_decay_scores(
        self,
        signal_returns: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """Batch compute decay scores for multiple signals.

        Returns {signal_name: decay_score}.  Signals with insufficient
        history are omitted.
        """
        scores: dict[str, float] = {}
        for name, rets in signal_returns.items():
            rets = np.asarray(rets, dtype=np.float64).ravel()
            if len(rets) < self._cfg.min_history:
                continue
            sharpe_series = self._rolling_sharpe(rets)
            if len(sharpe_series) < 2:
                continue
            scores[name] = self._bocpd_decay_score(sharpe_series)
        return scores

    def rolling_sharpe(self, returns: np.ndarray) -> np.ndarray:
        """Public accessor for the rolling Sharpe series (diagnostics)."""
        return self._rolling_sharpe(np.asarray(returns, dtype=np.float64).ravel())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _rolling_sharpe(self, returns: np.ndarray) -> np.ndarray:
        """Compute annualized rolling Sharpe with ε-floor on volatility."""
        w = self._cfg.rolling_window
        if len(returns) < w:
            return np.array([], dtype=np.float64)

        n = len(returns) - w + 1
        sharpes = np.empty(n, dtype=np.float64)
        eps = 1e-8
        sqrt_p = np.sqrt(self._cfg.periods_per_year)

        for i in range(n):
            window = returns[i : i + w]
            mu = window.mean()
            sigma = max(window.std(), eps)
            sharpes[i] = (mu / sigma) * sqrt_p

        return sharpes

    def _bocpd_decay_score(self, sharpe_series: np.ndarray) -> float:
        """Run BOCPD and compute decay score.

        Strategy: find the last changepoint via BOCPD.  If one exists,
        compare the mean Sharpe before and after it.  Decay score =
        clipped fractional drop in Sharpe.

        If no changepoint is found, fall back to the max recent
        changepoint probability.
        """
        bocpd = BOCPD(hazard_lambda=self._cfg.bocpd_hazard_lambda)
        result = bocpd.fit(sharpe_series)

        # Try to detect changepoints via expected run-length drops
        cps = result.changepoints(min_drop_frac=0.5, min_prev_rl=10)

        if cps:
            last_cp = cps[-1]
            pre = sharpe_series[:last_cp]
            post = sharpe_series[last_cp:]
            if len(pre) >= 2 and len(post) >= 2:
                pre_mean = float(np.mean(pre))
                post_mean = float(np.mean(post))
                # Decay = fractional drop (only positive when post < pre)
                denom = max(abs(pre_mean), 1e-8)
                drop = (pre_mean - post_mean) / denom
                return float(np.clip(drop, 0.0, 1.0))

        # Fallback: max changepoint probability in the entire series
        return float(np.max(result.changepoint_probs))
