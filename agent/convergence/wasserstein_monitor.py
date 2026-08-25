"""
TirraMind — Wasserstein Distribution Shift Detector (Idea 8)

Detects distributional drift in each data feed using the Wasserstein-1
(Earth Mover's Distance) between a short rolling window and a long baseline.

Problem
-------
TirraMind ingests 51+ data sources with wildly different update frequencies
and value distributions.  Three failure modes are currently invisible:

  1. **API / pipeline failures** — a feed quietly stops updating or starts
     returning clipped/constant values.  Stale-timestamp flags fire only
     after a hard cutoff; gradual degradation is missed entirely.

  2. **Regime changes** — commodity markets flip from contango to
     backwardation; AIS traffic patterns shift after a geopolitical event.
     The GNN keeps running on stale priors until the next retrain.

  3. **Prediction drift** — the model's input distribution drifts away from
     training distribution.  Predictions degrade silently.

All three manifest as a distributional shift in the daily observation pattern
of the affected tool.

Solution — Wasserstein-1 on Daily Activity Distributions
----------------------------------------------------------
For each source_tool with sufficient history:

    baseline_365 = daily observation counts over the past 365 days
    window_30    = daily observation counts over the past 30 days

    W1_count   = Wasserstein-1 distance(window_30, baseline_365)  # count channel
    W1_value   = Wasserstein-1 distance(window_30, baseline_365)  # mean-value channel
    drift_score = (W1_count + W1_value) / 2

Wasserstein-1 for 1D empirical distributions is computed **analytically**:

    W1(P, Q) = ∫|F_P(x) − F_Q(x)| dx  ≈  mean |sort(P) − interp(sort(Q), len(P))|

This requires **no external dependencies** — just numpy.  No POT is needed
for 1D; POT is optionally used for higher-dimensional Sinkhorn if available.

If ``use_sinkhorn=True`` (requires ``pip install POT``), a richer 4-feature
Sinkhorn distance replaces the 1D W1 score.  The 1D fallback is always
available and is the default.

Output
------
``WassersteinMonitor.run(store)`` returns
``dict[tool_name, WassersteinResult]``.  Results are:

  - Stored as ``wasserstein.<tool_name>.drift`` signals in the pipeline store
    (for ConvergenceDetector to see as cross-category evidence).
  - Logged as warnings when ``is_alarm=True``.
  - Available for inline inspection in ``Trainer.build_model()`` when
    ``use_wasserstein=True``.

References
----------
    Villani, C. (2008). Optimal Transport: Old and New.
        Chapter 6 — The Kantorovich problem; W1 duality theorem.
    Flamary, R. et al. (2021). POT: Python Optimal Transport.
        JMLR 22(78):1−8.  https://pythonot.github.io/
    PythonOT/POT (MIT) — optional; 2.5k⭐.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Value extraction priority (mirrors ts2vec_encoder.py and signature_encoder.py)
_VALUE_KEYS = (
    "close",
    "usd_amount",
    "value",
    "estimated_value",
    "goldstein_scale",
    "btc_amount",
    "log_return",
    "num_articles",
)

_SECS_PER_DAY: float = 86_400.0


# ═══════════════════════════════════════════════════════════════════════════
# Result dataclass
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class WassersteinResult:
    """Distribution shift result for a single source tool.

    Attributes
    ----------
    tool_name : str
        Source tool identifier (matches entity_observations.source_tool).
    drift_score : float
        Combined Wasserstein-1 drift score.  Larger = more drift.
        Computed as average of count-channel and value-channel W1 distances,
        each normalised by the baseline standard deviation so the threshold
        is dimensionless.
    is_alarm : bool
        True when drift_score exceeds the configured threshold.
    w1_count : float
        W1 distance on the daily observation count distribution.
    w1_value : float
        W1 distance on the daily mean-value distribution.
    short_count : int
        Total observations in the short (30-day) window.
    baseline_count : int
        Total observations in the long (365-day) baseline window.
    computed_at : float
        Unix timestamp of computation.
    threshold : float
        Alarm threshold used for this result.
    """

    tool_name: str
    drift_score: float
    is_alarm: bool
    w1_count: float
    w1_value: float
    short_count: int
    baseline_count: int
    computed_at: float
    threshold: float
    details: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# WassersteinMonitor
# ═══════════════════════════════════════════════════════════════════════════


class WassersteinMonitor:
    """Monitor distributional drift in each data feed.

    Parameters
    ----------
    short_days : int
        Length of the short rolling window in days.  Default 30.
    long_days : int
        Length of the long baseline window in days.  Default 365.
    alarm_threshold : float
        Normalised W1 distance above which ``is_alarm=True``.
        The score is normalised by the baseline standard deviation, so
        threshold=1.0 means one-standard-deviation shift.  Default 1.0.
    min_baseline_obs : int
        Minimum observations in the long baseline to attempt scoring.
        Tools with fewer observations are skipped.  Default 10.
    use_sinkhorn : bool
        If True and POT is installed, use 4-feature Sinkhorn distance
        instead of 1D W1.  Falls back to 1D W1 if POT is unavailable.
        Default False (always use the dependency-free path).
    """

    def __init__(
        self,
        short_days: int = 30,
        long_days: int = 365,
        alarm_threshold: float = 1.0,
        min_baseline_obs: int = 10,
        use_sinkhorn: bool = False,
    ) -> None:
        self.short_days = short_days
        self.long_days = long_days
        self.alarm_threshold = alarm_threshold
        self.min_baseline_obs = min_baseline_obs
        self.use_sinkhorn = use_sinkhorn

    # ── Public API ─────────────────────────────────────────────────────────

    def run(
        self,
        store: Any,
        as_of: float | None = None,
    ) -> dict[str, WassersteinResult]:
        """Run distribution shift monitoring for all source tools.

        Args:
            store: PipelineStore instance.
            as_of: Reference time (unix epoch).  Defaults to ``time.time()``.

        Returns:
            Dict mapping tool_name → WassersteinResult.
            Only tools with sufficient history are included.
        """
        if as_of is None:
            as_of = time.time()

        t_short_start = as_of - self.short_days * _SECS_PER_DAY
        t_long_start = as_of - self.long_days * _SECS_PER_DAY

        # Load observations from the long baseline window
        all_obs = store.query_all_observations(since=t_long_start, until=as_of)

        if not all_obs:
            log.info("WassersteinMonitor: no observations in baseline window.")
            return {}

        # Group by source_tool
        by_tool: dict[str, list[dict[str, Any]]] = {}
        for obs in all_obs:
            tool = obs.get("source_tool", "unknown")
            by_tool.setdefault(tool, []).append(obs)

        results: dict[str, WassersteinResult] = {}

        for tool_name, obs_list in sorted(by_tool.items()):
            if len(obs_list) < self.min_baseline_obs:
                log.debug(
                    "WassersteinMonitor: skipping %r (%d obs < min=%d)",
                    tool_name, len(obs_list), self.min_baseline_obs,
                )
                continue

            result = self._compute_for_tool(
                tool_name, obs_list, t_short_start, t_long_start, as_of
            )
            if result is not None:
                results[tool_name] = result
                if result.is_alarm:
                    log.warning(
                        "WassersteinMonitor: ALARM for %r — drift_score=%.3f "
                        "(threshold=%.3f, short=%d obs, baseline=%d obs)",
                        tool_name, result.drift_score, result.threshold,
                        result.short_count, result.baseline_count,
                    )
                else:
                    log.debug(
                        "WassersteinMonitor: %r drift_score=%.3f (ok)",
                        tool_name, result.drift_score,
                    )

        log.info(
            "WassersteinMonitor: scored %d tools, %d alarm(s)",
            len(results),
            sum(1 for r in results.values() if r.is_alarm),
        )
        return results

    def store_results(
        self,
        results: dict[str, WassersteinResult],
        store: Any,
    ) -> int:
        """Persist drift scores as pipeline signals.

        Signal names follow the convention:
            ``wasserstein.<tool_name>.drift``

        The ConvergenceDetector can register these signals in its
        SignalRegistry under a dedicated ``"data_quality"`` category
        to detect cross-feed regime changes.

        Returns:
            Number of signals successfully stored.
        """
        count = 0
        for tool_name, result in results.items():
            signal_name = f"wasserstein.{tool_name}.drift"
            try:
                store.store_signal(
                    signal_name=signal_name,
                    value=result.drift_score,
                    metadata={
                        "is_alarm": result.is_alarm,
                        "w1_count": result.w1_count,
                        "w1_value": result.w1_value,
                        "short_count": result.short_count,
                        "baseline_count": result.baseline_count,
                        "threshold": result.threshold,
                        "computed_at": result.computed_at,
                        "short_days": self.short_days,
                        "long_days": self.long_days,
                    },
                )
                count += 1
            except Exception:
                log.warning(
                    "WassersteinMonitor: failed to store signal for %r",
                    tool_name, exc_info=True,
                )
        log.info("WassersteinMonitor: stored %d drift signals.", count)
        return count

    # ── Internal: per-tool scoring ─────────────────────────────────────────

    def _compute_for_tool(
        self,
        tool_name: str,
        obs_list: list[dict[str, Any]],
        t_short_start: float,
        t_long_start: float,
        as_of: float,
    ) -> WassersteinResult | None:
        """Compute W1 drift score for one tool."""
        now = as_of

        # Split into windows
        short_obs = [o for o in obs_list if o.get("observed_at", 0.0) >= t_short_start]
        long_obs = obs_list  # already filtered to [t_long_start, as_of]

        if len(long_obs) < self.min_baseline_obs:
            return None

        # Build daily feature arrays
        long_count, long_value = self._daily_features(
            long_obs, t_long_start, now, n_bins=self.long_days
        )
        short_count, short_value = self._daily_features(
            short_obs, t_short_start, now, n_bins=self.short_days
        )

        if self.use_sinkhorn:
            drift_score, w1_count, w1_value = self._sinkhorn_score(
                short_count, short_value, long_count, long_value
            )
        else:
            w1_count = _w1_1d_normalised(short_count, long_count)
            w1_value = _w1_1d_normalised(short_value, long_value)
            drift_score = (w1_count + w1_value) / 2.0

        return WassersteinResult(
            tool_name=tool_name,
            drift_score=float(drift_score),
            is_alarm=drift_score > self.alarm_threshold,
            w1_count=float(w1_count),
            w1_value=float(w1_value),
            short_count=len(short_obs),
            baseline_count=len(long_obs),
            computed_at=now,
            threshold=self.alarm_threshold,
        )

    # ── Internal: feature extraction ──────────────────────────────────────

    @staticmethod
    def _daily_features(
        obs: list[dict[str, Any]],
        t_start: float,
        t_end: float,
        n_bins: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build daily count and mean-value arrays over [t_start, t_end].

        Returns:
            count_bins  : ndarray shape (n_bins,) — obs per day, float
            value_bins  : ndarray shape (n_bins,) — tanh mean value per day
        """
        span = max(t_end - t_start, 1.0)
        bin_dur = span / n_bins

        count_arr = np.zeros(n_bins, dtype=np.float64)
        value_sums = np.zeros(n_bins, dtype=np.float64)
        value_counts = np.zeros(n_bins, dtype=np.float64)

        for o in obs:
            t = float(o.get("observed_at", t_start))
            b = min(int((t - t_start) / bin_dur), n_bins - 1)
            count_arr[b] += 1.0

            v = o.get("value", {})
            if isinstance(v, dict):
                for k in _VALUE_KEYS:
                    if k in v:
                        try:
                            val = float(v[k])
                            if math.isfinite(val):
                                value_sums[b] += val
                                value_counts[b] += 1.0
                        except (TypeError, ValueError):
                            pass
                        break

        with np.errstate(invalid="ignore", divide="ignore"):
            mean_values = np.where(
                value_counts > 0, value_sums / value_counts, 0.0
            )
        value_arr = np.tanh(mean_values / (np.abs(mean_values) + 1.0))

        return count_arr, value_arr

    # ── Internal: Sinkhorn (optional, requires POT) ────────────────────────

    @staticmethod
    def _sinkhorn_score(
        sc: np.ndarray,
        sv: np.ndarray,
        lc: np.ndarray,
        lv: np.ndarray,
    ) -> tuple[float, float, float]:
        """4-feature Sinkhorn distance using POT.

        Falls back to 1D W1 if POT is not installed.
        """
        try:
            import ot  # noqa: PLC0415

            X_short = np.stack([sc, sv], axis=1).astype(np.float64)
            X_long = np.stack([lc, lv], axis=1).astype(np.float64)

            n_s, n_l = len(X_short), len(X_long)
            a = np.ones(n_s) / n_s
            b = np.ones(n_l) / n_l
            M = ot.dist(X_short, X_long, metric="euclidean")
            score = float(ot.sinkhorn2(a, b, M, reg=0.05)[0])

            # Also compute 1D channels for reporting
            w1_count = _w1_1d_normalised(sc, lc)
            w1_value = _w1_1d_normalised(sv, lv)
            return score, w1_count, w1_value

        except ImportError:
            log.debug(
                "POT not available — falling back to 1D W1. "
                "Install with: pip install POT"
            )
        except Exception as exc:
            log.warning("Sinkhorn computation failed: %s — falling back to W1", exc)

        w1_count = _w1_1d_normalised(sc, lc)
        w1_value = _w1_1d_normalised(sv, lv)
        return (w1_count + w1_value) / 2.0, w1_count, w1_value


# ═══════════════════════════════════════════════════════════════════════════
# Pure-numpy Wasserstein-1 (no external deps)
# ═══════════════════════════════════════════════════════════════════════════


def _w1_1d_normalised(
    short: np.ndarray,
    long: np.ndarray,
) -> float:
    """Normalised Wasserstein-1 distance between two 1D empirical distributions.

    Normalisation: divide by the baseline (long window) standard deviation
    so the result is dimensionless and threshold=1.0 means one-sigma shift.

    W1(P, Q) = ∫|F_P(x) − F_Q(x)| dx
             ≈ mean |sort(P) − interp_to_len_P(sort(Q))|

    Both arrays are sorted and the shorter is linearly interpolated to
    match the length of the longer before computing the L1 difference.
    This is the exact Wasserstein-1 for 1D distributions evaluated on a
    shared support grid (Villani 2008, Proposition 2.17).
    """
    a = np.asarray(short, dtype=np.float64).ravel()
    b = np.asarray(long, dtype=np.float64).ravel()

    if len(a) == 0 or len(b) == 0:
        return 0.0

    sa = np.sort(a)
    sb = np.sort(b)

    # Interpolate the longer array to the length of the shorter
    if len(sa) != len(sb):
        n = max(len(sa), len(sb))
        xs = np.linspace(0.0, 1.0, len(sa))
        xb = np.linspace(0.0, 1.0, len(sb))
        grid = np.linspace(0.0, 1.0, n)
        sa = np.interp(grid, xs, sa)
        sb = np.interp(grid, xb, sb)

    raw_w1 = float(np.mean(np.abs(sa - sb)))

    # Normalise by baseline std (long window)
    baseline_std = float(np.std(b, ddof=1)) if len(b) > 1 else 1.0
    if baseline_std < 1e-10:
        # Baseline is constant — any non-zero difference is a large shift
        return raw_w1 * 10.0 if raw_w1 > 1e-10 else 0.0

    return raw_w1 / baseline_std
