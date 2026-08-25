"""Canonical forward-return label construction for GNN training and IC eval.

Single owner for 21-day (configurable) forward holding-period returns from
instrument_daily close prices.  Trainer, backtest, and baseline audits must
all import from here — never duplicate label logic.

References:
    Lewellen (2015) — cross-sectional expected returns use holding-period returns.
"""

from __future__ import annotations

import bisect
import math
from typing import Any

import numpy as np

# 1 trading day ≈ 7/5 calendar days (252 trading / 365 calendar).
CALENDAR_SECS_PER_TRADING_DAY = 86400.0 * 7 / 5
DEFAULT_HORIZON_DAYS = 21
DEFAULT_TOLERANCE_SECS = 3 * 86400.0


def build_forward_return_lookup(
    observations: list[dict],
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    *,
    tolerance_secs: float = DEFAULT_TOLERANCE_SECS,
    require_future_after: float | None = None,
) -> dict[tuple[str, int], float]:
    """Precompute N-day forward simple returns from instrument_daily closes.

    forward_return = (close_{t+N} - close_t) / |close_t|

    Parameters
    ----------
    observations
        Full observation list (instrument_daily rows used).
    horizon_days
        Holding period in trading days (default 21).
    tolerance_secs
        Accept future close within ±this many seconds of target lag.
    require_future_after
        If set, drop entries whose matching future close timestamp is
        <= this value (point-in-time guard for window-end audits).

    Returns
    -------
    dict mapping (entity_id, int(observed_at)) → forward_return.
    """
    by_entity = _group_instrument_closes(observations)
    target_lag_secs = horizon_days * CALENDAR_SECS_PER_TRADING_DAY
    lookup: dict[tuple[str, int], float] = {}

    for eid, sorted_obs in by_entity.items():
        timestamps = [x[0] for x in sorted_obs]
        for i, (ts_now, close_now) in enumerate(sorted_obs):
            if abs(close_now) < 1e-8:
                continue
            target_ts = ts_now + target_lag_secs
            j = bisect.bisect_left(timestamps, target_ts)
            best_close: float | None = None
            best_future_ts: float | None = None
            best_dist = float("inf")
            for cand in (j - 1, j):
                if 0 <= cand < len(sorted_obs) and cand != i:
                    fut_ts, fut_close = sorted_obs[cand]
                    dist = abs(fut_ts - target_ts)
                    if dist < tolerance_secs and dist < best_dist:
                        best_dist = dist
                        best_close = fut_close
                        best_future_ts = fut_ts
            if best_close is None or best_future_ts is None:
                continue
            if require_future_after is not None and best_future_ts <= require_future_after:
                continue
            fwd_ret = (best_close - close_now) / abs(close_now)
            if not math.isfinite(fwd_ret):
                continue
            lookup[(eid, int(ts_now))] = fwd_ret

    return lookup


def forward_return_vector(
    lookup: dict[tuple[str, int], float],
    entity_ids: list[str],
    anchor_ts: float,
) -> np.ndarray:
    """Forward returns for *entity_ids* at *anchor_ts*; NaN when missing."""
    key = int(anchor_ts)
    out = np.full(len(entity_ids), np.nan, dtype=np.float64)
    for i, eid in enumerate(entity_ids):
        v = lookup.get((eid, key))
        if v is not None and math.isfinite(v):
            out[i] = float(v)
    return out


def _latest_lookup_ts_per_entity_on_date(
    lookup: dict[tuple[str, int], float],
    iso_date: str,
) -> dict[str, int]:
    """For each entity, latest lookup key ts on calendar day *iso_date*."""
    from datetime import datetime, timezone

    day_start = (
        datetime.fromisoformat(iso_date).replace(tzinfo=timezone.utc).timestamp()
    )
    day_end = day_start + 86400.0
    best: dict[str, tuple[int, int]] = {}  # eid → (ts, priority=ts)
    for (eid, ts), _ in lookup.items():
        if not (day_start <= ts < day_end):
            continue
        prev = best.get(eid)
        if prev is None or ts > prev[0]:
            best[eid] = (ts, ts)
    return {eid: ts for eid, (ts, _) in best.items()}


def forward_return_vector_for_date(
    lookup: dict[tuple[str, int], float],
    entity_ids: list[str],
    iso_date: str,
) -> np.ndarray:
    """Forward returns aligned to walk-forward calendar fold dates.

    Walk-forward folds use ISO date strings (midnight UTC).  Observation
    timestamps in the DB are typically market-close times on that day.
    This helper maps each entity's latest label on *iso_date* into a vector.
    """
    ts_by_eid = _latest_lookup_ts_per_entity_on_date(lookup, iso_date)
    out = np.full(len(entity_ids), np.nan, dtype=np.float64)
    for i, eid in enumerate(entity_ids):
        ts = ts_by_eid.get(eid)
        if ts is None:
            continue
        v = lookup.get((eid, ts))
        if v is not None and math.isfinite(v):
            out[i] = float(v)
    return out


def mean_daily_log_return_vector(
    returns: np.ndarray,
    split: int,
    test_size: int,
) -> np.ndarray:
    """Legacy IC target: mean daily log_return over test window (phase40 default)."""
    return returns[split : split + test_size].mean(axis=0)


def sum_log_return_vector(
    daily_by_entity: dict[str, list[tuple[float, float]]],
    entity_ids: list[str],
    anchor_ts: float,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> np.ndarray:
    """train_raw_head label: sum of daily log_returns over horizon window."""
    buffer = horizon_days * CALENDAR_SECS_PER_TRADING_DAY
    out = np.full(len(entity_ids), np.nan, dtype=np.float64)
    for i, eid in enumerate(entity_ids):
        rows = daily_by_entity.get(eid, [])
        vals = [
            lr
            for ts, lr in rows
            if anchor_ts <= ts <= anchor_ts + buffer and math.isfinite(lr)
        ]
        if len(vals) >= max(1, horizon_days // 3):
            out[i] = float(sum(vals))
    return out


def label_distribution_stats(values: np.ndarray) -> dict[str, float]:
    """Summary stats for a forward-return array."""
    v = values[np.isfinite(values)]
    if len(v) == 0:
        return {
            "n": 0.0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "pct_zero": float("nan"),
        }
    return {
        "n": float(len(v)),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "min": float(v.min()),
        "max": float(v.max()),
        "pct_zero": float((np.abs(v) < 1e-8).mean()),
    }


def compare_lookups(
    lookup_a: dict[tuple[str, int], float],
    lookup_b: dict[tuple[str, int], float],
) -> dict[str, float]:
    """Compare two forward-return lookups on shared keys."""
    keys = set(lookup_a) & set(lookup_b)
    if not keys:
        return {"n_shared": 0.0, "spearman": float("nan"), "mean_abs_diff": float("nan")}
    a = np.array([lookup_a[k] for k in keys], dtype=np.float64)
    b = np.array([lookup_b[k] for k in keys], dtype=np.float64)
    from scipy.stats import spearmanr

    rho, _ = spearmanr(a, b)
    return {
        "n_shared": float(len(keys)),
        "spearman": float(rho) if math.isfinite(rho) else float("nan"),
        "mean_abs_diff": float(np.mean(np.abs(a - b))),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "std_a": float(a.std()),
        "std_b": float(b.std()),
    }


def audit_window_label_leakage(
    windows: list[tuple[float, float, list[dict]]],
    lookup: dict[tuple[str, int], float],
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> dict[str, float]:
    """Fraction of labels whose future close falls inside the feature window.

    For each training window (t_start, t_end, obs), check instrument_daily
    obs in the window: if forward-return future close ts <= t_end, count leak.
    """
    min_lag = horizon_days * CALENDAR_SECS_PER_TRADING_DAY - DEFAULT_TOLERANCE_SECS
    total = 0
    leaked = 0
    by_entity = _group_instrument_closes(
        [o for _, _, obs in windows for o in obs if o.get("observation_type") == "instrument_daily"]
    )

    for _t_start, t_end, curr_obs in windows:
        for o in curr_obs:
            if o.get("observation_type") != "instrument_daily":
                continue
            eid = o.get("entity_id")
            if not eid:
                continue
            ts_now = float(o.get("observed_at", 0.0))
            key = (eid, int(ts_now))
            if key not in lookup:
                continue
            total += 1
            sorted_obs = by_entity.get(eid, [])
            if not sorted_obs:
                continue
            timestamps = [x[0] for x in sorted_obs]
            target_ts = ts_now + horizon_days * CALENDAR_SECS_PER_TRADING_DAY
            j = bisect.bisect_left(timestamps, target_ts)
            best_future_ts: float | None = None
            best_dist = float("inf")
            for cand in (j - 1, j):
                if 0 <= cand < len(sorted_obs):
                    fut_ts = sorted_obs[cand][0]
                    dist = abs(fut_ts - target_ts)
                    if dist < DEFAULT_TOLERANCE_SECS and dist < best_dist:
                        best_dist = dist
                        best_future_ts = fut_ts
            if best_future_ts is not None and best_future_ts <= t_end:
                leaked += 1

    pct = (leaked / total) if total else 0.0
    return {
        "n_labels_checked": float(total),
        "n_leaked": float(leaked),
        "pct_leaked": pct,
    }


def _group_instrument_closes(
    observations: list[dict],
) -> dict[str, list[tuple[float, float]]]:
    """entity_id → sorted [(ts, close), ...]."""
    by_entity: dict[str, list[tuple[float, float]]] = {}
    for o in observations:
        if o.get("observation_type") != "instrument_daily":
            continue
        v = o.get("value", {})
        if not isinstance(v, dict):
            continue
        close = v.get("close")
        if close is None:
            continue
        try:
            close_f = float(close)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(close_f):
            continue
        eid = o.get("entity_id")
        if not eid:
            continue
        ts = float(o.get("observed_at", 0.0))
        by_entity.setdefault(eid, []).append((ts, close_f))
    for eid in by_entity:
        by_entity[eid].sort(key=lambda x: x[0])
    return by_entity


def _group_daily_log_returns(
    observations: list[dict],
) -> dict[str, list[tuple[float, float]]]:
    """entity_id → sorted [(ts, log_return), ...]."""
    out: dict[str, list[tuple[float, float]]] = {}
    for o in observations:
        if o.get("observation_type") != "instrument_daily":
            continue
        v = o.get("value", {})
        if not isinstance(v, dict):
            continue
        lr = v.get("log_return")
        if lr is None:
            continue
        try:
            lr_f = float(lr)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(lr_f):
            continue
        eid = o.get("entity_id")
        if not eid:
            continue
        ts = float(o.get("observed_at", 0.0))
        out.setdefault(eid, []).append((ts, lr_f))
    for eid in out:
        out[eid].sort(key=lambda x: x[0])
    return out


def label_method_correlation(
    observations: list[dict],
    entity_ids: list[str],
    anchor_timestamps: list[float],
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> dict[str, float]:
    """Spearman correlation between label definitions on shared samples."""
    from scipy.stats import spearmanr

    lookup = build_forward_return_lookup(observations, horizon_days=horizon_days)
    daily = _group_daily_log_returns(observations)

    simple: list[float] = []
    log_sum: list[float] = []
    mean_daily: list[float] = []

    for anchor_ts in anchor_timestamps:
        for i, eid in enumerate(entity_ids):
            key = (eid, int(anchor_ts))
            s = lookup.get(key)
            if s is None:
                continue
            rows = daily.get(eid, [])
            buffer = horizon_days * CALENDAR_SECS_PER_TRADING_DAY
            lrs = [
                lr
                for ts, lr in rows
                if anchor_ts <= ts <= anchor_ts + buffer and math.isfinite(lr)
            ]
            if len(lrs) < max(1, horizon_days // 3):
                continue
            simple.append(s)
            log_sum.append(sum(lrs))
            # mean daily over ~21 calendar-step returns in window
            mean_daily.append(float(np.mean(lrs[: min(len(lrs), horizon_days)])))

    if len(simple) < 10:
        return {"n": 0.0, "simple_vs_logsum": float("nan"), "simple_vs_meandaily": float("nan")}

    a = np.array(simple)
    b = np.array(log_sum)
    c = np.array(mean_daily)
    r1, _ = spearmanr(a, b)
    r2, _ = spearmanr(a, c)
    return {
        "n": float(len(simple)),
        "simple_vs_logsum": float(r1) if math.isfinite(r1) else float("nan"),
        "simple_vs_meandaily": float(r2) if math.isfinite(r2) else float("nan"),
        "mean_abs_simple_minus_logsum": float(np.mean(np.abs(a - b))),
    }
