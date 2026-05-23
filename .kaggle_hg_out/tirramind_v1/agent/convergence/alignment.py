"""Temporal alignment for convergence detection.

Aligns multi-frequency evidence streams to common time grids using
last-observation-carried-forward (LOCF).

**Critical invariants (future leakage prevention):**

- LOCF only — never interpolate, never use future observations.
- Always align to the COARSER of two grids — never upsample.
- Staleness enforcement via TTL — expired observations become NaN.
- Grid boundaries snap backward (floor), never forward.
"""

from __future__ import annotations

import enum
import logging
import math

import numpy as np

from agent.convergence.evidence import Evidence
from agent.convergence.taxonomy import SignalMeta

log = logging.getLogger(__name__)


# ── TimeGrid ───────────────────────────────────────────────────


class TimeGrid(enum.IntEnum):
    """Canonical time grid resolutions, ordered finest → coarsest.

    IntEnum ordering ensures ``max(a, b)`` returns the coarser grid.
    """

    INTRADAY = 1
    DAILY = 2
    WEEKLY = 3
    MONTHLY = 4

    def period_seconds(self) -> int:
        """Canonical period length in seconds."""
        return _PERIOD_SECONDS[self]

    @staticmethod
    def coarser(a: TimeGrid, b: TimeGrid) -> TimeGrid:
        """Return the coarser (lower-frequency) of two grids."""
        return max(a, b)


_PERIOD_SECONDS: dict[TimeGrid, int] = {
    TimeGrid.INTRADAY: 3_600,
    TimeGrid.DAILY: 86_400,
    TimeGrid.WEEKLY: 604_800,
    TimeGrid.MONTHLY: 2_592_000,  # 30 days
}


# ── Frequency → Grid mapping ──────────────────────────────────


FREQUENCY_TO_GRID: dict[str, TimeGrid] = {
    "intraday": TimeGrid.INTRADAY,
    "daily": TimeGrid.DAILY,
    "weekly": TimeGrid.WEEKLY,
    "monthly": TimeGrid.MONTHLY,
    "event": TimeGrid.DAILY,  # Events → daily binary flags
}


# ── Staleness ──────────────────────────────────────────────────


def is_stale(evidence: Evidence, as_of: float) -> bool:
    """Check whether an evidence observation has exceeded its TTL.

    Parameters
    ----------
    evidence : Evidence
        The observation to check.
    as_of : float
        Reference time (Unix epoch) to check staleness against.

    Returns
    -------
    bool
        True if ``as_of - evidence.timestamp`` exceeds ``evidence.ttl``.
    """
    return (as_of - evidence.timestamp) > evidence.ttl


# ── Grid generation ────────────────────────────────────────────


def _make_grid(start: float, end: float, period: int) -> np.ndarray:
    """Create grid-aligned timestamps covering [start, end].

    Snaps ``start`` down to the nearest period boundary and generates
    evenly spaced timestamps up to (and possibly including) ``end``.

    Returns empty array if ``start > end`` or ``period <= 0``.
    """
    if period <= 0 or start > end:
        return np.array([], dtype=np.float64)

    grid_start = math.floor(start / period) * period
    n_points = int(math.floor((end - grid_start) / period)) + 1
    if n_points <= 0:
        return np.array([], dtype=np.float64)

    return grid_start + np.arange(n_points, dtype=np.float64) * period


# ── LOCF alignment ─────────────────────────────────────────────


def _align_locf(
    sorted_ev: list[Evidence],
    timestamps: np.ndarray,
) -> np.ndarray:
    """LOCF (last observation carried forward) alignment with staleness.

    For each grid point *t*, uses the value of the most recent observation
    whose timestamp is ≤ *t*.  If that observation is stale
    (``t - obs.timestamp > obs.ttl``), the value is NaN.

    Never interpolates.  Never uses future observations.
    """
    values = np.full(len(timestamps), np.nan)
    ev_idx = 0
    last_ev: Evidence | None = None
    n_ev = len(sorted_ev)

    for i, t in enumerate(timestamps):
        # Advance to the latest evidence at or before t
        while ev_idx < n_ev and sorted_ev[ev_idx].timestamp <= t:
            last_ev = sorted_ev[ev_idx]
            ev_idx += 1

        if last_ev is not None and not is_stale(last_ev, t):
            values[i] = last_ev.value

    return values


# ── Event-mode alignment ──────────────────────────────────────


def _align_events(
    sorted_ev: list[Evidence],
    timestamps: np.ndarray,
    period: int,
) -> np.ndarray:
    """Binary-flag alignment for event-driven signals.

    For each grid period ``[t, t + period)``, the value is 1.0 if at
    least one event falls within the period, 0.0 otherwise.
    """
    values = np.zeros(len(timestamps), dtype=np.float64)

    if not sorted_ev:
        return values

    ev_times = np.array([e.timestamp for e in sorted_ev], dtype=np.float64)

    for i, t in enumerate(timestamps):
        lo = int(np.searchsorted(ev_times, t, side="left"))
        hi = int(np.searchsorted(ev_times, t + period, side="left"))
        if lo < hi:
            values[i] = 1.0

    return values


# ── Public API ─────────────────────────────────────────────────


def align_to_grid(
    series: list[Evidence],
    grid: TimeGrid,
    start: float,
    end: float,
    *,
    event_mode: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Align an evidence series to a regular time grid.

    Parameters
    ----------
    series : list[Evidence]
        Evidence observations for a single signal stream.
    grid : TimeGrid
        Target grid resolution.
    start, end : float
        Time range (Unix epoch) for the output grid.
    event_mode : bool
        If True, produce binary flags (1 = event in period, 0 = none)
        instead of LOCF values.  Use for event-driven signals.

    Returns
    -------
    timestamps : np.ndarray
        Grid-aligned timestamp array.
    values : np.ndarray
        Aligned values (NaN where no valid observation exists).
    """
    period = grid.period_seconds()
    timestamps = _make_grid(start, end, period)

    if len(timestamps) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    if not series:
        if event_mode:
            return timestamps, np.zeros(len(timestamps), dtype=np.float64)
        return timestamps, np.full(len(timestamps), np.nan)

    sorted_ev = sorted(series, key=lambda e: e.timestamp)

    if event_mode:
        return timestamps, _align_events(sorted_ev, timestamps, period)
    else:
        return timestamps, _align_locf(sorted_ev, timestamps)


def align_pair(
    series_a: list[Evidence],
    series_b: list[Evidence],
    meta_a: SignalMeta,
    meta_b: SignalMeta,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align two evidence series to a common time grid.

    The target grid is the coarser of the two signals' native
    frequencies.  The time range is the intersection of both series'
    observation windows.

    Parameters
    ----------
    series_a, series_b : list[Evidence]
        Evidence observations for signals A and B.
    meta_a, meta_b : SignalMeta
        Metadata providing frequency information.

    Returns
    -------
    timestamps : np.ndarray
        Common grid-aligned timestamps.
    values_a, values_b : np.ndarray
        Aligned values for each signal (NaN where missing / stale).

    Notes
    -----
    Returns three empty arrays when either series is empty or the
    time ranges do not overlap.
    """
    empty = np.array([], dtype=np.float64)

    if not series_a or not series_b:
        return empty, empty, empty

    # Determine target grid (coarser of the two)
    grid_a = FREQUENCY_TO_GRID.get(meta_a.frequency, TimeGrid.DAILY)
    grid_b = FREQUENCY_TO_GRID.get(meta_b.frequency, TimeGrid.DAILY)
    target_grid = TimeGrid.coarser(grid_a, grid_b)

    # Time range: intersection of both series
    times_a = [e.timestamp for e in series_a]
    times_b = [e.timestamp for e in series_b]

    start = max(min(times_a), min(times_b))
    end = min(max(times_a), max(times_b))

    if start > end:
        return empty, empty, empty

    event_a = meta_a.frequency == "event"
    event_b = meta_b.frequency == "event"

    ts_a, vals_a = align_to_grid(series_a, target_grid, start, end, event_mode=event_a)
    ts_b, vals_b = align_to_grid(series_b, target_grid, start, end, event_mode=event_b)

    # Both calls use the same (start, end, grid) → identical timestamps
    return ts_a, vals_a, vals_b
