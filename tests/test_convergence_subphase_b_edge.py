"""Edge-case test suite for Sub-phase 7c-B (Temporal Alignment + Atomic Signals).

Covers the hardest corners from the spec:
- All-NaN series, σ=0, window=1, window>data
- Out-of-order and duplicate timestamps
- Staleness exact boundary
- Ties in percentile, all-anomalous series
- Extreme asymmetry in series lengths
- Alignment edge cases (all-identical timestamps, non-overlapping)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from agent.convergence.alignment import (
    TimeGrid,
    _make_grid,
    align_pair,
    align_to_grid,
    is_stale,
)
from agent.convergence.atomic_signals import (
    AtomicSignalResult,
    RollingStats,
    SignalStream,
    compute_anomaly,
    normalize_direction,
)
from agent.convergence.evidence import Evidence
from agent.convergence.taxonomy import SignalMeta


# ── Helpers ────────────────────────────────────────────────────

DAY = 86_400
HOUR = 3_600
WEEK = 604_800
T0 = 1743465600.0  # 2026-04-01 00:00:00 UTC (daily boundary)


def _ev(
    value: float = 1.0,
    timestamp: float = T0,
    ttl: int = DAY,
    **kw,
) -> Evidence:
    defaults = dict(
        source="test_tool",
        signal_id="test.signal",
        direction=1,
        confidence=0.8,
        category="positioning",
        tags=(),
    )
    defaults.update(kw)
    return Evidence(value=value, timestamp=timestamp, ttl=ttl, **defaults)


def _meta(
    frequency: str = "daily",
    flip_sign: bool = False,
    min_observations: int = 3,
    signal_id: str = "test.signal",
    **kw,
) -> SignalMeta:
    defaults = dict(
        source="test_tool",
        category="positioning",
        direction_semantics="higher = stress",
    )
    defaults.update(kw)
    return SignalMeta(
        signal_id=signal_id,
        frequency=frequency,
        flip_sign=flip_sign,
        min_observations=min_observations,
        **defaults,
    )


# ═══════════════════════════════════════════════════════════════
# Alignment edge cases
# ═══════════════════════════════════════════════════════════════


class TestAlignmentAllNaN:
    """Alignment with series containing only NaN values."""

    def test_all_nan_values_locf(self):
        series = [
            _ev(value=float("nan"), timestamp=T0, ttl=3 * DAY),
            _ev(value=float("nan"), timestamp=T0 + DAY, ttl=3 * DAY),
        ]
        ts, vals = align_to_grid(series, TimeGrid.DAILY, T0, T0 + 2 * DAY)
        # NaN is a valid value — it's carried forward by LOCF
        assert len(vals) == 3
        assert all(np.isnan(vals))

    def test_all_nan_values_event_mode(self):
        """Event mode doesn't care about value — only presence."""
        series = [
            _ev(value=float("nan"), timestamp=T0 + 100),
        ]
        ts, vals = align_to_grid(series, TimeGrid.DAILY, T0, T0 + DAY, event_mode=True)
        assert vals[0] == 1.0  # event present regardless of value


class TestAlignmentDuplicateTimestamps:
    """Alignment with all-identical timestamps."""

    def test_all_same_timestamp_locf(self):
        """Multiple evidence at the same time — LOCF uses the last one sorted."""
        series = [
            _ev(value=10.0, timestamp=T0, ttl=3 * DAY),
            _ev(value=20.0, timestamp=T0, ttl=3 * DAY),
            _ev(value=30.0, timestamp=T0, ttl=3 * DAY),
        ]
        ts, vals = align_to_grid(series, TimeGrid.DAILY, T0, T0 + DAY)
        # sorted() is stable — all have same timestamp, last in list = 30.0
        assert vals[0] == 30.0

    def test_duplicates_in_event_mode(self):
        series = [
            _ev(timestamp=T0 + 50),
            _ev(timestamp=T0 + 50),
            _ev(timestamp=T0 + 50),
        ]
        ts, vals = align_to_grid(series, TimeGrid.DAILY, T0, T0, event_mode=True)
        assert vals[0] == 1.0  # still binary


class TestAlignmentAsymmetricLengths:
    """One series has 1 observation, the other has 1000."""

    def test_short_vs_long_pair(self):
        short = [_ev(value=1.0, timestamp=T0, ttl=1000 * DAY)]
        long = [
            _ev(
                value=float(i),
                timestamp=T0 + i * DAY,
                ttl=2 * DAY,
                signal_id="test.long",
            )
            for i in range(1000)
        ]
        meta_s = _meta(signal_id="test.short")
        meta_l = _meta(signal_id="test.long")

        ts, va, vb = align_pair(short, long, meta_s, meta_l)
        # Intersection starts at T0, ends at T0 + 999*DAY
        # Short has 1 obs at T0 with huge TTL → carried forward everywhere
        assert len(ts) > 0
        assert va[0] == 1.0
        # Long values should be present
        assert not np.isnan(vb[0])


class TestAlignmentStalenessExactBoundary:
    """evidence.timestamp + evidence.ttl exactly equals as_of."""

    def test_exact_ttl_boundary_in_alignment(self):
        """At timestamp + ttl, the observation is NOT stale (> not >=)."""
        ev = _ev(value=42.0, timestamp=T0, ttl=DAY)
        assert not is_stale(ev, T0 + DAY)  # age == ttl → not stale

    def test_one_tick_past_boundary(self):
        ev = _ev(value=42.0, timestamp=T0, ttl=DAY)
        assert is_stale(ev, T0 + DAY + 0.001)  # age > ttl → stale

    def test_staleness_boundary_in_align_to_grid(self):
        """Grid point exactly at timestamp + ttl → value present."""
        series = [_ev(value=7.0, timestamp=T0, ttl=DAY)]
        # Grid: T0, T0+DAY. At T0+DAY: age=DAY == ttl → not stale
        ts, vals = align_to_grid(series, TimeGrid.DAILY, T0, T0 + DAY)
        assert vals[0] == 7.0
        assert vals[1] == 7.0  # exactly at boundary → still valid


class TestAlignmentNonOverlapping:
    """Series with no temporal overlap."""

    def test_non_overlapping_returns_empty(self):
        a = [_ev(value=1.0, timestamp=T0)]
        b = [_ev(value=2.0, timestamp=T0 + 100 * DAY)]
        ts, va, vb = align_pair(a, b, _meta(), _meta(signal_id="b"))
        assert len(ts) == 0
        assert len(va) == 0
        assert len(vb) == 0

    def test_barely_overlapping(self):
        """Series overlap at a single timestamp."""
        a = [_ev(value=1.0, timestamp=T0)]
        b = [_ev(value=2.0, timestamp=T0)]
        ts, va, vb = align_pair(a, b, _meta(), _meta(signal_id="b"))
        assert len(ts) >= 1
        assert va[0] == 1.0
        assert vb[0] == 2.0


# ═══════════════════════════════════════════════════════════════
# RollingStats edge cases
# ═══════════════════════════════════════════════════════════════


class TestRollingStatsWindow1:
    """Window=1: only the latest value matters."""

    def test_window_1_z_score(self):
        rs = RollingStats(window=1)
        rs.update(np.array([10.0, 20.0, 30.0]))
        # Only [30.0] kept. std=0 → z=0
        assert rs.n_observations == 1
        assert rs.z_score(30.0) == 0.0
        assert rs.z_score(100.0) == 0.0

    def test_window_1_percentile(self):
        rs = RollingStats(window=1)
        rs.update(np.array([5.0]))
        assert rs.percentile(5.0) == 1.0
        assert rs.percentile(0.0) == 0.0

    def test_window_1_mean(self):
        rs = RollingStats(window=1)
        rs.update(np.array([7.0]))
        assert rs.mean == 7.0


class TestRollingStatsWindowLargerThanData:
    """Window larger than available data: all data retained."""

    def test_window_exceeds_data(self):
        rs = RollingStats(window=1000)
        rs.update(np.array([1.0, 2.0, 3.0]))
        assert rs.n_observations == 3
        assert rs.mean == pytest.approx(2.0)

    def test_z_score_works_with_small_n(self):
        rs = RollingStats(window=1000)
        rs.update(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        z = rs.z_score(5.0)
        expected = (5.0 - 3.0) / np.std([1, 2, 3, 4, 5], ddof=1)
        assert z == pytest.approx(expected)


class TestRollingStatsZeroSeries:
    """Series of all zeros (σ=0)."""

    def test_all_zeros_z_score(self):
        rs = RollingStats(window=10)
        rs.update(np.zeros(10))
        assert rs.z_score(0.0) == 0.0
        assert rs.z_score(1.0) == 0.0  # σ≈0 → z=0

    def test_all_zeros_percentile(self):
        rs = RollingStats(window=10)
        rs.update(np.zeros(10))
        assert rs.percentile(0.0) == 1.0  # all 10 are ≤ 0
        assert rs.percentile(-1.0) == 0.0  # none are ≤ -1


class TestRollingStatsAllIdentical:
    """All-identical nonzero values (σ=0)."""

    def test_identical_values(self):
        rs = RollingStats(window=10)
        rs.update(np.full(10, 42.0))
        assert rs.mean == 42.0
        assert rs.std < 1e-10
        assert rs.z_score(42.0) == 0.0
        assert rs.z_score(100.0) == 0.0  # σ≈0 → z=0


class TestRollingStatsPercentileTies:
    """Multiple identical values — ties in percentile ranking."""

    def test_all_ties(self):
        """All values identical: percentile(that_value) = 1.0."""
        rs = RollingStats(window=10)
        rs.update(np.full(10, 5.0))
        assert rs.percentile(5.0) == 1.0
        assert rs.percentile(4.9) == 0.0
        assert rs.percentile(5.1) == 1.0

    def test_partial_ties(self):
        """[1, 1, 1, 1, 5]: percentile(1) = 4/5 = 0.8."""
        rs = RollingStats(window=10)
        rs.update(np.array([1.0, 1.0, 1.0, 1.0, 5.0]))
        assert rs.percentile(1.0) == pytest.approx(0.8)
        assert rs.percentile(5.0) == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════
# SignalStream edge cases
# ═══════════════════════════════════════════════════════════════


class TestSignalStreamOutOfOrder:
    """Ingest with out-of-order timestamps — should sort correctly."""

    def test_out_of_order_ingest(self):
        stream = SignalStream("sig", _meta())
        evs = [
            _ev(value=30.0, timestamp=T0 + 2 * DAY),
            _ev(value=10.0, timestamp=T0),
            _ev(value=20.0, timestamp=T0 + DAY),
        ]
        stream.ingest(evs)
        np.testing.assert_array_equal(stream.history(), [10.0, 20.0, 30.0])


class TestSignalStreamDuplicateTimestamps:
    """Duplicate timestamps — should deduplicate, keep latest."""

    def test_same_timestamp_multiple_ingests(self):
        """Second ingest overwrites first for same timestamp."""
        stream = SignalStream("sig", _meta())
        stream.ingest([_ev(value=1.0, timestamp=T0)])
        stream.ingest([_ev(value=99.0, timestamp=T0)])
        np.testing.assert_array_equal(stream.history(), [99.0])

    def test_same_timestamp_single_ingest(self):
        """Within one ingest, last value for a timestamp wins."""
        stream = SignalStream("sig", _meta())
        stream.ingest(
            [
                _ev(value=1.0, timestamp=T0),
                _ev(value=2.0, timestamp=T0),
                _ev(value=3.0, timestamp=T0),
            ]
        )
        np.testing.assert_array_equal(stream.history(), [3.0])

    def test_duplicate_then_compute(self):
        stream = SignalStream("sig", _meta(min_observations=1))
        stream.ingest(
            [
                _ev(value=5.0, timestamp=T0),
                _ev(value=10.0, timestamp=T0),
            ]
        )
        result = stream.compute(as_of=T0)
        assert result is not None
        assert result.raw_value == 10.0


class TestSignalStreamFlipSignOnZero:
    """Direction normalization with flip_sign on zero value."""

    def test_flip_zero(self):
        assert normalize_direction(0.0, flip_sign=True) == 0.0
        # Negative zero should also be zero
        assert normalize_direction(-0.0, flip_sign=True) == 0.0

    def test_stream_flip_zero(self):
        m = _meta(flip_sign=True, min_observations=1)
        stream = SignalStream("sig", m)
        stream.ingest([_ev(value=0.0, timestamp=T0)])
        np.testing.assert_array_equal(stream.history(), [0.0])


class TestAnomalyAllExtreme:
    """All values in the series are extreme — anomaly flag still works."""

    def test_all_extreme_z_scores(self):
        """When every value is extreme, latest is still flagged."""
        m = _meta(min_observations=3)
        stream = SignalStream("sig", m, window=52)
        # 30 values, then massive spike — all normal, one extreme
        evs = [_ev(value=10.0, timestamp=T0 + i * DAY) for i in range(30)]
        evs.append(_ev(value=1000.0, timestamp=T0 + 30 * DAY))
        stream.ingest(evs)
        result = stream.compute(as_of=T0 + 30 * DAY)
        assert result.is_anomaly is True

    def test_uniform_extreme_no_anomaly(self):
        """All identical extreme values → σ=0 → z=0 → not anomalous
        (unless percentile triggers)."""
        m = _meta(min_observations=3)
        stream = SignalStream("sig", m, window=52)
        evs = [_ev(value=9999.0, timestamp=T0 + i * DAY) for i in range(30)]
        stream.ingest(evs)
        result = stream.compute(as_of=T0 + 29 * DAY)
        # z=0 (σ=0), percentile=1.0 which is > 0.95 → anomalous
        assert result.z_score == 0.0
        assert result.percentile == 1.0
        assert result.is_anomaly is True


class TestSignalStreamNaNValues:
    """NaN values in evidence ingested into SignalStream."""

    def test_nan_value_ingested(self):
        stream = SignalStream("sig", _meta(min_observations=1))
        stream.ingest(
            [
                _ev(value=float("nan"), timestamp=T0),
                _ev(value=5.0, timestamp=T0 + DAY),
            ]
        )
        # History should include the NaN (direction-normalized)
        h = stream.history()
        assert np.isnan(h[0])
        assert h[1] == 5.0

    def test_compute_with_nan_latest(self):
        """If the latest value is NaN, compute returns None."""
        stream = SignalStream("sig", _meta(min_observations=1))
        stream.ingest(
            [
                _ev(value=5.0, timestamp=T0),
                _ev(value=float("nan"), timestamp=T0 + DAY),
            ]
        )
        result = stream.compute(as_of=T0 + DAY)
        assert result is None

    def test_nan_excluded_from_rolling_stats(self):
        """NaN values should not count toward n_observations."""
        rs = RollingStats(window=10)
        rs.update(np.array([1.0, np.nan, 3.0, np.nan, 5.0]))
        assert rs.n_observations == 3


# ═══════════════════════════════════════════════════════════════
# Cross-module integration edge cases
# ═══════════════════════════════════════════════════════════════


class TestAlignmentIntoAtomicSignal:
    """Full pipeline: align evidence → compute atomic signal."""

    def test_aligned_series_fed_to_signal_stream(self):
        """align_to_grid output → Evidence reconstruction → SignalStream."""
        raw_evs = [
            _ev(value=float(i), timestamp=T0 + i * DAY, ttl=3 * DAY) for i in range(30)
        ]
        ts, vals = align_to_grid(raw_evs, TimeGrid.DAILY, T0, T0 + 29 * DAY)

        # Create reconstructed evidence from aligned data
        aligned_evs = []
        for t, v in zip(ts, vals):
            if not np.isnan(v):
                aligned_evs.append(_ev(value=v, timestamp=t, ttl=3 * DAY))

        m = _meta(min_observations=5)
        stream = SignalStream("aligned.sig", m, window=52)
        stream.ingest(aligned_evs)
        result = stream.compute(as_of=T0 + 29 * DAY)

        assert result is not None
        assert result.signal_id == "aligned.sig"
        assert isinstance(result.z_score, float)
        assert isinstance(result.percentile, float)
        assert result.direction in (1, -1)

    def test_stale_data_becomes_nan_then_filtered(self):
        """Stale gaps produce NaN in alignment; NaN excluded from stats."""
        raw_evs = [
            _ev(value=10.0, timestamp=T0, ttl=DAY),  # expires after 1 day
            _ev(value=20.0, timestamp=T0 + 5 * DAY, ttl=DAY),
        ]
        ts, vals = align_to_grid(raw_evs, TimeGrid.DAILY, T0, T0 + 6 * DAY)

        # Days 0-1: 10.0, Days 2-4: NaN (stale), Days 5-6: 20.0
        non_nan_count = sum(1 for v in vals if not np.isnan(v))
        nan_count = sum(1 for v in vals if np.isnan(v))
        assert non_nan_count >= 2
        assert nan_count >= 2  # gap in the middle


class TestAlignPairEdgeCases:
    """Additional align_pair edge cases."""

    def test_event_plus_weekly(self):
        """Event-driven + weekly → weekly grid, events as binary."""
        events = [
            _ev(value=6.5, timestamp=T0 + DAY),
            _ev(value=7.2, timestamp=T0 + 8 * DAY),
        ]
        weekly = [
            _ev(value=100.0, timestamp=T0, ttl=2 * WEEK, signal_id="weekly.sig"),
            _ev(value=200.0, timestamp=T0 + WEEK, ttl=2 * WEEK, signal_id="weekly.sig"),
        ]
        meta_ev = _meta(frequency="event", signal_id="events.sig")
        meta_w = _meta(frequency="weekly", signal_id="weekly.sig")

        ts, v_ev, v_w = align_pair(events, weekly, meta_ev, meta_w)

        # Grid should be weekly (coarser)
        if len(ts) >= 2:
            assert ts[1] - ts[0] == WEEK
        # Event values should be binary
        for v in v_ev:
            assert v in (0.0, 1.0) or np.isnan(v)

    def test_both_event_frequency(self):
        """Two event-driven signals → daily grid (event maps to daily)."""
        a = [_ev(value=1.0, timestamp=T0 + 100)]
        b = [_ev(value=2.0, timestamp=T0 + 200, signal_id="b")]
        meta_a = _meta(frequency="event")
        meta_b = _meta(frequency="event", signal_id="b")

        ts, va, vb = align_pair(a, b, meta_a, meta_b)

        # Both should be binary
        if len(ts) > 0:
            for v in va:
                assert v in (0.0, 1.0) or np.isnan(v)
            for v in vb:
                assert v in (0.0, 1.0) or np.isnan(v)


class TestMakeGridEdgeCases:
    """Grid generation corner cases."""

    def test_very_large_range(self):
        """365 days of hourly grid → performance check."""
        grid = _make_grid(T0, T0 + 365 * DAY, HOUR)
        assert len(grid) == 365 * 24 + 1  # inclusive

    def test_fractional_timestamps(self):
        """Floating-point timestamps snap correctly."""
        grid = _make_grid(T0 + 0.5, T0 + DAY + 0.5, DAY)
        assert grid[0] == T0  # floor(T0+0.5 / DAY)*DAY == T0
        assert len(grid) >= 1
