"""Tests for atomic signal computation (Phase 7c-B.2).

Covers: RollingStats (z-score, percentile, windowing, NaN handling),
compute_anomaly thresholds, normalize_direction, SignalStream
(ingest, compute, history, cold-start).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

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
T0 = 1743465600.0  # 2026-04-01 00:00:00 UTC


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
    min_observations: int = 5,
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
# RollingStats — construction and validation
# ═══════════════════════════════════════════════════════════════


class TestRollingStatsConstruction:
    def test_default_window(self):
        rs = RollingStats()
        assert rs.window == 52

    def test_custom_window(self):
        rs = RollingStats(window=10)
        assert rs.window == 10

    def test_window_must_be_positive(self):
        with pytest.raises(ValueError, match="window must be >= 1"):
            RollingStats(window=0)
        with pytest.raises(ValueError):
            RollingStats(window=-5)

    def test_empty_initial_state(self):
        rs = RollingStats()
        assert rs.n_observations == 0
        assert math.isnan(rs.mean)
        assert rs.std == 0.0


# ═══════════════════════════════════════════════════════════════
# RollingStats — update and windowing
# ═══════════════════════════════════════════════════════════════


class TestRollingStatsUpdate:
    def test_basic_update(self):
        rs = RollingStats(window=5)
        rs.update(np.array([1.0, 2.0, 3.0]))
        assert rs.n_observations == 3

    def test_window_trims_oldest(self):
        rs = RollingStats(window=3)
        rs.update(np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
        assert rs.n_observations == 3
        # Should keep [30, 40, 50]
        assert rs.mean == pytest.approx(40.0)

    def test_nan_values_filtered(self):
        rs = RollingStats(window=10)
        rs.update(np.array([1.0, np.nan, 3.0, np.nan, 5.0]))
        assert rs.n_observations == 3
        assert rs.mean == pytest.approx(3.0)

    def test_all_nan(self):
        rs = RollingStats(window=10)
        rs.update(np.array([np.nan, np.nan, np.nan]))
        assert rs.n_observations == 0

    def test_update_is_full_refresh(self):
        """Calling update replaces previous buffer, not appends."""
        rs = RollingStats(window=10)
        rs.update(np.array([1.0, 2.0, 3.0]))
        assert rs.n_observations == 3
        rs.update(np.array([10.0, 20.0]))
        assert rs.n_observations == 2
        assert rs.mean == pytest.approx(15.0)

    def test_empty_array(self):
        rs = RollingStats(window=10)
        rs.update(np.array([]))
        assert rs.n_observations == 0


# ═══════════════════════════════════════════════════════════════
# RollingStats — z-score
# ═══════════════════════════════════════════════════════════════


class TestRollingStatsZScore:
    def test_hand_computed_z_score(self):
        """Known values: [1, 2, 3, 4, 5]. mean=3, std≈1.581."""
        rs = RollingStats(window=10)
        rs.update(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        # z(5) = (5 - 3) / std([1,2,3,4,5], ddof=1) = 2 / 1.5811... ≈ 1.265
        z = rs.z_score(5.0)
        expected = (5.0 - 3.0) / np.std([1, 2, 3, 4, 5], ddof=1)
        assert z == pytest.approx(expected, rel=1e-6)

    def test_z_score_of_mean_is_zero(self):
        rs = RollingStats(window=10)
        rs.update(np.array([10.0, 20.0, 30.0]))
        assert rs.z_score(rs.mean) == pytest.approx(0.0)

    def test_z_score_with_zero_std(self):
        """All identical values → σ ≈ 0 → z = 0."""
        rs = RollingStats(window=10)
        rs.update(np.array([5.0, 5.0, 5.0, 5.0]))
        assert rs.z_score(5.0) == 0.0
        assert rs.z_score(100.0) == 0.0  # Still 0, not Inf

    def test_z_score_single_observation(self):
        rs = RollingStats(window=10)
        rs.update(np.array([42.0]))
        assert rs.z_score(42.0) == 0.0  # std=0 → z=0

    def test_z_score_empty_buffer(self):
        rs = RollingStats()
        assert rs.z_score(10.0) == 0.0  # no data, std=0

    def test_negative_z_score(self):
        rs = RollingStats(window=10)
        rs.update(np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
        z = rs.z_score(10.0)
        assert z < 0


# ═══════════════════════════════════════════════════════════════
# RollingStats — percentile
# ═══════════════════════════════════════════════════════════════


class TestRollingStatsPercentile:
    def test_hand_computed_percentile(self):
        """[1, 2, 3, 4, 5]: percentile(3) = 3/5 = 0.6."""
        rs = RollingStats(window=10)
        rs.update(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert rs.percentile(3.0) == pytest.approx(0.6)

    def test_min_value_percentile(self):
        rs = RollingStats(window=10)
        rs.update(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert rs.percentile(1.0) == pytest.approx(0.2)

    def test_max_value_percentile(self):
        rs = RollingStats(window=10)
        rs.update(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert rs.percentile(5.0) == pytest.approx(1.0)

    def test_below_all_values(self):
        rs = RollingStats(window=10)
        rs.update(np.array([10.0, 20.0, 30.0]))
        assert rs.percentile(5.0) == pytest.approx(0.0)

    def test_above_all_values(self):
        rs = RollingStats(window=10)
        rs.update(np.array([10.0, 20.0, 30.0]))
        assert rs.percentile(100.0) == pytest.approx(1.0)

    def test_empty_buffer_returns_neutral(self):
        rs = RollingStats()
        assert rs.percentile(42.0) == 0.5

    def test_ties(self):
        """[1, 1, 1, 2, 3]: percentile(1) = 3/5 = 0.6."""
        rs = RollingStats(window=10)
        rs.update(np.array([1.0, 1.0, 1.0, 2.0, 3.0]))
        assert rs.percentile(1.0) == pytest.approx(0.6)


# ═══════════════════════════════════════════════════════════════
# compute_anomaly
# ═══════════════════════════════════════════════════════════════


class TestComputeAnomaly:
    def test_z_above_threshold(self):
        assert compute_anomaly(z=2.5, pct=0.5) is True

    def test_z_below_negative_threshold(self):
        assert compute_anomaly(z=-2.5, pct=0.5) is True

    def test_pct_below_lo(self):
        assert compute_anomaly(z=0.5, pct=0.03) is True

    def test_pct_above_hi(self):
        assert compute_anomaly(z=0.5, pct=0.97) is True

    def test_not_anomalous(self):
        assert compute_anomaly(z=1.0, pct=0.5) is False

    def test_exactly_at_z_threshold_not_anomalous(self):
        # |z| > threshold, not >=
        assert compute_anomaly(z=2.0, pct=0.5) is False

    def test_exactly_at_pct_lo_not_anomalous(self):
        # pct < pct_lo, not <=
        assert compute_anomaly(z=0.0, pct=0.05) is False

    def test_exactly_at_pct_hi_not_anomalous(self):
        # pct > pct_hi, not >=
        assert compute_anomaly(z=0.0, pct=0.95) is False

    def test_custom_thresholds(self):
        assert compute_anomaly(z=1.5, pct=0.5, z_threshold=1.0) is True
        assert compute_anomaly(z=0.5, pct=0.08, pct_lo=0.10) is True
        assert compute_anomaly(z=0.5, pct=0.85, pct_hi=0.80) is True

    def test_multiple_triggers(self):
        """All three triggers simultaneously → anomalous."""
        assert compute_anomaly(z=3.0, pct=0.01) is True


# ═══════════════════════════════════════════════════════════════
# normalize_direction
# ═══════════════════════════════════════════════════════════════


class TestNormalizeDirection:
    def test_no_flip(self):
        assert normalize_direction(10.0, flip_sign=False) == 10.0

    def test_flip(self):
        assert normalize_direction(10.0, flip_sign=True) == -10.0

    def test_flip_negative(self):
        assert normalize_direction(-5.0, flip_sign=True) == 5.0

    def test_zero_unaffected(self):
        assert normalize_direction(0.0, flip_sign=True) == 0.0
        assert normalize_direction(0.0, flip_sign=False) == 0.0

    def test_nan_preserved(self):
        assert math.isnan(normalize_direction(float("nan"), flip_sign=False))
        assert math.isnan(normalize_direction(float("nan"), flip_sign=True))


# ═══════════════════════════════════════════════════════════════
# SignalStream — ingest
# ═══════════════════════════════════════════════════════════════


class TestSignalStreamIngest:
    def test_basic_ingest(self):
        stream = SignalStream("test.sig", _meta())
        stream.ingest([_ev(value=1.0), _ev(value=2.0, timestamp=T0 + DAY)])
        assert len(stream.history()) == 2

    def test_ingest_sorts_by_timestamp(self):
        stream = SignalStream("test.sig", _meta())
        evs = [
            _ev(value=3.0, timestamp=T0 + 2 * DAY),
            _ev(value=1.0, timestamp=T0),
            _ev(value=2.0, timestamp=T0 + DAY),
        ]
        stream.ingest(evs)
        np.testing.assert_array_equal(stream.history(), [1.0, 2.0, 3.0])

    def test_ingest_deduplicates_timestamps(self):
        """Duplicate timestamp → last-write-wins."""
        stream = SignalStream("test.sig", _meta())
        stream.ingest(
            [
                _ev(value=10.0, timestamp=T0),
                _ev(value=99.0, timestamp=T0),  # overwrites
            ]
        )
        np.testing.assert_array_equal(stream.history(), [99.0])

    def test_ingest_applies_flip_sign(self):
        m = _meta(flip_sign=True)
        stream = SignalStream("test.sig", m)
        stream.ingest([_ev(value=5.0, timestamp=T0)])
        np.testing.assert_array_equal(stream.history(), [-5.0])

    def test_multiple_ingest_calls_accumulate(self):
        stream = SignalStream("test.sig", _meta())
        stream.ingest([_ev(value=1.0, timestamp=T0)])
        stream.ingest([_ev(value=2.0, timestamp=T0 + DAY)])
        assert len(stream.history()) == 2

    def test_empty_ingest(self):
        stream = SignalStream("test.sig", _meta())
        stream.ingest([])
        assert len(stream.history()) == 0


# ═══════════════════════════════════════════════════════════════
# SignalStream — compute
# ═══════════════════════════════════════════════════════════════


class TestSignalStreamCompute:
    def _build_stream(self, n: int = 30, flip: bool = False) -> SignalStream:
        """Build a stream with n observations: values 1..n."""
        m = _meta(flip_sign=flip, min_observations=5)
        stream = SignalStream("test.sig", m, window=52)
        evs = [_ev(value=float(i + 1), timestamp=T0 + i * DAY) for i in range(n)]
        stream.ingest(evs)
        return stream

    def test_basic_compute(self):
        stream = self._build_stream(30)
        result = stream.compute(as_of=T0 + 29 * DAY)
        assert result is not None
        assert isinstance(result, AtomicSignalResult)
        assert result.signal_id == "test.sig"
        assert result.raw_value == 30.0
        assert result.timestamp == T0 + 29 * DAY

    def test_z_score_is_plausible(self):
        stream = self._build_stream(30)
        result = stream.compute(as_of=T0 + 29 * DAY)
        # Value 30 is the max — should be positive z-score
        assert result.z_score > 0

    def test_direction_positive_for_high_z(self):
        stream = self._build_stream(30)
        result = stream.compute(as_of=T0 + 29 * DAY)
        assert result.direction == 1

    def test_direction_negative_for_low_value(self):
        """Compute at early timestamp where the value is below mean."""
        m = _meta(min_observations=3)
        stream = SignalStream("test.sig", m, window=52)
        evs = [_ev(value=float(i), timestamp=T0 + i * DAY) for i in range(20)]
        stream.ingest(evs)
        # At T0 + 4*DAY: values [0..4], latest=4, mean=2
        # z = (4-2)/std > 0 → direction = +1
        # At T0: only value 0, can't compute mean properly with 1 obs
        # Let's use T0 + 3*DAY: values [0,1,2,3], mean=1.5, z(3) > 0
        # Need a case where z < 0: latest value below mean
        # Inject: values [10, 20, 30, 1, 2] — the last value (2) is below mean
        m2 = _meta(min_observations=3)
        stream2 = SignalStream("test.sig2", m2, window=52)
        stream2.ingest(
            [
                _ev(value=10.0, timestamp=T0),
                _ev(value=20.0, timestamp=T0 + DAY),
                _ev(value=30.0, timestamp=T0 + 2 * DAY),
                _ev(value=1.0, timestamp=T0 + 3 * DAY),
                _ev(value=2.0, timestamp=T0 + 4 * DAY),
            ]
        )
        result2 = stream2.compute(as_of=T0 + 4 * DAY)
        assert result2.z_score < 0
        assert result2.direction == -1

    def test_insufficient_observations_returns_none(self):
        m = _meta(min_observations=10)
        stream = SignalStream("test.sig", m)
        stream.ingest([_ev(value=1.0, timestamp=T0 + i * DAY) for i in range(5)])
        result = stream.compute(as_of=T0 + 4 * DAY)
        assert result is None

    def test_as_of_filters_future_observations(self):
        """Observations after as_of should not be used."""
        m = _meta(min_observations=3)
        stream = SignalStream("test.sig", m, window=52)
        stream.ingest([_ev(value=float(i), timestamp=T0 + i * DAY) for i in range(10)])
        # Only first 5 obs (0..4) should be used
        result = stream.compute(as_of=T0 + 4 * DAY)
        assert result is not None
        assert result.raw_value == 4.0
        assert result.timestamp == T0 + 4 * DAY

    def test_empty_stream_returns_none(self):
        stream = SignalStream("test.sig", _meta())
        assert stream.compute(as_of=T0) is None

    def test_compute_with_flip_sign(self):
        m = _meta(flip_sign=True, min_observations=3)
        stream = SignalStream("test.sig", m, window=52)
        stream.ingest(
            [
                _ev(value=10.0, timestamp=T0),
                _ev(value=20.0, timestamp=T0 + DAY),
                _ev(value=30.0, timestamp=T0 + 2 * DAY),
            ]
        )
        result = stream.compute(as_of=T0 + 2 * DAY)
        # Flipped: latest raw=-30, history=[-10,-20,-30]
        assert result.raw_value == -30.0

    def test_anomaly_flag_set(self):
        """Extreme value should trigger anomaly."""
        m = _meta(min_observations=3)
        stream = SignalStream("test.sig", m, window=52)
        # 29 normal values, then one extreme
        evs = [_ev(value=10.0, timestamp=T0 + i * DAY) for i in range(29)]
        evs.append(_ev(value=10000.0, timestamp=T0 + 29 * DAY))
        stream.ingest(evs)
        result = stream.compute(as_of=T0 + 29 * DAY)
        assert result is not None
        assert result.is_anomaly is True
        assert result.z_score > 2.0


# ═══════════════════════════════════════════════════════════════
# SignalStream — history
# ═══════════════════════════════════════════════════════════════


class TestSignalStreamHistory:
    def test_empty_history(self):
        stream = SignalStream("test.sig", _meta())
        h = stream.history()
        assert len(h) == 0

    def test_history_preserves_order(self):
        stream = SignalStream("test.sig", _meta())
        stream.ingest(
            [
                _ev(value=3.0, timestamp=T0 + 2 * DAY),
                _ev(value=1.0, timestamp=T0),
                _ev(value=2.0, timestamp=T0 + DAY),
            ]
        )
        np.testing.assert_array_equal(stream.history(), [1.0, 2.0, 3.0])

    def test_history_reflects_flip_sign(self):
        m = _meta(flip_sign=True)
        stream = SignalStream("test.sig", m)
        stream.ingest([_ev(value=5.0, timestamp=T0)])
        np.testing.assert_array_equal(stream.history(), [-5.0])


# ═══════════════════════════════════════════════════════════════
# Integration: hand-computed end-to-end
# ═══════════════════════════════════════════════════════════════


class TestAtomicSignalIntegration:
    def test_known_z_score_percentile(self):
        """Known values 1-10: verify z-score and percentile for value 7."""
        m = _meta(min_observations=5)
        stream = SignalStream("test.sig", m, window=20)
        evs = [_ev(value=float(i), timestamp=T0 + i * DAY) for i in range(1, 11)]
        stream.ingest(evs)
        result = stream.compute(as_of=T0 + 10 * DAY)

        assert result is not None
        # Latest value: 10.0
        # Values in stats: [1..10], mean=5.5, std(ddof=1) ≈ 3.0277
        expected_z = (10.0 - 5.5) / np.std(np.arange(1, 11, dtype=float), ddof=1)
        assert result.z_score == pytest.approx(expected_z, rel=1e-6)
        # Percentile of 10 in [1..10]: 10/10 = 1.0
        assert result.percentile == pytest.approx(1.0)

    def test_rolling_window_drops_old_values(self):
        """With window=5, only last 5 values used for stats."""
        m = _meta(min_observations=3)
        stream = SignalStream("test.sig", m, window=5)
        evs = [_ev(value=float(i), timestamp=T0 + i * DAY) for i in range(10)]
        stream.ingest(evs)
        result = stream.compute(as_of=T0 + 9 * DAY)

        assert result is not None
        # Window=5: values [5, 6, 7, 8, 9], mean=7.0
        expected_mean = 7.0
        expected_z = (9.0 - expected_mean) / np.std([5, 6, 7, 8, 9], ddof=1)
        assert result.z_score == pytest.approx(expected_z, rel=1e-6)
