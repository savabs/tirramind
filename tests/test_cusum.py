"""Tests for CUSUM sequential monitor — detection, no false alarms, edge cases."""

from __future__ import annotations

import pytest

from agent.fusion.cusum import CUSUMMonitor

# ── Construction ───────────────────────────────────────────────


class TestCUSUMConstruction:
    def test_default_params(self) -> None:
        mon = CUSUMMonitor()
        assert mon.get_state("any") == 0.0

    def test_custom_params(self) -> None:
        mon = CUSUMMonitor(k=1.0, h=10.0)
        assert mon._k == 1.0
        assert mon._h == 10.0

    def test_negative_k_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be >= 0"):
            CUSUMMonitor(k=-0.1)

    def test_zero_h_raises(self) -> None:
        with pytest.raises(ValueError, match="h must be > 0"):
            CUSUMMonitor(h=0.0)

    def test_negative_h_raises(self) -> None:
        with pytest.raises(ValueError, match="h must be > 0"):
            CUSUMMonitor(h=-1.0)


# ── Detection: persistent mean shift ──────────────────────────


class TestCUSUMDetection:
    def test_2sigma_shift_detected(self) -> None:
        """A 2σ persistent shift should trigger CUSUM within ~10 updates."""
        mon = CUSUMMonitor(k=0.5, h=5.0)
        triggered_at = None
        for i in range(20):
            val, alert = mon.update("ent1", 2.0)  # persistent 2σ shift
            if alert and triggered_at is None:
                triggered_at = i
        assert triggered_at is not None
        assert triggered_at <= 10  # should trigger well within 10 steps

    def test_1sigma_shift_detected_eventually(self) -> None:
        """A 1σ shift accumulates more slowly but still triggers."""
        mon = CUSUMMonitor(k=0.5, h=5.0)
        triggered = False
        for i in range(30):
            _, alert = mon.update("ent1", 1.0)
            if alert:
                triggered = True
                break
        assert triggered

    def test_increasing_cusum_under_shift(self) -> None:
        """CUSUM should monotonically increase under persistent positive shift."""
        mon = CUSUMMonitor(k=0.5, h=5.0)
        prev = 0.0
        for _ in range(10):
            val, _ = mon.update("ent1", 2.0)
            assert val >= prev
            prev = val


# ── No false alarm on noise ────────────────────────────────────


class TestCUSUMNoFalseAlarm:
    def test_zero_input_no_alert(self) -> None:
        """Zero input (mean = 0) should never trigger."""
        mon = CUSUMMonitor(k=0.5, h=5.0)
        for _ in range(200):
            _, alert = mon.update("ent1", 0.0)
            assert not alert

    def test_negative_input_no_alert(self) -> None:
        """Negative input keeps CUSUM at 0 (one-sided upper)."""
        mon = CUSUMMonitor(k=0.5, h=5.0)
        for _ in range(100):
            val, alert = mon.update("ent1", -1.0)
            assert val == 0.0
            assert not alert

    def test_below_allowance_no_alert(self) -> None:
        """Input just below allowance k should not accumulate."""
        mon = CUSUMMonitor(k=0.5, h=5.0)
        for _ in range(1000):
            val, alert = mon.update("ent1", 0.49)
            assert not alert
        assert mon.get_state("ent1") == 0.0


# ── Reset behavior ─────────────────────────────────────────────


class TestCUSUMReset:
    def test_reset_clears_state(self) -> None:
        mon = CUSUMMonitor()
        mon.update("ent1", 3.0)
        mon.update("ent1", 3.0)
        assert mon.get_state("ent1") > 0
        mon.reset("ent1")
        assert mon.get_state("ent1") == 0.0

    def test_reset_unknown_entity(self) -> None:
        mon = CUSUMMonitor()
        mon.reset("unknown")  # should not raise
        assert mon.get_state("unknown") == 0.0

    def test_reaccumulation_after_reset(self) -> None:
        mon = CUSUMMonitor(k=0.5, h=5.0)
        for _ in range(10):
            mon.update("ent1", 3.0)
        mon.reset("ent1")
        val, _ = mon.update("ent1", 3.0)
        assert val == 2.5  # max(0, 0 + 3.0 - 0.5)


# ── Multiple entities ──────────────────────────────────────────


class TestCUSUMMultipleEntities:
    def test_independent_entities(self) -> None:
        mon = CUSUMMonitor()
        mon.update("a", 3.0)
        mon.update("b", 0.0)
        assert mon.get_state("a") > 0
        assert mon.get_state("b") == 0.0

    def test_get_all_states(self) -> None:
        mon = CUSUMMonitor()
        mon.update("a", 1.0)
        mon.update("b", 2.0)
        states = mon.get_all_states()
        assert "a" in states
        assert "b" in states
        assert len(states) == 2

    def test_get_all_states_snapshot(self) -> None:
        """get_all_states returns a copy, not a reference."""
        mon = CUSUMMonitor()
        mon.update("a", 1.0)
        snap = mon.get_all_states()
        mon.update("a", 5.0)
        assert snap["a"] != mon.get_state("a")


# ── Edge cases: extreme values ─────────────────────────────────


class TestCUSUMEdgeCases:
    def test_very_large_z_score(self) -> None:
        """Cap prevents float overflow."""
        mon = CUSUMMonitor(k=0.5, h=5.0, cap_multiplier=10.0)
        for _ in range(100):
            val, _ = mon.update("ent1", 1e6)
        assert val <= 5.0 * 10.0  # capped at h * cap_multiplier

    def test_zero_z_score_stream(self) -> None:
        mon = CUSUMMonitor()
        for _ in range(100):
            val, alert = mon.update("ent1", 0.0)
        assert val == 0.0
        assert not alert

    def test_alternating_positive_negative(self) -> None:
        """Alternating +1, -1 should not accumulate (resets to 0 each cycle)."""
        mon = CUSUMMonitor(k=0.5, h=5.0)
        for _ in range(100):
            mon.update("ent1", 1.0)
            mon.update("ent1", -1.0)
        assert mon.get_state("ent1") == 0.0

    def test_unknown_entity_state(self) -> None:
        mon = CUSUMMonitor()
        assert mon.get_state("nonexistent") == 0.0
