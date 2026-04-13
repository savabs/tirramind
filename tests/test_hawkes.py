"""Tests for Hawkes process intensity estimator — spikes, decay, bursts, edge cases."""

from __future__ import annotations

import math

import pytest

from agent.fusion.hawkes import HawkesIntensity

# ── Construction ───────────────────────────────────────────────


class TestHawkesConstruction:
    def test_default_params(self) -> None:
        h = HawkesIntensity()
        assert h.intensity_at("any", 0.0) == 0.1  # baseline mu

    def test_custom_params(self) -> None:
        h = HawkesIntensity(mu=0.2, alpha=0.3, beta=0.5)
        assert h._mu == 0.2

    def test_negative_mu_raises(self) -> None:
        with pytest.raises(ValueError, match="mu must be >= 0"):
            HawkesIntensity(mu=-0.1)

    def test_negative_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha must be >= 0"):
            HawkesIntensity(alpha=-0.1)

    def test_zero_beta_raises(self) -> None:
        with pytest.raises(ValueError, match="beta must be > 0"):
            HawkesIntensity(beta=0.0)

    def test_supercritical_raises(self) -> None:
        with pytest.raises(ValueError, match="Branching ratio"):
            HawkesIntensity(alpha=1.5, beta=1.0)  # ratio = 1.5 >= 1

    def test_critical_raises(self) -> None:
        with pytest.raises(ValueError, match="Branching ratio"):
            HawkesIntensity(alpha=1.0, beta=1.0)  # ratio = 1.0 >= 1


# ── Single event spike and decay ───────────────────────────────


class TestHawkesSingleEvent:
    def test_first_event_returns_mu(self) -> None:
        """First event has no prior self-excitation, so λ(t₁) = μ + α·0 = μ."""
        h = HawkesIntensity(mu=0.1, alpha=0.5, beta=1.0)
        intensity = h.update("ent1", 100.0)
        assert intensity == pytest.approx(0.1)

    def test_second_event_spike(self) -> None:
        """Second event shortly after first should spike due to excitation."""
        h = HawkesIntensity(mu=0.1, alpha=0.5, beta=1.0)
        h.update("ent1", 100.0)
        # dt = 0.1, A = exp(-1.0*0.1) * (0 + 1) = 0.9048...
        intensity = h.update("ent1", 100.1)
        expected_a = math.exp(-1.0 * 0.1) * 1.0
        expected = 0.1 + 0.5 * expected_a
        assert intensity == pytest.approx(expected, rel=1e-6)

    def test_exponential_decay(self) -> None:
        """Intensity should decay exponentially after a single event."""
        h = HawkesIntensity(mu=0.1, alpha=0.5, beta=1.0)
        h.update("ent1", 0.0)
        # Query at increasing time offsets
        prev = h.intensity_at("ent1", 0.01)
        for dt in [0.5, 1.0, 2.0, 5.0, 10.0]:
            curr = h.intensity_at("ent1", dt)
            assert curr < prev or (curr == pytest.approx(0.1, abs=0.001))
            prev = curr

    def test_returns_to_baseline(self) -> None:
        """After long gap, intensity → μ."""
        h = HawkesIntensity(mu=0.1, alpha=0.5, beta=1.0)
        h.update("ent1", 0.0)
        intensity = h.intensity_at("ent1", 100.0)
        assert intensity == pytest.approx(0.1, abs=1e-6)


# ── Burst detection ────────────────────────────────────────────


class TestHawkesBurst:
    def test_rapid_burst_high_intensity(self) -> None:
        """5 rapid events should produce high intensity."""
        h = HawkesIntensity(mu=0.1, alpha=0.5, beta=1.0)
        for i in range(5):
            intensity = h.update("ent1", float(i) * 0.1)
        assert intensity > 0.1 * 5  # well above baseline

    def test_burst_then_decay(self) -> None:
        """After burst, intensity should decay back toward baseline."""
        h = HawkesIntensity(mu=0.1, alpha=0.5, beta=1.0)
        for i in range(5):
            h.update("ent1", float(i) * 0.1)
        peak = h.intensity_at("ent1", 0.5)
        after = h.intensity_at("ent1", 10.0)
        assert after < peak
        assert after == pytest.approx(0.1, abs=0.01)


# ── Multiple entities ──────────────────────────────────────────


class TestHawkesMultipleEntities:
    def test_independent_entities(self) -> None:
        h = HawkesIntensity()
        h.update("a", 0.0)
        h.update("a", 0.1)
        h.update("b", 0.0)  # only 1 event for b
        assert h.intensity_at("a", 0.2) > h.intensity_at("b", 0.2)

    def test_get_all_entities(self) -> None:
        h = HawkesIntensity()
        h.update("x", 0.0)
        h.update("y", 1.0)
        assert set(h.get_all_entities()) == {"x", "y"}

    def test_get_state(self) -> None:
        h = HawkesIntensity()
        assert h.get_state("missing") is None
        h.update("a", 5.0)
        state = h.get_state("a")
        assert state is not None
        assert state[0] == 5.0


# ── Time ordering ──────────────────────────────────────────────


class TestHawkesTimeOrdering:
    def test_backward_time_raises(self) -> None:
        h = HawkesIntensity()
        h.update("ent1", 10.0)
        with pytest.raises(ValueError, match="monotonically non-decreasing"):
            h.update("ent1", 9.0)

    def test_same_time_ok(self) -> None:
        """Simultaneous events are valid."""
        h = HawkesIntensity()
        h.update("ent1", 10.0)
        intensity = h.update("ent1", 10.0)  # dt = 0
        assert intensity >= 0.1

    def test_query_before_last_event_raises(self) -> None:
        h = HawkesIntensity()
        h.update("ent1", 10.0)
        with pytest.raises(ValueError, match="Query time"):
            h.intensity_at("ent1", 5.0)


# ── Edge cases ─────────────────────────────────────────────────


class TestHawkesEdgeCases:
    def test_unknown_entity_intensity(self) -> None:
        h = HawkesIntensity(mu=0.1)
        assert h.intensity_at("unknown", 999.0) == 0.1

    def test_very_large_time_gap(self) -> None:
        """Large time gap should not cause overflow."""
        h = HawkesIntensity()
        h.update("ent1", 0.0)
        intensity = h.intensity_at("ent1", 1e12)
        assert math.isfinite(intensity)
        assert intensity == pytest.approx(0.1, abs=1e-6)

    def test_many_events_no_overflow(self) -> None:
        """1000 rapid events should not overflow."""
        h = HawkesIntensity(mu=0.1, alpha=0.3, beta=1.0)
        for i in range(1000):
            intensity = h.update("ent1", float(i) * 0.01)
        assert math.isfinite(intensity)

    def test_zero_alpha_constant_baseline(self) -> None:
        """With alpha=0, intensity is always mu regardless of events."""
        h = HawkesIntensity(mu=0.5, alpha=0.0, beta=1.0)
        h.update("ent1", 0.0)
        h.update("ent1", 0.01)
        h.update("ent1", 0.02)
        assert h.intensity_at("ent1", 0.03) == pytest.approx(0.5)
