"""Tests for EntityBaseline — rolling baseline, abnormal scoring, edge cases."""

from __future__ import annotations

import pytest

from agent.fusion.entity_baseline import EntityBaseline

# ── Construction ───────────────────────────────────────────────


class TestEntityBaselineConstruction:
    def test_default_params(self) -> None:
        b = EntityBaseline()
        assert b.observation_count("any") == 0

    def test_custom_params(self) -> None:
        b = EntityBaseline(window=50, gap=10, min_obs=15)
        assert b._window == 50

    def test_zero_window_raises(self) -> None:
        with pytest.raises(ValueError, match="Window must be >= 1"):
            EntityBaseline(window=0)

    def test_negative_gap_raises(self) -> None:
        with pytest.raises(ValueError, match="Gap must be >= 0"):
            EntityBaseline(gap=-1)

    def test_min_obs_one_raises(self) -> None:
        with pytest.raises(ValueError, match="min_obs must be >= 2"):
            EntityBaseline(min_obs=1)

    def test_min_obs_greater_than_window_raises(self) -> None:
        with pytest.raises(ValueError, match="min_obs.*> window"):
            EntityBaseline(window=5, min_obs=10)


# ── Known distribution ─────────────────────────────────────────


class TestEntityBaselineKnownDistribution:
    def test_3sigma_event_detected(self) -> None:
        """Feed N(0,1) baseline, then inject 3σ event → AS ≈ 3.0."""
        b = EntityBaseline(window=30, gap=0, min_obs=10)
        # Simulate N(0,1)-like values: use fixed values with mean≈0, std≈1
        baseline_vals = [
            0.1,
            -0.2,
            0.3,
            -0.1,
            0.05,
            -0.3,
            0.2,
            0.15,
            -0.05,
            0.1,
            -0.15,
            0.25,
            -0.25,
            0.0,
            0.1,
        ]
        for v in baseline_vals:
            b.add_observation("ent1", v)
        score = b.abnormal_score("ent1", 3.0)
        assert score is not None
        assert score > 2.0  # should be well above 2σ

    def test_normal_event_low_score(self) -> None:
        """Event at the mean should have score ≈ 0."""
        b = EntityBaseline(window=30, gap=0, min_obs=10)
        for v in [1.0] * 20:
            b.add_observation("ent1", v)
        score = b.abnormal_score("ent1", 1.0)
        assert score is not None
        assert abs(score) < 0.01

    def test_below_baseline_negative_score(self) -> None:
        """Event below baseline → negative abnormal score."""
        b = EntityBaseline(window=30, gap=0, min_obs=10)
        for v in [5.0] * 15:
            b.add_observation("ent1", v)
        # Inject slight variance
        b.add_observation("ent1", 5.1)
        b.add_observation("ent1", 4.9)
        score = b.abnormal_score("ent1", 2.0)
        assert score is not None
        assert score < 0


# ── Insufficient history ───────────────────────────────────────


class TestEntityBaselineInsufficientHistory:
    def test_no_observations(self) -> None:
        b = EntityBaseline(min_obs=10)
        assert b.abnormal_score("ent1", 5.0) is None

    def test_few_observations(self) -> None:
        b = EntityBaseline(min_obs=10, gap=0)
        for i in range(9):
            b.add_observation("ent1", float(i))
        assert b.abnormal_score("ent1", 5.0) is None

    def test_exactly_min_obs(self) -> None:
        b = EntityBaseline(min_obs=10, gap=0, window=30)
        for i in range(10):
            b.add_observation("ent1", float(i))
        score = b.abnormal_score("ent1", 100.0)
        assert score is not None

    def test_gap_eats_into_estimation(self) -> None:
        """With gap=5, need min_obs + gap observations total."""
        b = EntityBaseline(window=30, gap=5, min_obs=10)
        for i in range(14):
            b.add_observation("ent1", float(i))
        assert b.abnormal_score("ent1", 100.0) is None  # 14 - 5 = 9 < 10

        b.add_observation("ent1", 1.0)  # now 15 - 5 = 10 >= 10
        score = b.abnormal_score("ent1", 100.0)
        assert score is not None


# ── Gap window ─────────────────────────────────────────────────


class TestEntityBaselineGap:
    def test_gap_excludes_recent(self) -> None:
        """Gap window should exclude most recent observations from baseline."""
        b = EntityBaseline(window=30, gap=5, min_obs=10)
        # 15 normal observations
        for _ in range(15):
            b.add_observation("ent1", 1.0)
        # 5 extreme observations (in gap window, should be excluded)
        for _ in range(5):
            b.add_observation("ent1", 100.0)
        score = b.abnormal_score("ent1", 1.0)
        assert score is not None
        # Score should be near 0 because the estimation window only sees 1.0 values
        assert abs(score) < 0.01

    def test_zero_gap(self) -> None:
        """With gap=0, all observations used in estimation."""
        b = EntityBaseline(window=30, gap=0, min_obs=2)
        b.add_observation("ent1", 1.0)
        b.add_observation("ent1", 1.0)
        score = b.abnormal_score("ent1", 5.0)
        assert score is not None


# ── Sliding window ─────────────────────────────────────────────


class TestEntityBaselineSlidingWindow:
    def test_window_slides_over_time(self) -> None:
        """Old observations should fall out of the window."""
        b = EntityBaseline(window=5, gap=0, min_obs=3)
        # Fill with 1.0
        for _ in range(10):
            b.add_observation("ent1", 1.0)
        # Now fill with 10.0 to replace the window
        for _ in range(10):
            b.add_observation("ent1", 10.0)
        score = b.abnormal_score("ent1", 10.0)
        assert score is not None
        assert abs(score) < 0.01  # 10.0 is the new normal


# ── Cumulative abnormal score ──────────────────────────────────


class TestEntityBaselineCumulative:
    def test_car_single_event(self) -> None:
        b = EntityBaseline(window=30, gap=0, min_obs=2)
        for _ in range(10):
            b.add_observation("ent1", 0.0)
        b.add_observation("ent1", 0.1)  # add slight variance
        car = b.cumulative_abnormal_score("ent1", [3.0])
        single = b.abnormal_score("ent1", 3.0)
        assert car is not None and single is not None
        assert car == pytest.approx(single)

    def test_car_multiple_events(self) -> None:
        b = EntityBaseline(window=30, gap=0, min_obs=2)
        for _ in range(10):
            b.add_observation("ent1", 0.0)
        b.add_observation("ent1", 0.1)
        car = b.cumulative_abnormal_score("ent1", [3.0, 3.0, 3.0])
        single = b.abnormal_score("ent1", 3.0)
        assert car is not None and single is not None
        assert car == pytest.approx(3.0 * single)

    def test_car_empty_values(self) -> None:
        b = EntityBaseline()
        car = b.cumulative_abnormal_score("ent1", [])
        assert car == 0.0

    def test_car_insufficient_history(self) -> None:
        b = EntityBaseline(min_obs=10)
        assert b.cumulative_abnormal_score("ent1", [1.0, 2.0]) is None


# ── Edge cases ─────────────────────────────────────────────────


class TestEntityBaselineEdgeCases:
    def test_constant_baseline_zero_score(self) -> None:
        """Constant series → σ=0 → score=0."""
        b = EntityBaseline(window=30, gap=0, min_obs=10)
        for _ in range(20):
            b.add_observation("ent1", 5.0)
        score = b.abnormal_score("ent1", 100.0)
        assert score == 0.0  # constant → no measurable abnormality

    def test_nan_observation_skipped(self) -> None:
        b = EntityBaseline(window=30, gap=0, min_obs=2)
        b.add_observation("ent1", 1.0)
        b.add_observation("ent1", float("nan"))
        b.add_observation("ent1", 2.0)
        assert b.observation_count("ent1") == 2  # NaN was skipped

    def test_inf_observation_skipped(self) -> None:
        b = EntityBaseline(window=30, gap=0, min_obs=2)
        b.add_observation("ent1", 1.0)
        b.add_observation("ent1", float("inf"))
        assert b.observation_count("ent1") == 1

    def test_multiple_entities_independent(self) -> None:
        b = EntityBaseline(window=30, gap=0, min_obs=2)
        for _ in range(10):
            b.add_observation("a", 1.0)
            b.add_observation("b", 100.0)
        b.add_observation("a", 1.1)
        b.add_observation("b", 100.1)
        assert b.observation_count("a") == 11
        assert b.observation_count("b") == 11

    def test_unknown_entity_count(self) -> None:
        b = EntityBaseline()
        assert b.observation_count("nonexistent") == 0
