"""Tests for EdgeDecayMonitor — BOCPD on rolling Sharpe."""

from __future__ import annotations

import numpy as np
import pytest

from agent.adversarial.config import EdgeDecayConfig
from agent.adversarial.edge_decay import EdgeDecayMonitor


@pytest.fixture
def monitor() -> EdgeDecayMonitor:
    return EdgeDecayMonitor(
        EdgeDecayConfig(
            rolling_window=20,
            bocpd_hazard_lambda=50.0,
            decay_threshold=0.3,
            min_history=20,
            periods_per_year=52,
        )
    )


class TestEdgeDecayBasic:
    """Core functionality tests."""

    def test_constant_returns_no_flag(self, monitor: EdgeDecayMonitor):
        """Constant positive returns → stable Sharpe → no decay flag."""
        rng = np.random.default_rng(42)
        # Strong positive signal with noise
        returns = 0.01 + 0.002 * rng.standard_normal(200)
        flags = monitor.update("test_signal", returns)
        assert len(flags) == 0

    def test_decaying_signal_produces_flag(self):
        """Signal whose Sharpe drops dramatically at a known point → flag."""
        monitor = EdgeDecayMonitor(
            EdgeDecayConfig(
                rolling_window=20,
                bocpd_hazard_lambda=30.0,  # shorter expected run length
                decay_threshold=0.1,  # low threshold → sensitive detection
                min_history=20,
                periods_per_year=52,
            )
        )
        rng = np.random.default_rng(123)
        n = 300
        returns = np.empty(n)
        # First 150: very strong signal (high Sharpe)
        returns[:150] = 0.04 + 0.003 * rng.standard_normal(150)
        # Last 150: negative signal (Sharpe flips sign)
        returns[150:] = -0.02 + 0.003 * rng.standard_normal(150)

        flags = monitor.update("decaying_signal", returns, timestamp=1000.0)
        # Should detect decay
        assert len(flags) >= 1
        f = flags[0]
        assert f.flag_type == "edge_decay"
        assert f.signal_name == "decaying_signal"
        assert f.severity > 0
        assert f.timestamp == 1000.0

    def test_insufficient_history_skipped(self, monitor: EdgeDecayMonitor):
        """Fewer than min_history observations → no flag."""
        returns = np.array([0.01] * 10)
        flags = monitor.update("short_signal", returns)
        assert flags == []

    def test_exactly_min_history(self, monitor: EdgeDecayMonitor):
        """Exactly min_history observations → should still work (1 Sharpe value)."""
        rng = np.random.default_rng(42)
        returns = 0.01 + 0.002 * rng.standard_normal(20)
        # Only 1 Sharpe value → won't have enough for BOCPD (need ≥ 2)
        flags = monitor.update("min_signal", returns)
        # Should not crash, may or may not flag
        assert isinstance(flags, list)


class TestRollingSharpe:
    """Tests for the rolling Sharpe calculation."""

    def test_all_zeros(self, monitor: EdgeDecayMonitor):
        """All-zero returns → Sharpe = 0 everywhere."""
        returns = np.zeros(100)
        sharpes = monitor.rolling_sharpe(returns)
        assert len(sharpes) == 100 - 20 + 1
        np.testing.assert_allclose(sharpes, 0.0, atol=1e-6)

    def test_constant_positive_mean(self, monitor: EdgeDecayMonitor):
        """Constant positive returns → positive Sharpe."""
        returns = np.full(100, 0.01)
        sharpes = monitor.rolling_sharpe(returns)
        # std ≈ 0 (floored at eps) → Sharpe should be very large
        assert np.all(sharpes > 0)

    def test_shape_is_correct(self, monitor: EdgeDecayMonitor):
        """Output length = len(returns) - window + 1."""
        n = 150
        returns = np.random.default_rng(99).standard_normal(n)
        sharpes = monitor.rolling_sharpe(returns)
        assert len(sharpes) == n - 20 + 1

    def test_short_series(self, monitor: EdgeDecayMonitor):
        """Series shorter than window → empty."""
        returns = np.array([0.01] * 10)
        sharpes = monitor.rolling_sharpe(returns)
        assert len(sharpes) == 0


class TestDecayScoreBatch:
    """Tests for get_decay_scores (batch mode)."""

    def test_multiple_signals(self, monitor: EdgeDecayMonitor):
        rng = np.random.default_rng(42)
        signals = {
            "stable": 0.01 + 0.002 * rng.standard_normal(200),
            "short": rng.standard_normal(10),  # too short
        }
        scores = monitor.get_decay_scores(signals)
        assert "stable" in scores
        assert "short" not in scores  # insufficient history
        assert 0.0 <= scores["stable"] <= 1.0

    def test_empty_dict(self, monitor: EdgeDecayMonitor):
        scores = monitor.get_decay_scores({})
        assert scores == {}


class TestEdgeDecayEdgeCases:
    """Edge cases and boundary conditions."""

    def test_nan_in_returns(self, monitor: EdgeDecayMonitor):
        """NaN in returns should not crash (propagates through Sharpe)."""
        returns = np.ones(100) * 0.01
        returns[50] = np.nan
        # Should not raise — NaN propagates into Sharpe but BOCPD may still run
        # The key thing is no unhandled exception
        try:
            flags = monitor.update("nan_signal", returns)
            assert isinstance(flags, list)
        except (ValueError, FloatingPointError):
            pass  # acceptable to raise on NaN input

    def test_inf_in_returns(self, monitor: EdgeDecayMonitor):
        """Inf in returns handled gracefully."""
        returns = np.ones(100) * 0.01
        returns[50] = np.inf
        try:
            flags = monitor.update("inf_signal", returns)
            assert isinstance(flags, list)
        except (ValueError, FloatingPointError, OverflowError):
            pass

    def test_very_long_series(self):
        """500+ periods — performance and correctness."""
        monitor = EdgeDecayMonitor(
            EdgeDecayConfig(
                rolling_window=20,
                min_history=20,
                bocpd_hazard_lambda=50.0,
            )
        )
        rng = np.random.default_rng(42)
        returns = 0.005 + 0.01 * rng.standard_normal(500)
        flags = monitor.update("long_signal", returns)
        assert isinstance(flags, list)

    def test_all_negative_returns(self, monitor: EdgeDecayMonitor):
        """All returns negative → Sharpe negative → no crash."""
        rng = np.random.default_rng(42)
        returns = -0.01 + 0.002 * rng.standard_normal(200)
        flags = monitor.update("losing_signal", returns)
        assert isinstance(flags, list)

    def test_regime_switch_vs_decay(self):
        """Regime switch (up→down→up) differs from permanent decay."""
        monitor = EdgeDecayMonitor(
            EdgeDecayConfig(
                rolling_window=20,
                bocpd_hazard_lambda=50.0,
                decay_threshold=0.3,
                min_history=20,
            )
        )
        rng = np.random.default_rng(42)
        n = 300
        returns = np.empty(n)
        returns[:100] = 0.02 + 0.005 * rng.standard_normal(100)
        returns[100:200] = 0.0 + 0.005 * rng.standard_normal(100)
        returns[200:] = 0.02 + 0.005 * rng.standard_normal(100)
        flags = monitor.update("regime_switch", returns)
        # With recovery, the monitor may or may not flag depending on
        # the recency of the down regime. This just ensures no crash.
        assert isinstance(flags, list)
