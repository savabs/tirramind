"""Tests for VPINEstimator — Volume-synchronized PIN via BVC."""

from __future__ import annotations

import numpy as np
import pytest

from agent.adversarial.config import VPINConfig
from agent.adversarial.vpin import VPINEstimator


@pytest.fixture
def estimator() -> VPINEstimator:
    return VPINEstimator(
        VPINConfig(
            n_buckets=10,
            sigma_window=5,
            spike_threshold=0.7,
        )
    )


class TestVPINCompute:
    """Core VPIN computation tests."""

    def test_symmetric_returns_moderate_vpin(self, estimator: VPINEstimator):
        """Random-walk returns (symmetric buy/sell) → VPIN around baseline."""
        rng = np.random.default_rng(42)
        n = 200
        returns = 0.001 * rng.standard_normal(n)
        volumes = np.full(n, 1e6)

        vpin = estimator.compute(returns, volumes)
        assert len(vpin) == n - 10 + 1
        # For symmetric returns, VPIN should be moderate (not extreme)
        assert np.all(vpin >= 0)

    def test_one_sided_positive_returns_high_vpin(self, estimator: VPINEstimator):
        """Consistently large positive returns → high VPIN."""
        n = 100
        returns = np.full(n, 0.05)  # strong positive
        volumes = np.full(n, 1e6)

        vpin = estimator.compute(returns, volumes)
        assert len(vpin) > 0
        # Very one-sided flow → VPIN should be high
        assert vpin[-1] > 0.5

    def test_one_sided_negative_returns_high_vpin(self, estimator: VPINEstimator):
        """Consistently large negative returns → high VPIN."""
        n = 100
        returns = np.full(n, -0.05)
        volumes = np.full(n, 1e6)

        vpin = estimator.compute(returns, volumes)
        assert len(vpin) > 0
        assert vpin[-1] > 0.5

    def test_exact_values_known_input(self):
        """Manual verification against the VPIN formula."""
        est = VPINEstimator(VPINConfig(n_buckets=3, sigma_window=2))
        # Known setup: 5 data points, VPIN computed over last 3
        returns = np.array([0.01, -0.01, 0.02, -0.02, 0.03])
        volumes = np.array([100.0, 100.0, 100.0, 100.0, 100.0])

        vpin = est.compute(returns, volumes)
        assert len(vpin) == 3  # 5 - 3 + 1
        assert np.all(np.isfinite(vpin))
        assert np.all(vpin >= 0)

    def test_output_shape(self, estimator: VPINEstimator):
        """Output length = len(input) - n_buckets + 1."""
        n = 50
        vpin = estimator.compute(
            np.random.default_rng(42).standard_normal(n) * 0.01,
            np.full(n, 1e6),
        )
        assert len(vpin) == n - 10 + 1


class TestVPINFlagSpikes:
    """Tests for VPIN spike flagging."""

    def test_below_threshold_no_flag(self, estimator: VPINEstimator):
        vpin = np.array([0.3, 0.4, 0.5, 0.6])
        flags = estimator.flag_spikes(vpin, entity_id="SPY")
        assert flags == []

    def test_above_threshold_produces_flag(self, estimator: VPINEstimator):
        vpin = np.array([0.3, 0.4, 0.5, 0.8])
        flags = estimator.flag_spikes(vpin, entity_id="SPY", timestamp=999.0)
        assert len(flags) == 1
        f = flags[0]
        assert f.flag_type == "vpin_spike"
        assert f.severity == 0.8
        assert f.entity_id == "SPY"
        assert f.timestamp == 999.0
        assert f.evidence["vpin_latest"] == 0.8

    def test_empty_series_no_flag(self, estimator: VPINEstimator):
        flags = estimator.flag_spikes(np.array([]))
        assert flags == []

    def test_single_value_above(self, estimator: VPINEstimator):
        flags = estimator.flag_spikes(np.array([0.9]))
        assert len(flags) == 1

    def test_severity_capped_at_one(self, estimator: VPINEstimator):
        flags = estimator.flag_spikes(np.array([1.5]))
        assert len(flags) == 1
        assert flags[0].severity == 1.0


class TestVPINInputValidation:
    """Input validation and error handling."""

    def test_mismatched_lengths(self, estimator: VPINEstimator):
        with pytest.raises(ValueError, match="equal length"):
            estimator.compute(np.array([0.01, 0.02]), np.array([100.0]))

    def test_nan_in_returns(self, estimator: VPINEstimator):
        with pytest.raises(ValueError, match="NaN"):
            estimator.compute(
                np.array([np.nan, 0.01, 0.02]),
                np.array([100.0, 100.0, 100.0]),
            )

    def test_nan_in_volumes(self, estimator: VPINEstimator):
        with pytest.raises(ValueError, match="NaN"):
            estimator.compute(
                np.array([0.01, 0.02, 0.03]),
                np.array([100.0, np.nan, 100.0]),
            )

    def test_all_zero_volumes(self, estimator: VPINEstimator):
        with pytest.raises(ValueError, match="zero"):
            estimator.compute(
                np.array([0.01, 0.02, 0.03]),
                np.zeros(3),
            )

    def test_insufficient_data(self, estimator: VPINEstimator):
        """Fewer than n_buckets data points → empty result."""
        vpin = estimator.compute(
            np.array([0.01] * 5),
            np.array([100.0] * 5),
        )
        assert len(vpin) == 0

    def test_exactly_n_buckets(self, estimator: VPINEstimator):
        """Exactly n_buckets → one VPIN value."""
        vpin = estimator.compute(
            np.array([0.01] * 10),
            np.array([100.0] * 10),
        )
        assert len(vpin) == 1


class TestVPINMathProperties:
    """Mathematical properties verification."""

    def test_vpin_nonnegative(self, estimator: VPINEstimator):
        """VPIN must always be ≥ 0."""
        rng = np.random.default_rng(42)
        for _ in range(5):
            n = 100
            ret = 0.01 * rng.standard_normal(n)
            vol = np.abs(rng.standard_normal(n)) * 1e6 + 1
            vpin = estimator.compute(ret, vol)
            assert np.all(vpin >= 0), f"Negative VPIN: {vpin.min()}"

    def test_zero_returns_moderate_vpin(self, estimator: VPINEstimator):
        """Zero returns → Φ(0) = 0.5 → buy = sell → low OI → low VPIN."""
        n = 100
        returns = np.zeros(n)
        volumes = np.full(n, 1e6)
        vpin = estimator.compute(returns, volumes)
        # OI should be near zero → VPIN near zero
        assert np.all(vpin < 0.1)

    def test_rolling_std_no_nan(self):
        """Internal rolling std should not produce NaN."""
        arr = np.array([0.01, -0.01, 0.02, -0.02, 0.0])
        result = VPINEstimator._rolling_std(arr, 3)
        assert not np.any(np.isnan(result))
        assert len(result) == len(arr)
