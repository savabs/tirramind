"""Tests for RewardFunction — mathematical correctness of reward signals.

The tests prove:
    1. Vol-floor prevents division-by-zero → extrinsic always finite.
    2. CVaR penalty is non-negative and only activates on tail loss.
    3. Intrinsic reward = mean surprise (linear, unbiased).
    4. λ(t) decays linearly from λ₀ to 0 over training.
    5. Combined reward = ext + λ(t) · int (additive decomposition).
"""

import numpy as np
import pytest

from agent.learning.policy.config import RewardConfig
from agent.learning.policy.reward_fn import RewardFunction


@pytest.fixture()
def rf() -> RewardFunction:
    return RewardFunction()


# ── Extrinsic reward ─────────────────────────────────────────


class TestExtrinsic:
    def test_positive_return_positive_reward(self, rf: RewardFunction) -> None:
        """A positive return in a low-vol environment → positive reward."""
        rolling = np.array([0.01, 0.02, -0.005, 0.015, 0.01])
        r = rf.extrinsic(0.02, rolling)
        assert r > 0

    def test_zero_return(self, rf: RewardFunction) -> None:
        rolling = np.array([0.01, -0.01, 0.01, -0.01])
        r = rf.extrinsic(0.0, rolling)
        # Sharpe-normalised 0 return = 0, but CVaR penalty may apply
        # CVaR tail includes the -0.01 values, penalty ≥ 0
        assert np.isfinite(r)

    def test_vol_floor_prevents_inf(self) -> None:
        """With constant returns, vol → 0 → floor kicks in."""
        config = RewardConfig(vol_floor=1e-8)
        rf = RewardFunction(config)
        rolling = np.array([0.01, 0.01, 0.01, 0.01])  # zero vol
        r = rf.extrinsic(0.01, rolling)
        assert np.isfinite(r)
        # 0.01 / 1e-8 = 1e6 (large but finite)
        assert r > 0

    def test_cvar_penalty_activates_on_loss(self) -> None:
        """Large losses → CVaR is negative → penalty shrinks reward."""
        config = RewardConfig(cvar_penalty=2.0)
        rf = RewardFunction(config)
        # Tail is heavily negative
        rolling = np.array([-0.1, -0.08, -0.05, 0.01, 0.02])
        r_with_penalty = rf.extrinsic(0.01, rolling)
        # Without penalty
        config_no_pen = RewardConfig(cvar_penalty=0.0)
        rf_no_pen = RewardFunction(config_no_pen)
        r_no_penalty = rf_no_pen.extrinsic(0.01, rolling)
        assert r_with_penalty < r_no_penalty

    def test_cvar_penalty_zero_on_positive_tail(self, rf: RewardFunction) -> None:
        """When all returns are positive, CVaR > 0 → penalty = 0."""
        rolling = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        # The 5% tail is [0.01], CVaR = 0.01 > 0 → max(0, -0.01) = 0
        r = rf.extrinsic(0.02, rolling)
        # Should equal just the Sharpe-normalised part
        vol = max(float(np.std(rolling)), 1e-8)
        expected_sharpe = 0.02 / vol
        assert abs(r - expected_sharpe) < 1e-6

    def test_empty_rolling(self, rf: RewardFunction) -> None:
        assert rf.extrinsic(0.01, np.array([])) == 0.0

    def test_single_element_rolling(self, rf: RewardFunction) -> None:
        r = rf.extrinsic(0.01, np.array([0.01]))
        assert np.isfinite(r)


# ── Intrinsic reward ─────────────────────────────────────────


class TestIntrinsic:
    def test_mean_surprise(self, rf: RewardFunction) -> None:
        scores = np.array([2.0, 4.0, 6.0])
        assert rf.intrinsic(scores) == pytest.approx(4.0)

    def test_empty_surprise(self, rf: RewardFunction) -> None:
        assert rf.intrinsic(np.array([])) == 0.0

    def test_single_surprise(self, rf: RewardFunction) -> None:
        assert rf.intrinsic(np.array([3.5])) == pytest.approx(3.5)

    def test_zero_surprise(self, rf: RewardFunction) -> None:
        assert rf.intrinsic(np.array([0.0, 0.0])) == 0.0


# ── Lambda decay ─────────────────────────────────────────────


class TestLambdaDecay:
    def test_decay_at_start(self) -> None:
        """step=0 → λ = λ₀."""
        config = RewardConfig(intrinsic_weight_initial=0.1, intrinsic_decay=True)
        rf = RewardFunction(config)
        _, bd = rf.combined(
            0.0, np.array([0.01]), np.array([1.0]), step=0, total_steps=100
        )
        assert bd["lambda_t"] == pytest.approx(0.1)

    def test_decay_at_midpoint(self) -> None:
        """step=T/2 → λ = λ₀/2."""
        config = RewardConfig(intrinsic_weight_initial=0.1, intrinsic_decay=True)
        rf = RewardFunction(config)
        _, bd = rf.combined(
            0.0, np.array([0.01]), np.array([1.0]), step=50, total_steps=100
        )
        assert bd["lambda_t"] == pytest.approx(0.05)

    def test_decay_at_end(self) -> None:
        """step=T → λ = 0."""
        config = RewardConfig(intrinsic_weight_initial=0.1, intrinsic_decay=True)
        rf = RewardFunction(config)
        _, bd = rf.combined(
            0.0, np.array([0.01]), np.array([1.0]), step=100, total_steps=100
        )
        assert bd["lambda_t"] == pytest.approx(0.0)

    def test_no_decay(self) -> None:
        """intrinsic_decay=False → λ = λ₀ always."""
        config = RewardConfig(intrinsic_weight_initial=0.1, intrinsic_decay=False)
        rf = RewardFunction(config)
        _, bd = rf.combined(
            0.0, np.array([0.01]), np.array([1.0]), step=99, total_steps=100
        )
        assert bd["lambda_t"] == pytest.approx(0.1)


# ── Combined reward ──────────────────────────────────────────


class TestCombined:
    def test_additive_decomposition(self) -> None:
        """r = ext + λ(t) · int − adv_penalty — verify the additive structure."""
        rf = RewardFunction(
            RewardConfig(intrinsic_weight_initial=0.5, intrinsic_decay=False)
        )
        rolling = np.array([0.01, -0.02, 0.03, 0.005, -0.01])
        surprise = np.array([2.0, 4.0])

        total, bd = rf.combined(0.01, rolling, surprise, step=0, total_steps=100)
        reconstructed = (
            bd["extrinsic"]
            + bd["lambda_t"] * bd["intrinsic"]
            - bd["adversarial_penalty"]
        )
        assert total == pytest.approx(reconstructed, abs=1e-10)

    def test_combined_empty_surprise(self) -> None:
        rf = RewardFunction()
        total, bd = rf.combined(
            0.01, np.array([0.01, -0.01]), np.array([]), step=0, total_steps=100
        )
        assert total == pytest.approx(bd["extrinsic"])
        assert bd["intrinsic"] == 0.0

    def test_combined_empty_rolling(self) -> None:
        rf = RewardFunction()
        total, bd = rf.combined(
            0.01, np.array([]), np.array([3.0]), step=0, total_steps=100
        )
        # extrinsic returns 0 for empty rolling
        assert bd["extrinsic"] == 0.0
        assert total == pytest.approx(bd["lambda_t"] * bd["intrinsic"])

    def test_breakdown_keys(self, rf: RewardFunction) -> None:
        _, bd = rf.combined(
            0.01, np.array([0.01]), np.array([1.0]), step=0, total_steps=10
        )
        expected_keys = {
            "extrinsic",
            "intrinsic",
            "lambda_t",
            "raw_return",
            "vol",
            "cvar",
            "adversarial_penalty",
        }
        assert set(bd.keys()) == expected_keys

    def test_adversarial_penalty_zero_without_flags(self, rf: RewardFunction) -> None:
        """No adversarial flags → penalty = 0 (backward compatible)."""
        total, bd = rf.combined(
            0.01, np.array([0.01]), np.array([1.0]), step=0, total_steps=10
        )
        assert bd["adversarial_penalty"] == 0.0
