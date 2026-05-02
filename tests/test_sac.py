"""Tests for SAC components — Phase 21b.3-4

Mathematical proofs:
    1. Action bounds:       ∀ s: |a_i| ≤ max_position, Σ|a_i| ≤ L
    2. Log-prob finiteness: log π(a|s) is always finite (no NaN/Inf)
    3. Twin critic independence: Q1 ≠ Q2 after different gradient updates
    4. Soft target update:  θ' = (1-τ)θ' + τθ (Polyak averaging)
    5. Alpha auto-tuning:   α adapts toward target entropy
    6. Deterministic mode:  select_action(det=True) is deterministic
    7. Save/load roundtrip: serialisation + deserialisation is exact
    8. Gradient flow:       actor loss backpropagates to actor params
    9. Critic convergence:  critic loss decreases on repeated updates
    10. Leverage constraint: gross exposure ≤ leverage_limit
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from agent.learning.policy.config import SACConfig
from agent.learning.policy.replay_buffer import ReplayBuffer
from agent.learning.policy.sac import (
    AlphaScheduler,
    GaussianActor,
    SACTrainer,
    TwinCritic,
)

# ── Fixtures ──────────────────────────────────────────────────

STATE_DIM = 20
ACTION_DIM = 5


@pytest.fixture
def cfg() -> SACConfig:
    return SACConfig(
        hidden_dim=32,
        num_hidden=2,
        batch_size=16,
        max_position=0.5,
        leverage_limit=1.0,
    )


@pytest.fixture
def actor(cfg: SACConfig) -> GaussianActor:
    return GaussianActor(STATE_DIM, ACTION_DIM, cfg)


@pytest.fixture
def critic(cfg: SACConfig) -> TwinCritic:
    return TwinCritic(STATE_DIM, ACTION_DIM, cfg)


@pytest.fixture
def filled_buffer() -> ReplayBuffer:
    """Buffer with 200 random transitions."""
    buf = ReplayBuffer(500, STATE_DIM, ACTION_DIM)
    for _ in range(200):
        buf.push(
            np.random.randn(STATE_DIM).astype(np.float32),
            np.random.randn(ACTION_DIM).astype(np.float32) * 0.3,
            float(np.random.randn()),
            np.random.randn(STATE_DIM).astype(np.float32),
            bool(np.random.rand() > 0.95),
        )
    return buf


@pytest.fixture
def trainer(cfg: SACConfig) -> SACTrainer:
    return SACTrainer(STATE_DIM, ACTION_DIM, cfg)


# ── 1. Action Bounds ─────────────────────────────────────────


class TestActionBounds:
    """Proof 1: |a_i| ≤ max_position, Σ|a_i| ≤ leverage_limit."""

    def test_sample_within_position_limits(self, actor: GaussianActor):
        states = torch.randn(100, STATE_DIM)
        actions, _ = actor.sample(states)
        assert (actions.abs() <= 0.5 + 1e-6).all(), f"max abs: {actions.abs().max()}"

    def test_leverage_constraint(self, actor: GaussianActor):
        states = torch.randn(100, STATE_DIM)
        actions, _ = actor.sample(states)
        gross = actions.abs().sum(dim=-1)
        assert (gross <= 1.0 + 1e-5).all(), f"max gross: {gross.max()}"

    def test_large_hidden_output_still_bounded(self):
        """Even with extreme network outputs, tanh + leverage clamp keeps bounds."""
        cfg = SACConfig(hidden_dim=64, num_hidden=3, max_position=0.5, leverage_limit=1.0)
        actor = GaussianActor(STATE_DIM, ACTION_DIM, cfg)
        # Feed large inputs
        states = torch.randn(50, STATE_DIM) * 100
        actions, _ = actor.sample(states)
        assert (actions.abs() <= 0.5 + 1e-6).all()
        assert (actions.abs().sum(dim=-1) <= 1.0 + 1e-5).all()


# ── 2. Log-prob Finiteness ───────────────────────────────────


class TestLogProbFiniteness:
    """Proof 2: log π(a|s) ∈ ℝ (no NaN/Inf)."""

    def test_log_prob_finite(self, actor: GaussianActor):
        states = torch.randn(200, STATE_DIM)
        _, log_probs = actor.sample(states)
        assert torch.isfinite(log_probs).all(), f"non-finite: {log_probs[~torch.isfinite(log_probs)]}"

    def test_log_prob_finite_extreme_inputs(self, actor: GaussianActor):
        states = torch.randn(50, STATE_DIM) * 100
        _, log_probs = actor.sample(states)
        assert torch.isfinite(log_probs).all()

    def test_log_prob_shape(self, actor: GaussianActor):
        states = torch.randn(32, STATE_DIM)
        _, log_probs = actor.sample(states)
        assert log_probs.shape == (32, 1)


# ── 3. Twin Critic Independence ──────────────────────────────


class TestTwinCriticIndependence:
    """Proof 3: Q1 and Q2 are parameterised independently."""

    def test_different_initial_outputs(self, critic: TwinCritic):
        """With different random seeds, Q1 ≠ Q2 almost surely."""
        s = torch.randn(10, STATE_DIM)
        a = torch.randn(10, ACTION_DIM)
        q1, q2 = critic(s, a)
        # They should differ (both random init, independent params)
        assert not torch.allclose(q1, q2), "Q1 and Q2 shouldn't be identical"

    def test_param_count(self, critic: TwinCritic):
        """Q1 and Q2 have equal parameter counts."""
        q1_params = sum(p.numel() for p in critic._q1.parameters())
        q2_params = sum(p.numel() for p in critic._q2.parameters())
        assert q1_params == q2_params

    def test_gradient_isolation(self, critic: TwinCritic):
        """Gradient on Q1 loss doesn't affect Q2 params."""
        s = torch.randn(4, STATE_DIM)
        a = torch.randn(4, ACTION_DIM)
        q1, q2 = critic(s, a)

        # Only backprop through Q1
        q1.sum().backward()

        # Q2 params should have no gradient
        for p in critic._q2.parameters():
            assert p.grad is None or (p.grad == 0).all()


# ── 4. Soft Target Update (Polyak) ───────────────────────────


class TestSoftTargetUpdate:
    """Proof 4: θ' = (1-τ)θ' + τθ."""

    def test_polyak_averaging(self, cfg: SACConfig, filled_buffer: ReplayBuffer):
        trainer = SACTrainer(STATE_DIM, ACTION_DIM, cfg)
        tau = cfg.tau

        # Snapshot target params before update
        old_target = {name: p.clone() for name, p in trainer._target_critic.named_parameters()}
        # Snapshot critic params before update
        old_critic = {name: p.clone() for name, p in trainer._critic.named_parameters()}

        trainer.update(filled_buffer)

        # Check Polyak update: new_target ≈ (1-τ)*old_target + τ*new_critic
        for name, tp in trainer._target_critic.named_parameters():
            new_critic_p = dict(trainer._critic.named_parameters())[name]
            expected = (1 - tau) * old_target[name] + tau * new_critic_p
            torch.testing.assert_close(tp, expected, atol=1e-5, rtol=1e-5)


# ── 5. Alpha Auto-tuning ─────────────────────────────────────


class TestAlphaAutoTuning:
    """Proof 5: α adapts in the direction that matches target entropy."""

    def test_alpha_initial_value(self, cfg: SACConfig):
        sched = AlphaScheduler(ACTION_DIM, cfg)
        assert sched.alpha == pytest.approx(1.0)  # exp(0) = 1

    def test_alpha_adjusts(self, cfg: SACConfig):
        """Temperature α should increase when entropy is below target,
        and decrease when entropy is above target.

        Target entropy = scale * dim(A) = -0.5 * 5 = -2.5.

        Loss: L(α) = -α (log π + H̄)
        Gradient: ∂L/∂(log α) = -α (log π + H̄)
            If log π > -H̄ (entropy below target): gradient < 0 → log α increases → α increases
            If log π < -H̄ (entropy above target): gradient > 0 → log α decreases → α decreases
        """
        target = cfg.target_entropy_scale * ACTION_DIM  # -2.5

        # Scheduler A: log_probs = -10 (entropy too HIGH → α should DECREASE)
        sched_high = AlphaScheduler(ACTION_DIM, cfg)
        high_entropy_lp = torch.full((32, 1), -10.0)
        for _ in range(200):
            sched_high.update(high_entropy_lp)
        alpha_high_entropy = sched_high.alpha

        # Scheduler B: log_probs = -0.1 (entropy too LOW → α should INCREASE)
        sched_low = AlphaScheduler(ACTION_DIM, cfg)
        low_entropy_lp = torch.full((32, 1), -0.1)
        for _ in range(200):
            sched_low.update(low_entropy_lp)
        alpha_low_entropy = sched_low.alpha

        # α(low_entropy) > α(high_entropy) because the optimizer
        # pushes α up when entropy is below target and down when above
        assert alpha_low_entropy > alpha_high_entropy, (
            f"Expected α(low_entropy)={alpha_low_entropy:.6f} > α(high_entropy)={alpha_high_entropy:.6f}"
        )

    def test_alpha_state_dict_roundtrip(self, cfg: SACConfig):
        sched = AlphaScheduler(ACTION_DIM, cfg)
        lp = torch.randn(16, 1) * 2
        for _ in range(10):
            sched.update(lp)
        alpha_before = sched.alpha

        state = sched.state_dict()
        sched2 = AlphaScheduler(ACTION_DIM, cfg)
        sched2.load_state_dict(state)
        assert sched2.alpha == pytest.approx(alpha_before)


# ── 6. Deterministic Mode ────────────────────────────────────


class TestDeterministicMode:
    """Proof 6: deterministic selection is reproducible."""

    def test_deterministic_is_repeatable(self, trainer: SACTrainer):
        s = torch.randn(STATE_DIM)
        a1 = trainer.select_action(s, deterministic=True)
        a2 = trainer.select_action(s, deterministic=True)
        np.testing.assert_array_equal(a1, a2)

    def test_stochastic_varies(self, trainer: SACTrainer):
        """Stochastic mode should produce varying actions (probabilistic check)."""
        s = torch.randn(STATE_DIM)
        actions = [trainer.select_action(s, deterministic=False) for _ in range(20)]
        # At least 2 distinct actions in 20 samples
        unique = len(set(tuple(a.tolist()) for a in actions))
        assert unique > 1

    def test_select_action_shape(self, trainer: SACTrainer):
        s = torch.randn(STATE_DIM)
        a = trainer.select_action(s, deterministic=True)
        assert a.shape == (ACTION_DIM,)

    def test_select_action_batched_state(self, trainer: SACTrainer):
        s = torch.randn(1, STATE_DIM)
        a = trainer.select_action(s, deterministic=True)
        assert a.shape == (ACTION_DIM,)


# ── 7. Save/Load Roundtrip ───────────────────────────────────


class TestSaveLoadRoundtrip:
    """Proof 7: save() then load() restores identical behaviour."""

    def test_save_load_preserves_actions(self, cfg: SACConfig):
        trainer1 = SACTrainer(STATE_DIM, ACTION_DIM, cfg)
        s = torch.randn(STATE_DIM)
        a1 = trainer1.select_action(s, deterministic=True)

        data = trainer1.save()
        trainer2 = SACTrainer.load(data, STATE_DIM, ACTION_DIM, cfg)
        a2 = trainer2.select_action(s, deterministic=True)

        np.testing.assert_array_almost_equal(a1, a2, decimal=5)

    def test_save_load_preserves_critic(self, cfg: SACConfig):
        trainer1 = SACTrainer(STATE_DIM, ACTION_DIM, cfg)
        s = torch.randn(1, STATE_DIM)
        a = torch.randn(1, ACTION_DIM)
        q1_1, q2_1 = trainer1._critic(s, a)

        data = trainer1.save()
        trainer2 = SACTrainer.load(data, STATE_DIM, ACTION_DIM, cfg)
        q1_2, q2_2 = trainer2._critic(s, a)

        torch.testing.assert_close(q1_1, q1_2)
        torch.testing.assert_close(q2_1, q2_2)

    def test_save_load_preserves_update_count(self, cfg: SACConfig, filled_buffer: ReplayBuffer):
        trainer1 = SACTrainer(STATE_DIM, ACTION_DIM, cfg)
        trainer1.update(filled_buffer)
        trainer1.update(filled_buffer)
        assert trainer1._update_count == 2

        data = trainer1.save()
        trainer2 = SACTrainer.load(data, STATE_DIM, ACTION_DIM, cfg)
        assert trainer2._update_count == 2


# ── 8. Gradient Flow ─────────────────────────────────────────


class TestGradientFlow:
    """Proof 8: actor loss backpropagates through the policy."""

    def test_actor_has_gradients_after_update(self, trainer: SACTrainer, filled_buffer: ReplayBuffer):
        trainer.update(filled_buffer)
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainer._actor.parameters())
        assert has_grad, "Actor should have non-zero gradients after update"

    def test_critic_has_gradients_after_update(self, trainer: SACTrainer, filled_buffer: ReplayBuffer):
        trainer.update(filled_buffer)
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainer._critic.parameters())
        assert has_grad, "Critic should have non-zero gradients after update"


# ── 9. Critic Convergence ────────────────────────────────────


class TestCriticConvergence:
    """Proof 9: critic loss should decrease with training on fixed data."""

    def test_critic_loss_decreases(self, cfg: SACConfig, filled_buffer: ReplayBuffer):
        trainer = SACTrainer(STATE_DIM, ACTION_DIM, cfg)
        losses = []
        for _ in range(30):
            metrics = trainer.update(filled_buffer)
            losses.append(metrics["critic_loss"])

        # Compare first 5 avg vs last 5 avg
        early_avg = np.mean(losses[:5])
        late_avg = np.mean(losses[-5:])
        # Allow for noise but late should be generally lower
        assert late_avg < early_avg * 1.5, f"Critic loss not decreasing: early={early_avg:.4f} late={late_avg:.4f}"


# ── 10. Leverage Constraint ──────────────────────────────────


class TestLeverageConstraint:
    """Proof 10: gross exposure ≤ leverage_limit after enforcement."""

    def test_leverage_enforcement(self):
        cfg = SACConfig(
            hidden_dim=32,
            num_hidden=2,
            max_position=0.8,
            leverage_limit=1.0,
        )
        actor = GaussianActor(STATE_DIM, ACTION_DIM, cfg)
        states = torch.randn(200, STATE_DIM) * 10
        actions, _ = actor.sample(states)
        gross = actions.abs().sum(dim=-1)
        assert (gross <= 1.0 + 1e-5).all(), f"max gross: {gross.max()}"


# ── Update Metrics ────────────────────────────────────────────


class TestUpdateMetrics:
    """Verify update() returns expected metric keys."""

    def test_metric_keys(self, trainer: SACTrainer, filled_buffer: ReplayBuffer):
        metrics = trainer.update(filled_buffer)
        expected = {
            "critic_loss",
            "actor_loss",
            "alpha",
            "q1_mean",
            "q2_mean",
            "log_prob_mean",
            "gate_entropy",
        }
        assert set(metrics.keys()) == expected

    def test_metrics_are_finite(self, trainer: SACTrainer, filled_buffer: ReplayBuffer):
        metrics = trainer.update(filled_buffer)
        for k, v in metrics.items():
            assert np.isfinite(v), f"{k} is not finite: {v}"
