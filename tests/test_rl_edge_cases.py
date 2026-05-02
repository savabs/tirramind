"""Edge case tests for Phase 21a (weight learner) and Phase 21b (SAC).

Covers: invalid inputs, boundary values, numerical stability,
degenerate data, serialisation corruption, and adversarial scenarios.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from agent.fusion.alert import EntityAlert
from agent.learning.policy.asset_mapper import AssetMapper
from agent.learning.policy.config import (
    RewardConfig,
    SACConfig,
    WeightLearnerConfig,
)
from agent.learning.policy.replay_buffer import ReplayBuffer
from agent.learning.policy.reward_fn import RewardFunction
from agent.learning.policy.sac import (
    AlphaScheduler,
    GaussianActor,
    SACTrainer,
)
from agent.learning.policy.state_assembler import StateAssembler
from agent.learning.policy.symlog import symexp, symlog, symlog_np
from agent.learning.policy.weight_learner import SurpriseWeightLearner
from agent.models.belief import BeliefState

# ── Helpers ───────────────────────────────────────────────────


def _make_alert(
    entity_id: str = "e1",
    composite: float = 1.0,
    obs: float = 0.1,
    temporal: float = 0.2,
    value: float = 0.3,
    neighborhood: float = 0.4,
    drift: float = 0.5,
) -> EntityAlert:
    return EntityAlert(
        entity_id=entity_id,
        entity_type="company",
        entity_name=f"Name-{entity_id}",
        alert_time=time.time(),
        obs_type_surprise=obs,
        temporal_surprise=temporal,
        value_surprise=value,
        neighborhood_surprise=neighborhood,
        memory_drift=drift,
        cusum_statistic=0.0,
        hawkes_intensity=0.0,
        event_study_score=0.0,
        composite_surprise=composite,
        observation_count=1,
        evidence_sources=("test",),
    )


def _make_belief(entity_id: str = "e1") -> BeliefState:
    now = time.time()
    return BeliefState(
        variable_name="regime.test",
        version=1,
        effective_at=now,
        computed_at=now,
        dist_type="gaussian",
        mean=0.5,
        variance=0.1,
        confidence=0.8,
        entity_id=entity_id,
    )


def _mock_mapper(mappings: dict[str, str]) -> AssetMapper:
    mapper = MagicMock(spec=AssetMapper)
    mapper.resolve.side_effect = lambda eid: mappings.get(eid)
    mapper.tradeable_entities.return_value = dict(mappings)
    return mapper


# ═══════════════════════════════════════════════════════════════
# Phase 21a Edge Cases
# ═══════════════════════════════════════════════════════════════


class TestSymlogEdge:
    """Numerical stability under extreme inputs."""

    def test_very_large_input(self):
        x = torch.tensor([1e10, -1e10])
        y = symlog(x)
        assert torch.isfinite(y).all()
        assert torch.allclose(symexp(y), x, rtol=1e-4)

    def test_very_small_input(self):
        x = torch.tensor([1e-15, -1e-15])
        y = symlog(x)
        assert torch.isfinite(y).all()

    def test_nan_propagation(self):
        x = torch.tensor([float("nan")])
        y = symlog(x)
        assert torch.isnan(y).all()

    def test_inf_propagation(self):
        x = torch.tensor([float("inf")])
        y = symlog(x)
        assert torch.isinf(y).all()

    def test_numpy_matches_torch(self):
        x_np = np.array([0.0, 1.0, -1.0, 100.0, -100.0, 1e-10])
        x_t = torch.from_numpy(x_np)
        np.testing.assert_allclose(symlog_np(x_np), symlog(x_t).numpy(), atol=1e-7)


class TestRewardEdge:
    """RewardFunction under degenerate conditions."""

    def test_zero_return_zero_vol(self):
        rf = RewardFunction(RewardConfig())
        r = rf.extrinsic(0.0, np.zeros(20))
        assert np.isfinite(r)

    def test_all_same_returns(self):
        rf = RewardFunction(RewardConfig())
        rolling = np.full(20, 0.01)
        r = rf.extrinsic(0.01, rolling)
        assert np.isfinite(r)

    def test_single_return(self):
        rf = RewardFunction(RewardConfig())
        r = rf.extrinsic(0.05, np.array([0.05]))
        assert np.isfinite(r)

    def test_intrinsic_empty_surprises(self):
        rf = RewardFunction(RewardConfig())
        r = rf.intrinsic(np.array([]))
        assert r == 0.0

    def test_combined_all_zeros(self):
        rf = RewardFunction(RewardConfig())
        total, breakdown = rf.combined(
            portfolio_return=0.0,
            rolling_returns=np.zeros(20),
            surprise_scores=np.zeros(5),
            step=0,
            total_steps=100,
        )
        assert np.isfinite(total)
        assert "extrinsic" in breakdown
        assert "intrinsic" in breakdown

    def test_negative_cvar(self):
        """Deep tail loss should trigger CVaR penalty."""
        rf = RewardFunction(RewardConfig(cvar_penalty=2.0))
        rolling = np.concatenate([np.full(19, 0.01), [-0.5]])
        r = rf.extrinsic(-0.5, rolling)
        assert np.isfinite(r)


class TestWeightLearnerEdge:
    """Weight learner under adversarial data conditions."""

    def test_constant_surprises(self):
        """All surprise channels identical → weights should converge to uniform."""
        surprises = np.ones((200, 5)) * 0.5
        returns = np.random.randn(200) * 0.01
        cfg = WeightLearnerConfig(
            min_train_periods=50,
            test_periods=25,
            walk_forward_step=25,
            max_epochs=50,
            patience=10,
        )
        learner = SurpriseWeightLearner(cfg)
        learner.fit(surprises, returns)
        w = learner.get_learned_weights()
        # With identical channels, weights should be ≈ uniform (0.2 each)
        np.testing.assert_allclose(w, 0.2, atol=0.15)

    def test_zero_returns(self):
        """Zero returns everywhere → Sharpe undefined, should not crash."""
        surprises = np.random.randn(200, 5)
        returns = np.zeros(200)
        cfg = WeightLearnerConfig(min_train_periods=50, test_periods=25, walk_forward_step=25, max_epochs=50)
        learner = SurpriseWeightLearner(cfg)
        learner.fit(surprises, returns)
        w = learner.get_learned_weights()
        assert all(np.isfinite(x) for x in w)
        assert abs(sum(w) - 1.0) < 1e-6


# ═══════════════════════════════════════════════════════════════
# Phase 21b Edge Cases
# ═══════════════════════════════════════════════════════════════


class TestStateAssemblerEdge:
    """State assembler under adversarial inputs."""

    def test_all_non_tradeable(self):
        """All alerts are non-tradeable → state is all zeros except normalised count."""
        sa = StateAssembler(max_entities=5, market_dim=0)
        alerts = [_make_alert(f"e{i}", composite=5.0) for i in range(10)]
        state, meta = sa.assemble(alerts, [], {}, {})
        assert meta["n_active"] == 0
        # All zeros including normalised count (0/5 = 0)
        np.testing.assert_array_equal(state.numpy(), 0.0)

    def test_belief_without_entity_id(self):
        """Beliefs with None entity_id are gracefully skipped."""
        sa = StateAssembler(max_entities=2, surprise_dim=5, belief_dim=4, market_dim=0)
        alerts = [_make_alert("e1", composite=2.0)]
        belief_no_id = BeliefState(
            variable_name="regime.test",
            version=1,
            effective_at=time.time(),
            computed_at=time.time(),
            dist_type="gaussian",
            mean=0.5,
            variance=0.1,
            confidence=0.8,
            entity_id=None,
        )
        state, _ = sa.assemble(alerts, [belief_no_id], {}, {"e1": "AAPL"})
        assert state.shape == (sa.state_dim,)

    def test_duplicate_entity_alerts(self):
        """Multiple alerts for same entity → only top-K considered."""
        sa = StateAssembler(max_entities=2)
        alerts = [
            _make_alert("e1", composite=3.0),
            _make_alert("e1", composite=5.0),
            _make_alert("e1", composite=1.0),
        ]
        state, meta = sa.assemble(alerts, [], {}, {"e1": "AAPL"})
        # Should keep top 2 by composite (5.0 and 3.0)
        assert meta["n_active"] == 2

    def test_max_entities_one(self):
        sa = StateAssembler(max_entities=1, surprise_dim=5, belief_dim=4, market_dim=2)
        alerts = [_make_alert("e1", composite=1.0)]
        state, meta = sa.assemble(alerts, [], {"a": 1.0}, {"e1": "T"})
        assert state.shape == (sa.state_dim,)
        assert meta["n_active"] == 1


class TestSACEdge:
    """SAC under degenerate conditions."""

    def test_single_asset(self):
        """Action dim = 1 (single tradeable asset)."""
        cfg = SACConfig(hidden_dim=16, num_hidden=1)
        trainer = SACTrainer(10, 1, cfg)
        s = torch.randn(10)
        a = trainer.select_action(s, deterministic=True)
        assert a.shape == (1,)
        assert abs(a[0]) <= cfg.max_position + 1e-6

    def test_large_state_dim(self):
        """Very large state space doesn't crash."""
        cfg = SACConfig(hidden_dim=32, num_hidden=1)
        trainer = SACTrainer(1000, 5, cfg)
        s = torch.randn(1000)
        a = trainer.select_action(s, deterministic=True)
        assert a.shape == (5,)

    def test_update_with_minimal_buffer(self):
        """Training with only batch_size transitions should work."""
        cfg = SACConfig(hidden_dim=16, num_hidden=1, batch_size=4)
        trainer = SACTrainer(10, 3, cfg)
        buf = ReplayBuffer(100, 10, 3)
        for _ in range(4):
            buf.push(
                np.random.randn(10).astype(np.float32),
                np.random.randn(3).astype(np.float32),
                0.0,
                np.random.randn(10).astype(np.float32),
                False,
            )
        metrics = trainer.update(buf)
        assert np.isfinite(metrics["critic_loss"])

    def test_all_same_transitions(self):
        """Degenerate: all transitions identical → should not NaN."""
        cfg = SACConfig(hidden_dim=16, num_hidden=1, batch_size=8)
        trainer = SACTrainer(5, 2, cfg)
        buf = ReplayBuffer(100, 5, 2)
        s = np.zeros(5, dtype=np.float32)
        a = np.zeros(2, dtype=np.float32)
        for _ in range(20):
            buf.push(s, a, 0.0, s, False)
        metrics = trainer.update(buf)
        for k, v in metrics.items():
            assert np.isfinite(v), f"{k} = {v}"

    def test_gradients_finite_after_100_updates(self):
        """Numerical stability: gradients stay finite through 100 updates."""
        cfg = SACConfig(hidden_dim=16, num_hidden=1, batch_size=16)
        trainer = SACTrainer(10, 3, cfg)
        buf = ReplayBuffer(1000, 10, 3)
        for _ in range(100):
            buf.push(
                np.random.randn(10).astype(np.float32),
                np.random.randn(3).astype(np.float32) * 0.3,
                float(np.random.randn()),
                np.random.randn(10).astype(np.float32),
                bool(np.random.rand() > 0.95),
            )

        for i in range(100):
            metrics = trainer.update(buf)
            for k, v in metrics.items():
                assert np.isfinite(v), f"Step {i}, {k} = {v}"

    def test_action_at_boundary(self):
        """Actions near ±max_position are valid (tanh asymptote)."""
        cfg = SACConfig(hidden_dim=16, num_hidden=1, max_position=0.5)
        actor = GaussianActor(10, 3, cfg)
        # Extreme states to push tanh near ±1
        states = torch.randn(1000, 10) * 100
        actions, log_probs = actor.sample(states)
        assert torch.isfinite(actions).all()
        assert torch.isfinite(log_probs).all()


class TestReplayBufferEdge:
    """Replay buffer boundary conditions."""

    def test_capacity_one(self):
        buf = ReplayBuffer(1, 3, 2)
        buf.push(np.ones(3), np.ones(2), 1.0, np.ones(3), False)
        s, a, r, ns, d = buf.sample(1)
        assert s.shape == (1, 3)
        np.testing.assert_array_equal(s[0].numpy(), 1.0)

    def test_capacity_one_overwrite(self):
        buf = ReplayBuffer(1, 2, 1)
        buf.push(np.array([1.0, 1.0]), np.array([1.0]), 1.0, np.array([1.0, 1.0]), False)
        buf.push(np.array([2.0, 2.0]), np.array([2.0]), 2.0, np.array([2.0, 2.0]), True)
        assert len(buf) == 1
        s, _, r, _, d = buf.sample(1)
        assert s[0, 0].item() == 2.0
        assert r[0, 0].item() == 2.0


class TestSerializationEdge:
    """SAC serialisation edge cases."""

    def test_corrupt_bytes_raises(self):
        with pytest.raises(Exception):
            SACTrainer.load(b"corrupt_data", 10, 3)

    def test_empty_bytes_raises(self):
        with pytest.raises(Exception):
            SACTrainer.load(b"", 10, 3)

    def test_save_load_after_updates(self):
        """Save/load after many updates must preserve exact policy."""
        cfg = SACConfig(hidden_dim=16, num_hidden=1, batch_size=8)
        trainer = SACTrainer(10, 3, cfg)
        buf = ReplayBuffer(500, 10, 3)
        for _ in range(50):
            buf.push(
                np.random.randn(10).astype(np.float32),
                np.random.randn(3).astype(np.float32),
                float(np.random.randn()),
                np.random.randn(10).astype(np.float32),
                False,
            )
        for _ in range(20):
            trainer.update(buf)

        data = trainer.save()
        trainer2 = SACTrainer.load(data, 10, 3, cfg)

        s = torch.randn(10)
        a1 = trainer.select_action(s, deterministic=True)
        a2 = trainer2.select_action(s, deterministic=True)
        np.testing.assert_array_almost_equal(a1, a2, decimal=5)


class TestAlphaSchedulerEdge:
    """Temperature scheduler edge cases."""

    def test_alpha_stays_positive(self):
        """α = exp(log_α) must always be positive."""
        cfg = SACConfig()
        sched = AlphaScheduler(5, cfg)
        lp = torch.full((32, 1), -100.0)
        for _ in range(500):
            sched.update(lp)
        assert sched.alpha > 0

    def test_extreme_log_probs(self):
        cfg = SACConfig()
        sched = AlphaScheduler(5, cfg)
        lp = torch.full((32, 1), -1000.0)
        for _ in range(100):
            a = sched.update(lp)
            assert np.isfinite(a)


class TestRewardCombinedEdge:
    """Combined reward under edge conditions."""

    def test_intrinsic_decay_at_final_step(self):
        """At step=total_steps, λ(t) should be 0 (no intrinsic reward)."""
        rf = RewardFunction(RewardConfig(intrinsic_weight_initial=0.5, intrinsic_decay=True))
        total, bd = rf.combined(
            portfolio_return=0.01,
            rolling_returns=np.full(20, 0.01),
            surprise_scores=np.ones(5),
            step=100,
            total_steps=100,
        )
        # λ(T) = λ₀ * (1 - T/T) = 0 → intrinsic contribution = 0
        assert bd["lambda_t"] == pytest.approx(0.0, abs=1e-10)

    def test_intrinsic_no_decay(self):
        """With decay=False, λ stays constant."""
        rf = RewardFunction(RewardConfig(intrinsic_weight_initial=0.5, intrinsic_decay=False))
        _, bd1 = rf.combined(0.01, np.full(20, 0.01), surprise_scores=np.ones(5), step=0, total_steps=100)
        _, bd2 = rf.combined(
            0.01,
            np.full(20, 0.01),
            surprise_scores=np.ones(5),
            step=99,
            total_steps=100,
        )
        assert bd1["lambda_t"] == pytest.approx(bd2["lambda_t"])
