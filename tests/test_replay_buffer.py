"""Tests for ReplayBuffer — Phase 21b.2

Mathematical proofs:
    1. FIFO invariant:      oldest transitions overwritten first in circular mode
    2. Capacity bound:      len(buffer) ≤ capacity at all times
    3. Uniform sampling:    sample indices are in [0, size) with P = 1/size
    4. Shape correctness:   tensors have expected dimensions
    5. Value preservation:  stored data equals retrieved data exactly
    6. Boundary conditions: edge cases (size=1, sample==size, etc.)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from agent.learning.policy.replay_buffer import ReplayBuffer

# ── Helpers ───────────────────────────────────────────────────


def _push_n(buf: ReplayBuffer, n: int, state_dim: int, action_dim: int) -> list[float]:
    """Push n transitions with reward = i for traceability. Return rewards."""
    rewards = []
    for i in range(n):
        buf.push(
            state=np.full(state_dim, float(i), dtype=np.float32),
            action=np.full(action_dim, float(i), dtype=np.float32),
            reward=float(i),
            next_state=np.full(state_dim, float(i + 100), dtype=np.float32),
            done=(i % 2 == 0),
        )
        rewards.append(float(i))
    return rewards


# ── Test Classes ──────────────────────────────────────────────


class TestConstructorValidation:
    def test_zero_capacity_raises(self):
        with pytest.raises(ValueError, match="capacity"):
            ReplayBuffer(0, 4, 2)

    def test_negative_capacity_raises(self):
        with pytest.raises(ValueError, match="capacity"):
            ReplayBuffer(-1, 4, 2)

    def test_zero_state_dim_raises(self):
        with pytest.raises(ValueError, match="state_dim"):
            ReplayBuffer(10, 0, 2)

    def test_zero_action_dim_raises(self):
        with pytest.raises(ValueError, match="action_dim"):
            ReplayBuffer(10, 4, 0)


class TestCapacityBound:
    """Proof 2: len(buffer) ≤ capacity at all times."""

    def test_size_grows_to_capacity(self):
        buf = ReplayBuffer(5, 3, 2)
        for i in range(5):
            buf.push(np.zeros(3), np.zeros(2), 0.0, np.zeros(3), False)
            assert len(buf) == i + 1
        assert len(buf) == 5

    def test_size_capped_at_capacity(self):
        buf = ReplayBuffer(5, 3, 2)
        _push_n(buf, 100, 3, 2)
        assert len(buf) == 5

    def test_capacity_property(self):
        buf = ReplayBuffer(42, 3, 2)
        assert buf.capacity == 42

    def test_is_ready_threshold(self):
        buf = ReplayBuffer(1000, 3, 2)
        _push_n(buf, 255, 3, 2)
        assert not buf.is_ready
        _push_n(buf, 1, 3, 2)
        assert buf.is_ready


class TestFIFOInvariant:
    """Proof 1: circular buffer overwrites oldest first."""

    def test_oldest_overwritten(self):
        buf = ReplayBuffer(3, 1, 1)
        # Push 5 transitions, only last 3 should remain
        for i in range(5):
            buf.push(
                np.array([float(i)]),
                np.array([float(i)]),
                float(i),
                np.array([float(i)]),
                False,
            )
        # Buffer should contain transitions 2, 3, 4
        assert len(buf) == 3
        # Sample all to verify
        remaining_rewards = set()
        for _ in range(100):
            _, _, r, _, _ = buf.sample(1)
            remaining_rewards.add(float(r.squeeze().item()))
        assert remaining_rewards == {2.0, 3.0, 4.0}


class TestShapeCorrectness:
    """Proof 4: tensor shapes match specifications."""

    @pytest.mark.parametrize(
        "state_dim,action_dim,batch",
        [(4, 2, 1), (10, 5, 16), (1, 1, 3)],
    )
    def test_sample_shapes(self, state_dim, action_dim, batch):
        buf = ReplayBuffer(100, state_dim, action_dim)
        _push_n(buf, 50, state_dim, action_dim)
        s, a, r, ns, d = buf.sample(batch)

        assert s.shape == (batch, state_dim)
        assert a.shape == (batch, action_dim)
        assert r.shape == (batch, 1)
        assert ns.shape == (batch, state_dim)
        assert d.shape == (batch, 1)

    def test_tensor_dtype(self):
        buf = ReplayBuffer(10, 3, 2)
        _push_n(buf, 5, 3, 2)
        s, a, r, ns, d = buf.sample(2)
        assert s.dtype == torch.float32
        assert a.dtype == torch.float32
        assert r.dtype == torch.float32
        assert d.dtype == torch.float32


class TestValuePreservation:
    """Proof 5: stored values exactly equal retrieved values."""

    def test_single_transition_roundtrip(self):
        buf = ReplayBuffer(10, 3, 2)
        state = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        action = np.array([0.5, -0.5], dtype=np.float32)
        reward = 1.234
        next_state = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        done = True

        buf.push(state, action, reward, next_state, done)

        s, a, r, ns, d = buf.sample(1)
        np.testing.assert_array_equal(s[0].numpy(), state)
        np.testing.assert_array_equal(a[0].numpy(), action)
        assert r[0, 0].item() == pytest.approx(reward)
        np.testing.assert_array_equal(ns[0].numpy(), next_state)
        assert d[0, 0].item() == pytest.approx(1.0)

    def test_done_encoding(self):
        buf = ReplayBuffer(10, 1, 1)
        buf.push(np.zeros(1), np.zeros(1), 0.0, np.zeros(1), False)
        buf.push(np.zeros(1), np.zeros(1), 0.0, np.zeros(1), True)

        # Check done encoding across samples
        done_vals = set()
        for _ in range(200):
            _, _, _, _, d = buf.sample(1)
            done_vals.add(float(d.squeeze().item()))
        assert done_vals == {0.0, 1.0}


class TestSamplingBoundary:
    """Proof 6: boundary conditions."""

    def test_sample_more_than_size_raises(self):
        buf = ReplayBuffer(10, 3, 2)
        _push_n(buf, 5, 3, 2)
        with pytest.raises(ValueError, match="Cannot sample"):
            buf.sample(6)

    def test_sample_exact_size_works(self):
        buf = ReplayBuffer(10, 3, 2)
        _push_n(buf, 5, 3, 2)
        s, a, r, ns, d = buf.sample(5)
        assert s.shape[0] == 5

    def test_sample_size_one(self):
        buf = ReplayBuffer(10, 3, 2)
        _push_n(buf, 1, 3, 2)
        s, a, r, ns, d = buf.sample(1)
        assert s.shape == (1, 3)

    def test_empty_buffer_sample_raises(self):
        buf = ReplayBuffer(10, 3, 2)
        with pytest.raises(ValueError, match="Cannot sample"):
            buf.sample(1)


class TestUniformSampling:
    """Proof 3: sample draws uniformly from [0, size)."""

    def test_all_transitions_reachable(self):
        """With enough samples, every stored transition appears."""
        buf = ReplayBuffer(5, 1, 1)
        for i in range(5):
            buf.push(
                np.array([float(i)]),
                np.zeros(1),
                float(i),
                np.zeros(1),
                False,
            )

        seen_rewards = set()
        for _ in range(500):
            _, _, r, _, _ = buf.sample(1)
            seen_rewards.add(float(r.squeeze().item()))
        assert seen_rewards == {0.0, 1.0, 2.0, 3.0, 4.0}
