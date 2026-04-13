"""TirraMind — Off-Policy Replay Buffer

Circular numpy-backed buffer for SAC training.  All data stored as
contiguous float32 arrays for cache-friendly batch sampling.

Complexity:
    push()   : O(1)
    sample() : O(batch_size)  (numpy fancy indexing)
    memory   : O(capacity × (2·state_dim + action_dim + 2))

The buffer stores (s, a, r, s', done) tuples.  When full, oldest
transitions are overwritten (circular FIFO).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


class ReplayBuffer:
    """Fixed-capacity circular replay buffer backed by numpy arrays."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be ≥ 1, got {capacity}")
        if state_dim < 1 or action_dim < 1:
            raise ValueError("state_dim and action_dim must be ≥ 1")

        self._capacity = capacity
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._ptr = 0
        self._size = 0

        # Pre-allocate contiguous arrays
        self._states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition, overwriting oldest if full."""
        self._states[self._ptr] = state
        self._actions[self._ptr] = action
        self._rewards[self._ptr] = reward
        self._next_states[self._ptr] = next_state
        self._dones[self._ptr] = float(done)

        self._ptr = (self._ptr + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Uniformly sample a batch. Returns tensors on CPU.

        Returns
        -------
        (states, actions, rewards, next_states, dones) each as Tensor.
        rewards and dones have shape (batch_size, 1) for broadcasting.

        Raises
        ------
        ValueError
            If batch_size > current buffer size.
        """
        if batch_size > self._size:
            raise ValueError(
                f"Cannot sample {batch_size} from buffer of size {self._size}"
            )

        idx = np.random.randint(0, self._size, size=batch_size)

        return (
            torch.from_numpy(self._states[idx]),
            torch.from_numpy(self._actions[idx]),
            torch.from_numpy(self._rewards[idx]).unsqueeze(1),
            torch.from_numpy(self._next_states[idx]),
            torch.from_numpy(self._dones[idx]).unsqueeze(1),
        )

    def __len__(self) -> int:
        return self._size

    @property
    def is_ready(self) -> bool:
        """True if buffer has enough samples for at least one batch of 256."""
        return self._size >= 256

    @property
    def capacity(self) -> int:
        return self._capacity
