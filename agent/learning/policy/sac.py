"""TirraMind — Soft Actor-Critic (SAC)

Implements SAC (Haarnoja et al. 2018, arXiv:1801.01290) for continuous
portfolio weight allocation.

Mathematical framework
──────────────────────
SAC maximises the maximum-entropy objective:

    J(π) = Σ_t E[r_t + α H(π(·|s_t))]

where H is entropy and α is an auto-tuned temperature.

Three learnable components:
    1. GaussianActor   — π_φ(a|s) = tanh(μ + σ·ε), ε ~ N(0,I)
    2. TwinCritic       — Q_θ₁(s,a), Q_θ₂(s,a) (clipped double-Q)
    3. AlphaScheduler   — log(α) tuned to target entropy

Update rules (per Haarnoja 2018):
    Critic loss:  L(θ) = E[(Q_θ(s,a) − y)²]
        y = r + γ(min(Q'₁, Q'₂)(s',ã') − α log π(ã'|s'))
        ã' ~ π(·|s')

    Actor loss:   L(φ) = E[α log π(ã|s) − min(Q₁, Q₂)(s,ã)]
        ã ~ π(·|s)  (reparameterised)

    Temperature:  L(α) = E[−α(log π(ã|s) + H̄)]
        H̄ = target entropy = −dim(A)/2

Tanh squashing correction:
    log π(a|s) = log μ(u|s) − Σᵢ log(1 − tanh²(uᵢ))

Numerical stability:
    - log_std clamped to [-20, 2] (prevents variance collapse/explosion)
    - 1 − tanh²(u) floored at 1e-6 (prevents log(0))
    - Gradient norm clipped at 1.0

Trusted sources:
    - Haarnoja et al. (2018) arXiv:1801.01290 — SAC algorithm
    - Haarnoja et al. (2018) arXiv:1812.05905 — automatic temperature
    - Fujimoto et al. (2018) arXiv:1802.09477 — clipped double-Q
"""

from __future__ import annotations

import copy
import io
import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from agent.learning.policy.config import SACConfig
from agent.learning.policy.replay_buffer import ReplayBuffer

log = logging.getLogger(__name__)

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
TANH_EPS = 1e-6


# ── MLP builder ──────────────────────────────────────────────


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_hidden: int,
) -> nn.Sequential:
    """Build a ReLU MLP with the given architecture."""
    layers: list[nn.Module] = []
    in_d = input_dim
    for _ in range(num_hidden):
        layers.append(nn.Linear(in_d, hidden_dim))
        layers.append(nn.ReLU())
        in_d = hidden_dim
    layers.append(nn.Linear(in_d, output_dim))
    return nn.Sequential(*layers)


# ── Gaussian Actor ───────────────────────────────────────────


class GaussianActor(nn.Module):
    """Squashed Gaussian policy: s → (μ, log σ) → tanh(μ + σε).

    Output actions are in (−1, 1)^d, then rescaled by max_position
    and subject to leverage limit enforcement.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: SACConfig,
    ) -> None:
        super().__init__()
        self._action_dim = action_dim
        self._max_pos = config.max_position
        self._leverage = config.leverage_limit

        self._trunk = _build_mlp(
            state_dim, config.hidden_dim, config.hidden_dim, config.num_hidden - 1
        )
        self._mu_head = nn.Linear(config.hidden_dim, action_dim)
        self._log_std_head = nn.Linear(config.hidden_dim, action_dim)

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        """Return (mean, log_std) of the Gaussian before squashing."""
        h = self._trunk(state)
        mu = self._mu_head(h)
        log_std = self._log_std_head(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, state: Tensor) -> tuple[Tensor, Tensor]:
        """Sample action via reparameterisation trick.

        Returns
        -------
        (action, log_prob) where:
            action in [-max_pos, max_pos]^d, leverage-constrained
            log_prob accounts for tanh squashing correction
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()

        # Reparameterisation: u = μ + σ·ε
        dist = torch.distributions.Normal(mu, std)
        u = dist.rsample()

        # Tanh squashing
        raw_action = torch.tanh(u)

        # Log-prob with tanh correction
        # log π(a|s) = log N(u; μ, σ) − Σ log(1 − tanh²(u))
        log_prob = dist.log_prob(u) - torch.log(1.0 - raw_action.pow(2) + TANH_EPS)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # Scale to position limits
        action = raw_action * self._max_pos

        # Enforce leverage constraint: Σ|w_i| ≤ L
        action = self._enforce_leverage(action)

        return action, log_prob

    def _enforce_leverage(self, action: Tensor) -> Tensor:
        """Hard-clamp total gross exposure to leverage limit."""
        gross = action.abs().sum(dim=-1, keepdim=True)
        scale = torch.where(
            gross > self._leverage,
            self._leverage / (gross + 1e-8),
            torch.ones_like(gross),
        )
        return action * scale


# ── Twin Critic ──────────────────────────────────────────────


class TwinCritic(nn.Module):
    """Two independent Q-networks for clipped double-Q.

    min(Q₁, Q₂) provides pessimistic value estimates,
    mitigating overestimation bias (Fujimoto 2018).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: SACConfig,
    ) -> None:
        super().__init__()
        self._q1 = _build_mlp(
            state_dim + action_dim, 1, config.hidden_dim, config.num_hidden
        )
        self._q2 = _build_mlp(
            state_dim + action_dim, 1, config.hidden_dim, config.num_hidden
        )

    def forward(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        """Return (Q1, Q2) values."""
        sa = torch.cat([state, action], dim=-1)
        return self._q1(sa), self._q2(sa)


# ── Temperature Scheduler ────────────────────────────────────


class AlphaScheduler:
    """Auto-tuned entropy temperature (Haarnoja 2018b, §5).

    Maintains learnable log(α) and updates toward target entropy:
        L(α) = E[−α (log π(a|s) + H̄)]

    Target entropy H̄ = scale × dim(A), where scale = −0.5
    (Haarnoja's heuristic for continuous action spaces).
    """

    def __init__(self, action_dim: int, config: SACConfig) -> None:
        self._target = config.target_entropy_scale * action_dim
        self._log_alpha = torch.tensor(0.0, requires_grad=True)
        self._optimizer = torch.optim.Adam([self._log_alpha], lr=config.alpha_lr)

    @property
    def alpha(self) -> float:
        return float(self._log_alpha.exp().item())

    @property
    def alpha_tensor(self) -> Tensor:
        return self._log_alpha.exp().detach()

    def update(self, log_probs: Tensor) -> float:
        """Update temperature. Returns current alpha."""
        alpha_loss = -(
            self._log_alpha.exp() * (log_probs.detach() + self._target)
        ).mean()
        self._optimizer.zero_grad()
        alpha_loss.backward()
        self._optimizer.step()
        return self.alpha

    def state_dict(self) -> dict[str, Any]:
        return {
            "log_alpha": self._log_alpha.data.item(),
            "optimizer": self._optimizer.state_dict(),
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self._log_alpha.data.fill_(d["log_alpha"])
        self._optimizer.load_state_dict(d["optimizer"])


# ── SAC Trainer ──────────────────────────────────────────────


class SACTrainer:
    """Orchestrates SAC training: critics → actor → temperature → soft update.

    One call to ``update()`` performs a full SAC step on a mini-batch
    sampled from the replay buffer.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: SACConfig | None = None,
    ) -> None:
        self._config = config or SACConfig()
        self._state_dim = state_dim
        self._action_dim = action_dim
        cfg = self._config

        # Networks
        self._actor = GaussianActor(state_dim, action_dim, cfg)
        self._critic = TwinCritic(state_dim, action_dim, cfg)
        self._target_critic = copy.deepcopy(self._critic)
        # Freeze target parameters
        for p in self._target_critic.parameters():
            p.requires_grad = False

        # Optimisers
        self._actor_optim = torch.optim.Adam(self._actor.parameters(), lr=cfg.actor_lr)
        self._critic_optim = torch.optim.Adam(
            self._critic.parameters(), lr=cfg.critic_lr
        )

        # Temperature
        self._alpha_sched = AlphaScheduler(action_dim, cfg)

        self._update_count = 0

    def update(self, buffer: ReplayBuffer) -> dict[str, float]:
        """One SAC update step. Returns loss metrics.

        Steps:
            1. Sample batch from buffer
            2. Compute critic targets: y = r + γ(min(Q'₁,Q'₂) − α log π)
            3. Update critics (MSE loss)
            4. Update actor: maximise Q − α log π
            5. Update temperature α
            6. Soft-update target networks (Polyak averaging)
        """
        cfg = self._config
        batch_size = min(cfg.batch_size, len(buffer))
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        alpha = self._alpha_sched.alpha_tensor

        # ── Step 2: Critic targets ────────────────────────────
        with torch.no_grad():
            next_actions, next_log_probs = self._actor.sample(next_states)
            tq1, tq2 = self._target_critic(next_states, next_actions)
            target_q = torch.min(tq1, tq2) - alpha * next_log_probs
            y = rewards + cfg.gamma * (1.0 - dones) * target_q

        # ── Step 3: Update critics ────────────────────────────
        q1, q2 = self._critic(states, actions)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self._critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._critic.parameters(), 1.0)
        self._critic_optim.step()

        # ── Step 4: Update actor ──────────────────────────────
        new_actions, log_probs = self._actor.sample(states)
        q1_new, q2_new = self._critic(states, new_actions)
        q_min = torch.min(q1_new, q2_new)
        actor_loss = (alpha * log_probs - q_min).mean()

        self._actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._actor.parameters(), 1.0)
        self._actor_optim.step()

        # ── Step 5: Update temperature ────────────────────────
        new_alpha = self._alpha_sched.update(log_probs)

        # ── Step 6: Soft update targets ───────────────────────
        with torch.no_grad():
            for p, tp in zip(
                self._critic.parameters(), self._target_critic.parameters()
            ):
                tp.data.mul_(1.0 - cfg.tau).add_(p.data, alpha=cfg.tau)

        self._update_count += 1

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": new_alpha,
            "q1_mean": float(q1.mean().item()),
            "q2_mean": float(q2.mean().item()),
            "log_prob_mean": float(log_probs.mean().item()),
        }

    def select_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> np.ndarray:
        """Return action as numpy array.

        deterministic=True uses tanh(μ) (no sampling noise);
        deterministic=False samples from π.
        """
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            if deterministic:
                mu, _ = self._actor(state)
                action = torch.tanh(mu) * self._config.max_position
                action = self._actor._enforce_leverage(action)
            else:
                action, _ = self._actor.sample(state)
        return action.squeeze(0).cpu().numpy()

    # ── Serialisation ─────────────────────────────────────────

    def save(self) -> bytes:
        """Serialise all state to bytes."""
        buf = io.BytesIO()
        torch.save(
            {
                "actor": self._actor.state_dict(),
                "critic": self._critic.state_dict(),
                "target_critic": self._target_critic.state_dict(),
                "actor_optim": self._actor_optim.state_dict(),
                "critic_optim": self._critic_optim.state_dict(),
                "alpha": self._alpha_sched.state_dict(),
                "update_count": self._update_count,
            },
            buf,
        )
        return buf.getvalue()

    @classmethod
    def load(
        cls,
        data: bytes,
        state_dim: int,
        action_dim: int,
        config: SACConfig | None = None,
    ) -> SACTrainer:
        """Deserialise from bytes."""
        trainer = cls(state_dim, action_dim, config)
        buf = io.BytesIO(data)
        checkpoint = torch.load(buf, map_location="cpu", weights_only=False)
        trainer._actor.load_state_dict(checkpoint["actor"])
        trainer._critic.load_state_dict(checkpoint["critic"])
        trainer._target_critic.load_state_dict(checkpoint["target_critic"])
        trainer._actor_optim.load_state_dict(checkpoint["actor_optim"])
        trainer._critic_optim.load_state_dict(checkpoint["critic_optim"])
        trainer._alpha_sched.load_state_dict(checkpoint["alpha"])
        trainer._update_count = checkpoint["update_count"]
        return trainer
