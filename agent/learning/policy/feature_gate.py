"""TirraMind — Learned Feature Gate (Change 11)

Regime-conditioned soft gating over feature groups.  The gate learns which
feature groups (surprise, belief, market, entity_count, adversarial) contribute
to policy quality under each HMM regime, replacing the implicit assumption
that all features are equally relevant.

Architecture:
    regime_context → MLP → sigmoid → gate_values ∈ [floor, 1.0]
    state_flat ⊙ expanded_gates → gated_state

Math:
    g_k = (1 - floor) · σ(MLP(r))_k + floor
    gated_x = [g_0 · x_surprise ; g_1 · x_belief ; g_2 · x_market ; ...]
    L_gate  = −λ Σ_k [ĝ_k log ĝ_k + (1−ĝ_k) log(1−ĝ_k)]

    where ĝ_k = (g_k − floor) / (1 − floor) is the normalised gate.

Reference: Feature-Gating MoE (Polson et al. 2026) — adaptive sparsity via
    gating with heavy-tailed priors.  We use a simpler sigmoid-MLP variant
    conditioned on HMM regime posterior.

Trained end-to-end with the SAC actor loss via the state encoder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True)
class FeatureGateConfig:
    """Hyperparameters for the regime-conditioned feature gate.

    Parameters
    ----------
    n_feature_groups : int
        Number of feature groups to gate (default 5: surprise, belief,
        market, entity_count, adversarial).
    regime_dim : int
        Dimensionality of the regime context vector (HMM posterior).
    gate_hidden_dim : int
        Hidden dimension of the gating MLP.
    gate_floor : float
        Minimum gate value in [0, 1).  Prevents total feature suppression.
        Set to 0 for unconstrained gating.
    entropy_weight : float
        Weight λ for the entropy regularisation loss that prevents gate
        collapse (all gates → 0 or 1).
    group_dims : tuple[int, ...]
        Per-group feature dimensionality.  Must sum to the total state dim.
        Default matches StateAssembler with E=50, M=8:
        (250, 200, 8, 1, 4) = surprise, belief, market, count, adversarial.
    regime_clamp : float
        Clamp regime context values to [-clamp, clamp] to prevent NaN.
    """

    n_feature_groups: int = 5
    regime_dim: int = 4
    gate_hidden_dim: int = 16
    gate_floor: float = 0.05
    entropy_weight: float = 0.01
    group_dims: tuple[int, ...] = (250, 200, 8, 1, 4)
    regime_clamp: float = 10.0


class FeatureGate(nn.Module):
    """Regime-conditioned soft gating over feature groups.

    Given a flat state vector and a regime context vector, produces a gated
    state where each feature group is scaled by a learned gate ∈ [floor, 1].

    Parameters
    ----------
    config : FeatureGateConfig
        Gate hyperparameters.

    Raises
    ------
    ValueError
        If ``len(group_dims) != n_feature_groups``.
    """

    def __init__(self, config: FeatureGateConfig | None = None) -> None:
        super().__init__()
        cfg = config or FeatureGateConfig()
        self._cfg = cfg

        if len(cfg.group_dims) != cfg.n_feature_groups:
            raise ValueError(
                f"len(group_dims)={len(cfg.group_dims)} != "
                f"n_feature_groups={cfg.n_feature_groups}"
            )

        self._group_dims = cfg.group_dims
        self._total_dim = sum(cfg.group_dims)
        self._floor = cfg.gate_floor
        self._entropy_weight = cfg.entropy_weight
        self._regime_clamp = cfg.regime_clamp

        # Gating MLP: regime_dim → hidden → n_groups (one gate per group)
        self._gate_mlp = nn.Sequential(
            nn.Linear(cfg.regime_dim, cfg.gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.gate_hidden_dim, cfg.n_feature_groups),
        )

        # Store last computed gate values for diagnostics
        self._last_gates: Tensor | None = None

        # Group boundary indices (precomputed)
        boundaries: list[int] = [0]
        for d in cfg.group_dims:
            boundaries.append(boundaries[-1] + d)
        # Register as buffer so it moves to correct device automatically
        self.register_buffer(
            "_boundaries",
            torch.tensor(boundaries, dtype=torch.long),
            persistent=False,
        )

    @property
    def config(self) -> FeatureGateConfig:
        """Return the gate configuration."""
        return self._cfg

    @property
    def total_dim(self) -> int:
        """Expected input state dimensionality."""
        return self._total_dim

    def gate_values(self, regime_context: Tensor) -> Tensor:
        """Compute gate values for a given regime context.

        Parameters
        ----------
        regime_context : Tensor of shape (regime_dim,) or (batch, regime_dim)

        Returns
        -------
        Tensor of shape (n_groups,) or (batch, n_groups) in [floor, 1.0].
        """
        squeezed = False
        if regime_context.dim() == 1:
            regime_context = regime_context.unsqueeze(0)
            squeezed = True

        # Clamp to prevent NaN from extreme values
        r = regime_context.clamp(-self._regime_clamp, self._regime_clamp)

        # MLP → sigmoid → scale to [floor, 1]
        raw = self._gate_mlp(r)  # (B, n_groups)
        g = torch.sigmoid(raw)  # (B, n_groups) in (0, 1)
        g = self._floor + (1.0 - self._floor) * g  # (B, n_groups) in [floor, 1]

        if squeezed:
            g = g.squeeze(0)

        return g

    def forward(
        self,
        state_flat: Tensor,
        regime_context: Tensor,
    ) -> Tensor:
        """Apply regime-conditioned gating to the state vector.

        Parameters
        ----------
        state_flat : Tensor of shape (batch, total_dim) or (total_dim,)
        regime_context : Tensor of shape (batch, regime_dim) or (regime_dim,)

        Returns
        -------
        Tensor of same shape as state_flat, with each feature group scaled.
        """
        squeezed = False
        if state_flat.dim() == 1:
            state_flat = state_flat.unsqueeze(0)
            regime_context = regime_context.unsqueeze(0) if regime_context.dim() == 1 else regime_context
            squeezed = True
        elif regime_context.dim() == 1:
            regime_context = regime_context.unsqueeze(0).expand(state_flat.shape[0], -1)

        B = state_flat.shape[0]

        # Compute gates: (B, n_groups)
        gates = self.gate_values(regime_context)
        self._last_gates = gates.detach()

        # Expand gates to match feature dimensions
        # gates[k] is broadcast to all dims in group k
        gate_expanded = torch.zeros(B, self._total_dim, device=state_flat.device)
        for k, d in enumerate(self._group_dims):
            start = self._boundaries[k].item()
            end = self._boundaries[k + 1].item()
            gate_expanded[:, start:end] = gates[:, k : k + 1].expand(-1, d)

        gated = state_flat * gate_expanded

        if squeezed:
            gated = gated.squeeze(0)

        return gated

    def entropy_loss(self) -> Tensor:
        """Compute entropy regularization loss on the last forward pass's gates.

        Encourages gates away from 0 or 1 (prevents collapse).
        Returns weighted scalar loss.  Returns 0.0 if no forward pass yet.

        Math:
            L = −λ Σ_k [ĝ_k log ĝ_k + (1−ĝ_k) log(1−ĝ_k)]
            where ĝ_k = (g_k − floor) / (1 − floor)
        """
        if self._last_gates is None:
            return torch.tensor(0.0)

        # Normalize to [0, 1] range for entropy computation
        eps = 1e-8
        if self._floor >= 1.0 - eps:
            return torch.tensor(0.0)

        g_norm = (self._last_gates - self._floor) / (1.0 - self._floor)
        g_norm = g_norm.clamp(eps, 1.0 - eps)

        # Binary entropy per gate
        ent = -(g_norm * g_norm.log() + (1 - g_norm) * (1 - g_norm).log())

        # Mean over batch and groups, negate (we want to maximize entropy)
        # Negate because this is a *loss* to minimize (negative entropy)
        return -self._entropy_weight * ent.mean()

    def gate_diagnostics(self, regime_context: Tensor) -> dict[str, Any]:
        """Return diagnostic information for the given regime context.

        Parameters
        ----------
        regime_context : Tensor of shape (regime_dim,)

        Returns
        -------
        dict with keys:
            group_names : list[str]
            gate_values : list[float]
            entropy : float
        """
        group_names = ["surprise", "belief", "market", "entity_count", "adversarial"]
        if len(group_names) < len(self._group_dims):
            # Extend with generic names if more groups than defaults
            for i in range(len(group_names), len(self._group_dims)):
                group_names.append(f"group_{i}")

        with torch.no_grad():
            g = self.gate_values(regime_context)  # (n_groups,)

            # Compute entropy of normalized gates
            eps = 1e-8
            g_norm = (g - self._floor) / max(1.0 - self._floor, eps)
            g_norm = g_norm.clamp(eps, 1.0 - eps)
            ent = -(g_norm * g_norm.log() + (1 - g_norm) * (1 - g_norm).log())

        return {
            "group_names": group_names[: len(self._group_dims)],
            "gate_values": g.tolist(),
            "entropy": float(ent.mean()),
        }
