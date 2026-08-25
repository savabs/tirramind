"""
TirraMind — Heterogeneous CDE Drift Function (M1 Component)

Replaces the standalone CDEFunc in CDEMemoryEncoder with a drift function
that is conditioned on:

    1. The current entity state z (the CDE hidden variable)
    2. A graph message m_i from the entity's neighbourhood (frozen per window)
    3. A Mamba context vector pre-computed from the entity's event history

The drift function maps these three inputs to the CDE vector field matrix F:

    F(z, m_i, ctx) ∈ R^{hidden_dim × d_z}

such that the CDE increment is:

    dz = F(z, m_i, ctx) · dZ(t)

where dZ(t) is the control path derivative at time t.

Curriculum phases
-----------------
    Phase B:  m_i = zeros,  ctx = zeros  →  pure CDE with MLP drift
    Phase C:  m_i = prev memory state,  ctx = Mamba(event history)
    Phase D:  same + Hawkes-driven Z(t) with path signatures

Context is SET ONCE before calling the Euler-Maruyama solver (not at every
step), keeping Mamba O(1) per entity per window.

Architecture
------------
    combined = cat([z, m_i, ctx])                      → R^{3H}
    h        = Tanh( Linear(combined) )                → R^{4H}
    F_flat   = Linear(h)                               → R^{H × d_z}
    F        = reshape(F_flat, [hidden_dim, d_z])      → drift matrix

The 4H inner dimension follows Kidger et al. 2020 §3 recommendation.
Column-norm clipping to 1.0 is applied during the forward pass to prevent
CDE solver instability (see spec Risk 1).

References
----------
    Kidger et al. (2020). "Neural Controlled Differential Equations for
        Irregular Time Series."  NeurIPS 2020.  §3.1: CDEFunc architecture.
    Chen et al. (2018). "Neural ODEs."  NeurIPS 2018.  Adjoint method.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

_COL_NORM_CLIP: float = 1.0  # Maximum column norm of the drift matrix F


class HeterogeneousCDEFunc(nn.Module):
    """Graph-conditioned CDE vector field for M1.

    Computes the drift matrix F(z, m_i, ctx) at each Euler-Maruyama step.

    Parameters
    ----------
    hidden_dim : int
        Entity state / memory dimension H.
    d_z : int
        Control path dimension d_z.
    memory_dim : int
        Dimension of the graph message m_i and Mamba context ctx.
        Usually equals hidden_dim.
    """

    def __init__(
        self,
        hidden_dim: int,
        d_z: int,
        memory_dim: int,
    ) -> None:
        super().__init__()
        self._hidden_dim = hidden_dim
        self._d_z = d_z
        self._memory_dim = memory_dim

        # Input: [z | m_i | ctx]
        in_features = hidden_dim + memory_dim + memory_dim
        inner = in_features * 4  # Kidger et al. 2020 §3 recommendation

        self.net = nn.Sequential(
            nn.Linear(in_features, inner),
            nn.Tanh(),
            nn.Linear(inner, hidden_dim * d_z),
        )

        # Non-zero bias init: prevents dZ/dt = 0 when z=0 (cold-start issue)
        torch.nn.init.uniform_(self.net[0].bias, -0.1, 0.1)

        # Context buffers — set externally before each CDE solve
        self._graph_msg: torch.Tensor | None = None  # (batch, memory_dim)
        self._mamba_ctx: torch.Tensor | None = None  # (batch, memory_dim)

    # ──────────────────────────────────────────────────────────────────────

    def set_context(
        self,
        graph_msg: torch.Tensor | None,
        mamba_ctx: torch.Tensor | None,
    ) -> None:
        """Set graph message and Mamba context before starting the CDE solve.

        Both tensors are fixed for the duration of the CDE integration
        (one entity, one window).  Call this before every entity's solve.

        Args:
            graph_msg: (batch, memory_dim) neighbourhood message or prev state.
                       Pass None to use zeros (Phase B).
            mamba_ctx: (batch, memory_dim) Mamba summary of event history.
                       Pass None to use zeros (Phase B/C without Mamba).
        """
        self._graph_msg = graph_msg
        self._mamba_ctx = mamba_ctx

    # ──────────────────────────────────────────────────────────────────────

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute the CDE drift matrix F at state z and time t.

        Called by the Euler-Maruyama solver at each step.  Returns F such
        that the CDE increment is  dz = F @ dZ  (matrix-vector product).

        Args:
            t: Scalar time tensor (not used in autonomous formulation).
            z: (batch, hidden_dim) current latent state.

        Returns:
            (batch, hidden_dim, d_z) drift matrix F.
        """
        batch = z.shape[0]
        device = z.device

        # Resolve context — zeros if not set (Phase B)
        if self._graph_msg is not None:
            m = self._graph_msg.to(device)
            if m.shape[0] != batch:
                m = m.expand(batch, -1)
        else:
            m = torch.zeros(batch, self._memory_dim, device=device)

        if self._mamba_ctx is not None:
            ctx = self._mamba_ctx.to(device)
            if ctx.shape[0] != batch:
                ctx = ctx.expand(batch, -1)
        else:
            ctx = torch.zeros(batch, self._memory_dim, device=device)

        # Concatenate and compute drift
        combined = torch.cat([z, m, ctx], dim=-1)  # (batch, H+M+M)
        F_flat = self.net(combined)                # (batch, H * d_z)
        F = F_flat.view(batch, self._hidden_dim, self._d_z)  # (batch, H, d_z)

        # Column-norm clipping: each column of F has L2 norm ≤ 1.0
        # Prevents drift instability per spec Risk 1
        col_norms = F.norm(dim=1, keepdim=True).clamp(min=1.0)  # (batch, 1, d_z)
        F = F / col_norms * _COL_NORM_CLIP

        return F

    # ──────────────────────────────────────────────────────────────────────

    def clear_context(self) -> None:
        """Reset context buffers to None after a CDE solve completes."""
        self._graph_msg = None
        self._mamba_ctx = None
