"""
TirraMind — M1: Continuous-Time Heterogeneous World Model

Neural SDE on a Heterogeneous Event Graph.  Replaces the GRU-based
HeteroMemory update in HetTGN with a continuous-time CDE+SDE integration
over the full event path within each training window.

Equation
--------
    dX_i(t) = f_theta( X_i(t), G(t), Z(t) ) dZ(t)   [CDE drift — always on]
            + g_phi( X_i(t) ) dW_i(t)                 [SDE noise — Phase E]

where:
    X_i(t)   entity state (memory_dim,), initialised from HeteroMemory
    Z(t)     control path built from Time2Vec(dt) + projected messages
             [Phase D adds Hawkes hidden states + path signatures]
    G(t)     graph-conditioned neighbourhood message, frozen per window
    f_theta  HeterogeneousCDEFunc: MLP conditioned on (X, G, Mamba_ctx)
    g_phi    DiagonalDiffusionHead: per-dimension positive noise scale
    W_i(t)   standard Brownian motion per entity

Curriculum Phases
-----------------
    B  CDE only, no graph context, no Mamba, no diffusion.
       Z(t) = LinearInterp( Time2Vec(dt) ‖ proj_msg )
       Drift = HeterogeneousCDEFunc with graph_msg=0, mamba_ctx=0

    C  Add Mamba context + graph message from previous window memory.
       Mamba runs ONCE per entity per window (not at every solver step).
       Z(t) unchanged.  Unfreeze Mamba.

    D  Add Hawkes hidden states + path signatures to Z(t).
       Requires NeuralHawkesEncoder._model to be trained.
       Full joint training.

    E  Add DiagonalDiffusionHead.  Anneal lambda_kl 0→0.01 over 10 epochs.
       g_phi outputs per-dimension uncertainty alongside predictions.

Integration
-----------
This module is a WRAPPER around the existing HetTGN.  It:
  1. Reads the previous window's HeteroMemory state as z0.
  2. Replaces update_memory_from_events() with a CDE solve.
  3. Writes z_T back to HeteroMemory.
  4. The regular HetTGN.forward() still runs unchanged after this.

References
----------
    Chen et al. (2018). Neural ODEs. NeurIPS 2018.
    Kidger et al. (2020). Neural CDEs. NeurIPS 2020.
    Li et al. (2020). Scalable Gradients for SDEs. AISTATS 2020.
    Gu & Dao (2024). Mamba. ICLR 2024.
    Mei & Eisner (2017). Neural Hawkes Process. NeurIPS 2017.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from agent.models.gnn.diffusion_head import DiagonalDiffusionHead
from agent.models.gnn.heterogeneous_cde_func import HeterogeneousCDEFunc
from agent.models.gnn.signature_path import SignaturePathBuilder, compute_d_z

if TYPE_CHECKING:
    from agent.models.gnn.het_tgn import HeteroMemory
    from agent.models.gnn.graph_builder import IDMap
    from agent.models.gnn.mamba_encoder import MambaMemoryEncoder

log = logging.getLogger(__name__)


# ─── Minimal Time2Vec (copy-free — avoids circular import via het_tgn) ─────


class _Time2Vec(nn.Module):
    """Learnable periodic time encoding (Kazemi et al., 2019)."""

    def __init__(self, out_features: int = 16) -> None:
        super().__init__()
        self.linear = nn.Linear(1, out_features)
        self.periodic = nn.Linear(1, out_features - 1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(-1).float()
        trend = self.linear(t)
        periodic = torch.sin(self.periodic(t))
        return torch.cat([trend[..., :1], periodic], dim=-1)


# ═══════════════════════════════════════════════════════════════════════════
# ContinuousWorldModel
# ═══════════════════════════════════════════════════════════════════════════


class ContinuousWorldModel(nn.Module):
    """M1 synthesis: wraps HetTGN and replaces GRU memory with CDE+SDE.

    Parameters
    ----------
    hidden_dim : int
        Entity state / memory dimension (= HeteroMemory.memory_dim = 64).
    ctrl_time_dim : int
        Time2Vec output channels for the control path.  Default 16.
    ctrl_msg_dim : int
        Projected message channels for the control path.  Default 32.
    n_euler_steps : int
        Number of Euler-Maruyama steps across the window.  Default 20.
        More steps = better approximation, slower training.
    use_signatures : bool
        Add log-signature features to the control path (Phase D).
        Requires messages fed to forward() to be available.
    use_mamba_ctx : bool
        Pre-compute Mamba context before each CDE solve (Phase C/D).
        Requires mamba_encoder to be provided.
    use_diffusion : bool
        Add stochastic diffusion term g_phi(X) dW (Phase E).
    mamba_encoder : MambaMemoryEncoder | None
        Shared Mamba module.  Required when use_mamba_ctx=True.
    sig_builder : SignaturePathBuilder | None
        Path signature builder.  Required when use_signatures=True.
    hawkes_encoder : Any | None
        NeuralHawkesEncoder for Phase D Hawkes Z(t).  Optional.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        ctrl_time_dim: int = 16,
        ctrl_msg_dim: int = 32,
        n_euler_steps: int = 20,
        use_signatures: bool = False,
        use_mamba_ctx: bool = False,
        use_diffusion: bool = False,
        mamba_encoder: "MambaMemoryEncoder | None" = None,
        sig_builder: SignaturePathBuilder | None = None,
        hawkes_encoder: Any | None = None,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.ctrl_time_dim = ctrl_time_dim
        self.ctrl_msg_dim = ctrl_msg_dim
        self.n_euler_steps = n_euler_steps
        self.use_signatures = use_signatures
        self.use_mamba_ctx = use_mamba_ctx
        self.use_diffusion = use_diffusion

        # Control path dimension: time + message [+ signatures]
        self.d_z: int = compute_d_z(ctrl_time_dim, ctrl_msg_dim, sig_builder if use_signatures else None)

        # Time encoding for control path
        self.time_enc = _Time2Vec(ctrl_time_dim)

        # Message projection for control path (hidden_dim → ctrl_msg_dim)
        self.msg_proj = nn.Linear(hidden_dim, ctrl_msg_dim, bias=False)

        # CDE drift function
        self.cde_func = HeterogeneousCDEFunc(
            hidden_dim=hidden_dim,
            d_z=self.d_z,
            memory_dim=hidden_dim,  # graph_msg and mamba_ctx both in R^hidden_dim
        )

        # Optional components
        self.sig_builder: SignaturePathBuilder | None = None
        if use_signatures and sig_builder is not None:
            self.sig_builder = sig_builder

        self.mamba_encoder: "MambaMemoryEncoder | None" = None
        if use_mamba_ctx and mamba_encoder is not None:
            self.mamba_encoder = mamba_encoder

        self.diffusion_head: DiagonalDiffusionHead | None = None
        if use_diffusion:
            self.diffusion_head = DiagonalDiffusionHead(hidden_dim)

        self.hawkes_encoder = hawkes_encoder

        log.info(
            "ContinuousWorldModel: d_z=%d n_euler_steps=%d "
            "use_signatures=%s use_mamba=%s use_diffusion=%s",
            self.d_z,
            n_euler_steps,
            use_signatures,
            use_mamba_ctx,
            use_diffusion,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def update_memories(
        self,
        events: list[dict[str, Any]],
        memory: "HeteroMemory",
        id_map: "IDMap",
        embeddings: dict[str, torch.Tensor],
        training: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Replace GRU memory update with CDE integration for all active entities.

        For each entity that received at least one event in this window:
          1. Read previous state z0 from HeteroMemory.
          2. Build control path Z(t) from the entity's events.
          3. Pre-compute graph message m_i and Mamba context ctx.
          4. Solve CDE (+SDE if Phase E) via Euler-Maruyama.
          5. Write z_T back to HeteroMemory.

        Args:
            events:     All events for this training window.
            memory:     HeteroMemory module (modified in place).
            id_map:     Entity → global/local ID mapping.
            embeddings: Current GNN output embeddings per entity type.
            training:   If True, add stochastic diffusion (Phase E only).

        Returns:
            {"kl_loss": Tensor(scalar), "n_updated": int}
        """
        # Group events by entity global ID
        node_events: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for ev in events:
            etype = ev.get("entity_type")
            eid = ev.get("entity_id")
            if etype is None or eid is None:
                continue
            gid = id_map.global_id(etype, eid)
            if gid is None:
                continue
            if gid >= memory.num_nodes:
                continue
            node_events[gid].append(ev)

        if not node_events:
            device = memory.memory.device
            return {
                "kl_loss": torch.tensor(0.0, device=device),
                "n_updated": torch.tensor(0, device=device),
            }

        kl_acc = torch.tensor(0.0, device=memory.memory.device)
        n_updated = 0

        for gid, evts in node_events.items():
            # Sort chronologically
            evts = sorted(evts, key=lambda e: float(e.get("observed_at", 0.0)))
            times = torch.tensor(
                [float(e.get("observed_at", 0.0)) for e in evts],
                dtype=torch.float32,
                device=memory.memory.device,
            )

            # Entity's previous state
            z0 = memory.memory[gid].unsqueeze(0)  # (1, hidden_dim)
            t_prev = float(memory.last_update[gid].item())

            # Gather messages from embeddings
            msgs = self._gather_messages(evts, embeddings, id_map, z0)  # (n, hidden_dim)

            # Build control path knots: (n, d_z)
            knots = self._build_knots(msgs, times, t_prev)

            # Pre-compute graph message and Mamba context
            graph_msg, mamba_ctx = self._compute_context(
                gid, evts, msgs, times, t_prev, memory, id_map
            )

            # Set context on CDE func
            self.cde_func.set_context(
                graph_msg=graph_msg.unsqueeze(0),  # (1, hidden_dim)
                mamba_ctx=mamba_ctx.unsqueeze(0),  # (1, hidden_dim)
            )

            # Euler-Maruyama CDE solve
            z_T, kl = self._euler_maruyama(z0, knots, times, t_prev, training)

            kl_acc = kl_acc + kl
            n_updated += 1

            # Write result back to memory (detached to avoid autograd leaking
            # through the memory buffer into future windows)
            with torch.no_grad():
                memory.memory[gid] = z_T.squeeze(0).detach()
                memory.last_update[gid] = float(times[-1].item())

            # Clean up context
            self.cde_func.clear_context()

        device = memory.memory.device
        return {
            "kl_loss": kl_acc / max(n_updated, 1),
            "n_updated": torch.tensor(n_updated, device=device),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Control path construction
    # ──────────────────────────────────────────────────────────────────────

    def _build_knots(
        self,
        msgs: torch.Tensor,        # (n, hidden_dim)
        times: torch.Tensor,       # (n,)
        t_prev: float,
    ) -> torch.Tensor:
        """Build control path knot values Z_k at each event time.

        knot_k = cat( Time2Vec(dt_k) , proj_msg_k [, log_sig_k] )

        Args:
            msgs:   (n, hidden_dim) — entity messages/embeddings.
            times:  (n,) — absolute event timestamps.
            t_prev: previous window's last event time.

        Returns:
            (n, d_z) control path knot tensor.
        """
        device = msgs.device

        # Time deltas relative to previous window end
        dt = (times - t_prev).clamp(min=0.0)  # (n,)
        time_feats = self.time_enc(dt)         # (n, ctrl_time_dim)

        # Projected messages
        msg_feats = self.msg_proj(msgs)         # (n, ctrl_msg_dim)

        parts = [time_feats, msg_feats]

        # Path signatures (Phase D)
        if self.use_signatures and self.sig_builder is not None:
            sigs = self.sig_builder(msg_feats)  # (n, sig_dim)
            parts.append(sigs)

        return torch.cat(parts, dim=-1)         # (n, d_z)

    # ──────────────────────────────────────────────────────────────────────
    # Context computation
    # ──────────────────────────────────────────────────────────────────────

    def _compute_context(
        self,
        gid: int,
        evts: list[dict[str, Any]],
        msgs: torch.Tensor,
        times: torch.Tensor,
        t_prev: float,
        memory: "HeteroMemory",
        id_map: "IDMap",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute graph message m_i and Mamba context for a single entity.

        Phase B: both are zeros.
        Phase C: graph_msg = own prev memory, mamba_ctx = Mamba(event tokens).

        Returns:
            (graph_msg, mamba_ctx) — both shape (hidden_dim,).
        """
        device = memory.memory.device

        # Graph message: use entity's own previous memory state as m_i.
        # (neighbourhood-aggregated HGT context added in later phases)
        if self.use_mamba_ctx:
            graph_msg = memory.memory[gid].detach()  # (hidden_dim,)
        else:
            graph_msg = torch.zeros(self.hidden_dim, device=device)

        # Mamba context: encode event token sequence
        if self.use_mamba_ctx and self.mamba_encoder is not None:
            tokens = self._build_mamba_tokens(msgs, times, t_prev, memory, gid)
            mamba_ctx = self.mamba_encoder.encode_tokens(tokens)  # (memory_dim,)
        else:
            mamba_ctx = torch.zeros(self.hidden_dim, device=device)

        return graph_msg, mamba_ctx

    def _build_mamba_tokens(
        self,
        msgs: torch.Tensor,
        times: torch.Tensor,
        t_prev: float,
        memory: "HeteroMemory",
        gid: int,
    ) -> torch.Tensor:
        """Build Mamba input token sequence for an entity.

        Prepends the entity's previous memory state as a "history token".
        Token_k = input_proj( msg_k ‖ Time2Vec(dt_k) )

        Returns: (K+1, memory_dim) — prev_state token + K event tokens.
        """
        h_prev = memory.memory[gid].detach()  # (memory_dim,)
        dt = (times - t_prev).clamp(min=0.0)  # (n,)

        tokens: list[torch.Tensor] = [h_prev.unsqueeze(0)]  # (1, mem_dim)
        for k in range(len(times)):
            tok = self.mamba_encoder.build_token(msgs[k], float(dt[k].item()))
            tokens.append(tok.unsqueeze(0))  # (1, mem_dim)

        return torch.cat(tokens, dim=0)  # (K+1, memory_dim)

    # ──────────────────────────────────────────────────────────────────────
    # Euler-Maruyama solver
    # ──────────────────────────────────────────────────────────────────────

    def _euler_maruyama(
        self,
        z0: torch.Tensor,      # (1, hidden_dim)
        knots: torch.Tensor,   # (n, d_z)
        times: torch.Tensor,   # (n,)
        t_prev: float,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Euler-Maruyama CDE + optional SDE integration.

        At each event interval [t_{k-1}, t_k]:
            dZ = knots[k] - knots[k-1]          control path increment
            F  = cde_func(t_k, z)               drift matrix (hidden_dim, d_z)
            z  ← z + einsum('bid,bd->bi', F, dZ) drift step
            z  ← z + g(z) * sqrt(dt) * eps       diffusion step (Phase E)

        Returns:
            (z_T, kl_scalar) where z_T: (1, hidden_dim), kl_scalar: Tensor(0-d).
        """
        z = z0   # (1, hidden_dim)
        kl_acc = torch.tensor(0.0, device=z.device)
        n = knots.shape[0]

        if n == 0:
            return z, kl_acc

        # Previous knot (before first event): zeros in d_z space
        prev_knot = torch.zeros(1, self.d_z, device=z.device)

        for k in range(n):
            # Time step
            t_k = float(times[k].item()) if k < len(times) else t_prev
            dt = t_k - (float(times[k - 1].item()) if k > 0 else t_prev)
            dt = max(dt, 1.0)  # floor at 1 second to avoid zero steps

            # Control path increment dZ
            cur_knot = knots[k].unsqueeze(0)          # (1, d_z)
            dZ = cur_knot - prev_knot                  # (1, d_z)
            prev_knot = cur_knot

            # Drift step: dz = F(z) @ dZ
            t_tensor = torch.tensor([t_k], device=z.device)
            F = self.cde_func(t_tensor, z)            # (1, hidden_dim, d_z)
            z = z + torch.einsum("bid,bd->bi", F, dZ) # (1, hidden_dim)

            # Diffusion step (Phase E only)
            if self.use_diffusion and self.diffusion_head is not None:
                noise, sigma = self.diffusion_head.sample_noise(z, dt, training)
                z = z + noise
                # Accumulate KL
                kl_acc = kl_acc + self.diffusion_head.kl_divergence(z, sigma)

        return z, kl_acc

    # ──────────────────────────────────────────────────────────────────────
    # Message gathering helper
    # ──────────────────────────────────────────────────────────────────────

    def _gather_messages(
        self,
        evts: list[dict[str, Any]],
        embeddings: dict[str, torch.Tensor],
        id_map: "IDMap",
        z0: torch.Tensor,
    ) -> torch.Tensor:
        """Gather entity embedding messages for each event.

        If the entity is in the current GNN embeddings, use its embedding.
        Otherwise fall back to z0 (previous memory state).

        Returns: (n, hidden_dim) messages in chronological order.
        """
        msgs: list[torch.Tensor] = []
        device = z0.device

        for ev in evts:
            etype = ev.get("entity_type")
            eid = ev.get("entity_id")
            msg = None

            if etype is not None and eid is not None and etype in embeddings:
                local_idx = id_map.local_id(etype, eid)
                emb = embeddings[etype]
                if local_idx is not None and local_idx < emb.shape[0]:
                    msg = emb[local_idx].detach()  # (embed_dim,)
                    # Pad or truncate to hidden_dim
                    if msg.shape[0] > self.hidden_dim:
                        msg = msg[: self.hidden_dim]
                    elif msg.shape[0] < self.hidden_dim:
                        pad = torch.zeros(
                            self.hidden_dim - msg.shape[0], device=device
                        )
                        msg = torch.cat([msg, pad])

            if msg is None:
                msg = z0.squeeze(0).detach()

            msgs.append(msg.to(device))

        if not msgs:
            return z0.squeeze(0).unsqueeze(0)  # (1, hidden_dim) fallback

        return torch.stack(msgs, dim=0)  # (n, hidden_dim)
