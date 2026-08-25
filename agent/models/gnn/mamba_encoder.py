"""
TirraMind — Mamba SSM Memory Encoder (Idea 4)

Replaces HeteroMemory's fixed-size GRU step with a selective State Space Model
(Mamba) that can encode arbitrarily long event histories without O(n²) attention.

Problem
-------
HeteroMemory compresses each entity's **entire history** into a fixed
``memory_dim=64`` vector updated by a GRU cell:

    h_t  = GRU(msg_t ‖ Δt, h_{t-1})

A GRU cell performs **one** update per event and must summarise all prior
history in 64 floats.  For a vessel with 3 years of AIS pings, decades of
routing patterns are forced through a 64-float bottleneck.  Long-range
temporal patterns are structurally unreachable because GRU's update gate
is driven by the current message alone — it cannot selectively retain
old state based on the context of today's event.

Solution — Mamba Selective SSM
-------------------------------
Mamba (Gu & Dao, 2023, NeurIPS) is an S4-variant with **input-selective
state transitions**:

    h'(t) = A(x_t)·h(t) + B(x_t)·x_t
    y(t)  = C(x_t)·h(t) + D·x_t

where A, B, C are *input-dependent* — the model learns to forget or retain
state based on the content of each token.  This lets it:
  1. Ignore irrelevant mid-voyage pings and remember a rare sanctions event.
  2. Carry context across thousands of tokens at linear O(n) cost.
  3. Compress years of history into a single ``memory_dim``-vector without
     discarding long-range patterns.

Architectural Integration
--------------------------
The encoder operates per-node on the sequence of events received in the
current training window.  Each event contributes a token:

    token_k = Linear( msg_k ‖ Time2Vec(Δt_k) )  →  ℝ^{memory_dim}

The current memory vector is prepended as a "history token" so the Mamba
block conditions each window on accumulated prior state:

    sequence = [h_prev, token_1, ..., token_K]

After Mamba processes the sequence, the last output token becomes h_new:

    h_new = Mamba(sequence)[-1]

This preserves the same semantics as the GRU path (prev state + new messages)
while giving the model the expressiveness to selectively retain structure.

Fallback
--------
When ``mambapy`` is unavailable the module silently falls back to GRU,
preserving existing behaviour.  Training with ``use_mamba=False`` (default)
is also unaffected — existing TrainerConfig checkpoints are backward-compatible.

References
----------
    Gu, A. & Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling with
        Selective State Spaces." NeurIPS 2023. arXiv:2312.00752.
    Ma, X. et al. (2024). "Mamba-2." ICML 2024. arXiv:2405.21060.
    mambapy (MIT) — pure-PyTorch re-implementation. pip install mambapy.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from agent.models.gnn.het_tgn import HeteroMemory
    from agent.models.gnn.graph_builder import IDMap

log = logging.getLogger(__name__)

# Maximum sequence length fed to Mamba per node per window.
# Events older than this within the window are discarded (recency bias).
_MAX_SEQ_LEN: int = 128


# ─── Time encoding (copy-free import to avoid circular) ────────────────────


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
# MambaMemoryEncoder
# ═══════════════════════════════════════════════════════════════════════════


class MambaMemoryEncoder(nn.Module):
    """Replace the GRU in HeteroMemory with a Mamba selective SSM block.

    Parameters
    ----------
    memory_dim : int
        Dimension of the HeteroMemory state vector.  Mamba d_model = memory_dim.
    message_dim : int
        Dimension of incoming messages (current node embedding slices).
    time_dim : int
        Dimension of the Time2Vec encoding prepended to each message.
    n_layers : int
        Number of stacked Mamba blocks.  1 is fast; 2 adds modelling power.
    d_state : int
        Mamba SSM state size per channel.  16 is the Mamba-paper default.
    expand_factor : int
        Channel expansion factor inside Mamba block (2 = standard).
    max_seq_len : int
        Clip node event sequences at this length (recency truncation).
    """

    def __init__(
        self,
        memory_dim: int = 64,
        message_dim: int = 64,
        time_dim: int = 16,
        n_layers: int = 1,
        d_state: int = 16,
        expand_factor: int = 2,
        max_seq_len: int = _MAX_SEQ_LEN,
    ) -> None:
        super().__init__()
        self.memory_dim = memory_dim
        self.message_dim = message_dim
        self.time_dim = time_dim
        self.max_seq_len = max_seq_len

        # Project (message_dim + time_dim) → memory_dim so Mamba sees uniform width
        self.input_proj = nn.Linear(message_dim + time_dim, memory_dim)

        # Time encoding
        self.time_enc = _Time2Vec(out_features=time_dim)

        # ── Try Mamba; fall back to GRU if unavailable ──────────────────
        self._has_mamba = False
        try:
            from mambapy.mamba import Mamba, MambaConfig  # noqa: PLC0415

            cfg = MambaConfig(
                d_model=memory_dim,
                n_layers=n_layers,
                d_state=d_state,
                expand_factor=expand_factor,
            )
            self.mamba: nn.Module = Mamba(cfg)
            self._has_mamba = True
            log.debug(
                "MambaMemoryEncoder: using Mamba SSM "
                "(d_model=%d, n_layers=%d, d_state=%d)",
                memory_dim,
                n_layers,
                d_state,
            )
        except ImportError:
            log.warning(
                "mambapy not available — MambaMemoryEncoder falling back to GRU. "
                "Install with: pip install mambapy"
            )
            self.gru_cell = nn.GRUCell(memory_dim, memory_dim)

    # ──────────────────────────────────────────────────────────────────────
    # M1 API — encode a token sequence, return the last output vector
    # ──────────────────────────────────────────────────────────────────────

    def encode_tokens(self, token_seq: torch.Tensor) -> torch.Tensor:
        """Run Mamba (or GRU fallback) over a token sequence and return the
        last output vector.

        Used by HeterogeneousCDEFunc to pre-compute the Mamba context before
        the CDE solve begins.  The returned vector summarises all past events
        in the sequence and is held FIXED throughout the CDE integration
        (one call per entity per window, not per solver step).

        Args:
            token_seq: (seq_len, memory_dim) or (1, seq_len, memory_dim)

        Returns:
            (memory_dim,) — last output token from Mamba (or GRU fallback).
        """
        if token_seq.dim() == 2:
            token_seq = token_seq.unsqueeze(0)  # (1, seq_len, memory_dim)

        if self._has_mamba:
            out = self.mamba(token_seq)  # (1, seq_len, memory_dim)
            return out[0, -1]  # (memory_dim,)
        else:
            h = torch.zeros(self.memory_dim, device=token_seq.device)
            for tok in token_seq[0]:
                h = self.gru_cell(tok.unsqueeze(0), h.unsqueeze(0)).squeeze(0)
            return h  # (memory_dim,)

    def build_token(
        self,
        msg: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Build a single Mamba input token from a message and time delta.

        token = input_proj( msg || Time2Vec(dt) )  → R^{memory_dim}

        Args:
            msg: (message_dim,) entity message/embedding.
            dt:  seconds since previous event.

        Returns:
            (memory_dim,) projected token.
        """
        dt_t = torch.tensor([dt], device=msg.device, dtype=msg.dtype)
        time_feat = self.time_enc(dt_t)  # (1, time_dim)
        msg_cat = torch.cat([msg.unsqueeze(0), time_feat], dim=-1)  # (1, msg+time)
        return self.input_proj(msg_cat).squeeze(0)  # (memory_dim,)

    # ──────────────────────────────────────────────────────────────────────
    # Main entry point (mirrors CDEMemoryEncoder.update_memory_from_events)
    # ──────────────────────────────────────────────────────────────────────

    def update_memory_from_events(
        self,
        events: list[dict[str, Any]],
        embeddings: dict[str, torch.Tensor],
        id_map: "IDMap",
        memory: "HeteroMemory",
    ) -> None:
        """Update HeteroMemory for every node that received events.

        Args:
            events:     Observation dicts (entity_id, entity_type, observed_at).
            embeddings: Current GNN output embeddings per entity type.
            id_map:     Entity → global ID mapping.
            memory:     HeteroMemory module whose state buffers are updated.
        """
        if not events:
            return

        # ── 1. Group events per node, collecting (message, timestamp) ──
        node_events: dict[int, list[tuple[torch.Tensor, float]]] = defaultdict(list)

        for ev in events:
            etype = ev.get("entity_type")
            eid = ev.get("entity_id")
            t = float(ev.get("observed_at", 0.0))

            if etype is None or eid is None:
                continue
            if etype not in embeddings:
                continue

            gid = id_map.global_id(etype, eid)
            local_idx = id_map.local_id(etype, eid)
            if gid is None or local_idx is None:
                continue

            emb = embeddings[etype]
            if local_idx >= emb.size(0):
                continue

            msg = emb[local_idx].detach()
            if msg.size(0) != self.message_dim:
                if msg.size(0) > self.message_dim:
                    msg = msg[: self.message_dim]
                else:
                    msg = torch.cat(
                        [
                            msg,
                            torch.zeros(
                                self.message_dim - msg.size(0), device=msg.device
                            ),
                        ]
                    )

            node_events[gid].append((msg, t))

        if not node_events:
            return

        device = next(iter(node_events.values()))[0][0].device

        # ── 2. Process each node's event sequence ──────────────────────
        for gid, evt_list in node_events.items():
            # Clip to max_seq_len (keep most recent)
            evt_list = evt_list[-self.max_seq_len :]

            msgs = torch.stack([e[0] for e in evt_list])  # (K, message_dim)
            times = torch.tensor([e[1] for e in evt_list], device=device)

            self._update_single_node(gid, msgs, times, memory, device)

    def _update_single_node(
        self,
        gid: int,
        msgs: torch.Tensor,
        times: torch.Tensor,
        memory: "HeteroMemory",
        device: torch.device,
    ) -> None:
        """Run Mamba (or GRU fallback) for one node and write new memory.

        Sequence fed to Mamba:
            [h_prev_token, token_1, ..., token_K]
        where token_k = input_proj(msg_k ‖ Time2Vec(Δt_k)).

        h_prev is the existing memory vector, projected as-is since it is
        already in ℝ^{memory_dim}.
        """
        # Current memory for this node
        if gid < memory.num_nodes:
            h_prev = memory.memory[gid].detach()  # (memory_dim,)
            t_prev = float(memory.last_update[gid].item())
        else:
            h_prev = torch.zeros(self.memory_dim, device=device)
            t_prev = 0.0

        # Time deltas relative to last update
        dt = (times - t_prev).clamp(min=0.0)  # (K,)
        time_feat = self.time_enc(dt)  # (K, time_dim)

        # Input tokens: project each (msg ‖ time_feat) → memory_dim
        tokens = self.input_proj(
            torch.cat([msgs, time_feat], dim=-1)
        )  # (K, memory_dim)

        if self._has_mamba:
            # Prepend h_prev as context token → sequence of length K+1
            h_prev_tok = h_prev.unsqueeze(0)  # (1, memory_dim)
            seq = torch.cat([h_prev_tok, tokens], dim=0)  # (K+1, memory_dim)
            seq = seq.unsqueeze(0)  # (1, K+1, memory_dim) — batch=1

            out = self.mamba(seq)  # (1, K+1, memory_dim)
            h_new = out[0, -1]  # last token → new memory
        else:
            # GRU fallback: process tokens sequentially
            h = h_prev
            for tok in tokens:
                h = self.gru_cell(tok.unsqueeze(0), h.unsqueeze(0)).squeeze(0)
            h_new = h

        # Write back to HeteroMemory buffer
        t_last = float(times[-1].item())
        with torch.no_grad():
            if gid < memory.num_nodes:
                memory.memory[gid] = h_new.detach()
                memory.last_update[gid] = t_last
            # Silently skip out-of-range nodes (new entities added after build)
