"""
TirraMind — Heterogeneous Temporal Graph Network (Phase 12c)

Provides:
    HeteroMemory — Type-aware temporal memory for heterogeneous graphs.
                   Replaces PyG's TGNMemory which only handles homogeneous
                   (flat-integer) node IDs.
    HetTGN       — Full model: per-type projection → HGT convolution →
                   HeteroMemory → event prediction head.

Architecture:
    1. Per-type linear projection maps variable-dim node features to
       a common ``hidden_dim``.
    2. ``num_layers`` HGT convolution layers (Hu et al., WWW 2020) perform
       type-aware message passing with multi-head attention.
    3. HeteroMemory maintains a GRU-updated memory cell per node,
       incorporating Time2Vec-encoded time deltas between events.
    4. Event prediction head predicts (which entity, which obs_type,
       when) for the next event — the self-supervised pre-training signal.

References:
    HGT:       Hu et al. 2020, arXiv:2003.01332.
    TGN:       Rossi et al. 2020, arXiv:2006.10637 (inspiration for memory).
    Time2Vec:  Kazemi et al. 2019, arXiv:1907.05321.
    Spec step: 12c.1, 12c.2.
"""

from __future__ import annotations

import logging
import math as _math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv
from torch_geometric.utils import softmax as pyg_softmax

from agent.models.gnn.graph_builder import ENTITY_TYPES, IDMap, OBSERVATION_TYPES
from agent.models.gnn.temporal import Time2Vec

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# AttentionCapturingHGTConv
# ═══════════════════════════════════════════════════════════════


class AttentionCapturingHGTConv(HGTConv):
    """HGTConv that optionally captures per-edge attention weights.

    When ``capture_attention`` is True, the post-softmax attention
    tensor from ``message()`` is detached and stored.  After ``forward()``
    completes, call ``get_edge_attention()`` to retrieve per-edge-type
    attention tensors.

    Overhead is zero when ``capture_attention`` is False (the default),
    so this class can be used as a drop-in replacement for HGTConv
    during training.

    Implementation note:
        PyG 2.7 HGTConv concatenates all edge types into a single
        bipartite graph, calls ``propagate()`` once, then slices
        back.  ``construct_bipartite_edge_index`` preserves the
        iteration order of ``edge_index_dict``, so we record edge
        counts per type in ``forward()`` and split the captured
        attention tensor in ``get_edge_attention()``.

    References:
        * HGT: Hu et al. 2020, arXiv:2003.01332.
        * PyG source: torch_geometric.nn.conv.hgt_conv (v2.7).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.capture_attention: bool = False
        self._captured_alpha: torch.Tensor | None = None
        self._fwd_edge_types: list[tuple[str, str, str]] = []
        self._fwd_edge_counts: list[int] = []

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor | None]:
        if self.capture_attention:
            self._fwd_edge_types = list(edge_index_dict.keys())
            self._fwd_edge_counts = [
                edge_index_dict[et].size(1) for et in self._fwd_edge_types
            ]
            self._captured_alpha = None
        return super().forward(x_dict, edge_index_dict)

    def message(
        self,
        k_j: torch.Tensor,
        q_i: torch.Tensor,
        v_j: torch.Tensor,
        edge_attr: torch.Tensor,
        index: torch.Tensor,
        ptr: torch.Tensor | None,
        size_i: int | None,
    ) -> torch.Tensor:
        alpha = (q_i * k_j).sum(dim=-1) * edge_attr
        alpha = alpha / _math.sqrt(q_i.size(-1))
        alpha = pyg_softmax(alpha, index, ptr, size_i)
        if self.capture_attention:
            self._captured_alpha = alpha.detach()
        out = v_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)

    def get_edge_attention(self) -> dict[tuple[str, str, str], torch.Tensor]:
        """Split captured attention by edge type.

        Returns:
            Dict mapping edge_type → per-edge mean attention
            (averaged across heads), shape ``(num_edges_of_type,)``.
            Empty dict if no attention has been captured.
        """
        if self._captured_alpha is None:
            return {}
        result: dict[tuple[str, str, str], torch.Tensor] = {}
        offset = 0
        for etype, count in zip(self._fwd_edge_types, self._fwd_edge_counts):
            if count > 0:
                chunk = self._captured_alpha[offset : offset + count]
                result[etype] = chunk.mean(dim=-1)  # avg across heads
            offset += count
        return result


# ═══════════════════════════════════════════════════════════════
# HeteroMemory
# ═══════════════════════════════════════════════════════════════


class HeteroMemory(nn.Module):
    """Per-node temporal memory for heterogeneous graphs.

    Each node gets a ``memory_dim``-length vector that is updated via
    GRU whenever the node is involved in an event.  The time delta
    since the previous update is encoded with Time2Vec and concatenated
    to the incoming message before the GRU step.

    Unlike PyG's TGNMemory, this module accepts (node_type, local_id)
    and maps them to the internal flat tensor via an IDMap.

    The memory tensor is **not** a parameter — it is detached state
    (like a running mean in BatchNorm).  Only the GRU and Time2Vec
    weights are learned.
    """

    def __init__(
        self,
        num_nodes: int,
        memory_dim: int = 64,
        message_dim: int = 64,
        time_dim: int = 16,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.message_dim = message_dim
        self.time_dim = time_dim

        # GRU cell: input = [message || time_encoding]
        self.gru = nn.GRUCell(message_dim + time_dim, memory_dim)
        self.time_enc = Time2Vec(out_features=time_dim)

        # Persistent state (not parameters)
        self.register_buffer("memory", torch.zeros(num_nodes, memory_dim))
        self.register_buffer("last_update", torch.zeros(num_nodes))

    def reset(self) -> None:
        """Zero all memory and timestamps."""
        self.memory.zero_()
        self.last_update.zero_()

    def get_memory(self, node_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve memory vectors and last-update times for given nodes.

        Args:
            node_ids: 1-D int tensor of *global* node IDs.

        Returns:
            (memory, last_update) — shapes (N, memory_dim) and (N,).
        """
        return self.memory[node_ids].detach(), self.last_update[node_ids].detach()

    def update_memory(
        self,
        node_ids: torch.Tensor,
        messages: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> None:
        """Update memory for a batch of nodes.

        Args:
            node_ids:   1-D int tensor of global node IDs, shape (B,).
            messages:   Tensor of shape (B, message_dim).
            timestamps: Tensor of shape (B,) — event times.
        """
        if node_ids.numel() == 0:
            return

        # Time delta since previous update
        dt = timestamps - self.last_update[node_ids]
        dt = dt.clamp(min=0.0)
        time_feat = self.time_enc(dt)  # (B, time_dim)

        # GRU update
        gru_input = torch.cat(
            [messages, time_feat], dim=-1
        )  # (B, message_dim + time_dim)
        old_mem = self.memory[node_ids]
        new_mem = self.gru(gru_input, old_mem)

        # Write back (in-place, detached from graph for next step)
        with torch.no_grad():
            self.memory[node_ids] = new_mem.detach()
            self.last_update[node_ids] = timestamps.detach()

    def get_all_memory(self) -> torch.Tensor:
        """Return full memory tensor.  Used during forward pass."""
        return self.memory.detach()


# ═══════════════════════════════════════════════════════════════
# SupervisedHead (Phase 15b)
# ═══════════════════════════════════════════════════════════════


class SupervisedHead(nn.Module):
    """Bilinear scorer for co-occurrence prediction.

    Computes  σ(h_src^T W h_dst + b)  where W is a learned
    (hidden_dim × hidden_dim) weight matrix and b is a scalar bias.

    This is the supervised head for outcome-labeled fine-tuning:
    given embeddings of a (source, destination) entity pair, it
    predicts the probability that a target observation occurs on
    the destination entity within a time window after a source
    observation.

    Math:
        P(hit) = sigmoid( src^T W dst + b )

    Choosing bilinear over MLP: fewer parameters (d² + 1 vs
    d × h + h × 1), lower overfitting risk on small entity graphs,
    and the bilinear form directly measures learned interaction
    between entity representations.

    References:
        Research doc: gnn_pattern_and_finetuning.md §Math/Algorithm Survey.
        Pre-training GNNs (Hu et al. ICLR 2020) — two-phase training.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        emb_src: torch.Tensor,
        emb_dst: torch.Tensor,
    ) -> torch.Tensor:
        """Compute co-occurrence probabilities.

        Args:
            emb_src: (B, hidden_dim)
            emb_dst: (B, hidden_dim)

        Returns:
            Probabilities (B,) in [0, 1].
        """
        # src^T W dst + b  →  sigmoid
        scores = (emb_src @ self.weight * emb_dst).sum(dim=-1) + self.bias
        return torch.sigmoid(scores)


# ═══════════════════════════════════════════════════════════════
# HetTGN — full model
# ═══════════════════════════════════════════════════════════════


class HetTGN(nn.Module):
    """Heterogeneous Temporal Graph Network.

    Combines:
        - Per-type input projections
        - HGT convolution layers
        - HeteroMemory for temporal state
        - Event prediction head (entity, obs_type, time_delta)

    Parameters
    ----------
    metadata : tuple[list[str], list[tuple[str,str,str]]]
        PyG metadata = (node_types, edge_types).
    in_channels : dict[str, int]
        Input feature dims per node type.
    hidden_dim : int
        Hidden representation dimension (all types project here).
    time_dim : int
        Time2Vec output dimension.
    memory_dim : int
        HeteroMemory vector size per node.
    message_dim : int
        HeteroMemory message dimension.
    num_heads : int
        Multi-head attention heads for HGT.
    num_layers : int
        Number of HGT convolution layers.
    num_nodes : int
        Total nodes across all types (for memory allocation).
    num_obs_types : int
        Number of distinct observation types for prediction head.
    """

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        in_channels: dict[str, int],
        hidden_dim: int = 64,
        time_dim: int = 16,
        memory_dim: int = 64,
        message_dim: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        num_nodes: int = 0,
        num_obs_types: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        _num_obs_types = num_obs_types or len(OBSERVATION_TYPES)

        # ── Per-type input projections ──────────────────────────
        self.type_projections = nn.ModuleDict()
        for ntype in self.node_types:
            in_dim = in_channels.get(ntype, hidden_dim)
            self.type_projections[ntype] = nn.Linear(in_dim, hidden_dim)

        # ── HeteroMemory ───────────────────────────────────────
        self.memory = HeteroMemory(
            num_nodes=max(num_nodes, 1),
            memory_dim=memory_dim,
            message_dim=message_dim,
            time_dim=time_dim,
        )

        # ── Combiner: projected features + memory → HGT input ─
        self.combiner = nn.Linear(hidden_dim + memory_dim, hidden_dim)

        # ── HGT layers (attention-capturing) ─────────────────
        self.hgt_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.hgt_layers.append(
                AttentionCapturingHGTConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=metadata,
                    heads=num_heads,
                )
            )

        # ── Event prediction head ──────────────────────────────
        # Predicts: which obs_type + time_delta + value (per-node)
        self.obs_type_head = nn.Linear(hidden_dim, _num_obs_types)
        self.time_delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # time delta is non-negative
        )
        # Value prediction head (Phase 20): predict magnitude of next observation
        self.value_pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # ── Entity-pair score for link prediction ──────────────
        # Score(u, v) = u^T W v
        self.link_weight = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # ── Supervised bilinear head (Phase 15b) ──────────────
        # Score(u, v) = σ(u^T W v + b)  →  P(co-occurrence)
        self.supervised_head = SupervisedHead(hidden_dim)

    def forward(
        self,
        data: HeteroData,
        id_map: IDMap,
    ) -> dict[str, torch.Tensor]:
        """Forward pass: project, combine with memory, run HGT.

        Args:
            data: PyG HeteroData with ``data[type].x`` node features.
            id_map: Mapping (type, entity_id) → global node ID.

        Returns:
            Dict mapping node_type → embedding tensor [N_type, hidden_dim].
        """
        x_dict: dict[str, torch.Tensor] = {}

        for ntype in self.node_types:
            if ntype not in data.node_types:
                continue
            x = data[ntype].x  # (N_type, in_dim)
            projected = self.type_projections[ntype](x)  # (N_type, hidden_dim)

            # Retrieve memory for these nodes
            local_map = id_map.type_local.get(ntype, {})
            if local_map:
                # Build global IDs in local order
                global_ids = torch.zeros(len(local_map), dtype=torch.long)
                for eid, local_idx in local_map.items():
                    gid = id_map.global_id(ntype, eid)
                    if gid is not None:
                        global_ids[local_idx] = gid
                mem, _ = self.memory.get_memory(global_ids)
                combined = torch.cat([projected, mem], dim=-1)
            else:
                zero_mem = torch.zeros(
                    projected.size(0),
                    self.memory_dim,
                    device=projected.device,
                )
                combined = torch.cat([projected, zero_mem], dim=-1)

            x_dict[ntype] = F.relu(self.combiner(combined))

        # Build edge_index_dict from data
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor] = {}
        for etype in self.edge_types:
            if etype in data.edge_types:
                edge_index_dict[etype] = data[etype].edge_index

        # HGT layers
        if edge_index_dict:
            for hgt in self.hgt_layers:
                x_dict_new = hgt(x_dict, edge_index_dict)
                # Preserve node types not returned by HGTConv
                # (e.g. types that only appear as edge sources)
                for ntype in x_dict:
                    if ntype not in x_dict_new:
                        x_dict_new[ntype] = x_dict[ntype]
                x_dict = x_dict_new

        return x_dict

    def predict_obs_type(
        self,
        embeddings: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Predict next observation type per node.

        Returns:
            Dict node_type → logits [N_type, num_obs_types].
        """
        return {ntype: self.obs_type_head(emb) for ntype, emb in embeddings.items()}

    def predict_time_delta(
        self,
        embeddings: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Predict time until next event per node.

        Returns:
            Dict node_type → time_delta [N_type, 1].
        """
        return {ntype: self.time_delta_head(emb) for ntype, emb in embeddings.items()}

    def predict_value(
        self,
        embeddings: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Predict value (magnitude) of next observation per node.

        Returns:
            Dict node_type → value [N_type, 1].
        """
        return {ntype: self.value_pred_head(emb) for ntype, emb in embeddings.items()}

    def link_score(self, emb_u: torch.Tensor, emb_v: torch.Tensor) -> torch.Tensor:
        """Bilinear link score: u^T W v.

        Args:
            emb_u: (B, hidden_dim)
            emb_v: (B, hidden_dim)

        Returns:
            Scalar scores (B,).
        """
        return (self.link_weight(emb_u) * emb_v).sum(dim=-1)

    def predict_outcome(
        self,
        emb_src: torch.Tensor,
        emb_dst: torch.Tensor,
    ) -> torch.Tensor:
        """Predict co-occurrence probability for (src, dst) pairs.

        Uses the supervised bilinear head: σ(src^T W dst + b).

        Args:
            emb_src: (B, hidden_dim) — source entity embeddings.
            emb_dst: (B, hidden_dim) — destination entity embeddings.

        Returns:
            Probabilities (B,) in [0, 1].
        """
        return self.supervised_head(emb_src, emb_dst)

    def update_memory_from_events(
        self,
        events: list[dict[str, Any]],
        embeddings: dict[str, torch.Tensor],
        id_map: IDMap,
    ) -> None:
        """Push event information into HeteroMemory.

        For each event, take the current embedding of the involved entity
        as the message and update its memory cell.

        Args:
            events: Observation dicts with entity_id, entity_type,
                    observed_at, observation_type.
            embeddings: Current node embeddings from forward().
            id_map: Entity → global ID mapping.
        """
        if not events:
            return

        node_ids_list: list[int] = []
        messages_list: list[torch.Tensor] = []
        timestamps_list: list[float] = []

        for ev in events:
            etype = ev.get("entity_type")
            eid = ev.get("entity_id")
            t = ev.get("observed_at", 0.0)

            if etype is None or eid is None:
                continue

            gid = id_map.global_id(etype, eid)
            local_idx = id_map.local_id(etype, eid)
            if gid is None or local_idx is None:
                continue
            if etype not in embeddings:
                continue

            emb = embeddings[etype]
            if local_idx >= emb.size(0):
                continue

            # Use the node's current embedding as message
            msg = emb[local_idx]  # (hidden_dim,)
            # Project to message_dim if needed
            if msg.size(0) != self.memory.message_dim:
                # Truncate or pad
                if msg.size(0) > self.memory.message_dim:
                    msg = msg[: self.memory.message_dim]
                else:
                    pad = torch.zeros(
                        self.memory.message_dim - msg.size(0),
                        device=msg.device,
                    )
                    msg = torch.cat([msg, pad])

            node_ids_list.append(gid)
            messages_list.append(msg.detach())
            timestamps_list.append(t)

        if not node_ids_list:
            return

        node_ids = torch.tensor(node_ids_list, dtype=torch.long)
        messages = torch.stack(messages_list)
        timestamps = torch.tensor(timestamps_list, dtype=torch.float)

        self.memory.update_memory(node_ids, messages, timestamps)

    def reset_memory(self) -> None:
        """Reset all temporal memory (e.g. between training episodes)."""
        self.memory.reset()

    def get_attention_weights(
        self,
        data: HeteroData,
        id_map: IDMap,
    ) -> dict[tuple[str, str, str], float]:
        """Compute per-edge-type mean attention from HGT layers.

        Enables attention capture on all ``AttentionCapturingHGTConv``
        layers, runs a forward pass, collects per-edge-type attention,
        and returns the mean attention weight for each edge type
        (averaged across edges, heads, and layers).

        The model is set to eval mode during capture and restored
        afterwards.  Capture is disabled before returning so that
        subsequent training steps incur no overhead.

        Returns:
            Dict mapping (src_type, rel, dst_type) → mean attention (float).
        """
        was_training = self.training
        self.eval()

        for hgt in self.hgt_layers:
            hgt.capture_attention = True

        try:
            with torch.no_grad():
                self.forward(data, id_map)

            accum: dict[tuple[str, str, str], list[float]] = {}
            for hgt in self.hgt_layers:
                edge_attn = hgt.get_edge_attention()
                for etype, attn in edge_attn.items():
                    accum.setdefault(etype, []).append(attn.mean().item())

            return {etype: sum(vals) / len(vals) for etype, vals in accum.items()}
        finally:
            for hgt in self.hgt_layers:
                hgt.capture_attention = False
            if was_training:
                self.train()
