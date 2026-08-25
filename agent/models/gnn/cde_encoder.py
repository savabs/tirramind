"""
TirraMind — Neural Controlled Differential Equation Memory Encoder (Idea 1)

Replaces the per-window GRU memory update in HeteroMemory with a continuous-time
Neural CDE integration over the full event path within each window.

Architecture
------------
Current (discrete):
    For each event (t_i, msg_i): memory = GRU([msg || Time2Vec(dt)], memory)

CDE (continuous):
    Given n events per node: {(t_0, x_0), ..., (t_n, x_n)}
    1. Build interpolated control path X(t) via natural cubic spline
    2. Integrate: dZ = f_θ(Z) dX  from t=0 to t=T
    3. Z(T) becomes the updated memory state

Why this is better:
    - Processes the full event path, not one event at a time
    - Handles irregular sampling natively — no window approximation
    - Mathematically principled (rough path theory, Kidger et al. NeurIPS 2020)
    - Nodes with many events get richer integration; single-event nodes fall back to GRU

References
----------
    Kidger et al. 2020 "Neural Controlled Differential Equations for Irregular
    Time Series" NeurIPS Spotlight. arXiv:2005.08926.
    torchcde 0.2.x API: https://github.com/patrick-kidger/torchcde
    torchdiffeq 0.2.x API: https://github.com/rtqichen/torchdiffeq
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from agent.models.gnn.het_tgn import HeteroMemory

log = logging.getLogger(__name__)

try:
    import torchcde
    import torchdiffeq  # noqa: F401 — imported to verify available; cdeint uses it internally

    _CDE_AVAILABLE = True
except ImportError:
    _CDE_AVAILABLE = False
    log.warning(
        "torchcde/torchdiffeq not installed — CDEMemoryEncoder will fall back to GRU. "
        "Install with: pip install torchdiffeq torchcde"
    )


# ═══════════════════════════════════════════════════════════════
# CDEFunc — Neural vector field for the CDE
# ═══════════════════════════════════════════════════════════════


class CDEFunc(nn.Module):
    """Neural vector field for the controlled differential equation.

    Implements f_θ: R^hidden_dim → R^(hidden_dim × input_channels) such that
    the CDE is: dZ/dt = f_θ(Z) · dX/dt

    The output shape (hidden_dim, input_channels) acts as a matrix multiplication
    with the derivative dX/dt ∈ R^input_channels to produce dZ/dt ∈ R^hidden_dim.

    Architecture follows Kidger et al. 2020 §3.1: two-layer MLP with tanh
    activation and output reshaped to the matrix form.

    Parameters
    ----------
    hidden_dim : int
        Dimension of the latent state Z (= memory_dim in HeteroMemory).
    input_channels : int
        Channels in the control path X (= message_dim + 1 time channel).
    """

    def __init__(self, hidden_dim: int, input_channels: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_channels = input_channels
        _inner = hidden_dim * 4
        self.linear1 = nn.Linear(hidden_dim, _inner)
        self.linear2 = nn.Linear(_inner, hidden_dim * input_channels)
        # Non-zero bias init ensures f(0) ≠ 0 so memory updates from cold start
        # (default PyTorch bias=0 → tanh(0)=0 → dZ/dt=0 → no update when z0=zeros)
        torch.nn.init.uniform_(self.linear1.bias, -0.1, 0.1)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute the CDE vector field at state z and time t.

        Args:
            t: Scalar time (not used explicitly — autonomous CDE).
            z: (batch, hidden_dim) current latent state.

        Returns:
            (batch, hidden_dim, input_channels) — matrix to multiply with dX/dt.
        """
        z = F.tanh(self.linear1(z))
        z = self.linear2(z)
        return z.view(z.shape[0], self.hidden_dim, self.input_channels)


# ═══════════════════════════════════════════════════════════════
# CDEMemoryEncoder — drop-in replacement for GRU memory update
# ═══════════════════════════════════════════════════════════════


class CDEMemoryEncoder(nn.Module):
    """Continuous-time memory encoder using Neural CDEs.

    Processes the path of events for each node within a temporal window,
    integrating a Neural CDE to produce updated memory states.  Falls back
    to the GRU in HeteroMemory for nodes with fewer than ``min_events``
    events (CDE requires ≥2 points to form a path).

    Parameters
    ----------
    memory_dim : int
        Hidden state / memory dimension (= HeteroMemory.memory_dim).
    message_dim : int
        Incoming message dimension (= HeteroMemory.message_dim).
    n_time_points : int
        Number of interpolation points on the common time grid [0, 1].
        More points = more accurate integration but slower.
        Default 10 is a good balance for intra-window dynamics.
    min_events : int
        Minimum events per node to use CDE; below this falls back to GRU.
    """

    def __init__(
        self,
        memory_dim: int,
        message_dim: int,
        n_time_points: int = 10,
        min_events: int = 2,
    ) -> None:
        super().__init__()
        self.memory_dim = memory_dim
        self.message_dim = message_dim
        self.n_time_points = n_time_points
        self.min_events = min_events
        # input_channels = message_dim + 1 (the +1 is the normalized time channel)
        self._input_channels = message_dim + 1
        self.cde_func = CDEFunc(
            hidden_dim=memory_dim,
            input_channels=self._input_channels,
        )

    @property
    def available(self) -> bool:
        """True if torchcde/torchdiffeq are installed."""
        return _CDE_AVAILABLE

    def update_memory_from_events(
        self,
        events: list[dict],
        embeddings: dict[str, torch.Tensor],
        id_map: "Any",
        memory: "HeteroMemory",
        t_start: float,
        t_end: float,
    ) -> None:
        """Replace HeteroMemory.update_memory calls with CDE integration.

        Groups events by node, then:
        - Nodes with ≥ min_events: integrate CDE over their event path.
        - Nodes with < min_events: delegate to HeteroMemory.update_memory (GRU).

        Args:
            events:     Observation dicts (entity_id, entity_type, observed_at).
            embeddings: Current node embeddings {ntype: (N, hidden_dim)}.
            id_map:     Entity → global ID mapping.
            memory:     HeteroMemory instance to update.
            t_start:    Window start timestamp.
            t_end:      Window end timestamp (must be > t_start).
        """
        if not events:
            return

        # ── Group events by global node ID ─────────────────────────────
        node_events: dict[int, list[tuple[float, torch.Tensor]]] = {}

        for ev in events:
            etype = ev.get("entity_type")
            eid = ev.get("entity_id")
            t = float(ev.get("observed_at", 0.0))
            if etype is None or eid is None:
                continue
            gid = id_map.global_id(etype, eid)
            local_idx = id_map.local_id(etype, eid)
            if gid is None or local_idx is None or etype not in embeddings:
                continue
            emb = embeddings[etype]
            if local_idx >= emb.size(0):
                continue

            msg = emb[local_idx].detach()
            # Normalise to message_dim
            if msg.size(0) > self.message_dim:
                msg = msg[: self.message_dim]
            elif msg.size(0) < self.message_dim:
                pad = torch.zeros(self.message_dim - msg.size(0), device=msg.device)
                msg = torch.cat([msg, pad])

            node_events.setdefault(gid, []).append((t, msg))

        if not node_events:
            return

        # ── Partition: CDE nodes vs GRU fallback nodes ─────────────────
        cde_nodes = [
            (gid, evs)
            for gid, evs in node_events.items()
            if len(evs) >= self.min_events and _CDE_AVAILABLE
        ]
        gru_nodes = [
            (gid, evs)
            for gid, evs in node_events.items()
            if len(evs) < self.min_events or not _CDE_AVAILABLE
        ]

        # ── GRU fallback: process single-event nodes with HeteroMemory ─
        if gru_nodes:
            _device = memory.memory.device
            node_ids_list: list[int] = []
            msgs_list: list[torch.Tensor] = []
            ts_list: list[float] = []
            for gid, evs in gru_nodes:
                # Take the last event for nodes with < min_events
                t_ev, msg_ev = evs[-1]
                node_ids_list.append(gid)
                msgs_list.append(msg_ev.to(_device))
                ts_list.append(t_ev)
            memory.update_memory(
                torch.tensor(node_ids_list, dtype=torch.long, device=_device),
                torch.stack(msgs_list),
                torch.tensor(ts_list, dtype=torch.float, device=_device),
            )

        # ── CDE path: integrate over multi-event node trajectories ─────
        if cde_nodes:
            self._update_cde_batch(cde_nodes, memory, t_start, t_end)

    def _update_cde_batch(
        self,
        cde_nodes: list[tuple[int, list[tuple[float, torch.Tensor]]]],
        memory: "HeteroMemory",
        t_start: float,
        t_end: float,
    ) -> None:
        """Integrate CDE for a batch of nodes with ≥ min_events events.

        Builds a (batch, n_time_points, input_channels) control path tensor
        by interpolating each node's event messages onto a common normalised
        time grid [0, 1], then calls torchcde.cdeint.

        The updated Z(1) values are written back to memory.memory in-place.

        Args:
            cde_nodes: List of (global_node_id, [(t_i, msg_i), ...]).
            memory:    HeteroMemory to update.
            t_start:   Window start (for normalisation denominator).
            t_end:     Window end.
        """
        device = memory.memory.device
        t_span = max(t_end - t_start, 1.0)
        N = len(cde_nodes)
        M = self.n_time_points
        C = self._input_channels  # message_dim + 1

        # Common normalised time grid shared across all nodes in batch
        t_grid = torch.linspace(0.0, 1.0, M, device=device)  # (M,)

        # Build path tensor: (N, M, C)
        # Channel 0 = normalised time, channels 1..C = message features
        batch_paths = torch.zeros(N, M, C, device=device)
        batch_paths[:, :, 0] = t_grid.unsqueeze(0)  # broadcast time channel

        node_ids_list: list[int] = []
        last_ts_list: list[float] = []

        for b, (gid, evs) in enumerate(cde_nodes):
            node_ids_list.append(gid)
            last_ts_list.append(evs[-1][0])

            # Normalise event timestamps to [0, 1]
            ts = torch.tensor(
                [(e[0] - t_start) / t_span for e in evs],
                dtype=torch.float,
                device=device,
            ).clamp(0.0, 1.0)
            msgs = torch.stack([e[1].to(device) for e in evs])  # (n_evs, message_dim)

            # Step interpolation: for each grid point, use the last event at or before it
            for j in range(M):
                t_j = t_grid[j].item()
                valid = (ts <= t_j).nonzero(as_tuple=False)
                if valid.numel() > 0:
                    last_ev_idx = valid[-1].item()
                    batch_paths[b, j, 1:] = msgs[last_ev_idx]
                # else: stays zero (before any event in this window)

        # Natural cubic spline coefficients over the common grid
        # torchcde expects (batch, length, channels)
        coeffs = torchcde.natural_cubic_spline_coeffs(batch_paths, t_grid)
        X = torchcde.CubicSpline(coeffs)

        # Initial memory states Z(0) from current memory
        node_ids_t = torch.tensor(node_ids_list, dtype=torch.long, device=device)
        z0, _ = memory.get_memory(node_ids_t)  # (N, memory_dim) — already detached

        # Integrate CDE from t=0 to t=1
        # adjoint=False: no gradient through the ODE solver (memory update is detached)
        # method="rk4": fast fixed-step Runge-Kutta 4; rtol/atol loosen for speed
        solution = torchcde.cdeint(
            X=X,
            func=self.cde_func,
            z0=z0,
            t=torch.tensor([0.0, 1.0], device=device),
            adjoint=False,
            method="rk4",
            options={"step_size": 0.1},
        )
        # torchcde returns (batch, len_t, memory_dim) — axes are (batch, time, state)
        # sol[:, -1, :] gives the state at t=1 for all batch elements: (N, memory_dim)
        z_T = solution[:, -1, :].detach()  # (N, memory_dim)

        # Write back: only for valid node IDs within memory bounds
        with torch.no_grad():
            valid_mask = node_ids_t < memory.num_nodes
            safe_ids = node_ids_t.clamp(max=memory.num_nodes - 1)
            memory.memory[safe_ids[valid_mask]] = z_T[valid_mask]
            last_ts_t = torch.tensor(last_ts_list, dtype=torch.float, device=device)
            memory.last_update[safe_ids[valid_mask]] = last_ts_t[valid_mask]

        log.debug(
            "CDEMemoryEncoder: updated %d/%d nodes via CDE (step_size=0.1, method=rk4)",
            int(valid_mask.sum().item()),
            N,
        )
