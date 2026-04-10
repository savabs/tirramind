"""
TirraMind — Temporal Feature Encoding (Phase 12b)

Provides:
    Time2Vec   — Learnable periodic + linear time representation.
    TemporalEncoder — Per-entity observation history → fixed-length vector.

References:
    Time2Vec: Kazemi et al. 2019, arXiv:1907.05321.
      t2v(τ)[0] = ω₀τ + φ₀                (linear component)
      t2v(τ)[i] = sin(ωᵢτ + φᵢ)  i > 0    (periodic components)

    The periodic components learn dominant frequencies in the event
    stream (e.g., trading-day cadence, weekly geopolitical cycles).
    The linear component captures trend/drift.

Spec step: 12b.1, 12b.2.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from agent.models.gnn.graph_builder import OBSERVATION_TYPES

# ── Time2Vec ───────────────────────────────────────────────────


class Time2Vec(nn.Module):
    """Learnable time representation (Kazemi et al. 2019).

    Input:  (*, 1) — timestamps (any unit, learnable).
    Output: (*, out_features) — time embedding vector.

    First dimension is linear; the rest are sinusoidal with learned
    frequency ω and phase φ.
    """

    def __init__(self, out_features: int = 16) -> None:
        super().__init__()
        if out_features < 1:
            raise ValueError("out_features must be >= 1")
        self.out_features = out_features
        # ω and φ for periodic components (indices 1..out_features-1)
        periodic_dim = max(out_features - 1, 0)
        self.omega = nn.Parameter(torch.randn(periodic_dim))
        self.phi = nn.Parameter(torch.randn(periodic_dim))
        # Linear component (index 0)
        self.w_linear = nn.Parameter(torch.randn(1))
        self.b_linear = nn.Parameter(torch.randn(1))
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize frequencies spread across common timescales."""
        # Start with frequencies spanning 1 hour to 30 days
        periodic_dim = self.out_features - 1
        if periodic_dim > 0:
            # Log-spaced initial frequencies (radians per second)
            min_period = 3600.0  # 1 hour
            max_period = 30 * 86400.0  # 30 days
            periods = torch.logspace(
                math.log10(min_period),
                math.log10(max_period),
                periodic_dim,
            )
            with torch.no_grad():
                self.omega.copy_(2.0 * math.pi / periods)
                self.phi.zero_()
        with torch.no_grad():
            self.w_linear.fill_(1e-6)  # Small linear slope
            self.b_linear.zero_()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Encode timestamps.

        Args:
            t: Tensor of shape (*, 1) or (*,). Timestamps in seconds.

        Returns:
            Tensor of shape (*, out_features).
        """
        if t.dim() == 0:
            t = t.reshape(1, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)

        # Linear component: shape (*, 1)
        linear = self.w_linear * t + self.b_linear

        if self.out_features == 1:
            return linear

        # Periodic components: shape (*, periodic_dim)
        periodic = torch.sin(self.omega * t + self.phi)

        return torch.cat([linear, periodic], dim=-1)


# ── TemporalEncoder ────────────────────────────────────────────


class TemporalEncoder(nn.Module):
    """Per-entity observation history → fixed-length feature vector.

    For each entity, take its last K observations and produce a vector:
        [obs_type_counts | inter_event_stats | Time2Vec(Δt_last)]

    where:
        obs_type_counts: count per observation type in recent window (len = num_obs_types)
        inter_event_stats: [mean_Δt, std_Δt, min_Δt, max_Δt] (4 dims)
        Time2Vec(Δt_last): time since last observation encoding (time_dim dims)

    Total output dim = num_obs_types + 4 + time_dim
    """

    def __init__(
        self,
        time_dim: int = 16,
        obs_types: list[str] | None = None,
        max_history: int = 32,
    ) -> None:
        super().__init__()
        self.obs_types = obs_types or list(OBSERVATION_TYPES)
        self.obs_type_to_idx = {t: i for i, t in enumerate(self.obs_types)}
        self.max_history = max_history
        self.time_dim = time_dim
        self.time2vec = Time2Vec(out_features=time_dim)
        self._output_dim = len(self.obs_types) + 4 + time_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        entity_obs: list[dict[str, Any]],
        current_time: float,
    ) -> torch.Tensor:
        """Encode a single entity's observation history.

        Args:
            entity_obs: List of observation dicts with 'observed_at' and
                'observation_type' keys, pre-filtered for one entity.
                Need not be sorted.
            current_time: Reference time for Δt computation.

        Returns:
            1-D tensor of shape (output_dim,).
        """
        # Sort by time, keep last K
        sorted_obs = sorted(
            entity_obs,
            key=lambda o: o.get("observed_at", 0.0),
        )[-self.max_history :]

        # Obs-type counts
        type_counts = torch.zeros(len(self.obs_types))
        for o in sorted_obs:
            otype = o.get("observation_type", "")
            idx = self.obs_type_to_idx.get(otype)
            if idx is not None:
                type_counts[idx] += 1.0

        # Inter-event time statistics
        if len(sorted_obs) >= 2:
            times = torch.tensor(
                [o.get("observed_at", 0.0) for o in sorted_obs],
                dtype=torch.float,
            )
            deltas = times[1:] - times[:-1]
            mean_dt = deltas.mean()
            std_dt = deltas.std() if len(deltas) > 1 else torch.tensor(0.0)
            min_dt = deltas.min()
            max_dt = deltas.max()
        else:
            mean_dt = torch.tensor(0.0)
            std_dt = torch.tensor(0.0)
            min_dt = torch.tensor(0.0)
            max_dt = torch.tensor(0.0)
        inter_event = torch.stack([mean_dt, std_dt, min_dt, max_dt])

        # Time2Vec of time since last observation
        if sorted_obs:
            last_t = sorted_obs[-1].get("observed_at", 0.0)
            dt_last = max(0.0, current_time - last_t)
        else:
            dt_last = 0.0
        t2v = self.time2vec(torch.tensor([dt_last], dtype=torch.float)).squeeze(0)

        return torch.cat([type_counts, inter_event, t2v])

    def encode_batch(
        self,
        obs_by_entity: dict[str, list[dict[str, Any]]],
        entity_ids: list[str],
        current_time: float,
    ) -> torch.Tensor:
        """Encode multiple entities in batch.

        Args:
            obs_by_entity: {entity_id: [obs_dicts]}
            entity_ids: Ordered list of entity IDs (determines row order).
            current_time: Reference time.

        Returns:
            Tensor of shape (len(entity_ids), output_dim).
        """
        rows = []
        for eid in entity_ids:
            ent_obs = obs_by_entity.get(eid, [])
            rows.append(self.forward(ent_obs, current_time))
        if not rows:
            return torch.zeros(0, self._output_dim)
        return torch.stack(rows)
