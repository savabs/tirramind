"""
TirraMind — SurpriseExtractor

Extracts per-entity prediction surprise from a trained HetTGN model.

The five surprise signals (the primary anomaly signal):
    1. obs_type_surprise:       -log P(actual_obs_type | h_i)
    2. temporal_surprise:       |dt_pred - dt_actual|, z-scored per entity type
    3. value_surprise:          |v_pred - v_actual| / sigma_type
    4. neighborhood_surprise:   attention-weighted avg of neighbor composite surprise
    5. memory_drift:            L2 norm of memory state change

This is the **core innovation** of the prediction-surprise paradigm:
the GNN's own self-supervised prediction errors become the anomaly signal.

References:
    - Spec: docs/specs/signal_fusion_spec.md (Step 20.7)
    - SL-GAD (Zheng 2021): prediction error as anomaly signal
    - GraphMAE (Hou 2022 KDD): reconstruction error for anomaly
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from agent.models.gnn.graph_builder import IDMap, OBSERVATION_TYPES

log = logging.getLogger(__name__)


@dataclass
class EntitySurprise:
    """Per-entity surprise scores from GNN prediction."""

    entity_id: str
    entity_type: str
    obs_type_surprise: float
    temporal_surprise: float
    value_surprise: float
    neighborhood_surprise: float
    memory_drift: float
    composite_surprise: float

    def surprise_vector(self) -> tuple[float, ...]:
        """Return the 5 surprise signals as a vector (for cosine similarity)."""
        return (
            self.obs_type_surprise,
            self.temporal_surprise,
            self.value_surprise,
            self.neighborhood_surprise,
            self.memory_drift,
        )


class SurpriseExtractor:
    """Extract prediction surprise from a trained HetTGN.

    Usage::

        extractor = SurpriseExtractor()
        surprises = extractor.extract(model, data, id_map, observations, memory_before)
    """

    def __init__(
        self,
        *,
        weights: tuple[float, ...] | None = None,
        obs_type_weight: float = 0.3,
        temporal_weight: float = 0.15,
        value_weight: float = 0.25,
        neighborhood_weight: float = 0.2,
        memory_weight: float = 0.1,
    ) -> None:
        if weights is not None:
            if len(weights) != 5:
                raise ValueError(f"weights must have 5 elements, got {len(weights)}")
            (
                obs_type_weight,
                temporal_weight,
                value_weight,
                neighborhood_weight,
                memory_weight,
            ) = weights
        total = (
            obs_type_weight
            + temporal_weight
            + value_weight
            + neighborhood_weight
            + memory_weight
        )
        self._weights = {
            "obs_type": obs_type_weight / total,
            "temporal": temporal_weight / total,
            "value": value_weight / total,
            "neighborhood": neighborhood_weight / total,
            "memory": memory_weight / total,
        }
        # Per-type rolling statistics for z-scoring
        self._type_stats: dict[str, _RollingStats] = defaultdict(_RollingStats)

    def extract(
        self,
        model: torch.nn.Module,
        data: object,
        id_map: IDMap,
        observations: list[dict],
        memory_before: torch.Tensor | None = None,
    ) -> dict[str, EntitySurprise]:
        """Extract per-entity prediction surprise.

        Args:
            model: Trained HetTGN (in eval mode).
            data: HeteroData graph snapshot.
            id_map: Current ID mappings.
            observations: Observations in the current window (to compare against predictions).
            memory_before: Memory state snapshot from before the update (for memory_drift).
                Shape (num_nodes, memory_dim). If None, memory_drift = 0.

        Returns:
            Dict mapping entity_id → EntitySurprise.
        """
        model.eval()

        with torch.no_grad():
            embeddings = model(data, id_map)
            obs_logits = model.predict_obs_type(embeddings)
            dt_preds = model.predict_time_delta(embeddings)
            val_preds = model.predict_value(embeddings)

        # Memory after update (current state)
        memory_after = model.memory.memory.detach()

        # Group observations by entity (take most recent per entity)
        entity_obs: dict[str, dict] = {}
        for o in observations:
            eid = o.get("entity_id")
            if eid:
                entity_obs[eid] = o  # last one wins (latest)

        results: dict[str, EntitySurprise] = {}

        for eid, obs in entity_obs.items():
            # Find this entity in the id_map
            etype = None
            for (et, ei), gid in id_map.typed_to_global.items():
                if ei == eid:
                    etype = et
                    break
            if etype is None or etype not in embeddings:
                continue

            local_idx = id_map.local_id(etype, eid)
            if local_idx is None or local_idx >= embeddings[etype].size(0):
                continue
            gid = id_map.global_id(etype, eid)
            if gid is None:
                continue

            # 1. obs_type_surprise: -log P(actual | h_i)
            logits = obs_logits.get(etype)
            obs_type_s = 0.0
            if logits is not None and local_idx < logits.size(0):
                probs = F.softmax(logits[local_idx], dim=0)
                actual_type = obs.get("observation_type", "")
                actual_idx = None
                for j, ot in enumerate(OBSERVATION_TYPES):
                    if ot == actual_type:
                        actual_idx = j
                        break
                if actual_idx is not None and actual_idx < probs.size(0):
                    p = probs[actual_idx].item()
                    obs_type_s = -math.log(max(p, 1e-10))

            # 2. temporal_surprise: |dt_pred - dt_actual|
            dt_pred_dict = dt_preds.get(etype)
            temporal_s = 0.0
            if dt_pred_dict is not None and local_idx < dt_pred_dict.size(0):
                dt_pred_val = dt_pred_dict[local_idx].item()
                dt_actual = obs.get("observed_at", 0.0)
                # We use the raw absolute error, z-scored later
                temporal_s = abs(dt_pred_val - dt_actual)

            # 3. value_surprise: |v_pred - v_actual| / sigma_type
            val_pred_dict = val_preds.get(etype)
            value_s = 0.0
            if val_pred_dict is not None and local_idx < val_pred_dict.size(0):
                v_pred = val_pred_dict[local_idx].item()
                v_actual = self._extract_value(obs)
                stats = self._type_stats[etype]
                sigma = stats.std()
                if sigma > 0:
                    value_s = abs(v_pred - v_actual) / sigma
                else:
                    value_s = abs(v_pred - v_actual)
                stats.update(v_actual)

            # 4. neighborhood_surprise: computed after all entities processed
            # (set to 0 for now, filled in second pass)
            neighborhood_s = 0.0

            # 5. memory_drift: L2 norm of memory change
            memory_d = 0.0
            if (
                memory_before is not None
                and gid < memory_before.size(0)
                and gid < memory_after.size(0)
            ):
                diff = memory_after[gid] - memory_before[gid]
                memory_d = torch.norm(diff, p=2).item()

            # Composite (without neighborhood for now)
            composite = (
                self._weights["obs_type"] * obs_type_s
                + self._weights["temporal"] * temporal_s
                + self._weights["value"] * value_s
                + self._weights["memory"] * memory_d
            )

            results[eid] = EntitySurprise(
                entity_id=eid,
                entity_type=etype,
                obs_type_surprise=obs_type_s,
                temporal_surprise=temporal_s,
                value_surprise=value_s,
                neighborhood_surprise=neighborhood_s,
                memory_drift=memory_d,
                composite_surprise=composite,
            )

        # Second pass: compute neighborhood surprise
        self._compute_neighborhood_surprise(results, id_map, data)

        return results

    def _compute_neighborhood_surprise(
        self,
        results: dict[str, EntitySurprise],
        id_map: IDMap,
        data: object,
    ) -> None:
        """Fill in neighborhood_surprise by averaging connected neighbors' composite surprise."""
        # Build adjacency from the data's edge indices
        neighbors: dict[str, list[str]] = defaultdict(list)

        if hasattr(data, "edge_types"):
            for etype in data.edge_types:
                if hasattr(data[etype], "edge_index"):
                    edge_index = data[etype].edge_index
                    src_type, rel, dst_type = etype
                    for j in range(edge_index.size(1)):
                        src_local = edge_index[0, j].item()
                        dst_local = edge_index[1, j].item()
                        # Map local indices back to entity IDs
                        src_eid = self._local_to_eid(id_map, src_type, src_local)
                        dst_eid = self._local_to_eid(id_map, dst_type, dst_local)
                        if src_eid and dst_eid:
                            neighbors[src_eid].append(dst_eid)
                            neighbors[dst_eid].append(src_eid)

        for eid, surprise in results.items():
            neighbor_ids = neighbors.get(eid, [])
            if not neighbor_ids:
                continue
            neighbor_composites = [
                results[nid].composite_surprise
                for nid in neighbor_ids
                if nid in results
            ]
            if neighbor_composites:
                neigh_s = sum(neighbor_composites) / len(neighbor_composites)
                # Recompute composite with neighborhood
                composite = (
                    self._weights["obs_type"] * surprise.obs_type_surprise
                    + self._weights["temporal"] * surprise.temporal_surprise
                    + self._weights["value"] * surprise.value_surprise
                    + self._weights["neighborhood"] * neigh_s
                    + self._weights["memory"] * surprise.memory_drift
                )
                # Replace with updated surprise (dataclass is not frozen)
                results[eid] = EntitySurprise(
                    entity_id=eid,
                    entity_type=surprise.entity_type,
                    obs_type_surprise=surprise.obs_type_surprise,
                    temporal_surprise=surprise.temporal_surprise,
                    value_surprise=surprise.value_surprise,
                    neighborhood_surprise=neigh_s,
                    memory_drift=surprise.memory_drift,
                    composite_surprise=composite,
                )

    @staticmethod
    def _local_to_eid(id_map: IDMap, ntype: str, local_idx: int) -> str | None:
        """Convert (node_type, local_index) back to entity_id."""
        local_map = id_map.type_local.get(ntype, {})
        for eid, lidx in local_map.items():
            if lidx == local_idx:
                return eid
        return None

    @staticmethod
    def _extract_value(obs: dict) -> float:
        """Extract numeric value from observation."""
        v = obs.get("value", {})
        if isinstance(v, dict):
            for k in (
                "usd_amount",
                "btc_amount",
                "value",
                "estimated_value",
                "goldstein_scale",
                "num_articles",
            ):
                if k in v:
                    try:
                        return float(v[k])
                    except (TypeError, ValueError):
                        pass
        return 0.0


class _RollingStats:
    """Online mean/variance tracker (Welford's algorithm)."""

    __slots__ = ("_n", "_mean", "_m2")

    def __init__(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0

    def update(self, x: float) -> None:
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        delta2 = x - self._mean
        self._m2 += delta * delta2

    def std(self) -> float:
        if self._n < 2:
            return 0.0
        return math.sqrt(self._m2 / (self._n - 1))
