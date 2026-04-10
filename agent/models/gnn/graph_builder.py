"""
TirraMind — Graph Builder (Phase 12a)

Converts the PipelineStore entity graph into a PyG HeteroData object
suitable for training heterogeneous temporal graph networks.

Design:
    1. Query all entities → typed node feature tensors.
    2. Query all entity_links → typed edge_index tensors.
    3. Query all observations → per-node temporal event lists.
    4. Build bidirectional (type, entity_id) ↔ global int ID mappings.

The resulting HeteroData has:
    - data[node_type].x         — feature matrix [N_type, feat_dim]
    - data[node_type].node_ids  — original entity_id strings
    - data[src, rel, dst].edge_index — [2, E] connectivity
    - data[src, rel, dst].edge_attr  — [E, edge_feat_dim]

Temporal events are returned separately as a sorted list of dicts,
not embedded in the static graph (the trainer consumes them
sequentially).

References:
    HGT (Hu et al. 2020, arXiv:2003.01332) — motivates heterogeneous typing.
    PyG HeteroData — torch_geometric.data.HeteroData API.
    Spec step 12a.3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from torch_geometric.data import HeteroData

from agent.pipeline.store import PipelineStore

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────

# Canonical ordering — determines one-hot position in node features.
ENTITY_TYPES: list[str] = [
    "company",
    "country",
    "domain",
    "organization",
    "person",
    "protocol",
    "topic",
    "vessel",
    "wallet",
]

OBSERVATION_TYPES: list[str] = [
    "btc_transfer",
    "cert_issued",
    "cross_entity_pattern",
    "dns_change",
    "form144_filing",
    "geopolitical_event",
    "insider_trade",
    "lobbying_spend",
    "pageview_spike",
    "patent_filing",
    "port_call",
    "project_status",
    "sell_intent",
    "tvl_change",
    "vessel_position",
]

_ENTITY_TYPE_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(ENTITY_TYPES)}
_OBS_TYPE_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(OBSERVATION_TYPES)}


# ── ID mapping ─────────────────────────────────────────────────


@dataclass
class IDMap:
    """Bidirectional mapping between (entity_type, entity_id) and global int."""

    typed_to_global: dict[tuple[str, str], int] = field(default_factory=dict)
    global_to_typed: dict[int, tuple[str, str]] = field(default_factory=dict)
    # Per-type local indices: entity_type → {entity_id: local_idx}
    type_local: dict[str, dict[str, int]] = field(default_factory=dict)
    _next_global: int = 0

    def add(self, entity_type: str, entity_id: str) -> int:
        """Register an entity, return its global ID.  Idempotent."""
        key = (entity_type, entity_id)
        if key in self.typed_to_global:
            return self.typed_to_global[key]
        gid = self._next_global
        self._next_global += 1
        self.typed_to_global[key] = gid
        self.global_to_typed[gid] = key
        local = self.type_local.setdefault(entity_type, {})
        local[entity_id] = len(local)
        return gid

    def global_id(self, entity_type: str, entity_id: str) -> int | None:
        """Look up global ID.  Returns None if not registered."""
        return self.typed_to_global.get((entity_type, entity_id))

    def local_id(self, entity_type: str, entity_id: str) -> int | None:
        """Per-type local index (0-based within that type)."""
        local = self.type_local.get(entity_type)
        if local is None:
            return None
        return local.get(entity_id)

    @property
    def num_nodes(self) -> int:
        return self._next_global

    def num_nodes_of_type(self, entity_type: str) -> int:
        return len(self.type_local.get(entity_type, {}))


# ── Observation statistics helper ──────────────────────────────


def _compute_obs_stats(
    observations: list[dict[str, Any]],
    entity_id: str,
    current_time: float,
) -> dict[str, float]:
    """Aggregate observation statistics for a single entity.

    Returns: count, recency (seconds since last obs), mean_value.
    """
    ent_obs = [o for o in observations if o.get("entity_id") == entity_id]
    count = len(ent_obs)
    if count == 0:
        return {"count": 0.0, "recency": 0.0, "mean_value": 0.0}

    latest_t = max(o.get("observed_at", 0.0) for o in ent_obs)
    recency = current_time - latest_t if current_time > 0 else 0.0

    values: list[float] = []
    for o in ent_obs:
        v = o.get("value", {})
        if isinstance(v, dict):
            # Try common value fields
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
                        values.append(float(v[k]))
                    except (TypeError, ValueError):
                        pass
                    break
    mean_val = sum(values) / len(values) if values else 0.0

    return {"count": float(count), "recency": recency, "mean_value": mean_val}


# ── Node feature builder ──────────────────────────────────────


def _build_node_features(
    entity_type: str,
    entity_ids: list[str],
    observations: list[dict[str, Any]],
    current_time: float,
) -> torch.Tensor:
    """Build feature matrix for one node type.

    Features per node (dim = len(ENTITY_TYPES) + 3):
        [one_hot_entity_type..., obs_count, recency, mean_value]
    """
    type_idx = _ENTITY_TYPE_TO_IDX.get(entity_type)
    if type_idx is None:
        log.warning("Unknown entity type %r — defaulting to index 0", entity_type)
        type_idx = 0
    type_dim = len(ENTITY_TYPES)
    feat_dim = type_dim + 3  # +count, recency, mean_value

    if not entity_ids:
        return torch.zeros(0, feat_dim)

    # Pre-group observations by entity_id for efficiency
    obs_by_entity: dict[str, list[dict[str, Any]]] = {}
    for o in observations:
        eid = o.get("entity_id", "")
        obs_by_entity.setdefault(eid, []).append(o)

    features = torch.zeros(len(entity_ids), feat_dim)
    for local_idx, eid in enumerate(entity_ids):
        # One-hot entity type
        features[local_idx, type_idx] = 1.0
        # Observation stats
        ent_obs = obs_by_entity.get(eid, [])
        stats = _compute_obs_stats(ent_obs, eid, current_time)
        features[local_idx, type_dim] = stats["count"]
        features[local_idx, type_dim + 1] = stats["recency"]
        features[local_idx, type_dim + 2] = stats["mean_value"]

    return features


# ── Edge builder ───────────────────────────────────────────────


def _build_edge_data(
    links: list[dict[str, Any]],
    id_map: IDMap,
) -> dict[tuple[str, str, str], dict[str, torch.Tensor]]:
    """Group entity_links by (src_type, link_type, dst_type) and build tensors.

    Returns dict mapping edge-type triplet → {"edge_index": [2,E], "edge_attr": [E,2]}
    edge_attr columns: [confidence, age_days]
    """
    import time as _time

    now = _time.time()
    grouped: dict[tuple[str, str, str], dict[str, list[Any]]] = {}

    for link in links:
        eid_a = link.get("entity_id_a", "")
        eid_b = link.get("entity_id_b", "")
        ltype = link.get("link_type", "unknown")

        # Look up typed info from id_map
        gid_a = None
        gid_b = None
        type_a = type_b = None
        for (etype, eid), gid in id_map.typed_to_global.items():
            if eid == eid_a and gid_a is None:
                gid_a = gid
                type_a = etype
            if eid == eid_b and gid_b is None:
                gid_b = gid
                type_b = etype
            if gid_a is not None and gid_b is not None:
                break

        if type_a is None or type_b is None:
            log.debug("Skipping link %s→%s: entity not in id_map", eid_a, eid_b)
            continue

        local_a = id_map.local_id(type_a, eid_a)
        local_b = id_map.local_id(type_b, eid_b)
        if local_a is None or local_b is None:
            continue

        triplet = (type_a, ltype, type_b)
        bucket = grouped.setdefault(
            triplet, {"src": [], "dst": [], "conf": [], "age": []}
        )
        bucket["src"].append(local_a)
        bucket["dst"].append(local_b)
        bucket["conf"].append(link.get("confidence", 1.0))
        created = link.get("created_at", now)
        age_days = max(0.0, (now - created) / 86400.0)
        bucket["age"].append(age_days)

    result: dict[tuple[str, str, str], dict[str, torch.Tensor]] = {}
    for triplet, bucket in grouped.items():
        src = torch.tensor(bucket["src"], dtype=torch.long)
        dst = torch.tensor(bucket["dst"], dtype=torch.long)
        edge_index = torch.stack([src, dst], dim=0)
        edge_attr = torch.tensor(
            list(zip(bucket["conf"], bucket["age"])),
            dtype=torch.float,
        )
        result[triplet] = {"edge_index": edge_index, "edge_attr": edge_attr}

    return result


# ── GraphBuilder ───────────────────────────────────────────────


class GraphBuilder:
    """Converts PipelineStore into PyG HeteroData + ID mappings.

    Usage::

        builder = GraphBuilder(store)
        data, id_map, events = builder.build()
    """

    def __init__(self, store: PipelineStore) -> None:
        self._store = store

    def build(
        self,
        *,
        since: float | None = None,
        until: float | None = None,
    ) -> tuple[HeteroData, IDMap, list[dict[str, Any]]]:
        """Build the full heterogeneous graph.

        Args:
            since: Only include observations from this timestamp onward.
            until: Only include observations up to this timestamp.

        Returns:
            (HeteroData, IDMap, events) where events is a time-sorted list
            of observation dicts for the trainer to consume sequentially.
        """
        # 1. Fetch all raw data
        entities = self._store.query_all_entities()
        observations = self._store.query_all_observations(since=since, until=until)
        links = self._store.query_all_entity_links()

        # 2. Build ID map
        id_map = IDMap()
        for ent in entities:
            id_map.add(ent["entity_type"], ent["entity_id"])

        # 3. Determine current_time for recency features
        if observations:
            current_time = max(o.get("observed_at", 0.0) for o in observations)
        else:
            current_time = 0.0

        # 4. Build HeteroData
        data = HeteroData()

        # Node features per type — iterate all types present in id_map,
        # not just ENTITY_TYPES, so dynamically-added types get features too.
        all_types = sorted(set(ENTITY_TYPES) | set(id_map.type_local.keys()))
        for etype in all_types:
            local_map = id_map.type_local.get(etype, {})
            if not local_map:
                continue
            # Reconstruct ordered list by local index
            ordered_ids = [""] * len(local_map)
            for eid, lidx in local_map.items():
                ordered_ids[lidx] = eid
            features = _build_node_features(
                etype, ordered_ids, observations, current_time
            )
            data[etype].x = features
            data[etype].node_ids = ordered_ids

        # Edge data per (src_type, link_type, dst_type)
        edge_data = _build_edge_data(links, id_map)
        for triplet, tensors in edge_data.items():
            data[triplet].edge_index = tensors["edge_index"]
            data[triplet].edge_attr = tensors["edge_attr"]

        # 5. Events = observations sorted by time (for sequential training)
        events = sorted(observations, key=lambda o: o.get("observed_at", 0.0))

        node_ct = sum(id_map.num_nodes_of_type(t) for t in all_types)
        edge_ct = sum(tensors["edge_index"].size(1) for tensors in edge_data.values())
        log.info(
            "Built HeteroData: %d node types, %d nodes, %d edge types, %d edges, %d events",
            len([t for t in all_types if id_map.num_nodes_of_type(t) > 0]),
            node_ct,
            len(edge_data),
            edge_ct,
            len(events),
        )

        return data, id_map, events
