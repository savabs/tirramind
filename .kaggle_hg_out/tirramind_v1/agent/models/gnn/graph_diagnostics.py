"""TirraMind — Graph Diagnostics (Phase 34)

Utility for diagnosing entity graph health: counts per type, orphan
detection, observation coverage gaps.  Used for post-phase verification
and ongoing graph monitoring.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.pipeline.store import PipelineStore

log = logging.getLogger(__name__)


def diagnose_graph(store: PipelineStore) -> dict[str, Any]:
    """Run a comprehensive health check on the entity graph.

    Returns a dict with:
        entity_counts      — {entity_type: count}
        observation_counts — {observation_type: count}
        link_counts        — {link_type: count}
        orphan_entities    — list of {entity_id, entity_type, canonical_name}
                             with zero links (neither side)
        entity_types_without_obs  — list of entity types with zero observations
        obs_types_without_instances — list of obs types with zero stored instances
        total_entities, total_observations, total_links — int
    """
    entities = store.query_all_entities()
    observations = store.query_all_observations()
    links = store.query_all_entity_links()

    # ── Entity counts by type ──
    entity_type_counter: Counter[str] = Counter()
    for ent in entities:
        entity_type_counter[ent.get("entity_type", "unknown")] += 1

    # ── Observation counts by type ──
    obs_type_counter: Counter[str] = Counter()
    for obs in observations:
        obs_type_counter[obs.get("observation_type", "unknown")] += 1

    # ── Link counts by type ──
    link_type_counter: Counter[str] = Counter()
    for link in links:
        link_type_counter[link.get("link_type", "unknown")] += 1

    # ── Orphan detection (entities with zero links on either side) ──
    linked_eids: set[str] = set()
    for link in links:
        linked_eids.add(link.get("entity_id_a", ""))
        linked_eids.add(link.get("entity_id_b", ""))

    orphans: list[dict[str, str]] = []
    for ent in entities:
        eid = ent.get("entity_id", "")
        if eid not in linked_eids:
            orphans.append(
                {
                    "entity_id": eid,
                    "entity_type": ent.get("entity_type", "unknown"),
                    "canonical_name": ent.get("canonical_name", ""),
                }
            )

    # ── Entity types with zero observations ──
    entity_types_present = set(entity_type_counter.keys())
    # Find which entity types have at least one observation
    obs_entity_ids = {obs.get("entity_id", "") for obs in observations}
    entity_types_with_obs: set[str] = set()
    for ent in entities:
        if ent.get("entity_id", "") in obs_entity_ids:
            entity_types_with_obs.add(ent.get("entity_type", "unknown"))
    entity_types_without_obs = sorted(entity_types_present - entity_types_with_obs)

    # ── Observation types with zero instances ──
    from agent.models.gnn.graph_builder import OBSERVATION_TYPES

    obs_types_without = sorted(set(OBSERVATION_TYPES) - set(obs_type_counter.keys()))

    result = {
        "entity_counts": dict(entity_type_counter.most_common()),
        "observation_counts": dict(obs_type_counter.most_common()),
        "link_counts": dict(link_type_counter.most_common()),
        "orphan_entities": orphans,
        "entity_types_without_obs": entity_types_without_obs,
        "obs_types_without_instances": obs_types_without,
        "total_entities": len(entities),
        "total_observations": len(observations),
        "total_links": len(links),
    }

    log.info(
        "Graph diagnostics: %d entities (%d types), %d observations (%d types), %d links (%d types), %d orphans",
        result["total_entities"],
        len(entity_type_counter),
        result["total_observations"],
        len(obs_type_counter),
        result["total_links"],
        len(link_type_counter),
        len(orphans),
    )

    return result
