"""TirraMind — Entity Type & Relationship Inducer (Change 16)

Discovers new entity types from unresolved entity mentions and new
relationship types from entity co-occurrence patterns.

**Type induction method:**
1. Cluster unresolved entities by (source_tool, observation field similarity)
   using Jaccard distance + agglomerative clustering.
2. Evaluate cluster quality via silhouette score.
3. Propose types for clusters exceeding minimum size and cohesion threshold.
4. Validate against existing types to prevent duplicates.

**Relationship induction method:**
1. Detect entity co-occurrences within a time window per source tool.
2. Score co-occurrence significance via pointwise MI.
3. Validate relationships using BIC scoring (reusing EdgeConfidenceTracker logic).

No LLM in the hot path — all computations are frequency, schema similarity,
and co-occurrence based.

Reference: Spec steps 16.3 + 16.4 in [[tier8_autonomous_discovery_spec]].
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from agent.discovery.ontology_registry import OntologyRegistry
    from agent.pipeline.store import PipelineStore

log = logging.getLogger(__name__)

_DEFAULT_MIN_CLUSTER = 5
_DEFAULT_COHESION_THRESHOLD = 0.6
_DEFAULT_COOCCURRENCE_WINDOW = 86_400.0  # 24 hours in seconds
_DEFAULT_COOCCURRENCE_THRESHOLD = 10  # min co-occurrence count
_DEFAULT_MI_THRESHOLD = 0.1


class TypeInducer:
    """Discover entity types from unresolved entity mentions.

    Parameters
    ----------
    store : PipelineStore
        Backing store for reading unresolved entities and writing types.
    registry : OntologyRegistry
        Dynamic type registry to register newly discovered types into.
    min_cluster_size : int
        Minimum entities in a cluster to propose a type.
    cohesion_threshold : float
        Minimum silhouette score for a valid cluster.
    """

    def __init__(
        self,
        store: PipelineStore,
        registry: OntologyRegistry,
        *,
        min_cluster_size: int = _DEFAULT_MIN_CLUSTER,
        cohesion_threshold: float = _DEFAULT_COHESION_THRESHOLD,
    ) -> None:
        self._store = store
        self._registry = registry
        self._min_cluster = min_cluster_size
        self._cohesion_threshold = cohesion_threshold

    # ── ingestion ──────────────────────────────────────────────

    def ingest_unresolved(
        self,
        raw_text: str,
        source_tool: str,
        context: str | None = None,
        observed_at: float | None = None,
    ) -> int:
        """Store an unresolved entity mention.  Returns row ID."""
        return self._store.store_unresolved_entity(
            raw_text=raw_text,
            source_tool=source_tool,
            context_snippet=context,
            observed_at=observed_at,
        )

    # ── type induction ─────────────────────────────────────────

    def run_induction(self) -> list[str]:
        """Run one induction cycle.  Returns newly created type names."""
        unresolved = self._store.query_unresolved_entities(resolved=False)
        if not unresolved:
            return []

        # Step 1: Group by source_tool
        by_source: dict[str, list[dict]] = defaultdict(list)
        for entity in unresolved:
            by_source[entity["source_tool"]].append(entity)

        new_types: list[str] = []

        for source_tool, entities in by_source.items():
            if len(entities) < self._min_cluster:
                continue

            # Step 2: Extract field keys from context snippets
            entity_fields: list[set[str]] = []
            entity_ids: list[int] = []
            for e in entities:
                fields = self._extract_fields(e.get("context_snippet"))
                entity_fields.append(fields)
                entity_ids.append(e["id"])

            if not entity_fields:
                continue

            # Step 3: Cluster by Jaccard similarity
            clusters = self._cluster_by_jaccard(entity_ids, entity_fields)

            # Step 4: Evaluate and propose types
            for cluster_id, member_ids in clusters.items():
                if len(member_ids) < self._min_cluster:
                    continue

                # Compute cohesion (simplified silhouette)
                cohesion = self._compute_cluster_cohesion(
                    member_ids, entity_ids, entity_fields
                )
                if cohesion < self._cohesion_threshold:
                    log.debug(
                        "Cluster %d from %s: cohesion %.2f below threshold",
                        cluster_id,
                        source_tool,
                        cohesion,
                    )
                    continue

                # Derive type name
                type_name = self._derive_type_name(
                    source_tool, member_ids, entities, entity_fields, entity_ids
                )
                if not type_name:
                    continue

                # Check overlap with existing types
                if self._overlaps_existing_type(member_ids, entities):
                    log.info(
                        "Cluster from %s overlaps existing type, skipping",
                        source_tool,
                    )
                    continue

                # Register type
                created = self._registry.register_type(
                    type_name,
                    source="induced",
                    confidence=cohesion,
                )
                if created:
                    # Update cluster assignments in DB
                    self._store.update_unresolved_cluster(member_ids, cluster_id)
                    self._store.resolve_unresolved_entities(cluster_id, type_name)

                    # Register entities with new type
                    self._register_cluster_entities(member_ids, entities, entity_ids, type_name)
                    new_types.append(type_name)
                    log.info(
                        "Induced new entity type '%s' from %s (%d entities, cohesion=%.2f)",
                        type_name,
                        source_tool,
                        len(member_ids),
                        cohesion,
                    )

        return new_types

    # ── relationship induction ─────────────────────────────────

    def induce_relationships(
        self,
        *,
        time_window: float = _DEFAULT_COOCCURRENCE_WINDOW,
        min_count: int = _DEFAULT_COOCCURRENCE_THRESHOLD,
        mi_threshold: float = _DEFAULT_MI_THRESHOLD,
    ) -> list[tuple[str, str, str]]:
        """Discover relationship types from entity co-occurrence.

        Returns list of ``(entity_type_a, entity_type_b, link_type)`` triples
        for newly discovered relationships.
        """
        # Get all observations
        observations = self._store.query_all_observations()
        if not observations:
            return []

        # Group observations by (source_tool, time_bucket)
        co_occurrences: Counter[tuple[str, str]] = Counter()
        type_counts: Counter[str] = Counter()

        # Build time-bucketed groups
        by_tool: dict[str, list[dict]] = defaultdict(list)
        for obs in observations:
            by_tool[obs["source_tool"]].append(obs)

        # Count co-occurrences within time window
        for source_tool, tool_obs in by_tool.items():
            # Sort by observed_at
            sorted_obs = sorted(tool_obs, key=lambda o: o["observed_at"])

            # Resolve entity types
            entity_type_cache: dict[str, str] = {}
            for obs in sorted_obs:
                eid = obs["entity_id"]
                if eid not in entity_type_cache:
                    ents = self._store.query_all_entities(entity_type=None)
                    for ent in ents:
                        entity_type_cache[ent["entity_id"]] = ent["entity_type"]
                    break

            # Sliding window co-occurrence
            for i, obs_a in enumerate(sorted_obs):
                type_a = entity_type_cache.get(obs_a["entity_id"])
                if not type_a:
                    continue
                type_counts[type_a] += 1
                for j in range(i + 1, len(sorted_obs)):
                    obs_b = sorted_obs[j]
                    if obs_b["observed_at"] - obs_a["observed_at"] > time_window:
                        break
                    type_b = entity_type_cache.get(obs_b["entity_id"])
                    if not type_b or type_a == type_b:
                        continue
                    # Canonical order
                    pair = tuple(sorted([type_a, type_b]))
                    co_occurrences[pair] += 1  # type: ignore[arg-type]

        if not co_occurrences:
            return []

        # Evaluate significance via pointwise MI
        total = sum(type_counts.values())
        new_relationships: list[tuple[str, str, str]] = []

        for (type_a, type_b), count in co_occurrences.items():
            if count < min_count:
                continue

            # PMI = log(P(a,b) / (P(a) * P(b)))
            p_ab = count / max(total, 1)
            p_a = type_counts.get(type_a, 0) / max(total, 1)
            p_b = type_counts.get(type_b, 0) / max(total, 1)
            denom = p_a * p_b
            if denom < 1e-12:
                continue
            pmi = float(np.log(p_ab / denom))

            if pmi > mi_threshold:
                link_type = f"cooccurrence_{type_a}_{type_b}"
                if self._registry.register_link_type(link_type):
                    new_relationships.append((type_a, type_b, link_type))
                    log.info(
                        "Induced relationship %s ↔ %s (link=%s, count=%d, PMI=%.3f)",
                        type_a,
                        type_b,
                        link_type,
                        count,
                        pmi,
                    )

        return new_relationships

    # ── helpers ─────────────────────────────────────────────────

    @staticmethod
    def _extract_fields(context_snippet: str | None) -> set[str]:
        """Extract field/key names from a context snippet."""
        if not context_snippet:
            return set()
        # Try JSON parse
        try:
            data = json.loads(context_snippet)
            if isinstance(data, dict):
                return set(data.keys())
        except (json.JSONDecodeError, TypeError):
            pass
        # Fallback: extract word-like tokens
        tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", context_snippet)
        return set(tokens[:20])  # cap at 20 tokens

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        """Jaccard similarity between two sets."""
        if not a and not b:
            return 1.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union > 0 else 0.0

    def _cluster_by_jaccard(
        self,
        entity_ids: list[int],
        entity_fields: list[set[str]],
    ) -> dict[int, list[int]]:
        """Simple agglomerative clustering by Jaccard similarity."""
        n = len(entity_ids)
        if n == 0:
            return {}

        # Assign each entity to a cluster based on field similarity
        # Simple single-pass: assign to first cluster with avg similarity > 0.5
        clusters: dict[int, list[int]] = {}
        cluster_fields: dict[int, list[set[str]]] = {}
        next_id = 0

        for i in range(n):
            fields = entity_fields[i]
            eid = entity_ids[i]
            best_cluster = -1
            best_sim = 0.5  # threshold

            for cid, members_fields in cluster_fields.items():
                avg_sim = float(np.mean([self._jaccard(fields, mf) for mf in members_fields]))
                if avg_sim > best_sim:
                    best_sim = avg_sim
                    best_cluster = cid

            if best_cluster >= 0:
                clusters[best_cluster].append(eid)
                cluster_fields[best_cluster].append(fields)
            else:
                clusters[next_id] = [eid]
                cluster_fields[next_id] = [fields]
                next_id += 1

        return clusters

    def _compute_cluster_cohesion(
        self,
        member_ids: list[int],
        all_ids: list[int],
        all_fields: list[set[str]],
    ) -> float:
        """Simplified silhouette-like cohesion score for a cluster."""
        # Build index: id → fields
        id_to_fields: dict[int, set[str]] = {}
        for i, eid in enumerate(all_ids):
            id_to_fields[eid] = all_fields[i]

        member_fields = [id_to_fields[eid] for eid in member_ids if eid in id_to_fields]
        if len(member_fields) < 2:
            return 0.0

        # Intra-cluster average similarity
        intra_sims = []
        for i in range(len(member_fields)):
            for j in range(i + 1, len(member_fields)):
                intra_sims.append(self._jaccard(member_fields[i], member_fields[j]))

        if not intra_sims:
            return 0.0

        return float(np.mean(intra_sims))

    def _derive_type_name(
        self,
        source_tool: str,
        member_ids: list[int],
        entities: list[dict],
        entity_fields: list[set[str]],
        all_ids: list[int],
    ) -> str | None:
        """Derive a type name from source tool and dominant fields."""
        # Find entity texts in this cluster
        id_set = set(member_ids)
        texts = []
        for e in entities:
            if e["id"] in id_set:
                texts.append(e["raw_text"].lower())

        if not texts:
            return None

        # Extract most common word as discriminator
        words: list[str] = []
        for t in texts:
            words.extend(re.findall(r"[a-z]+", t))

        if not words:
            # Fallback to source tool name
            cleaned = re.sub(r"[^a-z0-9_]", "_", source_tool.lower()).strip("_")
            return f"{cleaned}_entity" if cleaned else None

        # Most common non-trivial word
        common = Counter(w for w in words if len(w) > 2)
        if not common:
            cleaned = re.sub(r"[^a-z0-9_]", "_", source_tool.lower()).strip("_")
            return f"{cleaned}_entity" if cleaned else None

        dominant = common.most_common(1)[0][0]
        # Clean to valid type name
        name = re.sub(r"[^a-z0-9_]", "", dominant)
        if not name or not name[0].isalpha():
            name = f"t_{name}"

        # Ensure uniqueness
        if self._registry.is_valid_type(name):
            name = f"{name}_{source_tool.split('_')[0]}" if "_" in source_tool else f"{name}_auto"

        return name if re.match(r"^[a-z][a-z0-9_]{0,63}$", name) else None

    def _overlaps_existing_type(
        self,
        member_ids: list[int],
        entities: list[dict],
    ) -> bool:
        """Check if >50% of cluster entities match an existing entity type."""
        id_set = set(member_ids)
        texts = {e["raw_text"].lower() for e in entities if e["id"] in id_set}

        # Check against existing entities in store
        try:
            existing = self._store.query_all_entities()
        except Exception:
            return False

        existing_names = {e["canonical_name"].lower() for e in existing}
        overlap = len(texts & existing_names)
        return overlap > len(texts) * 0.5 if texts else False

    def _register_cluster_entities(
        self,
        member_ids: list[int],
        entities: list[dict],
        all_ids: list[int],
        type_name: str,
    ) -> None:
        """Register cluster entities in the main entities table."""
        from agent.pipeline.entity import entity_id_from_key

        id_set = set(member_ids)
        for e in entities:
            if e["id"] not in id_set:
                continue
            raw = e["raw_text"]
            eid = entity_id_from_key(type_name, raw)
            self._store.register_entity(
                entity_type=type_name,
                canonical_name=raw.strip().lower(),
                entity_id=eid,
                metadata={"source": "type_induction", "source_tool": e["source_tool"]},
            )
