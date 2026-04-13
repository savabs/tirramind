"""
TirraMind — ConvergenceCluster + ConvergenceDetector

Immutable record of correlated prediction surprise across graph neighborhoods,
plus the detector that produces them.

Convergence = multiple linked entities showing elevated GNN prediction surprise
simultaneously. Detected via GNN attention weights and embedding similarity,
NOT via hand-coded graph traversal or archetype matching.

Design principles:
    1. Immutable (frozen dataclass) — never mutated after creation.
    2. GNN-native — correlated_surprise_score from cosine similarity of surprise vectors.
    3. No archetypes — no pattern matching, no named convergence categories.
    4. Minimum 2 member entities required for a valid cluster.

References:
    - Spec: docs/specs/signal_fusion_spec.md (Step 20.1, Step 20.8)
    - Research: docs/research/signal_fusion.md (Paradigm Revision)
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass

from agent.fusion.alert import EntityAlert
from agent.fusion.surprise import EntitySurprise

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConvergenceCluster:
    """Cluster of entities with correlated prediction surprise.

    Attributes:
        cluster_id:                 Unique identifier (hash of sorted member IDs + time).
        cluster_time:               Unix epoch of detection.
        member_alerts:              Tuple of EntityAlert for each member entity.
        correlated_surprise_score:  Mean pairwise cosine similarity of surprise vectors.
        temporal_span_hours:        Hours between earliest and latest member alert_time.
        contributing_domains:       Descriptive only — source domains of member entities.
        contributing_tools:         Descriptive only — tools that produced member observations.
        metadata:                   Optional freeform metadata dict.
    """

    cluster_id: str
    cluster_time: float  # unix epoch
    member_alerts: tuple[EntityAlert, ...]
    correlated_surprise_score: float
    temporal_span_hours: float
    contributing_domains: tuple[str, ...]  # descriptive only
    contributing_tools: tuple[str, ...]  # descriptive only
    metadata: dict | None = None

    def __post_init__(self) -> None:
        if len(self.member_alerts) < 2:
            raise ValueError(
                f"ConvergenceCluster requires >= 2 member alerts, got {len(self.member_alerts)}"
            )


class ConvergenceDetector:
    """Detect convergence as correlated prediction surprise in graph neighborhoods.

    Algorithm:
        1. Filter to entities with composite_surprise > threshold
        2. Build subgraph of elevated entities connected by entity_links
        3. Find connected components in subgraph
        4. For each component with 2+ entities → ConvergenceCluster
        5. correlated_surprise_score = mean pairwise cosine similarity of surprise vectors

    No domain diversity counting. No temporal window. No archetype matching.
    """

    def detect(
        self,
        entity_surprises: dict[str, EntitySurprise],
        entity_links: list[dict],
        *,
        surprise_threshold: float = 2.0,
    ) -> list[ConvergenceCluster]:
        """Find clusters of connected entities with correlated high surprise.

        Args:
            entity_surprises: Per-entity surprise scores from SurpriseExtractor.
            entity_links: Edge list from PipelineStore (dicts with src_id/dst_id).
            surprise_threshold: Minimum composite_surprise to include an entity.

        Returns:
            List of ConvergenceCluster (may be empty).
        """
        # 1. Filter to elevated entities
        elevated = {
            eid: s
            for eid, s in entity_surprises.items()
            if s.composite_surprise > surprise_threshold
        }
        if len(elevated) < 2:
            return []

        # 2. Build adjacency among elevated entities
        adj: dict[str, set[str]] = defaultdict(set)
        for link in entity_links:
            src = link.get("src_id") or link.get("source_id") or link.get("src")
            dst = link.get("dst_id") or link.get("target_id") or link.get("dst")
            if src in elevated and dst in elevated:
                adj[src].add(dst)
                adj[dst].add(src)

        # 3. Find connected components via BFS
        visited: set[str] = set()
        components: list[list[str]] = []
        for eid in elevated:
            if eid in visited:
                continue
            # BFS
            component: list[str] = []
            queue = [eid]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                for neighbor in adj.get(node, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)
            if len(component) >= 2:
                components.append(component)

        # 4. Build clusters
        clusters: list[ConvergenceCluster] = []
        now = time.time()
        for comp in components:
            surprises = [elevated[eid] for eid in comp]
            # Compute mean pairwise cosine similarity
            cos_score = self._mean_pairwise_cosine(surprises)

            # Build placeholder EntityAlerts (will be replaced by real alerts in scorer)
            member_alerts = tuple(
                EntityAlert(
                    entity_id=s.entity_id,
                    entity_type=s.entity_type,
                    entity_name=s.entity_id,  # name = id for now
                    alert_time=now,
                    obs_type_surprise=s.obs_type_surprise,
                    temporal_surprise=s.temporal_surprise,
                    value_surprise=s.value_surprise,
                    neighborhood_surprise=s.neighborhood_surprise,
                    memory_drift=s.memory_drift,
                    composite_surprise=s.composite_surprise,
                    cusum_statistic=0.0,
                    hawkes_intensity=0.0,
                    event_study_score=0.0,
                    observation_count=0,
                    evidence_sources=(),
                )
                for s in surprises
            )

            # Cluster ID from sorted member IDs
            sorted_ids = sorted(s.entity_id for s in surprises)
            cluster_id = hashlib.sha256(
                ("_".join(sorted_ids) + f"_{now}").encode()
            ).hexdigest()[:16]

            # Contributing info
            domains = tuple(sorted({s.entity_type for s in surprises}))
            tools: tuple[str, ...] = ()  # filled by scorer

            clusters.append(
                ConvergenceCluster(
                    cluster_id=cluster_id,
                    cluster_time=now,
                    member_alerts=member_alerts,
                    correlated_surprise_score=cos_score,
                    temporal_span_hours=0.0,  # same moment
                    contributing_domains=domains,
                    contributing_tools=tools,
                )
            )

        return clusters

    @staticmethod
    def _mean_pairwise_cosine(surprises: list[EntitySurprise]) -> float:
        """Compute mean pairwise cosine similarity of surprise vectors.

        Returns 1.0 for perfectly aligned surprise patterns, 0.0 for orthogonal.
        """
        n = len(surprises)
        if n < 2:
            return 0.0

        vectors = [s.surprise_vector() for s in surprises]
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                cos = _cosine_similarity(vectors[i], vectors[j])
                total += cos
                count += 1
        return total / count if count > 0 else 0.0


def _cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Cosine similarity between two vectors. Returns 0.0 if either is zero."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)
