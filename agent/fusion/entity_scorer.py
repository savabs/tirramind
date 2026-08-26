"""
TirraMind — EntityAnomalyScorer

Orchestrates the full entity-level anomaly scoring pipeline:
    1. Compute statistical enrichment features (CUSUM, Hawkes, EventStudy)
    2. Build enriched graph via GraphBuilder
    3. GNN forward pass
    4. Extract prediction surprise via SurpriseExtractor
    5. Detect convergence via ConvergenceDetector
    6. Emit EntityAlerts + ConvergenceClusters

This is the top-level entry point for Phase 20 entity scoring.

References:
    - Spec: docs/specs/signal_fusion_spec.md (Step 20.9)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import torch

from agent.fusion.alert import EntityAlert
from agent.fusion.convergence import ConvergenceCluster, ConvergenceDetector
from agent.fusion.cusum import CUSUMMonitor
from agent.fusion.entity_baseline import EntityBaseline
from agent.fusion.hawkes import HawkesIntensity
from agent.fusion.surprise import EntitySurprise, SurpriseExtractor
from agent.models.gnn.graph_builder import GraphBuilder
from agent.pipeline.store import PipelineStore

log = logging.getLogger(__name__)


@dataclass
class ScorerConfig:
    """Configuration for EntityAnomalyScorer."""

    # Enrichment monitor params
    cusum_k: float = 0.5
    cusum_h: float = 5.0
    hawkes_mu: float = 0.1
    hawkes_alpha: float = 0.5
    hawkes_beta: float = 1.0
    baseline_window: int = 30
    baseline_gap: int = 5

    # Convergence detection
    surprise_threshold: float = 2.0

    # Time window for observations (seconds) — None = all
    lookback_seconds: float | None = None


class EntityAnomalyScorer:
    """Orchestrate entity-level anomaly scoring.

    Usage::

        scorer = EntityAnomalyScorer(store, model, config=ScorerConfig())
        alerts, clusters = scorer.score_entities(as_of=time.time())
    """

    def __init__(
        self,
        store: PipelineStore,
        model: torch.nn.Module,
        *,
        config: ScorerConfig | None = None,
    ) -> None:
        self._store = store
        self._model = model
        self._config = config or ScorerConfig()

        # Sub-components
        self._cusum: dict[str, CUSUMMonitor] = {}
        self._hawkes: dict[str, HawkesIntensity] = {}
        self._baselines: dict[str, EntityBaseline] = {}
        self._surprise_extractor = SurpriseExtractor()
        self._convergence_detector = ConvergenceDetector()
        self._graph_builder = GraphBuilder(store)

    def score_entities(
        self,
        as_of: float,
    ) -> tuple[list[EntityAlert], list[ConvergenceCluster]]:
        """Run the full entity scoring pipeline.

        Args:
            as_of: Unix timestamp — score using observations up to this time.

        Returns:
            (alerts, clusters) — per-entity alerts and convergence clusters.
        """
        cfg = self._config

        # 1. Compute enrichment features from observations
        since = None
        if cfg.lookback_seconds is not None:
            since = as_of - cfg.lookback_seconds

        observations = self._store.query_all_observations(since=since, until=as_of)
        if not observations:
            return [], []

        enrichment = self._compute_enrichment(observations, as_of)

        # 2. Build the graph WITHOUT enrichment for the forward pass.
        #
        # The enrichment block (ENRICHMENT_DIM extra dims per node) is used for
        # alert construction in step 6, NOT as model input. Passing it here
        # widened every node vector — e.g. cftc_contract 15 -> 76 — while the
        # GNN's type_projections were trained on the un-enriched widths, giving:
        #
        #   RuntimeError: mat1 and mat2 shapes cannot be multiplied (40x76 and 15x64)
        #
        # This scorer was the ONLY caller building with enrichment; no training
        # path supports it at all (`grep enrich trainer.py` → zero hits), so the
        # model has never seen an enriched feature vector and cannot consume one.
        # The mismatch was misread as checkpoint schema drift requiring a
        # retrain; it is neither — the model's own widths are self-consistent.
        data, id_map, events = self._graph_builder.build(since=since, until=as_of)

        if id_map.num_nodes == 0:
            return [], []

        # 3. Snapshot memory before forward pass
        memory_before = self._model.memory.memory.detach().clone()

        # 4. GNN forward pass (updates memory)
        self._model.eval()
        with torch.no_grad():
            self._model(data, id_map)

        # 5. Extract prediction surprise
        entity_surprises = self._surprise_extractor.extract(
            self._model,
            data,
            id_map,
            observations,
            memory_before=memory_before,
        )

        if not entity_surprises:
            return [], []

        # 6. Build alerts
        entity_info = {e["entity_id"]: e for e in self._store.query_all_entities()}
        alerts = self._build_alerts(entity_surprises, enrichment, entity_info, as_of)

        # 7. Detect convergence
        links = self._store.query_all_entity_links()
        # Map PipelineStore link format to ConvergenceDetector format
        detector_links = [{"src_id": link["entity_id_a"], "dst_id": link["entity_id_b"]} for link in links]
        clusters = self._convergence_detector.detect(
            entity_surprises,
            detector_links,
            surprise_threshold=cfg.surprise_threshold,
        )

        return alerts, clusters

    def _compute_enrichment(
        self,
        observations: list[dict],
        as_of: float,
    ) -> dict[str, dict[str, float]]:
        """Compute per-entity enrichment features from observations."""
        cfg = self._config

        # Group observations by entity
        by_entity: dict[str, list[dict]] = defaultdict(list)
        for obs in observations:
            eid = obs.get("entity_id")
            if eid:
                by_entity[eid].append(obs)

        enrichment: dict[str, dict[str, float]] = {}

        for eid, eid_obs in by_entity.items():
            # Sort by time
            eid_obs.sort(key=lambda o: o.get("observed_at", 0.0))

            # CUSUM
            if eid not in self._cusum:
                self._cusum[eid] = CUSUMMonitor(k=cfg.cusum_k, h=cfg.cusum_h)
            cusum = self._cusum[eid]
            cusum_val = 0.0
            for obs in eid_obs:
                val = self._extract_value(obs)
                cusum_val, _ = cusum.update(eid, val)

            # Hawkes
            if eid not in self._hawkes:
                self._hawkes[eid] = HawkesIntensity(mu=cfg.hawkes_mu, alpha=cfg.hawkes_alpha, beta=cfg.hawkes_beta)
            hawkes = self._hawkes[eid]
            hawkes_val = 0.0
            for obs in eid_obs:
                t = obs.get("observed_at", 0.0)
                hawkes_val = hawkes.update(eid, t)

            # Entity baseline (event study)
            if eid not in self._baselines:
                self._baselines[eid] = EntityBaseline(window=cfg.baseline_window, gap=cfg.baseline_gap)
            baseline = self._baselines[eid]
            event_study = 0.0
            last_val = 0.0
            for obs in eid_obs:
                val = self._extract_value(obs)
                baseline.add_observation(eid, val)
                last_val = val
            score = baseline.abnormal_score(eid, last_val)
            event_study = score if score is not None else 0.0

            enrichment[eid] = {
                "cusum": cusum_val,
                "hawkes": hawkes_val,
                "event_study": event_study,
                "bocpd": 0.0,  # TODO: integrate BOCPD when available
            }

        return enrichment

    def _build_alerts(
        self,
        entity_surprises: dict[str, EntitySurprise],
        enrichment: dict[str, dict[str, float]],
        entity_info: dict[str, dict],
        as_of: float,
    ) -> list[EntityAlert]:
        """Convert EntitySurprise results into EntityAlert records."""
        alerts = []
        for eid, surprise in entity_surprises.items():
            enrich = enrichment.get(eid, {})
            info = entity_info.get(eid, {})
            alerts.append(
                EntityAlert(
                    entity_id=eid,
                    entity_type=surprise.entity_type,
                    entity_name=info.get("canonical_name", eid),
                    alert_time=as_of,
                    obs_type_surprise=surprise.obs_type_surprise,
                    temporal_surprise=surprise.temporal_surprise,
                    value_surprise=surprise.value_surprise,
                    neighborhood_surprise=surprise.neighborhood_surprise,
                    memory_drift=surprise.memory_drift,
                    composite_surprise=surprise.composite_surprise,
                    cusum_statistic=enrich.get("cusum", 0.0),
                    hawkes_intensity=enrich.get("hawkes", 0.0),
                    event_study_score=enrich.get("event_study", 0.0),
                    observation_count=len(self._store.query_entity_observations(eid, until=as_of)),
                    evidence_sources=tuple(
                        sorted(
                            {
                                obs.get("source_tool", "unknown")
                                for obs in self._store.query_entity_observations(eid, until=as_of)
                            }
                        )
                    ),
                )
            )
        return alerts

    @staticmethod
    def _extract_value(obs: dict) -> float:
        """Extract numeric value from observation."""
        v = obs.get("value_json")
        if isinstance(v, str):
            import json

            try:
                v = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                return 0.0
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
