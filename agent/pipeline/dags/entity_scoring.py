"""
TirraMind — Entity Scoring DAG

Runs after gnn_inference.  Loads the trained HetTGN model, scores all entities
via prediction surprise, detects convergence clusters, and persists results.

Schedule: weekdays at 18:45 UTC (15 min after gnn_inference, 15 min before
feature_generation).

All functions follow the FunctionOperator contract:
    fn(params: dict, upstream_results: dict) -> dict
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from agent.pipeline.dag import DAG
from agent.pipeline.store import PipelineStore

log = logging.getLogger(__name__)

DAG_NAME = "entity_scoring"
DEPENDS_ON = ["gnn_inference"]

# Minimum entity count to justify scoring
_MIN_ENTITY_COUNT = 5

# Default model checkpoint location (must match gnn_inference DAG)
_DEFAULT_MODEL_PATH = ".tirra_pipeline/gnn_model.pt"


def _merge_learned_thresholds(
    scorer_config: dict[str, Any],
    threshold_dir: Path,
) -> dict[str, Any]:
    """Merge GP-BO learned thresholds into scorer_config.

    Loads the ThresholdOptimizer from *threshold_dir* and fills in
    CUSUM/Hawkes best-so-far values.  Explicit keys already in
    *scorer_config* take precedence (user overrides beat learned).

    Returns a new dict (does not mutate the input).
    """
    from agent.learning.threshold_optimizer import ThresholdOptimizer

    try:
        opt = ThresholdOptimizer(persist_dir=threshold_dir)
    except Exception:
        log.warning("Could not load ThresholdOptimizer from %s", threshold_dir)
        return scorer_config

    merged = dict(scorer_config)

    # Mapping: ThresholdOptimizer param name → ScorerConfig field name
    _CUSUM_MAP = {"k": "cusum_k", "h": "cusum_h"}
    _HAWKES_MAP = {"mu": "hawkes_mu", "alpha": "hawkes_alpha", "beta": "hawkes_beta"}

    for detector, field_map in [("cusum", _CUSUM_MAP), ("hawkes", _HAWKES_MAP)]:
        best = opt.current_best(detector)
        if best is None:
            continue
        for bo_key, cfg_key in field_map.items():
            if cfg_key not in merged and bo_key in best:
                merged[cfg_key] = best[bo_key]

    return merged


def run_entity_scoring(params: dict, upstream: dict) -> dict:
    """FunctionOperator callback for the entity_scoring DAG step.

    1. Open PipelineStore.
    2. Load trained GNN model — skip if not available.
    3. Score all entities via EntityAnomalyScorer.
    4. Persist alerts and convergence clusters.
    5. Return summary dict.

    Parameters (from ``params``):
        db_path : str
            Path to the pipeline SQLite database.
        model_path : str
            Path to saved GNN model checkpoint.
        as_of : float | None
            Unix timestamp for scoring window (default: now).
        scorer_config : dict | None
            Override ScorerConfig parameters.
    """
    db_path: str = params.get("db_path", ".tirra_pipeline/pipeline.db")
    model_path = Path(params.get("model_path", _DEFAULT_MODEL_PATH))
    as_of: float = params.get("as_of") or time.time()
    scorer_config: dict[str, Any] = params.get("scorer_config") or {}

    # Merge learned detector thresholds from GP-BO (Tier 3, Change 7)
    # ThresholdOptimizer persists best CUSUM/Hawkes params — load them
    # and let explicit scorer_config overrides take precedence.
    threshold_dir = params.get("threshold_dir")
    if threshold_dir:
        scorer_config = _merge_learned_thresholds(scorer_config, Path(threshold_dir))
    else:
        # Default location: alongside the pipeline DB
        default_dir = Path(db_path).parent / "threshold_bo"
        if default_dir.exists():
            scorer_config = _merge_learned_thresholds(scorer_config, default_dir)

    store = PipelineStore(db_path)
    try:
        # Check entity count
        entities = store.query_all_entities()
        entity_count = len(entities)
        log.info("Entity scoring: %d entities in store.", entity_count)

        if entity_count < _MIN_ENTITY_COUNT:
            log.info(
                "Skipping entity scoring: %d entities < threshold %d.",
                entity_count,
                _MIN_ENTITY_COUNT,
            )
            return {
                "status": "skipped",
                "reason": "insufficient_entities",
                "entity_count": entity_count,
                "alerts_stored": 0,
                "clusters_stored": 0,
            }

        # Load GNN model — skip gracefully if unavailable
        try:
            import torch
            from agent.models.gnn.trainer import Trainer
        except ImportError:
            log.warning("torch not available — skipping entity scoring.")
            return {
                "status": "skipped",
                "reason": "torch_not_available",
                "entity_count": entity_count,
                "alerts_stored": 0,
                "clusters_stored": 0,
            }

        if not model_path.exists():
            log.warning(
                "GNN model not found at %s — skipping entity scoring.",
                model_path,
            )
            return {
                "status": "skipped",
                "reason": "model_not_found",
                "entity_count": entity_count,
                "alerts_stored": 0,
                "clusters_stored": 0,
            }

        try:
            trainer = Trainer.load_model(model_path, store)
        except Exception:
            log.exception("Failed to load GNN model from %s.", model_path)
            return {
                "status": "error",
                "reason": "model_load_failed",
                "entity_count": entity_count,
                "alerts_stored": 0,
                "clusters_stored": 0,
            }

        model = trainer.model

        # Score entities
        from agent.fusion.entity_scorer import EntityAnomalyScorer, ScorerConfig

        cfg = ScorerConfig(**scorer_config)
        scorer = EntityAnomalyScorer(store, model, config=cfg)

        alerts, clusters = scorer.score_entities(as_of)

        # Persist alerts
        alerts_stored = 0
        for alert in alerts:
            try:
                store.store_entity_alert(
                    entity_id=alert.entity_id,
                    entity_type=alert.entity_type,
                    entity_name=alert.entity_name,
                    alert_time=alert.alert_time,
                    obs_type_surprise=alert.obs_type_surprise,
                    temporal_surprise=alert.temporal_surprise,
                    value_surprise=alert.value_surprise,
                    neighborhood_surprise=alert.neighborhood_surprise,
                    memory_drift=alert.memory_drift,
                    cusum_statistic=alert.cusum_statistic,
                    hawkes_intensity=alert.hawkes_intensity,
                    event_study_score=alert.event_study_score,
                    composite_surprise=alert.composite_surprise,
                    observation_count=alert.observation_count,
                    evidence_sources=alert.evidence_sources,
                    metadata=alert.metadata,
                )
                alerts_stored += 1
            except Exception:
                log.warning(
                    "Failed to store alert for entity %s.",
                    alert.entity_id,
                    exc_info=True,
                )

        # Persist clusters
        clusters_stored = 0
        for cluster in clusters:
            try:
                store.store_convergence_cluster(
                    cluster_id=cluster.cluster_id,
                    cluster_time=cluster.cluster_time,
                    member_entity_ids=[a.entity_id for a in cluster.member_alerts],
                    correlated_surprise_score=cluster.correlated_surprise_score,
                    temporal_span_hours=cluster.temporal_span_hours,
                    contributing_domains=cluster.contributing_domains,
                    contributing_tools=cluster.contributing_tools,
                    metadata=cluster.metadata,
                )
                clusters_stored += 1
            except Exception:
                log.warning(
                    "Failed to store cluster %s.",
                    cluster.cluster_id,
                    exc_info=True,
                )

        log.info(
            "Entity scoring complete: %d alerts, %d clusters stored.",
            alerts_stored,
            clusters_stored,
        )

        return {
            "status": "completed",
            "entity_count": entity_count,
            "alerts_total": len(alerts),
            "alerts_stored": alerts_stored,
            "clusters_total": len(clusters),
            "clusters_stored": clusters_stored,
            "as_of": as_of,
        }

    finally:
        store.close()


def build_entity_scoring_dag(
    db_path: str = ".tirra_pipeline/pipeline.db",
    model_path: str = _DEFAULT_MODEL_PATH,
) -> DAG:
    """Build the entity_scoring DAG.

    Single node: ``score_entities``.
    Schedule: weekdays at 18:45 UTC (after gnn_inference at 18:30,
    before feature_generation at 19:00).
    """
    dag = DAG(
        name=DAG_NAME,
        schedule="45 18 * * 1-5",
        description=(
            "Entity scoring: run prediction surprise analysis on all entities, "
            "detect convergence clusters, persist alerts and clusters"
        ),
    )

    dag.add(
        "score_entities",
        operator=run_entity_scoring,
        params={"db_path": db_path, "model_path": model_path},
    )

    return dag
