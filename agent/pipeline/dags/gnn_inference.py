"""
TirraMind — GNN Inference DAG

Runs after daily_collection, before feature_generation.  Trains or loads the
HetTGN model and saves a persistent checkpoint.

Schedule: weekdays at 18:30 UTC (30 min after daily_collection, 30 min before
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

# Minimum entity count to justify training a GNN
_MIN_ENTITY_COUNT = 10

# Default model checkpoint location
_DEFAULT_MODEL_PATH = ".tirra_pipeline/gnn_model.pt"


def run_gnn_inference(params: dict, upstream: dict) -> dict:
    """FunctionOperator callback for the gnn_inference DAG step.

    1. Open PipelineStore.
    2. Check entity count — skip if below threshold.
    3. Load or train GNN model.
    4. Save model to checkpoint path.
    5. Return summary dict.

    Parameters (from ``params``):
        db_path : str
            Path to the pipeline SQLite database.
        model_path : str
            Path to save/load GNN model checkpoint.
        min_entities : int
            Minimum entity count to justify training (default: 10).
        trainer_config : dict | None
            Override TrainerConfig parameters.
    """
    db_path: str = params.get("db_path", ".tirra_pipeline/pipeline.db")
    model_path = Path(params.get("model_path", _DEFAULT_MODEL_PATH))
    min_entities: int = params.get("min_entities", _MIN_ENTITY_COUNT)
    trainer_config: dict[str, Any] = params.get("trainer_config") or {}

    store = PipelineStore(db_path)
    try:
        # Check entity count
        entities = store.query_all_entities()
        entity_count = len(entities)
        log.info("GNN inference: %d entities in store.", entity_count)

        if entity_count < min_entities:
            log.info(
                "Skipping GNN training: %d entities < threshold %d.",
                entity_count,
                min_entities,
            )
            return {
                "status": "skipped",
                "reason": "insufficient_entities",
                "entity_count": entity_count,
                "model_path": str(model_path),
                "trained": False,
            }

        # Lazy import to avoid torch at module level
        try:
            from agent.models.gnn.trainer import Trainer, TrainerConfig
        except ImportError:
            log.warning("torch not available — skipping GNN inference.")
            return {
                "status": "skipped",
                "reason": "torch_not_available",
                "entity_count": entity_count,
                "model_path": str(model_path),
                "trained": False,
            }

        cfg = TrainerConfig(**trainer_config)
        trained = False

        if model_path.exists():
            try:
                trainer = Trainer.load_model(model_path, store)
                log.info("Loaded existing GNN model from %s.", model_path)
            except Exception:
                log.warning("Failed to load model from %s — retraining.", model_path)
                trainer = Trainer(store, cfg)
                trainer.build_model()
                trainer.train()
                trained = True
        else:
            trainer = Trainer(store, cfg)
            trainer.build_model()
            trainer.train()
            trained = True

        # Save model
        try:
            trainer.save_model(model_path)
            log.info("Saved GNN model to %s.", model_path)
        except Exception:
            log.warning("Failed to save GNN model to %s.", model_path)

        return {
            "status": "completed",
            "entity_count": entity_count,
            "model_path": str(model_path),
            "trained": trained,
        }

    finally:
        store.close()


def build_gnn_inference_dag(
    db_path: str = ".tirra_pipeline/pipeline.db",
    model_path: str = _DEFAULT_MODEL_PATH,
) -> DAG:
    """Build the gnn_inference DAG.

    Single node: ``train_gnn``.
    Schedule: weekdays at 18:30 UTC, 30 min before feature_generation.
    """
    dag = DAG(
        name="gnn_inference",
        schedule="30 18 * * 1-5",
        description=(
            "GNN inference: train or load HetTGN model on entity graph, "
            "save checkpoint for feature_generation to consume"
        ),
    )

    dag.add(
        "train_gnn",
        operator=run_gnn_inference,
        params={"db_path": db_path, "model_path": model_path},
    )

    return dag
