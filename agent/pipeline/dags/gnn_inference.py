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

from agent.learning.meta_scheduler import MetaScheduler, compute_refit_reward
from agent.pipeline.dag import DAG
from agent.pipeline.regime_gate import get_current_regime, is_high_changepoint
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
    3. Load or create MetaScheduler for dynamic epoch selection (Change 14).
    4. Load or train GNN model.
    5. Save model + scheduler state.
    6. Return summary dict.

    Parameters (from ``params``):
        db_path : str
            Path to the pipeline SQLite database.
        model_path : str
            Path to save/load GNN model checkpoint.
        min_entities : int
            Minimum entity count to justify training (default: 10).
        trainer_config : dict | None
            Override TrainerConfig parameters.
        use_scheduler : bool
            Whether to use MetaScheduler for epoch selection (default: True).
        gnn_epochs : int | None
            Explicit override for epochs. If None and use_scheduler, scheduler decides.
    """
    db_path: str = params.get("db_path", ".tirra_pipeline/pipeline.db")
    model_path = Path(params.get("model_path", _DEFAULT_MODEL_PATH))
    min_entities: int = params.get("min_entities", _MIN_ENTITY_COUNT)
    trainer_config: dict[str, Any] = params.get("trainer_config") or {}
    use_scheduler: bool = params.get("use_scheduler", True)
    explicit_epochs: int | None = params.get("gnn_epochs")

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

        # ── MetaScheduler for dynamic epochs (Change 14) ─────
        scheduler: MetaScheduler | None = None
        scheduler_path = Path(db_path).parent / "meta_scheduler.json"
        epochs_arm: int = trainer_config.get("epochs", 10)

        if explicit_epochs is not None:
            epochs_arm = explicit_epochs
        elif use_scheduler:
            try:
                scheduler = MetaScheduler(persist_path=scheduler_path)
                epochs_arm = scheduler.suggest("gnn_epochs")
            except Exception as exc:
                log.warning("Failed to load MetaScheduler for GNN: %s", exc)
                scheduler = None

        # Apply epochs to trainer config
        trainer_config["epochs"] = epochs_arm
        log.info(
            "GNN training epochs: %d (scheduler=%s)", epochs_arm, scheduler is not None
        )

        cfg = TrainerConfig(**trainer_config)
        trained = False

        # ── Phase 49b: Force retrain when regime changepoint is high ────
        # If a structural break was detected (changepoint_posterior ≥ 0.9),
        # the existing model's learned representations may no longer be
        # valid for the new regime.  Force a full retrain rather than an
        # incremental EWC step, which cannot recover from a large structural
        # shift.  The old checkpoint is deleted so the else-branch below runs.
        regime_ctx = get_current_regime(store)
        if model_path.exists() and is_high_changepoint(store, ctx=regime_ctx):
            log.warning(
                "Phase 49b: high changepoint detected (posterior=%.3f) — "
                "forcing GNN retrain instead of loading checkpoint.",
                regime_ctx.changepoint_posterior,
            )
            try:
                model_path.unlink()
            except OSError as exc:
                log.warning("Could not remove old checkpoint: %s", exc)

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

        # ── Phase 46: EWC online update ─────────────────────────────────
        # When we LOADED an existing model (not retrained from scratch), check
        # whether enough new observations have accumulated since the last online
        # update.  If so, apply one EWC-regularised gradient step to the loaded
        # model before saving it back.  When we just ran a full train(), the
        # EWC anchor is already set to now, so `since_ts` will return an empty
        # list and we skip the step (correct — no point doing a second update
        # immediately after training).
        online_updated: bool = False
        online_result: dict[str, float] = {}

        if not trained and trainer._ewc_state is not None:
            since_ts = trainer._ewc_state.last_update_ts
            try:
                new_obs = store.query_all_observations(since=since_ts)
                log.info(
                    "Phase 46 EWC check: %d new observations since last update "
                    "(ts=%.0f), threshold=%d.",
                    len(new_obs),
                    since_ts,
                    cfg.online_batch_threshold,
                )
                if len(new_obs) >= cfg.online_batch_threshold:
                    log.info(
                        "Triggering EWC online continual-learning step " "(%d events).",
                        len(new_obs),
                    )
                    update_result = trainer.online_update(new_obs)
                    online_updated = True
                    online_result = update_result
                    log.info(
                        "EWC online update complete: loss_new=%.4f "
                        "loss_ewc=%.4f loss_total=%.4f n_events=%d",
                        update_result["loss_new"],
                        update_result["loss_ewc"],
                        update_result["loss_total"],
                        int(update_result["n_events"]),
                    )
                else:
                    log.info(
                        "EWC online update skipped: %d new observations "
                        "< threshold %d.",
                        len(new_obs),
                        cfg.online_batch_threshold,
                    )
            except Exception as exc:
                # Non-fatal: online update failure should not abort the DAG run.
                # The loaded model is still valid for inference.
                log.warning(
                    "EWC online update failed (non-fatal, continuing with "
                    "loaded model): %s",
                    exc,
                )
        elif not trained and trainer._ewc_state is None:
            log.info(
                "Phase 46 EWC: loaded model has no EWC state "
                "(pre-Phase-46 checkpoint). Skipping online update. "
                "Run a full retrain to compute Fisher diagonal."
            )

        # Save model (includes updated EWC bookkeeping when online_updated)
        try:
            trainer.save_model(model_path)
            log.info("Saved GNN model to %s.", model_path)
        except Exception:
            log.warning("Failed to save GNN model to %s.", model_path)

        # Record outcome for scheduler (Change 14)
        if scheduler is not None and trained:
            try:
                val_loss_after = getattr(trainer, "last_val_loss", None)
                after_metrics = (
                    {"val_loss": val_loss_after} if val_loss_after is not None else {}
                )
                reward = compute_refit_reward("gnn_epochs", {}, after_metrics)
                scheduler.record_outcome("gnn_epochs", epochs_arm, reward)
                scheduler.save()
                try:
                    store.store_component_performance(
                        "gnn_epochs",
                        time.time(),
                        epochs_arm,
                        reward,
                        {"val_loss": val_loss_after, "entity_count": entity_count},
                    )
                except Exception:
                    pass
            except Exception as exc:
                log.debug("Failed to record gnn_epochs outcome: %s", exc)

        return {
            "status": "completed",
            "entity_count": entity_count,
            "model_path": str(model_path),
            "trained": trained,
            "epochs": epochs_arm,
            "using_scheduler": scheduler is not None,
            # Phase 46
            "online_updated": online_updated,
            "online_update_result": online_result,
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
