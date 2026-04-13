"""TirraMind — RL Training DAG

Scheduled after entity_scoring, this DAG:
    1. Loads historical alerts, beliefs, and market features from PipelineStore
    2. Trains/continues surprise weight learner (Phase 21a)
    3. If sufficient data, trains/updates SAC policy (Phase 21b)
    4. Saves checkpoints to rl_policy_checkpoints table
    5. Returns training metrics

Schedule: weekdays at 19:30 UTC (45 min after entity_scoring at 18:45).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import numpy as np
import torch

from agent.learning.policy.config import PolicyConfig
from agent.learning.policy.replay_buffer import ReplayBuffer
from agent.learning.policy.reward_fn import RewardFunction
from agent.learning.policy.sac import SACTrainer
from agent.learning.policy.state_assembler import StateAssembler
from agent.learning.policy.asset_mapper import AssetMapper
from agent.learning.policy.weight_learner import SurpriseWeightLearner
from agent.pipeline.dag import DAG
from agent.pipeline.store import PipelineStore

log = logging.getLogger(__name__)

DAG_NAME = "rl_training"
DEPENDS_ON = ["entity_scoring"]

# Minimum alerts required before RL training starts
_MIN_ALERTS_FOR_WEIGHTS = 200
_MIN_ALERTS_FOR_SAC = 500


def run_rl_training(params: dict, upstream: dict) -> dict:
    """FunctionOperator callback for the rl_training DAG step.

    Parameters (from ``params``):
        db_path : str
        config : dict | None — merged into PolicyConfig

    Returns
    -------
    Dict with keys: weight_learner_trained, sac_trained, weights, metrics
    """
    db_path: str = params.get("db_path", ".tirra_pipeline/pipeline.db")
    config_overrides: dict = params.get("config") or {}

    store = PipelineStore(db_path)
    config = PolicyConfig(**config_overrides) if config_overrides else PolicyConfig()

    result: dict[str, Any] = {
        "weight_learner_trained": False,
        "sac_trained": False,
        "weights": None,
        "metrics": {},
    }

    # ── Load historical data ──────────────────────────────────
    alerts = _load_alert_history(store)
    if len(alerts) < _MIN_ALERTS_FOR_WEIGHTS:
        log.info(
            "Not enough alerts for RL training (%d < %d)",
            len(alerts),
            _MIN_ALERTS_FOR_WEIGHTS,
        )
        return result

    # ── Phase 21a: Surprise weight learning ───────────────────
    surprise_matrix, returns = _build_surprise_returns(alerts, store)

    if (
        surprise_matrix is not None
        and len(surprise_matrix) >= config.weight_learner.min_train_periods
    ):
        try:
            learner = SurpriseWeightLearner(config.weight_learner)
            learner.fit(surprise_matrix, returns)
            weights = learner.get_learned_weights()
            result["weights"] = weights
            result["weight_learner_trained"] = True
            result["metrics"]["weight_learner"] = {
                "weights": list(weights),
                "oos_sharpe": (
                    float(learner.oos_sharpe)
                    if hasattr(learner, "oos_sharpe")
                    else None
                ),
            }
            log.info("Weight learner trained: %s", weights)
        except Exception as e:
            log.warning("Weight learner failed: %s", e)

    # ── Phase 21b: SAC training ───────────────────────────────
    if len(alerts) < _MIN_ALERTS_FOR_SAC:
        log.info(
            "Not enough alerts for SAC training (%d < %d)",
            len(alerts),
            _MIN_ALERTS_FOR_SAC,
        )
        return result

    try:
        sac_metrics = _train_sac(store, config, alerts)
        result["sac_trained"] = True
        result["metrics"]["sac"] = sac_metrics
    except Exception as e:
        log.warning("SAC training failed: %s", e)

    return result


def _load_alert_history(store: PipelineStore) -> list[dict]:
    """Load alert history from store. Returns list of alert dicts."""
    try:
        rows = store.query_entity_alerts(limit=10000)
        return rows if rows else []
    except Exception:
        return []


def _build_surprise_returns(
    alerts: list[dict], store: PipelineStore
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Build aligned surprise matrix and returns array from alert history.

    Returns (surprise_matrix, returns) or (None, None) if insufficient.
    """
    if not alerts:
        return None, None

    # Group alerts by timestamp bucket (weekly)
    by_week: dict[int, list[dict]] = {}
    for a in alerts:
        ts = a.get("alert_time", a.get("timestamp", 0))
        week = int(ts // (7 * 86400))
        by_week.setdefault(week, []).append(a)

    if len(by_week) < 10:
        return None, None

    weeks = sorted(by_week.keys())
    T = len(weeks)

    # Average surprise per week
    surprise_matrix = np.zeros((T, 5), dtype=np.float64)
    for i, w in enumerate(weeks):
        week_alerts = by_week[w]
        for key_idx, key in enumerate(
            [
                "obs_type_surprise",
                "temporal_surprise",
                "value_surprise",
                "neighborhood_surprise",
                "memory_drift",
            ]
        ):
            vals = [a.get(key, 0.0) for a in week_alerts]
            surprise_matrix[i, key_idx] = float(np.mean(vals)) if vals else 0.0

    # Synthetic returns placeholder — in production, loaded from market data
    # Use composite surprise as a proxy for now
    returns = np.diff(surprise_matrix.mean(axis=1))
    returns = np.concatenate([[0.0], returns])

    return surprise_matrix, returns


def _train_sac(
    store: PipelineStore,
    config: PolicyConfig,
    alerts: list[dict],
) -> dict[str, float]:
    """Train SAC for one epoch on historical transitions.

    Returns training metrics.
    """
    assembler = StateAssembler()
    state_dim = assembler.state_dim
    mapper = AssetMapper(store)
    tradeable = mapper.tradeable_entities()
    action_dim = max(len(tradeable), 1)  # at least 1

    cfg = config.sac

    # Try to load existing checkpoint
    trainer = None
    checkpoint = store.load_latest_rl_checkpoint("sac")
    if checkpoint is not None:
        try:
            trainer = SACTrainer.load(
                checkpoint["state_dict_blob"],
                state_dim,
                action_dim,
                cfg,
            )
            log.info("Resumed SAC from checkpoint")
        except Exception as e:
            log.warning("Could not load SAC checkpoint: %s", e)

    if trainer is None:
        trainer = SACTrainer(state_dim, action_dim, cfg)

    # Load existing transitions from store
    transitions = store.query_rl_transitions(limit=cfg.buffer_size)
    buffer = ReplayBuffer(cfg.buffer_size, state_dim, action_dim)

    for t in transitions:
        try:
            state = np.array(json.loads(t["state_json"]), dtype=np.float32)
            action = np.array(json.loads(t["action_json"]), dtype=np.float32)
            next_state = np.array(json.loads(t["next_state_json"]), dtype=np.float32)
            buffer.push(state, action, t["reward"], next_state, bool(t["done"]))
        except (json.JSONDecodeError, KeyError, ValueError):
            continue

    if len(buffer) < cfg.batch_size:
        log.info(
            "Not enough transitions for SAC training (%d < %d)",
            len(buffer),
            cfg.batch_size,
        )
        return {"status": "insufficient_data", "buffer_size": len(buffer)}

    # Train for a fixed number of steps
    n_updates = min(100, len(buffer) // cfg.batch_size)
    metrics_history: list[dict[str, float]] = []

    for _ in range(n_updates):
        m = trainer.update(buffer)
        metrics_history.append(m)

    # Save checkpoint
    state_blob = trainer.save()
    avg_critic = float(np.mean([m["critic_loss"] for m in metrics_history]))
    avg_actor = float(np.mean([m["actor_loss"] for m in metrics_history]))

    store.store_rl_checkpoint(
        policy_type="sac",
        config_json=json.dumps({"state_dim": state_dim, "action_dim": action_dim}),
        state_dict_blob=state_blob,
        metrics_json=json.dumps(
            {"avg_critic_loss": avg_critic, "avg_actor_loss": avg_actor}
        ),
        is_best=False,
    )

    return {
        "n_updates": n_updates,
        "avg_critic_loss": avg_critic,
        "avg_actor_loss": avg_actor,
        "final_alpha": metrics_history[-1]["alpha"] if metrics_history else 0.0,
        "buffer_size": len(buffer),
    }


def build_rl_training_dag(
    db_path: str = ".tirra_pipeline/pipeline.db",
) -> DAG:
    """Build the rl_training DAG.

    Single node: ``train_rl_policy``.
    Schedule: weekdays at 19:30 UTC (after entity_scoring at 18:45).
    """
    dag = DAG(
        name=DAG_NAME,
        schedule="30 19 * * 1-5",
        description=(
            "RL policy training: learn surprise weights (21a) and "
            "train SAC actor-critic (21b) from accumulated entity alerts"
        ),
    )

    dag.add(
        "train_rl_policy",
        operator=run_rl_training,
        params={"db_path": db_path},
    )

    return dag
