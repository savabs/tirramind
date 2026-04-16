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
from agent.learning.policy.feature_gate import FeatureGate, FeatureGateConfig
from agent.learning.policy.replay_buffer import ReplayBuffer
from agent.learning.policy.reward_fn import RewardFunction
from agent.learning.policy.sac import SACTrainer
from agent.learning.policy.state_assembler import InstrumentStateAssembler
from agent.learning.policy.state_assembler import DifferentiableStateAssembler
from agent.learning.policy.state_encoder import LearnedStateEncoder, StateEncoderConfig
from agent.learning.policy.asset_mapper import AssetMapper
from agent.learning.policy.weight_learner import SurpriseWeightLearner
from agent.models.diff_kalman import DifferentiableKalmanFilter
from agent.tools.instrument_universe import tradeable_instruments
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
    instruments = tradeable_instruments()
    tickers = [inst.ticker for inst in instruments]
    assembler = InstrumentStateAssembler(instrument_tickers=tickers)
    state_dim = assembler.state_dim
    action_dim = max(len(tickers), 1)  # at least 1

    cfg = config.sac

    # Build encoder if configured
    encoder = None
    enc_cfg = config.state_encoder
    if enc_cfg is not None:
        encoder = LearnedStateEncoder(enc_cfg)

    # Attach feature gate (Change 11) if configured
    if encoder is not None and config.feature_gate is not None:
        gate = FeatureGate(config.feature_gate)
        encoder.set_feature_gate(gate)
        log.info(
            "Feature gate attached: %d groups, floor=%.3f",
            config.feature_gate.n_feature_groups,
            config.feature_gate.gate_floor,
        )

    # Try to load existing checkpoint
    trainer = None
    checkpoint = store.load_latest_rl_checkpoint("sac")
    if checkpoint is not None:
        try:
            trainer = SACTrainer.load(
                checkpoint["state_dict_bytes"],
                state_dim,
                action_dim,
                cfg,
            )
            log.info("Resumed SAC from checkpoint")
        except Exception as e:
            log.warning("Could not load SAC checkpoint: %s", e)

    if trainer is None:
        trainer = SACTrainer(state_dim, action_dim, cfg, encoder=encoder)

    # Set regime context for feature gate (Change 11)
    # Use zero vector during cold start; future: load HMM posterior from store
    if config.feature_gate is not None:
        regime_dim = config.feature_gate.regime_dim
        regime_ctx = torch.zeros(regime_dim)
        trainer.set_regime_context(regime_ctx)

    # Load existing transitions from store
    transitions = store.query_rl_transitions(limit=cfg.buffer_size)
    buffer = ReplayBuffer(cfg.buffer_size, state_dim, action_dim)

    for t in transitions:
        try:
            state = np.array(t["state"], dtype=np.float32)
            action = np.array(t["action"], dtype=np.float32)
            next_state = np.array(t["next_state"], dtype=np.float32)
            buffer.push(state, action, t["reward"], next_state, bool(t["done"]))
        except (KeyError, ValueError, TypeError):
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

    # ── Model-based Kalman augmentation (Phase B) ─────────────
    aux_metrics = _kalman_augmentation(
        store, config, trainer, assembler, tickers, alerts, action_dim,
    )

    # Save checkpoint
    state_blob = trainer.save()
    avg_critic = float(np.mean([m["critic_loss"] for m in metrics_history]))
    avg_actor = float(np.mean([m["actor_loss"] for m in metrics_history]))

    store.store_rl_checkpoint(
        policy_type="sac",
        config={
            "state_dim": state_dim,
            "action_dim": action_dim,
            "has_encoder": encoder is not None,
        },
        state_dict_bytes=state_blob,
        metrics={"avg_critic_loss": avg_critic, "avg_actor_loss": avg_actor},
        is_best=False,
    )

    return {
        "n_updates": n_updates,
        "avg_critic_loss": avg_critic,
        "avg_actor_loss": avg_actor,
        "final_alpha": metrics_history[-1]["alpha"] if metrics_history else 0.0,
        "buffer_size": len(buffer),
        **aux_metrics,
    }


def _kalman_augmentation(
    store: PipelineStore,
    config: PolicyConfig,
    trainer: SACTrainer,
    assembler: InstrumentStateAssembler,
    tickers: list[str],
    alerts: list[dict],
    action_dim: int,
) -> dict[str, float]:
    """Model-based augmentation: backprop SAC actor loss through DiffKalman.

    This creates a differentiable path:
        DiffKalman.predict → update → get_beliefs_differentiable()
        → DifferentiableStateAssembler → Actor.sample → auxiliary_loss
        → backward() → Kalman param gradients

    The auxiliary loss is scaled by ``config.sac.aux_kalman_weight`` (default
    0.01) so it fine-tunes Kalman params toward policy-useful states without
    overwhelming the EM-fitted priors.

    Analogous to Dreamer (Hafner et al. 2020): real replay transitions are
    the primary learning signal; the model-based path is auxiliary.

    Returns dict of auxiliary metrics (empty if augmentation is skipped).
    """
    cfg = config.sac
    if cfg.aux_kalman_weight <= 0.0:
        return {}

    # Try to load DiffKalman from world model checkpoint
    diff_kalman = _load_diff_kalman(store)
    if diff_kalman is None:
        log.info("DiffKalman not available — skipping model-based augmentation")
        return {}

    diff_assembler = DifferentiableStateAssembler(instrument_tickers=tickers)

    # Build a small observation batch from the most recent alerts
    obs_batch = _build_observation_batch(alerts, diff_kalman.obs_dim, max_steps=10)
    if not obs_batch:
        log.info("No observations for Kalman augmentation")
        return {}

    # Separate optimizer for Kalman parameters only
    kalman_optim = torch.optim.Adam(diff_kalman.parameters(), lr=cfg.kalman_lr)

    total_aux_loss = 0.0
    n_aux_steps = 0

    for obs_np, regime in obs_batch:
        kalman_optim.zero_grad()

        # Forward through Kalman (preserving autograd)
        diff_kalman.predict(regime)
        obs_tensor = torch.from_numpy(obs_np.astype(np.float32))
        diff_kalman.update(obs_tensor)

        # Get differentiable beliefs
        means, variances = diff_kalman.get_beliefs_differentiable()

        # Build differentiable state
        # Use empty instrument surprises / alerts / market for the auxiliary path
        # — the point is to get gradients through the belief block, not accurate
        # non-belief features (those come from replay for the primary update).
        diff_state, _ = diff_assembler.assemble(
            instrument_surprises={},
            entity_alerts=[],
            belief_means=means,
            belief_variances=variances,
            market_features={},
        )
        diff_state = diff_state.unsqueeze(0)  # batch dim

        # Encode if trainer has an encoder
        encoded = trainer._encode_state(diff_state)

        # Actor forward (no gradient to actor params — we only want Kalman grads)
        with torch.no_grad():
            alpha = trainer._alpha_sched.alpha_tensor

        action, log_prob = trainer._actor.sample(encoded)

        # Q-value estimate (detach critic to avoid corrupting its params)
        with torch.no_grad():
            q1, q2 = trainer._critic(encoded.detach(), action.detach())
            q_min = torch.min(q1, q2)

        # Auxiliary actor loss — same form as SAC actor loss
        aux_loss = cfg.aux_kalman_weight * (alpha * log_prob - q_min).mean()

        # Backward through: aux_loss → log_prob → actor → encoded state
        #   → diff_state (via belief_block) → means/variances → Kalman params
        aux_loss.backward()

        # Clip Kalman gradients
        torch.nn.utils.clip_grad_norm_(
            diff_kalman.parameters(), cfg.kalman_grad_clip
        )
        kalman_optim.step()

        total_aux_loss += float(aux_loss.item())
        n_aux_steps += 1

        # Reset Kalman graph to prevent memory accumulation
        diff_kalman.reset()

    # Compute gradient norm for monitoring
    grad_norm = 0.0
    for p in diff_kalman.parameters():
        if p.grad is not None:
            grad_norm += float(p.grad.data.norm(2).item() ** 2)
    grad_norm = grad_norm**0.5

    avg_aux = total_aux_loss / max(n_aux_steps, 1)
    log.info(
        "Kalman augmentation: %d steps, avg_aux_loss=%.6f, grad_norm=%.6f",
        n_aux_steps,
        avg_aux,
        grad_norm,
    )

    return {
        "aux_kalman_steps": n_aux_steps,
        "aux_kalman_loss": avg_aux,
        "kalman_grad_norm": grad_norm,
    }


def _load_diff_kalman(store: PipelineStore) -> DifferentiableKalmanFilter | None:
    """Try to load a DifferentiableKalmanFilter from world model checkpoint.

    Returns None if no checkpoint is available.
    """
    try:
        checkpoint = store.load_latest_rl_checkpoint("diff_kalman")
        if checkpoint is None:
            return None
        state_bytes = checkpoint.get("state_dict_bytes")
        if state_bytes is None:
            return None
        import io

        buf = io.BytesIO(state_bytes)
        state_dict = torch.load(buf, map_location="cpu", weights_only=True)
        # Infer dimensions from state_dict
        # F has shape (state_dim, state_dim), H has shape (obs_dim, state_dim)
        regime_names = [
            k.split(".")[1] for k in state_dict if k.startswith("_F.")
        ]
        if not regime_names:
            return None
        first_F = state_dict[f"_F.{regime_names[0]}"]
        state_dim = first_F.shape[0]
        obs_dim = state_dict["_H"].shape[0]

        kalman = DifferentiableKalmanFilter(
            state_dim=state_dim, obs_dim=obs_dim, regime_names=regime_names
        )
        kalman.load_state_dict(state_dict)
        return kalman
    except Exception as e:
        log.warning("Could not load DiffKalman: %s", e)
        return None


def _build_observation_batch(
    alerts: list[dict],
    obs_dim: int,
    max_steps: int = 10,
) -> list[tuple[np.ndarray, str]]:
    """Convert recent alerts into (observation_vector, regime) pairs.

    Each alert's 5 surprise fields are mapped into an observation vector
    zero-padded to obs_dim.  The regime is set to 'expansion' as default
    (the actual regime would come from the HMM, but for the auxiliary
    gradient path the regime choice has minor impact on gradient quality).
    """
    # Take most recent alerts
    recent = sorted(
        alerts,
        key=lambda a: a.get("alert_time", a.get("timestamp", 0)),
        reverse=True,
    )[:max_steps]

    batch: list[tuple[np.ndarray, str]] = []
    for alert in recent:
        obs = np.zeros(obs_dim, dtype=np.float32)
        for j, key in enumerate(
            [
                "obs_type_surprise",
                "temporal_surprise",
                "value_surprise",
                "neighborhood_surprise",
                "memory_drift",
            ]
        ):
            if j < obs_dim:
                obs[j] = float(alert.get(key, 0.0))
        # Default regime — override with HMM posterior if available
        regime = alert.get("regime", "expansion")
        batch.append((obs, regime))

    return batch


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
