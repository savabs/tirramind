"""TirraMind — Inference DAG

Nightly inference pipeline: load models → GNN forward pass → SAC allocation → emit weights.

Four strictly sequential FunctionOperator nodes:
    1. load_models     — verify GNN checkpoint (file) + SAC checkpoint (DB) exist
    2. gnn_inference   — build today's entity graph, forward pass, extract surprises
    3. sac_inference   — assemble instrument state, deterministic SAC action → weights
    4. emit_portfolio  — persist weights to portfolio_weights table, compute paper P&L

Schedule: 45 19 * * 1-5  (19:45 UTC weekdays, after all upstream DAGs complete)

Graceful degradation:
    If either model is missing, load_models returns status="skipped" and
    downstream nodes propagate the skip without error. This lets the DAG
    run harmlessly before enough training data has accumulated.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from agent.pipeline.dag import DAG
from agent.pipeline.store import PipelineStore

log = logging.getLogger(__name__)

DAG_NAME = "inference"
_DEFAULT_MODEL_PATH = ".tirra_pipeline/gnn_model.pt"


# ── Node 1: load_models ──────────────────────────────────────


def _load_models(params: dict[str, Any], upstream: dict[str, Any]) -> dict[str, Any]:
    """Verify that GNN and SAC models are available for inference.

    Does NOT load the models into memory (torch import is deferred to
    nodes that need it). Instead, checks existence and returns the
    paths/config that downstream nodes use.

    Returns
    -------
    Dict with keys:
        status : "ready" | "skipped"
        reason : str  (only when skipped)
        gnn_model_path : str
        sac_config : dict  (state_dim, action_dim from checkpoint)
        has_gnn : bool
        has_sac : bool
    """
    db_path: str = params.get("db_path", ".tirra_pipeline/pipeline.db")
    model_path = params.get("model_path", _DEFAULT_MODEL_PATH)

    result: dict[str, Any] = {
        "status": "ready",
        "gnn_model_path": str(model_path),
        "sac_config": {},
        "has_gnn": False,
        "has_sac": False,
    }

    # Check GNN model file
    if Path(model_path).exists():
        result["has_gnn"] = True
        log.info("GNN model found at %s", model_path)
    else:
        log.info("GNN model not found at %s — will skip GNN inference.", model_path)

    # Check SAC checkpoint in DB
    store = PipelineStore(db_path)
    try:
        checkpoint = store.load_latest_rl_checkpoint("sac")
        if checkpoint is not None:
            result["has_sac"] = True
            result["sac_config"] = checkpoint.get("config", {})
            log.info(
                "SAC checkpoint found (saved_at=%.0f)", checkpoint.get("saved_at", 0)
            )
        else:
            log.info("No SAC checkpoint found — will skip SAC inference.")
    finally:
        store.close()

    if not result["has_gnn"] and not result["has_sac"]:
        result["status"] = "skipped"
        result["reason"] = "no_models_available"
        log.info("Inference DAG skipped: no models available.")

    return result


# ── Node 2: gnn_inference ────────────────────────────────────


def _gnn_inference(params: dict[str, Any], upstream: dict[str, Any]) -> dict[str, Any]:
    """Run GNN forward pass and extract per-instrument surprises.

    Loads the trained HetTGN, builds the current entity graph,
    runs inference, and uses SurpriseExtractor to produce per-entity
    surprise vectors. Filters for instrument-type entities and returns
    {ticker → surprise_vector} for the SAC inference node.

    Returns
    -------
    Dict with keys:
        status : "completed" | "skipped"
        instrument_surprises : dict[str, list[float]]  (ticker → 5 floats)
        entity_count : int
    """
    load_result = upstream.get("load_models", {})
    if load_result.get("status") == "skipped":
        return {
            "status": "skipped",
            "reason": "upstream_skipped",
            "instrument_surprises": {},
        }

    if not load_result.get("has_gnn"):
        return {
            "status": "skipped",
            "reason": "no_gnn_model",
            "instrument_surprises": {},
        }

    db_path: str = params.get("db_path", ".tirra_pipeline/pipeline.db")
    model_path = load_result.get("gnn_model_path", _DEFAULT_MODEL_PATH)

    try:
        from agent.models.gnn.trainer import Trainer
    except ImportError:
        log.warning("torch not available — skipping GNN inference.")
        return {
            "status": "skipped",
            "reason": "torch_not_available",
            "instrument_surprises": {},
        }

    store = PipelineStore(db_path)
    try:
        trainer = Trainer.load_model(model_path, store)

        # Forward pass: embeddings + id_map
        embeddings, id_map = trainer.infer()
        if not embeddings:
            return {
                "status": "skipped",
                "reason": "empty_graph",
                "instrument_surprises": {},
            }

        # Extract per-entity surprises using SurpriseExtractor
        from agent.fusion.surprise import SurpriseExtractor

        # Get recent observations for surprise comparison
        recent_obs = store.query_all_observations(
            since=_days_ago(7),
        )

        extractor = SurpriseExtractor()
        # We need the model, graph data, id_map, and observations
        data, _, _ = trainer._graph_builder.build()
        surprises = extractor.extract(
            model=trainer.model,
            data=data,
            id_map=id_map,
            observations=recent_obs,
        )

        # Filter for instrument-type entities → {ticker: surprise_vector}
        instrument_surprises: dict[str, list[float]] = {}
        for entity_id, es in surprises.items():
            if es.entity_type == "instrument":
                # entity_id for instruments is the ticker
                instrument_surprises[entity_id] = list(es.surprise_vector())

        log.info(
            "GNN inference: %d total entities, %d instrument surprises.",
            len(surprises),
            len(instrument_surprises),
        )

        return {
            "status": "completed",
            "instrument_surprises": instrument_surprises,
            "entity_count": len(surprises),
        }

    except Exception as exc:
        log.warning("GNN inference failed: %s", exc)
        return {"status": "error", "reason": str(exc), "instrument_surprises": {}}
    finally:
        store.close()


# ── Node 3: sac_inference ────────────────────────────────────


def _sac_inference(params: dict[str, Any], upstream: dict[str, Any]) -> dict[str, Any]:
    """Run SAC policy to produce per-instrument weight allocation.

    Assembles the instrument state vector from GNN surprises, stored
    entity alerts, beliefs, and market features, then queries the SAC
    policy for a deterministic action.

    Returns
    -------
    Dict with keys:
        status : "completed" | "skipped"
        weights : dict[str, float]  (ticker → weight)
        instrument_tickers : list[str]
    """
    load_result = upstream.get("load_models", {})
    gnn_result = upstream.get("gnn_inference", {})

    if load_result.get("status") == "skipped":
        return {"status": "skipped", "reason": "upstream_skipped", "weights": {}}

    if not load_result.get("has_sac"):
        return {"status": "skipped", "reason": "no_sac_model", "weights": {}}

    db_path: str = params.get("db_path", ".tirra_pipeline/pipeline.db")
    sac_config = load_result.get("sac_config", {})
    state_dim = sac_config.get("state_dim")
    action_dim = sac_config.get("action_dim")

    if state_dim is None or action_dim is None:
        return {"status": "skipped", "reason": "missing_sac_dimensions", "weights": {}}

    try:
        import torch  # noqa: F401
        from agent.learning.policy.sac import SACTrainer
        from agent.learning.policy.state_assembler import InstrumentStateAssembler
        from agent.tools.instrument_universe import tradeable_instruments
        from agent.fusion.alert import EntityAlert
        from agent.models.belief import BeliefState
    except ImportError as exc:
        log.warning("Required imports not available: %s", exc)
        return {"status": "skipped", "reason": "import_error", "weights": {}}

    store = PipelineStore(db_path)
    try:
        # Load SAC checkpoint
        checkpoint = store.load_latest_rl_checkpoint("sac")
        if checkpoint is None:
            return {"status": "skipped", "reason": "checkpoint_vanished", "weights": {}}

        trainer = SACTrainer.load(
            checkpoint["state_dict_bytes"],
            state_dim,
            action_dim,
        )

        # Build instrument ticker list (same ordering SAC was trained with)
        instruments = tradeable_instruments()
        tickers = [inst.ticker for inst in instruments]

        # Assemble state
        assembler = InstrumentStateAssembler(instrument_tickers=tickers)

        # Instrument surprises from GNN node
        instrument_surprises: dict[str, tuple[float, ...]] = {}
        raw_surprises = gnn_result.get("instrument_surprises", {})
        for ticker, sv in raw_surprises.items():
            instrument_surprises[ticker] = tuple(sv)

        # Load recent entity alerts from store
        recent_alert_rows = store.query_entity_alerts(
            since=_days_ago(7),
            limit=200,
        )
        entity_alerts = [
            EntityAlert(
                entity_id=r["entity_id"],
                entity_type=r["entity_type"],
                entity_name=r["entity_name"],
                alert_time=r["alert_time"],
                obs_type_surprise=r["obs_type_surprise"],
                temporal_surprise=r["temporal_surprise"],
                value_surprise=r["value_surprise"],
                neighborhood_surprise=r["neighborhood_surprise"],
                memory_drift=r["memory_drift"],
                cusum_statistic=r["cusum_statistic"],
                hawkes_intensity=r["hawkes_intensity"],
                event_study_score=r["event_study_score"],
                composite_surprise=r["composite_surprise"],
            )
            for r in recent_alert_rows
        ]

        # ── Load beliefs from world model store ──────────────
        belief_rows = store.query_all_latest_beliefs()
        beliefs: list[BeliefState] = []
        for row in belief_rows:
            try:
                beliefs.append(BeliefState.from_dict(row))
            except (KeyError, TypeError):
                pass  # skip malformed rows

        # ── Market features: pack global belief means ────────
        # World-model latent beliefs (no entity_id) provide global
        # state context that SAC can use even when per-entity beliefs
        # are not yet available.
        market_features: dict[str, float] = {}
        for b in beliefs:
            if b.entity_id is None and b.mean is not None:
                market_features[f"belief.{b.variable_name}"] = b.mean

        state_tensor, meta = assembler.assemble(
            instrument_surprises=instrument_surprises,
            entity_alerts=entity_alerts,
            beliefs=beliefs,
            market_features=market_features,
        )

        # Deterministic SAC action → N-dim weight vector
        action = trainer.select_action(state_tensor, deterministic=True)

        # Map action dimensions to tickers
        weights: dict[str, float] = {}
        for i, ticker in enumerate(tickers):
            if i < len(action):
                weights[ticker] = float(action[i])
            else:
                weights[ticker] = 0.0

        log.info(
            "SAC inference: %d weights, sum=%.4f, max=%.4f",
            len(weights),
            sum(weights.values()),
            max(abs(w) for w in weights.values()) if weights else 0.0,
        )

        return {
            "status": "completed",
            "weights": weights,
            "instrument_tickers": tickers,
            # For transition storage — emit_portfolio uses these
            "state_vector": state_tensor.numpy().tolist(),
            "action_vector": [float(a) for a in action],
        }

    except Exception as exc:
        log.warning("SAC inference failed: %s", exc)
        return {"status": "error", "reason": str(exc), "weights": {}}
    finally:
        store.close()


# ── Node 4: emit_portfolio ───────────────────────────────────


def _emit_portfolio(params: dict[str, Any], upstream: dict[str, Any]) -> dict[str, Any]:
    """Persist portfolio weights, compute paper-trade P&L, and run alert checks.

    Steps:
        1. Write today's weights to ``portfolio_weights`` table.
        2. Load yesterday's weights from the table.
        3. Compute today's per-instrument returns from entity observations.
        4. P&L = dot(yesterday_weights, today_returns).
        5. Write to ``paper_trade_pnl`` table.
        6. Run alert checks: concentration, drawdown, Sharpe, edge decay.

    Returns
    -------
    Dict with summary keys: status, date, n_instruments, portfolio_return,
    alerts (list of alert dicts), etc.
    """
    sac_result = upstream.get("sac_inference", {})

    if sac_result.get("status") != "completed":
        reason = sac_result.get("reason", "upstream_not_completed")
        return {"status": "skipped", "reason": reason}

    weights = sac_result.get("weights", {})
    if not weights:
        return {"status": "skipped", "reason": "empty_weights"}

    db_path: str = params.get("db_path", ".tirra_pipeline/pipeline.db")
    today = params.get("as_of_date", date.today().isoformat())
    yesterday = params.get(
        "yesterday_date",
        (date.fromisoformat(today) - timedelta(days=1)).isoformat(),
    )

    store = PipelineStore(db_path)
    try:
        # 1. Write today's weights
        store.store_portfolio_weights(today, weights)
        log.info("Stored %d portfolio weights for %s", len(weights), today)

        # 1b. Store today's pending RL transition (state + action, reward TBD)
        state_vector = sac_result.get("state_vector")
        action_vector = sac_result.get("action_vector")
        transition_stored = False
        if state_vector is not None and action_vector is not None:
            import time as _time

            store.store_pending_transition(
                date=today,
                timestamp=_time.time(),
                state=state_vector,
                action=action_vector,
                metadata={
                    "instrument_tickers": sac_result.get("instrument_tickers", [])
                },
            )
            transition_stored = True
            log.info("Stored pending RL transition for %s", today)

        # 6a. Concentration alert — check BEFORE P&L (uses today's weights)
        alerts: list[dict[str, Any]] = []
        alerts.extend(_check_concentration(weights))

        # 2. Load yesterday's weights
        yesterday_weights = store.query_portfolio_weights(yesterday)

        if not yesterday_weights:
            log.info("No weights for %s — skipping P&L computation.", yesterday)
            return {
                "status": "completed",
                "date": today,
                "n_instruments": len(weights),
                "pnl_computed": False,
                "reason": "no_previous_weights",
                "alerts": alerts,
            }

        # 3. Compute today's returns from instrument observations
        today_returns = _compute_daily_returns(store, today)

        if not today_returns:
            log.info("No return data for %s — skipping P&L.", today)
            return {
                "status": "completed",
                "date": today,
                "n_instruments": len(weights),
                "pnl_computed": False,
                "reason": "no_return_data",
                "alerts": alerts,
            }

        # 4. P&L = dot(yesterday_weights, today_returns)
        portfolio_return = 0.0
        benchmark_return = 0.0  # equal-weight benchmark
        tickers_with_returns = set(today_returns.keys()) & set(yesterday_weights.keys())
        n_bench = len(today_returns) if today_returns else 1

        for ticker in tickers_with_returns:
            portfolio_return += yesterday_weights[ticker] * today_returns[ticker]

        for ticker, ret in today_returns.items():
            benchmark_return += ret / n_bench

        # 4b. Complete yesterday's pending RL transition with realized reward
        transition_completed = store.complete_pending_transition(
            date=yesterday,
            reward=portfolio_return,
            next_state=state_vector if state_vector is not None else [],
            done=False,
        )
        if transition_completed:
            log.info(
                "Completed RL transition for %s with reward=%.6f",
                yesterday,
                portfolio_return,
            )

        # 5. Cumulative return (load previous cumulative or start at 0)
        prev_pnl = store.query_paper_pnl(end_date=yesterday, limit=1)
        prev_cumulative = prev_pnl[-1]["cumulative_return"] if prev_pnl else 0.0
        cumulative_return = prev_cumulative + portfolio_return

        # 6. Write paper P&L
        store.store_paper_pnl(
            date=today,
            portfolio_return=portfolio_return,
            benchmark_return=benchmark_return,
            cumulative_return=cumulative_return,
            metadata={
                "n_tickers_with_returns": len(tickers_with_returns),
                "n_total_weights": len(yesterday_weights),
            },
        )

        log.info(
            "Paper P&L for %s: port=%.6f bench=%.6f cum=%.6f",
            today,
            portfolio_return,
            benchmark_return,
            cumulative_return,
        )

        # 6b. Drawdown and Sharpe alerts — need historical P&L
        alerts.extend(_check_drawdown(store, today))
        alerts.extend(_check_sharpe(store, today))

        # 6c. Edge decay alerts — check held instruments
        alerts.extend(_check_edge_decay(store, weights, today))

        return {
            "status": "completed",
            "date": today,
            "n_instruments": len(weights),
            "pnl_computed": True,
            "portfolio_return": portfolio_return,
            "benchmark_return": benchmark_return,
            "cumulative_return": cumulative_return,
            "transition_stored": transition_stored,
            "transition_completed": transition_completed,
            "alerts": alerts,
        }

    except Exception as exc:
        log.warning("emit_portfolio failed: %s", exc)
        return {"status": "error", "reason": str(exc)}
    finally:
        store.close()


# ── Alert Checks ─────────────────────────────────────────────

# Thresholds from spec 24f.2
_CONCENTRATION_THRESHOLD = 0.30  # single instrument > 30%
_DRAWDOWN_THRESHOLD = 0.05  # drawdown > 5% from peak
_SHARPE_THRESHOLD = -0.5  # cumulative Sharpe < -0.5
_SHARPE_MIN_DAYS = 30  # only check Sharpe after 30 calendar days


def _check_concentration(weights: dict[str, float]) -> list[dict[str, Any]]:
    """Warn if any single instrument weight exceeds 30%.

    Parameters
    ----------
    weights : {ticker → weight} for today.

    Returns list of alert dicts (may be empty).
    """
    alerts: list[dict[str, Any]] = []
    for ticker, w in weights.items():
        if abs(w) > _CONCENTRATION_THRESHOLD:
            msg = (
                f"Concentration alert: {ticker} weight={w:.4f} "
                f"exceeds {_CONCENTRATION_THRESHOLD:.0%} threshold"
            )
            log.warning(msg)
            alerts.append(
                {
                    "level": "WARNING",
                    "type": "concentration",
                    "ticker": ticker,
                    "weight": w,
                    "threshold": _CONCENTRATION_THRESHOLD,
                    "message": msg,
                }
            )
    return alerts


def _check_drawdown(
    store: PipelineStore,
    today: str,
) -> list[dict[str, Any]]:
    """Warn if cumulative return drawdown from peak exceeds 5%.

    Loads all historical P&L to compute peak and current drawdown.
    """
    pnl_history = store.query_paper_pnl(end_date=today, limit=10000)
    if not pnl_history:
        return []

    # Build cumulative wealth curve
    cumulative_returns = [row["cumulative_return"] for row in pnl_history]
    # Convert cumulative log return to wealth
    wealth = [math.exp(cr) for cr in cumulative_returns]
    peak = max(wealth)
    current = wealth[-1]

    if peak <= 0:
        return []

    drawdown = (peak - current) / peak

    if drawdown > _DRAWDOWN_THRESHOLD:
        msg = (
            f"Drawdown alert: {drawdown:.2%} from peak "
            f"(peak={peak:.6f}, current={current:.6f}) "
            f"exceeds {_DRAWDOWN_THRESHOLD:.0%} threshold"
        )
        log.warning(msg)
        return [
            {
                "level": "WARNING",
                "type": "drawdown",
                "drawdown": drawdown,
                "peak_wealth": peak,
                "current_wealth": current,
                "threshold": _DRAWDOWN_THRESHOLD,
                "message": msg,
            }
        ]
    return []


def _check_sharpe(
    store: PipelineStore,
    today: str,
) -> list[dict[str, Any]]:
    """Log CRITICAL if cumulative Sharpe < -0.5 after 30 calendar days.

    Uses daily portfolio returns from paper_trade_pnl to compute
    an annualised Sharpe ratio.
    """
    pnl_history = store.query_paper_pnl(end_date=today, limit=10000)
    if len(pnl_history) < _SHARPE_MIN_DAYS:
        return []

    # Check calendar day span
    first_date = date.fromisoformat(pnl_history[0]["date"])
    last_date = date.fromisoformat(pnl_history[-1]["date"])
    calendar_days = (last_date - first_date).days
    if calendar_days < _SHARPE_MIN_DAYS:
        return []

    daily_returns = np.array(
        [row["portfolio_return"] for row in pnl_history],
        dtype=np.float64,
    )
    mean_ret = float(daily_returns.mean())
    std_ret = float(daily_returns.std(ddof=1)) if len(daily_returns) > 1 else 0.0

    if std_ret < 1e-12:
        return []

    annualised_sharpe = (mean_ret / std_ret) * math.sqrt(252)

    if annualised_sharpe < _SHARPE_THRESHOLD:
        msg = (
            f"Sharpe alert: annualised Sharpe={annualised_sharpe:.3f} "
            f"after {calendar_days} calendar days ({len(daily_returns)} trading days) "
            f"is below {_SHARPE_THRESHOLD} threshold"
        )
        log.critical(msg)
        return [
            {
                "level": "CRITICAL",
                "type": "sharpe",
                "annualised_sharpe": annualised_sharpe,
                "calendar_days": calendar_days,
                "trading_days": len(daily_returns),
                "threshold": _SHARPE_THRESHOLD,
                "message": msg,
            }
        ]
    return []


def _check_edge_decay(
    store: PipelineStore,
    weights: dict[str, float],
    today: str,
) -> list[dict[str, Any]]:
    """Warn if adversarial edge_decay flags exist for held instruments.

    Queries recent entity observations of type ``"adversarial_flag"``
    for held tickers.  Falls back gracefully if no adversarial data is
    available (the adversarial scan table may not yet exist).
    """
    held_tickers = {t for t, w in weights.items() if abs(w) > 1e-6}
    if not held_tickers:
        return []

    try:
        # Look for adversarial flags in entity observations (last 48h)
        recent_obs = store.query_all_observations(
            since=_days_ago(2),
        )
    except Exception:
        log.debug("Could not query observations for edge decay check.")
        return []

    alerts: list[dict[str, Any]] = []
    for obs in recent_obs:
        otype = obs.get("observation_type", "")
        if otype != "adversarial_flag":
            continue
        entity_id = obs.get("entity_id", "")
        if entity_id not in held_tickers:
            continue
        value = obs.get("value", {})
        if not isinstance(value, dict):
            continue
        if value.get("flag_type") != "edge_decay":
            continue

        msg = (
            f"Edge decay alert: {entity_id} has edge_decay flag "
            f"(severity={value.get('severity', '?')}) "
            f"while held with weight={weights.get(entity_id, 0):.4f}"
        )
        log.warning(msg)
        alerts.append(
            {
                "level": "WARNING",
                "type": "edge_decay",
                "ticker": entity_id,
                "weight": weights.get(entity_id, 0.0),
                "severity": value.get("severity"),
                "message": msg,
            }
        )

    return alerts


# ── Helpers ──────────────────────────────────────────────────


def _days_ago(n: int) -> float:
    """Return unix timestamp for n days ago."""
    import time

    return time.time() - n * 86400


def _compute_daily_returns(store: PipelineStore, as_of: str) -> dict[str, float]:
    """Extract daily log returns for instruments on a given date.

    Looks up entity observations of type ``"daily_return"`` stored by
    the instrument ingest tool.  Returns ``{ticker: return_value}``.
    """
    # Parse date to approximate unix timestamp range
    target = date.fromisoformat(as_of)
    start_ts = _date_to_timestamp(target)
    end_ts = start_ts + 86400  # 24h window

    obs = store.query_all_observations(since=start_ts, until=end_ts)

    returns: dict[str, float] = {}
    for o in obs:
        if o.get("observation_type") == "daily_return":
            entity_id = o.get("entity_id", "")
            value = o.get("value", {})
            if isinstance(value, dict):
                ret = value.get("log_return", value.get("return", 0.0))
            else:
                ret = float(value) if value else 0.0
            returns[entity_id] = float(ret)

    return returns


def _date_to_timestamp(d: date) -> float:
    """Convert a date to midnight UTC unix timestamp."""
    from datetime import datetime, timezone

    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp()


# ── DAG builder ──────────────────────────────────────────────


def build_inference_dag(
    db_path: str = ".tirra_pipeline/pipeline.db",
    model_path: str = _DEFAULT_MODEL_PATH,
) -> DAG:
    """Build the inference DAG.

    Four sequential nodes:
        load_models → gnn_inference → sac_inference → emit_portfolio

    Schedule: weekdays at 19:45 UTC (after GNN, scoring, and RL
    training DAGs have completed their runs).
    """
    dag = DAG(
        name=DAG_NAME,
        schedule="45 19 * * 1-5",
        description=(
            "Inference pipeline: load trained models → GNN forward pass → "
            "SAC allocation → emit portfolio weights and paper P&L"
        ),
    )

    shared_params = {"db_path": db_path, "model_path": model_path}

    dag.add(
        "load_models",
        operator=_load_models,
        params=shared_params,
    )

    dag.add(
        "gnn_inference",
        operator=_gnn_inference,
        params={"db_path": db_path},
        depends_on=["load_models"],
    )

    dag.add(
        "sac_inference",
        operator=_sac_inference,
        params={"db_path": db_path},
        depends_on=["gnn_inference"],
    )

    dag.add(
        "emit_portfolio",
        operator=_emit_portfolio,
        params={"db_path": db_path},
        depends_on=["sac_inference"],
    )

    return dag
