"""
TirraMind — World Model Update DAG

Runs after feature_generation completes.  Reads the latest EngineeredFeatures,
builds / loads the WorldModel, runs the update cycle, and persists beliefs.

Schedule: weekdays at 19:30 UTC (30 min after feature_generation).

All functions follow the FunctionOperator contract:
    fn(params: dict, upstream_results: dict) -> dict
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from agent.features.protocol import EngineeredFeature
from agent.models.belief import BeliefState
from agent.models.initial_graph import ALL_NODES, build_initial_graph
from agent.models.propagator import BeliefPropagator
from agent.models.state_filter import ContinuousStateFilter, RegimeConfig
from agent.models.world_model import WorldModel
from agent.pipeline.dag import DAG
from agent.pipeline.store import PipelineStore

log = logging.getLogger(__name__)

# Feature names that the world model expects (from initial_graph NodeSpecs).
_FEATURE_NAMES = [
    spec.feature_name for spec in ALL_NODES if spec.feature_name is not None
]

# Kalman filter configuration (initial expert setup)
_STATE_DIM = 3
_OBS_DIM = 17
_CONTINUOUS_STATE_NAMES = [
    "latent.stress_level",
    "latent.macro_momentum",
    "latent.liquidity_state",
]
_FEATURE_TO_OBS_INDEX = {
    # Original 6 macro/convergence features
    "macro.rate_momentum.30d": 0,
    "macro.yield_curve_slope.spot": 1,
    "macro.liquidity_pressure.30d": 2,
    "convergence.stress_breadth.7d": 3,
    "convergence.stress_intensity.7d": 4,
    "convergence.regime_persistence.7d": 5,
    # GNN entity anomaly features (Phase 19c)
    "gnn.person_anomaly.spot": 6,
    "gnn.company_anomaly.spot": 7,
    "gnn.wallet_anomaly.spot": 8,
    "gnn.country_anomaly.spot": 9,
    "gnn.vessel_anomaly.spot": 10,
    # GNN entity activity features
    "gnn.person_activity.spot": 11,
    "gnn.company_activity.spot": 12,
    "gnn.wallet_activity.spot": 13,
    "gnn.country_activity.spot": 14,
    "gnn.vessel_activity.spot": 15,
    # GNN cross-entity correlation
    "gnn.cross_entity.spot": 16,
}

_REGIME_CONFIGS = {
    "expansion": RegimeConfig(
        name="expansion",
        F=np.diag([0.99, 0.98, 0.97]),
        Q=np.diag([0.01, 0.01, 0.01]),
    ),
    "contraction": RegimeConfig(
        name="contraction",
        F=np.diag([0.97, 0.96, 0.95]),
        Q=np.diag([0.02, 0.02, 0.02]),
    ),
    "crisis": RegimeConfig(
        name="crisis",
        F=np.diag([0.90, 0.88, 0.85]),
        Q=np.diag([0.10, 0.10, 0.10]),
    ),
}


def _build_world_model() -> WorldModel:
    """Construct the default WorldModel from the expert DAG."""
    graph = build_initial_graph()
    propagator = BeliefPropagator(graph)

    H = np.zeros((_OBS_DIM, _STATE_DIM))
    # Original macro/convergence features
    H[0, 0] = 1.0  # rate_momentum → stress_level
    H[1, 0] = 1.0  # yield_curve_slope → stress_level
    H[2, 1] = 1.0  # liquidity_pressure → macro_momentum
    H[3, 1] = 1.0  # stress_breadth → macro_momentum
    H[4, 2] = 1.0  # stress_intensity → liquidity_state
    H[5, 2] = 1.0  # regime_persistence → liquidity_state
    # GNN anomaly features → stress_level (column 0)
    H[6, 0] = 0.5  # person_anomaly
    H[7, 0] = 0.5  # company_anomaly
    H[8, 0] = 0.5  # wallet_anomaly
    H[9, 0] = 0.5  # country_anomaly
    H[10, 0] = 0.5  # vessel_anomaly
    # GNN activity features → macro_momentum (column 1)
    H[11, 1] = 0.3  # person_activity
    H[12, 1] = 0.3  # company_activity
    H[13, 1] = 0.3  # wallet_activity
    H[14, 1] = 0.3  # country_activity
    H[15, 1] = 0.3  # vessel_activity
    # GNN cross_entity → liquidity_state (column 2)
    H[16, 2] = 0.4  # cross_entity
    # Higher noise for GNN features (0.3) vs established features (0.1)
    R = np.diag([0.1] * 6 + [0.3] * 11)

    state_filter = ContinuousStateFilter(
        state_dim=_STATE_DIM,
        obs_dim=_OBS_DIM,
        regime_configs=_REGIME_CONFIGS,
        H=H,
        R=R,
    )

    return WorldModel(
        graph=graph,
        propagator=propagator,
        state_filter=state_filter,
        regime_node="regime.macro",
        continuous_state_names=_CONTINUOUS_STATE_NAMES,
        feature_to_obs_index=_FEATURE_TO_OBS_INDEX,
    )


def run_world_model_update(params: dict, upstream: dict) -> dict:
    """FunctionOperator callback for the world_model_update DAG.

    1. Open PipelineStore.
    2. Load latest features.
    3. Build WorldModel.
    4. Run update cycle.
    5. Persist beliefs.

    Parameters (from ``params``):
        db_path : str
            Path to the pipeline SQLite database.
        as_of : float | None
            Reference time (unix epoch). Defaults to now.
    """
    db_path: str = params.get("db_path", ".tirra_pipeline/pipeline.db")
    as_of: float = params.get("as_of") or time.time()

    store = PipelineStore(db_path)
    try:
        # Load latest features
        features: list[EngineeredFeature] = []
        for feat_name in _FEATURE_NAMES:
            row = store.get_latest_feature(feat_name)
            if row:
                features.append(EngineeredFeature.from_dict(row))

        log.info(
            "World model update: %d/%d features available.",
            len(features),
            len(_FEATURE_NAMES),
        )

        # Build and run world model
        wm = _build_world_model()
        beliefs = wm.update(features, as_of)

        # Persist beliefs
        stale_count = sum(1 for b in beliefs if b.stale)
        stored_count = 0
        if beliefs:
            try:
                row_ids = store.store_beliefs_batch(beliefs)
                stored_count = len(row_ids)
            except ValueError:
                log.warning("Batch belief store failed — falling back to individual.")
                for belief in beliefs:
                    try:
                        store.store_belief(belief)
                        stored_count += 1
                    except ValueError:
                        log.warning(
                            "Skipping invalid belief: %s",
                            belief.variable_name,
                        )

        log.info(
            "World model update complete: %d beliefs (%d stale), %d stored.",
            len(beliefs),
            stale_count,
            stored_count,
        )

        return {
            "beliefs_count": len(beliefs),
            "stale": stale_count,
            "stored": stored_count,
            "graph_hash": wm.get_graph_hash(),
            "as_of": as_of,
            "features_available": len(features),
        }

    finally:
        store.close()


def build_world_model_dag(
    db_path: str = ".tirra_pipeline/pipeline.db",
) -> DAG:
    """Build the world_model_update DAG.

    Single node: ``update_beliefs`` (FunctionOperator).
    Schedule: weekdays at 19:30 UTC, 30 min after feature_generation.
    """
    dag = DAG(
        name="world_model_update",
        schedule="30 19 * * 1-5",
        description=(
            "World model update: propagate feature evidence through "
            "causal DAG and Kalman filter to produce posterior beliefs"
        ),
    )

    dag.add(
        "update_beliefs",
        operator=run_world_model_update,
        params={"db_path": db_path},
        timeout=180,
        retries=1,
        store_result=True,
    )

    return dag
