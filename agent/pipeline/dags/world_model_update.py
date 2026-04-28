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
from collections import defaultdict
from typing import Any

import numpy as np

from pathlib import Path

import pandas as pd

from agent.features.protocol import EngineeredFeature
from agent.learning.meta_scheduler import MetaScheduler, compute_refit_reward
from agent.models.belief import BeliefState
from agent.models.edge_tracker import EdgeConfidenceTracker
from agent.models.initial_graph import ALL_NODES, build_initial_graph
from agent.models.propagator import BeliefPropagator
from agent.models.state_filter import ContinuousStateFilter, RegimeConfig
from agent.models.world_model import WorldModel
from agent.pipeline.dag import DAG
from agent.pipeline.regime_gate import get_current_regime, world_model_prior_decay
from agent.pipeline.store import PipelineStore
from agent.models.gnn.alignment import (
    compute_belief_log_likelihood_delta,
    store_entity_alignment,
)

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


def _load_learned_edges(
    store: PipelineStore,
) -> list[tuple[str, str]] | None:
    """Load the most recently persisted learned edge set, or None."""
    rows = store.query_data(_LEARNED_EDGES_SOURCE, limit=1)
    if not rows:
        return None
    data = rows[0].get("data", {})
    edges_raw = data.get("edges")
    if not edges_raw or not isinstance(edges_raw, list):
        return None
    try:
        return [(str(p), str(c)) for p, c in edges_raw]
    except (TypeError, ValueError):
        log.warning("Corrupt learned_graph_edges data, falling back to expert.")
        return None


def _persist_learned_edges(
    store: PipelineStore,
    graph_edges: list[tuple[str, str]],
    as_of: float,
    refine_result: dict[str, Any],
) -> None:
    """Persist the full current edge set after structure refinement."""
    store.store_data(
        _LEARNED_EDGES_SOURCE,
        {"as_of": as_of, "edge_count": len(graph_edges)},
        {
            "edges": [[p, c] for p, c in sorted(graph_edges)],
            "edges_added": refine_result.get("edges_added", []),
            "edges_removed": refine_result.get("edges_removed", []),
        },
    )


def _build_world_model(
    learned_edges: list[tuple[str, str]] | None = None,
    *,
    use_differentiable_filter: bool = False,
) -> WorldModel:
    """Construct the WorldModel, optionally applying a learned edge set.

    If *learned_edges* is provided, the expert graph is built first (to get
    all nodes and CPDs), then edges are diffed and updated to match the
    learned set.  CPDs for nodes whose parents changed are stripped (they
    will be re-fit by ``_maybe_fit_params`` on the next cycle).

    If *use_differentiable_filter* is True, the numpy ContinuousStateFilter is
    built with expert matrices first, then converted to a torch
    DifferentiableKalmanFilter via ``from_numpy_filter()``.  This enables
    autograd through predict/update while preserving the same initial params.
    """
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

    # ── Apply learned edges if available ──────────────────────
    if learned_edges is not None:
        expert_edges = set(graph.edges)
        target_edges = set(learned_edges)
        # Only diff edges where both endpoints exist in the graph
        valid_nodes = set(graph.node_names)
        target_edges = {
            (p, c) for p, c in target_edges if p in valid_nodes and c in valid_nodes
        }
        to_remove = expert_edges - target_edges
        to_add = target_edges - expert_edges
        for parent, child in to_remove:
            try:
                graph.remove_edge(parent, child)
            except ValueError:
                pass
        for parent, child in to_add:
            try:
                graph.add_edge(parent, child)
            except ValueError as exc:
                log.warning(
                    "Skipping learned edge %s→%s: %s",
                    parent,
                    child,
                    exc,
                )
        if to_remove or to_add:
            # Restore uniform CPDs for any node that lost its CPD due to
            # parent changes.  These will be re-fit by _maybe_fit_params.
            from pgmpy.factors.discrete import TabularCPD

            for spec in ALL_NODES:
                if graph.get_cpd(spec.name) is None:
                    parents = graph.get_parents(spec.name)
                    parent_cards = [graph.get_node(p).cardinality for p in parents]
                    ncols = max(1, int(np.prod(parent_cards))) if parents else 1
                    uniform = [[1.0 / spec.cardinality] * ncols] * spec.cardinality
                    evidence = parents if parents else None
                    evidence_card = parent_cards if parents else None
                    cpd = TabularCPD(
                        variable=spec.name,
                        variable_card=spec.cardinality,
                        values=uniform,
                        evidence=evidence,
                        evidence_card=evidence_card,
                        state_names={spec.name: list(spec.states)},
                    )
                    graph.set_cpd(spec.name, cpd)

            log.info(
                "Applied learned structure: +%d -%d edges (now %d).",
                len(to_add),
                len(to_remove),
                len(graph.edges),
            )

    # ── Optionally upgrade to differentiable (torch) filter ───
    active_filter: ContinuousStateFilter | Any = state_filter
    if use_differentiable_filter:
        from agent.models.diff_kalman import DifferentiableKalmanFilter

        active_filter = DifferentiableKalmanFilter.from_numpy_filter(state_filter)
        log.info(
            "Upgraded to DifferentiableKalmanFilter "
            "(state_dim=%d, obs_dim=%d, regimes=%s).",
            active_filter.state_dim,
            active_filter.obs_dim,
            active_filter.regime_names,
        )

    return WorldModel(
        graph=graph,
        propagator=propagator,
        state_filter=active_filter,
        regime_node="regime.macro",
        continuous_state_names=_CONTINUOUS_STATE_NAMES,
        feature_to_obs_index=_FEATURE_TO_OBS_INDEX,
    )


# ── Periodic parameter fitting helpers ─────────────────────────

_FIT_SOURCE = "world_model_fit"
_STRUCTURE_FIT_SOURCE = "world_model_structure_fit"
_LEARNED_EDGES_SOURCE = "learned_graph_edges"
_EDGE_TRACKER_STATE_SOURCE = "edge_confidence_tracker_state"
_META_SCHEDULER_SOURCE = "meta_scheduler_state"
_DAY_SECONDS = 86_400


def _apply_prior_decay(wm: WorldModel, decay: float) -> None:
    """Soften world model priors in-place to reflect regime uncertainty.

    Called once after building a fresh WorldModel and before wm.update()
    when the regime gate signals a structural change (decay < 1.0).
    Has no effect when decay == 1.0 (stable regime).

    Two operations:

    1. **Kalman covariance inflation** — sets ``P_0 ← (1/decay) · P_0``.
       The filter starts fresh each run with ``P = I``.  Inflating it
       widens the prior uncertainty, which increases the Kalman gain K
       during the first update step.  The filter then trusts incoming
       observations more relative to the (stale) prior state, which is
       exactly the right behaviour after a regime shift.

       Derivation: K = P·H^T·(H·P·H^T + R)^{-1}.  Scaling P by α > 1
       increases K monotonically (since H·P·H^T dominates for large α),
       so the updated state x_{t|t} = x_{t|t-1} + K·innovation places
       more weight on the innovation than on the prediction.

       Reference: Sarkka (2013), Ch. 4 — process noise inflation as an
       adaptive filter heuristic for non-stationary environments.

    2. **CPD softening** — blends each TabularCPD toward the uniform
       distribution: ``cpd_new = decay · cpd_old + (1-decay) · uniform``.
       This is equivalent to reducing the effective Dirichlet concentration
       parameter, expressing "our learned conditional distribution was
       fitted on data from the previous regime; we are now less certain".
       The softening preserves the MAP state ordering (the most probable
       states remain most probable) while shrinking probability differences.

       Mathematical note: for a CPD column p (a probability simplex vector
       of length k), the softened version is:
           p_new = decay · p + (1-decay) · [1/k, ..., 1/k]
       which is a convex combination.  p_new still sums to 1 (valid CPD)
       and is strictly positive (avoids zero-probability states).

    Args:
        wm: Freshly-built WorldModel (not yet updated with today's features).
        decay: Factor in (0, 1].  1.0 = no change (stable regime).
               0.8 = 20% blend toward uniform + 25% covariance inflation.
    """
    if decay >= 1.0:
        return  # stable regime — nothing to do

    inflation = 1.0 / decay  # e.g. 0.8 → 1.25

    # ── 1. Kalman covariance inflation ─────────────────────────────
    try:
        sf = wm._filter
        sf._P = sf._P * inflation
        # Symmetrize (numerical safety after in-place scaling)
        sf._P = 0.5 * (sf._P + sf._P.T)
        log.debug(
            "Prior decay: Kalman P inflated by %.3f (decay=%.2f).",
            inflation,
            decay,
        )
    except AttributeError:
        # DifferentiableKalmanFilter has a torch covariance — skip for now
        log.debug(
            "Prior decay: skipping Kalman inflation for DifferentiableKalmanFilter "
            "(torch covariance not directly mutable)."
        )

    # ── 2. CPD softening ────────────────────────────────────────
    # Blend each discrete node's CPD toward uniform.
    # We only soften non-regime, non-latent nodes — regime and latent nodes
    # are inferred (not directly observed) and their CPDs are structural;
    # softening them would distort the causal graph topology itself.
    uniform_weight = 1.0 - decay  # e.g. 0.2 when decay=0.8
    softened_count = 0

    from pgmpy.factors.discrete import TabularCPD

    for spec in ALL_NODES:
        if spec.node_type not in ("observed",):
            continue  # leave regime and latent CPDs unchanged
        cpd = wm._graph.get_cpd(spec.name)
        if cpd is None:
            continue
        if spec.cardinality is None or spec.cardinality < 2:
            continue

        k = spec.cardinality
        values = cpd.get_values()  # shape (k, n_parent_configs)
        # Blend each column toward uniform 1/k
        uniform_col = np.full(k, 1.0 / k)
        new_values = decay * values + uniform_weight * uniform_col[:, np.newaxis]
        # Renormalise columns (floating-point safety)
        col_sums = new_values.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums == 0, 1.0, col_sums)  # avoid divide by zero
        new_values = new_values / col_sums

        parents = wm._graph.get_parents(spec.name)
        parent_cards = [wm._graph.get_node(p).cardinality for p in parents]
        new_cpd = TabularCPD(
            variable=spec.name,
            variable_card=k,
            values=new_values,
            evidence=parents if parents else None,
            evidence_card=parent_cards if parent_cards else None,
            state_names={spec.name: list(spec.states)} if spec.states else None,
        )
        wm._graph.set_cpd(spec.name, new_cpd)
        softened_count += 1

    log.debug(
        "Prior decay: softened %d observed-node CPDs "
        "(blend=%.0f%% uniform, decay=%.2f).",
        softened_count,
        uniform_weight * 100,
        decay,
    )


def _snapshots_to_discretized_df(
    wm: WorldModel,
    snapshots: list[list[EngineeredFeature]],
) -> pd.DataFrame:
    """Convert feature snapshots into a discretized DataFrame.

    Reuses the WorldModel's graph specs and discretization logic to produce
    a DataFrame suitable for BIC scoring (EdgeConfidenceTracker).
    """
    specs = wm._graph.node_specs
    rows: list[dict[str, str]] = []
    for snapshot in snapshots:
        feat_by_name = {f.feature_name: f for f in snapshot}
        row: dict[str, str] = {}
        for node_name, spec in specs.items():
            if spec.feature_name is None or spec.bin_edges is None:
                continue
            feat = feat_by_name.get(spec.feature_name)
            if feat is None or feat.value is None:
                continue
            state = wm._discretize(feat.value, spec.bin_edges, spec.states)
            if state is not None:
                row[node_name] = state
        if len(row) >= 2:
            rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _build_windowed_dataframes(
    wm: WorldModel,
    snapshots: list[list[EngineeredFeature]],
    as_of: float,
    windows_days: tuple[int, ...],
) -> list[pd.DataFrame]:
    """Build one discretized DataFrame per window period for edge tracking.

    Slices the snapshots (oldest-first) to the most recent N days for each
    window in *windows_days*, then discretizes.
    """
    dfs: list[pd.DataFrame] = []
    for window in windows_days:
        cutoff = as_of - window * _DAY_SECONDS
        windowed = [
            snap for snap in snapshots if snap and snap[0].effective_at >= cutoff
        ]
        df = _snapshots_to_discretized_df(wm, windowed)
        dfs.append(df)
    return dfs


def _get_protected_edges(wm: WorldModel) -> set[tuple[str, str]]:
    """Return edges from regime/latent nodes that should never be removed."""
    protected: set[tuple[str, str]] = set()
    regime_names = {s.name for s in wm._graph.get_regime_nodes()}
    latent_names = {s.name for s in wm._graph.get_latent_nodes()}
    anchor_names = regime_names | latent_names
    for parent, child in wm._graph.edges:
        if parent in anchor_names:
            protected.add((parent, child))
    return protected


def _load_edge_tracker(
    store: PipelineStore,
    wm: WorldModel,
) -> EdgeConfidenceTracker:
    """Load persisted EdgeConfidenceTracker state, or create a fresh one."""
    rows = store.query_data(_EDGE_TRACKER_STATE_SOURCE, limit=1)
    if rows:
        try:
            data = rows[0].get("data", {})
            if isinstance(data, dict) and "node_names" in data:
                return EdgeConfidenceTracker.from_dict(data)
        except Exception as exc:
            log.warning("Failed to load edge tracker state: %s", exc)
    return EdgeConfidenceTracker(node_names=wm._graph.node_names)


def _save_edge_tracker(
    store: PipelineStore,
    tracker: EdgeConfidenceTracker,
    as_of: float,
) -> None:
    """Persist EdgeConfidenceTracker state for next run."""
    try:
        store.store_data(
            _EDGE_TRACKER_STATE_SOURCE,
            {"as_of": as_of},
            tracker.to_dict(),
        )
    except Exception as exc:
        log.warning("Failed to save edge tracker state: %s", exc)


def _should_fit(
    store: PipelineStore,
    as_of: float,
    fit_interval_days: int,
    *,
    source: str = _FIT_SOURCE,
) -> tuple[bool, str]:
    """Check whether enough logical time has passed since the last parameter fit.

    Uses the ``as_of`` timestamp stored in the fit marker (not wall-clock
    ``fetched_at``) so that back-testing with historical ``as_of`` values
    also respects the interval correctly.
    """
    markers = store.query_data(source, limit=1)
    if not markers:
        return True, "no previous fit found"

    marker_params = markers[0].get("params", {})
    last_fit_as_of = marker_params.get("as_of", markers[0].get("fetched_at", 0))
    days_since = (as_of - last_fit_as_of) / _DAY_SECONDS
    if days_since >= fit_interval_days:
        return True, f"{days_since:.1f}d since last fit"
    return False, f"only {days_since:.1f}d since last fit (need {fit_interval_days}d)"


def _load_feature_history(
    store: PipelineStore,
    since: float,
    until: float,
) -> list[list[EngineeredFeature]]:
    """Load historical features from the store and group into daily snapshots.

    Each snapshot is a list of EngineeredFeature values from one calendar day
    (UTC), suitable for passing to ``WorldModel.fit_cpds()``.

    Returns snapshots sorted oldest-first.
    """
    day_buckets: dict[int, list[EngineeredFeature]] = defaultdict(list)
    max_rows = int((until - since) / _DAY_SECONDS) + 20

    for feat_name in _FEATURE_NAMES:
        rows = store.query_features(
            feat_name,
            since=since,
            until=until,
            limit=max_rows,
        )
        for row in rows:
            try:
                feat = EngineeredFeature.from_dict(row)
            except (KeyError, TypeError):
                continue
            day_key = int(feat.effective_at // _DAY_SECONDS)
            day_buckets[day_key].append(feat)

    return [day_buckets[k] for k in sorted(day_buckets.keys()) if day_buckets[k]]


def _load_regime_labels(
    store: PipelineStore,
    day_keys: list[int],
    default_regime: str = "expansion",
) -> list[str]:
    """Load MAP regime labels from stored ``regime.macro`` beliefs.

    Returns one regime label per entry in *day_keys*.  Days without a
    stored belief get *default_regime* (safe fallback — Kalman EM will
    lump those observations under the default regime's F/Q).
    """
    if not day_keys:
        return []

    since = min(day_keys) * _DAY_SECONDS
    until = (max(day_keys) + 1) * _DAY_SECONDS
    max_rows = len(day_keys) + 20

    belief_rows = store.query_beliefs(
        "regime.macro",
        since=since,
        until=until,
        limit=max_rows,
    )

    # Index by day — query returns DESC order so first hit per day is latest
    day_to_regime: dict[int, str] = {}
    for row in belief_rows:
        ea = row.get("effective_at", 0)
        probs = row.get("probabilities")
        if probs and isinstance(probs, dict):
            day_key = int(ea // _DAY_SECONDS)
            if day_key not in day_to_regime:
                day_to_regime[day_key] = max(probs, key=probs.get)

    return [day_to_regime.get(dk, default_regime) for dk in day_keys]


def _build_observation_sequence(
    snapshots: list[list[EngineeredFeature]],
) -> list[np.ndarray]:
    """Convert daily feature snapshots into Kalman observation vectors.

    Each snapshot becomes a ``(_OBS_DIM,)`` array.  Features not present
    in ``_FEATURE_TO_OBS_INDEX`` or with ``value is None`` are left as
    ``NaN`` (the Kalman filter treats NaN rows as missing observations).
    """
    obs_seq: list[np.ndarray] = []
    for snapshot in snapshots:
        obs = np.full(_OBS_DIM, np.nan)
        for feat in snapshot:
            idx = _FEATURE_TO_OBS_INDEX.get(feat.feature_name)
            if idx is not None and feat.value is not None:
                obs[idx] = feat.value
        obs_seq.append(obs)
    return obs_seq


def _maybe_fit_params(
    store: PipelineStore,
    wm: WorldModel,
    as_of: float,
    *,
    fit_enabled: bool = True,
    fit_interval_days: int = 7,
    history_window_days: int = 90,
) -> dict[str, Any]:
    """Periodically fit CPD and Kalman filter parameters from historical data.

    Orchestrates the full fitting flow:

    1. Check periodicity marker — skip if last fit was too recent.
    2. Load feature history from the store (last *history_window_days* days).
    3. Group into daily snapshots.
    4. Fit DAG CPDs via Bayesian estimation with BDeu priors (Change 2a).
    5. Load regime labels from stored beliefs.
    6. Build observation sequences for the Kalman filter.
    7. Fit Kalman F/Q/H/R via Shumway-Stoffer EM (Change 2b).
    8. Store a fit marker so the next run respects the interval.

    On any failure, logs a warning and falls back to current parameters.
    Fitting is a best-effort improvement — the world model always has
    valid (expert or previously-fitted) params to use regardless of
    whether this step succeeds.
    """
    if not fit_enabled:
        return {"skipped": True, "reason": "fit_enabled=False"}

    should_fit, reason = _should_fit(store, as_of, fit_interval_days)
    if not should_fit:
        log.debug("Skipping parameter fit: %s", reason)
        return {"skipped": True, "reason": reason}

    log.info("Starting periodic parameter fit: %s", reason)

    since = as_of - history_window_days * _DAY_SECONDS

    # ── Load and group feature history ────────────────────────
    snapshots = _load_feature_history(store, since=since, until=as_of)
    if len(snapshots) < 10:
        log.info("Only %d daily snapshots — skipping fit.", len(snapshots))
        return {"skipped": True, "reason": f"only {len(snapshots)} snapshots"}

    # ── Step 1: Fit CPDs (Change 2a.2) ────────────────────────
    cpd_result: dict[str, Any] = {"fitted": False}
    try:
        cpd_result = wm.fit_cpds(snapshots)
    except Exception as exc:
        log.warning("CPD fitting failed: %s", exc)
        cpd_result = {"fitted": False, "error": str(exc)}

    # ── Step 2: Fit Kalman parameters (Change 2b.2) ───────────
    # Observation sequences from the same feature history
    obs_seq = _build_observation_sequence(snapshots)

    # Regime labels from stored beliefs (one per daily snapshot)
    day_keys = [int(s[0].effective_at // _DAY_SECONDS) for s in snapshots if s]
    default_regime = (
        list(wm._filter._regime_configs.keys())[0]
        if wm._filter._regime_configs
        else "expansion"
    )
    regime_labels = _load_regime_labels(
        store,
        day_keys,
        default_regime=default_regime,
    )

    kalman_result: dict[str, Any] = {"fitted": False}
    if len(obs_seq) >= 30 and len(regime_labels) == len(obs_seq):
        try:
            # Check if we're using the differentiable filter backend
            from agent.models.diff_kalman import DifferentiableKalmanFilter

            if isinstance(wm._filter, DifferentiableKalmanFilter):
                # EM runs on a temporary numpy filter, then params transfer
                numpy_params = wm._filter.to_numpy_params()
                tmp_filter = ContinuousStateFilter(
                    state_dim=_STATE_DIM,
                    obs_dim=_OBS_DIM,
                    regime_configs=_REGIME_CONFIGS,
                    H=numpy_params["H"],
                    R=numpy_params["R"],
                )
                # Seed with current learned F/Q per regime
                for rname, rp in numpy_params["regimes"].items():
                    if rname in tmp_filter._regime_configs:
                        tmp_filter._regime_configs[rname] = RegimeConfig(
                            name=rname,
                            F=rp["F"],
                            Q=rp["Q"],
                        )
                kalman_result = tmp_filter.fit_filter_params(
                    obs_seq,
                    regime_labels,
                )
                # Transfer EM-fitted params back to diff filter
                if kalman_result.get("fitted"):
                    import torch

                    with torch.no_grad():
                        new_diff = DifferentiableKalmanFilter.from_numpy_filter(
                            tmp_filter,
                        )
                        wm._filter.load_state_dict(new_diff.state_dict())
                    kalman_result["backend"] = "differentiable+EM"
            else:
                kalman_result = wm._filter.fit_filter_params(
                    obs_seq,
                    regime_labels,
                )
        except Exception as exc:
            log.warning("Kalman EM fitting failed: %s", exc)
            kalman_result = {"fitted": False, "error": str(exc)}
    else:
        kalman_result = {
            "fitted": False,
            "reason": (
                f"insufficient data: {len(obs_seq)} obs, "
                f"{len(regime_labels)} labels (need >=30)"
            ),
        }

    # ── Store fit marker for periodicity control ──────────────
    fit_summary = {
        "cpd_fitted": cpd_result.get("fitted", False),
        "kalman_fitted": kalman_result.get("fitted", False),
        "n_snapshots": len(snapshots),
    }
    try:
        store.store_data(
            _FIT_SOURCE,
            {"as_of": as_of, **fit_summary},
            {"cpd_result": cpd_result, "kalman_result": kalman_result},
        )
    except Exception as exc:
        log.warning("Failed to store fit marker: %s", exc)

    log.info(
        "Parameter fit complete: CPD=%s, Kalman=%s (%d snapshots).",
        cpd_result.get("fitted"),
        kalman_result.get("fitted"),
        len(snapshots),
    )

    return {
        "skipped": False,
        "cpd_result": cpd_result,
        "kalman_result": kalman_result,
        "n_snapshots": len(snapshots),
    }


def _maybe_refine_structure(
    store: PipelineStore,
    wm: WorldModel,
    as_of: float,
    *,
    structure_fit_enabled: bool = True,
    structure_fit_interval_days: int = 90,
    history_window_days: int = 90,
) -> dict[str, Any]:
    """Periodically refine DAG structure via constrained hill-climb (Change 3).

    Uses BIC-discrete scoring with the expert graph as warm start.
    After hill-climb, the EdgeConfidenceTracker (Change 13) validates
    proposed changes via BIC-δ hysteresis to prevent flip-flopping.
    Much less frequent than CPD fitting — quarterly by default, since
    causal structure should be stable.
    """
    if not structure_fit_enabled:
        return {"skipped": True, "reason": "structure_fit_enabled=False"}

    should_fit, reason = _should_fit(
        store, as_of, structure_fit_interval_days, source=_STRUCTURE_FIT_SOURCE
    )
    if not should_fit:
        log.debug("Skipping structure refinement: %s", reason)
        return {"skipped": True, "reason": reason}

    log.info("Starting periodic structure refinement: %s", reason)

    since = as_of - history_window_days * _DAY_SECONDS
    snapshots = _load_feature_history(store, since=since, until=as_of)
    if len(snapshots) < 50:
        log.info(
            "Only %d daily snapshots — skipping structure refinement.", len(snapshots)
        )
        return {"skipped": True, "reason": f"only {len(snapshots)} snapshots"}

    result: dict[str, Any] = {"refined": False}
    try:
        result = wm.refine_structure(snapshots)
    except Exception as exc:
        log.warning("Structure refinement failed: %s", exc)
        result = {"refined": False, "error": str(exc)}

    # ── Change 13: Edge confidence tracking + hysteresis ──────
    tracker_result: dict[str, Any] = {"tracked": False}
    try:
        tracker = _load_edge_tracker(store, wm)
        windowed_dfs = _build_windowed_dataframes(
            wm, snapshots, as_of, tracker.windows_days
        )

        current_edges = wm._graph.edges
        edge_confidences = tracker.evaluate(current_edges, windowed_dfs)

        if edge_confidences:
            protected = _get_protected_edges(wm)
            suggestion = tracker.suggest_changes(
                edge_confidences,
                set(current_edges),
                protected_edges=protected,
            )

            tracker_edges_added: list[tuple[str, str]] = []
            tracker_edges_removed: list[tuple[str, str]] = []

            for edge in suggestion.edges_to_remove:
                try:
                    wm._graph.remove_edge(*edge)
                    tracker.reset_consecutive(edge)
                    tracker_edges_removed.append(edge)
                    log.info("EdgeTracker removed: %s → %s", edge[0], edge[1])
                except Exception as exc:
                    log.debug("EdgeTracker remove failed %s: %s", edge, exc)

            for edge in suggestion.edges_to_add:
                try:
                    wm._graph.add_edge(*edge)
                    tracker.reset_consecutive(edge)
                    tracker_edges_added.append(edge)
                    log.info("EdgeTracker added: %s → %s", edge[0], edge[1])
                except Exception as exc:
                    log.debug("EdgeTracker add failed %s: %s", edge, exc)

            tracker_result = {
                "tracked": True,
                "n_edges_evaluated": len(edge_confidences),
                "edges_added": [[p, c] for p, c in tracker_edges_added],
                "edges_removed": [[p, c] for p, c in tracker_edges_removed],
            }

            # Mark beliefs stale if tracker changed structure (Change 13.6)
            if tracker_edges_added or tracker_edges_removed:
                result["refined"] = True
                result.setdefault("edges_added", []).extend(
                    [[p, c] for p, c in tracker_edges_added]
                )
                result.setdefault("edges_removed", []).extend(
                    [[p, c] for p, c in tracker_edges_removed]
                )
                try:
                    store.mark_beliefs_stale(
                        reason="structure_change",
                        dag_version=wm.dag_version,
                    )
                except Exception as exc:
                    log.warning("Failed to mark beliefs stale: %s", exc)

            # Store edge confidence scores
            try:
                conf_dict = {
                    f"{e[0]}|{e[1]}": {
                        "confidence": c.confidence,
                        "stability": c.stability,
                    }
                    for e, c in edge_confidences.items()
                }
                store.store_edge_confidences(as_of, wm.dag_version, conf_dict)
            except Exception as exc:
                log.warning("Failed to store edge confidences: %s", exc)

        # Persist tracker state for next run
        _save_edge_tracker(store, tracker, as_of)

    except Exception as exc:
        log.warning("Edge confidence tracking failed: %s", exc)
        tracker_result = {"tracked": False, "error": str(exc)}

    result["tracker"] = tracker_result

    # Store marker for periodicity control
    try:
        store.store_data(
            _STRUCTURE_FIT_SOURCE,
            {"as_of": as_of, **{k: v for k, v in result.items() if k != "tracker"}},
            {"detail": result},
        )
    except Exception as exc:
        log.warning("Failed to store structure fit marker: %s", exc)

    # Persist the full learned edge set if structure was actually refined
    if result.get("refined"):
        try:
            _persist_learned_edges(store, wm._graph.edges, as_of, result)
        except Exception as exc:
            log.warning("Failed to persist learned edges: %s", exc)

    log.info("Structure refinement complete: %s", result.get("refined"))
    return result


def run_world_model_update(params: dict, upstream: dict) -> dict:
    """FunctionOperator callback for the world_model_update DAG.

    1. Open PipelineStore.
    2. Load or create MetaScheduler (Change 14) for dynamic intervals.
    3. Load latest features.
    4. Build WorldModel.
    5. Periodic structure refinement (with edge confidence tracking).
    6. Periodic parameter fitting (CPD + Kalman EM) if interval elapsed.
    7. Run update cycle.
    8. Persist beliefs + scheduler state.

    Parameters (from ``params``):
        db_path : str
            Path to the pipeline SQLite database.
        as_of : float | None
            Reference time (unix epoch). Defaults to now.
        fit_enabled : bool
            Whether periodic CPD/Kalman fitting is active (default True).
        fit_interval_days : int | None
            Explicit override for CPD fit interval. If None, MetaScheduler
            suggests the interval (Change 14).
        history_window_days : int | None
            Explicit override for history window. If None, MetaScheduler
            suggests the window size (Change 14).
        structure_fit_enabled : bool
            Whether periodic DAG structure learning is active (default True).
        structure_fit_interval_days : int | None
            Explicit override for structure refinement interval. If None,
            MetaScheduler suggests the interval (Change 14).
        use_scheduler : bool
            Whether to use MetaScheduler for interval selection (default True).
    """
    db_path: str = params.get("db_path", ".tirra_pipeline/pipeline.db")
    as_of: float = params.get("as_of") or time.time()

    fit_enabled: bool = params.get("fit_enabled", True)
    structure_fit_enabled: bool = params.get("structure_fit_enabled", True)
    use_scheduler: bool = params.get("use_scheduler", True)

    # Explicit overrides (for backtesting) — None means use scheduler
    explicit_fit_interval: int | None = params.get("fit_interval_days")
    explicit_window: int | None = params.get("history_window_days")
    explicit_structure_interval: int | None = params.get("structure_fit_interval_days")

    store = PipelineStore(db_path)
    try:
        # ── Load or create MetaScheduler (Change 14) ─────────
        scheduler: MetaScheduler | None = None
        scheduler_path = Path(db_path).parent / "meta_scheduler.json"
        if use_scheduler:
            try:
                scheduler = MetaScheduler(persist_path=scheduler_path)
            except Exception as exc:
                log.warning("Failed to load MetaScheduler: %s — using defaults", exc)
                scheduler = None

        # Resolve intervals: explicit override > scheduler > hardcoded default
        if explicit_fit_interval is not None:
            fit_interval_days = explicit_fit_interval
            cpd_arm = explicit_fit_interval
        elif scheduler is not None:
            cpd_arm = scheduler.suggest("cpd_fit")
            fit_interval_days = cpd_arm
        else:
            fit_interval_days = 7
            cpd_arm = 7

        if explicit_window is not None:
            history_window_days = explicit_window
            window_arm = explicit_window
        elif scheduler is not None:
            window_arm = scheduler.suggest("history_window")
            history_window_days = window_arm
        else:
            history_window_days = 90
            window_arm = 90

        if explicit_structure_interval is not None:
            structure_fit_interval_days = explicit_structure_interval
            structure_arm = explicit_structure_interval
        elif scheduler is not None:
            structure_arm = scheduler.suggest("structure_refine")
            structure_fit_interval_days = structure_arm
        else:
            structure_fit_interval_days = 90
            structure_arm = 90

        log.info(
            "Intervals: cpd=%dd, structure=%dd, window=%dd (scheduler=%s)",
            fit_interval_days,
            structure_fit_interval_days,
            history_window_days,
            scheduler is not None,
        )

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

        # Load learned graph structure (if any previous refinement persisted one)
        learned_edges = _load_learned_edges(store)

        # Build and run world model
        wm = _build_world_model(learned_edges=learned_edges)

        # Periodic structure refinement (before param fitting — if structure
        # changes, subsequent CPD fitting will re-fit for new parent sets)
        structure_result = _maybe_refine_structure(
            store,
            wm,
            as_of,
            structure_fit_enabled=structure_fit_enabled,
            structure_fit_interval_days=structure_fit_interval_days,
            history_window_days=history_window_days,
        )

        # Record structure refinement outcome for scheduler
        if scheduler is not None and not structure_result.get("skipped"):
            try:
                n_changes = len(structure_result.get("edges_added", [])) + len(
                    structure_result.get("edges_removed", [])
                )
                reward = compute_refit_reward(
                    "structure_refine",
                    {},
                    {"n_confident_changes": float(n_changes)},
                )
                scheduler.record_outcome("structure_refine", structure_arm, reward)
                store.store_component_performance(
                    "structure_refine",
                    as_of,
                    structure_arm,
                    reward,
                    {"n_changes": n_changes},
                )
            except Exception as exc:
                log.debug("Failed to record structure_refine outcome: %s", exc)

        # Periodic parameter fitting (before normal update so fitted params
        # are used immediately in this run's belief propagation + Kalman step)
        fit_result = _maybe_fit_params(
            store,
            wm,
            as_of,
            fit_enabled=fit_enabled,
            fit_interval_days=fit_interval_days,
            history_window_days=history_window_days,
        )

        # Record CPD fit outcome for scheduler
        if scheduler is not None and not fit_result.get("skipped"):
            try:
                cpd_result = fit_result.get("cpd_result", {})
                reward = compute_refit_reward(
                    "cpd_fit",
                    {},
                    {"total_bic": 1.0 if cpd_result.get("fitted") else 0.0},
                )
                scheduler.record_outcome("cpd_fit", cpd_arm, reward)
                store.store_component_performance(
                    "cpd_fit",
                    as_of,
                    cpd_arm,
                    reward,
                    cpd_result,
                )
            except Exception as exc:
                log.debug("Failed to record cpd_fit outcome: %s", exc)

        # Record history window outcome
        if scheduler is not None and not fit_result.get("skipped"):
            try:
                reward = compute_refit_reward(
                    "history_window",
                    {},
                    {
                        "held_out_bic": (
                            1.0
                            if fit_result.get("cpd_result", {}).get("fitted")
                            else 0.0
                        )
                    },
                )
                scheduler.record_outcome("history_window", window_arm, reward)
            except Exception as exc:
                log.debug("Failed to record history_window outcome: %s", exc)

        # ── Phase 49b: Apply prior decay on regime change ─────────────
        # Query the current regime state.  When the regime label has just
        # changed (regime_changed=True), soften the world model priors
        # before running the update cycle:
        #   - Kalman P_0 inflated by 1/decay (widens prior uncertainty)
        #   - Observed-node CPDs blended toward uniform by (1-decay)
        # Both operations act on the freshly-built in-memory WorldModel
        # and do NOT mutate any persisted data.
        # See _apply_prior_decay() for the full mathematical justification.
        try:
            regime_ctx = get_current_regime(store)
            decay = world_model_prior_decay(regime_ctx)
            if decay < 1.0:
                log.info(
                    "Phase 49b: regime change detected — applying prior decay "
                    "(decay=%.2f, regime=%s, changepoint_posterior=%.3f).",
                    decay,
                    regime_ctx.regime_label,
                    regime_ctx.changepoint_posterior,
                )
                _apply_prior_decay(wm, decay)
            else:
                log.debug(
                    "Phase 49b: regime stable (decay=1.0, regime=%s) — no prior decay.",
                    regime_ctx.regime_label,
                )
        except Exception as exc:
            log.warning(
                "Phase 49b: prior decay check failed — continuing without decay: %s",
                exc,
            )

        # ── Phase 49: Capture beliefs before update for alignment delta ─
        try:
            beliefs_before = store.query_all_latest_beliefs()
        except Exception as exc:
            log.debug("Phase 49: could not load prior beliefs for alignment: %s", exc)
            beliefs_before = []

        beliefs = wm.update(features, as_of)

        # ── Phase 49: Compute and store GNN alignment delta ────────────
        # Compare beliefs before and after the update to measure how much
        # world-model beliefs sharpened.  Results feed back into GNN training
        # as per-entity-type loss weights (see alignment.py for math).
        try:
            beliefs_after_dicts = [b.to_dict() for b in beliefs]
            variable_deltas = compute_belief_log_likelihood_delta(
                beliefs_before, beliefs_after_dicts
            )
            if variable_deltas:
                store_entity_alignment(store, variable_deltas, as_of=as_of)
                log.debug(
                    "Phase 49: stored %d alignment deltas " "(mean_delta=%.4f).",
                    len(variable_deltas),
                    sum(variable_deltas.values()) / len(variable_deltas),
                )
        except Exception as exc:
            log.warning(
                "Phase 49: alignment delta computation failed — "
                "continuing without alignment signal: %s",
                exc,
            )

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

        # Save scheduler state
        if scheduler is not None:
            try:
                scheduler.save()
            except Exception as exc:
                log.warning("Failed to save MetaScheduler: %s", exc)

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
            "fit_result": fit_result,
            "structure_result": structure_result,
            "scheduler": {
                "cpd_fit_interval": fit_interval_days,
                "structure_interval": structure_fit_interval_days,
                "history_window": history_window_days,
                "using_scheduler": scheduler is not None,
            },
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
