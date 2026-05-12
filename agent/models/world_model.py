"""
TirraMind — World Model Orchestrator

Combines the causal DAG (BeliefPropagator) and continuous state filter
(ContinuousStateFilter) into a single coherent update cycle.

Update pipeline per tick:
    1. Map EngineeredFeatures to graph nodes by feature_name
    2. Discretize for DAG, extract raw values for Kalman
    3. Run DAG propagation → categorical beliefs (regime posteriors)
    4. Extract MAP regime from regime.macro posterior
    5. Kalman predict with that regime
    6. Kalman update with continuous feature values
    7. Collect all beliefs (DAG + Kalman) into unified output

Design principles:
    - One-directional coupling: DAG → Kalman (Phase 9a).
    - No LLM calls, no randomness at inference (PRNG seed only in filter init).
    - Output is always list[BeliefState] — full distributions.

References:
    - Spec: docs/specs/world_model_spec.md (sub-phase 9.5)
"""

from __future__ import annotations

import logging
from typing import Any, Union

import numpy as np

from agent.features.protocol import EngineeredFeature
from agent.models.belief import BeliefState
from agent.models.graph import WorldModelGraph
from agent.models.propagator import BeliefPropagator
from agent.models.state_filter import ContinuousStateFilter

log = logging.getLogger(__name__)

# Type alias for either filter backend
_AnyStateFilter = Union[
    ContinuousStateFilter, "DifferentiableKalmanFilter"  # noqa: F821
]


class WorldModel:
    """Top-level orchestrator combining DAG + Kalman into unified belief system.

    Args:
        graph: The causal DAG with CPDs.
        propagator: Belief propagation engine for DAG inference.
        state_filter: Regime-conditioned Kalman filter.
        regime_node: Name of the regime node used to condition the Kalman.
        continuous_state_names: Names for each Kalman state dimension
            (must match state_filter.state_dim).
        feature_to_obs_index: Maps EngineeredFeature.feature_name to
            observation vector index for the Kalman filter.
    """

    def __init__(
        self,
        graph: WorldModelGraph,
        propagator: BeliefPropagator,
        state_filter: _AnyStateFilter,
        regime_node: str = "regime.macro",
        continuous_state_names: list[str] | None = None,
        feature_to_obs_index: dict[str, int] | None = None,
    ) -> None:
        self._graph = graph
        self._propagator = propagator
        self._filter = state_filter
        self._regime_node = regime_node
        self._continuous_state_names = continuous_state_names or []
        self._feature_to_obs_index = feature_to_obs_index or {}

        # Cache latest beliefs
        self._latest_beliefs: dict[str, BeliefState] = {}

        # DAG version history: list of (timestamp, hash) pairs (Change 13)
        self._dag_version_history: list[tuple[float, str]] = []

    @property
    def dag_version(self) -> str:
        """Current DAG structure hash (deterministic SHA-256)."""
        return self._graph.graph_hash()

    @property
    def dag_version_history(self) -> list[tuple[float, str]]:
        """History of (timestamp, hash) pairs tracking structure changes."""
        return list(self._dag_version_history)

    def update(
        self,
        features: list[EngineeredFeature],
        as_of: float,
        version: int = 1,
    ) -> list[BeliefState]:
        """Execute the full update cycle.

        Args:
            features: Current EngineeredFeature values.
            as_of: Unix epoch for effective_at.
            version: World model schema version.

        Returns:
            All beliefs (DAG categorical + Kalman Gaussian).
        """
        # Step 1: Map features to DAG evidence
        dag_evidence: dict[str, Any] = {}
        dag_quality: dict[str, float] = {}

        feature_by_name = {f.feature_name: f for f in features}
        for spec in self._graph.get_observed_nodes():
            if spec.feature_name and spec.feature_name in feature_by_name:
                feat = feature_by_name[spec.feature_name]
                if feat.value is not None:
                    dag_evidence[spec.name] = feat.value
                    dag_quality[spec.name] = feat.quality

        # Step 2: DAG propagation
        dag_beliefs = self._propagator.propagate(
            evidence=dag_evidence,
            as_of=as_of,
            quality=dag_quality,
            version=version,
        )

        # Step 3: Extract MAP regime
        regime = self._extract_map_regime(dag_beliefs)

        # Step 4: Kalman predict + update (if filter is configured)
        kalman_beliefs: list[BeliefState] = []
        if self._continuous_state_names and self._filter.state_dim > 0:
            import numpy as np

            # Predict
            self._filter.predict(regime)

            # Build observation vector and quality for Kalman
            obs = np.full(self._filter.obs_dim, np.nan)
            obs_quality = np.ones(self._filter.obs_dim)

            for feat_name, obs_idx in self._feature_to_obs_index.items():
                if feat_name in feature_by_name:
                    feat = feature_by_name[feat_name]
                    if feat.value is not None:
                        obs[obs_idx] = feat.value
                        obs_quality[obs_idx] = feat.quality

            # Update
            self._filter.update(obs, obs_quality)

            # Get Kalman beliefs
            graph_hash = self._graph.graph_hash()
            kalman_beliefs = self._filter.get_beliefs(
                variable_names=self._continuous_state_names,
                as_of=as_of,
                graph_hash=graph_hash,
                version=version,
            )
            # Set evidence count
            n_evidence = sum(1 for v in obs if not np.isnan(v))
            kalman_beliefs = [
                BeliefState(
                    variable_name=b.variable_name,
                    version=b.version,
                    effective_at=b.effective_at,
                    computed_at=b.computed_at,
                    dist_type=b.dist_type,
                    mean=b.mean,
                    variance=b.variance,
                    probabilities=b.probabilities,
                    evidence_count=n_evidence,
                    model_graph_hash=b.model_graph_hash,
                    confidence=b.confidence,
                    stale=n_evidence == 0,
                )
                for b in kalman_beliefs
            ]

        all_beliefs = dag_beliefs + kalman_beliefs

        # Cache
        self._latest_beliefs = {b.variable_name: b for b in all_beliefs}

        return all_beliefs

    def query(self, variable_name: str) -> BeliefState | None:
        """Return the most recent in-memory belief for a variable."""
        return self._latest_beliefs.get(variable_name)

    def get_graph_hash(self) -> str:
        """Deterministic hash of the underlying DAG structure."""
        return self._graph.graph_hash()

    def _extract_map_regime(self, dag_beliefs: list[BeliefState]) -> str:
        """Extract MAP (most probable) state from the regime node posterior.

        Falls back to the first available regime config if regime node
        not found in beliefs.
        """
        for b in dag_beliefs:
            if b.variable_name == self._regime_node:
                if b.probabilities:
                    return max(b.probabilities, key=b.probabilities.get)

        # Fallback: use first regime config
        configs = list(self._filter._regime_configs.keys())
        if configs:
            return configs[0]

        raise RuntimeError(
            f"No regime posterior found for '{self._regime_node}' and no regime configs available"
        )

    # ── CPD fitting from data (Change 2a) ─────────────────────

    def fit_cpds(
        self,
        feature_history: list[list[EngineeredFeature]],
        *,
        equivalent_sample_size: float = 10.0,
        min_samples: int = 50,
    ) -> dict[str, Any]:
        """Fit CPD parameters from historical feature data via Bayesian estimation.

        Uses pgmpy's BayesianEstimator with BDeu (Bayesian Dirichlet equivalent
        uniform) priors.  The existing expert CPDs serve as warm-start priors
        through the equivalent_sample_size parameter — higher ESS gives more
        weight to the prior relative to data.

        Math:
            P_hat(X_i=x | Pa(X_i)=pi) = (N(x,pi) + alpha) / (N(pi) + alpha*|X_i|)
            where alpha = equivalent_sample_size / num_parent_configs

        Parameters
        ----------
        feature_history : List of feature snapshots (each snapshot is a list
            of EngineeredFeature values from one time step).
        equivalent_sample_size : BDeu hyperparameter. Higher = more prior
            influence, lower = more data-driven.  Default 10 corresponds to
            about 10 pseudo-observations from the prior.
        min_samples : Minimum data rows required before fitting.  If fewer,
            expert CPDs are retained unchanged.

        Returns
        -------
        Dict with 'fitted': bool, 'n_samples': int, 'nodes_fitted': list[str].
        """
        import pandas as pd
        from pgmpy.estimators import BayesianEstimator

        specs = self._graph.node_specs

        # Build a DataFrame of discretized feature observations
        rows: list[dict[str, str]] = []
        for snapshot in feature_history:
            feat_by_name = {f.feature_name: f for f in snapshot}
            row: dict[str, str] = {}
            for node_name, spec in specs.items():
                if spec.feature_name is None or spec.bin_edges is None:
                    continue
                feat = feat_by_name.get(spec.feature_name)
                if feat is None or feat.value is None:
                    continue
                # Discretize: find which bin the value falls in
                state = self._discretize(feat.value, spec.bin_edges, spec.states)
                if state is not None:
                    row[node_name] = state

            # Only include rows with at least some observed nodes
            if len(row) >= 2:
                rows.append(row)

        n_samples = len(rows)
        if n_samples < min_samples:
            log.info(
                "fit_cpds: only %d samples (need %d), keeping expert CPDs.",
                n_samples,
                min_samples,
            )
            return {"fitted": False, "n_samples": n_samples, "nodes_fitted": []}

        df = pd.DataFrame(rows)

        # Fill missing columns with NaN (pgmpy handles partial evidence)
        for node_name in specs:
            if node_name not in df.columns:
                df[node_name] = np.nan

        # Fit using BayesianEstimator with BDeu priors
        bn = self._graph.bn
        estimator = BayesianEstimator(bn, df)

        nodes_fitted: list[str] = []
        for node_name, spec in specs.items():
            if spec.cardinality is None:
                continue
            # Skip nodes with no observed data — BDeu with all-NaN produces
            # malformed CPDs (e.g., latent/regime nodes with no feature_name).
            if node_name not in df.columns or df[node_name].notna().sum() == 0:
                continue
            try:
                cpd = estimator.estimate_cpd(
                    node_name,
                    prior_type="BDeu",
                    equivalent_sample_size=equivalent_sample_size,
                )
                self._graph.set_cpd(node_name, cpd)
                nodes_fitted.append(node_name)
            except Exception as exc:
                log.warning("fit_cpds: failed for %s: %s", node_name, exc)

        # Validate the graph is still well-formed after fitting
        errors = self._graph.validate()
        if errors:
            log.warning("fit_cpds: graph validation errors after fitting: %s", errors)

        log.info(
            "fit_cpds: fitted %d/%d nodes from %d samples (ESS=%.1f).",
            len(nodes_fitted),
            len(specs),
            n_samples,
            equivalent_sample_size,
        )
        return {
            "fitted": True,
            "n_samples": n_samples,
            "nodes_fitted": nodes_fitted,
        }

    @staticmethod
    def _discretize(
        value: float,
        bin_edges: tuple[str, ...] | tuple[float, ...],
        states: tuple[str, ...] | None,
    ) -> str | None:
        """Discretize a continuous value into a state label using bin edges."""
        if states is None or bin_edges is None:
            return None
        edges = [float(e) for e in bin_edges]
        for i in range(len(edges) - 1):
            if edges[i] <= value < edges[i + 1]:
                if i < len(states):
                    return states[i]
                return None
        # Handle upper boundary (value == last edge)
        if len(states) > 0 and value >= edges[-1]:
            return states[-1]
        return None

    # ── Causal DAG structure learning (Change 3) ──────────────

    def refine_structure(
        self,
        feature_history: list[list[EngineeredFeature]],
        *,
        min_samples: int = 200,
        max_indegree: int = 4,
        epsilon: float = 1e-4,
    ) -> dict[str, Any]:
        """Learn causal DAG structure from accumulated data via constrained hill-climb.

        Uses pgmpy's HillClimbSearch with BIC-discrete scoring, warm-started
        from the current expert graph.  Structural constraints enforce domain
        invariants (regime nodes as roots, observed cannot parent latent,
        acyclicity, bounded in-degree).

        Math — score-based structure learning with BIC:
            BIC(G) = sum_i [log P(X_i | Pa_G(X_i)) - (d_i / 2) * log N]
        where d_i is the number of free CPD parameters for node i.  BIC
        naturally penalizes complexity, preventing spurious edges on limited
        data.  Hill-climbing iterates: try all single-edge add/remove/reverse,
        accept the best-scoring move, stop when no improvement exceeds epsilon.

        Trusted source: Koller & Friedman, "Probabilistic Graphical Models"
        (2009), Ch. 18 (Structure Learning).  pgmpy implements the standard
        score-based approach with optional expert constraints.

        Parameters
        ----------
        feature_history : List of feature snapshots (each snapshot is a list
            of EngineeredFeature values from one time step).
        min_samples : Minimum samples required for structure learning.
        max_indegree : Maximum number of parents per node.
        epsilon : Minimum BIC improvement to accept any edge change.

        Returns
        -------
        Dict with keys: 'refined', 'n_samples', 'edges_added', 'edges_removed',
        'new_edge_count', 'old_edge_count'.
        """
        import pandas as pd
        from pgmpy.base import DAG
        from pgmpy.causal_discovery import ExpertKnowledge, HillClimbSearch

        specs = self._graph.node_specs

        # ── Step 1: Build discretized DataFrame ───────────────
        rows: list[dict[str, str]] = []
        for snapshot in feature_history:
            feat_by_name = {f.feature_name: f for f in snapshot}
            row: dict[str, str] = {}
            for node_name, spec in specs.items():
                if spec.feature_name is None or spec.bin_edges is None:
                    continue
                feat = feat_by_name.get(spec.feature_name)
                if feat is None or feat.value is None:
                    continue
                state = self._discretize(feat.value, spec.bin_edges, spec.states)
                if state is not None:
                    row[node_name] = state
            if len(row) >= 2:
                rows.append(row)

        n_samples = len(rows)
        if n_samples < min_samples:
            log.info(
                "refine_structure: only %d samples (need %d), keeping current structure.",
                n_samples,
                min_samples,
            )
            return {
                "refined": False,
                "n_samples": n_samples,
                "edges_added": [],
                "edges_removed": [],
            }

        df = pd.DataFrame(rows)

        # Only include columns (nodes) that have data — regime/latent nodes
        # without feature_names won't appear and that's correct (structure
        # learning operates on observed variables only)
        observed_cols = [c for c in df.columns if c in specs]
        if len(observed_cols) < 3:
            log.info("refine_structure: fewer than 3 observed columns, skipping.")
            return {
                "refined": False,
                "n_samples": n_samples,
                "edges_added": [],
                "edges_removed": [],
            }

        df = df[observed_cols].dropna(how="all")

        # ── Step 2: Build warm-start DAG from current graph ───
        start_dag = DAG()
        start_dag.add_nodes_from(observed_cols)
        current_edges = set(self._graph.edges)
        for parent, child in current_edges:
            if parent in observed_cols and child in observed_cols:
                start_dag.add_edge(parent, child)

        old_observed_edges = set(start_dag.edges())

        # ── Step 3: Build ExpertKnowledge constraints ─────────
        regime_names = {s.name for s in self._graph.get_regime_nodes()}
        latent_names = {s.name for s in self._graph.get_latent_nodes()}

        forbidden: list[tuple[str, str]] = []
        for col in observed_cols:
            # Observed → regime forbidden
            for r in regime_names:
                if r in observed_cols:
                    forbidden.append((col, r))
            # Observed → latent forbidden
            for lt in latent_names:
                if lt in observed_cols:
                    forbidden.append((col, lt))
            # Latent → regime forbidden
            for lt in latent_names:
                if lt in observed_cols:
                    for r in regime_names:
                        if r in observed_cols:
                            forbidden.append((lt, r))

        # Deduplicate
        forbidden = list(set(forbidden))

        expert_knowledge = ExpertKnowledge(forbidden_edges=forbidden)

        # ── Step 4: Run HillClimbSearch ───────────────────────
        hc = HillClimbSearch(
            scoring_method="bic-d",
            start_dag=start_dag,
            max_indegree=max_indegree,
            expert_knowledge=expert_knowledge,
            return_type="dag",
            show_progress=False,
            epsilon=epsilon,
        )

        try:
            hc.fit(df)
        except Exception as exc:
            log.warning("refine_structure: HillClimbSearch failed: %s", exc)
            return {
                "refined": False,
                "n_samples": n_samples,
                "error": str(exc),
                "edges_added": [],
                "edges_removed": [],
            }

        learned_dag = hc.causal_graph_
        learned_observed_edges = set(learned_dag.edges())

        # ── Step 5: Diff edges ────────────────────────────────
        edges_added = sorted(learned_observed_edges - old_observed_edges)
        edges_removed = sorted(old_observed_edges - learned_observed_edges)

        if not edges_added and not edges_removed:
            log.info(
                "refine_structure: no structural changes found (%d samples).", n_samples
            )
            return {
                "refined": False,
                "n_samples": n_samples,
                "edges_added": [],
                "edges_removed": [],
            }

        # ── Step 6: Apply changes to WorldModelGraph ──────────
        for parent, child in edges_removed:
            try:
                self._graph.remove_edge(parent, child)
            except ValueError:
                log.warning(
                    "refine_structure: could not remove edge %s → %s", parent, child
                )

        for parent, child in edges_added:
            try:
                self._graph.add_edge(parent, child)
            except ValueError as exc:
                log.warning(
                    "refine_structure: could not add edge %s → %s: %s",
                    parent,
                    child,
                    exc,
                )

        new_edge_count = len(self._graph.edges)

        log.info(
            "refine_structure: %d added, %d removed (%d→%d edges, %d samples).",
            len(edges_added),
            len(edges_removed),
            len(current_edges),
            new_edge_count,
            n_samples,
        )

        # Record DAG version change (Change 13)
        import time as _time

        new_hash = self._graph.graph_hash()
        self._dag_version_history.append((_time.time(), new_hash))

        return {
            "refined": True,
            "n_samples": n_samples,
            "edges_added": edges_added,
            "edges_removed": edges_removed,
            "old_edge_count": len(current_edges),
            "new_edge_count": new_edge_count,
            "dag_version": new_hash,
        }
