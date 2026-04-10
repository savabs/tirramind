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

from typing import Any

from agent.features.protocol import EngineeredFeature
from agent.models.belief import BeliefState
from agent.models.graph import WorldModelGraph
from agent.models.propagator import BeliefPropagator
from agent.models.state_filter import ContinuousStateFilter


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
        state_filter: ContinuousStateFilter,
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
            f"No regime posterior found for '{self._regime_node}' "
            "and no regime configs available"
        )
