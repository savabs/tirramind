"""
TirraMind — Intervention Engine

Supports causal "what if" queries using do-calculus.  Intervening on a
variable (do(X=x)) severs its incoming edges and forces it to a value,
then computes the resulting posterior distribution.

Mathematical basis:
    P(Y | do(X=x)) ≠ P(Y | X=x) in general.
    do() creates a mutilated graph where all edges into X are removed
    and X is set to x with probability 1.
    The interventional distribution is computed via variable elimination
    on the mutilated graph.

Design principles:
    1. Never modifies the original graph — creates temporary copies.
    2. Supports both latent/regime and observed interventions.
    3. compare_intervention() provides observational vs interventional contrast.
    4. Causal effect measured as KL divergence for categoricals.

References:
    - Pearl, "Causality" (2009), Ch. 3 (do-calculus)
    - pgmpy.inference.CausalInference
    - Spec: docs/specs/world_model_spec.md (sub-phase 9.6)
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from pgmpy.inference import CausalInference, VariableElimination

from agent.models.belief import BeliefState
from agent.models.graph import WorldModelGraph


def _kl_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """KL(P || Q) for categorical distributions.

    Returns inf if Q has zero mass where P is nonzero.
    """
    kl = 0.0
    for state in p:
        pi = p[state]
        qi = q.get(state, 0.0)
        if pi > 0:
            if qi <= 0:
                return float("inf")
            kl += pi * np.log(pi / qi)
    return float(kl)


class InterventionEngine:
    """Causal intervention engine for do-calculus queries.

    Args:
        graph: The causal DAG with CPDs.
    """

    def __init__(self, graph: WorldModelGraph) -> None:
        self._graph = graph

    def intervene(
        self,
        do_variable: str,
        do_value: str,
        evidence: dict[str, str] | None = None,
        version: int = 1,
        as_of: float | None = None,
    ) -> list[BeliefState]:
        """Compute beliefs under a causal intervention do(X=x).

        Args:
            do_variable: Node to intervene on.
            do_value: State label to force the node to.
            evidence: Optional additional observed evidence (not intervened).
            version: World model schema version.
            as_of: Timestamp for beliefs. Defaults to now.

        Returns:
            List of BeliefState for all non-do variables under the intervention.

        Raises:
            ValueError: If do_variable not in graph or do_value not valid.
        """
        self._validate_intervention(do_variable, do_value)

        as_of = as_of or time.time()
        computed_at = time.time()
        graph_hash = self._graph.graph_hash()
        evidence = evidence or {}

        ci = CausalInference(self._graph.bn)

        beliefs: list[BeliefState] = []
        for node_name, spec in self._graph.node_specs.items():
            if node_name == do_variable:
                # Intervened node gets delta distribution
                probs = {s: 0.0 for s in spec.states}
                probs[do_value] = 1.0
                beliefs.append(
                    BeliefState(
                        variable_name=node_name,
                        version=version,
                        effective_at=as_of,
                        computed_at=computed_at,
                        dist_type="categorical",
                        probabilities=probs,
                        evidence_count=0,
                        model_graph_hash=graph_hash,
                        confidence=1.0,
                        stale=False,
                        metadata={"intervention": True},
                    )
                )
                continue

            if node_name in evidence:
                # Evidence node → delta
                probs = {s: 0.0 for s in spec.states}
                probs[evidence[node_name]] = 1.0
                beliefs.append(
                    BeliefState(
                        variable_name=node_name,
                        version=version,
                        effective_at=as_of,
                        computed_at=computed_at,
                        dist_type="categorical",
                        probabilities=probs,
                        evidence_count=len(evidence),
                        model_graph_hash=graph_hash,
                        confidence=1.0,
                        stale=False,
                    )
                )
                continue

            # Query interventional distribution
            try:
                result = ci.query(
                    variables=[node_name],
                    do={do_variable: do_value},
                    evidence=evidence if evidence else None,
                    show_progress=False,
                )
                probs = {state: float(result.values[i]) for i, state in enumerate(spec.states)}
            except Exception:
                # Fallback to uniform
                n = len(spec.states)
                probs = {s: 1.0 / n for s in spec.states}

            beliefs.append(
                BeliefState(
                    variable_name=node_name,
                    version=version,
                    effective_at=as_of,
                    computed_at=computed_at,
                    dist_type="categorical",
                    probabilities=probs,
                    evidence_count=len(evidence),
                    model_graph_hash=graph_hash,
                    confidence=1.0,
                    stale=False,
                    metadata={"intervention": True, "do": {do_variable: do_value}},
                )
            )

        return beliefs

    def compare_intervention(
        self,
        do_variable: str,
        do_value: str,
        evidence: dict[str, str] | None = None,
        version: int = 1,
        as_of: float | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Compare observational vs interventional posteriors.

        Returns:
            Dict mapping variable_name → {
                "observational": BeliefState,
                "interventional": BeliefState,
                "causal_effect": float (KL divergence),
            }
        """
        self._validate_intervention(do_variable, do_value)

        as_of = as_of or time.time()
        evidence = evidence or {}

        # Observational: condition on X=x (not do)
        ve = VariableElimination(self._graph.bn)
        obs_evidence = dict(evidence)
        obs_evidence[do_variable] = do_value

        # Interventional: do(X=x)
        int_beliefs = self.intervene(
            do_variable,
            do_value,
            evidence,
            version,
            as_of,
        )
        int_by_name = {b.variable_name: b for b in int_beliefs}

        computed_at = time.time()
        graph_hash = self._graph.graph_hash()
        result: dict[str, dict[str, Any]] = {}

        for node_name, spec in self._graph.node_specs.items():
            if node_name == do_variable:
                continue
            if node_name in evidence:
                continue

            # Observational posterior
            try:
                obs_result = ve.query(
                    variables=[node_name],
                    evidence=obs_evidence,
                    show_progress=False,
                )
                obs_probs = {state: float(obs_result.values[i]) for i, state in enumerate(spec.states)}
            except Exception:
                n = len(spec.states)
                obs_probs = {s: 1.0 / n for s in spec.states}

            obs_belief = BeliefState(
                variable_name=node_name,
                version=version,
                effective_at=as_of,
                computed_at=computed_at,
                dist_type="categorical",
                probabilities=obs_probs,
                evidence_count=len(evidence) + 1,
                model_graph_hash=graph_hash,
                confidence=1.0,
                stale=False,
            )

            int_belief = int_by_name.get(node_name)
            if int_belief is None:
                continue

            causal_effect = _kl_divergence(
                int_belief.probabilities,
                obs_probs,
            )

            result[node_name] = {
                "observational": obs_belief,
                "interventional": int_belief,
                "causal_effect": causal_effect,
            }

        return result

    def _validate_intervention(
        self,
        do_variable: str,
        do_value: str,
    ) -> None:
        """Check do_variable and do_value are valid."""
        if do_variable not in self._graph.node_specs:
            raise ValueError(f"do_variable '{do_variable}' not in graph. Available: {self._graph.node_names}")
        spec = self._graph.node_specs[do_variable]
        if spec.states is None:
            raise ValueError(f"Node '{do_variable}' has no states — cannot intervene")
        if do_value not in spec.states:
            raise ValueError(f"do_value '{do_value}' not in states for '{do_variable}': {spec.states}")
