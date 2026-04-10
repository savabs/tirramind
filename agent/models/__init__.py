"""
TirraMind — World Model (Layer 3)

Bayesian causal graph + continuous state-space filter for maintaining
probabilistic beliefs over hidden economic variables.

Modules:
    belief      — BeliefState protocol dataclass + validation
    graph       — WorldModelGraph DAG wrapper (pgmpy)
    propagator  — Evidence injection + posterior inference
    state_filter — Regime-conditioned Kalman filter
    initial_graph — Expert-specified initial DAG
    intervention — do-calculus counterfactual queries
    discovery   — Tigramite PCMCI causal structure learning (optional)
    world_model — Top-level orchestrator

References:
    - Research: docs/research/world_model.md
    - Spec: docs/specs/world_model_spec.md
"""

from agent.models.belief import BeliefState, validate_belief

__all__ = ["BeliefState", "validate_belief"]
