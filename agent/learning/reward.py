"""
TirraMind Agent — Reward Computation (RL Layer)

Converts an Evaluation into a scalar reward ∈ [0, 1] for the bandit.
Pure numeric — no LLM in the loop. This is what makes the learning real:
the reward signal drives parameter updates, not prompt engineering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from agent.learning.evaluator import Evaluation
from agent.learning.param_optimizer import BayesianParamOptimizer, ParamSpace

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RewardWeights:
    """Configurable weights for the reward components.

    All components are combined linearly and clamped to [0, 1].
    Adjust these to change what the agent optimizes for:
    - Higher eval_weight → trust the evaluator score more
    - Higher sharpe_weight → prioritize strategy edge
    - Higher facts_weight → prioritize knowledge discovery
    - Higher novelty_bonus → reward exploring new arms
    - Higher dead_end_penalty → punish wasted effort harder
    """

    eval_weight: float = 0.4
    sharpe_weight: float = 0.3
    facts_weight: float = 0.2
    novelty_bonus: float = 0.1
    dead_end_penalty: float = 0.3


# Sensible defaults
DEFAULT_WEIGHTS = RewardWeights()


def compute_reward(
    evaluation: Evaluation,
    is_first_pull: bool = False,
    weights: RewardWeights = DEFAULT_WEIGHTS,
) -> float:
    """Convert an Evaluation into a scalar reward for the bandit.

    Components:
    1. eval_score: The evaluator's 0-1 score (LLM + heuristic blend)
    2. sharpe_quality: If backtest metrics exist, normalize Sharpe → [0,1]
    3. knowledge_gain: new_facts_count / 5, capped at 1.0
    4. novelty: Small bonus for first pull on an arm (encourage exploration)
    5. dead_end: Penalty subtracted if the evaluation flagged a dead end

    Args:
        evaluation: The Evaluation from the evaluator.
        is_first_pull: Whether this is the first time the arm was pulled.
        weights: Configurable component weights.

    Returns:
        Scalar reward in [0, 1].
    """
    # Component 1: base evaluation score (already 0-1)
    eval_component = weights.eval_weight * evaluation.score

    # Component 2: Sharpe quality (backtest-specific)
    sharpe_component = 0.0
    if evaluation.strategy_metrics:
        sharpe = evaluation.strategy_metrics.get("sharpe")
        if sharpe is not None:
            # Normalize: Sharpe -0.5 → 0, Sharpe 1.5 → 1.0
            sharpe_normalized = max(0.0, min(1.0, (sharpe + 0.5) / 2.0))
            sharpe_component = weights.sharpe_weight * sharpe_normalized

    # Component 3: knowledge gain
    facts_normalized = min(1.0, evaluation.new_facts_count / 5.0)
    facts_component = weights.facts_weight * facts_normalized

    # Component 4: novelty bonus for first pulls
    novelty_component = weights.novelty_bonus if is_first_pull else 0.0

    # Component 5: dead-end penalty
    dead_end_component = weights.dead_end_penalty if evaluation.dead_end else 0.0

    # Combine
    raw_reward = (
        eval_component
        + sharpe_component
        + facts_component
        + novelty_component
        - dead_end_component
    )

    # Clamp to [0, 1]
    reward = max(0.0, min(1.0, raw_reward))

    log.info(
        "Reward: %.3f (eval=%.3f sharpe=%.3f facts=%.3f novelty=%.3f dead_end=-%.3f)",
        reward,
        eval_component,
        sharpe_component,
        facts_component,
        novelty_component,
        dead_end_component,
    )

    return reward


# ---------------------------------------------------------------------------
# Reward Weight Optimizer (Change 5 — Tier 3)
# ---------------------------------------------------------------------------

# Bounds for each reward weight component.
# All weights live in [0.01, 1.0]: zero weights collapse the signal;
# values > 1 would dominate the [0,1] clamp and make reward constant.
_REWARD_WEIGHT_SPACE = ParamSpace(
    names=[
        "eval_weight",
        "sharpe_weight",
        "facts_weight",
        "novelty_bonus",
        "dead_end_penalty",
    ],
    bounds=[(0.01, 1.0)] * 5,
)


class RewardWeightOptimizer:
    """GP-Bayesian optimization over the 5 reward weight dimensions.

    Objective: rolling portfolio Sharpe ratio (or any scalar metric
    measured after K autonomous iterations with a given weight vector).

    Usage::

        opt = RewardWeightOptimizer(persist_path=Path("reward_bo.json"))
        weights = opt.suggest_weights()       # → RewardWeights
        # … run K autonomous iterations with these weights …
        opt.record_trial(weights, sharpe=1.2)
        best = opt.current_best()             # → RewardWeights | None
    """

    def __init__(
        self,
        persist_path: Path | None = None,
        *,
        n_random: int = 5,
        seed: int | None = None,
    ) -> None:
        self._bo = BayesianParamOptimizer(
            _REWARD_WEIGHT_SPACE,
            persist_path=persist_path,
            n_random=n_random,
            seed=seed,
        )

    @property
    def n_trials(self) -> int:
        return self._bo.n_trials

    def suggest_weights(self) -> RewardWeights:
        """Suggest the next reward weight vector to evaluate."""
        params = self._bo.suggest()
        return RewardWeights(**params)

    def record_trial(
        self,
        weights: RewardWeights,
        objective: float,
        metadata: dict | None = None,
    ) -> None:
        """Record a completed trial.

        Args:
            weights: The RewardWeights used during the trial.
            objective: Scalar performance metric (higher = better).
            metadata: Optional extra info (e.g., n_iterations, date range).
        """
        params = {
            "eval_weight": weights.eval_weight,
            "sharpe_weight": weights.sharpe_weight,
            "facts_weight": weights.facts_weight,
            "novelty_bonus": weights.novelty_bonus,
            "dead_end_penalty": weights.dead_end_penalty,
        }
        self._bo.record(params, objective, metadata)

    def current_best(self) -> RewardWeights | None:
        """Return the best-performing RewardWeights found so far."""
        best = self._bo.best_params()
        if best is None:
            return None
        return RewardWeights(**best)
