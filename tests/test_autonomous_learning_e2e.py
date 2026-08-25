"""End-to-end proof: the full autonomous learning loop actually improves.

Keeps the REAL StrategyBandit (pure RL) and REAL compute_reward (success-anchored
reward after the 2026-08-24 fix). Replaces only the LLM components
(reflector / goal generator / evaluator / orchestrator) with deterministic
fakes over a mock world where some arms succeed and others fail.

The claim under test: the bandit's arm selection shifts toward the successful
arms over the course of a run — i.e. the full loop compounds, end to end.
"""

from __future__ import annotations

import random

import pytest

from agent.config.settings import AgentConfig, LLMConfig
from agent.core.autonomous import AutonomousRunner
from agent.core.orchestrator import AgentResult
from agent.learning.bandit import GoalArm
from agent.learning.evaluator import Evaluation
from agent.learning.goal_generator import Goal
from agent.learning.reflection import ReflectionResult

# ── Mock world: which arms succeed ───────────────────────────────────────────
SUCCESS_ARMS = {
    "backtest_strategy": 0.95,
    "research_market": 0.90,
    "insider_flow": 0.85,
}
FAIL_ARMS = {
    "weather_disruption": 0.10,
    "seismic_risk": 0.05,
    "pandemic_surveillance": 0.15,
    "novel_exploration": 0.20,
    "crypto_whale_flows": 0.25,
}


def _arm_success_rate(arm_name: str) -> float:
    return SUCCESS_ARMS.get(arm_name, FAIL_ARMS.get(arm_name, 0.30))


# ── Deterministic fakes for the LLM components ──────────────────────────────
class _FakeReflector:
    def reflect(self, episodes, semantic_facts, attempted_goals) -> ReflectionResult:
        return ReflectionResult(
            what_worked=["mock"],
            what_failed=[],
            open_questions=[],
        )


class _FakeGoalGenerator:
    def generate_for_arm(self, arm: GoalArm, reflection, attempted_goals=None, max_retries=3) -> Goal:
        # Deterministic: goal.description = arm name (uniquely, no dupes)
        return Goal(description=arm.name, rationale="mock", expected_tool="mock")


class _FakeEvaluator:
    def evaluate(self, result: AgentResult, goal: Goal) -> Evaluation:
        return Evaluation(
            success=result.success,
            score=1.0 if result.success else 0.0,
            new_facts_count=1 if result.success else 0,
            strategy_metrics=None,
            dead_end=not result.success,
        )


class _FakeOrchestrator:
    """Mock world: returns success per the arm's hidden true success rate."""

    def __init__(self, config, tool_registry) -> None:
        self._rng = random.Random(0)

    def run(self, goal_description: str) -> AgentResult:
        ok = self._rng.random() < _arm_success_rate(goal_description)
        return AgentResult(
            goal=goal_description,
            success=ok,
            output="edge found" if ok else "failed",
            steps_taken=3 if ok else 1,
            plan_summary="mock",
            episodes=[],
        )


@pytest.fixture
def runner_factory(tmp_path):
    def _make():
        cfg = AgentConfig(
            llm=LLMConfig(provider="openai", api_key="test", model="mock"),
            memory_dir=str(tmp_path),
        )
        runner = AutonomousRunner(config=cfg, tool_registry=[], max_iterations=120, max_consecutive_failures=500)
        # Deterministic Thompson sampling + deterministic reward-weight suggestions
        # so the learning proof is stable in any test order (the reward-weight BO
        # and bandit RNG were previously non-hermetic, causing run-order flakiness).
        runner._bandit._rng.seed(7)
        from agent.learning.reward import RewardWeightOptimizer
        runner._reward_optimizer = RewardWeightOptimizer(
            persist_path=runner._reward_optimizer._bo._persist_path, seed=7, n_random=5
        )
        runner._reflector = _FakeReflector()
        runner._goal_generator = _FakeGoalGenerator()
        runner._evaluator = _FakeEvaluator()
        return runner

    return _make


def test_autonomous_loop_arm_selection_improves_over_time(runner_factory, monkeypatch):
    import agent.core.autonomous as autonomous_mod

    # Patch the inline Orchestrator with the mock world.
    monkeypatch.setattr(autonomous_mod, "Orchestrator", _FakeOrchestrator)

    runner = runner_factory()
    summary = runner.run()

    # Collect the arm chosen in each iteration window.
    iterations = summary.iterations
    assert len(iterations) == 120

    # Early window = first third; late window = last third.
    third = len(iterations) // 3
    early_arms = [it.arm.name for it in iterations[:third]]
    late_arms = [it.arm.name for it in iterations[2 * third :]]

    def success_share(arms):
        return sum(1 for a in arms if a in SUCCESS_ARMS) / max(len(arms), 1)

    early_share = success_share(early_arms)
    late_share = success_share(late_arms)

    # The bandit must shift toward the successful arms over time.
    assert late_share > early_share, (
        f"no shift: early={early_share:.2f} late={late_share:.2f}"
    )
    # And the majority of late selections should be successful arms.
    assert late_share > 0.5, f"late selections still mostly failing: {late_share:.2f}"
