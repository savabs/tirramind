"""
TirraMind Agent — Autonomous Runner

The self-directed loop with RL-driven goal selection:
  reflect → bandit chooses arm → LLM fills specifics → execute → reward → bandit updates

The bandit (Thompson Sampling) makes the strategic decision of WHAT TYPE of work
to do, trained by numeric rewards from past iterations. The LLM fills in details
within that category. This is real learning: parameters update, behavior changes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agent.config.settings import AgentConfig
from agent.core.orchestrator import AgentResult, Orchestrator
from agent.learning.bandit import GoalArm, StrategyBandit, DEFAULT_ARMS
from agent.learning.evaluator import Evaluation, Evaluator
from agent.learning.goal_generator import Goal, GoalGenerator
from agent.learning.reflection import ReflectionResult, Reflector
from agent.learning.reward import RewardWeightOptimizer, compute_reward
from agent.memory.candidates import CandidateStore
from agent.memory.store import (
    EpisodicMemory,
    LearningEntry,
    SemanticMemory,
    WorkingMemory,
)
from agent.reasoning.llm_client import LLMClient
from agent.tools.base import ToolRegistry

log = logging.getLogger(__name__)


@dataclass
class LoopIteration:
    """Record of one autonomous loop iteration."""

    iteration: int
    arm: GoalArm  # which arm the bandit chose
    goal: Goal
    result: AgentResult
    evaluation: Evaluation
    reflection: ReflectionResult
    reward: float  # scalar reward fed back to bandit
    timestamp: float = field(default_factory=time.time)


@dataclass
class AutonomousRunSummary:
    """Summary of an entire autonomous session."""

    iterations_completed: int
    total_goals_attempted: int
    successful_goals: int
    failed_goals: int
    dead_ends_found: int
    stop_reason: str
    bandit_report: str = ""  # bandit stats after the run
    iterations: list[LoopIteration] = field(default_factory=list)

    def report(self) -> str:
        lines = [
            f"Autonomous Run Summary",
            f"  Iterations: {self.iterations_completed}",
            f"  Goals attempted: {self.total_goals_attempted}",
            f"  Successes: {self.successful_goals}",
            f"  Failures: {self.failed_goals}",
            f"  Dead ends: {self.dead_ends_found}",
            f"  Stop reason: {self.stop_reason}",
            "",
        ]
        for it in self.iterations:
            status = "✓" if it.evaluation.success else "✗"
            lines.append(
                f"  [{status}] #{it.iteration} arm={it.arm.name:20s} "
                f"reward={it.reward:.3f}  {it.goal.description[:60]}"
            )
        if self.bandit_report:
            lines.append("")
            lines.append(self.bandit_report)
        return "\n".join(lines)


class AutonomousRunner:
    """Self-directed agent loop with RL-driven goal selection.

    Decision flow per iteration:
      1. REFLECT on history (LLM — context gathering)
      2. BANDIT chooses arm category (RL — Thompson Sampling, no LLM)
      3. LLM generates specific goal within that category
      4. ORCHESTRATOR executes the goal
      5. EVALUATOR scores the outcome
      6. REWARD computed numerically (no LLM)
      7. BANDIT updates arm parameters (actual learning step)

    Guardrails:
    - max_iterations: hard cap on goals per session
    - goal deduplication: don't repeat attempted goals
    - stuck detection: consecutive failures → pause
    - bandit persistence: learning survives across sessions
    """

    def __init__(
        self,
        config: AgentConfig,
        tool_registry: ToolRegistry,
        max_iterations: int = 5,
        max_consecutive_failures: int = 3,
        on_iteration: Callable[[LoopIteration], None] | None = None,
    ) -> None:
        self._config = config
        self._tool_registry = tool_registry
        self._max_iterations = max_iterations
        self._max_consecutive_failures = max_consecutive_failures
        self._on_iteration = on_iteration

        # Shared LLM client for learning components
        self._llm = LLMClient(config.llm)

        # RL layer: bandit for goal-type selection
        mem_dir = Path(config.memory_dir)
        self._bandit = StrategyBandit(
            arms=DEFAULT_ARMS,
            persist_path=mem_dir / "bandit_state.json",
        )

        # GP-Bayesian optimization of reward weights (Tier 3, Change 5)
        self._reward_optimizer = RewardWeightOptimizer(
            persist_path=mem_dir / "reward_bo.json",
        )

        # Learning components (LLM-based, for language tasks)
        self._reflector = Reflector(self._llm)
        self._goal_generator = GoalGenerator(self._llm)
        self._evaluator = Evaluator(self._llm)

        # Shared memory (across all iterations)
        self._episodic = EpisodicMemory(persist_path=mem_dir / "episodic.jsonl")
        self._semantic = SemanticMemory(persist_path=mem_dir / "semantic.jsonl")
        self._candidates = CandidateStore(
            persist_path=mem_dir / "candidates.jsonl",
            min_support=config.lesson_min_support,
            min_runs=config.lesson_min_runs,
        )

    def run(self) -> AutonomousRunSummary:
        """Execute the autonomous loop with RL-driven decisions.

        Returns:
            Summary of the entire session including bandit stats.
        """
        log.info(
            "Autonomous runner starting (max_iterations=%d, bandit_pulls=%d)",
            self._max_iterations,
            self._bandit.total_pulls,
        )

        # Suggest reward weights for this run via GP-BO (Tier 3, Change 5)
        learned_weights = self._reward_optimizer.suggest_weights()
        log.info(
            "Using learned reward weights: eval=%.3f sharpe=%.3f "
            "facts=%.3f novelty=%.3f dead_end=%.3f",
            learned_weights.eval_weight,
            learned_weights.sharpe_weight,
            learned_weights.facts_weight,
            learned_weights.novelty_bonus,
            learned_weights.dead_end_penalty,
        )

        run_id = f"run_{int(time.time())}"
        iterations: list[LoopIteration] = []
        consecutive_failures = 0
        stop_reason = "max_iterations_reached"

        for i in range(1, self._max_iterations + 1):
            log.info("=== Autonomous iteration %d/%d ===", i, self._max_iterations)

            # 1. REFLECT on history (LLM — context gathering)
            # Use validated learnings for dead-end checks;
            # all goals for dedup (avoid re-attempting even unvalidated ones)
            attempted = self._semantic.get_attempted_goals()
            reflection = self._reflector.reflect(
                episodes=self._episodic.recent(20),
                semantic_facts=self._semantic.all_facts(),
                attempted_goals=attempted,
            )
            log.info(
                "Reflection: %d worked, %d failed, %d open questions",
                len(reflection.what_worked),
                len(reflection.what_failed),
                len(reflection.open_questions),
            )

            # 2. BANDIT chooses arm (RL — Thompson Sampling, NO LLM)
            arm = self._bandit.choose()
            log.info("Bandit chose arm: %s (%s)", arm.name, arm.description)

            # 3. LLM generates specific goal within arm category
            goal = self._goal_generator.generate_for_arm(
                arm=arm,
                reflection=reflection,
                attempted_goals=attempted,
            )
            log.info("Goal: %s (tool=%s)", goal.description, goal.expected_tool)

            # 4. EXECUTE via orchestrator
            orchestrator = Orchestrator(
                config=self._config,
                tool_registry=self._tool_registry,
            )
            result = orchestrator.run(goal.description)

            # Merge episodes from this run into shared memory
            for ep in result.episodes:
                if ep not in self._episodic.all():
                    self._episodic.add(ep)

            # 5. EVALUATE outcome (LLM + quantitative extraction)
            evaluation = self._evaluator.evaluate(result, goal)
            log.info(
                "Evaluation: success=%s score=%.2f dead_end=%s",
                evaluation.success,
                evaluation.score,
                evaluation.dead_end,
            )

            # 6. COMPUTE REWARD (pure numeric — NO LLM)
            reward = compute_reward(
                evaluation=evaluation,
                is_first_pull=self._bandit.is_first_pull(arm.name),
                weights=learned_weights,
            )

            # 7. BANDIT UPDATE (actual RL learning step)
            self._bandit.update(arm.name, reward)

            # 7b. NOVEL ARM RECORDING (Tier 3, Change 8)
            # If the bandit chose novel_exploration, extract which tools
            # were actually used and record the pull for auto-promotion.
            if arm.name == "novel_exploration" and result.episodes:
                tools_used = list({ep.action for ep in result.episodes if ep.action})
                promoted = self._bandit.record_novel_pull(
                    tools_used=tools_used,
                    reward=reward,
                    description=goal.description,
                )
                if promoted:
                    log.info(
                        "Novel exploration promoted new arm: '%s' (tools=%s)",
                        promoted.name,
                        promoted.tools,
                    )

            # 8. Store learning with arm + reward info
            learning_entry = LearningEntry(
                goal=goal.description,
                score=evaluation.score,
                success=evaluation.success,
                dead_end=evaluation.dead_end,
                lessons=evaluation.lessons,
                arm=arm.name,
                reward=reward,
                run_id=run_id,
            )
            self._semantic.store_learning(learning_entry)

            # 9. CANDIDATE PROMOTION — stage + evaluate lessons
            promo = self._candidates.process([learning_entry], run_id=run_id)
            if promo.promoted:
                for key in promo.promoted:
                    # Mark the originating LearningEntry as validated
                    self._semantic.mark_validated(goal.description, run_id)
                log.info(
                    "Candidate pipeline: %d updated, %d promoted, %d rejected",
                    promo.candidates_updated,
                    len(promo.promoted),
                    len(promo.rejected),
                )

            # Record iteration
            iteration = LoopIteration(
                iteration=i,
                arm=arm,
                goal=goal,
                result=result,
                evaluation=evaluation,
                reflection=reflection,
                reward=reward,
            )
            iterations.append(iteration)

            if self._on_iteration:
                self._on_iteration(iteration)

            # Guardrail: consecutive failure detection
            if evaluation.success:
                consecutive_failures = 0
            else:
                consecutive_failures += 1

            if consecutive_failures >= self._max_consecutive_failures:
                stop_reason = f"stuck: {consecutive_failures} consecutive failures"
                log.warning("Autonomous runner stuck — %s", stop_reason)
                break
        else:
            stop_reason = "max_iterations_reached"

        # Record reward weight trial (Tier 3, Change 5)
        # Objective: mean reward across iterations in this run — higher = better weights
        if iterations:
            mean_reward = sum(it.reward for it in iterations) / len(iterations)
            self._reward_optimizer.record_trial(learned_weights, objective=mean_reward)
            log.info(
                "Recorded reward weight trial: mean_reward=%.3f (n=%d iterations)",
                mean_reward,
                len(iterations),
            )

        # Episodic decay — remove old episodes, archive before deletion
        archive_dir = Path(self._config.memory_dir) / "episodic_archive"
        decayed = self._episodic.decay(
            max_age_days=self._config.episode_ttl_days,
            archive_dir=archive_dir,
        )
        if decayed:
            log.info("Decayed %d old episodes (TTL=%dd)", decayed, self._config.episode_ttl_days)

        summary = AutonomousRunSummary(
            iterations_completed=len(iterations),
            total_goals_attempted=len(iterations),
            successful_goals=sum(1 for it in iterations if it.evaluation.success),
            failed_goals=sum(1 for it in iterations if not it.evaluation.success),
            dead_ends_found=sum(1 for it in iterations if it.evaluation.dead_end),
            stop_reason=stop_reason,
            bandit_report=self._bandit.stats_report(),
            iterations=iterations,
        )

        log.info("Autonomous run complete:\n%s", summary.report())
        return summary
