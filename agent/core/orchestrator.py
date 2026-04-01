"""
TirraMind Agent — Orchestrator

The main agent loop follows a strict phased pipeline:

    Goal → Research → Plan (Specification) → Execute (Implementation) → Synthesize

This is the central nervous system. It coordinates:
  - Research: understand the problem space before acting
  - Planning: decompose into a structured task tree (the specification)
  - Execution: walk the tree, run tools, replan on failure
  - Synthesis: produce a final report from all results
  - Memory: persist knowledge across runs
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.config.settings import AgentConfig
from agent.memory.store import (
    Episode,
    EpisodicMemory,
    Fact,
    SemanticMemory,
    WorkingMemory,
)
from agent.planner.task_planner import Task, TaskPlanner, TaskStatus
from agent.reasoning.llm_client import LLMClient
from agent.tools.base import ToolRegistry, ToolResult

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are TirraMind, an autonomous machine intelligence agent.
Identity: A self-improving system that observes the full state of the global system — \
physical reality, human decisions, information flows, and market prices across every \
country and asset class — and turns that understanding into money.

Worldview: Markets are outputs. Reality is the input. You operate at Layer 0 (physical \
world: shipping, weather, factories, energy) and Layer 1 (human decisions: policy, trades, \
production, conflict) to understand what generates Layer 2 (information flows) and Layer 3 \
(market prices). Price is a symptom, not a cause. See reality before the scoreboard moves.

Scope: Global. Every equity market, bond market, commodity, currency pair, money market, \
volatility surface, credit spread, physical flow, and behavioral signal on Earth is your \
sensory surface. No country is excluded. No data source is irrelevant until proven so.

Bar: Renaissance Technologies (quantitative rigor) + frontier AI (deep learning, causal \
inference, world models).
Goal: Find information asymmetries that others miss. Predict. Profit.

You receive tasks and execute them using available tools.
After each tool result, decide:
1. Task is complete → report the result
2. Task needs more work → use another tool
3. Task is failing → explain what went wrong

Be precise. Cite sources. Carry uncertainty explicitly.
"""


@dataclass
class AgentResult:
    """Final output from an agent run."""
    goal: str
    success: bool
    output: str
    steps_taken: int
    plan_summary: str
    episodes: list[Episode]


class Orchestrator:
    """Core agent loop."""

    def __init__(self, config: AgentConfig, tool_registry: ToolRegistry) -> None:
        self._config = config
        self._llm = LLMClient(config.llm)
        self._tools = tool_registry
        self._planner = TaskPlanner(
            llm=self._llm,
            available_tools=self._tools.list_names(),
            max_depth=config.max_plan_depth,
        )
        mem_dir = Path(config.memory_dir)
        self._episodic = EpisodicMemory(persist_path=mem_dir / "episodic.jsonl")
        self._semantic = SemanticMemory(persist_path=mem_dir / "semantic.jsonl")
        self._working = WorkingMemory(system_prompt=_SYSTEM_PROMPT)
        self._step = 0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, goal: str, on_step: Any = None) -> AgentResult:
        """Execute the full agent loop for a given goal.

        Pipeline: Research → Plan → Execute → Synthesize.

        Args:
            goal: The high-level objective.
            on_step: Optional callback(step: int, task: Task, result: ToolResult)
                     for real-time progress reporting.
        """
        log.info("Agent starting: %s", goal)
        self._step = 0
        self._working.clear()
        self._working.add_user(f"Goal: {goal}")

        # Create task tracking file
        task_slug = self._slugify(goal)
        self._write_task_file(task_slug, goal, status="active")

        # Phase 1: Research — understand before acting
        research_context = self._research(goal)
        log.info("Research complete:\n%s", research_context[:300])

        # Phase 2: Plan (Specification) — structured task decomposition
        combined_context = self._semantic.summary()
        if research_context:
            combined_context += f"\n\nResearch findings:\n{research_context}"
        plan = self._planner.plan(goal, context=combined_context)
        log.info("Plan generated:\n%s", plan.progress_summary())

        # Phase 3: Execute (Implementation) — walk task tree
        replan_count = 0
        max_replans = 2

        while self._step < self._config.max_steps:
            task = plan.next_pending()
            if task is None:
                # All tasks completed
                break

            task.status = TaskStatus.IN_PROGRESS
            self._step += 1

            log.info("Step %d: %s → %s(%s)",
                     self._step, task.description, task.tool, task.tool_args)

            # Execute
            result = self._execute_task(task)

            # Record episode
            episode = Episode(
                timestamp=time.time(),
                step=self._step,
                action=task.tool or "unknown",
                input_summary=f"{task.description} | {task.tool_args}",
                output_summary=result.output[:200],
                success=result.success,
            )
            self._episodic.add(episode)

            # Update task status
            if result.success:
                task.status = TaskStatus.COMPLETED
                task.result = result.output
                # Store successful results as facts in semantic memory
                self._semantic.store(Fact(
                    key=f"result:{task.id}:{self._step}",
                    content=result.output[:300],
                    source=task.tool or "unknown",
                    confidence=0.8,
                    tags=["execution_result", task.tool or "unknown"],
                ))
            else:
                task.status = TaskStatus.FAILED
                task.result = result.output
                log.warning("Task %s failed: %s", task.id, result.output[:100])

                # Try re-planning
                if replan_count < max_replans:
                    replan_count += 1
                    log.info("Re-planning (attempt %d/%d)", replan_count, max_replans)
                    plan = self._planner.replan(
                        goal=goal,
                        completed_summary=self._episodic.summary(),
                        failure_reason=result.output[:300],
                    )

            # Progress callback
            if on_step:
                on_step(self._step, task, result)

            # Update working memory with result
            self._working.add_user(
                f"Step {self._step} ({task.tool}): {result.output[:500]}"
            )

        # Phase 4: Synthesize final output
        all_done = plan.next_pending() is None
        output = self._synthesize(goal, plan)

        # Update task tracking file
        final_status = "completed" if all_done else "partial"
        self._write_task_file(task_slug, goal, status=final_status)

        return AgentResult(
            goal=goal,
            success=all_done,
            output=output,
            steps_taken=self._step,
            plan_summary=plan.progress_summary(),
            episodes=self._episodic.all(),
        )

    # ------------------------------------------------------------------
    # Phase 1: Research
    # ------------------------------------------------------------------

    def _research(self, goal: str) -> str:
        """Analyze the goal before planning.

        Asks the LLM to identify what needs to be understood, what
        information is required, and what risks exist — without taking
        any action yet. The output feeds into the planner as context.
        """
        prompt = (
            f"Goal: {goal}\n\n"
            f"Available tools: {', '.join(self._tools.list_names())}\n\n"
            "Before creating an execution plan, perform a research analysis:\n"
            "1. What information is needed to accomplish this goal?\n"
            "2. What tools and approaches are most relevant?\n"
            "3. What risks or edge cases should the plan account for?\n"
            "4. What is the expected structure of the solution?\n\n"
            "Be concise. This analysis will inform the planning phase."
        )
        research = self._llm.ask(prompt, system=_SYSTEM_PROMPT)

        # Record as an episode
        self._episodic.add(Episode(
            timestamp=time.time(),
            step=0,
            action="research",
            input_summary=f"Research for: {goal[:100]}",
            output_summary=research[:200],
            success=True,
        ))

        # Store research findings in semantic memory for cross-run persistence
        self._semantic.store(Fact(
            key=f"research:{self._slugify(goal)}",
            content=research[:500],
            source="research_phase",
            confidence=0.7,
            tags=["research", "goal_analysis"],
        ))

        # Store in working memory so the planner has full context
        self._working.add_assistant(f"Research analysis:\n{research}")

        return research

    # ------------------------------------------------------------------
    # Phase 3: Execution helpers
    # ------------------------------------------------------------------

    def _execute_task(self, task: Task) -> ToolResult:
        """Execute a single task via the tool registry."""
        if not task.tool:
            return ToolResult(success=False, output="Task has no tool assigned")

        # Let the LLM refine tool arguments if needed
        if not task.tool_args:
            task.tool_args = self._infer_tool_args(task)

        return self._tools.execute(task.tool, **task.tool_args)

    def _infer_tool_args(self, task: Task) -> dict[str, Any]:
        """Use LLM to figure out the right arguments for a tool call."""
        tool = self._tools.get(task.tool)
        if tool is None:
            return {}

        # Build rich context for the LLM
        recent_history = self._episodic.summary(n=5)
        prompt = (
            f"Goal: {self._working.get_messages()[1]['content'] if len(self._working.get_messages()) > 1 else 'unknown'}\n"
            f"Recent history:\n{recent_history}\n\n"
            f"Current task: {task.description}\n"
            f"Tool: {task.tool}\n"
            f"Tool parameters schema: {tool.parameters}\n\n"
            "Based on the goal and history, provide the correct arguments as a JSON object."
        )
        result = self._llm.structured_output(prompt)
        if isinstance(result, dict):
            return result
        return {}

    def _synthesize(self, goal: str, plan: Task) -> str:
        """Ask the LLM to synthesize a final report from all collected results."""
        results_text = []
        for task in self._collect_completed(plan):
            results_text.append(f"[{task.id}] {task.description}:\n{task.result[:500]}\n")

        prompt = (
            f"Original goal: {goal}\n\n"
            f"Completed task results:\n{''.join(results_text)}\n\n"
            "Synthesize a comprehensive final report. Include:\n"
            "1. Key findings\n"
            "2. Supporting evidence with sources\n"
            "3. Confidence assessment\n"
            "4. Recommendations for further investigation\n"
        )
        return self._llm.ask(prompt, system=_SYSTEM_PROMPT)

    def _collect_completed(self, task: Task) -> list[Task]:
        """Gather all completed leaf tasks."""
        completed = []
        if task.status == TaskStatus.COMPLETED and task.is_leaf and task.result:
            completed.append(task)
        for sub in task.subtasks:
            completed.extend(self._collect_completed(sub))
        return completed

    # ------------------------------------------------------------------
    # Task tracking
    # ------------------------------------------------------------------

    @staticmethod
    def _slugify(text: str) -> str:
        """Convert goal text to a filesystem-safe slug."""
        slug = text.lower().strip()[:60]
        slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
        return slug or "task"

    @staticmethod
    def _write_task_file(slug: str, goal: str, status: str) -> None:
        """Write or update a task tracking file in tasks/active/."""
        tasks_dir = Path("tasks/active")
        tasks_dir.mkdir(parents=True, exist_ok=True)
        task_path = tasks_dir / f"{slug}.md"
        content = (
            f"# Task: {slug}\n\n"
            f"Status: {status}\n"
            f"Goal: {goal}\n"
        )
        task_path.write_text(content)
