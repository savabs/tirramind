"""
TirraMind Agent — Hierarchical Task Planner

Decomposes a high-level goal into a tree of executable tasks.
Phase 1: LLM-based decomposition.
Phase 4+: MCTS-augmented planning with world model simulation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agent.reasoning.llm_client import LLMClient

log = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    """A single node in the task tree."""
    id: str
    description: str
    tool: str | None = None          # which tool to use (None = needs further decomposition)
    tool_args: dict[str, Any] = field(default_factory=dict)
    success_criteria: str = ""
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    subtasks: list[Task] = field(default_factory=list)
    depth: int = 0

    @property
    def is_leaf(self) -> bool:
        return len(self.subtasks) == 0

    @property
    def is_actionable(self) -> bool:
        return self.tool is not None and self.is_leaf

    def next_pending(self) -> Task | None:
        """Find the next pending leaf task (depth-first)."""
        if self.status == TaskStatus.PENDING and self.is_actionable:
            return self
        for sub in self.subtasks:
            nxt = sub.next_pending()
            if nxt:
                return nxt
        # If this task has no tool but all subtasks are done, it's done too
        if not self.is_leaf and all(s.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED) for s in self.subtasks):
            self.status = TaskStatus.COMPLETED
        return None

    def progress_summary(self) -> str:
        """Human-readable progress."""
        lines = []
        self._collect_summary(lines)
        return "\n".join(lines)

    def _collect_summary(self, lines: list[str], indent: int = 0) -> None:
        prefix = "  " * indent
        icon = {"pending": "○", "in_progress": "◐", "completed": "●", "failed": "✗", "skipped": "—"}
        lines.append(f"{prefix}{icon.get(self.status, '?')} {self.id}: {self.description[:70]}")
        for sub in self.subtasks:
            sub._collect_summary(lines, indent + 1)


_PLAN_SYSTEM_PROMPT = """\
You are the planning module of TirraMind, an autonomous intelligence agent.
Your identity: An autonomous agent that discovers mathematical structure across heterogeneous data through trial and error.

Given a goal, decompose it into a structured task tree.
Each task should be concrete and executable by one of the available tools.

Available tools: {tool_names}

Respond with a JSON array of tasks. Each task has:
- "id": short unique id like "1", "1.1", "1.2", etc.
- "description": what this step does
- "tool": which tool to call (must be one of the available tools, or null if it needs sub-decomposition)
- "tool_args": dict of arguments for the tool
- "success_criteria": how to verify this step worked
- "subtasks": array of child tasks (empty for leaf tasks)

Be thorough but practical. Use 2-3 levels of depth max.
Respond ONLY with the JSON array, no other text.
"""


class TaskPlanner:
    """LLM-powered hierarchical planner."""

    def __init__(self, llm: LLMClient, available_tools: list[str], max_depth: int = 3) -> None:
        self._llm = llm
        self._tools = available_tools
        self._max_depth = max_depth

    def plan(self, goal: str, context: str = "") -> Task:
        """Generate a task tree for the given goal."""
        system = _PLAN_SYSTEM_PROMPT.format(tool_names=", ".join(self._tools))

        prompt = f"Goal: {goal}"
        if context:
            prompt += f"\n\nRelevant context:\n{context}"

        raw = self._llm.structured_output(prompt, system=system)

        if isinstance(raw, list):
            tasks_data = raw
        elif isinstance(raw, str):
            # Try to extract JSON from the string
            log.warning("Planner returned string instead of JSON, attempting parse")
            try:
                tasks_data = json.loads(raw)
            except json.JSONDecodeError:
                # Fallback: create a single research task
                return self._fallback_plan(goal)
        else:
            return self._fallback_plan(goal)

        root = Task(id="root", description=goal)
        root.subtasks = [self._parse_task(t, depth=1) for t in tasks_data]
        return root

    def replan(self, goal: str, completed_summary: str, failure_reason: str) -> Task:
        """Re-plan after a failure or when new information changes the approach."""
        context = (
            f"Previous progress:\n{completed_summary}\n\n"
            f"Reason for re-planning: {failure_reason}\n\n"
            "Adjust the plan based on what we've learned. Do not repeat completed steps."
        )
        return self.plan(goal, context=context)

    def _parse_task(self, data: dict[str, Any], depth: int) -> Task:
        subtasks = []
        if depth < self._max_depth and "subtasks" in data:
            subtasks = [self._parse_task(st, depth + 1) for st in (data["subtasks"] or [])]

        return Task(
            id=str(data.get("id", "?")),
            description=data.get("description", ""),
            tool=data.get("tool"),
            tool_args=data.get("tool_args", {}),
            success_criteria=data.get("success_criteria", ""),
            subtasks=subtasks,
            depth=depth,
        )

    def _fallback_plan(self, goal: str) -> Task:
        """If planning fails, create a minimal viable plan."""
        log.warning("Planning failed — using fallback plan")
        root = Task(id="root", description=goal)
        root.subtasks = [
            Task(id="1", description=f"Search the web for: {goal}", tool="web_search",
                 tool_args={"query": goal}, depth=1),
            Task(id="2", description="Analyze results and produce report", tool="execute_python",
                 tool_args={"code": f"print('Analysis of: {goal}')"}, depth=1),
        ]
        return root
