"""
TirraMind Agent — Goal Generator

Given a reflection and available tools, proposes the most valuable next action.
Includes deduplication against previously attempted goals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent.learning.reflection import ReflectionResult
from agent.reasoning.llm_client import LLMClient

if TYPE_CHECKING:
    from agent.learning.bandit import GoalArm

log = logging.getLogger(__name__)

_GOAL_SYSTEM = """\
You are the goal generator for TirraMind, an autonomous market intelligence system.
Your job: decide the single most valuable next action for the agent to take.
The goal must be concrete and map to a specific tool call. No vague aspirations.
Respond ONLY with valid JSON matching the schema below.
"""

_GOAL_PROMPT = """\
Based on the reflection below, generate the single most valuable next goal.

### Reflection
What worked: {what_worked}
What failed: {what_failed}
Open questions: {open_questions}
Suggested next actions: {suggested_next_actions}

### Available Tools
{tools}

### Previously Attempted Goals (DO NOT repeat these)
{attempted}

Produce a JSON object:
{{
  "description": "Concrete goal description — what the agent should accomplish",
  "rationale": "Why this is the most valuable next step",
  "expected_tool": "The primary tool this goal will use (from the available tools list)",
  "priority": 1 to 5 (5 = highest),
  "is_novel": true if this goal has NOT been attempted before
}}

Rules:
- The description must be specific enough that the agent can execute it without clarification.
- BAD: "Understand markets better" — too vague.
- GOOD: "Run a backtest comparing regime-avoid vs buy-and-hold on SPY using 2020-2024 data" — concrete.
- The expected_tool must be one of the available tools listed above.
- If all suggested actions have been attempted, propose a genuinely new angle.
"""

_ARM_GOAL_PROMPT = """\
The RL policy has selected the action category: **{arm_name}**
Category description: {arm_description}
Allowed tools for this category: {arm_tools}

Generate ONE concrete goal within this category.

### Context from Reflection
What worked: {what_worked}
What failed: {what_failed}
Open questions: {open_questions}

### Example Goals for This Category
{examples}

### Previously Attempted Goals (DO NOT repeat these)
{attempted}

Produce a JSON object:
{{
  "description": "Concrete goal within the '{arm_name}' category",
  "rationale": "Why this specific goal is valuable right now",
  "expected_tool": "Must be one of: {arm_tools}",
  "priority": 1 to 5 (5 = highest),
  "is_novel": true if this goal has NOT been attempted before
}}

Rules:
- The goal MUST use one of the allowed tools: {arm_tools}
- Be specific. Include asset names, date ranges, parameter values.
- Do NOT generate goals outside the '{arm_name}' category.
"""


@dataclass
class Goal:
    """A proposed next action for the autonomous agent."""

    description: str
    rationale: str
    expected_tool: str
    priority: int = 3
    is_novel: bool = True


class GoalGenerator:
    """Proposes the most valuable next goal given reflection and tools."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    # ------------------------------------------------------------------
    # Bandit-driven goal generation (RL layer)
    # ------------------------------------------------------------------

    def generate_for_arm(
        self,
        arm: GoalArm,
        reflection: ReflectionResult,
        attempted_goals: list[str] | None = None,
        max_retries: int = 3,
    ) -> Goal:
        """Generate a goal constrained to a specific arm category.

        The bandit picks the arm (what TYPE of work). This method asks
        the LLM to fill in the specifics within that category.

        Args:
            arm: The GoalArm chosen by the bandit.
            reflection: Output from the Reflector.
            attempted_goals: Goals already tried (for dedup).
            max_retries: Retries if the goal is a duplicate.

        Returns:
            A concrete Goal constrained to the arm's tools.
        """
        attempted = set(g.lower().strip() for g in (attempted_goals or []))

        for attempt in range(max_retries):
            goal = self._generate_for_arm_once(arm, reflection, attempted_goals or [])

            if goal.description.lower().strip() not in attempted:
                log.info(
                    "Generated arm goal (arm=%s, attempt %d): %s",
                    arm.name,
                    attempt + 1,
                    goal.description,
                )
                return goal

            log.info("Arm goal attempt %d was a duplicate, retrying", attempt + 1)

        log.warning("All %d arm goal attempts were duplicates", max_retries)
        goal.is_novel = False
        return goal

    def _generate_for_arm_once(
        self,
        arm: GoalArm,
        reflection: ReflectionResult,
        attempted_goals: list[str],
    ) -> Goal:
        prompt = _ARM_GOAL_PROMPT.format(
            arm_name=arm.name,
            arm_description=arm.description,
            arm_tools=", ".join(arm.tools),
            what_worked=", ".join(reflection.what_worked) or "(none)",
            what_failed=", ".join(reflection.what_failed) or "(none)",
            open_questions=", ".join(reflection.open_questions) or "(none)",
            examples="\n".join(f"- {ex}" for ex in arm.examples) or "(none)",
            attempted="\n".join(f"- {g}" for g in attempted_goals) or "(none)",
        )

        raw = self._llm.structured_output(prompt, system=_GOAL_SYSTEM)

        if isinstance(raw, dict):
            return self._parse_goal(raw, arm.tools)

        log.warning("Arm goal generator LLM failed, using arm fallback")
        return self._arm_fallback_goal(arm)

    @staticmethod
    def _arm_fallback_goal(arm: GoalArm) -> Goal:
        """Fallback: use the first example from the arm definition."""
        if arm.examples:
            desc = arm.examples[0]
        elif arm.tools:
            desc = f"Explore using {arm.tools[0]}"
        else:
            # Novel exploration arm — tools=[], no specific tool constraint
            desc = "Open-ended exploration: combine multiple data sources to find novel patterns"
        tool = arm.tools[0] if arm.tools else "web_search"
        return Goal(
            description=desc,
            rationale=f"Fallback goal for {arm.name} category",
            expected_tool=tool,
            priority=3,
            is_novel=True,
        )

    # ------------------------------------------------------------------
    # Unconstrained goal generation (Phase 4 original — kept for compat)
    # ------------------------------------------------------------------

    def generate(
        self,
        reflection: ReflectionResult,
        available_tools: list[str],
        attempted_goals: list[str] | None = None,
        max_retries: int = 3,
    ) -> Goal:
        """Generate a single concrete goal.

        Args:
            reflection: Output from the Reflector.
            available_tools: Names of registered tools.
            attempted_goals: Goals already tried (for dedup).
            max_retries: Retries if the goal is a duplicate.

        Returns:
            A concrete Goal.
        """
        attempted = set(g.lower().strip() for g in (attempted_goals or []))

        for attempt in range(max_retries):
            goal = self._generate_once(reflection, available_tools, attempted_goals or [])

            # Dedup check: reject if semantically duplicate
            if goal.description.lower().strip() not in attempted:
                log.info("Generated goal (attempt %d): %s", attempt + 1, goal.description)
                return goal

            log.info("Goal attempt %d was a duplicate, retrying", attempt + 1)

        # All retries produced duplicates — return the last one anyway with is_novel=False
        log.warning("All %d goal generation attempts were duplicates", max_retries)
        goal.is_novel = False
        return goal

    def _generate_once(
        self,
        reflection: ReflectionResult,
        available_tools: list[str],
        attempted_goals: list[str],
    ) -> Goal:
        prompt = _GOAL_PROMPT.format(
            what_worked=", ".join(reflection.what_worked) or "(none)",
            what_failed=", ".join(reflection.what_failed) or "(none)",
            open_questions=", ".join(reflection.open_questions) or "(none)",
            suggested_next_actions=", ".join(reflection.suggested_next_actions) or "(none)",
            tools=", ".join(available_tools),
            attempted="\n".join(f"- {g}" for g in attempted_goals) or "(none)",
        )

        raw = self._llm.structured_output(prompt, system=_GOAL_SYSTEM)

        if isinstance(raw, dict):
            return self._parse_goal(raw, available_tools)

        # Fallback: LLM returned garbage
        log.warning("Goal generator LLM did not return valid JSON, using fallback")
        return self._fallback_goal(reflection, available_tools)

    @staticmethod
    def _parse_goal(data: dict, available_tools: list[str]) -> Goal:
        expected_tool = data.get("expected_tool", "")
        # Validate tool name exists
        if expected_tool not in available_tools and available_tools:
            expected_tool = available_tools[0]

        return Goal(
            description=data.get("description", "Explore current market state"),
            rationale=data.get("rationale", ""),
            expected_tool=expected_tool,
            priority=int(data.get("priority", 3)),
            is_novel=bool(data.get("is_novel", True)),
        )

    @staticmethod
    def _fallback_goal(reflection: ReflectionResult, available_tools: list[str]) -> Goal:
        """Generate a reasonable default when LLM parsing fails."""
        # Pick from suggested next actions if available
        if reflection.suggested_next_actions:
            desc = reflection.suggested_next_actions[0]
        else:
            desc = "Fetch current liquidity regime state to assess market conditions"

        tool = (
            "liquidity_regime"
            if "liquidity_regime" in available_tools
            else (available_tools[0] if available_tools else "unknown")
        )
        return Goal(
            description=desc,
            rationale="Fallback goal from reflection suggestions",
            expected_tool=tool,
            priority=3,
            is_novel=True,
        )
