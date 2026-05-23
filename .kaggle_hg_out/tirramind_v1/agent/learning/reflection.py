"""
TirraMind Agent — Reflection Engine

Reviews recent episodes and semantic facts to produce structured assessments:
what worked, what failed, what open questions remain, and what to try next.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from agent.memory.store import Episode, Fact
from agent.reasoning.llm_client import LLMClient

log = logging.getLogger(__name__)

_REFLECTION_SYSTEM = """\
You are the reflection engine of TirraMind, an autonomous market intelligence system.
Your job: review past actions and their outcomes, then produce a structured assessment.
Be honest about what failed. Be specific about what to try next.
Respond ONLY with valid JSON matching the schema below.
"""

_REFLECTION_PROMPT = """\
Review the following episodes (agent actions and outcomes) and known facts.

### Recent Episodes
{episodes}

### Known Facts
{facts}

### Previously Attempted Goals
{attempted}

Produce a JSON object with these fields:
{{
  "what_worked": ["list of strategies/actions that produced useful results"],
  "what_failed": ["list of strategies/actions that failed or were unproductive"],
  "open_questions": ["list of unresolved questions worth investigating"],
  "suggested_next_actions": ["list of concrete, specific next steps the agent should take"],
  "confidence": 0.0 to 1.0
}}

Rules:
- Each suggested_next_action must be concrete enough to map to a specific tool call.
- Do NOT suggest actions that appear in the previously attempted goals list unless you have a new reason to retry.
- If there are no episodes (cold start), suggest exploratory actions.
- Be concise. 3-5 items per list maximum.
"""


@dataclass
class ReflectionResult:
    """Structured output from a reflection pass."""

    what_worked: list[str] = field(default_factory=list)
    what_failed: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    suggested_next_actions: list[str] = field(default_factory=list)
    confidence: float = 0.5

    @classmethod
    def cold_start(cls) -> ReflectionResult:
        """Default reflection when there's no history to review."""
        return cls(
            what_worked=[],
            what_failed=[],
            open_questions=[
                "What data sources are available and reliable?",
                "What market regimes are currently active?",
            ],
            suggested_next_actions=[
                "Fetch current liquidity regime state using liquidity_regime tool",
                "Run a backtest on regime-based strategies to establish baselines",
                "Search for recent macro data releases that could shift regimes",
            ],
            confidence=0.3,
        )


class Reflector:
    """Reviews past episodes and facts to produce structured assessments."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def reflect(
        self,
        episodes: list[Episode],
        semantic_facts: list[Fact],
        attempted_goals: list[str] | None = None,
    ) -> ReflectionResult:
        """Analyze recent history and produce a reflection.

        Args:
            episodes: Recent agent episodes to review.
            semantic_facts: Known facts from semantic memory.
            attempted_goals: Goals already tried (for dedup).

        Returns:
            Structured ReflectionResult.
        """
        if not episodes:
            log.info("No episodes to reflect on — returning cold start reflection")
            return ReflectionResult.cold_start()

        episodes_text = self._format_episodes(episodes)
        facts_text = self._format_facts(semantic_facts)
        attempted_text = "\n".join(f"- {g}" for g in (attempted_goals or [])) or "(none)"

        prompt = _REFLECTION_PROMPT.format(
            episodes=episodes_text,
            facts=facts_text,
            attempted=attempted_text,
        )

        raw = self._llm.structured_output(prompt, system=_REFLECTION_SYSTEM)

        if isinstance(raw, dict):
            return self._parse_result(raw)

        # LLM returned unparseable output — fallback
        log.warning("Reflection LLM output was not valid JSON, using fallback")
        return self._fallback_reflection(episodes)

    @staticmethod
    def _format_episodes(episodes: list[Episode]) -> str:
        lines = []
        for ep in episodes[-20:]:  # Cap at 20 most recent
            status = "SUCCESS" if ep.success else "FAILED"
            lines.append(
                f"- [{status}] Step {ep.step}: {ep.action}({ep.input_summary[:80]}) → {ep.output_summary[:120]}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_facts(facts: list[Fact]) -> str:
        if not facts:
            return "(no facts stored)"
        lines = []
        for f in facts[-15:]:  # Cap at 15 most recent
            lines.append(f"- [{f.key}] {f.content[:100]} (confidence={f.confidence:.1f})")
        return "\n".join(lines)

    @staticmethod
    def _parse_result(data: dict) -> ReflectionResult:
        return ReflectionResult(
            what_worked=data.get("what_worked", []),
            what_failed=data.get("what_failed", []),
            open_questions=data.get("open_questions", []),
            suggested_next_actions=data.get("suggested_next_actions", []),
            confidence=float(data.get("confidence", 0.5)),
        )

    @staticmethod
    def _fallback_reflection(episodes: list[Episode]) -> ReflectionResult:
        """Generate a basic reflection without LLM when parsing fails."""
        successes = [ep for ep in episodes if ep.success]
        failures = [ep for ep in episodes if not ep.success]
        return ReflectionResult(
            what_worked=[f"{ep.action}: {ep.output_summary[:60]}" for ep in successes[-3:]],
            what_failed=[f"{ep.action}: {ep.output_summary[:60]}" for ep in failures[-3:]],
            open_questions=["What approach should be tried next?"],
            suggested_next_actions=[
                "Run liquidity regime detection to check current market state",
                "Run a backtest to evaluate strategy performance",
            ],
            confidence=0.3,
        )
