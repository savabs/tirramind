"""
TirraMind Agent — Run Evaluator

Scores agent run outcomes: did it produce new knowledge? Did a strategy show edge?
Detects dead ends. Records structured evaluations for the learning loop.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from agent.core.orchestrator import AgentResult
from agent.learning.goal_generator import Goal
from agent.reasoning.llm_client import LLMClient

log = logging.getLogger(__name__)

_EVAL_SYSTEM = """\
You are the evaluation engine for TirraMind, an autonomous market intelligence system.
Your job: objectively score the outcome of an agent run against its stated goal.
Be honest. A failed run is valuable data if the failure is recorded accurately.
Respond ONLY with valid JSON matching the schema below.
"""

_EVAL_PROMPT = """\
Evaluate the following agent run.

### Goal
{goal}

### Expected Tool
{expected_tool}

### Run Outcome
Success: {success}
Steps taken: {steps}
Output (truncated):
{output}

Produce a JSON object:
{{
  "success": true/false,
  "score": 0.0 to 1.0 (overall quality of outcome),
  "new_facts_count": integer (how many new pieces of knowledge were produced),
  "strategy_metrics": {{}} or null (if backtest: include sharpe, max_drawdown, etc. as found in output),
  "dead_end": true/false (should this approach be abandoned?),
  "lessons": ["list of 1-3 key lessons from this run"]
}}

Scoring guide:
- 0.0: Complete failure, no useful output
- 0.3: Failed but produced some reusable information
- 0.5: Partial success, some new knowledge
- 0.7: Mostly successful, clear new findings
- 1.0: Full success with actionable edge or insight
"""


@dataclass
class Evaluation:
    """Structured assessment of an agent run."""
    success: bool
    score: float  # 0-1
    new_facts_count: int = 0
    strategy_metrics: dict | None = None
    dead_end: bool = False
    lessons: list[str] = field(default_factory=list)


class Evaluator:
    """Scores agent run outcomes against their goals."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def evaluate(self, result: AgentResult, goal: Goal) -> Evaluation:
        """Evaluate an agent run outcome.

        Uses a combination of:
        1. Quantitative extraction (Sharpe, significance from output text)
        2. LLM-based qualitative scoring

        Args:
            result: The AgentResult from orchestrator.run().
            goal: The Goal that was being pursued.

        Returns:
            Structured Evaluation.
        """
        # Try quantitative extraction first for backtest results
        quant_metrics = self._extract_quant_metrics(result.output)

        # Get LLM evaluation
        eval_result = self._llm_evaluate(result, goal)

        # Merge quantitative metrics if found
        if quant_metrics:
            eval_result.strategy_metrics = quant_metrics
            # Boost score if Sharpe is significant
            sharpe = quant_metrics.get("sharpe")
            if sharpe is not None and sharpe > 0.5:
                eval_result.score = max(eval_result.score, 0.7)

        return eval_result

    def _llm_evaluate(self, result: AgentResult, goal: Goal) -> Evaluation:
        prompt = _EVAL_PROMPT.format(
            goal=goal.description,
            expected_tool=goal.expected_tool,
            success=result.success,
            steps=result.steps_taken,
            output=result.output[:1500],
        )

        raw = self._llm.structured_output(prompt, system=_EVAL_SYSTEM)

        if isinstance(raw, dict):
            return self._parse_evaluation(raw)

        log.warning("Evaluator LLM did not return valid JSON, using heuristic")
        return self._heuristic_evaluation(result)

    @staticmethod
    def _parse_evaluation(data: dict) -> Evaluation:
        return Evaluation(
            success=bool(data.get("success", False)),
            score=float(data.get("score", 0.3)),
            new_facts_count=int(data.get("new_facts_count", 0)),
            strategy_metrics=data.get("strategy_metrics"),
            dead_end=bool(data.get("dead_end", False)),
            lessons=data.get("lessons", []),
        )

    @staticmethod
    def _heuristic_evaluation(result: AgentResult) -> Evaluation:
        """Basic scoring when LLM parsing fails."""
        return Evaluation(
            success=result.success,
            score=0.6 if result.success else 0.2,
            new_facts_count=1 if result.success else 0,
            dead_end=not result.success and result.steps_taken <= 1,
            lessons=[
                f"Run {'succeeded' if result.success else 'failed'} "
                f"in {result.steps_taken} steps"
            ],
        )

    @staticmethod
    def _extract_quant_metrics(output: str) -> dict | None:
        """Extract numerical metrics from backtest output text."""
        metrics: dict = {}

        patterns = {
            "sharpe": r"[Ss]harpe[:\s]+(-?\d+\.?\d*)",
            "sortino": r"[Ss]ortino[:\s]+(-?\d+\.?\d*)",
            "max_drawdown": r"[Mm]ax[_ ][Dd]rawdown[:\s]+(-?\d+\.?\d*)",
            "calmar": r"[Cc]almar[:\s]+(-?\d+\.?\d*)",
            "hit_rate": r"[Hh]it[_ ][Rr]ate[:\s]+(-?\d+\.?\d*)",
        }

        for name, pattern in patterns.items():
            match = re.search(pattern, output)
            if match:
                metrics[name] = float(match.group(1))

        return metrics if metrics else None
