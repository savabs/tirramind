"""Heuristic, free, offline classifier.

Uses weighted keyword / regex matching per category. Returns bounded
confidence (``heuristic_confidence_ceiling`` caps it so that high-stakes
decisions always have room for an LLM confirmation in hybrid mode).

This classifier is deterministic and cheap enough to run on every turn.
It is also the fallback used when the LLM classifier is unavailable
(no API key, no credits, network error).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from agent.awos.classifiers.base import Classification
from agent.awos.events.schema import TriggerCategory


@dataclass(frozen=True)
class KeywordRule:
    pattern: re.Pattern[str]
    weight: float


# --- category -> keyword rules ------------------------------------------
# Patterns are deliberately short and high-precision. The goal is to be a
# strong signal when a principle is explicitly stated, not to catch every
# implicit mention.
_RULES: dict[TriggerCategory, list[KeywordRule]] = {
    TriggerCategory.ARCHITECTURAL: [
        KeywordRule(re.compile(r"\barchitectur(e|al|ally)\b", re.I), 1.0),
        KeywordRule(re.compile(r"\bdesign decision\b", re.I), 1.0),
        KeywordRule(re.compile(r"\badr\b", re.I), 0.7),
        KeywordRule(re.compile(r"\blayer\s+\d+\b", re.I), 0.5),
        KeywordRule(re.compile(r"\bboundary\b|\bownership\b", re.I), 0.3),
        KeywordRule(re.compile(r"\b(separate|split)\s+(layers|concerns)\b", re.I), 0.8),
    ],
    TriggerCategory.WORKFLOW_PATTERN: [
        KeywordRule(re.compile(r"\bworkflow\b", re.I), 0.7),
        KeywordRule(re.compile(r"\bpattern\b", re.I), 0.4),
        KeywordRule(re.compile(r"\bwe should always\b", re.I), 1.0),
        KeywordRule(re.compile(r"\bwe should never\b", re.I), 1.0),
        KeywordRule(re.compile(r"\bnever\s+(commit|push|delete|force)\b", re.I), 1.0),
        KeywordRule(re.compile(r"\brule[:\s]", re.I), 0.5),
        KeywordRule(re.compile(r"\bconvention\b", re.I), 0.6),
        KeywordRule(re.compile(r"\bstanding\s+instruction\b", re.I), 1.2),
        KeywordRule(re.compile(r"\bprotocol\b", re.I), 0.5),
        KeywordRule(re.compile(r"\bcheckpoint\b", re.I), 0.3),
    ],
    TriggerCategory.LESSON: [
        KeywordRule(re.compile(r"\blesson(\s+learned)?\b", re.I), 1.2),
        KeywordRule(re.compile(r"\bpost[- ]mortem\b", re.I), 1.0),
        KeywordRule(re.compile(r"\bwhat went wrong\b", re.I), 1.0),
        KeywordRule(re.compile(r"\bnext time\b", re.I), 0.8),
        KeywordRule(re.compile(r"\bshould have\b", re.I), 0.5),
    ],
    TriggerCategory.ROADMAP_SHIFT: [
        KeywordRule(re.compile(r"\broadmap\b", re.I), 1.0),
        KeywordRule(re.compile(r"\breorder|resequence|reschedul\w+\b", re.I), 0.6),
        KeywordRule(re.compile(r"\b(defer|delay|bring forward)\s+phase", re.I), 1.0),
        KeywordRule(re.compile(r"\bsupersed(e|es|ed)\b", re.I), 0.6),
        KeywordRule(re.compile(r"\bnext phase\b", re.I), 0.4),
    ],
    TriggerCategory.DECISION: [
        KeywordRule(re.compile(r"\bdecided\b|\bdecision\b", re.I), 0.7),
        KeywordRule(re.compile(r"\bapprove(d)?\b|\blets go\b", re.I), 0.5),
        KeywordRule(re.compile(r"\bwe'll go with\b", re.I), 0.9),
        KeywordRule(re.compile(r"\block(ed)?\s+in\b", re.I), 0.5),
    ],
    TriggerCategory.ROUTINE: [
        KeywordRule(re.compile(r"\btypo\b", re.I), 1.5),
        KeywordRule(re.compile(r"\bfix\s+(import|lint|format)\b", re.I), 1.0),
        KeywordRule(re.compile(r"\brun\s+tests\b", re.I), 0.7),
    ],
}


# self-written AWOS updates carry a marker so the classifier never
# re-classifies its own output
_SELF_MARKER = "<!-- awos:self -->"


class HeuristicClassifier:
    """Fast, deterministic, free keyword-weighted classifier."""

    name = "heuristic"

    def __init__(self, confidence_ceiling: float = 0.7) -> None:
        self.ceiling = float(confidence_ceiling)

    # ------------------------------------------------------------------
    def classify(
        self, text: str, context: dict | None = None
    ) -> Classification:
        if not text or _SELF_MARKER in text:
            return Classification(
                category=TriggerCategory.ROUTINE,
                confidence=0.0,
                rationale="empty or self-written",
                classifier=self.name,
            )

        scores: dict[TriggerCategory, float] = {}
        matched: dict[TriggerCategory, list[str]] = {}
        for cat, rules in _RULES.items():
            hits = 0.0
            matches: list[str] = []
            for rule in rules:
                found = rule.pattern.findall(text)
                if found:
                    hits += rule.weight * min(len(found), 3)
                    matches.append(rule.pattern.pattern)
            if hits > 0:
                scores[cat] = hits
                matched[cat] = matches

        if not scores:
            return Classification(
                category=TriggerCategory.UNKNOWN,
                confidence=0.2,
                rationale="no keyword hits",
                classifier=self.name,
            )

        # top category + score normalization
        top_cat, top_score = max(scores.items(), key=lambda kv: kv[1])
        # density normalization: saturate around 2.0 of weighted hits
        density = min(top_score / 2.0, 1.0)
        confidence = self.ceiling * density

        return Classification(
            category=top_cat,
            confidence=round(confidence, 3),
            rationale=f"matched: {', '.join(matched[top_cat])}",
            classifier=self.name,
            suggested_section=_suggest_section(top_cat),
        )


def _suggest_section(cat: TriggerCategory) -> str:
    # mapping from category to AWOS section header hints
    return {
        TriggerCategory.ARCHITECTURAL: "5. Codebase Structure",
        TriggerCategory.WORKFLOW_PATTERN: "3. Agent Operating Principles",
        TriggerCategory.LESSON: "Lessons",
        TriggerCategory.ROADMAP_SHIFT: "Roadmap Notes",
        TriggerCategory.DECISION: "Decisions",
        TriggerCategory.ROUTINE: "Changelog",
        TriggerCategory.UNKNOWN: "Changelog",
    }.get(cat, "Changelog")


__all__ = ["HeuristicClassifier"]


def _iter_patterns() -> Iterable[str]:  # pragma: no cover - introspection
    for rules in _RULES.values():
        for r in rules:
            yield r.pattern.pattern
