"""Composite classifier: heuristic + LLM, with mode selection.

Modes
-----
``off``       — returns UNKNOWN, 0.0 confidence (classifier disabled).
``heuristic`` — heuristic only; free, deterministic, lower precision.
``llm``       — Anthropic only; silently falls back to heuristic if the
                API is unavailable. Always preserves safety.
``hybrid``    — run heuristic first; if it returns UNKNOWN or confidence
                below ``llm_confidence_floor``, escalate to the LLM.
                Otherwise keep the heuristic result. This minimises spend
                while still confirming the high-stakes cases.
"""

from __future__ import annotations

from agent.awos.classifiers.anthropic import AnthropicClassifier
from agent.awos.classifiers.base import Classification
from agent.awos.classifiers.heuristic import HeuristicClassifier
from agent.awos.config import AWOSConfig, ClassifierMode
from agent.awos.events.schema import TriggerCategory


class CompositeClassifier:
    name = "composite"

    def __init__(
        self,
        *,
        mode: ClassifierMode,
        heuristic: HeuristicClassifier,
        llm: AnthropicClassifier,
        llm_floor: float,
    ) -> None:
        self.mode = mode
        self.heuristic = heuristic
        self.llm = llm
        self.llm_floor = float(llm_floor)

    @classmethod
    def from_config(cls, cfg: AWOSConfig) -> CompositeClassifier:
        return cls(
            mode=cfg.classifier_mode,
            heuristic=HeuristicClassifier(cfg.heuristic_confidence_ceiling),
            llm=AnthropicClassifier(
                api_key=cfg.anthropic_api_key,
                model=cfg.anthropic_model,
                max_tokens=cfg.anthropic_max_tokens,
                timeout_s=cfg.anthropic_timeout_s,
            ),
            llm_floor=cfg.llm_confidence_floor,
        )

    # ------------------------------------------------------------------
    def classify(self, text: str, context: dict | None = None) -> Classification:
        if self.mode == "off":
            return Classification(
                category=TriggerCategory.UNKNOWN,
                confidence=0.0,
                rationale="classifier mode=off",
                classifier="off",
            )

        if self.mode == "heuristic":
            return self.heuristic.classify(text, context)

        if self.mode == "llm":
            result = self.llm.classify(text, context)
            if result.confidence == 0.0:
                return self.heuristic.classify(text, context)
            return result

        # hybrid: heuristic first, escalate only when uncertain
        h = self.heuristic.classify(text, context)
        if h.category not in (TriggerCategory.UNKNOWN,) and h.confidence >= self.llm_floor:
            return h
        llm = self.llm.classify(text, context)
        if llm.confidence == 0.0:
            return h  # LLM unavailable → keep heuristic
        return llm if llm.confidence >= h.confidence else h


__all__ = ["CompositeClassifier"]
