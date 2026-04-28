"""Classifier protocol + shared result model."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from agent.awos.events.schema import TriggerCategory


class Classification(BaseModel):
    """Result of classifying a chunk of conversation text."""

    category: TriggerCategory
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    extracted_principle: str | None = None
    suggested_section: str | None = None
    classifier: str = "unknown"

    @property
    def is_actionable(self) -> bool:
        """True when this classification is worth dispatching to policies."""
        return self.category not in {
            TriggerCategory.ROUTINE,
            TriggerCategory.UNKNOWN,
        } and self.confidence > 0.0


class Classifier(Protocol):
    name: str

    def classify(
        self, text: str, context: dict | None = None
    ) -> Classification: ...


__all__ = ["Classification", "Classifier"]
