"""Event schema for the AWOS event bus."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone; UTC = timezone.utc
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TriggerCategory(str, Enum):
    """What kind of thing happened — drives policy matching."""

    ARCHITECTURAL = "architectural"
    WORKFLOW_PATTERN = "workflow_pattern"
    LESSON = "lesson"
    ROADMAP_SHIFT = "roadmap_shift"
    DECISION = "decision"
    DRIFT = "drift"
    STALENESS = "staleness"
    ROUTINE = "routine"
    UNKNOWN = "unknown"


class EventStatus(str, Enum):
    """Where the event is in its lifecycle."""

    NEW = "new"
    PROCESSED = "processed"
    IGNORED = "ignored"
    ERRORED = "errored"


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_uuid() -> str:
    return str(uuid.uuid4())


class Event(BaseModel):
    """An atomic, append-only record of something that happened."""

    id: str = Field(default_factory=_new_uuid)
    ts: datetime = Field(default_factory=_utcnow)
    source: str
    category: TriggerCategory
    confidence: float = Field(ge=0.0, le=1.0)
    status: EventStatus = EventStatus.NEW
    payload: dict[str, Any] = Field(default_factory=dict)
    rationale: str | None = None
    parent_event_id: str | None = None
    dedup_hash: str | None = None
    payload_truncated: bool = False


__all__ = ["Event", "EventStatus", "TriggerCategory"]
