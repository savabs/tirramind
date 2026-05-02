"""Evidence protocol — the universal data format for convergence detection.

Every tool's raw output is converted into Evidence objects via extractors.
The EvidenceBus collects evidence for a single detection cycle.

Evidence is frozen (immutable) to prevent accidental mutation after creation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from agent.convergence.taxonomy import CATEGORIES

log = logging.getLogger(__name__)

# Re-export for backward compatibility — taxonomy.py is the canonical owner
VALID_CATEGORIES: frozenset[str] = CATEGORIES


@dataclass(frozen=True)
class Evidence:
    """A single normalized observation from any data tool.

    Parameters
    ----------
    source : str
        Tool name (e.g., "cftc", "weather_alerts").
    signal_id : str
        Unique signal identifier (e.g., "cftc.crude_oil.mm_net_long").
    timestamp : float
        Unix epoch of the observation.
    value : float
        Numeric value. NaN is valid for categorical-encoded signals.
    direction : int
        +1 = stress/expansion/risk-on, -1 = relief/contraction/risk-off,
        0 = neutral.
    confidence : float
        Source-quality weight in [0.0, 1.0].
    category : str
        Taxonomy bucket (must be in VALID_CATEGORIES).
    tags : tuple[str, ...]
        Immutable metadata labels (country, sector, entity, etc.).
    ttl : int
        Seconds until this evidence is considered stale.
    """

    source: str
    signal_id: str
    timestamp: float
    value: float
    direction: int
    confidence: float
    category: str
    tags: tuple[str, ...]
    ttl: int

    def __post_init__(self) -> None:
        if not self.signal_id:
            raise ValueError("signal_id must be non-empty")
        if self.timestamp <= 0:
            raise ValueError(f"timestamp must be positive, got {self.timestamp}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
        if self.direction not in (-1, 0, 1):
            raise ValueError(f"direction must be -1, 0, or 1, got {self.direction}")
        if self.category not in VALID_CATEGORIES:
            raise ValueError(f"category must be one of {sorted(VALID_CATEGORIES)}, got {self.category!r}")
        if self.ttl <= 0:
            raise ValueError(f"ttl must be positive, got {self.ttl}")


class EvidenceBus:
    """Collects Evidence objects for a single detection cycle.

    Thread-safe is NOT required — the pipeline runs single-threaded per DAG node.
    """

    def __init__(self) -> None:
        self._items: list[Evidence] = []

    def submit(self, evidence: Evidence) -> None:
        """Validate type and append evidence to the bus."""
        if not isinstance(evidence, Evidence):
            raise TypeError(f"Expected Evidence, got {type(evidence).__name__}")
        self._items.append(evidence)

    def flush(self) -> list[Evidence]:
        """Return all collected evidence and clear the bus."""
        result = list(self._items)
        self._items.clear()
        return result

    def snapshot(self) -> list[Evidence]:
        """Return a copy of collected evidence without clearing."""
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return len(self._items) > 0

    def __repr__(self) -> str:
        return f"EvidenceBus({len(self._items)} items)"
