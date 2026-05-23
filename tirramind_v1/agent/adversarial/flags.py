"""TirraMind — AdversarialFlag output contract

Frozen dataclass emitted by every adversarial detector.  Consumed by
the reward function (penalty term), the state assembler (feature block),
and the pipeline DAG (logging / alerting).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AdversarialFlag:
    """Single adversarial detection event.

    Attributes
    ----------
    entity_id : target entity, or ``None`` for market-wide flags.
    flag_type : one of ``"edge_decay"``, ``"vpin_spike"``, ``"crowding_risk"``.
    severity : normalised intensity in [0, 1].
    confidence : posterior confidence in [0, 1].
    signal_name : name of the affected signal (optional).
    evidence : free-form supporting metrics dict.
    timestamp : unix epoch of flag creation.
    """

    flag_type: str
    severity: float
    confidence: float
    entity_id: str | None = None
    signal_name: str | None = None
    evidence: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    _VALID_TYPES = frozenset({"edge_decay", "vpin_spike", "crowding_risk"})

    def __post_init__(self) -> None:
        if self.flag_type not in self._VALID_TYPES:
            raise ValueError(f"Invalid flag_type {self.flag_type!r}; must be one of {sorted(self._VALID_TYPES)}")
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(f"severity must be in [0, 1], got {self.severity}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
