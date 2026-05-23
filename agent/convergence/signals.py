"""Convergence signal emission and pipeline store integration.

Converts :class:`~agent.convergence.detector.DetectionResult` objects
into :class:`ConvergenceSignal` records and persists them to the
pipeline store's ``signals`` table.

This module is deterministic and LLM-free (Pipeline Layer contract).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone; UTC = timezone.utc
from typing import Any

from agent.pipeline.store import PipelineStore

log = logging.getLogger(__name__)

# Allowed characters in event_type / signal_name components.
_SAFE_RE = re.compile(r"^[a-zA-Z0-9_]+$")


# ── Signal name formatting ─────────────────────────────────────


def format_signal_name(event_type: str, date: str) -> str:
    """Build a canonical convergence signal name.

    Parameters
    ----------
    event_type : str
        Template name or ``"unknown_pattern"``.
        Must be alphanumeric + underscore only.
    date : str
        ISO date string (``YYYY-MM-DD``).
        Must be alphanumeric + hyphen only.

    Returns
    -------
    str
        ``"convergence.<event_type>.<date>"``

    Raises
    ------
    ValueError
        If *event_type* contains unsafe characters.
    """
    if not _SAFE_RE.match(event_type):
        raise ValueError(f"event_type must be alphanumeric + underscore, got {event_type!r}")
    # Date allows hyphens
    if not re.match(r"^[a-zA-Z0-9_-]+$", date):
        raise ValueError(f"date contains unsafe characters: {date!r}")
    return f"convergence.{event_type}.{date}"


# ── ConvergenceSignal ──────────────────────────────────────────


@dataclass
class ConvergenceSignal:
    """A convergence event ready for storage in the pipeline signals table.

    Attributes
    ----------
    signal_name : str
        Canonical name: ``"convergence.<event_type>.<iso_date>"``.
    computed_at : float
        Unix timestamp of detection.
    value : float
        Convergence score in [0, 1].
    event_type : str
        Template name or ``"unknown_pattern"``.
    signals_involved : list[str]
        Signal IDs in the convergence clique.
    categories_involved : list[str]
        Distinct taxonomy categories.
    cross_category_count : int
        Number of distinct categories.
    p_value : float
        Fisher's combined p-value for the clique.
    persistence_days : int
        Consecutive detection cycles this pattern has been active.
    template_match : float
        Template match score in [0, 1].
    direction : int
        +1 stress/expansion, -1 relief/contraction.
    lead_signal : str
        Earliest-activating signal in the clique.
    lag_signals : list[str]
        Later-activating signals.
    """

    signal_name: str
    computed_at: float
    value: float
    event_type: str
    signals_involved: list[str]
    categories_involved: list[str]
    cross_category_count: int
    p_value: float
    persistence_days: int
    template_match: float
    direction: int
    lead_signal: str
    lag_signals: list[str] = field(default_factory=list)

    def to_metadata_dict(self) -> dict[str, Any]:
        """Serialize non-core fields into a JSON-safe dict.

        The core fields (``signal_name``, ``computed_at``, ``value``)
        are stored in their own columns; everything else goes here.
        """
        return {
            "event_type": self.event_type,
            "signals_involved": self.signals_involved,
            "categories_involved": self.categories_involved,
            "cross_category_count": self.cross_category_count,
            "p_value": self.p_value,
            "persistence_days": self.persistence_days,
            "template_match": self.template_match,
            "direction": self.direction,
            "lead_signal": self.lead_signal,
            "lag_signals": self.lag_signals,
        }


# ── Factory from DetectionResult ───────────────────────────────


def from_detection_result(
    result: Any,  # DetectionResult — Any to avoid circular import
    persistence_count: int = 0,
    as_of: float | None = None,
) -> ConvergenceSignal:
    """Convert a :class:`DetectionResult` into a :class:`ConvergenceSignal`.

    Parameters
    ----------
    result : DetectionResult
        Output from :meth:`ConvergenceDetector.detect`.
    persistence_count : int
        How many consecutive cycles this clique has been detected.
    as_of : float | None
        Detection timestamp.  Defaults to now.

    Returns
    -------
    ConvergenceSignal
    """
    if as_of is None:
        as_of = time.time()

    iso_date = datetime.fromtimestamp(as_of, tz=UTC).strftime("%Y-%m-%d")
    signal_name = format_signal_name(result.event_type, iso_date)

    clique = result.clique

    # Direction from pairwise coincidence score aggregation
    direction = getattr(result, "direction", 1)

    # Fisher's combined p-value (attached by FDR layer)
    p_value = getattr(clique, "p_values_combined", 1.0)
    if p_value is None:
        p_value = 1.0

    return ConvergenceSignal(
        signal_name=signal_name,
        computed_at=as_of,
        value=result.boosted_score,
        event_type=result.event_type,
        signals_involved=list(clique.signals),
        categories_involved=list(clique.categories),
        cross_category_count=len(clique.categories),
        p_value=float(p_value),
        persistence_days=persistence_count,
        template_match=result.template_match,
        direction=direction,
        lead_signal=result.lead_signal or "",
        lag_signals=list(result.lag_signals) if result.lag_signals else [],
    )


# ── Emission ───────────────────────────────────────────────────


def emit_signals(
    signals: list[ConvergenceSignal],
    store: PipelineStore,
) -> int:
    """Persist convergence signals to the pipeline store.

    Parameters
    ----------
    signals : list[ConvergenceSignal]
        Signals to store.
    store : PipelineStore
        Target store.

    Returns
    -------
    int
        Number of signals successfully stored.
    """
    count = 0
    for sig in signals:
        try:
            store.store_signal(
                signal_name=sig.signal_name,
                value=sig.value,
                metadata=sig.to_metadata_dict(),
            )
            count += 1
        except Exception:
            log.warning("Failed to store signal %s", sig.signal_name, exc_info=True)
    if count:
        log.info("Emitted %d convergence signal(s).", count)
    return count
