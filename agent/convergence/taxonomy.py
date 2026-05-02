"""Signal taxonomy and metadata registry for convergence detection.

This module is the single source of truth for:
- The 11 taxonomy categories that classify every evidence signal
- Valid observation frequencies
- Per-signal metadata (SignalMeta) describing expected behavior
- The SignalRegistry that catalogues all known signals

Other modules (evidence.py, extractors.py, etc.) import CATEGORIES from here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

# ── Taxonomy categories ────────────────────────────────────────
#
# Every piece of evidence belongs to exactly one category.
# Cross-category coincidence is more informative than within-category.

CATEGORIES: frozenset[str] = frozenset(
    {
        "physical_flow",  # AIS vessels, transport throughput, energy supply
        "physical_disruption",  # Weather, earthquake, internet outage
        "financial_stress",  # Sovereign debt, creditor filings, bankruptcy, DeFi
        "monetary_policy",  # Central bank balance, rate monitor, capital flows
        "regulatory_action",  # Sanctions, drug regulatory, gazette, FOIA
        "behavioral_intent",  # Patents, lobbying, job postings, Wikipedia, certs
        "positioning",  # CFTC, FINRA short, Polymarket whales, insider
        "macro_momentum",  # PMI, consumer sentiment, building permits, receipts
        "biological",  # Disease surveillance, food security
        "geopolitical",  # Political risk, GDELT, migration flows
        "supply_chain",  # Supply chain monitor, interconnection queue, gov contracts
    }
)

# ── Valid frequencies ──────────────────────────────────────────

VALID_FREQUENCIES: frozenset[str] = frozenset(
    {
        "intraday",  # Hourly buckets (electricity, weather, internet)
        "daily",  # Daily close (treasury receipts, most pipeline runs)
        "weekly",  # Weekly (CFTC, FINRA, energy supply)
        "monthly",  # Monthly (job postings, PMI, building permits)
        "event",  # Irregular/event-driven (earthquake, sanctions)
    }
)


# ── SignalMeta ─────────────────────────────────────────────────


@dataclass(frozen=True)
class SignalMeta:
    """Metadata describing a single signal stream's expected behavior.

    Parameters
    ----------
    signal_id : str
        Unique identifier matching Evidence.signal_id
        (e.g., "cftc.crude_oil.mm_net_long").
    source : str
        Tool name that produces this signal.
    category : str
        Taxonomy category (must be in CATEGORIES).
    frequency : str
        Expected update cadence (must be in VALID_FREQUENCIES).
    direction_semantics : str
        Human description of what "up" means
        (e.g., "higher = more speculative longs").
    flip_sign : bool
        True if the raw value convention is opposite ours.
        When True, the value is negated before z-scoring so that
        positive always means stress/expansion.
    default_ttl : int
        Seconds before an observation is considered stale.
    min_observations : int
        Minimum history needed for meaningful z-scores.
    """

    signal_id: str
    source: str
    category: str
    frequency: str
    direction_semantics: str
    flip_sign: bool = False
    default_ttl: int = 86_400  # 24 h
    min_observations: int = 30

    def __post_init__(self) -> None:
        if not self.signal_id:
            raise ValueError("signal_id must be non-empty")
        if not self.source:
            raise ValueError("source must be non-empty")
        if self.category not in CATEGORIES:
            raise ValueError(f"category must be one of {sorted(CATEGORIES)}, got {self.category!r}")
        if self.frequency not in VALID_FREQUENCIES:
            raise ValueError(f"frequency must be one of {sorted(VALID_FREQUENCIES)}, got {self.frequency!r}")
        if self.default_ttl <= 0:
            raise ValueError(f"default_ttl must be positive, got {self.default_ttl}")
        if self.min_observations < 1:
            raise ValueError(f"min_observations must be >= 1, got {self.min_observations}")


# ── SignalRegistry ─────────────────────────────────────────────


class SignalRegistry:
    """In-memory catalogue of all known signal streams.

    Provides efficient lookup by signal_id, source, and category.
    Used by the ConvergenceDetector to know what signals exist and
    how to interpret them.
    """

    def __init__(self) -> None:
        self._by_id: dict[str, SignalMeta] = {}
        self._by_source: dict[str, list[SignalMeta]] = {}
        self._by_category: dict[str, list[SignalMeta]] = {}

    # ── Mutation ───────────────────────────────────────────────

    def register(self, meta: SignalMeta) -> None:
        """Add a signal to the registry. Raises on duplicate signal_id."""
        if not isinstance(meta, SignalMeta):
            raise TypeError(f"Expected SignalMeta, got {type(meta).__name__}")
        if meta.signal_id in self._by_id:
            raise ValueError(f"Duplicate signal_id: {meta.signal_id!r} already registered")
        self._by_id[meta.signal_id] = meta
        self._by_source.setdefault(meta.source, []).append(meta)
        self._by_category.setdefault(meta.category, []).append(meta)

    # ── Lookup ─────────────────────────────────────────────────

    def get(self, signal_id: str) -> SignalMeta | None:
        """Return metadata for a signal, or None if unknown."""
        return self._by_id.get(signal_id)

    def by_source(self, source: str) -> list[SignalMeta]:
        """Return all signals from a given tool. Empty list if none."""
        return list(self._by_source.get(source, []))

    def by_category(self, category: str) -> list[SignalMeta]:
        """Return all signals in a taxonomy category. Empty list if none."""
        return list(self._by_category.get(category, []))

    def all_ids(self) -> list[str]:
        """Return all registered signal_ids (sorted for determinism)."""
        return sorted(self._by_id.keys())

    def frequencies(self) -> dict[str, list[str]]:
        """Group signal_ids by their frequency.

        Returns dict like {"daily": ["sig_a", "sig_b"], "weekly": ["sig_c"]}.
        """
        groups: dict[str, list[str]] = {}
        for sid, meta in sorted(self._by_id.items()):
            groups.setdefault(meta.frequency, []).append(sid)
        return groups

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, signal_id: str) -> bool:
        return signal_id in self._by_id

    def __repr__(self) -> str:
        return f"SignalRegistry({len(self._by_id)} signals)"
