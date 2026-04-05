"""TirraMind — Convergence Detection Layer (Phase 7c).

Detects when normally-uncorrelated signals from 60+ data tools begin moving
together — the mathematical signature of a hidden cause propagating through
observable reality.

Layer placement: L2 (Feature Engineering) / L3 (World Model boundary).
No LLM calls. Pure math and rules.
"""

from __future__ import annotations

from agent.convergence.evidence import Evidence, EvidenceBus
from agent.convergence.extractors import extract_evidence, registered_tools
from agent.convergence.taxonomy import CATEGORIES, SignalMeta, SignalRegistry

__all__ = [
    "Evidence",
    "EvidenceBus",
    "CATEGORIES",
    "SignalMeta",
    "SignalRegistry",
    "extract_evidence",
    "registered_tools",
]
