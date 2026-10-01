"""TirraMind — ghost chain discovery (automatic candidate generation and its publication).

A "ghost chain" is a multi-hop path through the entity graph carrying a lag and
independent time-varying evidence. Until now every published chain came from a
hand-written YAML template in ``templates/ghost_chains/mp1/``: a human found the
pattern and the system executed it. The automatic scorer
(``agent/models/gnn/pattern_extractor.py``) ranked meta-paths by
``mean_attention * log(frequency)`` from an HGT that was never trained, so the
attention came back as uniform 1/k fractions and the ranking was frequency
wearing a costume.

This package is the replacement, and ``agent.ghost.publish`` is the part that
decides what the world is allowed to see. Its single organising rule:

    the candidate count is the product.

A chain finder that reports survivors without the denominator is a p-hacking
machine. ``publish.Scoreboard`` therefore refuses to exist unless
``enumerated == untestable + tested``, and ``render_scoreboard`` leads with that
arithmetic before it prints a single survivor.
"""

from __future__ import annotations

from agent.ghost.enumerate import (
    DEFAULT_LAG_WINDOWS_DAYS,
    DEFAULT_MAX_SOURCE_STALENESS_DAYS,
    DEFAULT_MIN_SOURCE_OBS_PER_ENTITY,
    DEFAULT_MIN_TARGET_PRICE_OBS,
    CandidateList,
    CandidateSpaceTooLargeError,
    ChainCandidate,
    EmptyUniverseError,
    FilterLedger,
    FilterStep,
    GhostEnumerationError,
    ObservationChannel,
    PriceTarget,
    Universe,
    chain_id,
    count_candidate_space,
    describe_candidate_space,
    enumerate_chains,
    iter_chains,
    load_universe,
    variation_report,
)

# Additive on purpose: `agent.ghost.publish` and `agent.ghost.score` are written
# by sibling modules that also append to this file, and a bare re-assignment here
# would silently drop their exports.
__all__ = [
    *globals().get("__all__", ()),
    "DEFAULT_LAG_WINDOWS_DAYS",
    "DEFAULT_MAX_SOURCE_STALENESS_DAYS",
    "DEFAULT_MIN_SOURCE_OBS_PER_ENTITY",
    "DEFAULT_MIN_TARGET_PRICE_OBS",
    "CandidateList",
    "CandidateSpaceTooLargeError",
    "ChainCandidate",
    "EmptyUniverseError",
    "FilterLedger",
    "FilterStep",
    "GhostEnumerationError",
    "ObservationChannel",
    "PriceTarget",
    "Universe",
    "chain_id",
    "count_candidate_space",
    "describe_candidate_space",
    "enumerate_chains",
    "iter_chains",
    "load_universe",
    "variation_report",
]
