"""TirraMind — ghost chain candidate enumeration, and the denominator it owes you.

A *ghost chain* is a falsifiable claim of the shape

    "<source entity>'s <observation type> moves, and <target instrument>'s price
     responds <lag window> later, through <this path> in the entity graph,
     witnessed by <these independent time-varying sources>."

This module generates those claims. It does NOT test them; it is the thing that
decides how many tests the next stage is about to run, which is the number that
governs whether any survivor means anything.

WHY THIS MODULE EXISTS AT ALL
-----------------------------
Three chain briefs were published in June 2026 and all three came from
hand-written YAML in ``templates/ghost_chains/mp1/``. A human found the pattern;
the system executed it. The automatic finder
(``agent/models/gnn/pattern_extractor.py``) ranked meta-paths by
``mean_attention * log(frequency)`` read off an HGT that was never trained, so
``mean_attention`` came back as uniform ``1/k`` fractions — 0.0, 0.333, 0.5,
0.667, 1.0, with 9 of 20 edge types at exactly 0.0. It was ranking by frequency
in a costume, and it never produced a candidate a human had not already written
down.

The substrate here is different: ``agent.mechanism.routes.top_routes`` is
personalised-PageRank path enumeration over the real link table, with zero
learned parameters and therefore nothing to leave untrained. This module adds no
scoring of its own. It enumerates and it counts.

THE DANGER THIS MODULE IS SHAPED AROUND
---------------------------------------
With 12 node types, 18 link types and a lag grid, a chain finder can manufacture
hundreds of thousands of candidates, so thousands will clear an uncorrected
p-value threshold by construction. Measured precedents in this repository:

  * the CFTC study: 0 of 51 hypotheses survived Benjamini-Hochberg. 9 of 51
    would have "worked" uncorrected.
  * re-run on more data, 3 of 54 "survived" — at 5.1-7.6% power, one with n=3.
  * "12 positions, 3 bets" occurred in 0 of 200 real portfolios.

Every headline this project has produced died when it was measured properly.
So: ``count_candidate_space`` exists to make the denominator knowable BEFORE
anything is generated, ``CandidateList.space`` carries the realised denominator
afterwards, and every filter reports a count. "3 chains found" is not a result
this module can produce. "3 of 40,000 tested" is.

WHERE THIS MODULE MISLEADS — read before quoting anything it returns
--------------------------------------------------------------------
  * A candidate is a HYPOTHESIS, not a finding. Nothing here has been tested
    against a price series. ``len(enumerate_chains(...))`` is a count of
    unfalsified guesses.
  * The candidate count is a function of the options. ``k``, ``max_hops`` and
    ``lag_windows`` each multiply it. Two runs with different options have
    different denominators and their survivors are NOT comparable.
  * ``k`` truncates per pair. Every candidate list is therefore a subset of the
    graph's chains, and ``CandidateList.space["pairs_truncated"]`` says how many
    pairs had more routes than were taken. A BH correction over the returned
    candidates is a correction over what was looked at, which is the right
    denominator for *this* run and an understatement of the search space that
    generated the option choices.
  * ``independent_sources`` counts distinct ingest pipelines on time-varying
    hops. Two pipelines fed by one upstream publisher count as two. It is an
    upper bound, as ``agent.mechanism.routes.independence`` documents at length.
  * Candidates are not independent tests of independent things. Four lag windows
    over one (path, target) share a price series; many paths share a target.
    Benjamini-Hochberg assumes nothing about independence, but the effective
    number of distinct claims is far below the candidate count, so a BH-adjusted
    survivor is still a single un-replicated observation.
  * A path's existence is not a mechanism. ``produced_in`` is static geography:
    the single highest-PageRank Russia -> WTI route is a DIRECT ``produced_in``
    edge with zero evidential weight. ``require_evidence_hop=True`` (the
    default) drops those, and the count is reported.

READ-ONLY. This module opens the pipeline DB with ``mode=ro`` and contains no
INSERT, UPDATE or DELETE. It never calls a collector, a trainer or a backfill.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

from agent.mechanism.routes import (
    DEFAULT_DB_PATH,
    Route,
    RouteError,
    independence,
    link_kind,
    load_route_graph,
    top_routes,
)

__all__ = [
    "DEFAULT_LAG_WINDOWS_DAYS",
    "DEFAULT_MAX_SOURCE_STALENESS_DAYS",
    "DEFAULT_MIN_SOURCE_OBS_PER_ENTITY",
    "DEFAULT_MIN_TARGET_PRICE_OBS",
    "PRICE_OBSERVATION_TYPE",
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
    "witness_groups",
]

# ── Defaults, every one of them a judgement call ───────────────────────────

PRICE_OBSERVATION_TYPE = "instrument_daily"
"""The observation_type that constitutes a testable price series.

Measured on the live DB: written by ``instrument_universe`` for 89 entities,
840-1201 rows each. The ``entities`` table holds 93 rows of entity_type
``instrument``, so 4 "instruments" have no price series at all and cannot be the
target of any testable chain.
"""

DEFAULT_MIN_TARGET_PRICE_OBS = 250
"""Minimum daily price observations for a target to be testable.

~250 is one trading year. Below it, a block bootstrap over non-overlapping
blocks has too few blocks for the tail of the null to be estimable — the repo's
own precedent is a "survivor" with n=3, where an i.i.d. bootstrap over 3 points
takes at most 27 distinct values and a p-value below 1/27 is arithmetically
impossible.

Misleads: on the live DB this filter currently removes NOTHING, because every
one of the 89 priced instruments has >= 840 observations. It is a guard against
a future thin instrument, not an active constraint today, and reporting it as
"89 of 89 passed" is the honest way to say so.
"""

DEFAULT_MIN_SOURCE_OBS_PER_ENTITY = 20
"""Minimum observations of one (entity, source_tool, observation_type) channel.

A chain's source term is a z-score of that channel. Below ~20 points the
z-score's own denominator is noise: the standard error of a sample sd at n=20 is
about 16% of the sd, and it grows as 1/sqrt(2n) below that.

Misleads: 20 is a floor on *estimability*, not on *power*. A channel with 21
observations passes this filter and will still fail any honest power
calculation. This filter's job is to stop untestable chains being counted in the
denominator; it is not a quality bar.
"""

DEFAULT_MAX_SOURCE_STALENESS_DAYS = 90.0
"""How long after a channel's last observation it stops being a live source.

Measured on the live DB (data horizon 2026-09-23): ``ais_vessel``'s
``vessel_position`` is 106 days stale and averages 3.1 observations per entity,
and ``baltic_activity_proxy`` is 125 days stale. Chains routed through vessel
positioning CANNOT be tested on current data. They are excluded here, by name
and with a count, rather than silently producing candidates that no downstream
stage can evaluate.

Misleads: staleness is measured per (entity, source_tool, observation_type)
channel against the observation cutoff, so a live source tool with one abandoned
entity loses only that entity. The rollup in
``Universe.excluded_channel_kinds`` aggregates by source tool so the exclusion
is legible, but the filter itself is per channel.
"""

DEFAULT_LAG_WINDOWS_DAYS: tuple[tuple[int, int], ...] = (
    (1, 5),
    (5, 21),
    (21, 63),
    (63, 126),
)
"""Default lag grid, in CALENDAR days, as half-open ``[low, high)`` windows.

One trading week, month, quarter and half-year. Disjoint on purpose: overlapping
windows would test the same price response several times and inflate the
denominator with near-duplicates, which makes a BH correction look stricter than
it is while testing nothing new.

Misleads:
  * calendar days, not trading days. ``observed_at`` is epoch seconds; a 5-day
    window spanning a weekend contains 3 trading days. The downstream test owns
    the calendar; this module only owns the label.
  * the grid is DECLARED, not learned. ``agent.models.gnn.pattern_extractor
    .extract_temporal_lags`` can estimate an empirical lag distribution per
    meta-path, and ``lag_windows=`` accepts its output — but an empirical grid
    chosen after looking at the data is a fitted parameter, and every candidate
    generated from it inherits that look. A fixed declared grid is the version
    whose denominator is honest.
  * four windows multiply the candidate count by four and the *distinct claims*
    by far less than four.
"""

_MAX_CHAIN_ID_CANDIDATES_SAFE = 10_000_000
"""Above this many candidates the 80-bit chain_id collision risk stops being negligible."""


# ── Errors: loud, and specific about what the caller should change ─────────


class GhostEnumerationError(ValueError):
    """Base class for every loud failure in ghost chain enumeration."""


class EmptyUniverseError(GhostEnumerationError):
    """No testable targets, or no testable source channels, survived the filters.

    Raised rather than returning an empty candidate list, because "zero chains
    exist" and "zero chains are testable with this data" are different facts and
    must not share the value 0.
    """


class CandidateSpaceTooLargeError(GhostEnumerationError):
    """Generation would exceed ``max_candidates``.

    Carries the realised count at the point of refusal and the up-front bound
    from ``count_candidate_space``, so the caller can narrow ``k``,
    ``max_hops`` or ``lag_windows`` instead of receiving a silently sampled
    subset. Sampling would make the denominator unknowable, which is the one
    failure this module exists to prevent.
    """


# ── Value types ───────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class FilterStep:
    """One filter, with the count it removed and why it is defensible.

    ``before == after + dropped`` is enforced by ``FilterLedger``, not merely
    hoped for. A filter that cannot account for its own arithmetic is a filter
    that dropped something silently.
    """

    name: str
    before: int
    after: int
    reason: str
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def dropped(self) -> int:
        """How many units this step removed."""
        return self.before - self.after

    def describe(self) -> str:
        """One line: ``name  before -> after  (-dropped)  reason``."""
        return f"{self.name:32s} {self.before:8d} -> {self.after:8d}  (-{self.dropped})  {self.reason}"


@dataclass(frozen=True, slots=True)
class FilterLedger:
    """An ordered, arithmetically closed account of everything dropped.

    Construction raises ``GhostEnumerationError`` if any step's arithmetic does
    not close, or if consecutive steps do not chain (``step[i].after !=
    step[i+1].before``). That makes "nothing was dropped silently" a checked
    property rather than a claim in a docstring.

    Misleads: a closed ledger proves the counts are consistent. It does not
    prove the filters are *right* — a principled-looking filter with a
    conveniently chosen threshold closes just as neatly.
    """

    unit: str
    steps: tuple[FilterStep, ...]

    def __post_init__(self) -> None:
        for i, step in enumerate(self.steps):
            if step.after > step.before:
                raise GhostEnumerationError(
                    f"filter step {step.name!r} reports after={step.after} > before={step.before}: "
                    "a filter cannot add units."
                )
            if step.after < 0 or step.before < 0:
                raise GhostEnumerationError(f"filter step {step.name!r} has a negative count: {step}")
            if i and self.steps[i - 1].after != step.before:
                raise GhostEnumerationError(
                    f"filter ledger does not chain: {self.steps[i - 1].name!r} ended at "
                    f"{self.steps[i - 1].after} but {step.name!r} starts at {step.before}. "
                    "A gap here means units vanished between filters."
                )

    @property
    def initial(self) -> int:
        """Count before any filter. 0 for an empty ledger."""
        return self.steps[0].before if self.steps else 0

    @property
    def final(self) -> int:
        """Count after every filter. 0 for an empty ledger."""
        return self.steps[-1].after if self.steps else 0

    @property
    def total_dropped(self) -> int:
        """Units removed across all steps."""
        return self.initial - self.final

    def describe(self) -> str:
        """Multi-line rendering, one line per step plus a total."""
        head = f"{self.unit}: {self.initial} -> {self.final} ({self.total_dropped} dropped)"
        return "\n".join([head, *(f"  {s.describe()}" for s in self.steps)])


@dataclass(frozen=True, slots=True)
class PriceTarget:
    """A graph entity with a price series long enough to test a chain against."""

    entity_id: str
    canonical_name: str
    entity_type: str
    n_price_obs: int
    first_observed_at: float
    last_observed_at: float


@dataclass(frozen=True, slots=True)
class ObservationChannel:
    """One (entity, source_tool, observation_type) series usable as a chain's source term.

    ``n_obs`` counts rows at or before the universe's observation cutoff, so the
    same channel has a smaller ``n_obs`` at an earlier ``as_of``. That is the
    point: a channel that is z-scoreable today but was not in 2019 must not
    generate a candidate in a 2019-as-of run (F-04).
    """

    entity_id: str
    canonical_name: str
    entity_type: str
    source_tool: str
    observation_type: str
    n_obs: int
    first_observed_at: float
    last_observed_at: float

    @property
    def kind(self) -> tuple[str, str]:
        """``(source_tool, observation_type)`` — the channel's data-source identity."""
        return (self.source_tool, self.observation_type)


@dataclass(frozen=True, slots=True)
class Universe:
    """The testable source channels and price targets at one observation cutoff.

    ``target_ledger`` and ``channel_ledger`` are arithmetically closed accounts
    of every entity and channel removed. ``excluded_channel_kinds`` is the
    by-source-tool rollup of exclusions: it is how a dead source appears BY NAME
    in a report instead of as an absence.
    """

    db_path: str
    observation_cutoff: float
    as_of: float | None
    targets: tuple[PriceTarget, ...]
    channels: tuple[ObservationChannel, ...]
    target_ledger: FilterLedger
    channel_ledger: FilterLedger
    excluded_channel_kinds: Mapping[tuple[str, str], Mapping[str, Any]]
    observation_rows_total: int
    observation_rows_after_cutoff: int
    options: Mapping[str, Any]

    @property
    def source_entity_ids(self) -> tuple[str, ...]:
        """Distinct source entities, sorted. Fewer than ``len(channels)``.

        An entity with four usable channels is ONE route search and FOUR
        candidates per route. Conflating the two numbers either over-states the
        cost or under-states the denominator.
        """
        return tuple(sorted({c.entity_id for c in self.channels}))

    def channels_by_entity(self) -> dict[str, tuple[ObservationChannel, ...]]:
        """Usable channels grouped by source entity, each group sorted by kind."""
        out: dict[str, list[ObservationChannel]] = {}
        for ch in self.channels:
            out.setdefault(ch.entity_id, []).append(ch)
        return {eid: tuple(sorted(chs, key=lambda c: c.kind)) for eid, chs in out.items()}

    def restrict(
        self,
        *,
        target_ids: Iterable[str] | None = None,
        source_tools: Iterable[str] | None = None,
        observation_types: Iterable[str] | None = None,
        source_entity_types: Iterable[str] | None = None,
        label: str = "caller_restriction",
    ) -> Universe:
        """A narrower universe, with the narrowing APPENDED to both ledgers.

        ``count_candidate_space`` tells a caller whose space is too large to
        narrow the options rather than sample. This is how: the result is a real
        ``Universe`` whose ledgers still close, so the narrowed run can state
        its own denominator AND what it excluded to get there. Narrowing by hand
        (rebuilding the dataclass) would lose that, and a denominator whose
        provenance is lost is not a denominator.

        Every argument left as None means "do not restrict on this". An empty
        iterable restricts to nothing and raises ``EmptyUniverseError``, because
        an empty universe returned quietly would produce a confident zero.

        Misleads: a restricted run's survivors are NOT comparable with the full
        run's. Narrowing changes the denominator, which changes every
        BH-adjusted p-value. Choosing the narrowing after seeing which chains
        looked promising is p-hacking with extra steps, and nothing in this
        module can detect that you did it.
        """
        targets = self.targets
        channels = self.channels
        tsteps = list(self.target_ledger.steps)
        csteps = list(self.channel_ledger.steps)
        if target_ids is not None:
            keep = frozenset(target_ids)
            new = tuple(t for t in targets if t.entity_id in keep)
            tsteps.append(
                FilterStep(f"{label}:target_ids", len(targets), len(new), f"caller kept {len(keep)} named targets")
            )
            targets = new
        for name, values, attr in (
            ("source_tools", source_tools, "source_tool"),
            ("observation_types", observation_types, "observation_type"),
            ("source_entity_types", source_entity_types, "entity_type"),
        ):
            if values is None:
                continue
            keep = frozenset(values)
            new = tuple(c for c in channels if getattr(c, attr) in keep)
            csteps.append(
                FilterStep(f"{label}:{name}", len(channels), len(new), f"caller kept {attr} in {sorted(keep)}")
            )
            channels = new
        out = Universe(
            db_path=self.db_path,
            observation_cutoff=self.observation_cutoff,
            as_of=self.as_of,
            targets=targets,
            channels=channels,
            target_ledger=FilterLedger(self.target_ledger.unit, tuple(tsteps)),
            channel_ledger=FilterLedger(self.channel_ledger.unit, tuple(csteps)),
            excluded_channel_kinds=self.excluded_channel_kinds,
            observation_rows_total=self.observation_rows_total,
            observation_rows_after_cutoff=self.observation_rows_after_cutoff,
            options={**self.options, "restricted_by": label},
        )
        if not out.targets:
            raise EmptyUniverseError(f"restrict({label}) left no price targets:\n{out.target_ledger.describe()}")
        if not out.channels:
            raise EmptyUniverseError(f"restrict({label}) left no source channels:\n{out.channel_ledger.describe()}")
        return out

    def describe(self) -> str:
        """Human rendering of both ledgers and the named exclusions."""
        lines = [
            f"universe from {self.db_path}",
            f"  observation cutoff {_iso(self.observation_cutoff)} (as_of={self.as_of!r})",
            f"  observation rows {self.observation_rows_total} total, "
            f"{self.observation_rows_after_cutoff} dated AFTER the cutoff and excluded",
            self.target_ledger.describe(),
            self.channel_ledger.describe(),
        ]
        if self.excluded_channel_kinds:
            lines.append("  EXCLUDED SOURCE KINDS (named, not silent):")
            for (tool, obs), info in sorted(self.excluded_channel_kinds.items()):
                lines.append(
                    f"    {tool}/{obs}: -{info['channels_excluded']} channels "
                    f"(thin {info['below_min_obs']}, stale {info['stale']}, option {info['excluded_by_option']}); "
                    f"obs/entity {info['obs_per_entity']:.1f}, kind stale {info['staleness_days']:.1f}d; "
                    f"{info['reason']}"
                )
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ChainCandidate:
    """One falsifiable chain hypothesis. NOT a finding, NOT a tested claim.

    Identity is ``chain_id``: a blake2b digest over the source channel, the node
    sequence, the link-type sequence and the lag window. The same hypothesis
    therefore carries the same id across runs, across ``as_of`` values and
    across option changes that do not alter the path — which is what makes
    "this candidate survived in March and died in September" a statement about
    the world rather than about the run.

    ``as_of`` is deliberately NOT part of the id. Two runs over different
    evidence windows testing the same path are the same hypothesis, and being
    able to say that is the whole point of a stable id.

    Field notes, and where each one misleads:
      ``hop_classes``      per hop: "evidence" (time-varying), "scaffold"
          (static geography) or "unclassified" (a link_type nobody has decided
          about). Recomputed by ``agent.mechanism.routes.link_kind`` so one
          definition governs. Unclassified is NOT a middle ground; it is a
          to-do.
      ``independent_sources``  distinct ingest pipelines on this route's
          EVIDENCE hops only. An UPPER bound: two pipelines over one publisher
          count as two. Identical by construction to what
          ``agent.mechanism.routes.independence`` counts over a one-route list.
      ``independent_sources_conservative`` / ``witness_groups``  the LOWER
          bound, from ``witness_groups``: sources that co-write the same
          evidence relation are collapsed into one witness. Quote this number,
          not ``independent_sources``, when claiming corroboration. Measured on
          the live graph, every candidate that passes
          ``min_evidence_sources=2`` collapses to a conservative count of 1.
      ``pair_has_co_dependent_sources``  True when ANY route for this pair had
          two sources on one relation, even if this route did not. It is the
          flag that says "read the pair-level count with suspicion".
      ``pair_independent_sources``  the same count over ALL routes returned for
          this (source, target) pair. Larger than ``independent_sources``
          whenever a sibling route carries a different witness, and it is the
          number to quote for "how many datasets support this connection at
          all" — not for "how many support THIS chain".
      ``mass``  the route's personalised-PageRank contribution. Comparable only
          within one (source, target) query, and a lower bound on the pair's
          total mass because only simple paths are enumerated. A high mass means
          SHORT and LOW-DEGREE, not well-evidenced.
      ``hub_dominated``  the route passes an intermediate node at or above the
          graph's 99th-degree percentile. Such a route connects almost
          everything to almost everything and is weak evidence by construction.
      ``lag_low_days`` / ``lag_high_days``  a half-open calendar-day window, and
          a DECLARED hypothesis rather than a measurement.
      ``source_obs_count`` / ``target_obs_count``  series lengths at the
          universe's cutoff. These bound the achievable power and are the
          numbers that made the repo's earlier "3 of 54 survivors" a 5.1-7.6%
          power result.
    """

    chain_id: str
    source_entity: str
    source_name: str
    source_entity_type: str
    source_tool: str
    observation_type: str
    source_obs_count: int
    source_last_observed_at: float
    target_entity: str
    target_name: str
    target_entity_type: str
    target_obs_count: int
    path: tuple[str, ...]
    path_names: tuple[str, ...]
    path_types: tuple[str, ...]
    hops: int
    link_types: tuple[str, ...]
    hop_sources: tuple[str, ...]
    hop_classes: tuple[str, ...]
    evidence_hops: int
    scaffold_hops: int
    unclassified_hops: int
    independent_sources: int
    independent_sources_conservative: int
    witness_groups: tuple[tuple[str, ...], ...]
    evidence_sources: tuple[str, ...]
    pair_independent_sources: int
    pair_has_co_dependent_sources: bool
    pair_routes_considered: int
    pair_route_set_complete: bool
    mass: float
    max_intermediate_degree: int
    hub_dominated: bool
    lag_low_days: int
    lag_high_days: int
    as_of: float | None

    @property
    def is_pure_scaffold(self) -> bool:
        """True when no hop varies with time. Always False under the default filter.

        Kept as a property so a caller that lowers ``require_evidence_hop`` can
        still ask the question, and so a test can assert the default never emits
        one.
        """
        return self.evidence_hops == 0

    def describe(self) -> str:
        """One line naming the mechanism, the lag and the evidence."""
        parts = [self.path_names[0]]
        for i in range(self.hops):
            tag = {"evidence": "EVID", "scaffold": "SCAF", "unclassified": "????"}[self.hop_classes[i]]
            parts.append(f"-[{self.link_types[i]}/{self.hop_sources[i]}/{tag}]->")
            parts.append(self.path_names[i + 1])
        hub = "  HUB" if self.hub_dominated else ""
        return (
            f"{self.chain_id}  {self.source_tool}/{self.observation_type} "
            f"(n={self.source_obs_count})  lag [{self.lag_low_days},{self.lag_high_days})d  "
            f"src={self.independent_sources}  " + " ".join(parts) + hub
        )


class CandidateList(list):
    """A ``list[ChainCandidate]`` that carries its own denominator.

    It IS a list, so ``enumerate_chains`` honours its documented return type,
    but it also answers "how many did you look at, and what did you drop?" via
    ``.space`` and ``.ledger``. Slicing or copying yields a plain list and LOSES
    that provenance — deliberately, exactly as ``RouteList`` does: a candidate
    set stripped of its denominator must not still look authoritative.
    """

    def __init__(
        self,
        candidates: Iterable[ChainCandidate],
        *,
        space: Mapping[str, Any],
        ledger: FilterLedger,
        universe: Universe,
    ) -> None:
        super().__init__(candidates)
        self.space = dict(space)
        self.ledger = ledger
        self.universe = universe

    def headline(self) -> str:
        """The only honest one-liner: tested count first, survivors are not here.

        Deliberately does not say "found". Nothing in this list has been tested,
        so the number of candidates is the number of tests the next stage owes a
        multiple-comparison correction over.
        """
        s = self.space
        return (
            f"{len(self)} chain candidates ENUMERATED (not tested) from "
            f"{s['pairs_searched']} (source, target) route searches over "
            f"{s['n_source_channels']} source channels x {s['n_targets']} priced targets "
            f"x {s['n_lag_windows']} lag windows; up-front upper bound was "
            f"{s['upper_bound_candidates']}. Any survivor must be corrected over "
            f"{len(self)} tests, not over the survivors."
        )


# ── helpers ───────────────────────────────────────────────────────────────


def _iso(ts: float | None) -> str:
    """Epoch seconds as a UTC ISO string, or ``'none'``."""
    if ts is None:
        return "none"
    import datetime as _dt

    return _dt.datetime.fromtimestamp(ts, _dt.UTC).isoformat()


def _connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    """Open the pipeline DB strictly read-only.

    The live DB is the only irreplaceable asset here, so ``mode=ro`` is built
    from a resolved path and SQLite rejects writes at the engine level — a
    stronger guarantee than "this module contains no INSERT".
    """
    path = Path(db_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"pipeline DB not found at {path}. Enumeration never creates a database: "
            "an empty one would report a candidate space of zero and look like a clean bill of health."
        )
    return sqlite3.connect(f"file:{quote(str(path.resolve()))}?mode=ro", uri=True)


def _normalise_lag_windows(windows: Iterable[Sequence[int]]) -> tuple[tuple[int, int], ...]:
    """Validate and canonicalise a lag grid, raising on anything ambiguous.

    Rejects a reversed or empty window, a negative low bound (a chain whose
    cause follows its effect is not a lag, it is leakage) and duplicates.
    """
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for w in windows:
        pair = tuple(int(x) for x in w)
        if len(pair) != 2:
            raise GhostEnumerationError(f"lag window {w!r} must be a (low_days, high_days) pair")
        low, high = pair
        if low < 0:
            raise GhostEnumerationError(
                f"lag window {pair} has a negative low bound: a target responding BEFORE its "
                "source is data leakage, not a lag (F-04)."
            )
        if high <= low:
            raise GhostEnumerationError(f"lag window {pair} is empty or reversed; high must exceed low")
        if pair in seen:
            raise GhostEnumerationError(
                f"lag window {pair} is duplicated. A repeated window would double this "
                "candidate's weight in the multiple-testing denominator while testing nothing new."
            )
        seen.add(pair)
        out.append((low, high))
    if not out:
        raise GhostEnumerationError("lag_windows is empty: a chain with no lag window is not falsifiable")
    return tuple(out)


def chain_id(
    source_entity: str,
    source_tool: str,
    observation_type: str,
    path: Sequence[str],
    link_types: Sequence[str],
    lag_window: Sequence[int],
) -> str:
    """A stable 20-hex-character id for one chain hypothesis.

    Deterministic across runs, processes and Python versions: blake2b over a
    canonical ``|``-joined string with ``>`` between path nodes. Not derived
    from ``id()``, dict order or insertion order, any of which would make the
    "same candidate across runs" claim false.

    ``as_of`` is intentionally excluded (see ``ChainCandidate``), and so are
    ``mass`` and the hub flag, which are properties of the ranking rather than
    of the hypothesis.

    Every field is LENGTH-PREFIXED before hashing. A plain ``"|".join`` is
    forgeable: ``source_entity="x|y", source_tool="t"`` and
    ``source_entity="x", source_tool="y|t"`` produce the same joined string and
    therefore the same id, and ``source_tool`` / ``observation_type`` are free
    text written by collectors. Two distinct hypotheses sharing an id would be
    counted once in the multiple-testing denominator and silently merged in any
    store keyed by id — the exact class of quiet miscount this module exists to
    prevent. A test asserts the two cases above differ.

    Misleads: 80 bits. At 10^6 candidates the birthday collision probability is
    about 4 x 10^-13; at 10^8 it is about 4 x 10^-9. ``enumerate_chains``
    verifies uniqueness within its own output and raises on a collision rather
    than letting two hypotheses share a row.
    """
    low, high = (int(x) for x in lag_window)
    parts = [source_entity, source_tool, observation_type, *path, "->", *link_types, f"{low}-{high}"]
    payload = "".join(f"{len(p)}:{p}|" for p in parts)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=10).hexdigest()


# ── the universe: what is testable at all ─────────────────────────────────


def load_universe(
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    as_of: float | None = None,
    min_target_price_obs: int = DEFAULT_MIN_TARGET_PRICE_OBS,
    min_source_obs_per_entity: int = DEFAULT_MIN_SOURCE_OBS_PER_ENTITY,
    max_source_staleness_days: float = DEFAULT_MAX_SOURCE_STALENESS_DAYS,
    price_observation_type: str = PRICE_OBSERVATION_TYPE,
    exclude_source_tools: Iterable[str] = (),
    exclude_observation_types: Iterable[str] = (),
    conn: sqlite3.Connection | None = None,
) -> Universe:
    """Read the DB once and decide what could be tested, with a count per filter.

    Targets are entities with at least ``min_target_price_obs`` observations of
    ``price_observation_type`` at or before the cutoff. Source channels are
    (entity, source_tool, observation_type) triples with at least
    ``min_source_obs_per_entity`` observations at or before the cutoff whose last
    observation is within ``max_source_staleness_days`` of it.

    THE CUTOFF. ``as_of`` is the observation cutoff. When it is None the cutoff
    is the wall-clock time of the call, NOT infinity — the live DB contains
    ``gov_contracts`` rows dated into 2030 (award dates written as observation
    times), and a channel counted with future rows is a leakage bug wearing a
    completeness costume. ``Universe.observation_rows_after_cutoff`` reports how
    many rows that removed.

    Raises ``EmptyUniverseError`` when either side is empty, rather than
    returning an empty universe that would make ``enumerate_chains`` produce a
    confident zero.

    Where it misleads:
      * this is a filter on ESTIMABILITY, not on power or on quality. A channel
        with 21 observations passes.
      * the graph is not consulted here. A channel whose entity is isolated in
        the graph still appears in ``channels``; ``enumerate_chains`` applies
        that filter and counts it separately, because "not in the graph" is a
        property of the graph's ``as_of``, not of the observation table.
      * ``min_target_price_obs`` currently removes nothing on the live DB (all
        89 priced instruments have >= 840 rows). It is reported as a step that
        dropped 0, which is the honest way to show an inactive guard.
    """
    if min_target_price_obs < 1:
        raise GhostEnumerationError(f"min_target_price_obs must be >= 1, got {min_target_price_obs}")
    if min_source_obs_per_entity < 2:
        raise GhostEnumerationError(
            f"min_source_obs_per_entity must be >= 2, got {min_source_obs_per_entity}: "
            "a z-score needs at least two points to have a denominator at all."
        )
    if max_source_staleness_days <= 0:
        raise GhostEnumerationError(f"max_source_staleness_days must be > 0, got {max_source_staleness_days}")

    cutoff = float(as_of) if as_of is not None else time.time()
    owned = conn is None
    c = conn if conn is not None else _connect_readonly(db_path)
    try:
        entities = {
            row[0]: (row[1], row[2]) for row in c.execute("SELECT entity_id, canonical_name, entity_type FROM entities")
        }
        rows_total = int(c.execute("SELECT COUNT(*) FROM entity_observations").fetchone()[0])
        rows_after = int(
            c.execute("SELECT COUNT(*) FROM entity_observations WHERE observed_at > ?", (cutoff,)).fetchone()[0]
        )
        chan_rows = c.execute(
            "SELECT entity_id, source_tool, observation_type, COUNT(*), MIN(observed_at), MAX(observed_at) "
            "FROM entity_observations WHERE observed_at <= ? GROUP BY 1, 2, 3",
            (cutoff,),
        ).fetchall()
    finally:
        if owned:
            c.close()

    # ── targets ────────────────────────────────────────────────
    price_rows = [r for r in chan_rows if r[2] == price_observation_type]
    n_typed_instruments = sum(1 for _eid, (_n, t) in entities.items() if t == "instrument")
    target_steps = [
        FilterStep(
            "entities_total",
            len(entities),
            len(entities),
            f"all entities in the DB ({n_typed_instruments} of entity_type 'instrument')",
            {"instrument_typed": n_typed_instruments},
        ),
        FilterStep(
            "has_price_series",
            len(entities),
            len(price_rows),
            f"has >=1 {price_observation_type} observation at or before the cutoff",
            {"price_observation_type": price_observation_type},
        ),
    ]
    priced_known = [r for r in price_rows if r[0] in entities]
    target_steps.append(
        FilterStep(
            "price_entity_known",
            len(price_rows),
            len(priced_known),
            "the priced entity_id exists in `entities` (an orphan observation has no type or name)",
        )
    )
    kept_targets = [r for r in priced_known if r[3] >= min_target_price_obs]
    target_steps.append(
        FilterStep(
            "min_target_price_obs",
            len(priced_known),
            len(kept_targets),
            f">= {min_target_price_obs} daily price observations (block bootstrap needs blocks)",
            {"threshold": min_target_price_obs},
        )
    )
    targets = tuple(
        sorted(
            (
                PriceTarget(
                    entity_id=eid,
                    canonical_name=entities[eid][0],
                    entity_type=entities[eid][1],
                    n_price_obs=int(n),
                    first_observed_at=float(mn),
                    last_observed_at=float(mx),
                )
                for eid, _tool, _obs, n, mn, mx in kept_targets
            ),
            key=lambda t: t.entity_id,
        )
    )

    # ── source channels ────────────────────────────────────────
    ex_tools = frozenset(exclude_source_tools)
    ex_obs = frozenset(exclude_observation_types)
    stale_cutoff = cutoff - max_source_staleness_days * 86400.0

    kind_totals: dict[tuple[str, str], list[int]] = {}
    for _eid, tool, obs, n, _mn, mx in chan_rows:
        agg = kind_totals.setdefault((tool, obs), [0, 0, 0])
        agg[0] += int(n)
        agg[1] += 1
        agg[2] = max(agg[2], int(mx))

    chan_steps = [
        FilterStep(
            "channels_total",
            len(chan_rows),
            len(chan_rows),
            "distinct (entity, source_tool, observation_type) triples at or before the cutoff",
        )
    ]
    s1 = [r for r in chan_rows if r[0] in entities]
    chan_steps.append(
        FilterStep("channel_entity_known", len(chan_rows), len(s1), "the observed entity_id exists in `entities`")
    )
    s2 = [r for r in s1 if r[1] not in ex_tools and r[2] not in ex_obs]
    chan_steps.append(
        FilterStep(
            "channel_excluded_by_option",
            len(s1),
            len(s2),
            f"caller excluded source tools {sorted(ex_tools)} and observation types {sorted(ex_obs)}",
            {"exclude_source_tools": sorted(ex_tools), "exclude_observation_types": sorted(ex_obs)},
        )
    )
    s3 = [r for r in s2 if r[3] >= min_source_obs_per_entity]
    chan_steps.append(
        FilterStep(
            "min_source_obs_per_entity",
            len(s2),
            len(s3),
            f">= {min_source_obs_per_entity} observations, so the channel is z-scoreable at all",
            {"threshold": min_source_obs_per_entity},
        )
    )
    s4 = [r for r in s3 if float(r[5]) >= stale_cutoff]
    chan_steps.append(
        FilterStep(
            "channel_live",
            len(s3),
            len(s4),
            f"last observation within {max_source_staleness_days:g} days of the cutoff "
            f"({_iso(stale_cutoff)}); a dead source cannot test anything",
            {"stale_cutoff": stale_cutoff, "max_source_staleness_days": max_source_staleness_days},
        )
    )

    excluded: dict[tuple[str, str], dict[str, Any]] = {}
    kept_keys = {(r[0], r[1], r[2]) for r in s4}
    for eid, tool, obs, n_obs, _mn, mx in s1:
        if (eid, tool, obs) in kept_keys:
            continue
        key = (tool, obs)
        info = excluded.setdefault(
            key,
            {
                "channels_excluded": 0,
                "excluded_by_option": 0,
                "below_min_obs": 0,
                "stale": 0,
                "max_obs_on_an_excluded_channel": 0,
                "reasons": set(),
                "obs_per_entity": kind_totals[key][0] / max(kind_totals[key][1], 1),
                "staleness_days": (cutoff - kind_totals[key][2]) / 86400.0,
            },
        )
        info["channels_excluded"] += 1
        info["max_obs_on_an_excluded_channel"] = max(info["max_obs_on_an_excluded_channel"], int(n_obs))
        if tool in ex_tools or obs in ex_obs:
            info["excluded_by_option"] += 1
            info["reasons"].add("excluded by caller option")
        if n_obs < min_source_obs_per_entity:
            info["below_min_obs"] += 1
            info["reasons"].add(f"< {min_source_obs_per_entity} obs, not z-scoreable")
        if float(mx) < stale_cutoff:
            info["stale"] += 1
            info["reasons"].add(f"last observation > {max_source_staleness_days:g} days before the cutoff")
    for info in excluded.values():
        info["reason"] = "; ".join(sorted(info.pop("reasons"))) or "unknown"

    channels = tuple(
        sorted(
            (
                ObservationChannel(
                    entity_id=eid,
                    canonical_name=entities[eid][0],
                    entity_type=entities[eid][1],
                    source_tool=tool,
                    observation_type=obs,
                    n_obs=int(n),
                    first_observed_at=float(mn),
                    last_observed_at=float(mx),
                )
                for eid, tool, obs, n, mn, mx in s4
            ),
            key=lambda c: (c.entity_id, c.source_tool, c.observation_type),
        )
    )

    universe = Universe(
        db_path=str(db_path),
        observation_cutoff=cutoff,
        as_of=float(as_of) if as_of is not None else None,
        targets=targets,
        channels=channels,
        target_ledger=FilterLedger("price targets", tuple(target_steps)),
        channel_ledger=FilterLedger("source channels", tuple(chan_steps)),
        excluded_channel_kinds=excluded,
        observation_rows_total=rows_total,
        observation_rows_after_cutoff=rows_after,
        options={
            "min_target_price_obs": min_target_price_obs,
            "min_source_obs_per_entity": min_source_obs_per_entity,
            "max_source_staleness_days": max_source_staleness_days,
            "price_observation_type": price_observation_type,
            "exclude_source_tools": sorted(ex_tools),
            "exclude_observation_types": sorted(ex_obs),
        },
    )
    if not targets:
        raise EmptyUniverseError(
            f"no entity has >= {min_target_price_obs} {price_observation_type} observations at or before "
            f"{_iso(cutoff)}. There is nothing to test a chain AGAINST, which is not the same as "
            f"'no chains exist'.\n{universe.target_ledger.describe()}"
        )
    if not channels:
        raise EmptyUniverseError(
            f"no (entity, source_tool, observation_type) channel has >= {min_source_obs_per_entity} "
            f"observations within {max_source_staleness_days:g} days of {_iso(cutoff)}. Every chain's "
            f"source term would be unestimable.\n{universe.channel_ledger.describe()}"
        )
    return universe


# ── the denominator, before anything is generated ─────────────────────────


def count_candidate_space(
    graph: Any = None,
    *,
    universe: Universe | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
    as_of: float | None = None,
    max_hops: int = 3,
    k: int = 5,
    lag_windows: Iterable[Sequence[int]] = DEFAULT_LAG_WINDOWS_DAYS,
    require_evidence_hop: bool = True,
    min_evidence_sources: int = 1,
    reachability_fraction: float | None = None,
    **universe_opts: Any,
) -> dict[str, Any]:
    """How many candidates could these options produce? Answered BEFORE generating.

    The multiple-testing denominator has to be knowable up front, not discovered
    after a run has already chosen its thresholds. This is arithmetic over the
    universe plus the option grid; it walks no paths and costs milliseconds.

    Returns a dict whose load-bearing keys are:
      ``upper_bound_candidates``  ``n_source_channels * n_targets * k *
          n_lag_windows``, minus the self-pair term. An UPPER bound: it assumes
          every source reaches every target by ``k`` distinct routes. On the
          live graph the realised count is roughly an order of magnitude below
          it, because most pairs are unreachable within ``max_hops``.
      ``pairs_to_search``  distinct (source ENTITY, target) route searches. This
          is the cost predictor, and it is smaller than
          ``n_source_channels * n_targets`` because one entity can carry several
          channels.
      ``too_large``  True when the bound exceeds ``advisory_limit`` (10^6). The
          honest response is to narrow ``k``, ``max_hops`` or ``lag_windows``,
          not to sample: a sampled run has an unknowable denominator.
      ``projected_candidates``  present only when the caller supplies
          ``reachability_fraction`` measured from an earlier run. It is an
          extrapolation and is labelled as one.

    Where it misleads:
      * ``graph`` is accepted and, apart from reading its link count for the
        report, UNUSED. Reachability is a property of the graph that cannot be
        had without walking it, so this function declines to guess and returns
        a bound instead of a pretend-exact number.
      * the bound ignores ``require_evidence_hop`` and ``min_evidence_sources``,
        both of which only ever reduce the count. It is a bound, not a forecast.
      * a bound is not a denominator. The denominator for a BH correction is the
        realised ``len(enumerate_chains(...))``, which ``CandidateList.space``
        reports next to this bound so the gap is visible.
    """
    if k < 1:
        raise GhostEnumerationError(f"k must be >= 1, got {k}")
    if max_hops < 1:
        raise GhostEnumerationError(f"max_hops must be >= 1, got {max_hops}")
    if min_evidence_sources < 0:
        raise GhostEnumerationError(f"min_evidence_sources must be >= 0, got {min_evidence_sources}")
    if min_evidence_sources > max_hops:
        raise GhostEnumerationError(
            f"min_evidence_sources={min_evidence_sources} exceeds max_hops={max_hops}: no path of at "
            "most that many hops can carry that many distinct evidence sources, so every candidate "
            "would be filtered out and the run would report a confident zero."
        )
    windows = _normalise_lag_windows(lag_windows)
    if universe is None:
        universe = load_universe(db_path, as_of=as_of, **universe_opts)
    elif universe_opts:
        raise GhostEnumerationError(
            f"universe= was supplied together with universe options {sorted(universe_opts)}. "
            "The options would be silently ignored and the reported filter counts would describe a "
            "different universe from the one used."
        )

    target_ids = {t.entity_id for t in universe.targets}
    n_targets = len(universe.targets)
    n_channels = len(universe.channels)
    n_lags = len(windows)
    src_entities = universe.source_entity_ids

    # A chain from an instrument to itself has no mechanism to explain.
    self_channels = sum(1 for c in universe.channels if c.entity_id in target_ids)
    channel_target_pairs = n_channels * n_targets - self_channels
    pairs_to_search = len(src_entities) * n_targets - sum(1 for e in src_entities if e in target_ids)

    upper = channel_target_pairs * k * n_lags
    advisory_limit = 1_000_000
    space: dict[str, Any] = {
        "bound_kind": "combinatorial_upper_bound",
        "n_targets": n_targets,
        "n_source_channels": n_channels,
        "n_source_entities": len(src_entities),
        "n_lag_windows": n_lags,
        "lag_windows": list(windows),
        "k": k,
        "max_hops": max_hops,
        "require_evidence_hop": require_evidence_hop,
        "min_evidence_sources": min_evidence_sources,
        "channel_target_pairs": channel_target_pairs,
        "self_pairs_excluded": self_channels,
        "pairs_to_search": pairs_to_search,
        "upper_bound_candidates": upper,
        "advisory_limit": advisory_limit,
        "too_large": upper > advisory_limit,
        "observation_cutoff": universe.observation_cutoff,
        "as_of": universe.as_of,
        "graph_links": int(getattr(graph, "n_links", 0) or 0) if graph is not None else None,
        "graph_nodes": int(getattr(graph, "n_nodes", 0) or 0) if graph is not None else None,
        "reachability_measured": False,
        "notes": [
            "upper_bound_candidates assumes every source channel reaches every target by k distinct "
            "routes; it is an upper bound, not a forecast.",
            "require_evidence_hop and min_evidence_sources are NOT applied to this bound; both only "
            "reduce the realised count.",
            "the denominator for a multiple-comparison correction is the realised candidate count, "
            "not this bound. Compare them in CandidateList.space.",
        ],
    }
    if space["too_large"]:
        space["notes"].append(
            f"BOUND EXCEEDS {advisory_limit}: narrow k (now {k}), max_hops (now {max_hops}) or "
            f"lag_windows (now {n_lags}) before generating. Do NOT sample — a sampled run cannot "
            "state its own denominator, and the denominator is the product."
        )
    if reachability_fraction is not None:
        if not 0.0 <= reachability_fraction <= 1.0:
            raise GhostEnumerationError(f"reachability_fraction must be in [0, 1], got {reachability_fraction}")
        space["reachability_fraction"] = float(reachability_fraction)
        space["projected_candidates"] = int(round(upper * reachability_fraction))
        space["notes"].append(
            "projected_candidates is an EXTRAPOLATION from a reachability fraction the caller supplied "
            "from an earlier run, on earlier options. It is not a measurement of this run."
        )
    return space


def describe_candidate_space(space: Mapping[str, Any]) -> str:
    """Human rendering of ``count_candidate_space``, bound and caveats together."""
    lines = [
        f"candidate space ({space['bound_kind']}):",
        f"  {space['n_source_channels']} source channels ({space['n_source_entities']} entities)"
        f" x {space['n_targets']} priced targets"
        f" x k={space['k']} routes x {space['n_lag_windows']} lag windows",
        f"  self-pairs excluded: {space['self_pairs_excluded']}",
        f"  route searches to run: {space['pairs_to_search']}",
        f"  UPPER BOUND candidates: {space['upper_bound_candidates']}",
    ]
    if space.get("projected_candidates") is not None:
        lines.append(
            f"  projected (extrapolated, x{space['reachability_fraction']:.3f}): {space['projected_candidates']}"
        )
    lines += [f"  note: {n}" for n in space["notes"]]
    return "\n".join(lines)


# ── enumeration ───────────────────────────────────────────────────────────


def _route_evidence_sources(route: Route) -> tuple[str, ...]:
    """Distinct ingest pipelines on this route's EVIDENCE hops, sorted.

    Identical by construction to what ``agent.mechanism.routes.independence``
    counts: it reads ``Route.evidence_sources``, the same tested property
    ``independence`` iterates. Used per route instead of calling
    ``independence`` once per route because ``independence`` builds a 25-key
    verdict dict, and at 10^5 routes that is a lot of dictionaries for one
    integer. The pair-level ``independence`` call IS made, once per pair, and
    its provenance is carried on every candidate.
    """
    return tuple(sorted(route.evidence_sources))


def witness_groups(route: Route) -> tuple[tuple[str, ...], ...]:
    """Collapse evidence sources that co-write a relation into single witnesses.

    ``agent.mechanism.routes.independence`` documents that two ``source`` labels
    on the SAME ``link_type`` are probably one witness, and names the measured
    case: ``topic_relates_to_instrument`` is written by both ``polymarket`` and
    ``repair_topic_links``, the latter being a repair pass over the former. It
    therefore calls its own ``independent_sources`` an UPPER bound.

    This is the lower bound. Sources on a route are joined when they appear on
    the same evidence ``link_type``, transitively, and each connected component
    counts as ONE witness.

    Measured on the live graph (2026-09-30, 371,368 candidates enumerated):
    4,948 candidates survive ``min_evidence_sources=2``, and 1,300 of them carry
    exactly ``('polymarket', 'repair_topic_links')`` — a naive count of 2 that
    collapses to a conservative count of 1. Add ``exclude_hub_dominated=True``
    and 88 routes remain, ALL of them that same pair, so the conservative floor
    takes the survivor count to zero: not one chain in the space is corroborated
    by two independent witnesses on a non-hub route.

    Returns one sorted tuple per witness group, sorted. Empty for a route with
    no evidence hop.

    Misleads: this is a lower bound, not the truth. Two sources that never share
    a link_type can still be one publisher behind two pipelines, and no
    structural rule in this repository can see that. It collapses what is
    provably co-dependent; it cannot collapse what is merely likely to be.
    """
    by_link_type: dict[str, set[str]] = {}
    for hop in route.hops_detail:
        if hop.kind == "evidence":
            by_link_type.setdefault(hop.link_type, set()).add(hop.source)
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for sources in by_link_type.values():
        ordered = sorted(sources)
        for src in ordered:
            parent.setdefault(src, src)  # a lone source is still a witness — seed it
        for other in ordered[1:]:
            ra, rb = find(ordered[0]), find(other)
            if ra != rb:
                parent[rb] = ra
    groups: dict[str, set[str]] = {}
    for src in list(parent):
        groups.setdefault(find(src), set()).add(src)
    return tuple(sorted(tuple(sorted(g)) for g in groups.values()))


def iter_chains(
    graph: Any = None,
    *,
    universe: Universe | None = None,
    db_path: str | Path | None = None,
    max_hops: int = 3,
    min_evidence_sources: int = 1,
    as_of: float | None = None,
    k: int = 5,
    lag_windows: Iterable[Sequence[int]] = DEFAULT_LAG_WINDOWS_DAYS,
    require_evidence_hop: bool = True,
    exclude_hub_dominated: bool = False,
    min_conservative_sources: int = 0,
    stats: dict[str, Any] | None = None,
    **universe_opts: Any,
) -> Iterator[ChainCandidate]:
    """Stream chain candidates, and write the drop counts into ``stats``.

    The streaming form of ``enumerate_chains``, for a caller that will test each
    candidate as it arrives rather than hold 10^5 of them in memory. Pass a dict
    as ``stats`` to receive the same counters ``CandidateList.space`` carries;
    without it the counts are computed and discarded, which is exactly the
    silent-drop failure this module is built against — so pass it.

    Traversal is ``agent.mechanism.routes.top_routes`` (tested, sub-second, exact
    top-k by personalised-PageRank mass) and evidence counting is
    ``agent.mechanism.routes.independence``. This module writes neither.

    Where it misleads:
      * the generator yields nothing until the first reachable pair is found, so
        a long initial pause is normal on a sparse graph.
      * ``stats`` is only complete after the generator is exhausted. Reading it
        mid-iteration gives a partial account that looks like a final one.
    """
    if min_evidence_sources < 0:
        raise GhostEnumerationError(f"min_evidence_sources must be >= 0, got {min_evidence_sources}")
    if min_evidence_sources > 0 and not require_evidence_hop:
        raise GhostEnumerationError(
            f"min_evidence_sources={min_evidence_sources} with require_evidence_hop=False is "
            "contradictory: the source floor already implies an evidence hop. Set one or the other."
        )
    windows = _normalise_lag_windows(lag_windows)

    if graph is None:
        graph = load_route_graph(db_path if db_path is not None else DEFAULT_DB_PATH, as_of=as_of)
    if db_path is None:
        db_path = getattr(graph, "stats", {}).get("db_path") or DEFAULT_DB_PATH
    if universe is None:
        universe = load_universe(db_path, as_of=as_of, **universe_opts)
    elif universe_opts:
        raise GhostEnumerationError(
            f"universe= was supplied together with universe options {sorted(universe_opts)}; "
            "the options would be silently ignored."
        )

    graph_as_of = getattr(graph, "stats", {}).get("as_of")
    counters: dict[str, Any] = {} if stats is None else stats
    counters.update(
        {
            "graph_as_of": graph_as_of,
            "observation_cutoff": universe.observation_cutoff,
            "as_of": universe.as_of,
            "n_targets": len(universe.targets),
            "n_source_channels": len(universe.channels),
            "n_source_entities": len(universe.source_entity_ids),
            "n_lag_windows": len(windows),
            "lag_windows": list(windows),
            "k": k,
            "max_hops": max_hops,
            "min_evidence_sources": min_evidence_sources,
            "require_evidence_hop": require_evidence_hop,
            "exclude_hub_dominated": exclude_hub_dominated,
            "min_conservative_sources": min_conservative_sources,
            "channels_entity_not_in_graph": 0,
            "channels_entity_isolated_in_graph": 0,
            "source_entities_not_in_graph": 0,
            "source_entities_usable": 0,
            "pairs_searched": 0,
            "pairs_self_skipped": 0,
            "pairs_unreachable": 0,
            "pairs_with_routes": 0,
            "pairs_truncated": 0,
            "pairs_budget_exhausted": 0,
            "routes_found": 0,
            "routes_dropped_no_evidence_hop": 0,
            "routes_dropped_below_min_sources": 0,
            "routes_dropped_hub_dominated": 0,
            "routes_dropped_below_min_conservative_sources": 0,
            "routes_kept": 0,
            "candidates_emitted": 0,
            "candidates_hub_dominated": 0,
            "candidates_single_source": 0,
            "candidates_conservative_single_source": 0,
            "candidates_with_co_dependent_sources": 0,
            "evidence_source_candidate_counts": {},
            "route_errors": {},
            "warnings": [],
        }
    )
    if graph_as_of != universe.as_of:
        counters["warnings"].append(
            f"GRAPH/OBSERVATION CUTOFF MISMATCH: the graph was loaded with as_of={graph_as_of!r} but "
            f"observations are cut at {_iso(universe.observation_cutoff)} (as_of={universe.as_of!r}). "
            "A link visible after the observation cutoff can route a chain the source data could not "
            "have known about (F-04). Load both at the same as_of."
        )

    adj = getattr(graph, "adj", None)
    if adj is None:
        raise GhostEnumerationError(
            f"{type(graph).__name__} exposes no adjacency; pass a RouteGraph from "
            "agent.mechanism.routes.load_route_graph, a MechanismGraph, or a DB path."
        )

    by_entity = universe.channels_by_entity()
    usable: dict[str, tuple[ObservationChannel, ...]] = {}
    for eid, chs in by_entity.items():
        if eid not in adj:
            counters["source_entities_not_in_graph"] += 1
            counters["channels_entity_not_in_graph"] += len(chs)
            continue
        if not adj[eid]:
            counters["source_entities_isolated_in_graph"] = counters.get("source_entities_isolated_in_graph", 0) + 1
            counters["channels_entity_isolated_in_graph"] += len(chs)
            continue
        usable[eid] = chs
    counters["source_entities_usable"] = len(usable)
    counters["channels_usable"] = sum(len(v) for v in usable.values())
    if counters["channels_entity_not_in_graph"]:
        counters["warnings"].append(
            f"{counters['channels_entity_not_in_graph']} z-scoreable channels on "
            f"{counters['source_entities_not_in_graph']} entities have NO link in this graph and can "
            "never appear in a chain. They are testable data with no mechanism, not missing data."
        )

    if not usable:
        raise EmptyUniverseError(
            "every z-scoreable source channel belongs to an entity with no links in this graph. "
            "There is no mechanism to enumerate, which is not the same as no chains existing."
        )

    seen_ids: set[str] = set()
    for target in universe.targets:
        if target.entity_id not in adj or not adj[target.entity_id]:
            counters["targets_isolated_in_graph"] = counters.get("targets_isolated_in_graph", 0) + 1
            continue
        for src_id, channels in usable.items():
            if src_id == target.entity_id:
                counters["pairs_self_skipped"] += 1
                continue
            counters["pairs_searched"] += 1
            try:
                routes = top_routes(graph, src_id, target.entity_id, max_hops=max_hops, k=k)
            except RouteError as exc:  # isolated / unknown at this as_of — count, never swallow
                name = type(exc).__name__
                counters["route_errors"][name] = counters["route_errors"].get(name, 0) + 1
                continue
            search = routes.search
            if not routes:
                counters["pairs_unreachable"] += 1
                continue
            counters["pairs_with_routes"] += 1
            if search.more_routes_exist:
                counters["pairs_truncated"] += 1
            if search.budget_exhausted:
                counters["pairs_budget_exhausted"] += 1
            counters["routes_found"] += len(routes)

            pair_ind = independence(routes)
            pair_n = int(pair_ind["independent_sources"])
            pair_complete = bool(pair_ind["complete"])
            pair_codep = bool(pair_ind["shared_relation_sources"])

            for route in routes:
                if require_evidence_hop and not route.has_evidence:
                    counters["routes_dropped_no_evidence_hop"] += 1
                    continue
                ev_sources = _route_evidence_sources(route)
                if len(ev_sources) < min_evidence_sources:
                    counters["routes_dropped_below_min_sources"] += 1
                    continue
                if exclude_hub_dominated and route.hub_dominated:
                    counters["routes_dropped_hub_dominated"] += 1
                    continue
                groups = witness_groups(route)
                if len(groups) < min_conservative_sources:
                    counters["routes_dropped_below_min_conservative_sources"] += 1
                    continue
                counters["routes_kept"] += 1
                kinds = tuple(link_kind(lt) for lt in route.link_types)
                ev_hops = sum(1 for kk in kinds if kk == "evidence")
                sc_hops = sum(1 for kk in kinds if kk == "scaffold")
                un_hops = sum(1 for kk in kinds if kk == "unclassified")
                path_types = tuple(getattr(graph, "etype", {}).get(n, "unknown") for n in route.nodes)
                for ch in channels:
                    for low, high in windows:
                        cid = chain_id(
                            ch.entity_id,
                            ch.source_tool,
                            ch.observation_type,
                            route.nodes,
                            route.link_types,
                            (low, high),
                        )
                        if cid in seen_ids:
                            raise GhostEnumerationError(
                                f"chain_id collision or duplicate candidate: {cid} was already emitted. "
                                "Two hypotheses sharing an id would be counted once in the denominator "
                                "and merged in any store keyed by id."
                            )
                        seen_ids.add(cid)
                        counters["candidates_emitted"] += 1
                        if route.hub_dominated:
                            counters["candidates_hub_dominated"] += 1
                        if len(ev_sources) == 1:
                            counters["candidates_single_source"] += 1
                        if len(groups) == 1:
                            counters["candidates_conservative_single_source"] += 1
                        if len(groups) < len(ev_sources):
                            counters["candidates_with_co_dependent_sources"] += 1
                        esc = counters["evidence_source_candidate_counts"]
                        for s in ev_sources:
                            esc[s] = esc.get(s, 0) + 1
                        yield ChainCandidate(
                            chain_id=cid,
                            source_entity=ch.entity_id,
                            source_name=ch.canonical_name,
                            source_entity_type=ch.entity_type,
                            source_tool=ch.source_tool,
                            observation_type=ch.observation_type,
                            source_obs_count=ch.n_obs,
                            source_last_observed_at=ch.last_observed_at,
                            target_entity=target.entity_id,
                            target_name=target.canonical_name,
                            target_entity_type=target.entity_type,
                            target_obs_count=target.n_price_obs,
                            path=route.nodes,
                            path_names=route.node_names,
                            path_types=path_types,
                            hops=route.hops,
                            link_types=route.link_types,
                            hop_sources=route.sources,
                            hop_classes=kinds,
                            evidence_hops=ev_hops,
                            scaffold_hops=sc_hops,
                            unclassified_hops=un_hops,
                            independent_sources=len(ev_sources),
                            independent_sources_conservative=len(groups),
                            witness_groups=groups,
                            evidence_sources=ev_sources,
                            pair_independent_sources=pair_n,
                            pair_has_co_dependent_sources=pair_codep,
                            pair_routes_considered=len(routes),
                            pair_route_set_complete=pair_complete,
                            mass=route.mass,
                            max_intermediate_degree=route.max_intermediate_degree,
                            hub_dominated=route.hub_dominated,
                            lag_low_days=low,
                            lag_high_days=high,
                            as_of=universe.as_of,
                        )

    if counters["pairs_truncated"]:
        counters["warnings"].append(
            f"{counters['pairs_truncated']} of {counters['pairs_with_routes']} pairs had MORE routes "
            f"than k={k} returned. Every candidate list is a subset of the graph's chains, and this "
            "run's denominator is a count of what was looked at."
        )
    if counters["pairs_budget_exhausted"]:
        counters["warnings"].append(
            f"{counters['pairs_budget_exhausted']} pairs exhausted the route search budget: their "
            "routes are not guaranteed to be the top-k."
        )
    if counters["routes_dropped_no_evidence_hop"]:
        counters["warnings"].append(
            f"{counters['routes_dropped_no_evidence_hop']} routes were pure static geography (no "
            "time-varying hop) and were dropped. On the live graph the single highest-mass "
            "Russia -> WTI route is exactly this: a direct produced_in edge with zero evidential weight."
        )
    emitted = counters["candidates_emitted"]
    if emitted:
        hub_frac = counters["candidates_hub_dominated"] / emitted
        if hub_frac > 0.5:
            counters["warnings"].append(
                f"HUB SATURATION: {hub_frac:.1%} of candidates ({counters['candidates_hub_dominated']} of "
                f"{emitted}) route through a node at or above the graph's 99th degree percentile. The live "
                "graph's largest hub has degree 2010, so 'X located_in United States' connects almost "
                "everything to almost everything. Pass exclude_hub_dominated=True to see how much of this "
                "space survives without it — and note that doing so changes the denominator, so the two "
                "runs' survivors are not comparable."
            )
        counters["candidates_single_source_fraction"] = counters["candidates_single_source"] / emitted
        top = max(counters["evidence_source_candidate_counts"].items(), key=lambda kv: kv[1])
        counters["dominant_evidence_source"] = top[0]
        counters["dominant_evidence_source_fraction"] = top[1] / emitted
        if top[1] / emitted > 0.5:
            counters["warnings"].append(
                f"EVIDENCE MONOCULTURE: {top[0]!r} is the witness on {top[1] / emitted:.1%} of candidates "
                f"({top[1]} of {emitted}). A failure, revision or coverage change in that one pipeline takes "
                "the whole candidate space with it. This is not a diversified search; it is one dataset "
                "enumerated many ways, and any BH survivor drawn from it is a claim about that dataset."
            )
        codep = counters["candidates_with_co_dependent_sources"]
        if codep:
            counters["warnings"].append(
                f"CO-DEPENDENT WITNESSES: {codep} of {emitted} candidates ({codep / emitted:.1%}) count two "
                "or more sources that co-write the SAME relation, which independence() documents as one "
                "witness wearing several hats (topic_relates_to_instrument is written by both polymarket "
                "and repair_topic_links, a repair pass over it). Their independent_sources is an upper "
                "bound; quote independent_sources_conservative instead."
            )
        cons1 = counters["candidates_conservative_single_source"]
        if cons1 and cons1 > counters["candidates_single_source"]:
            counters["warnings"].append(
                f"{cons1} of {emitted} candidates collapse to exactly ONE witness once co-dependent "
                f"pipelines are merged, against {counters['candidates_single_source']} by the naive count. "
                "The gap is the amount of corroboration this space does not actually have."
            )
        if counters["candidates_single_source"] / emitted > 0.9:
            counters["warnings"].append(
                f"{counters['candidates_single_source'] / emitted:.1%} of candidates rest on exactly ONE "
                "independent time-varying source. Corroboration is not available for them at any k, so "
                "'independently confirmed' is not a phrase this run can support."
            )


def enumerate_chains(
    graph: Any = None,
    *,
    max_hops: int = 3,
    min_evidence_sources: int = 1,
    as_of: float | None = None,
    universe: Universe | None = None,
    db_path: str | Path | None = None,
    k: int = 5,
    lag_windows: Iterable[Sequence[int]] = DEFAULT_LAG_WINDOWS_DAYS,
    require_evidence_hop: bool = True,
    exclude_hub_dominated: bool = False,
    min_conservative_sources: int = 0,
    max_candidates: int = 500_000,
    **universe_opts: Any,
) -> CandidateList:
    """Generate every bounded chain candidate, and report the denominator with it.

    A candidate is a source channel (entity + source_tool + observation_type),
    a path through the entity graph, a priced target, and a lag window. Paths
    come from ``agent.mechanism.routes.top_routes``; evidence independence from
    ``agent.mechanism.routes.independence``. Nothing here is scored or tested.

    Order of operations, each step counted:
      1. ``load_universe`` — targets with a long enough price series, source
         channels that are z-scoreable and live. Two closed filter ledgers.
      2. ``count_candidate_space`` — the up-front upper bound, computed BEFORE
         any path is walked and stored in ``.space['upper_bound_candidates']``.
      3. per (source entity, target): ``top_routes``, then ``independence``.
      4. per route: require an evidence hop, require ``min_evidence_sources``
         distinct time-varying sources (the upper bound, matching
         ``independence``), optionally drop hub-dominated routes, then require
         ``min_conservative_sources`` witnesses after collapsing pipelines that
         co-write one relation (the lower bound). Measured on the live graph:
         ``min_evidence_sources=2`` leaves 4,948 candidates,
         ``min_conservative_sources=2`` cuts that to 3,648, and adding
         ``exclude_hub_dominated=True`` leaves 0 — the 88 non-hub two-source
         routes are all ``polymarket`` plus its own repair pass.
      5. per (route, channel, lag window): one ``ChainCandidate``.

    Returns a ``CandidateList``: a real list that also carries ``.space`` (the
    bound and every realised counter), ``.ledger`` (the route/candidate funnel,
    arithmetically closed) and ``.universe``. Use ``.headline()`` to report it —
    it leads with the tested count, because a survivor count without a
    denominator is the specific lie this module exists to prevent.

    Raises ``CandidateSpaceTooLargeError`` when generation would exceed
    ``max_candidates``, rather than sampling. A sampled run cannot state its own
    denominator.

    Where it misleads:
      * it is exhaustive over the OPTIONS, not over the graph. ``k`` and
        ``max_hops`` truncate, and ``.space['pairs_truncated']`` says how often.
      * every returned candidate is a guess. A run that returns 40,000
        candidates has done no work toward showing any of them is real, and the
        repo's own record is that essentially all of them will die: 0 of 51 in
        the CFTC study, and a 45% rolling beat rate where one window said 14%.
      * candidates are correlated. Four lag windows share a price series and
        many paths share a target, so the number of DISTINCT claims is well
        below ``len(...)``. Treat the count as a multiple-testing denominator,
        not as a discovery inventory.
    """
    if max_candidates < 1:
        raise GhostEnumerationError(f"max_candidates must be >= 1, got {max_candidates}")
    windows = _normalise_lag_windows(lag_windows)

    if graph is None:
        graph = load_route_graph(db_path if db_path is not None else DEFAULT_DB_PATH, as_of=as_of)
    resolved_db = db_path if db_path is not None else (getattr(graph, "stats", {}).get("db_path") or DEFAULT_DB_PATH)
    if universe is None:
        universe = load_universe(resolved_db, as_of=as_of, **universe_opts)
        universe_opts = {}

    space = count_candidate_space(
        graph,
        universe=universe,
        max_hops=max_hops,
        k=k,
        lag_windows=windows,
        require_evidence_hop=require_evidence_hop,
        min_evidence_sources=min_evidence_sources,
    )

    stats: dict[str, Any] = {}
    out: list[ChainCandidate] = []
    started = time.monotonic()
    for cand in iter_chains(
        graph,
        universe=universe,
        db_path=resolved_db,
        max_hops=max_hops,
        min_evidence_sources=min_evidence_sources,
        as_of=as_of,
        k=k,
        lag_windows=windows,
        require_evidence_hop=require_evidence_hop,
        exclude_hub_dominated=exclude_hub_dominated,
        min_conservative_sources=min_conservative_sources,
        stats=stats,
    ):
        out.append(cand)
        if len(out) > max_candidates:
            raise CandidateSpaceTooLargeError(
                f"generation passed max_candidates={max_candidates} (up-front upper bound was "
                f"{space['upper_bound_candidates']}, realised so far {len(out)} after "
                f"{stats['pairs_searched']} of {space['pairs_to_search']} route searches). "
                "Narrow k, max_hops or lag_windows, or raise max_candidates deliberately. This "
                "function will not sample: a sampled run cannot state its own denominator, and the "
                "denominator is the product."
            )
    elapsed = time.monotonic() - started

    space.update(stats)
    space["elapsed_seconds"] = elapsed
    space["candidates"] = len(out)
    space["bound_over_realised"] = (space["upper_bound_candidates"] / len(out)) if out else None
    space["reachability_measured"] = True
    space["reachability_fraction"] = (
        stats["pairs_with_routes"] / stats["pairs_searched"] if stats["pairs_searched"] else 0.0
    )
    if len(out) > _MAX_CHAIN_ID_CANDIDATES_SAFE:
        space["notes"].append(
            f"{len(out)} candidates exceeds {_MAX_CHAIN_ID_CANDIDATES_SAFE}: the 80-bit chain_id "
            "collision probability is no longer negligible. Widen the digest before persisting these."
        )

    after_min_sources = (
        stats["routes_found"] - stats["routes_dropped_no_evidence_hop"] - stats["routes_dropped_below_min_sources"]
    )
    after_hub = after_min_sources - stats["routes_dropped_hub_dominated"]
    funnel = FilterLedger(
        "routes -> candidates",
        (
            FilterStep(
                "routes_found",
                stats["routes_found"],
                stats["routes_found"],
                f"top_routes over {stats['pairs_searched']} searches "
                f"({stats['pairs_unreachable']} pairs unreachable within {max_hops} hops)",
                {"pairs_searched": stats["pairs_searched"], "pairs_unreachable": stats["pairs_unreachable"]},
            ),
            FilterStep(
                "require_evidence_hop",
                stats["routes_found"],
                stats["routes_found"] - stats["routes_dropped_no_evidence_hop"],
                "at least one time-varying hop; a pure-scaffold path is static geography",
                {"enabled": require_evidence_hop},
            ),
            FilterStep(
                "min_evidence_sources",
                stats["routes_found"] - stats["routes_dropped_no_evidence_hop"],
                after_min_sources,
                f">= {min_evidence_sources} distinct independent time-varying sources on the route",
                {"threshold": min_evidence_sources},
            ),
            FilterStep(
                "exclude_hub_dominated",
                after_min_sources,
                after_hub,
                "route passes no node at or above the graph's 99th degree percentile",
                {"enabled": exclude_hub_dominated},
            ),
            FilterStep(
                "min_conservative_sources",
                after_hub,
                stats["routes_kept"],
                f">= {min_conservative_sources} witnesses after collapsing sources that co-write one "
                "relation (the lower bound on corroboration)",
                {"threshold": min_conservative_sources},
            ),
            FilterStep(
                "expand_channels_x_lags",
                stats["routes_kept"],
                stats["routes_kept"],
                f"each kept route becomes (usable channels on its source) x {len(windows)} lag windows "
                f"= {len(out)} candidates",
                {"candidates": len(out), "n_lag_windows": len(windows)},
            ),
        ),
    )
    return CandidateList(out, space=space, ledger=funnel, universe=universe)


# ── F-18 guard: a number that does not vary is not a number ───────────────


def variation_report(candidates: Sequence[ChainCandidate]) -> dict[str, Any]:
    """Distinct-value counts for every scalar field, and the ones that are FROZEN.

    This exists because of the exact defect that made the previous discovery
    engine fake. ``pattern_extractor`` ranked meta-paths by an attention that
    came back as uniform ``1/k`` fractions — 0.0, 0.333, 0.5, 0.667, 1.0 — with
    9 of 20 edge types at exactly 0.0, and nobody printed the series. Under
    LESSONS F-18 a number is not to be called good until its full series has
    been printed and seen to vary; a frozen model was once ranked first here for
    having a good random initialisation.

    Returns ``{"n": int, "fields": {name: {...}}, "frozen_fields": [...],
    "duplicate_chain_ids": int}``. A field in ``frozen_fields`` took one value
    across more than one candidate: either the option grid collapsed it (``k=1``
    freezes ``pair_routes_considered``, one lag window freezes the lag bounds)
    or something upstream is returning a constant dressed as a measurement.
    Both cases must be explained before any ranking is quoted.

    Misleads: variation is necessary, not sufficient. A field can vary and still
    be meaningless — ``mass`` varies beautifully and says nothing about whether
    a chain is real.
    """
    n = len(candidates)
    numeric = (
        "hops",
        "evidence_hops",
        "scaffold_hops",
        "unclassified_hops",
        "independent_sources",
        "pair_independent_sources",
        "pair_routes_considered",
        "mass",
        "max_intermediate_degree",
        "source_obs_count",
        "target_obs_count",
        "lag_low_days",
        "lag_high_days",
    )
    categorical = (
        "source_tool",
        "observation_type",
        "source_entity_type",
        "target_entity",
        "source_entity",
        "hub_dominated",
    )
    fields: dict[str, Any] = {}
    for name in numeric:
        vals = [getattr(c, name) for c in candidates]
        uniq = sorted(set(vals))
        fields[name] = {
            "kind": "numeric",
            "n_distinct": len(uniq),
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
            "sample_distinct": uniq[:12],
        }
    for name in categorical:
        vals = [getattr(c, name) for c in candidates]
        uniq = sorted({str(v) for v in vals})
        fields[name] = {"kind": "categorical", "n_distinct": len(uniq), "sample_distinct": uniq[:12]}
    frozen = [name for name, info in fields.items() if n > 1 and info["n_distinct"] <= 1]
    ids = [c.chain_id for c in candidates]
    return {
        "n": n,
        "fields": fields,
        "frozen_fields": frozen,
        "duplicate_chain_ids": len(ids) - len(set(ids)),
    }
