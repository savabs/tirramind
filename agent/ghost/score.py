"""Honest scoring of automatically-generated ghost-chain candidates.

WHAT THIS MODULE IS FOR
    An automatic chain finder is a p-hacking machine unless the
    multiple-testing accounting is airtight. With 12 node types, 18 link types
    and a lag grid you can generate tens of thousands of candidate chains, so
    thousands of them will look significant by chance. This module runs every
    candidate through the SAME event study the published work used
    (``agent.verify.study.verify`` — it is not reimplemented here), then
    applies Benjamini-Hochberg across the WHOLE family and refuses to report a
    survivor without the family size next to it.

    Measured precedents in this repository, all of which this module exists to
    make visible rather than to avoid:

    * the CFTC study: 0 of 51 hypotheses survived BH. 9 of 51 would have
      "worked" uncorrected.
    * re-running it on more data gave 3 of 54 "survivors" at 5.1-7.6% power,
      one of them with n=3 (an i.i.d. bootstrap over 3 points takes at most 27
      distinct values, so its p-value has a resolution of about 0.04).
    * "12 positions, 3 bets" happened 0 times in 200 real portfolios.
    * "picking cost you 14%" was one window; the honest rolling record is a 45%
      beat rate.

    Every headline this project has produced died when measured properly.
    Assume the next one will too. The deliverable is the arithmetic.

THE THREE ACCOUNTING RULES, ENFORCED IN CODE NOT IN PROSE
    1.  ``ScoreBoard.n_tested`` is the FULL family size: every candidate times
        every horizon that was actually run. ``ScoreBoard.report()`` cannot
        print survivors without it, and ``ChainCell.survived`` is ``None``
        until a family has been corrected, so a lone ``score_chain`` result
        cannot be misread as a discovery.
    2.  A cell with ``n_events < 30`` or fewer than 20 effective clusters is
        ``UNTESTABLE``, not insignificant. It goes in its own bucket with a
        named reason. "Cannot tell" is a legitimate verdict and is the one
        this product differentiates on.
    3.  Power accompanies every p-value. A "significant" result at 5% power is
        closer to the false-positive rate than to evidence; this repository has
        already published that mistake once, so ``ChainCell.verdict`` spells it
        ``survived_underpowered`` rather than ``survived``.

WHERE THIS MODULE MISLEADS — read before quoting anything out of it
    * **It corrects the family it is GIVEN.** If the candidate generator
      pre-filtered, or if you drop cells before calling ``score_all``, m
      shrinks and every adjusted p is too small. ``ScoreBoard.n_generated`` is
      provided so a caller can record the pre-filter count, and
      ``report()`` prints it as "of N generated" whenever it is set. Nothing
      here can detect a family that was trimmed before it arrived.
    * **BH corrects within ONE look.** Scoring the same family again on more
      data is a second look and the correction does not cover it. There is no
      valid way to combine two ``ScoreBoard`` objects; ``relook_warnings``
      exists to say so out loud rather than to make it possible.
    * **Untestable cells are excluded from the BH denominator by default**,
      which is what the reference study did (it filtered non-computable cells
      out and reported them as "not testable"). That choice makes survival
      EASIER. ``ScoreBoard.bh_conservative`` re-runs the correction over the
      full family with untestable cells entered at p=1.0, and
      ``report()`` prints both counts. If they disagree, the exclusion is
      carrying the result and the result is not real.
    * **The clustering window is a knob that moves verdicts.**
      ``cluster_window_s`` (default 7 days) decides how many "effective
      clusters" a cell has and therefore whether it is testable at all. It is
      reported on every board and every cell. ``n_event_clusters`` (distinct
      event timestamps, the unit ``verify`` itself uses for power) is reported
      beside it; when the two disagree badly the timestamps were overstating
      independence.
    * **``power`` is stipulated, not measured.** It comes from
      ``agent.verify.study.power_two_sided``, which scales a reference
      t-statistic (2.14 at n=1451, from ``docs/publications/cot_null_result.md``)
      to this cell's cluster count. Change the reference and the power changes.
      It is a sensitivity figure, not a property of this sample. Post-hoc power
      computed from the observed effect would be a deterministic restatement of
      the p-value and is deliberately not offered.
    * **An effect is measured against the UNCONDITIONAL baseline**, not zero —
      that is ``verify``'s doing, and it is why ``edge`` can be negative while
      the raw event return is positive. In a trending market a zero null prints
      an "edge" equal to the drift.
    * **Nothing here validates the mechanism.** A surviving cell is a
      statistical association along one link type. Whether a route through the
      graph explains it, and how many genuinely independent sources witness it,
      is ``agent.mechanism.routes.top_routes`` / ``independence``, not this
      module.
    * **``extract_lags`` inherits a defect it can only route around.** See its
      docstring: the upstream lag extractor matches ``edge_type`` against the
      raw ``entity_links.link_type`` string, so a ``rev_`` or composite
      ``X_via_Y`` name silently yields ``mean_lag=0.0`` — a real-looking number
      meaning "no data". This module refuses those names instead of inheriting
      the zero.

COMPOSING WITH THE GENERATOR
    ``agent.ghost.enumerate`` emits ROUTE-level candidates (a specific source
    entity, a link-type sequence, a declared lag window). The event study is
    stated over a CHANNEL (source tool, observation type, numeric field,
    z-threshold, direction). ``from_enumerated`` adapts one to the other and
    returns, alongside the candidate, a list of exactly what that adaptation
    widened or invented — chiefly that the single-entity restriction is lost,
    because ``Hypothesis`` has no entity scope. Discard those warnings and a
    channel-level verdict gets reported as a route-level one.

READ-ONLY BY CONSTRUCTION
    Every connection this module opens is ``mode=ro`` via a ``file:`` URI, and
    it never instantiates ``PipelineStore`` (whose constructor runs
    ``_init_schema`` and therefore writes). ``extract_lags`` reuses the upstream
    ``extract_temporal_lags`` by handing it a read-only projection object that
    satisfies the three ``query_all_*`` methods that function calls.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from agent.verify.stats import benjamini_hochberg, effective_sample_size
from agent.verify.study import MIN_TESTABLE_EVENTS, N_BOOTSTRAP, Hypothesis, StudyResult, verify

__all__ = [
    "BH_EXCLUDES_UNTESTABLE",
    "DEFAULT_CLUSTER_WINDOW_S",
    "UNTESTABLE_MIN_CLUSTERS",
    "UNTESTABLE_MIN_EVENTS",
    "UNTESTABLE_REASONS",
    "CandidateError",
    "ChainCandidate",
    "ChainCell",
    "ChainResult",
    "ScoreBoard",
    "coerce_candidate",
    "extract_lags",
    "from_enumerated",
    "horizon_from_lag_window",
    "relook_warnings",
    "score_all",
    "score_chain",
]

# --------------------------------------------------------------------------- #
# thresholds — stated once, reported on every board
# --------------------------------------------------------------------------- #

#: Below this many graded events a cell is UNTESTABLE, not insignificant.
#: ``verify`` itself refuses to bootstrap below ``MIN_TESTABLE_EVENTS`` (3);
#: 30 is the point at which the normal approximation behind ``power_two_sided``
#: stops being wildly optimistic, and it is deliberately far above 3 because
#: this repository has already published a p-value computed from n=3.
UNTESTABLE_MIN_EVENTS = 30

#: Below this many effective (time-bucketed) clusters a cell is UNTESTABLE.
#: 123 events on 67 weeks are not 123 observations; in the published CFTC study
#: switching from an i.i.d. bootstrap over events to a cluster bootstrap over
#: weeks moved the headline adjusted p from 0.102 to 0.294.
UNTESTABLE_MIN_CLUSTERS = 20

#: Width of the independence bucket, in seconds. A knob that moves verdicts:
#: it decides ``effective_clusters`` and therefore testability.
DEFAULT_CLUSTER_WINDOW_S = 7 * 86400.0

#: Default policy for the BH denominator. True reproduces the reference study
#: (untestable cells filtered out before correcting); ``ScoreBoard`` always
#: computes the conservative alternative as well, so the choice is visible.
BH_EXCLUDES_UNTESTABLE = True

_P_BASES = ("clustered", "iid")

# Reasons a cell can be untestable. Enumerated so a caller can count them
# without discovering them empirically.
UNTESTABLE_REASONS = (
    "no_events",
    "below_verify_bootstrap_minimum",
    "n_events_below_minimum",
    "effective_clusters_below_minimum",
    "p_value_not_finite",
    "bookkeeping_did_not_reconcile",
)


class CandidateError(ValueError):
    """A candidate could not be read as a falsifiable chain specification."""


# --------------------------------------------------------------------------- #
# 1. the candidate contract
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ChainCandidate:
    """One falsifiable ghost-chain specification, independent of who generated it.

    This is deliberately a *contract*, not a class the generator must import:
    ``coerce_candidate`` accepts any mapping or any object carrying these
    attribute names, so the discovery side and the scoring side can be written
    concurrently without sharing a type.

    The statistical fields are exactly the ones ``agent.verify.study.Hypothesis``
    needs; the structural fields (``event_entity_type``, ``link_types``,
    ``metapath``) describe the chain's shape and are what ``extract_lags``
    reads. ``chain_id`` is what appears in every report row, so it should be
    stable across runs — a family whose ids move cannot be re-accounted.

    Misleads: ``horizon_days`` is NOT here. Horizons are a dimension of the
    family, supplied to ``score_chain``/``score_all``, because a candidate
    tested at four horizons is four tests and must count as four. A candidate
    that carried its own horizon would make that miscount easy.
    """

    event_source: str
    event_obs_type: str
    event_field: str
    z_threshold: float
    direction: str
    target_entity_type: str
    target_obs_type: str = "instrument_daily"
    target_value_field: str = "close"
    publication_lag_s: float = 0.0
    link_types: tuple[str, ...] | None = None
    event_entity_type: str | None = None
    metapath: tuple[str, str, str] | None = None
    chain_id: str = ""
    label: str = ""
    provenance: str = ""

    def __post_init__(self) -> None:
        if not self.chain_id:
            object.__setattr__(self, "chain_id", self.auto_id())
        if not self.label:
            object.__setattr__(self, "label", self.describe())

    def auto_id(self) -> str:
        """A stable id derived from the specification itself.

        Two generators that emit the same chain get the same id, which is what
        makes ``relook_warnings`` able to say "same family" at all.
        """
        parts = (
            self.event_source,
            self.event_obs_type,
            self.event_field,
            f"{float(self.z_threshold):g}",
            self.direction,
            self.target_entity_type,
            self.target_obs_type,
            self.target_value_field,
            f"{float(self.publication_lag_s):g}",
            "|".join(self.link_types or ()),
        )
        digest = hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:10]
        return f"{self.event_source}.{self.event_field}.{self.direction}{float(self.z_threshold):g}.{digest}"

    def describe(self) -> str:
        links = ",".join(self.link_types) if self.link_types else "any-link"
        return (
            f"{self.event_source}/{self.event_obs_type}/{self.event_field} "
            f"[{self.direction} z {float(self.z_threshold):g}] --{links}--> "
            f"{self.target_entity_type}.{self.target_obs_type}.{self.target_value_field}"
        )

    def hypothesis(self, horizon_days: int) -> Hypothesis:
        """The ``Hypothesis`` this candidate becomes at one horizon.

        Every knob is passed explicitly. ``Hypothesis.__post_init__`` does the
        validation (direction, z >= 0, horizon >= 1, lag >= 0), so an invalid
        candidate raises here rather than producing a study of nothing.
        """
        return Hypothesis(
            event_source=self.event_source,
            event_obs_type=self.event_obs_type,
            event_field=self.event_field,
            z_threshold=float(self.z_threshold),
            direction=self.direction,  # type: ignore[arg-type]
            target_entity_type=self.target_entity_type,
            target_obs_type=self.target_obs_type,
            horizon_days=int(horizon_days),
            publication_lag_s=float(self.publication_lag_s),
            target_value_field=self.target_value_field,
            link_types=tuple(self.link_types) if self.link_types else None,
            label=f"{self.label} @ {int(horizon_days)} steps",
        )


_REQUIRED_CANDIDATE_FIELDS = (
    "event_source",
    "event_obs_type",
    "event_field",
    "z_threshold",
    "direction",
    "target_entity_type",
)

_OPTIONAL_CANDIDATE_FIELDS = (
    "target_obs_type",
    "target_value_field",
    "publication_lag_s",
    "link_types",
    "event_entity_type",
    "metapath",
    "chain_id",
    "label",
    "provenance",
)


def coerce_candidate(obj: Any) -> ChainCandidate:
    """Read a ``ChainCandidate`` out of a mapping or any attribute-carrying object.

    Accepts a ``ChainCandidate`` unchanged, a ``Mapping``, or anything with the
    required attribute names — so a concurrently-written generator does not
    have to import this module's dataclass.

    Raises ``CandidateError`` naming EVERY missing required field at once, and
    naming an unknown key in a mapping rather than ignoring it. A generator
    that misspells ``event_field`` as ``field`` must find out immediately: a
    silently-defaulted specification produces a study of the wrong thing and
    still prints a p-value.

    Misleads: it validates *shape*, not sense. ``event_field="close"`` on a
    GDELT observation type is well-formed and will simply grade zero events.
    """
    if isinstance(obj, ChainCandidate):
        return obj

    if isinstance(obj, Mapping):
        present = dict(obj)
        unknown = sorted(set(present) - set(_REQUIRED_CANDIDATE_FIELDS) - set(_OPTIONAL_CANDIDATE_FIELDS))
        if unknown:
            raise CandidateError(
                f"unknown candidate key(s) {unknown}; known keys are "
                f"{sorted(_REQUIRED_CANDIDATE_FIELDS + _OPTIONAL_CANDIDATE_FIELDS)}. "
                "Refusing to ignore a key: a misspelled field becomes a silent default."
            )
        getter = present.get
        has = present.__contains__
    else:

        def getter(name: str, default: Any = None) -> Any:
            return getattr(obj, name, default)

        def has(name: str) -> bool:
            return hasattr(obj, name)

    missing = [f for f in _REQUIRED_CANDIDATE_FIELDS if not has(f) or getter(f) is None]
    if missing:
        raise CandidateError(
            f"candidate is missing required field(s) {missing} "
            f"(got a {type(obj).__name__}). Required: {list(_REQUIRED_CANDIDATE_FIELDS)}."
        )

    kwargs: dict[str, Any] = {f: getter(f) for f in _REQUIRED_CANDIDATE_FIELDS}
    for f in _OPTIONAL_CANDIDATE_FIELDS:
        if has(f) and getter(f) is not None:
            kwargs[f] = getter(f)

    if "link_types" in kwargs:
        lt = kwargs["link_types"]
        if isinstance(lt, str):
            raise CandidateError(
                f"link_types must be a sequence of link types, got the string {lt!r}; "
                "a bare string would be iterated character by character."
            )
        kwargs["link_types"] = tuple(str(x) for x in lt) or None
    if "metapath" in kwargs:
        mp = tuple(str(x) for x in kwargs["metapath"])
        if len(mp) != 3:
            raise CandidateError(f"metapath must be (src_type, edge_type, dst_type); got {mp!r}")
        kwargs["metapath"] = mp

    try:
        kwargs["z_threshold"] = float(kwargs["z_threshold"])
    except (TypeError, ValueError) as exc:
        raise CandidateError(f"z_threshold must be a real number, got {kwargs['z_threshold']!r}") from exc
    if "publication_lag_s" in kwargs:
        kwargs["publication_lag_s"] = float(kwargs["publication_lag_s"])

    try:
        return ChainCandidate(**kwargs)
    except (TypeError, ValueError) as exc:
        raise CandidateError(f"candidate rejected: {exc}") from exc


_ENUMERATED_ALIASES = {
    "event_source": "source_tool",
    "event_obs_type": "observation_type",
    "target_entity_type": "target_entity_type",
}


def from_enumerated(
    enumerated: Any,
    *,
    event_field: str,
    z_threshold: float,
    direction: str,
    target_obs_type: str = "instrument_daily",
    target_value_field: str = "close",
    publication_lag_s: float | None = None,
) -> tuple[ChainCandidate, list[str]]:
    """Adapt a route-level candidate from ``agent.ghost.enumerate`` into a testable one.

    The generator emits a chain as a ROUTE: a specific source entity, a specific
    target entity, a link-type sequence and a declared lag window. The event
    study is stated over a CHANNEL: a source tool, an observation type, a
    numeric field, a z-threshold and a direction. The two do not carry the same
    information, so this adapter maps what it can and returns, alongside the
    candidate, the list of things it had to widen or invent. Nothing is silent.

    What it maps
        ``source_tool`` -> ``event_source``, ``observation_type`` ->
        ``event_obs_type``, ``target_entity_type``, ``link_types``,
        ``source_entity_type`` -> ``event_entity_type``, and the first hop as
        the ``metapath``.

    What the CALLER must supply, because a route does not contain it
        ``event_field``, ``z_threshold`` and ``direction``. These are extra
        dimensions of the search: one route tested over 5 fields x 5 thresholds
        x 3 directions is 75 tests, and they all belong in the same BH family.

    WHERE THIS LOSES INFORMATION — both losses are returned as warnings
        * **The entity restriction is lost.** ``agent.verify.study.Hypothesis``
          has no single-entity scope, so scoring this candidate tests EVERY
          entity carrying that channel against EVERY linked target of that
          type, not the specific route the generator found. The verdict is
          therefore about the channel-and-link-type, and is strictly weaker
          than a claim about ``source_entity -> target_entity``.
        * **A multi-hop route collapses to its link types.** ``verify`` walks
          one hop of ``entity_links``, so a 2-hop route cannot be tested as a
          chain; the ``link_types`` are passed through as a filter and the
          intermediate node is not enforced.

    The lag window maps to ``publication_lag_s = lag_low_days`` (entry is
    delayed to the start of the declared window) and the horizon is
    ``lag_high_days - lag_low_days``, available from
    ``horizon_from_lag_window``. Note the unit mismatch: the window is CALENDAR
    days and ``horizon_days`` is SERIES STEPS, so on a daily close series the
    horizon is trading days and is shorter in wall-clock time than the window.

    Returns
    -------
    (ChainCandidate, warnings)
        The warnings are not decoration. Discard them and the board will report
        a channel-level verdict as if it were a route-level one.
    """

    def get(name: str, default: Any = None) -> Any:
        if isinstance(enumerated, Mapping):
            return enumerated.get(name, default)
        return getattr(enumerated, name, default)

    missing = [src for dst, src in _ENUMERATED_ALIASES.items() if get(src) is None]
    if missing:
        raise CandidateError(
            f"not an enumerated chain candidate: missing {missing}. Expected the field names of "
            "agent.ghost.enumerate.ChainCandidate (source_tool, observation_type, target_entity_type)."
        )

    link_types = tuple(str(x) for x in (get("link_types") or ()))
    path_types = tuple(str(x) for x in (get("path_types") or ()))
    hops = int(get("hops") or len(link_types) or 1)
    low = get("lag_low_days")
    lag = float(publication_lag_s) if publication_lag_s is not None else (float(low) * 86400.0 if low else 0.0)

    warnings = [
        "entity restriction LOST: verify() has no single-entity scope, so this tests the whole "
        f"{get('source_tool')}/{get('observation_type')} channel against every linked "
        f"{get('target_entity_type')}, not {get('source_entity')!r} -> {get('target_entity')!r}.",
        f"event_field={event_field!r}, z_threshold={z_threshold!r} and direction={direction!r} were supplied by "
        "the caller, not by the route. Every combination you try is another member of the BH family.",
    ]
    if hops > 1:
        warnings.append(
            f"{hops}-hop route collapsed to a link_types filter {link_types}: verify() walks ONE hop, so the "
            "intermediate node is not enforced and this is a weaker claim than the route."
        )
    if publication_lag_s is None and not low:
        warnings.append(
            "publication_lag_s defaulted to 0 because the route declared no lag_low_days; assert the real "
            "release delay instead — measuring from an as-of stamp grants free lookahead."
        )

    cand = ChainCandidate(
        event_source=str(get("source_tool")),
        event_obs_type=str(get("observation_type")),
        event_field=str(event_field),
        z_threshold=float(z_threshold),
        direction=str(direction),
        target_entity_type=str(get("target_entity_type")),
        target_obs_type=target_obs_type,
        target_value_field=target_value_field,
        publication_lag_s=lag,
        link_types=link_types or None,
        event_entity_type=(str(get("source_entity_type")) if get("source_entity_type") else None),
        metapath=(
            (str(path_types[0]), link_types[0], str(path_types[1])) if len(path_types) >= 2 and link_types else None
        ),
        provenance=f"from_enumerated({get('chain_id')})",
    )
    return cand, warnings


def horizon_from_lag_window(enumerated: Any) -> int:
    """``lag_high_days - lag_low_days`` as a horizon in SERIES STEPS, at least 1.

    Misleads: the window is calendar days and the horizon is series steps, so a
    21-calendar-day window becomes 21 trading days — about 29 calendar days on a
    daily close series. Use it as a starting grid, not as a faithful conversion.
    """

    def get(name: str) -> Any:
        if isinstance(enumerated, Mapping):
            return enumerated.get(name)
        return getattr(enumerated, name, None)

    low = int(get("lag_low_days") or 0)
    high = int(get("lag_high_days") or 0)
    return max(high - low, 1)


# --------------------------------------------------------------------------- #
# 2. one tested cell
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ChainCell:
    """One candidate at one horizon: the atom of the family, and one test.

    A cell is a *test*, and the family is every cell. Everything a reader needs
    in order to disbelieve the number is on the object: the sample size, the
    cluster count that the sample size overstates, the stipulated power, both
    p-values, the effect against the unconditional baseline, the bootstrap CI,
    and the attrition.

    ``p_adjusted`` and ``survived`` are ``None`` until ``score_all`` has
    corrected a family. That is not laziness — a single cell has no family, and
    a ``survived`` flag on it would be the exact lie this module exists to
    prevent.
    """

    chain_id: str
    label: str
    horizon_days: int

    # --- sample ----------------------------------------------------------- #
    n_events: int
    n_event_clusters: int
    effective_clusters: int
    max_cluster_size: int
    pseudo_replication: float
    cluster_window_s: float
    cluster_origin: float
    n_baseline: int
    n_pairs_used: int
    n_pairs_entered: int

    # --- effect ----------------------------------------------------------- #
    mean_event_return: float
    baseline_mean_return: float
    edge: float
    ci_low: float
    ci_high: float
    hit_rate_event: float
    hit_rate_baseline: float

    # --- inference -------------------------------------------------------- #
    p_uncorrected: float
    p_iid: float
    p_clustered: float
    p_basis: str
    power: float
    power_if_independent: float
    power_reference: str
    p_resolution: float
    """Smallest non-zero p the bootstrap can express: 1 / N_BOOTSTRAP.

    ``verify`` runs 2000 resamples, so a reported p of exactly 0.0 means
    "p < 5e-4", not zero. Quoting it as 0 is a claim the estimator cannot make.
    """

    # --- accounting ------------------------------------------------------- #
    testable: bool
    untestable_reason: str | None
    reconciled: bool
    complete: bool
    study_verdict: str
    drop_reasons: tuple[tuple[str, int], ...]
    excluded: tuple[str, ...]
    error: str | None = None

    # --- filled in only by score_all, across the whole family ------------- #
    p_adjusted: float | None = None
    survived: bool | None = None
    family_size: int | None = None
    bh_denominator: int | None = None

    @property
    def verdict(self) -> str:
        """The single word to put in front of a customer.

        ``untestable`` beats everything: a cell with 4 events has no verdict to
        give, and calling that "not supported" would be claiming evidence of
        absence from nothing. ``survived_underpowered`` exists because a
        survivor at 8% power is closer to the false-positive rate than to
        evidence, and this repository has already shipped that confusion once.
        ``uncorrected_only`` is what an un-BH'd cell gets — never "supported".
        """
        if self.error is not None:
            return "error"
        if not self.testable:
            return "untestable"
        if self.survived is None:
            return "uncorrected_only" if self.p_uncorrected < 0.05 else "not_supported"
        if not self.survived:
            return "not_supported"
        return "survived_underpowered" if self.power < 0.5 else "survived"

    def row(self) -> str:
        """One fixed-width line. Power and n are never printed without each other."""
        padj = "     -" if self.p_adjusted is None else f"{self.p_adjusted:6.4f}"
        return (
            f"{self.chain_id[:44]:<44} h={self.horizon_days:>3} "
            f"n={self.n_events:>5} clus={self.effective_clusters:>4} "
            f"edge={self.edge * 100:+7.3f}% "
            f"CI=[{self.ci_low * 100:+7.3f},{self.ci_high * 100:+7.3f}] "
            f"p={self.p_uncorrected:6.4f} padj={padj} pow={self.power:5.1%} "
            f"{self.verdict}"
        )


@dataclass(frozen=True)
class ChainResult:
    """One candidate across every horizon it was tested at.

    ``score_chain`` returns this. Note that ``len(cells)`` tests were run, and
    that this object carries no survival flags: correcting a single candidate's
    horizons as if they were the whole family is exactly the miscount that
    turns 3-of-40,000 into "3 chains found".
    """

    candidate: ChainCandidate
    cells: tuple[ChainCell, ...]
    db_path: str
    as_of: float | None
    studies: tuple[StudyResult, ...] = ()

    @property
    def n_tests(self) -> int:
        return len(self.cells)

    def summary(self) -> str:
        head = f"CANDIDATE {self.candidate.chain_id}\n  {self.candidate.describe()}\n  {self.n_tests} test(s), uncorrected:"
        return "\n".join([head, *(f"  {c.row()}" for c in self.cells)])


# --------------------------------------------------------------------------- #
# 3. scoring one candidate
# --------------------------------------------------------------------------- #


def _cluster_diagnostics(
    cluster_ts: Sequence[float],
    *,
    window: float,
) -> dict[str, Any]:
    """Bucket event timestamps and report how many genuinely distinct times they are.

    Thin wrapper over ``agent.verify.stats.effective_sample_size`` that pins
    ``origin`` to the FIRST event timestamp. The raw epoch grid starts on a
    Thursday (the epoch was a Thursday) and the bucket phase changes the
    cluster count, so leaving ``origin`` at 0.0 makes the number depend on an
    accident of 1970. Pinning it to the first event makes the count
    reproducible and translation-invariant; it is returned so it can be quoted.

    Returns a zeroed dict with ``n_clusters=0`` for an empty input rather than
    raising, because "this cell graded no events" is a normal outcome of a
    40,000-cell search and must not abort the search.
    """
    ts = [float(t) for t in cluster_ts if math.isfinite(float(t))]
    if not ts:
        return {
            "n_observations": 0,
            "n_clusters": 0,
            "max_cluster_size": 0,
            "pseudo_replication": float("nan"),
            "window": float(window),
            "origin": 0.0,
        }
    origin = min(ts)
    out = effective_sample_size(ts, window=float(window), origin=origin)
    return out


def _untestable_reason(
    study: StudyResult,
    *,
    effective_clusters: int,
    min_events: int,
    min_clusters: int,
) -> str | None:
    """Why this cell cannot be tested, or ``None`` if it can.

    Order matters: the most basic failure is named first, so a cell with 0
    events is reported as ``no_events`` rather than as a cluster shortfall.
    Every branch returns a member of ``UNTESTABLE_REASONS``; there is no
    catch-all, because "untestable for some reason" is not a reportable bucket.
    """
    if study.n_events <= 0:
        return "no_events"
    if study.n_events < MIN_TESTABLE_EVENTS:
        return "below_verify_bootstrap_minimum"
    if study.n_events < min_events:
        return "n_events_below_minimum"
    if effective_clusters < min_clusters:
        return "effective_clusters_below_minimum"
    if not math.isfinite(study.p_value) or not math.isfinite(study.p_value_clustered):
        return "p_value_not_finite"
    if not study.reconciled:
        return "bookkeeping_did_not_reconcile"
    return None


_POWER_REFERENCE = (
    "stipulated: power_two_sided(n_clusters), reference t=2.14 at n=1451 "
    "(docs/publications/cot_null_result.md). NOT post-hoc power."
)


def _cell_from_study(
    candidate: ChainCandidate,
    study: StudyResult,
    *,
    horizon_days: int,
    cluster_window_s: float,
    p_basis: str,
    min_events: int,
    min_clusters: int,
) -> ChainCell:
    diag = _cluster_diagnostics(study.event_cluster_ts, window=cluster_window_s)
    effective_clusters = int(diag["n_clusters"])
    reason = _untestable_reason(
        study,
        effective_clusters=effective_clusters,
        min_events=min_events,
        min_clusters=min_clusters,
    )
    p_iid = float(study.p_value)
    p_clustered = float(study.p_value_clustered)
    p_used = p_clustered if p_basis == "clustered" else p_iid
    return ChainCell(
        chain_id=candidate.chain_id,
        label=candidate.label,
        horizon_days=int(horizon_days),
        n_events=int(study.n_events),
        n_event_clusters=int(study.n_event_clusters),
        effective_clusters=effective_clusters,
        max_cluster_size=int(diag["max_cluster_size"]),
        pseudo_replication=float(diag["pseudo_replication"]),
        cluster_window_s=float(cluster_window_s),
        cluster_origin=float(diag["origin"]),
        n_baseline=int(study.n_baseline),
        n_pairs_used=int(study.n_pairs_used),
        n_pairs_entered=int(study.n_pairs_entered),
        mean_event_return=float(study.mean_event_return),
        baseline_mean_return=float(study.baseline_mean_return),
        edge=float(study.edge),
        ci_low=float(study.ci_low),
        ci_high=float(study.ci_high),
        hit_rate_event=float(study.hit_rate_event),
        hit_rate_baseline=float(study.hit_rate_baseline),
        p_uncorrected=p_used,
        p_iid=p_iid,
        p_clustered=p_clustered,
        p_basis=p_basis,
        power=float(study.power),
        power_if_independent=float(study.power_n_events),
        power_reference=_POWER_REFERENCE,
        p_resolution=1.0 / float(N_BOOTSTRAP),
        testable=reason is None,
        untestable_reason=reason,
        reconciled=bool(study.reconciled),
        complete=bool(study.complete),
        study_verdict=study.verdict,
        drop_reasons=tuple(sorted(study.drop_reasons.items(), key=lambda kv: (-kv[1], kv[0]))),
        excluded=tuple(study.excluded),
    )


def _error_cell(
    candidate: ChainCandidate,
    *,
    horizon_days: int,
    cluster_window_s: float,
    p_basis: str,
    message: str,
) -> ChainCell:
    """A cell that could not be run at all.

    It still occupies a slot in the family. A candidate that crashed is a
    candidate that was attempted, and quietly dropping it would shrink m.
    Its p is NaN, so it can never enter the BH numerator; it is counted as
    untestable with the reason carried verbatim in ``error``.
    """
    nan = float("nan")
    return ChainCell(
        chain_id=candidate.chain_id,
        label=candidate.label,
        horizon_days=int(horizon_days),
        n_events=0,
        n_event_clusters=0,
        effective_clusters=0,
        max_cluster_size=0,
        pseudo_replication=nan,
        cluster_window_s=float(cluster_window_s),
        cluster_origin=0.0,
        n_baseline=0,
        n_pairs_used=0,
        n_pairs_entered=0,
        mean_event_return=nan,
        baseline_mean_return=nan,
        edge=nan,
        ci_low=nan,
        ci_high=nan,
        hit_rate_event=nan,
        hit_rate_baseline=nan,
        p_uncorrected=nan,
        p_iid=nan,
        p_clustered=nan,
        p_basis=p_basis,
        power=nan,
        power_if_independent=nan,
        power_reference=_POWER_REFERENCE,
        p_resolution=1.0 / float(N_BOOTSTRAP),
        testable=False,
        untestable_reason="no_events",
        reconciled=False,
        complete=False,
        study_verdict="error",
        drop_reasons=(),
        excluded=(f"study raised: {message}",),
        error=message,
    )


def score_chain(
    candidate: Any,
    db_path: str,
    *,
    horizons: Sequence[int],
    as_of: float | None = None,
    cluster_window_s: float = DEFAULT_CLUSTER_WINDOW_S,
    p_basis: str = "clustered",
    min_events: int = UNTESTABLE_MIN_EVENTS,
    min_clusters: int = UNTESTABLE_MIN_CLUSTERS,
    keep_studies: bool = False,
) -> ChainResult:
    """Run the event study for one candidate at every horizon. No correction.

    The event study is NOT reimplemented here: each (candidate, horizon) pair
    becomes an ``agent.verify.study.Hypothesis`` and is handed to
    ``agent.verify.study.verify``, which opens the database ``mode=ro``,
    computes causal z-scores against prior points only, enters every gradable
    point into the UNCONDITIONAL baseline, and reports its own attrition.

    Per cell you get: ``n_events``; the cluster count that ``n_events``
    overstates (``effective_clusters``, buckets of ``cluster_window_s``);
    stipulated ``power`` at that cluster count; the uncorrected p (clustered by
    default, i.i.d. also carried); the event-versus-unconditional-baseline
    ``edge``; and the circular-block-bootstrap CI.

    Parameters
    ----------
    p_basis
        Which p-value enters ``p_uncorrected`` and therefore the family:
        ``"clustered"`` (resamples event timestamps — the honest default, and
        typically several times the i.i.d. figure) or ``"iid"``. Both are
        always recorded, and the basis is stamped on every cell so a board
        cannot be compared against one built on the other basis.
    keep_studies
        Keep the full ``StudyResult`` objects on the result. Off by default
        because a 40,000-cell family of them does not fit in memory; turn it on
        for the handful of cells you intend to publish, where the leakage audit
        and pair attrition are the deliverable.

    Returns
    -------
    ChainResult
        With ``p_adjusted`` and ``survived`` ``None`` on every cell. A single
        candidate is not a family.

    Raises
    ------
    CandidateError
        For an unreadable candidate, an empty/invalid ``horizons``, a duplicate
        horizon (the same test twice would double-count in the family), or a
        ``p_basis`` outside ``{"clustered", "iid"}``.

        A failure INSIDE the study is not raised: it becomes a cell with
        ``error`` set, so that one bad candidate cannot silently shrink a
        40,000-cell family.
    """
    cand = coerce_candidate(candidate)
    if p_basis not in _P_BASES:
        raise CandidateError(f"p_basis must be one of {_P_BASES}, got {p_basis!r}")
    hs = [int(h) for h in horizons]
    if not hs:
        raise CandidateError("horizons is empty: a candidate tested at no horizon is not a test.")
    if len(set(hs)) != len(hs):
        raise CandidateError(
            f"duplicate horizon(s) in {hs}: the same test would enter the family twice and inflate the BH denominator."
        )
    if any(h < 1 for h in hs):
        raise CandidateError(f"horizons must all be >= 1 series steps, got {hs}")

    cells: list[ChainCell] = []
    studies: list[StudyResult] = []
    for h in hs:
        try:
            study = verify(db_path, cand.hypothesis(h), as_of=as_of)
        except Exception as exc:  # noqa: BLE001 - the cell must survive to keep m honest
            cells.append(
                _error_cell(
                    cand,
                    horizon_days=h,
                    cluster_window_s=cluster_window_s,
                    p_basis=p_basis,
                    message=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        cells.append(
            _cell_from_study(
                cand,
                study,
                horizon_days=h,
                cluster_window_s=cluster_window_s,
                p_basis=p_basis,
                min_events=min_events,
                min_clusters=min_clusters,
            )
        )
        if keep_studies:
            studies.append(study)

    return ChainResult(
        candidate=cand,
        cells=tuple(cells),
        db_path=str(db_path),
        as_of=as_of,
        studies=tuple(studies),
    )


# --------------------------------------------------------------------------- #
# 4. the family
# --------------------------------------------------------------------------- #


def _db_fingerprint(db_path: str, as_of: float | None = None) -> str:
    """Size, mtime, row counts AND the as_of cutoff, so a second look is detectable.

    ``as_of`` is part of the fingerprint because it is part of the EVIDENCE
    WINDOW. Re-running one family at a later cutoff is the repository's own
    documented failure — the CFTC family gave 0 of 51 on the first look and
    "3 of 54 survivors" at 5.1-7.6% power when re-run on more data — and
    without the cutoff here, two boards over the same file would compare as
    "identical data, no new evidence" while actually being two looks.

    Not a content hash: hashing a multi-gigabyte SQLite file on every board
    would dominate the runtime. Row counts are read ``mode=ro``. A fingerprint
    that cannot be taken returns ``"unavailable:<reason>"`` rather than a
    plausible-looking empty string.
    """
    try:
        st = os.stat(db_path)
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            counts = [
                con.execute(f"select count(*) from {t}").fetchone()[0]  # noqa: S608 - literal table names
                for t in ("entities", "entity_observations", "entity_links")
            ]
        finally:
            con.close()
        window = "all" if as_of is None else f"{as_of:.0f}"
        return f"size={st.st_size}|mtime={int(st.st_mtime)}|rows={'/'.join(str(c) for c in counts)}|as_of={window}"
    except Exception as exc:  # noqa: BLE001
        window = "all" if as_of is None else f"{as_of:.0f}"
        return f"unavailable:{type(exc).__name__}|as_of={window}"


@dataclass(frozen=True)
class ScoreBoard:
    """A whole family of tests, corrected once, with the family size attached.

    ``n_tested`` is the FULL family: every candidate times every horizon that
    was run, including the cells that turned out to be untestable and the cells
    that crashed. It is the product. "3 of 40,000" is honest and defensible;
    "3 chains found" from the same run is a lie by omission.

    Two corrections are always computed:

    ``bh``
        over the testable cells only — the reference study's behaviour, and the
        one whose survivors ``survivors`` lists.
    ``bh_conservative``
        over the full family with every untestable cell entered at p=1.0. A
        survivor that appears in ``bh`` but not here is being carried by the
        exclusion rather than by the data, and ``warnings`` says so.

    Misleads: the board can only correct the cells it was handed. If the
    generator filtered candidates before scoring, set ``n_generated`` so the
    report prints "of N generated"; nothing here can detect a family that was
    trimmed upstream.
    """

    cells: tuple[ChainCell, ...]
    results: tuple[ChainResult, ...]
    alpha: float
    p_basis: str
    cluster_window_s: float
    min_events: int
    min_clusters: int
    horizons: tuple[int, ...]
    n_candidates: int
    n_generated: int | None
    db_path: str
    db_fingerprint: str
    as_of: float | None
    family_id: str
    bh: Any
    bh_conservative: Any
    bh_excludes_untestable: bool

    # ---- the counts that are the product --------------------------------- #
    @property
    def n_tested(self) -> int:
        """The FULL family size: every candidate x every horizon actually run."""
        return len(self.cells)

    @property
    def n_testable(self) -> int:
        return sum(1 for c in self.cells if c.testable)

    @property
    def n_untestable(self) -> int:
        return sum(1 for c in self.cells if not c.testable)

    @property
    def n_errors(self) -> int:
        return sum(1 for c in self.cells if c.error is not None)

    @property
    def untestable_reasons(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.cells:
            if not c.testable:
                out[c.untestable_reason or "unclassified"] = out.get(c.untestable_reason or "unclassified", 0) + 1
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))

    @property
    def survivors(self) -> tuple[ChainCell, ...]:
        """Cells that survived BH over the testable subset, strongest first."""
        return tuple(sorted((c for c in self.cells if c.survived), key=lambda c: c.p_uncorrected))

    @property
    def survivors_conservative(self) -> tuple[ChainCell, ...]:
        """Survivors when untestable cells are entered at p=1.0 instead of dropped."""
        ids = self._conservative_survivor_keys
        return tuple(
            sorted((c for c in self.cells if (c.chain_id, c.horizon_days) in ids), key=lambda c: c.p_uncorrected)
        )

    @property
    def n_would_survive_uncorrected(self) -> int:
        """How many testable cells have p < alpha with NO correction at all.

        The gap between this and ``len(survivors)`` is the multiple-testing
        problem stated as a number. In the CFTC study it was 9 versus 0.
        """
        return sum(1 for c in self.cells if c.testable and c.p_uncorrected < self.alpha)

    @property
    def n_underpowered_survivors(self) -> int:
        return sum(1 for c in self.survivors if not math.isfinite(c.power) or c.power < 0.5)

    _conservative_survivor_keys: frozenset = frozenset()

    # ---- reporting -------------------------------------------------------- #
    def warnings(self) -> list[str]:
        """Everything that should stop a reader believing the survivors.

        Emitted as data rather than printed, so a caller cannot format the
        board without having had the chance to format these.
        """
        w: list[str] = []
        if self.n_tested == 0:
            return ["empty family: nothing was tested, so nothing survived and nothing failed."]
        surv = self.survivors
        if self.n_untestable:
            w.append(
                f"{self.n_untestable} of {self.n_tested} cells ({self.n_untestable / self.n_tested:.1%}) were "
                f"UNTESTABLE (n_events < {self.min_events} or effective clusters < {self.min_clusters}); "
                f"they are excluded from the BH denominator of {self.bh.n_tested}, which makes survival easier."
            )
        cons = {(c.chain_id, c.horizon_days) for c in self.survivors_conservative}
        lost = [c for c in surv if (c.chain_id, c.horizon_days) not in cons]
        if lost:
            w.append(
                f"{len(lost)} survivor(s) do NOT survive the conservative correction over the full family of "
                f"{self.bh_conservative.n_tested}: {[c.chain_id for c in lost]}. Those are carried by the "
                "untestable-cell exclusion, not by the data."
            )
        if surv and self.n_underpowered_survivors:
            w.append(
                f"{self.n_underpowered_survivors} of {len(surv)} survivor(s) have power below 50% "
                f"(min {min(c.power for c in surv):.1%}). At that power a 'significant' result is closer to the "
                "false-positive rate than to evidence; this repository has published that mistake once."
            )
        tiny = [c for c in surv if c.effective_clusters < 3 * self.min_clusters]
        if tiny:
            w.append(
                f"{len(tiny)} of {len(surv)} survivor(s) sit just above the testability floor "
                f"(effective clusters < {3 * self.min_clusters}); their p-values have coarse resolution."
            )
        floored = [c for c in surv if c.p_uncorrected <= 0.0]
        if floored:
            w.append(
                f"{len(floored)} survivor(s) report p == 0.0, which is the bootstrap RESOLUTION FLOOR "
                f"({N_BOOTSTRAP} resamples): the honest statement is p < {1.0 / N_BOOTSTRAP:g}, not 0."
            )
        unrec = sum(1 for c in self.cells if c.testable and not c.reconciled)
        if unrec:
            w.append(f"{unrec} testable cell(s) did not reconcile their own bookkeeping; their numbers are unusable.")
        if self.n_errors:
            w.append(f"{self.n_errors} cell(s) raised and were counted as untestable, keeping the family size honest.")
        if self.n_generated is not None and self.n_generated > self.n_tested:
            w.append(
                f"{self.n_generated - self.n_tested} generated candidate-cells were filtered out before "
                f"scoring; BH corrects only the {self.n_tested} that were run, so every adjusted p-value "
                "here is optimistic by that much."
            )
        w.append(
            f"BH corrects within ONE look. This board is look 1 over family {self.family_id} at "
            f"db {self.db_fingerprint}. Re-scoring on new data is a second look and is not covered."
        )
        return w

    def variation_report(self) -> str:
        """Confirm the numbers actually VARY across the family (LESSONS F-18).

        The discovery engine this module replaces ranked meta-paths by an
        attention score that came back as uniform 1/k fractions from an
        untrained network — 9 of 20 edge types at exactly 0.0 — and nobody
        noticed, because a constant ranked first looks exactly like a winner.
        A frozen model was once ranked first here for having a good random
        initialisation.

        So: print the distinct-value count and the full range of every column
        that a verdict depends on, and say DEGENERATE when a column takes one
        value across more than one cell. A degenerate p-column is a broken
        engine, not a finding.
        """
        lines = [f"VARIATION over {self.n_tested} cells (family {self.family_id})"]
        cols: list[tuple[str, list[float]]] = [
            ("p_uncorrected", [c.p_uncorrected for c in self.cells]),
            ("p_iid", [c.p_iid for c in self.cells]),
            ("p_clustered", [c.p_clustered for c in self.cells]),
            ("edge", [c.edge for c in self.cells]),
            ("n_events", [float(c.n_events) for c in self.cells]),
            ("effective_clusters", [float(c.effective_clusters) for c in self.cells]),
            ("power", [c.power for c in self.cells]),
        ]
        for name, raw in cols:
            vals = np.asarray([v for v in raw if math.isfinite(v)], dtype=float)
            if vals.size == 0:
                lines.append(f"  {name:<20} NO FINITE VALUES (all {len(raw)} cells non-finite)")
                continue
            uniq = np.unique(np.round(vals, 12))
            flag = "DEGENERATE" if uniq.size == 1 and vals.size > 1 else "ok"
            lines.append(
                f"  {name:<20} finite={vals.size:<6} distinct={uniq.size:<6} "
                f"min={vals.min():<12.6g} med={float(np.median(vals)):<12.6g} max={vals.max():<12.6g} "
                f"std={float(vals.std()):<12.6g} {flag}"
            )
        return "\n".join(lines)

    def report(self, *, top: int = 20) -> str:
        """The customer-facing block. It is impossible to print without n_tested."""
        gen = "" if self.n_generated is None else f" (of {self.n_generated} generated)"
        head = [
            "GHOST CHAIN SCOREBOARD",
            f"  family            {self.family_id}",
            f"  db                {self.db_path}  [{self.db_fingerprint}]",
            f"  as_of             {'whole history' if self.as_of is None else f'{self.as_of:.0f}'}",
            f"  TESTED            {self.n_tested} cells{gen} "
            f"= {self.n_candidates} candidates x {len(self.horizons)} horizons {list(self.horizons)}",
            f"  testable          {self.n_testable}   untestable {self.n_untestable}   errors {self.n_errors}",
            f"  p basis           {self.p_basis} (alpha={self.alpha}, cluster window "
            f"{self.cluster_window_s / 86400:g}d, floors n>={self.min_events} clusters>={self.min_clusters})",
            f"  SURVIVED BH       {len(self.survivors)} of {self.n_tested} tested "
            f"(BH denominator {self.bh.n_tested}); conservative over full family: "
            f"{len(self.survivors_conservative)} of {self.bh_conservative.n_tested}",
            f"  would 'work' uncorrected: {self.n_would_survive_uncorrected}",
        ]
        if self.untestable_reasons:
            head.append("  untestable by reason")
            head.extend(f"    {k:<38} {v}" for k, v in self.untestable_reasons.items())
        body = ["  SURVIVORS (BH over testable cells)"]
        if self.survivors:
            body.extend(f"    {c.row()}" for c in self.survivors[:top])
        else:
            body.append(f"    none. 0 of {self.n_tested} tested survived correction.")
        strongest = sorted((c for c in self.cells if c.testable), key=lambda c: c.p_uncorrected)[:top]
        body.append("  STRONGEST TESTABLE CELLS (uncorrected order — NOT discoveries)")
        body.extend(f"    {c.row()}" for c in strongest)
        warn = ["  WARNINGS"] + [f"    - {w}" for w in self.warnings()]
        return "\n".join(head + body + warn)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able summary. Includes ``n_tested`` unconditionally, by construction."""
        return {
            "family_id": self.family_id,
            "db_path": self.db_path,
            "db_fingerprint": self.db_fingerprint,
            "as_of": self.as_of,
            "n_tested": self.n_tested,
            "n_generated": self.n_generated,
            "n_candidates": self.n_candidates,
            "horizons": list(self.horizons),
            "alpha": self.alpha,
            "p_basis": self.p_basis,
            "cluster_window_s": self.cluster_window_s,
            "min_events": self.min_events,
            "min_clusters": self.min_clusters,
            "n_testable": self.n_testable,
            "n_untestable": self.n_untestable,
            "n_errors": self.n_errors,
            "untestable_reasons": self.untestable_reasons,
            "bh_denominator": self.bh.n_tested,
            "n_survivors": len(self.survivors),
            "n_survivors_conservative": len(self.survivors_conservative),
            "n_would_survive_uncorrected": self.n_would_survive_uncorrected,
            "n_underpowered_survivors": self.n_underpowered_survivors,
            "survivors": [
                {
                    "chain_id": c.chain_id,
                    "label": c.label,
                    "horizon_days": c.horizon_days,
                    "n_events": c.n_events,
                    "effective_clusters": c.effective_clusters,
                    "edge": c.edge,
                    "ci": [c.ci_low, c.ci_high],
                    "p_uncorrected": c.p_uncorrected,
                    "p_adjusted": c.p_adjusted,
                    "power": c.power,
                    "verdict": c.verdict,
                }
                for c in self.survivors
            ],
            "warnings": self.warnings(),
        }


def _family_id(candidates: Sequence[ChainCandidate], horizons: Sequence[int], alpha: float, p_basis: str) -> str:
    """Stable hash of the family SPECIFICATION (not of its results).

    Two runs with the same id tested the same family; if the ids differ, BH
    over one says nothing about the other. Candidate ids are sorted, so the
    generator's emission order cannot change the family identity.
    """
    payload = json.dumps(
        {
            "candidates": sorted(c.chain_id for c in candidates),
            "horizons": sorted(int(h) for h in horizons),
            "alpha": float(alpha),
            "p_basis": p_basis,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def score_all(
    candidates: Sequence[Any],
    db_path: str,
    *,
    horizons: Sequence[int],
    alpha: float = 0.05,
    as_of: float | None = None,
    cluster_window_s: float = DEFAULT_CLUSTER_WINDOW_S,
    p_basis: str = "clustered",
    min_events: int = UNTESTABLE_MIN_EVENTS,
    min_clusters: int = UNTESTABLE_MIN_CLUSTERS,
    n_generated: int | None = None,
    progress: Any = None,
) -> ScoreBoard:
    """Score every candidate at every horizon, then correct across the WHOLE family.

    The family is ``len(candidates) * len(horizons)`` cells and that number is
    ``ScoreBoard.n_tested``. Benjamini-Hochberg is applied ONCE, across every
    cell — not per candidate, not per horizon, not over the survivors.
    ``agent.verify.stats.benjamini_hochberg`` does the correction; it is not
    reimplemented here.

    Untestable cells (``n_events < min_events`` or fewer than ``min_clusters``
    effective clusters) are reported in their own bucket with a named reason.
    By default they are kept OUT of the BH denominator, which is what the
    reference study did and which makes survival easier; the board therefore
    also computes ``bh_conservative`` over the full family with those cells
    entered at p=1.0, and warns whenever a survivor exists only under the
    easier accounting.

    Parameters
    ----------
    n_generated
        The number of candidate-cells the generator produced BEFORE any
        pre-filtering, if that is larger than what you are handing in. Recorded
        and reported. This module cannot detect upstream trimming; supplying
        this is how you stay honest about it.
    progress
        Optional ``callable(i, n, candidate)`` invoked before each candidate, so
        a 40,000-cell run is observable. It is not allowed to influence
        anything.

    Returns
    -------
    ScoreBoard
        With ``p_adjusted``, ``survived``, ``family_size`` and
        ``bh_denominator`` filled in on every cell.

    Raises
    ------
    CandidateError
        For an empty candidate list, duplicate ``chain_id`` values (two
        identical tests in one family), an alpha outside (0, 1), or an invalid
        ``horizons``/``p_basis``.
    """
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise CandidateError(f"alpha must be strictly between 0 and 1, got {alpha!r}")
    if p_basis not in _P_BASES:
        raise CandidateError(f"p_basis must be one of {_P_BASES}, got {p_basis!r}")
    cands = [coerce_candidate(c) for c in candidates]
    if not cands:
        raise CandidateError(
            "no candidates: an empty family has no BH denominator, and returning a board "
            "with 0 survivors would read as 'nothing worked' rather than 'nothing was tried'."
        )
    ids = [c.chain_id for c in cands]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise CandidateError(
            f"duplicate chain_id(s) {dupes}: the same test would enter the family twice. "
            "Deduplicate before scoring, and count what you dropped."
        )

    hs = tuple(int(h) for h in horizons)
    results: list[ChainResult] = []
    for i, cand in enumerate(cands):
        if progress is not None:
            progress(i, len(cands), cand)
        results.append(
            score_chain(
                cand,
                db_path,
                horizons=hs,
                as_of=as_of,
                cluster_window_s=cluster_window_s,
                p_basis=p_basis,
                min_events=min_events,
                min_clusters=min_clusters,
            )
        )

    cells = [c for r in results for c in r.cells]
    family_size = len(cells)

    # ---- BH over the testable subset (reference behaviour) ---------------- #
    testable_idx = [i for i, c in enumerate(cells) if c.testable]
    testable_p = [cells[i].p_uncorrected for i in testable_idx]
    bh = benjamini_hochberg(testable_p, alpha=alpha) if testable_p else benjamini_hochberg([], alpha=alpha)

    # ---- BH over the FULL family, untestable cells entered at p=1.0 ------- #
    full_p = [c.p_uncorrected if c.testable and math.isfinite(c.p_uncorrected) else 1.0 for c in cells]
    bh_cons = benjamini_hochberg(full_p, alpha=alpha)
    cons_keys = frozenset((cells[i].chain_id, cells[i].horizon_days) for i in range(family_size) if bh_cons.rejected[i])

    adj: list[float | None] = [None] * family_size
    surv: list[bool | None] = [None] * family_size
    for slot, i in enumerate(testable_idx):
        adj[i] = float(bh.p_adjusted[slot])
        surv[i] = bool(bh.rejected[slot])
    # An untestable cell did not fail the test; it was not tested. `survived`
    # stays False (it is not a discovery) but `p_adjusted` stays None so it can
    # never be quoted as a corrected p-value.
    for i, c in enumerate(cells):
        if not c.testable:
            surv[i] = False

    corrected = tuple(
        replace(c, p_adjusted=adj[i], survived=surv[i], family_size=family_size, bh_denominator=bh.n_tested)
        for i, c in enumerate(cells)
    )
    by_key = {(c.chain_id, c.horizon_days): c for c in corrected}
    results = [replace(r, cells=tuple(by_key[(c.chain_id, c.horizon_days)] for c in r.cells)) for r in results]

    board = ScoreBoard(
        cells=corrected,
        results=tuple(results),
        alpha=float(alpha),
        p_basis=p_basis,
        cluster_window_s=float(cluster_window_s),
        min_events=int(min_events),
        min_clusters=int(min_clusters),
        horizons=hs,
        n_candidates=len(cands),
        n_generated=None if n_generated is None else int(n_generated),
        db_path=str(db_path),
        db_fingerprint=_db_fingerprint(str(db_path), as_of),
        as_of=as_of,
        family_id=_family_id(cands, hs, alpha, p_basis),
        bh=bh,
        bh_conservative=bh_cons,
        bh_excludes_untestable=BH_EXCLUDES_UNTESTABLE,
    )
    object.__setattr__(board, "_conservative_survivor_keys", cons_keys)
    return board


def relook_warnings(previous: ScoreBoard, current: ScoreBoard) -> list[str]:
    """Say out loud what re-scoring a family costs, since the correction cannot pay it.

    BH controls the false-discovery rate within ONE look at ONE family. There
    is no arithmetic that combines two looks after the fact — the alpha was
    spent — so this returns prose for the report rather than a corrected
    number, and there is deliberately no function here that merges two boards.

    Distinguishes the two cases that matter:

    * same ``family_id``, different ``db_fingerprint`` — the same search
      re-run on changed data. This is the CFTC re-run: 0 of 51 became
      "3 of 54 survivors" at 5-8% power. The second look is uncorrected.
    * different ``family_id`` — a different search. The two boards' adjusted
      p-values are not comparable (BH adjusted values are not comparable
      across families of different size), and if the second family was chosen
      after seeing the first, its alpha is spent too.
    """
    out: list[str] = []
    if previous.family_id == current.family_id and previous.db_fingerprint == current.db_fingerprint:
        out.append(
            f"identical family {current.family_id} on identical data — this is the same look, "
            "not a replication. It provides no new evidence."
        )
        return out
    if previous.family_id == current.family_id:
        window = ""
        if previous.as_of != current.as_of:
            window = (
                f" The as_of cutoff moved ({previous.as_of} -> {current.as_of}), so the second look sees "
                "evidence the first did not; that is the widening, and it is uncorrected."
            )
        out.append(
            f"SECOND LOOK at family {current.family_id}: same search, different data "
            f"({previous.db_fingerprint} -> {current.db_fingerprint}). BH corrects within one look; the "
            f"false-discovery rate across both looks is NOT {current.alpha}. Measured precedent: the CFTC "
            "family gave 0 of 51 on the first look and 3 of 54 'survivors' at 5.1-7.6% power on the second." + window
        )
    else:
        out.append(
            f"DIFFERENT family ({previous.family_id} -> {current.family_id}, "
            f"{previous.n_tested} -> {current.n_tested} cells). BH-adjusted p-values are not comparable "
            "across families of different size. If this family was chosen after reading the last board, "
            "the selection is part of the search and is uncorrected."
        )
    if previous.p_basis != current.p_basis:
        out.append(f"p basis changed ({previous.p_basis} -> {current.p_basis}); the two boards test different things.")
    if (previous.min_events, previous.min_clusters) != (current.min_events, current.min_clusters):
        out.append(
            f"testability floors changed (n>={previous.min_events}/clusters>={previous.min_clusters} -> "
            f"n>={current.min_events}/clusters>={current.min_clusters}); that moves cells into and out of the "
            "BH denominator and therefore moves every adjusted p."
        )
    return out


# --------------------------------------------------------------------------- #
# 5. lags — reusing the upstream extractor, routing around its defect
# --------------------------------------------------------------------------- #

_REV_PREFIX = "rev_"
_VIA_SEP = "_via_"


class _ReadOnlyLagStore:
    """The three ``query_all_*`` methods ``extract_temporal_lags`` calls, read-only.

    ``PipelineStore.__init__`` runs ``_init_schema``, which issues DDL and
    therefore writes; the live database is read-only by rule. This object opens
    ``mode=ro`` and serves the same three queries.

    It PROJECTS: observations come back with only ``entity_id`` and
    ``observed_at`` (the only two keys ``_compute_lags_for_pattern`` reads from
    an observation), and only for entities of the two types in the meta-path.
    Both projections are lossless *for this computation* and are covered by a
    test that compares projected against unprojected lags on a fixture
    database. If the upstream extractor ever starts reading a third key, that
    test fails rather than this returning a quietly different lag.
    """

    def __init__(self, db_path: str, *, entity_types: Sequence[str] | None = None) -> None:
        self._db_path = str(db_path)
        self._entity_types = None if entity_types is None else tuple(dict.fromkeys(entity_types))
        self._con = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
        self._con.row_factory = sqlite3.Row

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> _ReadOnlyLagStore:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def link_type_counts(self) -> dict[str, int]:
        return {
            r["link_type"]: int(r["n"])
            for r in self._con.execute("select link_type, count(*) n from entity_links group by 1")
        }

    def metapath_pair_count(self, src_type: str, edge_type: str, dst_type: str) -> int:
        """Directed (a -> b) pair count, matching what the extractor indexes.

        Directed on purpose: ``_compute_lags_for_pattern`` indexes links by
        ``(entity_id_a, link_type)`` only, so an ``instrument -> country``
        ``produced_in`` link is invisible to a ``country -> instrument``
        meta-path. Counting the same direction the extractor walks is what lets
        this module say "no pairs" instead of returning 0.0.
        """
        row = self._con.execute(
            "select count(*) from entity_links l "
            "join entities ea on ea.entity_id=l.entity_id_a "
            "join entities eb on eb.entity_id=l.entity_id_b "
            "where l.link_type=? and ea.entity_type=? and eb.entity_type=?",
            (edge_type, src_type, dst_type),
        ).fetchone()
        return int(row[0])

    def entity_types_for(self, source_tool: str, observation_type: str) -> dict[str, int]:
        return {
            r["entity_type"]: int(r["n"])
            for r in self._con.execute(
                "select e.entity_type, count(distinct o.entity_id) n from entity_observations o "
                "join entities e on e.entity_id=o.entity_id "
                "where o.source_tool=? and o.observation_type=? group by 1 order by 2 desc",
                (source_tool, observation_type),
            )
        }

    # ---- the PipelineStore surface extract_temporal_lags uses ------------- #
    def query_all_entities(self, **_: Any) -> list[dict[str, Any]]:
        return [
            {"entity_id": r["entity_id"], "entity_type": r["entity_type"]}
            for r in self._con.execute("select entity_id, entity_type from entities")
        ]

    def query_all_observations(self, **_: Any) -> list[dict[str, Any]]:
        if self._entity_types is None:
            cur = self._con.execute("select entity_id, observed_at from entity_observations")
        else:
            marks = ",".join("?" * len(self._entity_types))
            cur = self._con.execute(
                "select o.entity_id, o.observed_at from entity_observations o "  # noqa: S608 - marks are literals
                f"join entities e on e.entity_id=o.entity_id where e.entity_type in ({marks})",
                self._entity_types,
            )
        return [{"entity_id": r["entity_id"], "observed_at": r["observed_at"]} for r in cur]

    def query_all_entity_links(self, **_: Any) -> list[dict[str, Any]]:
        return [
            {"entity_id_a": r["entity_id_a"], "entity_id_b": r["entity_id_b"], "link_type": r["link_type"]}
            for r in self._con.execute("select entity_id_a, entity_id_b, link_type from entity_links")
        ]


def _normalise_edge(src_type: str, edge_type: str, dst_type: str) -> dict[str, Any]:
    """Resolve a meta-path edge name to something the upstream extractor can match.

    Two shapes are handled, and this is the whole point of the function:

    ``rev_X``
        A reversed edge. The extractor only indexes ``(entity_id_a, link_type)``
        and only matches the raw string, so ``rev_produced_in`` matches nothing
        and returns ``mean_lag=0.0``. Here it becomes ``X`` with ``src`` and
        ``dst`` swapped, which is the same meta-path walked in the direction the
        data is stored in.
    ``X_via_Y``
        A composite (multi-hop) name. There is no single ``link_type`` equal to
        it, so the extractor again returns 0.0. It is REFUSED with the
        components named: a two-hop lag is not a one-hop lag, and inventing one
        by matching the first component would be fabrication.
    """
    if _VIA_SEP in edge_type:
        return {
            "resolved_link_type": None,
            "src_type": src_type,
            "dst_type": dst_type,
            "normalisation": "composite_refused",
            "components": tuple(edge_type.split(_VIA_SEP)),
            "status": "composite_metapath_unsupported",
        }
    if edge_type.startswith(_REV_PREFIX):
        return {
            "resolved_link_type": edge_type[len(_REV_PREFIX) :],
            "src_type": dst_type,
            "dst_type": src_type,
            "normalisation": "reversed",
            "components": (),
            "status": None,
        }
    return {
        "resolved_link_type": edge_type,
        "src_type": src_type,
        "dst_type": dst_type,
        "normalisation": "identity",
        "components": (),
        "status": None,
    }


def extract_lags(candidate: Any, db_path: str, *, top_k: int = 1) -> dict[str, Any]:
    """The observed lag distribution along a candidate's meta-path edge(s).

    Reuses ``agent.models.gnn.pattern_extractor.extract_temporal_lags`` rather
    than reimplementing it: for every directed ``(src_type) -[edge]-> (dst_type)``
    pair it collects ``(src_obs_time, nearest later dst_obs_time)`` and reports
    the mean, sd and quartiles of that gap.

    KNOWN DEFECT IN THE UPSTREAM FUNCTION, ROUTED AROUND HERE RATHER THAN
    INHERITED
        ``_compute_lags_for_pattern`` matches ``pattern.edge_type`` against the
        raw ``entity_links.link_type`` string and indexes links by
        ``(entity_id_a, link_type)`` only. So:

        * a ``rev_`` name matches no link and the pattern keeps its dataclass
          default of ``mean_lag=0.0`` — a real-looking "simultaneous" answer
          that actually means "no data";
        * a composite ``X_via_Y`` name does the same;
        * a meta-path stated in the direction opposite to how the links are
          stored (``country -> instrument`` for ``produced_in``, which runs
          ``instrument -> country``) also silently yields 0.0.

        This function (a) normalises ``rev_`` by swapping the endpoints,
        (b) REFUSES composites with their components named, (c) checks the
        link type exists and that the DIRECTED meta-path has at least one pair
        before calling, and (d) seeds ``mean_lag`` with NaN so that a
        non-writing extractor is detectable rather than looking like zero.
        ``mean_lag`` is ``None`` — never 0.0 — whenever there is no data.

    Returns
    -------
    dict
        ``status`` (``"ok"`` if any edge produced lags, else the first
        blocking reason), ``metapaths`` (one dict per edge with its own
        ``status``, ``n_pairs``, ``mean_lag`` / ``mean_lag_days``, ``lag_std``,
        ``lag_p25``, ``lag_p75``, ``normalisation``), ``known_link_types``,
        ``resolved_from`` (how the meta-path was derived) and ``warnings``.

    Where it misleads
        * A lag here is a gap between OBSERVATION timestamps on linked
          entities, not a causal delay. Two entities that both get daily
          observations will show a ~1-day "lag" whatever the mechanism.
        * It is one hop. A chain's lag is not the sum of its hops' lags.
        * ``top_k`` is passed through to the upstream function, which only
          populates the first ``top_k`` patterns. It defaults to the number of
          meta-paths here so that nothing is silently left unpopulated.
    """
    from agent.models.gnn.pattern_extractor import MetaPathPattern, extract_temporal_lags

    cand = coerce_candidate(candidate)
    warnings: list[str] = []

    entity_types_needed: set[str] = set()
    edges: list[tuple[str, str, str]] = []
    resolved_from: str

    if cand.metapath is not None:
        edges = [cand.metapath]
        resolved_from = "candidate.metapath"
    else:
        with _ReadOnlyLagStore(db_path) as probe:
            src = cand.event_entity_type
            if src is None:
                observed = probe.entity_types_for(cand.event_source, cand.event_obs_type)
                if not observed:
                    return {
                        "status": "no_event_entities",
                        "metapaths": [],
                        "known_link_types": tuple(sorted(probe.link_type_counts())),
                        "resolved_from": "db lookup of event entity type",
                        "warnings": [
                            f"no entity carries {cand.event_source}/{cand.event_obs_type} observations, so the "
                            "meta-path source type cannot be derived and no lag exists to compute."
                        ],
                    }
                src = max(observed.items(), key=lambda kv: kv[1])[0]
                if len(observed) > 1:
                    warnings.append(
                        f"event observations span {len(observed)} entity types {observed}; taking the most common "
                        f"({src!r}) as the meta-path source. Set candidate.event_entity_type to be explicit."
                    )
            link_types = cand.link_types or tuple(sorted(probe.link_type_counts()))
            if not cand.link_types:
                warnings.append(
                    f"candidate declares no link_types; every one of the {len(link_types)} link types in the "
                    "graph was tried, which is a search over link types and is not corrected anywhere."
                )
        edges = [(src, lt, cand.target_entity_type) for lt in link_types]
        resolved_from = "derived from event_entity_type/link_types/target_entity_type"

    known: dict[str, int]
    rows: list[dict[str, Any]] = []
    patterns: list[MetaPathPattern] = []
    plan: list[dict[str, Any]] = []

    with _ReadOnlyLagStore(db_path) as probe:
        known = probe.link_type_counts()
        for src_type, edge_type, dst_type in edges:
            norm = _normalise_edge(src_type, edge_type, dst_type)
            entry: dict[str, Any] = {
                "src_type": src_type,
                "edge_type": edge_type,
                "dst_type": dst_type,
                "resolved_link_type": norm["resolved_link_type"],
                "resolved_src_type": norm["src_type"],
                "resolved_dst_type": norm["dst_type"],
                "normalisation": norm["normalisation"],
                "n_pairs": 0,
                # The upstream extractor reports moments, not a count, so the
                # number of (src_obs, later dst_obs) gaps behind a mean_lag is
                # genuinely unavailable. It stays None rather than being
                # invented, because a fabricated n is what this module is for.
                "n_lags": None,
                "mean_lag": None,
                "mean_lag_days": None,
                "lag_std": None,
                "lag_p25": None,
                "lag_p75": None,
                "status": norm["status"] or "pending",
            }
            if norm["status"] == "composite_metapath_unsupported":
                entry["components"] = list(norm["components"])
                warnings.append(
                    f"refused composite edge {edge_type!r} (components {list(norm['components'])}): the upstream "
                    "extractor is one-hop and matches raw link_type, so it would have returned mean_lag=0.0 — "
                    "a number meaning 'no data'."
                )
                rows.append(entry)
                continue
            lt = str(norm["resolved_link_type"])
            if lt not in known:
                entry["status"] = "unknown_link_type"
                warnings.append(
                    f"link type {lt!r} does not exist in entity_links (known: {len(known)} types); reporting "
                    "'unknown_link_type' rather than the 0.0 the upstream extractor would have returned."
                )
                rows.append(entry)
                continue
            n_pairs = probe.metapath_pair_count(norm["src_type"], lt, norm["dst_type"])
            entry["n_pairs"] = n_pairs
            if n_pairs == 0:
                entry["status"] = "no_directed_pairs"
                warnings.append(
                    f"meta-path {norm['src_type']} -[{lt}]-> {norm['dst_type']} has 0 directed pairs "
                    f"(the link exists {known[lt]} times but not in this direction/type combination); "
                    "reporting 'no_directed_pairs' rather than 0.0."
                )
                rows.append(entry)
                continue
            entity_types_needed.update({norm["src_type"], norm["dst_type"]})
            patterns.append(
                MetaPathPattern(
                    src_type=str(norm["src_type"]),
                    edge_type=lt,
                    dst_type=str(norm["dst_type"]),
                    score=float("nan"),
                    mean_attention=float("nan"),
                    frequency=int(known[lt]),
                    # NaN, not the dataclass default of 0.0: if the extractor
                    # writes nothing, that must be visible.
                    mean_lag=float("nan"),
                    lag_std=float("nan"),
                    lag_p25=float("nan"),
                    lag_p75=float("nan"),
                )
            )
            plan.append(entry)
            rows.append(entry)

    if patterns:
        with _ReadOnlyLagStore(db_path, entity_types=sorted(entity_types_needed)) as store:
            extract_temporal_lags(patterns, store, top_k=max(int(top_k), len(patterns)))
        for entry, pat in zip(plan, patterns, strict=True):
            if math.isnan(pat.mean_lag):
                entry["status"] = "no_lag_observations"
                warnings.append(
                    f"meta-path {pat.src_type} -[{pat.edge_type}]-> {pat.dst_type} has {entry['n_pairs']} pairs but "
                    "produced no (src_obs, later dst_obs) gap at all; mean_lag is None, not 0.0."
                )
                continue
            entry["status"] = "ok"
            entry["mean_lag"] = float(pat.mean_lag)
            entry["mean_lag_days"] = float(pat.mean_lag) / 86400.0
            entry["lag_std"] = float(pat.lag_std)
            entry["lag_p25"] = float(pat.lag_p25)
            entry["lag_p75"] = float(pat.lag_p75)

    ok = [r for r in rows if r["status"] == "ok"]
    if ok:
        status = "ok"
    elif rows:
        status = rows[0]["status"]
    else:
        status = "no_metapath"
    return {
        "status": status,
        "chain_id": cand.chain_id,
        "metapaths": rows,
        "n_metapaths": len(rows),
        "n_with_lags": len(ok),
        "known_link_types": tuple(sorted(known)),
        "resolved_from": resolved_from,
        "warnings": warnings,
    }
