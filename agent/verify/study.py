"""Generalised forward-return event study: verify one stated hypothesis, and say what it cost.

The product this serves is pattern VERIFICATION, not prediction. A customer states
"when <event_field> on <event_source> goes extreme, the linked <target_entity_type>
moves within N days" and gets back a verdict *plus the arithmetic*, including every
observation that could not be graded. We do not sell positive results. A `StudyResult`
whose verdict is `not_supported` is a complete delivery.

This module is `scripts/cftc_event_study.py` with CFTC cut out of it. That script is the
reference implementation and is NOT modified by this module; the discipline lifted from
it, item by item:

*   **Causal z-scores.** `z_t = (x_t - mean(x_{<t})) / std(x_{<t})`, expanding window,
    at least `min_history` prior points, degenerate-variance guard. The current point is
    never in its own history (F-04).
*   **Publication lag.** An event is actionable only after its release. Entry is the
    first target close at or after `event_ts + publication_lag_s`, within
    `entry_tolerance_s`; beyond that the event is DROPPED, never silently shifted to a
    later close. Measuring from the as-of timestamp grants free lookahead and is the
    single easiest way to manufacture a false positive.
*   **The baseline is the unconditional return distribution**, not zero. Every entry
    with a computable z enters the population regardless of whether it fired. In a
    trending market a zero null prints an "edge" equal to the drift.
*   **Attrition is reported, not absorbed.** `drop_reasons` names every excluded
    observation, `pair_attrition` names every excluded pair, and `reconciled` asserts
    the arithmetic closes. `complete` is True only when NOTHING was dropped.
*   **A leakage audit sample** of `(event_ts, publication_ts, entry_ts)` triples so a
    reader can check the ordering by hand instead of trusting this docstring.

WHERE THIS MODULE MISLEADS — read before quoting a number out of it:

*   **`p_value` is uncorrected.** One hypothesis is one test, and nobody tests one
    hypothesis. Multiplicity correction is the caller's job across the whole family
    they actually ran; `benjamini_hochberg` here is the same BH used by the reference.
    The published COT result had a best uncorrected p of 0.002 and a BH-adjusted 0.102.
    `verdict == "supported_uncorrected"` is deliberately not spelled "supported".
*   **`n_events` is not `n` independent observations.** One event entity linked to
    fifteen targets emits fifteen returns from one z-score, and correlated targets
    firing on the same date are closer to one observation than to many.
    `n_event_clusters` (distinct event timestamps) is the honest unit;
    `p_value_clustered` resamples those, and `power` is computed from clusters, not
    events. Expect `p_value_clustered` to be several times `p_value`.
*   **`direction` is the sign of the SIGNAL, not of the expected return.** The test is
    two-sided against the population mean. `direction="abs"` pools "extremely long" and
    "extremely short" into one bucket and therefore averages a genuinely signed effect
    toward zero by construction — that is a specification choice, not a finding.
*   **Graph structure is not point-in-time in the way prices are.** A pair exists here
    because a link exists in `entity_links` now (or by `as_of`). 119 of the live
    `produced_in` links have a NULL `effective_from` and cannot be dated at all;
    `n_links_undatable` counts them and `n_events_before_link_known` counts events that
    precede the date their own routing link was first recorded. Neither is dropped
    unless `require_link_predates_event=True`, because dropping undatable geography
    deletes every commodity route. This is a measured leak, not a sealed one.
*   **A static link routes evidence without witnessing anything.** "Canada produced oil"
    was true in 1990. Five `source` values along a scaffold path are one observation.
    This module counts returns; it does not count independence. Ask
    `agent.mechanism.independence()` before reading a confirmation as five datasets.
*   **`power` is a stipulated benchmark**, scaled from a reference t-statistic at a
    reference n (defaults: t=2.14 at n=1451, the figure used in
    `docs/publications/cot_null_result.md`). Substitute your own; it is a sensitivity
    knob, not a measurement of this sample.

Read-only by construction: the connection is opened `mode=ro`.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Literal

import numpy as np
from scipy.stats import norm

from agent.quant.scoring import block_bootstrap_ci

__all__ = [
    "Hypothesis",
    "PairAttrition",
    "LeakageSample",
    "StudyResult",
    "verify",
    "benjamini_hochberg",
    "causal_zscore",
    "power_two_sided",
]

DEFAULT_MIN_HISTORY = 20
DEFAULT_MIN_TARGET_POINTS = 40
DEFAULT_ENTRY_TOLERANCE_S = 7 * 86400.0
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 7
MIN_TESTABLE_EVENTS = 3

AGGREGATES = ("last", "mean", "sum", "count", "max", "min")
DIRECTIONS = ("up", "down", "abs")

# Every bucket an observation can land in. Named here so `reconciled` can check that
# the buckets sum to what entered, and so a caller can enumerate reasons without
# discovering them empirically.
PAIR_DROP_REASONS = (
    "pair_no_event_points",
    "pair_event_history_below_min",
    "pair_target_points_below_min",
    "pair_no_calendar_overlap",
)
EVENT_DROP_REASONS = (
    "event_z_not_computable",
    "event_link_not_yet_known",
    "event_no_entry_close_within_tolerance",
    "event_entry_price_invalid",
    "event_no_close_at_horizon",
    "event_forward_price_invalid",
)


@dataclass(frozen=True)
class Hypothesis:
    """A single falsifiable claim of the form "extreme <event_field> precedes a move".

    The nine fields the product exposes come first. Everything after `publication_lag_s`
    is an explicit knob with a default chosen to match the reference implementation, and
    is spelled out rather than hardcoded because a silent default is how this repo's
    failures hide.

    Parameters
    ----------
    event_source : `entity_observations.source_tool`, e.g. "gdelt", "cftc".
    event_obs_type : `entity_observations.observation_type`, e.g. "geopolitical_event".
    event_field : key inside the observation's `value_json`. Dotted paths ("a.b") index
        nested dicts. Must resolve to a real number (bool is rejected: `True` is not 1).
    z_threshold : |z| (or signed z, see `direction`) at which the event fires.
    direction : "up" (z >= +t), "down" (z <= -t), or "abs" (|z| >= t). This is the sign
        of the SIGNAL. It does not assert the sign of the return, and "abs" cancels a
        signed effect by construction.
    target_entity_type : `entities.entity_type` of the thing that is supposed to move.
    target_obs_type : observation type carrying the price series.
    horizon_days : forward horizon in SERIES STEPS of the target, not calendar days. For
        a daily close series these are trading days; a weekly target makes `horizon_days`
        weeks, and the name will lie to you.
    publication_lag_s : seconds between the event's `observed_at` (its as-of stamp) and
        the moment it became public. 0 means "actionable immediately" — assert that, do
        not default into it.
    """

    event_source: str
    event_obs_type: str
    event_field: str
    z_threshold: float
    direction: Literal["up", "down", "abs"]
    target_entity_type: str
    target_obs_type: str = "instrument_daily"
    horizon_days: int = 5
    publication_lag_s: float = 0.0
    # --- explicit knobs -------------------------------------------------------
    target_value_field: str = "close"
    target_source: str | None = None
    link_types: tuple[str, ...] | None = None
    event_aggregate: str = "last"
    target_aggregate: str = "last"
    min_history: int = DEFAULT_MIN_HISTORY
    min_target_points: int = DEFAULT_MIN_TARGET_POINTS
    entry_tolerance_s: float = DEFAULT_ENTRY_TOLERANCE_S
    require_link_predates_event: bool = False
    label: str = ""

    def __post_init__(self) -> None:
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}, got {self.direction!r}")
        if self.event_aggregate not in AGGREGATES:
            raise ValueError(f"event_aggregate must be one of {AGGREGATES}, got {self.event_aggregate!r}")
        if self.target_aggregate not in AGGREGATES:
            raise ValueError(f"target_aggregate must be one of {AGGREGATES}, got {self.target_aggregate!r}")
        if self.z_threshold < 0:
            raise ValueError(f"z_threshold must be >= 0, got {self.z_threshold}")
        if self.horizon_days < 1:
            raise ValueError(f"horizon_days must be >= 1, got {self.horizon_days}")
        if self.publication_lag_s < 0:
            raise ValueError(f"publication_lag_s must be >= 0, got {self.publication_lag_s}")
        if self.min_history < 2:
            raise ValueError(f"min_history must be >= 2 (a z-score needs a history), got {self.min_history}")
        if self.entry_tolerance_s < 0:
            raise ValueError(f"entry_tolerance_s must be >= 0, got {self.entry_tolerance_s}")

    def describe(self) -> str:
        arrow = {"up": ">= +", "down": "<= -", "abs": "|z| >= "}[self.direction]
        sig = f"{arrow}{self.z_threshold:g}" if self.direction == "abs" else f"z {arrow}{self.z_threshold:g}"
        return (
            f"{self.event_source}/{self.event_obs_type}/{self.event_field} {sig} "
            f"-> {self.target_entity_type}.{self.target_obs_type}.{self.target_value_field} "
            f"@ {self.horizon_days} steps (pub lag {self.publication_lag_s / 86400:g}d)"
        )


@dataclass(frozen=True)
class PairAttrition:
    """One (event entity, target entity) candidate and why it did or did not survive."""

    event_entity_id: str
    event_name: str
    target_entity_id: str
    target_name: str
    link_type: str
    link_effective_from: float | None
    n_event_points: int
    n_target_points: int
    status: str  # "OK" or one of PAIR_DROP_REASONS with detail
    used: bool


@dataclass(frozen=True)
class LeakageSample:
    """One graded event, emitted so a reader can check `event <= publication <= entry` by hand."""

    event_name: str
    target_name: str
    event_ts: float
    publication_ts: float
    entry_ts: float
    exit_ts: float
    z: float
    entry_price: float
    log_return: float

    @property
    def ordered(self) -> bool:
        return self.event_ts <= self.publication_ts <= self.entry_ts < self.exit_ts

    def as_row(self) -> str:
        def d(ts: float) -> str:
            return dt.datetime.fromtimestamp(ts, tz=dt.UTC).date().isoformat()

        return (
            f"{self.event_name:<18} {self.target_name:<18} event={d(self.event_ts)} "
            f"pub={d(self.publication_ts)} entry={d(self.entry_ts)} exit={d(self.exit_ts)} "
            f"z={self.z:+.2f} ret={self.log_return * 100:+.3f}%"
        )


@dataclass
class StudyResult:
    """Everything the study found, everything it dropped, and whether those two add up.

    `complete` is True only when nothing whatsoever was excluded. On real data it is
    almost always False, and that is the point: the published COT study could not grade
    one event in three. `reconciled` is the weaker but load-bearing claim that every
    observation which entered is accounted for in exactly one bucket. A False
    `reconciled` means this module lost rows and its numbers are not to be used.
    """

    hypothesis: Hypothesis
    as_of: float | None

    # --- counts that must add up ---------------------------------------------
    n_event_entities: int = 0
    n_target_entities: int = 0
    n_pairs_entered: int = 0
    n_pairs_used: int = 0
    n_event_observations_raw: int = 0
    n_event_series_points: int = 0
    n_z_computable: int = 0
    n_warmup_excluded: int = 0
    n_degenerate_excluded: int = 0
    n_below_threshold: int = 0
    n_fired: int = 0
    n_events: int = 0
    n_event_clusters: int = 0
    max_events_per_cluster: int = 0
    n_baseline: int = 0
    drop_reasons: dict[str, int] = dc_field(default_factory=dict)
    pair_attrition: list[PairAttrition] = dc_field(default_factory=list)

    # --- effect --------------------------------------------------------------
    mean_event_return: float = float("nan")
    median_event_return: float = float("nan")
    baseline_mean_return: float = float("nan")
    baseline_median_return: float = float("nan")
    edge: float = float("nan")
    hit_rate_event: float = float("nan")
    hit_rate_baseline: float = float("nan")
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    p_value: float = float("nan")
    p_value_clustered: float = float("nan")
    effective_sample_size: int = 0
    bootstrap_time_ordered: bool = False
    """True when the event sample was sorted by event time before resampling.

    The circular block bootstrap draws CONTIGUOUS runs, so it only means
    anything on a time-ordered array. Reported rather than assumed, because the
    reference implementation does not do it and its p-values move by up to
    0.059 purely with row order.
    """
    power: float = float("nan")
    power_n_events: float = float("nan")
    verdict: str = "not_testable"

    # --- audit ---------------------------------------------------------------
    complete: bool = False
    reconciled: bool = False
    excluded: list[str] = dc_field(default_factory=list)
    leakage_audit: list[LeakageSample] = dc_field(default_factory=list)
    n_links_undatable: int = 0
    n_events_before_link_known: int = 0
    n_unparsable_observations: int = 0
    n_collapsed_observations: int = 0
    event_returns: tuple[float, ...] = ()
    baseline_returns: tuple[float, ...] = ()
    event_cluster_ts: tuple[float, ...] = ()

    # --- names `agent/verify/report.py` looks a quantity up under -------------
    # That renderer resolves each quantity through an alias table rather than
    # importing this dataclass, so a field it cannot find becomes "not computed"
    # — a silent hole in a customer-facing report. These aliases are the reference
    # implementation's own row-dict spellings; they are read-only views, so there
    # is no second copy of any number to drift.
    @property
    def mean_event_ret(self) -> float:
        return self.mean_event_return

    @property
    def mean_baseline_ret(self) -> float:
        return self.baseline_mean_return

    @property
    def n_clusters(self) -> int:
        return self.n_event_clusters

    @property
    def period_name(self) -> str:
        return "distinct event timestamp"

    @property
    def field(self) -> str:
        return self.hypothesis.event_field

    @property
    def z_threshold(self) -> float:
        return self.hypothesis.z_threshold

    @property
    def horizon_days(self) -> int:
        return self.hypothesis.horizon_days

    @property
    def label(self) -> str:
        return self.hypothesis.label or self.hypothesis.describe()

    def summary(self) -> str:
        lines = [
            f"HYPOTHESIS  {self.hypothesis.label or self.hypothesis.describe()}",
            f"  pairs       {self.n_pairs_used} used / {self.n_pairs_entered} entered "
            f"({self.n_event_entities} event entities, {self.n_target_entities} targets)",
            f"  events      {self.n_events} graded / {self.n_fired} fired / {self.n_z_computable} z-computable "
            f"(from {self.n_event_series_points} series points, {self.n_event_observations_raw} raw rows)",
            f"  warm-up     {self.n_warmup_excluded} points had no history yet (structural: min_history-1 per pair); "
            f"{self.n_degenerate_excluded} more had a degenerate history variance",
            f"  clusters    {self.n_event_clusters} distinct event timestamps "
            f"(max {self.max_events_per_cluster} events on one), n_eff={self.effective_sample_size}",
            f"  effect      mean_event={self.mean_event_return * 100:+.3f}%  "
            f"baseline={self.baseline_mean_return * 100:+.3f}% (n={self.n_baseline})  "
            f"edge={self.edge * 100:+.3f}%",
            f"  hit rate    event={self.hit_rate_event:.2%}  baseline={self.hit_rate_baseline:.2%}",
            f"  CI(95%)     [{self.ci_low * 100:+.3f}%, {self.ci_high * 100:+.3f}%] (circular block bootstrap)",
            f"  p           {self.p_value:.4f} uncorrected / {self.p_value_clustered:.4f} cluster-resampled",
            f"  power       {self.power:.1%} at n_eff={self.effective_sample_size} "
            f"({self.power_n_events:.1%} if events were independent)",
            f"  VERDICT     {self.verdict}",
            f"  complete    {self.complete}   reconciled  {self.reconciled}",
        ]
        if self.drop_reasons:
            lines.append("  drops")
            for k, v in sorted(self.drop_reasons.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {k:<42} {v}")
        if self.excluded:
            lines.append("  EXCLUDED")
            lines.extend(f"    - {e}" for e in self.excluded)
        if self.leakage_audit:
            lines.append("  leakage audit (event <= publication <= entry < exit)")
            lines.extend(f"    {s.as_row()}" for s in self.leakage_audit)
            bad = [s for s in self.leakage_audit if not s.ordered]
            lines.append(f"    ordering violations in sample: {len(bad)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def causal_zscore(values: list[float] | np.ndarray, min_history: int = DEFAULT_MIN_HISTORY) -> float | None:
    """z-score of the LAST element against only the elements before it.

    Replica of `zscore_causal` in `scripts/cftc_event_study.py`, which is itself a
    replica of `_zscore_anomaly` in the shipped product: expanding window, `x[:-1]` as
    history, `min_history` prior points required, degenerate-variance guard.

    Returns None when the z-score is not computable. None is not 0.0 — a silent zero
    here would turn "we could not tell" into "nothing happened", which is F-17's
    signature (a target of all zeros owning 99% of an objective).

    Misleads: `np.std` is the POPULATION standard deviation (ddof=0), so for short
    histories this z is larger in magnitude than a sample-std z would be. That matches
    what the product shipped, which is the point of grading it this way.
    """
    if len(values) < min_history:
        return None
    x = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(x)):
        return None
    hist = x[:-1]
    if float(np.std(hist)) < 1e-12:
        return None
    return float((x[-1] - np.mean(hist)) / (np.std(hist) + 1e-12))


def benjamini_hochberg(pvalues: list[float] | np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg (1995) step-up FDR control.

    Returns `(reject, p_adjusted)`. Equivalent to
    `statsmodels.stats.multitest.multipletests(..., method="fdr_bh")` (asserted in the
    test suite) but implemented on numpy so this module depends on numpy and scipy only.

    Non-finite p-values are not tests: they are excluded from the family, returned as
    NaN, and do not inflate `m`. Silently coercing them to 1.0 would enlarge the family
    and make every real p look better.

    Misleads: BH controls the false discovery RATE over the family you pass. Passing the
    one hypothesis you liked is not a correction, it is laundering.
    """
    p = np.asarray(pvalues, dtype=float)
    reject = np.zeros(p.shape, dtype=bool)
    adjusted = np.full(p.shape, np.nan, dtype=float)
    finite = np.isfinite(p)
    m = int(finite.sum())
    if m == 0:
        return reject, adjusted
    idx = np.flatnonzero(finite)
    order = idx[np.argsort(p[idx], kind="stable")]
    ranked = p[order] * m / np.arange(1, m + 1)
    # step-up: enforce monotonicity from the largest p downwards
    adj_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)
    adjusted[order] = adj_sorted
    reject[order] = adj_sorted <= alpha
    return reject, adjusted


def power_two_sided(
    n: int,
    reference_t: float = 2.14,
    reference_n: int = 1451,
    alpha: float = 0.05,
) -> float:
    """Power of a two-sided test at sample size `n`, against a scaled reference effect.

    `t_expected = reference_t * sqrt(n / reference_n)`; power is
    `P(|Z + t_expected| > z_{1-alpha/2})`. This is the calculation in
    `docs/publications/cot_null_result.md` section 7.1 (n=123 -> 9.6%, n=160 -> 11.0%),
    reproduced so a null carries its own power rather than being quoted as proof.

    Misleads: `reference_t` is STIPULATED from published literature, not estimated here.
    Change it and the power changes; that is a feature (report the sensitivity), not a
    measurement of this dataset.
    """
    if n <= 0 or reference_n <= 0:
        return float("nan")
    t_exp = reference_t * float(np.sqrt(n / reference_n))
    crit = float(norm.isf(alpha / 2.0))
    return float(norm.cdf(t_exp - crit) + norm.cdf(-t_exp - crit))


def _bootstrap_p_iid(events: np.ndarray, baseline_mean: float, seed: int = BOOTSTRAP_SEED) -> float:
    """Two-sided percentile bootstrap p for the event mean against the population mean.

    Resamples events i.i.d. with replacement, exactly as the reference does, including
    the seed and B, so a reproduction can compare against it.

    Misleads: i.i.d. resampling assumes the events are independent draws. They are not
    when correlated targets fire on the same date — see `_bootstrap_p_clustered`.
    """
    rng = np.random.default_rng(seed)
    n = len(events)
    boot = np.array([np.mean(rng.choice(events, size=n, replace=True)) for _ in range(N_BOOTSTRAP)])
    p_low = float((boot <= baseline_mean).mean())
    p_high = float((boot >= baseline_mean).mean())
    return min(1.0, float(2 * min(p_low, p_high)))


def _bootstrap_p_clustered(
    events: np.ndarray,
    cluster_ids: np.ndarray,
    baseline_mean: float,
    seed: int = BOOTSTRAP_SEED,
    n_bootstrap: int = N_BOOTSTRAP,
) -> float:
    """Same two-sided bootstrap, resampling event TIMESTAMPS instead of events.

    The correct unit when several targets fire on one date from one macro event. In the
    published COT study this moved the best uncorrected p from 0.002 to 0.010 — a factor
    of five, which is the honest measure of how much of `n` was real.
    """
    uniq = np.unique(cluster_ids)
    if len(uniq) < 2:
        return float("nan")
    groups = [events[cluster_ids == c] for c in uniq]
    rng = np.random.default_rng(seed)
    k = len(groups)
    boot = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        picks = rng.integers(0, k, size=k)
        boot[b] = float(np.mean(np.concatenate([groups[i] for i in picks])))
    p_low = float((boot <= baseline_mean).mean())
    p_high = float((boot >= baseline_mean).mean())
    return min(1.0, float(2 * min(p_low, p_high)))


# ---------------------------------------------------------------------------
# data access (read-only)
# ---------------------------------------------------------------------------


def _connect_ro(db_path: str) -> sqlite3.Connection:
    """Open `db_path` read-only. A verification product must not be able to edit its evidence."""
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def _dig(value: Any, path: str) -> Any:
    """Index a nested dict by a dotted path. Returns None when any hop is missing."""
    cur = value
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _numeric(v: Any) -> float | None:
    """Coerce to float, rejecting bool (True is not 1.0) and non-finite values."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return f if np.isfinite(f) else None


@dataclass
class _Series:
    ts: np.ndarray
    val: np.ndarray
    n_raw: int
    n_unparsable: int
    n_missing_field: int
    n_collapsed: int  # raw rows discarded by a lossy aggregate ("last"/"max"/"min")

    def __len__(self) -> int:
        return len(self.ts)


def _load_series(
    con: sqlite3.Connection,
    entity_id: str,
    obs_type: str,
    source_tool: str | None,
    field_path: str,
    aggregate: str,
    as_of: float | None,
) -> _Series:
    """Load one entity's numeric series for `field_path`, collapsing repeated timestamps.

    Several rows can share an `observed_at` (gdelt files dozens of events per country per
    day; duplicate ingestion has been a live failure mode here). `aggregate` says what to
    do about it, and `n_collapsed` counts the rows a LOSSY aggregate threw away so the
    caller can refuse to call the result complete. The reference's "keep max(rowid)"
    behaviour is `aggregate="last"`.

    Misleads: with `aggregate="last"` on a source that files many events per timestamp,
    the series is one arbitrary event per day and `n_collapsed` will be large. That is
    reported, not corrected.
    """
    sql = "select observed_at, value_json from entity_observations where entity_id=? and observation_type=?"
    params: list[Any] = [entity_id, obs_type]
    if source_tool is not None:
        sql += " and source_tool=?"
        params.append(source_tool)
    if as_of is not None:
        sql += " and observed_at<=?"
        params.append(as_of)
    sql += " order by observed_at asc, id asc"
    rows = con.execute(sql, params).fetchall()

    buckets: dict[float, list[float]] = defaultdict(list)
    n_unparsable = 0
    n_missing = 0
    for ts, vj in rows:
        try:
            parsed = json.loads(vj)
        except (json.JSONDecodeError, TypeError):
            n_unparsable += 1
            continue
        if not isinstance(parsed, dict):
            n_unparsable += 1
            continue
        num = _numeric(_dig(parsed, field_path))
        if num is None:
            n_missing += 1
            continue
        buckets[float(ts)].append(num)

    reducer = {
        "last": lambda xs: xs[-1],
        "mean": lambda xs: float(np.mean(xs)),
        "sum": lambda xs: float(np.sum(xs)),
        "count": lambda xs: float(len(xs)),
        "max": lambda xs: float(np.max(xs)),
        "min": lambda xs: float(np.min(xs)),
    }[aggregate]
    lossy = aggregate in ("last", "max", "min")

    ts_sorted = sorted(buckets)
    vals = [reducer(buckets[t]) for t in ts_sorted]
    n_collapsed = sum(len(buckets[t]) - 1 for t in ts_sorted) if lossy else 0
    return _Series(
        ts=np.asarray(ts_sorted, dtype=float),
        val=np.asarray(vals, dtype=float),
        n_raw=len(rows),
        n_unparsable=n_unparsable,
        n_missing_field=n_missing,
        n_collapsed=n_collapsed,
    )


def _event_entities(con: sqlite3.Connection, h: Hypothesis, as_of: float | None) -> list[tuple[str, str]]:
    sql = (
        "select distinct o.entity_id, e.canonical_name from entity_observations o "
        "join entities e on e.entity_id=o.entity_id "
        "where o.source_tool=? and o.observation_type=?"
    )
    params: list[Any] = [h.event_source, h.event_obs_type]
    if as_of is not None:
        sql += " and o.observed_at<=?"
        params.append(as_of)
    sql += " order by o.entity_id"
    return [(r[0], r[1]) for r in con.execute(sql, params).fetchall()]


def _linked_targets(
    con: sqlite3.Connection,
    event_entity_id: str,
    h: Hypothesis,
    as_of: float | None,
) -> list[tuple[str, str, str, float | None]]:
    """(target_entity_id, name, link_type, effective_from) for targets linked in EITHER direction.

    `entity_links` is stored undirected-ish: `produced_in` runs instrument -> country,
    `cftc_tracks` runs contract -> instrument. A study that only looked at
    `entity_id_a` would silently find zero commodity pairs, so both columns are checked.

    Links with `effective_from > as_of` are excluded. Links with a NULL `effective_from`
    are KEPT and counted as undatable by the caller: all 119 live `produced_in` links are
    NULL, and a strict gate deletes every commodity route at any historical date.
    """
    out: list[tuple[str, str, str, float | None]] = []
    for near, far in (("entity_id_a", "entity_id_b"), ("entity_id_b", "entity_id_a")):
        sql = (
            f"select l.{far}, e.canonical_name, l.link_type, l.effective_from "  # noqa: S608 - column names are literals above
            f"from entity_links l join entities e on e.entity_id=l.{far} "
            f"where l.{near}=? and e.entity_type=? and l.{far}<>l.{near}"
        )
        params: list[Any] = [event_entity_id, h.target_entity_type]
        if h.link_types:
            sql += f" and l.link_type in ({','.join('?' * len(h.link_types))})"
            params.extend(h.link_types)
        if as_of is not None:
            sql += " and (l.effective_from is null or l.effective_from<=?)"
            params.append(as_of)
        for tid, name, ltype, eff in con.execute(sql, params).fetchall():
            out.append((tid, name, ltype, None if eff is None else float(eff)))
    # one row per (target, link_type); keep the earliest datable effective_from
    best: dict[tuple[str, str], tuple[str, str, str, float | None]] = {}
    for tid, name, ltype, eff in out:
        key = (tid, ltype)
        prev = best.get(key)
        if prev is None or (eff is not None and (prev[3] is None or eff < prev[3])):
            best[key] = (tid, name, ltype, eff)
    return [best[k] for k in sorted(best)]


def _reconciles(
    *,
    entered: int,
    n_not_computable: int,
    n_z_computable: int,
    n_graded: int,
    n_downstream_drops: int,
    n_fired: int,
    n_below_threshold: int,
    n_events: int,
) -> bool:
    """Does every observation that entered the study land in exactly one bucket?

    Four identities, all of which must hold:

    1.  Every event-series point of a usable pair either has a computable causal z or
        does not: `entered == n_not_computable + n_z_computable`.
    2.  Every point WITH a z is either graded into the baseline or dropped for a named
        downstream reason: `n_z_computable == n_graded + n_downstream_drops`.
    3.  Every point with a z either fired or sat below the threshold:
        `n_z_computable == n_fired + n_below_threshold`.
    4.  The event set is a subset of both the graded points and the fired points.

    A False here means this module lost rows, and none of its other numbers should be
    quoted. This is the arithmetic that a 5xx silently dropping 72% of every window
    (F-16) would have failed at the time instead of two years later.

    Misleads: it checks that counts ADD UP, not that any of them is the right count.
    A gate that wrongly drops a whole pair reconciles perfectly.
    """
    return (
        entered == n_not_computable + n_z_computable
        and n_z_computable == n_graded + n_downstream_drops
        and n_z_computable == n_fired + n_below_threshold
        and n_events <= n_graded
        and n_events <= n_fired
    )


# ---------------------------------------------------------------------------
# the study
# ---------------------------------------------------------------------------


def verify(
    db_path: str, hypothesis: Hypothesis, *, as_of: float | None = None, max_audit_samples: int = 8
) -> StudyResult:
    """Run the event study for one hypothesis against a read-only pipeline DB.

    Steps, in order, each of which reports its own attrition:

    1.  Resolve event entities (every entity with an observation of
        `event_source`/`event_obs_type`), then the `target_entity_type` entities linked
        to each in either direction.
    2.  Per pair, load both series and apply availability gates: enough event history to
        compute a z-score at all, `min_target_points` target points, and overlapping
        calendars after the publication lag. Pairs are dropped on DATA only, never on
        how they performed.
    3.  Walk the event series forward. At each point compute a causal z from prior
        points only. Every point with a computable z enters the POPULATION baseline;
        points that also pass `direction`/`z_threshold` enter the event set.
    4.  Entry is the first target observation at or after `event_ts + publication_lag_s`,
        within `entry_tolerance_s`. Exit is `horizon_days` series steps later. Return is
        `log(exit/entry)`.
    5.  Test the event mean against the population mean by two-sided bootstrap, both
        i.i.d. and resampling event timestamps; CI by circular block bootstrap.

    `as_of` bounds BOTH series and the links, so a historical run cannot see a price,
    an event or a graph edge that did not exist yet. Note it is a bound, not a
    reconstruction: see the module docstring on undatable links.

    Misleads: `verify` applies no multiplicity correction, because it can see only the
    one hypothesis it was given. Run your family through `benjamini_hochberg`, and
    include the hypotheses you discarded in it.
    """
    h = hypothesis
    result = StudyResult(hypothesis=h, as_of=as_of)
    drops: dict[str, int] = defaultdict(int)
    con = _connect_ro(db_path)
    try:
        events = _event_entities(con, h, as_of)
        result.n_event_entities = len(events)

        # ---- step 1 & 2: pairs and their attrition --------------------------
        event_cache: dict[str, _Series] = {}
        target_cache: dict[str, _Series] = {}
        pairs: list[dict[str, Any]] = []
        target_ids: set[str] = set()
        for eid, ename in events:
            if eid not in event_cache:
                event_cache[eid] = _load_series(
                    con, eid, h.event_obs_type, h.event_source, h.event_field, h.event_aggregate, as_of
                )
            eser = event_cache[eid]
            for tid, tname, ltype, eff in _linked_targets(con, eid, h, as_of):
                target_ids.add(tid)
                if tid not in target_cache:
                    target_cache[tid] = _load_series(
                        con, tid, h.target_obs_type, h.target_source, h.target_value_field, h.target_aggregate, as_of
                    )
                tser = target_cache[tid]
                if eff is None:
                    result.n_links_undatable += 1

                status = "OK"
                if len(eser) == 0:
                    status = "pair_no_event_points: 0 usable event points"
                elif len(eser) < h.min_history:
                    status = f"pair_event_history_below_min: {len(eser)} event points < {h.min_history}"
                elif len(tser) < h.min_target_points:
                    status = f"pair_target_points_below_min: {len(tser)} target points < {h.min_target_points}"
                else:
                    first_pub = float(eser.ts[0]) + h.publication_lag_s
                    last_pub = float(eser.ts[-1]) + h.publication_lag_s
                    if float(tser.ts[-1]) < first_pub:
                        status = "pair_no_calendar_overlap: target series ends before first publication"
                    elif float(tser.ts[0]) > last_pub:
                        status = "pair_no_calendar_overlap: target series starts after last publication"
                used = status == "OK"
                if not used:
                    drops[status.split(":", 1)[0]] += 1
                pairs.append(
                    {
                        "event_id": eid,
                        "event_name": ename,
                        "target_id": tid,
                        "target_name": tname,
                        "link_type": ltype,
                        "eff": eff,
                        "eser": eser,
                        "tser": tser,
                        "used": used,
                    }
                )
                result.pair_attrition.append(
                    PairAttrition(
                        event_entity_id=eid,
                        event_name=ename,
                        target_entity_id=tid,
                        target_name=tname,
                        link_type=ltype,
                        link_effective_from=eff,
                        n_event_points=len(eser),
                        n_target_points=len(tser),
                        status=status,
                        used=used,
                    )
                )

        result.n_target_entities = len(target_ids)
        result.n_pairs_entered = len(pairs)
        usable = [p for p in pairs if p["used"]]
        result.n_pairs_used = len(usable)
        result.n_event_observations_raw = sum(s.n_raw for s in event_cache.values())
        result.n_event_series_points = sum(len(s) for s in event_cache.values())
        result.n_unparsable_observations = sum(s.n_unparsable for s in event_cache.values()) + sum(
            s.n_unparsable for s in target_cache.values()
        )
        result.n_collapsed_observations = sum(s.n_collapsed for s in event_cache.values()) + sum(
            s.n_collapsed for s in target_cache.values()
        )

        # ---- step 3 & 4: events, entries, forward returns -------------------
        ev_rets: list[float] = []
        ev_clusters: list[float] = []
        pop_rets: list[float] = []
        n_z_computable = 0
        n_below_threshold = 0
        n_fired = 0
        for p in usable:
            eser: _Series = p["eser"]
            tser: _Series = p["tser"]
            hist: list[float] = []
            for i in range(len(eser)):
                event_ts = float(eser.ts[i])
                hist.append(float(eser.val[i]))
                z = causal_zscore(hist, h.min_history)
                if z is None:
                    drops["event_z_not_computable"] += 1
                    continue
                n_z_computable += 1
                fired = (
                    z >= h.z_threshold
                    if h.direction == "up"
                    else (z <= -h.z_threshold if h.direction == "down" else abs(z) >= h.z_threshold)
                )
                if fired:
                    n_fired += 1
                    if p["eff"] is not None and event_ts < p["eff"]:
                        result.n_events_before_link_known += 1
                        if h.require_link_predates_event:
                            drops["event_link_not_yet_known"] += 1
                            continue
                else:
                    n_below_threshold += 1

                pub_ts = event_ts + h.publication_lag_s
                idx = int(np.searchsorted(tser.ts, pub_ts, side="left"))
                if idx >= len(tser) or (float(tser.ts[idx]) - pub_ts) > h.entry_tolerance_s:
                    drops["event_no_entry_close_within_tolerance"] += 1
                    continue
                entry_price = float(tser.val[idx])
                if entry_price <= 0 or not np.isfinite(entry_price):
                    drops["event_entry_price_invalid"] += 1
                    continue
                exit_idx = idx + h.horizon_days
                if exit_idx >= len(tser):
                    drops["event_no_close_at_horizon"] += 1
                    continue
                exit_price = float(tser.val[exit_idx])
                if exit_price <= 0 or not np.isfinite(exit_price):
                    drops["event_forward_price_invalid"] += 1
                    continue
                log_ret = float(np.log(exit_price / entry_price))
                pop_rets.append(log_ret)
                if fired:
                    ev_rets.append(log_ret)
                    ev_clusters.append(event_ts)
                    if len(result.leakage_audit) < max_audit_samples:
                        result.leakage_audit.append(
                            LeakageSample(
                                event_name=p["event_name"],
                                target_name=p["target_name"],
                                event_ts=event_ts,
                                publication_ts=pub_ts,
                                entry_ts=float(tser.ts[idx]),
                                exit_ts=float(tser.ts[exit_idx]),
                                z=z,
                                entry_price=entry_price,
                                log_return=log_ret,
                            )
                        )
    finally:
        con.close()

    result.n_z_computable = n_z_computable
    result.n_below_threshold = n_below_threshold
    result.n_fired = n_fired
    result.n_events = len(ev_rets)
    result.n_baseline = len(pop_rets)
    result.drop_reasons = {k: v for k, v in sorted(drops.items()) if v}

    # ---- step 5: effect and inference ---------------------------------------
    ev = np.asarray(ev_rets, dtype=float)
    pop = np.asarray(pop_rets, dtype=float)
    clusters = np.asarray(ev_clusters, dtype=float)
    result.event_returns = tuple(float(x) for x in ev)
    result.baseline_returns = tuple(float(x) for x in pop)
    uniq_clusters = np.unique(clusters) if len(clusters) else np.asarray([])
    result.event_cluster_ts = tuple(float(x) for x in uniq_clusters)
    result.n_event_clusters = int(len(uniq_clusters))
    result.max_events_per_cluster = (
        int(max(int((clusters == c).sum()) for c in uniq_clusters)) if len(uniq_clusters) else 0
    )
    result.effective_sample_size = result.n_event_clusters
    result.power = power_two_sided(result.effective_sample_size)
    result.power_n_events = power_two_sided(result.n_events)

    if len(pop):
        result.baseline_mean_return = float(pop.mean())
        result.baseline_median_return = float(np.median(pop))
        result.hit_rate_baseline = float((pop > 0).mean())
    if len(ev):
        result.mean_event_return = float(ev.mean())
        result.median_event_return = float(np.median(ev))
        result.hit_rate_event = float((ev > 0).mean())
        if len(pop):
            result.edge = result.mean_event_return - result.baseline_mean_return
    if len(ev) >= MIN_TESTABLE_EVENTS and len(pop):
        # Sort by event time before resampling.  `block_bootstrap_ci` is a
        # CIRCULAR BLOCK bootstrap: it draws contiguous runs to preserve
        # autocorrelation, which is only meaningful when the array is in time
        # order.  Events are accumulated pair-by-pair and only chronologically
        # WITHIN a pair, so the raw array is a concatenation of per-ticker
        # series — blocks straddle ticker boundaries, and because `rng` draws
        # POSITIONS the answer depends on the order SQL happened to return the
        # links in.  Measured against scripts/cftc_event_study.py, which has
        # this defect: 49 of 54 cells disagreed on p, max |delta| 0.059, from
        # nothing but a different row order.  A p-value that moves when you
        # re-sort your data is not a p-value.
        #
        # Sorting makes the blocks capture genuine temporal clustering and makes
        # the result reproducible.  It does not change the estimator.
        _order = np.argsort(clusters, kind="stable") if len(clusters) == len(ev) else np.arange(len(ev))
        ev_t = ev[_order]
        clusters_t = clusters[_order] if len(clusters) == len(ev) else clusters
        result.bootstrap_time_ordered = True
        _, result.ci_low, result.ci_high = block_bootstrap_ci(ev_t, np.mean, n_bootstrap=N_BOOTSTRAP)
        result.p_value = _bootstrap_p_iid(ev_t, result.baseline_mean_return)
        result.p_value_clustered = _bootstrap_p_clustered(ev_t, clusters_t, result.baseline_mean_return)

    # ---- audit: what was excluded, and does the arithmetic close? -----------
    # The first `min_history - 1` points of every usable pair can never carry a causal
    # z-score: that is arithmetic, not coverage, and it is reported separately so that
    # `complete` is a flag that can actually be True. Anything ABOVE that floor is a
    # degenerate history variance, which IS lost evidence and IS an exclusion.
    n_not_computable = result.drop_reasons.get("event_z_not_computable", 0)
    result.n_warmup_excluded = min(n_not_computable, result.n_pairs_used * (h.min_history - 1))
    result.n_degenerate_excluded = n_not_computable - result.n_warmup_excluded

    excluded: list[str] = []
    for pair_reason in PAIR_DROP_REASONS:
        n = result.drop_reasons.get(pair_reason, 0)
        if n:
            excluded.append(f"{n} (event, target) pairs dropped on data availability: {pair_reason}")
    if result.n_degenerate_excluded:
        excluded.append(
            f"{result.n_degenerate_excluded} event-observations had a degenerate history variance and "
            f"could not be z-scored (event_z_not_computable beyond the "
            f"{result.n_warmup_excluded}-point structural warm-up)"
        )
    for reason in EVENT_DROP_REASONS:
        if reason == "event_z_not_computable":
            continue  # split into warm-up (structural) and degenerate (lost) above
        n = result.drop_reasons.get(reason, 0)
        if n:
            excluded.append(f"{n} event-observations dropped: {reason}")
    if result.n_collapsed_observations:
        excluded.append(
            f"{result.n_collapsed_observations} raw observations discarded by lossy aggregation "
            f"(event_aggregate={h.event_aggregate!r}, target_aggregate={h.target_aggregate!r}); "
            f"use 'mean'/'sum'/'count' to keep them"
        )
    if result.n_unparsable_observations:
        excluded.append(f"{result.n_unparsable_observations} observations had unparsable or non-dict value_json")
    if result.n_links_undatable:
        excluded.append(
            f"{result.n_links_undatable} routing links have a NULL effective_from and could not be "
            f"date-gated; pairs were kept and this is an unsealed leak, not a clean one"
        )
    if result.n_events_before_link_known and not h.require_link_predates_event:
        excluded.append(
            f"{result.n_events_before_link_known} fired events predate the effective_from of their own "
            f"routing link and were KEPT (require_link_predates_event=False)"
        )
    if as_of is not None:
        excluded.append(f"everything after as_of={as_of} was out of scope by request")
    if len(ev) and len(ev) < MIN_TESTABLE_EVENTS:
        excluded.append(f"not tested: {len(ev)} events < {MIN_TESTABLE_EVENTS} required for a bootstrap")
    result.excluded = excluded
    result.complete = not excluded

    result.reconciled = _reconciles(
        entered=sum(len(p["eser"]) for p in usable),
        n_not_computable=result.drop_reasons.get("event_z_not_computable", 0),
        n_z_computable=n_z_computable,
        n_graded=len(pop),
        n_downstream_drops=sum(
            result.drop_reasons.get(r, 0)
            for r in (
                "event_no_entry_close_within_tolerance",
                "event_entry_price_invalid",
                "event_no_close_at_horizon",
                "event_forward_price_invalid",
                "event_link_not_yet_known",
            )
        ),
        n_fired=n_fired,
        n_below_threshold=n_below_threshold,
        n_events=result.n_events,
    )

    # ---- verdict -------------------------------------------------------------
    if result.n_events < MIN_TESTABLE_EVENTS or not np.isfinite(result.p_value):
        result.verdict = "not_testable"
    elif not result.reconciled:
        result.verdict = "invalid_bookkeeping"
    elif result.p_value < 0.05:
        result.verdict = "supported_uncorrected"
    else:
        result.verdict = "not_supported"
    return result
