"""Layer 2 — derived CFTC positioning features.

Pure feature engineering over ``futures_positioning`` observations that the
Layer 1 CFTC tool (``agent/tools/cftc.py``) has already stored.  No network,
no fetching: this module reads rows the pipeline already owns and writes
``futures_positioning_derived`` rows back through ``PipelineStore``.

Why it lives here and not in ``scripts/``
-----------------------------------------
This math used to live in ``scripts/add_cftc_derived_features.py``, which is
not an importable package (``scripts/`` has no ``__init__.py``), so no DAG
could call it.  The consequence, found by the 2026-09-23 audit (P6.4): the
derivation ran exactly once, on 2026-05-03, and then never again — the raw
``cftc`` source kept collecting while ``cftc_derived`` sat frozen at
``observed_at`` 2026-04-14 for 162 days.  There was no failing node to notice,
because there was no node at all.

Derived features (all per-entity, time-ordered, trailing windows only — a row
never sees data from its own future, so re-deriving history is stable):

  ``cftc_mm_pct_52w_rank`` : float [0, 1]
      Rolling 52-observation percentile rank of ``mm_net_pct_oi``.  Tells the
      GNN whether speculative positioning is historically crowded (near 1.0)
      or bare (near 0.0) — the raw value is not interpretable without it.

  ``cftc_mm_direction_change`` : float {-1, 0, +1}
      Sign of (current ``mm_net_pct_oi`` − previous week's).  Momentum of
      positioning.

  ``cftc_oi_vs_52w_avg`` : float
      Current ``open_interest`` / rolling 52-observation mean.  > 1.0 means
      unusually heavy participation; < 1.0 thinning liquidity.

  ``mm_net_pct_oi_raw`` : float | None
      The raw value passed through for reference.

What a re-run does, and what it cannot do
-----------------------------------------
Every window is trailing and every input is already in the database, so
``derive_and_store`` is a pure re-derivation over whatever raw weeks the
database holds.

It does **not** "refill a gap".  It can only re-derive report dates that
``fetch_cftc`` actually collected; a COT week the collector never fetched is
not recoverable here, and the ``derive_cftc_features`` node only ever runs
``fetch_cftc`` in ``mode="latest"`` (current week) — ``mode="historical"`` is
the only backfill path and no node calls it.  Measured on the live database
on 2026-09-23: 335 distinct report dates exist against 9,622 weekly slots
(53%), and only **3** report dates land in the 162 days after the derived
table froze.  A run that re-derives those 3 weeks is not a repaired gap, so
``derive_and_store`` reports :func:`coverage_stats` alongside its row counts
and logs a warning when the trailing window is mostly holes.  Fixing the holes
is a Layer 1 job (``agent/tools/cftc.py``), not this module's.

Re-running is *idempotent by timestamp*, not merely by value.  A derived
feature is a function of the entity's history up to ``observed_at``, so
exactly one derived row may exist per ``(entity_id, observed_at)``.  When a
re-derivation produces a different value for a timestamp that already has a
row — which is what happens the first time this module runs over the 5,080
rows the old script left behind — the new row **supersedes** the old one
instead of landing beside it.  ``OBSERVATION_UNIQUE_KEY`` includes
``value_json`` (deliberately, for GDELT's sake — see ``store.py``), so the
store alone cannot collapse a changed value; without the supersede the live
table would go from 80 to 209 same-timestamp conflicting groups in one run,
and ``microstructure_signals`` / ``graph_builder`` would pick between them by
SELECT order.  The supersede is counted and returned as ``rows_superseded`` so
a re-derivation that rewrites history is visible rather than silent.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from agent.pipeline.store import PipelineStore

log = logging.getLogger(__name__)

#: Source tool recorded on the rows this module writes.
SOURCE_TOOL = "cftc_derived"
#: Observation type read (written by ``agent/tools/cftc.py``).
RAW_OBSERVATION_TYPE = "futures_positioning"
#: Observation type written.
DERIVED_OBSERVATION_TYPE = "futures_positioning_derived"
#: Rolling window, in observations.  CFTC COT is weekly → 52 ≈ one year.
WINDOW_OBSERVATIONS = 52
#: Depth level of a derived observation (L2 = engineered on top of an L1 fetch).
DEPTH_LEVEL = 2
#: Seconds between two CFTC Commitments of Traders report dates.
SECONDS_PER_WEEK = 7 * 86400.0
#: Trailing window used for the collector-coverage check, in report dates.  A
#: quarter is short enough that a stopped collector shows up within weeks
#: instead of being diluted by a year of healthy history.
COVERAGE_WINDOW_WEEKS = 12
#: Below this fraction of expected report dates the derivation logs a warning:
#: its output is thin because its *input* is thin, not because it failed.
COVERAGE_WARN_RATIO = 0.8


class RawPositioning(NamedTuple):
    """What :func:`load_raw_positioning` found, including what it discarded."""

    #: entity_id → ``(observed_at, raw_value)`` pairs, ascending, one per timestamp.
    by_entity: dict[str, list[tuple[float, dict[str, Any]]]]
    #: Byte-identical duplicate raw rows dropped (real: the live table holds 442).
    duplicates_dropped: int
    #: Same ``(entity, observed_at)`` carrying a *different* payload.  Kept the
    #: most recently ingested one; this counter exists so that never happens
    #: quietly.
    conflicts_collapsed: int
    #: Rows whose ``value_json`` would not parse or was not an object.
    unparseable: int


def rolling_percentile_rank(series: list[float], window: int) -> list[float]:
    """For each element *i*, its rank among the last *window* values (inclusive).

    Returns a list the same length as *series*.  Positions with fewer than two
    finite observations in the window get 0.5 (neutral) rather than a rank
    invented from a single point.  Trailing-only by construction: element *i*
    never sees ``series[i + 1:]`` (F-04 — no lookahead into the future).
    """
    ranks: list[float] = []
    for i, val in enumerate(series):
        start = max(0, i - window + 1)
        window_vals = [v for v in series[start : i + 1] if v == v]  # drop NaN
        if len(window_vals) < 2:
            ranks.append(0.5)
            continue
        below = sum(1 for v in window_vals if v < val)
        rank = below / (len(window_vals) - 1)
        ranks.append(rank)
    return ranks


def derive_series(
    observations: list[tuple[float, dict[str, Any]]],
    window: int = WINDOW_OBSERVATIONS,
) -> list[tuple[float, dict[str, Any]]]:
    """Derive one entity's feature series.  Pure function, no I/O.

    Parameters
    ----------
    observations
        ``(observed_at, raw_value)`` pairs for a single entity, **ascending by
        observed_at**.  Order is the caller's responsibility; the rolling
        windows are meaningless otherwise.
    window
        Rolling window length in observations.

    Returns
    -------
    list of ``(observed_at, derived_value)``, one per input observation, in the
    same order.
    """
    if not observations:
        return []

    mm_pct_series: list[float] = []
    oi_series: list[float] = []
    for _, val in observations:
        mm_pct = val.get("mm_net_pct_oi")
        oi = val.get("open_interest")
        mm_pct_series.append(float(mm_pct) if mm_pct is not None else float("nan"))
        oi_series.append(float(oi) if oi is not None else float("nan"))

    # NaN → 0.0 before ranking so a missing week does not shift every later rank.
    valid_mm = [v if v == v else 0.0 for v in mm_pct_series]
    pct_ranks = rolling_percentile_rank(valid_mm, window)

    derived: list[tuple[float, dict[str, Any]]] = []
    for i, (ts, _raw) in enumerate(observations):
        mm_pct = mm_pct_series[i]
        oi = oi_series[i]
        prev_mm = mm_pct_series[i - 1] if i > 0 else mm_pct

        # Week-over-week direction of speculative positioning.
        if mm_pct == mm_pct and prev_mm == prev_mm:  # both finite
            delta = mm_pct - prev_mm
            direction = 1.0 if delta > 0 else (-1.0 if delta < 0 else 0.0)
        else:
            direction = 0.0

        # Open interest against its own trailing mean.
        start = max(0, i - window + 1)
        window_oi = [v for v in oi_series[start : i + 1] if v == v]
        if window_oi and oi == oi:
            mean_oi = sum(window_oi) / len(window_oi)
            oi_vs_avg = oi / mean_oi if mean_oi else 1.0
        else:
            oi_vs_avg = 1.0  # neutral default

        derived.append(
            (
                ts,
                {
                    "cftc_mm_pct_52w_rank": round(pct_ranks[i], 4),
                    "cftc_mm_direction_change": direction,
                    "cftc_oi_vs_52w_avg": round(oi_vs_avg, 4),
                    "mm_net_pct_oi_raw": mm_pct if mm_pct == mm_pct else None,
                },
            )
        )
    return derived


def load_raw_positioning(store: PipelineStore) -> RawPositioning:
    """Read every ``futures_positioning`` observation, one row per entity-week.

    The COT publishes one report per contract per report date, so
    ``(entity_id, observed_at)`` is single-valued upstream and this function
    enforces that on the way in.  It has to: the live table has no UNIQUE index
    (audit P2.2) and holds 442 byte-identical duplicate rows out of 5,522, and
    counting the same week twice shifts every rolling window it sits in — which
    is exactly what the old ``scripts/`` implementation did.

    Two kinds of duplicate, counted separately:

    * **byte-identical** — the same report stored twice.  Dropped silently
      (well, counted); there is nothing to choose between.
    * **conflicting** — the same ``(entity, observed_at)`` with a *different*
      payload, i.e. a revision or a collector bug.  The most recently ingested
      row wins, deterministically, and the count is logged.  Keeping both would
      double-count the week *and* push the ambiguity into the derived table,
      where the consumers resolve it by SELECT order.

    Read-only: ``PipelineStore`` exposes no query-by-observation-type API, so
    this borrows its connection the same way ``agent/models/gnn/graph_builder``
    does.  Writes go through ``store_entity_observation`` and nowhere else.
    """
    conn = store._get_conn()  # noqa: SLF001 — store exposes no query-by-type API
    # ingested_at then id ascending: the last row seen for a timestamp is the
    # newest, which matches the keep-rule of the planned P2.2 dedup migration.
    rows = conn.execute(
        "SELECT entity_id, observed_at, value_json, ingested_at, id "
        "FROM entity_observations WHERE observation_type=? "
        "ORDER BY entity_id, observed_at, ingested_at, id",
        (RAW_OBSERVATION_TYPE,),
    ).fetchall()

    # entity_id → observed_at → (value_json, parsed value); dict preserves the
    # ascending insertion order the query guarantees.
    staged: dict[str, dict[float, tuple[str, dict[str, Any]]]] = defaultdict(dict)
    duplicates_dropped = 0
    conflicts_collapsed = 0
    unparseable = 0

    for row in rows:
        entity_id = row["entity_id"]
        raw_json = row["value_json"]
        try:
            value = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
        except (json.JSONDecodeError, TypeError):
            log.warning(
                "Unparseable %s value_json for entity=%s — skipped",
                RAW_OBSERVATION_TYPE,
                entity_id,
            )
            unparseable += 1
            continue
        if not isinstance(value, dict):
            unparseable += 1
            continue

        observed_at = float(row["observed_at"])
        canonical = raw_json if isinstance(raw_json, str) else json.dumps(raw_json, sort_keys=True)
        prior = staged[entity_id].get(observed_at)
        if prior is not None:
            if prior[0] == canonical:
                duplicates_dropped += 1
            else:
                conflicts_collapsed += 1
        staged[entity_id][observed_at] = (canonical, value)

    by_entity: dict[str, list[tuple[float, dict[str, Any]]]] = {
        entity_id: [(ts, payload) for ts, (_json, payload) in sorted(per_ts.items())]
        for entity_id, per_ts in staged.items()
    }

    if duplicates_dropped or conflicts_collapsed or unparseable:
        log.info(
            "%s: collapsed %d byte-identical duplicate rows, %d conflicting rows "
            "(newest ingested_at kept), skipped %d unparseable",
            RAW_OBSERVATION_TYPE,
            duplicates_dropped,
            conflicts_collapsed,
            unparseable,
        )
    if conflicts_collapsed:
        log.warning(
            "%s holds %d (entity_id, observed_at) pairs with more than one distinct "
            "payload — the COT publishes one report per contract per date, so this is "
            "a revision or a collector bug, not two observations.",
            RAW_OBSERVATION_TYPE,
            conflicts_collapsed,
        )

    return RawPositioning(
        by_entity=by_entity,
        duplicates_dropped=duplicates_dropped,
        conflicts_collapsed=conflicts_collapsed,
        unparseable=unparseable,
    )


def coverage_stats(
    by_entity: dict[str, list[tuple[float, dict[str, Any]]]],
    weeks: int = COVERAGE_WINDOW_WEEKS,
) -> dict[str, Any]:
    """How many of the last *weeks* COT report dates the input actually holds.

    The derivation can only be as complete as ``fetch_cftc`` made it, so a
    caller that sees ``rows_written > 0`` still learns nothing about whether
    the weeks are there.  This counts distinct report dates present in the
    trailing window ending at the newest observation, snapping each timestamp
    to its weekly slot (real COT dates drift by a day around holidays, so the
    grid is nearest-week, not exact-multiple).

    Returns zeros with ``ratio = 0.0`` for empty input.
    """
    dates = sorted({ts for observations in by_entity.values() for ts, _ in observations})
    if not dates or weeks < 1:
        return {
            "window_weeks": weeks,
            "report_dates_expected": weeks,
            "report_dates_present": 0,
            "ratio": 0.0,
            "latest_observed_at": None,
        }

    latest = dates[-1]
    slots = {round((ts - latest) / SECONDS_PER_WEEK) for ts in dates}
    present = {slot for slot in slots if -(weeks - 1) <= slot <= 0}
    return {
        "window_weeks": weeks,
        "report_dates_expected": weeks,
        "report_dates_present": len(present),
        "ratio": round(len(present) / weeks, 4),
        "latest_observed_at": latest,
    }


def _existing_derived_index(store: PipelineStore) -> dict[tuple[str, float], list[tuple[int, str]]]:
    """``(entity_id, observed_at)`` → ``[(row_id, value_json), ...]`` already stored.

    Includes the rows the old ``scripts/`` implementation wrote: they carry the
    same ``source_tool`` and ``observation_type``, so they are superseded by
    the same code path as any other stale row — no one-off migration needed.
    """
    conn = store._get_conn()  # noqa: SLF001 — store exposes no query-by-type API
    rows = conn.execute(
        "SELECT id, entity_id, observed_at, value_json FROM entity_observations "
        "WHERE source_tool=? AND observation_type=? "
        "ORDER BY ingested_at DESC, id DESC",
        (SOURCE_TOOL, DERIVED_OBSERVATION_TYPE),
    ).fetchall()

    index: dict[tuple[str, float], list[tuple[int, str]]] = defaultdict(list)
    for row in rows:
        index[(row["entity_id"], float(row["observed_at"]))].append((int(row["id"]), row["value_json"]))
    return dict(index)


def _delete_superseded(store: PipelineStore, row_ids: list[int]) -> int:
    """Delete derived rows made obsolete by a fresher value at the same timestamp.

    Scoped to ``(source_tool, observation_type)`` in the WHERE clause as well as
    by id, so a mis-collected id cannot reach a row this module does not own.
    This is the one statement here that is not a ``store_entity_observation``
    call: the store has no replace-on-timestamp method, and adding one belongs
    to ``store.py``'s owner.  It removes rows only — never inserts, never edits
    a value — so the write boundary's ``observed_at`` guard is untouched.
    """
    if not row_ids:
        return 0
    conn = store._get_conn()  # noqa: SLF001
    deleted = 0
    for start in range(0, len(row_ids), 500):  # keep well under SQLITE_MAX_VARIABLE_NUMBER
        chunk = row_ids[start : start + 500]
        placeholders = ",".join("?" * len(chunk))
        cursor = conn.execute(
            f"DELETE FROM entity_observations WHERE id IN ({placeholders}) "  # noqa: S608 — placeholders only
            "AND source_tool=? AND observation_type=?",
            (*chunk, SOURCE_TOOL, DERIVED_OBSERVATION_TYPE),
        )
        deleted += cursor.rowcount
    conn.commit()
    return deleted


def derive_and_store(store: PipelineStore) -> dict[str, Any]:
    """Derive features for every CFTC entity and persist them.

    Inserts go through :meth:`PipelineStore.store_entity_observation` — the
    single write boundary — so the ``observed_at`` range guard and the
    idempotency collapse both apply.  On top of that this function enforces the
    invariant the store cannot: **exactly one derived row per
    ``(entity_id, observed_at)``**.  A timestamp whose value changed has its old
    row deleted after the new one lands; a timestamp whose value is unchanged is
    not written at all.

    Not writing the unchanged rows matters beyond cost.  The store refreshes
    ``ingested_at`` on a collapsed duplicate, so the previous version of this
    function touched all 5,080 rows on a run that wrote nothing — which pins
    ``max_ingested_at`` to "now" every day and permanently disables both
    write-activity detectors in ``scripts/check_freshness.py``
    (``REASON_SILENT``, ``REASON_NO_NEW_ROWS``) for this source.  A no-op run
    must now leave the clock alone.

    Returns
    -------
    dict with ``rows_written`` (rows genuinely inserted), ``rows_superseded``
    (stale rows deleted because the value at that timestamp changed),
    ``rows_unchanged`` / ``deduplicated`` (already stored, not rewritten),
    ``rows_derived`` (rows considered), ``entities``, ``max_observed_at``
    (``None`` when nothing was derived), ``raw_duplicates_dropped``,
    ``raw_conflicts_collapsed`` and ``coverage`` (see :func:`coverage_stats`).
    """
    raw = load_raw_positioning(store)
    by_entity = raw.by_entity
    if not by_entity:
        log.warning("No %s observations in the database — nothing to derive.", RAW_OBSERVATION_TYPE)
        return {
            "rows_written": 0,
            "rows_superseded": 0,
            "rows_unchanged": 0,
            "rows_orphaned": 0,
            "rows_derived": 0,
            "entities": 0,
            "max_observed_at": None,
            "deduplicated": 0,
            "raw_duplicates_dropped": raw.duplicates_dropped,
            "raw_conflicts_collapsed": raw.conflicts_collapsed,
            "coverage": coverage_stats({}),
        }

    existing = _existing_derived_index(store)

    rows_derived = 0
    rows_written = 0
    rows_unchanged = 0
    rows_superseded = 0
    max_observed_at: float | None = None

    for entity_id, observations in by_entity.items():
        for observed_at, value in derive_series(observations):
            rows_derived += 1
            if max_observed_at is None or observed_at > max_observed_at:
                max_observed_at = observed_at

            key = (entity_id, float(observed_at))
            prior = existing.pop(key, [])
            # Must match PipelineStore.store_entity_observation byte for byte,
            # or an unchanged row looks changed and gets rewritten every day.
            value_json = json.dumps(value, default=str)
            identical = [row_id for row_id, stored_json in prior if stored_json == value_json]

            if len(prior) == 1 and identical:
                # Already correct and unambiguous — do not touch ingested_at.
                rows_unchanged += 1
                continue

            row_id = store.store_entity_observation(
                entity_id=entity_id,
                source_tool=SOURCE_TOOL,
                observed_at=observed_at,
                observation_type=DERIVED_OBSERVATION_TYPE,
                depth_level=DEPTH_LEVEL,
                value=value,
                metadata={"source": "agent.quant.cftc_features"},
            )
            if identical:
                rows_unchanged += 1
            else:
                rows_written += 1
            # Delete the old row(s) immediately, not in a batch at the end: if
            # this run dies half way, every key it has already touched is left
            # single-valued rather than holding both the old and new value.
            rows_superseded += _delete_superseded(store, [prior_id for prior_id, _ in prior if prior_id != row_id])

    coverage = coverage_stats(by_entity)
    # Keys left in `existing` were never re-derived: a derived row whose raw
    # parent is no longer in the database.  Counted, not deleted — this run did
    # not author them and cannot prove they are wrong.
    rows_orphaned = sum(len(v) for v in existing.values())

    if rows_superseded:
        log.warning(
            "cftc_derived: superseded %d stale derived row(s) whose value changed at an "
            "already-stored (entity_id, observed_at) — re-derivation rewrote history.",
            rows_superseded,
        )
    if rows_orphaned:
        log.warning(
            "cftc_derived: %d derived row(s) have no %s row left at their "
            "(entity_id, observed_at) — left in place, not superseded.",
            rows_orphaned,
            RAW_OBSERVATION_TYPE,
        )
    if coverage["ratio"] < COVERAGE_WARN_RATIO:
        log.warning(
            "cftc_derived: only %d of the last %d weekly COT report dates are present in "
            "%s — the derivation cannot create weeks the Layer 1 collector never fetched "
            "(fetch_cftc runs mode='latest'; mode='historical' is the only backfill path).",
            coverage["report_dates_present"],
            coverage["report_dates_expected"],
            RAW_OBSERVATION_TYPE,
        )

    log.info(
        "cftc_derived: %d inserted, %d superseded, %d unchanged (%d derived) across %d "
        "entities, max observed_at=%s, %d-week coverage=%.0f%%",
        rows_written,
        rows_superseded,
        rows_unchanged,
        rows_derived,
        len(by_entity),
        max_observed_at,
        coverage["window_weeks"],
        coverage["ratio"] * 100,
    )
    return {
        "rows_written": rows_written,
        "rows_superseded": rows_superseded,
        "rows_unchanged": rows_unchanged,
        "rows_orphaned": rows_orphaned,
        "rows_derived": rows_derived,
        "entities": len(by_entity),
        "max_observed_at": max_observed_at,
        # Back-compat alias: the DAG node reports this as "already stored".
        "deduplicated": rows_unchanged,
        "raw_duplicates_dropped": raw.duplicates_dropped,
        "raw_conflicts_collapsed": raw.conflicts_collapsed,
        "coverage": coverage,
    }
