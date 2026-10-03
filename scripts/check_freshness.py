#!/usr/bin/env python3
"""Collection freshness watchdog — is every source still actually landing rows?

Why this exists
---------------
Collection stopped on 2026-08-27 and nobody noticed for 26 days.  Nothing was
broken in a way anyone could see: the DAG nodes were green, the collectors
returned HTTP 200, and "no new rows today" is byte-for-byte indistinguishable
from "ran fine, the source had nothing new".  For a *time-gated* source — one
whose API only serves a recent window — those 26 days are gone at any price.
That data is the project's moat, so silence has to be an alarm, not a default.

The watchdog reads the live database **read-only** and answers three different
questions, because the 26-day outage looks different under each:

1. **Is the data stale?**  ``now - MAX(observed_at)`` per source, against a
   per-source cadence in :data:`CADENCE_DAYS`.  Catches "the source stopped
   publishing" and "we stopped collecting", without distinguishing them.

2. **Has the collector written anything at all?**  ``now - MAX(ingested_at)``
   per source, against :data:`COLLECTOR_SILENCE_DAYS`.  This is the clock that
   would have caught the outage on day 4 instead of day 26.  It is a *separate*
   clock on purpose: ``sanctions_monitor`` on the live DB right now has data 63
   days old (OFAC simply has not published) but wrote rows an hour ago — healthy
   collector, quiet source.  ``cert_transparency`` has data 37 days old and has
   not written in 27 — dead collector.  One staleness number cannot tell those
   apart; two can.
   Note this clock works even when a source is re-fetching unchanged records:
   ``PipelineStore.store_entity_observation`` refreshes ``ingested_at`` on a
   duplicate instead of dropping it, so a live collector keeps the clock moving.

3. **Did a run that reported success actually write rows?**  With
   ``--since-snapshot <db copy>``, per-source ``COUNT(*)`` then vs now.  A green
   node with zero growth is this project's signature bug, and neither clock
   above sees it on the day it happens.  A *negative* delta is graded too and
   separately (``ROWS_LOST``): a run that deleted 250 rows while writing one is
   not a healthy run, and the destructive-migration shape has already been
   caught once in this repo.

4. **Did the run yield the volume it normally yields?**  Zero is not the only
   failing row count.  The F-16 EDGAR bug returned HTTP 200, wrote rows, and
   silently dropped 72% of every window — every clock above reads green on that
   day.  ``LOW_VOLUME`` compares the run's yield against the source's own
   trailing median rows-per-active-day, so a collector that falls off a cliff
   without falling to zero is still visible.

5. **Is a collection node running but landing nothing?**  ``node ran`` and
   ``node produced observations`` are two different signals and this repo has
   confused them before.  The expected-source list is therefore *not* derived
   from the data being audited — it is derived from the table names the shipped
   collection DAG declares, so a collector that has never landed a single row
   is MISSING rather than simply absent from the report.

Usage
-----
    python scripts/check_freshness.py
    python scripts/check_freshness.py --db .tirra_pipeline/pipeline.db

    # before a collection run — VACUUM INTO, never `cp`.  pipeline.db is WAL
    # with a live writer, so a plain file copy can omit everything committed
    # since the last checkpoint; the resulting baseline is too small, new_rows
    # is inflated, and NO_NEW_ROWS is suppressed.  The bias of a torn copy is
    # toward false green, which is the one direction this script must never
    # fail in.  VACUUM INTO is read-only, WAL-correct and atomic.
    sqlite3 .tirra_pipeline/pipeline.db "VACUUM INTO '/tmp/snap.db'"
    python scripts/check_freshness.py --since-snapshot /tmp/snap.db   # after

Exit codes (a watchdog that only prints is a watchdog nobody wires to cron):

    0  everything inside its threshold
    1  a BACKFILLABLE source is stale/silent — refill it, no permanent loss
    2  a TIME_GATED source is stale/silent — LOUDER: the gap is unrecoverable
    3  the watchdog itself could not run (missing/unreadable DB)

Safety
------
This script never writes to the database it inspects.  It opens every path with
``mode=ro`` through a ``file:`` URI and additionally sets ``PRAGMA query_only``,
so even a coding mistake downstream fails closed with an sqlite3 error rather
than mutating the project's only irreplaceable asset.
"""

from __future__ import annotations

import argparse
import ast
import sqlite3
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DB_PATH = ".tirra_pipeline/pipeline.db"

# ── classification ────────────────────────────────────────────
#
# TIME_GATED: the upstream API serves only a recent window, so a day missed is
# a day that can never be bought back.  A gap here is permanent data loss and
# gets the loud exit code.
#
# Everything else is BACKFILLABLE: the API serves history, so a gap is an
# annoyance and a backfill script, not a loss.
TIME_GATED: frozenset[str] = frozenset(
    {
        "ais_vessel",  # AIS feeds serve live/recent positions only; no position history API
        "whale_alert",  # free tier serves ~recent transactions; the archive is paywalled
        "form144",  # EDGAR keeps the filing, but the daily index window we scrape rolls
        "options_chain",  # quotes are point-in-time; historical chains are a paid product
        "dns_monitor",  # a DNS answer is observable only while it is live
        "finra_short_volume",  # daily files age out of the free endpoint
        "bankruptcy_court",  # free court feeds expose a recent docket window
        "political_risk",  # event feed, no historical query
        "regulatory_gazette",  # gazette front pages roll; older issues need paid archives
        "academic_preprints",  # the "recent submissions" listing is a rolling window
        "global_pmi",  # releases are republished with revisions; the first print is not re-servable
        "cert_transparency",  # CT log tailing is positional — you cannot ask for last month
    }
)

# ── per-source cadence thresholds (days) ──────────────────────
#
# "How many days of silence is still normal for this source?"  Each number is
# the source's real publication rhythm plus slack for the collector's own
# schedule (daily_collection runs ``0 18 * * 1-5`` — weekdays only, so any daily
# source must tolerate a weekend: 3 days minimum, never 1).
#
# The numbers are grounded in the median inter-observation gap measured on the
# live DB, not guessed; where the measured gap is unstable because the source is
# new or the pipeline itself was down, the upstream publication schedule wins.
CADENCE_DAYS: dict[str, float] = {
    # — daily-publishing sources (measured median gap 1 day) —
    "ais_vessel": 3.0,  # vessel snapshot every weekday run; weekend + one miss
    "defi_flows": 3.0,  # on-chain, continuous; largest table in the DB
    "instrument_universe": 3.0,  # daily closes; only trading days, so weekend slack
    "polymarket": 3.0,  # live market quotes, refreshed every run
    "sovereign_debt": 3.0,  # daily yield curve points
    "whale_alert": 3.0,  # continuous transaction stream; silence means we stopped
    "gdelt": 3.0,  # 15-minute feed stored day-granular; never legitimately quiet
    # — business-day sources (need weekend slack plus a settlement lag) —
    "form144": 4.0,  # SEC filings land each business day, index is T+0
    "insider_filings": 4.0,  # Forms 3/4/5, business days
    "finra_short_volume": 4.0,  # daily short-volume file, published T+1
    "options_chain": 4.0,  # quotes on trading days only
    "dns_monitor": 4.0,  # our own probe; cadence is the DAG's, not a vendor's
    "cert_transparency": 4.0,  # measured median gap 2.5 days across log tailing runs
    "wikipedia_pageviews": 4.0,  # REST API publishes with a 1–2 day lag
    "academic_preprints": 4.0,  # arXiv posts every weekday; quiet weekends are normal
    "creditor_filings": 5.0,  # measured median gap 2 days, p90 6
    "gov_contracts": 5.0,  # award notices post on business days
    "regulatory_gazette": 5.0,  # gazettes publish on business days, some skip
    "bankruptcy_court": 5.0,  # filings are business-day, lumpy
    "supply_chain_monitor": 5.0,  # port/logistics indices, business-day updates
    "transport_throughput": 5.0,  # same family, same rhythm
    # — weekly sources —
    "cftc": 10.0,  # COT publishes Friday for Tuesday; one skipped week is survivable
    "cftc_derived": 10.0,  # derived from cftc by scripts/add_cftc_derived_features.py,
    #                        so it inherits the weekly cadence — but it is a *script*,
    #                        not a DAG collector, which is why it drifts furthest
    "energy_supply": 10.0,  # EIA weekly petroleum status report (measured gap: exactly 7)
    "disease_surveillance": 10.0,  # WHO/ECDC weekly bulletins
    # — monthly / irregular sources —
    "central_bank_balance": 40.0,  # monthly balance-sheet releases (measured gap 29)
    "consumer_sentiment": 40.0,  # monthly index print
    "global_pmi": 40.0,  # monthly PMI print, first business day of the month
    "dividend_data": 40.0,  # measured median gap 21, p90 35 — declaration-driven
    "capital_flows": 45.0,  # TIC-style monthly data with a ~2 month publication lag
    "comtrade": 45.0,  # UN Comtrade monthly, long and irregular lag
    "drug_regulatory": 30.0,  # approvals are event-driven; measured median gap 19
    "political_risk": 14.0,  # event feed: genuinely quiet stretches, but it is time-gated,
    #                          so the threshold stays tighter than the data's own p90
    "sanctions_monitor": 21.0,  # OFAC/EU updates are irregular; the collector-silence
    #                             clock below is the real guard for this one
}

#: Sources present in the DB but absent from :data:`CADENCE_DAYS` are checked
#: against this.  Deliberately loose — an unrecognised source gets reported as
#: UNKNOWN so it is added to the table above, rather than silently policed by a
#: number nobody chose.
DEFAULT_CADENCE_DAYS = 7.0

#: How long a collector may write *nothing at all* before that is a failure,
#: regardless of what its source is publishing.  daily_collection runs weekdays
#: (``0 18 * * 1-5``), so a Friday run followed by a Monday run is 3 days apart;
#: 4 allows one missed run on top of a weekend.  This is the clock that turns a
#: 26-day outage into a day-4 alarm.
COLLECTOR_SILENCE_DAYS = 4.0

#: ``observed_at`` may legitimately lead wall clock by this much (timezone slop,
#: vendor clock skew).  Mirrors ``PipelineStore.OBSERVED_AT_MAX_AHEAD_SECONDS``.
#: Anything beyond it is a mis-parsed date, not an observation — the live DB
#: holds gov_contracts rows dated 2030, and a naive ``now - MAX(observed_at)``
#: would score that source as 1225 days *fresh*.
FUTURE_SKEW_DAYS = 1.0

SECONDS_PER_DAY = 86_400.0

# ── where the expected-source list actually comes from ────────
#
# The original version of this script derived its universe from the database it
# was auditing (``set(rows) | set(CADENCE_DAYS)``), and CADENCE_DAYS itself was
# measured off that same database.  That makes a collector which has *never*
# landed a row structurally invisible: it is absent from the data, so it is
# absent from the report, and the MISSING guard can never fire.  On the live DB
# that was provably a no-op — every configured source already had rows.
#
# The universe therefore comes from the shipped collection DAG instead: the
# ``table_name=`` each collection node declares (executor.py stores each node's
# envelope under ``node.table_name or node.id``).  The DAG module is *parsed*,
# never imported — this script is stdlib-only by design and the unit that runs
# it must not depend on importing the whole agent package.
DAG_MODULE_PATH = Path(__file__).resolve().parent.parent / "agent" / "pipeline" / "dags" / "daily_collection.py"

#: A few collection nodes declare a ``table_name`` that differs from the
#: ``source_tool`` their observations actually carry.  Without this, the two
#: below would be reported MISSING forever while their data lands under
#: another name.
NODE_TABLE_ALIASES: dict[str, str] = {
    "ais_vessel_tracking": "ais_vessel",
    "supply_chain_prices": "supply_chain_monitor",
}

#: Collection nodes that only ever land a raw ``pipeline_data`` envelope: their
#: tool module contains no ``store_entity_observation`` call at all, so grading
#: them against ``entity_observations`` would be a permanent false alarm — and
#: an unclearable red teaches an operator to ignore the whole report.  They are
#: still graded, on the node clock, against ``pipeline_data.fetched_at``.
#:
#: This set is not a silencer: ``tests/test_freshness_watchdog.py`` asserts each
#: member's tool module really has no observation write, so a broken collector
#: cannot be quieted by adding its name here.
OBSERVATION_EXEMPT_TABLES: frozenset[str] = frozenset(
    {
        "building_permits",
        "earthquake_proximity",
        "internet_infrastructure",
        "job_postings",
        "labor_disruptions",
        "macro_data",
        "power_grid",
        "satellite_activity",
        "treasury_receipts",
        "weather_alerts",
    }
)

#: How long a *node* may go without storing a ``pipeline_data`` envelope.  Same
#: reasoning as COLLECTOR_SILENCE_DAYS: weekdays plus one missed run.
NODE_SILENCE_DAYS = 4.0

# ── volume floor ──────────────────────────────────────────────
#
# "rows > 0" is not "the collector worked".  F-16: a transient 5xx dropped 72%
# of every EDGAR window while the node stayed green and kept writing rows.  The
# floor compares a run's yield against the source's own trailing median
# rows-per-active-day, which is robust to the lumpy backfill days that dominate
# this DB's history.
#
# The thresholds are deliberately conservative — measured against the live DB,
# they flag 2 of 34 sources, both genuine collapses (ais_vessel 4 rows against a
# 50-row median, gdelt 1,452 against 45,379).  A noisy volume alarm would be
# worse than none: it retrains the operator to ignore the report.
VOLUME_FLOOR_FRACTION = 0.25
#: Below this median the day-to-day noise of a small source swamps the signal,
#: so it is not graded on volume at all (comtrade has 6 rows *lifetime*).
VOLUME_MIN_MEDIAN_ROWS = 50.0
#: Fewer completed active days than this and there is no baseline worth the name.
VOLUME_MIN_ACTIVE_DAYS = 3

# ── exit codes ────────────────────────────────────────────────
EXIT_OK = 0
EXIT_BACKFILLABLE = 1
EXIT_TIME_GATED = 2
EXIT_WATCHDOG_ERROR = 3

# Reason codes, ordered loosely by how much they should scare you.
REASON_STALE = "STALE"  # data older than the source's cadence
REASON_SILENT = "SILENT"  # collector has written nothing recently
REASON_MISSING = "MISSING"  # configured source has no rows at all
REASON_NO_NEW_ROWS = "NO_NEW_ROWS"  # snapshot mode: no new rows AND no writes at all
REASON_ROWS_LOST = "ROWS_LOST"  # snapshot mode: the table *shrank*
REASON_LOW_VOLUME = "LOW_VOLUME"  # wrote rows, but far fewer than it normally does
REASON_FUTURE = "FUTURE_DATED"  # observed_at beyond now + skew
REASON_UNKNOWN = "UNKNOWN_SOURCE"  # not in CADENCE_DAYS
REASON_NODE_SILENT = "NODE_SILENT"  # node clock: no pipeline_data envelope recently
REASON_NODE_NEVER_RAN = "NODE_NEVER_RAN"  # node clock: no envelope, ever

#: Reasons that describe a data-quality defect rather than a collection
#: failure.  They are printed, but they do not page.
#:
#: REASON_FUTURE is here because of a concrete trap: the live DB holds 912
#: gov_contracts rows dated 2030, all ingested before 2026-08-27, and
#: ``PipelineStore``'s observed_at guard now rejects such writes — so no
#: collector run can ever clear them and only a DELETE on the live DB could.
#: Left as severity 1 it pinned ``tirra-freshness.service`` in ``failed``
#: permanently, which is precisely the alarm fatigue that let a 26-day outage go
#: unnoticed.  It cannot hide staleness either: the ageing query already ignores
#: future timestamps.  So it is a report, not an alarm.
NON_PAGING_REASONS: frozenset[str] = frozenset({REASON_FUTURE})


class WatchdogError(RuntimeError):
    """The watchdog could not inspect the database (exit code 3)."""


@dataclass(frozen=True)
class SourceRow:
    """Raw per-source aggregates straight out of ``entity_observations``."""

    source: str
    rows: int
    max_observed_at: float | None  # ignoring implausible future timestamps
    max_observed_at_raw: float | None  # including them, to detect the anomaly
    max_ingested_at: float | None


@dataclass
class SourceStatus:
    """A source's verdict."""

    source: str
    time_gated: bool
    threshold_days: float
    rows: int
    data_age_days: float | None
    write_age_days: float | None
    new_rows: int | None = None  # snapshot mode only
    reasons: list[str] = field(default_factory=list)

    @property
    def source_class(self) -> str:
        return "TIME_GATED" if self.time_gated else "BACKFILLABLE"

    @property
    def ok(self) -> bool:
        return not self.reasons

    @property
    def verdict(self) -> str:
        return "OK" if self.ok else "+".join(self.reasons)

    @property
    def severity(self) -> int:
        """0 fine, 1 needs a backfill, 2 permanent loss in progress."""
        return _severity(self.reasons, time_gated=self.time_gated)


@dataclass
class NodeStatus:
    """A *collection node*'s verdict, graded on ``pipeline_data`` envelopes.

    Deliberately a separate namespace from :class:`SourceStatus`: a node is
    keyed by the ``table_name`` it declares in the DAG, an observation by the
    ``source_tool`` it carries, and those two are not the same set (``gdelt`` is
    both; ``ais_vessel_tracking`` is only ever the former).  Collapsing them
    into one table is how "the node ran" and "the node produced data" got
    confused in the first place.
    """

    table: str
    time_gated: bool
    envelope_age_days: float | None
    envelopes: int
    observation_source: str | None  # the source_tool its rows land under, if any
    reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.reasons

    @property
    def verdict(self) -> str:
        return "OK" if self.ok else "+".join(self.reasons)

    @property
    def severity(self) -> int:
        return _severity(self.reasons, time_gated=self.time_gated)


def _severity(reasons: list[str], *, time_gated: bool) -> int:
    """0 fine, 1 needs a backfill, 2 permanent loss in progress."""
    if not reasons:
        return EXIT_OK
    paging = [r for r in reasons if r not in NON_PAGING_REASONS]
    if not paging:
        # Only data-quality notes.  Printed, never paged — see
        # NON_PAGING_REASONS for why an unclearable red is its own failure.
        return EXIT_OK
    # An unrecognised source is a bookkeeping gap, not a lost-data event, so it
    # never escalates to the time-gated alarm on its own.
    hard = [r for r in paging if r != REASON_UNKNOWN]
    if not hard:
        return EXIT_BACKFILLABLE
    return EXIT_TIME_GATED if time_gated else EXIT_BACKFILLABLE


def connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    """Open *db_path* strictly read-only, or raise :class:`WatchdogError`.

    Two independent locks: the ``mode=ro`` URI (sqlite refuses writes and will
    not create the file) and ``PRAGMA query_only`` (refuses writes even on a
    connection that somehow got opened writable).
    """
    path = Path(db_path)
    if not path.exists():
        raise WatchdogError(f"database not found: {path}")
    uri = f"file:{path}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
        conn.execute("PRAGMA query_only = 1")
    except sqlite3.Error as exc:  # pragma: no cover - depends on fs state
        raise WatchdogError(f"cannot open {path} read-only: {exc}") from exc
    conn.row_factory = sqlite3.Row
    return conn


def read_source_rows(conn: sqlite3.Connection, *, now: float) -> dict[str, SourceRow]:
    """Aggregate ``entity_observations`` per ``source_tool``."""
    horizon = now + FUTURE_SKEW_DAYS * SECONDS_PER_DAY
    try:
        cursor = conn.execute(
            """
            SELECT source_tool                                    AS source,
                   COUNT(*)                                       AS rows_total,
                   MAX(CASE WHEN observed_at <= ? THEN observed_at END) AS max_observed,
                   MAX(observed_at)                               AS max_observed_raw,
                   MAX(ingested_at)                               AS max_ingested
            FROM entity_observations
            GROUP BY source_tool
            """,
            (horizon,),
        )
        fetched = cursor.fetchall()
    except sqlite3.Error as exc:
        raise WatchdogError(f"cannot read entity_observations: {exc}") from exc

    return {
        row["source"]: SourceRow(
            source=row["source"],
            rows=int(row["rows_total"]),
            max_observed_at=row["max_observed"],
            max_observed_at_raw=row["max_observed_raw"],
            max_ingested_at=row["max_ingested"],
        )
        for row in fetched
    }


def read_node_envelopes(conn: sqlite3.Connection) -> dict[str, tuple[int, float]]:
    """Per ``pipeline_data.source``: ``(envelope count, MAX(fetched_at))``.

    ``pipeline_data`` is where :mod:`agent.pipeline.executor` records that a
    node ran and stored *something*, under ``node.table_name or node.id``.  It
    answers a question ``entity_observations`` cannot: did the node execute at
    all?  A node can run every day forever and never land an observation, and
    before this the watchdog could not tell that apart from a node that was
    deleted.
    """
    try:
        cursor = conn.execute(
            "SELECT source, COUNT(*) AS n, MAX(fetched_at) AS last FROM pipeline_data GROUP BY source"
        )
        fetched = cursor.fetchall()
    except sqlite3.Error as exc:
        raise WatchdogError(f"cannot read pipeline_data: {exc}") from exc
    return {row["source"]: (int(row["n"]), row["last"]) for row in fetched}


def read_daily_write_counts(conn: sqlite3.Connection) -> dict[str, list[tuple[int, int]]]:
    """Per source, ``[(epoch day, rows written that day), ...]`` ascending.

    Keyed on ``ingested_at`` (when *we* wrote it), not ``observed_at`` (when the
    world made it), because the question is "how much did the collector yield
    on the days it ran", not "how much did the source publish".
    """
    try:
        cursor = conn.execute(
            """
            SELECT source_tool                        AS source,
                   CAST(ingested_at / 86400 AS INTEGER) AS day,
                   COUNT(*)                           AS n
            FROM entity_observations
            WHERE ingested_at IS NOT NULL
            GROUP BY source_tool, day
            ORDER BY source_tool, day
            """
        )
        fetched = cursor.fetchall()
    except sqlite3.Error as exc:
        raise WatchdogError(f"cannot read entity_observations: {exc}") from exc

    out: dict[str, list[tuple[int, int]]] = {}
    for row in fetched:
        out.setdefault(row["source"], []).append((int(row["day"]), int(row["n"])))
    return out


def declared_collection_tables(path: Path | str = DAG_MODULE_PATH) -> frozenset[str]:
    """Every ``table_name="..."`` the collection DAG declares.

    Parsed with :mod:`ast`, not imported: ``build_daily_collection_dag``'s
    default ``db_path`` is the live database, and this script must never be one
    import away from opening it writable.  Parsing also keeps the watchdog
    stdlib-only, which is why the systemd unit needs no EnvironmentFile.

    A missing or unparseable DAG module raises :class:`WatchdogError` rather
    than quietly degrading to "no nodes to check" — a watchdog that reports
    green because it could not find the thing it audits is the bug, not a
    graceful fallback.
    """
    source_path = Path(path)
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    except (OSError, SyntaxError) as exc:
        raise WatchdogError(f"cannot read collection DAG at {source_path}: {exc}") from exc

    tables: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg == "table_name" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                tables.add(kw.value.value)
    if not tables:
        raise WatchdogError(f"no table_name= declarations found in {source_path} — parser or DAG changed shape")
    return frozenset(tables)


def expected_sources(tables: frozenset[str]) -> frozenset[str]:
    """The source_tool names that *must* have rows, from the DAG plus cadences.

    Not derived from the database being audited.  That is the whole point: a
    collector which has never written a row has to be able to show up as
    MISSING, and it cannot do that if its name only exists in the data it
    failed to produce.
    """
    from_dag = {NODE_TABLE_ALIASES.get(table, table) for table in tables if table not in OBSERVATION_EXEMPT_TABLES}
    return frozenset(from_dag | set(CADENCE_DAYS))


def _age_days(timestamp: float | None, now: float) -> float | None:
    if timestamp is None:
        return None
    return (now - timestamp) / SECONDS_PER_DAY


def _volume_baseline(days: list[tuple[int, int]], *, before_day: int | None = None) -> float | None:
    """Trailing median rows-per-active-day, or ``None`` if there is no baseline.

    ``before_day`` drops the current (still-running) day: grading a partial day
    against complete ones manufactures a LOW_VOLUME every afternoon.  The most
    recent complete day is also excluded from the median itself, because it is
    the day being *graded*.

    Median, not mean: this DB's history is dominated by a handful of enormous
    backfill days, and a mean baseline would be set by them alone.
    """
    complete = [n for day, n in days if before_day is None or day < before_day]
    if len(complete) < VOLUME_MIN_ACTIVE_DAYS:
        return None
    prior = complete[:-1] if before_day is not None else complete
    if not prior:
        return None
    median = float(statistics.median(prior))
    return median if median >= VOLUME_MIN_MEDIAN_ROWS else None


def evaluate(
    rows: dict[str, SourceRow],
    *,
    now: float,
    snapshot_rows: dict[str, SourceRow] | None = None,
    require_all_configured: bool = True,
    expected: frozenset[str] | None = None,
    daily_counts: dict[str, list[tuple[int, int]]] | None = None,
    snapshot_daily_counts: dict[str, list[tuple[int, int]]] | None = None,
) -> list[SourceStatus]:
    """Turn raw aggregates into verdicts, worst first.

    ``snapshot_rows`` enables the green-but-zero-rows check.  Row growth alone
    is too blunt to fail on: a healthy collector that re-fetches an unchanged
    record adds no row.  But it is not silent either —
    ``store_entity_observation`` refreshes ``ingested_at`` on a duplicate — so
    the failing shape is **both** clocks flat: no new rows *and* no write
    activity at all since the snapshot.  That is a run that reported success
    and touched nothing, which is exactly the bug that hides behind a green
    checkmark.  It is independent of staleness: a source can be well inside its
    cadence and still have done nothing during the run being audited.

    A *shrinking* table is graded separately and unconditionally
    (``ROWS_LOST``).  It used to be folded into the ``new_rows <= 0`` test above
    and therefore suppressed by ``wrote_since``: a run that deleted 249 rows and
    wrote one rendered ``NEW = -249`` in the table and ``OK`` in the verdict
    column of the same line.  A legitimate negative delta exists — the P2.2
    dedup migration collapses 121,597 duplicates — so this is a reason an
    operator acknowledges, not a reason to stay quiet.

    ``daily_counts`` (and ``snapshot_daily_counts``) enable the volume floor,
    which is the only check here that fires while rows are still landing.

    ``require_all_configured`` decides what an expected source with no rows at
    all means.  On the live database it means a collector that has never landed
    anything, which is a failure (MISSING).  Against a partial database (a
    probe, a single-source fixture, a freshly initialised store) it would just
    be noise, so it can be turned off.
    """
    if expected is None:
        expected = frozenset(CADENCE_DAYS)
    statuses: list[SourceStatus] = []
    today = int(now // SECONDS_PER_DAY)
    # Expected sources that have never written a row still deserve a line.
    for source in sorted(set(rows) | set(expected)):
        row = rows.get(source)
        known = source in CADENCE_DAYS
        threshold = CADENCE_DAYS.get(source, DEFAULT_CADENCE_DAYS)
        status = SourceStatus(
            source=source,
            time_gated=source in TIME_GATED,
            threshold_days=threshold,
            rows=row.rows if row else 0,
            data_age_days=_age_days(row.max_observed_at, now) if row else None,
            write_age_days=_age_days(row.max_ingested_at, now) if row else None,
        )

        if not known and row is not None and row.rows:
            # UNKNOWN means "this landed rows and nobody chose a threshold for
            # it".  A DAG-declared collector that has never written anything is
            # MISSING, not unknown — tagging it both ways buries the actionable
            # word under a bookkeeping one.
            status.reasons.append(REASON_UNKNOWN)

        if row is None or row.rows == 0:
            if not require_all_configured:
                continue
            status.reasons.append(REASON_MISSING)
            statuses.append(status)
            continue

        if status.data_age_days is None or status.data_age_days > threshold:
            # data_age_days is None when *every* row of the source is
            # future-dated: there is no usable recent observation at all.
            status.reasons.append(REASON_STALE)

        if status.write_age_days is not None and status.write_age_days > COLLECTOR_SILENCE_DAYS:
            status.reasons.append(REASON_SILENT)

        if row.max_observed_at_raw is not None and row.max_observed_at_raw > now + FUTURE_SKEW_DAYS * SECONDS_PER_DAY:
            status.reasons.append(REASON_FUTURE)

        if snapshot_rows is not None:
            before = snapshot_rows.get(source)
            status.new_rows = row.rows - (before.rows if before else 0)
            previous_write = before.max_ingested_at if before else None
            wrote_since = (
                row.max_ingested_at is not None and previous_write is not None and row.max_ingested_at > previous_write
            ) or (previous_write is None and row.max_ingested_at is not None)
            if status.new_rows == 0 and not wrote_since:
                status.reasons.append(REASON_NO_NEW_ROWS)
            if status.new_rows < 0:
                # Never suppressed by wrote_since: writing one row does not
                # excuse deleting two hundred.
                status.reasons.append(REASON_ROWS_LOST)

        # ── volume floor ──
        # Snapshot mode grades the run's own delta against the history that
        # existed before it; without a snapshot it grades the most recent
        # *complete* day against the days before it.  Either way, a source that
        # wrote nothing at all is already covered by NO_NEW_ROWS / SILENT, so
        # the floor only speaks about runs that did produce rows.
        if snapshot_rows is not None and snapshot_daily_counts is not None:
            baseline = _volume_baseline(snapshot_daily_counts.get(source, []))
            if (
                baseline is not None
                and status.new_rows is not None
                and 0 < status.new_rows < VOLUME_FLOOR_FRACTION * baseline
            ):
                status.reasons.append(REASON_LOW_VOLUME)
        elif daily_counts is not None:
            days = daily_counts.get(source, [])
            baseline = _volume_baseline(days, before_day=today)
            complete = [n for day, n in days if day < today]
            if baseline is not None and complete and 0 < complete[-1] < VOLUME_FLOOR_FRACTION * baseline:
                status.reasons.append(REASON_LOW_VOLUME)

        statuses.append(status)

    statuses.sort(key=lambda s: (-s.severity, -(s.data_age_days or 0.0), s.source))
    return statuses


def evaluate_nodes(
    tables: frozenset[str],
    envelopes: dict[str, tuple[int, float]],
    *,
    now: float,
    require_all_nodes: bool = True,
) -> list[NodeStatus]:
    """Grade every DAG-declared collection node on whether it still *runs*.

    This is the signal ``entity_observations`` cannot carry.  Fifteen
    collection nodes on the live DB stored an envelope within the last three
    hours and have contributed zero observations, ever; the source table could
    not show them at all, because a source with no rows has no name in it.
    Here they at least have a line, and the ones whose tool does write
    observations are additionally reported MISSING by :func:`evaluate`.

    Only nodes the *current* DAG declares are graded.  ``pipeline_data`` still
    holds envelopes from node ids retired months ago (``fetch_gdelt``,
    ``train_gnn``); grading those would produce permanently red lines nobody can
    clear, and an unclearable red is how an alarm gets ignored.
    """
    statuses: list[NodeStatus] = []
    for table in sorted(tables):
        source = NODE_TABLE_ALIASES.get(table, table)
        count, last = envelopes.get(table, (0, None))
        status = NodeStatus(
            table=table,
            time_gated=source in TIME_GATED,
            envelope_age_days=_age_days(last, now),
            envelopes=count,
            observation_source=None if table in OBSERVATION_EXEMPT_TABLES else source,
        )
        if count == 0 or last is None:
            if not require_all_nodes:
                # Same contract as require_all_configured: against a probe or a
                # single-source fixture, "this node has no envelopes" is the
                # fixture's shape, not a finding.
                continue
            status.reasons.append(REASON_NODE_NEVER_RAN)
        elif status.envelope_age_days is not None and status.envelope_age_days > NODE_SILENCE_DAYS:
            status.reasons.append(REASON_NODE_SILENT)
        statuses.append(status)
    statuses.sort(key=lambda s: (-s.severity, -(s.envelope_age_days or 0.0), s.table))
    return statuses


def exit_code_for(statuses: list[SourceStatus], nodes: list[NodeStatus] | None = None) -> int:
    severities = [s.severity for s in statuses]
    severities += [n.severity for n in nodes or []]
    return max(severities, default=EXIT_OK)


def _fmt_age(value: float | None) -> str:
    return "  never" if value is None else f"{value:7.1f}"


def render_table(statuses: list[SourceStatus], *, snapshot: bool = False) -> str:
    """A fixed-width table: source, ages, class, verdict."""
    head = f"{'SOURCE':<22} {'CLASS':<12} {'DATA AGE':>8} {'LIMIT':>6} {'LAST WRITE':>10} {'ROWS':>8}"
    if snapshot:
        head += f" {'NEW':>7}"
    head += "  VERDICT"
    lines = [head, "-" * len(head)]
    for s in statuses:
        line = (
            f"{s.source:<22} {s.source_class:<12} {_fmt_age(s.data_age_days):>8} "
            f"{s.threshold_days:6.0f} {_fmt_age(s.write_age_days):>10} {s.rows:8d}"
        )
        if snapshot:
            line += f" {'-' if s.new_rows is None else s.new_rows:>7}"
        line += f"  {s.verdict}"
        lines.append(line)
    return "\n".join(lines)


def render_node_table(nodes: list[NodeStatus]) -> str:
    """A second table, keyed by DAG ``table_name``: did the node run at all?"""
    head = f"{'COLLECTION NODE':<26} {'LAST ENVELOPE':>13} {'ENVELOPES':>10} {'OBS SOURCE':<22}  VERDICT"
    lines = [head, "-" * len(head)]
    for n in nodes:
        lines.append(
            f"{n.table:<26} {_fmt_age(n.envelope_age_days):>13} {n.envelopes:10d} "
            f"{(n.observation_source or '(envelope-only)'):<22}  {n.verdict}"
        )
    return "\n".join(lines)


def render_alarm(statuses: list[SourceStatus], nodes: list[NodeStatus] | None = None) -> str:
    """The loud part: which sources are losing data that cannot be bought back."""
    graded: list[SourceStatus | NodeStatus] = [*statuses, *(nodes or [])]
    lost = [s for s in graded if s.severity == EXIT_TIME_GATED]
    degraded = [s for s in graded if s.severity == EXIT_BACKFILLABLE]
    notes = [s for s in statuses if s.severity == EXIT_OK and not s.ok]
    out: list[str] = []
    if lost:
        out.append("")
        out.append("!" * 78)
        out.append("!! TIME-GATED SOURCES ARE NOT LANDING DATA — THIS GAP IS UNRECOVERABLE")
        out.append("!! Every day below is data no backfill and no amount of money can restore.")
        out.append("!" * 78)
        for s in lost:
            out.append(f"!!   {_alarm_line(s)}")
        out.append("!" * 78)
    if degraded:
        out.append("")
        out.append("-- BACKFILLABLE sources behind (recoverable, schedule a backfill):")
        for s in degraded:
            out.append(f"--   {_alarm_line(s)}")
    if notes:
        # Printed, not paged.  See NON_PAGING_REASONS: an alarm whose red state
        # no permitted action can clear is an alarm the operator learns to skip.
        out.append("")
        out.append("~~ DATA QUALITY (reported, does not fail the run):")
        for s in notes:
            out.append(f"~~   {s.source:<22} [{s.verdict}]")
    if not lost and not degraded:
        out.append("")
        if graded:
            out.append("All sources fresh within their cadence.")
        else:
            # Reaching here used to print the reassuring line above against an
            # empty database.  The watchdog built to break green-on-nothing had
            # a green-on-nothing path of its own; check() now refuses first, and
            # this is the second lock.
            out.append("NOTHING WAS GRADED — this is not a pass.")
    return "\n".join(out)


def _alarm_line(status: SourceStatus | NodeStatus) -> str:
    if isinstance(status, NodeStatus):
        ran = "never" if status.envelope_age_days is None else f"{status.envelope_age_days:.1f}d ago"
        return f"{status.table:<22} node last ran {ran:>12}  [{status.verdict}]"
    age = "never" if status.data_age_days is None else f"{status.data_age_days:.1f}d old"
    wrote = "never" if status.write_age_days is None else f"{status.write_age_days:.1f}d ago"
    return f"{status.source:<22} data {age:>12}, last write {wrote:>12}  [{status.verdict}]"


def check(
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    now: float | None = None,
    snapshot_path: str | Path | None = None,
    require_all_configured: bool = True,
    dag_path: str | Path = DAG_MODULE_PATH,
) -> tuple[list[SourceStatus], list[NodeStatus], int]:
    """Inspect *db_path* read-only and return ``(sources, nodes, exit_code)``."""
    now = time.time() if now is None else now
    tables = declared_collection_tables(dag_path)

    conn = connect_readonly(db_path)
    try:
        rows = read_source_rows(conn, now=now)
        daily_counts = read_daily_write_counts(conn)
        envelopes = read_node_envelopes(conn)
    finally:
        conn.close()

    total_rows = sum(r.rows for r in rows.values())
    if total_rows == 0:
        # An empty observations table is never a pass, with or without
        # --allow-absent-sources.  Before this, the flag dropped every zero-row
        # source, left `statuses` empty, and max(..., default=EXIT_OK) returned
        # 0 under the banner "All sources fresh within their cadence."
        raise WatchdogError(
            f"entity_observations is empty in {db_path} — there is nothing to grade, so nothing is fresh"
        )

    snapshot_rows: dict[str, SourceRow] | None = None
    snapshot_daily: dict[str, list[tuple[int, int]]] | None = None
    if snapshot_path is not None:
        _reject_torn_snapshot(snapshot_path)
        snap_conn = connect_readonly(snapshot_path)
        try:
            snapshot_rows = read_source_rows(snap_conn, now=now)
            snapshot_daily = read_daily_write_counts(snap_conn)
        finally:
            snap_conn.close()
        if sum(r.rows for r in snapshot_rows.values()) == 0:
            raise WatchdogError(
                f"snapshot {snapshot_path} holds zero observations while {db_path} holds {total_rows} — "
                "that is a torn copy, not a baseline; take it with "
                "sqlite3 DB \"VACUUM INTO 'snap.db'\", never cp (pipeline.db is WAL)"
            )

    statuses = evaluate(
        rows,
        now=now,
        snapshot_rows=snapshot_rows,
        require_all_configured=require_all_configured,
        expected=expected_sources(tables),
        daily_counts=daily_counts,
        snapshot_daily_counts=snapshot_daily,
    )
    nodes = evaluate_nodes(tables, envelopes, now=now, require_all_nodes=require_all_configured)
    if not statuses:
        raise WatchdogError(f"no source in {db_path} was graded — a report about nothing is not a pass")
    return statuses, nodes, exit_code_for(statuses, nodes)


def _reject_torn_snapshot(snapshot_path: str | Path) -> None:
    """Refuse a snapshot that was copied out from under an active WAL writer.

    ``cp`` of a WAL database omits everything committed since the last
    checkpoint.  Reproduced: a writer committed 2,000 rows, ``shutil.copy`` of
    the main file produced a copy whose *table did not exist*.  The partially
    checkpointed case is the silent version — a too-small baseline that inflates
    ``new_rows`` and suppresses NO_NEW_ROWS.  The bias of a torn copy is toward
    false green, so it fails closed.
    """
    sidecar = Path(f"{snapshot_path}-wal")
    if sidecar.exists() and sidecar.stat().st_size > 0:
        raise WatchdogError(
            f"{sidecar} is non-empty: this snapshot was copied from a live WAL database and is torn. "
            "Take it with sqlite3 DB \"VACUUM INTO 'snap.db'\" instead of cp."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail loudly when a collector stops landing rows.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="exit 0 = fresh, 1 = backfillable gap, 2 = TIME-GATED gap (unrecoverable), 3 = watchdog error",
    )
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help=f"database to inspect (default: {DEFAULT_DB_PATH})")
    parser.add_argument(
        "--since-snapshot",
        default=None,
        metavar="PATH",
        help="a copy of the DB taken earlier; also report sources whose row count did not grow since",
    )
    parser.add_argument(
        "--allow-absent-sources",
        action="store_true",
        help="do not fail on expected sources/nodes that have no rows at all (use against a partial DB). It cannot turn an empty database green: zero observations is a watchdog error either way.",
    )
    parser.add_argument(
        "--now",
        type=float,
        default=None,
        help="override wall clock (epoch seconds) — for tests and replaying a past outage",
    )
    args = parser.parse_args(argv)

    try:
        statuses, nodes, code = check(
            args.db,
            now=args.now,
            snapshot_path=args.since_snapshot,
            require_all_configured=not args.allow_absent_sources,
        )
    except WatchdogError as exc:
        print(f"FRESHNESS WATCHDOG COULD NOT RUN: {exc}", file=sys.stderr)
        return EXIT_WATCHDOG_ERROR

    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(args.now if args.now is not None else time.time()))
    print(f"Collection freshness — {args.db} @ {stamp}Z (ages in days)")
    if args.since_snapshot:
        print(f"Row-growth baseline: {args.since_snapshot}")
    print()
    print(render_table(statuses, snapshot=args.since_snapshot is not None))
    print()
    print(render_node_table(nodes))
    print(render_alarm(statuses, nodes))
    print()
    print(f"exit={code}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
