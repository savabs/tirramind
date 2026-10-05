"""The freshness watchdog: does it see the outage that nobody saw for 26 days?

Collection stopped on 2026-08-27 and ran green the whole time.  These tests
reproduce each shape that outage can take and assert the watchdog separates
them, because the shapes look identical from the DAG's point of view:

* a source that stopped publishing          → STALE, collector fine
* a collector that stopped writing          → SILENT, source fine
* a quiet source with a live collector      → OK (must NOT page anyone)
* a run that reported success and wrote 0   → NO_NEW_ROWS (snapshot mode)
* a run that wrote one row and deleted 250  → ROWS_LOST (never suppressed)
* a run that wrote 2 rows where it writes 200 → LOW_VOLUME (the F-16 shape)
* a collector that has never landed a row   → MISSING, from the DAG's own list
* a node that runs daily and lands nothing  → NODE_SILENT / envelope-only
* a mis-parsed future date hiding staleness → FUTURE_DATED + STALE
* an empty database, or an empty report     → never a pass

Every test writes to a pytest ``tmp_path`` database.  The live DB is never
opened here, and the watchdog itself only ever opens ``mode=ro``.

Clock discipline: observations are written at fixture times anchored by
``T`` (the store rejects ``observed_at`` before 1990-01-01), and the watchdog
is handed the *same* fixture clock via ``now=``.  ``ingested_at`` is stamped by
the store with real wall-clock time, so where a test cares about the
collector-silence clock it sets ``ingested_at`` explicitly on its own throwaway
database — shifting only the writes would let a silence test pass for the wrong
reason.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from agent.pipeline.store import PipelineStore
from scripts.check_freshness import (
    CADENCE_DAYS,
    COLLECTOR_SILENCE_DAYS,
    EXIT_BACKFILLABLE,
    EXIT_OK,
    EXIT_TIME_GATED,
    EXIT_WATCHDOG_ERROR,
    OBSERVATION_EXEMPT_TABLES,
    REASON_FUTURE,
    REASON_LOW_VOLUME,
    REASON_MISSING,
    REASON_NO_NEW_ROWS,
    REASON_NODE_SILENT,
    REASON_ROWS_LOST,
    REASON_SILENT,
    REASON_STALE,
    REASON_UNKNOWN,
    TIME_GATED,
    WatchdogError,
    check,
    connect_readonly,
    declared_collection_tables,
    expected_sources,
    main,
    render_alarm,
)
from tests.fixture_time import T

DAY = 86_400.0

#: The fixture "today": 2025-01-01, i.e. one year after ``TEST_BASE_TIME``.
#: Far enough past 1990 that the store accepts every derived timestamp, and far
#: enough before real wall clock that a "future-dated" fixture row is still a
#: valid write.
NOW = T(365 * DAY)

ENTITY = "e_freshness_probe"


@pytest.fixture()
def db(tmp_path):
    """A throwaway database path — never the live pipeline DB."""
    return tmp_path / "probe_freshness.db"


def _store(db) -> PipelineStore:
    store = PipelineStore(db_path=str(db))
    store.register_entity(
        entity_type="instrument",
        canonical_name="Freshness Probe Entity",
        entity_id=ENTITY,
    )
    return store


def _seed(
    store: PipelineStore,
    source: str,
    *,
    data_age_days: float,
    write_age_days: float | None = None,
    count: int = 1,
    value_seed: int = 0,
) -> None:
    """Write *count* observations for *source*, aged as requested.

    ``data_age_days`` sets ``observed_at`` relative to :data:`NOW`; a negative
    age puts the observation in the future *as far as the watchdog's clock is
    concerned*, which is how the live DB's 2030-dated gov_contracts rows look.

    ``write_age_days`` sets ``ingested_at``.  The store stamps it with real
    wall-clock time, so it is rewritten here through the throwaway DB's own
    connection; defaults to ``data_age_days`` (collector wrote when it observed).
    """
    observed_at = NOW - data_age_days * DAY
    for i in range(count):
        store.store_entity_observation(
            entity_id=ENTITY,
            source_tool=source,
            observed_at=observed_at,
            observation_type="probe",
            value={"n": value_seed + i},
        )
    ingested_at = NOW - (data_age_days if write_age_days is None else write_age_days) * DAY
    conn = store._get_conn()
    conn.execute(
        "UPDATE entity_observations SET ingested_at = ? WHERE source_tool = ?",
        (ingested_at, source),
    )
    conn.commit()


def _seed_history(
    store: PipelineStore,
    source: str,
    per_day: list[tuple[float, int]],
    *,
    value_seed: int = 0,
) -> int:
    """Write *count* observations on each ``(days_ago, count)`` day.

    The volume floor is a *per-day* statistic, so a fixture for it has to lay
    rows down across distinct ``ingested_at`` days; ``_seed``'s single blanket
    UPDATE would collapse them all onto one day and the baseline would never
    exist.  Returns the next unused value seed.
    """
    conn = store._get_conn()
    value = value_seed
    for days_ago, count in per_day:
        high_water = conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM entity_observations").fetchone()[0]
        for _ in range(count):
            store.store_entity_observation(
                entity_id=ENTITY,
                source_tool=source,
                observed_at=NOW - days_ago * DAY,
                observation_type="probe",
                value={"n": value},
            )
            value += 1
        conn.execute(
            "UPDATE entity_observations SET ingested_at = ? WHERE rowid > ? AND source_tool = ?",
            (NOW - days_ago * DAY, high_water, source),
        )
    conn.commit()
    return value


def _seed_envelope(store: PipelineStore, table: str, *, age_days: float, count: int = 1) -> None:
    """Record *count* ``pipeline_data`` envelopes for a DAG node, aged as asked.

    This is what ``executor.py`` writes under ``node.table_name or node.id`` for
    every node that ran — the only evidence that a collector executed at all.
    """
    conn = store._get_conn()
    for i in range(count):
        store.store_data(source=table, params={"i": i}, data={"ok": True})
    conn.execute("UPDATE pipeline_data SET fetched_at = ? WHERE source = ?", (NOW - age_days * DAY, table))
    conn.commit()


def _fake_dag(tmp_path, tables: list[str]):
    """A stand-in collection DAG module, so a test is not hostage to the real one."""
    path = tmp_path / "fake_dag.py"
    body = "\n".join(f'    add_node(id="n{i}", table_name="{t}")' for i, t in enumerate(tables))
    path.write_text(f"def build():\n{body or '    pass'}\n", encoding="utf-8")
    return path


def _by_source(statuses):
    return {s.source: s for s in statuses}


def _by_table(nodes):
    return {n.table: n for n in nodes}


# ── the thresholds table itself ───────────────────────────────


def test_every_time_gated_source_has_an_explicit_threshold():
    """A time-gated source falling back to the default is a silent downgrade."""
    missing = sorted(TIME_GATED - set(CADENCE_DAYS))
    assert missing == [], f"time-gated sources with no cadence entry: {missing}"


def test_daily_thresholds_tolerate_a_weekend():
    """daily_collection runs ``0 18 * * 1-5``; a 1-day threshold would page every Monday."""
    assert all(days >= 3.0 for days in CADENCE_DAYS.values())


# ── read-only safety ──────────────────────────────────────────


def test_watchdog_connection_refuses_writes(db):
    store = _store(db)
    _seed(store, "ais_vessel", data_age_days=0.0)
    store.close()

    conn = connect_readonly(db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM entity_observations").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM entity_observations")
    finally:
        conn.close()


def test_missing_database_is_a_watchdog_error_not_a_pass(tmp_path):
    with pytest.raises(WatchdogError):
        check(tmp_path / "nope.db", now=NOW)
    assert main(["--db", str(tmp_path / "nope.db")]) == EXIT_WATCHDOG_ERROR


# ── staleness, and the two clocks ─────────────────────────────


def test_all_fresh_sources_exit_zero(db):
    store = _store(db)
    for source in ("ais_vessel", "gdelt", "cftc"):
        _seed(store, source, data_age_days=0.5)
    store.close()

    statuses, _nodes, code = check(db, now=NOW, require_all_configured=False)
    assert code == EXIT_OK
    present = [s for s in statuses if s.rows]
    assert {s.source for s in present} == {"ais_vessel", "gdelt", "cftc"}
    assert all(s.verdict == "OK" for s in present)


def test_time_gated_staleness_is_louder_than_backfillable(db):
    """Same 10-day gap, two classes: only the unrecoverable one escalates to 2."""
    store = _store(db)
    _seed(store, "gdelt", data_age_days=10.0)  # BACKFILLABLE, threshold 3
    store.close()
    statuses, _nodes, code = check(db, now=NOW, require_all_configured=False)
    assert code == EXIT_BACKFILLABLE
    assert _by_source(statuses)["gdelt"].reasons == [REASON_STALE, REASON_SILENT]

    store = _store(db)
    _seed(store, "ais_vessel", data_age_days=10.0)  # TIME_GATED, threshold 3
    store.close()
    statuses, _nodes, code = check(db, now=NOW, require_all_configured=False)
    assert code == EXIT_TIME_GATED
    ais = _by_source(statuses)["ais_vessel"]
    assert ais.source_class == "TIME_GATED"
    assert REASON_STALE in ais.reasons


def test_a_source_just_inside_its_own_cadence_is_not_flagged(db):
    """cftc publishes weekly (threshold 10) — 8 days quiet is normal, not an outage."""
    store = _store(db)
    _seed(store, "cftc", data_age_days=8.0, write_age_days=0.5)
    store.close()

    statuses, _nodes, code = check(db, now=NOW, require_all_configured=False)
    assert code == EXIT_OK
    assert _by_source(statuses)["cftc"].verdict == "OK"


def test_silent_collector_is_caught_even_when_data_age_looks_fine(db):
    """The 26-day outage, exactly: data still inside cadence, but nothing is writing.

    ``sanctions_monitor`` has a 21-day cadence, so 15-day-old data passes the
    staleness check.  The collector has not written for 26 days.  A one-clock
    watchdog says OK here; this is why there are two.
    """
    store = _store(db)
    _seed(store, "sanctions_monitor", data_age_days=15.0, write_age_days=26.0)
    store.close()

    statuses, _nodes, code = check(db, now=NOW, require_all_configured=False)
    status = _by_source(statuses)["sanctions_monitor"]
    assert status.reasons == [REASON_SILENT]
    assert status.write_age_days == pytest.approx(26.0)
    assert code == EXIT_BACKFILLABLE


def test_quiet_source_with_a_live_collector_stays_green(db):
    """The complement: same old data, collector wrote today. Must not page anyone."""
    store = _store(db)
    _seed(store, "sanctions_monitor", data_age_days=15.0, write_age_days=0.02)
    store.close()

    statuses, _nodes, code = check(db, now=NOW, require_all_configured=False)
    assert _by_source(statuses)["sanctions_monitor"].verdict == "OK"
    assert code == EXIT_OK


def test_silence_threshold_boundary(db):
    """One missed weekday run plus a weekend is tolerated; more is not."""
    store = _store(db)
    _seed(store, "cftc", data_age_days=1.0, write_age_days=COLLECTOR_SILENCE_DAYS - 0.1)
    store.close()
    assert check(db, now=NOW, require_all_configured=False)[2] == EXIT_OK

    store = _store(db)
    _seed(store, "cftc", data_age_days=1.0, write_age_days=COLLECTOR_SILENCE_DAYS + 0.1, value_seed=99)
    store.close()
    statuses, _nodes, code = check(db, now=NOW, require_all_configured=False)
    assert REASON_SILENT in _by_source(statuses)["cftc"].reasons
    assert code == EXIT_BACKFILLABLE


# ── the anomalies that hide staleness ─────────────────────────


def test_future_dated_rows_cannot_mask_a_stale_source(db):
    """A mis-parsed date must not score a dead source as fresh.

    The live DB holds gov_contracts rows dated 2030; ``now - MAX(observed_at)``
    reports that source as 1225 days *fresh*.  The watchdog ignores timestamps
    beyond now + skew when ageing, and reports the anomaly separately.
    """
    store = _store(db)
    _seed(store, "gov_contracts", data_age_days=40.0, write_age_days=0.5)
    _seed(store, "gov_contracts", data_age_days=-400.0, write_age_days=0.5, value_seed=7)
    store.close()

    statuses, _nodes, code = check(db, now=NOW, require_all_configured=False)
    status = _by_source(statuses)["gov_contracts"]
    assert status.data_age_days == pytest.approx(40.0)
    assert REASON_STALE in status.reasons
    assert REASON_FUTURE in status.reasons
    assert code == EXIT_BACKFILLABLE


def test_configured_source_with_no_rows_is_missing_not_absent(db):
    """A collector that never landed a row must appear in the table, loudly."""
    store = _store(db)
    _seed(store, "gdelt", data_age_days=0.5)
    store.close()

    statuses, _nodes, code = check(db, now=NOW)  # default: every configured source must exist
    whale = _by_source(statuses)["whale_alert"]  # time-gated, configured, never wrote
    assert whale.reasons == [REASON_MISSING]
    assert whale.rows == 0
    assert whale.data_age_days is None
    assert code == EXIT_TIME_GATED


def test_unrecognised_source_is_reported_rather_than_policed_silently(db):
    store = _store(db)
    _seed(store, "brand_new_tool", data_age_days=0.1, write_age_days=0.1)
    store.close()

    statuses, _nodes, code = check(db, now=NOW, require_all_configured=False)
    status = _by_source(statuses)["brand_new_tool"]
    assert status.reasons == [REASON_UNKNOWN]
    assert code == EXIT_BACKFILLABLE  # never escalates to the time-gated alarm


# ── green-but-zero-rows (snapshot mode) ───────────────────────


def test_snapshot_mode_catches_the_run_that_wrote_nothing(db, tmp_path):
    """A run that reported success and appended zero rows is invisible to both clocks.

    Both sources here are comfortably inside their cadence *and* inside the
    collector-silence window after the run, so neither STALE nor SILENT fires.
    The only thing separating them is that one wrote during the run and one did
    nothing at all — which is precisely the green-checkmark-zero-rows bug.
    """
    store = _store(db)
    _seed(store, "ais_vessel", data_age_days=1.0, write_age_days=1.0, count=2)
    _seed(store, "gdelt", data_age_days=1.0, write_age_days=1.0, count=2)
    store.close()

    snapshot = tmp_path / "snapshot.db"
    shutil.copy(db, snapshot)

    # The run: gdelt lands fresh rows, ais_vessel reports success and writes none.
    store = _store(db)
    _seed(store, "gdelt", data_age_days=0.1, write_age_days=0.1, count=3, value_seed=500)
    store.close()

    statuses, _nodes, code = check(db, now=NOW, snapshot_path=snapshot, require_all_configured=False)
    by_source = _by_source(statuses)

    assert by_source["gdelt"].new_rows == 3
    assert by_source["gdelt"].verdict == "OK"

    ais = by_source["ais_vessel"]
    assert ais.new_rows == 0
    assert ais.reasons == [REASON_NO_NEW_ROWS]  # NOT stale, NOT silent — only this
    assert code == EXIT_TIME_GATED  # ais_vessel is time-gated: the gap is permanent


def test_snapshot_mode_does_not_condemn_a_collector_that_refetched_unchanged_rows(db, tmp_path):
    """Zero row growth is normal when the source had nothing new to say.

    ``store_entity_observation`` treats a byte-identical observation as the same
    observation and refreshes ``ingested_at`` instead of appending, so a quiet
    monthly source that re-fetched its last release grows by zero rows while
    still proving the collector ran.  Failing on row count alone would page
    someone every day for comtrade.
    """
    store = _store(db)
    _seed(store, "comtrade", data_age_days=1.0, write_age_days=1.0, count=2)  # 45-day cadence
    store.close()

    snapshot = tmp_path / "snapshot.db"
    shutil.copy(db, snapshot)

    # The run: same two observations fetched again — dedup refreshes, count is flat.
    store = _store(db)
    _seed(store, "comtrade", data_age_days=1.0, write_age_days=0.02, count=2)
    store.close()

    statuses, _nodes, code = check(db, now=NOW, snapshot_path=snapshot, require_all_configured=False)
    status = _by_source(statuses)["comtrade"]
    assert status.new_rows == 0  # no growth ...
    assert status.verdict == "OK"  # ... but the collector demonstrably wrote
    assert code == EXIT_OK


# ── the CLI surface a cron job actually uses ──────────────────


def test_main_prints_a_table_and_returns_the_exit_code(db, capsys):
    store = _store(db)
    _seed(store, "ais_vessel", data_age_days=30.0, write_age_days=30.0)
    _seed(store, "cftc", data_age_days=1.0, write_age_days=0.5)
    store.close()

    code = main(["--db", str(db), "--now", str(NOW), "--allow-absent-sources"])
    out = capsys.readouterr().out

    assert code == EXIT_TIME_GATED
    assert "ais_vessel" in out and "TIME_GATED" in out
    assert "UNRECOVERABLE" in out  # the loud banner fired
    assert "exit=2" in out


# ── the watchdog's own green-on-nothing path ──────────────────


def test_an_empty_database_is_never_a_pass(db, capsys):
    """The alarm built to break green-on-nothing had a green-on-nothing path.

    With ``--allow-absent-sources`` every zero-row source was dropped, the
    status list came out empty, ``max(..., default=EXIT_OK)`` returned 0, and
    the reassuring "All sources fresh within their cadence." was printed over a
    database holding not one observation.
    """
    _store(db).close()  # schema created, zero rows

    with pytest.raises(WatchdogError):
        check(db, now=NOW, require_all_configured=False)

    code = main(["--db", str(db), "--now", str(NOW), "--allow-absent-sources"])
    out = capsys.readouterr().out
    assert code == EXIT_WATCHDOG_ERROR
    assert "All sources fresh" not in out


def test_the_reassuring_line_requires_something_to_have_been_graded():
    """Second lock, at the rendering layer: no statuses, no green sentence."""
    assert "All sources fresh" not in render_alarm([])
    assert "NOTHING WAS GRADED" in render_alarm([])


# ── row loss (snapshot mode) ──────────────────────────────────


def test_a_run_that_deleted_rows_is_not_ok_just_because_it_wrote_one(db, tmp_path):
    """``new_rows <= 0 and not wrote_since`` let a net loss render NEW=-249, verdict OK.

    One write suppressed the whole check, so a destructive migration that
    dropped half a table passed with exit 0 while printing the negative delta on
    the same line.
    """
    store = _store(db)
    _seed_history(store, "defi_flows", [(3.0, 200), (2.0, 200), (1.0, 100)])
    store.close()

    snapshot = tmp_path / "snapshot.db"
    shutil.copy(db, snapshot)

    conn = sqlite3.connect(db)
    conn.execute("DELETE FROM entity_observations WHERE source_tool = 'defi_flows' AND rowid % 2 = 0")
    conn.commit()
    conn.close()

    store = _store(db)
    _seed_history(store, "defi_flows", [(0.1, 1)], value_seed=10_000)
    store.close()

    statuses, _nodes, code = check(db, now=NOW, snapshot_path=snapshot, require_all_configured=False)
    status = _by_source(statuses)["defi_flows"]
    assert status.new_rows < 0
    assert REASON_ROWS_LOST in status.reasons
    assert status.verdict != "OK"
    assert code != EXIT_OK


def test_no_new_rows_still_fires_only_when_nothing_was_written(db, tmp_path):
    """Splitting ROWS_LOST out must not turn NO_NEW_ROWS into a row-count alarm.

    A collector that re-fetched unchanged records grows by zero rows and is
    healthy; that distinction is the whole reason NO_NEW_ROWS also consults the
    write clock, and it has to survive the split.
    """
    store = _store(db)
    _seed(store, "comtrade", data_age_days=1.0, write_age_days=1.0, count=2)
    store.close()
    snapshot = tmp_path / "snapshot.db"
    shutil.copy(db, snapshot)

    store = _store(db)
    _seed(store, "comtrade", data_age_days=1.0, write_age_days=0.02, count=2)  # dedup: refresh, no growth
    store.close()

    statuses, _nodes, code = check(db, now=NOW, snapshot_path=snapshot, require_all_configured=False)
    status = _by_source(statuses)["comtrade"]
    assert status.new_rows == 0
    assert REASON_NO_NEW_ROWS not in status.reasons
    assert REASON_ROWS_LOST not in status.reasons
    assert code == EXIT_OK


# ── the volume floor (the F-16 shape) ─────────────────────────


def test_a_collector_that_collapses_to_a_trickle_is_caught_in_snapshot_mode(db, tmp_path):
    """F-16 verbatim: HTTP 200, rows written, 72% of the window silently dropped.

    Every other check reads green here — the data is fresh, the collector wrote,
    the row count grew.  Only the comparison against this source's own history
    sees that the run yielded 2 rows where it normally yields 200.
    """
    store = _store(db)
    _seed_history(store, "gdelt", [(4.0, 200), (3.0, 200), (2.0, 200)])
    store.close()

    snapshot = tmp_path / "snapshot.db"
    shutil.copy(db, snapshot)

    store = _store(db)
    _seed_history(store, "gdelt", [(0.1, 2)], value_seed=50_000)
    store.close()

    statuses, _nodes, code = check(db, now=NOW, snapshot_path=snapshot, require_all_configured=False)
    status = _by_source(statuses)["gdelt"]
    assert status.new_rows == 2  # it DID write — that is the point
    assert REASON_LOW_VOLUME in status.reasons
    assert code == EXIT_BACKFILLABLE


def test_the_volume_floor_also_works_without_a_snapshot(db):
    """The shipped systemd unit passes no snapshot, so the floor cannot need one.

    Without a baseline file the most recent *complete* day is graded against the
    days before it.  The current, still-running day is excluded: grading a
    partial day against whole ones would manufacture an alarm every afternoon.
    """
    store = _store(db)
    _seed_history(store, "gdelt", [(4.0, 200), (3.0, 200), (2.0, 200), (1.0, 3)])
    store.close()

    statuses, _nodes, code = check(db, now=NOW, require_all_configured=False)
    assert REASON_LOW_VOLUME in _by_source(statuses)["gdelt"].reasons
    assert code == EXIT_BACKFILLABLE


def test_a_healthy_run_is_not_flagged_by_the_volume_floor(db):
    """The complement, and the reason the thresholds are conservative."""
    store = _store(db)
    _seed_history(store, "gdelt", [(4.0, 200), (3.0, 200), (2.0, 200), (1.0, 190)])
    store.close()

    statuses, _nodes, code = check(db, now=NOW, require_all_configured=False)
    assert _by_source(statuses)["gdelt"].verdict == "OK"
    assert code == EXIT_OK


def test_tiny_sources_are_not_graded_on_volume(db):
    """comtrade holds 6 rows lifetime; a "median" of 2 would page on every quiet day."""
    store = _store(db)
    _seed_history(store, "comtrade", [(4.0, 3), (3.0, 3), (2.0, 3), (1.0, 1)])
    store.close()

    statuses, _nodes, code = check(db, now=NOW, require_all_configured=False)
    assert _by_source(statuses)["comtrade"].verdict == "OK"
    assert code == EXIT_OK


# ── where the expected-source list comes from ─────────────────


def test_expected_sources_are_read_from_the_dag_not_from_the_data(tmp_path):
    """A collector invisible in the data must still be expected.

    The original universe was ``set(rows) | set(CADENCE_DAYS)`` with
    CADENCE_DAYS itself measured off that same database — so a collector that
    had never written a row could not appear, and MISSING was a provable no-op.
    """
    dag = _fake_dag(tmp_path, ["gdelt", "ghost_collector", "weather_alerts"])
    tables = declared_collection_tables(dag)
    assert tables == {"gdelt", "ghost_collector", "weather_alerts"}

    expected = expected_sources(tables)
    assert "ghost_collector" in expected  # declared by the DAG, unknown to the DB
    assert "ghost_collector" not in CADENCE_DAYS  # ... and unknown to the cadence table


def test_a_dag_collector_that_never_landed_a_row_is_missing(db, tmp_path):
    store = _store(db)
    _seed(store, "gdelt", data_age_days=0.5)
    store.close()

    dag = _fake_dag(tmp_path, ["gdelt", "ghost_collector"])
    statuses, _nodes, code = check(db, now=NOW, dag_path=dag)
    ghost = _by_source(statuses)["ghost_collector"]  # a name the database has never seen
    assert ghost.reasons == [REASON_MISSING]
    assert ghost.rows == 0
    assert code != EXIT_OK


def test_envelope_only_collectors_are_not_reported_missing(db, tmp_path):
    """Several collection nodes never call ``store_entity_observation`` at all.

    Grading those against ``entity_observations`` would produce permanently red
    lines no code change in this file can clear, which is how an operator learns
    to ignore the report.  They are graded on the node clock instead.
    """
    store = _store(db)
    _seed(store, "gdelt", data_age_days=0.5)
    _seed_envelope(store, "weather_alerts", age_days=0.1)
    _seed_envelope(store, "gdelt", age_days=0.1)
    store.close()

    dag = _fake_dag(tmp_path, ["gdelt", "weather_alerts"])
    statuses, nodes, _code = check(db, now=NOW, dag_path=dag)
    assert "weather_alerts" not in _by_source(statuses)
    weather = _by_table(nodes)["weather_alerts"]
    assert weather.verdict == "OK"
    assert weather.observation_source is None  # graded on envelopes, by design


def test_the_exempt_list_cannot_be_used_to_silence_a_real_collector():
    """Every exempt table's tool module must genuinely contain no observation write.

    Without this, ``OBSERVATION_EXEMPT_TABLES`` is a mute button: any collector
    that stopped landing rows could be "fixed" by adding its name to the set.
    """
    tools = Path(__file__).resolve().parent.parent / "agent" / "tools"
    for table in sorted(OBSERVATION_EXEMPT_TABLES):
        module = tools / f"{table}.py"
        assert module.exists(), f"{table} is exempt but has no tool module to justify it"
        assert "store_entity_observation" not in module.read_text(encoding="utf-8"), (
            f"{table} DOES write entity observations — it must be graded, not exempt"
        )


def test_an_unparseable_dag_is_an_error_not_an_empty_node_list(tmp_path):
    """A watchdog that reports green because it lost the thing it audits is the bug."""
    with pytest.raises(WatchdogError):
        declared_collection_tables(tmp_path / "absent.py")

    empty = tmp_path / "no_nodes.py"
    empty.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(WatchdogError):
        declared_collection_tables(empty)


# ── the node clock: "it ran" vs "it produced rows" ────────────


def test_a_node_that_stopped_running_is_caught_even_with_fresh_rows(db, tmp_path):
    """``pipeline_data`` answers a question ``entity_observations`` cannot.

    A node can execute daily forever and never land an observation, and a node
    can have rows on file long after it stopped executing.  Two tables, two
    signals.
    """
    store = _store(db)
    _seed(store, "gdelt", data_age_days=0.5)
    _seed_envelope(store, "gdelt", age_days=30.0)
    store.close()

    dag = _fake_dag(tmp_path, ["gdelt"])
    statuses, nodes, code = check(db, now=NOW, dag_path=dag, require_all_configured=False)
    assert _by_source(statuses)["gdelt"].verdict == "OK"  # rows look perfectly fine
    assert _by_table(nodes)["gdelt"].reasons == [REASON_NODE_SILENT]
    assert code == EXIT_BACKFILLABLE


def test_retired_node_ids_are_not_graded(db, tmp_path):
    """``pipeline_data`` still holds envelopes from nodes deleted months ago.

    Grading every source it has ever seen would pin the alarm red on
    ``train_gnn`` and ``fetch_gdelt`` forever.  Only what the DAG declares now.
    """
    store = _store(db)
    _seed(store, "gdelt", data_age_days=0.5)
    _seed_envelope(store, "gdelt", age_days=0.1)
    _seed_envelope(store, "fetch_gdelt_retired", age_days=400.0)
    store.close()

    dag = _fake_dag(tmp_path, ["gdelt"])
    _statuses, nodes, code = check(db, now=NOW, dag_path=dag, require_all_configured=False)
    assert "fetch_gdelt_retired" not in _by_table(nodes)
    assert code == EXIT_OK


# ── an alarm nobody can clear is an alarm nobody reads ────────


def test_future_dated_rows_alone_do_not_pin_the_watchdog_red(db, tmp_path):
    """912 gov_contracts rows dated 2030 held the unit in ``failed`` permanently.

    They were all ingested before 2026-08-27, the store's observed_at guard now
    refuses such writes, and only a DELETE on the live DB could clear them —
    which this project forbids.  FUTURE_DATED cannot mask staleness (the ageing
    query already ignores future timestamps), so it is a data-quality report:
    printed, never paging.
    """
    store = _store(db)
    _seed(store, "gov_contracts", data_age_days=1.0, write_age_days=0.5)
    _seed(store, "gov_contracts", data_age_days=-400.0, write_age_days=0.5, value_seed=7)
    store.close()

    dag = _fake_dag(tmp_path, ["gov_contracts"])
    statuses, _nodes, code = check(db, now=NOW, dag_path=dag, require_all_configured=False)
    status = _by_source(statuses)["gov_contracts"]
    assert status.reasons == [REASON_FUTURE]
    assert status.verdict == "FUTURE_DATED"  # still on the table, not swept away
    assert code == EXIT_OK
    assert "DATA QUALITY" in render_alarm(statuses)


# ── the snapshot itself has to be trustworthy ─────────────────


def test_a_cp_torn_wal_snapshot_is_refused(db, tmp_path):
    """``cp`` of a WAL database can omit everything since the last checkpoint.

    Reproduced directly: a writer committed 2,000 rows and ``shutil.copy`` of
    the main file produced a copy whose table did not exist.  The partially
    checkpointed case is the silent version — a too-small baseline inflates
    ``new_rows`` and suppresses NO_NEW_ROWS, biasing toward false green.
    """
    store = _store(db)
    _seed(store, "gdelt", data_age_days=0.5, count=2)
    store.close()

    snapshot = tmp_path / "snapshot.db"
    shutil.copy(db, snapshot)
    Path(f"{snapshot}-wal").write_bytes(b"\x00" * 32)  # the sidecar a cp leaves behind

    with pytest.raises(WatchdogError, match="torn"):
        check(db, now=NOW, snapshot_path=snapshot, require_all_configured=False)


def test_an_empty_snapshot_against_a_populated_db_is_refused(db, tmp_path):
    """The silent shape of the same defect: a baseline of zero makes everything look new."""
    empty_snapshot = tmp_path / "empty_snapshot.db"
    _store(empty_snapshot).close()

    store = _store(db)
    _seed(store, "gdelt", data_age_days=0.5, count=2)
    store.close()

    with pytest.raises(WatchdogError, match="torn copy"):
        check(db, now=NOW, snapshot_path=empty_snapshot, require_all_configured=False)


def test_the_documented_snapshot_recipe_is_wal_correct():
    """The usage block is the only instruction anyone follows — it must not say ``cp``."""
    doc = Path(__file__).resolve().parent.parent / "scripts" / "check_freshness.py"
    usage = doc.read_text(encoding="utf-8").split("Exit codes")[0]
    assert "VACUUM INTO" in usage
    assert "cp .tirra_pipeline/pipeline.db" not in usage
