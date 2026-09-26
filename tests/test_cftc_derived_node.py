"""Regression tests for the ``derive_cftc_features`` DAG node (audit P6.4).

The bug these lock down is not a collector that failed — it is a collector
that was never scheduled.  ``cftc_derived`` observations were produced only by
a hand-run script that no DAG imported, so the table froze at ``observed_at``
2026-04-14 while its parent ``cftc`` source kept collecting for another 162
days.  Nothing went red, because nothing ran.

So the tests assert three separate things, each of which alone was enough to
hide the failure:

1. The node exists in ``daily_collection`` and depends on ``fetch_cftc`` —
   against the unfixed tree this fails at ``"derive_cftc_features" in nodes``.
2. Running it writes real rows, and the derived ``MAX(observed_at)`` catches up
   to the raw one — a fix that runs but writes nothing fails here.
3. Re-running writes zero *new* rows and duplicates nothing, and reports
   ``status="skipped"`` so ``classify_payload_status`` cannot paint an empty
   run green.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from agent.pipeline.dags.daily_collection import (
    build_daily_collection_dag,
    run_cftc_feature_derivation,
)
from agent.pipeline.operators import DOMAIN_TABLES_PARAM_KEY, classify_payload_status
from agent.pipeline.store import PipelineStore
from agent.quant.cftc_features import (
    COVERAGE_WINDOW_WEEKS,
    coverage_stats,
    derive_and_store,
    load_raw_positioning,
)
from tests.fixture_time import T

WEEK = 7 * 86400.0
ENTITIES = ("cftc_contract:088691", "cftc_contract:067651")
WEEKS_PER_ENTITY = 8


def _seed_raw(store: PipelineStore) -> float:
    """Seed weekly ``futures_positioning`` rows for two entities.

    Returns the maximum ``observed_at`` seeded.
    """
    max_ts = 0.0
    for e_idx, entity_id in enumerate(ENTITIES):
        for week in range(WEEKS_PER_ENTITY):
            observed_at = T(week * WEEK + e_idx * 3600.0)
            store.store_entity_observation(
                entity_id=entity_id,
                source_tool="cftc",
                observed_at=observed_at,
                observation_type="futures_positioning",
                depth_level=2,
                value={
                    "open_interest": 500_000 + 1_000 * week + 10_000 * e_idx,
                    "mm_net": 50_000 - 500 * week,
                    "mm_net_pct_oi": 0.10 + 0.01 * week - 0.002 * e_idx,
                },
            )
            max_ts = max(max_ts, observed_at)
    return max_ts


def _derived_rows(db_path: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT entity_id, observed_at, value_json FROM entity_observations "
            "WHERE source_tool='cftc_derived' ORDER BY entity_id, observed_at"
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


@pytest.fixture
def seeded_db(tmp_path: Path) -> tuple[Path, float]:
    db_path = tmp_path / "probe_cftc_derived.db"
    store = PipelineStore(str(db_path))
    try:
        raw_max = _seed_raw(store)
    finally:
        store.close()
    return db_path, raw_max


class TestDeriveCftcFeaturesNodeIsScheduled:
    """The node has to exist in the DAG — that is the whole bug."""

    def test_node_present_and_depends_on_fetch_cftc(self) -> None:
        dag = build_daily_collection_dag(db_path=".tirra_pipeline/pipeline.db")
        assert "derive_cftc_features" in dag.nodes, (
            "derive_cftc_features is not a node in daily_collection — the derivation "
            "can only run by hand again, which is exactly how cftc_derived froze for 162 days."
        )
        node = dag.nodes["derive_cftc_features"]
        assert node.depends_on == ["fetch_cftc"]

    def test_node_declares_a_real_domain_table(self) -> None:
        """Without this the zero-rows guard counts the result envelope, not rows."""
        node = build_daily_collection_dag().nodes["derive_cftc_features"]
        assert node.params[DOMAIN_TABLES_PARAM_KEY] == ["entity_observations"]

    def test_dag_still_validates(self) -> None:
        build_daily_collection_dag().validate()


class TestDeriveCftcFeaturesWritesRows:
    def test_writes_one_derived_row_per_raw_row(self, seeded_db: tuple[Path, float]) -> None:
        db_path, raw_max = seeded_db
        result = run_cftc_feature_derivation({"db_path": str(db_path)}, {})

        expected = len(ENTITIES) * WEEKS_PER_ENTITY
        assert result["rows_written"] == expected
        assert result["entities"] == len(ENTITIES)
        assert classify_payload_status(result) == "completed"

        rows = _derived_rows(db_path)
        assert len(rows) == expected
        assert max(r["observed_at"] for r in rows) == pytest.approx(raw_max)

    def test_derived_values_are_populated(self, seeded_db: tuple[Path, float]) -> None:
        db_path, _ = seeded_db
        run_cftc_feature_derivation({"db_path": str(db_path)}, {})

        values = [json.loads(r["value_json"]) for r in _derived_rows(db_path)]
        assert all("cftc_mm_pct_52w_rank" in v for v in values)
        # mm_net_pct_oi rises monotonically per entity, so the last week is the
        # most crowded reading in its own trailing window.
        assert values[WEEKS_PER_ENTITY - 1]["cftc_mm_pct_52w_rank"] == pytest.approx(1.0)
        assert values[WEEKS_PER_ENTITY - 1]["cftc_mm_direction_change"] == 1.0


class TestDeriveCftcFeaturesIsIdempotent:
    def test_second_run_writes_nothing_and_duplicates_nothing(self, seeded_db: tuple[Path, float]) -> None:
        db_path, _ = seeded_db
        first = run_cftc_feature_derivation({"db_path": str(db_path)}, {})
        rows_after_first = _derived_rows(db_path)

        second = run_cftc_feature_derivation({"db_path": str(db_path)}, {})
        rows_after_second = _derived_rows(db_path)

        assert first["rows_written"] > 0
        assert second["rows_written"] == 0
        assert second["deduplicated"] == first["rows_derived"]
        assert len(rows_after_second) == len(rows_after_first)

        keys = [(r["entity_id"], r["observed_at"]) for r in rows_after_second]
        assert len(set(keys)) == len(keys), "re-running the derivation duplicated rows"

    def test_empty_run_reports_skipped_not_completed(self, tmp_path: Path) -> None:
        """A run with nothing to derive must not look like a successful one."""
        db_path = tmp_path / "empty.db"
        PipelineStore(str(db_path)).close()

        result = run_cftc_feature_derivation({"db_path": str(db_path)}, {})

        assert result["rows_written"] == 0
        assert classify_payload_status(result) == "skipped"
        assert result["reason"]

    def test_repeat_run_also_reports_skipped(self, seeded_db: tuple[Path, float]) -> None:
        db_path, _ = seeded_db
        run_cftc_feature_derivation({"db_path": str(db_path)}, {})
        assert classify_payload_status(run_cftc_feature_derivation({"db_path": str(db_path)}, {})) == "skipped"


# ── Legacy-state transition ───────────────────────────────────────────────
#
# Everything above runs new-code-over-new-code on a virgin derived table, which
# is the one state the live database is *not* in.  It already holds 5,080
# derived rows written by the old ``scripts/`` implementation, computed from a
# raw series that counted 442 byte-identical duplicate rows as separate weeks.
# Re-deriving changes 188 of those values, and ``OBSERVATION_UNIQUE_KEY``
# includes ``value_json`` — so without an explicit supersede the store *inserts*
# a second, contradictory row at an (entity_id, observed_at) that already has
# one, and reports it as rows_written.  Measured on a replica of the live table:
# same-timestamp conflicting groups went 80 -> 209 in a single run.


def _raw_insert(db_path: Path, rows: list[dict[str, Any]]) -> None:
    """Insert rows straight into the table, bypassing the store.

    Deliberately not ``store_entity_observation``: these fixtures have to
    reproduce state the write boundary refuses to create — a byte-identical
    duplicate row, or a stale derived value sitting at a live timestamp.  That
    state exists on disk because it was written before the boundary did.
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.executemany(
            "INSERT INTO entity_observations "
            "(entity_id, source_tool, observed_at, ingested_at, observation_type, "
            " depth_level, value_json, metadata_json) "
            "VALUES (:entity_id, :source_tool, :observed_at, :ingested_at, "
            ":observation_type, :depth_level, :value_json, :metadata_json)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def _derived_by_key(db_path: Path) -> dict[tuple[str, float], list[dict[str, Any]]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in _derived_rows(db_path):
        grouped.setdefault((row["entity_id"], row["observed_at"]), []).append(json.loads(row["value_json"]))
    return grouped


def _derive(db_path: Path) -> dict[str, Any]:
    """Call the Layer 2 function directly, for the stats the DAG node drops.

    ``run_cftc_feature_derivation`` rebuilds its own result envelope from a
    fixed set of keys, so ``rows_superseded``, ``raw_duplicates_dropped`` and
    ``coverage`` never reach the pipeline.  Assert them at the source; the
    node-level tests above and below still go through the node.
    """
    store = PipelineStore(str(db_path))
    try:
        return derive_and_store(store)
    finally:
        store.close()


def _ingested_at(db_path: Path) -> dict[int, float]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return {
            int(i): float(t)
            for i, t in conn.execute("SELECT id, ingested_at FROM entity_observations WHERE source_tool='cftc_derived'")
        }
    finally:
        conn.close()


class TestLegacyDerivedRowsAreSuperseded:
    """The live transition: pre-existing derived rows holding a *different* value."""

    def test_stale_value_is_replaced_not_appended(self, seeded_db: tuple[Path, float]) -> None:
        db_path, _ = seeded_db
        # One legacy row per entity-week, shaped like the old script's output
        # (same source_tool and observation_type, different values).
        legacy = [
            {
                "entity_id": entity_id,
                "source_tool": "cftc_derived",
                "observed_at": T(week * WEEK + e_idx * 3600.0),
                "ingested_at": 1.0,
                "observation_type": "futures_positioning_derived",
                "depth_level": 2,
                "value_json": json.dumps(
                    {
                        "cftc_mm_pct_52w_rank": 0.1234,
                        "cftc_mm_direction_change": -1.0,
                        "cftc_oi_vs_52w_avg": 9.9999,
                        "mm_net_pct_oi_raw": 0.0,
                    }
                ),
                "metadata_json": '{"source": "add_cftc_derived_features.py"}',
            }
            for e_idx, entity_id in enumerate(ENTITIES)
            for week in range(WEEKS_PER_ENTITY)
        ]
        _raw_insert(db_path, legacy)
        assert len(_derived_rows(db_path)) == len(legacy)

        result = _derive(db_path)

        grouped = _derived_by_key(db_path)
        offenders = {key: values for key, values in grouped.items() if len(values) > 1}
        assert not offenders, (
            f"re-deriving a changed value appended instead of superseding: {offenders}. "
            "microstructure_signals picks between same-timestamp rows by SELECT order."
        )
        assert len(grouped) == len(legacy)
        # And the surviving row is the freshly derived one, not the legacy one.
        assert all(values[0]["cftc_oi_vs_52w_avg"] != 9.9999 for values in grouped.values())
        assert result["rows_superseded"] == len(legacy)
        assert result["rows_written"] == len(legacy)

    def test_legacy_row_with_an_unchanged_value_is_left_alone(self, seeded_db: tuple[Path, float]) -> None:
        """Only the rows that actually changed may be rewritten."""
        db_path, _ = seeded_db
        run_cftc_feature_derivation({"db_path": str(db_path)}, {})
        before = _ingested_at(db_path)

        second = _derive(db_path)

        assert second["rows_superseded"] == 0
        assert second["rows_written"] == 0
        assert _ingested_at(db_path) == before


class TestNoOpRunLeavesTheWriteClockAlone:
    """A run that writes nothing must not look like a run that wrote.

    ``store_entity_observation`` refreshes ``ingested_at`` on a collapsed
    duplicate, so re-offering every row daily pins ``MAX(ingested_at)`` to
    "now" forever.  That silently disables both write-activity detectors in
    ``scripts/check_freshness.py`` — ``REASON_SILENT`` and
    ``REASON_NO_NEW_ROWS`` — for source ``cftc_derived``, leaving only the
    ``observed_at`` staleness check.  The watchdog's own docstring calls that
    shape "the bug that hides behind a green checkmark".
    """

    def test_repeat_run_updates_no_ingested_at(self, seeded_db: tuple[Path, float]) -> None:
        db_path, _ = seeded_db
        run_cftc_feature_derivation({"db_path": str(db_path)}, {})
        before = _ingested_at(db_path)
        assert before, "first run wrote nothing — the rest of this test proves nothing"

        run_cftc_feature_derivation({"db_path": str(db_path)}, {})

        after = _ingested_at(db_path)
        moved = {row_id for row_id, ts in after.items() if before.get(row_id) != ts}
        assert not moved, (
            f"{len(moved)} of {len(after)} rows had ingested_at refreshed by a run that "
            "wrote nothing — check_freshness can no longer tell this source apart from a live one"
        )


class TestRawDuplicateCollapseIsLoadBearing:
    """The 442 duplicate raw rows on the live table shift every window they sit in."""

    def test_duplicate_raw_rows_do_not_move_the_rolling_mean(self, seeded_db: tuple[Path, float]) -> None:
        db_path, _ = seeded_db
        clean = _derive(db_path)
        clean_values = _derived_by_key(db_path)

        # Hand-computed, duplicate-free: entity 0's open interest is
        # 500_000 + 1_000 * week over 8 weeks, so the week-7 trailing mean is
        # 503_500 and cftc_oi_vs_52w_avg is 507_000 / 503_500.
        last_key = (ENTITIES[0], T((WEEKS_PER_ENTITY - 1) * WEEK))
        assert clean_values[last_key][0]["cftc_oi_vs_52w_avg"] == pytest.approx(round(507_000 / 503_500, 4))

        # Now duplicate week 0 for each entity, byte-identically, the way the
        # live table does.  Written raw: the store would collapse them.
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            originals = conn.execute(
                "SELECT * FROM entity_observations WHERE observation_type='futures_positioning' "
                "ORDER BY entity_id, observed_at"
            ).fetchall()
        finally:
            conn.close()
        firsts: dict[str, sqlite3.Row] = {}
        for row in originals:
            firsts.setdefault(row["entity_id"], row)
        _raw_insert(
            db_path,
            [
                {
                    "entity_id": row["entity_id"],
                    "source_tool": row["source_tool"],
                    "observed_at": row["observed_at"],
                    "ingested_at": row["ingested_at"],
                    "observation_type": row["observation_type"],
                    "depth_level": row["depth_level"],
                    "value_json": row["value_json"],
                    "metadata_json": row["metadata_json"],
                }
                for row in firsts.values()
            ],
        )

        after = _derive(db_path)

        assert after["raw_duplicates_dropped"] == len(ENTITIES)
        # Nothing changed, because the duplicate week was collapsed away.
        assert after["rows_written"] == 0
        assert after["rows_superseded"] == 0
        assert _derived_by_key(db_path)[last_key][0]["cftc_oi_vs_52w_avg"] == pytest.approx(
            round(507_000 / 503_500, 4)
        ), "a byte-identical duplicate raw row was counted as a second week and moved the trailing mean"
        assert after["rows_derived"] == clean["rows_derived"]

    def test_conflicting_raw_rows_collapse_to_the_newest(self, seeded_db: tuple[Path, float]) -> None:
        """Two *different* payloads at one (entity, week) is a revision, not two weeks."""
        db_path, _ = seeded_db
        entity_id = ENTITIES[0]
        observed_at = T((WEEKS_PER_ENTITY - 1) * WEEK)
        _raw_insert(
            db_path,
            [
                {
                    "entity_id": entity_id,
                    "source_tool": "cftc",
                    "observed_at": observed_at,
                    "ingested_at": 4_102_444_800.0,  # newest by a wide margin
                    "observation_type": "futures_positioning",
                    "depth_level": 2,
                    "value_json": json.dumps({"open_interest": 1_000_000, "mm_net": 1, "mm_net_pct_oi": 0.5}),
                    "metadata_json": None,
                }
            ],
        )

        result = _derive(db_path)

        assert result["raw_conflicts_collapsed"] == 1
        grouped = _derived_by_key(db_path)
        assert len(grouped[(entity_id, observed_at)]) == 1, "a revised raw week produced two derived rows"
        # The revision won: open_interest 1_000_000 against a ~503k trailing mean.
        assert grouped[(entity_id, observed_at)][0]["cftc_oi_vs_52w_avg"] > 1.5


class TestCoverageIsReported:
    """Row counts alone cannot tell a full refill from a 13% one.

    On the live database only 3 COT report dates land in the 162 days after the
    derived table froze — roughly 23 weekly publications are missing, and the
    derivation cannot create a week ``fetch_cftc`` never fetched (the node runs
    it in ``mode="latest"``; ``mode="historical"`` is the only backfill path and
    no node calls it).
    """

    def test_a_gappy_input_reports_partial_coverage(self, seeded_db: tuple[Path, float]) -> None:
        db_path, _ = seeded_db
        result = _derive(db_path)

        coverage = result["coverage"]
        assert coverage["report_dates_expected"] == COVERAGE_WINDOW_WEEKS
        assert coverage["report_dates_present"] == WEEKS_PER_ENTITY
        assert coverage["ratio"] == pytest.approx(WEEKS_PER_ENTITY / COVERAGE_WINDOW_WEEKS, abs=1e-4)
        assert coverage["ratio"] < 1.0, "8 weeks of input inside a 12-week window is not full coverage"

    def test_a_complete_window_reports_full_coverage(self, seeded_db: tuple[Path, float]) -> None:
        db_path, _ = seeded_db
        store = PipelineStore(str(db_path))
        try:
            by_entity = load_raw_positioning(store).by_entity
        finally:
            store.close()

        assert coverage_stats(by_entity, weeks=WEEKS_PER_ENTITY)["ratio"] == pytest.approx(1.0)

    def test_empty_input_is_zero_not_one(self) -> None:
        """No data must not read as complete coverage."""
        empty = coverage_stats({})
        assert empty["ratio"] == 0.0
        assert empty["latest_observed_at"] is None
