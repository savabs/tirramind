"""Write-path guards on ``entity_observations`` (audit 2026-09-23, P1.1 / P1.4).

``PipelineStore.store_entity_observation`` is the single write boundary every
collector passes through.  Before these guards it accepted anything:

* **P1.1** — no sanity bound on ``observed_at``.  The live database holds rows
  dated 1920-01-01, 1972-03-24 and 2030-01-30.  ``trainer.py`` derives its
  train/val/test boundaries from ``MIN``/``MAX(observed_at)``, so a single
  future-dated row drags the test window out to 2030.
* **P1.4** — no uniqueness constraint and no dedup.  121,597 byte-identical
  duplicate rows, 31.6% of the table, from a backfill that ran three times.

Every test below fails against the pre-fix ``store_entity_observation``
(a plain ``INSERT`` with no validation): the rejection tests because nothing
raised, the dedup tests because the row count grew.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import threading
import time

import pytest

from agent.pipeline.store import (
    OBSERVED_AT_MAX_AHEAD_SECONDS,
    OBSERVED_AT_MIN_EPOCH,
    ObservationRejected,
    PipelineStore,
)

ENTITY = "e_guard_test"


def _epoch(year: int, month: int = 1, day: int = 1) -> float:
    return dt.datetime(year, month, day, tzinfo=dt.UTC).timestamp()


@pytest.fixture()
def store() -> PipelineStore:
    s = PipelineStore(db_path=":memory:")
    s.register_entity(
        entity_type="instrument",
        canonical_name="Guard Test Entity",
        entity_id=ENTITY,
    )
    yield s
    s.close()


def _count(store: PipelineStore) -> int:
    row = store._get_conn().execute("SELECT COUNT(*) AS n FROM entity_observations").fetchone()
    return int(row["n"])


# ── P1.1 · observed_at range guard ────────────────────────────


class TestObservedAtRangeGuard:
    """Implausible timestamps are refused loudly, never clamped."""

    def test_ancient_timestamp_is_rejected(self, store: PipelineStore) -> None:
        # The real row: gdelt/geopolitical_event at 1920-01-01.
        with pytest.raises(ObservationRejected, match="implausibly old"):
            store.store_entity_observation(
                entity_id=ENTITY,
                source_tool="gdelt",
                observed_at=_epoch(1920),
                observation_type="geopolitical_event",
                value={"tone": -3.1},
            )
        assert _count(store) == 0

    def test_pre_1990_timestamp_is_rejected(self, store: PipelineStore) -> None:
        # The real rows: drug_regulatory/drug_approval from 1972-03-24.
        with pytest.raises(ObservationRejected):
            store.store_entity_observation(
                entity_id=ENTITY,
                source_tool="drug_regulatory",
                observed_at=_epoch(1972, 3, 24),
                observation_type="drug_approval",
                value={"n": 1},
            )
        assert _count(store) == 0

    def test_future_timestamp_is_rejected(self, store: PipelineStore) -> None:
        # The real rows: gov_contracts/contract_award out to 2030-01-30.
        with pytest.raises(ObservationRejected, match="future"):
            store.store_entity_observation(
                entity_id=ENTITY,
                source_tool="gov_contracts",
                observed_at=_epoch(2030, 1, 30),
                observation_type="contract_award",
                value={"amount_usd": 1_000_000},
            )
        assert _count(store) == 0

    def test_rejection_does_not_clamp_to_now(self, store: PipelineStore) -> None:
        """A bad date must not become a plausible-looking recent observation."""
        with pytest.raises(ObservationRejected):
            store.store_entity_observation(
                entity_id=ENTITY,
                source_tool="gov_contracts",
                observed_at=_epoch(2030, 1, 30),
                observation_type="contract_award",
                value={"amount_usd": 1},
            )
        rows = store._get_conn().execute("SELECT * FROM entity_observations").fetchall()
        assert rows == [], "rejected row must not be stored under any timestamp"

    def test_boundary_valid_timestamps_are_accepted(self, store: PipelineStore) -> None:
        now = time.time()
        accepted = [
            ("floor", OBSERVED_AT_MIN_EPOCH),  # exactly 1990-01-01
            ("recent", now - 3600.0),
            ("near_horizon", now + OBSERVED_AT_MAX_AHEAD_SECONDS - 60.0),
        ]
        for label, ts in accepted:
            row_id = store.store_entity_observation(
                entity_id=ENTITY,
                source_tool="cftc",
                observed_at=ts,
                observation_type=f"positioning_{label}",
                value={"net": 1.0},
            )
            assert row_id > 0, label
        assert _count(store) == len(accepted)

    def test_just_below_floor_is_rejected(self, store: PipelineStore) -> None:
        with pytest.raises(ObservationRejected):
            store.store_entity_observation(
                entity_id=ENTITY,
                source_tool="cftc",
                observed_at=OBSERVED_AT_MIN_EPOCH - 1.0,
                observation_type="positioning",
                value={"net": 1.0},
            )
        assert _count(store) == 0

    def test_zero_and_negative_epochs_are_rejected(self, store: PipelineStore) -> None:
        for ts in (0.0, -1.0):
            with pytest.raises(ObservationRejected):
                store.store_entity_observation(
                    entity_id=ENTITY,
                    source_tool="gdelt",
                    observed_at=ts,
                    observation_type="geopolitical_event",
                    value={},
                )
        assert _count(store) == 0

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_timestamps_are_rejected(self, store: PipelineStore, bad: float) -> None:
        """NaN fails every comparison, so a bare range check would admit it."""
        with pytest.raises(ObservationRejected, match="finite"):
            store.store_entity_observation(
                entity_id=ENTITY,
                source_tool="defi_flows",
                observed_at=bad,
                observation_type="tvl_change",
                value={"tvl": 1.0},
            )
        assert _count(store) == 0

    def test_non_numeric_timestamp_is_rejected(self, store: PipelineStore) -> None:
        with pytest.raises(ObservationRejected, match="not a number"):
            store.store_entity_observation(
                entity_id=ENTITY,
                source_tool="defi_flows",
                observed_at="2030-01-30",  # type: ignore[arg-type]
                observation_type="tvl_change",
                value={"tvl": 1.0},
            )
        with pytest.raises(ObservationRejected, match="not a number"):
            store.store_entity_observation(
                entity_id=ENTITY,
                source_tool="defi_flows",
                observed_at=None,  # type: ignore[arg-type]
                observation_type="tvl_change",
                value={"tvl": 1.0},
            )
        assert _count(store) == 0

    def test_rejections_are_counted_and_logged(self, store: PipelineStore, caplog: pytest.LogCaptureFixture) -> None:
        """The guard must be loud and countable, not a quiet drop."""
        with caplog.at_level(logging.ERROR, logger="agent.pipeline.store"):
            for _ in range(3):
                with pytest.raises(ObservationRejected):
                    store.store_entity_observation(
                        entity_id=ENTITY,
                        source_tool="gov_contracts",
                        observed_at=_epoch(2030, 1, 30),
                        observation_type="contract_award",
                        value={},
                    )
        stats = store.observation_write_stats()
        assert stats["rejected"] == {"gov_contracts/contract_award": 3}
        assert len([r for r in caplog.records if r.levelno >= logging.ERROR]) == 3
        assert "gov_contracts/contract_award" in caplog.text

    def test_max_observed_at_stays_inside_the_window(self, store: PipelineStore) -> None:
        """The downstream symptom: MAX(observed_at) drives the trainer's split."""
        store.store_entity_observation(
            entity_id=ENTITY,
            source_tool="cftc",
            observed_at=time.time() - 86_400.0,
            observation_type="positioning",
            value={"net": 1.0},
        )
        with pytest.raises(ObservationRejected):
            store.store_entity_observation(
                entity_id=ENTITY,
                source_tool="gov_contracts",
                observed_at=_epoch(2030, 1, 30),
                observation_type="contract_award",
                value={},
            )
        row = (
            store._get_conn()
            .execute("SELECT MAX(observed_at) AS hi, MIN(observed_at) AS lo FROM entity_observations")
            .fetchone()
        )
        assert row["hi"] <= time.time() + OBSERVED_AT_MAX_AHEAD_SECONDS
        assert row["lo"] >= OBSERVED_AT_MIN_EPOCH


# ── P1.4 · write-path idempotency ─────────────────────────────


class TestDuplicateObservationGuard:
    """Re-running a collector must not double-count."""

    KEY = dict(
        entity_id=ENTITY,
        source_tool="defi_flows",
        observed_at=1_750_000_000.0,
        observation_type="tvl_change",
    )

    def test_identical_write_twice_stores_one_row(self, store: PipelineStore) -> None:
        first = store.store_entity_observation(**self.KEY, value={"tvl": 1.0})
        second = store.store_entity_observation(**self.KEY, value={"tvl": 1.0})
        assert _count(store) == 1
        assert first == second, "the duplicate write must return the existing row id"

    def test_triple_rerun_of_a_backfill_stores_one_row(self, store: PipelineStore) -> None:
        """The live database's 105,172 tvl_change duplicates came from this."""
        for _ in range(3):
            store.store_entity_observation(**self.KEY, value={"tvl": 1.0})
        assert _count(store) == 1

    def test_non_key_columns_take_the_last_write(self, store: PipelineStore) -> None:
        store.store_entity_observation(**self.KEY, value={"tvl": 1.0})
        store.store_entity_observation(
            **self.KEY,
            value={"tvl": 1.0},
            depth_level=3,
            metadata={"revision": "corrected"},
        )
        row = store._get_conn().execute("SELECT * FROM entity_observations").fetchone()
        assert json.loads(row["value_json"]) == {"tvl": 1.0}
        assert row["depth_level"] == 3
        assert json.loads(row["metadata_json"]) == {"revision": "corrected"}

    def test_ingested_at_advances_on_the_collapsed_row(self, store: PipelineStore) -> None:
        store.store_entity_observation(**self.KEY, value={"tvl": 1.0})
        before = store._get_conn().execute("SELECT ingested_at FROM entity_observations").fetchone()
        time.sleep(0.01)
        store.store_entity_observation(**self.KEY, value={"tvl": 1.0})
        after = store._get_conn().execute("SELECT ingested_at FROM entity_observations").fetchone()
        assert after["ingested_at"] > before["ingested_at"]

    def test_each_key_column_distinguishes_rows(self, store: PipelineStore) -> None:
        """Only an exact key match collapses — nothing legitimate is lost."""
        store.register_entity(entity_type="instrument", canonical_name="Other", entity_id="e_other")
        store.store_entity_observation(**self.KEY, value={"tvl": 1.0})
        variants: list[tuple[dict, dict]] = [
            ({**self.KEY, "entity_id": "e_other"}, {"tvl": 1.0}),
            ({**self.KEY, "source_tool": "whale_alert"}, {"tvl": 1.0}),
            ({**self.KEY, "observation_type": "tvl_snapshot"}, {"tvl": 1.0}),
            ({**self.KEY, "observed_at": self.KEY["observed_at"] + 1.0}, {"tvl": 1.0}),
            (dict(self.KEY), {"tvl": 2.0}),  # same key columns, different value
        ]
        for key, value in variants:
            store.store_entity_observation(**key, value=value)
        assert _count(store) == 1 + len(variants)

    def test_distinct_events_sharing_a_timestamp_are_all_kept(self, store: PipelineStore) -> None:
        """The reason ``value_json`` is part of the uniqueness key.

        GDELT timestamps are day-granular and a country legitimately has
        hundreds of distinct events per day — one sampled entity/day in the
        live database holds 255 rows with 255 distinct ``event_id`` values.
        A key of ``(entity_id, source_tool, observation_type, observed_at)``
        alone would collapse them all into one, turning a duplicate-row bug
        into silent data loss.
        """
        day = 1_591_099_200.0  # a real day-granular GDELT stamp
        for event_id in range(255):
            store.store_entity_observation(
                entity_id=ENTITY,
                source_tool="gdelt",
                observed_at=day,
                observation_type="geopolitical_event",
                value={"event_id": str(926_908_863 + event_id), "goldstein": -6.5},
            )
        assert _count(store) == 255
        # ...and a genuine re-run of that same day is still idempotent.
        for event_id in range(255):
            store.store_entity_observation(
                entity_id=ENTITY,
                source_tool="gdelt",
                observed_at=day,
                observation_type="geopolitical_event",
                value={"event_id": str(926_908_863 + event_id), "goldstein": -6.5},
            )
        assert _count(store) == 255
        assert store.observation_write_stats()["deduplicated"] == {"gdelt/geopolitical_event": 255}

    def test_depth_level_alone_does_not_create_a_second_row(self, store: PipelineStore) -> None:
        """depth_level is not part of the uniqueness key — it is overwritten."""
        store.store_entity_observation(**self.KEY, value={"tvl": 1.0}, depth_level=1)
        store.store_entity_observation(**self.KEY, value={"tvl": 1.0}, depth_level=2)
        assert _count(store) == 1

    def test_collapses_are_counted_and_logged(self, store: PipelineStore, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="agent.pipeline.store"):
            store.store_entity_observation(**self.KEY, value={"tvl": 1.0})
            store.store_entity_observation(**self.KEY, value={"tvl": 1.0})
            store.store_entity_observation(**self.KEY, value={"tvl": 1.0})
        stats = store.observation_write_stats()
        assert stats["deduplicated"] == {"defi_flows/tvl_change": 2}
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings, "a collapsed duplicate must not be silent"
        assert "defi_flows/tvl_change" in caplog.text

    def test_a_clean_run_reports_nothing(self, store: PipelineStore) -> None:
        store.store_entity_observation(**self.KEY, value={"tvl": 1.0})
        assert store.observation_write_stats() == {"rejected": {}, "deduplicated": {}}

    def test_query_returns_the_single_row(self, store: PipelineStore) -> None:
        """The dedup must be visible through the public read path too."""
        store.store_entity_observation(**self.KEY, value={"tvl": 1.0})
        store.store_entity_observation(**self.KEY, value={"tvl": 1.0})
        rows = store.query_entity_observations(ENTITY)
        assert len(rows) == 1
        assert rows[0]["value"] == {"tvl": 1.0}
        assert store.count_entity_observations(ENTITY) == 1

    def test_concurrent_identical_writes_collapse(self, store: PipelineStore) -> None:
        """DAGExecutor runs collectors in a ThreadPoolExecutor on one store."""
        barrier = threading.Barrier(8)
        errors: list[BaseException] = []

        def write() -> None:
            try:
                barrier.wait(timeout=10)
                store.store_entity_observation(**self.KEY, value={"tvl": 1.0})
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                errors.append(exc)

        threads = [threading.Thread(target=write) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert not errors, errors
        assert _count(store) == 1


# ── the two guards together ───────────────────────────────────


def test_rejected_row_is_not_resurrected_by_a_later_valid_write(store: PipelineStore) -> None:
    with pytest.raises(ObservationRejected):
        store.store_entity_observation(
            entity_id=ENTITY,
            source_tool="gov_contracts",
            observed_at=_epoch(2030, 1, 30),
            observation_type="contract_award",
            value={"amount_usd": 5},
        )
    good = time.time() - 60.0
    store.store_entity_observation(
        entity_id=ENTITY,
        source_tool="gov_contracts",
        observed_at=good,
        observation_type="contract_award",
        value={"amount_usd": 5},
    )
    rows = store._get_conn().execute("SELECT observed_at FROM entity_observations").fetchall()
    assert [r["observed_at"] for r in rows] == [good]
