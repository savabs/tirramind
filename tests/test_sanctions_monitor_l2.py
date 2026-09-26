"""Tests for sanctions_monitor L2 entity persistence."""

from __future__ import annotations

import contextlib
import json
import os
import time
from datetime import UTC, date, datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.pipeline.store import OBSERVED_AT_MIN_EPOCH
from agent.tools.sanctions_monitor import (
    _PROGRAM_COUNTRY,
    SanctionsMonitorTool,
    _parse_source_date,
)

# ── Helpers ──────────────────────────────────────────────────────


def _make_record(
    *,
    name: str = "Test Entity",
    sdn_type: str = "entity",
    source: str = "ofac",
    entity_id: str = "12345",
    programs: list[str] | None = None,
    listed_date: str | None = None,
    nationality: str | None = None,
    aliases: list[str] | None = None,
    remarks: str = "",
) -> dict[str, Any]:
    return {
        "source": source,
        "entity_id": entity_id,
        "name": name,
        "type": sdn_type,
        "programs": programs or [],
        "listed_date": listed_date,
        "last_updated": None,
        "nationality": nationality,
        "aliases": aliases or [],
        "remarks": remarks,
    }


def _make_mock_store() -> MagicMock:
    store = MagicMock()
    store.register_entity.return_value = "mock_eid"
    store.store_entity_observation.return_value = 1
    store.link_entities.return_value = 1
    store.add_entity_alias.return_value = None
    return store


# ── No pipeline_store → graceful skip ──


class TestNoPipelineStore:
    def test_persist_entities_no_store(self):
        """No pipeline_store → _persist_entities is a no-op."""
        tool = SanctionsMonitorTool(cache=None)
        # Should not raise
        tool._persist_entities([_make_record()])

    def test_persist_entities_empty_results(self):
        """Empty results → no calls."""
        store = _make_mock_store()
        tool = SanctionsMonitorTool(cache=None, pipeline_store=store)
        tool._persist_entities([])
        store.register_entity.assert_not_called()
        store.store_entity_observation.assert_not_called()


# ── Entity type mapping ──


class TestEntityTypeMapping:
    def test_individual_maps_to_person(self):
        store = _make_mock_store()
        tool = SanctionsMonitorTool(cache=None, pipeline_store=store)
        rec = _make_record(name="John Doe", sdn_type="individual")
        tool._persist_entities([rec])
        reg_call = store.register_entity.call_args_list[0]
        assert reg_call.kwargs["entity_type"] == "person"

    def test_entity_maps_to_organization(self):
        store = _make_mock_store()
        tool = SanctionsMonitorTool(cache=None, pipeline_store=store)
        rec = _make_record(name="Evil Corp LLC", sdn_type="entity")
        tool._persist_entities([rec])
        reg_call = store.register_entity.call_args_list[0]
        assert reg_call.kwargs["entity_type"] == "organization"

    def test_vessel_maps_to_vessel(self):
        store = _make_mock_store()
        tool = SanctionsMonitorTool(cache=None, pipeline_store=store)
        rec = _make_record(name="MV Shadow", sdn_type="vessel")
        tool._persist_entities([rec])
        reg_call = store.register_entity.call_args_list[0]
        assert reg_call.kwargs["entity_type"] == "vessel"

    def test_aircraft_maps_to_organization(self):
        store = _make_mock_store()
        tool = SanctionsMonitorTool(cache=None, pipeline_store=store)
        rec = _make_record(name="Air Korea", sdn_type="aircraft")
        tool._persist_entities([rec])
        reg_call = store.register_entity.call_args_list[0]
        assert reg_call.kwargs["entity_type"] == "organization"

    def test_unknown_type_defaults_to_organization(self):
        store = _make_mock_store()
        tool = SanctionsMonitorTool(cache=None, pipeline_store=store)
        rec = _make_record(name="Unk Thing", sdn_type="widget")
        tool._persist_entities([rec])
        reg_call = store.register_entity.call_args_list[0]
        assert reg_call.kwargs["entity_type"] == "organization"


# ── Observation storage ──


class TestObservationStorage:
    def test_observation_stored_with_correct_type(self):
        store = _make_mock_store()
        tool = SanctionsMonitorTool(cache=None, pipeline_store=store)
        rec = _make_record(programs=["IRAN"])
        tool._persist_entities([rec])
        obs_call = store.store_entity_observation.call_args
        assert obs_call.kwargs["observation_type"] == "sanctions_listing"
        assert obs_call.kwargs["depth_level"] == 2
        assert obs_call.kwargs["source_tool"] == "sanctions_monitor"

    def test_observation_value_contains_programs(self):
        store = _make_mock_store()
        tool = SanctionsMonitorTool(cache=None, pipeline_store=store)
        rec = _make_record(programs=["IRAN", "SDGT"])
        tool._persist_entities([rec])
        obs_call = store.store_entity_observation.call_args
        assert obs_call.kwargs["value"]["programs"] == ["IRAN", "SDGT"]

    def test_listed_date_used_for_timestamp(self):
        store = _make_mock_store()
        tool = SanctionsMonitorTool(cache=None, pipeline_store=store)
        rec = _make_record(listed_date="2024-01-15")
        tool._persist_entities([rec])
        obs_call = store.store_entity_observation.call_args
        # Should be a float timestamp
        assert isinstance(obs_call.kwargs["observed_at"], float)

    def test_no_date_uses_now(self):
        store = _make_mock_store()
        tool = SanctionsMonitorTool(cache=None, pipeline_store=store)
        rec = _make_record(listed_date=None)
        before = time.time()
        tool._persist_entities([rec])
        after = time.time()
        obs_call = store.store_entity_observation.call_args
        ts = obs_call.kwargs["observed_at"]
        assert before <= ts <= after


# ── Program → Country links ──


class TestProgramCountryLinks:
    def test_known_program_creates_link(self):
        store = _make_mock_store()
        tool = SanctionsMonitorTool(cache=None, pipeline_store=store)
        rec = _make_record(name="Evil Corp", sdn_type="entity", programs=["IRAN"])
        tool._persist_entities([rec])
        # Should register country entity + link
        assert store.link_entities.call_count == 1
        link_call = store.link_entities.call_args
        assert link_call.kwargs["link_type"] == "sanctioned_under"
        assert link_call.kwargs["confidence"] == 0.95

    def test_multi_country_program_skips_link(self):
        store = _make_mock_store()
        tool = SanctionsMonitorTool(cache=None, pipeline_store=store)
        rec = _make_record(programs=["SDGT"])  # global, maps to None
        tool._persist_entities([rec])
        store.link_entities.assert_not_called()

    def test_unknown_program_skips_link(self):
        store = _make_mock_store()
        tool = SanctionsMonitorTool(cache=None, pipeline_store=store)
        rec = _make_record(programs=["TOTALLY_UNKNOWN_PROG"])
        tool._persist_entities([rec])
        store.link_entities.assert_not_called()

    def test_multiple_programs_create_multiple_links(self):
        store = _make_mock_store()
        tool = SanctionsMonitorTool(cache=None, pipeline_store=store)
        rec = _make_record(programs=["IRAN", "RUSSIA"])
        tool._persist_entities([rec])
        # Should create 2 links (IRAN→IR, RUSSIA→RU)
        assert store.link_entities.call_count == 2

    def test_country_entity_registered(self):
        store = _make_mock_store()
        tool = SanctionsMonitorTool(cache=None, pipeline_store=store)
        rec = _make_record(programs=["CUBA"])
        tool._persist_entities([rec])
        # Find the country registration call
        country_calls = [c for c in store.register_entity.call_args_list if c.kwargs.get("entity_type") == "country"]
        assert len(country_calls) == 1
        assert country_calls[0].kwargs["canonical_name"] == "CU"

    def test_no_programs_no_links(self):
        store = _make_mock_store()
        tool = SanctionsMonitorTool(cache=None, pipeline_store=store)
        rec = _make_record(programs=[])
        tool._persist_entities([rec])
        store.link_entities.assert_not_called()


# ── Deduplication ──


class TestDeduplication:
    def test_same_entity_twice_registered_once(self):
        store = _make_mock_store()
        tool = SanctionsMonitorTool(cache=None, pipeline_store=store)
        rec1 = _make_record(name="Evil Corp", entity_id="111")
        rec2 = _make_record(name="Evil Corp", entity_id="222")
        tool._persist_entities([rec1, rec2])
        # Same name → same entity_id → registered once
        # But 2 observations and 2 alias calls
        assert store.register_entity.call_count == 1
        assert store.store_entity_observation.call_count == 2

    def test_different_entities_each_registered(self):
        store = _make_mock_store()
        tool = SanctionsMonitorTool(cache=None, pipeline_store=store)
        rec1 = _make_record(name="Corp A", entity_id="111")
        rec2 = _make_record(name="Corp B", entity_id="222")
        tool._persist_entities([rec1, rec2])
        # Different names → different entity_ids → both registered
        assert store.register_entity.call_count == 2


# ── Edge cases ──


class TestEdgeCases:
    def test_empty_name_skipped(self):
        store = _make_mock_store()
        tool = SanctionsMonitorTool(cache=None, pipeline_store=store)
        rec = _make_record(name="")
        tool._persist_entities([rec])
        store.register_entity.assert_not_called()

    def test_none_name_skipped(self):
        store = _make_mock_store()
        tool = SanctionsMonitorTool(cache=None, pipeline_store=store)
        rec = _make_record()
        rec["name"] = None
        tool._persist_entities([rec])
        store.register_entity.assert_not_called()

    def test_unicode_name(self):
        """Cyrillic/Arabic names should not crash."""
        store = _make_mock_store()
        tool = SanctionsMonitorTool(cache=None, pipeline_store=store)
        rec = _make_record(name="Компания Зло", sdn_type="entity")
        tool._persist_entities([rec])
        assert store.register_entity.call_count == 1

    def test_persistence_exception_is_counted_not_swallowed(self):
        """A store blowing up must not propagate — and must not report 0/0 either.

        It used to only assert "does not raise", which is the shape of test
        that lets a lossy write report a healthy steady state: written=0 and
        rejected=0 is exactly what a run where nothing changed looks like.
        """
        store = _make_mock_store()
        store.register_entity.side_effect = RuntimeError("DB error")
        tool = SanctionsMonitorTool(cache=None, pipeline_store=store)
        outcome = tool._persist_entities([_make_record(), _make_record(name="Other", entity_id="2")])
        assert outcome.written == 0
        assert outcome.rejected == 2, "a lost record must be reported, not folded into 'nothing new'"

    def test_alias_stored_with_source_prefix(self):
        store = _make_mock_store()
        tool = SanctionsMonitorTool(cache=None, pipeline_store=store)
        rec = _make_record(source="ofac", entity_id="99999")
        tool._persist_entities([rec])
        alias_call = store.add_entity_alias.call_args
        assert alias_call[0][1] == "sanctions_ofac"
        assert alias_call[0][2] == "99999"

    def test_un_source_alias(self):
        store = _make_mock_store()
        tool = SanctionsMonitorTool(cache=None, pipeline_store=store)
        rec = _make_record(source="un", entity_id="UN-42")
        tool._persist_entities([rec])
        alias_call = store.add_entity_alias.call_args
        assert alias_call[0][1] == "sanctions_un"


# ── _PROGRAM_COUNTRY coverage ──


class TestProgramCountryMap:
    def test_all_none_values_are_multi_country(self):
        """Programs mapping to None should be intentionally multi-country."""
        for prog, code in _PROGRAM_COUNTRY.items():
            if code is None:
                # These are known multi-country / transnational programs
                assert prog in {
                    "BALKANS",
                    "SDGT",
                    "SDNTK",
                    "FTO",
                    "ISIL",
                    "TCO",
                    "GLOMAG",
                    "CYBER2",
                }, f"Unexpected None mapping for {prog}"

    def test_all_country_codes_are_2_letter(self):
        for prog, code in _PROGRAM_COUNTRY.items():
            if code is not None:
                assert len(code) == 2, f"{prog} maps to non-2-letter code: {code}"
                assert code == code.upper(), f"{prog} maps to non-uppercase: {code}"


# ── Regression: dateless OFAC records must reach the store ──
#
# Reproduces the defect found on 2026-09-23, not a smoke test.
#
# The DAG wired this node as mode="recent", days_back=90. `_execute_recent`
# filters on a per-entry date, and `_parse_ofac_csv` hard-codes
# `listed_date=None`/`last_updated=None` because the OFAC SDN CSV genuinely
# has none. Measured against the live feeds: 19,393 OFAC records, every one
# dateless, all silently discarded before any write. Exactly 11 dated UN rows
# fell inside the window, and because their observed_at came from the source
# listing date they were byte-identical run to run — so every later run took
# the store's UPDATE branch and the node reported SUCCESS with zero rows.
#
# Against the unfixed code every assertion below fails: `snapshot` is not a
# valid mode, and no path persists a dateless record.


class TestSnapshotModePersistsDatelessRecords:
    def _tool(self, store, records):
        tool = SanctionsMonitorTool(cache=None, pipeline_store=store)
        tool._get_records = lambda source: (  # type: ignore[method-assign]
            records,
            {"ofac": {"records": len(records), "body_sha256": "deadbeef"}},
            None,
        )
        return tool

    def _dateless_ofac(self, n: int = 3) -> list[dict[str, Any]]:
        return [
            _make_record(
                name=f"Dateless Corp {i}",
                entity_id=str(9000 + i),
                source="ofac",
                programs=["IRAN"],
            )
            for i in range(n)
        ]

    def test_dateless_ofac_records_are_persisted(self, tmp_path):
        """The whole bug: dateless OFAC records must produce observations."""
        from agent.pipeline.store import PipelineStore

        store = PipelineStore(db_path=str(tmp_path / "probe.db"))
        records = self._dateless_ofac()
        assert all(r["listed_date"] is None and r["last_updated"] is None for r in records)

        result = self._tool(store, records).execute(mode="snapshot")
        assert result.success

        rows = (
            store._get_conn()
            .execute(
                "SELECT observation_type, COUNT(*) FROM entity_observations "
                "WHERE source_tool='sanctions_monitor' GROUP BY 1"
            )
            .fetchall()
        )
        counts = {r[0]: r[1] for r in rows}

        # Pre-fix this is 0 — the date filter dropped every one of them.
        assert counts.get("sanctions_listing") == 3
        assert counts.get("sanctions_roster_snapshot") == 1
        assert result.data["new_listings"] == 3

    def test_dateless_records_get_collection_time_not_epoch_zero(self, tmp_path):
        """No source date → observed_at is collection time, not a bogus epoch."""
        from agent.pipeline.store import PipelineStore

        store = PipelineStore(db_path=str(tmp_path / "probe.db"))
        before = time.time()
        self._tool(store, self._dateless_ofac(1)).execute(mode="snapshot")
        after = time.time()

        row = (
            store._get_conn()
            .execute("SELECT observed_at FROM entity_observations WHERE observation_type='sanctions_listing'")
            .fetchone()
        )
        assert before <= row[0] <= after

    def test_second_identical_run_writes_roster_row_only(self, tmp_path):
        """Steady state: no duplicate listings, but never zero rows either."""
        from agent.pipeline.store import PipelineStore

        store = PipelineStore(db_path=str(tmp_path / "probe.db"))
        records = self._dateless_ofac()

        self._tool(store, records).execute(mode="snapshot")
        conn = store._get_conn()
        after_first = conn.execute("SELECT COUNT(*) FROM entity_observations").fetchone()[0]

        result2 = self._tool(store, records).execute(mode="snapshot")
        after_second = conn.execute("SELECT COUNT(*) FROM entity_observations").fetchone()[0]

        # The roster row is stamped with collection time, so it can never
        # dedup away: the node can never again be green with zero writes.
        assert after_second - after_first == 1
        assert result2.data["new_listings"] == 0
        listings = conn.execute(
            "SELECT COUNT(*) FROM entity_observations WHERE observation_type='sanctions_listing'"
        ).fetchone()[0]
        assert listings == 3

    def test_duplicate_uids_for_one_name_each_get_an_alias(self, tmp_path):
        """Two uids → one canonical entity, but BOTH aliases must be written.

        If only the first uid got an alias the second would never resolve, so
        the first-seen gate would re-admit it on every single run.
        """
        from agent.pipeline.store import PipelineStore

        store = PipelineStore(db_path=str(tmp_path / "probe.db"))
        records = [
            _make_record(name="Twin Corp", entity_id="111", source="ofac"),
            _make_record(name="Twin Corp", entity_id="222", source="ofac"),
        ]
        self._tool(store, records).execute(mode="snapshot")

        aliases = (
            store._get_conn()
            .execute("SELECT external_id FROM entity_aliases WHERE source='sanctions_ofac' ORDER BY external_id")
            .fetchall()
        )
        assert [a[0] for a in aliases] == ["111", "222"]


class TestRecentFilterDatePreference:
    """A record must never be stored under a date older than the one that admitted it.

    Pre-fix the `recent` filter preferred `last_updated` while the persist
    path preferred `listed_date`, so an entity updated recently but listed
    years ago was admitted on the new date and then back-dated to the old one.
    The live DB holds one: entity 5cfe20e6…, listed 2025-06-16 / updated
    2026-08-14, stored at observed_at 2025-06-15.

    Driven through the real HTTP + XML path so it runs identically against the
    unfixed code, where it fails on the assertion rather than on a signature.
    """

    @staticmethod
    def _un_xml(listed_on: str, last_updated: str) -> str:
        return (
            "<CONSOLIDATED_LIST><INDIVIDUALS><INDIVIDUAL>"
            "<DATAID>555</DATAID>"
            "<FIRST_NAME>Stale Listing Person</FIRST_NAME>"
            "<UN_LIST_TYPE>Test</UN_LIST_TYPE>"
            f"<LISTED_ON>{listed_on}</LISTED_ON>"
            f"<LAST_DAY_UPDATED><VALUE>{last_updated}</VALUE></LAST_DAY_UPDATED>"
            "<COMMENTS1>fixture</COMMENTS1>"
            "</INDIVIDUAL></INDIVIDUALS></CONSOLIDATED_LIST>"
        )

    def test_filter_and_persist_agree_on_which_date_wins(self, tmp_path):
        from agent.pipeline.store import PipelineStore

        xml = self._un_xml("2022-06-16", time.strftime("%Y-%m-%d"))
        resp = MagicMock()
        resp.text = xml
        resp.content = xml.encode()
        resp.raise_for_status.return_value = None

        store = PipelineStore(db_path=str(tmp_path / "probe.db"))
        tool = SanctionsMonitorTool(cache=None, pipeline_store=store)
        with patch("agent.tools.sanctions_monitor.httpx.get", return_value=resp):
            result = tool.execute(mode="recent", source="un", days_back=30)

        # listed_date (2022) is what the persist path stamps the row with, and
        # it is outside the 30d window — so the filter must reject the record
        # rather than admit it on last_updated and back-date the stored row.
        assert result.data["count"] == 0
        stored = (
            store._get_conn()
            .execute("SELECT COUNT(*) FROM entity_observations WHERE observation_type='sanctions_listing'")
            .fetchone()[0]
        )
        assert stored == 0

    def test_recently_listed_record_is_still_admitted(self, tmp_path):
        """The alignment must not turn the filter into a no-op."""
        from agent.pipeline.store import PipelineStore

        today = time.strftime("%Y-%m-%d")
        xml = self._un_xml(today, today)
        resp = MagicMock()
        resp.text = xml
        resp.content = xml.encode()
        resp.raise_for_status.return_value = None

        store = PipelineStore(db_path=str(tmp_path / "probe.db"))
        tool = SanctionsMonitorTool(cache=None, pipeline_store=store)
        with patch("agent.tools.sanctions_monitor.httpx.get", return_value=resp):
            result = tool.execute(mode="recent", source="un", days_back=30)

        assert result.data["count"] == 1


# ── Regression: source dates are UTC, not the host box's timezone ──
#
# `_parse_source_date` called `.timestamp()` on a naive datetime, which
# interprets it in the HOST machine's local zone. `UN LISTED_ON` is a bare
# "YYYY-MM-DD", so on the +05:30 box this was measured on, "2025-06-16" became
# 2025-06-15T18:30:00Z — every dated row landed on the previous calendar day,
# and the same feed produced different observed_at values on different
# machines. The live DB holds one: entity 5cfe20e6…, UN LISTED_ON 2025-06-16,
# stored at observed_at 2025-06-15 18:30:00.


class TestParseSourceDateIsUTC:
    @staticmethod
    @contextlib.contextmanager
    def _tz(tz: str):
        """Pin the process timezone for the duration of the block."""
        if not hasattr(time, "tzset"):  # pragma: no cover - Windows
            pytest.skip("time.tzset unavailable on this platform")
        previous = os.environ.get("TZ")
        os.environ["TZ"] = tz
        time.tzset()
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("TZ", None)
            else:
                os.environ["TZ"] = previous
            time.tzset()

    @pytest.mark.parametrize("tz", ["UTC", "Asia/Kolkata", "America/Los_Angeles", "Pacific/Kiritimati"])
    def test_bare_date_is_utc_midnight_in_every_timezone(self, tz):
        """The parsed instant must not depend on which machine ran the collector."""
        with self._tz(tz):
            assert _parse_source_date("2025-06-16") == datetime(2025, 6, 16, tzinfo=UTC).timestamp()

    def test_bare_date_does_not_land_on_the_previous_day(self):
        """The exact live-DB symptom: 2025-06-16 stored as 2025-06-15."""
        with self._tz("Asia/Kolkata"):
            parsed = _parse_source_date("2025-06-16")
        assert datetime.fromtimestamp(parsed, UTC).date() == date(2025, 6, 16)

    def test_explicit_zone_is_respected_not_overridden(self):
        """Forcing UTC must apply to naive values only."""
        assert (
            _parse_source_date("2025-06-16T00:00:00+05:30")
            == datetime(2025, 6, 16, tzinfo=timezone(timedelta(hours=5, minutes=30))).timestamp()
        )
        assert _parse_source_date("2025-06-16T00:00:00Z") == datetime(2025, 6, 16, tzinfo=UTC).timestamp()

    def test_unusable_values_still_return_none(self):
        for raw in ("", "   ", "not-a-date", None, 17, []):
            assert _parse_source_date(raw) is None


# ── Regression: an out-of-range source date must not abort the batch ──
#
# The store refuses observed_at outside [1990-01-01, now+1d] by raising
# (agent/pipeline/store.py, OBSERVED_AT_MIN_EPOCH). `_observation_timestamp`
# guarded only the FUTURE side, and only in snapshot mode, so one source
# record carrying a placeholder like "1899-01-01" reached the store, raised,
# and — caught by the batch-level except — aborted the remaining 20k records
# while the run still reported success with new_listings=0.
#
# This is a fallback for an unusable date, not a relaxation of the store's
# guard: the store still rejects anything out of range that reaches it, and
# the substitution is recorded in the row's metadata rather than hidden.


class TestObservationTimestampGuardIsSymmetric:
    def test_implausibly_old_date_falls_back_to_collection_time(self):
        now = 1_800_000_000.0
        ts, prov = SanctionsMonitorTool._observation_timestamp(_make_record(listed_date="1899-01-01"), now_ts=now)
        assert ts == now
        assert prov == "out_of_range_source_date"
        assert ts >= OBSERVED_AT_MIN_EPOCH

    def test_future_date_falls_back_to_collection_time(self):
        now = 1_800_000_000.0
        ts, prov = SanctionsMonitorTool._observation_timestamp(_make_record(listed_date="2099-01-01"), now_ts=now)
        assert ts == now
        assert prov == "out_of_range_source_date"

    def test_usable_date_is_kept_verbatim(self):
        now = datetime(2026, 9, 23, tzinfo=UTC).timestamp()
        ts, prov = SanctionsMonitorTool._observation_timestamp(_make_record(listed_date="2025-06-16"), now_ts=now)
        assert ts == datetime(2025, 6, 16, tzinfo=UTC).timestamp()
        assert prov == "source_listed_date"

    def test_dateless_record_is_collection_time(self):
        now = 1_800_000_000.0
        ts, prov = SanctionsMonitorTool._observation_timestamp(_make_record(listed_date=None), now_ts=now)
        assert ts == now
        assert prov == "collection_time"

    def test_unparsable_date_is_marked_distinctly(self):
        now = 1_800_000_000.0
        ts, prov = SanctionsMonitorTool._observation_timestamp(_make_record(listed_date="15th of Never"), now_ts=now)
        assert ts == now
        assert prov == "unparsed_source_date"


class TestOutOfRangeDateDoesNotZeroTheRun:
    def _tool(self, store, records):
        tool = SanctionsMonitorTool(cache=None, pipeline_store=store)
        tool._get_records = lambda source: (  # type: ignore[method-assign]
            records,
            {"un": {"records": len(records), "body_sha256": "deadbeef"}},
            None,
        )
        return tool

    def test_one_bad_date_among_five_still_writes_five(self, tmp_path):
        """Pre-fix: 2 rows written, new_listings=0, success=True."""
        from agent.pipeline.store import PipelineStore

        store = PipelineStore(db_path=str(tmp_path / "probe.db"))
        records = [
            _make_record(name=f"Roster Corp {i}", entity_id=str(7000 + i), source="un", listed_date="2025-06-16")
            for i in range(5)
        ]
        records[2]["listed_date"] = "1899-01-01"

        result = self._tool(store, records).execute(mode="snapshot")

        conn = store._get_conn()
        written = conn.execute(
            "SELECT COUNT(*) FROM entity_observations WHERE observation_type='sanctions_listing'"
        ).fetchone()[0]
        assert written == 5
        assert result.data["new_listings"] == 5
        assert result.data["rejected"] == 0
        assert result.success

        roster = json.loads(
            conn.execute(
                "SELECT value_json FROM entity_observations WHERE observation_type='sanctions_roster_snapshot'"
            ).fetchone()[0]
        )
        assert roster["new_listings"] == written
        assert roster["rejected"] == 0
        assert roster["complete"] is True

    def test_the_bad_record_is_stamped_and_labelled(self, tmp_path):
        from agent.pipeline.store import PipelineStore

        store = PipelineStore(db_path=str(tmp_path / "probe.db"))
        before = time.time()
        self._tool(store, [_make_record(name="Ancient Corp", entity_id="7777", listed_date="1899-01-01")]).execute(
            mode="snapshot"
        )
        after = time.time()

        row = (
            store._get_conn()
            .execute(
                "SELECT observed_at, metadata_json FROM entity_observations WHERE observation_type='sanctions_listing'"
            )
            .fetchone()
        )
        assert before <= row[0] <= after
        meta = json.loads(row[1])
        assert meta["observed_at_source"] == "out_of_range_source_date"
        assert meta["source_date_raw"] == "1899-01-01"


# ── Regression: a refused record must not bury itself or fake a green run ──
#
# `_persist_entities` wrapped the WHOLE loop in one try/except and returned 0.
# So a single ObservationRejected mid-batch: (a) abandoned every later record,
# (b) reported new_listings=0 while rows had in fact been written, (c) still
# returned success=True, and (d) permanently suppressed the offending record,
# because its alias was written BEFORE its observation — the first-seen gate
# reads that alias, so the record was skipped on every future run.


class TestRejectedRecordMidBatch:
    def _records(self, n: int = 5) -> list[dict[str, Any]]:
        return [
            _make_record(name=f"Batch Corp {i}", entity_id=str(8000 + i), source="ofac", programs=["IRAN"])
            for i in range(n)
        ]

    def _tool(self, store, records):
        tool = SanctionsMonitorTool(cache=None, pipeline_store=store)
        tool._get_records = lambda source: (  # type: ignore[method-assign]
            records,
            {"ofac": {"records": len(records), "body_sha256": "deadbeef"}},
            None,
        )
        return tool

    @staticmethod
    def _break_nth_listing(store, n: int):
        """Make the n-th (1-based) sanctions_listing write raise, as the store does."""
        from agent.pipeline.store import ObservationRejected

        real = store.store_entity_observation
        calls = {"n": 0}

        def flaky(**kwargs):
            if kwargs.get("observation_type") == "sanctions_listing":
                calls["n"] += 1
                if calls["n"] == n:
                    raise ObservationRejected("synthetic refusal")
            return real(**kwargs)

        store.store_entity_observation = flaky  # type: ignore[method-assign]
        return calls

    def test_remaining_records_still_persist(self, tmp_path):
        """Pre-fix: records 3-5 were never attempted."""
        from agent.pipeline.store import PipelineStore

        store = PipelineStore(db_path=str(tmp_path / "probe.db"))
        records = self._records()
        self._break_nth_listing(store, 3)

        result = self._tool(store, records).execute(mode="snapshot")

        written = (
            store._get_conn()
            .execute("SELECT COUNT(*) FROM entity_observations WHERE observation_type='sanctions_listing'")
            .fetchone()[0]
        )
        assert written == 4
        assert result.data["new_listings"] == 4
        assert result.data["rejected"] == 1

    def test_run_does_not_report_success_with_an_understated_count(self, tmp_path):
        """Green + "0 new listings" while rows were written is the bug itself."""
        from agent.pipeline.store import PipelineStore

        store = PipelineStore(db_path=str(tmp_path / "probe.db"))
        self._break_nth_listing(store, 3)
        result = self._tool(store, self._records()).execute(mode="snapshot")

        assert result.success is False, "a lossy write must let the node's retries fire"
        assert "FAILED" in result.output

    def test_roster_row_records_the_loss(self, tmp_path):
        from agent.pipeline.store import PipelineStore

        store = PipelineStore(db_path=str(tmp_path / "probe.db"))
        self._break_nth_listing(store, 3)
        self._tool(store, self._records()).execute(mode="snapshot")

        roster = json.loads(
            store._get_conn()
            .execute("SELECT value_json FROM entity_observations WHERE observation_type='sanctions_roster_snapshot'")
            .fetchone()[0]
        )
        assert roster["rejected"] == 1
        assert roster["new_listings"] == 4
        assert roster["total_records"] == 5

    def test_refused_record_is_retried_not_buried(self, tmp_path):
        """Its alias must not exist, or the first-seen gate skips it forever."""
        from agent.pipeline.store import PipelineStore

        store = PipelineStore(db_path=str(tmp_path / "probe.db"))
        records = self._records()
        self._break_nth_listing(store, 3)
        self._tool(store, records).execute(mode="snapshot")

        conn = store._get_conn()
        aliases = {r[0] for r in conn.execute("SELECT external_id FROM entity_aliases").fetchall()}
        assert "8002" not in aliases, "alias written for a record whose observation failed"

        # Run 2 on a healthy store: the buried record is recovered, and the
        # four that already landed are NOT re-counted as new designations.
        store2 = PipelineStore(db_path=str(tmp_path / "probe.db"))
        result2 = self._tool(store2, records).execute(mode="snapshot")
        assert result2.data["new_listings"] == 1
        assert result2.success
        total = conn.execute(
            "SELECT COUNT(*) FROM entity_observations WHERE observation_type='sanctions_listing'"
        ).fetchone()[0]
        assert total == 5


# ── Regression: a single-source outage is not a complete roster ──
#
# `_get_records` dropped its errors list whenever the other source returned
# anything, so a 503 on sdn.csv produced a roster row holding only UN's 1,011
# records with no marker that OFAC was missing — and success=True, so the DAG
# node's retries=2 never fired. A downstream diff of consecutive roster rows
# reads that as 19,391 delistings, then 19,391 re-designations the next day:
# noise indistinguishable from the exact signal this tool exists to produce.


class TestPartialSourceFailure:
    _UN_XML = (
        "<CONSOLIDATED_LIST><ENTITIES><ENTITY>"
        "<DATAID>901</DATAID><FIRST_NAME>UN Only Corp</FIRST_NAME>"
        "<UN_LIST_TYPE>Test</UN_LIST_TYPE><LISTED_ON>2025-06-16</LISTED_ON>"
        "<COMMENTS1>fixture</COMMENTS1>"
        "</ENTITY></ENTITIES></CONSOLIDATED_LIST>"
    )

    def _patched_get(self):
        ok = MagicMock()
        ok.text = self._UN_XML
        ok.content = self._UN_XML.encode()
        ok.raise_for_status.return_value = None

        def fake_get(url, **_kwargs):
            if "sdn.csv" in url:
                raise httpx.HTTPStatusError("503", request=MagicMock(), response=MagicMock(status_code=503))
            return ok

        return fake_get

    def test_snapshot_with_one_dead_source_is_not_success(self, tmp_path):
        from agent.pipeline.store import PipelineStore

        store = PipelineStore(db_path=str(tmp_path / "probe.db"))
        tool = SanctionsMonitorTool(cache=None, pipeline_store=store)
        with patch("agent.tools.sanctions_monitor.httpx.get", side_effect=self._patched_get()):
            result = tool.execute(mode="snapshot")

        assert result.success is False, "a partial fetch must let the node's retries fire"
        assert result.data["complete"] is False
        assert result.data["failed_sources"] == ["ofac"]

    def test_failed_source_is_named_in_the_stats_not_omitted(self, tmp_path):
        from agent.pipeline.store import PipelineStore

        store = PipelineStore(db_path=str(tmp_path / "probe.db"))
        tool = SanctionsMonitorTool(cache=None, pipeline_store=store)
        with patch("agent.tools.sanctions_monitor.httpx.get", side_effect=self._patched_get()):
            result = tool.execute(mode="snapshot")

        # Pre-fix there was no "ofac" key at all, so the roster row was
        # indistinguishable from a world with no OFAC list in it.
        assert "ofac" in result.data["sources"]
        assert result.data["sources"]["ofac"]["records"] == 0
        assert "503" in result.data["sources"]["ofac"]["error"]

    def test_roster_row_is_flagged_incomplete(self, tmp_path):
        """The row still gets written — losing the evidence of the outage is worse."""
        from agent.pipeline.store import PipelineStore

        store = PipelineStore(db_path=str(tmp_path / "probe.db"))
        tool = SanctionsMonitorTool(cache=None, pipeline_store=store)
        with patch("agent.tools.sanctions_monitor.httpx.get", side_effect=self._patched_get()):
            tool.execute(mode="snapshot")

        roster = json.loads(
            store._get_conn()
            .execute("SELECT value_json FROM entity_observations WHERE observation_type='sanctions_roster_snapshot'")
            .fetchone()[0]
        )
        assert roster["complete"] is False
        assert roster["sources"]["ofac"]["error"]
        assert roster["total_records"] == 1

    def test_all_sources_dead_still_errors(self, tmp_path):
        from agent.pipeline.store import PipelineStore

        store = PipelineStore(db_path=str(tmp_path / "probe.db"))
        tool = SanctionsMonitorTool(cache=None, pipeline_store=store)

        def dead(url, **_kwargs):
            raise httpx.ConnectError("down")

        with patch("agent.tools.sanctions_monitor.httpx.get", side_effect=dead):
            result = tool.execute(mode="snapshot")
        assert result.success is False
        assert store._get_conn().execute("SELECT COUNT(*) FROM entity_observations").fetchone()[0] == 0


# ── Regression: collection-time rows must be distinguishable from designations ──
#
# 19,365 of the 20,376 rows the live backfill writes carry collection time,
# because only UN records have dates at all. With no provenance marker, day
# one is indistinguishable downstream from 19k designations in one second —
# and after the backfill a genuine new OFAC designation is stamped the same
# way. metadata_json is not part of OBSERVATION_UNIQUE_KEY, so this costs
# nothing in dedup terms.


class TestObservationProvenanceMetadata:
    def _tool(self, store, records):
        tool = SanctionsMonitorTool(cache=None, pipeline_store=store)
        tool._get_records = lambda source: (  # type: ignore[method-assign]
            records,
            {"ofac": {"records": len(records), "body_sha256": "deadbeef"}},
            None,
        )
        return tool

    def test_dateless_and_dated_rows_are_separable(self, tmp_path):
        from agent.pipeline.store import PipelineStore

        store = PipelineStore(db_path=str(tmp_path / "probe.db"))
        records = [
            _make_record(name="Dateless Corp", entity_id="6001", source="ofac"),
            _make_record(name="Dated Corp", entity_id="6002", source="un", listed_date="2025-06-16"),
        ]
        self._tool(store, records).execute(mode="snapshot")

        rows = (
            store._get_conn()
            .execute(
                "SELECT observed_at, metadata_json FROM entity_observations WHERE observation_type='sanctions_listing'"
            )
            .fetchall()
        )
        by_prov = {json.loads(r[1])["observed_at_source"]: r for r in rows}
        assert set(by_prov) == {"collection_time", "source_listed_date"}
        assert by_prov["source_listed_date"][0] == datetime(2025, 6, 16, tzinfo=UTC).timestamp()
        assert all(json.loads(r[1])["first_seen_backfill"] is True for r in rows)

    def test_metadata_does_not_create_a_duplicate_row(self, tmp_path):
        """metadata_json is outside the idempotency key — dedup must be unaffected."""
        from agent.pipeline.store import PipelineStore

        store = PipelineStore(db_path=str(tmp_path / "probe.db"))
        records = [_make_record(name="Dateless Corp", entity_id="6001", source="ofac")]
        self._tool(store, records).execute(mode="snapshot")
        self._tool(store, records).execute(mode="snapshot")

        listings = (
            store._get_conn()
            .execute("SELECT COUNT(*) FROM entity_observations WHERE observation_type='sanctions_listing'")
            .fetchone()[0]
        )
        assert listings == 1


# ── Regression: the roster row is the guarantee, so failing to write it must fail the run ──
#
# The module docstring calls the roster observation the thing that makes this
# node unable to report green while writing nothing. That was only ever true
# of the *dedup* behaviour of the row; failing to write it at all left
# success=True. Two live paths reached that:
#
#   1. snapshot mode with no pipeline store — the single path that writes
#      nothing at all, and the single path the warning explicitly excluded
#      (`if not roster_written and self._store is not None`).
#   2. the roster write raising. sqlite3.OperationalError("database is
#      locked") is live-reachable: DAGExecutor runs nodes in worker threads
#      against a database a backfill may be writing.
#
# In both, ToolOperator.execute (agent/pipeline/operators.py:173) keys off
# result.success, so the node's retries=2 never fired.


class TestRosterWriteFailureFailsTheRun:
    _RECORDS = [_make_record(name=f"Corp {i}", entity_id=str(7000 + i), source="ofac") for i in range(10)]
    _STATS = {"ofac": {"records": 10, "body_sha256": "aa"}}

    def _tool(self, store):
        tool = SanctionsMonitorTool(cache=None, pipeline_store=store)
        tool._get_records = lambda source: (self._RECORDS, self._STATS, None)  # type: ignore[method-assign]
        return tool

    def test_snapshot_without_a_store_is_not_success(self):
        """Nothing was written. That is the definition of a failed collection run."""
        result = self._tool(None).execute(mode="snapshot")
        assert result.success is False
        assert result.data["roster_observation_written"] is False
        assert result.data["new_listings"] == 0

    def test_the_zero_write_path_also_warns(self):
        """The one path that writes nothing used to be the one path excluded from the warning."""
        result = self._tool(None).execute(mode="snapshot")
        assert "FAILED" in result.output
        assert "not diffable" in result.output

    def test_locked_database_on_the_roster_write_is_not_success(self, tmp_path):
        import sqlite3

        from agent.pipeline.store import PipelineStore

        store = PipelineStore(db_path=str(tmp_path / "locked.db"))
        real = store.store_entity_observation

        def flaky(*args, **kwargs):
            if kwargs.get("observation_type") == "sanctions_roster_snapshot":
                raise sqlite3.OperationalError("database is locked")
            return real(*args, **kwargs)

        store.store_entity_observation = flaky  # type: ignore[method-assign]
        result = self._tool(store).execute(mode="snapshot")

        assert result.success is False, "an undiffable run must let the node's retries fire"
        assert result.data["new_listings"] == 10, "the listing rows really did land; say so"
        assert "database is locked" in result.data["roster_error"]
        assert "database is locked" in result.output

        rows = (
            store._get_conn()
            .execute("SELECT observation_type, COUNT(*) FROM entity_observations GROUP BY 1")
            .fetchall()
        )
        assert dict(rows) == {"sanctions_listing": 10}

    def test_healthy_run_is_still_success_and_writes_the_row(self, tmp_path):
        """The guard must not fire on the path it is guarding."""
        from agent.pipeline.store import PipelineStore

        store = PipelineStore(db_path=str(tmp_path / "healthy.db"))
        result = self._tool(store).execute(mode="snapshot")

        assert result.success is True
        assert result.data["roster_observation_written"] is True
        assert result.data["roster_error"] is None
        rows = (
            store._get_conn()
            .execute("SELECT observation_type, COUNT(*) FROM entity_observations GROUP BY 1")
            .fetchall()
        )
        assert dict(rows) == {"sanctions_listing": 10, "sanctions_roster_snapshot": 1}
