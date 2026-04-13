"""Tests for sanctions_monitor L2 entity persistence."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from agent.tools.sanctions_monitor import SanctionsMonitorTool, _PROGRAM_COUNTRY


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
        country_calls = [
            c
            for c in store.register_entity.call_args_list
            if c.kwargs.get("entity_type") == "country"
        ]
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

    def test_persistence_exception_non_fatal(self):
        """_persist_entities_inner exception should not propagate."""
        store = _make_mock_store()
        store.register_entity.side_effect = RuntimeError("DB error")
        tool = SanctionsMonitorTool(cache=None, pipeline_store=store)
        rec = _make_record()
        # Should not raise
        tool._persist_entities([rec])

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
