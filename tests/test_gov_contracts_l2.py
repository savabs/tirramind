"""Tests for gov_contracts L2 entity persistence."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent.tools.gov_contracts import GovContractsTool


# ── Helpers ──────────────────────────────────────────────────────


def _make_award(
    *,
    recipient: str = "Lockheed Martin Corp",
    agency: str = "Department of Defense",
    award_id: str = "W911QY-24-C-0001",
    amount_usd: float = 1_500_000.0,
    award_type: str = "Contract",
    start_date: str = "2024-06-01",
    end_date: str = "2025-06-01",
    description: str = "Test contract",
) -> dict[str, Any]:
    return {
        "award_id": award_id,
        "recipient": recipient,
        "amount_usd": amount_usd,
        "agency": agency,
        "sub_agency": "US Army",
        "award_type": award_type,
        "start_date": start_date,
        "end_date": end_date,
        "description": description,
    }


def _make_uk_award(
    *,
    recipient: str = "BAE Systems plc",
    agency: str = "Ministry of Defence",
    award_id: str = "ocds-b5fd17-0001",
    amount: float = 750_000.0,
) -> dict[str, Any]:
    return {
        "award_id": award_id,
        "recipient": recipient,
        "amount": amount,
        "currency": "GBP",
        "agency": agency,
        "award_type": "open",
        "start_date": "2024-05-01",
        "end_date": "2025-05-01",
        "description": "UK test contract",
        "region": "uk",
    }


def _make_mock_store() -> MagicMock:
    store = MagicMock()
    store.register_entity.return_value = "mock_eid"
    store.store_entity_observation.return_value = 1
    store.link_entities.return_value = 1
    return store


# ── No pipeline_store → graceful skip ──


class TestNoPipelineStore:
    def test_no_store_is_noop(self):
        tool = GovContractsTool(cache=None)
        tool._persist_entities([_make_award()], "US")

    def test_empty_awards_no_calls(self):
        store = _make_mock_store()
        tool = GovContractsTool(cache=None, pipeline_store=store)
        tool._persist_entities([], "US")
        # Only country entity is NOT registered (empty → early return)
        store.register_entity.assert_not_called()


# ── Entity registration ──


class TestEntityRegistration:
    def test_company_entity_registered(self):
        store = _make_mock_store()
        tool = GovContractsTool(cache=None, pipeline_store=store)
        tool._persist_entities([_make_award()], "US")
        company_calls = [
            c
            for c in store.register_entity.call_args_list
            if c.kwargs.get("entity_type") == "company"
        ]
        assert len(company_calls) == 1

    def test_agency_entity_registered_as_organization(self):
        store = _make_mock_store()
        tool = GovContractsTool(cache=None, pipeline_store=store)
        tool._persist_entities([_make_award()], "US")
        org_calls = [
            c
            for c in store.register_entity.call_args_list
            if c.kwargs.get("entity_type") == "organization"
        ]
        assert len(org_calls) == 1

    def test_country_entity_registered(self):
        store = _make_mock_store()
        tool = GovContractsTool(cache=None, pipeline_store=store)
        tool._persist_entities([_make_award()], "US")
        country_calls = [
            c
            for c in store.register_entity.call_args_list
            if c.kwargs.get("entity_type") == "country"
        ]
        assert len(country_calls) == 1
        assert country_calls[0].kwargs["canonical_name"] == "US"


# ── Observation storage ──


class TestObservationStorage:
    def test_observation_type_is_contract_award(self):
        store = _make_mock_store()
        tool = GovContractsTool(cache=None, pipeline_store=store)
        tool._persist_entities([_make_award()], "US")
        obs_call = store.store_entity_observation.call_args
        assert obs_call.kwargs["observation_type"] == "contract_award"
        assert obs_call.kwargs["depth_level"] == 2
        assert obs_call.kwargs["source_tool"] == "gov_contracts"

    def test_observation_value_contains_award_info(self):
        store = _make_mock_store()
        tool = GovContractsTool(cache=None, pipeline_store=store)
        award = _make_award(amount_usd=5_000_000.0, agency="DOE")
        tool._persist_entities([award], "US")
        obs_call = store.store_entity_observation.call_args
        val = obs_call.kwargs["value"]
        assert val["amount_usd"] == 5_000_000.0
        assert val["agency"] == "DOE"
        assert val["country"] == "US"

    def test_start_date_used_for_timestamp(self):
        store = _make_mock_store()
        tool = GovContractsTool(cache=None, pipeline_store=store)
        tool._persist_entities([_make_award(start_date="2024-03-15")], "US")
        obs_call = store.store_entity_observation.call_args
        assert isinstance(obs_call.kwargs["observed_at"], float)

    def test_missing_date_uses_now(self):
        store = _make_mock_store()
        tool = GovContractsTool(cache=None, pipeline_store=store)
        before = time.time()
        tool._persist_entities([_make_award(start_date="")], "US")
        after = time.time()
        obs_call = store.store_entity_observation.call_args
        ts = obs_call.kwargs["observed_at"]
        assert before <= ts <= after


# ── Link creation ──


class TestLinks:
    def test_awarded_by_link_created(self):
        store = _make_mock_store()
        tool = GovContractsTool(cache=None, pipeline_store=store)
        tool._persist_entities([_make_award()], "US")
        awarded_by_calls = [
            c
            for c in store.link_entities.call_args_list
            if c.kwargs.get("link_type") == "awarded_by"
        ]
        assert len(awarded_by_calls) == 1
        assert awarded_by_calls[0].kwargs["confidence"] == 1.0

    def test_operates_in_link_created(self):
        store = _make_mock_store()
        tool = GovContractsTool(cache=None, pipeline_store=store)
        tool._persist_entities([_make_award()], "US")
        operates_in_calls = [
            c
            for c in store.link_entities.call_args_list
            if c.kwargs.get("link_type") == "operates_in"
        ]
        assert len(operates_in_calls) == 1
        assert operates_in_calls[0].kwargs["confidence"] == 0.9

    def test_us_country_code(self):
        store = _make_mock_store()
        tool = GovContractsTool(cache=None, pipeline_store=store)
        tool._persist_entities([_make_award()], "US")
        country_calls = [
            c
            for c in store.register_entity.call_args_list
            if c.kwargs.get("entity_type") == "country"
        ]
        assert country_calls[0].kwargs["canonical_name"] == "US"

    def test_gb_country_code_for_uk(self):
        store = _make_mock_store()
        tool = GovContractsTool(cache=None, pipeline_store=store)
        tool._persist_entities([_make_uk_award()], "GB")
        country_calls = [
            c
            for c in store.register_entity.call_args_list
            if c.kwargs.get("entity_type") == "country"
        ]
        assert country_calls[0].kwargs["canonical_name"] == "GB"

    def test_no_agency_skips_org_registration_and_awarded_by(self):
        store = _make_mock_store()
        tool = GovContractsTool(cache=None, pipeline_store=store)
        award = _make_award(agency="")
        tool._persist_entities([award], "US")
        org_calls = [
            c
            for c in store.register_entity.call_args_list
            if c.kwargs.get("entity_type") == "organization"
        ]
        assert len(org_calls) == 0
        awarded_by = [
            c
            for c in store.link_entities.call_args_list
            if c.kwargs.get("link_type") == "awarded_by"
        ]
        assert len(awarded_by) == 0


# ── Deduplication ──


class TestDeduplication:
    def test_same_company_multiple_awards_registered_once(self):
        store = _make_mock_store()
        tool = GovContractsTool(cache=None, pipeline_store=store)
        a1 = _make_award(recipient="Raytheon", award_id="A001")
        a2 = _make_award(recipient="Raytheon", award_id="A002")
        tool._persist_entities([a1, a2], "US")
        company_calls = [
            c
            for c in store.register_entity.call_args_list
            if c.kwargs.get("entity_type") == "company"
        ]
        assert len(company_calls) == 1
        # But 2 observations
        assert store.store_entity_observation.call_count == 2

    def test_same_agency_multiple_awards_registered_once(self):
        store = _make_mock_store()
        tool = GovContractsTool(cache=None, pipeline_store=store)
        a1 = _make_award(recipient="Corp A", agency="DOD")
        a2 = _make_award(recipient="Corp B", agency="DOD")
        tool._persist_entities([a1, a2], "US")
        org_calls = [
            c
            for c in store.register_entity.call_args_list
            if c.kwargs.get("entity_type") == "organization"
        ]
        assert len(org_calls) == 1


# ── Edge cases ──


class TestEdgeCases:
    def test_empty_recipient_skipped(self):
        store = _make_mock_store()
        tool = GovContractsTool(cache=None, pipeline_store=store)
        award = _make_award(recipient="")
        tool._persist_entities([award], "US")
        # Country registered, but no company
        company_calls = [
            c
            for c in store.register_entity.call_args_list
            if c.kwargs.get("entity_type") == "company"
        ]
        assert len(company_calls) == 0

    def test_none_recipient_skipped(self):
        store = _make_mock_store()
        tool = GovContractsTool(cache=None, pipeline_store=store)
        award = _make_award()
        award["recipient"] = None
        tool._persist_entities([award], "US")
        company_calls = [
            c
            for c in store.register_entity.call_args_list
            if c.kwargs.get("entity_type") == "company"
        ]
        assert len(company_calls) == 0

    def test_persistence_exception_non_fatal(self):
        store = _make_mock_store()
        store.register_entity.side_effect = RuntimeError("DB boom")
        tool = GovContractsTool(cache=None, pipeline_store=store)
        # Should not raise
        tool._persist_entities([_make_award()], "US")

    def test_company_name_normalization(self):
        """Corp suffix should be stripped by normalize_company_name."""
        store = _make_mock_store()
        tool = GovContractsTool(cache=None, pipeline_store=store)
        a1 = _make_award(recipient="Acme Corp")
        a2 = _make_award(recipient="ACME CORP")
        tool._persist_entities([a1, a2], "US")
        company_calls = [
            c
            for c in store.register_entity.call_args_list
            if c.kwargs.get("entity_type") == "company"
        ]
        # Both should normalize to the same name → registered once
        assert len(company_calls) == 1

    def test_uk_award_with_amount_field(self):
        """UK awards use 'amount' not 'amount_usd'."""
        store = _make_mock_store()
        tool = GovContractsTool(cache=None, pipeline_store=store)
        award = _make_uk_award(amount=999_999.0)
        tool._persist_entities([award], "GB")
        obs_call = store.store_entity_observation.call_args
        val = obs_call.kwargs["value"]
        # Should pick up 'amount' since 'amount_usd' is None
        assert val["amount_usd"] == 999_999.0
