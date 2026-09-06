"""L2 persistence tests for creditor_filings tool."""

from __future__ import annotations

from datetime import UTC
from unittest.mock import MagicMock

from agent.tools.creditor_filings import CreditorFilingsTool

# ── Helpers ───────────────────────────────────────────────────


def _make_store() -> MagicMock:
    store = MagicMock()
    store.register_entity = MagicMock()
    store.store_entity_observation = MagicMock()
    store.link_entities = MagicMock()
    return store


def _make_tool(store: MagicMock | None = None) -> CreditorFilingsTool:
    return CreditorFilingsTool(cache=None, pipeline_store=store)


def _sec_entry(
    name: str = "Acme Corp",
    cik: str = "12345",
    date: str = "2025-06-01",
    items: list | None = None,
) -> dict:
    return {
        "company_name": name,
        "cik": cik,
        "file_date": date,
        "form": "8-K",
        "items": items or ["1.01", "2.03"],
    }


def _uk_charge(
    status: str = "outstanding",
    creditors: list | None = None,
    charge_num: int = 1,
    date: str = "2025-01-15",
) -> dict:
    return {
        "charge_number": charge_num,
        "status": status,
        "created_on": date,
        "classification": "debenture",
        "persons_entitled": creditors or ["Big Bank PLC"],
        "particulars": "All assets",
    }


# ── No store → no-op ─────────────────────────────────────────


class TestNoPersistenceWithoutStore:
    def test_no_store_no_crash(self):
        tool = _make_tool(store=None)
        tool._persist_entities([_sec_entry()])

    def test_empty_entries_no_crash(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities([])
        store.register_entity.assert_not_called()


# ── SEC EDGAR entity registration ────────────────────────────


class TestSECEntityRegistration:
    def test_single_entry_registered(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities([_sec_entry()])

        store.register_entity.assert_called_once()
        call = store.register_entity.call_args
        assert call.kwargs["entity_type"] == "company"
        assert isinstance(call.kwargs["entity_id"], str)

    def test_multiple_entries_different_companies(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities(
            [
                _sec_entry("Acme Corp", "111"),
                _sec_entry("Beta Inc", "222"),
            ]
        )

        assert store.register_entity.call_count == 2
        assert store.store_entity_observation.call_count == 2

    def test_duplicate_company_registered_once(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities(
            [
                _sec_entry("Acme Corp", "111", "2025-06-01"),
                _sec_entry("Acme Corp", "111", "2025-06-02"),
            ]
        )

        store.register_entity.assert_called_once()
        assert store.store_entity_observation.call_count == 2

    def test_missing_company_name_skipped(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities([{"company_name": "", "cik": "111", "file_date": "2025-01-01"}])

        store.register_entity.assert_not_called()


# ── SEC EDGAR observations ───────────────────────────────────


class TestSECObservations:
    def test_observation_type_is_creditor_filing(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities([_sec_entry()])

        call = store.store_entity_observation.call_args
        assert call.kwargs["observation_type"] == "creditor_filing"
        assert call.kwargs["depth_level"] == 2

    def test_stress_signal_detected(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities([_sec_entry(items=["1.01", "2.03"])])

        value = store.store_entity_observation.call_args.kwargs["value"]
        assert value["is_stress_signal"] is True

    def test_non_stress_items(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities([_sec_entry(items=["5.02"])])

        value = store.store_entity_observation.call_args.kwargs["value"]
        assert value["is_stress_signal"] is False

    def test_timestamp_parsed_from_file_date(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities([_sec_entry(date="2025-03-15")])

        from datetime import datetime

        ts = store.store_entity_observation.call_args.kwargs["observed_at"]
        expected = datetime(2025, 3, 15, tzinfo=UTC).timestamp()
        assert ts == expected


# ── UK charges + debtor-creditor links ───────────────────────


class TestUKChargesEntityLinking:
    def test_debtor_registered(self):
        store = _make_store()
        tool = _make_tool(store=store)
        company_info = {"company_name": "UK Debtor Ltd", "company_number": "12345678"}
        tool._persist_entities([], [_uk_charge()], company_info)

        # Debtor + creditor = 2 entities
        assert store.register_entity.call_count == 2

    def test_charge_observation_stored(self):
        store = _make_store()
        tool = _make_tool(store=store)
        company_info = {"company_name": "UK Debtor Ltd", "company_number": "12345678"}
        tool._persist_entities([], [_uk_charge()], company_info)

        obs_call = store.store_entity_observation.call_args
        assert obs_call.kwargs["observation_type"] == "creditor_filing"
        assert obs_call.kwargs["value"]["is_red_flag"] is True

    def test_debtor_creditor_link_created(self):
        store = _make_store()
        tool = _make_tool(store=store)
        company_info = {"company_name": "UK Debtor Ltd", "company_number": "12345678"}
        tool._persist_entities([], [_uk_charge(creditors=["Big Bank PLC"])], company_info)

        store.link_entities.assert_called_once()
        link_call = store.link_entities.call_args
        assert link_call.kwargs["link_type"] == "debtor_of"
        assert link_call.kwargs["source"] == "creditor_filings"
        assert link_call.kwargs["confidence"] == 0.8

    def test_multiple_creditors_per_charge(self):
        store = _make_store()
        tool = _make_tool(store=store)
        company_info = {"company_name": "Debtor Inc"}
        charge = _uk_charge(creditors=["Bank A", "Bank B", "Bank C"])
        tool._persist_entities([], [charge], company_info)

        assert store.link_entities.call_count == 3

    def test_self_link_prevented(self):
        """Debtor name == creditor name → no link created."""
        store = _make_store()
        tool = _make_tool(store=store)
        company_info = {"company_name": "Same Corp"}
        charge = _uk_charge(creditors=["Same Corp"])
        tool._persist_entities([], [charge], company_info)

        store.link_entities.assert_not_called()

    def test_satisfied_charge_not_red_flag(self):
        store = _make_store()
        tool = _make_tool(store=store)
        company_info = {"company_name": "UK Co"}
        charge = _uk_charge(status="fully-satisfied")
        tool._persist_entities([], [charge], company_info)

        value = store.store_entity_observation.call_args.kwargs["value"]
        assert value["is_red_flag"] is False

    def test_no_company_info_skips_charges(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities([], [_uk_charge()], None)

        # No charge observations stored (no debtor identity)
        store.store_entity_observation.assert_not_called()

    def test_empty_creditor_name_skipped(self):
        store = _make_store()
        tool = _make_tool(store=store)
        company_info = {"company_name": "Debtor Inc"}
        charge = _uk_charge(creditors=["", "  "])
        tool._persist_entities([], [charge], company_info)

        store.link_entities.assert_not_called()


# ── Exception safety ─────────────────────────────────────────


class TestExceptionSafety:
    def test_persist_exception_non_fatal(self):
        store = _make_store()
        store.register_entity.side_effect = RuntimeError("DB error")
        tool = _make_tool(store=store)

        # Should not raise
        tool._persist_entities([_sec_entry()])
