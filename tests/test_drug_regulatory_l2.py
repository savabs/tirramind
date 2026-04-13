"""L2 persistence tests for drug_regulatory tool."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.tools.drug_regulatory import DrugRegulatoryTool


# ── Helpers ───────────────────────────────────────────────────


def _make_store() -> MagicMock:
    store = MagicMock()
    store.register_entity = MagicMock()
    store.store_entity_observation = MagicMock()
    store.link_entities = MagicMock()
    return store


def _make_tool(store: MagicMock | None = None) -> DrugRegulatoryTool:
    return DrugRegulatoryTool(cache=None, pipeline_store=store)


def _approval_rec(
    sponsor: str = "Pfizer",
    app_num: str = "NDA001",
    brands: list | None = None,
    sub_type: str = "ORIG",
    sub_date: str = "20250601",
    priority: str = "STANDARD",
) -> dict:
    return {
        "sponsor": sponsor,
        "application_number": app_num,
        "brands": brands or ["Brandol"],
        "latest_submission_type": sub_type,
        "latest_submission_date": sub_date,
        "review_priority": priority,
    }


def _adverse_rec(
    drugs: list | None = None,
    reactions: list | None = None,
    serious: bool = True,
    date: str = "20250315",
) -> dict:
    return {
        "drugs": drugs or ["Aspirin"],
        "reactions": reactions or ["Nausea"],
        "serious": serious,
        "date": date,
    }


# ── No store → no-op ─────────────────────────────────────────


class TestNoPersistenceWithoutStore:
    def test_no_store_no_crash(self):
        tool = _make_tool(store=None)
        tool._persist_entities("approvals", [_approval_rec()])

    def test_empty_results_no_crash(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities("approvals", [])
        store.register_entity.assert_not_called()


# ── Approvals: company entities + drug_approval observations ──


class TestApprovalsEntityRegistration:
    def test_single_approval_registered(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities("approvals", [_approval_rec()])

        # company + US country = 2 registrations
        assert store.register_entity.call_count == 2
        company_call = store.register_entity.call_args_list[0]
        assert company_call.kwargs["entity_type"] == "company"

    def test_multiple_sponsors_deduplicated(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities(
            "approvals",
            [
                _approval_rec("Pfizer", "NDA001"),
                _approval_rec("Pfizer", "NDA002"),
                _approval_rec("Merck", "NDA003"),
            ],
        )

        # 2 companies + 1 US country = 3 registrations
        assert store.register_entity.call_count == 3
        # 3 observations (one per approval)
        assert store.store_entity_observation.call_count == 3

    def test_missing_sponsor_skipped(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities("approvals", [_approval_rec(sponsor="")])

        store.register_entity.assert_not_called()
        store.store_entity_observation.assert_not_called()


class TestApprovalsObservations:
    def test_observation_type_is_drug_approval(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities("approvals", [_approval_rec()])

        call = store.store_entity_observation.call_args
        assert call.kwargs["observation_type"] == "drug_approval"
        assert call.kwargs["depth_level"] == 2
        assert call.kwargs["source_tool"] == "drug_regulatory"

    def test_value_contains_approval_fields(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities(
            "approvals",
            [
                _approval_rec(
                    app_num="NDA999",
                    brands=["Drug A", "Drug B"],
                    sub_type="ORIG",
                    sub_date="20250601",
                    priority="PRIORITY",
                )
            ],
        )

        value = store.store_entity_observation.call_args.kwargs["value"]
        assert value["application_number"] == "NDA999"
        assert value["brand_names"] == ["Drug A", "Drug B"]
        assert value["submission_type"] == "ORIG"
        assert value["submission_date"] == "20250601"
        assert value["review_priority"] == "PRIORITY"

    def test_timestamp_parsed_from_date(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities("approvals", [_approval_rec(sub_date="20250315")])

        from datetime import datetime, timezone

        ts = store.store_entity_observation.call_args.kwargs["observed_at"]
        expected = datetime(2025, 3, 15, tzinfo=timezone.utc).timestamp()
        assert ts == expected

    def test_bad_date_falls_back(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities("approvals", [_approval_rec(sub_date="bad")])

        ts = store.store_entity_observation.call_args.kwargs["observed_at"]
        assert isinstance(ts, float)
        assert ts > 0


# ── Approvals: US market authorization link ───────────────────


class TestMarketAuthorizationLink:
    def test_link_created_for_approval(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities("approvals", [_approval_rec()])

        store.link_entities.assert_called_once()
        lc = store.link_entities.call_args
        assert lc.kwargs["link_type"] == "market_authorized_in"
        assert lc.kwargs["source"] == "drug_regulatory"
        assert lc.kwargs["confidence"] == 1.0
        assert lc.kwargs["entity_id_a"] != lc.kwargs["entity_id_b"]

    def test_us_country_entity_registered(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities("approvals", [_approval_rec()])

        # Second register_entity call is the US country
        country_call = store.register_entity.call_args_list[1]
        assert country_call.kwargs["entity_type"] == "country"
        assert country_call.kwargs["canonical_name"] == "US"

    def test_us_country_registered_once_for_multiple_sponsors(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities(
            "approvals",
            [
                _approval_rec("Pfizer"),
                _approval_rec("Merck"),
            ],
        )

        country_calls = [
            c
            for c in store.register_entity.call_args_list
            if c.kwargs.get("entity_type") == "country"
        ]
        assert len(country_calls) == 1

    def test_link_per_approval(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities(
            "approvals",
            [
                _approval_rec("Pfizer", "NDA001"),
                _approval_rec("Pfizer", "NDA002"),
            ],
        )

        # One link per approval record, not per unique company
        assert store.link_entities.call_count == 2


# ── Adverse events: drug name entities ────────────────────────


class TestAdverseEventsEntities:
    def test_drug_entity_registered(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities("adverse_events", [_adverse_rec(drugs=["Aspirin"])])

        store.register_entity.assert_called_once()
        assert store.register_entity.call_args.kwargs["entity_type"] == "company"

    def test_observation_includes_seriousness(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities(
            "adverse_events",
            [_adverse_rec(serious=True, date="20250101")],
            signals={"seriousness_ratio": 0.75},
        )

        value = store.store_entity_observation.call_args.kwargs["value"]
        assert value["serious"] is True
        assert value["seriousness_ratio"] == 0.75

    def test_multiple_drugs_per_event(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities(
            "adverse_events",
            [
                _adverse_rec(drugs=["DrugA", "DrugB"]),
            ],
        )

        assert store.register_entity.call_count == 2
        assert store.store_entity_observation.call_count == 2

    def test_question_mark_drug_skipped(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities("adverse_events", [_adverse_rec(drugs=["?"])])

        store.register_entity.assert_not_called()

    def test_empty_drug_name_skipped(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities("adverse_events", [_adverse_rec(drugs=["", "  "])])

        store.register_entity.assert_not_called()

    def test_no_market_link_for_adverse_events(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities("adverse_events", [_adverse_rec()])

        store.link_entities.assert_not_called()


# ── Labels mode: no persistence ───────────────────────────────


class TestLabelsNoPersistence:
    def test_labels_mode_skipped(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities("labels", [{"brand_name": "TestDrug"}])

        store.register_entity.assert_not_called()
        store.store_entity_observation.assert_not_called()


# ── Exception safety ─────────────────────────────────────────


class TestExceptionSafety:
    def test_persist_exception_non_fatal(self):
        store = _make_store()
        store.register_entity.side_effect = RuntimeError("DB error")
        tool = _make_tool(store=store)

        # Should not raise
        tool._persist_entities("approvals", [_approval_rec()])


# ── Execute integration (persist called from execute) ─────────


class TestExecuteCallsPersist:
    def test_approvals_mode_triggers_persist(self):
        store = _make_store()
        tool = _make_tool(store=store)

        # Mock the _fetch to return approval data
        fake_payload = {
            "meta": {"results": {"total": 1}},
            "results": [
                {
                    "application_number": "NDA123",
                    "sponsor_name": "TestCo",
                    "products": [{"brand_name": "TestBrand"}],
                    "submissions": [
                        {
                            "submission_type": "ORIG",
                            "submission_status_date": "20250601",
                            "review_priority": "PRIORITY",
                        }
                    ],
                }
            ],
        }
        with patch.object(tool, "_fetch", return_value=fake_payload):
            result = tool.execute(mode="approvals")

        assert result.success
        store.register_entity.assert_called()
        store.store_entity_observation.assert_called()
