"""Collector write-correctness regressions (audit 2026-09-23, findings 3/19/31/46).

Two defects, both of the same family: a collector writing a *plausible* value
that is not the value it claims to be.

(A) ``agent/tools/gov_contracts.py`` stamped ``observed_at`` with the contract
    period-of-performance START date — a future event for any active contract.
    86% of its rows were future-dated and 912 landed beyond today, dragging the
    trainer's test window out to 2030. ``observed_at`` must be when the fact
    became KNOWN (the award/obligation date).

(B) ``agent/tools/gdelt.py`` keyed country entities on the raw CAMEO actor code
    (alpha-3 ``USA``) and named them from ``Actor1Name`` (which is why the graph
    held ALASKA, SASKATCHEWAN and TOYOTA as countries). The 2026-09-23 merge
    unified countries onto ISO alpha-2 as a one-off DB migration; without this
    fix the next GDELT run re-creates the alpha-3 entities and re-splits the
    graph. Regional blocs (EUR, MEA, SEA) are not countries and must not be
    written at all.

Every test here fails against the pre-fix code.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent.pipeline.country_codes import resolve_country_key
from agent.pipeline.entity import entity_id_from_key
from agent.tools.gdelt import GDELTTool
from agent.tools.gov_contracts import GovContractsTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store() -> MagicMock:
    store = MagicMock()
    store.register_entity.return_value = "mock_eid"
    store.add_entity_alias.return_value = None
    store.store_entity_observation.return_value = 1
    store.link_entities.return_value = 1
    return store


def _registrations(store: MagicMock, entity_type: str) -> list[Any]:
    return [c for c in store.register_entity.call_args_list if c.kwargs.get("entity_type") == entity_type]


def _make_gdelt_event(
    *,
    event_id: str = "1001",
    date: str = "20260401",
    actor1_name: str | None = "UNITED STATES",
    actor1_country: str = "USA",
    actor2_name: str | None = "CHINA",
    actor2_country: str = "CHN",
    goldstein: float | None = -9.0,
) -> dict[str, Any]:
    """A parsed GDELT event, in the shape ``_parse_events`` emits.

    Actor country codes are CAMEO alpha-3, as the real GDELT event table
    supplies them — the test fixtures that used alpha-2 here are why the
    alpha-3 split went unnoticed.
    """
    return {
        "id": event_id,
        "date": date,
        "actor1": {"name": actor1_name, "country": actor1_country, "type": "GOV"},
        "actor2": {"name": actor2_name, "country": actor2_country, "type": "GOV"},
        "event_code": "190",
        "event_root": "19",
        "event_description": "Fight",
        "quad_class": 4,
        "quad_label": "Material Conflict",
        "goldstein": goldstein,
        "num_mentions": 50,
        "num_sources": 10,
        "avg_tone": -5.2,
        "location": {"name": "South China Sea", "country": "CH", "lat": 15.0, "lon": 115.0},
        "source_url": "https://example.com/article",
    }


def _make_award(
    *,
    award_id: str = "W911QY-26-C-0001",
    recipient: str = "Lockheed Martin Corp",
    agency: str = "Department of Defense",
    award_date: str | None = "2026-01-15",
    start_date: str = "2027-06-01",
    end_date: str = "2030-01-30",
    amount_usd: float = 1_500_000.0,
) -> dict[str, Any]:
    """A gov_contracts record whose period of performance starts in the future.

    This is the live shape: USAspending's "Start Date" is the period-of-
    performance start, and for 86% of collected rows it is ahead of the
    observation date.
    """
    award: dict[str, Any] = {
        "award_id": award_id,
        "recipient": recipient,
        "amount_usd": amount_usd,
        "agency": agency,
        "sub_agency": "US Army",
        "award_type": "Contract",
        "start_date": start_date,
        "end_date": end_date,
        "description": "Test contract",
    }
    if award_date is not None:
        award["award_date"] = award_date
    return award


def _iso_to_ts(iso: str) -> float:
    from datetime import UTC, datetime

    return datetime.fromisoformat(iso).replace(tzinfo=UTC).timestamp()


# ===========================================================================
# (A) gov_contracts — observed_at is the award date, never the PoP start
# ===========================================================================


class TestGovContractsObservedAt:
    def test_future_pop_start_is_stamped_with_award_date(self) -> None:
        """The exact defect: PoP start 2027-06-01, award obligated 2026-01-15."""
        store = _make_store()
        tool = GovContractsTool(cache=None, pipeline_store=store)
        tool._persist_entities([_make_award(award_date="2026-01-15", start_date="2027-06-01")], "US")

        calls = store.store_entity_observation.call_args_list
        assert calls, "award with a valid award date must still be stored"
        for call in calls:
            ts = call.kwargs["observed_at"]
            assert ts == pytest.approx(_iso_to_ts("2026-01-15"))
            # The old code produced the PoP start; assert it explicitly so the
            # failure message names the bug.
            assert ts != pytest.approx(_iso_to_ts("2027-06-01"))

    def test_no_observation_is_future_dated(self) -> None:
        """912 live rows landed beyond today. None may, for any award."""
        store = _make_store()
        tool = GovContractsTool(cache=None, pipeline_store=store)
        awards = [
            _make_award(award_id="A1", award_date="2026-01-15", start_date="2027-06-01"),
            _make_award(award_id="A2", recipient="Raytheon", award_date="2025-11-02", start_date="2030-01-30"),
            _make_award(award_id="A3", recipient="Boeing", award_date="2024-03-09", start_date="2028-12-31"),
        ]
        tool._persist_entities(awards, "US")

        now = time.time()
        stamps = [c.kwargs["observed_at"] for c in store.store_entity_observation.call_args_list]
        assert stamps, "awards with valid award dates must be stored"
        assert all(ts <= now for ts in stamps), f"future-dated observed_at written: {stamps}"

    def test_pop_start_moves_into_the_value_payload(self) -> None:
        store = _make_store()
        tool = GovContractsTool(cache=None, pipeline_store=store)
        tool._persist_entities([_make_award(start_date="2027-06-01", end_date="2030-01-30")], "US")

        val = store.store_entity_observation.call_args_list[0].kwargs["value"]
        assert val["period_of_performance_start"] == "2027-06-01"
        assert val["period_of_performance_end"] == "2030-01-30"
        assert val["award_date"] == "2026-01-15"

    def test_future_award_date_is_dropped_not_rewritten_to_now(self) -> None:
        """A future obligation date is a data error.

        The old guard replaced it with ``datetime.now()``, turning the error
        into a plausible-looking recent observation. The record must be dropped
        instead, and the drop must be logged.
        """
        store = _make_store()
        tool = GovContractsTool(cache=None, pipeline_store=store)
        tool._persist_entities([_make_award(award_date="2029-07-04")], "US")

        assert store.store_entity_observation.call_count == 0

    def test_unparseable_award_date_is_dropped(self) -> None:
        store = _make_store()
        tool = GovContractsTool(cache=None, pipeline_store=store)
        tool._persist_entities([_make_award(award_date="not-a-date")], "US")

        assert store.store_entity_observation.call_count == 0

    def test_a_bad_record_does_not_drop_its_neighbours(self) -> None:
        store = _make_store()
        tool = GovContractsTool(cache=None, pipeline_store=store)
        tool._persist_entities(
            [
                _make_award(award_id="BAD", award_date="2029-07-04"),
                _make_award(award_id="GOOD", recipient="Raytheon", award_date="2026-02-20"),
            ],
            "US",
        )
        ids = {c.kwargs["value"]["award_id"] for c in store.store_entity_observation.call_args_list}
        assert ids == {"GOOD"}

    def test_drops_are_logged_loudly(self, caplog: pytest.LogCaptureFixture) -> None:
        store = _make_store()
        tool = GovContractsTool(cache=None, pipeline_store=store)
        with caplog.at_level("ERROR", logger="agent.tools.gov_contracts"):
            tool._persist_entities([_make_award(award_date="2029-07-04")], "US")
        assert any(r.levelname == "ERROR" for r in caplog.records), "a silent drop is the failure mode being fixed"

    def test_us_parser_maps_base_obligation_date_to_award_date(self) -> None:
        """The API row must surface an award date distinct from "Start Date"."""
        tool = GovContractsTool(cache=None)
        api_row = {
            "Award ID": "X1",
            "Recipient Name": "Acme",
            "Award Amount": 1.0,
            "Awarding Agency": "DOD",
            "Awarding Sub Agency": "Army",
            "Award Type": "A",
            "Base Obligation Date": "2026-01-15",
            "Start Date": "2027-06-01",
            "End Date": "2030-01-30",
            "Description": "d",
        }
        captured: dict[str, Any] = {}

        def _fake_post(url: str, payload: dict) -> dict:
            captured["payload"] = payload
            return {"results": [api_row], "page_metadata": {"total": 1}}

        tool._post_json = _fake_post  # type: ignore[method-assign]
        result = tool._query_awards("2026-01-01", "2026-09-23", 5, "Base Obligation Date", "desc")

        assert result.success
        award = result.data["awards"][0]
        assert award["award_date"] == "2026-01-15"
        assert award["start_date"] == "2027-06-01"
        assert "Base Obligation Date" in captured["payload"]["fields"]

    def test_recent_mode_does_not_sort_by_period_of_performance_start(self) -> None:
        """Sorting by PoP start descending returns the *furthest future* work."""
        tool = GovContractsTool(cache=None)
        seen: dict[str, Any] = {}

        def _fake_query(*args: Any, **kwargs: Any) -> Any:
            seen["sort_field"] = kwargs.get("sort_field", args[3] if len(args) > 3 else None)
            return MagicMock(success=True)

        tool._query_awards = _fake_query  # type: ignore[method-assign]
        tool._execute_us("recent", "2026-01-01", "2026-09-23", 5, {})
        assert seen["sort_field"] != "Start Date"
        assert seen["sort_field"] == "Base Obligation Date"

    def test_uk_parser_takes_the_award_date_not_the_contract_period(self) -> None:
        tool = GovContractsTool(cache=None)
        releases = [
            {
                "ocid": "ocds-b5fd17-0001",
                "date": "2026-02-01T09:00:00Z",
                "buyer": {"name": "Ministry of Defence"},
                "tender": {
                    "title": "Vehicles",
                    "description": "d",
                    "procurementMethod": "open",
                    "contractPeriod": {"startDate": "2027-05-01", "endDate": "2030-05-01"},
                },
                "awards": [
                    {
                        "date": "2026-01-20T00:00:00Z",
                        "value": {"amount": 750_000.0, "currency": "GBP"},
                        "suppliers": [{"name": "BAE Systems plc"}],
                    }
                ],
            }
        ]
        parsed = tool._parse_uk_releases(releases)
        assert len(parsed) == 1
        assert parsed[0]["award_date"] == "2026-01-20T00:00:00Z"
        assert parsed[0]["start_date"] == "2027-05-01"

    def test_uk_award_without_award_date_uses_release_publication_date(self) -> None:
        tool = GovContractsTool(cache=None)
        releases = [
            {
                "ocid": "ocds-b5fd17-0002",
                "date": "2026-02-01T09:00:00Z",
                "buyer": {"name": "Ministry of Defence"},
                "tender": {
                    "title": "Radios",
                    "procurementMethod": "open",
                    "contractPeriod": {"startDate": "2028-01-01", "endDate": "2031-01-01"},
                },
                "awards": [{"value": {"amount": 1.0, "currency": "GBP"}, "suppliers": [{"name": "Acme"}]}],
            }
        ]
        parsed = tool._parse_uk_releases(releases)
        assert parsed[0]["award_date"] == "2026-02-01T09:00:00Z"


# ===========================================================================
# (B) gdelt — country entities go through the shared resolver
# ===========================================================================


class TestGdeltCountryResolution:
    def test_usa_and_eur_batch_writes_one_us_country_and_no_eur(self) -> None:
        """The headline assertion.

        A batch with a USA actor and an EUR (regional bloc) actor must produce
        exactly one ``country:US`` entity and zero ``country:EUR`` entities.
        """
        store = _make_store()
        tool = GDELTTool(pipeline_store=store)
        events = [
            _make_gdelt_event(event_id="1", actor1_country="USA", actor2_country="CHN"),
            _make_gdelt_event(event_id="2", actor1_country="EUR", actor2_country="EUR"),
            _make_gdelt_event(event_id="3", actor1_country="USA", actor2_country="EUR"),
        ]
        tool._persist_entities_inner(events)

        country_calls = _registrations(store, "country")
        ids = [c.kwargs["entity_id"] for c in country_calls]

        us_id = entity_id_from_key("country", "US")
        assert ids.count(us_id) == 1, "US must be registered exactly once"
        assert entity_id_from_key("country", "USA") not in ids, "raw CAMEO alpha-3 key re-splits the graph"
        assert entity_id_from_key("country", "EUR") not in ids, "EUR is a regional bloc, not a country"
        assert entity_id_from_key("country", "CHN") not in ids
        assert entity_id_from_key("country", "CN") in ids

    def test_no_observation_is_written_for_a_regional_bloc(self) -> None:
        store = _make_store()
        tool = GDELTTool(pipeline_store=store)
        tool._persist_entities_inner([_make_gdelt_event(actor1_country="EUR", actor2_country="MEA")])

        assert store.register_entity.call_count == 0
        assert store.store_entity_observation.call_count == 0
        assert store.link_entities.call_count == 0

    def test_unresolved_codes_are_logged_not_silently_dropped(self, caplog: pytest.LogCaptureFixture) -> None:
        store = _make_store()
        tool = GDELTTool(pipeline_store=store)
        with caplog.at_level("WARNING", logger="agent.tools.gdelt"):
            tool._persist_entities_inner([_make_gdelt_event(actor1_country="EUR", actor2_country="MEA")])
        assert any("EUR" in r.getMessage() for r in caplog.records)

    def test_canonical_name_comes_from_the_code_not_the_actor_name(self) -> None:
        """Actor1Name is why ALASKA, SASKATCHEWAN and TOYOTA were countries."""
        store = _make_store()
        tool = GDELTTool(pipeline_store=store)
        tool._persist_entities_inner(
            [_make_gdelt_event(actor1_name="ALASKA", actor1_country="USA", actor2_name="TOYOTA", actor2_country="JPN")]
        )

        names = {c.kwargs["entity_id"]: c.kwargs["canonical_name"] for c in _registrations(store, "country")}
        assert names[entity_id_from_key("country", "US")] == "United States"
        assert names[entity_id_from_key("country", "JP")] == "Japan"
        assert "ALASKA" not in names.values()
        assert "TOYOTA" not in names.values()

    def test_alpha3_and_alpha2_inputs_collapse_to_one_entity(self) -> None:
        store = _make_store()
        tool = GDELTTool(pipeline_store=store)
        tool._persist_entities_inner(
            [
                _make_gdelt_event(event_id="1", actor1_country="USA", actor2_country="CHN"),
                _make_gdelt_event(event_id="2", actor1_country="US", actor2_country="CHN"),
            ]
        )
        ids = [c.kwargs["entity_id"] for c in _registrations(store, "country")]
        assert ids.count(entity_id_from_key("country", "US")) == 1
        # Two countries in the batch, therefore two entities — never three.
        # Pre-fix, "USA" and "US" were separate entities and this was 3.
        assert set(ids) == {entity_id_from_key("country", "US"), entity_id_from_key("country", "CN")}

    def test_observations_attach_to_the_resolved_entity(self) -> None:
        store = _make_store()
        tool = GDELTTool(pipeline_store=store)
        tool._persist_entities_inner([_make_gdelt_event(actor1_country="USA", actor2_country="CHN")])

        obs_ids = {c.kwargs["entity_id"] for c in store.store_entity_observation.call_args_list}
        assert obs_ids == {entity_id_from_key("country", "US"), entity_id_from_key("country", "CN")}

    def test_event_involves_link_uses_resolved_ids(self) -> None:
        store = _make_store()
        tool = GDELTTool(pipeline_store=store)
        tool._persist_entities_inner([_make_gdelt_event(actor1_country="USA", actor2_country="CHN")])

        link = store.link_entities.call_args_list[0].kwargs
        assert link["entity_id_a"] == entity_id_from_key("country", "US")
        assert link["entity_id_b"] == entity_id_from_key("country", "CN")

    def test_no_link_when_one_side_is_a_bloc(self) -> None:
        store = _make_store()
        tool = GDELTTool(pipeline_store=store)
        tool._persist_entities_inner([_make_gdelt_event(actor1_country="USA", actor2_country="EUR")])
        assert store.link_entities.call_count == 0

    def test_raw_cameo_code_is_kept_as_an_alias(self) -> None:
        """The alpha-3 code stays reachable — as an alias of the alpha-2 entity."""
        store = _make_store()
        tool = GDELTTool(pipeline_store=store)
        tool._persist_entities_inner([_make_gdelt_event(actor1_country="USA", actor2_country="CHN")])

        aliases = {(c.args[0], c.args[2]) for c in store.add_entity_alias.call_args_list}
        assert (entity_id_from_key("country", "US"), "USA") in aliases

    def test_events_mode_entity_ids_match_the_persisted_ids(self) -> None:
        """The ids handed to downstream consumers must be the ids in the store."""
        from unittest.mock import patch

        store = _make_store()
        tool = GDELTTool(pipeline_store=store)
        ev = _make_gdelt_event(actor1_country="USA", actor2_country="EUR")

        with (
            patch.object(tool, "_fetch_event_batches", return_value=["fake"]),
            patch.object(tool, "_parse_events", return_value=[ev]),
        ):
            result = tool.execute(mode="events", hours_back=1, quad_class="all")

        assert result.success
        out = result.data["events"][0]
        assert out["actor1"]["entity_id"] == entity_id_from_key("country", "US")
        assert "entity_id" not in out["actor2"], "no country entity exists for a regional bloc"


class TestResolverIsActuallyReached:
    """F-14: a correct resolver nobody calls is not a fix.

    ``tests/test_country_codes.py`` exercises ``resolve_country_key`` in
    isolation and passed throughout the period the graph was split. These
    assertions are about the *collector*.
    """

    def test_gdelt_imports_the_shared_resolver(self) -> None:
        import agent.tools.gdelt as gdelt_mod

        assert gdelt_mod.resolve_country_key is resolve_country_key

    def test_gdelt_source_has_no_raw_country_entity_key(self) -> None:
        """AST guard: every entity_id_from_key("country", X) in gdelt.py must
        take a resolved key, never a raw actor code."""
        import ast
        import pathlib

        src = pathlib.Path(__file__).resolve().parents[1] / "agent" / "tools" / "gdelt.py"
        tree = ast.parse(src.read_text(encoding="utf-8"))

        offenders: list[int] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Name) and func.id == "entity_id_from_key"):
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            if node.args[0].value != "country":
                continue
            key_arg = node.args[1]
            resolved = (
                isinstance(key_arg, ast.Call)
                and isinstance(key_arg.func, ast.Name)
                and key_arg.func.id == "resolve_country_key"
            ) or (isinstance(key_arg, ast.Name) and "country_key" in key_arg.id)
            if not resolved:
                offenders.append(node.lineno)

        assert not offenders, f"raw country key passed to entity_id_from_key at gdelt.py lines {offenders}"
