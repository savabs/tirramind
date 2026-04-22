"""Phase 32 edge-case tests for Trade + Disease + Political L2 persistence.

Covers: guard checks (no store, no entity_id_from_key), exception safety,
empty/missing data, mode-specific entity extraction, graph builder obs types.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.models.gnn.graph_builder import ENRICHMENT_DIM, OBSERVATION_TYPES
from agent.tools.comtrade import ComtradeTool, _ISO3_TO_ISO2
from agent.tools.disease_surveillance import DiseaseSurveillanceTool
from agent.tools.political_risk import PoliticalRiskTool
from agent.tools.transport_throughput import TransportThroughputTool


# ── Helpers ──────────────────────────────────────────────────


def _store() -> MagicMock:
    s = MagicMock()
    s.register_entity = MagicMock()
    s.store_entity_observation = MagicMock(return_value=1)
    return s


# =====================================================================
# Graph builder constants
# =====================================================================


class TestGraphBuilderPhase32:
    def test_obs_types_sorted(self):
        assert OBSERVATION_TYPES == sorted(OBSERVATION_TYPES)

    def test_new_obs_types_present(self):
        for ot in (
            "border_throughput",
            "campaign_finance",
            "pathogen_level",
            "trade_flow",
        ):
            assert ot in OBSERVATION_TYPES, f"{ot} missing from OBSERVATION_TYPES"

    def test_obs_count(self):
        assert len(OBSERVATION_TYPES) == 46

    def test_enrichment_dim(self):
        assert ENRICHMENT_DIM == 9 + len(OBSERVATION_TYPES)
        assert ENRICHMENT_DIM == 55


# =====================================================================
# Comtrade L2
# =====================================================================


class TestComtradeL2:
    def test_no_store_returns_zero(self):
        tool = ComtradeTool()
        assert tool._persist_entities({"reporter": "USA"}, "flows") == {
            "trade_flow_obs": 0
        }

    @patch("agent.tools.comtrade._entity_id_from_key", None)
    def test_no_entity_id_returns_zero(self):
        tool = ComtradeTool(pipeline_store=_store())
        assert tool._persist_entities({"reporter": "USA"}, "flows") == {
            "trade_flow_obs": 0
        }

    def test_exception_caught(self):
        store = _store()
        store.register_entity.side_effect = RuntimeError("boom")
        tool = ComtradeTool(pipeline_store=store)
        result = tool._persist_entities({"reporter": "USA", "records": []}, "flows")
        assert result == {"trade_flow_obs": 0}

    def test_unknown_iso3_skipped(self):
        store = _store()
        tool = ComtradeTool(pipeline_store=store)
        result = tool._persist_entities_inner(
            {"reporter": "ZZZ", "records": []}, "flows"
        )
        assert result == {"trade_flow_obs": 0}
        assert store.register_entity.call_count == 0

    def test_empty_reporter_skipped(self):
        store = _store()
        tool = ComtradeTool(pipeline_store=store)
        result = tool._persist_entities_inner({"reporter": "", "records": []}, "flows")
        assert result == {"trade_flow_obs": 0}

    def test_valid_flows_persisted(self):
        store = _store()
        tool = ComtradeTool(pipeline_store=store)
        data = {
            "mode": "flows",
            "reporter": "USA",
            "partner": "CHN",
            "flow": "X",
            "record_count": 5,
            "records": [
                {
                    "commodity_code": "8542",
                    "trade_value_usd": 1000000,
                    "period": "2024",
                },
            ],
        }
        result = tool._persist_entities_inner(data, "flows")
        assert result == {"trade_flow_obs": 1}
        assert store.register_entity.call_count == 1
        assert store.register_entity.call_args.args[:2] == ("country", "US")
        obs_kwargs = store.store_entity_observation.call_args.kwargs
        assert obs_kwargs["observation_type"] == "trade_flow"
        assert obs_kwargs["depth_level"] == 2
        assert obs_kwargs["value"]["partner"] == "CHN"
        assert obs_kwargs["value"]["trade_value_usd"] == 1000000

    def test_commodity_mode_persisted(self):
        store = _store()
        tool = ComtradeTool(pipeline_store=store)
        data = {
            "mode": "commodity",
            "reporter": "DEU",
            "records": [{"commodity_code": "2709", "trade_value_usd": 500}],
        }
        result = tool._persist_entities_inner(data, "commodity")
        assert result == {"trade_flow_obs": 1}
        assert store.register_entity.call_args.args[1] == "DE"

    def test_partners_mode_persisted(self):
        store = _store()
        tool = ComtradeTool(pipeline_store=store)
        data = {"mode": "partners", "reporter": "JPN", "records": []}
        result = tool._persist_entities_inner(data, "partners")
        assert result == {"trade_flow_obs": 1}
        assert store.register_entity.call_args.args[1] == "JP"

    def test_no_records_still_persists(self):
        store = _store()
        tool = ComtradeTool(pipeline_store=store)
        data = {"reporter": "USA", "records": []}
        result = tool._persist_entities_inner(data, "flows")
        assert result == {"trade_flow_obs": 1}
        obs_val = store.store_entity_observation.call_args.kwargs["value"]
        assert obs_val["trade_value_usd"] is None  # empty top_record

    def test_iso3_to_iso2_mapping_coverage(self):
        """All M49 codes should have ISO2 mappings."""
        from agent.tools.comtrade import M49_CODES

        for iso3 in M49_CODES:
            assert iso3 in _ISO3_TO_ISO2, f"{iso3} missing from _ISO3_TO_ISO2"


# =====================================================================
# Transport Throughput L2
# =====================================================================


class TestTransportThroughputL2:
    def test_no_store_returns_zero(self):
        tool = TransportThroughputTool()
        assert tool._persist_entities({"records": []}, "recent") == {
            "border_throughput_obs": 0
        }

    @patch("agent.tools.transport_throughput._entity_id_from_key", None)
    def test_no_entity_id_returns_zero(self):
        tool = TransportThroughputTool(pipeline_store=_store())
        assert tool._persist_entities({"records": []}, "recent") == {
            "border_throughput_obs": 0
        }

    def test_exception_caught(self):
        store = _store()
        store.register_entity.side_effect = RuntimeError("boom")
        tool = TransportThroughputTool(pipeline_store=store)
        result = tool._persist_entities(
            {"records": [{"border": "US-Canada Border"}]}, "recent"
        )
        assert result == {"border_throughput_obs": 0}

    def test_empty_records_zero(self):
        store = _store()
        tool = TransportThroughputTool(pipeline_store=store)
        result = tool._persist_entities_inner({"records": []}, "recent")
        assert result == {"border_throughput_obs": 0}

    def test_canada_border_persists_two_countries(self):
        store = _store()
        tool = TransportThroughputTool(pipeline_store=store)
        data = {
            "records": [
                {"border": "US-Canada Border", "measure": "Trucks", "total": 100}
            ],
            "period": "2024-01",
        }
        result = tool._persist_entities_inner(data, "recent")
        assert result == {"border_throughput_obs": 2}
        registered = [c.args[1] for c in store.register_entity.call_args_list]
        assert "CA" in registered
        assert "US" in registered

    def test_mexico_border_persists_two_countries(self):
        store = _store()
        tool = TransportThroughputTool(pipeline_store=store)
        data = {"records": [{"border": "US-Mexico Border"}]}
        result = tool._persist_entities_inner(data, "recent")
        assert result == {"border_throughput_obs": 2}
        registered = [c.args[1] for c in store.register_entity.call_args_list]
        assert "MX" in registered
        assert "US" in registered

    def test_both_borders_deduplicates_us(self):
        store = _store()
        tool = TransportThroughputTool(pipeline_store=store)
        data = {
            "records": [
                {"border": "US-Canada Border"},
                {"border": "US-Mexico Border"},
            ]
        }
        result = tool._persist_entities_inner(data, "recent")
        # US, CA, MX = 3 unique countries
        assert result == {"border_throughput_obs": 3}

    def test_unknown_border_skipped(self):
        store = _store()
        tool = TransportThroughputTool(pipeline_store=store)
        data = {"records": [{"border": "Unknown Border"}]}
        result = tool._persist_entities_inner(data, "recent")
        assert result == {"border_throughput_obs": 0}

    def test_series_key_used_for_trend(self):
        store = _store()
        tool = TransportThroughputTool(pipeline_store=store)
        data = {
            "series": [
                {"border": "US-Canada Border", "date": "2024-01-01", "total": 500},
            ]
        }
        result = tool._persist_entities_inner(data, "trend")
        assert result == {"border_throughput_obs": 2}

    def test_comparison_key_used_for_compare(self):
        store = _store()
        tool = TransportThroughputTool(pipeline_store=store)
        data = {"comparison": [{"date": "2024-01", "canada": 100, "mexico": 200}]}
        # comparison records don't have "border" key → no countries found
        result = tool._persist_entities_inner(data, "compare")
        assert result == {"border_throughput_obs": 0}

    def test_ports_key_used(self):
        store = _store()
        tool = TransportThroughputTool(pipeline_store=store)
        data = {
            "ports": [
                {"border": "US-Mexico Border", "port": "Laredo", "value": 999},
            ]
        }
        result = tool._persist_entities_inner(data, "port")
        assert result == {"border_throughput_obs": 2}


# =====================================================================
# Disease Surveillance L2
# =====================================================================


class TestDiseaseSurveillanceL2:
    def test_no_store_returns_zero(self):
        tool = DiseaseSurveillanceTool()
        assert tool._persist_entities({}, "wastewater") == {"pathogen_level_obs": 0}

    @patch("agent.tools.disease_surveillance._entity_id_from_key", None)
    def test_no_entity_id_returns_zero(self):
        tool = DiseaseSurveillanceTool(pipeline_store=_store())
        assert tool._persist_entities({}, "wastewater") == {"pathogen_level_obs": 0}

    def test_exception_caught(self):
        store = _store()
        store.register_entity.side_effect = RuntimeError("boom")
        tool = DiseaseSurveillanceTool(pipeline_store=store)
        result = tool._persist_entities(
            {"pathogen": "covid", "total_samples": 100}, "wastewater"
        )
        assert result == {"pathogen_level_obs": 0}

    # ── wastewater ──

    def test_wastewater_persists_to_us(self):
        store = _store()
        tool = DiseaseSurveillanceTool(pipeline_store=store)
        data = {
            "pathogen": "sars-cov-2",
            "total_samples": 500,
            "states_count": 10,
            "hot_states": 3,
        }
        result = tool._persist_entities_inner(data, "wastewater")
        assert result == {"pathogen_level_obs": 1}
        assert store.register_entity.call_args.args[:2] == ("country", "US")
        obs = store.store_entity_observation.call_args.kwargs
        assert obs["observation_type"] == "pathogen_level"
        assert obs["value"]["mode"] == "wastewater"
        assert obs["value"]["total_samples"] == 500

    # ── outbreaks ──

    def test_outbreaks_extracts_countries(self):
        store = _store()
        tool = DiseaseSurveillanceTool(pipeline_store=store)
        data = {
            "entries": [
                {"country_parsed": "Bangladesh", "disease_parsed": "Nipah"},
                {"country_parsed": "United States", "disease_parsed": "Mpox"},
                {"country_parsed": "Bangladesh", "disease_parsed": "Cholera"},
            ],
            "count": 3,
        }
        result = tool._persist_entities_inner(data, "outbreaks")
        # "Ba" and "Un" — 2 unique country prefixes
        assert result == {"pathogen_level_obs": 2}
        assert store.register_entity.call_count == 2

    def test_outbreaks_empty_country_skipped(self):
        store = _store()
        tool = DiseaseSurveillanceTool(pipeline_store=store)
        data = {
            "entries": [
                {"country_parsed": "", "disease_parsed": "Unknown"},
                {"country_parsed": "  ", "disease_parsed": "Unknown"},
            ]
        }
        result = tool._persist_entities_inner(data, "outbreaks")
        assert result == {"pathogen_level_obs": 0}

    def test_outbreaks_single_char_country_skipped(self):
        store = _store()
        tool = DiseaseSurveillanceTool(pipeline_store=store)
        data = {"entries": [{"country_parsed": "X", "disease_parsed": "test"}]}
        result = tool._persist_entities_inner(data, "outbreaks")
        assert result == {"pathogen_level_obs": 0}

    # ── eu_surveillance ──

    def test_eu_surveillance_extracts_country_codes(self):
        store = _store()
        tool = DiseaseSurveillanceTool(pipeline_store=store)
        data = {
            "records": [
                {"country_code": "DE", "year_week": "2024-W01"},
                {"country_code": "FR", "year_week": "2024-W01"},
                {"country_code": "DE", "year_week": "2024-W02"},
            ]
        }
        result = tool._persist_entities_inner(data, "eu_surveillance")
        assert result == {"pathogen_level_obs": 2}
        registered = sorted(c.args[1] for c in store.register_entity.call_args_list)
        assert registered == ["DE", "FR"]

    def test_eu_surveillance_empty_country_skipped(self):
        store = _store()
        tool = DiseaseSurveillanceTool(pipeline_store=store)
        data = {"records": [{"country_code": ""}, {"country": ""}]}
        result = tool._persist_entities_inner(data, "eu_surveillance")
        assert result == {"pathogen_level_obs": 0}

    def test_eu_surveillance_3char_country_skipped(self):
        """ISO-2 only — skip 3-char codes."""
        store = _store()
        tool = DiseaseSurveillanceTool(pipeline_store=store)
        data = {"records": [{"country_code": "DEU"}]}
        result = tool._persist_entities_inner(data, "eu_surveillance")
        assert result == {"pathogen_level_obs": 0}

    # ── genomics ──

    def test_genomics_not_persisted(self):
        """Genomics mode has no country dimension — skipped in execute()."""
        store = _store()
        tool = DiseaseSurveillanceTool(pipeline_store=store)
        # Genomics never reaches _persist_entities — it returns directly
        # But if someone calls _persist_entities_inner, it should return 0
        result = tool._persist_entities_inner(
            {"organism": "SARS-CoV-2", "signal": "STABLE"}, "genomics"
        )
        assert result == {"pathogen_level_obs": 0}

    def test_no_entries_in_data(self):
        store = _store()
        tool = DiseaseSurveillanceTool(pipeline_store=store)
        result = tool._persist_entities_inner({}, "outbreaks")
        assert result == {"pathogen_level_obs": 0}


# =====================================================================
# Political Risk L2
# =====================================================================


class TestPoliticalRiskL2:
    def test_no_store_returns_zero(self):
        tool = PoliticalRiskTool()
        assert tool._persist_entities({"records": []}, "candidates") == {
            "campaign_finance_obs": 0
        }

    @patch("agent.tools.political_risk._entity_id_from_key", None)
    def test_no_entity_id_returns_zero(self):
        tool = PoliticalRiskTool(pipeline_store=_store())
        assert tool._persist_entities({"records": []}, "candidates") == {
            "campaign_finance_obs": 0
        }

    def test_exception_caught(self):
        store = _store()
        store.register_entity.side_effect = RuntimeError("boom")
        tool = PoliticalRiskTool(pipeline_store=store)
        result = tool._persist_entities(
            {"records": [{"candidate_id": "P00000001"}], "result_type": "candidates"},
            "candidates",
        )
        assert result == {"campaign_finance_obs": 0}

    # ── filings skipped ──

    def test_filings_skipped(self):
        store = _store()
        tool = PoliticalRiskTool(pipeline_store=store)
        result = tool._persist_entities_inner(
            {"records": [{"committee_id": "C00703975"}], "result_type": "filings"},
            "filings",
        )
        assert result == {"campaign_finance_obs": 0}
        assert store.register_entity.call_count == 0

    # ── candidates ──

    def test_candidates_persisted(self):
        store = _store()
        tool = PoliticalRiskTool(pipeline_store=store)
        data = {
            "result_type": "candidates",
            "records": [
                {
                    "candidate_id": "P00000001",
                    "name": "Test Candidate",
                    "party": "DEM",
                    "office": "P",
                    "state": "US",
                    "has_raised_funds": True,
                    "candidate_status": "C",
                },
            ],
        }
        result = tool._persist_entities_inner(data, "candidates")
        assert result == {"campaign_finance_obs": 1}
        assert store.register_entity.call_args.args[:2] == ("person", "P00000001")
        obs = store.store_entity_observation.call_args.kwargs
        assert obs["observation_type"] == "campaign_finance"
        assert obs["value"]["party"] == "DEM"
        assert obs["depth_level"] == 2

    def test_candidates_empty_id_skipped(self):
        store = _store()
        tool = PoliticalRiskTool(pipeline_store=store)
        data = {
            "result_type": "candidates",
            "records": [{"candidate_id": "", "name": "No ID"}],
        }
        result = tool._persist_entities_inner(data, "candidates")
        assert result == {"campaign_finance_obs": 0}

    def test_multiple_candidates(self):
        store = _store()
        tool = PoliticalRiskTool(pipeline_store=store)
        data = {
            "result_type": "candidates",
            "records": [
                {"candidate_id": "P00000001", "name": "A", "party": "DEM"},
                {"candidate_id": "P00000002", "name": "B", "party": "REP"},
                {"candidate_id": "", "name": "C"},  # skipped
            ],
        }
        result = tool._persist_entities_inner(data, "candidates")
        assert result == {"campaign_finance_obs": 2}

    # ── expenditures ──

    def test_expenditures_aggregated_per_candidate(self):
        store = _store()
        tool = PoliticalRiskTool(pipeline_store=store)
        data = {
            "result_type": "expenditures",
            "records": [
                {
                    "candidate_id": "P00000001",
                    "candidate_name": "Test",
                    "support_oppose": "S",
                    "expenditure_amount": 5000,
                },
                {
                    "candidate_id": "P00000001",
                    "candidate_name": "Test",
                    "support_oppose": "O",
                    "expenditure_amount": 3000,
                },
                {
                    "candidate_id": "P00000002",
                    "candidate_name": "Other",
                    "support_oppose": "S",
                    "expenditure_amount": 1000,
                },
            ],
        }
        result = tool._persist_entities_inner(data, "expenditures")
        assert result == {"campaign_finance_obs": 2}
        # Check the aggregated values for candidate P00000001
        calls = store.store_entity_observation.call_args_list
        for call in calls:
            val = call.kwargs["value"]
            if val.get("name") == "Test":
                assert val["total_support"] == 5000.0
                assert val["total_oppose"] == 3000.0
                assert val["total_spent"] == 8000.0

    def test_expenditures_empty_candidate_id_skipped(self):
        store = _store()
        tool = PoliticalRiskTool(pipeline_store=store)
        data = {
            "result_type": "expenditures",
            "records": [
                {"candidate_id": "", "support_oppose": "S", "expenditure_amount": 100},
            ],
        }
        result = tool._persist_entities_inner(data, "expenditures")
        assert result == {"campaign_finance_obs": 0}

    def test_expenditures_none_amount_treated_as_zero(self):
        store = _store()
        tool = PoliticalRiskTool(pipeline_store=store)
        data = {
            "result_type": "expenditures",
            "records": [
                {
                    "candidate_id": "P00000001",
                    "candidate_name": "X",
                    "support_oppose": "S",
                    "expenditure_amount": None,
                },
            ],
        }
        result = tool._persist_entities_inner(data, "expenditures")
        assert result == {"campaign_finance_obs": 1}
        val = store.store_entity_observation.call_args.kwargs["value"]
        assert val["total_spent"] == 0.0

    def test_empty_records(self):
        store = _store()
        tool = PoliticalRiskTool(pipeline_store=store)
        result = tool._persist_entities_inner(
            {"records": [], "result_type": "candidates"}, "candidates"
        )
        assert result == {"campaign_finance_obs": 0}
