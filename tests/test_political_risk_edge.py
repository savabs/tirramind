"""
Edge case tests for PoliticalRiskTool (FEC Campaign Finance API).

Covers: mode validation, cycle validation, office validation, FEC API fetch,
response parsing (candidates/filings/expenditures), signal computation
(party breakdown, cash on hand, support/oppose ratio, top targets), cache
interaction, HTTP errors (429/422/500/timeout), empty data, malformed
responses, API key handling, output formatting, tool metadata, registry +
bandit integration.
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.tools.political_risk import (
    _FEC_BASE,
    VALID_MODES,
    VALID_OFFICES,
    PoliticalRiskTool,
    _compute_signals,
    _format_summary,
    _get_api_key,
    _parse_candidates,
    _parse_expenditures,
    _parse_filings,
)

# ── Fixtures ──────────────────────────────────────────────────


def _tool(cache=None) -> PoliticalRiskTool:
    return PoliticalRiskTool(cache=cache)


def _mock_resp(body: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        text=json.dumps(body),
        request=httpx.Request("GET", "http://test"),
    )


def _fec_response(results: list, total: int = 0) -> dict:
    return {
        "results": results,
        "pagination": {"count": total or len(results), "page": 1, "pages": 1},
    }


def _candidate(
    name: str = "SMITH, JOHN",
    party: str = "DEM",
    office: str = "P",
    state: str = "",
    has_raised: bool = True,
) -> dict:
    return {
        "candidate_id": "P00000001",
        "name": name,
        "party": party,
        "office": office,
        "office_full": "President" if office == "P" else "Senate",
        "state": state,
        "district": "",
        "incumbent_challenge": "C",
        "cycles": [2024],
        "has_raised_funds": has_raised,
        "candidate_status": "C",
    }


def _filing(
    committee_name: str = "SMITH FOR PRESIDENT",
    cash: float | None = 5000000,
    receipts: float | None = 2000000,
) -> dict:
    return {
        "committee_id": "C00703975",
        "committee_name": committee_name,
        "form_type": "F3P",
        "receipt_date": "2024-07-15",
        "coverage_start_date": "2024-04-01",
        "coverage_end_date": "2024-06-30",
        "total_receipts": receipts,
        "total_disbursements": 1500000,
        "cash_on_hand_end_period": cash,
        "debts_owed_by_committee": 100000,
        "document_description": "Quarterly report",
    }


def _expenditure(
    candidate: str = "SMITH, JOHN",
    support_oppose: str = "S",
    amount: float = 50000,
    committee_name: str = "AMERICANS FOR FREEDOM PAC",
) -> dict:
    return {
        "committee_id": "C00800001",
        "committee": {"name": committee_name},
        "candidate_id": "P00000001",
        "candidate_name": candidate,
        "support_oppose_indicator": support_oppose,
        "expenditure_amount": amount,
        "expenditure_date": "2024-10-15",
        "payee_name": "MEDIA CONSULTING LLC",
        "expenditure_description": "TV AD",
        "candidate_office": "P",
        "candidate_office_state": "",
    }


# ── TestToolMetadata ──────────────────────────────────────────


class TestToolMetadata:
    def test_name(self):
        assert _tool().name == "political_risk"

    def test_description_mentions_fec(self):
        assert "FEC" in _tool().description

    def test_parameters_mode_required(self):
        assert "mode" in _tool().parameters["required"]

    def test_parameters_office_enum(self):
        enum = _tool().parameters["properties"]["office"]["enum"]
        assert set(enum) == VALID_OFFICES


# ── TestInputValidation ───────────────────────────────────────


class TestInputValidation:
    def test_invalid_mode(self):
        r = _tool().execute(mode="invalid")
        assert not r.success
        assert "Invalid mode" in r.output

    def test_empty_mode(self):
        r = _tool().execute(mode="")
        assert not r.success

    def test_odd_cycle_rejected(self):
        r = _tool().execute(mode="candidates", cycle=2023)
        assert not r.success
        assert "even year" in r.output

    def test_even_cycle_accepted(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.return_value = _mock_resp(_fec_response([_candidate()]))
            r = _tool().execute(mode="candidates", cycle=2024)
        assert r.success

    def test_invalid_office_in_candidates(self):
        with patch("httpx.Client"):
            r = _tool().execute(mode="candidates", office="X")
        assert not r.success
        assert "Invalid office" in r.output

    def test_limit_clamped_to_100(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.return_value = _mock_resp(_fec_response([_candidate()]))
            r = _tool().execute(mode="candidates", limit=500)
        assert r.success
        call_params = mc.return_value.get.call_args[1]["params"]
        assert call_params["per_page"] == "100"  # clamped from 500 to 100


# ── TestAPIKeyHandling ────────────────────────────────────────


class TestAPIKeyHandling:
    def test_default_is_demo_key(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove TIRRA_FEC_API_KEY if present
            os.environ.pop("TIRRA_FEC_API_KEY", None)
            assert _get_api_key() == "DEMO_KEY"

    def test_custom_key_from_env(self):
        with patch.dict(os.environ, {"TIRRA_FEC_API_KEY": "MY_KEY"}):
            assert _get_api_key() == "MY_KEY"

    def test_api_key_passed_in_params(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.return_value = _mock_resp(_fec_response([_candidate()]))
            _tool().execute(mode="candidates")
        call_params = mc.return_value.get.call_args[1]["params"]
        assert "api_key" in call_params


# ── TestCandidateParsing ──────────────────────────────────────


class TestCandidateParsing:
    def test_parse_basic(self):
        records = _parse_candidates([_candidate()])
        assert len(records) == 1
        assert records[0]["name"] == "SMITH, JOHN"
        assert records[0]["party"] == "DEM"
        assert records[0]["has_raised_funds"] is True

    def test_parse_empty(self):
        assert _parse_candidates([]) == []

    def test_parse_missing_fields(self):
        records = _parse_candidates([{}])
        assert records[0]["name"] == ""
        assert records[0]["party"] == ""
        assert records[0]["has_raised_funds"] is False

    def test_parse_multiple(self):
        records = _parse_candidates(
            [
                _candidate(name="A", party="DEM"),
                _candidate(name="B", party="REP"),
            ]
        )
        assert len(records) == 2
        assert records[0]["name"] == "A"
        assert records[1]["party"] == "REP"


# ── TestFilingParsing ─────────────────────────────────────────


class TestFilingParsing:
    def test_parse_basic(self):
        records = _parse_filings([_filing()])
        assert len(records) == 1
        assert records[0]["cash_on_hand_end"] == 5000000
        assert records[0]["total_receipts"] == 2000000

    def test_parse_null_cash(self):
        records = _parse_filings([_filing(cash=None)])
        assert records[0]["cash_on_hand_end"] is None

    def test_parse_empty(self):
        assert _parse_filings([]) == []


# ── TestExpenditureParsing ────────────────────────────────────


class TestExpenditureParsing:
    def test_parse_basic(self):
        records = _parse_expenditures([_expenditure()])
        assert records[0]["support_oppose"] == "S"
        assert records[0]["expenditure_amount"] == 50000
        assert records[0]["committee_name"] == "AMERICANS FOR FREEDOM PAC"

    def test_parse_committee_as_string(self):
        """When committee is not a dict, fall back to committee_name."""
        raw = _expenditure()
        del raw["committee"]
        raw["committee_name"] = "FLAT PAC"
        records = _parse_expenditures([raw])
        assert records[0]["committee_name"] == "FLAT PAC"

    def test_parse_oppose(self):
        records = _parse_expenditures([_expenditure(support_oppose="O")])
        assert records[0]["support_oppose"] == "O"

    def test_parse_empty(self):
        assert _parse_expenditures([]) == []


# ── TestSignalComputation ─────────────────────────────────────


class TestSignalComputation:
    def test_empty_records(self):
        assert _compute_signals([], "candidates") == {}

    def test_candidate_party_breakdown(self):
        records = [
            {"party": "DEM", "office_full": "President", "has_raised_funds": True},
            {"party": "DEM", "office_full": "Senate", "has_raised_funds": False},
            {"party": "REP", "office_full": "President", "has_raised_funds": True},
        ]
        signals = _compute_signals(records, "candidates")
        assert signals["party_breakdown"] == {"DEM": 2, "REP": 1}
        assert signals["active_fundraisers"] == 2

    def test_candidate_office_breakdown(self):
        records = [
            {"party": "DEM", "office_full": "President", "has_raised_funds": True},
            {"party": "REP", "office_full": "President", "has_raised_funds": True},
        ]
        signals = _compute_signals(records, "candidates")
        assert signals["office_breakdown"]["President"] == 2

    def test_filing_avg_cash(self):
        records = [
            {"cash_on_hand_end": 1000000, "total_receipts": 500000},
            {"cash_on_hand_end": 3000000, "total_receipts": 700000},
        ]
        signals = _compute_signals(records, "filings")
        assert signals["avg_cash_on_hand"] == 2000000.0
        assert signals["total_receipts_sum"] == 1200000.0

    def test_filing_null_cash_excluded_from_avg(self):
        records = [
            {"cash_on_hand_end": None, "total_receipts": 500000},
            {"cash_on_hand_end": 2000000, "total_receipts": None},
        ]
        signals = _compute_signals(records, "filings")
        assert signals["avg_cash_on_hand"] == 2000000.0

    def test_expenditure_support_oppose(self):
        records = [
            {"support_oppose": "S", "expenditure_amount": 100000, "candidate_name": "A"},
            {"support_oppose": "O", "expenditure_amount": 300000, "candidate_name": "A"},
            {"support_oppose": "S", "expenditure_amount": 50000, "candidate_name": "B"},
        ]
        signals = _compute_signals(records, "expenditures")
        assert signals["support_total"] == 150000.0
        assert signals["oppose_total"] == 300000.0
        assert signals["support_count"] == 2
        assert signals["oppose_count"] == 1
        assert signals["oppose_ratio"] == pytest.approx(300000 / 450000, rel=0.01)

    def test_expenditure_top_targets(self):
        records = [
            {"support_oppose": "S", "expenditure_amount": 100000, "candidate_name": "A"},
            {"support_oppose": "O", "expenditure_amount": 200000, "candidate_name": "B"},
        ]
        signals = _compute_signals(records, "expenditures")
        assert len(signals["top_targets"]) == 2
        assert signals["top_targets"][0]["candidate"] == "B"  # highest

    def test_expenditure_null_amounts(self):
        records = [
            {"support_oppose": "S", "expenditure_amount": None, "candidate_name": "A"},
        ]
        signals = _compute_signals(records, "expenditures")
        assert signals["support_total"] == 0

    def test_expenditure_unknown_support_oppose(self):
        records = [
            {"support_oppose": "?", "expenditure_amount": 5000, "candidate_name": "A"},
        ]
        signals = _compute_signals(records, "expenditures")
        assert signals["support_total"] == 0
        assert signals["oppose_total"] == 0


# ── TestOutputFormatting ──────────────────────────────────────


class TestOutputFormatting:
    def test_candidates_format(self):
        records = [
            {
                "name": "SMITH",
                "party": "DEM",
                "office_full": "President",
                "office": "P",
                "state": "",
                "district": "",
                "has_raised_funds": True,
            }
        ]
        signals = {"party_breakdown": {"DEM": 1}, "active_fundraisers": 1}
        out = _format_summary(records, signals, "candidates", 1)
        assert "SMITH" in out
        assert "DEM" in out
        assert "fundraising" in out

    def test_filings_format(self):
        records = [
            {
                "committee_name": "TEST COMMITTEE",
                "form_type": "F3P",
                "receipt_date": "2024-07-15",
                "cash_on_hand_end": 5000000,
            }
        ]
        signals = {"avg_cash_on_hand": 5000000}
        out = _format_summary(records, signals, "filings", 1)
        assert "TEST COMMITTEE" in out
        assert "$5,000,000" in out

    def test_expenditures_format(self):
        records = [
            {
                "support_oppose": "O",
                "candidate_name": "JONES",
                "expenditure_amount": 100000,
                "committee_name": "PAC X",
                "expenditure_date": "2024-10-01",
            }
        ]
        signals = {
            "support_total": 0,
            "oppose_total": 100000,
            "support_count": 0,
            "oppose_count": 1,
            "oppose_ratio": 1.0,
            "top_targets": [{"candidate": "JONES", "total_spent": 100000}],
        }
        out = _format_summary(records, signals, "expenditures", 1)
        assert "OPPOSE" in out
        assert "JONES" in out

    def test_empty_results(self):
        out = _format_summary([], {}, "candidates", 0)
        assert "0 returned" in out


# ── TestHTTPErrors ────────────────────────────────────────────


class TestHTTPErrors:
    def test_timeout(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.side_effect = httpx.TimeoutException("t")
            r = _tool().execute(mode="candidates")
        assert not r.success
        assert "timed out" in r.output

    def test_generic_http_error(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.side_effect = httpx.HTTPError("e")
            r = _tool().execute(mode="candidates")
        assert not r.success
        assert "HTTP error" in r.output

    def test_rate_limit_429(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.return_value = _mock_resp({}, 429)
            r = _tool().execute(mode="candidates")
        assert not r.success
        assert "rate limit" in r.output.lower()

    def test_validation_422(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.return_value = _mock_resp({}, 422)
            r = _tool().execute(mode="candidates")
        assert not r.success
        assert "validation" in r.output.lower()

    def test_server_error_500(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.return_value = _mock_resp({}, 500)
            r = _tool().execute(mode="candidates")
        assert not r.success
        assert "500" in r.output

    def test_malformed_json(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            resp = httpx.Response(
                status_code=200,
                text="<!DOCTYPE html>",
                request=httpx.Request("GET", "http://test"),
            )
            mc.return_value.get.return_value = resp
            r = _tool().execute(mode="candidates")
        assert not r.success
        assert "parse" in r.output.lower()


# ── TestCache ─────────────────────────────────────────────────


class TestCache:
    def test_cache_hit(self):
        cache = MagicMock()
        cache.get.return_value = {
            "output": "cached",
            "data": {"records": [], "cached": True},
        }
        r = _tool(cache=cache).execute(mode="candidates")
        assert r.success
        assert r.output == "cached"

    def test_cache_miss_stores_result(self):
        cache = MagicMock()
        cache.get.return_value = None
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.return_value = _mock_resp(_fec_response([_candidate()]))
            r = _tool(cache=cache).execute(mode="candidates")
        assert r.success
        cache.put.assert_called_once()

    def test_no_cache_works(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.return_value = _mock_resp(_fec_response([_candidate()]))
            r = _tool(cache=None).execute(mode="candidates")
        assert r.success


# ── TestConstants ─────────────────────────────────────────────


class TestConstants:
    def test_valid_modes(self):
        assert {"candidates", "filings", "expenditures"} == VALID_MODES

    def test_valid_offices(self):
        assert {"P", "S", "H"} == VALID_OFFICES

    def test_fec_base_url(self):
        assert _FEC_BASE.startswith("https://api.open.fec.gov")


# ── TestRegistryAndBandit ─────────────────────────────────────


class TestRegistryAndBandit:
    def test_tool_in_cli_registry(self):
        from agent.cli import build_tool_registry

        registry = build_tool_registry()
        assert "political_risk" in registry.list_names()

    def test_bandit_arm_exists(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm_names = [a.name for a in DEFAULT_ARMS]
        assert "political_risk_monitor" in arm_names

    def test_bandit_arm_references_tool(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm = next(a for a in DEFAULT_ARMS if a.name == "political_risk_monitor")
        assert "political_risk" in arm.tools


# ── TestCandidatesMode ────────────────────────────────────────


class TestCandidatesMode:
    def test_search_by_name(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.return_value = _mock_resp(_fec_response([_candidate(name="BIDEN")]))
            r = _tool().execute(mode="candidates", query="BIDEN")
        assert r.success
        call_params = mc.return_value.get.call_args[1]["params"]
        assert call_params["q"] == "BIDEN"

    def test_filter_by_office(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.return_value = _mock_resp(_fec_response([_candidate(office="S")]))
            r = _tool().execute(mode="candidates", office="S")
        assert r.success

    def test_filter_by_cycle(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.return_value = _mock_resp(_fec_response([_candidate()]))
            r = _tool().execute(mode="candidates", cycle=2024)
        assert r.success
        call_params = mc.return_value.get.call_args[1]["params"]
        assert call_params["election_year"] == "2024"


# ── TestFilingsMode ───────────────────────────────────────────


class TestFilingsMode:
    def test_filings_by_committee(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.return_value = _mock_resp(_fec_response([_filing()]))
            r = _tool().execute(mode="filings", query="C00703975")
        assert r.success
        call_params = mc.return_value.get.call_args[1]["params"]
        assert call_params["committee_id"] == "C00703975"

    def test_filings_sort_asc(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.return_value = _mock_resp(_fec_response([_filing()]))
            r = _tool().execute(mode="filings", sort_order="asc")
        call_params = mc.return_value.get.call_args[1]["params"]
        assert call_params["sort"] == "receipt_date"


# ── TestExpendituresMode ─────────────────────────────────────


class TestExpendituresMode:
    def test_expenditures_success(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.return_value = _mock_resp(_fec_response([_expenditure()]))
            r = _tool().execute(mode="expenditures")
        assert r.success
        assert r.data["result_type"] == "expenditures"

    def test_expenditures_by_candidate(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.return_value = _mock_resp(_fec_response([_expenditure()]))
            r = _tool().execute(mode="expenditures", query="P00000001")
        call_params = mc.return_value.get.call_args[1]["params"]
        assert call_params["candidate_id"] == "P00000001"


# ── TestEdgeCombinations ──────────────────────────────────────


class TestEdgeCombinations:
    def test_empty_results(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.return_value = _mock_resp(_fec_response([], total=0))
            r = _tool().execute(mode="candidates")
        assert r.success
        assert r.data["count"] == 0

    def test_no_query_candidates(self):
        """Candidates mode without query should still work."""
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.return_value = _mock_resp(_fec_response([_candidate()]))
            r = _tool().execute(mode="candidates")
        assert r.success
        call_params = mc.return_value.get.call_args[1]["params"]
        assert "q" not in call_params

    def test_all_modes_return_correct_type(self):
        for mode, fixture in [
            ("candidates", _candidate()),
            ("filings", _filing()),
            ("expenditures", _expenditure()),
        ]:
            with patch("httpx.Client") as mc:
                mc.return_value.__enter__ = lambda s: s
                mc.return_value.__exit__ = MagicMock(return_value=False)
                mc.return_value.get.return_value = _mock_resp(_fec_response([fixture]))
                r = _tool().execute(mode=mode)
            assert r.success
            assert r.data["result_type"] == mode

    def test_missing_pagination(self):
        """Response without pagination should still work."""
        body = {"results": [_candidate()]}
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.return_value = _mock_resp(body)
            r = _tool().execute(mode="candidates")
        assert r.success
