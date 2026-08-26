"""
Edge case tests for GovContractsTool (USASpending.gov — US federal contracts).

Covers: mode validation, required params, date defaults, limit clamping,
recent/top/agency/search modes, payload construction, HTTP errors, timeout,
empty responses, malformed data, output formatting, registry + bandit.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.tools.gov_contracts import (
    _AWARDS_URL,
    _CONTRACT_CODES,
    _FIELDS,
    _UK_OCDS_URL,
    VALID_MODES,
    VALID_REGIONS,
    GovContractsTool,
)

# ── Fixtures ──────────────────────────────────────────────────


def _tool(cache=None) -> GovContractsTool:
    return GovContractsTool(cache=cache)


SAMPLE_AWARDS_RESPONSE = {
    "spending_level": "awards",
    "limit": 20,
    "results": [
        {
            "Award ID": "HT940216C0001",
            "Recipient Name": "ACME DEFENSE INC",
            "Award Amount": 500_000_000,
            "Total Outlays": 300_000_000,
            "Awarding Agency": "Department of Defense",
            "Awarding Sub Agency": "Army",
            "Award Type": "Contract",
            "Start Date": "2024-06-01",
            "End Date": "2026-06-01",
            "Description": "Tactical vehicle maintenance and parts supply for CONUS operations",
        },
        {
            "Award ID": "75N91024C0042",
            "Recipient Name": "PHARMA INNOVATIONS LLC",
            "Award Amount": 100_000_000,
            "Total Outlays": 50_000_000,
            "Awarding Agency": "Department of Health and Human Services",
            "Awarding Sub Agency": "NIH",
            "Award Type": "Contract",
            "Start Date": "2024-03-15",
            "End Date": "2025-03-15",
            "Description": "Clinical trial support for vaccine development program",
        },
    ],
    "page_metadata": {
        "total": 2,
        "page": 1,
        "limit": 20,
        "hasNext": False,
    },
}


# ── 1. Tool Metadata ─────────────────────────────────────────


class TestToolMetadata:
    def test_name(self):
        assert _tool().name == "gov_contracts"

    def test_description_nonempty(self):
        assert len(_tool().description) > 50

    def test_parameters_schema(self):
        params = _tool().parameters
        assert params["type"] == "object"
        props = params["properties"]
        assert "mode" in props
        assert "agency" in props
        assert "query" in props
        assert "start_date" in props
        assert "end_date" in props
        assert "limit" in props

    def test_mode_enum(self):
        modes = _tool().parameters["properties"]["mode"]["enum"]
        assert set(modes) == {"recent", "top", "agency", "search"}

    def test_required_fields(self):
        assert _tool().parameters["required"] == ["mode"]


# ── 2. Input Validation ──────────────────────────────────────


class TestInputValidation:
    def test_invalid_mode(self):
        r = _tool().execute(mode="invalid")
        assert not r.success
        assert "Invalid mode" in r.output

    def test_empty_mode(self):
        r = _tool().execute(mode="")
        assert not r.success

    def test_no_mode(self):
        r = _tool().execute()
        assert not r.success

    def test_mode_case_sensitive(self):
        r = _tool().execute(mode="RECENT")
        assert not r.success

    def test_agency_mode_requires_agency(self):
        r = _tool().execute(mode="agency")
        assert not r.success
        assert "agency" in r.output.lower()

    def test_agency_mode_empty_agency(self):
        r = _tool().execute(mode="agency", agency="")
        assert not r.success

    def test_search_mode_requires_query(self):
        r = _tool().execute(mode="search")
        assert not r.success
        assert "query" in r.output.lower()

    def test_search_mode_empty_query(self):
        r = _tool().execute(mode="search", query="")
        assert not r.success

    def test_extra_kwargs_ignored(self):
        with patch.object(GovContractsTool, "_post_json", return_value=SAMPLE_AWARDS_RESPONSE):
            r = _tool().execute(mode="recent", bogus="thing")
            assert r.success


# ── 3. Recent Mode ───────────────────────────────────────────


class TestRecentMode:
    def test_basic_recent(self):
        with patch.object(GovContractsTool, "_post_json", return_value=SAMPLE_AWARDS_RESPONSE):
            r = _tool().execute(mode="recent")
            assert r.success
            assert "awards" in r.data
            assert r.data["count"] == 2

    def test_recent_sorted_by_date(self):
        with patch.object(GovContractsTool, "_post_json", return_value=SAMPLE_AWARDS_RESPONSE) as mock:
            r = _tool().execute(mode="recent")
            call_payload = mock.call_args[0][1]
            assert call_payload["sort"] == "Start Date"
            assert call_payload["order"] == "desc"

    def test_recent_default_dates(self):
        with patch.object(GovContractsTool, "_post_json", return_value=SAMPLE_AWARDS_RESPONSE) as mock:
            r = _tool().execute(mode="recent")
            call_payload = mock.call_args[0][1]
            time_period = call_payload["filters"]["time_period"][0]
            assert "start_date" in time_period
            assert "end_date" in time_period

    def test_recent_custom_dates(self):
        with patch.object(GovContractsTool, "_post_json", return_value=SAMPLE_AWARDS_RESPONSE) as mock:
            r = _tool().execute(mode="recent", start_date="2024-01-01", end_date="2024-12-31")
            call_payload = mock.call_args[0][1]
            time_period = call_payload["filters"]["time_period"][0]
            assert time_period["start_date"] == "2024-01-01"
            assert time_period["end_date"] == "2024-12-31"


# ── 4. Top Mode ──────────────────────────────────────────────


class TestTopMode:
    def test_basic_top(self):
        with patch.object(GovContractsTool, "_post_json", return_value=SAMPLE_AWARDS_RESPONSE):
            r = _tool().execute(mode="top")
            assert r.success
            assert r.data["count"] == 2

    def test_top_sorted_by_amount(self):
        with patch.object(GovContractsTool, "_post_json", return_value=SAMPLE_AWARDS_RESPONSE) as mock:
            r = _tool().execute(mode="top")
            call_payload = mock.call_args[0][1]
            assert call_payload["sort"] == "Award Amount"
            assert call_payload["order"] == "desc"


# ── 5. Agency Mode ───────────────────────────────────────────


class TestAgencyMode:
    def test_basic_agency(self):
        with patch.object(GovContractsTool, "_post_json", return_value=SAMPLE_AWARDS_RESPONSE) as mock:
            r = _tool().execute(mode="agency", agency="Department of Defense")
            assert r.success
            call_payload = mock.call_args[0][1]
            agencies = call_payload["filters"]["agencies"]
            assert len(agencies) == 1
            assert agencies[0]["name"] == "Department of Defense"

    def test_agency_filter_structure(self):
        with patch.object(GovContractsTool, "_post_json", return_value=SAMPLE_AWARDS_RESPONSE) as mock:
            r = _tool().execute(mode="agency", agency="NASA")
            call_payload = mock.call_args[0][1]
            agency = call_payload["filters"]["agencies"][0]
            assert agency["type"] == "awarding"
            assert agency["tier"] == "toptier"


# ── 6. Search Mode ──────────────────────────────────────────


class TestSearchMode:
    def test_basic_search(self):
        with patch.object(GovContractsTool, "_post_json", return_value=SAMPLE_AWARDS_RESPONSE) as mock:
            r = _tool().execute(mode="search", query="semiconductor")
            assert r.success
            call_payload = mock.call_args[0][1]
            assert "semiconductor" in call_payload["filters"]["keywords"]

    def test_search_output_mentions_query(self):
        with patch.object(GovContractsTool, "_post_json", return_value=SAMPLE_AWARDS_RESPONSE):
            r = _tool().execute(mode="search", query="AI")
            assert "AI" in r.output


# ── 7. Payload Construction ──────────────────────────────────


class TestPayloadConstruction:
    def test_contract_codes(self):
        with patch.object(GovContractsTool, "_post_json", return_value=SAMPLE_AWARDS_RESPONSE) as mock:
            r = _tool().execute(mode="recent")
            call_payload = mock.call_args[0][1]
            assert call_payload["filters"]["award_type_codes"] == _CONTRACT_CODES

    def test_fields_requested(self):
        with patch.object(GovContractsTool, "_post_json", return_value=SAMPLE_AWARDS_RESPONSE) as mock:
            r = _tool().execute(mode="recent")
            call_payload = mock.call_args[0][1]
            assert call_payload["fields"] == _FIELDS

    def test_limit_in_payload(self):
        with patch.object(GovContractsTool, "_post_json", return_value=SAMPLE_AWARDS_RESPONSE) as mock:
            r = _tool().execute(mode="recent", limit=5)
            call_payload = mock.call_args[0][1]
            assert call_payload["limit"] == 5

    def test_limit_clamped_low(self):
        with patch.object(GovContractsTool, "_post_json", return_value=SAMPLE_AWARDS_RESPONSE) as mock:
            r = _tool().execute(mode="recent", limit=0)
            call_payload = mock.call_args[0][1]
            assert call_payload["limit"] == 1

    def test_limit_clamped_high(self):
        with patch.object(GovContractsTool, "_post_json", return_value=SAMPLE_AWARDS_RESPONSE) as mock:
            r = _tool().execute(mode="recent", limit=999)
            call_payload = mock.call_args[0][1]
            assert call_payload["limit"] == 50


# ── 8. Result Parsing ────────────────────────────────────────


class TestResultParsing:
    def test_award_fields(self):
        with patch.object(GovContractsTool, "_post_json", return_value=SAMPLE_AWARDS_RESPONSE):
            r = _tool().execute(mode="recent")
            a = r.data["awards"][0]
            assert a["award_id"] == "HT940216C0001"
            assert a["recipient"] == "ACME DEFENSE INC"
            assert a["amount_usd"] == 500_000_000
            assert a["agency"] == "Department of Defense"
            assert a["sub_agency"] == "Army"
            assert a["start_date"] == "2024-06-01"

    def test_description_truncated(self):
        with patch.object(GovContractsTool, "_post_json", return_value=SAMPLE_AWARDS_RESPONSE):
            r = _tool().execute(mode="recent")
            for a in r.data["awards"]:
                assert len(a["description"]) <= 200

    def test_total_count(self):
        with patch.object(GovContractsTool, "_post_json", return_value=SAMPLE_AWARDS_RESPONSE):
            r = _tool().execute(mode="recent")
            assert r.data["total"] == 2

    def test_empty_results(self):
        empty = {"results": [], "page_metadata": {"total": 0}}
        with patch.object(GovContractsTool, "_post_json", return_value=empty):
            r = _tool().execute(mode="recent")
            assert r.success
            assert r.data["count"] == 0

    def test_missing_description(self):
        data = {
            "results": [{"Award ID": "X", "Recipient Name": "Y", "Award Amount": 100}],
            "page_metadata": {"total": 1},
        }
        with patch.object(GovContractsTool, "_post_json", return_value=data):
            r = _tool().execute(mode="recent")
            assert r.success

    def test_null_fields(self):
        data = {
            "results": [{"Award ID": None, "Recipient Name": None, "Award Amount": None}],
            "page_metadata": {"total": 1},
        }
        with patch.object(GovContractsTool, "_post_json", return_value=data):
            r = _tool().execute(mode="recent")
            assert r.success


# ── 9. HTTP Error Handling ────────────────────────────────────


class TestHTTPErrors:
    def test_timeout(self):
        with patch.object(
            GovContractsTool,
            "_post_json",
            side_effect=httpx.TimeoutException("timeout"),
        ):
            r = _tool().execute(mode="recent")
            assert not r.success
            assert "timed out" in r.output

    def test_http_error(self):
        with patch.object(
            GovContractsTool,
            "_post_json",
            side_effect=httpx.HTTPStatusError(
                "500",
                request=httpx.Request("POST", "http://test"),
                response=httpx.Response(500),
            ),
        ):
            r = _tool().execute(mode="recent")
            assert not r.success

    def test_connection_error(self):
        with patch.object(GovContractsTool, "_post_json", side_effect=httpx.ConnectError("fail")):
            r = _tool().execute(mode="recent")
            assert not r.success

    def test_generic_exception(self):
        with patch.object(GovContractsTool, "_post_json", side_effect=RuntimeError("boom")):
            r = _tool().execute(mode="recent")
            assert not r.success

    def test_fetch_returns_none(self):
        with patch.object(GovContractsTool, "_post_json", return_value=None):
            r = _tool().execute(mode="recent")
            assert not r.success
            assert "Failed" in r.output


# ── 10. Output Formatting ────────────────────────────────────


class TestOutputFormatting:
    def test_recent_output(self):
        with patch.object(GovContractsTool, "_post_json", return_value=SAMPLE_AWARDS_RESPONSE):
            r = _tool().execute(mode="recent")
            assert "federal contract" in r.output.lower()

    def test_agency_output_mentions_agency(self):
        with patch.object(GovContractsTool, "_post_json", return_value=SAMPLE_AWARDS_RESPONSE):
            r = _tool().execute(mode="agency", agency="DoD")
            assert "DoD" in r.output

    def test_search_output_mentions_keyword(self):
        with patch.object(GovContractsTool, "_post_json", return_value=SAMPLE_AWARDS_RESPONSE):
            r = _tool().execute(mode="search", query="chips")
            assert "chips" in r.output


# ── 11. Constants ─────────────────────────────────────────────


class TestConstants:
    def test_valid_modes(self):
        assert {"recent", "top", "agency", "search"} == VALID_MODES

    def test_contract_codes(self):
        assert ["A", "B", "C", "D"] == _CONTRACT_CODES

    def test_fields_nonempty(self):
        assert len(_FIELDS) >= 5

    def test_awards_url_https(self):
        assert _AWARDS_URL.startswith("https://")


# ── 12. Registry + Bandit Integration ────────────────────────


class TestRegistryAndBandit:
    def test_tool_count(self):
        try:
            from agent.cli import build_tool_registry
        except (ImportError, ModuleNotFoundError):
            pytest.skip("optional dep not installed")

        mock_config = MagicMock()
        mock_config.tool_timeout = 30
        mock_config.fred_api_key = ""
        registry = build_tool_registry(mock_config)
        assert len(registry._tools) == 61

    def test_gov_contracts_registered(self):
        try:
            from agent.cli import build_tool_registry
        except (ImportError, ModuleNotFoundError):
            pytest.skip("optional dep not installed")

        mock_config = MagicMock()
        mock_config.tool_timeout = 30
        mock_config.fred_api_key = ""
        registry = build_tool_registry(mock_config)
        assert "gov_contracts" in registry._tools

    def test_bandit_arm_count(self):
        from agent.learning.bandit import DEFAULT_ARMS

        assert len(DEFAULT_ARMS) == 48

    def test_government_spending_arm_exists(self):
        from agent.learning.bandit import DEFAULT_ARMS

        names = {a.name for a in DEFAULT_ARMS}
        assert "government_spending" in names

    def test_arm_references_tool(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm = next(a for a in DEFAULT_ARMS if a.name == "government_spending")
        assert "gov_contracts" in arm.tools


# ══════════════════════════════════════════════════════════════
# UK CONTRACTS FINDER (OCDS) — EDGE CASE TESTS
# ══════════════════════════════════════════════════════════════

SAMPLE_UK_OCDS_RESPONSE = {
    "uri": "https://www.contractsfinder.service.gov.uk/Published/Notices/OCDS/Search",
    "releases": [
        {
            "ocid": "ocds-b5fd17-cf-12345",
            "id": "cf-12345-award-1",
            "buyer": {"name": "Ministry of Defence"},
            "tender": {
                "title": "Military satellite communications upgrade",
                "description": "Upgrade of SATCOM systems across UK MoD facilities",
                "procurementMethod": "open",
                "contractPeriod": {
                    "startDate": "2024-09-01T00:00:00Z",
                    "endDate": "2026-09-01T00:00:00Z",
                },
            },
            "awards": [
                {
                    "value": {"amount": 75_000_000, "currency": "GBP"},
                    "suppliers": [{"name": "BAE Systems plc"}],
                }
            ],
        },
        {
            "ocid": "ocds-b5fd17-cf-67890",
            "id": "cf-67890-award-1",
            "buyer": {"name": "NHS England"},
            "tender": {
                "title": "GP digital records platform",
                "description": "National rollout of digital patient records for general practice",
                "procurementMethod": "selective",
                "tenderPeriod": {
                    "startDate": "2024-06-15T00:00:00Z",
                    "endDate": "2025-06-15T00:00:00Z",
                },
            },
            "awards": [
                {
                    "value": {"amount": 12_500_000, "currency": "GBP"},
                    "suppliers": [{"name": "NHS Digital Solutions Ltd"}],
                }
            ],
        },
        {
            "ocid": "ocds-b5fd17-cf-11111",
            "id": "cf-11111-award-1",
            "buyer": {"name": "Ministry of Defence"},
            "tender": {
                "title": "Vehicle fleet maintenance",
                "description": "Routine maintenance of MoD vehicle fleet",
                "procurementMethod": "open",
                "contractPeriod": {
                    "startDate": "2024-01-10T00:00:00Z",
                    "endDate": "2025-01-10T00:00:00Z",
                },
            },
            "awards": [
                {
                    "value": {"amount": 3_200_000, "currency": "GBP"},
                    "suppliers": [{"name": "Fleet Support Ltd"}],
                }
            ],
        },
    ],
}


def _uk_ocds_releases():
    """Return just the releases list for mocking _fetch_uk_contracts."""
    return SAMPLE_UK_OCDS_RESPONSE["releases"]


# ── 13. Region Parameter ─────────────────────────────────────


class TestRegionParameter:
    def test_valid_regions_constant(self):
        assert {"us", "uk"} == VALID_REGIONS

    def test_invalid_region_rejected(self):
        r = _tool().execute(mode="recent", region="fr")
        assert not r.success
        assert "Invalid region" in r.output

    def test_empty_region_defaults_to_us(self):
        with patch.object(GovContractsTool, "_post_json", return_value=SAMPLE_AWARDS_RESPONSE):
            r = _tool().execute(mode="recent", region="")
            assert r.success

    def test_none_region_defaults_to_us(self):
        with patch.object(GovContractsTool, "_post_json", return_value=SAMPLE_AWARDS_RESPONSE):
            r = _tool().execute(mode="recent", region=None)
            assert r.success

    def test_region_case_insensitive(self):
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=_uk_ocds_releases()):
            r = _tool().execute(mode="recent", region="UK")
            assert r.success
            assert r.data["region"] == "uk"

    def test_region_whitespace_stripped(self):
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=_uk_ocds_releases()):
            r = _tool().execute(mode="recent", region="  uk  ")
            assert r.success

    def test_us_region_explicit(self):
        with patch.object(GovContractsTool, "_post_json", return_value=SAMPLE_AWARDS_RESPONSE):
            r = _tool().execute(mode="recent", region="us")
            assert r.success
            # US results should not have "region" key in data
            assert r.data.get("region") is None

    def test_region_in_parameters_schema(self):
        params = _tool().parameters
        assert "region" in params["properties"]
        region_schema = params["properties"]["region"]
        assert region_schema["type"] == "string"
        assert set(region_schema["enum"]) == {"uk", "us"}


# ── 14. UK Recent Mode ───────────────────────────────────────


class TestUKRecentMode:
    def test_basic_recent(self):
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=_uk_ocds_releases()):
            r = _tool().execute(mode="recent", region="uk")
            assert r.success
            assert "awards" in r.data
            assert r.data["count"] == 3
            assert r.data["region"] == "uk"

    def test_recent_sorted_by_date_descending(self):
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=_uk_ocds_releases()):
            r = _tool().execute(mode="recent", region="uk")
            dates = [a["start_date"] for a in r.data["awards"]]
            assert dates == sorted(dates, reverse=True)

    def test_recent_custom_dates(self):
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=_uk_ocds_releases()) as mock:
            r = _tool().execute(
                mode="recent",
                region="uk",
                start_date="2024-01-01",
                end_date="2024-12-31",
            )
            mock.assert_called_once_with("2024-01-01", "2024-12-31")
            assert r.success

    def test_recent_output_mentions_uk(self):
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=_uk_ocds_releases()):
            r = _tool().execute(mode="recent", region="uk")
            assert "UK" in r.output


# ── 15. UK Top Mode ──────────────────────────────────────────


class TestUKTopMode:
    def test_basic_top(self):
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=_uk_ocds_releases()):
            r = _tool().execute(mode="top", region="uk")
            assert r.success
            assert r.data["count"] == 3

    def test_top_sorted_by_amount_descending(self):
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=_uk_ocds_releases()):
            r = _tool().execute(mode="top", region="uk")
            amounts = [a["amount"] for a in r.data["awards"]]
            assert amounts == sorted(amounts, reverse=True)

    def test_top_largest_first(self):
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=_uk_ocds_releases()):
            r = _tool().execute(mode="top", region="uk")
            assert r.data["awards"][0]["amount"] == 75_000_000


# ── 16. UK Agency Mode ───────────────────────────────────────


class TestUKAgencyMode:
    def test_agency_requires_param(self):
        r = _tool().execute(mode="agency", region="uk")
        assert not r.success
        assert "agency" in r.output.lower()

    def test_agency_empty_rejected(self):
        r = _tool().execute(mode="agency", region="uk", agency="")
        assert not r.success

    def test_agency_filters_by_buyer(self):
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=_uk_ocds_releases()):
            r = _tool().execute(mode="agency", region="uk", agency="Ministry of Defence")
            assert r.success
            assert r.data["count"] == 2
            for a in r.data["awards"]:
                assert "Ministry of Defence" in a["agency"]

    def test_agency_case_insensitive_filter(self):
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=_uk_ocds_releases()):
            r = _tool().execute(mode="agency", region="uk", agency="ministry of defence")
            assert r.success
            assert r.data["count"] == 2

    def test_agency_partial_match(self):
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=_uk_ocds_releases()):
            r = _tool().execute(mode="agency", region="uk", agency="NHS")
            assert r.success
            assert r.data["count"] == 1
            assert r.data["awards"][0]["agency"] == "NHS England"

    def test_agency_no_match(self):
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=_uk_ocds_releases()):
            r = _tool().execute(mode="agency", region="uk", agency="NONEXISTENT_DEPT")
            assert r.success
            assert r.data["count"] == 0

    def test_agency_output_mentions_buyer(self):
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=_uk_ocds_releases()):
            r = _tool().execute(mode="agency", region="uk", agency="NHS England")
            assert "NHS England" in r.output


# ── 17. UK Search Mode ───────────────────────────────────────


class TestUKSearchMode:
    def test_search_requires_query(self):
        r = _tool().execute(mode="search", region="uk")
        assert not r.success
        assert "query" in r.output.lower()

    def test_search_empty_rejected(self):
        r = _tool().execute(mode="search", region="uk", query="")
        assert not r.success

    def test_search_by_title(self):
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=_uk_ocds_releases()):
            r = _tool().execute(mode="search", region="uk", query="satellite")
            assert r.success
            assert r.data["count"] == 1
            assert "satellite" in r.data["awards"][0]["description"].lower()

    def test_search_by_description(self):
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=_uk_ocds_releases()):
            r = _tool().execute(mode="search", region="uk", query="patient records")
            assert r.success
            assert r.data["count"] == 1

    def test_search_by_buyer_name(self):
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=_uk_ocds_releases()):
            r = _tool().execute(mode="search", region="uk", query="NHS")
            assert r.success
            assert r.data["count"] == 1

    def test_search_case_insensitive(self):
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=_uk_ocds_releases()):
            r = _tool().execute(mode="search", region="uk", query="SATELLITE")
            assert r.success
            assert r.data["count"] == 1

    def test_search_no_match(self):
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=_uk_ocds_releases()):
            r = _tool().execute(mode="search", region="uk", query="quantum_unicorn_xyz")
            assert r.success
            assert r.data["count"] == 0

    def test_search_output_mentions_query(self):
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=_uk_ocds_releases()):
            r = _tool().execute(mode="search", region="uk", query="digital")
            assert "digital" in r.output


# ── 18. UK Result Parsing ────────────────────────────────────


class TestUKResultParsing:
    def test_award_fields_present(self):
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=_uk_ocds_releases()):
            r = _tool().execute(mode="recent", region="uk")
            a = r.data["awards"][0]
            assert "award_id" in a
            assert "recipient" in a
            assert "amount" in a
            assert "currency" in a
            assert "agency" in a
            assert "award_type" in a
            assert "start_date" in a
            assert "end_date" in a
            assert "description" in a
            assert "region" in a

    def test_first_award_values(self):
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=_uk_ocds_releases()):
            r = _tool().execute(mode="recent", region="uk")
            # Most recent (by start_date) should be the MoD SATCOM one: 2024-09-01
            a = r.data["awards"][0]
            assert a["award_id"] == "ocds-b5fd17-cf-12345"
            assert a["recipient"] == "BAE Systems plc"
            assert a["amount"] == 75_000_000
            assert a["currency"] == "GBP"
            assert a["agency"] == "Ministry of Defence"
            assert a["award_type"] == "open"
            assert a["start_date"] == "2024-09-01"
            assert a["region"] == "uk"

    def test_description_truncated_at_200(self):
        long_title = "X" * 300
        releases = [
            {
                "ocid": "test-long",
                "buyer": {"name": "Test"},
                "tender": {"title": long_title, "description": ""},
                "awards": [],
            }
        ]
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=releases):
            r = _tool().execute(mode="recent", region="uk")
            assert len(r.data["awards"][0]["description"]) <= 200

    def test_total_and_count(self):
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=_uk_ocds_releases()):
            r = _tool().execute(mode="recent", region="uk")
            assert r.data["total"] == 3
            assert r.data["count"] == 3

    def test_limit_applied(self):
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=_uk_ocds_releases()):
            r = _tool().execute(mode="recent", region="uk", limit=1)
            assert r.data["count"] == 1
            assert r.data["total"] == 3

    def test_empty_releases(self):
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=[]):
            r = _tool().execute(mode="recent", region="uk")
            assert r.success
            assert r.data["count"] == 0
            assert r.data["awards"] == []

    def test_release_missing_tender(self):
        releases = [{"ocid": "no-tender", "buyer": {"name": "Test"}, "awards": []}]
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=releases):
            r = _tool().execute(mode="recent", region="uk")
            assert r.success
            assert r.data["count"] == 1

    def test_release_missing_buyer(self):
        releases = [
            {
                "ocid": "no-buyer",
                "tender": {"title": "Test", "description": ""},
                "awards": [{"value": {"amount": 100}, "suppliers": []}],
            }
        ]
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=releases):
            r = _tool().execute(mode="recent", region="uk")
            assert r.success
            assert r.data["awards"][0]["agency"] == ""

    def test_release_missing_awards_array(self):
        releases = [
            {
                "ocid": "no-awards",
                "buyer": {"name": "Test"},
                "tender": {"title": "Test", "description": ""},
            }
        ]
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=releases):
            r = _tool().execute(mode="recent", region="uk")
            assert r.success
            a = r.data["awards"][0]
            assert a["amount"] is None
            assert a["recipient"] == ""

    def test_release_empty_suppliers(self):
        releases = [
            {
                "ocid": "empty-suppliers",
                "buyer": {"name": "Test"},
                "tender": {"title": "Test", "description": ""},
                "awards": [{"value": {"amount": 500}, "suppliers": []}],
            }
        ]
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=releases):
            r = _tool().execute(mode="recent", region="uk")
            assert r.data["awards"][0]["recipient"] == ""

    def test_release_no_value_in_award(self):
        releases = [
            {
                "ocid": "no-value",
                "buyer": {"name": "Test"},
                "tender": {"title": "Test", "description": ""},
                "awards": [{"suppliers": [{"name": "Someone"}]}],
            }
        ]
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=releases):
            r = _tool().execute(mode="recent", region="uk")
            assert r.data["awards"][0]["amount"] is None
            assert r.data["awards"][0]["recipient"] == "Someone"

    def test_uses_tender_period_fallback(self):
        """When contractPeriod is absent, falls back to tenderPeriod."""
        releases = [
            {
                "ocid": "tender-period",
                "buyer": {"name": "Test"},
                "tender": {
                    "title": "Test",
                    "description": "",
                    "tenderPeriod": {
                        "startDate": "2024-03-01T00:00:00Z",
                        "endDate": "2025-03-01T00:00:00Z",
                    },
                },
                "awards": [],
            }
        ]
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=releases):
            r = _tool().execute(mode="recent", region="uk")
            a = r.data["awards"][0]
            assert a["start_date"] == "2024-03-01"
            assert a["end_date"] == "2025-03-01"

    def test_currency_defaults_to_gbp(self):
        releases = [
            {
                "ocid": "no-currency",
                "buyer": {"name": "Test"},
                "tender": {"title": "Test", "description": ""},
                "awards": [{"value": {"amount": 100}, "suppliers": []}],
            }
        ]
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=releases):
            r = _tool().execute(mode="recent", region="uk")
            assert r.data["awards"][0]["currency"] == "GBP"

    def test_non_gbp_currency_preserved(self):
        releases = [
            {
                "ocid": "eur-award",
                "buyer": {"name": "Test"},
                "tender": {"title": "Test", "description": ""},
                "awards": [{"value": {"amount": 500, "currency": "EUR"}, "suppliers": []}],
            }
        ]
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=releases):
            r = _tool().execute(mode="recent", region="uk")
            assert r.data["awards"][0]["currency"] == "EUR"

    def test_ocid_used_as_award_id(self):
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=_uk_ocds_releases()):
            r = _tool().execute(mode="recent", region="uk")
            ids = {a["award_id"] for a in r.data["awards"]}
            assert "ocds-b5fd17-cf-12345" in ids

    def test_fallback_to_id_when_no_ocid(self):
        releases = [
            {
                "id": "fallback-id-123",
                "buyer": {"name": "Test"},
                "tender": {"title": "Test", "description": ""},
                "awards": [],
            }
        ]
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=releases):
            r = _tool().execute(mode="recent", region="uk")
            assert r.data["awards"][0]["award_id"] == "fallback-id-123"


# ── 19. UK HTTP Error Handling ────────────────────────────────


class TestUKHTTPErrors:
    def test_timeout(self):
        with patch.object(
            GovContractsTool,
            "_fetch_uk_contracts",
            side_effect=httpx.TimeoutException("timeout"),
        ):
            r = _tool().execute(mode="recent", region="uk")
            assert not r.success
            assert "timed out" in r.output.lower() or "Contracts Finder" in r.output

    def test_http_error(self):
        with patch.object(
            GovContractsTool,
            "_fetch_uk_contracts",
            side_effect=httpx.HTTPStatusError(
                "500",
                request=httpx.Request("GET", "http://test"),
                response=httpx.Response(500),
            ),
        ):
            r = _tool().execute(mode="recent", region="uk")
            assert not r.success
            assert "Contracts Finder" in r.output

    def test_connection_error(self):
        with patch.object(
            GovContractsTool,
            "_fetch_uk_contracts",
            side_effect=httpx.ConnectError("fail"),
        ):
            r = _tool().execute(mode="recent", region="uk")
            assert not r.success

    def test_generic_exception(self):
        with patch.object(
            GovContractsTool,
            "_fetch_uk_contracts",
            side_effect=RuntimeError("boom"),
        ):
            r = _tool().execute(mode="recent", region="uk")
            assert not r.success

    def test_fetch_returns_none(self):
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=None):
            r = _tool().execute(mode="recent", region="uk")
            assert not r.success
            assert "Failed" in r.output

    def test_us_timeout_says_usaspending(self):
        with patch.object(
            GovContractsTool,
            "_post_json",
            side_effect=httpx.TimeoutException("timeout"),
        ):
            r = _tool().execute(mode="recent", region="us")
            assert "USASpending" in r.output

    def test_uk_timeout_says_contracts_finder(self):
        with patch.object(
            GovContractsTool,
            "_fetch_uk_contracts",
            side_effect=httpx.TimeoutException("timeout"),
        ):
            r = _tool().execute(mode="recent", region="uk")
            assert "Contracts Finder" in r.output


# ── 20. UK Output Formatting ─────────────────────────────────


class TestUKOutputFormatting:
    def test_recent_output_mentions_uk(self):
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=_uk_ocds_releases()):
            r = _tool().execute(mode="recent", region="uk")
            assert "UK" in r.output

    def test_agency_output_mentions_buyer(self):
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=_uk_ocds_releases()):
            r = _tool().execute(mode="agency", region="uk", agency="NHS England")
            assert "NHS England" in r.output

    def test_search_output_mentions_keyword(self):
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=_uk_ocds_releases()):
            r = _tool().execute(mode="search", region="uk", query="satellite")
            assert "satellite" in r.output

    def test_output_includes_count(self):
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=_uk_ocds_releases()):
            r = _tool().execute(mode="recent", region="uk")
            assert "3" in r.output


# ── 21. UK Constants ─────────────────────────────────────────


class TestUKConstants:
    def test_uk_ocds_url_https(self):
        assert _UK_OCDS_URL.startswith("https://")

    def test_uk_ocds_url_contains_ocds(self):
        assert "OCDS" in _UK_OCDS_URL

    def test_modes_shared_across_regions(self):
        """Both US and UK support the same 4 modes."""
        with patch.object(GovContractsTool, "_post_json", return_value=SAMPLE_AWARDS_RESPONSE):
            for m in VALID_MODES:
                kw = {"mode": m, "region": "us"}
                if m == "agency":
                    kw["agency"] = "Test"
                if m == "search":
                    kw["query"] = "test"
                r = _tool().execute(**kw)
                assert r.success, f"US mode '{m}' failed"

        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=_uk_ocds_releases()):
            for m in VALID_MODES:
                kw = {"mode": m, "region": "uk"}
                if m == "agency":
                    kw["agency"] = "Ministry of Defence"
                if m == "search":
                    kw["query"] = "satellite"
                r = _tool().execute(**kw)
                assert r.success, f"UK mode '{m}' failed"


# ── 22. Top-mode tie-breaking with None amounts ──────────────


class TestUKTopModeEdge:
    def test_top_with_none_amounts(self):
        """None amounts should not crash sorting."""
        releases = [
            {
                "ocid": "has-amount",
                "buyer": {"name": "A"},
                "tender": {"title": "T1", "description": ""},
                "awards": [{"value": {"amount": 1000}, "suppliers": []}],
            },
            {
                "ocid": "no-amount",
                "buyer": {"name": "B"},
                "tender": {"title": "T2", "description": ""},
                "awards": [],
            },
        ]
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=releases):
            r = _tool().execute(mode="top", region="uk")
            assert r.success
            assert r.data["awards"][0]["amount"] == 1000
            assert r.data["awards"][1]["amount"] is None

    def test_top_all_none_amounts(self):
        releases = [
            {
                "ocid": f"x-{i}",
                "buyer": {"name": "Test"},
                "tender": {"title": "T", "description": ""},
                "awards": [],
            }
            for i in range(3)
        ]
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=releases):
            r = _tool().execute(mode="top", region="uk")
            assert r.success
            assert r.data["count"] == 3

    def test_recent_with_missing_dates(self):
        """Missing start_date should not crash date sorting."""
        releases = [
            {
                "ocid": "has-date",
                "buyer": {"name": "A"},
                "tender": {
                    "title": "T1",
                    "description": "",
                    "contractPeriod": {"startDate": "2024-05-01T00:00:00Z"},
                },
                "awards": [],
            },
            {
                "ocid": "no-date",
                "buyer": {"name": "B"},
                "tender": {"title": "T2", "description": ""},
                "awards": [],
            },
        ]
        with patch.object(GovContractsTool, "_fetch_uk_contracts", return_value=releases):
            r = _tool().execute(mode="recent", region="uk")
            assert r.success
            assert r.data["count"] == 2
