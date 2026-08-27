"""
Edge-case tests for LobbyingTool (7b-J).

Coverage targets:
- Invalid / missing / boundary parameters
- Empty / malformed API responses
- Cache hit / miss paths
- Search: registrant, client, year, quarter combinations
- Spending: yearly aggregation, anomaly detection
- Issues: issue code listing, filtering, registrant aggregation
- HTTP errors, timeouts
- Mode validation
- Year boundary validation (2008-current)
- Quarter mapping
- Helper functions (_parse_filing, _detect_spend_anomaly, _fetch_lda)
- API key handling (optional)
- Integration: tool count = 44, arm count = 32
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.tools.lobbying import (
    ISSUE_AREAS,
    VALID_MODES,
    LobbyingTool,
    _detect_spend_anomaly,
    _fetch_lda,
    _parse_filing,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tool():
    cache = MagicMock()
    cache.get.return_value = None
    t = LobbyingTool(cache=cache)
    t._api_key = "test-key"
    return t


@pytest.fixture
def tool_no_key():
    cache = MagicMock()
    cache.get.return_value = None
    t = LobbyingTool(cache=cache)
    t._api_key = None
    return t


@pytest.fixture
def tool_no_cache():
    t = LobbyingTool(cache=None)
    t._api_key = "test-key"
    return t


def _filing(
    registrant: str = "Lobby Corp",
    client: str = "TechCo",
    year: int = 2024,
    amount_income: str = "100000.00",
    issue_codes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "filing_uuid": "test-uuid-1234",
        "filing_type": "RR",
        "filing_type_display": "Registration",
        "filing_year": year,
        "filing_period": "first_quarter",
        "income": amount_income,
        "expenses": "0.00",
        "dt_posted": "2024-04-15T09:30:00Z",
        "registrant": {"id": 1, "name": registrant},
        "client": {"id": 2, "name": client},
        "lobbying_activities": [
            {"general_issue_code": code, "description": f"Activity for {code}"} for code in (issue_codes or ["HCR"])
        ],
    }


def _lda_response(
    results: list[dict] | None = None,
    count: int | None = None,
) -> dict[str, Any]:
    r = results or []
    return {
        "count": count if count is not None else len(r),
        "next": None,
        "previous": None,
        "results": r,
    }


def _mock_response(data: dict, status: int = 200) -> httpx.Response:
    r = MagicMock(spec=httpx.Response)
    r.status_code = status
    r.json.return_value = data
    return r


# ═══════════════════════════════════════════════════════════════════════════
# 1. Mode validation
# ═══════════════════════════════════════════════════════════════════════════


class TestModeValidation:
    def test_valid_modes(self, tool):
        assert {"search", "spending", "issues"} == VALID_MODES

    def test_empty_mode(self, tool):
        r = tool.execute(mode="")
        assert not r.success
        assert "Invalid mode" in r.output

    def test_invalid_mode(self, tool):
        r = tool.execute(mode="bogus")
        assert not r.success
        assert "bogus" in r.output

    def test_none_mode(self, tool):
        r = tool.execute(mode=None)
        assert not r.success

    def test_mode_case_insensitive(self, tool):
        # search mode without params → error, but not "Invalid mode"
        r = tool.execute(mode="SEARCH")
        assert "Invalid mode" not in r.output

    def test_mode_whitespace(self, tool):
        r = tool.execute(mode="  search  ")
        assert "Invalid mode" not in r.output


# ═══════════════════════════════════════════════════════════════════════════
# 2. Search mode
# ═══════════════════════════════════════════════════════════════════════════


class TestSearchMode:
    def test_search_no_params(self, tool):
        r = tool.execute(mode="search")
        assert not r.success
        assert "registrant" in r.output or "client" in r.output

    @patch("agent.tools.lobbying._fetch_lda")
    def test_search_by_registrant(self, mock_fetch, tool):
        mock_fetch.return_value = _lda_response([_filing()])
        r = tool.execute(mode="search", registrant="Lobby Corp")
        assert r.success
        assert r.data["mode"] == "search"

    @patch("agent.tools.lobbying._fetch_lda")
    def test_search_by_client(self, mock_fetch, tool):
        mock_fetch.return_value = _lda_response([_filing(client="BigCo")])
        r = tool.execute(mode="search", client="BigCo")
        assert r.success

    @patch("agent.tools.lobbying._fetch_lda")
    def test_search_by_year(self, mock_fetch, tool):
        mock_fetch.return_value = _lda_response([_filing(year=2023)])
        r = tool.execute(mode="search", year=2023)
        assert r.success

    def test_search_year_too_low(self, tool):
        r = tool.execute(mode="search", year=2005)
        assert not r.success

    def test_search_year_too_high(self, tool):
        r = tool.execute(mode="search", year=3000)
        assert not r.success

    @patch("agent.tools.lobbying._fetch_lda")
    def test_search_with_quarter(self, mock_fetch, tool):
        mock_fetch.return_value = _lda_response([_filing()])
        r = tool.execute(mode="search", registrant="Test", quarter="Q2")
        assert r.success

    @patch("agent.tools.lobbying._fetch_lda")
    def test_search_empty_results(self, mock_fetch, tool):
        mock_fetch.return_value = _lda_response([])
        r = tool.execute(mode="search", registrant="NoOne")
        assert r.success
        assert r.data["total_count"] == 0

    @patch("agent.tools.lobbying._fetch_lda")
    def test_search_api_unavailable(self, mock_fetch, tool):
        mock_fetch.return_value = None
        r = tool.execute(mode="search", registrant="Test")
        assert not r.success
        assert "unavailable" in r.output.lower()

    @patch("agent.tools.lobbying._fetch_lda")
    def test_search_cache_hit(self, mock_fetch, tool):
        tool._cache.get.return_value = {
            "filings": [_parse_filing(_filing())],
            "total_count": 1,
            "returned": 1,
        }
        r = tool.execute(mode="search", registrant="Cached")
        assert r.success
        assert "(cached)" in r.output
        mock_fetch.assert_not_called()

    @patch("agent.tools.lobbying._fetch_lda")
    def test_search_cache_write(self, mock_fetch, tool):
        mock_fetch.return_value = _lda_response([_filing()])
        tool.execute(mode="search", registrant="Test")
        tool._cache.put.assert_called()


# ═══════════════════════════════════════════════════════════════════════════
# 3. Spending mode
# ═══════════════════════════════════════════════════════════════════════════


class TestSpendingMode:
    def test_spending_no_params(self, tool):
        r = tool.execute(mode="spending")
        assert not r.success
        assert "registrant" in r.output or "client" in r.output

    @patch("agent.tools.lobbying._fetch_lda")
    def test_spending_basic(self, mock_fetch, tool):
        mock_fetch.return_value = _lda_response([_filing(amount_income="250000.00")])
        r = tool.execute(mode="spending", registrant="Lobby Corp")
        assert r.success
        assert r.data["mode"] == "spending"
        assert "yearly_totals" in r.data

    @patch("agent.tools.lobbying._fetch_lda")
    def test_spending_by_client(self, mock_fetch, tool):
        mock_fetch.return_value = _lda_response([_filing()])
        r = tool.execute(mode="spending", client="TechCo")
        assert r.success

    @patch("agent.tools.lobbying._fetch_lda")
    def test_spending_anomaly_detected(self, mock_fetch, tool):
        # Return escalating amounts across years
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            year = call_count[0]
            if year <= 4:
                return _lda_response([_filing(amount_income="50000.00")])
            else:
                # Last year: 5x spike
                return _lda_response([_filing(amount_income="500000.00")])

        mock_fetch.side_effect = side_effect
        r = tool.execute(mode="spending", registrant="SpikeCorp")
        assert r.success
        assert r.data["anomaly"]["anomaly"] is True

    @patch("agent.tools.lobbying._fetch_lda")
    def test_spending_no_anomaly(self, mock_fetch, tool):
        mock_fetch.return_value = _lda_response([_filing(amount_income="100000.00")])
        r = tool.execute(mode="spending", registrant="SteadyCorp")
        assert r.success
        assert r.data["anomaly"]["anomaly"] is False

    @patch("agent.tools.lobbying._fetch_lda")
    def test_spending_all_api_failures(self, mock_fetch, tool):
        mock_fetch.return_value = None
        r = tool.execute(mode="spending", registrant="FailCorp")
        assert not r.success

    @patch("agent.tools.lobbying._fetch_lda")
    def test_spending_cache_hit(self, mock_fetch, tool):
        tool._cache.get.return_value = {"filings": [_parse_filing(_filing())]}
        r = tool.execute(mode="spending", registrant="CachedCorp")
        assert r.success
        mock_fetch.assert_not_called()

    @patch("agent.tools.lobbying._fetch_lda")
    def test_spending_partial_failure(self, mock_fetch, tool):
        # Some years succeed, some fail
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                return None
            return _lda_response([_filing()])

        mock_fetch.side_effect = side_effect
        r = tool.execute(mode="spending", registrant="PartialCorp")
        assert r.success  # At least some data


# ═══════════════════════════════════════════════════════════════════════════
# 4. Issues mode
# ═══════════════════════════════════════════════════════════════════════════


class TestIssuesMode:
    def test_issues_no_code(self, tool):
        r = tool.execute(mode="issues")
        assert r.success
        assert "issue_areas" in r.data

    @patch("agent.tools.lobbying._fetch_lda")
    def test_issues_with_code(self, mock_fetch, tool):
        mock_fetch.return_value = _lda_response([_filing(issue_codes=["HCR"])])
        r = tool.execute(mode="issues", issue_code="HCR")
        assert r.success
        assert r.data["issue_code"] == "HCR"
        assert r.data["issue_name"] == "Health Issues"

    @patch("agent.tools.lobbying._fetch_lda")
    def test_issues_by_year(self, mock_fetch, tool):
        mock_fetch.return_value = _lda_response([_filing(year=2023, issue_codes=["TAX"])])
        r = tool.execute(mode="issues", issue_code="TAX", year=2023)
        assert r.success
        assert r.data["year"] == 2023

    @patch("agent.tools.lobbying._fetch_lda")
    def test_issues_no_matching_filings(self, mock_fetch, tool):
        # Filings exist but none match the issue code
        mock_fetch.return_value = _lda_response([_filing(issue_codes=["DEF"])])
        r = tool.execute(mode="issues", issue_code="HCR")
        assert r.success
        assert r.data["total_filings"] == 0

    @patch("agent.tools.lobbying._fetch_lda")
    def test_issues_api_unavailable(self, mock_fetch, tool):
        mock_fetch.return_value = None
        r = tool.execute(mode="issues", issue_code="HCR")
        assert not r.success

    @patch("agent.tools.lobbying._fetch_lda")
    def test_issues_registrant_aggregation(self, mock_fetch, tool):
        filings = [
            _filing(registrant="FirmA", amount_income="100000", issue_codes=["HCR"]),
            _filing(registrant="FirmB", amount_income="200000", issue_codes=["HCR"]),
        ]
        mock_fetch.return_value = _lda_response(filings)
        r = tool.execute(mode="issues", issue_code="HCR")
        assert r.success
        assert len(r.data["registrant_spend"]) == 2

    @patch("agent.tools.lobbying._fetch_lda")
    def test_issues_cache_hit(self, mock_fetch, tool):
        tool._cache.get.return_value = {
            "issue_code": "HCR",
            "issue_name": "Health Issues",
            "year": 2024,
            "total_filings": 5,
            "total_searched": 10,
            "filings": [_parse_filing(_filing(issue_codes=["HCR"]))],
            "registrant_spend": {"FirmA": 100000},
        }
        r = tool.execute(mode="issues", issue_code="HCR")
        assert r.success
        assert "(cached)" in r.output
        mock_fetch.assert_not_called()

    @patch("agent.tools.lobbying._fetch_lda")
    def test_issues_invalid_year_clamped(self, mock_fetch, tool):
        mock_fetch.return_value = _lda_response([_filing(issue_codes=["TAX"])])
        r = tool.execute(mode="issues", issue_code="TAX", year=1990)
        assert r.success  # Clamps to current year


# ═══════════════════════════════════════════════════════════════════════════
# 5. Helper functions
# ═══════════════════════════════════════════════════════════════════════════


class TestHelpers:
    # -- _parse_filing --
    def test_parse_filing_basic(self):
        result = _parse_filing(_filing())
        assert result["registrant_name"] == "Lobby Corp"
        assert result["client_name"] == "TechCo"
        assert result["amount"] == 100000.0
        assert "HCR" in result["issue_codes"]

    def test_parse_filing_missing_registrant(self):
        f = _filing()
        f["registrant"] = None
        result = _parse_filing(f)
        assert result["registrant_name"] == "Unknown"

    def test_parse_filing_missing_client(self):
        f = _filing()
        f["client"] = None
        result = _parse_filing(f)
        assert result["client_name"] == "Unknown"

    def test_parse_filing_no_income(self):
        f = _filing(amount_income="0.00")
        f["expenses"] = "75000.00"
        result = _parse_filing(f)
        assert result["amount"] == 75000.0

    def test_parse_filing_no_amount(self):
        f = _filing(amount_income="0.00")
        f["expenses"] = "0.00"
        result = _parse_filing(f)
        assert result["amount"] == 0.0

    def test_parse_filing_invalid_amount(self):
        f = _filing()
        f["income"] = "not-a-number"
        f["expenses"] = "also-not"
        result = _parse_filing(f)
        assert result["amount"] == 0.0

    def test_parse_filing_no_activities(self):
        f = _filing()
        f["lobbying_activities"] = []
        result = _parse_filing(f)
        assert result["issue_codes"] == []

    def test_parse_filing_multiple_issues(self):
        f = _filing(issue_codes=["HCR", "TAX", "DEF"])
        result = _parse_filing(f)
        assert len(result["issue_codes"]) == 3

    # -- _detect_spend_anomaly --
    def test_anomaly_detected(self):
        result = _detect_spend_anomaly([100, 100, 100, 300])
        assert result["anomaly"] is True
        assert result["ratio"] >= 2.0

    def test_anomaly_not_detected(self):
        result = _detect_spend_anomaly([100, 100, 100, 100])
        assert result["anomaly"] is False
        assert result["ratio"] == pytest.approx(1.0)

    def test_anomaly_insufficient_data(self):
        result = _detect_spend_anomaly([100])
        assert result["anomaly"] is False
        assert result["ratio"] is None

    def test_anomaly_empty(self):
        result = _detect_spend_anomaly([])
        assert result["anomaly"] is False

    def test_anomaly_zero_average(self):
        result = _detect_spend_anomaly([0, 0, 0, 100])
        assert result["anomaly"] is False  # avg is 0

    def test_anomaly_custom_threshold(self):
        result = _detect_spend_anomaly([100, 100, 150], threshold_multiplier=1.2)
        assert result["anomaly"] is True  # 150/100 = 1.5 > 1.2


# ═══════════════════════════════════════════════════════════════════════════
# 6. LDA fetch helper
# ═══════════════════════════════════════════════════════════════════════════


class TestFetchLDA:
    @patch("agent.tools.lobbying.httpx.Client")
    def test_fetch_basic(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(_lda_response([_filing()]))
        result = _fetch_lda("filings", {"filing_year": "2024"}, "key")
        assert result is not None

    @patch("agent.tools.lobbying.httpx.Client")
    def test_fetch_with_auth(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(_lda_response())
        _fetch_lda("filings", {}, "my-token")
        # Check Authorization header was set
        headers = mock_client_cls.call_args[1].get("headers", {})
        assert "Authorization" in headers

    @patch("agent.tools.lobbying.httpx.Client")
    def test_fetch_no_auth(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(_lda_response())
        _fetch_lda("filings", {}, None)
        headers = mock_client_cls.call_args[1].get("headers", {})
        assert "Authorization" not in headers

    @patch("agent.tools.lobbying.httpx.Client")
    def test_fetch_http_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response({}, status=500)
        result = _fetch_lda("filings", {})
        assert result is None

    @patch("agent.tools.lobbying.httpx.Client")
    def test_fetch_timeout(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = httpx.ReadTimeout("timeout")
        result = _fetch_lda("filings", {})
        assert result is None

    @patch("agent.tools.lobbying.httpx.Client")
    def test_fetch_rate_limited(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response({}, status=429)
        result = _fetch_lda("filings", {})
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# 7. No-cache paths
# ═══════════════════════════════════════════════════════════════════════════


class TestNoCache:
    @patch("agent.tools.lobbying._fetch_lda")
    def test_search_no_cache(self, mock_fetch, tool_no_cache):
        mock_fetch.return_value = _lda_response([_filing()])
        r = tool_no_cache.execute(mode="search", registrant="Test")
        assert r.success

    @patch("agent.tools.lobbying._fetch_lda")
    def test_spending_no_cache(self, mock_fetch, tool_no_cache):
        mock_fetch.return_value = _lda_response([_filing()])
        r = tool_no_cache.execute(mode="spending", registrant="Test")
        assert r.success

    @patch("agent.tools.lobbying._fetch_lda")
    def test_issues_no_cache(self, mock_fetch, tool_no_cache):
        mock_fetch.return_value = _lda_response([_filing(issue_codes=["HCR"])])
        r = tool_no_cache.execute(mode="issues", issue_code="HCR")
        assert r.success


# ═══════════════════════════════════════════════════════════════════════════
# 8. API key handling
# ═══════════════════════════════════════════════════════════════════════════


class TestAPIKey:
    @patch("agent.tools.lobbying._fetch_lda")
    def test_works_without_key(self, mock_fetch, tool_no_key):
        # LDA API works without key (15 req/min)
        mock_fetch.return_value = _lda_response([_filing()])
        r = tool_no_key.execute(mode="search", registrant="Test")
        assert r.success

    @patch.dict("os.environ", {"TIRRA_LDA_API_KEY": "my-key"})
    def test_key_from_env(self):
        t = LobbyingTool(cache=None)
        assert t._api_key == "my-key"

    @patch.dict("os.environ", {"TIRRA_LDA_API_KEY": ""})
    def test_empty_key(self):
        t = LobbyingTool(cache=None)
        assert t._api_key is None

    @patch.dict("os.environ", {}, clear=True)
    def test_no_env_key(self):
        t = LobbyingTool(cache=None)
        assert t._api_key is None


# ═══════════════════════════════════════════════════════════════════════════
# 9. Tool metadata
# ═══════════════════════════════════════════════════════════════════════════


class TestToolMetadata:
    def test_tool_name(self, tool):
        assert tool.name == "lobbying"

    def test_tool_description(self, tool):
        assert "lobbying" in tool.description.lower()

    def test_parameters_schema(self, tool):
        props = tool.parameters["properties"]
        assert "mode" in props
        assert "registrant" in props
        assert "client" in props
        assert "issue_code" in props
        assert "year" in props
        assert "quarter" in props

    def test_required_params(self, tool):
        assert "mode" in tool.parameters["required"]


# ═══════════════════════════════════════════════════════════════════════════
# 10. Constants
# ═══════════════════════════════════════════════════════════════════════════


class TestConstants:
    def test_issue_areas_not_empty(self):
        assert len(ISSUE_AREAS) > 0

    def test_issue_areas_known_codes(self):
        assert "HCR" in ISSUE_AREAS
        assert "TAX" in ISSUE_AREAS
        assert "DEF" in ISSUE_AREAS


# ═══════════════════════════════════════════════════════════════════════════
# 11. Integration counts
# ═══════════════════════════════════════════════════════════════════════════


class TestIntegration:
    def test_tool_count(self):
        from agent.cli import build_tool_registry

        reg = build_tool_registry()
        # Was 60; commit 43de067 (2026-08-26) fixed nightlight_activity's
        # constructor kwarg mismatch (store= vs pipeline_store=) that silently
        # skipped its registration -- registry now correctly has 61 tools.
        assert len(reg.list_names()) == 61

    def test_arm_count(self):
        from agent.learning.bandit import DEFAULT_ARMS

        assert len(DEFAULT_ARMS) == 48
