"""
Edge-case tests for PatentFilingsTool (7b-H).

Coverage targets:
- Invalid / missing / boundary parameters
- Empty / malformed API responses
- Cache hit / miss paths
- Search: keyword, assignee, CPC class, date range combinations
- Trends: CPC class listing, yearly aggregation, trend direction
- Assignee: portfolio analysis, CPC distribution, filing velocity
- HTTP errors, timeouts
- Mode validation
- Helper functions (_parse_date, _year_range, _fetch_patents)
- Integration: tool count = 44, arm count = 32
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.tools.patent_filings import (
    _MAX_RESULTS,
    CPC_CLASSES,
    SIGNAL_CPC,
    VALID_MODES,
    PatentFilingsTool,
    _fetch_assignees,
    _fetch_patents,
    _parse_date,
    _year_range,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tool():
    cache = MagicMock()
    cache.get.return_value = None
    return PatentFilingsTool(cache=cache)


@pytest.fixture
def tool_no_cache():
    return PatentFilingsTool(cache=None)


def _patent(
    number: str = "US12345678",
    title: str = "Test Patent",
    date: str = "2025-01-15",
    assignee: str = "TestCorp",
    cpc: str = "G06N3/08",
) -> dict[str, Any]:
    return {
        "patent_number": number,
        "patent_title": title,
        "patent_date": date,
        "assignee_organization": assignee,
        "cpc_subgroup_id": cpc,
        "patent_abstract": "A method for...",
    }


def _api_response(
    patents: list[dict] | None = None,
    total: int | None = None,
) -> dict[str, Any]:
    p = patents or []
    return {
        "patents": p,
        "total_patent_count": total if total is not None else len(p),
        "count": len(p),
    }


def _mock_http_response(data: dict, status: int = 200) -> httpx.Response:
    r = MagicMock(spec=httpx.Response)
    r.status_code = status
    r.json.return_value = data
    return r


# ═══════════════════════════════════════════════════════════════════════════
# 1. Mode validation
# ═══════════════════════════════════════════════════════════════════════════


class TestModeValidation:
    def test_valid_modes(self, tool):
        assert {"search", "trends", "assignee"} == VALID_MODES

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
        assert "query" in r.output or "assignee" in r.output

    @patch("agent.tools.patent_filings._fetch_patents")
    def test_search_by_keyword(self, mock_fetch, tool):
        mock_fetch.return_value = _api_response([_patent()])
        r = tool.execute(mode="search", query="neural network")
        assert r.success
        assert r.data["mode"] == "search"
        assert r.data["total_count"] >= 1

    @patch("agent.tools.patent_filings._fetch_patents")
    def test_search_by_assignee(self, mock_fetch, tool):
        mock_fetch.return_value = _api_response([_patent(assignee="Apple")])
        r = tool.execute(mode="search", assignee="Apple")
        assert r.success

    @patch("agent.tools.patent_filings._fetch_patents")
    def test_search_by_cpc(self, mock_fetch, tool):
        mock_fetch.return_value = _api_response([_patent(cpc="G06N3/08")])
        r = tool.execute(mode="search", cpc_class="G06N")
        assert r.success

    @patch("agent.tools.patent_filings._fetch_patents")
    def test_search_with_dates(self, mock_fetch, tool):
        mock_fetch.return_value = _api_response([_patent()])
        r = tool.execute(mode="search", query="AI", date_from="2024-01-01", date_to="2025-01-01")
        assert r.success

    @patch("agent.tools.patent_filings._fetch_patents")
    def test_search_empty_results(self, mock_fetch, tool):
        mock_fetch.return_value = _api_response([])
        r = tool.execute(mode="search", query="xyzzy123")
        assert r.success
        assert r.data["total_count"] == 0

    @patch("agent.tools.patent_filings._fetch_patents")
    def test_search_api_unavailable(self, mock_fetch, tool):
        mock_fetch.return_value = None
        r = tool.execute(mode="search", query="test")
        assert not r.success
        assert "unavailable" in r.output.lower()

    @patch("agent.tools.patent_filings._fetch_patents")
    def test_search_limit_clamped(self, mock_fetch, tool):
        mock_fetch.return_value = _api_response([_patent()])
        r = tool.execute(mode="search", query="test", limit=999)
        assert r.success

    @patch("agent.tools.patent_filings._fetch_patents")
    def test_search_cache_hit(self, mock_fetch, tool):
        tool._cache.get.return_value = {
            "patents": [_patent()],
            "total_count": 1,
            "returned": 1,
        }
        r = tool.execute(mode="search", query="cached")
        assert r.success
        assert "(cached)" in r.output
        mock_fetch.assert_not_called()

    @patch("agent.tools.patent_filings._fetch_patents")
    def test_search_cache_write(self, mock_fetch, tool):
        mock_fetch.return_value = _api_response([_patent()])
        tool.execute(mode="search", query="save")
        tool._cache.put.assert_called()

    @patch("agent.tools.patent_filings._fetch_patents")
    def test_search_assignee_list(self, mock_fetch, tool):
        # assignee_organization can be a list
        p = _patent()
        p["assignee_organization"] = ["Apple Inc.", "Apple Labs"]
        mock_fetch.return_value = _api_response([p])
        r = tool.execute(mode="search", query="test")
        assert r.success
        assert "Apple Inc." in r.output

    @patch("agent.tools.patent_filings._fetch_patents")
    def test_search_cpc_list(self, mock_fetch, tool):
        p = _patent()
        p["cpc_subgroup_id"] = ["G06N3/08", "H04L67/10"]
        mock_fetch.return_value = _api_response([p])
        r = tool.execute(mode="search", query="test")
        assert r.success


# ═══════════════════════════════════════════════════════════════════════════
# 3. Trends mode
# ═══════════════════════════════════════════════════════════════════════════


class TestTrendsMode:
    def test_trends_no_cpc(self, tool):
        r = tool.execute(mode="trends")
        assert r.success
        assert "signal_classes" in r.data
        assert r.data["mode"] == "trends"

    @patch("agent.tools.patent_filings._fetch_patents")
    def test_trends_with_cpc(self, mock_fetch, tool):
        patents = [
            _patent(number=f"US{i}", date=f"202{y}-01-01") for i, y in enumerate([1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
        ]
        mock_fetch.return_value = _api_response(patents, total=100)
        r = tool.execute(mode="trends", cpc_class="G06N")
        assert r.success
        assert "yearly_counts" in r.data
        assert r.data["cpc_class"] == "G06N"

    @patch("agent.tools.patent_filings._fetch_patents")
    def test_trends_accelerating(self, mock_fetch, tool):
        patents = [_patent(number=f"US{i}", date="2020-01-01") for i in range(2)] + [
            _patent(number=f"US{i + 100}", date="2024-01-01") for i in range(10)
        ]
        mock_fetch.return_value = _api_response(patents, total=12)
        r = tool.execute(mode="trends", cpc_class="H01L")
        assert r.success
        assert "accelerating" in r.output.lower()

    @patch("agent.tools.patent_filings._fetch_patents")
    def test_trends_decelerating(self, mock_fetch, tool):
        patents = [_patent(number=f"US{i}", date="2020-01-01") for i in range(10)] + [
            _patent(number=f"US{i + 100}", date="2024-01-01") for i in range(2)
        ]
        mock_fetch.return_value = _api_response(patents, total=12)
        r = tool.execute(mode="trends", cpc_class="H01L")
        assert r.success
        assert "decelerating" in r.output.lower()

    @patch("agent.tools.patent_filings._fetch_patents")
    def test_trends_api_unavailable(self, mock_fetch, tool):
        mock_fetch.return_value = None
        r = tool.execute(mode="trends", cpc_class="G06N")
        assert not r.success

    @patch("agent.tools.patent_filings._fetch_patents")
    def test_trends_empty_results(self, mock_fetch, tool):
        mock_fetch.return_value = _api_response([])
        r = tool.execute(mode="trends", cpc_class="Z99")
        assert r.success
        assert "No filings found" in r.output

    @patch("agent.tools.patent_filings._fetch_patents")
    def test_trends_cache_hit(self, mock_fetch, tool):
        tool._cache.get.return_value = {
            "cpc_class": "G06N",
            "yearly_counts": {"2024": 50},
            "total_count": 50,
            "sample_size": 50,
        }
        r = tool.execute(mode="trends", cpc_class="G06N")
        assert r.success
        assert "(cached)" in r.output
        mock_fetch.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════
# 4. Assignee mode
# ═══════════════════════════════════════════════════════════════════════════


class TestAssigneeMode:
    def test_assignee_no_name(self, tool):
        r = tool.execute(mode="assignee")
        assert not r.success
        assert "assignee" in r.output.lower()

    @patch("agent.tools.patent_filings._fetch_patents")
    def test_assignee_basic(self, mock_fetch, tool):
        patents = [_patent(number=f"US{i}", date=f"2024-0{i + 1}-01", cpc=f"G06N{i}/00") for i in range(5)]
        mock_fetch.return_value = _api_response(patents, total=100)
        r = tool.execute(mode="assignee", assignee="Google")
        assert r.success
        assert r.data["mode"] == "assignee"
        assert r.data["assignee"] == "Google"
        assert "cpc_distribution" in r.data
        assert "yearly_velocity" in r.data

    @patch("agent.tools.patent_filings._fetch_patents")
    def test_assignee_api_unavailable(self, mock_fetch, tool):
        mock_fetch.return_value = None
        r = tool.execute(mode="assignee", assignee="Apple")
        assert not r.success

    @patch("agent.tools.patent_filings._fetch_patents")
    def test_assignee_empty_results(self, mock_fetch, tool):
        mock_fetch.return_value = _api_response([])
        r = tool.execute(mode="assignee", assignee="NonExistentCorp")
        assert r.success
        assert r.data["total_patents"] == 0

    @patch("agent.tools.patent_filings._fetch_patents")
    def test_assignee_cpc_list_handling(self, mock_fetch, tool):
        p = _patent()
        p["cpc_subgroup_id"] = ["G06N3/08", "H04L67/10"]
        mock_fetch.return_value = _api_response([p])
        r = tool.execute(mode="assignee", assignee="TestCorp")
        assert r.success
        assert len(r.data["cpc_distribution"]) >= 2

    @patch("agent.tools.patent_filings._fetch_patents")
    def test_assignee_cache_hit(self, mock_fetch, tool):
        tool._cache.get.return_value = {
            "assignee": "Google",
            "total_patents": 50,
            "returned": 10,
            "patents": [_patent()],
            "cpc_distribution": {"G06N": 5},
            "yearly_velocity": {"2024": 10},
        }
        r = tool.execute(mode="assignee", assignee="Google")
        assert r.success
        assert "(cached)" in r.output
        mock_fetch.assert_not_called()

    @patch("agent.tools.patent_filings._fetch_patents")
    def test_assignee_limit(self, mock_fetch, tool):
        mock_fetch.return_value = _api_response([_patent()])
        r = tool.execute(mode="assignee", assignee="Test", limit=5)
        assert r.success


# ═══════════════════════════════════════════════════════════════════════════
# 5. Helper functions
# ═══════════════════════════════════════════════════════════════════════════


class TestHelpers:
    def test_parse_date_iso(self):
        assert _parse_date("2025-03-15") == "2025-03-15"

    def test_parse_date_slash(self):
        assert _parse_date("2025/03/15") == "2025-03-15"

    def test_parse_date_us_format(self):
        assert _parse_date("03/15/2025") == "2025-03-15"

    def test_parse_date_invalid(self):
        assert _parse_date("not-a-date") == "not-a-date"

    def test_year_range(self):
        start, end = _year_range(5)
        assert len(start) == 10  # YYYY-MM-DD
        assert len(end) == 10

    def test_year_range_default(self):
        start, end = _year_range()
        assert start < end


class TestFetchPatents:
    @patch("agent.tools.patent_filings.httpx.Client")
    def test_fetch_basic(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_http_response(_api_response([_patent()]))
        result = _fetch_patents({"_text_any": {"patent_abstract": "test"}}, ["patent_number"])
        assert result is not None

    @patch("agent.tools.patent_filings.httpx.Client")
    def test_fetch_http_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_http_response({}, status=500)
        result = _fetch_patents({}, [])
        assert result is None

    @patch("agent.tools.patent_filings.httpx.Client")
    def test_fetch_timeout(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = httpx.ReadTimeout("timeout")
        result = _fetch_patents({}, [])
        assert result is None


class TestFetchAssignees:
    @patch("agent.tools.patent_filings.httpx.Client")
    def test_fetch_assignees_basic(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_http_response({"assignees": []})
        result = _fetch_assignees({"_text_any": {"assignee_organization": "test"}}, [])
        assert result is not None

    @patch("agent.tools.patent_filings.httpx.Client")
    def test_fetch_assignees_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_http_response({}, status=403)
        result = _fetch_assignees({}, [])
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# 6. No-cache paths
# ═══════════════════════════════════════════════════════════════════════════


class TestNoCache:
    @patch("agent.tools.patent_filings._fetch_patents")
    def test_search_no_cache(self, mock_fetch, tool_no_cache):
        mock_fetch.return_value = _api_response([_patent()])
        r = tool_no_cache.execute(mode="search", query="test")
        assert r.success

    @patch("agent.tools.patent_filings._fetch_patents")
    def test_trends_no_cache(self, mock_fetch, tool_no_cache):
        mock_fetch.return_value = _api_response([_patent()])
        r = tool_no_cache.execute(mode="trends", cpc_class="G06N")
        assert r.success

    @patch("agent.tools.patent_filings._fetch_patents")
    def test_assignee_no_cache(self, mock_fetch, tool_no_cache):
        mock_fetch.return_value = _api_response([_patent()])
        r = tool_no_cache.execute(mode="assignee", assignee="Test")
        assert r.success


# ═══════════════════════════════════════════════════════════════════════════
# 7. Tool metadata
# ═══════════════════════════════════════════════════════════════════════════


class TestToolMetadata:
    def test_tool_name(self, tool):
        assert tool.name == "patent_filings"

    def test_tool_description(self, tool):
        assert "patent" in tool.description.lower()

    def test_parameters_schema(self, tool):
        props = tool.parameters["properties"]
        assert "mode" in props
        assert "query" in props
        assert "assignee" in props
        assert "cpc_class" in props

    def test_required_params(self, tool):
        assert "mode" in tool.parameters["required"]


# ═══════════════════════════════════════════════════════════════════════════
# 8. Constants
# ═══════════════════════════════════════════════════════════════════════════


class TestConstants:
    def test_cpc_classes_not_empty(self):
        assert len(CPC_CLASSES) > 0

    def test_signal_cpc_not_empty(self):
        assert len(SIGNAL_CPC) > 0

    def test_signal_cpc_known_classes(self):
        assert "G06N" in SIGNAL_CPC
        assert "H01L" in SIGNAL_CPC

    def test_max_results(self):
        assert _MAX_RESULTS == 50


# ═══════════════════════════════════════════════════════════════════════════
# 9. Integration counts
# ═══════════════════════════════════════════════════════════════════════════


class TestIntegration:
    def test_tool_count(self):
        from agent.cli import build_tool_registry

        reg = build_tool_registry()
        assert len(reg.list_names()) == 60

    def test_arm_count(self):
        from agent.learning.bandit import DEFAULT_ARMS

        assert len(DEFAULT_ARMS) == 48
