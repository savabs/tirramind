"""
Edge case tests for FoiaRequestsTool (FOIA/FOI request log monitor).

Covers: mode validation, _parse_date (ISO 8601, date-only, fractional seconds,
None, empty, malformed), _normalize_status (known statuses, unknown, None,
empty), _normalize_muckrock (full record, missing fields, non-dict agency,
non-dict user), _normalize_wdtk (full record, missing fields, non-dict
public_body, non-dict user), _format_request (full, minimal, no URL),
_cache_key (deterministic, different params differ), _fetch_muckrock
(paginated, single page, empty, timeout, 429, 500, invalid JSON, non-dict
response, non-list non-dict), _fetch_wdtk (normal, dict response, list
response, empty, timeout, 429, 500, invalid JSON), search mode (results,
empty, date filter, jurisdiction filter, cache hit/miss, missing query),
agency_activity mode (surge detection, no surge, no baseline, empty results,
missing agency, agency name filtering), entity_cluster mode (convergence,
no convergence, multi-jurisdiction, empty, missing query), input validation
(invalid mode, bounds on days_back and limit, jurisdiction normalization),
tool metadata (name, description, parameters, required), integration of
count assertions (37 tools, 25 arms).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone; UTC = timezone.utc
from typing import Any
from unittest.mock import MagicMock, patch

import httpx

from agent.tools.base import ToolResult
from agent.tools.foia_requests import (
    VALID_MODES,
    FoiaRequestsTool,
    _cache_key,
    _fetch_muckrock,
    _fetch_wdtk,
    _format_request,
    _normalize_muckrock,
    _normalize_status,
    _normalize_wdtk,
    _parse_date,
)

# ── Timestamps ───────────────────────────────────────────────

NOW = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)
YESTERDAY_S = (NOW - timedelta(days=1)).strftime("%Y-%m-%d")
LAST_WEEK_S = (NOW - timedelta(days=7)).strftime("%Y-%m-%d")
LAST_MONTH_S = (NOW - timedelta(days=31)).strftime("%Y-%m-%d")
OLD_DATE_S = (NOW - timedelta(days=200)).strftime("%Y-%m-%d")


# ── Mock Data Factories ─────────────────────────────────────


def _muckrock_record(
    *,
    title: str = "Request about Boeing safety inspections",
    agency: dict | str = None,
    status: str = "done",
    date_submitted: str = YESTERDAY_S,
    date_done: str = "",
    user: dict | str = None,
    url: str = "https://www.muckrock.com/foi/united-states-of-america-10/test-12345/",
) -> dict[str, Any]:
    if agency is None:
        agency = {"name": "Federal Aviation Administration"}
    if user is None:
        user = {"username": "journalist1"}
    return {
        "title": title,
        "agency": agency,
        "agency_name": agency.get("name") if isinstance(agency, dict) else agency,
        "status": status,
        "date_submitted": date_submitted,
        "datetime_submitted": date_submitted,
        "date_done": date_done,
        "datetime_done": date_done,
        "jurisdiction_name": "United States of America",
        "absolute_url": url,
        "url": url,
        "user": user,
    }


def _wdtk_record(
    *,
    title: str = "FOI request about Boeing contracts",
    public_body: dict | str = None,
    described_state: str = "successful",
    created_at: str = YESTERDAY_S,
    url: str = "https://www.whatdotheyknow.com/request/test_12345",
    user: dict | str = None,
) -> dict[str, Any]:
    if public_body is None:
        public_body = {"name": "Ministry of Defence"}
    if user is None:
        user = {"name": "ResearcherA"}
    return {
        "title": title,
        "public_body": public_body,
        "described_state": described_state,
        "created_at": created_at,
        "url": url,
        "user": user,
    }


def _paginated_response(results: list[dict], *, count: int = 0, next_url: str | None = None) -> dict[str, Any]:
    return {
        "count": count or len(results),
        "next": next_url,
        "previous": None,
        "results": results,
    }


def _mock_response(data: Any, status_code: int = 200) -> httpx.Response:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(),
            response=resp,
        )
    return resp


# ═══════════════════════════════════════════════════════════
# 1. _parse_date
# ═══════════════════════════════════════════════════════════


class TestParseDate:
    def test_date_only(self):
        dt = _parse_date("2026-03-15")
        assert dt is not None
        assert dt.year == 2026
        assert dt.month == 3
        assert dt.day == 15
        assert dt.tzinfo == UTC

    def test_iso_datetime(self):
        dt = _parse_date("2026-03-15T14:30:00")
        assert dt is not None
        assert dt.hour == 14
        assert dt.minute == 30

    def test_iso_with_tz(self):
        dt = _parse_date("2026-03-15T14:30:00+00:00")
        assert dt is not None
        assert dt.day == 15

    def test_iso_with_fractional(self):
        dt = _parse_date("2026-03-15T14:30:00.123+00:00")
        assert dt is not None

    def test_none(self):
        assert _parse_date(None) is None

    def test_empty(self):
        assert _parse_date("") is None

    def test_malformed(self):
        assert _parse_date("not-a-date") is None

    def test_partial_garbage(self):
        # Date portion valid, rest garbage
        dt = _parse_date("2026-03-15GARBAGE")
        assert dt is not None  # Fallback to first 10 chars
        assert dt.day == 15

    def test_very_long_string(self):
        # Should not crash, uses [:32] slicing
        dt = _parse_date("2026-03-15T14:30:00.000+00:00" + "x" * 200)
        assert dt is not None


# ═══════════════════════════════════════════════════════════
# 2. _normalize_status
# ═══════════════════════════════════════════════════════════


class TestNormalizeStatus:
    def test_known_statuses(self):
        assert _normalize_status("done") == "Completed"
        assert _normalize_status("submitted") == "Submitted"
        assert _normalize_status("rejected") == "Rejected"
        assert _normalize_status("ack") == "Acknowledged"
        assert _normalize_status("no_docs") == "No Responsive Docs"
        assert _normalize_status("lawsuit") == "Lawsuit Filed"

    def test_case_insensitive(self):
        assert _normalize_status("DONE") == "Completed"
        assert _normalize_status("Done") == "Completed"

    def test_with_whitespace(self):
        assert _normalize_status("  done  ") == "Completed"

    def test_unknown_status(self):
        assert _normalize_status("weird_status") == "Weird_Status"

    def test_none(self):
        assert _normalize_status(None) == "Unknown"

    def test_empty(self):
        assert _normalize_status("") == "Unknown"


# ═══════════════════════════════════════════════════════════
# 3. _normalize_muckrock
# ═══════════════════════════════════════════════════════════


class TestNormalizeMuckrock:
    def test_full_record(self):
        rec = _muckrock_record()
        normed = _normalize_muckrock(rec)
        assert normed["title"] == "Request about Boeing safety inspections"
        assert normed["agency"] == "Federal Aviation Administration"
        assert normed["status"] == "Completed"
        assert normed["source"] == "muckrock"
        assert normed["jurisdiction"] == "United States of America"
        assert normed["requester"] == "journalist1"
        assert "muckrock.com" in normed["url"]

    def test_missing_title(self):
        rec = _muckrock_record(title="")
        rec["title"] = None
        normed = _normalize_muckrock(rec)
        assert normed["title"] == "Untitled"

    def test_agency_as_string(self):
        rec = _muckrock_record(agency="SEC")
        normed = _normalize_muckrock(rec)
        assert normed["agency"] == "SEC"

    def test_agency_as_int(self):
        rec = _muckrock_record()
        rec["agency"] = 12345
        rec["agency_name"] = None
        normed = _normalize_muckrock(rec)
        assert normed["agency"] == "12345"

    def test_user_as_string(self):
        rec = _muckrock_record(user="john_doe")
        normed = _normalize_muckrock(rec)
        assert normed["requester"] == "john_doe"

    def test_no_dates(self):
        rec = _muckrock_record(date_submitted="", date_done="")
        rec["datetime_submitted"] = ""
        rec["date_submitted"] = ""
        normed = _normalize_muckrock(rec)
        assert normed["date_filed"] == ""

    def test_whitespace_stripping(self):
        rec = _muckrock_record(title="  Padded Title  ", agency={"name": "  FBI  "})
        normed = _normalize_muckrock(rec)
        assert normed["title"] == "Padded Title"
        assert normed["agency"] == "FBI"


# ═══════════════════════════════════════════════════════════
# 4. _normalize_wdtk
# ═══════════════════════════════════════════════════════════


class TestNormalizeWdtk:
    def test_full_record(self):
        rec = _wdtk_record()
        normed = _normalize_wdtk(rec)
        assert normed["title"] == "FOI request about Boeing contracts"
        assert normed["agency"] == "Ministry of Defence"
        assert normed["status"] == "Successful"
        assert normed["source"] == "wdtk"
        assert normed["jurisdiction"] == "UK"
        assert normed["requester"] == "ResearcherA"

    def test_underscore_status(self):
        rec = _wdtk_record(described_state="waiting_response")
        normed = _normalize_wdtk(rec)
        assert normed["status"] == "Waiting Response"

    def test_public_body_as_string(self):
        rec = _wdtk_record(public_body="Home Office")
        normed = _normalize_wdtk(rec)
        assert normed["agency"] == "Home Office"

    def test_missing_public_body(self):
        rec = _wdtk_record()
        rec["public_body"] = None
        normed = _normalize_wdtk(rec)
        # Falls through to str(None)
        assert "None" in normed["agency"] or "Unknown" in normed["agency"]

    def test_user_as_string(self):
        rec = _wdtk_record(user="someone")
        normed = _normalize_wdtk(rec)
        assert normed["requester"] == "anonymous"  # str user → anonymous

    def test_missing_title(self):
        rec = _wdtk_record(title="")
        rec["title"] = None
        normed = _normalize_wdtk(rec)
        assert normed["title"] == "Untitled"


# ═══════════════════════════════════════════════════════════
# 5. _format_request
# ═══════════════════════════════════════════════════════════


class TestFormatRequest:
    def test_full_format(self):
        rec = {
            "title": "Test Request",
            "agency": "FBI",
            "jurisdiction": "US",
            "status": "Completed",
            "source": "muckrock",
            "date_filed": "2026-03-15",
            "url": "https://example.com/request/1",
        }
        out = _format_request(rec, index=1)
        assert "1. [2026-03-15]" in out
        assert "Test Request" in out
        assert "FBI" in out
        assert "US" in out
        assert "Completed" in out
        assert "muckrock" in out
        assert "https://example.com/request/1" in out

    def test_no_url(self):
        rec = {
            "title": "Test Request",
            "agency": "EPA",
            "jurisdiction": "US",
            "status": "Submitted",
            "source": "muckrock",
            "date_filed": "2026-01-01",
            "url": "",
        }
        out = _format_request(rec, index=5)
        assert "URL:" not in out  # No URL line emitted
        assert "5." in out

    def test_missing_date(self):
        rec = {
            "title": "Missing Date",
            "agency": "DOD",
            "jurisdiction": "US",
            "status": "Processing",
            "source": "muckrock",
            "date_filed": "",
            "url": "",
        }
        out = _format_request(rec, index=1)
        assert "[?]" in out


# ═══════════════════════════════════════════════════════════
# 6. _cache_key
# ═══════════════════════════════════════════════════════════


class TestCacheKey:
    def test_deterministic(self):
        k1 = _cache_key("search", query="boeing", days="90")
        k2 = _cache_key("search", query="boeing", days="90")
        assert k1 == k2

    def test_different_params_differ(self):
        k1 = _cache_key("search", query="boeing")
        k2 = _cache_key("search", query="airbus")
        assert k1 != k2

    def test_different_modes_differ(self):
        k1 = _cache_key("search", query="boeing")
        k2 = _cache_key("entity", query="boeing")
        assert k1 != k2

    def test_starts_with_prefix(self):
        k = _cache_key("search", query="test")
        assert k.startswith("foia:search:")

    def test_empty_values_ignored(self):
        k1 = _cache_key("search", query="test", empty="")
        k2 = _cache_key("search", query="test")
        assert k1 == k2


# ═══════════════════════════════════════════════════════════
# 7. _fetch_muckrock
# ═══════════════════════════════════════════════════════════


class TestFetchMuckrock:
    @patch("agent.tools.foia_requests.httpx.get")
    def test_single_page(self, mock_get):
        data = _paginated_response([_muckrock_record()])
        mock_get.return_value = _mock_response(data)
        results = _fetch_muckrock("foia", {"q": "boeing"})
        assert len(results) == 1
        assert results[0]["title"] == "Request about Boeing safety inspections"

    @patch("agent.tools.foia_requests.httpx.get")
    def test_multi_page(self, mock_get):
        page1 = _paginated_response(
            [_muckrock_record(title="Page 1")],
            count=2,
            next_url="https://www.muckrock.com/api_v1/foia/?page=2",
        )
        page2 = _paginated_response([_muckrock_record(title="Page 2")])
        mock_get.side_effect = [
            _mock_response(page1),
            _mock_response(page2),
        ]
        results = _fetch_muckrock("foia", {"q": "test"})
        assert len(results) == 2

    @patch("agent.tools.foia_requests.httpx.get")
    def test_empty_results(self, mock_get):
        mock_get.return_value = _mock_response(_paginated_response([]))
        results = _fetch_muckrock("foia", {"q": "zzz_nothing"})
        assert results == []

    @patch("agent.tools.foia_requests.httpx.get")
    def test_timeout(self, mock_get):
        mock_get.side_effect = httpx.TimeoutException("timeout")
        results = _fetch_muckrock("foia", {"q": "test"})
        assert results == []

    @patch("agent.tools.foia_requests.httpx.get")
    def test_rate_limit_429(self, mock_get):
        resp = _mock_response({}, status_code=429)
        resp.status_code = 429
        resp.raise_for_status = MagicMock()  # 429 checked before raise
        mock_get.return_value = resp
        results = _fetch_muckrock("foia", {"q": "test"})
        assert results == []

    @patch("agent.tools.foia_requests.httpx.get")
    def test_server_error_500(self, mock_get):
        mock_get.return_value = _mock_response({}, status_code=500)
        results = _fetch_muckrock("foia", {"q": "test"})
        assert results == []

    @patch("agent.tools.foia_requests.httpx.get")
    def test_invalid_json(self, mock_get):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = resp
        results = _fetch_muckrock("foia", {"q": "test"})
        assert results == []

    @patch("agent.tools.foia_requests.httpx.get")
    def test_list_response(self, mock_get):
        """Some endpoints may return a bare list."""
        mock_get.return_value = _mock_response([_muckrock_record()])
        results = _fetch_muckrock("foia", {"q": "test"})
        assert len(results) == 1

    @patch("agent.tools.foia_requests.httpx.get")
    def test_non_dict_non_list_response(self, mock_get):
        mock_get.return_value = _mock_response("just a string")
        results = _fetch_muckrock("foia", {"q": "test"})
        assert results == []

    @patch("agent.tools.foia_requests.httpx.get")
    def test_max_pages_respected(self, mock_get):
        """Should stop after max_pages even if next URL keeps appearing."""
        page = _paginated_response(
            [_muckrock_record()],
            next_url="https://www.muckrock.com/api_v1/foia/?page=999",
        )
        mock_get.return_value = _mock_response(page)
        results = _fetch_muckrock("foia", {"q": "test"}, max_pages=2)
        assert mock_get.call_count == 2

    @patch("agent.tools.foia_requests.httpx.get")
    def test_connection_error(self, mock_get):
        mock_get.side_effect = httpx.ConnectError("Connection refused")
        results = _fetch_muckrock("foia", {"q": "test"})
        assert results == []


# ═══════════════════════════════════════════════════════════
# 8. _fetch_wdtk
# ═══════════════════════════════════════════════════════════


class TestFetchWdtk:
    @patch("agent.tools.foia_requests.httpx.get")
    def test_list_response(self, mock_get):
        mock_get.return_value = _mock_response([_wdtk_record()])
        results = _fetch_wdtk("boeing")
        assert len(results) == 1

    @patch("agent.tools.foia_requests.httpx.get")
    def test_dict_response_with_requests(self, mock_get):
        mock_get.return_value = _mock_response({"requests": [_wdtk_record()]})
        results = _fetch_wdtk("boeing")
        assert len(results) == 1

    @patch("agent.tools.foia_requests.httpx.get")
    def test_dict_response_with_results(self, mock_get):
        mock_get.return_value = _mock_response({"results": [_wdtk_record()]})
        results = _fetch_wdtk("boeing")
        assert len(results) == 1

    @patch("agent.tools.foia_requests.httpx.get")
    def test_empty_response(self, mock_get):
        mock_get.return_value = _mock_response([])
        results = _fetch_wdtk("zzz_nothing")
        assert results == []

    @patch("agent.tools.foia_requests.httpx.get")
    def test_timeout(self, mock_get):
        mock_get.side_effect = httpx.TimeoutException("timeout")
        results = _fetch_wdtk("test")
        assert results == []

    @patch("agent.tools.foia_requests.httpx.get")
    def test_rate_limit_429(self, mock_get):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 429
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp
        results = _fetch_wdtk("test")
        assert results == []

    @patch("agent.tools.foia_requests.httpx.get")
    def test_server_error_500(self, mock_get):
        mock_get.return_value = _mock_response({}, status_code=500)
        results = _fetch_wdtk("test")
        assert results == []

    @patch("agent.tools.foia_requests.httpx.get")
    def test_invalid_json(self, mock_get):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.side_effect = ValueError("bad json")
        mock_get.return_value = resp
        results = _fetch_wdtk("test")
        assert results == []

    @patch("agent.tools.foia_requests.httpx.get")
    def test_max_results_respected(self, mock_get):
        records = [_wdtk_record(title=f"Req {i}") for i in range(10)]
        mock_get.return_value = _mock_response(records)
        results = _fetch_wdtk("test", max_results=3)
        assert len(results) == 3

    @patch("agent.tools.foia_requests.httpx.get")
    def test_non_list_non_dict(self, mock_get):
        mock_get.return_value = _mock_response(42)
        results = _fetch_wdtk("test")
        assert results == []


# ═══════════════════════════════════════════════════════════
# 9. Tool — Input Validation
# ═══════════════════════════════════════════════════════════


class TestToolValidation:
    def setup_method(self):
        self.tool = FoiaRequestsTool(cache=None)

    def test_invalid_mode(self):
        r = self.tool.execute(mode="bogus")
        assert not r.success
        assert "Invalid mode" in r.output

    def test_empty_mode(self):
        r = self.tool.execute(mode="")
        assert not r.success

    def test_search_missing_query(self):
        r = self.tool.execute(mode="search")
        assert not r.success
        assert "query" in r.output.lower()

    def test_agency_activity_missing_agency(self):
        r = self.tool.execute(mode="agency_activity")
        assert not r.success
        assert "agency" in r.output.lower()

    def test_entity_cluster_missing_query(self):
        r = self.tool.execute(mode="entity_cluster")
        assert not r.success
        assert "query" in r.output.lower()

    def test_days_back_clamped_low(self):
        # Should not error, just clamp to 1
        r = self.tool.execute(mode="search", query="test", days_back=-5)
        # Will proceed (may fail on network, but shouldn't fail on validation)
        # We just verify it didn't crash on days_back processing
        assert isinstance(r, ToolResult)

    def test_days_back_clamped_high(self):
        r = self.tool.execute(mode="search", query="test", days_back=9999)
        assert isinstance(r, ToolResult)

    def test_limit_clamped(self):
        r = self.tool.execute(mode="search", query="test", limit=-1)
        assert isinstance(r, ToolResult)
        r = self.tool.execute(mode="search", query="test", limit=9999)
        assert isinstance(r, ToolResult)

    def test_jurisdiction_normalization(self):
        r = self.tool.execute(mode="search", query="test", jurisdiction="INVALID")
        # Should default to "all", not crash
        assert isinstance(r, ToolResult)

    def test_jurisdiction_us(self):
        r = self.tool.execute(mode="search", query="test", jurisdiction="US")
        assert isinstance(r, ToolResult)

    def test_jurisdiction_uk(self):
        r = self.tool.execute(mode="search", query="test", jurisdiction="UK")
        assert isinstance(r, ToolResult)


# ═══════════════════════════════════════════════════════════
# 10. Tool — Search Mode
# ═══════════════════════════════════════════════════════════


class TestSearchMode:
    def setup_method(self):
        self.cache = MagicMock()
        self.cache.get.return_value = None
        self.tool = FoiaRequestsTool(cache=self.cache)

    @patch("agent.tools.foia_requests._fetch_wdtk", return_value=[])
    @patch("agent.tools.foia_requests._fetch_muckrock")
    def test_search_with_results(self, mock_mr, mock_wdtk):
        mock_mr.return_value = [_muckrock_record()]
        r = self.tool.execute(mode="search", query="boeing", days_back=365)
        assert r.success
        assert "boeing" in r.output.lower() or "Boeing" in r.output
        assert "Federal Aviation" in r.output
        self.cache.put.assert_called_once()

    @patch("agent.tools.foia_requests._fetch_wdtk", return_value=[])
    @patch("agent.tools.foia_requests._fetch_muckrock", return_value=[])
    def test_search_empty(self, mock_mr, mock_wdtk):
        r = self.tool.execute(mode="search", query="zzz_nothing", days_back=365)
        assert r.success
        assert "No matching" in r.output or "0" in r.output

    @patch("agent.tools.foia_requests._fetch_wdtk")
    @patch("agent.tools.foia_requests._fetch_muckrock")
    def test_search_both_sources(self, mock_mr, mock_wdtk):
        mock_mr.return_value = [_muckrock_record()]
        mock_wdtk.return_value = [_wdtk_record()]
        r = self.tool.execute(mode="search", query="boeing", jurisdiction="all", days_back=365)
        assert r.success
        assert "MuckRock" in r.output
        assert "WDTK" in r.output

    @patch("agent.tools.foia_requests._fetch_wdtk", return_value=[])
    @patch("agent.tools.foia_requests._fetch_muckrock")
    def test_search_us_only(self, mock_mr, mock_wdtk):
        mock_mr.return_value = [_muckrock_record()]
        r = self.tool.execute(mode="search", query="test", jurisdiction="us", days_back=365)
        assert r.success
        mock_wdtk.assert_not_called()

    @patch("agent.tools.foia_requests._fetch_muckrock", return_value=[])
    @patch("agent.tools.foia_requests._fetch_wdtk")
    def test_search_uk_only(self, mock_wdtk, mock_mr):
        mock_wdtk.return_value = [_wdtk_record()]
        r = self.tool.execute(mode="search", query="test", jurisdiction="uk", days_back=365)
        assert r.success
        mock_mr.assert_not_called()

    @patch("agent.tools.foia_requests._fetch_wdtk", return_value=[])
    @patch("agent.tools.foia_requests._fetch_muckrock")
    def test_search_date_filter(self, mock_mr, mock_wdtk):
        """Records outside the date window should be excluded."""
        mock_mr.return_value = [
            _muckrock_record(title="Recent", date_submitted=YESTERDAY_S),
            _muckrock_record(title="Old", date_submitted=OLD_DATE_S),
        ]
        r = self.tool.execute(mode="search", query="test", days_back=30)
        assert r.success
        assert "Recent" in r.output
        assert "Old" not in r.output

    def test_search_cache_hit(self):
        self.cache.get.return_value = "cached output"
        r = self.tool.execute(mode="search", query="boeing", days_back=90)
        assert r.success
        assert r.output == "cached output"

    @patch("agent.tools.foia_requests._fetch_wdtk", return_value=[])
    @patch("agent.tools.foia_requests._fetch_muckrock")
    def test_search_status_breakdown(self, mock_mr, mock_wdtk):
        mock_mr.return_value = [
            _muckrock_record(status="done"),
            _muckrock_record(title="Another", status="submitted"),
        ]
        r = self.tool.execute(mode="search", query="test", days_back=365)
        assert r.success
        assert "Status breakdown" in r.output
        assert "Completed" in r.output
        assert "Submitted" in r.output


# ═══════════════════════════════════════════════════════════
# 11. Tool — Agency Activity Mode
# ═══════════════════════════════════════════════════════════


class TestAgencyActivityMode:
    def setup_method(self):
        self.cache = MagicMock()
        self.cache.get.return_value = None
        self.tool = FoiaRequestsTool(cache=self.cache)

    @patch("agent.tools.foia_requests._fetch_muckrock")
    def test_surge_detected(self, mock_mr):
        """Many recent requests vs few baseline → surge."""
        recent_recs = [
            _muckrock_record(
                title=f"Recent {i}",
                date_submitted=YESTERDAY_S,
                agency={"name": "SEC"},
            )
            for i in range(6)
        ]
        old_rec = _muckrock_record(
            title="Baseline",
            date_submitted=OLD_DATE_S,
            agency={"name": "SEC"},
        )
        mock_mr.return_value = recent_recs + [old_rec]
        r = self.tool.execute(mode="agency_activity", agency="SEC", days_back=90)
        assert r.success
        assert "SURGE" in r.output

    @patch("agent.tools.foia_requests._fetch_muckrock")
    def test_no_surge(self, mock_mr):
        """Equal recent and baseline → no surge."""
        recent = _muckrock_record(date_submitted=YESTERDAY_S, agency={"name": "EPA"})
        old = _muckrock_record(date_submitted=OLD_DATE_S, agency={"name": "EPA"})
        mock_mr.return_value = [recent, old]
        r = self.tool.execute(mode="agency_activity", agency="EPA", days_back=180)
        assert r.success
        assert "Normal" in r.output or "SURGE" not in r.output

    @patch("agent.tools.foia_requests._fetch_muckrock")
    def test_no_baseline_with_recent(self, mock_mr):
        """No baseline data but some recent → surge (new activity)."""
        recs = [
            _muckrock_record(
                title=f"New {i}",
                date_submitted=YESTERDAY_S,
                agency={"name": "CISA"},
            )
            for i in range(3)
        ]
        mock_mr.return_value = recs
        r = self.tool.execute(mode="agency_activity", agency="CISA", days_back=90)
        assert r.success
        assert "SURGE" in r.output or "NEW" in r.output

    @patch("agent.tools.foia_requests._fetch_muckrock", return_value=[])
    def test_empty_results(self, mock_mr):
        r = self.tool.execute(mode="agency_activity", agency="NONEXISTENT")
        assert r.success
        assert "No recent" in r.output or "0" in r.output

    @patch("agent.tools.foia_requests._fetch_muckrock")
    def test_agency_name_filtering(self, mock_mr):
        """Only records matching the agency name should be counted."""
        mock_mr.return_value = [
            _muckrock_record(agency={"name": "FBI"}, date_submitted=YESTERDAY_S),
            _muckrock_record(agency={"name": "CIA"}, date_submitted=YESTERDAY_S),
        ]
        r = self.tool.execute(mode="agency_activity", agency="FBI", days_back=365)
        assert r.success
        assert "1" in r.output  # Only 1 FBI record

    def test_cache_hit(self):
        self.cache.get.return_value = "cached agency output"
        r = self.tool.execute(mode="agency_activity", agency="SEC")
        assert r.success
        assert r.output == "cached agency output"


# ═══════════════════════════════════════════════════════════
# 12. Tool — Entity Cluster Mode
# ═══════════════════════════════════════════════════════════


class TestEntityClusterMode:
    def setup_method(self):
        self.cache = MagicMock()
        self.cache.get.return_value = None
        self.tool = FoiaRequestsTool(cache=self.cache)

    @patch("agent.tools.foia_requests._fetch_wdtk")
    @patch("agent.tools.foia_requests._fetch_muckrock")
    def test_convergence_multi_agency(self, mock_mr, mock_wdtk):
        """Requests from 3+ agencies → convergence."""
        mock_mr.return_value = [
            _muckrock_record(agency={"name": "FBI"}, date_submitted=YESTERDAY_S),
            _muckrock_record(agency={"name": "SEC"}, date_submitted=YESTERDAY_S),
            _muckrock_record(agency={"name": "EPA"}, date_submitted=YESTERDAY_S),
        ]
        mock_wdtk.return_value = []
        r = self.tool.execute(mode="entity_cluster", query="MegaCorp", days_back=365)
        assert r.success
        assert "CONVERGENCE" in r.output
        assert "3" in r.output  # 3 distinct agencies

    @patch("agent.tools.foia_requests._fetch_wdtk")
    @patch("agent.tools.foia_requests._fetch_muckrock")
    def test_convergence_multi_jurisdiction(self, mock_mr, mock_wdtk):
        """Requests from 2+ jurisdictions → convergence."""
        mock_mr.return_value = [
            _muckrock_record(agency={"name": "SEC"}, date_submitted=YESTERDAY_S),
        ]
        mock_wdtk.return_value = [
            _wdtk_record(public_body={"name": "FCA"}, created_at=YESTERDAY_S),
        ]
        r = self.tool.execute(mode="entity_cluster", query="MegaCorp", days_back=365)
        assert r.success
        assert "CONVERGENCE" in r.output  # US + UK = 2 jurisdictions

    @patch("agent.tools.foia_requests._fetch_wdtk", return_value=[])
    @patch("agent.tools.foia_requests._fetch_muckrock")
    def test_no_convergence(self, mock_mr, mock_wdtk):
        """Only 1 agency, 1 jurisdiction → no convergence."""
        mock_mr.return_value = [
            _muckrock_record(agency={"name": "SEC"}, date_submitted=YESTERDAY_S),
        ]
        r = self.tool.execute(mode="entity_cluster", query="SmallCo", days_back=365)
        assert r.success
        assert "No convergence" in r.output

    @patch("agent.tools.foia_requests._fetch_wdtk", return_value=[])
    @patch("agent.tools.foia_requests._fetch_muckrock", return_value=[])
    def test_empty_cluster(self, mock_mr, mock_wdtk):
        r = self.tool.execute(mode="entity_cluster", query="zzz_nothing", days_back=365)
        assert r.success
        assert "No matching" in r.output or "0" in r.output

    @patch("agent.tools.foia_requests._fetch_wdtk")
    @patch("agent.tools.foia_requests._fetch_muckrock")
    def test_agency_breakdown_in_output(self, mock_mr, mock_wdtk):
        mock_mr.return_value = [
            _muckrock_record(agency={"name": "FBI"}, date_submitted=YESTERDAY_S),
            _muckrock_record(
                title="FBI 2",
                agency={"name": "FBI"},
                date_submitted=LAST_WEEK_S,
            ),
            _muckrock_record(agency={"name": "SEC"}, date_submitted=YESTERDAY_S),
        ]
        mock_wdtk.return_value = []
        r = self.tool.execute(mode="entity_cluster", query="MegaCorp", days_back=365)
        assert r.success
        assert "Agency breakdown" in r.output
        assert "FBI" in r.output
        assert "SEC" in r.output

    @patch("agent.tools.foia_requests._fetch_wdtk", return_value=[])
    @patch("agent.tools.foia_requests._fetch_muckrock")
    def test_date_filter(self, mock_mr, mock_wdtk):
        mock_mr.return_value = [
            _muckrock_record(title="Recent", date_submitted=YESTERDAY_S),
            _muckrock_record(title="Old", date_submitted=OLD_DATE_S),
        ]
        r = self.tool.execute(mode="entity_cluster", query="test", days_back=30)
        assert r.success
        assert "Recent" in r.output
        assert "Old" not in r.output

    def test_cache_hit(self):
        self.cache.get.return_value = "cached cluster"
        r = self.tool.execute(mode="entity_cluster", query="MegaCorp")
        assert r.success
        assert r.output == "cached cluster"

    @patch("agent.tools.foia_requests._fetch_wdtk", return_value=[])
    @patch("agent.tools.foia_requests._fetch_muckrock")
    def test_us_only_jurisdiction(self, mock_mr, mock_wdtk):
        mock_mr.return_value = [
            _muckrock_record(date_submitted=YESTERDAY_S),
        ]
        r = self.tool.execute(mode="entity_cluster", query="test", jurisdiction="us", days_back=365)
        assert r.success
        mock_wdtk.assert_not_called()


# ═══════════════════════════════════════════════════════════
# 13. Tool Metadata
# ═══════════════════════════════════════════════════════════


class TestToolMetadata:
    def setup_method(self):
        self.tool = FoiaRequestsTool()

    def test_name(self):
        assert self.tool.name == "foia_requests"

    def test_description(self):
        assert "FOIA" in self.tool.description
        assert "FOI" in self.tool.description
        assert "investigation" in self.tool.description

    def test_parameters_has_mode(self):
        props = self.tool.parameters["properties"]
        assert "mode" in props
        assert set(props["mode"]["enum"]) == VALID_MODES

    def test_parameters_has_query(self):
        assert "query" in self.tool.parameters["properties"]

    def test_parameters_has_agency(self):
        assert "agency" in self.tool.parameters["properties"]

    def test_required_field(self):
        assert "mode" in self.tool.parameters["required"]

    def test_valid_modes_frozen(self):
        assert isinstance(VALID_MODES, frozenset)
        assert {"search", "agency_activity", "entity_cluster"} == VALID_MODES


# ═══════════════════════════════════════════════════════════
# 14. Integration — Tool Count & Arm Count
# ═══════════════════════════════════════════════════════════


class TestIntegration:
    def test_tool_count(self):
        """Ensure FoiaRequestsTool is tool #37 in the registry."""
        from agent.cli import build_tool_registry

        registry = build_tool_registry()
        names = registry.list_names()
        assert "foia_requests" in names
        assert len(names) == 60, f"Expected 60 tools, got {len(names)}: {sorted(names)}"

    def test_bandit_arm_count(self):
        """Ensure investigation_signals is arm #25."""
        from agent.learning.bandit import DEFAULT_ARMS

        arm_names = [a.name for a in DEFAULT_ARMS]
        assert "investigation_signals" in arm_names
        assert len(DEFAULT_ARMS) == 48, f"Expected 48 arms, got {len(DEFAULT_ARMS)}"

    def test_arm_tools_reference_valid(self):
        """The arm's tools list should reference tools that exist in registry."""
        from agent.cli import build_tool_registry
        from agent.learning.bandit import DEFAULT_ARMS

        registry = build_tool_registry()
        tool_names = set(registry.list_names())
        arm = next(a for a in DEFAULT_ARMS if a.name == "investigation_signals")
        for t in arm.tools:
            assert t in tool_names, f"Arm references tool '{t}' which is not registered"

    def test_arm_has_examples(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm = next(a for a in DEFAULT_ARMS if a.name == "investigation_signals")
        assert len(arm.examples) >= 2


# ═══════════════════════════════════════════════════════════
# 15. Edge Cases — Graceful Degradation
# ═══════════════════════════════════════════════════════════


class TestGracefulDegradation:
    def setup_method(self):
        self.tool = FoiaRequestsTool(cache=None)

    @patch("agent.tools.foia_requests._fetch_wdtk", return_value=[])
    @patch("agent.tools.foia_requests._fetch_muckrock")
    def test_muckrock_timeout_wdtk_ok(self, mock_mr, mock_wdtk):
        """If MuckRock dies, WDTK results should still be returned."""
        mock_mr.return_value = []  # simulates timeout returning empty
        mock_wdtk.return_value = [_wdtk_record()]
        r = self.tool.execute(mode="search", query="test", days_back=365)
        assert r.success

    @patch("agent.tools.foia_requests._fetch_wdtk", return_value=[])
    @patch("agent.tools.foia_requests._fetch_muckrock", return_value=[])
    def test_both_sources_empty(self, mock_mr, mock_wdtk):
        r = self.tool.execute(mode="search", query="test", days_back=365)
        assert r.success
        assert "No matching" in r.output

    @patch("agent.tools.foia_requests._fetch_wdtk", return_value=[])
    @patch("agent.tools.foia_requests._fetch_muckrock")
    def test_unparseable_dates_skipped(self, mock_mr, mock_wdtk):
        """Records with unparseable dates should be silently skipped."""
        bad_rec = _muckrock_record(date_submitted="not-a-date")
        mock_mr.return_value = [bad_rec]
        r = self.tool.execute(mode="search", query="test", days_back=365)
        assert r.success
        # Bad date → _parse_date returns None → filter skips it
        assert "No matching" in r.output or "0" in r.output

    @patch("agent.tools.foia_requests._fetch_wdtk", return_value=[])
    @patch("agent.tools.foia_requests._fetch_muckrock")
    def test_unicode_in_titles(self, mock_mr, mock_wdtk):
        rec = _muckrock_record(title="Réquête sur les données 日本語")
        mock_mr.return_value = [rec]
        r = self.tool.execute(mode="search", query="données", days_back=365)
        assert r.success
        assert "données" in r.output or "Réquête" in r.output

    def test_no_cache_no_crash(self):
        """Tool with cache=None should not crash on any mode."""
        tool = FoiaRequestsTool(cache=None)
        # These will try to make real API calls, but we're testing the cache code path
        # The network calls will fail gracefully
        r = tool.execute(mode="search", query="test")
        assert isinstance(r, ToolResult)

    @patch("agent.tools.foia_requests._fetch_wdtk", return_value=[])
    @patch("agent.tools.foia_requests._fetch_muckrock")
    def test_limit_respected(self, mock_mr, mock_wdtk):
        """Should not return more than `limit` records."""
        recs = [_muckrock_record(title=f"Rec {i}", date_submitted=YESTERDAY_S) for i in range(20)]
        mock_mr.return_value = recs
        r = self.tool.execute(mode="search", query="test", limit=5, days_back=365)
        assert r.success
        assert "Results: 5" in r.output

    @patch("agent.tools.foia_requests._fetch_wdtk", return_value=[])
    @patch("agent.tools.foia_requests._fetch_muckrock")
    def test_extra_kwargs_ignored(self, mock_mr, mock_wdtk):
        """Unknown kwargs should be silently absorbed by **_."""
        mock_mr.return_value = [_muckrock_record()]
        r = self.tool.execute(
            mode="search",
            query="test",
            bogus_param="xyz",
            days_back=365,
        )
        assert isinstance(r, ToolResult)


# ── L2 Entity Persistence Tests ──────────────────────────────────────────────

import unittest


def _make_store_mock():
    """Build a mock PipelineStore for L2 persistence testing."""
    store = MagicMock()
    store.register_entity = MagicMock(side_effect=lambda **kw: kw["entity_id"])
    store.store_entity_observation = MagicMock(return_value=1)
    return store


class TestL2PersistenceNoStore(unittest.TestCase):
    """Persistence is a no-op when store is absent."""

    def test_no_store_returns_zeros(self):
        tool = FoiaRequestsTool()
        tool._store = None
        counts = tool._persist_entities({"records": [{"title": "Some request"}]}, "search")
        assert counts == {"investigation_signal_obs": 0}

    def test_no_entity_id_fn_returns_zeros(self):
        import agent.tools.foia_requests as foia_mod

        tool = FoiaRequestsTool()
        tool._store = _make_store_mock()
        original = foia_mod._entity_id_from_key
        try:
            foia_mod._entity_id_from_key = None
            counts = tool._persist_entities({"records": [{"title": "Some request"}]}, "search")
            assert counts == {"investigation_signal_obs": 0}
        finally:
            foia_mod._entity_id_from_key = original


class TestL2PersistenceSearch(unittest.TestCase):
    """search mode persists investigation_signal obs on entity nodes."""

    def test_persists_one_obs(self):
        tool = FoiaRequestsTool()
        store = _make_store_mock()
        tool._store = store
        data = {
            "records": [
                {
                    "title": "FOIA re: Acme Corp Safety Violations",
                    "agency": "EPA",
                    "jurisdiction": "US",
                    "source": "muckrock",
                    "date_filed": "2026-04-01",
                    "status": "Submitted",
                }
            ]
        }
        counts = tool._persist_entities(data, "search")
        assert counts["investigation_signal_obs"] == 1

    def test_obs_type_and_depth(self):
        tool = FoiaRequestsTool()
        store = _make_store_mock()
        tool._store = store
        data = {
            "records": [
                {
                    "title": "Records on XYZ Pharma",
                    "agency": "FDA",
                    "source": "muckrock",
                }
            ]
        }
        tool._persist_entities(data, "search")
        obs_call = store.store_entity_observation.call_args_list[0]
        assert obs_call.kwargs["observation_type"] == "investigation_signal"
        assert obs_call.kwargs["depth_level"] == 2
        assert obs_call.kwargs["source_tool"] == "foia_requests"

    def test_empty_title_skipped(self):
        tool = FoiaRequestsTool()
        store = _make_store_mock()
        tool._store = store
        data = {"records": [{"title": "", "agency": "SEC"}]}
        counts = tool._persist_entities(data, "search")
        assert counts["investigation_signal_obs"] == 0

    def test_multiple_records(self):
        tool = FoiaRequestsTool()
        store = _make_store_mock()
        tool._store = store
        data = {
            "records": [
                {"title": "Req A", "agency": "EPA", "source": "muckrock"},
                {"title": "Req B", "agency": "FDA", "source": "muckrock"},
                {"title": "Req C", "agency": "DOJ", "source": "wdtk"},
            ]
        }
        counts = tool._persist_entities(data, "search")
        assert counts["investigation_signal_obs"] == 3


class TestL2PersistenceAgencyActivity(unittest.TestCase):
    """agency_activity mode uses agency name as entity."""

    def test_persists_agency_entity(self):
        tool = FoiaRequestsTool()
        store = _make_store_mock()
        tool._store = store
        data = {
            "records": [
                {
                    "title": "Request about enforcement",
                    "agency": "Securities and Exchange Commission",
                    "jurisdiction": "US",
                    "source": "muckrock",
                }
            ]
        }
        counts = tool._persist_entities(data, "agency_activity")
        assert counts["investigation_signal_obs"] == 1
        # Agency name is used as entity, not title
        reg_call = store.register_entity.call_args_list[0]
        assert reg_call.kwargs["canonical_name"] == "Securities and Exchange Commission"

    def test_empty_agency_skipped(self):
        tool = FoiaRequestsTool()
        store = _make_store_mock()
        tool._store = store
        data = {"records": [{"title": "Some FOIA", "agency": ""}]}
        counts = tool._persist_entities(data, "agency_activity")
        assert counts["investigation_signal_obs"] == 0


class TestL2PersistenceEntityCluster(unittest.TestCase):
    """entity_cluster mode persists on entity from title."""

    def test_persists_cluster_obs(self):
        tool = FoiaRequestsTool()
        store = _make_store_mock()
        tool._store = store
        data = {
            "records": [
                {"title": "Entity Alpha", "agency": "FBI", "source": "muckrock"},
                {"title": "Entity Alpha", "agency": "DOJ", "source": "muckrock"},
            ]
        }
        counts = tool._persist_entities(data, "entity_cluster")
        assert counts["investigation_signal_obs"] == 2

    def test_obs_value_contains_mode(self):
        tool = FoiaRequestsTool()
        store = _make_store_mock()
        tool._store = store
        data = {"records": [{"title": "Test Entity", "agency": "SEC", "source": "muckrock"}]}
        tool._persist_entities(data, "entity_cluster")
        obs_call = store.store_entity_observation.call_args_list[0]
        assert obs_call.kwargs["value"]["mode"] == "entity_cluster"


class TestL2PersistenceExceptionHandling(unittest.TestCase):
    """Exception in persistence is non-fatal."""

    def test_exception_returns_zeros(self):
        tool = FoiaRequestsTool()
        store = _make_store_mock()
        store.register_entity.side_effect = RuntimeError("boom")
        tool._store = store
        counts = tool._persist_entities(
            {"records": [{"title": "Crash Request", "agency": "EPA"}]},
            "search",
        )
        assert counts == {"investigation_signal_obs": 0}

    def test_exception_does_not_propagate(self):
        tool = FoiaRequestsTool()
        store = _make_store_mock()
        store.store_entity_observation.side_effect = Exception("db error")
        tool._store = store
        counts = tool._persist_entities({"records": [{"title": "Crash Request"}]}, "search")
        assert counts == {"investigation_signal_obs": 0}


class TestL2PersistenceEmptyRecords(unittest.TestCase):
    """Empty records are handled gracefully."""

    def test_empty_records_all_modes(self):
        tool = FoiaRequestsTool()
        store = _make_store_mock()
        tool._store = store
        for mode in ("search", "agency_activity", "entity_cluster"):
            counts = tool._persist_entities({"records": []}, mode)
            assert counts["investigation_signal_obs"] == 0
        store.store_entity_observation.assert_not_called()

    def test_missing_records_key(self):
        tool = FoiaRequestsTool()
        store = _make_store_mock()
        tool._store = store
        counts = tool._persist_entities({}, "search")
        assert counts["investigation_signal_obs"] == 0
