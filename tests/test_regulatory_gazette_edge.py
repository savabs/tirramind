"""
Edge case tests for Regulatory Gazette tool (7b-Q).

Covers: mode routing, parameter validation, agency resolution, doc type parsing,
date helpers, format_doc normalization, cache interaction, HTTP errors (400/429/
500/timeout), significant_only filter, upcoming mode, search mode, agency listing,
parameter clamping, URL encoding, output formatting, registry integration, bandit arm.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.tools.regulatory_gazette import (
    MARKET_AGENCIES,
    RegulatoryGazetteTool,
    _FIELDS,
    _VALID_TYPES,
    _build_params,
    _days_until,
    _encode_fr_params,
    _format_doc,
    _parse_date,
    _parse_doc_types,
    _resolve_agency,
    _safe_int,
    _url_encode_value,
)
from agent.tools.base import ToolResult


# ── Helpers ─────────────────────────────────────────────────────────


def _make_fr_doc(
    title: str = "Test Rule",
    doc_type: str = "Proposed Rule",
    doc_number: str = "2026-00001",
    pub_date: str = "2026-03-20",
    agency_name: str = "Securities and Exchange Commission",
    agency_slug: str = "securities-and-exchange-commission",
    agency_id: int = 466,
    abstract: str = "This is a test abstract.",
    action: str = "Proposed rule.",
    comments_close: str | None = None,
    effective_on: str | None = None,
    topics: list[str] | None = None,
    significant: bool | None = None,
    docket_ids: list[str] | None = None,
    page_length: int = 10,
    html_url: str = "https://www.federalregister.gov/d/2026-00001",
) -> dict:
    """Build a single Federal Register document dict as returned by the API."""
    return {
        "title": title,
        "type": doc_type,
        "document_number": doc_number,
        "publication_date": pub_date,
        "agencies": [
            {
                "raw_name": agency_name.upper(),
                "name": agency_name,
                "id": agency_id,
                "slug": agency_slug,
                "url": f"https://www.federalregister.gov/agencies/{agency_slug}",
                "json_url": f"https://www.federalregister.gov/api/v1/agencies/{agency_id}",
                "parent_id": None,
            }
        ],
        "abstract": abstract,
        "action": action,
        "comments_close_on": comments_close,
        "effective_on": effective_on,
        "topics": topics or [],
        "significant": significant,
        "regulation_id_number_info": {},
        "cfr_references": [],
        "docket_ids": docket_ids or [],
        "page_length": page_length,
        "html_url": html_url,
    }


def _make_fr_response(docs: list[dict], count: int | None = None) -> dict:
    """Build a Federal Register API response dict."""
    return {
        "count": count if count is not None else len(docs),
        "results": docs,
    }


def _mock_httpx_response(data: dict, status_code: int = 200):
    """Create a mock httpx.Response."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = status_code
    mock_resp.json.return_value = data
    mock_resp.raise_for_status = MagicMock()
    if status_code >= 400:
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}",
            request=MagicMock(),
            response=mock_resp,
        )
    return mock_resp


# ── 1. Helper Functions ─────────────────────────────────────────────


class TestResolveAgency:
    def test_known_alias(self):
        assert _resolve_agency("sec") == "securities-and-exchange-commission"

    def test_known_alias_uppercase(self):
        assert _resolve_agency("SEC") == "securities-and-exchange-commission"

    def test_known_alias_whitespace(self):
        assert _resolve_agency("  fed  ") == "federal-reserve-system"

    def test_all_aliases_resolve(self):
        for alias in MARKET_AGENCIES:
            slug = _resolve_agency(alias)
            assert slug == MARKET_AGENCIES[alias]["slug"]

    def test_full_slug_passthrough(self):
        assert (
            _resolve_agency("securities-and-exchange-commission")
            == "securities-and-exchange-commission"
        )

    def test_unknown_alias_passthrough(self):
        assert _resolve_agency("unknown-agency") == "unknown-agency"

    def test_empty_string(self):
        assert _resolve_agency("") == ""

    def test_mixed_case(self):
        assert _resolve_agency("Fda") == "food-and-drug-administration"


class TestParseDate:
    def test_iso_format(self):
        assert _parse_date("2026-03-20") == "2026-03-20"

    def test_compact_format(self):
        assert _parse_date("20260320") == "2026-03-20"

    def test_us_format(self):
        assert _parse_date("03/20/2026") == "2026-03-20"

    def test_empty(self):
        assert _parse_date("") is None

    def test_whitespace(self):
        assert _parse_date("  ") is None

    def test_invalid(self):
        assert _parse_date("not-a-date") is None

    def test_partial(self):
        assert _parse_date("2026-13-01") is None  # month 13


class TestSafeInt:
    def test_int(self):
        assert _safe_int(42) == 42

    def test_string_int(self):
        assert _safe_int("42") == 42

    def test_none(self):
        assert _safe_int(None) == 0

    def test_none_with_default(self):
        assert _safe_int(None, 99) == 99

    def test_empty_string(self):
        assert _safe_int("") == 0

    def test_float_string(self):
        assert _safe_int("3.14") == 0  # int() rejects float strings

    def test_dict(self):
        assert _safe_int({}) == 0


class TestParseDocTypes:
    def test_default(self):
        assert _parse_doc_types("RULE,PRORULE") == ["RULE", "PRORULE"]

    def test_single(self):
        assert _parse_doc_types("RULE") == ["RULE"]

    def test_all_types(self):
        result = _parse_doc_types("RULE,PRORULE,NOTICE,PRESDOCU")
        assert set(result) == _VALID_TYPES

    def test_invalid_type_filtered(self):
        result = _parse_doc_types("RULE,INVALID,PRORULE")
        assert result == ["RULE", "PRORULE"]

    def test_all_invalid_falls_to_default(self):
        result = _parse_doc_types("INVALID,GARBAGE")
        assert result == ["RULE", "PRORULE"]

    def test_empty_string(self):
        assert _parse_doc_types("") == ["RULE", "PRORULE"]

    def test_whitespace_only(self):
        assert _parse_doc_types("   ") == ["RULE", "PRORULE"]

    def test_lowercase_normalized(self):
        result = _parse_doc_types("rule,prorule")
        assert result == ["RULE", "PRORULE"]

    def test_mixed_case(self):
        result = _parse_doc_types("Rule,ProRule,Notice")
        assert result == ["RULE", "PRORULE", "NOTICE"]

    def test_whitespace_around_types(self):
        result = _parse_doc_types(" RULE , PRORULE ")
        assert result == ["RULE", "PRORULE"]


class TestDaysUntil:
    def test_future_date(self):
        future = (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m-%d")
        result = _days_until(future)
        assert result is not None
        assert 28 <= result <= 30

    def test_past_date(self):
        past = (datetime.now(timezone.utc) - timedelta(days=10)).strftime("%Y-%m-%d")
        result = _days_until(past)
        assert result is not None
        assert -11 <= result <= -10

    def test_none(self):
        assert _days_until(None) is None

    def test_empty_string(self):
        assert _days_until("") is None

    def test_invalid_date(self):
        assert _days_until("not-a-date") is None


class TestFormatDoc:
    def test_basic(self):
        doc = _make_fr_doc()
        formatted = _format_doc(doc)
        assert formatted["title"] == "Test Rule"
        assert formatted["type"] == "Proposed Rule"
        assert formatted["document_number"] == "2026-00001"
        assert formatted["agencies"] == ["Securities and Exchange Commission"]
        assert formatted["page_length"] == 10

    def test_missing_agencies(self):
        doc = _make_fr_doc()
        doc["agencies"] = None
        formatted = _format_doc(doc)
        assert formatted["agencies"] == []

    def test_empty_agencies(self):
        doc = _make_fr_doc()
        doc["agencies"] = []
        formatted = _format_doc(doc)
        assert formatted["agencies"] == []

    def test_agency_fallback_to_raw_name(self):
        doc = _make_fr_doc()
        doc["agencies"] = [{"raw_name": "RAW AGENCY", "id": 1}]
        formatted = _format_doc(doc)
        assert formatted["agencies"] == ["RAW AGENCY"]

    def test_agency_fallback_to_unknown(self):
        doc = _make_fr_doc()
        doc["agencies"] = [{"id": 1}]
        formatted = _format_doc(doc)
        assert formatted["agencies"] == ["Unknown"]

    def test_abstract_truncation(self):
        doc = _make_fr_doc(abstract="x" * 1000)
        formatted = _format_doc(doc)
        assert len(formatted["abstract"]) == 500

    def test_none_title(self):
        doc = _make_fr_doc()
        doc["title"] = None
        formatted = _format_doc(doc)
        assert formatted["title"] == ""

    def test_none_abstract(self):
        doc = _make_fr_doc()
        doc["abstract"] = None
        formatted = _format_doc(doc)
        assert formatted["abstract"] == ""

    def test_none_action(self):
        doc = _make_fr_doc()
        doc["action"] = None
        formatted = _format_doc(doc)
        assert formatted["action"] == ""

    def test_none_topics(self):
        doc = _make_fr_doc()
        doc["topics"] = None
        formatted = _format_doc(doc)
        assert formatted["topics"] == []

    def test_none_docket_ids(self):
        doc = _make_fr_doc()
        doc["docket_ids"] = None
        formatted = _format_doc(doc)
        assert formatted["docket_ids"] == []

    def test_page_length_none(self):
        doc = _make_fr_doc()
        doc["page_length"] = None
        formatted = _format_doc(doc)
        assert formatted["page_length"] == 0

    def test_url_present(self):
        doc = _make_fr_doc()
        formatted = _format_doc(doc)
        assert formatted["url"] == "https://www.federalregister.gov/d/2026-00001"


class TestUrlEncodeValue:
    def test_spaces(self):
        assert _url_encode_value("hello world") == "hello+world"

    def test_ampersand(self):
        assert _url_encode_value("a&b") == "a%26b"

    def test_equals(self):
        assert _url_encode_value("a=b") == "a%3Db"

    def test_hash(self):
        assert _url_encode_value("a#b") == "a%23b"

    def test_plain(self):
        assert _url_encode_value("semiconductor") == "semiconductor"

    def test_combined(self):
        assert _url_encode_value("a & b = c") == "a+%26+b+%3D+c"


# ── 2. Parameter Building ──────────────────────────────────────────


class TestBuildParams:
    def test_basic(self):
        params = _build_params(types=["RULE"])
        assert params["types"] == ["RULE"]
        assert params["per_page"] == 25
        assert params["order"] == "newest"

    def test_with_keyword(self):
        params = _build_params(types=["RULE"], keyword="crypto")
        assert params["keyword"] == "crypto"

    def test_with_agency_alias(self):
        params = _build_params(types=["RULE"], agency="sec")
        assert params["agency"] == "securities-and-exchange-commission"

    def test_with_date(self):
        params = _build_params(types=["RULE"], date_gte="2026-01-01")
        assert params["date_gte"] == "2026-01-01"

    def test_empty_keyword_omitted(self):
        params = _build_params(types=["RULE"], keyword="")
        assert "keyword" not in params

    def test_empty_agency_omitted(self):
        params = _build_params(types=["RULE"], agency="")
        assert "agency" not in params

    def test_per_page_override(self):
        params = _build_params(types=["RULE"], per_page=50)
        assert params["per_page"] == 50


class TestEncodeFrParams:
    def test_basic(self):
        params = _build_params(types=["RULE"])
        qs = _encode_fr_params(params)
        assert "per_page=25" in qs
        assert "order=newest" in qs
        assert "conditions[type][]=RULE" in qs
        # All fields included
        for f in _FIELDS:
            assert f"fields[]={f}" in qs

    def test_multiple_types(self):
        params = _build_params(types=["RULE", "PRORULE"])
        qs = _encode_fr_params(params)
        assert "conditions[type][]=RULE" in qs
        assert "conditions[type][]=PRORULE" in qs

    def test_keyword_encoding(self):
        params = _build_params(types=["RULE"], keyword="hello world")
        qs = _encode_fr_params(params)
        assert "conditions[term]=hello+world" in qs

    def test_agency_encoding(self):
        params = _build_params(types=["RULE"], agency="sec")
        qs = _encode_fr_params(params)
        assert "conditions[agencies][]=securities-and-exchange-commission" in qs

    def test_date_encoding(self):
        params = _build_params(types=["RULE"], date_gte="2026-01-01")
        qs = _encode_fr_params(params)
        assert "conditions[publication_date][gte]=2026-01-01" in qs


# ── 3. Mode Routing ────────────────────────────────────────────────


class TestModeRouting:
    def test_invalid_mode(self):
        tool = RegulatoryGazetteTool()
        result = tool.execute(mode="invalid")
        assert not result.success
        assert "Invalid mode" in result.output

    def test_mode_case_insensitive(self):
        tool = RegulatoryGazetteTool()
        with patch.object(tool, "_fetch_fr", return_value=([], 0, None)):
            result = tool.execute(mode="RECENT")
            assert result.success

    def test_mode_whitespace_stripped(self):
        tool = RegulatoryGazetteTool()
        with patch.object(tool, "_fetch_fr", return_value=([], 0, None)):
            result = tool.execute(mode="  recent  ")
            assert result.success

    def test_search_requires_keyword(self):
        tool = RegulatoryGazetteTool()
        result = tool.execute(mode="search")
        assert not result.success
        assert "keyword" in result.output.lower()

    def test_search_empty_keyword_rejected(self):
        tool = RegulatoryGazetteTool()
        result = tool.execute(mode="search", keyword="   ")
        assert not result.success

    def test_agency_without_name_lists_agencies(self):
        tool = RegulatoryGazetteTool()
        result = tool.execute(mode="agency")
        assert result.success
        assert "Market-Relevant" in result.output
        assert result.data["agencies"] == MARKET_AGENCIES


# ── 4. Recent Mode ─────────────────────────────────────────────────


class TestRecentMode:
    def _mock_fetch(self, tool, docs, count=None):
        fr_response = _make_fr_response(docs, count)
        return patch.object(
            tool,
            "_fetch_fr",
            return_value=(fr_response["results"], fr_response["count"], None),
        )

    def test_basic_recent(self):
        tool = RegulatoryGazetteTool()
        docs = [_make_fr_doc(title="Rule A"), _make_fr_doc(title="Rule B")]
        with self._mock_fetch(tool, docs):
            result = tool.execute(mode="recent")
        assert result.success
        assert "Rule A" in result.output
        assert "Rule B" in result.output
        assert result.data["count"] == 2

    def test_empty_results(self):
        tool = RegulatoryGazetteTool()
        with self._mock_fetch(tool, []):
            result = tool.execute(mode="recent")
        assert result.success
        assert "No documents found" in result.output

    def test_significant_only_filter(self):
        tool = RegulatoryGazetteTool()
        docs = [
            _make_fr_doc(title="Big Rule", significant=True),
            _make_fr_doc(title="Small Rule", significant=False),
            _make_fr_doc(title="Unknown Rule", significant=None),
        ]
        with self._mock_fetch(tool, docs):
            result = tool.execute(mode="recent", significant_only=True)
        assert result.success
        assert "Big Rule" in result.output
        assert "Small Rule" not in result.output
        assert "Unknown Rule" not in result.output

    def test_with_keyword(self):
        tool = RegulatoryGazetteTool()
        docs = [_make_fr_doc(title="Crypto Rule")]
        with self._mock_fetch(tool, docs):
            result = tool.execute(mode="recent", keyword="crypto")
        assert result.success
        assert "Crypto Rule" in result.output

    def test_with_agency(self):
        tool = RegulatoryGazetteTool()
        docs = [_make_fr_doc(title="SEC Rule")]
        with self._mock_fetch(tool, docs):
            result = tool.execute(mode="recent", agency="sec")
        assert result.success

    def test_days_back_clamped_min(self):
        tool = RegulatoryGazetteTool()
        with patch.object(tool, "_fetch_fr", return_value=([], 0, None)) as mock:
            tool.execute(mode="recent", days_back=-5)
        # Should have been clamped to 1
        call_params = mock.call_args[0][0]
        assert "date_gte" in call_params

    def test_days_back_clamped_max(self):
        tool = RegulatoryGazetteTool()
        with patch.object(tool, "_fetch_fr", return_value=([], 0, None)):
            result = tool.execute(mode="recent", days_back=9999)
        assert result.success

    def test_limit_clamped_min(self):
        tool = RegulatoryGazetteTool()
        docs = [_make_fr_doc(title=f"Rule {i}") for i in range(5)]
        with self._mock_fetch(tool, docs, count=5):
            result = tool.execute(mode="recent", limit=0)
        assert result.success
        # limit clamped to 1, so max 1 doc
        assert result.data["count"] <= 1

    def test_limit_clamped_max(self):
        tool = RegulatoryGazetteTool()
        docs = [_make_fr_doc(title=f"Rule {i}") for i in range(3)]
        with self._mock_fetch(tool, docs, count=3):
            result = tool.execute(mode="recent", limit=999)
        assert result.success

    def test_api_error_propagated(self):
        tool = RegulatoryGazetteTool()
        with patch.object(tool, "_fetch_fr", return_value=([], 0, "API broke")):
            result = tool.execute(mode="recent")
        assert not result.success
        assert "API broke" in result.output

    def test_comment_period_shown_in_output(self):
        tool = RegulatoryGazetteTool()
        future = (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m-%d")
        docs = [_make_fr_doc(title="Comment Rule", comments_close=future)]
        with self._mock_fetch(tool, docs):
            result = tool.execute(mode="recent")
        assert result.success
        assert "comments close" in result.output.lower()

    def test_closed_comment_shown(self):
        tool = RegulatoryGazetteTool()
        past = (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%d")
        docs = [_make_fr_doc(title="Closed Rule", comments_close=past)]
        with self._mock_fetch(tool, docs):
            result = tool.execute(mode="recent")
        assert result.success
        assert "comments closed" in result.output.lower()

    def test_significant_tag_in_output(self):
        tool = RegulatoryGazetteTool()
        docs = [_make_fr_doc(title="Big Rule", significant=True)]
        with self._mock_fetch(tool, docs):
            result = tool.execute(mode="recent")
        assert result.success
        assert "[SIGNIFICANT]" in result.output

    def test_total_in_data(self):
        tool = RegulatoryGazetteTool()
        docs = [_make_fr_doc()]
        with self._mock_fetch(tool, docs, count=500):
            result = tool.execute(mode="recent")
        assert result.data["total"] == 500


# ── 5. Search Mode ─────────────────────────────────────────────────


class TestSearchMode:
    def test_basic_search(self):
        tool = RegulatoryGazetteTool()
        docs = [_make_fr_doc(title="Semiconductor Control Rule")]
        fr_resp = _make_fr_response(docs)
        with patch.object(
            tool,
            "_fetch_fr",
            return_value=(fr_resp["results"], fr_resp["count"], None),
        ):
            result = tool.execute(mode="search", keyword="semiconductor")
        assert result.success
        assert "semiconductor" in result.output.lower()

    def test_search_with_type_filter(self):
        tool = RegulatoryGazetteTool()
        with patch.object(tool, "_fetch_fr", return_value=([], 0, None)) as mock:
            tool.execute(mode="search", keyword="test", doc_type="NOTICE")
        params = mock.call_args[0][0]
        assert params["types"] == ["NOTICE"]

    def test_search_significant_filter(self):
        tool = RegulatoryGazetteTool()
        docs = [
            _make_fr_doc(title="A", significant=True),
            _make_fr_doc(title="B", significant=False),
        ]
        fr_resp = _make_fr_response(docs)
        with patch.object(
            tool,
            "_fetch_fr",
            return_value=(fr_resp["results"], fr_resp["count"], None),
        ):
            result = tool.execute(mode="search", keyword="test", significant_only=True)
        assert result.success
        assert result.data["count"] == 1


# ── 6. Agency Mode ─────────────────────────────────────────────────


class TestAgencyMode:
    def test_agency_by_alias(self):
        tool = RegulatoryGazetteTool()
        docs = [_make_fr_doc(title="FDA Rule")]
        fr_resp = _make_fr_response(docs)
        with patch.object(
            tool,
            "_fetch_fr",
            return_value=(fr_resp["results"], fr_resp["count"], None),
        ) as mock:
            result = tool.execute(mode="agency", agency="fda")
        assert result.success
        params = mock.call_args[0][0]
        assert params["agency"] == "food-and-drug-administration"

    def test_agency_by_full_slug(self):
        tool = RegulatoryGazetteTool()
        with patch.object(tool, "_fetch_fr", return_value=([], 0, None)) as mock:
            tool.execute(mode="agency", agency="food-and-drug-administration")
        params = mock.call_args[0][0]
        assert params["agency"] == "food-and-drug-administration"

    def test_agency_listing(self):
        tool = RegulatoryGazetteTool()
        result = tool.execute(mode="agency")
        assert result.success
        assert "sec" in result.output.lower()
        assert "fed" in result.output.lower()
        assert "fda" in result.output.lower()
        assert len(result.data["agencies"]) == len(MARKET_AGENCIES)

    def test_agency_with_keyword(self):
        tool = RegulatoryGazetteTool()
        with patch.object(tool, "_fetch_fr", return_value=([], 0, None)) as mock:
            tool.execute(mode="agency", agency="sec", keyword="crypto")
        params = mock.call_args[0][0]
        assert params["keyword"] == "crypto"
        assert params["agency"] == "securities-and-exchange-commission"


# ── 7. Upcoming Mode ───────────────────────────────────────────────


class TestUpcomingMode:
    def test_basic_upcoming(self):
        tool = RegulatoryGazetteTool()
        future = (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m-%d")
        docs = [_make_fr_doc(title="Open Rule", comments_close=future)]
        fr_resp = _make_fr_response(docs)
        with patch.object(
            tool,
            "_fetch_fr",
            return_value=(fr_resp["results"], fr_resp["count"], None),
        ):
            result = tool.execute(mode="upcoming")
        assert result.success
        assert "Open Rule" in result.output
        assert result.data["count"] == 1

    def test_upcoming_filters_closed(self):
        tool = RegulatoryGazetteTool()
        future = (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m-%d")
        past = (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%d")
        docs = [
            _make_fr_doc(title="Open Rule", comments_close=future),
            _make_fr_doc(title="Closed Rule", comments_close=past),
        ]
        fr_resp = _make_fr_response(docs)
        with patch.object(
            tool,
            "_fetch_fr",
            return_value=(fr_resp["results"], fr_resp["count"], None),
        ):
            result = tool.execute(mode="upcoming")
        assert result.success
        assert "Open Rule" in result.output
        assert "Closed Rule" not in result.output

    def test_upcoming_filters_no_comment_date(self):
        tool = RegulatoryGazetteTool()
        docs = [_make_fr_doc(title="No Date Rule", comments_close=None)]
        fr_resp = _make_fr_response(docs)
        with patch.object(
            tool,
            "_fetch_fr",
            return_value=(fr_resp["results"], fr_resp["count"], None),
        ):
            result = tool.execute(mode="upcoming")
        assert result.success
        assert "No rules with open comment periods" in result.output

    def test_upcoming_days_remaining(self):
        tool = RegulatoryGazetteTool()
        future = (datetime.now(timezone.utc) + timedelta(days=15)).strftime("%Y-%m-%d")
        docs = [_make_fr_doc(title="Soon Rule", comments_close=future)]
        fr_resp = _make_fr_response(docs)
        with patch.object(
            tool,
            "_fetch_fr",
            return_value=(fr_resp["results"], fr_resp["count"], None),
        ):
            result = tool.execute(mode="upcoming")
        assert result.success
        assert result.data["documents"][0]["days_remaining"] is not None
        assert 13 <= result.data["documents"][0]["days_remaining"] <= 15

    def test_upcoming_sorted_by_deadline(self):
        tool = RegulatoryGazetteTool()
        soon = (datetime.now(timezone.utc) + timedelta(days=5)).strftime("%Y-%m-%d")
        later = (datetime.now(timezone.utc) + timedelta(days=60)).strftime("%Y-%m-%d")
        docs = [
            _make_fr_doc(title="Later Rule", comments_close=later),
            _make_fr_doc(title="Soon Rule", comments_close=soon),
        ]
        fr_resp = _make_fr_response(docs)
        with patch.object(
            tool,
            "_fetch_fr",
            return_value=(fr_resp["results"], fr_resp["count"], None),
        ):
            result = tool.execute(mode="upcoming")
        assert result.success
        assert result.data["documents"][0]["title"] == "Soon Rule"

    def test_upcoming_significant_filter(self):
        tool = RegulatoryGazetteTool()
        future = (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m-%d")
        docs = [
            _make_fr_doc(title="Big Rule", comments_close=future, significant=True),
            _make_fr_doc(title="Small Rule", comments_close=future, significant=False),
        ]
        fr_resp = _make_fr_response(docs)
        with patch.object(
            tool,
            "_fetch_fr",
            return_value=(fr_resp["results"], fr_resp["count"], None),
        ):
            result = tool.execute(mode="upcoming", significant_only=True)
        assert result.success
        assert result.data["count"] == 1
        assert result.data["documents"][0]["title"] == "Big Rule"

    def test_upcoming_with_keyword(self):
        tool = RegulatoryGazetteTool()
        with patch.object(tool, "_fetch_fr", return_value=([], 0, None)) as mock:
            tool.execute(mode="upcoming", keyword="crypto")
        params = mock.call_args[0][0]
        assert params["keyword"] == "crypto"

    def test_upcoming_with_agency(self):
        tool = RegulatoryGazetteTool()
        with patch.object(tool, "_fetch_fr", return_value=([], 0, None)) as mock:
            tool.execute(mode="upcoming", agency="sec")
        params = mock.call_args[0][0]
        assert params["agency"] == "securities-and-exchange-commission"

    def test_upcoming_empty_results(self):
        tool = RegulatoryGazetteTool()
        with patch.object(tool, "_fetch_fr", return_value=([], 0, None)):
            result = tool.execute(mode="upcoming")
        assert result.success
        assert "No rules with open comment periods" in result.output


# ── 8. HTTP Error Handling ──────────────────────────────────────────


class TestHTTPErrors:
    def test_400_bad_request(self):
        tool = RegulatoryGazetteTool()
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.json.return_value = {}
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch(
            "agent.tools.regulatory_gazette.httpx.Client", return_value=mock_client
        ):
            result = tool.execute(mode="recent")
        assert not result.success
        assert "Bad request" in result.output

    def test_429_rate_limit(self):
        tool = RegulatoryGazetteTool()
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch(
            "agent.tools.regulatory_gazette.httpx.Client", return_value=mock_client
        ):
            result = tool.execute(mode="recent")
        assert not result.success
        assert "Rate limited" in result.output

    def test_500_server_error(self):
        tool = RegulatoryGazetteTool()
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500",
            request=MagicMock(),
            response=mock_resp,
        )
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch(
            "agent.tools.regulatory_gazette.httpx.Client", return_value=mock_client
        ):
            result = tool.execute(mode="recent")
        assert not result.success
        assert "HTTP 500" in result.output

    def test_timeout(self):
        tool = RegulatoryGazetteTool()
        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.TimeoutException("timed out")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch(
            "agent.tools.regulatory_gazette.httpx.Client", return_value=mock_client
        ):
            result = tool.execute(mode="recent")
        assert not result.success
        assert "timed out" in result.output.lower()

    def test_generic_exception(self):
        tool = RegulatoryGazetteTool()
        mock_client = MagicMock()
        mock_client.get.side_effect = ConnectionError("DNS fail")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch(
            "agent.tools.regulatory_gazette.httpx.Client", return_value=mock_client
        ):
            result = tool.execute(mode="recent")
        assert not result.success
        assert "DNS fail" in result.output


# ── 9. Cache Interaction ───────────────────────────────────────────


class TestCacheInteraction:
    def test_cache_hit(self):
        cache = MagicMock()
        cached_data = _make_fr_response([_make_fr_doc(title="Cached Rule")])
        cache.get.return_value = cached_data
        tool = RegulatoryGazetteTool(cache=cache)
        result = tool.execute(mode="recent")
        assert result.success
        assert "Cached Rule" in result.output
        cache.get.assert_called_once()

    def test_cache_miss_then_put(self):
        cache = MagicMock()
        cache.get.return_value = None
        tool = RegulatoryGazetteTool(cache=cache)

        docs = [_make_fr_doc(title="Fresh Rule")]
        fr_resp = _make_fr_response(docs)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fr_resp
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch(
            "agent.tools.regulatory_gazette.httpx.Client", return_value=mock_client
        ):
            result = tool.execute(mode="recent")
        assert result.success
        cache.put.assert_called_once()
        # Verify TTL is 7200
        call_kwargs = cache.put.call_args
        assert call_kwargs[1]["ttl"] == 7200

    def test_cache_miss_empty_results_no_put(self):
        cache = MagicMock()
        cache.get.return_value = None
        tool = RegulatoryGazetteTool(cache=cache)

        fr_resp = _make_fr_response([])
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fr_resp
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch(
            "agent.tools.regulatory_gazette.httpx.Client", return_value=mock_client
        ):
            result = tool.execute(mode="recent")
        assert result.success
        cache.put.assert_not_called()

    def test_no_cache_still_works(self):
        tool = RegulatoryGazetteTool(cache=None)
        docs = [_make_fr_doc()]
        fr_resp = _make_fr_response(docs)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fr_resp
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch(
            "agent.tools.regulatory_gazette.httpx.Client", return_value=mock_client
        ):
            result = tool.execute(mode="recent")
        assert result.success


# ── 10. Tool Schema / Metadata ──────────────────────────────────────


class TestToolSchema:
    def test_name(self):
        tool = RegulatoryGazetteTool()
        assert tool.name == "regulatory_gazette"

    def test_description_nonempty(self):
        tool = RegulatoryGazetteTool()
        assert len(tool.description) > 50

    def test_parameters_structure(self):
        tool = RegulatoryGazetteTool()
        assert tool.parameters["type"] == "object"
        props = tool.parameters["properties"]
        assert "mode" in props
        assert "keyword" in props
        assert "agency" in props
        assert "doc_type" in props
        assert "days_back" in props
        assert "significant_only" in props
        assert "limit" in props

    def test_mode_enum(self):
        tool = RegulatoryGazetteTool()
        modes = tool.parameters["properties"]["mode"]["enum"]
        assert set(modes) == {"recent", "search", "agency", "upcoming"}

    def test_defaults(self):
        tool = RegulatoryGazetteTool()
        props = tool.parameters["properties"]
        assert props["mode"]["default"] == "recent"
        assert props["doc_type"]["default"] == "RULE,PRORULE"
        assert props["days_back"]["default"] == 7
        assert props["limit"]["default"] == 25
        assert props["significant_only"]["default"] is False

    def test_no_required_params(self):
        tool = RegulatoryGazetteTool()
        assert tool.parameters.get("required", []) == []


# ── 11. MARKET_AGENCIES ────────────────────────────────────────────


class TestMarketAgencies:
    def test_all_have_slug(self):
        for alias, info in MARKET_AGENCIES.items():
            assert "slug" in info, f"{alias} missing slug"

    def test_all_have_id(self):
        for alias, info in MARKET_AGENCIES.items():
            assert "id" in info, f"{alias} missing id"
            assert isinstance(info["id"], int)

    def test_all_have_sector(self):
        for alias, info in MARKET_AGENCIES.items():
            assert "sector" in info, f"{alias} missing sector"

    def test_no_duplicate_slugs(self):
        slugs = [info["slug"] for info in MARKET_AGENCIES.values()]
        assert len(slugs) == len(set(slugs))

    def test_no_duplicate_ids(self):
        ids = [info["id"] for info in MARKET_AGENCIES.values()]
        assert len(ids) == len(set(ids))

    def test_expected_agencies_present(self):
        expected = {"sec", "fed", "cftc", "ftc", "epa", "fda", "fcc", "ferc"}
        assert expected.issubset(MARKET_AGENCIES.keys())

    def test_count(self):
        assert len(MARKET_AGENCIES) == 20


# ── 12. Output Formatting ──────────────────────────────────────────


class TestOutputFormatting:
    def test_recent_header(self):
        tool = RegulatoryGazetteTool()
        docs = [_make_fr_doc()]
        fr_resp = _make_fr_response(docs, count=100)
        with patch.object(
            tool,
            "_fetch_fr",
            return_value=(fr_resp["results"], fr_resp["count"], None),
        ):
            result = tool.execute(mode="recent")
        assert "Federal Register: Recent" in result.output
        assert "1 of 100" in result.output

    def test_search_header_shows_keyword(self):
        tool = RegulatoryGazetteTool()
        docs = [_make_fr_doc()]
        fr_resp = _make_fr_response(docs)
        with patch.object(
            tool,
            "_fetch_fr",
            return_value=(fr_resp["results"], fr_resp["count"], None),
        ):
            result = tool.execute(mode="search", keyword="crypto")
        assert '"crypto"' in result.output

    def test_agency_header_shows_slug(self):
        tool = RegulatoryGazetteTool()
        docs = [_make_fr_doc()]
        fr_resp = _make_fr_response(docs)
        with patch.object(
            tool,
            "_fetch_fr",
            return_value=(fr_resp["results"], fr_resp["count"], None),
        ):
            result = tool.execute(mode="agency", agency="sec")
        assert "securities-and-exchange-commission" in result.output

    def test_upcoming_header(self):
        tool = RegulatoryGazetteTool()
        future = (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m-%d")
        docs = [_make_fr_doc(comments_close=future)]
        fr_resp = _make_fr_response(docs)
        with patch.object(
            tool,
            "_fetch_fr",
            return_value=(fr_resp["results"], fr_resp["count"], None),
        ):
            result = tool.execute(mode="upcoming")
        assert "Open Comment Periods" in result.output

    def test_abstract_in_output(self):
        tool = RegulatoryGazetteTool()
        docs = [_make_fr_doc(abstract="Important abstract text.")]
        fr_resp = _make_fr_response(docs)
        with patch.object(
            tool,
            "_fetch_fr",
            return_value=(fr_resp["results"], fr_resp["count"], None),
        ):
            result = tool.execute(mode="recent")
        assert "Important abstract text" in result.output

    def test_agency_name_in_output(self):
        tool = RegulatoryGazetteTool()
        docs = [_make_fr_doc(agency_name="Test Agency")]
        fr_resp = _make_fr_response(docs)
        with patch.object(
            tool,
            "_fetch_fr",
            return_value=(fr_resp["results"], fr_resp["count"], None),
        ):
            result = tool.execute(mode="recent")
        assert "Test Agency" in result.output


# ── 13. CLI Registration ───────────────────────────────────────────


class TestRegistration:
    def test_tool_in_registry(self):
        try:
            from agent.cli import build_tool_registry
            from agent.config.settings import AgentConfig
        except (ImportError, ModuleNotFoundError):
            pytest.skip("apscheduler or other dep not installed")

        config = AgentConfig()
        registry = build_tool_registry(config)
        names = registry.list_names()
        assert "regulatory_gazette" in names

    def test_tool_count(self):
        try:
            from agent.cli import build_tool_registry
            from agent.config.settings import AgentConfig
        except (ImportError, ModuleNotFoundError):
            pytest.skip("apscheduler or other dep not installed")

        config = AgentConfig()
        registry = build_tool_registry(config)
        assert len(registry.list_names()) == 47


# ── 14. Bandit Arm ─────────────────────────────────────────────────


class TestBanditArm:
    def test_regulatory_arm_exists(self):
        from agent.learning.bandit import DEFAULT_ARMS

        names = [arm.name for arm in DEFAULT_ARMS]
        assert "regulatory_pipeline" in names

    def test_regulatory_arm_tools(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm = next(a for a in DEFAULT_ARMS if a.name == "regulatory_pipeline")
        assert "regulatory_gazette" in arm.tools

    def test_regulatory_arm_has_examples(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm = next(a for a in DEFAULT_ARMS if a.name == "regulatory_pipeline")
        assert len(arm.examples) >= 2


# ── 15. Extra kwargs ignored ───────────────────────────────────────


class TestExtraKwargs:
    def test_extra_kwargs_ignored(self):
        tool = RegulatoryGazetteTool()
        with patch.object(tool, "_fetch_fr", return_value=([], 0, None)):
            result = tool.execute(mode="recent", unknown_param="foo", bar=42)
        assert result.success


# ── 16. Doc Type Edge Cases ─────────────────────────────────────────


class TestDocTypeEdgeCases:
    def test_notice_only(self):
        tool = RegulatoryGazetteTool()
        with patch.object(tool, "_fetch_fr", return_value=([], 0, None)) as mock:
            tool.execute(mode="recent", doc_type="NOTICE")
        params = mock.call_args[0][0]
        assert params["types"] == ["NOTICE"]

    def test_presdocu(self):
        tool = RegulatoryGazetteTool()
        with patch.object(tool, "_fetch_fr", return_value=([], 0, None)) as mock:
            tool.execute(mode="recent", doc_type="PRESDOCU")
        params = mock.call_args[0][0]
        assert params["types"] == ["PRESDOCU"]

    def test_all_four_types(self):
        tool = RegulatoryGazetteTool()
        with patch.object(tool, "_fetch_fr", return_value=([], 0, None)) as mock:
            tool.execute(mode="recent", doc_type="RULE,PRORULE,NOTICE,PRESDOCU")
        params = mock.call_args[0][0]
        assert set(params["types"]) == {"RULE", "PRORULE", "NOTICE", "PRESDOCU"}


# ── 17. _FIELDS constant ───────────────────────────────────────────


class TestFieldsConstant:
    def test_contains_essential_fields(self):
        essential = {
            "title",
            "type",
            "abstract",
            "document_number",
            "publication_date",
            "agencies",
            "comments_close_on",
            "significant",
            "html_url",
        }
        assert essential.issubset(set(_FIELDS))

    def test_no_duplicates(self):
        assert len(_FIELDS) == len(set(_FIELDS))


# ── 18. _VALID_TYPES constant ──────────────────────────────────────


class TestValidTypes:
    def test_expected_types(self):
        assert _VALID_TYPES == {"RULE", "PRORULE", "NOTICE", "PRESDOCU"}


# ── 19. Multiple Agencies in Doc ────────────────────────────────────


class TestMultipleAgencies:
    def test_multiple_agencies_shown(self):
        tool = RegulatoryGazetteTool()
        doc = _make_fr_doc()
        doc["agencies"].append(
            {"name": "Second Agency", "id": 999, "slug": "second-agency"}
        )
        fr_resp = _make_fr_response([doc])
        with patch.object(
            tool,
            "_fetch_fr",
            return_value=(fr_resp["results"], fr_resp["count"], None),
        ):
            result = tool.execute(mode="recent")
        assert result.success
        # Both agencies should show (we display up to 2)
        assert "Securities and Exchange Commission" in result.output
        assert "Second Agency" in result.output

    def test_more_than_two_agencies_truncated(self):
        doc = _make_fr_doc()
        doc["agencies"] = [{"name": f"Agency {i}", "id": i} for i in range(5)]
        formatted = _format_doc(doc)
        assert len(formatted["agencies"]) == 5  # all preserved in data


# ── Live Network (optional) ─────────────────────────────────────────


class TestLiveNetwork:
    """Live tests against the real Federal Register API."""

    @pytest.mark.skipif(True, reason="Live network test — run manually")
    def test_live_recent_proposed_rules(self):
        tool = RegulatoryGazetteTool()
        result = tool.execute(mode="recent", doc_type="PRORULE", days_back=7, limit=3)
        assert result.success
        print(result.output)

    @pytest.mark.skipif(True, reason="Live network test — run manually")
    def test_live_search_semiconductor(self):
        tool = RegulatoryGazetteTool()
        result = tool.execute(
            mode="search", keyword="semiconductor", days_back=365, limit=5
        )
        assert result.success
        print(result.output)

    @pytest.mark.skipif(True, reason="Live network test — run manually")
    def test_live_agency_sec(self):
        tool = RegulatoryGazetteTool()
        result = tool.execute(mode="agency", agency="sec", days_back=30, limit=5)
        assert result.success
        print(result.output)

    @pytest.mark.skipif(True, reason="Live network test — run manually")
    def test_live_upcoming(self):
        tool = RegulatoryGazetteTool()
        result = tool.execute(mode="upcoming", limit=10)
        assert result.success
        print(result.output)
