"""
Edge case tests for Bankruptcy Court tool (7b-E).

Covers: mode routing, parameter validation, PACER RSS parsing, chapter detection,
court filtering, SEC enforcement RSS parsing, SEC EFTS parsing, UK Gazette Atom
parsing, GOV.UK SFO parsing, keyword filtering, cache integration, HTTP errors,
timeout handling, malformed XML/JSON, partial failures, result formatting,
tool schema, registry integration.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, PropertyMock
from xml.etree.ElementTree import Element, SubElement, tostring

import httpx
import pytest

from agent.tools.bankruptcy_court import (
    PACER_COURTS,
    BankruptcyCourtTool,
    _ATOM_NS,
    _CHAPTER_RE,
    _SEC_ADMIN_RSS,
    _SEC_EFTS,
    _SEC_LIT_RSS,
    _TITLE_RE,
    _UK_GAZETTE,
    _fetch_json,
    _fetch_pacer_court,
    _fetch_xml,
    _keyword_match,
    _parse_chapter,
    _parse_efts_hits,
    _parse_gazette_atom,
    _parse_govuk_results,
    _parse_pacer_feed,
    _parse_pub_date,
    _parse_sec_rss,
)
from agent.tools.base import ToolRegistry, ToolResult


# ── XML builders ──────────────────────────────────────────────────────────────


def _build_pacer_rss(items: list[tuple[str, str, str, str]]) -> bytes:
    """Build PACER RSS XML.  items = [(title, link, desc, pubDate), ...]"""
    rss = Element("rss", version="2.0")
    ch = SubElement(rss, "channel")
    SubElement(ch, "title").text = "Test Court - Recent Entries"
    for title, link, desc, pub_date in items:
        item = SubElement(ch, "item")
        SubElement(item, "title").text = title
        SubElement(item, "link").text = link
        SubElement(item, "description").text = desc
        SubElement(item, "pubDate").text = pub_date
    return tostring(rss)


def _build_sec_rss(items: list[tuple[str, str, str, str]]) -> bytes:
    """Build SEC RSS XML.  items = [(title, link, desc, pubDate), ...]"""
    rss = Element("rss", version="2.0")
    ch = SubElement(rss, "channel")
    SubElement(ch, "title").text = "Administrative Proceedings"
    for title, link, desc, pub_date in items:
        item = SubElement(ch, "item")
        SubElement(item, "title").text = title
        SubElement(item, "link").text = link
        SubElement(item, "description").text = desc
        SubElement(item, "pubDate").text = pub_date
    return tostring(rss)


def _build_atom_feed(entries: list[tuple[str, str, str, str]]) -> bytes:
    """Build Atom XML.  entries = [(title, link_href, updated, summary), ...]"""
    ns = "http://www.w3.org/2005/Atom"
    feed = Element(f"{{{ns}}}feed")
    for title, link_href, updated, summary in entries:
        entry = SubElement(feed, f"{{{ns}}}entry")
        SubElement(entry, f"{{{ns}}}title").text = title
        SubElement(entry, f"{{{ns}}}link", href=link_href)
        SubElement(entry, f"{{{ns}}}updated").text = updated
        SubElement(entry, f"{{{ns}}}summary").text = summary
    return tostring(feed)


def _build_efts_response(
    hits: list[tuple[str, str, str, list[str]]],
    total: int | None = None,
) -> dict:
    """Build EFTS JSON response.  hits = [(company, cik, date, items), ...]"""
    hit_list = []
    for company, cik, date, items in hits:
        hit_list.append(
            {
                "_source": {
                    "display_names": [company],
                    "ciks": [cik],
                    "file_date": date,
                    "form": "8-K",
                    "items": items,
                }
            }
        )
    return {
        "hits": {
            "total": {"value": total if total is not None else len(hits)},
            "hits": hit_list,
        }
    }


def _build_govuk_response(
    results: list[tuple[str, str, str, str]],
) -> dict:
    """Build GOV.UK search JSON.  results = [(title, link, timestamp, desc), ...]"""
    return {
        "total": len(results),
        "results": [
            {
                "title": t,
                "link": lnk,
                "public_timestamp": ts,
                "description": d,
            }
            for t, lnk, ts, d in results
        ],
    }


def _mock_response(
    content: bytes, status_code: int = 200, content_type: str = "text/xml"
):
    """Create a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.content = content
    resp.text = content.decode("utf-8", errors="replace")
    return resp


def _mock_json_response(data: dict, status_code: int = 200):
    """Create a mock httpx.Response for JSON."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = data
    return resp


# ── 1. Helper function tests ─────────────────────────────────────────────────


class TestParseChapter:
    def test_chapter_7(self):
        assert _parse_chapter("Chapter 7 filing") == "7"

    def test_chapter_11(self):
        assert _parse_chapter("Voluntary Chapter 11 petition") == "11"

    def test_chapter_13(self):
        assert _parse_chapter("ch. 13 wage earner") == "13"

    def test_chapter_15(self):
        assert _parse_chapter("Chapter 15 cross-border") == "15"

    def test_ch_dot_notation(self):
        assert _parse_chapter("ch.11 reorganization") == "11"

    def test_ch_space_notation(self):
        assert _parse_chapter("Ch 7 liquidation") == "7"

    def test_no_chapter(self):
        assert _parse_chapter("Motion to dismiss") is None

    def test_invalid_chapter_99(self):
        assert _parse_chapter("Chapter 99 filing") is None

    def test_empty_string(self):
        assert _parse_chapter("") is None

    def test_chapter_embedded(self):
        assert _parse_chapter("Debtor filed chapter 11 petition on March 1") == "11"


class TestParsePubDate:
    def test_rfc2822(self):
        result = _parse_pub_date("Thu, 26 Mar 2026 16:14:40 -0400")
        assert result == "2026-03-26 16:14"

    def test_iso_zulu(self):
        result = _parse_pub_date("2026-03-26T16:14:40Z")
        assert result == "2026-03-26 16:14"

    def test_iso_offset(self):
        result = _parse_pub_date("2026-03-26T16:14:40+0000")
        assert result == "2026-03-26 16:14"

    def test_date_only(self):
        result = _parse_pub_date("2026-03-26")
        assert result == "2026-03-26 00:00"

    def test_empty(self):
        assert _parse_pub_date("") == ""

    def test_unparseable(self):
        result = _parse_pub_date("not a date")
        assert result == "not a date"

    def test_whitespace_stripped(self):
        result = _parse_pub_date("  2026-03-26  ")
        assert result == "2026-03-26 00:00"


class TestKeywordMatch:
    def test_match(self):
        assert _keyword_match("SEC enforcement against FooBar Inc", "foobar")

    def test_no_match(self):
        assert not _keyword_match("SEC enforcement against FooBar Inc", "xyz")

    def test_empty_keyword_matches_all(self):
        assert _keyword_match("anything", "")

    def test_case_insensitive(self):
        assert _keyword_match("BANKRUPTCY filing", "bankruptcy")

    def test_empty_text(self):
        assert not _keyword_match("", "keyword")


class TestTitleRegex:
    def test_standard_case(self):
        m = _TITLE_RE.match("26-12345 John Doe Corp")
        assert m
        assert m.group(1) == "26-12345"
        assert m.group(2) == "John Doe Corp"

    def test_complex_case_number(self):
        m = _TITLE_RE.match("6:26-bk-12331 Coachella Valley Economic Partnership")
        assert m
        assert m.group(1) == "6:26-bk-12331"
        assert m.group(2) == "Coachella Valley Economic Partnership"

    def test_no_space(self):
        # Single word — no space means no debtor name split
        m = _TITLE_RE.match("26-12345")
        assert m is None  # need at least one space


# ── 2. Mode routing ──────────────────────────────────────────────────────────


class TestModeRouting:
    def setup_method(self):
        self.tool = BankruptcyCourtTool(cache=None)

    def test_invalid_mode(self):
        r = self.tool.execute(mode="invalid")
        assert not r.success
        assert "Invalid mode" in r.output

    def test_empty_mode(self):
        r = self.tool.execute(mode="")
        assert not r.success

    @patch("agent.tools.bankruptcy_court.httpx.Client")
    def test_us_bankruptcy_mode(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_resp = _mock_response(_build_pacer_rss([]))
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value = mock_client
        r = self.tool.execute(mode="us_bankruptcy", court="sdny")
        assert r.success

    @patch("agent.tools.bankruptcy_court.httpx.Client")
    def test_sec_enforcement_mode(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_resp = _mock_response(_build_sec_rss([]))
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value = mock_client
        r = self.tool.execute(mode="sec_enforcement")
        assert r.success

    @patch("agent.tools.bankruptcy_court.httpx.Client")
    def test_sec_bankruptcy_mode(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        data = _build_efts_response([])
        mock_resp = _mock_json_response(data)
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value = mock_client
        r = self.tool.execute(mode="sec_bankruptcy")
        assert r.success

    @patch("agent.tools.bankruptcy_court.httpx.Client")
    def test_uk_insolvency_mode(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        xml_resp = _mock_response(_build_atom_feed([]))
        json_resp = _mock_json_response(_build_govuk_response([]))
        mock_client.get.side_effect = [xml_resp, json_resp]
        mock_client_cls.return_value = mock_client
        r = self.tool.execute(mode="uk_insolvency")
        assert r.success

    def test_mode_case_insensitive(self):
        # Invalid mode but tests lowercasing
        r = self.tool.execute(mode="INVALID_MODE")
        assert not r.success
        assert "Invalid mode" in r.output


# ── 3. Parameter validation ──────────────────────────────────────────────────


class TestParameterValidation:
    def setup_method(self):
        self.tool = BankruptcyCourtTool(cache=None)

    def test_days_back_clamped_min(self):
        """days_back=0 should clamp to 1."""
        with patch.object(self.tool, "_sec_bankruptcy") as mock:
            mock.return_value = ToolResult(success=True, output="ok")
            self.tool.execute(mode="sec_bankruptcy", days_back=0)
            mock.assert_called_once_with(days_back=1, limit=25)

    def test_days_back_clamped_max(self):
        """days_back=999 should clamp to 90."""
        with patch.object(self.tool, "_sec_bankruptcy") as mock:
            mock.return_value = ToolResult(success=True, output="ok")
            self.tool.execute(mode="sec_bankruptcy", days_back=999)
            mock.assert_called_once_with(days_back=90, limit=25)

    def test_limit_clamped_min(self):
        """limit=0 should clamp to 1."""
        with patch.object(self.tool, "_sec_enforcement") as mock:
            mock.return_value = ToolResult(success=True, output="ok")
            self.tool.execute(mode="sec_enforcement", limit=0)
            mock.assert_called_once_with(keyword="", limit=1)

    def test_limit_clamped_max(self):
        """limit=999 should clamp to 100."""
        with patch.object(self.tool, "_sec_enforcement") as mock:
            mock.return_value = ToolResult(success=True, output="ok")
            self.tool.execute(mode="sec_enforcement", limit=999)
            mock.assert_called_once_with(keyword="", limit=100)

    def test_keyword_stripped(self):
        with patch.object(self.tool, "_sec_enforcement") as mock:
            mock.return_value = ToolResult(success=True, output="ok")
            self.tool.execute(mode="sec_enforcement", keyword="  hello  ")
            mock.assert_called_once_with(keyword="hello", limit=25)

    def test_invalid_court(self):
        r = self.tool.execute(mode="us_bankruptcy", court="fake_court")
        assert not r.success
        assert "Invalid court" in r.output

    def test_days_back_negative(self):
        """Negative days_back should clamp to 1."""
        with patch.object(self.tool, "_sec_bankruptcy") as mock:
            mock.return_value = ToolResult(success=True, output="ok")
            self.tool.execute(mode="sec_bankruptcy", days_back=-5)
            mock.assert_called_once_with(days_back=1, limit=25)


# ── 4. PACER RSS parsing ─────────────────────────────────────────────────────


class TestPACERParsing:
    def test_basic_item(self):
        xml = _build_pacer_rss(
            [
                (
                    "26-12345 FooCorp Inc",
                    "https://court.example.com/case/1",
                    "Chapter 11 voluntary petition",
                    "Thu, 26 Mar 2026 16:14:40 -0400",
                )
            ]
        )
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml)
        entries = _parse_pacer_feed(root, "sdny", "S.D. New York")
        assert len(entries) == 1
        e = entries[0]
        assert e["case_number"] == "26-12345"
        assert e["debtor_name"] == "FooCorp Inc"
        assert e["chapter"] == "11"
        assert e["court"] == "sdny"
        assert "2026-03-26" in e["pub_date"]

    def test_chapter_7_detection(self):
        xml = _build_pacer_rss(
            [("1:26-bk-999 John Doe", "http://x", "Ch 7 liquidation", "")]
        )
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml)
        entries = _parse_pacer_feed(root, "ndil", "N.D. Illinois")
        assert entries[0]["chapter"] == "7"

    def test_no_chapter_in_desc(self):
        xml = _build_pacer_rss(
            [("26-100 Jane Smith", "http://x", "Motion to dismiss case", "")]
        )
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml)
        entries = _parse_pacer_feed(root, "del", "Delaware")
        assert entries[0]["chapter"] is None

    def test_empty_feed(self):
        xml = _build_pacer_rss([])
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml)
        entries = _parse_pacer_feed(root, "sdtx", "S.D. Texas")
        assert entries == []

    def test_missing_channel(self):
        """RSS with no <channel> element."""
        rss = Element("rss")
        entries = _parse_pacer_feed(rss, "sdny", "S.D. New York")
        assert entries == []

    def test_title_without_space(self):
        """Title with no space — case_number empty, debtor is full title."""
        xml = _build_pacer_rss([("SINGLEWORD", "http://x", "desc", "")])
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml)
        entries = _parse_pacer_feed(root, "nj", "D. New Jersey")
        assert entries[0]["case_number"] == ""
        assert entries[0]["debtor_name"] == "SINGLEWORD"

    def test_unicode_debtor_name(self):
        xml = _build_pacer_rss(
            [("26-999 José García LLC", "http://x", "Chapter 11 filing", "")]
        )
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml)
        entries = _parse_pacer_feed(root, "cdca", "C.D. California")
        assert "José García" in entries[0]["debtor_name"]

    def test_description_truncated(self):
        long_desc = "A" * 500
        xml = _build_pacer_rss([("26-1 Test", "http://x", long_desc, "")])
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml)
        entries = _parse_pacer_feed(root, "sdny", "S.D. New York")
        assert len(entries[0]["description"]) <= 300

    def test_multiple_items(self):
        items = [
            (f"26-{i} Corp{i}", "http://x", f"Chapter {11 if i % 2 else 7}", "")
            for i in range(10)
        ]
        xml = _build_pacer_rss(items)
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml)
        entries = _parse_pacer_feed(root, "del", "Delaware")
        assert len(entries) == 10


class TestPACERCourtFiltering:
    def setup_method(self):
        self.tool = BankruptcyCourtTool(cache=None)

    @patch("agent.tools.bankruptcy_court.httpx.Client")
    def test_single_court(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        xml = _build_pacer_rss([("26-1 TestCo", "http://x", "Chapter 11", "")])
        mock_resp = _mock_response(xml)
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        r = self.tool.execute(mode="us_bankruptcy", court="sdny")
        assert r.success
        assert r.data["court_breakdown"]["S.D. New York"] == 1

    @patch("agent.tools.bankruptcy_court._fetch_pacer_court")
    @patch("agent.tools.bankruptcy_court.httpx.Client")
    def test_all_courts(self, mock_client_cls, mock_fetch):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        mock_fetch.return_value = [
            {
                "case_number": "1",
                "debtor_name": "X",
                "chapter": "11",
                "court": "sdny",
                "court_name": "S.D. New York",
                "link": "",
                "pub_date": "",
                "description": "",
            }
        ]

        r = self.tool.execute(mode="us_bankruptcy", court="all")
        assert r.success
        # Called 6 times (once per court)
        assert mock_fetch.call_count == 6

    def test_invalid_court_code(self):
        r = self.tool.execute(mode="us_bankruptcy", court="fake")
        assert not r.success
        assert "Invalid court" in r.output

    def test_court_case_insensitive(self):
        """Court code should be lowercased."""
        r = self.tool.execute(mode="us_bankruptcy", court="FAKE")
        assert not r.success  # still invalid, but was lowercased

    @patch("agent.tools.bankruptcy_court.httpx.Client")
    def test_each_valid_court(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        xml = _build_pacer_rss([])
        mock_resp = _mock_response(xml)
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        for court_code in PACER_COURTS:
            r = self.tool.execute(mode="us_bankruptcy", court=court_code)
            assert r.success, f"Court {court_code} failed"


# ── 5. PACER error handling ──────────────────────────────────────────────────


class TestPACERErrorHandling:
    def test_fetch_xml_404(self):
        client = MagicMock()
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 404
        client.get.return_value = resp
        result = _fetch_xml("http://example.com", client)
        assert result is None

    def test_fetch_xml_timeout(self):
        client = MagicMock()
        client.get.side_effect = httpx.ConnectTimeout("timeout")
        result = _fetch_xml("http://example.com", client)
        assert result is None

    def test_fetch_xml_malformed(self):
        client = MagicMock()
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.content = b"<not valid xml"
        client.get.return_value = resp
        result = _fetch_xml("http://example.com", client)
        assert result is None

    @patch("agent.tools.bankruptcy_court._fetch_pacer_court")
    @patch("agent.tools.bankruptcy_court.httpx.Client")
    def test_partial_failure(self, mock_client_cls, mock_fetch):
        """Some courts fail, others succeed — should return partial results."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        call_count = [0]

        def side_effect(court_code, client):
            call_count[0] += 1
            if court_code in ("sdny", "del"):
                return [
                    {
                        "case_number": "1",
                        "debtor_name": "X",
                        "chapter": "11",
                        "court": court_code,
                        "court_name": PACER_COURTS[court_code][0],
                        "link": "",
                        "pub_date": "",
                        "description": "",
                    }
                ]
            raise httpx.ConnectTimeout("timeout")

        mock_fetch.side_effect = side_effect

        tool = BankruptcyCourtTool(cache=None)
        r = tool.execute(mode="us_bankruptcy", court="all")
        assert r.success
        # Should have partial results from courts that succeeded
        assert r.data["count"] >= 2


# ── 6. SEC enforcement parsing ───────────────────────────────────────────────


class TestSECEnforcementParsing:
    def test_admin_proceedings(self):
        xml = _build_sec_rss(
            [
                (
                    "GrubMarket Inc.",
                    "https://sec.gov/lit/admin/123.pdf",
                    "GrubMarket Inc.",
                    "Thu, 26 Mar 2026 16:14:40 -0400",
                ),
            ]
        )
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml)
        entries = _parse_sec_rss(root, "admin")
        assert len(entries) == 1
        assert entries[0]["title"] == "GrubMarket Inc."
        assert entries[0]["type"] == "admin"

    def test_litigation_release(self):
        xml = _build_sec_rss(
            [
                (
                    "SEC v. Bad Actor",
                    "https://sec.gov/lit/123.htm",
                    "Securities fraud",
                    "2026-03-26T12:00:00Z",
                ),
            ]
        )
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml)
        entries = _parse_sec_rss(root, "litigation")
        assert entries[0]["type"] == "litigation"

    def test_empty_rss(self):
        xml = _build_sec_rss([])
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml)
        entries = _parse_sec_rss(root, "admin")
        assert entries == []

    def test_missing_channel(self):
        rss = Element("rss")
        entries = _parse_sec_rss(rss, "admin")
        assert entries == []

    @patch("agent.tools.bankruptcy_court._fetch_xml")
    @patch("agent.tools.bankruptcy_court.httpx.Client")
    def test_both_feeds_merged(self, mock_client_cls, mock_fetch_xml):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        # Build two different RSS feeds
        admin_xml = _build_sec_rss(
            [
                ("Admin Case 1", "http://x", "desc", "2026-03-26T12:00:00Z"),
            ]
        )
        lit_xml = _build_sec_rss(
            [
                ("Lit Case 1", "http://y", "desc", "2026-03-25T12:00:00Z"),
            ]
        )

        import xml.etree.ElementTree as ET

        call_count = [0]

        def side_effect(url, client):
            call_count[0] += 1
            if "admin" in url:
                return ET.fromstring(admin_xml)
            return ET.fromstring(lit_xml)

        mock_fetch_xml.side_effect = side_effect

        tool = BankruptcyCourtTool(cache=None)
        r = tool.execute(mode="sec_enforcement")
        assert r.success
        assert r.data["count"] == 2
        assert r.data["type_breakdown"]["admin"] == 1
        assert r.data["type_breakdown"]["litigation"] == 1

    @patch("agent.tools.bankruptcy_court._fetch_xml")
    @patch("agent.tools.bankruptcy_court.httpx.Client")
    def test_keyword_filter(self, mock_client_cls, mock_fetch_xml):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        xml = _build_sec_rss(
            [
                ("SEC v. FooBar Inc", "http://x", "fraud", "2026-03-26T12:00:00Z"),
                (
                    "SEC v. Other Co",
                    "http://y",
                    "insider trading",
                    "2026-03-25T12:00:00Z",
                ),
            ]
        )
        import xml.etree.ElementTree as ET

        call_count = [0]

        def side_effect(url, client):
            call_count[0] += 1
            if call_count[0] == 1:
                return ET.fromstring(xml)
            return None  # second feed returns nothing

        mock_fetch_xml.side_effect = side_effect

        tool = BankruptcyCourtTool(cache=None)
        r = tool.execute(mode="sec_enforcement", keyword="FooBar")
        assert r.success
        assert r.data["count"] == 1
        assert "FooBar" in r.data["entries"][0]["title"]


# ── 7. SEC enforcement errors ────────────────────────────────────────────────


class TestSECEnforcementErrors:
    @patch("agent.tools.bankruptcy_court._fetch_xml")
    @patch("agent.tools.bankruptcy_court.httpx.Client")
    def test_both_feeds_fail(self, mock_client_cls, mock_fetch_xml):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client
        mock_fetch_xml.return_value = None  # both fail

        tool = BankruptcyCourtTool(cache=None)
        r = tool.execute(mode="sec_enforcement")
        assert r.success  # still succeeds with 0 entries
        assert r.data["count"] == 0

    @patch("agent.tools.bankruptcy_court._fetch_xml")
    @patch("agent.tools.bankruptcy_court.httpx.Client")
    def test_one_feed_fails(self, mock_client_cls, mock_fetch_xml):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        xml = _build_sec_rss([("Test", "http://x", "desc", "")])
        import xml.etree.ElementTree as ET

        call_count = [0]

        def side_effect(url, client):
            call_count[0] += 1
            if call_count[0] == 1:
                return ET.fromstring(xml)
            return None  # second feed fails

        mock_fetch_xml.side_effect = side_effect

        tool = BankruptcyCourtTool(cache=None)
        r = tool.execute(mode="sec_enforcement")
        assert r.success
        assert r.data["count"] == 1


# ── 8. SEC EFTS bankruptcy parsing ───────────────────────────────────────────


class TestSECBankruptcyParsing:
    def test_basic_parse(self):
        data = _build_efts_response(
            [
                ("FooCorp (CIK 123)", "0000000123", "2026-03-20", ["1.03"]),
            ]
        )
        entries = _parse_efts_hits(data)
        assert len(entries) == 1
        assert entries[0]["company_name"] == "FooCorp (CIK 123)"
        assert entries[0]["cik"] == "0000000123"
        assert entries[0]["items"] == ["1.03"]

    def test_multiple_hits(self):
        data = _build_efts_response(
            [
                ("A Corp", "1", "2026-03-20", ["1.03"]),
                ("B Corp", "2", "2026-03-19", ["1.03", "2.04"]),
            ]
        )
        entries = _parse_efts_hits(data)
        assert len(entries) == 2

    def test_empty_hits(self):
        data = _build_efts_response([])
        entries = _parse_efts_hits(data)
        assert entries == []

    def test_missing_fields(self):
        """Hit with minimal _source data."""
        data = {"hits": {"total": {"value": 1}, "hits": [{"_source": {}}]}}
        entries = _parse_efts_hits(data)
        assert len(entries) == 1
        assert entries[0]["company_name"] == "Unknown"
        assert entries[0]["cik"] == ""

    @patch("agent.tools.bankruptcy_court._fetch_json")
    @patch("agent.tools.bankruptcy_court.httpx.Client")
    def test_sec_bankruptcy_mode_full(self, mock_client_cls, mock_fetch_json):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        data = _build_efts_response(
            [("TestCo", "123", "2026-03-20", ["1.03"])],
            total=42,
        )
        mock_fetch_json.return_value = data

        tool = BankruptcyCourtTool(cache=None)
        r = tool.execute(mode="sec_bankruptcy", days_back=7)
        assert r.success
        assert r.data["total"] == 42
        assert r.data["count"] == 1


# ── 9. SEC EFTS errors ───────────────────────────────────────────────────────


class TestSECBankruptcyErrors:
    @patch("agent.tools.bankruptcy_court._fetch_json")
    @patch("agent.tools.bankruptcy_court.httpx.Client")
    def test_efts_returns_none(self, mock_client_cls, mock_fetch_json):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client
        mock_fetch_json.return_value = None  # HTTP error

        tool = BankruptcyCourtTool(cache=None)
        r = tool.execute(mode="sec_bankruptcy")
        assert not r.success
        assert "unavailable" in r.output.lower()

    def test_fetch_json_http_error(self):
        client = MagicMock()
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 403
        client.get.return_value = resp
        result = _fetch_json("http://example.com", client)
        assert result is None

    def test_fetch_json_timeout(self):
        client = MagicMock()
        client.get.side_effect = httpx.ReadTimeout("timeout")
        result = _fetch_json("http://example.com", client)
        assert result is None

    def test_fetch_json_invalid_json(self):
        client = MagicMock()
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.side_effect = ValueError("bad json")
        client.get.return_value = resp
        result = _fetch_json("http://example.com", client)
        assert result is None


# ── 10. UK Gazette Atom parsing ──────────────────────────────────────────────


class TestGazetteAtomParsing:
    def test_basic_entry(self):
        xml = _build_atom_feed(
            [
                (
                    "Winding-Up Petition: FooCo Ltd",
                    "https://gazette.co.uk/notice/123",
                    "2026-03-26T12:00:00Z",
                    "Petition for winding up",
                ),
            ]
        )
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml)
        entries = _parse_gazette_atom(root)
        assert len(entries) == 1
        assert entries[0]["title"] == "Winding-Up Petition: FooCo Ltd"
        assert entries[0]["source"] == "gazette"
        assert "2026-03-26" in entries[0]["pub_date"]

    def test_empty_feed(self):
        xml = _build_atom_feed([])
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml)
        entries = _parse_gazette_atom(root)
        assert entries == []

    def test_missing_link(self):
        """Entry without <link>."""
        ns = "http://www.w3.org/2005/Atom"
        feed = Element(f"{{{ns}}}feed")
        entry = SubElement(feed, f"{{{ns}}}entry")
        SubElement(entry, f"{{{ns}}}title").text = "Test"
        SubElement(entry, f"{{{ns}}}updated").text = "2026-01-01"
        entries = _parse_gazette_atom(feed)
        assert len(entries) == 1
        assert entries[0]["link"] == ""

    def test_summary_truncated(self):
        long_summary = "B" * 500
        xml = _build_atom_feed([("Title", "http://x", "2026-01-01", long_summary)])
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml)
        entries = _parse_gazette_atom(root)
        assert len(entries[0]["description"]) <= 300


# ── 11. GOV.UK SFO parsing ──────────────────────────────────────────────────


class TestGovUKParsing:
    def test_basic_results(self):
        data = _build_govuk_response(
            [
                (
                    "SFO Investigation into XYZ",
                    "/government/news/sfo-xyz",
                    "2026-03-26T12:00:00Z",
                    "The SFO has opened an investigation...",
                ),
            ]
        )
        entries = _parse_govuk_results(data)
        assert len(entries) == 1
        assert entries[0]["source"] == "sfo"
        assert entries[0]["link"] == "https://www.gov.uk/government/news/sfo-xyz"
        assert "2026-03-26" in entries[0]["pub_date"]

    def test_empty_results(self):
        data = _build_govuk_response([])
        entries = _parse_govuk_results(data)
        assert entries == []

    def test_missing_timestamp(self):
        data = _build_govuk_response([("Title", "/link", "", "desc")])
        entries = _parse_govuk_results(data)
        assert entries[0]["pub_date"] == ""

    def test_description_truncated(self):
        data = _build_govuk_response([("T", "/l", "2026-01-01T00:00:00Z", "C" * 500)])
        entries = _parse_govuk_results(data)
        assert len(entries[0]["description"]) <= 300


# ── 12. UK insolvency combined ───────────────────────────────────────────────


class TestUKInsolvencyCombined:
    @patch("agent.tools.bankruptcy_court._fetch_json")
    @patch("agent.tools.bankruptcy_court._fetch_xml")
    @patch("agent.tools.bankruptcy_court.httpx.Client")
    def test_both_sources(self, mock_client_cls, mock_fetch_xml, mock_fetch_json):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        xml = _build_atom_feed(
            [("Gazette Notice", "http://x", "2026-03-26T12:00:00Z", "desc")]
        )
        import xml.etree.ElementTree as ET

        mock_fetch_xml.return_value = ET.fromstring(xml)
        mock_fetch_json.return_value = _build_govuk_response(
            [
                ("SFO Case", "/sfo", "2026-03-25T12:00:00Z", "desc"),
            ]
        )

        tool = BankruptcyCourtTool(cache=None)
        r = tool.execute(mode="uk_insolvency")
        assert r.success
        assert r.data["count"] == 2
        assert r.data["source_breakdown"]["gazette"] == 1
        assert r.data["source_breakdown"]["sfo"] == 1

    @patch("agent.tools.bankruptcy_court._fetch_json")
    @patch("agent.tools.bankruptcy_court._fetch_xml")
    @patch("agent.tools.bankruptcy_court.httpx.Client")
    def test_keyword_filter_uk(self, mock_client_cls, mock_fetch_xml, mock_fetch_json):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        xml = _build_atom_feed(
            [
                ("Winding up FooCo", "http://x", "2026-01-01T00:00:00Z", "desc"),
                ("Admin of BarCo", "http://y", "2026-01-01T00:00:00Z", "desc"),
            ]
        )
        import xml.etree.ElementTree as ET

        mock_fetch_xml.return_value = ET.fromstring(xml)
        mock_fetch_json.return_value = _build_govuk_response([])

        tool = BankruptcyCourtTool(cache=None)
        r = tool.execute(mode="uk_insolvency", keyword="FooCo")
        assert r.success
        assert r.data["count"] == 1

    @patch("agent.tools.bankruptcy_court._fetch_json")
    @patch("agent.tools.bankruptcy_court._fetch_xml")
    @patch("agent.tools.bankruptcy_court.httpx.Client")
    def test_both_fail(self, mock_client_cls, mock_fetch_xml, mock_fetch_json):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client
        mock_fetch_xml.return_value = None
        mock_fetch_json.return_value = None

        tool = BankruptcyCourtTool(cache=None)
        r = tool.execute(mode="uk_insolvency")
        assert r.success
        assert r.data["count"] == 0


# ── 13. Cache integration ────────────────────────────────────────────────────


class TestCacheIntegration:
    def test_pacer_cache_hit(self):
        cache = MagicMock()
        cached_entries = [
            {
                "case_number": "1",
                "debtor_name": "CachedCo",
                "chapter": "11",
                "court": "sdny",
                "court_name": "S.D. New York",
                "link": "",
                "pub_date": "",
                "description": "",
            }
        ]
        cache.get.return_value = cached_entries

        tool = BankruptcyCourtTool(cache=cache)
        r = tool.execute(mode="us_bankruptcy", court="sdny")
        assert r.success
        assert "(cached)" in r.output
        cache.get.assert_called_once()

    def test_pacer_cache_miss(self):
        cache = MagicMock()
        cache.get.return_value = None

        with patch("agent.tools.bankruptcy_court.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_resp = _mock_response(_build_pacer_rss([]))
            mock_client.get.return_value = mock_resp
            mock_cls.return_value = mock_client

            tool = BankruptcyCourtTool(cache=cache)
            r = tool.execute(mode="us_bankruptcy", court="sdny")
            assert r.success
            cache.set.assert_called_once()

    def test_sec_enforcement_cache_hit(self):
        cache = MagicMock()
        cache.get.return_value = [
            {
                "title": "Cached",
                "type": "admin",
                "link": "",
                "pub_date": "",
                "description": "",
            }
        ]

        tool = BankruptcyCourtTool(cache=cache)
        r = tool.execute(mode="sec_enforcement")
        assert r.success
        assert "(cached)" in r.output

    def test_sec_bankruptcy_cache_hit(self):
        cache = MagicMock()
        cache.get.return_value = [
            {
                "company_name": "Cached",
                "cik": "1",
                "file_date": "",
                "form": "8-K",
                "items": ["1.03"],
            }
        ]

        tool = BankruptcyCourtTool(cache=cache)
        r = tool.execute(mode="sec_bankruptcy")
        assert r.success
        assert "(cached)" in r.output

    def test_uk_cache_hit(self):
        cache = MagicMock()
        cache.get.return_value = [
            {
                "title": "Cached",
                "source": "gazette",
                "link": "",
                "pub_date": "",
                "description": "",
            }
        ]

        tool = BankruptcyCourtTool(cache=cache)
        r = tool.execute(mode="uk_insolvency")
        assert r.success
        assert "(cached)" in r.output

    def test_no_cache(self):
        """Tool works without cache (cache=None)."""
        with patch("agent.tools.bankruptcy_court.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_resp = _mock_response(_build_pacer_rss([]))
            mock_client.get.return_value = mock_resp
            mock_cls.return_value = mock_client

            tool = BankruptcyCourtTool(cache=None)
            r = tool.execute(mode="us_bankruptcy", court="sdny")
            assert r.success


# ── 14. Tool schema ──────────────────────────────────────────────────────────


class TestToolSchema:
    def test_schema_structure(self):
        tool = BankruptcyCourtTool()
        assert tool.parameters["type"] == "object"
        assert "mode" in tool.parameters["properties"]
        assert "court" in tool.parameters["properties"]
        assert "keyword" in tool.parameters["properties"]
        assert "days_back" in tool.parameters["properties"]
        assert "limit" in tool.parameters["properties"]

    def test_mode_enum(self):
        tool = BankruptcyCourtTool()
        modes = tool.parameters["properties"]["mode"]["enum"]
        assert "us_bankruptcy" in modes
        assert "sec_enforcement" in modes
        assert "sec_bankruptcy" in modes
        assert "uk_insolvency" in modes
        assert len(modes) == 4

    def test_required_fields(self):
        tool = BankruptcyCourtTool()
        assert tool.parameters["required"] == ["mode"]


# ── 15. Registry integration ─────────────────────────────────────────────────


class TestRegistryIntegration:
    def test_register_and_lookup(self):
        registry = ToolRegistry()
        tool = BankruptcyCourtTool()
        registry.register(tool)
        assert registry.get("bankruptcy_court") is tool

    def test_name(self):
        tool = BankruptcyCourtTool()
        assert tool.name == "bankruptcy_court"

    def test_description_nonempty(self):
        tool = BankruptcyCourtTool()
        assert len(tool.description) > 50


# ── 16. Result formatting ────────────────────────────────────────────────────


class TestResultFormat:
    def test_pacer_result_data_shape(self):
        cache = MagicMock()
        cache.get.return_value = [
            {
                "case_number": f"{i}",
                "debtor_name": f"Corp{i}",
                "chapter": "11",
                "court": "sdny",
                "court_name": "S.D. New York",
                "link": "",
                "pub_date": "2026-03-26",
                "description": "",
            }
            for i in range(5)
        ]
        tool = BankruptcyCourtTool(cache=cache)
        r = tool.execute(mode="us_bankruptcy", court="sdny")
        assert r.data["mode"] == "us_bankruptcy"
        assert r.data["count"] == 5
        assert "chapter_breakdown" in r.data
        assert "court_breakdown" in r.data
        assert len(r.data["entries"]) == 5

    def test_sec_enforce_result_data_shape(self):
        cache = MagicMock()
        cache.get.return_value = [
            {
                "title": "X",
                "type": "admin",
                "link": "",
                "pub_date": "",
                "description": "",
            }
        ]
        tool = BankruptcyCourtTool(cache=cache)
        r = tool.execute(mode="sec_enforcement")
        assert r.data["mode"] == "sec_enforcement"
        assert "type_breakdown" in r.data

    def test_sec_bk_result_data_shape(self):
        cache = MagicMock()
        cache.get.return_value = [
            {
                "company_name": "X",
                "cik": "1",
                "file_date": "",
                "form": "8-K",
                "items": ["1.03"],
            }
        ]
        tool = BankruptcyCourtTool(cache=cache)
        r = tool.execute(mode="sec_bankruptcy")
        assert r.data["mode"] == "sec_bankruptcy"
        assert "total" in r.data
        assert "days_back" in r.data

    def test_uk_result_data_shape(self):
        cache = MagicMock()
        cache.get.return_value = [
            {
                "title": "X",
                "source": "gazette",
                "link": "",
                "pub_date": "",
                "description": "",
            }
        ]
        tool = BankruptcyCourtTool(cache=cache)
        r = tool.execute(mode="uk_insolvency")
        assert r.data["mode"] == "uk_insolvency"
        assert "source_breakdown" in r.data

    def test_limit_respected_in_output(self):
        """When more entries than limit, only limit entries returned."""
        cache = MagicMock()
        cache.get.return_value = [
            {
                "case_number": f"{i}",
                "debtor_name": f"Corp{i}",
                "chapter": "11",
                "court": "sdny",
                "court_name": "S.D. New York",
                "link": "",
                "pub_date": "",
                "description": "",
            }
            for i in range(50)
        ]
        tool = BankruptcyCourtTool(cache=cache)
        r = tool.execute(mode="us_bankruptcy", court="sdny", limit=10)
        assert r.data["count"] == 10
        assert len(r.data["entries"]) == 10

    def test_text_output_truncated_at_15(self):
        """Text output shows max 15 entries, with '... and N more'."""
        cache = MagicMock()
        cache.get.return_value = [
            {
                "case_number": f"{i}",
                "debtor_name": f"Corp{i}",
                "chapter": "11",
                "court": "sdny",
                "court_name": "S.D. New York",
                "link": "",
                "pub_date": "",
                "description": "",
            }
            for i in range(30)
        ]
        tool = BankruptcyCourtTool(cache=cache)
        r = tool.execute(mode="us_bankruptcy", court="sdny", limit=30)
        assert "... and 15 more" in r.output


# ── 17. PACER courts dict ────────────────────────────────────────────────────


class TestPACERCourtsDict:
    def test_six_courts(self):
        assert len(PACER_COURTS) == 6

    def test_all_have_name_and_domain(self):
        for code, (name, domain) in PACER_COURTS.items():
            assert name, f"Court {code} missing name"
            assert "uscourts.gov" in domain, f"Court {code} bad domain: {domain}"

    def test_expected_courts_present(self):
        expected = {"sdny", "del", "sdtx", "cdca", "ndil", "nj"}
        assert set(PACER_COURTS.keys()) == expected
