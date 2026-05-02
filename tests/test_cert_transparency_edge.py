"""
Edge case tests for CertTransparencyTool (crt.sh CT log monitor).

Covers: mode validation, parse_timestamp (happy path, fractional seconds,
None, empty, malformed), shorten_issuer (DN extraction, truncation, empty),
normalize_record (expired, active, no dates, boundary), format_cert
(expired, expiring soon, normal, brief mode), search mode (results, empty,
exclude_expired flag, limit), subdomains mode (wildcard discovery, concrete
vs wildcard separation, dedup, empty), recent mode (days_back filtering,
dedup by serial, empty, boundary dates), fetch_crtsh (timeout, 503,
HTTP error, connection error, invalid JSON, non-list response),
cache interaction (hit, miss, put), tool metadata (name, description,
parameters, required), input validation (missing domain, empty domain,
invalid mode, days_back bounds, limit bounds), output formatting,
integration of count assertions (32 tools, 21 arms).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.tools.cert_transparency import (
    _CACHE_TTL,
    VALID_MODES,
    CertTransparencyTool,
    _format_cert,
    _normalize_record,
    _parse_timestamp,
    _shorten_issuer,
)

# ── Timestamps ───────────────────────────────────────────────

NOW = datetime(2026, 3, 28, 12, 0, 0, tzinfo=UTC)
YESTERDAY = (NOW - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
LAST_WEEK = (NOW - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")
LAST_MONTH = (NOW - timedelta(days=31)).strftime("%Y-%m-%dT%H:%M:%S")
LAST_YEAR = (NOW - timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%S")
FUTURE_90D = (NOW + timedelta(days=90)).strftime("%Y-%m-%dT%H:%M:%S")
FUTURE_10D = (NOW + timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%S")
PAST_EXPIRED = (NOW - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%S")


# ── Mock Data ────────────────────────────────────────────────


def _make_record(
    *,
    crt_id: int = 1,
    common_name: str = "api.stripe.com",
    name_value: str = "api.stripe.com",
    issuer_name: str = 'C=US, O="DigiCert, Inc.", CN=DigiCert SHA2 Extended Validation Server CA',
    entry_timestamp: str = YESTERDAY,
    not_before: str = LAST_WEEK,
    not_after: str = FUTURE_90D,
    serial_number: str = "0a1b2c3d4e5f",
    result_count: int = 1,
) -> dict[str, Any]:
    return {
        "id": crt_id,
        "issuer_ca_id": 237361,
        "issuer_name": issuer_name,
        "common_name": common_name,
        "name_value": name_value,
        "entry_timestamp": entry_timestamp,
        "not_before": not_before,
        "not_after": not_after,
        "serial_number": serial_number,
        "result_count": result_count,
    }


MOCK_SEARCH_RESULTS = [
    _make_record(crt_id=1, entry_timestamp=YESTERDAY, not_after=FUTURE_90D),
    _make_record(
        crt_id=2,
        common_name="api.stripe.com",
        entry_timestamp=LAST_WEEK,
        not_after=PAST_EXPIRED,
        serial_number="expired1111",
    ),
    _make_record(
        crt_id=3,
        common_name="api.stripe.com",
        entry_timestamp=LAST_MONTH,
        not_after=FUTURE_10D,
        serial_number="expiring_soon",
    ),
]

MOCK_SUBDOMAIN_RESULTS = [
    _make_record(crt_id=10, common_name="api.stripe.com", serial_number="s1"),
    _make_record(crt_id=11, common_name="dashboard.stripe.com", serial_number="s2"),
    _make_record(crt_id=12, common_name="checkout.stripe.com", serial_number="s3"),
    _make_record(
        crt_id=13,
        common_name="api.stripe.com",
        serial_number="s4",
        entry_timestamp=LAST_WEEK,
    ),
    _make_record(crt_id=14, common_name="*.stripe.com", serial_number="s5"),
    _make_record(crt_id=15, common_name="ai.stripe.com", serial_number="s6"),
]

MOCK_BASE_RESULTS = [
    _make_record(crt_id=20, common_name="stripe.com", serial_number="b1"),
]

MOCK_RECENT_WILDCARD = [
    _make_record(
        crt_id=30,
        common_name="new.stripe.com",
        entry_timestamp=YESTERDAY,
        serial_number="r1",
    ),
    _make_record(
        crt_id=31,
        common_name="api.stripe.com",
        entry_timestamp=YESTERDAY,
        serial_number="r2",
    ),
    _make_record(
        crt_id=32,
        common_name="old.stripe.com",
        entry_timestamp=LAST_YEAR,
        serial_number="r3",
    ),
    # Duplicate (same common_name + serial_number)
    _make_record(
        crt_id=33,
        common_name="api.stripe.com",
        entry_timestamp=YESTERDAY,
        serial_number="r2",
    ),
]

MOCK_RECENT_BASE = [
    _make_record(
        crt_id=40,
        common_name="stripe.com",
        entry_timestamp=YESTERDAY,
        serial_number="rb1",
    ),
]


# ── Helper function tests ────────────────────────────────────


class TestParseTimestamp:
    def test_iso_no_fractional(self):
        result = _parse_timestamp("2026-03-27T07:49:06")
        assert result == datetime(2026, 3, 27, 7, 49, 6, tzinfo=UTC)

    def test_iso_with_fractional(self):
        result = _parse_timestamp("2026-03-27T07:49:06.083")
        assert result is not None
        assert result.year == 2026
        assert result.microsecond == 83000

    def test_none_input(self):
        assert _parse_timestamp(None) is None

    def test_empty_input(self):
        assert _parse_timestamp("") is None

    def test_malformed(self):
        assert _parse_timestamp("not-a-date") is None

    def test_partial_date(self):
        assert _parse_timestamp("2026-03-27") is None

    def test_date_only_no_time(self):
        assert _parse_timestamp("2026-03-27T") is None


class TestShortenIssuer:
    def test_extract_cn(self):
        result = _shorten_issuer('C=US, O="DigiCert, Inc.", CN=DigiCert SHA2 Extended Validation Server CA')
        assert result == "DigiCert SHA2 Extended Validation Server CA"

    def test_no_cn(self):
        result = _shorten_issuer("C=US, O=SomeCompany")
        assert "SomeCompany" in result

    def test_empty(self):
        assert _shorten_issuer("") == ""

    def test_long_truncation(self):
        long = "A" * 200
        result = _shorten_issuer(long)
        assert len(result) <= 81  # 80 + "…"
        assert result.endswith("…")

    def test_cn_at_start(self):
        result = _shorten_issuer("CN=My CA, O=Org")
        assert result == "My CA"


class TestNormalizeRecord:
    def test_active_cert(self):
        rec = _make_record(not_after=FUTURE_90D)
        result = _normalize_record(rec, NOW)
        assert result["is_expired"] is False
        assert result["days_remaining"] is not None
        assert result["days_remaining"] > 0

    def test_expired_cert(self):
        rec = _make_record(not_after=PAST_EXPIRED)
        result = _normalize_record(rec, NOW)
        assert result["is_expired"] is True
        assert result["days_remaining"] is None

    def test_no_not_after(self):
        rec = _make_record()
        rec["not_after"] = ""
        result = _normalize_record(rec, NOW)
        assert result["is_expired"] is None
        assert result["days_remaining"] is None

    def test_expiring_soon(self):
        rec = _make_record(not_after=FUTURE_10D)
        result = _normalize_record(rec, NOW)
        assert result["is_expired"] is False
        assert result["days_remaining"] is not None
        assert result["days_remaining"] <= 10

    def test_fields_preserved(self):
        rec = _make_record(crt_id=42, serial_number="abc123")
        result = _normalize_record(rec, NOW)
        assert result["id"] == 42
        assert result["serial_number"] == "abc123"
        assert result["common_name"] == "api.stripe.com"

    def test_issuer_shortened(self):
        rec = _make_record(issuer_name='C=US, O="DigiCert", CN=DigiCert Fancy CA')
        result = _normalize_record(rec, NOW)
        assert result["issuer"] == "DigiCert Fancy CA"
        assert "DigiCert" in result["issuer_full"]


class TestFormatCert:
    def test_expired_annotation(self):
        cert = _normalize_record(_make_record(not_after=PAST_EXPIRED), NOW)
        text = _format_cert(cert)
        assert "[EXPIRED]" in text

    def test_expiring_soon_annotation(self):
        cert = _normalize_record(_make_record(not_after=FUTURE_10D), NOW)
        text = _format_cert(cert)
        assert "EXPIRES IN" in text

    def test_normal_cert(self):
        cert = _normalize_record(_make_record(not_after=FUTURE_90D), NOW)
        text = _format_cert(cert)
        assert "[EXPIRED]" not in text
        assert "EXPIRES IN" not in text

    def test_brief_mode(self):
        cert = _normalize_record(_make_record(), NOW)
        brief = _format_cert(cert, brief=True)
        full = _format_cert(cert, brief=False)
        assert len(brief) < len(full)
        assert "Issuer" not in brief
        assert "Issuer" in full

    def test_common_name_in_output(self):
        cert = _normalize_record(_make_record(common_name="secret.example.com"), NOW)
        text = _format_cert(cert)
        assert "secret.example.com" in text


# ── Tool metadata ────────────────────────────────────────────


class TestToolMetadata:
    def test_name(self):
        tool = CertTransparencyTool()
        assert tool.name == "cert_transparency"

    def test_description_nonempty(self):
        tool = CertTransparencyTool()
        assert len(tool.description) > 20

    def test_parameters_has_domain(self):
        tool = CertTransparencyTool()
        assert "domain" in tool.parameters["properties"]

    def test_parameters_has_mode(self):
        tool = CertTransparencyTool()
        assert "mode" in tool.parameters["properties"]

    def test_domain_is_required(self):
        tool = CertTransparencyTool()
        assert "domain" in tool.parameters["required"]


# ── Input validation ─────────────────────────────────────────


class TestInputValidation:
    def test_missing_domain(self):
        tool = CertTransparencyTool()
        result = tool.execute(domain="")
        assert not result.success
        assert "domain" in result.output.lower()

    def test_whitespace_only_domain(self):
        tool = CertTransparencyTool()
        result = tool.execute(domain="   ")
        assert not result.success

    def test_invalid_mode(self):
        tool = CertTransparencyTool()
        result = tool.execute(mode="invalid", domain="example.com")
        assert not result.success
        assert "Invalid mode" in result.output

    def test_modes_match_constant(self):
        assert {"search", "subdomains", "recent"} == VALID_MODES

    def test_days_back_clamped_high(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh", return_value=([], None)):
            result = tool.execute(mode="recent", domain="x.com", days_back=9999)
            assert result.success  # doesn't error

    def test_days_back_clamped_low(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh", return_value=([], None)):
            result = tool.execute(mode="recent", domain="x.com", days_back=-5)
            assert result.success

    def test_limit_clamped_high(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh", return_value=(MOCK_SEARCH_RESULTS, None)):
            result = tool.execute(domain="stripe.com", limit=10000)
            assert result.success

    def test_limit_clamped_low(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh", return_value=(MOCK_SEARCH_RESULTS, None)):
            result = tool.execute(domain="stripe.com", limit=0)
            assert result.success
            # limit clamped to 1, so max 1 result
            assert len(result.data["certs"]) <= 1

    def test_extra_kwargs_ignored(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh", return_value=([], None)):
            result = tool.execute(domain="x.com", unknown_param="hi")
            assert result.success


# ── Search mode ──────────────────────────────────────────────


class TestSearchMode:
    def test_search_returns_certs(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh", return_value=(MOCK_SEARCH_RESULTS, None)):
            result = tool.execute(mode="search", domain="api.stripe.com")
        assert result.success
        assert result.data["count"] == 3
        assert result.data["active"] >= 1
        assert result.data["expired"] >= 1

    def test_search_empty(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh", return_value=([], None)):
            result = tool.execute(mode="search", domain="nonexistent.example.com")
        assert result.success
        assert result.data["count"] == 0
        assert "no certificates" in result.output.lower()

    def test_search_sorted_by_entry_timestamp_desc(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh", return_value=(MOCK_SEARCH_RESULTS, None)):
            result = tool.execute(mode="search", domain="api.stripe.com")
        certs = result.data["certs"]
        # Should be most recent first
        for i in range(len(certs) - 1):
            assert certs[i]["entry_timestamp"] >= certs[i + 1]["entry_timestamp"]

    def test_search_limit(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh", return_value=(MOCK_SEARCH_RESULTS, None)):
            result = tool.execute(mode="search", domain="stripe.com", limit=2)
        assert result.data["count"] == 2

    def test_search_exclude_expired_passed(self):
        """Verify exclude_expired is forwarded to _fetch_crtsh."""
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh", return_value=([], None)) as mock_fetch:
            tool.execute(mode="search", domain="x.com", exclude_expired=True)
        mock_fetch.assert_called_once_with(query="x.com", exclude_expired=True)

    def test_search_output_mentions_domain(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh", return_value=(MOCK_SEARCH_RESULTS, None)):
            result = tool.execute(mode="search", domain="api.stripe.com")
        assert "api.stripe.com" in result.output

    def test_search_fetch_error(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh", return_value=([], "crt.sh timed out")):
            result = tool.execute(mode="search", domain="x.com")
        assert not result.success
        assert "timed out" in result.output

    def test_default_mode_is_search(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh", return_value=(MOCK_SEARCH_RESULTS, None)):
            result = tool.execute(domain="api.stripe.com")
        assert result.success
        assert result.data["count"] == 3


# ── Subdomains mode ──────────────────────────────────────────


class TestSubdomainsMode:
    def test_subdomains_discovered(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh") as mock:
            mock.side_effect = [
                (MOCK_SUBDOMAIN_RESULTS, None),  # wildcard query
                (MOCK_BASE_RESULTS, None),  # base domain
            ]
            result = tool.execute(mode="subdomains", domain="stripe.com")
        assert result.success
        names = [s["subdomain"] for s in result.data["subdomains"]]
        assert "api.stripe.com" in names
        assert "dashboard.stripe.com" in names
        assert "stripe.com" in names

    def test_subdomains_count_aggregation(self):
        """api.stripe.com appears twice, should have cert_count=2."""
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh") as mock:
            mock.side_effect = [
                (MOCK_SUBDOMAIN_RESULTS, None),
                (MOCK_BASE_RESULTS, None),
            ]
            result = tool.execute(mode="subdomains", domain="stripe.com")
        api_sub = next(s for s in result.data["subdomains"] if s["subdomain"] == "api.stripe.com")
        assert api_sub["cert_count"] == 2

    def test_subdomains_wildcard_vs_concrete(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh") as mock:
            mock.side_effect = [
                (MOCK_SUBDOMAIN_RESULTS, None),
                (MOCK_BASE_RESULTS, None),
            ]
            result = tool.execute(mode="subdomains", domain="stripe.com")
        assert len(result.data["wildcards"]) >= 1
        assert len(result.data["concrete"]) >= 3

    def test_subdomains_empty(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh") as mock:
            mock.side_effect = [([], None), ([], None)]
            result = tool.execute(mode="subdomains", domain="nonexistent.example.com")
        assert result.success
        assert result.data["count"] == 0
        assert "no subdomains" in result.output.lower()

    def test_subdomains_wildcard_query(self):
        """Verify the wildcard query %.domain is used."""
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh") as mock:
            mock.side_effect = [([], None), ([], None)]
            tool.execute(mode="subdomains", domain="example.com")
        calls = mock.call_args_list
        assert calls[0][1]["query"] == "%.example.com"
        assert calls[1][1]["query"] == "example.com"

    def test_subdomains_limit(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh") as mock:
            mock.side_effect = [
                (MOCK_SUBDOMAIN_RESULTS, None),
                (MOCK_BASE_RESULTS, None),
            ]
            result = tool.execute(mode="subdomains", domain="stripe.com", limit=2)
        assert result.data["count"] <= 2 + len(result.data.get("wildcards", []))
        # Total subdomains capped at limit
        assert len(result.data["subdomains"]) <= 2

    def test_subdomains_sorted_by_count(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh") as mock:
            mock.side_effect = [
                (MOCK_SUBDOMAIN_RESULTS, None),
                (MOCK_BASE_RESULTS, None),
            ]
            result = tool.execute(mode="subdomains", domain="stripe.com")
        subs = result.data["subdomains"]
        for i in range(len(subs) - 1):
            assert subs[i]["cert_count"] >= subs[i + 1]["cert_count"]

    def test_subdomains_wildcard_fetch_error(self):
        """If wildcard fetch fails, returns error."""
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh", return_value=([], "crt.sh 503")):
            result = tool.execute(mode="subdomains", domain="x.com")
        assert not result.success

    def test_subdomains_base_fetch_error_partial(self):
        """If base fetch fails but wildcard succeeds, still returns results."""
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh") as mock:
            mock.side_effect = [
                (MOCK_SUBDOMAIN_RESULTS, None),  # wildcard ok
                ([], "timeout"),  # base fails — returns empty, not error tuple
            ]
            result = tool.execute(mode="subdomains", domain="stripe.com")
        assert result.success
        assert result.data["count"] > 0


# ── Recent mode ──────────────────────────────────────────────


class TestRecentMode:
    def test_recent_filters_by_date(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh") as mock:
            mock.side_effect = [
                (MOCK_RECENT_WILDCARD, None),
                (MOCK_RECENT_BASE, None),
            ]
            with patch("agent.tools.cert_transparency.datetime") as mock_dt:
                mock_dt.now.return_value = NOW
                mock_dt.strptime = datetime.strptime
                mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
                result = tool.execute(mode="recent", domain="stripe.com", days_back=30)
        assert result.success
        # old.stripe.com (LAST_YEAR) should be filtered out
        names = [c["common_name"] for c in result.data["certs"]]
        assert "old.stripe.com" not in names

    def test_recent_deduplicates(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh") as mock:
            mock.side_effect = [
                (MOCK_RECENT_WILDCARD, None),
                (MOCK_RECENT_BASE, None),
            ]
            with patch("agent.tools.cert_transparency.datetime") as mock_dt:
                mock_dt.now.return_value = NOW
                mock_dt.strptime = datetime.strptime
                mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
                result = tool.execute(mode="recent", domain="stripe.com", days_back=30)
        # Serial r2 appears twice; should be deduped
        serials = [c["serial_number"] for c in result.data["certs"]]
        assert serials.count("r2") <= 1

    def test_recent_empty(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh") as mock:
            mock.side_effect = [([], None), ([], None)]
            result = tool.execute(mode="recent", domain="x.com")
        assert result.success
        assert result.data["count"] == 0
        assert "no certificates" in result.output.lower()

    def test_recent_sorted_desc(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh") as mock:
            mock.side_effect = [
                (MOCK_RECENT_WILDCARD, None),
                (MOCK_RECENT_BASE, None),
            ]
            with patch("agent.tools.cert_transparency.datetime") as mock_dt:
                mock_dt.now.return_value = NOW
                mock_dt.strptime = datetime.strptime
                mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
                result = tool.execute(mode="recent", domain="stripe.com", days_back=30)
        certs = result.data["certs"]
        for i in range(len(certs) - 1):
            assert certs[i]["entry_timestamp"] >= certs[i + 1]["entry_timestamp"]

    def test_recent_wildcard_query_used(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh") as mock:
            mock.side_effect = [([], None), ([], None)]
            tool.execute(mode="recent", domain="example.com", days_back=7)
        calls = mock.call_args_list
        assert calls[0][1]["query"] == "%.example.com"
        assert calls[1][1]["query"] == "example.com"

    def test_recent_days_back_in_data(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh") as mock:
            mock.side_effect = [([], None), ([], None)]
            result = tool.execute(mode="recent", domain="x.com", days_back=14)
        assert result.data["days_back"] == 14

    def test_recent_unique_names_count(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh") as mock:
            mock.side_effect = [
                (MOCK_RECENT_WILDCARD, None),
                (MOCK_RECENT_BASE, None),
            ]
            with patch("agent.tools.cert_transparency.datetime") as mock_dt:
                mock_dt.now.return_value = NOW
                mock_dt.strptime = datetime.strptime
                mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
                result = tool.execute(mode="recent", domain="stripe.com", days_back=30)
        if result.data["count"] > 0:
            assert result.data["unique_names"] > 0

    def test_recent_fetch_error(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh", return_value=([], "connection failed")):
            result = tool.execute(mode="recent", domain="x.com")
        assert not result.success


# ── _fetch_crtsh ─────────────────────────────────────────────


class TestFetchCrtsh:
    def _make_mock_response(
        self,
        *,
        json_data: Any = None,
        status_code: int = 200,
        text: str = "[]",
    ) -> MagicMock:
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = json_data if json_data is not None else []
        resp.text = text
        resp.raise_for_status = MagicMock()
        if status_code >= 400:
            resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                f"HTTP {status_code}",
                request=MagicMock(),
                response=resp,
            )
        return resp

    @patch("agent.tools.cert_transparency.httpx.get")
    def test_success(self, mock_get):
        mock_get.return_value = self._make_mock_response(json_data=[_make_record()])
        tool = CertTransparencyTool()
        records, error = tool._fetch_crtsh(query="x.com")
        assert error is None
        assert len(records) == 1

    @patch("agent.tools.cert_transparency.httpx.get")
    def test_timeout(self, mock_get):
        mock_get.side_effect = httpx.TimeoutException("timed out")
        tool = CertTransparencyTool()
        records, error = tool._fetch_crtsh(query="x.com")
        assert records == []
        assert "timed out" in error.lower()

    @patch("agent.tools.cert_transparency.httpx.get")
    def test_503(self, mock_get):
        mock_get.return_value = self._make_mock_response(status_code=503)
        tool = CertTransparencyTool()
        records, error = tool._fetch_crtsh(query="x.com")
        assert records == []
        assert "503" in error

    @patch("agent.tools.cert_transparency.httpx.get")
    def test_other_http_error(self, mock_get):
        mock_get.return_value = self._make_mock_response(status_code=404)
        tool = CertTransparencyTool()
        records, error = tool._fetch_crtsh(query="x.com")
        assert records == []
        assert "404" in error

    @patch("agent.tools.cert_transparency.httpx.get")
    def test_connection_error(self, mock_get):
        mock_get.side_effect = httpx.ConnectError("refused")
        tool = CertTransparencyTool()
        records, error = tool._fetch_crtsh(query="x.com")
        assert records == []
        assert "connection failed" in error.lower()

    @patch("agent.tools.cert_transparency.httpx.get")
    def test_invalid_json(self, mock_get):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.side_effect = ValueError("bad json")
        mock_get.return_value = resp
        tool = CertTransparencyTool()
        records, error = tool._fetch_crtsh(query="x.com")
        assert records == []
        assert "invalid JSON" in error

    @patch("agent.tools.cert_transparency.httpx.get")
    def test_non_list_response(self, mock_get):
        mock_get.return_value = self._make_mock_response(json_data={"error": "bad request"})
        tool = CertTransparencyTool()
        records, error = tool._fetch_crtsh(query="x.com")
        assert records == []
        assert "unexpected" in error.lower()

    @patch("agent.tools.cert_transparency.httpx.get")
    def test_deduplicate_param_always_set(self, mock_get):
        mock_get.return_value = self._make_mock_response(json_data=[])
        tool = CertTransparencyTool()
        tool._fetch_crtsh(query="x.com")
        call_kwargs = mock_get.call_args
        params = call_kwargs[1]["params"]
        assert params["deduplicate"] == "Y"

    @patch("agent.tools.cert_transparency.httpx.get")
    def test_exclude_expired_param(self, mock_get):
        mock_get.return_value = self._make_mock_response(json_data=[])
        tool = CertTransparencyTool()
        tool._fetch_crtsh(query="x.com", exclude_expired=True)
        params = mock_get.call_args[1]["params"]
        assert params["exclude"] == "expired"

    @patch("agent.tools.cert_transparency.httpx.get")
    def test_no_exclude_expired_by_default(self, mock_get):
        mock_get.return_value = self._make_mock_response(json_data=[])
        tool = CertTransparencyTool()
        tool._fetch_crtsh(query="x.com")
        params = mock_get.call_args[1]["params"]
        assert "exclude" not in params


# ── Cache interaction ────────────────────────────────────────


class TestCacheInteraction:
    def test_cache_hit(self):
        cache = MagicMock()
        cache.get.return_value = [_make_record()]
        tool = CertTransparencyTool(cache=cache)
        records, error = tool._fetch_crtsh(query="cached.com")
        assert error is None
        assert len(records) == 1
        cache.get.assert_called_once()

    @patch("agent.tools.cert_transparency.httpx.get")
    def test_cache_miss_then_put(self, mock_get):
        cache = MagicMock()
        cache.get.return_value = None
        mock_get.return_value = MagicMock()
        mock_get.return_value.raise_for_status = MagicMock()
        mock_get.return_value.json.return_value = [_make_record()]
        tool = CertTransparencyTool(cache=cache)
        records, error = tool._fetch_crtsh(query="x.com")
        assert error is None
        cache.put.assert_called_once()
        put_args = cache.put.call_args
        assert put_args[0][0] == "cert_transparency"
        assert put_args[1]["ttl"] == _CACHE_TTL

    def test_no_cache_still_works(self):
        tool = CertTransparencyTool(cache=None)
        with patch("agent.tools.cert_transparency.httpx.get") as mock_get:
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = []
            mock_get.return_value = resp
            records, error = tool._fetch_crtsh(query="x.com")
        assert error is None


# ── Output formatting ────────────────────────────────────────


class TestOutputFormatting:
    def test_search_output_has_active_expired_count(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh", return_value=(MOCK_SEARCH_RESULTS, None)):
            result = tool.execute(domain="api.stripe.com")
        assert "active" in result.output.lower()
        assert "expired" in result.output.lower()

    def test_subdomains_output_has_concrete_wildcard_count(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh") as mock:
            mock.side_effect = [
                (MOCK_SUBDOMAIN_RESULTS, None),
                (MOCK_BASE_RESULTS, None),
            ]
            result = tool.execute(mode="subdomains", domain="stripe.com")
        assert "concrete" in result.output.lower()
        assert "wildcard" in result.output.lower()

    def test_recent_output_has_days(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh") as mock:
            mock.side_effect = [([], None), ([], None)]
            result = tool.execute(mode="recent", domain="x.com", days_back=14)
        assert "14d" in result.output


# ── Integration: tool registration counts ────────────────────


class TestIntegration:
    def _build_registry(self):
        try:
            from agent.cli import build_tool_registry
        except (ImportError, ModuleNotFoundError):
            pytest.skip("optional dep not installed")
        from unittest.mock import MagicMock as MM

        mock_config = MM()
        mock_config.tool_timeout = 30
        mock_config.fred_api_key = ""
        return build_tool_registry(mock_config)

    def test_tool_count(self):
        registry = self._build_registry()
        assert len(registry._tools) == 60, (
            f"Expected 60 tools, got {len(registry._tools)}: {sorted(registry._tools.keys())}"
        )

    def test_bandit_arm_count(self):
        from agent.learning.bandit import DEFAULT_ARMS

        assert len(DEFAULT_ARMS) == 48, f"Expected 48 arms, got {len(DEFAULT_ARMS)}: {[a.name for a in DEFAULT_ARMS]}"

    def test_cert_transparency_in_registry(self):
        registry = self._build_registry()
        assert "cert_transparency" in registry._tools

    def test_infrastructure_recon_arm(self):
        from agent.learning.bandit import DEFAULT_ARMS

        names = [a.name for a in DEFAULT_ARMS]
        assert "infrastructure_recon" in names
        arm = next(a for a in DEFAULT_ARMS if a.name == "infrastructure_recon")
        assert "cert_transparency" in arm.tools
