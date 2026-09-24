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

import json
import time
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.tools.cert_transparency import (
    _DEFAULT_TIME_BUDGET,
    _MIN_ATTEMPT_SECONDS,
    _RETRY_BACKOFF,
    _TIMEOUT,
    VALID_MODES,
    CertTransparencyTool,
    _format_cert,
    _normalize_record,
    _parse_timestamp,
    _shorten_issuer,
)

# ── Timestamps ───────────────────────────────────────────────

# Was a hardcoded calendar date (2026-03-28). Tests that inject NOW
# explicitly (_normalize_record(rec, NOW), mock_dt.now.return_value = NOW)
# are unaffected by the absolute value. But test_search_returns_certs calls
# tool.execute(...) uninjected, which uses real datetime.now(UTC)
# (agent/tools/cert_transparency.py) -- so a frozen NOW meant FUTURE_90D/
# FUTURE_10D silently drifted into the past as real time passed, and by
# 2026-08-27 every mock cert (even the 'future' ones) was already expired,
# giving active=0. Fixed 2026-08-27: derive NOW from real wall-clock time.
NOW = datetime.now(UTC)
YESTERDAY = (NOW - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
LAST_WEEK = (NOW - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")
LAST_MONTH = (NOW - timedelta(days=31)).strftime("%Y-%m-%dT%H:%M:%S")
LAST_YEAR = (NOW - timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%S")
FUTURE_90D = (NOW + timedelta(days=90)).strftime("%Y-%m-%dT%H:%M:%S")
FUTURE_10D = (NOW + timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%S")
PAST_EXPIRED = (NOW - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%S")


# ── Mock Data ────────────────────────────────────────────────

#: A full issuer DN, as crt.sh serves it. Named because two tests assert on
#: it: the persisted row must carry the whole DN, not the shortened CN.
_ISSUER_DN = 'C=US, O="DigiCert, Inc.", CN=DigiCert SHA2 Extended Validation Server CA'


def _make_record(
    *,
    crt_id: int = 1,
    common_name: str = "api.stripe.com",
    name_value: str = "api.stripe.com",
    issuer_name: str = _ISSUER_DN,
    entry_timestamp: str | None = None,
    not_before: str = LAST_WEEK,
    not_after: str = FUTURE_90D,
    serial_number: str = "0a1b2c3d4e5f",
    result_count: int = 1,
) -> dict[str, Any]:
    """A crt.sh JSON record, in the shape crt.sh actually serves today.

    ``entry_timestamp`` now defaults to **absent**, not to YESTERDAY. The old
    default is what hid the outage that cost 26 days of time-gated CT data:
    crt.sh stopped sending the field entirely (verified live 2026-09-23 —
    present on 0 of 247 records for ``%.aqr.com``, and on 0 for every other
    query shape), ``recent`` mode filtered exclusively on it, so every record
    was discarded before persistence and the DAG node still reported green.
    Every test in this file passed the whole time, against a schema that no
    longer exists.

    The collector backfills the field from ``not_before`` at the fetch
    boundary, so anything driving the tool through ``_fetch_crtsh`` sees a
    usable timestamp. Pass ``entry_timestamp=`` explicitly only when a test is
    specifically about a record that still carries one.
    """
    rec: dict[str, Any] = {
        "id": crt_id,
        "issuer_ca_id": 237361,
        "issuer_name": issuer_name,
        "common_name": common_name,
        "name_value": name_value,
        "not_before": not_before,
        "not_after": not_after,
        "serial_number": serial_number,
        "result_count": result_count,
    }
    if entry_timestamp is not None:
        rec["entry_timestamp"] = entry_timestamp
    return rec


def _live_shape_record(**kwargs: Any) -> dict[str, Any]:
    """A record in the exact live crt.sh key set — no ``entry_timestamp``."""
    rec = _make_record(**kwargs)
    rec.pop("entry_timestamp", None)
    return rec


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

    def test_partial_date_is_read_as_that_date_at_midnight_utc(self):
        """A bare ISO date is unambiguous; dropping the record is not better.

        ``tests/test_l2_integration.py`` and ``tests/test_digital_infra_l2.py``
        both feed ``not_before="2025-01-01"``. Refusing it skipped the write
        and logged a warning — a row lost to a parser gap, which is the same
        failure as a row lost to a bad filter. Midnight UTC is deterministic
        and stable across re-fetches, so idempotency is untouched.
        """
        assert _parse_timestamp("2026-03-27") == datetime(2026, 3, 27, tzinfo=UTC)

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

    def test_the_raw_crtsh_issuer_field_survives_normalization(self):
        """``issuer_name`` must reach the payload under crt.sh's own name.

        The persist path reads ``issuer_name`` with a fallback to
        ``issuer_full``, so dropping this key does NOT break persistence — it
        survived a mutation run on 2026-09-23 for exactly that reason. What it
        does break is ``data["certs"]``, which is the tool's public output, so
        the assertion belongs here, on the normalizer, rather than in a
        persistence test that the fallback would keep green.
        """
        result = _normalize_record(_make_record(), NOW)
        assert result["issuer_name"] == _ISSUER_DN
        # The shortened CN is a separate field, not a replacement for it.
        assert result["issuer"] != result["issuer_name"]


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

    def test_search_exclude_expired_passed_with_budget(self):
        """exclude_expired is forwarded, and so is a real wall-clock deadline.

        The deadline is not decoration: without it a single slow domain can
        burn the whole ``fetch_cert_domains`` node budget, and a node timeout
        marks the executor pool degraded, skipping every later node.
        """
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh", return_value=([], None)) as mock_fetch:
            tool.execute(mode="search", domain="x.com", exclude_expired=True)
        mock_fetch.assert_called_once()
        kwargs = mock_fetch.call_args[1]
        assert kwargs["query"] == "x.com"
        assert kwargs["exclude_expired"] is True
        assert kwargs["deadline"] is not None
        assert kwargs["deadline"] - time.monotonic() <= _DEFAULT_TIME_BUDGET

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
        """The single wildcard query carries the apex name too.

        Measured 2026-09-23: crt.sh answers ``q=aqr.com`` and ``q=%.aqr.com``
        with byte-identical payloads (247 records each, 0 ids unique to
        either), so the apex arrives in the one response the tool now makes.
        """
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh") as mock:
            mock.side_effect = [(MOCK_SUBDOMAIN_RESULTS + MOCK_BASE_RESULTS, None)]
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
            mock.side_effect = [(MOCK_SUBDOMAIN_RESULTS + MOCK_BASE_RESULTS, None)]
            result = tool.execute(mode="subdomains", domain="stripe.com")
        api_sub = next(s for s in result.data["subdomains"] if s["subdomain"] == "api.stripe.com")
        assert api_sub["cert_count"] == 2

    def test_subdomains_wildcard_vs_concrete(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh") as mock:
            mock.side_effect = [(MOCK_SUBDOMAIN_RESULTS + MOCK_BASE_RESULTS, None)]
            result = tool.execute(mode="subdomains", domain="stripe.com")
        assert len(result.data["wildcards"]) >= 1
        assert len(result.data["concrete"]) >= 3

    def test_subdomains_empty(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh") as mock:
            mock.side_effect = [([], None)]
            result = tool.execute(mode="subdomains", domain="nonexistent.example.com")
        assert result.success
        assert result.data["count"] == 0
        assert "no subdomains" in result.output.lower()

    def test_subdomains_issues_exactly_one_query(self):
        """One request per domain — the base-domain fetch was pure duplication.

        Two queries per domain doubled this node's wall time and its
        rate-limit pressure against a service that answered 502 on 4 of 8
        probe requests, for a payload measured byte-identical to the one the
        wildcard query already returns.
        """
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh") as mock:
            mock.side_effect = [(MOCK_SUBDOMAIN_RESULTS, None)]
            tool.execute(mode="subdomains", domain="example.com")
        assert mock.call_count == 1
        assert mock.call_args_list[0][1]["query"] == "%.example.com"

    def test_subdomains_limit(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh") as mock:
            mock.side_effect = [(MOCK_SUBDOMAIN_RESULTS + MOCK_BASE_RESULTS, None)]
            result = tool.execute(mode="subdomains", domain="stripe.com", limit=2)
        assert result.data["count"] <= 2 + len(result.data.get("wildcards", []))
        # Total subdomains capped at limit
        assert len(result.data["subdomains"]) <= 2

    def test_subdomains_sorted_by_count(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh") as mock:
            mock.side_effect = [(MOCK_SUBDOMAIN_RESULTS + MOCK_BASE_RESULTS, None)]
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

    def test_subdomains_never_makes_a_second_request(self):
        """The wildcard response is the whole answer; nothing follows it."""
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh") as mock:
            mock.side_effect = [(MOCK_SUBDOMAIN_RESULTS, None)]
            result = tool.execute(mode="subdomains", domain="stripe.com")
        assert result.success
        assert result.data["count"] > 0
        assert mock.call_count == 1


# ── Recent mode ──────────────────────────────────────────────


class TestRecentMode:
    def test_recent_filters_by_date(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh") as mock:
            mock.side_effect = [(MOCK_RECENT_WILDCARD + MOCK_RECENT_BASE, None)]
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
            mock.side_effect = [(MOCK_RECENT_WILDCARD + MOCK_RECENT_BASE, None)]
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
            mock.side_effect = [([], None)]
            result = tool.execute(mode="recent", domain="x.com")
        assert result.success
        assert result.data["count"] == 0
        assert "no certificates" in result.output.lower()

    def test_recent_sorted_desc(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh") as mock:
            mock.side_effect = [(MOCK_RECENT_WILDCARD + MOCK_RECENT_BASE, None)]
            with patch("agent.tools.cert_transparency.datetime") as mock_dt:
                mock_dt.now.return_value = NOW
                mock_dt.strptime = datetime.strptime
                mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
                result = tool.execute(mode="recent", domain="stripe.com", days_back=30)
        certs = result.data["certs"]
        for i in range(len(certs) - 1):
            assert certs[i]["entry_timestamp"] >= certs[i + 1]["entry_timestamp"]

    def test_recent_issues_exactly_one_wildcard_query(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh") as mock:
            mock.side_effect = [([], None)]
            tool.execute(mode="recent", domain="example.com", days_back=7)
        assert mock.call_count == 1
        assert mock.call_args_list[0][1]["query"] == "%.example.com"

    def test_recent_days_back_in_data(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh") as mock:
            mock.side_effect = [([], None)]
            result = tool.execute(mode="recent", domain="x.com", days_back=14)
        assert result.data["days_back"] == 14

    def test_recent_unique_names_count(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh") as mock:
            mock.side_effect = [(MOCK_RECENT_WILDCARD + MOCK_RECENT_BASE, None)]
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


@patch("agent.tools.cert_transparency.time.sleep")  # the ladder is asserted in TestRetryLadder
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
    def test_success(self, mock_get, _sleep):
        mock_get.return_value = self._make_mock_response(json_data=[_make_record()])
        tool = CertTransparencyTool()
        records, error = tool._fetch_crtsh(query="x.com")
        assert error is None
        assert len(records) == 1

    @patch("agent.tools.cert_transparency.httpx.get")
    def test_timeout(self, mock_get, _sleep):
        mock_get.side_effect = httpx.TimeoutException("timed out")
        tool = CertTransparencyTool()
        records, error = tool._fetch_crtsh(query="x.com")
        assert records == []
        assert "timed out" in error.lower()

    @patch("agent.tools.cert_transparency.httpx.get")
    def test_503(self, mock_get, _sleep):
        mock_get.return_value = self._make_mock_response(status_code=503)
        tool = CertTransparencyTool()
        records, error = tool._fetch_crtsh(query="x.com")
        assert records == []
        assert "503" in error

    @patch("agent.tools.cert_transparency.httpx.get")
    def test_other_http_error(self, mock_get, _sleep):
        mock_get.return_value = self._make_mock_response(status_code=404)
        tool = CertTransparencyTool()
        records, error = tool._fetch_crtsh(query="x.com")
        assert records == []
        assert "404" in error

    @patch("agent.tools.cert_transparency.httpx.get")
    def test_connection_error(self, mock_get, _sleep):
        mock_get.side_effect = httpx.ConnectError("refused")
        tool = CertTransparencyTool()
        records, error = tool._fetch_crtsh(query="x.com")
        assert records == []
        assert "connection failed" in error.lower()

    @patch("agent.tools.cert_transparency.httpx.get")
    def test_invalid_json(self, mock_get, _sleep):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.side_effect = ValueError("bad json")
        mock_get.return_value = resp
        tool = CertTransparencyTool()
        records, error = tool._fetch_crtsh(query="x.com")
        assert records == []
        assert "invalid JSON" in error

    @patch("agent.tools.cert_transparency.httpx.get")
    def test_non_list_response(self, mock_get, _sleep):
        mock_get.return_value = self._make_mock_response(json_data={"error": "bad request"})
        tool = CertTransparencyTool()
        records, error = tool._fetch_crtsh(query="x.com")
        assert records == []
        assert "unexpected" in error.lower()

    @patch("agent.tools.cert_transparency.httpx.get")
    def test_deduplicate_param_always_set(self, mock_get, _sleep):
        mock_get.return_value = self._make_mock_response(json_data=[])
        tool = CertTransparencyTool()
        tool._fetch_crtsh(query="x.com")
        call_kwargs = mock_get.call_args
        params = call_kwargs[1]["params"]
        assert params["deduplicate"] == "Y"

    @patch("agent.tools.cert_transparency.httpx.get")
    def test_exclude_expired_param(self, mock_get, _sleep):
        mock_get.return_value = self._make_mock_response(json_data=[])
        tool = CertTransparencyTool()
        tool._fetch_crtsh(query="x.com", exclude_expired=True)
        params = mock_get.call_args[1]["params"]
        assert params["exclude"] == "expired"

    @patch("agent.tools.cert_transparency.httpx.get")
    def test_no_exclude_expired_by_default(self, mock_get, _sleep):
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
            mock.side_effect = [(MOCK_SUBDOMAIN_RESULTS + MOCK_BASE_RESULTS, None)]
            result = tool.execute(mode="subdomains", domain="stripe.com")
        assert "concrete" in result.output.lower()
        assert "wildcard" in result.output.lower()

    def test_recent_output_has_days(self):
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh") as mock:
            mock.side_effect = [([], None)]
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
        assert len(registry._tools) == 61, (
            f"Expected 61 tools, got {len(registry._tools)}: {sorted(registry._tools.keys())}"
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


# ── Regression: crt.sh dropped entry_timestamp (2026-09-23) ──
#
# crt.sh's JSON API stopped serving ``entry_timestamp``. ``recent`` mode
# filtered exclusively on it, so every record was discarded before
# ``_persist_entities`` was ever reached — the early ``if not deduped`` return
# sits above the persist call. The tool returned success=True/count=0 and the
# DAG node recorded ``status=completed, error=null`` (live run 55233b8478c4).
# ``entity_observations`` for this source stayed frozen at 19 rows, max
# ingested_at 2026-08-27, for 26 days.
#
# That window IS recoverable — the earlier claim here, that it "cannot be
# refetched at any price", was wrong, and wrong in the expensive direction: it
# would stop a future session attempting a recovery that costs one request per
# domain. crt.sh's query carries no date parameter, so the full history arrives
# on every call and ``days_back`` filters it client-side on ``not_before``, a
# fixed property of each certificate. Measured on one live 247-record response
# for ``%.aqr.com`` (2026-09-23): 0 records inside 7d, 2 inside 30d, 11 inside
# 60d, 45 inside 365d, newest 2026-09-03. A catch-up run is ``days_back=365``
# and nothing else.
#
# These assert on ROWS IN A DATABASE, not on ``result.success`` — the whole
# failure mode is that success was already True.


class TestCrtShWithoutEntryTimestamp:
    """The live crt.sh schema: no ``entry_timestamp``, only ``not_before``."""

    @staticmethod
    def _mock_response(records: list[dict[str, Any]]) -> MagicMock:
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = records
        return resp

    def test_recent_mode_persists_rows_when_entry_timestamp_absent(self, tmp_path):
        """A payload in the live schema must still write rows to the store.

        Fails against the unfixed collector: with ``entry_timestamp`` missing,
        ``_execute_recent``'s cutoff filter drops all four records, ``deduped``
        is empty, the early return fires before ``_persist_entities``, and the
        table stays at zero rows while the tool reports success.
        """
        import sqlite3

        from agent.pipeline.store import PipelineStore

        records = [
            _live_shape_record(crt_id=101, common_name="new.aqr.com", serial_number="c1"),
            _live_shape_record(crt_id=102, common_name="api.aqr.com", serial_number="c2"),
            # Same common_name AND same issuance second as c2, different cert.
            # Before serial_number entered the persisted value these two
            # collapsed under the store's unique key and one was suppressed.
            _live_shape_record(crt_id=103, common_name="api.aqr.com", serial_number="c3"),
            # Outside the 30d window — must still be excluded.
            _live_shape_record(
                crt_id=104,
                common_name="ancient.aqr.com",
                serial_number="c4",
                not_before=LAST_YEAR,
            ),
        ]
        assert all("entry_timestamp" not in r for r in records), "fixture must mirror the live schema"

        db_path = tmp_path / "probe_cert.db"
        store = PipelineStore(str(db_path))
        try:
            tool = CertTransparencyTool(pipeline_store=store)
            with patch("agent.tools.cert_transparency.httpx.get") as mock_get:
                mock_get.return_value = self._mock_response(records)
                result = tool.execute(mode="recent", domain="aqr.com", days_back=30)
        finally:
            store.close()

        # The out-of-window record is still filtered; the three recent ones are not.
        assert result.data["count"] == 3, result.output
        assert result.data["rows_persisted"] == 3

        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT observed_at, value_json FROM entity_observations "
                "WHERE source_tool='cert_transparency' AND observation_type='cert_issued'"
            ).fetchall()
        finally:
            conn.close()

        # The assertion that matters: real rows in a real table.
        assert len(rows) == 3, f"expected 3 persisted observations, got {len(rows)}"

        values = [json.loads(r["value_json"]) for r in rows]
        # observed_at is the certificate's issuance instant, NOT ingestion time.
        # An ingestion-time stamp would defeat the store's idempotency key and
        # append a fresh copy of every cert on every daily run.
        expected = _parse_timestamp(LAST_WEEK)
        assert expected is not None
        assert {r["observed_at"] for r in rows} == {expected.timestamp()}

        # Distinct certs sharing a common_name and a second no longer collide.
        assert {v["serial_number"] for v in values} == {"c1", "c2", "c3"}
        # The issuer-switch signal: every live row stored "" before this fix.
        # Assert the FULL DN, not truthiness. Storing the shortened CN here
        # would still be truthy while dropping the O= and C= that make an
        # issuer switch legible, and `all(...)` cannot tell those apart —
        # mutation-tested 2026-09-23: renaming the normalizer's "issuer_name"
        # key left the truthiness form green, because the persist path falls
        # back to "issuer_full".
        assert {v["issuer_name"] for v in values} == {_ISSUER_DN}, values

    def test_node_raises_instead_of_reporting_green_on_zero_certs(self, tmp_path):
        """The DAG node must go red, not return a payload, when nothing lands.

        Fails against the unfixed ``run_cert_domain_collection``, which
        returned ``{"domains_scanned": 20, "total_certs": 0, ...}`` — a green
        checkmark indistinguishable from a quiet day.
        """
        from agent.pipeline.dags.daily_collection import run_cert_domain_collection

        with patch("agent.tools.cert_transparency.httpx.get") as mock_get:
            mock_get.return_value = self._mock_response([])
            with pytest.raises(RuntimeError) as exc:
                run_cert_domain_collection(
                    {
                        "db_path": str(tmp_path / "probe_empty.db"),
                        "domains": ["aqr.com"],
                        "days_back": 30,
                        "request_delay": 0,
                    },
                    {},
                )
        # Assert the FACTS the message must carry — that it raised, and that
        # it names the zero. Deliberately NOT asserting the "time-gated /
        # unrecoverable" wording: crt.sh's query takes no date parameter, so
        # the window IS recoverable by widening days_back, and that sentence in
        # daily_collection.py is wrong and owned by the DAG. Pinning it here
        # would make this file veto the correction.
        assert "wrote nothing" in str(exc.value)
        assert "0 rows written" in str(exc.value)


# ── Regression: the retry ladder, the time budget, the write path ──
#
# Mutation-tested 2026-09-23: before these classes existed, reverting
# _TIMEOUT, emptying _RETRY_STATUSES, collapsing _RETRY_BACKOFF to one
# attempt, dropping the cache-hit backfill and discarding base_error all left
# the file at "84 passed". Every assertion below fails against at least one of
# those reversions, which is the only reason it is here.


def _http_response(*, status_code: int = 200, records: list[dict[str, Any]] | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = [] if records is None else records
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}",
            request=MagicMock(),
            response=resp,
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


@patch("agent.tools.cert_transparency.time.sleep")
class TestRetryLadder:
    """Asserts the ladder *retries* — not just that the error string is nice.

    The three pre-existing error tests (timeout/503/connection) assert only on
    the final message, which is identical with and without retries. They passed
    against a single-attempt build for the whole time the collector was writing
    zero rows.
    """

    @patch("agent.tools.cert_transparency.httpx.get")
    def test_504_then_200_recovers_the_payload(self, mock_get, _sleep):
        """A transient 504 followed by a 200 must return the 200's records."""
        mock_get.side_effect = [
            _http_response(status_code=504),
            _http_response(records=[_live_shape_record()]),
        ]
        tool = CertTransparencyTool()
        records, error = tool._fetch_crtsh(query="%.nasdaq.com")
        assert error is None
        assert len(records) == 1
        assert mock_get.call_count == 2

    @patch("agent.tools.cert_transparency.httpx.get")
    def test_503_uses_every_attempt(self, mock_get, _sleep):
        mock_get.side_effect = [_http_response(status_code=503) for _ in _RETRY_BACKOFF]
        tool = CertTransparencyTool()
        records, error = tool._fetch_crtsh(query="x.com")
        assert records == []
        assert mock_get.call_count == len(_RETRY_BACKOFF)
        assert f"{len(_RETRY_BACKOFF)} attempt(s)" in error

    @patch("agent.tools.cert_transparency.httpx.get")
    def test_timeout_uses_every_attempt(self, mock_get, _sleep):
        mock_get.side_effect = httpx.TimeoutException("timed out")
        tool = CertTransparencyTool()
        _records, error = tool._fetch_crtsh(query="x.com")
        assert mock_get.call_count == len(_RETRY_BACKOFF)
        assert "timed out" in error.lower()

    @patch("agent.tools.cert_transparency.httpx.get")
    def test_connection_error_uses_every_attempt(self, mock_get, _sleep):
        mock_get.side_effect = httpx.ConnectError("refused")
        tool = CertTransparencyTool()
        tool._fetch_crtsh(query="x.com")
        assert mock_get.call_count == len(_RETRY_BACKOFF)

    @patch("agent.tools.cert_transparency.httpx.get")
    def test_404_is_retried_because_crtsh_404s_are_transient(self, mock_get, _sleep):
        """crt.sh answers an unknown domain ``200 []`` — a 404 is its backend.

        Observed 2026-09-23 on the identical ``%.nasdaq.com`` URL: 404, then
        504, then 200 with 3,347 records. Treating 404 as permanent dropped
        half of that domain's fetch for the day after a single attempt.
        """
        mock_get.side_effect = [
            _http_response(status_code=404),
            _http_response(records=[_live_shape_record()]),
        ]
        tool = CertTransparencyTool()
        records, error = tool._fetch_crtsh(query="nasdaq.com")
        assert error is None
        assert len(records) == 1
        assert mock_get.call_count == 2

    @patch("agent.tools.cert_transparency.httpx.get")
    def test_400_is_not_retried(self, mock_get, _sleep):
        """A malformed query is ours to fix; retrying it only burns budget."""
        mock_get.return_value = _http_response(status_code=400)
        tool = CertTransparencyTool()
        records, error = tool._fetch_crtsh(query="x.com")
        assert records == []
        assert mock_get.call_count == 1
        assert "400" in error

    @patch("agent.tools.cert_transparency.httpx.get")
    def test_backoff_is_applied_between_attempts(self, mock_get, sleep):
        mock_get.side_effect = [_http_response(status_code=502) for _ in _RETRY_BACKOFF]
        tool = CertTransparencyTool()
        tool._fetch_crtsh(query="x.com")
        # One sleep between each pair of attempts — never after the last.
        assert [call.args[0] for call in sleep.call_args_list] == list(_RETRY_BACKOFF[:-1])


class TestTimeBudget:
    """A slow domain must cost this node its own budget, not the whole DAG.

    ``executor.py`` sets ``_degraded`` when a node times out and then skips
    every node in every later layer, so overrunning ``fetch_cert_domains``'
    1200s is not "a slow node", it is a dead DAG layer (LESSONS F-13).
    """

    def test_the_ladder_cannot_sleep_away_the_budget(self):
        """A 502 storm must leave room for the request that finally works.

        crt.sh answers a 502 in ~0.6s, so an attempt count of 3 gave a
        storming domain up after 8s of a 55s budget (measured live on
        jpmorgan.com and goldmansachs.com, 2026-09-23).
        """
        assert len(_RETRY_BACKOFF) >= 5
        assert sum(_RETRY_BACKOFF[:-1]) < _DEFAULT_TIME_BUDGET

    def test_default_budget_fits_the_dag_node(self):
        """20 domains x budget + 19 x 2.5s inter-domain gap must fit 1200s."""
        node_timeout = 1200
        domains = 20
        inter_domain_gap = 2.5
        assert domains * _DEFAULT_TIME_BUDGET + (domains - 1) * inter_domain_gap < node_timeout
        # A single HTTP attempt must never be able to outlive the whole call.
        assert _TIMEOUT <= _DEFAULT_TIME_BUDGET

    def test_an_attempt_is_allowed_to_outlast_a_slow_success(self):
        """The per-attempt ceiling must not cut off answers crt.sh does give.

        This is the half of the fix with the most direct effect on rows — too
        low and every large domain times out — and it was the one mutation
        that survived the 2026-09-23 mutation run: dropping ``_TIMEOUT`` from
        50 back to the original 30 left all 116 tests green.

        Measured live 2026-09-23 (second pass), successful 200s only:
        ``aqr.com`` 58.9s, ``%.bis.org`` 42.1s, ``cftc.gov`` 22.6s,
        ``%.aqr.com`` 10.0s. At 30s the middle two are lost to a timeout the
        server never caused. The floor is deliberately below the slowest
        observation: 58.9s cannot fit a 55s call budget at a 20-domain roster,
        and the collector reports that domain "not reached" rather than
        pretending it was a quiet day.
        """
        assert _TIMEOUT >= 40, "a 30s ceiling times out answers crt.sh actually served"

    @patch("agent.tools.cert_transparency.httpx.get")
    def test_the_first_attempt_gets_the_full_per_attempt_ceiling(self, mock_get):
        """Behavioural companion: the ceiling must reach httpx, not just exist.

        Drives the real default-budget path (no explicit ``time_budget``), so
        a revert of either ``_TIMEOUT`` or the clamp arithmetic shows up as a
        number handed to the HTTP client.
        """
        mock_get.return_value = _http_response(records=[])
        tool = CertTransparencyTool()
        tool.execute(mode="search", domain="x.com")
        assert mock_get.call_args[1]["timeout"] >= 40

    @patch("agent.tools.cert_transparency.httpx.get")
    def test_spent_budget_makes_no_request_at_all(self, mock_get):
        tool = CertTransparencyTool()
        records, error = tool._fetch_crtsh(query="x.com", deadline=time.monotonic() - 1)
        assert records == []
        assert mock_get.call_count == 0
        assert "budget exhausted" in error
        assert "not reached" in error

    @patch("agent.tools.cert_transparency.httpx.get")
    def test_attempt_timeout_is_clamped_to_the_remaining_budget(self, mock_get):
        mock_get.return_value = _http_response(records=[])
        tool = CertTransparencyTool()
        tool._fetch_crtsh(query="x.com", deadline=time.monotonic() + 10)
        assert mock_get.call_args[1]["timeout"] <= 10
        assert mock_get.call_args[1]["timeout"] < _TIMEOUT

    @patch("agent.tools.cert_transparency.time.sleep")
    @patch("agent.tools.cert_transparency.httpx.get")
    def test_a_sliver_of_budget_is_not_spent_on_a_doomed_request(self, mock_get, sleep):
        """crt.sh's fastest good answer that hour was 5.0s; a 1s try is noise."""
        mock_get.return_value = _http_response(status_code=503)
        tool = CertTransparencyTool()
        records, error = tool._fetch_crtsh(query="x.com", deadline=time.monotonic() + 1)
        assert records == []
        assert mock_get.call_count == 0
        assert sleep.call_count == 0
        assert "budget exhausted" in error

    @patch("agent.tools.cert_transparency.time.sleep")
    @patch("agent.tools.cert_transparency.httpx.get")
    def test_a_usable_sliver_is_still_tried_once(self, mock_get, sleep):
        mock_get.return_value = _http_response(records=[_live_shape_record()])
        tool = CertTransparencyTool()
        records, error = tool._fetch_crtsh(query="x.com", deadline=time.monotonic() + _MIN_ATTEMPT_SECONDS + 1)
        assert error is None
        assert len(records) == 1
        assert mock_get.call_count == 1

    @patch("agent.tools.cert_transparency.httpx.get")
    def test_zero_budget_means_unbounded(self, mock_get):
        """CLI/interactive callers can opt out; the DAG never does."""
        mock_get.return_value = _http_response(records=[])
        tool = CertTransparencyTool()
        tool.execute(mode="search", domain="x.com", time_budget=0)
        assert mock_get.call_args[1]["timeout"] == _TIMEOUT


class TestCacheBackfill:
    def test_cache_hit_gets_entry_timestamp_backfilled(self):
        """A cache entry stored before the fix still lacks entry_timestamp.

        Without the backfill on the cache-hit path, every run inside the 1h
        TTL went straight back to discarding 100% of records in recent mode.
        """
        cached = [_live_shape_record(not_before=LAST_WEEK)]
        assert "entry_timestamp" not in cached[0]
        cache = MagicMock()
        cache.get.return_value = cached
        tool = CertTransparencyTool(cache=cache)
        with patch("agent.tools.cert_transparency.httpx.get") as mock_get:
            records, error = tool._fetch_crtsh(query="cached.com")
        assert error is None
        assert mock_get.call_count == 0
        assert records[0]["entry_timestamp"] == LAST_WEEK


class TestOneQueryPerDomain:
    """``q=domain`` and ``q=%.domain`` are the same request; only one is made.

    Measured against live crt.sh 2026-09-23, three domains, both query shapes:

    ====================  ==========  ==========  ====================
    domain                wildcard n  base n      ids unique to either
    ====================  ==========  ==========  ====================
    aqr.com               247         247         0
    cftc.gov              398         398         0
    sec.gov               923         923         0
    ====================  ==========  ==========  ====================

    Byte counts matched too (78,228 / 156,529 / 558,095). The second request
    bought nothing and spent half of every domain's time budget against a
    service that returned 502 on 4 of the 8 probe requests.
    """

    @patch("agent.tools.cert_transparency.httpx.get")
    def test_recent_mode_makes_one_http_request(self, mock_get):
        mock_get.return_value = _http_response(records=[_live_shape_record()])
        tool = CertTransparencyTool()
        result = tool.execute(mode="recent", domain="stripe.com", days_back=30)
        assert result.success
        assert mock_get.call_count == 1
        assert mock_get.call_args[1]["params"]["q"] == "%.stripe.com"

    @patch("agent.tools.cert_transparency.httpx.get")
    def test_subdomains_mode_makes_one_http_request(self, mock_get):
        mock_get.return_value = _http_response(records=MOCK_SUBDOMAIN_RESULTS)
        tool = CertTransparencyTool()
        result = tool.execute(mode="subdomains", domain="stripe.com")
        assert result.success
        assert mock_get.call_count == 1

    def test_no_dead_base_error_plumbing_survives(self):
        """The payload must not carry a field for a request that is not made."""
        tool = CertTransparencyTool()
        with patch.object(tool, "_fetch_crtsh", return_value=(MOCK_RECENT_WILDCARD, None)):
            recent = tool.execute(mode="recent", domain="stripe.com", days_back=30)
        with patch.object(tool, "_fetch_crtsh", return_value=(MOCK_SUBDOMAIN_RESULTS, None)):
            subs = tool.execute(mode="subdomains", domain="stripe.com")
        assert "base_error" not in recent.data
        assert "base_error" not in subs.data


class TestFetchedVersusWindowed:
    def test_recent_reports_how_many_records_were_fetched(self):
        """ "Fetched 3,347, 0 inside the window" must not read as "no certs".

        Live: ``%.nasdaq.com`` serves thousands of records whose newest
        not_before is 2021 — a permanent per-domain zero that the DAG's
        aggregate guard cannot see. The number is the only way to tell it
        apart from a domain that genuinely issued nothing.
        """
        stale = [
            _live_shape_record(crt_id=n, serial_number=f"s{n}", not_before=LAST_YEAR, common_name=f"n{n}.nasdaq.com")
            for n in range(3)
        ]
        tool = CertTransparencyTool()
        with patch("agent.tools.cert_transparency.httpx.get") as mock_get:
            mock_get.return_value = _http_response(records=stale)
            result = tool.execute(mode="recent", domain="nasdaq.com", days_back=30)
        assert result.success
        assert result.data["count"] == 0
        assert result.data["records_fetched"] == 3
        assert "3 records fetched" in result.output
        # The ceiling is the tell: this identity's newest cert predates the
        # window, which is a permanently silent domain, not a quiet day.
        assert result.data["newest_not_before"].startswith(LAST_YEAR[:10])
        assert LAST_YEAR[:10] in result.output

    def test_recent_reports_the_ceiling_when_rows_do_land(self):
        tool = CertTransparencyTool()
        with patch("agent.tools.cert_transparency.httpx.get") as mock_get:
            mock_get.return_value = _http_response(
                records=[_live_shape_record(crt_id=9, serial_number="s9", not_before=LAST_WEEK)]
            )
            result = tool.execute(mode="recent", domain="aqr.com", days_back=30)
        assert result.success
        assert result.data["newest_not_before"].startswith(LAST_WEEK[:10])


class TestPersistenceIsolation:
    """One refused record must cost one record, not the domain's whole batch."""

    @staticmethod
    def _rows(db_path) -> list[dict[str, Any]]:
        import sqlite3

        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT value_json FROM entity_observations WHERE source_tool='cert_transparency'"
                ).fetchall()
            ]
        finally:
            conn.close()

    def test_a_rejected_cert_does_not_discard_the_rest(self, tmp_path):
        """A future-dated not_before is legal CA pre-issuance; crt.sh serves it.

        It sits inside recent mode's window and outside the store's
        ``[1990, now+1d]`` guard, and the newest-first sort puts it FIRST — so
        with one try/except around the loop it took the whole batch with it,
        and the tool still reported success=True, count=3, rows=0.
        """
        from agent.pipeline.store import PipelineStore

        records = [
            _live_shape_record(crt_id=201, common_name="future.aqr.com", serial_number="f1", not_before=FUTURE_10D),
            _live_shape_record(crt_id=202, common_name="a.aqr.com", serial_number="c1"),
            _live_shape_record(crt_id=203, common_name="b.aqr.com", serial_number="c2"),
        ]
        db_path = tmp_path / "isolation.db"
        store = PipelineStore(str(db_path))
        try:
            tool = CertTransparencyTool(pipeline_store=store)
            with patch("agent.tools.cert_transparency.httpx.get") as mock_get:
                mock_get.side_effect = [_http_response(records=records), _http_response(records=[])]
                result = tool.execute(mode="recent", domain="aqr.com", days_back=30)
        finally:
            store.close()

        assert result.success, result.output
        assert result.data["count"] == 3
        assert result.data["rows_persisted"] == 2
        assert result.data["certs_rejected"] == 1
        rows = self._rows(db_path)
        assert len(rows) == 2, rows
        assert {json.loads(r["value_json"])["serial_number"] for r in rows} == {"c1", "c2"}

    def test_a_batch_that_lands_nothing_is_a_failure_not_a_skip(self, tmp_path):
        """count>0 / rows=0 must not reach the DAG as success.

        It used to: the node then recorded ``status='skipped'`` with the reason
        "every observation was already stored — no new certificates since the
        last run", on a database that had never held a row.
        """
        from agent.pipeline.store import PipelineStore

        records = [
            _live_shape_record(crt_id=301, common_name="future.aqr.com", serial_number="f1", not_before=FUTURE_10D),
        ]
        db_path = tmp_path / "all_rejected.db"
        store = PipelineStore(str(db_path))
        try:
            tool = CertTransparencyTool(pipeline_store=store)
            with patch("agent.tools.cert_transparency.httpx.get") as mock_get:
                mock_get.side_effect = [_http_response(records=records), _http_response(records=[])]
                result = tool.execute(mode="recent", domain="aqr.com", days_back=30)
        finally:
            store.close()

        assert not result.success
        assert result.data["count"] == 1
        assert result.data["rows_persisted"] == 0
        assert result.data["certs_rejected"] == 1
        assert "0 rows reached the store" in result.output
        assert self._rows(db_path) == []

    def test_a_write_error_is_reported_not_swallowed(self):
        """A locked database (a concurrent backfill is enough) must go red."""
        import sqlite3

        store = MagicMock()
        store.store_entity_observation.side_effect = sqlite3.OperationalError("database is locked")
        tool = CertTransparencyTool(pipeline_store=store)
        with patch("agent.tools.cert_transparency.httpx.get") as mock_get:
            mock_get.side_effect = [
                _http_response(records=[_live_shape_record(crt_id=401, serial_number="w1")]),
                _http_response(records=[]),
            ]
            result = tool.execute(mode="recent", domain="aqr.com", days_back=30)

        assert not result.success
        assert result.data["rows_persisted"] == 0
        assert "database is locked" in result.data["persist_error"]

    def test_no_store_configured_is_not_a_failure(self):
        """Most callers (CLI, tests) have no store; that is not data loss."""
        tool = CertTransparencyTool()
        with patch("agent.tools.cert_transparency.httpx.get") as mock_get:
            mock_get.side_effect = [
                _http_response(records=[_live_shape_record(crt_id=501, serial_number="n1")]),
                _http_response(records=[]),
            ]
            result = tool.execute(mode="recent", domain="aqr.com", days_back=30)
        assert result.success
        assert result.data["count"] == 1
        assert result.data["rows_persisted"] == 0
        assert result.data["persist_error"] is None


class TestLimitTruncationIsVisible:
    """``limit`` silently dropping in-window certs is a partial write."""

    def test_recent_reports_the_pre_limit_window_count(self):
        records = [
            _live_shape_record(crt_id=n, serial_number=f"t{n}", common_name=f"n{n}.aqr.com", not_before=LAST_WEEK)
            for n in range(5)
        ]
        tool = CertTransparencyTool()
        with patch("agent.tools.cert_transparency.httpx.get") as mock_get:
            mock_get.return_value = _http_response(records=records)
            result = tool.execute(mode="recent", domain="aqr.com", days_back=30, limit=2)
        assert result.data["count"] == 2
        assert result.data["certs_in_window"] == 5
        assert "TRUNCATED" in result.output
        assert "5 were in the window" in result.output

    def test_no_truncation_note_when_everything_fits(self):
        tool = CertTransparencyTool()
        with patch("agent.tools.cert_transparency.httpx.get") as mock_get:
            mock_get.return_value = _http_response(records=[_live_shape_record(crt_id=1, serial_number="u1")])
            result = tool.execute(mode="recent", domain="aqr.com", days_back=30, limit=50)
        assert result.data["certs_in_window"] == 1
        assert "TRUNCATED" not in result.output


class TestParseTimestampTolerance:
    """The parser is now the ONLY gate on observed_at, so its gaps drop rows.

    ``_persist_entities_inner`` used ``datetime.fromisoformat`` (which accepts
    a trailing Z) before this changeset and ``_parse_timestamp`` after it. Two
    formats narrower means records that used to persist now get skipped with a
    warning and no row — the silent-loss shape, one layer down.
    """

    def test_zulu_suffix_is_accepted(self):
        parsed = _parse_timestamp("2025-01-15T12:00:00Z")
        assert parsed == datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)

    def test_explicit_offset_is_accepted_and_normalised(self):
        parsed = _parse_timestamp("2025-01-15T12:00:00+00:00")
        assert parsed is not None
        assert parsed.tzinfo is not None
        assert parsed.timestamp() == datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC).timestamp()

    def test_bare_date_is_midnight_not_ingestion_time(self):
        """The one thing it must never do is stamp the row with "now"."""
        parsed = _parse_timestamp("2025-01-01")
        assert parsed == datetime(2025, 1, 1, tzinfo=UTC)
        assert abs(parsed.timestamp() - time.time()) > 86400

    def test_garbage_is_still_refused(self):
        assert _parse_timestamp("not-a-date") is None
        assert _parse_timestamp("2026-13-45T99:99:99") is None


class TestUnparseableTimestampIsCounted:
    def test_a_skipped_record_shows_up_in_the_payload(self):
        """A record dropped by the parser must not vanish without a number."""
        good = _live_shape_record(crt_id=1, serial_number="g1", not_before=LAST_WEEK)
        bad = _live_shape_record(crt_id=2, serial_number="b1", not_before=LAST_WEEK)
        bad["not_before"] = "not-a-date"
        bad["entry_timestamp"] = "not-a-date"
        store = MagicMock()
        tool = CertTransparencyTool(pipeline_store=store)
        # Both records are inside the window; only one is storable.
        outcome = tool._persist_entities("aqr.com", [_normalize_record(good, NOW), _normalize_record(bad, NOW)])
        assert outcome.written == 1
        assert outcome.rejected == 1
        assert "unparseable" in outcome.error


class TestIdempotentRerun:
    def test_a_second_run_of_the_same_payload_adds_no_rows(self, tmp_path):
        """observed_at is the issuance instant, so re-runs collapse.

        This is the whole reason the persist path refuses to fall back to
        ``time.time()``: an ingestion-time stamp defeats the store's unique
        key and every daily run appends a fresh copy of every cert crt.sh has
        ever served for the domain.
        """
        import sqlite3

        from agent.pipeline.store import PipelineStore

        records = [
            _live_shape_record(crt_id=n, serial_number=f"i{n}", common_name=f"n{n}.aqr.com", not_before=LAST_WEEK)
            for n in range(3)
        ]
        db_path = tmp_path / "idempotent.db"
        store = PipelineStore(str(db_path))

        def count() -> int:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            try:
                return conn.execute(
                    "SELECT COUNT(*) FROM entity_observations WHERE source_tool='cert_transparency'"
                ).fetchone()[0]
            finally:
                conn.close()

        try:
            tool = CertTransparencyTool(pipeline_store=store)
            with patch("agent.tools.cert_transparency.httpx.get") as mock_get:
                mock_get.return_value = _http_response(records=records)
                first = tool.execute(mode="recent", domain="aqr.com", days_back=30)
                after_first = count()
                second = tool.execute(mode="recent", domain="aqr.com", days_back=30)
        finally:
            store.close()

        assert first.data["rows_persisted"] == 3
        assert after_first == 3
        # The tool still reports 3 handed over; the store collapses them.
        assert second.data["rows_persisted"] == 3
        assert count() == 3
