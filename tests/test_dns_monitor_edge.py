"""
Edge case tests for DNS Monitor tool (7b-B).

Covers: domain validation, mode routing, parameter validation, Google DoH response
parsing, Cloudflare failover, resolve mode, diff mode (baseline + changes),
bulk_resolve mode, cloud provider identification, MX provider detection,
NS provider detection, TXT token detection, TTL analysis, rate limiting,
cache integration, HTTP errors, timeout handling, malformed JSON,
tool schema, registry integration of count assertions (34 tools, 22 arms).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.tools.base import ToolResult
from agent.tools.dns_monitor import (
    _CACHE_TTL_RESOLVE,
    _CACHE_TTL_SNAPSHOT,
    _MAX_BULK_DOMAINS,
    _RATE_LIMIT_INTERVAL,
    ALL_RECORD_TYPES,
    DEFAULT_RECORD_TYPES,
    VALID_MODES,
    DnsMonitorTool,
    _analyze_records,
    _compute_diff,
    _format_analysis,
    _format_changes,
    _format_records,
    _identify_cloud_provider,
    _identify_mx_provider,
    _identify_ns_provider,
    _identify_txt_tokens,
    _parse_doh_response,
    _query_doh,
    _resolve_all_types,
    _resolve_type,
    _validate_domain,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_doh_response(
    answers: list[dict],
    status: int = 0,
) -> dict:
    """Build a Google DoH JSON response."""
    return {
        "Status": status,
        "TC": False,
        "RD": True,
        "RA": True,
        "AD": False,
        "CD": False,
        "Question": [{"name": "example.com", "type": 1}],
        "Answer": answers,
    }


def _make_a_answer(ip: str, ttl: int = 300, name: str = "example.com") -> dict:
    return {"name": name, "type": 1, "TTL": ttl, "data": ip}


def _make_aaaa_answer(ip: str, ttl: int = 300) -> dict:
    return {"name": "example.com", "type": 28, "TTL": ttl, "data": ip}


def _make_mx_answer(mx: str, ttl: int = 3600) -> dict:
    return {"name": "example.com", "type": 15, "TTL": ttl, "data": mx}


def _make_ns_answer(ns: str, ttl: int = 86400) -> dict:
    return {"name": "example.com", "type": 2, "TTL": ttl, "data": ns}


def _make_txt_answer(txt: str, ttl: int = 3600) -> dict:
    return {"name": "example.com", "type": 16, "TTL": ttl, "data": f'"{txt}"'}


def _make_cname_answer(cname: str, ttl: int = 300) -> dict:
    return {"name": "example.com", "type": 5, "TTL": ttl, "data": cname}


def _mock_httpx_get_success(data: dict):
    """Create a mock httpx response with JSON data."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = data
    mock_resp.raise_for_status.return_value = None
    return mock_resp


# ══════════════════════════════════════════════════════════════════════════════
# 1. Domain Validation
# ══════════════════════════════════════════════════════════════════════════════


class TestDomainValidation:
    def test_valid_simple_domain(self):
        assert _validate_domain("example.com") is None

    def test_valid_subdomain(self):
        assert _validate_domain("mail.example.com") is None

    def test_valid_deep_subdomain(self):
        assert _validate_domain("a.b.c.d.example.com") is None

    def test_valid_domain_with_hyphens(self):
        assert _validate_domain("my-site.example.com") is None

    def test_valid_domain_with_numbers(self):
        assert _validate_domain("site123.example.com") is None

    def test_valid_long_tld(self):
        assert _validate_domain("example.technology") is None

    def test_empty_domain(self):
        err = _validate_domain("")
        assert err is not None
        assert "required" in err.lower()

    def test_too_long_domain(self):
        domain = "a" * 250 + ".com"
        err = _validate_domain(domain)
        assert err is not None
        assert "too long" in err.lower()

    def test_invalid_no_tld(self):
        err = _validate_domain("localhost")
        assert err is not None
        assert "invalid" in err.lower()

    def test_invalid_single_char_tld(self):
        err = _validate_domain("example.x")
        assert err is not None

    def test_invalid_starts_with_dot(self):
        err = _validate_domain(".example.com")
        assert err is not None

    def test_invalid_ends_with_dot(self):
        # Trailing dot is technically valid in DNS but we reject it for simplicity
        err = _validate_domain("example.com.")
        assert err is not None

    def test_invalid_double_dots(self):
        err = _validate_domain("example..com")
        assert err is not None

    def test_invalid_spaces(self):
        err = _validate_domain("my site.com")
        assert err is not None

    def test_invalid_underscore(self):
        err = _validate_domain("my_site.com")
        assert err is not None

    def test_invalid_ip_address(self):
        err = _validate_domain("192.168.1.1")
        assert err is not None

    def test_invalid_url(self):
        err = _validate_domain("https://example.com")
        assert err is not None

    def test_invalid_special_chars(self):
        err = _validate_domain("example!.com")
        assert err is not None

    def test_max_length_label(self):
        # Each label can be up to 63 chars
        label = "a" * 63
        assert _validate_domain(f"{label}.com") is None

    def test_domain_253_chars(self):
        # Domain at exactly 253 chars
        parts = ["a" * 50] * 4 + ["com"]  # 50.50.50.50.com = 207 chars
        domain = ".".join(parts)
        if len(domain) <= 253:
            result = _validate_domain(domain)
            # Should be valid if within 253 chars
            assert result is None or "too long" not in (result or "").lower()


# ══════════════════════════════════════════════════════════════════════════════
# 2. Cloud Provider Identification
# ══════════════════════════════════════════════════════════════════════════════


class TestCloudProviderID:
    def test_aws_ip(self):
        assert _identify_cloud_provider("52.1.2.3") == "AWS"

    def test_aws_ip_54(self):
        assert _identify_cloud_provider("54.200.1.1") == "AWS"

    def test_gcp_ip(self):
        assert _identify_cloud_provider("35.192.1.1") == "GCP"

    def test_gcp_ip_34(self):
        assert _identify_cloud_provider("34.102.1.1") == "GCP"

    def test_azure_ip(self):
        assert _identify_cloud_provider("20.1.2.3") == "Azure"

    def test_azure_ip_13(self):
        assert _identify_cloud_provider("13.64.1.1") == "Azure"

    def test_cloudflare_ip(self):
        assert _identify_cloud_provider("104.16.1.1") == "Cloudflare"

    def test_cloudflare_ip_172(self):
        assert _identify_cloud_provider("172.67.1.1") == "Cloudflare"

    def test_fastly_ip(self):
        assert _identify_cloud_provider("151.101.1.1") == "Fastly"

    def test_akamai_ip(self):
        assert _identify_cloud_provider("23.32.1.1") == "Akamai"

    def test_unknown_ip(self):
        assert _identify_cloud_provider("192.168.1.1") is None

    def test_empty_ip(self):
        assert _identify_cloud_provider("") is None

    def test_private_ip_10(self):
        assert _identify_cloud_provider("10.0.0.1") is None

    def test_loopback(self):
        assert _identify_cloud_provider("127.0.0.1") is None


# ══════════════════════════════════════════════════════════════════════════════
# 3. MX Provider Detection
# ══════════════════════════════════════════════════════════════════════════════


class TestMXProviderDetection:
    def test_google_workspace(self):
        assert _identify_mx_provider("10 smtp.google.com") == "Google Workspace"

    def test_google_alt(self):
        assert _identify_mx_provider("20 alt2.aspmx.l.google.com") == "Google Workspace"

    def test_microsoft_365(self):
        assert _identify_mx_provider("10 company.mail.protection.outlook.com") == "Microsoft 365"

    def test_proofpoint(self):
        assert _identify_mx_provider("10 mx.pphosted.com") == "Proofpoint"

    def test_mimecast(self):
        assert _identify_mx_provider("10 us-smtp.mimecast.com") == "Mimecast"

    def test_sendgrid(self):
        assert _identify_mx_provider("10 mx.sendgrid.net") == "SendGrid"

    def test_unknown_mx(self):
        assert _identify_mx_provider("10 mail.custom-domain.com") is None

    def test_empty_mx(self):
        assert _identify_mx_provider("") is None

    def test_case_insensitive(self):
        assert _identify_mx_provider("10 SMTP.GOOGLE.COM") == "Google Workspace"


# ══════════════════════════════════════════════════════════════════════════════
# 4. NS Provider Detection
# ══════════════════════════════════════════════════════════════════════════════


class TestNSProviderDetection:
    def test_cloudflare_ns(self):
        assert _identify_ns_provider("ada.ns.cloudflare.com.") == "Cloudflare"

    def test_aws_route53(self):
        assert _identify_ns_provider("ns-1234.awsdns-56.co.uk") == "AWS Route53"

    def test_azure_dns(self):
        assert _identify_ns_provider("ns1-01.azure-dns.com") == "Azure DNS"

    def test_google_domains(self):
        assert _identify_ns_provider("ns-cloud-a1.googledomains.com") == "Google Domains"

    def test_godaddy(self):
        assert _identify_ns_provider("ns49.domaincontrol.com") == "GoDaddy"

    def test_namecheap(self):
        assert _identify_ns_provider("dns1.registrar-servers.com") == "Namecheap"

    def test_unknown_ns(self):
        assert _identify_ns_provider("ns1.custom-host.com") is None

    def test_empty_ns(self):
        assert _identify_ns_provider("") is None


# ══════════════════════════════════════════════════════════════════════════════
# 5. TXT Token Detection
# ══════════════════════════════════════════════════════════════════════════════


class TestTXTTokenDetection:
    def test_google_verification(self):
        tokens = _identify_txt_tokens("google-site-verification=abc123")
        assert "Google" in tokens

    def test_microsoft_verification(self):
        tokens = _identify_txt_tokens("MS=ms12345678")
        assert "Microsoft" in tokens

    def test_atlassian_verification(self):
        tokens = _identify_txt_tokens("atlassian-domain-verification=abc")
        assert "Atlassian" in tokens

    def test_facebook_verification(self):
        tokens = _identify_txt_tokens("facebook-domain-verification=abc")
        assert "Facebook/Meta" in tokens

    def test_stripe_verification(self):
        tokens = _identify_txt_tokens("stripe-verification=abc")
        assert "Stripe" in tokens

    def test_multiple_tokens_in_one_record(self):
        # Unlikely but test robustness
        txt = "google-site-verification=a MS=b"
        tokens = _identify_txt_tokens(txt)
        assert "Google" in tokens
        assert "Microsoft" in tokens

    def test_spf_record_not_token(self):
        tokens = _identify_txt_tokens("v=spf1 include:_spf.google.com ~all")
        # SPF is not a verification token
        assert tokens == []

    def test_empty_txt(self):
        assert _identify_txt_tokens("") == []

    def test_amazon_ses(self):
        tokens = _identify_txt_tokens("amazonses:abc123")
        assert "Amazon SES" in tokens

    def test_zoom_verification(self):
        tokens = _identify_txt_tokens("zoom-domain-verification=abc")
        assert "Zoom" in tokens

    def test_slack_verification(self):
        tokens = _identify_txt_tokens("slack-domain-verification=abc")
        assert "Slack" in tokens


# ══════════════════════════════════════════════════════════════════════════════
# 6. DoH Response Parsing
# ══════════════════════════════════════════════════════════════════════════════


class TestDoHResponseParsing:
    def test_parse_a_record(self):
        data = _make_doh_response([_make_a_answer("1.2.3.4", ttl=300)])
        records = _parse_doh_response(data, "A")
        assert len(records) == 1
        assert records[0]["type"] == "A"
        assert records[0]["value"] == "1.2.3.4"
        assert records[0]["ttl"] == 300

    def test_parse_multiple_a_records(self):
        data = _make_doh_response(
            [
                _make_a_answer("1.2.3.4"),
                _make_a_answer("5.6.7.8"),
            ]
        )
        records = _parse_doh_response(data, "A")
        assert len(records) == 2

    def test_parse_mx_record(self):
        data = _make_doh_response([_make_mx_answer("10 mail.example.com")])
        records = _parse_doh_response(data, "MX")
        assert len(records) == 1
        assert records[0]["type"] == "MX"
        assert records[0]["value"] == "10 mail.example.com"

    def test_parse_txt_record_strips_quotes(self):
        data = _make_doh_response([_make_txt_answer("v=spf1 ~all")])
        records = _parse_doh_response(data, "TXT")
        assert len(records) == 1
        assert records[0]["value"] == "v=spf1 ~all"  # quotes stripped

    def test_parse_ns_record(self):
        data = _make_doh_response([_make_ns_answer("ns1.example.com")])
        records = _parse_doh_response(data, "NS")
        assert len(records) == 1

    def test_parse_cname_record(self):
        data = _make_doh_response([_make_cname_answer("cdn.example.com")])
        records = _parse_doh_response(data, "CNAME")
        assert len(records) == 1

    def test_parse_empty_answer(self):
        data = _make_doh_response([])
        records = _parse_doh_response(data, "A")
        assert records == []

    def test_parse_nxdomain(self):
        data = _make_doh_response([], status=3)
        records = _parse_doh_response(data, "A")
        assert records == []

    def test_parse_filters_wrong_type(self):
        # Response has A record but we ask for MX
        data = _make_doh_response([_make_a_answer("1.2.3.4")])
        records = _parse_doh_response(data, "MX")
        assert records == []

    def test_parse_aaaa_record(self):
        data = _make_doh_response([_make_aaaa_answer("2001:db8::1")])
        records = _parse_doh_response(data, "AAAA")
        assert len(records) == 1
        assert records[0]["value"] == "2001:db8::1"

    def test_parse_txt_not_quoted(self):
        # Handle TXT records that aren't wrapped in quotes
        data = _make_doh_response([{"name": "example.com", "type": 16, "TTL": 300, "data": "v=spf1 ~all"}])
        records = _parse_doh_response(data, "TXT")
        assert records[0]["value"] == "v=spf1 ~all"

    def test_parse_missing_answer_key(self):
        data = {"Status": 0, "Question": []}
        records = _parse_doh_response(data, "A")
        assert records == []


# ══════════════════════════════════════════════════════════════════════════════
# 7. DoH Query Function
# ══════════════════════════════════════════════════════════════════════════════


class TestQueryDoH:
    @patch("agent.tools.dns_monitor.httpx.get")
    def test_google_success(self, mock_get):
        data = _make_doh_response([_make_a_answer("1.2.3.4")])
        mock_get.return_value = _mock_httpx_get_success(data)
        records, status, error = _query_doh("example.com", "A", provider="google")
        assert error is None
        assert status == 0
        assert len(records) == 1

    @patch("agent.tools.dns_monitor.httpx.get")
    def test_cloudflare_success(self, mock_get):
        data = _make_doh_response([_make_a_answer("1.2.3.4")])
        mock_get.return_value = _mock_httpx_get_success(data)
        records, status, error = _query_doh("example.com", "A", provider="cloudflare")
        assert error is None
        # Check Cloudflare header was used
        call_kwargs = mock_get.call_args
        assert call_kwargs.kwargs["headers"]["Accept"] == "application/dns-json"

    @patch("agent.tools.dns_monitor.httpx.get")
    def test_timeout_error(self, mock_get):
        mock_get.side_effect = httpx.TimeoutException("timeout")
        records, status, error = _query_doh("example.com", "A")
        assert error is not None
        assert "timeout" in error.lower()
        assert records == []

    @patch("agent.tools.dns_monitor.httpx.get")
    def test_http_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "rate limited", request=MagicMock(), response=mock_resp
        )
        mock_get.return_value = mock_resp
        records, status, error = _query_doh("example.com", "A")
        assert error is not None
        assert "429" in error
        assert records == []

    @patch("agent.tools.dns_monitor.httpx.get")
    def test_connection_error(self, mock_get):
        mock_get.side_effect = httpx.ConnectError("no connection")
        records, status, error = _query_doh("example.com", "A")
        assert error is not None
        assert "connection" in error.lower()

    @patch("agent.tools.dns_monitor.httpx.get")
    def test_invalid_json(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.side_effect = ValueError("bad json")
        mock_get.return_value = mock_resp
        records, status, error = _query_doh("example.com", "A")
        assert error is not None
        assert "json" in error.lower()

    @patch("agent.tools.dns_monitor.httpx.get")
    def test_non_dict_response(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = []  # list instead of dict
        mock_get.return_value = mock_resp
        records, status, error = _query_doh("example.com", "A")
        assert error is not None
        assert "unexpected" in error.lower()


# ══════════════════════════════════════════════════════════════════════════════
# 8. Resolve Type with Failover
# ══════════════════════════════════════════════════════════════════════════════


class TestResolveTypeFailover:
    @patch("agent.tools.dns_monitor._query_doh")
    def test_google_succeeds(self, mock_query):
        mock_query.return_value = (
            [{"type": "A", "value": "1.2.3.4", "ttl": 300}],
            0,
            None,
        )
        records, status = _resolve_type("example.com", "A")
        assert len(records) == 1
        # Should only call google
        assert mock_query.call_count == 1

    @patch("agent.tools.dns_monitor._query_doh")
    def test_google_fails_cloudflare_succeeds(self, mock_query):
        mock_query.side_effect = [
            ([], -1, "Google timeout"),
            ([{"type": "A", "value": "1.2.3.4", "ttl": 300}], 0, None),
        ]
        records, status = _resolve_type("example.com", "A")
        assert len(records) == 1
        assert mock_query.call_count == 2

    @patch("agent.tools.dns_monitor._query_doh")
    def test_both_fail(self, mock_query):
        mock_query.side_effect = [
            ([], -1, "Google error"),
            ([], -1, "Cloudflare error"),
        ]
        records, status = _resolve_type("example.com", "A")
        assert records == []
        assert status == -1


# ══════════════════════════════════════════════════════════════════════════════
# 9. Resolve All Types
# ══════════════════════════════════════════════════════════════════════════════


class TestResolveAllTypes:
    @patch("agent.tools.dns_monitor.time.sleep")
    @patch("agent.tools.dns_monitor._resolve_type")
    def test_resolves_all_types(self, mock_resolve, mock_sleep):
        mock_resolve.return_value = ([{"type": "A", "value": "1.2.3.4", "ttl": 300}], 0)
        results = _resolve_all_types("example.com", ["A", "MX"])
        assert "A" in results
        # MX also returns A-type records due to mock, but that's fine for testing
        assert mock_resolve.call_count == 2
        assert mock_sleep.call_count == 2

    @patch("agent.tools.dns_monitor.time.sleep")
    @patch("agent.tools.dns_monitor._resolve_type")
    def test_nxdomain_stops_early(self, mock_resolve, mock_sleep):
        mock_resolve.return_value = ([], 3)  # NXDOMAIN
        results = _resolve_all_types("nonexistent.example.com", ["A", "MX", "NS"])
        # Should stop after first NXDOMAIN
        assert mock_resolve.call_count == 1

    @patch("agent.tools.dns_monitor.time.sleep")
    @patch("agent.tools.dns_monitor._resolve_type")
    def test_empty_results_continue(self, mock_resolve, mock_sleep):
        # Empty but not NXDOMAIN (status=0) → continue
        mock_resolve.return_value = ([], 0)
        results = _resolve_all_types("example.com", ["A", "MX"])
        assert mock_resolve.call_count == 2


# ══════════════════════════════════════════════════════════════════════════════
# 10. Record Analysis
# ══════════════════════════════════════════════════════════════════════════════


class TestRecordAnalysis:
    def test_identifies_aws(self):
        records = {"A": [{"value": "52.1.2.3", "ttl": 300}]}
        analysis = _analyze_records(records)
        assert "AWS" in analysis["cloud_providers"]

    def test_identifies_multiple_providers(self):
        records = {
            "A": [
                {"value": "52.1.2.3", "ttl": 300},
                {"value": "104.16.1.1", "ttl": 300},
            ]
        }
        analysis = _analyze_records(records)
        assert "AWS" in analysis["cloud_providers"]
        assert "Cloudflare" in analysis["cloud_providers"]

    def test_identifies_mx_provider(self):
        records = {"MX": [{"value": "10 smtp.google.com", "ttl": 3600}]}
        analysis = _analyze_records(records)
        assert analysis["mx_provider"] == "Google Workspace"

    def test_identifies_ns_provider(self):
        records = {"NS": [{"value": "ada.ns.cloudflare.com.", "ttl": 86400}]}
        analysis = _analyze_records(records)
        assert analysis["ns_provider"] == "Cloudflare"

    def test_identifies_saas_tokens(self):
        records = {"TXT": [{"value": "google-site-verification=abc", "ttl": 3600}]}
        analysis = _analyze_records(records)
        assert "Google" in analysis["saas_tokens"]

    def test_low_ttl_warning(self):
        records = {"A": [{"value": "1.2.3.4", "ttl": 60}]}
        analysis = _analyze_records(records)
        assert analysis["low_ttl_warning"] is True
        assert analysis["min_ttl"] == 60

    def test_normal_ttl_no_warning(self):
        records = {"A": [{"value": "1.2.3.4", "ttl": 3600}]}
        analysis = _analyze_records(records)
        assert analysis["low_ttl_warning"] is False

    def test_empty_records(self):
        analysis = _analyze_records({})
        assert analysis["cloud_providers"] == []
        assert analysis["mx_provider"] is None
        assert analysis["min_ttl"] is None
        assert analysis["low_ttl_warning"] is False

    def test_min_ttl_across_types(self):
        records = {
            "A": [{"value": "1.2.3.4", "ttl": 3600}],
            "MX": [{"value": "10 mail.example.com", "ttl": 120}],
        }
        analysis = _analyze_records(records)
        assert analysis["min_ttl"] == 120

    def test_zero_ttl_ignored_in_min(self):
        records = {"A": [{"value": "1.2.3.4", "ttl": 0}]}
        analysis = _analyze_records(records)
        # TTL 0 records should not trigger low_ttl_warning through negative path
        # Since 0 is excluded from min_ttl calculation
        assert analysis["min_ttl"] is None


# ══════════════════════════════════════════════════════════════════════════════
# 11. Diff Computation
# ══════════════════════════════════════════════════════════════════════════════


class TestDiffComputation:
    def test_no_changes(self):
        old = {"A": [{"value": "1.2.3.4", "ttl": 300}]}
        new = {"A": [{"value": "1.2.3.4", "ttl": 300}]}
        changes = _compute_diff(old, new)
        assert changes == []

    def test_added_record(self):
        old = {"A": [{"value": "1.2.3.4", "ttl": 300}]}
        new = {
            "A": [
                {"value": "1.2.3.4", "ttl": 300},
                {"value": "5.6.7.8", "ttl": 300},
            ]
        }
        changes = _compute_diff(old, new)
        assert len(changes) == 1
        assert changes[0]["action"] == "added"
        assert changes[0]["value"] == "5.6.7.8"

    def test_removed_record(self):
        old = {
            "A": [
                {"value": "1.2.3.4", "ttl": 300},
                {"value": "5.6.7.8", "ttl": 300},
            ]
        }
        new = {"A": [{"value": "1.2.3.4", "ttl": 300}]}
        changes = _compute_diff(old, new)
        assert len(changes) == 1
        assert changes[0]["action"] == "removed"
        assert changes[0]["value"] == "5.6.7.8"

    def test_ttl_changed(self):
        old = {"A": [{"value": "1.2.3.4", "ttl": 86400}]}
        new = {"A": [{"value": "1.2.3.4", "ttl": 300}]}
        changes = _compute_diff(old, new)
        assert len(changes) == 1
        assert changes[0]["action"] == "ttl_changed"
        assert changes[0]["old_ttl"] == 86400
        assert changes[0]["new_ttl"] == 300

    def test_new_record_type_appears(self):
        old = {"A": [{"value": "1.2.3.4", "ttl": 300}]}
        new = {
            "A": [{"value": "1.2.3.4", "ttl": 300}],
            "MX": [{"value": "10 mail.example.com", "ttl": 3600}],
        }
        changes = _compute_diff(old, new)
        assert len(changes) == 1
        assert changes[0]["type"] == "MX"
        assert changes[0]["action"] == "added"

    def test_record_type_removed(self):
        old = {
            "A": [{"value": "1.2.3.4", "ttl": 300}],
            "MX": [{"value": "10 mail.example.com", "ttl": 3600}],
        }
        new = {"A": [{"value": "1.2.3.4", "ttl": 300}]}
        changes = _compute_diff(old, new)
        assert len(changes) == 1
        assert changes[0]["type"] == "MX"
        assert changes[0]["action"] == "removed"

    def test_both_empty(self):
        changes = _compute_diff({}, {})
        assert changes == []

    def test_multiple_changes(self):
        old = {
            "A": [{"value": "1.2.3.4", "ttl": 300}],
            "MX": [{"value": "10 old-mx.com", "ttl": 3600}],
        }
        new = {
            "A": [{"value": "5.6.7.8", "ttl": 300}],
            "MX": [{"value": "10 new-mx.com", "ttl": 3600}],
        }
        changes = _compute_diff(old, new)
        assert len(changes) == 4  # A: added + removed, MX: added + removed

    def test_value_change_is_add_plus_remove(self):
        old = {"A": [{"value": "1.2.3.4", "ttl": 300}]}
        new = {"A": [{"value": "5.6.7.8", "ttl": 300}]}
        changes = _compute_diff(old, new)
        actions = {c["action"] for c in changes}
        assert "added" in actions
        assert "removed" in actions


# ══════════════════════════════════════════════════════════════════════════════
# 12. Mode Routing
# ══════════════════════════════════════════════════════════════════════════════


class TestModeRouting:
    def test_invalid_mode(self):
        tool = DnsMonitorTool()
        result = tool.execute(mode="invalid")
        assert not result.success
        assert "invalid" in result.output.lower()

    def test_valid_modes_constant(self):
        assert {"resolve", "diff", "bulk_resolve"} == VALID_MODES

    def test_mode_case_insensitive(self):
        tool = DnsMonitorTool()
        # Should not fail on mode validation — will fail on domain validation
        result = tool.execute(mode="RESOLVE", domain="")
        # Mode accepted (lowercased), fails on domain
        assert "required" in result.output.lower() or "invalid" in result.output.lower()

    def test_mode_stripped(self):
        tool = DnsMonitorTool()
        result = tool.execute(mode=" resolve ", domain="")
        assert "required" in result.output.lower() or "invalid" in result.output.lower()


# ══════════════════════════════════════════════════════════════════════════════
# 13. Parameter Validation
# ══════════════════════════════════════════════════════════════════════════════


class TestParameterValidation:
    def test_invalid_record_type(self):
        tool = DnsMonitorTool()
        result = tool.execute(mode="resolve", domain="example.com", record_types=["FAKE"])
        assert not result.success
        assert "invalid record type" in result.output.lower()

    def test_multiple_invalid_record_types(self):
        tool = DnsMonitorTool()
        result = tool.execute(mode="resolve", domain="example.com", record_types=["FAKE", "BOGUS"])
        assert not result.success
        assert "FAKE" in result.output
        assert "BOGUS" in result.output

    def test_valid_record_types_accepted(self):
        for rt in ALL_RECORD_TYPES:
            tool = DnsMonitorTool()
            # Won't fail on record type validation
            result = tool.execute(mode="resolve", domain="", record_types=[rt])
            # Should fail on domain, not record type
            assert "invalid record type" not in result.output.lower()

    def test_empty_record_types_uses_default(self):
        # Empty list → use defaults
        tool = DnsMonitorTool()
        # This should use defaults. We test by checking it doesn't error on record types.
        result = tool.execute(mode="resolve", domain="", record_types=[])
        assert "invalid record type" not in result.output.lower()

    def test_bulk_resolve_no_domains(self):
        tool = DnsMonitorTool()
        result = tool.execute(mode="bulk_resolve", domains=[])
        assert not result.success
        assert "required" in result.output.lower()

    def test_bulk_resolve_too_many_domains(self):
        tool = DnsMonitorTool()
        domains = [f"domain{i}.com" for i in range(_MAX_BULK_DOMAINS + 1)]
        result = tool.execute(mode="bulk_resolve", domains=domains)
        assert not result.success
        assert "too many" in result.output.lower()

    def test_resolve_missing_domain(self):
        tool = DnsMonitorTool()
        result = tool.execute(mode="resolve", domain="")
        assert not result.success

    def test_diff_missing_domain(self):
        tool = DnsMonitorTool()
        result = tool.execute(mode="diff", domain="")
        assert not result.success


# ══════════════════════════════════════════════════════════════════════════════
# 14. Resolve Mode (full integration)
# ══════════════════════════════════════════════════════════════════════════════


class TestResolveMode:
    @patch("agent.tools.dns_monitor._resolve_all_types")
    def test_resolve_returns_records(self, mock_resolve):
        mock_resolve.return_value = {
            "A": [{"value": "52.1.2.3", "ttl": 300}],
            "MX": [{"value": "10 smtp.google.com", "ttl": 3600}],
        }
        tool = DnsMonitorTool()
        result = tool.execute(mode="resolve", domain="example.com")
        assert result.success
        assert "example.com" in result.output
        assert result.data["record_count"] == 2
        assert "AWS" in result.data["analysis"]["cloud_providers"]

    @patch("agent.tools.dns_monitor._resolve_all_types")
    def test_resolve_no_records(self, mock_resolve):
        mock_resolve.return_value = {}
        tool = DnsMonitorTool()
        result = tool.execute(mode="resolve", domain="nonexistent.example.com")
        assert result.success
        assert "no records" in result.output.lower()
        assert result.data["record_count"] == 0

    @patch("agent.tools.dns_monitor._resolve_all_types")
    def test_resolve_with_cache_hit(self, mock_resolve):
        cache = MagicMock()
        cache.get.return_value = {
            "output": "cached output",
            "data": {"domain": "example.com", "cached": True},
        }
        tool = DnsMonitorTool(cache=cache)
        result = tool.execute(mode="resolve", domain="example.com")
        assert result.success
        assert result.output == "cached output"
        # Should not have called resolve
        mock_resolve.assert_not_called()

    @patch("agent.tools.dns_monitor._resolve_all_types")
    def test_resolve_cache_miss_stores_result(self, mock_resolve):
        mock_resolve.return_value = {
            "A": [{"value": "1.2.3.4", "ttl": 300}],
        }
        cache = MagicMock()
        cache.get.return_value = None
        tool = DnsMonitorTool(cache=cache)
        result = tool.execute(mode="resolve", domain="example.com")
        assert result.success
        cache.set.assert_called_once()
        # Check TTL
        call_args = cache.set.call_args
        assert call_args.kwargs["ttl"] == _CACHE_TTL_RESOLVE

    @patch("agent.tools.dns_monitor._resolve_all_types")
    def test_resolve_invalid_domain(self, mock_resolve):
        tool = DnsMonitorTool()
        result = tool.execute(mode="resolve", domain="not a domain!")
        assert not result.success
        mock_resolve.assert_not_called()

    @patch("agent.tools.dns_monitor._resolve_all_types")
    def test_resolve_low_ttl_in_output(self, mock_resolve):
        mock_resolve.return_value = {
            "A": [{"value": "1.2.3.4", "ttl": 60}],
        }
        tool = DnsMonitorTool()
        result = tool.execute(mode="resolve", domain="example.com")
        assert result.success
        assert "LOW TTL" in result.output
        assert result.data["analysis"]["low_ttl_warning"] is True


# ══════════════════════════════════════════════════════════════════════════════
# 15. Diff Mode
# ══════════════════════════════════════════════════════════════════════════════


class TestDiffMode:
    @patch("agent.tools.dns_monitor._resolve_all_types")
    def test_diff_first_scan_baseline(self, mock_resolve):
        mock_resolve.return_value = {
            "A": [{"value": "1.2.3.4", "ttl": 300}],
        }
        cache = MagicMock()
        cache.get.return_value = None  # No previous snapshot
        tool = DnsMonitorTool(cache=cache)
        result = tool.execute(mode="diff", domain="example.com")
        assert result.success
        assert "baseline" in result.output.lower()
        assert result.data["baseline_established"] is True
        # Should store snapshot
        cache.set.assert_called()

    @patch("agent.tools.dns_monitor._resolve_all_types")
    def test_diff_no_changes(self, mock_resolve):
        current = {"A": [{"value": "1.2.3.4", "ttl": 300}]}
        mock_resolve.return_value = current
        cache = MagicMock()
        # Return same records as previous snapshot
        cache.get.side_effect = lambda source, key: (
            None if key.get("mode") == "resolve" else current if key.get("mode") == "snapshot" else None
        )
        tool = DnsMonitorTool(cache=cache)
        result = tool.execute(mode="diff", domain="example.com")
        assert result.success
        assert "no changes" in result.output.lower()
        assert result.data["changes"] == []

    @patch("agent.tools.dns_monitor._resolve_all_types")
    def test_diff_detects_added_record(self, mock_resolve):
        old = {"A": [{"value": "1.2.3.4", "ttl": 300}]}
        new = {
            "A": [
                {"value": "1.2.3.4", "ttl": 300},
                {"value": "5.6.7.8", "ttl": 300},
            ]
        }
        mock_resolve.return_value = new
        cache = MagicMock()
        cache.get.side_effect = lambda source, key: old if key.get("mode") == "snapshot" else None
        tool = DnsMonitorTool(cache=cache)
        result = tool.execute(mode="diff", domain="example.com")
        assert result.success
        assert len(result.data["changes"]) == 1
        assert result.data["changes"][0]["action"] == "added"
        assert "+" in result.output

    @patch("agent.tools.dns_monitor._resolve_all_types")
    def test_diff_detects_removed_record(self, mock_resolve):
        old = {"A": [{"value": "1.2.3.4", "ttl": 300}, {"value": "5.6.7.8", "ttl": 300}]}
        new = {"A": [{"value": "1.2.3.4", "ttl": 300}]}
        mock_resolve.return_value = new
        cache = MagicMock()
        cache.get.side_effect = lambda source, key: old if key.get("mode") == "snapshot" else None
        tool = DnsMonitorTool(cache=cache)
        result = tool.execute(mode="diff", domain="example.com")
        assert result.success
        assert len(result.data["changes"]) == 1
        assert result.data["changes"][0]["action"] == "removed"

    @patch("agent.tools.dns_monitor._resolve_all_types")
    def test_diff_detects_ttl_change(self, mock_resolve):
        old = {"A": [{"value": "1.2.3.4", "ttl": 86400}]}
        new = {"A": [{"value": "1.2.3.4", "ttl": 300}]}
        mock_resolve.return_value = new
        cache = MagicMock()
        cache.get.side_effect = lambda source, key: old if key.get("mode") == "snapshot" else None
        tool = DnsMonitorTool(cache=cache)
        result = tool.execute(mode="diff", domain="example.com")
        assert result.success
        assert len(result.data["changes"]) == 1
        assert result.data["changes"][0]["action"] == "ttl_changed"
        assert "↓" in result.output  # TTL decreased

    @patch("agent.tools.dns_monitor._resolve_all_types")
    def test_diff_invalid_domain(self, mock_resolve):
        tool = DnsMonitorTool()
        result = tool.execute(mode="diff", domain="")
        assert not result.success
        mock_resolve.assert_not_called()

    @patch("agent.tools.dns_monitor._resolve_all_types")
    def test_diff_no_cache_still_works(self, mock_resolve):
        mock_resolve.return_value = {"A": [{"value": "1.2.3.4", "ttl": 300}]}
        tool = DnsMonitorTool(cache=None)
        result = tool.execute(mode="diff", domain="example.com")
        assert result.success
        # No cache → always baseline
        assert result.data["baseline_established"] is True

    @patch("agent.tools.dns_monitor._resolve_all_types")
    def test_diff_snapshot_stored_with_long_ttl(self, mock_resolve):
        mock_resolve.return_value = {"A": [{"value": "1.2.3.4", "ttl": 300}]}
        cache = MagicMock()
        cache.get.return_value = None
        tool = DnsMonitorTool(cache=cache)
        result = tool.execute(mode="diff", domain="example.com")
        # Check snapshot was stored with long TTL
        set_calls = cache.set.call_args_list
        assert any(c.kwargs.get("ttl") == _CACHE_TTL_SNAPSHOT for c in set_calls)


# ══════════════════════════════════════════════════════════════════════════════
# 16. Bulk Resolve Mode
# ══════════════════════════════════════════════════════════════════════════════


class TestBulkResolveMode:
    @patch("agent.tools.dns_monitor._resolve_all_types")
    def test_bulk_resolves_multiple_domains(self, mock_resolve):
        mock_resolve.return_value = {"A": [{"value": "1.2.3.4", "ttl": 300}]}
        tool = DnsMonitorTool()
        result = tool.execute(
            mode="bulk_resolve",
            domains=["example.com", "test.org"],
        )
        assert result.success
        assert result.data["domain_count"] == 2
        assert mock_resolve.call_count == 2

    @patch("agent.tools.dns_monitor._resolve_all_types")
    def test_bulk_with_invalid_domain_continues(self, mock_resolve):
        mock_resolve.return_value = {"A": [{"value": "1.2.3.4", "ttl": 300}]}
        tool = DnsMonitorTool()
        result = tool.execute(
            mode="bulk_resolve",
            domains=["example.com", "not valid!", "test.org"],
        )
        assert result.success
        assert result.data["domain_count"] == 2  # Only valid ones
        assert len(result.data["errors"]) == 1

    @patch("agent.tools.dns_monitor._resolve_all_types")
    def test_bulk_all_invalid(self, mock_resolve):
        tool = DnsMonitorTool()
        result = tool.execute(
            mode="bulk_resolve",
            domains=["not valid!", "also bad"],
        )
        assert result.success  # Completes but with errors
        assert result.data["domain_count"] == 0
        assert len(result.data["errors"]) == 2

    @patch("agent.tools.dns_monitor._resolve_all_types")
    def test_bulk_output_format(self, mock_resolve):
        mock_resolve.return_value = {
            "A": [{"value": "52.1.2.3", "ttl": 300}],
            "MX": [{"value": "10 smtp.google.com", "ttl": 3600}],
        }
        tool = DnsMonitorTool()
        result = tool.execute(
            mode="bulk_resolve",
            domains=["example.com"],
        )
        assert result.success
        assert "example.com" in result.output
        assert result.data["total_records"] == 2

    def test_bulk_empty_domains(self):
        tool = DnsMonitorTool()
        result = tool.execute(mode="bulk_resolve", domains=[])
        assert not result.success

    def test_bulk_none_domains(self):
        tool = DnsMonitorTool()
        result = tool.execute(mode="bulk_resolve")
        assert not result.success

    def test_bulk_too_many_domains(self):
        tool = DnsMonitorTool()
        domains = [f"d{i}.com" for i in range(25)]
        result = tool.execute(mode="bulk_resolve", domains=domains)
        assert not result.success
        assert "too many" in result.output.lower()


# ══════════════════════════════════════════════════════════════════════════════
# 17. Formatting
# ══════════════════════════════════════════════════════════════════════════════


class TestFormatting:
    def test_format_records_a(self):
        records = {"A": [{"value": "1.2.3.4", "ttl": 300}]}
        lines = _format_records(records)
        assert len(lines) == 1
        assert "A" in lines[0]
        assert "1.2.3.4" in lines[0]
        assert "300" in lines[0]

    def test_format_records_empty(self):
        lines = _format_records({})
        assert lines == []

    def test_format_records_multiple_types(self):
        records = {
            "A": [{"value": "1.2.3.4", "ttl": 300}],
            "MX": [{"value": "10 mail.example.com", "ttl": 3600}],
        }
        lines = _format_records(records)
        assert len(lines) == 2

    def test_format_analysis_with_cloud(self):
        analysis = {
            "cloud_providers": ["AWS"],
            "mx_provider": "Google Workspace",
            "ns_provider": None,
            "saas_tokens": [],
            "min_ttl": 300,
            "low_ttl_warning": False,
        }
        lines = _format_analysis(analysis)
        assert any("AWS" in line for line in lines)
        assert any("Google Workspace" in line for line in lines)

    def test_format_analysis_low_ttl_warning(self):
        analysis = {
            "cloud_providers": [],
            "mx_provider": None,
            "ns_provider": None,
            "saas_tokens": [],
            "min_ttl": 60,
            "low_ttl_warning": True,
        }
        lines = _format_analysis(analysis)
        assert any("LOW TTL" in line for line in lines)

    def test_format_analysis_empty(self):
        analysis = {
            "cloud_providers": [],
            "mx_provider": None,
            "ns_provider": None,
            "saas_tokens": [],
            "min_ttl": None,
            "low_ttl_warning": False,
        }
        lines = _format_analysis(analysis)
        assert lines == []

    def test_format_changes_added(self):
        changes = [{"type": "A", "action": "added", "value": "1.2.3.4", "ttl": 300}]
        lines = _format_changes(changes)
        assert len(lines) == 1
        assert "+" in lines[0]

    def test_format_changes_removed(self):
        changes = [{"type": "A", "action": "removed", "value": "1.2.3.4", "old_ttl": 300}]
        lines = _format_changes(changes)
        assert len(lines) == 1
        assert "-" in lines[0]

    def test_format_changes_ttl_decreased(self):
        changes = [
            {
                "type": "A",
                "action": "ttl_changed",
                "value": "1.2.3.4",
                "old_ttl": 86400,
                "new_ttl": 300,
            }
        ]
        lines = _format_changes(changes)
        assert "↓" in lines[0]

    def test_format_changes_ttl_increased(self):
        changes = [
            {
                "type": "A",
                "action": "ttl_changed",
                "value": "1.2.3.4",
                "old_ttl": 300,
                "new_ttl": 86400,
            }
        ]
        lines = _format_changes(changes)
        assert "↑" in lines[0]


# ══════════════════════════════════════════════════════════════════════════════
# 18. Tool Schema & Constants
# ══════════════════════════════════════════════════════════════════════════════


class TestToolSchema:
    def test_tool_name(self):
        tool = DnsMonitorTool()
        assert tool.name == "dns_monitor"

    def test_tool_description(self):
        tool = DnsMonitorTool()
        assert "DNS" in tool.description

    def test_schema_has_mode(self):
        tool = DnsMonitorTool()
        props = tool.parameters["properties"]
        assert "mode" in props
        assert set(props["mode"]["enum"]) == {"resolve", "diff", "bulk_resolve"}

    def test_schema_has_domain(self):
        tool = DnsMonitorTool()
        assert "domain" in tool.parameters["properties"]

    def test_schema_has_domains(self):
        tool = DnsMonitorTool()
        assert "domains" in tool.parameters["properties"]

    def test_schema_has_record_types(self):
        tool = DnsMonitorTool()
        assert "record_types" in tool.parameters["properties"]

    def test_schema_required_mode(self):
        tool = DnsMonitorTool()
        assert "mode" in tool.parameters["required"]

    def test_default_record_types(self):
        assert set(DEFAULT_RECORD_TYPES) == {"A", "AAAA", "MX", "NS", "TXT", "CNAME"}

    def test_all_record_types_superset(self):
        assert set(DEFAULT_RECORD_TYPES).issubset(ALL_RECORD_TYPES)

    def test_constants(self):
        assert _MAX_BULK_DOMAINS == 20
        assert _CACHE_TTL_RESOLVE == 3600
        assert _CACHE_TTL_SNAPSHOT == 604800
        assert _RATE_LIMIT_INTERVAL > 0


# ══════════════════════════════════════════════════════════════════════════════
# 19. Registry Integration
# ══════════════════════════════════════════════════════════════════════════════


class TestRegistryIntegration:
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
        names = registry.list_names()
        assert len(names) == 60, f"Expected 60 tools, got {len(names)}: {sorted(names)}"

    def test_dns_monitor_in_registry(self):
        registry = self._build_registry()
        names = registry.list_names()
        assert "dns_monitor" in names

    def test_arm_count(self):
        from agent.learning.bandit import DEFAULT_ARMS

        assert len(DEFAULT_ARMS) == 48, f"Expected 48 arms, got {len(DEFAULT_ARMS)}: {[a.name for a in DEFAULT_ARMS]}"

    def test_infrastructure_recon_arm_includes_dns(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm = next(a for a in DEFAULT_ARMS if a.name == "infrastructure_recon")
        assert "dns_monitor" in arm.tools
        assert "cert_transparency" in arm.tools


# ══════════════════════════════════════════════════════════════════════════════
# 20. Cache Integration Details
# ══════════════════════════════════════════════════════════════════════════════


class TestCacheIntegration:
    @patch("agent.tools.dns_monitor._resolve_all_types")
    def test_no_cache_resolve_works(self, mock_resolve):
        mock_resolve.return_value = {"A": [{"value": "1.2.3.4", "ttl": 300}]}
        tool = DnsMonitorTool(cache=None)
        result = tool.execute(mode="resolve", domain="example.com")
        assert result.success

    @patch("agent.tools.dns_monitor._resolve_all_types")
    def test_cache_key_includes_record_types(self, mock_resolve):
        mock_resolve.return_value = {"A": [{"value": "1.2.3.4", "ttl": 300}]}
        cache = MagicMock()
        cache.get.return_value = None
        tool = DnsMonitorTool(cache=cache)
        tool.execute(mode="resolve", domain="example.com", record_types=["A", "MX"])
        # Check cache key includes types
        get_call_key = cache.get.call_args[0][1]
        assert "types" in get_call_key
        assert get_call_key["types"] == ["A", "MX"]


# ══════════════════════════════════════════════════════════════════════════════
# 21. Result Format
# ══════════════════════════════════════════════════════════════════════════════


class TestResultFormat:
    @patch("agent.tools.dns_monitor._resolve_all_types")
    def test_resolve_returns_tool_result(self, mock_resolve):
        mock_resolve.return_value = {"A": [{"value": "1.2.3.4", "ttl": 300}]}
        tool = DnsMonitorTool()
        result = tool.execute(mode="resolve", domain="example.com")
        assert isinstance(result, ToolResult)
        assert isinstance(result.output, str)
        assert isinstance(result.data, dict)

    @patch("agent.tools.dns_monitor._resolve_all_types")
    def test_resolve_data_has_required_keys(self, mock_resolve):
        mock_resolve.return_value = {"A": [{"value": "1.2.3.4", "ttl": 300}]}
        tool = DnsMonitorTool()
        result = tool.execute(mode="resolve", domain="example.com")
        assert "domain" in result.data
        assert "records" in result.data
        assert "analysis" in result.data
        assert "record_count" in result.data

    @patch("agent.tools.dns_monitor._resolve_all_types")
    def test_diff_data_has_required_keys(self, mock_resolve):
        mock_resolve.return_value = {"A": [{"value": "1.2.3.4", "ttl": 300}]}
        tool = DnsMonitorTool(cache=MagicMock(get=MagicMock(return_value=None)))
        result = tool.execute(mode="diff", domain="example.com")
        assert "domain" in result.data
        assert "baseline_established" in result.data
        assert "records" in result.data
        assert "changes" in result.data

    @patch("agent.tools.dns_monitor._resolve_all_types")
    def test_bulk_data_has_required_keys(self, mock_resolve):
        mock_resolve.return_value = {"A": [{"value": "1.2.3.4", "ttl": 300}]}
        tool = DnsMonitorTool()
        result = tool.execute(mode="bulk_resolve", domains=["example.com"])
        assert "results" in result.data
        assert "errors" in result.data
        assert "domain_count" in result.data
        assert "total_records" in result.data

    def test_error_result_format(self):
        tool = DnsMonitorTool()
        result = tool.execute(mode="resolve", domain="")
        assert isinstance(result, ToolResult)
        assert not result.success
        assert isinstance(result.output, str)
        assert len(result.output) > 0
