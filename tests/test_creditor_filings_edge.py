"""
Edge-case tests for CreditorFilingsTool (7b-V).

Coverage targets:
- Invalid / missing / boundary parameters
- Empty / malformed API responses
- Cache hit / miss paths
- SEC EFTS parsing edge cases
- UK Companies House parsing edge cases
- Cluster detection logic
- Red-flag charge counting
- HTTP errors, timeouts
- Mode validation
- Integration: tool count = 38, arm count = 26
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.tools.creditor_filings import (
    _CACHE_TTL,
    _CHARGE_RED_FLAGS,
    _CREDIT_ITEMS,
    _CREDIT_TERMS,
    _DEFAULT_LIMIT,
    _MAX_LIMIT,
    VALID_MODES,
    CreditorFilingsTool,
    _classify_charge,
    _count_red_flag_charges,
    _detect_filing_clusters,
    _extract_particulars,
    _fetch_json,
    _get_ch_charges,
    _get_ch_key,
    _parse_efts_hits,
    _search_ch_company,
    _search_efts,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tool():
    cache = MagicMock()
    cache.get.return_value = None
    return CreditorFilingsTool(cache=cache)


@pytest.fixture
def tool_no_cache():
    return CreditorFilingsTool(cache=None)


def _efts_record(
    company: str = "Acme Corp",
    cik: str = "0001234567",
    file_date: str = "2025-06-15",
    form: str = "8-K",
    items: list[str] | None = None,
) -> dict[str, Any]:
    """Build a synthetic EFTS hit."""
    return {
        "_source": {
            "display_names": [company],
            "ciks": [cik],
            "file_date": file_date,
            "form": form,
            "items": items or ["1.01"],
        }
    }


def _efts_response(hits: list[dict] | None = None, total: int | None = None) -> dict[str, Any]:
    """Build a synthetic EFTS response envelope."""
    hit_list = hits or []
    return {
        "hits": {
            "total": {"value": total if total is not None else len(hit_list)},
            "hits": hit_list,
        }
    }


def _ch_charge(
    charge_number: int = 1,
    status: str = "satisfied",
    created_on: str = "2020-01-01",
    satisfied_on: str = "2021-01-01",
    classification: str = "fixed_charge",
    persons_entitled: list[str] | None = None,
    description: str = "A fixed charge over assets",
) -> dict[str, Any]:
    """Build a synthetic Companies House charge item."""
    return {
        "charge_number": charge_number,
        "status": status,
        "created_on": created_on,
        "delivered_on": created_on,
        "satisfied_on": satisfied_on if status == "satisfied" else "",
        "particulars": {"description": description},
        "persons_entitled": [{"name": n} for n in (persons_entitled or ["BigBank PLC"])],
    }


def _ch_charges_response(items: list[dict] | None = None) -> dict[str, Any]:
    return {"items": items or []}


def _ch_search_response(
    companies: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Build Companies House search response."""
    results = []
    for name, number in companies or []:
        results.append(
            {
                "title": name,
                "company_number": number,
                "company_status": "active",
                "date_of_creation": "2010-01-01",
            }
        )
    return {"items": results}


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
        assert {"search", "uk_charges", "stress_scan"} == VALID_MODES

    def test_empty_mode(self, tool):
        r = tool.execute(mode="")
        assert not r.success
        assert "Invalid mode" in r.output

    def test_invalid_mode(self, tool):
        r = tool.execute(mode="bogus")
        assert not r.success
        assert "bogus" in r.output

    def test_mode_case_insensitive(self, tool):
        r = tool.execute(mode="SEARCH")
        # Should not fail on mode validation (may fail on missing query)
        assert "Invalid mode" not in r.output

    def test_mode_whitespace(self, tool):
        r = tool.execute(mode="  search  ")
        assert "Invalid mode" not in r.output

    def test_none_mode(self, tool):
        r = tool.execute(mode=None)
        assert not r.success


# ═══════════════════════════════════════════════════════════════════════════
# 2. Search mode
# ═══════════════════════════════════════════════════════════════════════════


class TestSearchMode:
    def test_missing_query(self, tool):
        r = tool.execute(mode="search")
        assert not r.success
        assert "query" in r.output.lower()

    def test_empty_query(self, tool):
        r = tool.execute(mode="search", query="")
        assert not r.success

    def test_whitespace_query(self, tool):
        r = tool.execute(mode="search", query="   ")
        assert not r.success

    @patch("agent.tools.creditor_filings._get_ch_key", return_value=None)
    @patch("agent.tools.creditor_filings.httpx.Client")
    def test_search_basic(self, mock_client_cls, mock_ch_key, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(_efts_response([_efts_record()], total=1))

        r = tool.execute(mode="search", query="Acme")
        assert r.success
        assert "Acme" in r.output
        assert r.data["sec_count"] == 1

    @patch("agent.tools.creditor_filings._get_ch_key", return_value=None)
    @patch("agent.tools.creditor_filings.httpx.Client")
    def test_search_empty_results(self, mock_client_cls, mock_ch_key, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(_efts_response([], total=0))

        r = tool.execute(mode="search", query="NonexistentCorp")
        assert r.success
        assert r.data["sec_count"] == 0

    @patch("agent.tools.creditor_filings._get_ch_key", return_value=None)
    @patch("agent.tools.creditor_filings.httpx.Client")
    def test_search_many_results(self, mock_client_cls, mock_ch_key, tool):
        hits = [_efts_record(company=f"Corp{i}") for i in range(50)]
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(_efts_response(hits, total=200))

        r = tool.execute(mode="search", query="Corp", limit=50)
        assert r.success
        assert "... and" in r.output  # truncated display

    @patch("agent.tools.creditor_filings._get_ch_key", return_value=None)
    @patch("agent.tools.creditor_filings.httpx.Client")
    def test_search_http_error(self, mock_client_cls, mock_ch_key, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response({}, status=500)

        r = tool.execute(mode="search", query="Acme")
        assert r.success  # Empty results, not a failure
        assert r.data["sec_count"] == 0

    def test_search_cache_hit(self, tool):
        cached = {
            "entries": [
                {
                    "company_name": "CachedCorp",
                    "cik": "111",
                    "file_date": "2025-01-01",
                    "form": "8-K",
                    "items": ["1.01"],
                }
            ],
            "total": 1,
            "ch_charges": None,
        }
        tool._cache.get.return_value = cached

        r = tool.execute(mode="search", query="CachedCorp")
        assert r.success
        assert "(cached)" in r.output
        assert "CachedCorp" in r.output

    @patch("agent.tools.creditor_filings._get_ch_key", return_value="test_key")
    @patch("agent.tools.creditor_filings._ch_client")
    @patch("agent.tools.creditor_filings.httpx.Client")
    def test_search_with_ch(self, mock_client_cls, mock_ch_client, mock_ch_key, tool):
        # SEC client
        mock_sec = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_sec)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_sec.get.return_value = _mock_response(_efts_response([_efts_record()], total=1))

        # CH client
        mock_ch = MagicMock()
        mock_ch_client.return_value.__enter__ = MagicMock(return_value=mock_ch)
        mock_ch_client.return_value.__exit__ = MagicMock(return_value=False)

        # First call = company search, second = charges
        mock_ch.get.side_effect = [
            _mock_response(_ch_search_response([("Acme UK Ltd", "12345678")])),
            _mock_response(_ch_charges_response([_ch_charge()])),
        ]

        r = tool.execute(mode="search", query="Acme")
        assert r.success
        assert r.data["ch_charges"] is not None

    @patch("agent.tools.creditor_filings._get_ch_key", return_value=None)
    @patch("agent.tools.creditor_filings.httpx.Client")
    def test_search_no_ch_key_message(self, mock_client_cls, mock_ch_key, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(_efts_response([], total=0))

        r = tool.execute(mode="search", query="Acme")
        assert "TIRRA_COMPANIES_HOUSE_KEY" in r.output


# ═══════════════════════════════════════════════════════════════════════════
# 3. UK Charges mode
# ═══════════════════════════════════════════════════════════════════════════


class TestUkChargesMode:
    @patch("agent.tools.creditor_filings._get_ch_key", return_value=None)
    def test_no_api_key(self, mock_key, tool):
        r = tool.execute(mode="uk_charges", company_number="12345678")
        assert not r.success
        assert "TIRRA_COMPANIES_HOUSE_KEY" in r.output

    @patch("agent.tools.creditor_filings._get_ch_key", return_value="key")
    def test_no_company_or_query(self, mock_key, tool):
        r = tool.execute(mode="uk_charges")
        assert not r.success
        assert "company_number" in r.output

    @patch("agent.tools.creditor_filings._get_ch_key", return_value="key")
    @patch("agent.tools.creditor_filings._ch_client")
    def test_by_company_number(self, mock_ch_client, mock_key, tool):
        mock_ch = MagicMock()
        mock_ch_client.return_value.__enter__ = MagicMock(return_value=mock_ch)
        mock_ch_client.return_value.__exit__ = MagicMock(return_value=False)
        mock_ch.get.return_value = _mock_response(_ch_charges_response([_ch_charge(status="outstanding")]))

        r = tool.execute(mode="uk_charges", company_number="12345678")
        assert r.success
        assert r.data["charge_count"] == 1
        assert r.data["red_flag_count"] == 1

    @patch("agent.tools.creditor_filings._get_ch_key", return_value="key")
    @patch("agent.tools.creditor_filings._ch_client")
    def test_by_name_search(self, mock_ch_client, mock_key, tool):
        mock_ch = MagicMock()
        mock_ch_client.return_value.__enter__ = MagicMock(return_value=mock_ch)
        mock_ch_client.return_value.__exit__ = MagicMock(return_value=False)
        mock_ch.get.side_effect = [
            _mock_response(_ch_search_response([("TestCo", "99999999")])),
            _mock_response(_ch_charges_response([_ch_charge()])),
        ]

        r = tool.execute(mode="uk_charges", query="TestCo")
        assert r.success

    @patch("agent.tools.creditor_filings._get_ch_key", return_value="key")
    @patch("agent.tools.creditor_filings._ch_client")
    def test_company_not_found(self, mock_ch_client, mock_key, tool):
        mock_ch = MagicMock()
        mock_ch_client.return_value.__enter__ = MagicMock(return_value=mock_ch)
        mock_ch_client.return_value.__exit__ = MagicMock(return_value=False)
        mock_ch.get.return_value = _mock_response(_ch_search_response([]))

        r = tool.execute(mode="uk_charges", query="NoSuchCompany")
        assert not r.success
        assert "No UK company" in r.output

    @patch("agent.tools.creditor_filings._get_ch_key", return_value="key")
    @patch("agent.tools.creditor_filings._ch_client")
    def test_no_charges(self, mock_ch_client, mock_key, tool):
        mock_ch = MagicMock()
        mock_ch_client.return_value.__enter__ = MagicMock(return_value=mock_ch)
        mock_ch_client.return_value.__exit__ = MagicMock(return_value=False)
        mock_ch.get.return_value = _mock_response(_ch_charges_response([]))

        r = tool.execute(mode="uk_charges", company_number="00000001")
        assert r.success
        assert r.data["charge_count"] == 0
        assert r.data["red_flag_count"] == 0

    @patch("agent.tools.creditor_filings._get_ch_key", return_value="key")
    @patch("agent.tools.creditor_filings._ch_client")
    def test_high_stress_charges(self, mock_ch_client, mock_key, tool):
        charges = [_ch_charge(charge_number=i, status="outstanding") for i in range(5)]
        mock_ch = MagicMock()
        mock_ch_client.return_value.__enter__ = MagicMock(return_value=mock_ch)
        mock_ch_client.return_value.__exit__ = MagicMock(return_value=False)
        mock_ch.get.return_value = _mock_response(_ch_charges_response(charges))

        r = tool.execute(mode="uk_charges", company_number="11111111")
        assert r.success
        assert "HIGH STRESS" in r.output
        assert r.data["red_flag_count"] == 5

    @patch("agent.tools.creditor_filings._get_ch_key", return_value="key")
    @patch("agent.tools.creditor_filings._ch_client")
    def test_moderate_stress(self, mock_ch_client, mock_key, tool):
        charges = [_ch_charge(status="outstanding")]
        mock_ch = MagicMock()
        mock_ch_client.return_value.__enter__ = MagicMock(return_value=mock_ch)
        mock_ch_client.return_value.__exit__ = MagicMock(return_value=False)
        mock_ch.get.return_value = _mock_response(_ch_charges_response(charges))

        r = tool.execute(mode="uk_charges", company_number="22222222")
        assert r.success
        assert "MODERATE" in r.output

    @patch("agent.tools.creditor_filings._get_ch_key", return_value="key")
    def test_uk_charges_cache_hit(self, mock_key, tool):
        tool._cache.get.return_value = {
            "charges": [
                {
                    "status": "satisfied",
                    "charge_number": 1,
                    "created_on": "2020-01-01",
                    "satisfied_on": "2021-01-01",
                    "classification": "mortgage",
                    "persons_entitled": ["Bank"],
                    "particulars": "prop",
                }
            ],
            "company_info": {"company_name": "CachedUK", "company_number": "99"},
        }
        r = tool.execute(mode="uk_charges", company_number="99")
        assert r.success
        assert "(cached)" in r.output

    @patch("agent.tools.creditor_filings._get_ch_key", return_value="key")
    @patch("agent.tools.creditor_filings._ch_client")
    def test_ch_api_exception(self, mock_ch_client, mock_key, tool):
        mock_ch_client.return_value.__enter__ = MagicMock(side_effect=httpx.ConnectError("timeout"))
        mock_ch_client.return_value.__exit__ = MagicMock(return_value=False)

        r = tool.execute(mode="uk_charges", company_number="12345678")
        assert not r.success
        assert "error" in r.output.lower() or "Error" in r.output


# ═══════════════════════════════════════════════════════════════════════════
# 4. Stress scan mode
# ═══════════════════════════════════════════════════════════════════════════


class TestStressScanMode:
    @patch("agent.tools.creditor_filings.httpx.Client")
    def test_basic_scan(self, mock_client_cls, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        hits = [_efts_record(company="StressCo", file_date="2025-06-10")]
        mock_client.get.return_value = _mock_response(_efts_response(hits, total=1))

        r = tool.execute(mode="stress_scan")
        assert r.success
        assert r.data["mode"] == "stress_scan"
        assert r.data["sec_count"] == 1

    @patch("agent.tools.creditor_filings.httpx.Client")
    def test_scan_with_clusters(self, mock_client_cls, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        hits = [
            _efts_record(company="ClusterCo", file_date="2025-06-10"),
            _efts_record(company="ClusterCo", file_date="2025-06-12"),
            _efts_record(company="ClusterCo", file_date="2025-06-14"),
            _efts_record(company="Other", file_date="2025-06-11"),
        ]
        mock_client.get.return_value = _mock_response(_efts_response(hits, total=4))

        r = tool.execute(mode="stress_scan", days_back=30)
        assert r.success
        assert r.data["cluster_count"] >= 1
        # ClusterCo should be flagged
        cluster_names = [c["entity"] for c in r.data["clusters"]]
        assert "ClusterCo" in cluster_names

    @patch("agent.tools.creditor_filings.httpx.Client")
    def test_scan_empty(self, mock_client_cls, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(_efts_response([], total=0))

        r = tool.execute(mode="stress_scan")
        assert r.success
        assert r.data["cluster_count"] == 0

    @patch("agent.tools.creditor_filings.httpx.Client")
    def test_scan_custom_days(self, mock_client_cls, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(_efts_response([], total=0))

        r = tool.execute(mode="stress_scan", days_back=7)
        assert r.success
        assert "last 7d" in r.output

    def test_stress_scan_cache_hit(self, tool):
        tool._cache.get.return_value = {
            "entries": [],
            "total": 0,
            "clusters": [],
        }
        r = tool.execute(mode="stress_scan")
        assert r.success
        assert "(cached)" in r.output


# ═══════════════════════════════════════════════════════════════════════════
# 5. Parameter edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestParameterEdgeCases:
    @patch("agent.tools.creditor_filings._get_ch_key", return_value=None)
    @patch("agent.tools.creditor_filings.httpx.Client")
    def test_limit_zero_clamped(self, mock_client_cls, mock_ch_key, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(_efts_response([], total=0))

        r = tool.execute(mode="search", query="Test", limit=0)
        assert r.success  # Clamped to 1

    @patch("agent.tools.creditor_filings._get_ch_key", return_value=None)
    @patch("agent.tools.creditor_filings.httpx.Client")
    def test_limit_over_max_clamped(self, mock_client_cls, mock_ch_key, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(_efts_response([], total=0))

        r = tool.execute(mode="search", query="Test", limit=500)
        assert r.success

    @patch("agent.tools.creditor_filings.httpx.Client")
    def test_days_back_negative_clamped(self, mock_client_cls, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(_efts_response([], total=0))

        r = tool.execute(mode="stress_scan", days_back=-5)
        assert r.success  # Clamped to 1

    @patch("agent.tools.creditor_filings._get_ch_key", return_value=None)
    @patch("agent.tools.creditor_filings.httpx.Client")
    def test_default_limit(self, mock_client_cls, mock_ch_key, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(_efts_response([], total=0))

        # No limit provided — should use default
        r = tool.execute(mode="search", query="Test")
        assert r.success


# ═══════════════════════════════════════════════════════════════════════════
# 6. Helper function unit tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFetchJson:
    def test_success(self):
        client = MagicMock()
        client.get.return_value = _mock_response({"key": "value"})
        result = _fetch_json("https://example.com", client)
        assert result == {"key": "value"}

    def test_http_error(self):
        client = MagicMock()
        client.get.return_value = _mock_response({}, status=404)
        result = _fetch_json("https://example.com", client)
        assert result is None

    def test_network_error(self):
        client = MagicMock()
        client.get.side_effect = httpx.ConnectError("fail")
        result = _fetch_json("https://example.com", client)
        assert result is None

    def test_invalid_json(self):
        client = MagicMock()
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.side_effect = ValueError("bad json")
        client.get.return_value = resp
        result = _fetch_json("https://example.com", client)
        assert result is None

    def test_timeout(self):
        client = MagicMock()
        client.get.side_effect = httpx.ReadTimeout("timeout")
        result = _fetch_json("https://example.com", client)
        assert result is None


class TestParseEftsHits:
    def test_empty(self):
        assert _parse_efts_hits({}) == []
        assert _parse_efts_hits({"hits": {}}) == []
        assert _parse_efts_hits({"hits": {"hits": []}}) == []

    def test_single_hit(self):
        data = _efts_response([_efts_record()])
        entries = _parse_efts_hits(data)
        assert len(entries) == 1
        assert entries[0]["company_name"] == "Acme Corp"
        assert entries[0]["cik"] == "0001234567"

    def test_missing_display_names(self):
        hit = {"_source": {"ciks": ["111"], "file_date": "2025-01-01"}}
        entries = _parse_efts_hits({"hits": {"hits": [hit]}})
        assert entries[0]["company_name"] == "Unknown"

    def test_missing_ciks(self):
        hit = {"_source": {"display_names": ["Corp"], "file_date": "2025-01-01"}}
        entries = _parse_efts_hits({"hits": {"hits": [hit]}})
        assert entries[0]["cik"] == ""

    def test_empty_source(self):
        hit = {"_source": {}}
        entries = _parse_efts_hits({"hits": {"hits": [hit]}})
        assert len(entries) == 1
        assert entries[0]["company_name"] == "Unknown"

    def test_no_source_key(self):
        hit = {}
        entries = _parse_efts_hits({"hits": {"hits": [hit]}})
        assert len(entries) == 1


class TestSearchEfts:
    def test_success(self):
        client = MagicMock()
        client.get.return_value = _mock_response(_efts_response([_efts_record()], total=1))
        entries, total = _search_efts(client, query="test", days_back=30, limit=20)
        assert len(entries) == 1
        assert total == 1

    def test_none_response(self):
        client = MagicMock()
        client.get.return_value = _mock_response({}, status=500)
        entries, total = _search_efts(client, query="test", days_back=30, limit=20)
        assert entries == []
        assert total == 0

    def test_limit_applied(self):
        client = MagicMock()
        hits = [_efts_record(company=f"C{i}") for i in range(10)]
        client.get.return_value = _mock_response(_efts_response(hits, total=100))
        entries, total = _search_efts(client, query="test", days_back=30, limit=5)
        assert len(entries) == 5
        assert total == 100


class TestSearchChCompany:
    def test_success(self):
        client = MagicMock()
        client.get.return_value = _mock_response(_ch_search_response([("TestCo", "12345678")]))
        results = _search_ch_company("TestCo", client)
        assert len(results) == 1
        assert results[0]["company_name"] == "TestCo"

    def test_empty(self):
        client = MagicMock()
        client.get.return_value = _mock_response(_ch_search_response([]))
        results = _search_ch_company("nothing", client)
        assert results == []

    def test_none_response(self):
        client = MagicMock()
        client.get.return_value = _mock_response({}, status=500)
        results = _search_ch_company("test", client)
        assert results == []


class TestGetChCharges:
    def test_success(self):
        client = MagicMock()
        client.get.return_value = _mock_response(_ch_charges_response([_ch_charge()]))
        charges = _get_ch_charges("12345678", client)
        assert len(charges) == 1
        assert charges[0]["status"] == "satisfied"

    def test_empty(self):
        client = MagicMock()
        client.get.return_value = _mock_response(_ch_charges_response([]))
        charges = _get_ch_charges("12345678", client)
        assert charges == []

    def test_none_response(self):
        client = MagicMock()
        client.get.return_value = _mock_response({}, status=500)
        charges = _get_ch_charges("12345678", client)
        assert charges == []

    def test_special_chars_in_company_number(self):
        """Company number with special chars should be URL-encoded."""
        client = MagicMock()
        client.get.return_value = _mock_response(_ch_charges_response([]))
        _get_ch_charges("SC/123 456", client)
        # Should not raise
        assert client.get.called


class TestClassifyCharge:
    def test_debenture(self):
        assert _classify_charge({"particulars": {"description": "A debenture"}}) == "debenture"

    def test_floating(self):
        assert (
            _classify_charge({"particulars": {"description": "floating charge over all assets"}}) == "floating_charge"
        )

    def test_mortgage(self):
        assert _classify_charge({"particulars": {"description": "Legal mortgage"}}) == "mortgage"

    def test_fixed(self):
        assert _classify_charge({"particulars": {"description": "Fixed charge"}}) == "fixed_charge"

    def test_other(self):
        assert _classify_charge({"particulars": {"description": "something else"}}) == "other"

    def test_empty(self):
        assert _classify_charge({}) == "other"

    def test_no_description(self):
        assert _classify_charge({"particulars": {}}) == "other"


class TestExtractParticulars:
    def test_normal(self):
        item = {"particulars": {"description": "Some description"}}
        assert _extract_particulars(item) == "Some description"

    def test_truncation(self):
        long_desc = "A" * 300
        item = {"particulars": {"description": long_desc}}
        assert len(_extract_particulars(item)) == 200

    def test_empty(self):
        assert _extract_particulars({}) == ""
        assert _extract_particulars({"particulars": {}}) == ""

    def test_non_dict_particulars(self):
        item = {"particulars": "raw string"}
        result = _extract_particulars(item)
        assert isinstance(result, str)


class TestDetectFilingClusters:
    def test_no_clusters(self):
        entries = [
            {"company_name": "A", "file_date": "2025-01-01", "cik": "1"},
            {"company_name": "B", "file_date": "2025-01-02", "cik": "2"},
        ]
        clusters = _detect_filing_clusters(entries)
        assert clusters == []

    def test_single_cluster(self):
        entries = [
            {"company_name": "A", "file_date": "2025-01-01", "cik": "1"},
            {"company_name": "A", "file_date": "2025-01-05", "cik": "1"},
            {"company_name": "B", "file_date": "2025-01-03", "cik": "2"},
        ]
        clusters = _detect_filing_clusters(entries)
        assert len(clusters) == 1
        assert clusters[0]["entity"] == "A"
        assert clusters[0]["filing_count"] == 2

    def test_multiple_clusters(self):
        entries = [
            {"company_name": "X", "file_date": "2025-01-01", "cik": "1"},
            {"company_name": "X", "file_date": "2025-01-02", "cik": "1"},
            {"company_name": "Y", "file_date": "2025-01-03", "cik": "2"},
            {"company_name": "Y", "file_date": "2025-01-04", "cik": "2"},
            {"company_name": "Y", "file_date": "2025-01-05", "cik": "2"},
        ]
        clusters = _detect_filing_clusters(entries)
        assert len(clusters) == 2
        # Sorted by filing_count desc
        assert clusters[0]["entity"] == "Y"
        assert clusters[0]["filing_count"] == 3

    def test_empty_entries(self):
        assert _detect_filing_clusters([]) == []

    def test_missing_company_name(self):
        entries = [
            {"file_date": "2025-01-01", "cik": "1"},
            {"file_date": "2025-01-02", "cik": "1"},
        ]
        clusters = _detect_filing_clusters(entries)
        assert len(clusters) == 1
        assert clusters[0]["entity"] == "Unknown"


class TestCountRedFlagCharges:
    def test_all_satisfied(self):
        charges = [{"status": "satisfied"}, {"status": "satisfied"}]
        assert _count_red_flag_charges(charges) == 0

    def test_outstanding(self):
        charges = [{"status": "outstanding"}, {"status": "satisfied"}]
        assert _count_red_flag_charges(charges) == 1

    def test_part_satisfied(self):
        charges = [{"status": "part-satisfied"}]
        assert _count_red_flag_charges(charges) == 1

    def test_all_red(self):
        charges = [
            {"status": "outstanding"},
            {"status": "part-satisfied"},
            {"status": "outstanding"},
        ]
        assert _count_red_flag_charges(charges) == 3

    def test_empty(self):
        assert _count_red_flag_charges([]) == 0

    def test_missing_status(self):
        assert _count_red_flag_charges([{}]) == 0


class TestGetChKey:
    @patch.dict(os.environ, {"TIRRA_COMPANIES_HOUSE_KEY": "mykey"})
    def test_key_present(self):
        assert _get_ch_key() == "mykey"

    @patch.dict(os.environ, {}, clear=True)
    def test_key_missing(self):
        # Remove key if present
        os.environ.pop("TIRRA_COMPANIES_HOUSE_KEY", None)
        assert _get_ch_key() is None

    @patch.dict(os.environ, {"TIRRA_COMPANIES_HOUSE_KEY": "  "})
    def test_key_whitespace(self):
        assert _get_ch_key() is None

    @patch.dict(os.environ, {"TIRRA_COMPANIES_HOUSE_KEY": ""})
    def test_key_empty(self):
        assert _get_ch_key() is None


# ═══════════════════════════════════════════════════════════════════════════
# 7. No-cache tool
# ═══════════════════════════════════════════════════════════════════════════


class TestNoCacheTool:
    @patch("agent.tools.creditor_filings._get_ch_key", return_value=None)
    @patch("agent.tools.creditor_filings.httpx.Client")
    def test_search_no_cache(self, mock_client_cls, mock_ch_key, tool_no_cache):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(_efts_response([], total=0))

        r = tool_no_cache.execute(mode="search", query="Test")
        assert r.success

    @patch("agent.tools.creditor_filings.httpx.Client")
    def test_stress_scan_no_cache(self, mock_client_cls, tool_no_cache):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(_efts_response([], total=0))

        r = tool_no_cache.execute(mode="stress_scan")
        assert r.success


# ═══════════════════════════════════════════════════════════════════════════
# 8. Tool metadata
# ═══════════════════════════════════════════════════════════════════════════


class TestToolMetadata:
    def test_name(self, tool):
        assert tool.name == "creditor_filings"

    def test_description_nonempty(self, tool):
        assert len(tool.description) > 20

    def test_parameters_schema(self, tool):
        p = tool.parameters
        assert p["type"] == "object"
        assert "mode" in p["properties"]
        assert "required" in p
        assert "mode" in p["required"]

    def test_mode_enum(self, tool):
        modes = tool.parameters["properties"]["mode"]["enum"]
        assert set(modes) == VALID_MODES


# ═══════════════════════════════════════════════════════════════════════════
# 9. Constants sanity
# ═══════════════════════════════════════════════════════════════════════════


class TestConstants:
    def test_credit_terms_nonempty(self):
        assert len(_CREDIT_TERMS) >= 4

    def test_credit_items(self):
        assert "1.01" in _CREDIT_ITEMS
        assert "2.03" in _CREDIT_ITEMS

    def test_red_flag_statuses(self):
        assert "outstanding" in _CHARGE_RED_FLAGS
        assert "part-satisfied" in _CHARGE_RED_FLAGS
        assert "satisfied" not in _CHARGE_RED_FLAGS

    def test_cache_ttl(self):
        assert _CACHE_TTL > 0

    def test_defaults(self):
        assert _DEFAULT_LIMIT == 20
        assert _MAX_LIMIT == 100


# ═══════════════════════════════════════════════════════════════════════════
# 10. Integration: tool + arm counts
# ═══════════════════════════════════════════════════════════════════════════


class TestIntegration:
    def test_tool_count(self):
        """Verify total registered tools = 38."""
        from agent.cli import build_tool_registry

        registry = build_tool_registry()
        names = registry.list_names()
        # Was 60; commit 43de067 (2026-08-26) fixed nightlight_activity's
        # constructor kwarg mismatch (store= vs pipeline_store=) that silently
        # skipped its registration -- registry now correctly has 61 tools.
        assert len(names) == 61, f"Expected 61 tools, got {len(names)}: {sorted(names)}"

    def test_arm_count(self):
        """Verify total bandit arms = 26."""
        from agent.learning.bandit import DEFAULT_ARMS

        assert len(DEFAULT_ARMS) == 48, f"Expected 48 arms, got {len(DEFAULT_ARMS)}"

    def test_creditor_filings_registered(self):
        from agent.cli import build_tool_registry

        registry = build_tool_registry()
        names = set(registry.list_names())
        assert "creditor_filings" in names

    def test_creditor_stress_arm_exists(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm_names = [a.name for a in DEFAULT_ARMS]
        assert "creditor_stress" in arm_names

    def test_creditor_stress_arm_tools(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm = next(a for a in DEFAULT_ARMS if a.name == "creditor_stress")
        assert "creditor_filings" in arm.tools
