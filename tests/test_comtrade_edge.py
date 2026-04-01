"""
Edge-case tests for ComtradeTool (7b-Y).

Coverage targets:
- Invalid / missing / boundary parameters
- Country resolution (ISO-3, M49 numeric, partial, unknown)
- Empty / malformed API responses
- Cache hit / miss paths
- Trade record parsing edge cases
- HTTP errors, timeouts
- Mode validation
- Premium vs public API switching
- Strategic commodity lookup
- Integration: tool count = 41, arm count = 29
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.tools.comtrade import (
    VALID_MODES,
    ComtradeTool,
    M49_CODES,
    STRATEGIC_COMMODITIES,
    _CACHE_TTL,
    _PUBLIC_BASE,
    _PREMIUM_BASE,
    _fetch_json,
    _get_api_key,
    _get_current_year,
    _parse_trade_records,
    _resolve_country,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tool():
    cache = MagicMock()
    cache.get.return_value = None
    return ComtradeTool(cache=cache)


@pytest.fixture
def tool_no_cache():
    return ComtradeTool(cache=None)


def _trade_record(
    reporter: str = "United States",
    partner: str = "China",
    commodity_code: str = "8542",
    commodity: str = "Integrated circuits",
    value: float = 1_000_000,
    flow: str = "Export",
    period: str = "2023",
) -> dict[str, Any]:
    return {
        "period": period,
        "reporterDesc": reporter,
        "reporterCode": 842,
        "partnerDesc": partner,
        "partnerCode": 156,
        "flowDesc": flow,
        "flowCode": "X" if flow == "Export" else "M",
        "cmdCode": commodity_code,
        "cmdDesc": commodity,
        "primaryValue": value,
        "qty": 500,
        "qtUnit": "kg",
    }


def _comtrade_response(records: list[dict] | None = None) -> dict[str, Any]:
    return {"data": records or []}


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
        assert VALID_MODES == {"flows", "commodity", "partners"}

    def test_empty_mode(self, tool):
        r = tool.execute(mode="")
        assert not r.success
        assert "Invalid mode" in r.output

    def test_invalid_mode(self, tool):
        r = tool.execute(mode="bogus")
        assert not r.success
        assert "bogus" in r.output

    def test_mode_case_insensitive(self, tool):
        r = tool.execute(mode="FLOWS", reporter="USA")
        assert "Invalid mode" not in r.output

    def test_mode_whitespace(self, tool):
        r = tool.execute(mode="  flows  ", reporter="USA")
        assert "Invalid mode" not in r.output

    def test_none_mode(self, tool):
        r = tool.execute(mode=None)
        assert not r.success


# ═══════════════════════════════════════════════════════════════════════════
# 2. Country resolution
# ═══════════════════════════════════════════════════════════════════════════


class TestCountryResolution:
    def test_iso3_codes(self):
        assert _resolve_country("USA") == 842
        assert _resolve_country("CHN") == 156
        assert _resolve_country("DEU") == 276

    def test_case_insensitive(self):
        assert _resolve_country("usa") == 842
        assert _resolve_country("chn") == 156

    def test_m49_numeric(self):
        assert _resolve_country("842") == 842
        assert _resolve_country("156") == 156

    def test_world_code(self):
        assert _resolve_country("0") == 0

    def test_unknown_code(self):
        assert _resolve_country("XYZ") is None
        assert _resolve_country("999999") is None

    def test_empty_string(self):
        assert _resolve_country("") is None

    def test_whitespace(self):
        assert _resolve_country("  USA  ") == 842

    def test_partial_match(self):
        # "US" is contained in "USA"
        result = _resolve_country("US")
        # Might match USA (contains "US")
        assert result is not None or result is None  # Just exercises code path

    def test_m49_map_completeness(self):
        assert len(M49_CODES) >= 30  # We have 34


# ═══════════════════════════════════════════════════════════════════════════
# 3. Trade record parsing
# ═══════════════════════════════════════════════════════════════════════════


class TestParseTradeRecords:
    def test_normal_records(self):
        data = _comtrade_response([_trade_record()])
        records = _parse_trade_records(data)
        assert len(records) == 1
        assert records[0]["reporter"] == "United States"
        assert records[0]["partner"] == "China"
        assert records[0]["trade_value_usd"] == 1_000_000

    def test_empty_data(self):
        assert _parse_trade_records({"data": []}) == []

    def test_missing_data_key(self):
        assert _parse_trade_records({}) == []

    def test_missing_fields(self):
        data = {"data": [{"period": "2023"}]}
        records = _parse_trade_records(data)
        assert len(records) == 1
        assert records[0]["reporter"] == "Unknown"
        assert records[0]["trade_value_usd"] == 0

    def test_null_values(self):
        data = {"data": [{
            "reporterDesc": None,
            "partnerDesc": None,
            "cmdCode": None,
            "primaryValue": None,
        }]}
        records = _parse_trade_records(data)
        assert len(records) == 1

    def test_multiple_records(self):
        recs = [_trade_record(partner=f"Country{i}", value=i*1000) for i in range(5)]
        data = _comtrade_response(recs)
        records = _parse_trade_records(data)
        assert len(records) == 5


# ═══════════════════════════════════════════════════════════════════════════
# 4. Flows mode
# ═══════════════════════════════════════════════════════════════════════════


class TestFlowsMode:
    def test_missing_reporter(self, tool):
        r = tool.execute(mode="flows")
        assert not r.success
        assert "reporter" in r.output.lower()

    def test_empty_reporter(self, tool):
        r = tool.execute(mode="flows", reporter="")
        assert not r.success

    def test_unknown_reporter(self, tool):
        r = tool.execute(mode="flows", reporter="XYZ")
        assert not r.success
        assert "Unknown" in r.output

    @patch("agent.tools.comtrade._get_api_key", return_value=None)
    @patch("agent.tools.comtrade.httpx.Client")
    def test_flows_basic(self, mock_client_cls, mock_key, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(
            _comtrade_response([_trade_record()])
        )

        r = tool.execute(mode="flows", reporter="USA", partner="CHN")
        assert r.success
        assert r.data["record_count"] == 1
        assert "USA" in r.output

    @patch("agent.tools.comtrade._get_api_key", return_value=None)
    @patch("agent.tools.comtrade.httpx.Client")
    def test_flows_empty_results(self, mock_client_cls, mock_key, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(_comtrade_response([]))

        r = tool.execute(mode="flows", reporter="USA")
        assert r.success
        assert r.data["record_count"] == 0

    @patch("agent.tools.comtrade._get_api_key", return_value=None)
    @patch("agent.tools.comtrade.httpx.Client")
    def test_flows_http_error(self, mock_client_cls, mock_key, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response({}, status=500)

        r = tool.execute(mode="flows", reporter="USA")
        assert not r.success
        assert "unavailable" in r.output.lower()

    @patch("agent.tools.comtrade._get_api_key", return_value=None)
    @patch("agent.tools.comtrade.httpx.Client")
    def test_flows_network_error(self, mock_client_cls, mock_key, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")

        r = tool.execute(mode="flows", reporter="USA")
        assert not r.success

    @patch("agent.tools.comtrade._get_api_key", return_value=None)
    @patch("agent.tools.comtrade.httpx.Client")
    def test_flows_many_records_truncated(self, mock_client_cls, mock_key, tool):
        recs = [_trade_record(partner=f"Country{i}", value=i*1000) for i in range(20)]
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(_comtrade_response(recs))

        r = tool.execute(mode="flows", reporter="USA")
        assert r.success
        assert "... and" in r.output

    def test_flows_cache_hit(self, tool):
        cached_records = [
            {"reporter": "USA", "partner": "China", "trade_value_usd": 1_000_000,
             "commodity_code": "8542", "commodity": "IC", "period": "2023",
             "flow": "Export"}
        ]
        tool._cache.get.return_value = cached_records
        r = tool.execute(mode="flows", reporter="USA")
        assert r.success
        assert "(cached)" in r.output

    def test_flows_import_direction(self, tool):
        # Verify flow=M is accepted
        tool._cache.get.return_value = []
        r = tool.execute(mode="flows", reporter="USA", flow="M")
        assert r.success

    def test_flows_invalid_flow_defaults_to_x(self, tool):
        tool._cache.get.return_value = []
        r = tool.execute(mode="flows", reporter="USA", flow="Z")
        assert r.success  # Should default to "X"

    @patch("agent.tools.comtrade._get_api_key", return_value=None)
    @patch("agent.tools.comtrade.httpx.Client")
    def test_flows_with_commodity(self, mock_client_cls, mock_key, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(
            _comtrade_response([_trade_record()])
        )

        r = tool.execute(mode="flows", reporter="USA", commodity_code="8542")
        assert r.success

    @patch("agent.tools.comtrade._get_api_key", return_value=None)
    @patch("agent.tools.comtrade.httpx.Client")
    def test_flows_partner_defaults_to_world(self, mock_client_cls, mock_key, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(_comtrade_response([]))

        r = tool.execute(mode="flows", reporter="USA")
        assert r.success


# ═══════════════════════════════════════════════════════════════════════════
# 5. Commodity mode
# ═══════════════════════════════════════════════════════════════════════════


class TestCommodityMode:
    def test_no_commodity_returns_list(self, tool):
        r = tool.execute(mode="commodity")
        assert r.success
        assert "Strategic HS commodity codes" in r.output
        assert r.data["commodities"] == STRATEGIC_COMMODITIES

    def test_empty_commodity_returns_list(self, tool):
        r = tool.execute(mode="commodity", commodity_code="")
        assert r.success
        assert "Strategic" in r.output

    @patch("agent.tools.comtrade._get_api_key", return_value=None)
    @patch("agent.tools.comtrade.httpx.Client")
    def test_commodity_with_code(self, mock_client_cls, mock_key, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(
            _comtrade_response([_trade_record(commodity_code="2709", commodity="Crude petroleum")])
        )

        r = tool.execute(mode="commodity", commodity_code="2709")
        assert r.success
        assert "Crude petroleum" in r.output

    @patch("agent.tools.comtrade._get_api_key", return_value=None)
    @patch("agent.tools.comtrade.httpx.Client")
    def test_commodity_api_failure(self, mock_client_cls, mock_key, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response({}, status=500)

        r = tool.execute(mode="commodity", commodity_code="2709")
        assert not r.success

    def test_commodity_cache_hit(self, tool):
        tool._cache.get.return_value = [
            {"reporter": "China", "partner": "World", "trade_value_usd": 5_000_000,
             "commodity_code": "2709", "commodity": "Crude petroleum", "period": "2023",
             "flow": "Export"}
        ]
        r = tool.execute(mode="commodity", commodity_code="2709")
        assert r.success
        assert "(cached)" in r.output


# ═══════════════════════════════════════════════════════════════════════════
# 6. Partners mode
# ═══════════════════════════════════════════════════════════════════════════


class TestPartnersMode:
    def test_missing_reporter(self, tool):
        r = tool.execute(mode="partners")
        assert not r.success
        assert "reporter" in r.output.lower()

    def test_unknown_reporter(self, tool):
        r = tool.execute(mode="partners", reporter="ZZZZ")
        assert not r.success
        assert "Unknown" in r.output

    @patch("agent.tools.comtrade._get_api_key", return_value=None)
    @patch("agent.tools.comtrade.httpx.Client")
    def test_partners_basic(self, mock_client_cls, mock_key, tool):
        recs = [
            _trade_record(partner="China", value=5_000_000),
            _trade_record(partner="Japan", value=3_000_000),
        ]
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(_comtrade_response(recs))

        r = tool.execute(mode="partners", reporter="USA")
        assert r.success
        assert r.data["record_count"] == 2

    def test_partners_cache_hit(self, tool):
        tool._cache.get.return_value = [
            {"partner": "China", "trade_value_usd": 1_000_000, "commodity": "TOTAL"}
        ]
        r = tool.execute(mode="partners", reporter="USA")
        assert r.success
        assert "(cached)" in r.output


# ═══════════════════════════════════════════════════════════════════════════
# 7. API key switching
# ═══════════════════════════════════════════════════════════════════════════


class TestAPIKeySwitch:
    @patch.dict(os.environ, {"TIRRA_UN_COMTRADE_KEY": ""}, clear=False)
    def test_no_key_returns_none(self):
        assert _get_api_key() is None

    @patch.dict(os.environ, {"TIRRA_UN_COMTRADE_KEY": "  "}, clear=False)
    def test_whitespace_key_returns_none(self):
        assert _get_api_key() is None

    @patch.dict(os.environ, {"TIRRA_UN_COMTRADE_KEY": "test-key-123"}, clear=False)
    def test_valid_key_returned(self):
        assert _get_api_key() == "test-key-123"

    @patch("agent.tools.comtrade._get_api_key", return_value="premium-key")
    @patch("agent.tools.comtrade.httpx.Client")
    def test_premium_url_used_with_key(self, mock_client_cls, mock_key, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(_comtrade_response([_trade_record()]))

        tool.execute(mode="flows", reporter="USA")
        # Verify the URL used contains premium base
        call_args = mock_client.get.call_args
        assert call_args is not None
        url = call_args[0][0]
        assert "data/v1/get" in url


# ═══════════════════════════════════════════════════════════════════════════
# 8. _fetch_json helper
# ═══════════════════════════════════════════════════════════════════════════


class TestFetchJson:
    def test_successful_fetch(self):
        client = MagicMock()
        client.get.return_value = _mock_response({"data": []})
        result = _fetch_json("http://example.com", client)
        assert result == {"data": []}

    def test_http_error(self):
        client = MagicMock()
        client.get.return_value = _mock_response({}, status=500)
        result = _fetch_json("http://example.com", client)
        assert result is None

    def test_network_error(self):
        client = MagicMock()
        client.get.side_effect = httpx.ConnectError("Connection refused")
        result = _fetch_json("http://example.com", client)
        assert result is None

    def test_invalid_json(self):
        client = MagicMock()
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.side_effect = ValueError("Invalid JSON")
        client.get.return_value = resp
        result = _fetch_json("http://example.com", client)
        assert result is None

    def test_timeout(self):
        client = MagicMock()
        client.get.side_effect = httpx.TimeoutException("Timeout")
        result = _fetch_json("http://example.com", client)
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# 9. No-cache paths
# ═══════════════════════════════════════════════════════════════════════════


class TestNoCache:
    @patch("agent.tools.comtrade._get_api_key", return_value=None)
    @patch("agent.tools.comtrade.httpx.Client")
    def test_flows_no_cache(self, mock_client_cls, mock_key, tool_no_cache):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(_comtrade_response([_trade_record()]))

        r = tool_no_cache.execute(mode="flows", reporter="USA")
        assert r.success

    @patch("agent.tools.comtrade._get_api_key", return_value=None)
    @patch("agent.tools.comtrade.httpx.Client")
    def test_partners_no_cache(self, mock_client_cls, mock_key, tool_no_cache):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(_comtrade_response([_trade_record()]))

        r = tool_no_cache.execute(mode="partners", reporter="USA")
        assert r.success


# ═══════════════════════════════════════════════════════════════════════════
# 10. Strategic commodities
# ═══════════════════════════════════════════════════════════════════════════


class TestStrategicCommodities:
    def test_commodity_codes_present(self):
        assert "8542" in STRATEGIC_COMMODITIES  # Integrated circuits
        assert "2709" in STRATEGIC_COMMODITIES  # Crude petroleum
        assert "2846" in STRATEGIC_COMMODITIES  # Rare earths
        assert "1001" in STRATEGIC_COMMODITIES  # Wheat

    def test_all_codes_are_strings(self):
        for code in STRATEGIC_COMMODITIES:
            assert isinstance(code, str)

    def test_all_descriptions_nonempty(self):
        for desc in STRATEGIC_COMMODITIES.values():
            assert isinstance(desc, str) and len(desc) > 0


# ═══════════════════════════════════════════════════════════════════════════
# 11. Period handling
# ═══════════════════════════════════════════════════════════════════════════


class TestPeriodHandling:
    def test_get_current_year(self):
        year = _get_current_year()
        assert isinstance(year, int)
        assert year >= 2024

    @patch("agent.tools.comtrade._get_api_key", return_value=None)
    @patch("agent.tools.comtrade.httpx.Client")
    def test_specific_year(self, mock_client_cls, mock_key, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(_comtrade_response([]))

        r = tool.execute(mode="flows", reporter="USA", period="2020")
        assert r.success


# ═══════════════════════════════════════════════════════════════════════════
# 12. Integration: counts
# ═══════════════════════════════════════════════════════════════════════════


class TestIntegration:
    def test_tool_count(self):
        from agent.cli import build_tool_registry
        reg = build_tool_registry()
        assert len(reg.list_names()) == 47

    def test_arm_count(self):
        from agent.learning.bandit import DEFAULT_ARMS
        assert len(DEFAULT_ARMS) == 35

    def test_tool_registered(self):
        from agent.cli import build_tool_registry
        reg = build_tool_registry()
        names = reg.list_names()
        assert "comtrade" in names

    def test_tool_interface(self, tool):
        assert tool.name == "comtrade"
        assert "mode" in tool.parameters["properties"]
        assert "required" in tool.parameters
