"""
Edge case tests for EnergySupplyTool (EIA API v2).

Covers: mode validation, series validation, EIA API fetch, response parsing,
_safe_float, signal computation (petroleum stocks, rig counts, alerts for
surprise/tightening/building, 3-month rig change), cache interaction,
HTTP errors (429/500/timeout/404), empty data, malformed responses, API key
handling, output formatting, tool metadata, registry + bandit integration.
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import MagicMock, patch

import httpx

from agent.tools.energy_supply import (
    STOCK_SERIES,
    VALID_MODES,
    EnergySupplyTool,
    _compute_rig_signals,
    _compute_stock_signals,
    _fetch_eia,
    _format_petroleum_summary,
    _format_rig_summary,
    _get_api_key,
    _parse_eia_records,
    _safe_float,
)

# ── Fixtures ──────────────────────────────────────────────────


def _tool(cache=None) -> EnergySupplyTool:
    return EnergySupplyTool(cache=cache)


def _mock_resp(body: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        text=json.dumps(body),
        request=httpx.Request("GET", "http://test"),
    )


def _eia_response(data: list, total: int | None = None) -> dict:
    return {
        "response": {
            "total": str(total or len(data)),
            "data": data,
        }
    }


def _stock_record(
    period: str = "2024-06-28",
    series: str = "WCESTUS1",
    value: Any = 455000,
    units: str = "thousand barrels",
    area_name: str = "U.S.",
    product_name: str = "Crude Oil",
    process_name: str = "Ending Stocks Excluding SPR",
    series_desc: str = "Weekly U.S. Ending Stocks excluding SPR of Crude Oil",
) -> dict:
    return {
        "period": period,
        "series": series,
        "value": value,
        "units": units,
        "area-name": area_name,
        "product-name": product_name,
        "process-name": process_name,
        "series-description": series_desc,
    }


def _rig_record(
    period: str = "2024-06",
    value: Any = 583,
    units: str = "rigs",
) -> dict:
    return {
        "period": period,
        "series": "E_ERTRR0_XR0_NUS_M",
        "value": value,
        "units": units,
        "area-name": "U.S.",
        "product-name": "",
        "process-name": "Rotary Rigs in Operation",
        "series-description": "Monthly U.S. Rotary Rigs in Operation",
    }


# ── Tool metadata ────────────────────────────────────────────


class TestToolMetadata:
    def test_name(self):
        assert _tool().name == "energy_supply"

    def test_description(self):
        desc = _tool().description
        assert "EIA" in desc or "petroleum" in desc.lower() or "energy" in desc.lower()
        assert len(desc) > 50

    def test_parameters_schema(self):
        params = _tool().parameters
        assert params["type"] == "object"
        assert "mode" in params["properties"]
        assert params["required"] == ["mode"]

    def test_valid_modes_match_schema(self):
        enum = _tool().parameters["properties"]["mode"]["enum"]
        assert set(enum) == VALID_MODES

    def test_has_series_and_weeks_params(self):
        props = _tool().parameters["properties"]
        assert "series" in props
        assert "weeks" in props

    def test_constants(self):
        assert "crude_excl_spr" in STOCK_SERIES
        assert "gasoline_total" in STOCK_SERIES
        assert "distillate" in STOCK_SERIES
        assert "spr" in STOCK_SERIES


# ── Mode validation ──────────────────────────────────────────


class TestModeValidation:
    def test_invalid_mode(self):
        result = _tool().execute(mode="bogus")
        assert not result.success
        assert "Invalid mode" in result.output

    def test_empty_mode(self):
        result = _tool().execute(mode="")
        assert not result.success

    def test_missing_mode(self):
        result = _tool().execute()
        assert not result.success

    def test_case_sensitive_mode(self):
        result = _tool().execute(mode="Petroleum_Stocks")
        assert not result.success


# ── Series validation ────────────────────────────────────────


class TestSeriesValidation:
    @patch("agent.tools.energy_supply._fetch_eia")
    def test_invalid_stock_series(self, mock_fetch):
        result = _tool().execute(mode="petroleum_stocks", series="bogus")
        assert not result.success
        assert "Invalid series" in result.output

    @patch("agent.tools.energy_supply._fetch_eia")
    def test_invalid_supply_series(self, mock_fetch):
        result = _tool().execute(mode="petroleum_supply", series="bogus")
        assert not result.success
        assert "Invalid series" in result.output

    @patch("agent.tools.energy_supply._fetch_eia")
    def test_valid_stock_series(self, mock_fetch):
        mock_fetch.return_value = ([_stock_record()], None)
        result = _tool().execute(mode="petroleum_stocks", series="crude_excl_spr")
        assert result.success

    @patch("agent.tools.energy_supply._fetch_eia")
    def test_empty_series_gets_all(self, mock_fetch):
        mock_fetch.return_value = ([_stock_record()], None)
        result = _tool().execute(mode="petroleum_stocks", series="")
        assert result.success
        # Should call fetch for all 4 series
        assert mock_fetch.call_count == 4


# ── API key handling ─────────────────────────────────────────


class TestAPIKey:
    def test_default_demo_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("TIRRA_EIA_API_KEY", None)
            assert _get_api_key() == "DEMO_KEY"

    def test_custom_key(self):
        with patch.dict(os.environ, {"TIRRA_EIA_API_KEY": "MY_KEY"}):
            assert _get_api_key() == "MY_KEY"


# ── _safe_float ──────────────────────────────────────────────


class TestSafeFloat:
    def test_normal(self):
        assert _safe_float("455000") == 455000.0

    def test_decimal(self):
        assert _safe_float("455.5") == 455.5

    def test_none(self):
        assert _safe_float(None) is None

    def test_empty(self):
        assert _safe_float("") is None

    def test_non_numeric(self):
        assert _safe_float("N/A") is None

    def test_zero(self):
        assert _safe_float("0") == 0.0

    def test_negative(self):
        assert _safe_float("-100.5") == -100.5


# ── _parse_eia_records ───────────────────────────────────────


class TestParseEIARecords:
    def test_basic(self):
        raw = [_stock_record(period="2024-06-28", value=455000)]
        records = _parse_eia_records(raw)
        assert len(records) == 1
        assert records[0]["value"] == 455000.0
        assert records[0]["period"] == "2024-06-28"

    def test_chronological_sort(self):
        raw = [
            _stock_record(period="2024-07-05", value=460000),
            _stock_record(period="2024-06-28", value=455000),
            _stock_record(period="2024-06-21", value=450000),
        ]
        records = _parse_eia_records(raw)
        assert records[0]["period"] == "2024-06-21"
        assert records[-1]["period"] == "2024-07-05"

    def test_skip_null_value(self):
        raw = [_stock_record(value=None)]
        records = _parse_eia_records(raw)
        assert records == []

    def test_skip_empty_value(self):
        raw = [_stock_record(value="")]
        records = _parse_eia_records(raw)
        assert records == []

    def test_skip_non_numeric(self):
        raw = [_stock_record(value="N/A")]
        records = _parse_eia_records(raw)
        assert records == []

    def test_empty_input(self):
        records = _parse_eia_records([])
        assert records == []

    def test_fields_extracted(self):
        raw = [_stock_record()]
        r = _parse_eia_records(raw)[0]
        expected = {"period", "area", "product", "process", "series", "series_description", "value", "units"}
        assert expected == set(r.keys())


# ── Stock signal computation ─────────────────────────────────


class TestStockSignals:
    def test_empty(self):
        s = _compute_stock_signals([], "crude")
        assert s["status"] == "NO_DATA"

    def test_basic(self):
        records = [{"value": 450000.0, "units": "thousand barrels"}]
        s = _compute_stock_signals(records, "crude")
        assert s["latest_value"] == 450000.0
        assert s["total_weeks"] == 1

    def test_wow_change(self):
        records = [
            {"value": 450000.0, "units": "kb"},
            {"value": 455000.0, "units": "kb"},
        ]
        s = _compute_stock_signals(records, "crude")
        assert s["wow_change"] == 5000.0
        assert s["wow_pct"] is not None

    def test_single_record_no_wow(self):
        records = [{"value": 100.0, "units": ""}]
        s = _compute_stock_signals(records, "test")
        assert s["wow_change"] is None
        assert s["wow_pct"] is None

    def test_surprise_build(self):
        records = [
            {"value": 400000.0, "units": ""},
            {"value": 406000.0, "units": ""},  # +6000 > 5000 threshold
        ]
        s = _compute_stock_signals(records, "crude")
        assert "SURPRISE" in s["alert"]
        assert "build" in s["alert"]

    def test_surprise_draw(self):
        records = [
            {"value": 406000.0, "units": ""},
            {"value": 400000.0, "units": ""},  # -6000
        ]
        s = _compute_stock_signals(records, "crude")
        assert "SURPRISE" in s["alert"]
        assert "draw" in s["alert"]

    def test_consecutive_draws_alert(self):
        # 4 consecutive draws
        records = [
            {"value": 470000.0, "units": ""},
            {"value": 465000.0, "units": ""},
            {"value": 460000.0, "units": ""},
            {"value": 458000.0, "units": ""},
            {"value": 456000.0, "units": ""},
        ]
        s = _compute_stock_signals(records, "crude")
        assert s["direction"] == "draw"
        assert s["consecutive_weeks"] >= 3
        assert "TIGHTENING" in s["alert"]

    def test_consecutive_builds_alert(self):
        records = [
            {"value": 440000.0, "units": ""},
            {"value": 442000.0, "units": ""},
            {"value": 444000.0, "units": ""},
            {"value": 446000.0, "units": ""},
        ]
        s = _compute_stock_signals(records, "crude")
        assert s["direction"] == "build"
        assert s["consecutive_weeks"] >= 3
        assert "BUILDING" in s["alert"]

    def test_no_consecutive_no_direction_alert(self):
        # Alternating
        records = [
            {"value": 450000.0, "units": ""},
            {"value": 451000.0, "units": ""},
            {"value": 449000.0, "units": ""},
        ]
        s = _compute_stock_signals(records, "crude")
        assert s["consecutive_weeks"] <= 1

    def test_flat_no_direction(self):
        records = [
            {"value": 100.0, "units": ""},
            {"value": 100.0, "units": ""},
        ]
        s = _compute_stock_signals(records, "test")
        assert s["direction"] is None
        assert s["consecutive_weeks"] == 0

    def test_zero_prior_value_wow_pct(self):
        records = [
            {"value": 0.0, "units": ""},
            {"value": 100.0, "units": ""},
        ]
        s = _compute_stock_signals(records, "test")
        assert s["wow_pct"] is None


# ── Rig count signal computation ─────────────────────────────


class TestRigSignals:
    def test_empty(self):
        s = _compute_rig_signals([])
        assert s["status"] == "NO_DATA"

    def test_basic(self):
        records = [{"value": 583.0}]
        s = _compute_rig_signals(records)
        assert s["latest_value"] == 583.0
        assert s["total_months"] == 1

    def test_mom_change(self):
        records = [{"value": 580.0}, {"value": 583.0}]
        s = _compute_rig_signals(records)
        assert s["mom_change"] == 3.0
        assert s["mom_pct"] is not None

    def test_single_no_mom(self):
        records = [{"value": 583.0}]
        s = _compute_rig_signals(records)
        assert s["mom_change"] is None
        assert s["mom_pct"] is None

    def test_three_month_warning(self):
        # Down >10% over 3 months
        records = [{"value": 600.0}, {"value": 550.0}, {"value": 500.0}]
        s = _compute_rig_signals(records)
        assert s["three_month_change_pct"] is not None
        pct = s["three_month_change_pct"]
        # (500-600)/600 = -16.7%
        assert pct < -10
        assert "WARNING" in s["alert"]

    def test_three_month_up_notice(self):
        records = [{"value": 500.0}, {"value": 550.0}, {"value": 600.0}]
        s = _compute_rig_signals(records)
        # (600-500)/500 = +20%
        assert s["three_month_change_pct"] > 10
        assert "NOTICE" in s["alert"]

    def test_three_month_stable(self):
        records = [{"value": 580.0}, {"value": 582.0}, {"value": 584.0}]
        s = _compute_rig_signals(records)
        assert s["alert"] is None

    def test_trend_expanding(self):
        records = [
            {"value": 500.0},
            {"value": 510.0},
            {"value": 520.0},
            {"value": 570.0},
            {"value": 580.0},
            {"value": 590.0},
        ]
        s = _compute_rig_signals(records)
        assert s["trend"] == "EXPANDING"

    def test_trend_contracting(self):
        records = [
            {"value": 600.0},
            {"value": 590.0},
            {"value": 580.0},
            {"value": 520.0},
            {"value": 510.0},
            {"value": 500.0},
        ]
        s = _compute_rig_signals(records)
        assert s["trend"] == "CONTRACTING"

    def test_trend_stable(self):
        records = [{"value": 580.0}] * 6
        s = _compute_rig_signals(records)
        assert s["trend"] == "STABLE"

    def test_trend_insufficient(self):
        records = [{"value": 580.0}] * 3
        s = _compute_rig_signals(records)
        assert s["trend"] == "INSUFFICIENT_DATA"

    def test_zero_prior_no_mom_pct(self):
        records = [{"value": 0.0}, {"value": 100.0}]
        s = _compute_rig_signals(records)
        assert s["mom_pct"] is None

    def test_zero_3m_ago_no_pct(self):
        records = [{"value": 0.0}, {"value": 50.0}, {"value": 100.0}]
        s = _compute_rig_signals(records)
        assert s["three_month_change_pct"] is None


# ── EIA fetch (mocked HTTP) ─────────────────────────────────


class TestFetchEIA:
    @patch("agent.tools.energy_supply.httpx.Client")
    def test_success(self, mock_client_cls):
        body = _eia_response([_stock_record()])
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_resp(body)
        mock_client_cls.return_value = mock_client

        data, err = _fetch_eia("http://test", {"api_key": "DEMO"})
        assert err is None
        assert len(data) == 1

    @patch("agent.tools.energy_supply.httpx.Client")
    def test_timeout(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = httpx.TimeoutException("timeout")
        mock_client_cls.return_value = mock_client

        data, err = _fetch_eia("http://test", {})
        assert "timed out" in err

    @patch("agent.tools.energy_supply.httpx.Client")
    def test_rate_limit(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_resp({}, 429)
        mock_client_cls.return_value = mock_client

        data, err = _fetch_eia("http://test", {})
        assert "rate limit" in err.lower()

    @patch("agent.tools.energy_supply.httpx.Client")
    def test_404_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_resp({}, 404)
        mock_client_cls.return_value = mock_client

        data, err = _fetch_eia("http://test", {})
        assert "404" in err

    @patch("agent.tools.energy_supply.httpx.Client")
    def test_500_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_resp({}, 500)
        mock_client_cls.return_value = mock_client

        data, err = _fetch_eia("http://test", {})
        assert "500" in err

    @patch("agent.tools.energy_supply.httpx.Client")
    def test_malformed_json(self, mock_client_cls):
        resp = httpx.Response(
            status_code=200,
            text="<html>not json</html>",
            request=httpx.Request("GET", "http://test"),
        )
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = resp
        mock_client_cls.return_value = mock_client

        data, err = _fetch_eia("http://test", {})
        assert "parse" in err.lower()

    @patch("agent.tools.energy_supply.httpx.Client")
    def test_eia_error_in_body(self, mock_client_cls):
        body = {"error": "API key invalid"}
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_resp(body)
        mock_client_cls.return_value = mock_client

        data, err = _fetch_eia("http://test", {})
        assert "error" in err.lower()

    @patch("agent.tools.energy_supply.httpx.Client")
    def test_http_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = httpx.ConnectError("refused")
        mock_client_cls.return_value = mock_client

        data, err = _fetch_eia("http://test", {})
        assert err is not None
        assert "error" in err.lower()

    @patch("agent.tools.energy_supply.httpx.Client")
    def test_empty_data(self, mock_client_cls):
        body = _eia_response([])
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_resp(body)
        mock_client_cls.return_value = mock_client

        data, err = _fetch_eia("http://test", {})
        assert err is None
        assert data == []


# ── Formatting ───────────────────────────────────────────────


class TestFormatting:
    def test_petroleum_empty(self):
        text = _format_petroleum_summary(
            {"crude": []},
            {"crude": {"status": "NO_DATA"}},
            "petroleum_stocks",
            12,
        )
        assert "No data" in text

    def test_petroleum_with_data(self):
        records = _parse_eia_records([_stock_record()])
        signals = _compute_stock_signals(records, "crude")
        text = _format_petroleum_summary(
            {"crude": records},
            {"crude": signals},
            "petroleum_stocks",
            12,
        )
        assert "Petroleum Stocks" in text
        assert "Latest" in text

    def test_petroleum_supply_title(self):
        text = _format_petroleum_summary(
            {"crude": []},
            {"crude": {"status": "NO_DATA"}},
            "petroleum_supply",
            12,
        )
        assert "Petroleum Supply" in text

    def test_rig_empty(self):
        text = _format_rig_summary([], {"status": "NO_DATA"}, 12)
        assert "No rig count" in text

    def test_rig_with_data(self):
        records = _parse_eia_records([_rig_record()])
        signals = _compute_rig_signals(records)
        text = _format_rig_summary(records, signals, 12)
        assert "Rig Count" in text
        assert "rigs" in text.lower()


# ── Cache interaction ────────────────────────────────────────


class TestCache:
    @patch("agent.tools.energy_supply._fetch_eia")
    def test_rig_cache_hit(self, mock_fetch):
        cache = MagicMock()
        cache.get.return_value = {
            "output": "cached rig data",
            "data": {"records": [], "signals": {}},
        }
        tool = _tool(cache=cache)
        result = tool.execute(mode="rig_count")
        assert result.success
        assert "cached" in result.output
        mock_fetch.assert_not_called()

    @patch("agent.tools.energy_supply._fetch_eia")
    def test_rig_cache_miss(self, mock_fetch):
        mock_fetch.return_value = ([_rig_record()], None)
        cache = MagicMock()
        cache.get.return_value = None
        tool = _tool(cache=cache)
        result = tool.execute(mode="rig_count")
        assert result.success
        cache.put.assert_called_once()

    @patch("agent.tools.energy_supply._fetch_eia")
    def test_stocks_per_series_cache_hit(self, mock_fetch):
        # Single series with cache hit
        cache = MagicMock()
        cache.get.return_value = {
            "output": "",
            "data": {
                "records": [{"value": 100.0, "units": "kb", "series_description": "test", "period": "2024-01"}],
                "signals": {"latest_value": 100.0, "alert": None},
            },
        }
        tool = _tool(cache=cache)
        result = tool.execute(mode="petroleum_stocks", series="crude_excl_spr")
        assert result.success
        mock_fetch.assert_not_called()

    @patch("agent.tools.energy_supply._fetch_eia")
    def test_no_cache(self, mock_fetch):
        mock_fetch.return_value = ([_stock_record()], None)
        result = _tool(cache=None).execute(mode="petroleum_stocks", series="crude_excl_spr")
        assert result.success


# ── End-to-end with mocked fetch ─────────────────────────────


class TestEndToEnd:
    @patch("agent.tools.energy_supply._fetch_eia")
    def test_petroleum_stocks_single(self, mock_fetch):
        mock_fetch.return_value = ([_stock_record()], None)
        result = _tool().execute(mode="petroleum_stocks", series="crude_excl_spr")
        assert result.success
        assert "crude_excl_spr" in result.data["series"]

    @patch("agent.tools.energy_supply._fetch_eia")
    def test_petroleum_stocks_all(self, mock_fetch):
        mock_fetch.return_value = ([_stock_record()], None)
        result = _tool().execute(mode="petroleum_stocks")
        assert result.success
        assert mock_fetch.call_count == 4  # 4 series

    @patch("agent.tools.energy_supply._fetch_eia")
    def test_petroleum_supply(self, mock_fetch):
        mock_fetch.return_value = ([_stock_record()], None)
        result = _tool().execute(mode="petroleum_supply", series="crude_excl_spr")
        assert result.success

    @patch("agent.tools.energy_supply._fetch_eia")
    def test_rig_count(self, mock_fetch):
        mock_fetch.return_value = ([_rig_record()], None)
        result = _tool().execute(mode="rig_count")
        assert result.success
        assert result.data["months"] == 12

    @patch("agent.tools.energy_supply._fetch_eia")
    def test_rig_count_custom_months(self, mock_fetch):
        mock_fetch.return_value = ([_rig_record()], None)
        result = _tool().execute(mode="rig_count", weeks=6)
        assert result.success
        assert result.data["months"] == 6

    @patch("agent.tools.energy_supply._fetch_eia")
    def test_weeks_capped_at_52(self, mock_fetch):
        mock_fetch.return_value = ([_stock_record()], None)
        result = _tool().execute(mode="petroleum_stocks", series="crude_excl_spr", weeks=200)
        assert result.success
        # Verify length param sent to API
        call_args = mock_fetch.call_args
        assert call_args[0][1]["length"] == "52"

    @patch("agent.tools.energy_supply._fetch_eia")
    def test_fetch_error_propagates(self, mock_fetch):
        mock_fetch.return_value = ([], "EIA exploded")
        result = _tool().execute(mode="petroleum_stocks", series="crude_excl_spr")
        assert not result.success
        assert "exploded" in result.output

    @patch("agent.tools.energy_supply._fetch_eia")
    def test_rig_count_fetch_error(self, mock_fetch):
        mock_fetch.return_value = ([], "timeout")
        result = _tool().execute(mode="rig_count")
        assert not result.success

    @patch("agent.tools.energy_supply._fetch_eia")
    def test_default_weeks(self, mock_fetch):
        mock_fetch.return_value = ([_stock_record()], None)
        _tool().execute(mode="petroleum_stocks", series="crude_excl_spr")
        call_args = mock_fetch.call_args
        assert call_args[0][1]["length"] == "12"


# ── Registry and Bandit integration ──────────────────────────


class TestRegistryIntegration:
    def test_tool_importable(self):
        from agent.tools.energy_supply import EnergySupplyTool

        assert EnergySupplyTool is not None

    def test_bandit_arm_exists(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm_names = [a.name for a in DEFAULT_ARMS]
        assert "energy_supply_monitor" in arm_names

    def test_bandit_arm_references_tool(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm = next(a for a in DEFAULT_ARMS if a.name == "energy_supply_monitor")
        assert "energy_supply" in arm.tools
