"""
Edge case tests for FoodSecurityTool (World Bank Agricultural Indicators).

Covers: mode validation, country validation, year range validation, indicator
validation, World Bank API fetch, response parsing, signal computation
(YoY, trend, stress alerts, vulnerability), cache interaction, HTTP errors
(429/500/timeout), empty/null data, malformed responses, output formatting,
tool metadata, registry + bandit integration.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.tools.food_security import (
    _CACHE_TTL,
    _CEREAL_INDICATORS,
    _PRODUCTION_INDICATORS,
    _TRADE_INDICATORS,
    MAJOR_PRODUCERS,
    VALID_MODES,
    VULNERABLE_IMPORTERS,
    FoodSecurityTool,
    _compute_signals,
    _format_summary,
    _parse_wb_records,
)

# ── Fixtures ──────────────────────────────────────────────────


def _tool(cache=None) -> FoodSecurityTool:
    return FoodSecurityTool(cache=cache)


def _mock_resp(body: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        text=json.dumps(body),
        request=httpx.Request("GET", "http://test"),
    )


def _wb_record(
    country_id: str = "US",
    country_name: str = "United States",
    year: str = "2022",
    value: float | None = 105.2,
    indicator_id: str = "AG.PRD.FOOD.XD",
    indicator_name: str = "Food production index",
) -> dict:
    return {
        "country": {"id": country_id, "value": country_name},
        "date": year,
        "value": value,
        "indicator": {"id": indicator_id, "value": indicator_name},
    }


def _wb_response(records: list[dict], page: int = 1, total: int = 0) -> list:
    """World Bank API response format: [metadata, data_array]."""
    return [
        {"page": page, "pages": 1, "per_page": 50, "total": total or len(records)},
        records,
    ]


# ── TestToolMetadata ──────────────────────────────────────────


class TestToolMetadata:
    def test_name(self):
        assert _tool().name == "food_security"

    def test_description_contains_key_terms(self):
        desc = _tool().description
        assert "food security" in desc.lower()
        assert "World Bank" in desc

    def test_parameters_has_required_fields(self):
        params = _tool().parameters
        assert "mode" in params["properties"]
        assert "country" in params["properties"]
        assert params["required"] == ["mode", "country"]

    def test_parameters_mode_enum_matches_valid_modes(self):
        enum = params = _tool().parameters["properties"]["mode"]["enum"]
        assert set(enum) == VALID_MODES


# ── TestInputValidation ───────────────────────────────────────


class TestInputValidation:
    def test_invalid_mode(self):
        r = _tool().execute(mode="invalid", country="US")
        assert not r.success
        assert "Invalid mode" in r.output

    def test_empty_mode(self):
        r = _tool().execute(mode="", country="US")
        assert not r.success

    def test_missing_country(self):
        r = _tool().execute(mode="production")
        assert not r.success
        assert "country" in r.output.lower()

    def test_empty_country(self):
        r = _tool().execute(mode="production", country="")
        assert not r.success

    def test_country_too_long(self):
        r = _tool().execute(mode="production", country="ABCD")
        assert not r.success
        assert "2-3 characters" in r.output

    def test_country_single_char(self):
        r = _tool().execute(mode="production", country="X")
        assert not r.success

    def test_start_after_end_year(self):
        r = _tool().execute(
            mode="production",
            country="US",
            start_year=2025,
            end_year=2020,
        )
        assert not r.success
        assert "start_year" in r.output

    def test_invalid_production_indicator(self):
        with patch("httpx.Client") as mock_client:
            r = _tool().execute(
                mode="production",
                country="US",
                indicator="invalid",
            )
        assert not r.success
        assert "Invalid production indicator" in r.output

    def test_invalid_cereal_indicator(self):
        r = _tool().execute(
            mode="cereal_yield",
            country="US",
            indicator="bad",
        )
        assert not r.success
        assert "Invalid cereal indicator" in r.output

    def test_invalid_trade_indicator(self):
        r = _tool().execute(
            mode="food_trade",
            country="US",
            indicator="bad",
        )
        assert not r.success
        assert "Invalid trade indicator" in r.output

    def test_country_case_insensitive(self):
        """Country code should be uppercased internally."""
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = lambda s: s
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = _mock_resp(_wb_response([_wb_record()]))
            r = _tool().execute(mode="production", country="us")
        assert r.success

    def test_wld_country_accepted(self):
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = lambda s: s
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = _mock_resp(_wb_response([_wb_record(country_id="WLD")]))
            r = _tool().execute(mode="production", country="WLD")
        assert r.success

    def test_limit_clamped_to_100(self):
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = lambda s: s
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_resp = _mock_resp(_wb_response([_wb_record()]))
            mock_client.return_value.get.return_value = mock_resp
            r = _tool().execute(
                mode="production",
                country="US",
                limit=500,
            )
        assert r.success
        call_args = mock_client.return_value.get.call_args
        assert call_args[1]["params"]["per_page"] == "100"


# ── TestWBRecordParsing ───────────────────────────────────────


class TestWBRecordParsing:
    def test_parse_basic_records(self):
        raw = [_wb_record(), _wb_record(year="2021", value=103.0)]
        records = _parse_wb_records(raw, "US", "AG.PRD.FOOD.XD")
        assert len(records) == 2
        assert records[0]["country"] == "US"
        assert records[0]["year"] == "2022"
        assert records[0]["value"] == 105.2

    def test_parse_null_value(self):
        raw = [_wb_record(value=None)]
        records = _parse_wb_records(raw, "US", "AG.PRD.FOOD.XD")
        assert records[0]["value"] is None

    def test_parse_empty_list(self):
        records = _parse_wb_records([], "US", "AG.PRD.FOOD.XD")
        assert records == []

    def test_parse_missing_country_field(self):
        rec = {"date": "2022", "value": 100, "indicator": {"id": "X", "value": "Y"}}
        records = _parse_wb_records([rec], "US", "X")
        assert records[0]["country"] == "US"  # fallback

    def test_parse_missing_indicator_field(self):
        rec = {"country": {"id": "IN", "value": "India"}, "date": "2022", "value": 99}
        records = _parse_wb_records([rec], "IN", "AG.PRD.FOOD.XD")
        assert records[0]["indicator"] == "AG.PRD.FOOD.XD"  # fallback


# ── TestSignalComputation ─────────────────────────────────────


class TestSignalComputation:
    def test_empty_valid(self):
        assert _compute_signals([], "production:food") == {}

    def test_single_record_no_signals(self):
        valid = [{"value": 100, "year": "2022"}]
        signals = _compute_signals(valid, "production:food")
        assert signals == {}

    def test_yoy_increase(self):
        valid = [{"value": 100, "year": "2021"}, {"value": 110, "year": "2022"}]
        signals = _compute_signals(valid, "production:food")
        assert signals["yoy_change_pct"] == 10.0

    def test_yoy_decrease(self):
        valid = [{"value": 100, "year": "2021"}, {"value": 90, "year": "2022"}]
        signals = _compute_signals(valid, "production:food")
        assert signals["yoy_change_pct"] == -10.0

    def test_yoy_zero_previous(self):
        valid = [{"value": 0, "year": "2021"}, {"value": 10, "year": "2022"}]
        signals = _compute_signals(valid, "production:food")
        assert "yoy_change_pct" not in signals

    def test_period_average(self):
        valid = [
            {"value": 100, "year": "2020"},
            {"value": 110, "year": "2021"},
            {"value": 120, "year": "2022"},
        ]
        signals = _compute_signals(valid, "production:food")
        assert signals["period_average"] == 110.0

    def test_deviation_from_avg(self):
        valid = [
            {"value": 100, "year": "2020"},
            {"value": 100, "year": "2021"},
            {"value": 80, "year": "2022"},
        ]
        signals = _compute_signals(valid, "production:food")
        assert signals["deviation_from_avg_pct"] < 0

    def test_trend_up(self):
        valid = [
            {"value": 100, "year": "2020"},
            {"value": 110, "year": "2021"},
            {"value": 120, "year": "2022"},
        ]
        signals = _compute_signals(valid, "production:food")
        assert signals["trend_direction"] == "up"
        assert signals["consecutive_years"] == 2

    def test_trend_down(self):
        valid = [
            {"value": 120, "year": "2020"},
            {"value": 110, "year": "2021"},
            {"value": 100, "year": "2022"},
        ]
        signals = _compute_signals(valid, "production:food")
        assert signals["trend_direction"] == "down"
        assert signals["consecutive_years"] == 2

    def test_trend_reversal(self):
        valid = [
            {"value": 100, "year": "2020"},
            {"value": 90, "year": "2021"},
            {"value": 95, "year": "2022"},
        ]
        signals = _compute_signals(valid, "production:food")
        assert signals["trend_direction"] == "up"
        assert signals["consecutive_years"] == 1

    def test_flat_values_no_trend(self):
        valid = [
            {"value": 100, "year": "2021"},
            {"value": 100, "year": "2022"},
        ]
        signals = _compute_signals(valid, "production:food")
        assert "trend_direction" not in signals

    def test_stress_alert_declining_production(self):
        valid = [
            {"value": 100, "year": "2019"},
            {"value": 110, "year": "2020"},
            {"value": 105, "year": "2021"},
            {"value": 100, "year": "2022"},
        ]
        signals = _compute_signals(valid, "production:food")
        assert signals["trend_direction"] == "down"
        assert signals["consecutive_years"] == 2
        assert "stress_alert" in signals

    def test_stress_alert_below_average(self):
        valid = [
            {"value": 100, "year": "2020"},
            {"value": 100, "year": "2021"},
            {"value": 80, "year": "2022"},
        ]
        signals = _compute_signals(valid, "production:food")
        # deviation is about -14.3%, below -10% threshold
        assert "stress_alert" in signals

    def test_food_import_vulnerability_high(self):
        valid = [
            {"value": 25, "year": "2021"},
            {"value": 35, "year": "2022"},
        ]
        signals = _compute_signals(valid, "food_trade:food_import_pct")
        assert signals["vulnerability"] == "high"
        assert "vulnerability_note" in signals

    def test_food_import_vulnerability_moderate(self):
        valid = [
            {"value": 18, "year": "2021"},
            {"value": 25, "year": "2022"},
        ]
        signals = _compute_signals(valid, "food_trade:food_import_pct")
        assert signals["vulnerability"] == "moderate"

    def test_food_import_low_not_flagged(self):
        valid = [
            {"value": 10, "year": "2021"},
            {"value": 15, "year": "2022"},
        ]
        signals = _compute_signals(valid, "food_trade:food_import_pct")
        assert "vulnerability" not in signals

    def test_non_production_label_no_stress(self):
        """cereal_yield label should not trigger production stress alerts."""
        valid = [
            {"value": 100, "year": "2019"},
            {"value": 90, "year": "2020"},
            {"value": 80, "year": "2021"},
            {"value": 70, "year": "2022"},
        ]
        signals = _compute_signals(valid, "cereal_yield:yield_kg_per_ha")
        assert "stress_alert" not in signals


# ── TestOutputFormatting ──────────────────────────────────────


class TestOutputFormatting:
    def test_format_basic(self):
        records = [
            {"year": "2022", "value": 105, "indicator_name": "Food prod"},
        ]
        valid = records
        signals = {"yoy_change_pct": 5.0}
        out = _format_summary(records, valid, signals, "US", "production:food")
        assert "US" in out
        assert "105" in out
        assert "5.0%" in out

    def test_format_empty(self):
        out = _format_summary([], [], {}, "US", "production:food")
        assert "0 with data" in out

    def test_format_stress_alert_shown(self):
        valid = [
            {"year": "2022", "value": 80, "indicator_name": "X"},
        ]
        signals = {"stress_alert": "test alert"}
        out = _format_summary(valid, valid, signals, "US", "production:food")
        assert "STRESS" in out

    def test_format_vulnerability_shown(self):
        valid = [
            {"year": "2022", "value": 35, "indicator_name": "X"},
        ]
        signals = {"vulnerability": "high", "vulnerability_note": "35% imports"}
        out = _format_summary(valid, valid, signals, "EG", "food_trade:food_import_pct")
        assert "high" in out

    def test_recent_values_shown(self):
        valid = [{"year": str(y), "value": 100 + y, "indicator_name": "X"} for y in range(2017, 2023)]
        out = _format_summary(valid, valid, {}, "US", "production:food")
        assert "Recent values:" in out
        # Should show last 6
        assert "2022" in out
        assert "2017" in out


# ── TestProductionMode ────────────────────────────────────────


class TestProductionMode:
    def test_food_production_success(self):
        body = _wb_response(
            [
                _wb_record(year="2021", value=100),
                _wb_record(year="2022", value=105),
            ]
        )
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = lambda s: s
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = _mock_resp(body)
            r = _tool().execute(mode="production", country="US")
        assert r.success
        assert r.data["valid_count"] == 2
        assert "yoy_change_pct" in r.data["signals"]

    def test_crop_production(self):
        body = _wb_response([_wb_record(indicator_id="AG.PRD.CROP.XD")])
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = lambda s: s
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = _mock_resp(body)
            r = _tool().execute(
                mode="production",
                country="CN",
                indicator="crop",
            )
        assert r.success

    def test_livestock_production(self):
        body = _wb_response([_wb_record(indicator_id="AG.PRD.LVSK.XD")])
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = lambda s: s
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = _mock_resp(body)
            r = _tool().execute(
                mode="production",
                country="BR",
                indicator="livestock",
            )
        assert r.success


# ── TestCerealYieldMode ───────────────────────────────────────


class TestCerealYieldMode:
    def test_yield_success(self):
        body = _wb_response(
            [
                _wb_record(year="2021", value=6000, indicator_id="AG.YLD.CREL.KG"),
                _wb_record(year="2022", value=6200, indicator_id="AG.YLD.CREL.KG"),
            ]
        )
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = lambda s: s
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = _mock_resp(body)
            r = _tool().execute(mode="cereal_yield", country="IN")
        assert r.success

    def test_area_hectares(self):
        body = _wb_response([_wb_record(indicator_id="AG.LND.CREL.HA")])
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = lambda s: s
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = _mock_resp(body)
            r = _tool().execute(
                mode="cereal_yield",
                country="US",
                indicator="area_hectares",
            )
        assert r.success


# ── TestFoodTradeMode ─────────────────────────────────────────


class TestFoodTradeMode:
    def test_food_import_success(self):
        body = _wb_response(
            [
                _wb_record(year="2021", value=28.5, indicator_id="TM.VAL.FOOD.ZS.UN"),
                _wb_record(year="2022", value=31.2, indicator_id="TM.VAL.FOOD.ZS.UN"),
            ]
        )
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = lambda s: s
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = _mock_resp(body)
            r = _tool().execute(mode="food_trade", country="EG")
        assert r.success
        assert r.data["signals"].get("vulnerability") == "high"

    def test_food_export(self):
        body = _wb_response([_wb_record(indicator_id="TX.VAL.FOOD.ZS.UN")])
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = lambda s: s
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = _mock_resp(body)
            r = _tool().execute(
                mode="food_trade",
                country="BR",
                indicator="food_export_pct",
            )
        assert r.success


# ── TestHTTPErrors ────────────────────────────────────────────


class TestHTTPErrors:
    def test_timeout(self):
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = lambda s: s
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.side_effect = httpx.TimeoutException("timeout")
            r = _tool().execute(mode="production", country="US")
        assert not r.success
        assert "timed out" in r.output

    def test_http_error(self):
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = lambda s: s
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.side_effect = httpx.HTTPError("err")
            r = _tool().execute(mode="production", country="US")
        assert not r.success
        assert "HTTP error" in r.output

    def test_rate_limit_429(self):
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = lambda s: s
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = _mock_resp({}, 429)
            r = _tool().execute(mode="production", country="US")
        assert not r.success
        assert "rate limit" in r.output.lower()

    def test_server_error_500(self):
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = lambda s: s
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = _mock_resp({}, 500)
            r = _tool().execute(mode="production", country="US")
        assert not r.success
        assert "500" in r.output

    def test_malformed_json(self):
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = lambda s: s
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            resp = httpx.Response(
                status_code=200,
                text="not json",
                request=httpx.Request("GET", "http://test"),
            )
            mock_client.return_value.get.return_value = resp
            r = _tool().execute(mode="production", country="US")
        assert not r.success
        assert "parse" in r.output.lower()


# ── TestEmptyAndNullData ──────────────────────────────────────


class TestEmptyAndNullData:
    def test_null_data_array(self):
        body = [{"page": 1, "total": 0}, None]
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = lambda s: s
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = _mock_resp(body)
            r = _tool().execute(mode="production", country="US")
        assert r.success
        assert "No data" in r.output

    def test_empty_data_array(self):
        body = [{"page": 1, "total": 0}, []]
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = lambda s: s
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = _mock_resp(body)
            r = _tool().execute(mode="production", country="US")
        assert r.success
        assert r.data["valid_count"] == 0

    def test_all_null_values(self):
        body = _wb_response(
            [
                _wb_record(year="2021", value=None),
                _wb_record(year="2022", value=None),
            ]
        )
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = lambda s: s
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = _mock_resp(body)
            r = _tool().execute(mode="production", country="US")
        assert r.success
        assert r.data["valid_count"] == 0

    def test_non_list_body(self):
        """Non-list body should be handled gracefully."""
        body = {"error": "not found"}
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = lambda s: s
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = _mock_resp(body)
            r = _tool().execute(mode="production", country="US")
        assert r.success
        assert "No data" in r.output


# ── TestCache ─────────────────────────────────────────────────


class TestCache:
    def test_cache_hit(self):
        cache = MagicMock()
        cache.get.return_value = {
            "output": "cached output",
            "data": {"records": [], "cached": True},
        }
        r = _tool(cache=cache).execute(mode="production", country="US")
        assert r.success
        assert r.output == "cached output"
        assert r.data["cached"] is True

    def test_cache_miss_then_store(self):
        cache = MagicMock()
        cache.get.return_value = None
        body = _wb_response([_wb_record()])
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = lambda s: s
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = _mock_resp(body)
            r = _tool(cache=cache).execute(mode="production", country="US")
        assert r.success
        cache.set.assert_called_once()
        call_args = cache.set.call_args
        assert call_args[1]["ttl"] == _CACHE_TTL

    def test_no_cache_still_works(self):
        body = _wb_response([_wb_record()])
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = lambda s: s
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = _mock_resp(body)
            r = _tool(cache=None).execute(mode="production", country="US")
        assert r.success


# ── TestConstants ─────────────────────────────────────────────


class TestConstants:
    def test_production_indicators_all_valid(self):
        for key, code in _PRODUCTION_INDICATORS.items():
            assert code.startswith("AG.PRD.")

    def test_cereal_indicators_all_valid(self):
        for key, code in _CEREAL_INDICATORS.items():
            assert code.startswith("AG.")

    def test_trade_indicators_all_valid(self):
        for code in _TRADE_INDICATORS.values():
            assert "VAL.FOOD" in code

    def test_major_producers_are_iso2(self):
        for cc in MAJOR_PRODUCERS:
            assert len(cc) == 2

    def test_vulnerable_importers_are_iso2(self):
        for cc in VULNERABLE_IMPORTERS:
            assert len(cc) == 2

    def test_valid_modes(self):
        assert {"production", "cereal_yield", "food_trade"} == VALID_MODES


# ── TestRegistryAndBandit ─────────────────────────────────────


class TestRegistryAndBandit:
    def test_tool_in_cli_registry(self):
        from agent.cli import build_tool_registry

        registry = build_tool_registry()
        names = registry.list_names()
        assert "food_security" in names

    def test_bandit_arm_exists(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm_names = [a.name for a in DEFAULT_ARMS]
        assert "food_security_monitor" in arm_names

    def test_bandit_arm_references_tool(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm = next(a for a in DEFAULT_ARMS if a.name == "food_security_monitor")
        assert "food_security" in arm.tools


# ── TestEdgeCombinations ──────────────────────────────────────


class TestEdgeCombinations:
    def test_explicit_year_range(self):
        body = _wb_response(
            [
                _wb_record(year="2018", value=98),
                _wb_record(year="2019", value=100),
            ]
        )
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = lambda s: s
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = _mock_resp(body)
            r = _tool().execute(
                mode="production",
                country="IN",
                start_year=2018,
                end_year=2019,
            )
        assert r.success
        call_args = mock_client.return_value.get.call_args
        assert "2018:2019" in call_args[1]["params"]["date"]

    def test_same_start_end_year(self):
        body = _wb_response([_wb_record(year="2022")])
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = lambda s: s
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = _mock_resp(body)
            r = _tool().execute(
                mode="production",
                country="US",
                start_year=2022,
                end_year=2022,
            )
        assert r.success

    def test_mixed_null_and_valid_values(self):
        body = _wb_response(
            [
                _wb_record(year="2020", value=None),
                _wb_record(year="2021", value=100),
                _wb_record(year="2022", value=110),
            ]
        )
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = lambda s: s
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = _mock_resp(body)
            r = _tool().execute(mode="production", country="US")
        assert r.success
        assert r.data["valid_count"] == 2
        assert r.data["total_count"] == 3

    def test_negative_values(self):
        """Some indicators can have negative deviation values."""
        valid = [
            {"value": -5, "year": "2021"},
            {"value": -10, "year": "2022"},
        ]
        signals = _compute_signals(valid, "production:food")
        # Should handle negatives without crashing
        assert "yoy_change_pct" in signals

    def test_very_large_values(self):
        valid = [
            {"value": 1e12, "year": "2021"},
            {"value": 1.1e12, "year": "2022"},
        ]
        signals = _compute_signals(valid, "production:food")
        assert signals["yoy_change_pct"] == pytest.approx(10.0, rel=0.01)
