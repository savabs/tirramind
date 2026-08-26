"""
Edge case tests for MigrationFlowsTool (UNHCR + World Bank).

Covers: mode validation, country validation, UNHCR/WB fetch, response
parsing (displacement/asylum/remittances), _safe_int for UNHCR mixed types,
signal computation (displacement thresholds, acceptance rates, remittance
YoY/trend), cache interaction, HTTP errors (429/500/timeout), empty data,
malformed responses, output formatting, tool metadata, registry + bandit
integration.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import httpx

from agent.tools.migration_flows import (
    _UNHCR_BASE,
    VALID_MODES,
    WB_REMITTANCE_INDICATOR,
    MigrationFlowsTool,
    _compute_asylum_signals,
    _compute_displacement_signals,
    _compute_remittance_signals,
    _fetch_unhcr,
    _fetch_wb_remittances,
    _format_asylum_summary,
    _format_displacement_summary,
    _format_remittance_summary,
    _parse_asylum_records,
    _parse_population_records,
    _parse_wb_records,
    _safe_int,
)

# ── Fixtures ──────────────────────────────────────────────────


def _tool(cache=None) -> MigrationFlowsTool:
    return MigrationFlowsTool(cache=cache)


def _mock_resp(body: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        text=json.dumps(body),
        request=httpx.Request("GET", "http://test"),
    )


def _population_item(
    year: int = 2023,
    coa: str = "TUR",
    coa_name: str = "Türkiye",
    coo: str = "SYR",
    coo_name: str = "Syrian Arab Rep.",
    refugees: Any = 3250000,
    asylum_seekers: Any = 5000,
    idps: Any = 0,
    stateless: Any = 0,
) -> dict:
    return {
        "year": year,
        "coa": coa,
        "coa_name": coa_name,
        "coa_iso": coa,
        "coo": coo,
        "coo_name": coo_name,
        "coo_iso": coo,
        "refugees": refugees,
        "asylum_seekers": asylum_seekers,
        "returned_refugees": 0,
        "idps": idps,
        "returned_idps": 0,
        "stateless": stateless,
        "ooc": 0,
        "oip": 0,
        "hst": 0,
    }


def _asylum_item(
    year: int = 2023,
    dec_recognized: int = 1000,
    dec_rejected: int = 500,
    dec_closed: int = 100,
    dec_other: int = 50,
    dec_total: int = 1650,
) -> dict:
    return {
        "year": year,
        "coa": "DEU",
        "coa_name": "Germany",
        "coa_iso": "DEU",
        "coo": "SYR",
        "coo_name": "Syrian Arab Rep.",
        "coo_iso": "SYR",
        "procedure_type": "G",
        "dec_level": "FI",
        "dec_recognized": dec_recognized,
        "dec_rejected": dec_rejected,
        "dec_closed": dec_closed,
        "dec_other": dec_other,
        "dec_total": dec_total,
    }


def _wb_record(year: str = "2023", value: float = 3.9e10) -> dict:
    return {
        "date": year,
        "value": value,
        "country": {"id": "PH", "value": "Philippines"},
        "indicator": {"id": WB_REMITTANCE_INDICATOR, "value": "Personal remittances"},
    }


# ── Tool metadata ────────────────────────────────────────────


class TestToolMetadata:
    def test_name(self):
        assert _tool().name == "migration_flows"

    def test_description(self):
        desc = _tool().description
        assert "UNHCR" in desc or "migration" in desc.lower()
        assert len(desc) > 50

    def test_parameters_schema(self):
        params = _tool().parameters
        assert params["type"] == "object"
        assert "mode" in params["properties"]
        assert params["required"] == ["mode"]

    def test_valid_modes_match_schema(self):
        enum = _tool().parameters["properties"]["mode"]["enum"]
        assert set(enum) == VALID_MODES

    def test_has_country_param(self):
        props = _tool().parameters["properties"]
        assert "country" in props

    def test_has_role_param(self):
        props = _tool().parameters["properties"]
        assert "role" in props


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


# ── Country validation for remittances ───────────────────────


class TestRemittanceCountryValidation:
    def test_no_country(self):
        result = _tool().execute(mode="remittances")
        assert not result.success
        assert "country" in result.output.lower()

    def test_empty_country(self):
        result = _tool().execute(mode="remittances", country="")
        assert not result.success

    def test_single_char_country(self):
        result = _tool().execute(mode="remittances", country="X")
        assert not result.success
        assert "Invalid" in result.output or "character" in result.output

    def test_too_long_country(self):
        result = _tool().execute(mode="remittances", country="ABCD")
        assert not result.success


# ── _safe_int ────────────────────────────────────────────────


class TestSafeInt:
    def test_int(self):
        assert _safe_int(3250000) == 3250000

    def test_string_number(self):
        assert _safe_int("100") == 100

    def test_string_zero(self):
        assert _safe_int("0") == 0

    def test_dash(self):
        assert _safe_int("-") == 0

    def test_none(self):
        assert _safe_int(None) == 0

    def test_empty_string(self):
        assert _safe_int("") == 0

    def test_non_numeric(self):
        assert _safe_int("N/A") == 0

    def test_float_string(self):
        # UNHCR shouldn't return floats, but handle gracefully
        assert _safe_int("1.5") == 0  # int("1.5") raises ValueError

    def test_bool(self):
        assert _safe_int(True) == 1

    def test_negative_int(self):
        assert _safe_int(-5) == -5


# ── _parse_population_records ────────────────────────────────


class TestParsePopulationRecords:
    def test_basic(self):
        items = [_population_item()]
        records = _parse_population_records(items)
        assert len(records) == 1
        assert records[0]["refugees"] == 3250000
        assert records[0]["coa"] == "TUR"

    def test_mixed_types(self):
        item = _population_item(refugees="100", asylum_seekers="-", idps=None)
        records = _parse_population_records([item])
        assert records[0]["refugees"] == 100
        assert records[0]["asylum_seekers"] == 0
        assert records[0]["idps"] == 0

    def test_empty(self):
        records = _parse_population_records([])
        assert records == []

    def test_all_fields_present(self):
        items = [_population_item()]
        r = _parse_population_records(items)[0]
        expected_keys = {
            "year",
            "coo",
            "coo_name",
            "coo_iso",
            "coa",
            "coa_name",
            "coa_iso",
            "refugees",
            "asylum_seekers",
            "returned_refugees",
            "idps",
            "returned_idps",
            "stateless",
            "ooc",
            "oip",
            "hst",
        }
        assert expected_keys == set(r.keys())


# ── _parse_asylum_records ────────────────────────────────────


class TestParseAsylumRecords:
    def test_basic(self):
        items = [_asylum_item()]
        records = _parse_asylum_records(items)
        assert len(records) == 1
        assert records[0]["dec_recognized"] == 1000
        assert records[0]["dec_total"] == 1650

    def test_empty(self):
        records = _parse_asylum_records([])
        assert records == []


# ── _parse_wb_records ────────────────────────────────────────


class TestParseWBRecords:
    def test_basic(self):
        raw = [_wb_record("2023", 3.9e10), _wb_record("2022", 3.5e10)]
        records = _parse_wb_records(raw)
        assert len(records) == 2
        # Should be sorted chronologically
        assert records[0]["year"] == "2022"
        assert records[1]["year"] == "2023"

    def test_skip_null_value(self):
        raw = [{"date": "2023", "value": None, "country": {}, "indicator": {}}]
        records = _parse_wb_records(raw)
        assert records == []

    def test_skip_non_numeric(self):
        raw = [{"date": "2023", "value": "N/A", "country": {}, "indicator": {}}]
        records = _parse_wb_records(raw)
        assert records == []

    def test_empty(self):
        records = _parse_wb_records([])
        assert records == []


# ── Displacement signals ─────────────────────────────────────


class TestDisplacementSignals:
    def test_empty(self):
        s = _compute_displacement_signals([])
        assert s["status"] == "NO_DATA"

    def test_critical_alert(self):
        records = [
            {"refugees": 8_000_000, "asylum_seekers": 1_000_000, "idps": 2_000_000, "stateless": 100_000},
        ]
        s = _compute_displacement_signals(records)
        assert "CRITICAL" in s["alert"]
        assert s["total_displaced"] == 11_000_000

    def test_warning_alert(self):
        records = [
            {"refugees": 500_000, "asylum_seekers": 300_000, "idps": 300_000, "stateless": 0},
        ]
        s = _compute_displacement_signals(records)
        assert "WARNING" in s["alert"]

    def test_notice_alert(self):
        records = [
            {"refugees": 50_000, "asylum_seekers": 30_000, "idps": 30_000, "stateless": 0},
        ]
        s = _compute_displacement_signals(records)
        assert "NOTICE" in s["alert"]

    def test_no_alert(self):
        records = [
            {"refugees": 1_000, "asylum_seekers": 500, "idps": 0, "stateless": 0},
        ]
        s = _compute_displacement_signals(records)
        assert s["alert"] is None

    def test_composition_pct(self):
        records = [
            {"refugees": 600, "asylum_seekers": 200, "idps": 200, "stateless": 0},
        ]
        s = _compute_displacement_signals(records)
        assert s["refugee_pct"] == 60.0
        assert s["idp_pct"] == 20.0
        assert s["asylum_pct"] == 20.0

    def test_zero_displaced_no_pct(self):
        records = [
            {"refugees": 0, "asylum_seekers": 0, "idps": 0, "stateless": 100},
        ]
        s = _compute_displacement_signals(records)
        assert s["total_displaced"] == 0
        assert s["refugee_pct"] == 0


# ── Asylum signals ───────────────────────────────────────────


class TestAsylumSignals:
    def test_empty(self):
        s = _compute_asylum_signals([])
        assert s["status"] == "NO_DATA"

    def test_restrictive(self):
        records = [
            {"dec_recognized": 10, "dec_rejected": 90, "dec_closed": 5, "dec_other": 0, "dec_total": 105},
        ]
        s = _compute_asylum_signals(records)
        assert "RESTRICTIVE" in s["alert"]
        assert s["acceptance_rate"] == 10.0

    def test_liberal(self):
        records = [
            {"dec_recognized": 90, "dec_rejected": 10, "dec_closed": 5, "dec_other": 0, "dec_total": 105},
        ]
        s = _compute_asylum_signals(records)
        assert "LIBERAL" in s["alert"]
        assert s["acceptance_rate"] == 90.0

    def test_normal_no_alert(self):
        records = [
            {"dec_recognized": 50, "dec_rejected": 50, "dec_closed": 10, "dec_other": 5, "dec_total": 115},
        ]
        s = _compute_asylum_signals(records)
        assert s["alert"] is None
        assert s["acceptance_rate"] == 50.0

    def test_zero_substantive_no_rate(self):
        records = [
            {"dec_recognized": 0, "dec_rejected": 0, "dec_closed": 10, "dec_other": 0, "dec_total": 10},
        ]
        s = _compute_asylum_signals(records)
        assert s["acceptance_rate"] is None
        assert s["alert"] is None

    def test_closure_rate(self):
        records = [
            {"dec_recognized": 40, "dec_rejected": 40, "dec_closed": 20, "dec_other": 0, "dec_total": 100},
        ]
        s = _compute_asylum_signals(records)
        assert s["closure_rate"] == 20.0


# ── Remittance signals ──────────────────────────────────────


class TestRemittanceSignals:
    def test_empty(self):
        s = _compute_remittance_signals([])
        assert s["status"] == "NO_DATA"

    def test_basic(self):
        records = [{"value": 3.0e10}, {"value": 3.5e10}]
        s = _compute_remittance_signals(records)
        assert s["latest_value"] == 3.5e10
        assert s["latest_value_billions"] == 35.0

    def test_yoy_surge(self):
        records = [{"value": 1e10}, {"value": 2e10}]
        s = _compute_remittance_signals(records)
        assert s["yoy_change_pct"] == 100.0
        assert "SURGE" in s["alert"]

    def test_yoy_critical_drop(self):
        records = [{"value": 1e10}, {"value": 7e9}]
        s = _compute_remittance_signals(records)
        assert s["yoy_change_pct"] == -30.0
        assert "CRITICAL" in s["alert"]

    def test_yoy_warning_drop(self):
        records = [{"value": 1e10}, {"value": 8.5e9}]
        s = _compute_remittance_signals(records)
        assert s["yoy_change_pct"] == -15.0
        assert "WARNING" in s["alert"]

    def test_yoy_normal_no_alert(self):
        records = [{"value": 1e10}, {"value": 1.05e10}]
        s = _compute_remittance_signals(records)
        assert s["alert"] is None

    def test_single_record_no_yoy(self):
        records = [{"value": 1e10}]
        s = _compute_remittance_signals(records)
        assert s["yoy_change_pct"] is None

    def test_trend_growing(self):
        values = [1e10, 1.1e10, 1.2e10, 1.5e10, 1.6e10, 1.7e10]
        records = [{"value": v} for v in values]
        s = _compute_remittance_signals(records)
        assert s["trend"] == "GROWING"

    def test_trend_declining(self):
        values = [2e10, 1.9e10, 1.8e10, 1.0e10, 0.9e10, 0.8e10]
        records = [{"value": v} for v in values]
        s = _compute_remittance_signals(records)
        assert s["trend"] == "DECLINING"

    def test_trend_stable(self):
        values = [1e10] * 6
        records = [{"value": v} for v in values]
        s = _compute_remittance_signals(records)
        assert s["trend"] == "STABLE"

    def test_trend_insufficient(self):
        records = [{"value": 1e10}] * 3
        s = _compute_remittance_signals(records)
        assert s["trend"] == "INSUFFICIENT_DATA"

    def test_zero_prior_no_yoy(self):
        records = [{"value": 0}, {"value": 1e10}]
        s = _compute_remittance_signals(records)
        assert s["yoy_change_pct"] is None


# ── UNHCR fetch (mocked) ────────────────────────────────────


class TestFetchUNHCR:
    @patch("agent.tools.migration_flows.httpx.Client")
    def test_success(self, mock_client_cls):
        body = {"items": [_population_item()]}
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_resp(body)
        mock_client_cls.return_value = mock_client

        items, err = _fetch_unhcr(f"{_UNHCR_BASE}/population/", {"limit": 10})
        assert err is None
        assert len(items) == 1

    @patch("agent.tools.migration_flows.httpx.Client")
    def test_timeout(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = httpx.TimeoutException("timeout")
        mock_client_cls.return_value = mock_client

        items, err = _fetch_unhcr(f"{_UNHCR_BASE}/population/", {})
        assert "timed out" in err

    @patch("agent.tools.migration_flows.httpx.Client")
    def test_rate_limit(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_resp({}, 429)
        mock_client_cls.return_value = mock_client

        items, err = _fetch_unhcr(f"{_UNHCR_BASE}/population/", {})
        assert "rate limit" in err.lower()

    @patch("agent.tools.migration_flows.httpx.Client")
    def test_500_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_resp({}, 500)
        mock_client_cls.return_value = mock_client

        items, err = _fetch_unhcr(f"{_UNHCR_BASE}/population/", {})
        assert "500" in err

    @patch("agent.tools.migration_flows.httpx.Client")
    def test_malformed_json(self, mock_client_cls):
        resp = httpx.Response(
            status_code=200,
            text="not json",
            request=httpx.Request("GET", "http://test"),
        )
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = resp
        mock_client_cls.return_value = mock_client

        items, err = _fetch_unhcr(f"{_UNHCR_BASE}/population/", {})
        assert "parse" in err.lower()


# ── World Bank fetch (mocked) ───────────────────────────────


class TestFetchWorldBank:
    @patch("agent.tools.migration_flows.httpx.Client")
    def test_success(self, mock_client_cls):
        body = [{"page": 1}, [_wb_record()]]
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_resp(body)
        mock_client_cls.return_value = mock_client

        records, err = _fetch_wb_remittances("PH", 2020, 2023, 20)
        assert err is None
        assert len(records) == 1

    @patch("agent.tools.migration_flows.httpx.Client")
    def test_no_data_available(self, mock_client_cls):
        body = [{"page": 1}, None]
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_resp(body)
        mock_client_cls.return_value = mock_client

        records, err = _fetch_wb_remittances("ZZ", 2020, 2023, 20)
        assert err is None  # not an error, just no data
        assert records == []

    @patch("agent.tools.migration_flows.httpx.Client")
    def test_timeout(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = httpx.TimeoutException("timeout")
        mock_client_cls.return_value = mock_client

        records, err = _fetch_wb_remittances("PH", 2020, 2023, 20)
        assert "timed out" in err

    @patch("agent.tools.migration_flows.httpx.Client")
    def test_non_list_response(self, mock_client_cls):
        body = {"error": "bad request"}
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_resp(body)
        mock_client_cls.return_value = mock_client

        records, err = _fetch_wb_remittances("PH", 2020, 2023, 20)
        assert err is None  # non-list body treated as no data
        assert records == []


# ── Formatting ───────────────────────────────────────────────


class TestFormatting:
    def test_displacement_empty(self):
        text = _format_displacement_summary([], {"status": "NO_DATA"}, "", None, "asylum")
        assert "No displacement" in text

    def test_displacement_with_data(self):
        records = _parse_population_records([_population_item()])
        signals = _compute_displacement_signals(records)
        text = _format_displacement_summary(records, signals, "TUR", 2023, "asylum")
        assert "TUR" in text
        assert "Refugees" in text

    def test_asylum_empty(self):
        text = _format_asylum_summary([], {"status": "NO_DATA"}, "", None, "asylum")
        assert "No asylum" in text

    def test_asylum_with_data(self):
        records = _parse_asylum_records([_asylum_item()])
        signals = _compute_asylum_signals(records)
        text = _format_asylum_summary(records, signals, "DEU", 2023, "asylum")
        assert "Decisions" in text or "decisions" in text

    def test_remittance_empty(self):
        text = _format_remittance_summary([], {"status": "NO_DATA"}, "PH", 2020, 2023)
        assert "No remittance" in text

    def test_remittance_with_data(self):
        raw = [_wb_record("2022", 3.5e10), _wb_record("2023", 3.9e10)]
        records = _parse_wb_records(raw)
        signals = _compute_remittance_signals(records)
        text = _format_remittance_summary(records, signals, "PH", 2020, 2023)
        assert "PH" in text
        assert "$" in text


# ── Cache interaction ────────────────────────────────────────


class TestCache:
    @patch("agent.tools.migration_flows._fetch_unhcr")
    def test_displacement_cache_hit(self, mock_fetch):
        cache = MagicMock()
        cache.get.return_value = {
            "output": "cached displacement",
            "data": {"records": [], "signals": {}},
        }
        tool = _tool(cache=cache)
        result = tool.execute(mode="displacement")
        assert result.success
        assert "cached" in result.output
        mock_fetch.assert_not_called()

    @patch("agent.tools.migration_flows._fetch_unhcr")
    def test_displacement_cache_miss(self, mock_fetch):
        mock_fetch.return_value = ([], None)
        cache = MagicMock()
        cache.get.return_value = None
        tool = _tool(cache=cache)
        result = tool.execute(mode="displacement")
        assert result.success
        cache.put.assert_called_once()

    @patch("agent.tools.migration_flows._fetch_wb_remittances")
    def test_remittance_cache_hit(self, mock_fetch):
        cache = MagicMock()
        cache.get.return_value = {
            "output": "cached remittances",
            "data": {"records": [], "signals": {}},
        }
        tool = _tool(cache=cache)
        result = tool.execute(mode="remittances", country="PH")
        assert result.success
        mock_fetch.assert_not_called()


# ── End-to-end with mocked fetch ─────────────────────────────


class TestEndToEnd:
    @patch("agent.tools.migration_flows._fetch_unhcr")
    def test_displacement_success(self, mock_fetch):
        mock_fetch.return_value = ([_population_item()], None)
        result = _tool().execute(mode="displacement", country="TUR")
        assert result.success
        assert result.data["count"] == 1

    @patch("agent.tools.migration_flows._fetch_unhcr")
    def test_asylum_success(self, mock_fetch):
        mock_fetch.return_value = ([_asylum_item()], None)
        result = _tool().execute(mode="asylum", country="DEU")
        assert result.success
        assert result.data["count"] == 1

    @patch("agent.tools.migration_flows._fetch_wb_remittances")
    def test_remittance_success(self, mock_fetch):
        records = _parse_wb_records([_wb_record("2022", 3.5e10), _wb_record("2023", 3.9e10)])
        mock_fetch.return_value = (records, None)
        result = _tool().execute(mode="remittances", country="PH")
        assert result.success
        assert result.data["count"] == 2

    @patch("agent.tools.migration_flows._fetch_unhcr")
    def test_displacement_fetch_error(self, mock_fetch):
        mock_fetch.return_value = ([], "UNHCR exploded")
        result = _tool().execute(mode="displacement")
        assert not result.success
        assert "exploded" in result.output

    @patch("agent.tools.migration_flows._fetch_unhcr")
    def test_origin_role(self, mock_fetch):
        mock_fetch.return_value = ([], None)
        result = _tool().execute(mode="displacement", country="SYR", role="origin")
        assert result.success
        # Verify "coo" param was set in the call
        call_args = mock_fetch.call_args
        assert call_args[0][1].get("coo") == "SYR"

    @patch("agent.tools.migration_flows._fetch_unhcr")
    def test_asylum_role(self, mock_fetch):
        mock_fetch.return_value = ([], None)
        result = _tool().execute(mode="displacement", country="DEU", role="asylum")
        assert result.success
        call_args = mock_fetch.call_args
        assert call_args[0][1].get("coa") == "DEU"

    @patch("agent.tools.migration_flows._fetch_unhcr")
    def test_limit_capped_at_100(self, mock_fetch):
        mock_fetch.return_value = ([], None)
        result = _tool().execute(mode="displacement", limit=500)
        assert result.success
        call_args = mock_fetch.call_args
        assert call_args[0][1]["limit"] == 100

    @patch("agent.tools.migration_flows._fetch_unhcr")
    def test_country_uppercased(self, mock_fetch):
        mock_fetch.return_value = ([], None)
        _tool().execute(mode="displacement", country="tur")
        call_args = mock_fetch.call_args
        assert call_args[0][1].get("coa") == "TUR"


# ── Registry and Bandit integration ──────────────────────────


class TestRegistryIntegration:
    def test_tool_importable(self):
        from agent.tools.migration_flows import MigrationFlowsTool

        assert MigrationFlowsTool is not None

    def test_bandit_arm_exists(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm_names = [a.name for a in DEFAULT_ARMS]
        assert "migration_flow_monitor" in arm_names

    def test_bandit_arm_references_tool(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm = next(a for a in DEFAULT_ARMS if a.name == "migration_flow_monitor")
        assert "migration_flows" in arm.tools
