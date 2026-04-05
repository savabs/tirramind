"""
Edge case tests for InternetInfrastructureTool (IODA + OONI).

Covers: mode validation, parameter validation, IODA outages parsing,
IODA signals parsing, OONI censorship parsing, OONI incidents parsing,
HTTP errors, cache interaction, scoring thresholds, trend analysis,
malformed payloads, empty responses, boundary values, output formatting,
registry integration.
"""

from __future__ import annotations

import json
import time
from collections import namedtuple
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent.tools.internet_infrastructure import (
    CACHE_IODA_ALERTS,
    CACHE_IODA_SIGNALS,
    CACHE_OONI_AGGREGATION,
    CACHE_OONI_INCIDENTS,
    GTR_NORM_CRITICAL,
    GTR_NORM_WARNING,
    IODA_BASE,
    InternetInfrastructureTool,
    OONI_BASE,
    VALID_MODES,
    OONI_TEST_TYPES,
    _safe_float,
    _safe_int,
    _severity_from_gtr_norm,
    _ts_to_iso,
)


# ── Fixtures ─────────────────────────────────────────────────


def _tool(cache: Any = None) -> InternetInfrastructureTool:
    return InternetInfrastructureTool(cache=cache)


def _mock_response(status_code: int = 200, json_data: Any = None, text: str = ""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    if json_data is not None:
        resp.json.return_value = json_data
    else:
        resp.json.side_effect = ValueError("No JSON")
    return resp


def _ioda_alerts_response(alerts: list[dict]) -> dict:
    return {
        "type": "outages.alerts",
        "metadata": {},
        "data": alerts,
        "copyright": "Test copyright",
    }


def _ioda_alert(
    code: str = "US",
    name: str = "United States",
    ds: str = "bgp",
    level: str = "critical",
    value: float = 100,
    history: float = 200,
    ts: int = 1700000000,
) -> dict:
    return {
        "datasource": ds,
        "entity": {"code": code, "name": name, "type": "country"},
        "time": ts,
        "level": level,
        "condition": "< 0.99" if level == "critical" else "normal",
        "value": value,
        "historyValue": history,
        "method": "median",
    }


def _ioda_events_response(events: list[dict]) -> dict:
    return {
        "type": "outages.events",
        "metadata": {},
        "data": events,
        "copyright": "Test copyright",
    }


def _ioda_event(
    location: str = "country/US",
    score: float = 5000.0,
    ds: str = "bgp",
    start: int = 1700000000,
    duration: int = 3600,
) -> dict:
    return {
        "location": location,
        "score": score,
        "datasource": ds,
        "start": start,
        "duration": duration,
        "status": 0,
        "method": "median",
    }


def _ioda_signals_response(
    values: list,
    ds: str = "gtr-norm",
    step: int = 1800,
    from_ts: int = 1700000000,
) -> dict:
    return {
        "type": "signals",
        "data": [
            [
                {
                    "entityType": "country",
                    "entityCode": "US",
                    "entityName": "United States",
                    "datasource": ds,
                    "subtype": None,
                    "from": from_ts,
                    "until": from_ts + step * len(values),
                    "step": step,
                    "nativeStep": step,
                    "values": values,
                }
            ]
        ],
        "copyright": "Test copyright",
    }


def _ooni_aggregation_response(rows: list[dict]) -> dict:
    return {"result": rows}


def _ooni_agg_row(
    date: str = "2026-03-15",
    ok: int = 100,
    anomaly: int = 5,
    confirmed: int = 0,
) -> dict:
    return {
        "measurement_start_day": date,
        "ok_count": ok,
        "anomaly_count": anomaly,
        "confirmed_count": confirmed,
    }


def _ooni_incidents_response(incidents: list[dict]) -> dict:
    return {"incidents": incidents}


def _ooni_incident(
    title: str = "Test incident",
    ccs: list[str] | None = None,
    published: bool = True,
) -> dict:
    return {
        "title": title,
        "CCs": ccs or ["RU"],
        "published": published,
        "start_time": "2026-03-01T00:00:00Z",
    }


# ══════════════════════════════════════════════════════════════
# Test: Helper functions
# ══════════════════════════════════════════════════════════════


class TestSafeFloat:
    def test_normal(self):
        assert _safe_float(3.14) == 3.14

    def test_int(self):
        assert _safe_float(42) == 42.0

    def test_string(self):
        assert _safe_float("2.5") == 2.5

    def test_none(self):
        assert _safe_float(None) == 0.0

    def test_none_custom_default(self):
        assert _safe_float(None, -1.0) == -1.0

    def test_invalid_string(self):
        assert _safe_float("abc") == 0.0

    def test_list(self):
        assert _safe_float([1, 2]) == 0.0


class TestSafeInt:
    def test_normal(self):
        assert _safe_int(42) == 42

    def test_float(self):
        assert _safe_int(3.7) == 3

    def test_string(self):
        assert _safe_int("10") == 10

    def test_none(self):
        assert _safe_int(None) == 0

    def test_invalid(self):
        assert _safe_int("xyz") == 0


class TestTsToIso:
    def test_valid(self):
        result = _ts_to_iso(1700000000)
        assert "2023" in result
        assert "UTC" in result

    def test_none(self):
        assert _ts_to_iso(None) == "unknown"

    def test_zero(self):
        result = _ts_to_iso(0)
        assert "1970" in result

    def test_string_ts(self):
        # Should handle string timestamps
        result = _ts_to_iso("bad")
        assert result == "unknown"


class TestSeverityFromGtrNorm:
    def test_normal(self):
        assert _severity_from_gtr_norm(0.95) == "normal"

    def test_warning(self):
        assert _severity_from_gtr_norm(0.70) == "warning"

    def test_critical(self):
        assert _severity_from_gtr_norm(0.30) == "critical"

    def test_boundary_warning(self):
        assert _severity_from_gtr_norm(GTR_NORM_WARNING) == "normal"

    def test_just_below_warning(self):
        assert _severity_from_gtr_norm(GTR_NORM_WARNING - 0.001) == "warning"

    def test_boundary_critical(self):
        assert _severity_from_gtr_norm(GTR_NORM_CRITICAL) == "warning"

    def test_just_below_critical(self):
        assert _severity_from_gtr_norm(GTR_NORM_CRITICAL - 0.001) == "critical"

    def test_zero(self):
        assert _severity_from_gtr_norm(0.0) == "critical"

    def test_one(self):
        assert _severity_from_gtr_norm(1.0) == "normal"


# ══════════════════════════════════════════════════════════════
# Test: Tool metadata
# ══════════════════════════════════════════════════════════════


class TestToolMetadata:
    def test_name(self):
        assert _tool().name == "internet_infrastructure"

    def test_description_contains_ioda(self):
        assert "IODA" in _tool().description

    def test_description_contains_ooni(self):
        assert "OONI" in _tool().description

    def test_parameters_has_mode(self):
        assert "mode" in _tool().parameters["properties"]

    def test_parameters_mode_enum(self):
        enum = _tool().parameters["properties"]["mode"]["enum"]
        assert set(enum) == set(VALID_MODES)

    def test_parameters_has_country(self):
        assert "country" in _tool().parameters["properties"]

    def test_parameters_has_test(self):
        assert "test" in _tool().parameters["properties"]

    def test_parameters_has_hours_back(self):
        assert "hours_back" in _tool().parameters["properties"]

    def test_parameters_has_days_back(self):
        assert "days_back" in _tool().parameters["properties"]

    def test_parameters_has_limit(self):
        assert "limit" in _tool().parameters["properties"]


# ══════════════════════════════════════════════════════════════
# Test: Mode routing and parameter validation
# ══════════════════════════════════════════════════════════════


class TestModeRouting:
    def test_invalid_mode(self):
        r = _tool().execute(mode="invalid")
        assert not r.success
        assert "Invalid mode" in r.output

    def test_empty_mode(self):
        r = _tool().execute(mode="")
        assert not r.success

    def test_mode_case_insensitive(self):
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ioda_alerts_response([]))
            r = _tool().execute(mode="OUTAGES")
            assert r.success

    def test_mode_whitespace(self):
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ioda_alerts_response([]))
            r = _tool().execute(mode="  outages  ")
            assert r.success


class TestParameterValidation:
    def test_censorship_requires_country(self):
        r = _tool().execute(mode="censorship")
        assert not r.success
        assert "country" in r.output.lower()

    def test_censorship_rejects_single_char_country(self):
        r = _tool().execute(mode="censorship", country="U")
        assert not r.success

    def test_censorship_rejects_three_char_country(self):
        r = _tool().execute(mode="censorship", country="USA")
        assert not r.success

    def test_censorship_invalid_test(self):
        r = _tool().execute(mode="censorship", country="US", test="invalid_test")
        assert not r.success
        assert "Invalid test" in r.output

    def test_signals_requires_country(self):
        r = _tool().execute(mode="signals")
        assert not r.success
        assert "country" in r.output.lower()

    def test_signals_rejects_invalid_country(self):
        r = _tool().execute(mode="signals", country="X")
        assert not r.success

    def test_hours_back_clamped_min(self):
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ioda_alerts_response([]))
            r = _tool().execute(mode="outages", hours_back=-5)
            assert r.success

    def test_hours_back_clamped_max(self):
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ioda_alerts_response([]))
            r = _tool().execute(mode="outages", hours_back=9999)
            assert r.success

    def test_days_back_clamped(self):
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ooni_aggregation_response([]))
            r = _tool().execute(mode="censorship", country="US", days_back=999)
            assert r.success

    def test_limit_clamped_min(self):
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ooni_incidents_response([]))
            r = _tool().execute(mode="incidents", limit=0)
            assert r.success

    def test_limit_clamped_max(self):
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ooni_incidents_response([]))
            r = _tool().execute(mode="incidents", limit=9999)
            assert r.success

    def test_country_uppercased(self):
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ioda_signals_response([0.9, 0.8]))
            _tool().execute(mode="signals", country="us")
            call_url = mock.call_args[0][0]
            assert "/US" in call_url


# ══════════════════════════════════════════════════════════════
# Test: Outages mode (IODA)
# ══════════════════════════════════════════════════════════════


class TestOutagesMode:
    def test_empty_alerts_and_events(self):
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ioda_alerts_response([]))
            r = _tool().execute(mode="outages")
            assert r.success
            assert "No outage" in r.output

    def test_critical_alert_parsed(self):
        alerts = [
            _ioda_alert(code="IR", name="Iran", level="critical", value=50, history=200)
        ]
        with patch("httpx.get") as mock:
            mock.side_effect = [
                _mock_response(200, _ioda_alerts_response(alerts)),
                _mock_response(200, _ioda_events_response([])),
            ]
            r = _tool().execute(mode="outages")
            assert r.success
            assert "IR" in r.output
            assert "CRITICAL" in r.output

    def test_normal_alerts_filtered(self):
        """Normal-level alerts (recovery) should be excluded."""
        alerts = [_ioda_alert(level="normal")]
        with patch("httpx.get") as mock:
            mock.side_effect = [
                _mock_response(200, _ioda_alerts_response(alerts)),
                _mock_response(200, _ioda_events_response([])),
            ]
            r = _tool().execute(mode="outages")
            assert r.success
            assert "No outage" in r.output

    def test_events_parsed(self):
        events = [_ioda_event(location="country/BR", score=20000)]
        with patch("httpx.get") as mock:
            mock.side_effect = [
                _mock_response(200, _ioda_alerts_response([])),
                _mock_response(200, _ioda_events_response(events)),
            ]
            r = _tool().execute(mode="outages")
            assert r.success
            assert "BR" in r.output
            assert "20000" in r.output

    def test_country_filter_in_params(self):
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ioda_alerts_response([]))
            _tool().execute(mode="outages", country="RU")
            # Both calls should include entityCode
            for call in mock.call_args_list:
                params = call[1].get("params", {})
                if params:
                    assert params.get("entityCode") == "RU"

    def test_alerts_sorted_by_time_desc(self):
        alerts = [
            _ioda_alert(code="A1", level="critical", ts=100),
            _ioda_alert(code="A2", level="critical", ts=300),
            _ioda_alert(code="A3", level="critical", ts=200),
        ]
        with patch("httpx.get") as mock:
            mock.side_effect = [
                _mock_response(200, _ioda_alerts_response(alerts)),
                _mock_response(200, _ioda_events_response([])),
            ]
            r = _tool().execute(mode="outages")
            # A2 (ts=300) should appear first
            a2_pos = r.output.index("A2")
            a3_pos = r.output.index("A3")
            a1_pos = r.output.index("A1")
            assert a2_pos < a3_pos < a1_pos

    def test_events_sorted_by_score_desc(self):
        events = [
            _ioda_event(location="country/A1", score=100),
            _ioda_event(location="country/A2", score=500),
            _ioda_event(location="country/A3", score=300),
        ]
        with patch("httpx.get") as mock:
            mock.side_effect = [
                _mock_response(200, _ioda_alerts_response([])),
                _mock_response(200, _ioda_events_response(events)),
            ]
            r = _tool().execute(mode="outages")
            a2_pos = r.output.index("A2")
            a3_pos = r.output.index("A3")
            a1_pos = r.output.index("A1")
            assert a2_pos < a3_pos < a1_pos

    def test_malformed_alert_skipped(self):
        alerts = [
            "not a dict",
            {"entity": "not a dict", "level": "critical"},
            _ioda_alert(code="OK", level="critical"),
        ]
        with patch("httpx.get") as mock:
            mock.side_effect = [
                _mock_response(200, _ioda_alerts_response(alerts)),
                _mock_response(200, _ioda_events_response([])),
            ]
            r = _tool().execute(mode="outages")
            assert r.success
            assert "OK" in r.output

    def test_malformed_event_skipped(self):
        events = [
            "not a dict",
            _ioda_event(location="country/OK", score=999),
        ]
        with patch("httpx.get") as mock:
            mock.side_effect = [
                _mock_response(200, _ioda_alerts_response([])),
                _mock_response(200, _ioda_events_response(events)),
            ]
            r = _tool().execute(mode="outages")
            assert r.success
            assert "OK" in r.output

    def test_http_error_alerts(self):
        with patch("httpx.get") as mock:
            mock.side_effect = [
                _mock_response(500),  # alerts fail
                _mock_response(200, _ioda_events_response([])),
            ]
            r = _tool().execute(mode="outages")
            assert r.success  # graceful degradation

    def test_http_timeout(self):
        import httpx as httpx_mod

        with patch("httpx.get") as mock:
            mock.side_effect = httpx_mod.TimeoutException("timeout")
            r = _tool().execute(mode="outages")
            assert r.success  # graceful — returns "no outages" instead of crash
            assert "No outage" in r.output

    def test_duration_converted_to_minutes(self):
        events = [_ioda_event(duration=7200)]  # 2 hours
        with patch("httpx.get") as mock:
            mock.side_effect = [
                _mock_response(200, _ioda_alerts_response([])),
                _mock_response(200, _ioda_events_response(events)),
            ]
            r = _tool().execute(mode="outages")
            assert "120.0min" in r.output

    def test_output_header_global(self):
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ioda_alerts_response([]))
            r = _tool().execute(mode="outages")
            assert "Internet Outage Monitor" in r.output

    def test_output_header_country_specific(self):
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ioda_alerts_response([]))
            r = _tool().execute(mode="outages", country="DE")
            assert "(DE)" in r.output


# ══════════════════════════════════════════════════════════════
# Test: Censorship mode (OONI)
# ══════════════════════════════════════════════════════════════


class TestCensorshipMode:
    def test_empty_result(self):
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ooni_aggregation_response([]))
            r = _tool().execute(mode="censorship", country="US")
            assert r.success
            assert "No OONI" in r.output

    def test_normal_aggregation(self):
        rows = [
            _ooni_agg_row(date=f"2026-03-{d:02d}", ok=100, anomaly=5)
            for d in range(1, 16)
        ]
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ooni_aggregation_response(rows))
            r = _tool().execute(mode="censorship", country="IR")
            assert r.success
            assert "IR" in r.output
            assert "anomaly rate" in r.output.lower()

    def test_anomaly_rate_computed(self):
        rows = [_ooni_agg_row(ok=50, anomaly=50)]  # 50% anomaly rate
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ooni_aggregation_response(rows))
            r = _tool().execute(mode="censorship", country="CN")
            assert r.success
            assert "50.0%" in r.output

    def test_heavy_blocking_alert(self):
        rows = [_ooni_agg_row(ok=10, anomaly=90)]  # 90% anomaly rate
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ooni_aggregation_response(rows))
            r = _tool().execute(mode="censorship", country="CN")
            assert "HEAVY BLOCKING" in r.output

    def test_trend_rising(self):
        # First half: low anomaly, second half: high anomaly
        rows = []
        for i in range(14):
            anom = 2 if i < 7 else 20
            rows.append(_ooni_agg_row(date=f"2026-03-{i+1:02d}", ok=100, anomaly=anom))
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ooni_aggregation_response(rows))
            r = _tool().execute(mode="censorship", country="IR")
            assert "RISING" in r.output

    def test_trend_falling(self):
        rows = []
        for i in range(14):
            anom = 20 if i < 7 else 2
            rows.append(_ooni_agg_row(date=f"2026-03-{i+1:02d}", ok=100, anomaly=anom))
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ooni_aggregation_response(rows))
            r = _tool().execute(mode="censorship", country="IR")
            assert "FALLING" in r.output

    def test_trend_stable(self):
        rows = [
            _ooni_agg_row(date=f"2026-03-{d:02d}", ok=100, anomaly=5)
            for d in range(1, 15)
        ]
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ooni_aggregation_response(rows))
            r = _tool().execute(mode="censorship", country="US")
            assert "STABLE" in r.output

    def test_different_test_types(self):
        for test_type in ["telegram", "tor", "whatsapp", "signal"]:
            with patch("httpx.get") as mock:
                mock.return_value = _mock_response(
                    200, _ooni_aggregation_response([_ooni_agg_row(ok=10, anomaly=5)])
                )
                r = _tool().execute(mode="censorship", country="RU", test=test_type)
                assert r.success
                assert test_type in r.output

    def test_http_failure(self):
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(500)
            r = _tool().execute(mode="censorship", country="US")
            assert not r.success

    def test_malformed_rows_skipped(self):
        rows = ["not a dict", _ooni_agg_row(ok=100, anomaly=10)]
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ooni_aggregation_response(rows))
            r = _tool().execute(mode="censorship", country="US")
            assert r.success

    def test_zero_total_measurements(self):
        rows = [_ooni_agg_row(ok=0, anomaly=0, confirmed=0)]
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ooni_aggregation_response(rows))
            r = _tool().execute(mode="censorship", country="US")
            assert r.success
            assert "0.0%" in r.output

    def test_warning_marker_on_high_anomaly(self):
        rows = [_ooni_agg_row(ok=50, anomaly=50)]
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ooni_aggregation_response(rows))
            r = _tool().execute(mode="censorship", country="IR")
            assert "⚠" in r.output


# ══════════════════════════════════════════════════════════════
# Test: Signals mode (IODA)
# ══════════════════════════════════════════════════════════════


class TestSignalsMode:
    def test_normal_signal(self):
        values = [0.95, 0.96, 0.94, 0.97]
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ioda_signals_response(values))
            r = _tool().execute(mode="signals", country="US")
            assert r.success
            assert "nominal" in r.output.lower()

    def test_warning_signal(self):
        values = [0.95, 0.70, 0.95]
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ioda_signals_response(values))
            r = _tool().execute(mode="signals", country="US")
            assert r.success
            # Last value is 0.95 = normal current state
            assert "nominal" in r.output.lower() or "WARNING" in r.output

    def test_critical_current(self):
        values = [0.95, 0.30]
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ioda_signals_response(values))
            r = _tool().execute(mode="signals", country="IR")
            assert r.success
            assert "CRITICAL" in r.output

    def test_drops_detected(self):
        values = [0.95, 0.60, 0.40, 0.95]  # two drops
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ioda_signals_response(values))
            r = _tool().execute(mode="signals", country="RU")
            assert r.success
            assert "Connectivity Drops" in r.output

    def test_all_none_values(self):
        values = [None, None, None]
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ioda_signals_response(values))
            r = _tool().execute(mode="signals", country="US")
            assert r.success
            assert "No valid" in r.output

    def test_mixed_none_and_valid(self):
        values = [None, 0.95, None, 0.90, None]
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ioda_signals_response(values))
            r = _tool().execute(mode="signals", country="US")
            assert r.success
            assert "2 data points" in r.output

    def test_list_values_filtered(self):
        """IODA sometimes returns lists instead of floats — should be filtered."""
        values = [[1, 2], 0.95, [3], 0.90]
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ioda_signals_response(values))
            r = _tool().execute(mode="signals", country="US")
            assert r.success
            assert "2 data points" in r.output

    def test_empty_data(self):
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, {"type": "signals", "data": []})
            r = _tool().execute(mode="signals", country="US")
            assert r.success
            assert "No gtr-norm" in r.output

    def test_no_gtr_norm_datasource(self):
        """Response has data but not gtr-norm."""
        data = {
            "type": "signals",
            "data": [
                [
                    {
                        "datasource": "bgp",
                        "values": [100, 200],
                        "step": 1800,
                        "from": 1700000000,
                    }
                ]
            ],
        }
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, data)
            r = _tool().execute(mode="signals", country="US")
            assert r.success
            assert "No gtr-norm" in r.output

    def test_http_failure(self):
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(500)
            r = _tool().execute(mode="signals", country="US")
            assert not r.success

    def test_step_displayed(self):
        values = [0.95, 0.94]
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(
                200, _ioda_signals_response(values, step=1800)
            )
            r = _tool().execute(mode="signals", country="US")
            assert "1800s" in r.output
            assert "30min" in r.output

    def test_min_max_avg(self):
        values = [0.80, 0.90, 1.00]
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ioda_signals_response(values))
            r = _tool().execute(mode="signals", country="US")
            assert "0.8000" in r.output  # min
            assert "1.0000" in r.output  # max
            assert "0.9000" in r.output  # avg


# ══════════════════════════════════════════════════════════════
# Test: Incidents mode (OONI)
# ══════════════════════════════════════════════════════════════


class TestIncidentsMode:
    def test_empty_incidents(self):
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ooni_incidents_response([]))
            r = _tool().execute(mode="incidents")
            assert r.success
            assert "No ongoing" in r.output

    def test_normal_incidents(self):
        incidents = [
            _ooni_incident("Russia blocked Telegram", ["RU"]),
            _ooni_incident("Gabon blocked social media", ["GA"]),
        ]
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ooni_incidents_response(incidents))
            r = _tool().execute(mode="incidents")
            assert r.success
            assert "Russia blocked Telegram" in r.output
            assert "Gabon" in r.output

    def test_country_frequency(self):
        incidents = [
            _ooni_incident("Event 1", ["RU"]),
            _ooni_incident("Event 2", ["RU"]),
            _ooni_incident("Event 3", ["CN"]),
        ]
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ooni_incidents_response(incidents))
            r = _tool().execute(mode="incidents")
            assert "Most Affected" in r.output
            assert "RU: 2" in r.output

    def test_multi_country_incident(self):
        incidents = [_ooni_incident("Regional block", ["IR", "IQ", "SY"])]
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ooni_incidents_response(incidents))
            r = _tool().execute(mode="incidents")
            assert "IR, IQ, SY" in r.output

    def test_http_failure(self):
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(500)
            r = _tool().execute(mode="incidents")
            assert not r.success

    def test_malformed_incident_skipped(self):
        incidents = ["not a dict", _ooni_incident("Valid", ["US"])]
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ooni_incidents_response(incidents))
            r = _tool().execute(mode="incidents")
            assert r.success
            assert "Valid" in r.output

    def test_limit_respected(self):
        incidents = [_ooni_incident(f"Event {i}", ["US"]) for i in range(50)]
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ooni_incidents_response(incidents))
            r = _tool().execute(mode="incidents", limit=5)
            # Should only show first 5
            assert "Event 4" in r.output
            # Event 10 should not appear in the main listing
            lines = [l for l in r.output.split("\n") if "Event" in l and "[" in l]
            assert len(lines) <= 5


# ══════════════════════════════════════════════════════════════
# Test: Cache interaction
# ══════════════════════════════════════════════════════════════


class TestCacheInteraction:
    def test_cache_hit(self):
        cache = MagicMock()
        cache.get.return_value = {"result": [_ooni_agg_row()]}
        tool = _tool(cache=cache)
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(
                200, _ooni_aggregation_response([_ooni_agg_row()])
            )
            tool.execute(mode="censorship", country="US")
        # Cache was checked
        assert cache.get.called

    def test_cache_miss_then_set(self):
        cache = MagicMock()
        cache.get.return_value = None
        tool = _tool(cache=cache)
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(
                200, _ooni_aggregation_response([_ooni_agg_row()])
            )
            tool.execute(mode="censorship", country="US")
        assert cache.set.called

    def test_cache_set_with_correct_ttl_ooni(self):
        cache = MagicMock()
        cache.get.return_value = None
        tool = _tool(cache=cache)
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(
                200, _ooni_aggregation_response([_ooni_agg_row()])
            )
            tool.execute(mode="censorship", country="US")
        # Check ttl was CACHE_OONI_AGGREGATION
        for call in cache.set.call_args_list:
            assert call[1].get("ttl") == CACHE_OONI_AGGREGATION

    def test_no_cache_still_works(self):
        tool = _tool(cache=None)
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(
                200, _ooni_aggregation_response([_ooni_agg_row()])
            )
            r = tool.execute(mode="censorship", country="US")
            assert r.success


# ══════════════════════════════════════════════════════════════
# Test: HTTP error handling
# ══════════════════════════════════════════════════════════════


class TestHTTPErrors:
    def test_connection_error(self):
        import httpx as httpx_mod

        with patch("httpx.get") as mock:
            mock.side_effect = httpx_mod.ConnectError("connection refused")
            r = _tool().execute(mode="incidents")
            assert not r.success

    def test_timeout_error(self):
        import httpx as httpx_mod

        with patch("httpx.get") as mock:
            mock.side_effect = httpx_mod.TimeoutException("timeout")
            r = _tool().execute(mode="incidents")
            assert not r.success

    def test_404_response(self):
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(404)
            r = _tool().execute(mode="incidents")
            assert not r.success

    def test_429_rate_limit(self):
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(429)
            r = _tool().execute(mode="incidents")
            assert not r.success

    def test_non_json_response(self):
        with patch("httpx.get") as mock:
            resp = _mock_response(200)
            resp.json.side_effect = ValueError("bad json")
            mock.return_value = resp
            r = _tool().execute(mode="incidents")
            assert not r.success


# ══════════════════════════════════════════════════════════════
# Test: Registry integration
# ══════════════════════════════════════════════════════════════


class TestRegistryIntegration:
    def test_cli_import(self):
        from agent.tools.internet_infrastructure import InternetInfrastructureTool

        tool = InternetInfrastructureTool()
        assert tool.name == "internet_infrastructure"

    def test_in_cli_registry(self):
        try:
            from agent.cli import build_tool_registry
        except ImportError:
            pytest.skip("cli dependencies not available in test env")
        registry = build_tool_registry()
        names = (
            [t.name for t in registry._tools.values()]
            if hasattr(registry, "_tools")
            else []
        )
        if not names:
            names = [t.name for t in getattr(registry, "tools", {}).values()]
        assert "internet_infrastructure" in names

    def test_bandit_arm_exists(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm_names = [a.name for a in DEFAULT_ARMS]
        assert "internet_infrastructure_monitor" in arm_names

    def test_bandit_arm_includes_new_tool(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm = next(
            a for a in DEFAULT_ARMS if a.name == "internet_infrastructure_monitor"
        )
        assert "internet_infrastructure" in arm.tools


# ══════════════════════════════════════════════════════════════
# Test: Output format
# ══════════════════════════════════════════════════════════════


class TestOutputFormat:
    def test_outages_header(self):
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ioda_alerts_response([]))
            r = _tool().execute(mode="outages")
            assert "Source: IODA" in r.output

    def test_censorship_header(self):
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(
                200, _ooni_aggregation_response([_ooni_agg_row()])
            )
            r = _tool().execute(mode="censorship", country="IR")
            assert "Source: OONI" in r.output

    def test_signals_header(self):
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ioda_signals_response([0.95]))
            r = _tool().execute(mode="signals", country="US")
            assert "Source: IODA" in r.output

    def test_incidents_header(self):
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(
                200, _ooni_incidents_response([_ooni_incident()])
            )
            r = _tool().execute(mode="incidents")
            assert "Source: OONI" in r.output


# ══════════════════════════════════════════════════════════════
# Test: Edge combinations
# ══════════════════════════════════════════════════════════════


class TestEdgeCombinations:
    def test_outages_both_sources_fail(self):
        """Both IODA endpoints return None — should gracefully return empty."""
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(500)
            r = _tool().execute(mode="outages")
            assert r.success
            assert "No outage" in r.output

    def test_signal_single_datapoint(self):
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(200, _ioda_signals_response([0.42]))
            r = _tool().execute(mode="signals", country="US")
            assert r.success
            assert "1 data points" in r.output

    def test_censorship_single_day(self):
        with patch("httpx.get") as mock:
            mock.return_value = _mock_response(
                200, _ooni_aggregation_response([_ooni_agg_row()])
            )
            r = _tool().execute(mode="censorship", country="US")
            assert r.success

    def test_outages_with_both_alerts_and_events(self):
        alerts = [_ioda_alert(code="IR", level="critical")]
        events = [_ioda_event(location="country/SY", score=10000)]
        with patch("httpx.get") as mock:
            mock.side_effect = [
                _mock_response(200, _ioda_alerts_response(alerts)),
                _mock_response(200, _ioda_events_response(events)),
            ]
            r = _tool().execute(mode="outages")
            assert r.success
            assert "Active Alerts" in r.output
            assert "Outage Events" in r.output
