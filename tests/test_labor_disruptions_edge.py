"""
Edge case tests for LaborDisruptionsTool (BLS Work Stoppages).

Covers: mode validation, year validation, BLS API fetch, response parsing
(WSU001/WSU002), signal computation (workers, idle_days, overview, trends,
alerts), cache interaction, HTTP errors (429/500/timeout), empty data,
malformed responses, output formatting, _safe_float, _parse_bls_records,
footnotes/preliminary flag, tool metadata, registry + bandit integration.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.tools.labor_disruptions import (
    LaborDisruptionsTool,
    VALID_MODES,
    SERIES_WORKERS,
    SERIES_IDLE,
    _BLS_BASE,
    _CACHE_TTL,
    _fetch_bls_series,
    _parse_bls_records,
    _safe_float,
    _compute_single_signals,
    _compute_overview_signals,
    _format_single_summary,
    _format_overview_summary,
)
from agent.tools.base import ToolResult


# ── Fixtures ──────────────────────────────────────────────────


def _tool(cache=None) -> LaborDisruptionsTool:
    return LaborDisruptionsTool(cache=cache)


def _mock_resp(body: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        text=json.dumps(body),
        request=httpx.Request("POST", _BLS_BASE),
    )


def _bls_response(series_id: str, data: list) -> dict:
    return {
        "status": "REQUEST_SUCCEEDED",
        "responseTime": 42,
        "message": [],
        "Results": {
            "series": [
                {"seriesID": series_id, "data": data},
            ]
        },
    }


def _bls_record(
    year: str = "2024",
    period: str = "M06",
    value: str = "92.3",
    period_name: str = "June",
    footnotes: list | None = None,
) -> dict:
    return {
        "year": year,
        "period": period,
        "value": value,
        "periodName": period_name,
        "footnotes": footnotes or [{}],
    }


# ── Tool metadata ────────────────────────────────────────────


class TestToolMetadata:
    def test_name(self):
        assert _tool().name == "labor_disruptions"

    def test_description_non_empty(self):
        desc = _tool().description
        assert len(desc) > 50
        assert "BLS" in desc or "labor" in desc.lower()

    def test_parameters_schema(self):
        params = _tool().parameters
        assert params["type"] == "object"
        assert "mode" in params["properties"]
        assert params["required"] == ["mode"]

    def test_valid_modes_match_schema(self):
        enum = _tool().parameters["properties"]["mode"]["enum"]
        assert set(enum) == VALID_MODES

    def test_has_start_year_end_year_params(self):
        props = _tool().parameters["properties"]
        assert "start_year" in props
        assert "end_year" in props

    def test_constants(self):
        assert SERIES_WORKERS == "WSU001"
        assert SERIES_IDLE == "WSU002"


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
        result = _tool().execute(mode="Work_Stoppages")
        assert not result.success


# ── Year validation ──────────────────────────────────────────


class TestYearValidation:
    @patch("agent.tools.labor_disruptions._fetch_bls_series")
    def test_start_after_end(self, mock_fetch):
        result = _tool().execute(mode="work_stoppages", start_year=2025, end_year=2020)
        assert not result.success
        assert "cannot be after" in result.output

    @patch("agent.tools.labor_disruptions._fetch_bls_series")
    def test_clamp_start_to_1993(self, mock_fetch):
        mock_fetch.return_value = ([], None)
        result = _tool().execute(mode="work_stoppages", start_year=1990, end_year=1995)
        # Should clamp start to 1993, not error
        assert mock_fetch.called
        call_args = mock_fetch.call_args
        assert call_args[0][1] == 1993  # start_year arg

    @patch("agent.tools.labor_disruptions._fetch_bls_series")
    def test_default_span(self, mock_fetch):
        mock_fetch.return_value = ([], None)
        _tool().execute(mode="work_stoppages")
        assert mock_fetch.called


# ── _safe_float ──────────────────────────────────────────────


class TestSafeFloat:
    def test_normal_float(self):
        assert _safe_float("92.3") == 92.3

    def test_integer_string(self):
        assert _safe_float("100") == 100.0

    def test_zero(self):
        assert _safe_float("0") == 0.0

    def test_none(self):
        assert _safe_float(None) is None

    def test_empty_string(self):
        assert _safe_float("") is None

    def test_non_numeric(self):
        assert _safe_float("N/A") is None

    def test_dash(self):
        assert _safe_float("-") is None

    def test_negative(self):
        assert _safe_float("-5.3") == -5.3

    def test_bool(self):
        # bool is a subclass of int in Python
        assert _safe_float(True) == 1.0

    def test_list(self):
        assert _safe_float([1, 2]) is None


# ── _parse_bls_records ───────────────────────────────────────


class TestParseBLSRecords:
    def test_basic_parsing(self):
        raw = [_bls_record(), _bls_record(year="2024", period="M07", value="110.5")]
        records = _parse_bls_records(raw, "WSU001")
        assert len(records) == 2
        assert records[0]["value"] == 92.3
        assert records[1]["value"] == 110.5

    def test_chronological_sort(self):
        raw = [
            _bls_record(year="2024", period="M12", value="50"),
            _bls_record(year="2024", period="M01", value="10"),
            _bls_record(year="2023", period="M06", value="80"),
        ]
        records = _parse_bls_records(raw, "WSU001")
        assert records[0]["year"] == "2023"
        assert records[-1]["period"] == "M12"

    def test_skip_invalid_values(self):
        raw = [
            _bls_record(value="100"),
            _bls_record(value="N/A"),
            _bls_record(value=""),
        ]
        records = _parse_bls_records(raw, "WSU001")
        assert len(records) == 1

    def test_preliminary_flag(self):
        raw = [_bls_record(footnotes=[{"code": "P", "text": "Preliminary"}])]
        records = _parse_bls_records(raw, "WSU001")
        assert records[0]["preliminary"] is True

    def test_non_preliminary(self):
        raw = [_bls_record(footnotes=[{}])]
        records = _parse_bls_records(raw, "WSU001")
        assert records[0]["preliminary"] is False

    def test_footnotes_extracted(self):
        raw = [_bls_record(footnotes=[{"code": "R", "text": "Revised"}])]
        records = _parse_bls_records(raw, "WSU001")
        assert len(records[0]["footnotes"]) == 1
        assert "Revised" in records[0]["footnotes"][0]

    def test_empty_input(self):
        records = _parse_bls_records([], "WSU001")
        assert records == []

    def test_series_id_stored(self):
        raw = [_bls_record()]
        records = _parse_bls_records(raw, "WSU001")
        assert records[0]["series_id"] == "WSU001"


# ── Signal computation: single series ────────────────────────


class TestComputeSingleSignals:
    def test_empty_records(self):
        s = _compute_single_signals([], "workers")
        assert s["status"] == "NO_DATA"
        assert s["alert"] is None

    def test_workers_critical(self):
        records = [{"value": 600.0}]
        s = _compute_single_signals(records, "workers")
        assert "CRITICAL" in s["alert"]

    def test_workers_warning(self):
        records = [{"value": 200.0}]
        s = _compute_single_signals(records, "workers")
        assert "WARNING" in s["alert"]

    def test_workers_notice(self):
        records = [{"value": 10.0}]
        s = _compute_single_signals(records, "workers")
        assert "NOTICE" in s["alert"]

    def test_workers_zero_no_alert(self):
        records = [{"value": 0.0}]
        s = _compute_single_signals(records, "workers")
        assert s["alert"] is None

    def test_idle_critical(self):
        records = [{"value": 15000.0}]
        s = _compute_single_signals(records, "idle_days")
        assert "CRITICAL" in s["alert"]

    def test_idle_warning(self):
        records = [{"value": 5000.0}]
        s = _compute_single_signals(records, "idle_days")
        assert "WARNING" in s["alert"]

    def test_idle_notice(self):
        records = [{"value": 50.0}]
        s = _compute_single_signals(records, "idle_days")
        assert "NOTICE" in s["alert"]

    def test_idle_zero_no_alert(self):
        records = [{"value": 0.0}]
        s = _compute_single_signals(records, "idle_days")
        assert s["alert"] is None

    def test_active_months_count(self):
        records = [{"value": 10.0}, {"value": 0.0}, {"value": 5.0}]
        s = _compute_single_signals(records, "workers")
        assert s["active_months"] == 2
        assert s["total_months"] == 3

    def test_trend_escalating(self):
        # 12 months: first 6 low, last 6 high
        records = [{"value": 1.0}] * 6 + [{"value": 20.0}] * 6
        s = _compute_single_signals(records, "workers")
        assert s["trend"] == "ESCALATING"

    def test_trend_declining(self):
        records = [{"value": 100.0}] * 6 + [{"value": 5.0}] * 6
        s = _compute_single_signals(records, "workers")
        assert s["trend"] == "DECLINING"

    def test_trend_stable(self):
        records = [{"value": 50.0}] * 12
        s = _compute_single_signals(records, "workers")
        assert s["trend"] == "STABLE"

    def test_trend_insufficient(self):
        records = [{"value": 50.0}] * 5
        s = _compute_single_signals(records, "workers")
        assert s["trend"] == "INSUFFICIENT_DATA"

    def test_trend_new_activity(self):
        records = [{"value": 0.0}] * 6 + [{"value": 10.0}] * 6
        s = _compute_single_signals(records, "workers")
        # prior_avg == 0 → NEW_ACTIVITY
        assert s["trend"] == "NEW_ACTIVITY"

    def test_trend_quiet(self):
        records = [{"value": 0.0}] * 12
        s = _compute_single_signals(records, "workers")
        assert s["trend"] == "QUIET"

    def test_unknown_label_no_alert(self):
        records = [{"value": 999.0}]
        s = _compute_single_signals(records, "unknown")
        assert s["alert"] is None


# ── Signal computation: overview ─────────────────────────────


class TestComputeOverviewSignals:
    def test_overview_basic(self):
        workers = [{"value": 50.0}]
        idle = [{"value": 200.0}]
        s = _compute_overview_signals(workers, idle)
        assert s["workers"]["latest_value"] == 50.0
        assert s["idle_days"]["latest_value"] == 200.0
        assert s["intensity_ratio"] == 4.0

    def test_zero_workers_no_intensity(self):
        workers = [{"value": 0.0}]
        idle = [{"value": 100.0}]
        s = _compute_overview_signals(workers, idle)
        assert s["intensity_ratio"] is None

    def test_empty_workers(self):
        s = _compute_overview_signals([], [{"value": 100.0}])
        assert s["workers"]["status"] == "NO_DATA"

    def test_consecutive_active(self):
        workers = [{"value": 0.0}, {"value": 10.0}, {"value": 5.0}, {"value": 20.0}]
        idle = [{"value": 0.0}] * 4
        s = _compute_overview_signals(workers, idle)
        assert s["consecutive_active_months"] == 3

    def test_consecutive_none_active(self):
        workers = [{"value": 10.0}, {"value": 0.0}]
        idle = [{"value": 0.0}] * 2
        s = _compute_overview_signals(workers, idle)
        assert s["consecutive_active_months"] == 0

    def test_combined_alert_from_workers(self):
        workers = [{"value": 600.0}]
        idle = [{"value": 0.0}]
        s = _compute_overview_signals(workers, idle)
        assert "CRITICAL" in s["combined_alert"]


# ── Formatting ───────────────────────────────────────────────


class TestFormatting:
    def test_single_empty(self):
        text = _format_single_summary([], {"status": "NO_DATA"}, "workers", 2020, 2024)
        assert "No data" in text

    def test_single_with_data(self):
        records = [
            {"year": "2024", "period": "M01", "value": 10.0, "preliminary": False},
        ]
        signals = _compute_single_signals(records, "workers")
        text = _format_single_summary(records, signals, "workers", 2024, 2024)
        assert "workers" in text.lower() or "BLS" in text
        assert "2024" in text

    def test_overview_format(self):
        workers = [{"value": 20.0}]
        idle = [{"value": 80.0}]
        signals = _compute_overview_signals(workers, idle)
        text = _format_overview_summary(workers, idle, signals, 2020, 2024)
        assert "Overview" in text
        assert "Workers" in text or "workers" in text
        assert "intensity" in text.lower()


# ── BLS fetch (mocked HTTP) ─────────────────────────────────


class TestFetchBLS:
    @patch("agent.tools.labor_disruptions.httpx.Client")
    def test_success(self, mock_client_cls):
        body = _bls_response("WSU001", [_bls_record()])
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_resp(body)
        mock_client_cls.return_value = mock_client

        records, err = _fetch_bls_series("WSU001", 2020, 2024)
        assert err is None
        assert len(records) == 1

    @patch("agent.tools.labor_disruptions.httpx.Client")
    def test_timeout(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = httpx.TimeoutException("timeout")
        mock_client_cls.return_value = mock_client

        records, err = _fetch_bls_series("WSU001", 2020, 2024)
        assert err is not None
        assert "timed out" in err

    @patch("agent.tools.labor_disruptions.httpx.Client")
    def test_rate_limit(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_resp({}, 429)
        mock_client_cls.return_value = mock_client

        records, err = _fetch_bls_series("WSU001", 2020, 2024)
        assert "rate limit" in err.lower()

    @patch("agent.tools.labor_disruptions.httpx.Client")
    def test_500_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_resp({}, 500)
        mock_client_cls.return_value = mock_client

        records, err = _fetch_bls_series("WSU001", 2020, 2024)
        assert "500" in err

    @patch("agent.tools.labor_disruptions.httpx.Client")
    def test_request_failed_status(self, mock_client_cls):
        body = {
            "status": "REQUEST_NOT_PROCESSED",
            "message": ["Invalid series"],
            "Results": {},
        }
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_resp(body)
        mock_client_cls.return_value = mock_client

        records, err = _fetch_bls_series("BOGUS", 2020, 2024)
        assert err is not None
        assert "failed" in err.lower()

    @patch("agent.tools.labor_disruptions.httpx.Client")
    def test_empty_series(self, mock_client_cls):
        body = {
            "status": "REQUEST_SUCCEEDED",
            "Results": {"series": []},
        }
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_resp(body)
        mock_client_cls.return_value = mock_client

        records, err = _fetch_bls_series("WSU001", 2020, 2024)
        assert err is not None
        assert "No series" in err

    @patch("agent.tools.labor_disruptions.httpx.Client")
    def test_malformed_json(self, mock_client_cls):
        mock_resp = httpx.Response(
            status_code=200,
            text="not json at all",
            request=httpx.Request("POST", _BLS_BASE),
        )
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        records, err = _fetch_bls_series("WSU001", 2020, 2024)
        assert err is not None
        assert "parse" in err.lower()

    @patch("agent.tools.labor_disruptions.httpx.Client")
    def test_http_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = httpx.ConnectError("refused")
        mock_client_cls.return_value = mock_client

        records, err = _fetch_bls_series("WSU001", 2020, 2024)
        assert err is not None
        assert "error" in err.lower()


# ── Cache interaction ────────────────────────────────────────


class TestCache:
    @patch("agent.tools.labor_disruptions._fetch_bls_series")
    def test_cache_hit(self, mock_fetch):
        cache = MagicMock()
        cache.get.return_value = {
            "output": "cached summary",
            "data": {"records": [], "signals": {}},
        }
        tool = _tool(cache=cache)
        result = tool.execute(mode="work_stoppages")
        assert result.success
        assert "cached" in result.output
        mock_fetch.assert_not_called()

    @patch("agent.tools.labor_disruptions._fetch_bls_series")
    def test_cache_miss(self, mock_fetch):
        mock_fetch.return_value = ([], None)
        cache = MagicMock()
        cache.get.return_value = None
        tool = _tool(cache=cache)
        result = tool.execute(mode="work_stoppages")
        assert result.success
        cache.set.assert_called_once()

    @patch("agent.tools.labor_disruptions._fetch_bls_series")
    def test_no_cache(self, mock_fetch):
        mock_fetch.return_value = ([], None)
        result = _tool(cache=None).execute(mode="work_stoppages")
        assert result.success

    @patch("agent.tools.labor_disruptions._fetch_bls_series")
    def test_overview_cache_hit(self, mock_fetch):
        cache = MagicMock()
        cache.get.return_value = {
            "output": "cached overview",
            "data": {"workers": [], "idle_days": [], "signals": {}},
        }
        tool = _tool(cache=cache)
        result = tool.execute(mode="overview")
        assert result.success
        assert "cached" in result.output
        mock_fetch.assert_not_called()


# ── End-to-end with mocked fetch ─────────────────────────────


class TestEndToEnd:
    @patch("agent.tools.labor_disruptions._fetch_bls_series")
    def test_work_stoppages_success(self, mock_fetch):
        records = [
            {"year": "2024", "period": "M01", "value": 50.0, "preliminary": False,
             "series_id": "WSU001", "period_name": "January", "footnotes": []},
            {"year": "2024", "period": "M02", "value": 75.0, "preliminary": True,
             "series_id": "WSU001", "period_name": "February", "footnotes": []},
        ]
        mock_fetch.return_value = (records, None)
        result = _tool().execute(mode="work_stoppages")
        assert result.success
        assert result.data["count"] == 2

    @patch("agent.tools.labor_disruptions._fetch_bls_series")
    def test_overview_success(self, mock_fetch):
        workers = [
            {"year": "2024", "period": "M01", "value": 50.0, "preliminary": False,
             "series_id": "WSU001", "period_name": "January", "footnotes": []},
        ]
        idle = [
            {"year": "2024", "period": "M01", "value": 200.0, "preliminary": False,
             "series_id": "WSU002", "period_name": "January", "footnotes": []},
        ]
        mock_fetch.side_effect = [(workers, None), (idle, None)]
        result = _tool().execute(mode="overview")
        assert result.success
        assert "workers_count" in result.data

    @patch("agent.tools.labor_disruptions._fetch_bls_series")
    def test_idle_days_success(self, mock_fetch):
        records = [
            {"year": "2024", "period": "M05", "value": 1500.0, "preliminary": False,
             "series_id": "WSU002", "period_name": "May", "footnotes": []},
        ]
        mock_fetch.return_value = (records, None)
        result = _tool().execute(mode="idle_days")
        assert result.success
        assert result.data["label"] == "idle_days"

    @patch("agent.tools.labor_disruptions._fetch_bls_series")
    def test_fetch_error_propagates(self, mock_fetch):
        mock_fetch.return_value = ([], "API exploded")
        result = _tool().execute(mode="work_stoppages")
        assert not result.success
        assert "API exploded" in result.output

    @patch("agent.tools.labor_disruptions._fetch_bls_series")
    def test_overview_first_fetch_fails(self, mock_fetch):
        mock_fetch.return_value = ([], "Workers series: timeout")
        result = _tool().execute(mode="overview")
        assert not result.success
        assert "Workers" in result.output


# ── Registry and Bandit integration ──────────────────────────


class TestRegistryIntegration:
    def test_tool_name_in_imports(self):
        from agent.tools.labor_disruptions import LaborDisruptionsTool
        assert LaborDisruptionsTool is not None

    def test_bandit_arm_exists(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm_names = [a.name for a in DEFAULT_ARMS]
        assert "labor_disruption_monitor" in arm_names

    def test_bandit_arm_references_tool(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm = next(a for a in DEFAULT_ARMS if a.name == "labor_disruption_monitor")
        assert "labor_disruptions" in arm.tools
