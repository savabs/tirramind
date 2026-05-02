"""
Edge case tests for SupplyChainMonitorTool (BLS PPI + Import Prices).

Covers: mode validation, sector filtering, BLS multi-series fetch,
PPI signal computation (MoM, 3-month, cross-sector breadth, broad inflation),
import signal computation, pressure index scoring, cache interaction,
HTTP errors (429/500/timeout), empty data, malformed responses,
output formatting, _safe_float, _filter_ppi_series, tool metadata.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import httpx

from agent.tools.base import ToolResult
from agent.tools.supply_chain_monitor import (
    _BLS_BASE,
    _IMPORT_SERIES,
    _PPI_SERIES,
    VALID_MODES,
    SupplyChainMonitorTool,
    _compute_import_signals,
    _compute_ppi_signals,
    _compute_pressure_score,
    _fetch_bls_multi,
    _filter_ppi_series,
    _format_import_summary,
    _format_ppi_summary,
    _format_pressure_summary,
    _safe_float,
)

# ── Fixtures ──────────────────────────────────────────────────


def _tool(cache=None) -> SupplyChainMonitorTool:
    return SupplyChainMonitorTool(cache=cache)


def _mock_resp(body: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        text=json.dumps(body),
        request=httpx.Request("POST", _BLS_BASE),
    )


def _bls_multi_response(series_data: dict[str, list]) -> dict:
    """Build BLS multi-series response.
    series_data: {series_id: [{"year": ..., "period": ..., "value": ...}, ...]}
    """
    series = []
    for sid, data in series_data.items():
        series.append({"seriesID": sid, "data": data})

    return {
        "status": "REQUEST_SUCCEEDED",
        "responseTime": 42,
        "message": [],
        "Results": {"series": series},
    }


def _make_ppi_data(sid: str, values: list[float], start_year: str = "2025") -> list[dict]:
    """Build N months of BLS data entries."""
    return [
        {
            "year": start_year,
            "period": f"M{i + 1:02d}",
            "value": str(v),
        }
        for i, v in enumerate(values)
    ]


# ── 1. _safe_float ────────────────────────────────────────────


class TestSafeFloat:
    def test_normal(self):
        assert _safe_float("100.225") == 100.225

    def test_negative(self):
        assert _safe_float("-5.3") == -5.3

    def test_none(self):
        assert _safe_float(None) is None

    def test_empty(self):
        assert _safe_float("") is None

    def test_dot(self):
        assert _safe_float(".") is None

    def test_nan(self):
        assert _safe_float("NaN") is None

    def test_junk(self):
        assert _safe_float("n/a") is None

    def test_int(self):
        assert _safe_float(42) == 42.0


# ── 2. _filter_ppi_series ────────────────────────────────────


class TestFilterPpiSeries:
    def test_all(self):
        result = _filter_ppi_series("all")
        assert len(result) == len(_PPI_SERIES)

    def test_all_caps(self):
        result = _filter_ppi_series("ALL")
        assert len(result) == len(_PPI_SERIES)

    def test_single_sector(self):
        result = _filter_ppi_series("tech")
        assert len(result) == 2  # semiconductors + computers
        for sid in result:
            assert _PPI_SERIES[sid]["sector"] == "tech"

    def test_multi_sector(self):
        result = _filter_ppi_series("tech,materials")
        assert len(result) >= 3

    def test_unknown_sector(self):
        result = _filter_ppi_series("bogus")
        assert result == []

    def test_whitespace(self):
        result = _filter_ppi_series("  tech , materials  ")
        assert len(result) >= 3

    def test_empty(self):
        result = _filter_ppi_series("")
        assert result == []


# ── 3. Mode validation ───────────────────────────────────────


class TestModeValidation:
    def test_invalid_mode(self):
        r = _tool().execute(mode="bogus")
        assert not r.success
        assert "bogus" in r.output

    def test_missing_mode(self):
        r = _tool().execute()
        assert not r.success

    def test_valid_modes_match(self):
        assert {"producer_prices", "import_prices", "pressure_index"} == VALID_MODES


# ── 4. Parameter validation ──────────────────────────────────


class TestParameterValidation:
    def test_invalid_months(self):
        with patch("agent.tools.supply_chain_monitor._fetch_bls_multi") as m:
            m.return_value = ({}, None)
            r = _tool().execute(mode="producer_prices", months="abc")
            assert r.success  # gracefully defaults

    def test_months_clamped(self):
        with patch("agent.tools.supply_chain_monitor._fetch_bls_multi") as m:
            m.return_value = ({}, None)
            _tool().execute(mode="producer_prices", months=-5)

    def test_invalid_sectors(self):
        r = _tool().execute(mode="producer_prices", sectors="bogus_sector")
        assert not r.success
        assert "No PPI series" in r.output


# ── 5. BLS multi-series fetch ────────────────────────────────


class TestBlsMultiFetch:
    @patch("agent.tools.supply_chain_monitor.httpx.Client")
    def test_success(self, mock_client_cls):
        body = _bls_multi_response(
            {
                "PCU334413334413": _make_ppi_data("PCU334413334413", [30.0, 30.2, 30.5]),
                "PCU331110331110": _make_ppi_data("PCU331110331110", [280.0, 283.0, 285.0]),
            }
        )
        mock_resp = _mock_resp(body)
        mock_client_cls.return_value.__enter__ = MagicMock(
            return_value=MagicMock(post=MagicMock(return_value=mock_resp))
        )
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        data, err = _fetch_bls_multi(["PCU334413334413", "PCU331110331110"], 6)
        assert err is None
        assert "PCU334413334413" in data
        assert len(data["PCU334413334413"]) == 3

    @patch("agent.tools.supply_chain_monitor.httpx.Client")
    def test_timeout(self, mock_client_cls):
        mock_client_cls.return_value.__enter__ = MagicMock(
            return_value=MagicMock(post=MagicMock(side_effect=httpx.TimeoutException("timeout")))
        )
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        data, err = _fetch_bls_multi(["PCU334413334413"], 6)
        assert err is not None
        assert "timed out" in err

    @patch("agent.tools.supply_chain_monitor.httpx.Client")
    def test_rate_limit(self, mock_client_cls):
        mock_resp = _mock_resp({}, status=429)
        mock_client_cls.return_value.__enter__ = MagicMock(
            return_value=MagicMock(post=MagicMock(return_value=mock_resp))
        )
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        data, err = _fetch_bls_multi(["PCU334413334413"], 6)
        assert err is not None
        assert "rate limit" in err.lower()

    @patch("agent.tools.supply_chain_monitor.httpx.Client")
    def test_request_failed(self, mock_client_cls):
        body = {
            "status": "REQUEST_NOT_PROCESSED",
            "message": ["Invalid series"],
            "Results": {},
        }
        mock_resp = _mock_resp(body)
        mock_client_cls.return_value.__enter__ = MagicMock(
            return_value=MagicMock(post=MagicMock(return_value=mock_resp))
        )
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        data, err = _fetch_bls_multi(["PCU334413334413"], 6)
        assert err is not None
        assert "failed" in err.lower()

    @patch("agent.tools.supply_chain_monitor.httpx.Client")
    def test_m13_skipped(self, mock_client_cls):
        raw_data = [
            {"year": "2025", "period": "M13", "value": "999"},
            {"year": "2025", "period": "M01", "value": "30.0"},
        ]
        body = _bls_multi_response({"PCU334413334413": raw_data})
        mock_resp = _mock_resp(body)
        mock_client_cls.return_value.__enter__ = MagicMock(
            return_value=MagicMock(post=MagicMock(return_value=mock_resp))
        )
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        data, err = _fetch_bls_multi(["PCU334413334413"], 6)
        assert err is None
        assert len(data["PCU334413334413"]) == 1

    @patch("agent.tools.supply_chain_monitor.httpx.Client")
    def test_bad_json(self, mock_client_cls):
        resp = httpx.Response(
            status_code=200,
            text="not json",
            request=httpx.Request("POST", _BLS_BASE),
        )
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=MagicMock(post=MagicMock(return_value=resp)))
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        data, err = _fetch_bls_multi(["PCU334413334413"], 6)
        assert err is not None
        assert "parse" in err.lower()


# ── 6. PPI signal computation ───────────────────────────────


class TestPpiSignals:
    def test_basic_signals(self):
        data = {
            "PCU334413334413": [
                {"year": "2025", "period": "M01", "value": 30.0},
                {"year": "2025", "period": "M02", "value": 30.2},
                {"year": "2025", "period": "M03", "value": 31.0},
            ],
        }
        sig = _compute_ppi_signals(data)
        series_sig = sig["series"]["PCU334413334413"]
        assert series_sig["latest"] == 31.0
        assert series_sig["mom_pct"] is not None

    def test_rising_alert(self):
        data = {
            "PCU334413334413": [
                {"year": "2025", "period": "M02", "value": 30.0},
                {"year": "2025", "period": "M03", "value": 31.5},  # ~5% rise
            ],
        }
        sig = _compute_ppi_signals(data)
        assert sig["series"]["PCU334413334413"]["alert"] is not None
        assert "RISING" in sig["series"]["PCU334413334413"]["alert"]

    def test_falling_alert(self):
        data = {
            "PCU334413334413": [
                {"year": "2025", "period": "M02", "value": 30.0},
                {"year": "2025", "period": "M03", "value": 28.5},  # -5%
            ],
        }
        sig = _compute_ppi_signals(data)
        assert "FALLING" in sig["series"]["PCU334413334413"]["alert"]

    def test_three_month_trend(self):
        data = {
            "PCU334413334413": [
                {"year": "2025", "period": "M01", "value": 30.0},
                {"year": "2025", "period": "M02", "value": 30.5},
                {"year": "2025", "period": "M03", "value": 31.0},
                {"year": "2025", "period": "M04", "value": 32.0},
            ],
        }
        sig = _compute_ppi_signals(data)
        assert sig["series"]["PCU334413334413"]["three_month_pct"] is not None

    def test_broad_inflation(self):
        # 4+ sectors all rising
        data = {}
        for i, sid in enumerate(list(_PPI_SERIES.keys())[:5]):
            data[sid] = [
                {"year": "2025", "period": "M02", "value": 100.0},
                {"year": "2025", "period": "M03", "value": 102.0},  # +2%
            ]
        sig = _compute_ppi_signals(data)
        assert sig["broad_inflation"] is True
        assert sig["sectors_rising"] >= 4

    def test_empty_data(self):
        sig = _compute_ppi_signals({})
        assert sig["status"] == "NO_DATA"

    def test_single_record(self):
        data = {
            "PCU334413334413": [
                {"year": "2025", "period": "M03", "value": 30.0},
            ],
        }
        sig = _compute_ppi_signals(data)
        assert sig["series"]["PCU334413334413"]["mom_pct"] is None
        assert sig["series"]["PCU334413334413"]["three_month_pct"] is None

    def test_no_records_for_series(self):
        data = {"PCU334413334413": []}
        sig = _compute_ppi_signals(data)
        assert sig["series"]["PCU334413334413"]["latest"] is None


# ── 7. Import signal computation ────────────────────────────


class TestImportSignals:
    def test_basic(self):
        data = {
            "EIUIR": [
                {"year": "2025", "period": "M02", "value": 143.0},
                {"year": "2025", "period": "M03", "value": 144.0},
            ],
        }
        sig = _compute_import_signals(data)
        assert sig["series"]["EIUIR"]["latest"] == 144.0
        assert sig["series"]["EIUIR"]["mom_pct"] is not None

    def test_empty(self):
        sig = _compute_import_signals({})
        assert sig["status"] == "NO_DATA"

    def test_single_record(self):
        data = {
            "EIUIR": [{"year": "2025", "period": "M03", "value": 144.0}],
        }
        sig = _compute_import_signals(data)
        assert sig["series"]["EIUIR"]["mom_pct"] is None


# ── 8. Pressure score ───────────────────────────────────────


class TestPressureScore:
    def test_high_pressure(self):
        ppi_sig = {
            "series": {sid: {"latest": 100 + i, "mom_pct": 2.5} for i, sid in enumerate(_PPI_SERIES)},
            "avg_mom_pct": 2.5,
            "sectors_rising": 6,
            "sectors_falling": 0,
            "broad_inflation": True,
        }
        import_sig = {
            "series": {"EIUIR": {"latest": 150, "mom_pct": 2.0}},
        }
        pressure = _compute_pressure_score(ppi_sig, import_sig)
        assert pressure["score"] > 50
        assert "HIGH" in pressure["level"] or "MODERATE" in pressure["level"]

    def test_low_pressure(self):
        ppi_sig = {
            "series": {"PCU334413334413": {"latest": 30, "mom_pct": 0.1}},
            "avg_mom_pct": 0.1,
            "sectors_rising": 0,
            "sectors_falling": 0,
            "broad_inflation": False,
        }
        import_sig = {
            "series": {"EIUIR": {"latest": 144, "mom_pct": -0.1}},
        }
        pressure = _compute_pressure_score(ppi_sig, import_sig)
        assert pressure["score"] < 40

    def test_deflationary(self):
        ppi_sig = {
            "series": {},
            "avg_mom_pct": -2.0,
            "sectors_rising": 0,
            "sectors_falling": 4,
            "broad_inflation": False,
        }
        import_sig = {
            "series": {"EIUIR": {"latest": 140, "mom_pct": -2.0}},
        }
        pressure = _compute_pressure_score(ppi_sig, import_sig)
        assert pressure["score"] < 20

    def test_score_clamped_0_100(self):
        ppi_sig = {
            "series": {sid: {"latest": 100, "mom_pct": 10.0} for sid in _PPI_SERIES},
            "avg_mom_pct": 10.0,
            "sectors_rising": 6,
            "sectors_falling": 0,
            "broad_inflation": True,
        }
        import_sig = {"series": {"EIUIR": {"latest": 200, "mom_pct": 10.0}}}
        pressure = _compute_pressure_score(ppi_sig, import_sig)
        assert 0 <= pressure["score"] <= 100

    def test_empty_inputs(self):
        pressure = _compute_pressure_score({"series": {}}, {"series": {}})
        assert pressure["score"] == 0


# ── 9. Cache interaction ────────────────────────────────────


class TestCache:
    def test_cache_hit(self):
        cache = MagicMock()
        cache.get.return_value = {"output": "cached", "data": {"mode": "producer_prices"}}
        r = _tool(cache).execute(mode="producer_prices")
        assert r.success
        assert "cached" in r.output

    def test_cache_miss_then_set(self):
        cache = MagicMock()
        cache.get.return_value = None

        with patch("agent.tools.supply_chain_monitor._fetch_bls_multi") as m:
            m.return_value = (
                {"PCU334413334413": [{"year": "2025", "period": "M03", "value": 30.0}]},
                None,
            )
            r = _tool(cache).execute(mode="producer_prices")

        assert r.success
        cache.set.assert_called_once()

    def test_no_cache(self):
        with patch("agent.tools.supply_chain_monitor._fetch_bls_multi") as m:
            m.return_value = (
                {"PCU334413334413": [{"year": "2025", "period": "M03", "value": 30.0}]},
                None,
            )
            r = _tool(None).execute(mode="producer_prices")
            assert r.success

    def test_import_prices_cache(self):
        cache = MagicMock()
        cache.get.return_value = {"output": "import cached", "data": {}}
        r = _tool(cache).execute(mode="import_prices")
        assert r.success
        assert "import cached" in r.output

    def test_pressure_index_cache(self):
        cache = MagicMock()
        cache.get.return_value = {"output": "pressure cached", "data": {}}
        r = _tool(cache).execute(mode="pressure_index")
        assert r.success


# ── 10. Formatting ───────────────────────────────────────────


class TestFormatting:
    def test_ppi_summary(self):
        data = {
            "PCU334413334413": [
                {"year": "2025", "period": "M02", "value": 30.0},
                {"year": "2025", "period": "M03", "value": 30.5},
            ],
        }
        sig = _compute_ppi_signals(data)
        summary = _format_ppi_summary(data, sig, 6)
        assert "Producer Price" in summary
        assert "Semiconductor" in summary

    def test_import_summary(self):
        data = {
            "EIUIR": [
                {"year": "2025", "period": "M02", "value": 143.0},
                {"year": "2025", "period": "M03", "value": 144.0},
            ],
        }
        sig = _compute_import_signals(data)
        summary = _format_import_summary(data, sig, 6)
        assert "Import Price" in summary
        assert "All Imports" in summary

    def test_pressure_summary(self):
        ppi_sig = {
            "series": {"PCU334413334413": {"label": "Semiconductors", "mom_pct": 1.5}},
            "avg_mom_pct": 1.5,
            "broad_inflation": False,
        }
        import_sig = {
            "series": {"EIUIR": {"label": "All Imports", "mom_pct": 0.5}},
        }
        pressure = {"score": 35.0, "level": "MODERATE", "components": [20, 15]}
        summary = _format_pressure_summary(ppi_sig, import_sig, pressure, 6)
        assert "Pressure Index" in summary
        assert "35.0" in summary


# ── 11. Tool metadata ───────────────────────────────────────


class TestToolMetadata:
    def test_name(self):
        assert _tool().name == "supply_chain_prices"

    def test_description(self):
        desc = _tool().description
        assert "producer_prices" in desc
        assert "import_prices" in desc
        assert "pressure_index" in desc

    def test_parameters_structure(self):
        params = _tool().parameters
        assert "properties" in params
        assert "mode" in params["properties"]
        assert "months" in params["properties"]
        assert "sectors" in params["properties"]

    def test_returns_tool_result(self):
        r = _tool().execute(mode="invalid")
        assert isinstance(r, ToolResult)


# ── 12. PPI series constants ────────────────────────────────


class TestConstants:
    def test_ppi_series_has_six(self):
        assert len(_PPI_SERIES) == 6

    def test_import_series_has_three(self):
        assert len(_IMPORT_SERIES) == 3

    def test_all_ppi_have_label_and_sector(self):
        for sid, info in _PPI_SERIES.items():
            assert "label" in info
            assert "sector" in info
            assert info["sector"] in {"tech", "industrial", "materials", "energy", "chemicals"}

    def test_valid_modes_frozen(self):
        assert isinstance(VALID_MODES, frozenset)


# ── 13. End-to-end mode execution ───────────────────────────


class TestEndToEnd:
    @patch("agent.tools.supply_chain_monitor._fetch_bls_multi")
    def test_producer_prices_mode(self, mock_fetch):
        mock_fetch.return_value = (
            {
                "PCU334413334413": [
                    {"year": "2025", "period": "M02", "value": 30.0},
                    {"year": "2025", "period": "M03", "value": 30.5},
                ],
            },
            None,
        )
        r = _tool().execute(mode="producer_prices", sectors="tech")
        assert r.success
        assert r.data["mode"] == "producer_prices"

    @patch("agent.tools.supply_chain_monitor._fetch_bls_multi")
    def test_import_prices_mode(self, mock_fetch):
        mock_fetch.return_value = (
            {
                "EIUIR": [
                    {"year": "2025", "period": "M03", "value": 144.0},
                ],
            },
            None,
        )
        r = _tool().execute(mode="import_prices")
        assert r.success
        assert r.data["mode"] == "import_prices"

    @patch("agent.tools.supply_chain_monitor._fetch_bls_multi")
    def test_pressure_index_mode(self, mock_fetch):
        combined_data = {}
        for sid in _PPI_SERIES:
            combined_data[sid] = [
                {"year": "2025", "period": "M02", "value": 100.0},
                {"year": "2025", "period": "M03", "value": 101.0},
            ]
        combined_data["EIUIR"] = [
            {"year": "2025", "period": "M02", "value": 143.0},
            {"year": "2025", "period": "M03", "value": 144.0},
        ]
        mock_fetch.return_value = (combined_data, None)

        r = _tool().execute(mode="pressure_index")
        assert r.success
        assert r.data["mode"] == "pressure_index"
        assert "pressure" in r.data
        assert 0 <= r.data["pressure"]["score"] <= 100

    @patch("agent.tools.supply_chain_monitor._fetch_bls_multi")
    def test_fetch_error_propagates(self, mock_fetch):
        mock_fetch.return_value = ({}, "BLS down")
        r = _tool().execute(mode="producer_prices")
        assert not r.success
        assert "BLS down" in r.output
