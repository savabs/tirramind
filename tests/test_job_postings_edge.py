"""
Edge-case tests for JobPostingsTool (7b-F).

Coverage targets:
- Invalid / missing / boundary parameters
- FRED vs BLS fallback logic
- Empty / malformed API responses
- Cache hit / miss paths
- JOLTS data parsing, quits/layoffs ratio
- Market tightness computation
- Sector mode with BLS API
- labor_market mode requires FRED key
- HTTP errors, timeouts
- Mode validation
- Integration: tool count = 41, arm count = 29
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.tools.job_postings import (
    _CACHE_TTL,
    _JOLTS_SERIES,
    _LABOR_SERIES,
    _SECTOR_SERIES,
    VALID_MODES,
    JobPostingsTool,
    _compute_trend,
    _fetch_bls_series,
    _fetch_fred_series,
    _latest_value,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tool():
    cache = MagicMock()
    cache.get.return_value = None
    return JobPostingsTool(fred_api_key="test-key", cache=cache)


@pytest.fixture
def tool_no_key():
    cache = MagicMock()
    cache.get.return_value = None
    return JobPostingsTool(fred_api_key="", cache=cache)


@pytest.fixture
def tool_no_cache():
    return JobPostingsTool(fred_api_key="test-key", cache=None)


def _fred_obs(date: str = "2025-01-01", value: str = "1000") -> dict[str, str]:
    return {"date": date, "value": value}


def _fred_response(
    observations: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {"observations": observations or []}


def _bls_response(
    series: list[dict] | None = None,
    status: str = "REQUEST_SUCCEEDED",
) -> dict[str, Any]:
    return {
        "status": status,
        "Results": {"series": series or []},
    }


def _bls_series(
    series_id: str,
    data: list[dict] | None = None,
) -> dict[str, Any]:
    return {
        "seriesID": series_id,
        "data": data or [],
    }


def _bls_obs(
    year: str = "2025",
    period: str = "M01",
    value: str = "1000",
) -> dict[str, str]:
    return {"year": year, "period": period, "value": value}


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
        assert {"jolts", "sector", "labor_market"} == VALID_MODES

    def test_empty_mode(self, tool):
        r = tool.execute(mode="")
        assert not r.success
        assert "Invalid mode" in r.output

    def test_invalid_mode(self, tool):
        r = tool.execute(mode="bogus")
        assert not r.success
        assert "bogus" in r.output

    def test_mode_case_insensitive(self, tool):
        # Should normalize, not fail on mode validation
        r = tool.execute(mode="JOLTS")
        assert "Invalid mode" not in r.output

    def test_mode_whitespace(self, tool):
        r = tool.execute(mode="  jolts  ")
        assert "Invalid mode" not in r.output

    def test_none_mode(self, tool):
        r = tool.execute(mode=None)
        assert not r.success


# ═══════════════════════════════════════════════════════════════════════════
# 2. JOLTS mode (with FRED key)
# ═══════════════════════════════════════════════════════════════════════════


class TestJoltsMode:
    @patch("agent.tools.job_postings.httpx.Client")
    def test_jolts_basic(self, mock_client_cls, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(_fred_response([_fred_obs("2025-01-01", "8000")]))

        r = tool.execute(mode="jolts")
        assert r.success
        assert r.data["mode"] == "jolts"

    @patch("agent.tools.job_postings.httpx.Client")
    def test_jolts_empty_series(self, mock_client_cls, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(_fred_response([]))

        r = tool.execute(mode="jolts")
        assert r.success
        assert "N/A" in r.output

    @patch("agent.tools.job_postings.httpx.Client")
    def test_jolts_with_quits_layoffs_ratio(self, mock_client_cls, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        # Return different values per series
        def get_side_effect(*args, **kwargs):
            series_id = kwargs.get("params", {}).get("series_id", "")
            if series_id == "JTSQUL":
                return _mock_response(_fred_response([_fred_obs(value="4000")]))
            if series_id == "JTSLDR":
                return _mock_response(_fred_response([_fred_obs(value="1800")]))
            return _mock_response(_fred_response([_fred_obs(value="8000")]))

        mock_client.get.side_effect = get_side_effect

        r = tool.execute(mode="jolts")
        assert r.success
        # If ratio is computed, it shows in summary
        summary = r.data.get("summary", {})
        assert isinstance(summary, dict)

    @patch("agent.tools.job_postings.httpx.Client")
    def test_jolts_http_error(self, mock_client_cls, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response({}, status=500)

        r = tool.execute(mode="jolts")
        assert r.success  # Still returns with N/A values

    def test_jolts_cache_hit(self, tool):
        tool._cache.get.return_value = {
            "JTSJOL": [_fred_obs(value="8000")],
            "JTSQUL": [_fred_obs(value="4000")],
            "JTSHIL": [_fred_obs(value="6000")],
            "JTSLDR": [_fred_obs(value="1800")],
        }
        r = tool.execute(mode="jolts")
        assert r.success
        assert "(cached)" in r.output

    def test_jolts_months_default(self, tool):
        tool._cache.get.return_value = {}
        tool._cache.get.return_value = {k: [] for k in _JOLTS_SERIES}
        r = tool.execute(mode="jolts")
        assert r.success

    def test_jolts_months_clamped(self, tool):
        tool._cache.get.return_value = {k: [] for k in _JOLTS_SERIES}
        r = tool.execute(mode="jolts", months=0)
        assert r.success  # Clamped to 1

    def test_jolts_months_max(self, tool):
        tool._cache.get.return_value = {k: [] for k in _JOLTS_SERIES}
        r = tool.execute(mode="jolts", months=999)
        assert r.success  # Clamped to 60


# ═══════════════════════════════════════════════════════════════════════════
# 3. JOLTS via BLS fallback (no FRED key)
# ═══════════════════════════════════════════════════════════════════════════


class TestJoltsBLSFallback:
    @patch("agent.tools.job_postings.httpx.Client")
    def test_fallback_to_bls(self, mock_client_cls, tool_no_key):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        bls_data = _bls_response(
            [
                _bls_series("JTS000000000000000JOL", [_bls_obs()]),
                _bls_series("JTS000000000000000QUL", [_bls_obs()]),
                _bls_series("JTS000000000000000HIL", [_bls_obs()]),
                _bls_series("JTS000000000000000LDL", [_bls_obs()]),
            ]
        )
        mock_client.post.return_value = _mock_response(bls_data)

        r = tool_no_key.execute(mode="jolts")
        assert r.success

    @patch("agent.tools.job_postings.httpx.Client")
    def test_bls_request_failed(self, mock_client_cls, tool_no_key):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_response(_bls_response(status="REQUEST_FAILED"))

        r = tool_no_key.execute(mode="jolts")
        assert r.success  # Returns with N/A values

    @patch("agent.tools.job_postings.httpx.Client")
    def test_bls_http_error(self, mock_client_cls, tool_no_key):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_response({}, status=500)

        r = tool_no_key.execute(mode="jolts")
        assert r.success

    @patch("agent.tools.job_postings.httpx.Client")
    def test_bls_network_error(self, mock_client_cls, tool_no_key):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")

        r = tool_no_key.execute(mode="jolts")
        assert r.success


# ═══════════════════════════════════════════════════════════════════════════
# 4. Sector mode
# ═══════════════════════════════════════════════════════════════════════════


class TestSectorMode:
    @patch("agent.tools.job_postings.httpx.Client")
    def test_sector_basic(self, mock_client_cls, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        series_data = [_bls_series(sid, [_bls_obs(value=str(i * 100))]) for i, sid in enumerate(_SECTOR_SERIES.keys())]
        mock_client.post.return_value = _mock_response(_bls_response(series_data))

        r = tool.execute(mode="sector")
        assert r.success
        assert r.data["mode"] == "sector"

    @patch("agent.tools.job_postings.httpx.Client")
    def test_sector_empty_results(self, mock_client_cls, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_response(_bls_response([]))

        r = tool.execute(mode="sector")
        assert r.success

    def test_sector_cache_hit(self, tool):
        tool._cache.get.return_value = {sid: [] for sid in _SECTOR_SERIES}
        r = tool.execute(mode="sector")
        assert r.success
        assert "(cached)" in r.output

    @patch("agent.tools.job_postings.httpx.Client")
    def test_sector_strongest_weakest(self, mock_client_cls, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        series_data = [
            _bls_series(sid, [_bls_obs(value=str((i + 1) * 500))]) for i, sid in enumerate(_SECTOR_SERIES.keys())
        ]
        mock_client.post.return_value = _mock_response(_bls_response(series_data))

        r = tool.execute(mode="sector")
        assert r.success
        assert "Strongest" in r.output
        assert "Weakest" in r.output


# ═══════════════════════════════════════════════════════════════════════════
# 5. Labor market mode
# ═══════════════════════════════════════════════════════════════════════════


class TestLaborMarketMode:
    def test_requires_fred_key(self, tool_no_key):
        r = tool_no_key.execute(mode="labor_market")
        assert not r.success
        assert "FRED" in r.output

    @patch("agent.tools.job_postings.httpx.Client")
    def test_labor_market_basic(self, mock_client_cls, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        # Return realistic values for each series
        def get_side_effect(*args, **kwargs):
            series_id = kwargs.get("params", {}).get("series_id", "")
            if series_id == "UNRATE":
                return _mock_response(_fred_response([_fred_obs(value="3.8")]))
            if series_id == "ICSA":
                return _mock_response(_fred_response([_fred_obs(value="210000")]))
            if series_id == "PAYEMS":
                return _mock_response(_fred_response([_fred_obs(value="157000")]))
            return _mock_response(_fred_response([_fred_obs(value="8000")]))

        mock_client.get.side_effect = get_side_effect

        r = tool.execute(mode="labor_market")
        assert r.success
        assert r.data["mode"] == "labor_market"

    def test_labor_market_cache_hit(self, tool):
        tool._cache.get.return_value = {
            "JTSJOL": [_fred_obs(value="8000")],
            "JTSQUL": [_fred_obs(value="4000")],
            "JTSHIL": [_fred_obs(value="6000")],
            "JTSLDR": [_fred_obs(value="1800")],
            "UNRATE": [_fred_obs(value="3.8")],
            "ICSA": [_fred_obs(value="210000")],
            "PAYEMS": [_fred_obs(value="157000")],
        }
        r = tool.execute(mode="labor_market")
        assert r.success
        assert "(cached)" in r.output
        # Should compute market tightness
        assert "tightness" in r.output.lower()


# ═══════════════════════════════════════════════════════════════════════════
# 6. Helper functions
# ═══════════════════════════════════════════════════════════════════════════


class TestHelperFunctions:
    def test_latest_value_normal(self):
        obs = [_fred_obs("2025-01-01", "100"), _fred_obs("2024-12-01", "95")]
        date, val = _latest_value(obs)
        assert date == "2025-01-01"
        assert val == 100.0

    def test_latest_value_empty(self):
        date, val = _latest_value([])
        assert date == "N/A"
        assert val is None

    def test_latest_value_invalid(self):
        obs = [{"date": "2025-01-01", "value": "not_a_number"}]
        date, val = _latest_value(obs)
        assert date == "2025-01-01"
        assert val is None

    def test_latest_value_missing_keys(self):
        obs = [{"foo": "bar"}]
        date, val = _latest_value(obs)
        assert val is None

    def test_compute_trend_rising(self):
        obs = [
            _fred_obs(value="110"),
            _fred_obs(value="100"),
            _fred_obs(value="95"),
            _fred_obs(value="90"),
        ]
        assert _compute_trend(obs) == "rising"

    def test_compute_trend_falling(self):
        obs = [
            _fred_obs(value="85"),
            _fred_obs(value="95"),
            _fred_obs(value="100"),
            _fred_obs(value="110"),
        ]
        assert _compute_trend(obs) == "falling"

    def test_compute_trend_stable(self):
        obs = [
            _fred_obs(value="100"),
            _fred_obs(value="99"),
            _fred_obs(value="101"),
            _fred_obs(value="100"),
        ]
        assert _compute_trend(obs) == "stable"

    def test_compute_trend_insufficient(self):
        assert _compute_trend([]) == "insufficient data"
        assert _compute_trend([_fred_obs()]) == "insufficient data"

    def test_compute_trend_invalid_values(self):
        obs = [{"date": "d", "value": "bad"}, {"date": "d", "value": "bad"}]
        assert _compute_trend(obs) == "insufficient data"


# ═══════════════════════════════════════════════════════════════════════════
# 7. FRED series fetching
# ═══════════════════════════════════════════════════════════════════════════


class TestFetchFredSeries:
    @patch("agent.tools.job_postings.httpx.Client")
    def test_normal_fetch(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(_fred_response([_fred_obs("2025-01-01", "100")]))

        result = _fetch_fred_series("JTSJOL", "key")
        assert len(result) == 1
        assert result[0]["value"] == "100"

    @patch("agent.tools.job_postings.httpx.Client")
    def test_filters_dot_values(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(
            _fred_response(
                [
                    _fred_obs("2025-01-01", "100"),
                    {"date": "2024-12-01", "value": "."},
                    {"date": "2024-11-01", "value": ""},
                    {"date": "2024-10-01", "value": "95"},
                ]
            )
        )

        result = _fetch_fred_series("JTSJOL", "key")
        assert len(result) == 2
        assert result[0]["value"] == "100"
        assert result[1]["value"] == "95"

    @patch("agent.tools.job_postings.httpx.Client")
    def test_http_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response({}, status=500)

        result = _fetch_fred_series("JTSJOL", "key")
        assert result == []

    @patch("agent.tools.job_postings.httpx.Client")
    def test_network_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = httpx.ConnectError("fail")

        result = _fetch_fred_series("JTSJOL", "key")
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════
# 8. BLS series fetching
# ═══════════════════════════════════════════════════════════════════════════


class TestFetchBLSSeries:
    @patch("agent.tools.job_postings.httpx.Client")
    def test_normal_fetch(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        bls_data = _bls_response(
            [
                _bls_series("SER1", [_bls_obs("2025", "M01", "100")]),
            ]
        )
        mock_client.post.return_value = _mock_response(bls_data)

        result = _fetch_bls_series(["SER1"], start_year=2024, end_year=2025)
        assert "SER1" in result
        assert len(result["SER1"]) == 1
        assert result["SER1"][0]["date"] == "2025-01-01"

    @patch("agent.tools.job_postings.httpx.Client")
    def test_filters_non_monthly(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        bls_data = _bls_response(
            [
                _bls_series(
                    "SER1",
                    [
                        _bls_obs("2025", "M01", "100"),
                        {"year": "2025", "period": "Q01", "value": "300"},  # quarterly
                        {"year": "2025", "period": "A01", "value": "1200"},  # annual
                    ],
                ),
            ]
        )
        mock_client.post.return_value = _mock_response(bls_data)

        result = _fetch_bls_series(["SER1"], start_year=2024, end_year=2025)
        assert len(result["SER1"]) == 1  # Only M01 kept

    @patch("agent.tools.job_postings.httpx.Client")
    def test_request_failed_status(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_response(_bls_response(status="REQUEST_FAILED"))

        result = _fetch_bls_series(["SER1"], start_year=2024, end_year=2025)
        assert result == {}

    @patch("agent.tools.job_postings.httpx.Client")
    def test_http_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_response({}, status=500)

        result = _fetch_bls_series(["SER1"], start_year=2024, end_year=2025)
        assert result == {}

    @patch("agent.tools.job_postings.httpx.Client")
    def test_network_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = httpx.ConnectError("fail")

        result = _fetch_bls_series(["SER1"], start_year=2024, end_year=2025)
        assert result == {}


# ═══════════════════════════════════════════════════════════════════════════
# 9. No-cache paths
# ═══════════════════════════════════════════════════════════════════════════


class TestNoCache:
    @patch("agent.tools.job_postings.httpx.Client")
    def test_jolts_no_cache(self, mock_client_cls, tool_no_cache):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(_fred_response([_fred_obs()]))

        r = tool_no_cache.execute(mode="jolts")
        assert r.success

    @patch("agent.tools.job_postings.httpx.Client")
    def test_labor_market_no_cache(self, mock_client_cls, tool_no_cache):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(_fred_response([_fred_obs()]))

        r = tool_no_cache.execute(mode="labor_market")
        assert r.success


# ═══════════════════════════════════════════════════════════════════════════
# 10. Constants / data integrity
# ═══════════════════════════════════════════════════════════════════════════


class TestConstants:
    def test_jolts_series_keys(self):
        expected = {"JTSJOL", "JTSQUL", "JTSHIL", "JTSLDR"}
        assert set(_JOLTS_SERIES.keys()) == expected

    def test_labor_series_superset_of_jolts(self):
        for key in _JOLTS_SERIES:
            assert key in _LABOR_SERIES

    def test_sector_series_nonempty(self):
        assert len(_SECTOR_SERIES) >= 10

    def test_cache_ttl_reasonable(self):
        assert 3600 <= _CACHE_TTL <= 86400


# ═══════════════════════════════════════════════════════════════════════════
# 11. Integration: counts
# ═══════════════════════════════════════════════════════════════════════════


class TestIntegration:
    def test_tool_count(self):
        from agent.cli import build_tool_registry

        reg = build_tool_registry()
        # Was 60; commit 43de067 (2026-08-26) fixed nightlight_activity's
        # constructor kwarg mismatch (store= vs pipeline_store=) that silently
        # skipped its registration -- registry now correctly has 61 tools.
        assert len(reg.list_names()) == 61

    def test_arm_count(self):
        from agent.learning.bandit import DEFAULT_ARMS

        assert len(DEFAULT_ARMS) == 48

    def test_tool_registered(self):
        from agent.cli import build_tool_registry

        reg = build_tool_registry()
        names = reg.list_names()
        assert "job_postings" in names

    def test_tool_interface(self, tool):
        assert tool.name == "job_postings"
        assert "mode" in tool.parameters["properties"]
        assert "required" in tool.parameters
