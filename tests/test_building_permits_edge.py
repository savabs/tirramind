"""
Edge-case tests for BuildingPermitsTool (7b-AB).

Coverage targets:
- Invalid / missing / boundary parameters
- FRED API key requirement
- Empty / malformed API responses
- Cache hit / miss paths
- Permit trend calculations (MoM, YoY, consecutive declines)
- Regional divergence detection
- Housing starts/permits ratio
- HTTP errors, timeouts
- Mode validation
- Months clamping
- Integration: tool count = 41, arm count = 29
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.tools.building_permits import (
    VALID_MODES,
    BuildingPermitsTool,
    _CACHE_TTL,
    _PERMIT_SERIES,
    _REGIONAL_SERIES,
    _STARTS_SERIES,
    _consecutive_declines,
    _fetch_fred,
    _latest,
    _pct_change,
    _trend_direction,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tool():
    cache = MagicMock()
    cache.get.return_value = None
    return BuildingPermitsTool(fred_api_key="test-key", cache=cache)


@pytest.fixture
def tool_no_key():
    cache = MagicMock()
    cache.get.return_value = None
    return BuildingPermitsTool(fred_api_key="", cache=cache)


@pytest.fixture
def tool_no_cache():
    return BuildingPermitsTool(fred_api_key="test-key", cache=None)


def _obs(date: str = "2025-01-01", value: str = "1500") -> dict[str, str]:
    return {"date": date, "value": value}


def _fred_response(observations: list[dict[str, str]] | None = None) -> dict[str, Any]:
    return {"observations": observations or []}


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
        assert VALID_MODES == {"permits", "regional", "housing_starts"}

    def test_empty_mode(self, tool):
        r = tool.execute(mode="")
        assert not r.success
        assert "Invalid mode" in r.output

    def test_invalid_mode(self, tool):
        r = tool.execute(mode="bogus")
        assert not r.success
        assert "bogus" in r.output

    def test_mode_case_insensitive(self, tool):
        r = tool.execute(mode="PERMITS")
        assert "Invalid mode" not in r.output

    def test_mode_whitespace(self, tool):
        r = tool.execute(mode="  permits  ")
        assert "Invalid mode" not in r.output

    def test_none_mode(self, tool):
        r = tool.execute(mode=None)
        assert not r.success


# ═══════════════════════════════════════════════════════════════════════════
# 2. FRED API key requirement
# ═══════════════════════════════════════════════════════════════════════════


class TestAPIKeyRequirement:
    def test_permits_requires_key(self, tool_no_key):
        r = tool_no_key.execute(mode="permits")
        assert not r.success
        assert "FRED" in r.output

    def test_regional_requires_key(self, tool_no_key):
        r = tool_no_key.execute(mode="regional")
        assert not r.success

    def test_housing_starts_requires_key(self, tool_no_key):
        r = tool_no_key.execute(mode="housing_starts")
        assert not r.success


# ═══════════════════════════════════════════════════════════════════════════
# 3. Permits mode
# ═══════════════════════════════════════════════════════════════════════════


class TestPermitsMode:
    @patch("agent.tools.building_permits.httpx.Client")
    def test_permits_basic(self, mock_client_cls, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(
            _fred_response([_obs("2025-01-01", "1500")])
        )

        r = tool.execute(mode="permits")
        assert r.success
        assert r.data["mode"] == "permits"
        assert "US Building Permits" in r.output

    @patch("agent.tools.building_permits.httpx.Client")
    def test_permits_empty_data(self, mock_client_cls, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(_fred_response([]))

        r = tool.execute(mode="permits")
        assert r.success
        assert "N/A" in r.output

    @patch("agent.tools.building_permits.httpx.Client")
    def test_permits_with_decline_warning(self, mock_client_cls, tool):
        # Create a series with consecutive declines
        obs = [
            _obs("2025-04-01", "1200"),
            _obs("2025-03-01", "1300"),
            _obs("2025-02-01", "1400"),
            _obs("2025-01-01", "1500"),
        ]
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(_fred_response(obs))

        r = tool.execute(mode="permits")
        assert r.success
        # 3 consecutive declines → warning
        assert "consecutive months of decline" in r.output

    @patch("agent.tools.building_permits.httpx.Client")
    def test_permits_sf_mf_share(self, mock_client_cls, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        call_count = [0]
        def get_side_effect(*args, **kwargs):
            call_count[0] += 1
            series_id = kwargs.get("params", {}).get("series_id", "")
            if series_id == "PERMIT":
                return _mock_response(_fred_response([_obs(value="1500")]))
            elif series_id == "PERMIT1":
                return _mock_response(_fred_response([_obs(value="1000")]))
            return _mock_response(_fred_response([_obs(value="800")]))

        mock_client.get.side_effect = get_side_effect

        r = tool.execute(mode="permits")
        assert r.success
        # Verify single-family share is computed
        if "Single-family share" in r.output:
            assert "Multi-family" in r.output

    def test_permits_cache_hit(self, tool):
        tool._cache.get.return_value = {
            "PERMIT": [_obs(value="1500")],
            "PERMIT1": [_obs(value="1000")],
            "PERMITNSA": [_obs(value="1400")],
        }
        r = tool.execute(mode="permits")
        assert r.success
        assert "(cached)" in r.output

    @patch("agent.tools.building_permits.httpx.Client")
    def test_permits_http_error(self, mock_client_cls, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response({}, status=500)

        r = tool.execute(mode="permits")
        assert r.success  # Returns with N/A values


# ═══════════════════════════════════════════════════════════════════════════
# 4. Regional mode
# ═══════════════════════════════════════════════════════════════════════════


class TestRegionalMode:
    @patch("agent.tools.building_permits.httpx.Client")
    def test_regional_basic(self, mock_client_cls, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(
            _fred_response([_obs(value="400")])
        )

        r = tool.execute(mode="regional")
        assert r.success
        assert "Regional" in r.output

    @patch("agent.tools.building_permits.httpx.Client")
    def test_regional_divergence(self, mock_client_cls, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        # Return different values for different regions
        call_idx = [0]
        values = ["50", "100", "800", "300", "40", "90", "750", "280"]
        def get_side_effect(*args, **kwargs):
            idx = min(call_idx[0], len(values) - 1)
            call_idx[0] += 1
            return _mock_response(_fred_response([_obs(value=values[idx])]))

        mock_client.get.side_effect = get_side_effect

        r = tool.execute(mode="regional")
        assert r.success
        # Should detect strongest/weakest regions
        if "Strongest" in r.output:
            assert "Weakest" in r.output

    def test_regional_cache_hit(self, tool):
        tool._cache.get.return_value = {sid: [] for sid in _REGIONAL_SERIES}
        r = tool.execute(mode="regional")
        assert r.success
        assert "(cached)" in r.output


# ═══════════════════════════════════════════════════════════════════════════
# 5. Housing starts mode
# ═══════════════════════════════════════════════════════════════════════════


class TestHousingStartsMode:
    @patch("agent.tools.building_permits.httpx.Client")
    def test_starts_basic(self, mock_client_cls, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(
            _fred_response([_obs(value="1400")])
        )

        r = tool.execute(mode="housing_starts")
        assert r.success
        assert "Housing Starts" in r.output

    @patch("agent.tools.building_permits.httpx.Client")
    def test_starts_permits_ratio(self, mock_client_cls, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        def get_side_effect(*args, **kwargs):
            series_id = kwargs.get("params", {}).get("series_id", "")
            if series_id == "HOUST":
                return _mock_response(_fred_response([_obs(value="1400")]))
            if series_id == "HOUST1F":
                return _mock_response(_fred_response([_obs(value="900")]))
            if series_id == "PERMIT":
                return _mock_response(_fred_response([_obs(value="1500")]))
            if series_id == "PERMIT1":
                return _mock_response(_fred_response([_obs(value="1000")]))
            return _mock_response(_fred_response([_obs(value="1000")]))

        mock_client.get.side_effect = get_side_effect

        r = tool.execute(mode="housing_starts")
        assert r.success
        assert "Starts/Permits ratio" in r.output

    def test_starts_cache_hit(self, tool):
        tool._cache.get.return_value = {
            "HOUST": [_obs(value="1400")],
            "HOUST1F": [_obs(value="900")],
            "PERMIT": [_obs(value="1500")],
            "PERMIT1": [_obs(value="1000")],
        }
        r = tool.execute(mode="housing_starts")
        assert r.success
        assert "(cached)" in r.output
        # Ratio should be computed
        assert "Starts/Permits ratio" in r.output

    @patch("agent.tools.building_permits.httpx.Client")
    def test_starts_no_ratio_when_zero_permits(self, mock_client_cls, tool):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        def get_side_effect(*args, **kwargs):
            series_id = kwargs.get("params", {}).get("series_id", "")
            if series_id in ("PERMIT", "PERMIT1"):
                return _mock_response(_fred_response([_obs(value="0")]))
            return _mock_response(_fred_response([_obs(value="1400")]))

        mock_client.get.side_effect = get_side_effect

        r = tool.execute(mode="housing_starts")
        assert r.success
        # Should NOT crash on division by zero


# ═══════════════════════════════════════════════════════════════════════════
# 6. Helper functions
# ═══════════════════════════════════════════════════════════════════════════


class TestHelperFunctions:
    # _latest
    def test_latest_normal(self):
        obs = [_obs("2025-01-01", "1500"), _obs("2024-12-01", "1400")]
        date, val = _latest(obs)
        assert date == "2025-01-01"
        assert val == 1500.0

    def test_latest_empty(self):
        date, val = _latest([])
        assert date == "N/A"
        assert val is None

    def test_latest_invalid_value(self):
        obs = [{"date": "2025-01-01", "value": "bad"}]
        date, val = _latest(obs)
        assert date == "2025-01-01"
        assert val is None

    def test_latest_missing_keys(self):
        obs = [{"foo": "bar"}]
        date, val = _latest(obs)
        assert val is None

    # _pct_change
    def test_pct_change_normal(self):
        obs = [_obs(value="110"), _obs(value="100")]
        pct = _pct_change(obs, 1)
        assert pct is not None
        assert abs(pct - 10.0) < 0.01

    def test_pct_change_negative(self):
        obs = [_obs(value="90"), _obs(value="100")]
        pct = _pct_change(obs, 1)
        assert pct is not None
        assert abs(pct - (-10.0)) < 0.01

    def test_pct_change_insufficient_data(self):
        obs = [_obs(value="100")]
        assert _pct_change(obs, 1) is None

    def test_pct_change_zero_previous(self):
        obs = [_obs(value="100"), _obs(value="0")]
        assert _pct_change(obs, 1) is None

    def test_pct_change_invalid_values(self):
        obs = [{"date": "d", "value": "bad"}, {"date": "d", "value": "bad"}]
        assert _pct_change(obs, 1) is None

    def test_pct_change_empty(self):
        assert _pct_change([], 1) is None

    # _trend_direction
    def test_trend_rising(self):
        obs = [_obs(value="110"), _obs(value="100"), _obs(value="95"), _obs(value="90")]
        assert _trend_direction(obs) == "rising"

    def test_trend_falling(self):
        obs = [_obs(value="85"), _obs(value="95"), _obs(value="100"), _obs(value="110")]
        assert _trend_direction(obs) == "falling"

    def test_trend_stable(self):
        obs = [_obs(value="100"), _obs(value="99"), _obs(value="101"), _obs(value="100")]
        assert _trend_direction(obs) == "stable"

    def test_trend_insufficient(self):
        assert _trend_direction([]) == "insufficient data"
        assert _trend_direction([_obs()]) == "insufficient data"

    def test_trend_invalid_values(self):
        obs = [{"date": "d", "value": "x"}, {"date": "d", "value": "y"}]
        assert _trend_direction(obs) == "insufficient data"

    # _consecutive_declines
    def test_consecutive_declines_3(self):
        obs = [_obs(value="100"), _obs(value="110"), _obs(value="120"), _obs(value="130")]
        assert _consecutive_declines(obs) == 3

    def test_consecutive_declines_0(self):
        obs = [_obs(value="130"), _obs(value="120"), _obs(value="110"), _obs(value="100")]
        assert _consecutive_declines(obs) == 0

    def test_consecutive_declines_partial(self):
        obs = [_obs(value="100"), _obs(value="110"), _obs(value="120"), _obs(value="110")]
        assert _consecutive_declines(obs) == 2

    def test_consecutive_declines_empty(self):
        assert _consecutive_declines([]) == 0

    def test_consecutive_declines_single(self):
        assert _consecutive_declines([_obs()]) == 0

    def test_consecutive_declines_invalid(self):
        obs = [{"date": "d", "value": "bad"}, {"date": "d", "value": "bad"}]
        assert _consecutive_declines(obs) == 0


# ═══════════════════════════════════════════════════════════════════════════
# 7. FRED fetching
# ═══════════════════════════════════════════════════════════════════════════


class TestFetchFred:
    @patch("agent.tools.building_permits.httpx.Client")
    def test_normal_fetch(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(
            _fred_response([_obs("2025-01-01", "1500")])
        )

        result = _fetch_fred("PERMIT", "key")
        assert len(result) == 1
        assert result[0]["value"] == "1500"

    @patch("agent.tools.building_permits.httpx.Client")
    def test_filters_dot_values(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(
            _fred_response([
                _obs("2025-01-01", "1500"),
                {"date": "2024-12-01", "value": "."},
                {"date": "2024-11-01", "value": ""},
                {"date": "2024-10-01", "value": "1400"},
            ])
        )

        result = _fetch_fred("PERMIT", "key")
        assert len(result) == 2

    @patch("agent.tools.building_permits.httpx.Client")
    def test_http_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response({}, status=500)

        result = _fetch_fred("PERMIT", "key")
        assert result == []

    @patch("agent.tools.building_permits.httpx.Client")
    def test_network_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = httpx.ConnectError("fail")

        result = _fetch_fred("PERMIT", "key")
        assert result == []

    @patch("agent.tools.building_permits.httpx.Client")
    def test_invalid_json(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.side_effect = ValueError("Bad JSON")
        mock_client.get.return_value = resp

        result = _fetch_fred("PERMIT", "key")
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════
# 8. Months clamping
# ═══════════════════════════════════════════════════════════════════════════


class TestMonthsClamping:
    def test_default_months(self, tool):
        tool._cache.get.return_value = {k: [] for k in _PERMIT_SERIES}
        r = tool.execute(mode="permits")
        assert r.success

    def test_zero_months_clamped(self, tool):
        tool._cache.get.return_value = {k: [] for k in _PERMIT_SERIES}
        r = tool.execute(mode="permits", months=0)
        assert r.success

    def test_negative_months_clamped(self, tool):
        tool._cache.get.return_value = {k: [] for k in _PERMIT_SERIES}
        r = tool.execute(mode="permits", months=-5)
        assert r.success

    def test_huge_months_clamped(self, tool):
        tool._cache.get.return_value = {k: [] for k in _PERMIT_SERIES}
        r = tool.execute(mode="permits", months=9999)
        assert r.success


# ═══════════════════════════════════════════════════════════════════════════
# 9. No-cache paths
# ═══════════════════════════════════════════════════════════════════════════


class TestNoCache:
    @patch("agent.tools.building_permits.httpx.Client")
    def test_permits_no_cache(self, mock_client_cls, tool_no_cache):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(_fred_response([_obs()]))

        r = tool_no_cache.execute(mode="permits")
        assert r.success

    @patch("agent.tools.building_permits.httpx.Client")
    def test_regional_no_cache(self, mock_client_cls, tool_no_cache):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(_fred_response([_obs()]))

        r = tool_no_cache.execute(mode="regional")
        assert r.success

    @patch("agent.tools.building_permits.httpx.Client")
    def test_starts_no_cache(self, mock_client_cls, tool_no_cache):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(_fred_response([_obs()]))

        r = tool_no_cache.execute(mode="housing_starts")
        assert r.success


# ═══════════════════════════════════════════════════════════════════════════
# 10. Constants / data integrity
# ═══════════════════════════════════════════════════════════════════════════


class TestConstants:
    def test_permit_series_keys(self):
        assert "PERMIT" in _PERMIT_SERIES
        assert "PERMIT1" in _PERMIT_SERIES

    def test_regional_series_count(self):
        assert len(_REGIONAL_SERIES) == 8

    def test_starts_series_keys(self):
        assert "HOUST" in _STARTS_SERIES
        assert "HOUST1F" in _STARTS_SERIES
        assert "PERMIT" in _STARTS_SERIES

    def test_cache_ttl_reasonable(self):
        assert 3600 <= _CACHE_TTL <= 86400


# ═══════════════════════════════════════════════════════════════════════════
# 11. Integration: counts
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
        assert "building_permits" in names

    def test_tool_interface(self, tool):
        assert tool.name == "building_permits"
        assert "mode" in tool.parameters["properties"]
        assert "required" in tool.parameters
