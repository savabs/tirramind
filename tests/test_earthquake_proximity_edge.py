"""
Edge case tests for EarthquakeProximityTool (USGS).

Covers: mode validation, zone matching, parameter clamping, haversine distance,
magnitude labels, infrastructure lookup, quake formatting, cache interaction,
HTTP error handling, timeout handling, infrastructure listing, empty data,
malformed data, monitor mode, output formatting, registry + bandit.
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.tools.earthquake_proximity import (
    EarthquakeProximityTool,
    CRITICAL_INFRA,
    _find_nearby_infra,
    _format_quake,
    _haversine_km,
    _mag_label,
)
from agent.tools.base import ToolResult


# ── Fixtures ──────────────────────────────────────────────────


def _make_usgs_feature(
    mag: float = 5.5,
    place: str = "15km SW of Test City",
    time_ms: int = 1751400000000,
    lat: float = 24.78,
    lon: float = 120.99,
    depth: float = 10.0,
    alert: str | None = "yellow",
    tsunami: int = 0,
    sig: int = 500,
    url: str = "https://earthquake.usgs.gov/earthquakes/eventpage/test",
) -> dict[str, Any]:
    return {
        "type": "Feature",
        "properties": {
            "mag": mag,
            "place": place,
            "time": time_ms,
            "alert": alert,
            "tsunami": tsunami,
            "sig": sig,
            "url": url,
            "magType": "mw",
        },
        "geometry": {
            "type": "Point",
            "coordinates": [lon, lat, depth],
        },
    }


def _make_usgs_response(features: list[dict] | None = None) -> dict[str, Any]:
    if features is None:
        features = [
            _make_usgs_feature(),
            _make_usgs_feature(mag=4.2, place="10km NE of Nowhere", lat=0, lon=0),
        ]
    return {"type": "FeatureCollection", "features": features}


def _tool(cache=None) -> EarthquakeProximityTool:
    return EarthquakeProximityTool(cache=cache)


def _mock_resp(body: Any, status: int = 200) -> httpx.Response:
    import json

    return httpx.Response(
        status_code=status,
        text=json.dumps(body),
        request=httpx.Request("GET", "http://test"),
    )


# ── 1. Tool Metadata ─────────────────────────────────────────


class TestToolMetadata:
    def test_name(self):
        assert _tool().name == "earthquake_proximity"

    def test_description_nonempty(self):
        assert len(_tool().description) > 50

    def test_parameters_schema(self):
        params = _tool().parameters
        assert params["type"] == "object"
        props = params["properties"]
        assert "mode" in props
        assert "min_magnitude" in props
        assert "days_back" in props
        assert "zone" in props
        assert "infra_only" in props
        assert "limit" in props

    def test_mode_enum(self):
        modes = _tool().parameters["properties"]["mode"]["enum"]
        assert set(modes) == {"recent", "monitor", "infrastructure"}


# ── 2. Input Validation ──────────────────────────────────────


class TestInputValidation:
    def test_invalid_mode(self):
        r = _tool().execute(mode="bad")
        assert not r.success
        assert "Invalid mode" in r.output

    def test_days_back_clamped_low(self):
        with patch.object(EarthquakeProximityTool, "_fetch_usgs") as m:
            m.return_value = ([], None)
            r = _tool().execute(mode="recent", days_back=0)
        assert r.success

    def test_days_back_clamped_high(self):
        with patch.object(EarthquakeProximityTool, "_fetch_usgs") as m:
            m.return_value = ([], None)
            r = _tool().execute(mode="recent", days_back=999)
        assert r.success

    def test_limit_clamped_low(self):
        with patch.object(EarthquakeProximityTool, "_fetch_usgs") as m:
            m.return_value = ([], None)
            r = _tool().execute(mode="recent", limit=0)
        assert r.success

    def test_limit_clamped_high(self):
        feats = [_make_usgs_feature(mag=5.0 + i * 0.01) for i in range(250)]
        with patch.object(EarthquakeProximityTool, "_fetch_usgs") as m:
            m.return_value = (feats, None)
            r = _tool().execute(mode="recent", limit=999)
        assert r.success
        assert len(r.data["quakes"]) <= 200

    def test_min_magnitude_clamped(self):
        with patch.object(EarthquakeProximityTool, "_fetch_usgs") as m:
            m.return_value = ([], None)
            r = _tool().execute(mode="recent", min_magnitude=-5)
        assert r.success

    def test_min_magnitude_clamped_high(self):
        with patch.object(EarthquakeProximityTool, "_fetch_usgs") as m:
            m.return_value = ([], None)
            r = _tool().execute(mode="recent", min_magnitude=15)
        assert r.success

    def test_monitor_requires_zone(self):
        r = _tool().execute(mode="monitor")
        assert not r.success
        assert "zone" in r.output.lower()

    def test_extra_kwargs_ignored(self):
        with patch.object(EarthquakeProximityTool, "_fetch_usgs") as m:
            m.return_value = ([], None)
            r = _tool().execute(mode="recent", bogus="thing")
        assert r.success


# ── 3. Helper Functions ──────────────────────────────────────


class TestHaversine:
    def test_zero_distance(self):
        d = _haversine_km(0, 0, 0, 0)
        assert d == pytest.approx(0, abs=0.01)

    def test_known_distance(self):
        # NYC to LA ≈ 3944 km
        d = _haversine_km(40.71, -74.01, 34.05, -118.24)
        assert 3800 < d < 4100

    def test_same_lat(self):
        d = _haversine_km(45, 0, 45, 1)
        assert d > 0

    def test_antipodal(self):
        d = _haversine_km(0, 0, 0, 180)
        assert d > 15000

    def test_negative_coords(self):
        d = _haversine_km(-33.86, 151.21, -37.81, 144.96)  # Sydney to Melbourne
        assert 700 < d < 900


class TestMagLabel:
    def test_great(self):
        assert _mag_label(8.5) == "GREAT"

    def test_major(self):
        assert _mag_label(7.5) == "MAJOR"

    def test_strong(self):
        assert _mag_label(6.5) == "STRONG"

    def test_moderate(self):
        assert _mag_label(5.5) == "MODERATE"

    def test_light(self):
        assert _mag_label(4.5) == "LIGHT"

    def test_minor(self):
        assert _mag_label(3.0) == "MINOR"

    def test_boundary_8(self):
        assert _mag_label(8.0) == "GREAT"

    def test_boundary_7(self):
        assert _mag_label(7.0) == "MAJOR"

    def test_boundary_6(self):
        assert _mag_label(6.0) == "STRONG"

    def test_boundary_5(self):
        assert _mag_label(5.0) == "MODERATE"

    def test_boundary_4(self):
        assert _mag_label(4.0) == "LIGHT"

    def test_zero(self):
        assert _mag_label(0) == "MINOR"


class TestFindNearbyInfra:
    def test_near_tsmc(self):
        # Directly at TSMC Hsinchu coordinates
        result = _find_nearby_infra(24.78, 120.99)
        assert len(result) >= 1
        assert result[0]["name"] == "TSMC Hsinchu"

    def test_nowhere(self):
        # Middle of Pacific Ocean
        result = _find_nearby_infra(0, -160)
        assert result == []

    def test_sorted_by_distance(self):
        # Near Taiwan — might hit multiple
        result = _find_nearby_infra(24.0, 121.0)
        if len(result) >= 2:
            assert result[0]["distance_km"] <= result[1]["distance_km"]

    def test_all_infra_reachable(self):
        """Each infra zone should be findable when queried at its center."""
        for infra in CRITICAL_INFRA:
            result = _find_nearby_infra(infra["lat"], infra["lon"])
            names = [r["name"] for r in result]
            assert (
                infra["name"] in names
            ), f"{infra['name']} not findable at its own center"


class TestFormatQuake:
    def test_basic_format(self):
        feat = _make_usgs_feature()
        q = _format_quake(feat)
        assert q["magnitude"] == 5.5
        assert q["mag_label"] == "MODERATE"
        assert q["lat"] == 24.78
        assert q["lon"] == 120.99
        assert q["depth_km"] == 10.0
        assert q["tsunami"] is False
        assert q["alert"] == "yellow"
        assert isinstance(q["nearby_infrastructure"], list)

    def test_tsunami_true(self):
        feat = _make_usgs_feature(tsunami=1)
        q = _format_quake(feat)
        assert q["tsunami"] is True

    def test_missing_time(self):
        feat = _make_usgs_feature()
        feat["properties"]["time"] = None
        q = _format_quake(feat)
        assert q["time"] == ""

    def test_missing_mag(self):
        feat = _make_usgs_feature()
        feat["properties"]["mag"] = None
        q = _format_quake(feat)
        assert q["magnitude"] == 0

    def test_missing_geometry(self):
        feat = _make_usgs_feature()
        feat["geometry"] = {}
        q = _format_quake(feat)
        # Should default to 0,0,0
        assert q["lat"] == 0
        assert q["lon"] == 0

    def test_two_coord_geometry(self):
        feat = _make_usgs_feature()
        feat["geometry"]["coordinates"] = [120.0, 24.0]  # no depth
        q = _format_quake(feat)
        assert q["depth_km"] == 0


# ── 4. Recent Mode ───────────────────────────────────────────


class TestRecentMode:
    def test_basic_recent(self):
        feats = [_make_usgs_feature(), _make_usgs_feature(mag=4.2, lat=0, lon=0)]
        with patch.object(EarthquakeProximityTool, "_fetch_usgs") as m:
            m.return_value = (feats, None)
            r = _tool().execute(mode="recent")
        assert r.success
        assert r.data["count"] == 2

    def test_empty_recent(self):
        with patch.object(EarthquakeProximityTool, "_fetch_usgs") as m:
            m.return_value = ([], None)
            r = _tool().execute(mode="recent")
        assert r.success
        assert r.data["count"] == 0
        assert "No earthquakes" in r.output

    def test_infra_only_filter(self):
        feats = [
            _make_usgs_feature(lat=24.78, lon=120.99),  # near TSMC
            _make_usgs_feature(lat=0, lon=-160),  # Pacific Ocean
        ]
        with patch.object(EarthquakeProximityTool, "_fetch_usgs") as m:
            m.return_value = (feats, None)
            r = _tool().execute(mode="recent", infra_only=True)
        assert r.success
        assert r.data["count"] == 1  # only the one near TSMC

    def test_sorted_by_magnitude(self):
        feats = [
            _make_usgs_feature(mag=4.0),
            _make_usgs_feature(mag=7.0),
        ]
        with patch.object(EarthquakeProximityTool, "_fetch_usgs") as m:
            m.return_value = (feats, None)
            r = _tool().execute(mode="recent")
        assert r.data["quakes"][0]["magnitude"] == 7.0

    def test_fetch_error(self):
        with patch.object(EarthquakeProximityTool, "_fetch_usgs") as m:
            m.return_value = ([], "USGS API: Rate limited.")
            r = _tool().execute(mode="recent")
        assert not r.success

    def test_limit_respected(self):
        feats = [_make_usgs_feature(mag=5.0 + i * 0.01) for i in range(50)]
        with patch.object(EarthquakeProximityTool, "_fetch_usgs") as m:
            m.return_value = (feats, None)
            r = _tool().execute(mode="recent", limit=5)
        assert len(r.data["quakes"]) == 5

    def test_output_format(self):
        feats = [_make_usgs_feature(alert="red", tsunami=1)]
        with patch.object(EarthquakeProximityTool, "_fetch_usgs") as m:
            m.return_value = (feats, None)
            r = _tool().execute(mode="recent")
        assert "ALERT=RED" in r.output
        assert "TSUNAMI" in r.output

    def test_infrastructure_count_in_data(self):
        feats = [_make_usgs_feature(lat=24.78, lon=120.99)]  # near TSMC
        with patch.object(EarthquakeProximityTool, "_fetch_usgs") as m:
            m.return_value = (feats, None)
            r = _tool().execute(mode="recent")
        assert r.data["near_infrastructure"] >= 1


# ── 5. Monitor Mode ──────────────────────────────────────────


class TestMonitorMode:
    def test_monitor_tsmc(self):
        feats = [_make_usgs_feature(lat=24.78, lon=120.99, mag=4.5)]
        with patch.object(EarthquakeProximityTool, "_fetch_usgs") as m:
            m.return_value = (feats, None)
            r = _tool().execute(mode="monitor", zone="TSMC Hsinchu")
        assert r.success
        assert r.data["count"] >= 1

    def test_monitor_partial_match(self):
        feats = [_make_usgs_feature(lat=24.78, lon=120.99)]
        with patch.object(EarthquakeProximityTool, "_fetch_usgs") as m:
            m.return_value = (feats, None)
            r = _tool().execute(mode="monitor", zone="tsmc")
        assert r.success  # case-insensitive partial match

    def test_monitor_no_match(self):
        r = _tool().execute(mode="monitor", zone="NonexistentPlace123")
        assert not r.success
        assert "No infrastructure zone" in r.output

    def test_monitor_no_quakes(self):
        feats = [_make_usgs_feature(lat=0, lon=-160)]  # far from TSMC
        with patch.object(EarthquakeProximityTool, "_fetch_usgs") as m:
            m.return_value = (feats, None)
            r = _tool().execute(mode="monitor", zone="TSMC Hsinchu")
        assert r.success
        assert r.data["count"] == 0
        assert "quiet" in r.output.lower()

    def test_monitor_fetch_error(self):
        with patch.object(EarthquakeProximityTool, "_fetch_usgs") as m:
            m.return_value = ([], "USGS error")
            r = _tool().execute(mode="monitor", zone="TSMC")
        assert not r.success

    def test_monitor_empty_zone(self):
        r = _tool().execute(mode="monitor", zone="")
        assert not r.success

    def test_monitor_output_format(self):
        feats = [_make_usgs_feature(lat=24.78, lon=120.99, mag=5.0)]
        with patch.object(EarthquakeProximityTool, "_fetch_usgs") as m:
            m.return_value = (feats, None)
            r = _tool().execute(mode="monitor", zone="TSMC Hsinchu")
        assert "TSMC Hsinchu" in r.output
        assert "semiconductor" in r.output


# ── 6. Infrastructure Mode ───────────────────────────────────


class TestInfrastructureMode:
    def test_list(self):
        r = _tool().execute(mode="infrastructure")
        assert r.success
        assert r.data["count"] == len(CRITICAL_INFRA)
        assert r.data["count"] >= 19

    def test_list_output(self):
        r = _tool().execute(mode="infrastructure")
        assert "TSMC" in r.output
        assert "Escondida" in r.output
        assert "SEMICONDUCTOR" in r.output or "semiconductor" in r.output.lower()


# ── 7. USGS Fetch ────────────────────────────────────────────


class TestUSGSFetch:
    def test_successful_fetch(self):
        resp = _make_usgs_response()
        with patch("httpx.Client") as mock_client:
            mc = MagicMock()
            mc.__enter__ = MagicMock(return_value=mc)
            mc.__exit__ = MagicMock(return_value=False)
            mc.get.return_value = _mock_resp(resp)
            mock_client.return_value = mc

            feats, err = _tool()._fetch_usgs(min_magnitude=4.0, days_back=7)
        assert err is None
        assert len(feats) == 2

    def test_rate_limited(self):
        with patch("httpx.Client") as mock_client:
            mc = MagicMock()
            mc.__enter__ = MagicMock(return_value=mc)
            mc.__exit__ = MagicMock(return_value=False)
            mc.get.return_value = _mock_resp({}, status=429)
            mock_client.return_value = mc

            feats, err = _tool()._fetch_usgs(min_magnitude=4.0, days_back=7)
        assert feats == []
        assert "Rate limited" in err

    def test_http_500(self):
        with patch("httpx.Client") as mock_client:
            mc = MagicMock()
            mc.__enter__ = MagicMock(return_value=mc)
            mc.__exit__ = MagicMock(return_value=False)
            mc.get.return_value = _mock_resp({}, status=500)
            mock_client.return_value = mc

            feats, err = _tool()._fetch_usgs(min_magnitude=4.0, days_back=7)
        assert "500" in err

    def test_timeout(self):
        with patch("httpx.Client") as mock_client:
            mc = MagicMock()
            mc.__enter__ = MagicMock(return_value=mc)
            mc.__exit__ = MagicMock(return_value=False)
            mc.get.side_effect = httpx.TimeoutException("timeout")
            mock_client.return_value = mc

            feats, err = _tool()._fetch_usgs(min_magnitude=4.0, days_back=7)
        assert feats == []
        assert "timed out" in err.lower()

    def test_generic_exception(self):
        with patch("httpx.Client") as mock_client:
            mc = MagicMock()
            mc.__enter__ = MagicMock(return_value=mc)
            mc.__exit__ = MagicMock(return_value=False)
            mc.get.side_effect = ConnectionError("fail")
            mock_client.return_value = mc

            feats, err = _tool()._fetch_usgs(min_magnitude=4.0, days_back=7)
        assert "error" in err.lower()

    def test_cache_hit(self):
        cache = MagicMock()
        cache.get.return_value = {"features": [_make_usgs_feature()]}
        feats, err = EarthquakeProximityTool(cache=cache)._fetch_usgs(
            min_magnitude=4.0,
            days_back=7,
        )
        assert err is None
        assert len(feats) == 1

    def test_cache_miss_then_put(self):
        cache = MagicMock()
        cache.get.return_value = None
        resp = _make_usgs_response()
        with patch("httpx.Client") as mock_client:
            mc = MagicMock()
            mc.__enter__ = MagicMock(return_value=mc)
            mc.__exit__ = MagicMock(return_value=False)
            mc.get.return_value = _mock_resp(resp)
            mock_client.return_value = mc

            feats, err = EarthquakeProximityTool(cache=cache)._fetch_usgs(
                min_magnitude=4.0,
                days_back=7,
            )
        assert err is None
        cache.put.assert_called_once()


# ── 8. Critical Infrastructure Data ──────────────────────────


class TestInfraData:
    def test_all_zones_valid(self):
        for z in CRITICAL_INFRA:
            assert "name" in z
            assert "lat" in z
            assert "lon" in z
            assert "radius_km" in z
            assert "sector" in z
            assert "detail" in z
            assert -90 <= z["lat"] <= 90
            assert -180 <= z["lon"] <= 180
            assert z["radius_km"] > 0

    def test_unique_names(self):
        names = [z["name"] for z in CRITICAL_INFRA]
        assert len(names) == len(set(names))

    def test_expected_sectors(self):
        sectors = {z["sector"] for z in CRITICAL_INFRA}
        assert "semiconductor" in sectors
        assert "mining" in sectors
        assert "nuclear" in sectors
        assert "energy" in sectors
        assert "logistics" in sectors


# ── 9. Registry & Bandit Integration ─────────────────────────


class TestRegistryIntegration:
    def test_tool_registered(self):
        from agent.cli import build_tool_registry

        registry = build_tool_registry()
        assert "earthquake_proximity" in registry.list_names()

    def test_bandit_arm_exists(self):
        from agent.learning.bandit import DEFAULT_ARMS

        names = [a.name for a in DEFAULT_ARMS]
        assert "seismic_risk" in names

    def test_bandit_arm_tools(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm = next(a for a in DEFAULT_ARMS if a.name == "seismic_risk")
        assert "earthquake_proximity" in arm.tools
