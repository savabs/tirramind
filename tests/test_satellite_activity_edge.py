"""
Edge-case tests for SatelliteActivityTool (7b-I).

Coverage targets:
- Invalid / missing / boundary parameters
- Mode validation (fire, vegetation, events)
- Fire: API key handling, source validation, area parsing, FIRMS CSV parsing,
  cluster computation, FRP stats, cache hit/miss
- Vegetation: lat/lon bounds, date conversion, NDVI parsing, health classification,
  anomaly computation, empty series
- Events: category validation, status validation, EONET JSON parsing, category counts
- HTTP errors, empty responses, malformed data
- Helper functions (_ndvi_health, _cluster_hotspots, _date_to_modis, _safe_float)
- Integration: tool count = 47, arm count = 35
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from agent.tools.satellite_activity import (
    EONET_CATEGORIES,
    FIRMS_SOURCES,
    VALID_MODES,
    SatelliteActivityTool,
    _cluster_hotspots,
    _date_to_modis,
    _fetch_eonet,
    _fetch_firms,
    _fetch_ndvi,
    _ndvi_health,
    _safe_float,
)

# ── FIXTURES ──


@pytest.fixture
def tool(monkeypatch):
    # FeaturePreflight.for_api_key falls back to os.getenv(env_var) whenever
    # key_value is falsy, so setting t._firms_key alone does not fully
    # isolate these tests from a real/leaked TIRRA_NASA_FIRMS_KEY in the
    # ambient environment -- observed failing in-suite (passed alone) when
    # something upstream left the var set. delenv pins this hermetic.
    # Fixed 2026-08-27.
    monkeypatch.delenv("TIRRA_NASA_FIRMS_KEY", raising=False)
    cache = MagicMock()
    cache.get.return_value = None
    t = SatelliteActivityTool(cache=cache)
    t._firms_key = "test-key"
    return t


@pytest.fixture
def tool_no_key(monkeypatch):
    # See `tool` fixture above: pin hermetic against ambient env leakage.
    monkeypatch.delenv("TIRRA_NASA_FIRMS_KEY", raising=False)
    cache = MagicMock()
    cache.get.return_value = None
    t = SatelliteActivityTool(cache=cache)
    t._firms_key = None
    return t


# ── HELPER: mock response builder ──


def _mock_response(status_code=200, text="", json_data=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    if json_data is not None:
        resp.json.return_value = json_data
    else:
        resp.json.side_effect = ValueError("No JSON")
    return resp


def _firms_csv(*rows):
    """Build FIRMS-style CSV text from row dicts."""
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for r in rows:
        lines.append(",".join(str(r.get(h, "")) for h in headers))
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# 1. Mode validation
# ══════════════════════════════════════════════════════════════════════════


class TestModeValidation:
    def test_invalid_mode(self, tool):
        r = tool.execute(mode="invalid")
        assert not r.success
        assert "Invalid mode" in r.output

    def test_empty_mode(self, tool):
        r = tool.execute(mode="")
        assert not r.success

    def test_none_mode(self, tool):
        r = tool.execute()
        assert not r.success

    def test_valid_modes_constant(self):
        assert {"fire", "vegetation", "events"} == VALID_MODES

    def test_case_insensitive_mode(self, tool):
        # Fire mode without key returns API key error, proving mode was recognized
        tool._firms_key = None
        r = tool.execute(mode="FIRE")
        assert "TIRRA_NASA_FIRMS_KEY" in r.output

    def test_mode_whitespace(self, tool):
        tool._firms_key = None
        r = tool.execute(mode="  fire  ")
        assert "TIRRA_NASA_FIRMS_KEY" in r.output


# ══════════════════════════════════════════════════════════════════════════
# 2. Fire mode
# ══════════════════════════════════════════════════════════════════════════


class TestFireMode:
    def test_no_api_key(self, tool_no_key):
        r = tool_no_key.execute(mode="fire", area="USA")
        assert not r.success
        assert "TIRRA_NASA_FIRMS_KEY" in r.output

    def test_missing_area(self, tool):
        r = tool.execute(mode="fire")
        assert not r.success
        assert "area" in r.output.lower()

    def test_empty_area(self, tool):
        r = tool.execute(mode="fire", area="")
        assert not r.success

    def test_invalid_source(self, tool):
        r = tool.execute(mode="fire", area="USA", source="INVALID_SAT")
        assert not r.success
        assert "Invalid source" in r.output

    def test_valid_sources(self):
        assert "VIIRS_NOAA20_NRT" in FIRMS_SOURCES
        assert "MODIS_NRT" in FIRMS_SOURCES

    def test_days_clamped_min(self, tool):
        with patch("agent.tools.satellite_activity._fetch_firms", return_value=[]):
            r = tool.execute(mode="fire", area="USA", days=0)
            assert r.success

    def test_days_clamped_max(self, tool):
        with patch("agent.tools.satellite_activity._fetch_firms", return_value=[]):
            r = tool.execute(mode="fire", area="USA", days=99)
            assert r.success

    def test_days_non_numeric(self, tool):
        with patch("agent.tools.satellite_activity._fetch_firms", return_value=[]):
            r = tool.execute(mode="fire", area="USA", days="abc")
            assert r.success

    @patch("agent.tools.satellite_activity._fetch_firms")
    def test_empty_hotspots(self, mock_fetch, tool):
        mock_fetch.return_value = []
        r = tool.execute(mode="fire", area="USA")
        assert r.success
        assert "No thermal hotspots" in r.output

    @patch("agent.tools.satellite_activity._fetch_firms")
    def test_fetch_failure(self, mock_fetch, tool):
        mock_fetch.return_value = None
        r = tool.execute(mode="fire", area="USA")
        assert not r.success
        assert "Failed" in r.output

    @patch("agent.tools.satellite_activity._fetch_firms")
    def test_successful_fire(self, mock_fetch, tool):
        mock_fetch.return_value = [
            {
                "latitude": "30.0",
                "longitude": "-90.0",
                "frp": "25.5",
                "confidence": "high",
                "daynight": "D",
            },
            {
                "latitude": "30.01",
                "longitude": "-90.01",
                "frp": "12.3",
                "confidence": "nominal",
                "daynight": "D",
            },
        ]
        r = tool.execute(mode="fire", area="USA")
        assert r.success
        assert "FIRMS" in r.output
        assert "Total hotspots: 2" in r.output
        tool._cache.put.assert_called_once()

    @patch("agent.tools.satellite_activity._fetch_firms")
    def test_fire_cache_hit(self, mock_fetch, tool):
        tool._cache.get.return_value = "cached fire data"
        r = tool.execute(mode="fire", area="USA")
        assert r.success
        assert r.output == "cached fire data"
        mock_fetch.assert_not_called()

    @patch("agent.tools.satellite_activity._fetch_firms")
    def test_fire_frp_stats(self, mock_fetch, tool):
        mock_fetch.return_value = [
            {
                "latitude": "10",
                "longitude": "20",
                "frp": "100",
                "confidence": "high",
                "daynight": "N",
            },
            {
                "latitude": "10",
                "longitude": "20",
                "frp": "200",
                "confidence": "high",
                "daynight": "N",
            },
        ]
        r = tool.execute(mode="fire", area="BRA")
        assert r.success
        assert "max=200.0" in r.output
        assert "total=300.0" in r.output

    @patch("agent.tools.satellite_activity._fetch_firms")
    def test_fire_malformed_frp(self, mock_fetch, tool):
        mock_fetch.return_value = [
            {
                "latitude": "10",
                "longitude": "20",
                "frp": "not_a_number",
                "confidence": "?",
                "daynight": "?",
            },
        ]
        r = tool.execute(mode="fire", area="USA")
        assert r.success
        assert "Total hotspots: 1" in r.output

    @patch("agent.tools.satellite_activity._fetch_firms")
    def test_default_source(self, mock_fetch, tool):
        mock_fetch.return_value = []
        tool.execute(mode="fire", area="USA")
        mock_fetch.assert_called_once()
        call_args = mock_fetch.call_args
        assert call_args[0][1] == "VIIRS_NOAA20_NRT"  # default source


# ══════════════════════════════════════════════════════════════════════════
# 3. Vegetation mode
# ══════════════════════════════════════════════════════════════════════════


class TestVegetationMode:
    def test_missing_lat_lon(self, tool):
        r = tool.execute(mode="vegetation")
        assert not r.success
        assert "latitude" in r.output.lower()

    def test_missing_longitude(self, tool):
        r = tool.execute(mode="vegetation", latitude=40.0)
        assert not r.success

    def test_non_numeric_lat(self, tool):
        r = tool.execute(mode="vegetation", latitude="abc", longitude=0)
        assert not r.success
        assert "numeric" in r.output.lower()

    def test_lat_out_of_range(self, tool):
        r = tool.execute(
            mode="vegetation",
            latitude=91,
            longitude=0,
            start_date="2024-01-01",
            end_date="2024-06-01",
        )
        assert not r.success
        assert "-90" in r.output

    def test_lon_out_of_range(self, tool):
        r = tool.execute(
            mode="vegetation",
            latitude=0,
            longitude=181,
            start_date="2024-01-01",
            end_date="2024-06-01",
        )
        assert not r.success

    def test_missing_dates(self, tool):
        r = tool.execute(mode="vegetation", latitude=40, longitude=-90)
        assert not r.success
        assert "start_date" in r.output.lower() or "end_date" in r.output.lower()

    def test_km_radius_clamped(self, tool):
        with patch("agent.tools.satellite_activity._fetch_ndvi", return_value={"subset": []}):
            r = tool.execute(
                mode="vegetation",
                latitude=40,
                longitude=-90,
                start_date="2024-01-01",
                end_date="2024-06-01",
                km_radius=999,
            )
            assert r.success

    def test_km_radius_non_numeric(self, tool):
        with patch("agent.tools.satellite_activity._fetch_ndvi", return_value={"subset": []}):
            r = tool.execute(
                mode="vegetation",
                latitude=40,
                longitude=-90,
                start_date="2024-01-01",
                end_date="2024-06-01",
                km_radius="abc",
            )
            assert r.success

    @patch("agent.tools.satellite_activity._fetch_ndvi")
    def test_fetch_failure(self, mock_fetch, tool):
        mock_fetch.return_value = None
        r = tool.execute(
            mode="vegetation",
            latitude=40,
            longitude=-90,
            start_date="2024-01-01",
            end_date="2024-06-01",
        )
        assert not r.success
        assert "Failed" in r.output

    @patch("agent.tools.satellite_activity._fetch_ndvi")
    def test_empty_subset(self, mock_fetch, tool):
        mock_fetch.return_value = {"subset": []}
        r = tool.execute(
            mode="vegetation",
            latitude=40,
            longitude=-90,
            start_date="2024-01-01",
            end_date="2024-06-01",
        )
        assert r.success
        assert "No NDVI data" in r.output

    @patch("agent.tools.satellite_activity._fetch_ndvi")
    def test_successful_ndvi(self, mock_fetch, tool):
        mock_fetch.return_value = {
            "subset": [
                {
                    "calendar_date": "2024-01-01",
                    "scale": 0.0001,
                    "data": [3500, 3600, 3700],
                },
                {
                    "calendar_date": "2024-01-17",
                    "scale": 0.0001,
                    "data": [4000, 4100, 4200],
                },
            ]
        }
        r = tool.execute(
            mode="vegetation",
            latitude=40,
            longitude=-90,
            start_date="2024-01-01",
            end_date="2024-06-01",
        )
        assert r.success
        assert "NDVI" in r.output
        assert "Crop Health" in r.output or "Time series" in r.output
        tool._cache.put.assert_called_once()

    @patch("agent.tools.satellite_activity._fetch_ndvi")
    def test_ndvi_anomaly_computation(self, mock_fetch, tool):
        mock_fetch.return_value = {
            "subset": [
                {"calendar_date": "2024-01-01", "scale": 0.0001, "data": [5000]},
                {"calendar_date": "2024-02-01", "scale": 0.0001, "data": [5000]},
                {"calendar_date": "2024-03-01", "scale": 0.0001, "data": [2000]},
            ]
        }
        r = tool.execute(
            mode="vegetation",
            latitude=40,
            longitude=-90,
            start_date="2024-01-01",
            end_date="2024-06-01",
        )
        assert r.success
        assert "Anomaly" in r.output

    @patch("agent.tools.satellite_activity._fetch_ndvi")
    def test_vegetation_cache_hit(self, mock_fetch, tool):
        tool._cache.get.return_value = "cached veg data"
        r = tool.execute(
            mode="vegetation",
            latitude=40,
            longitude=-90,
            start_date="2024-01-01",
            end_date="2024-06-01",
        )
        assert r.success
        assert r.output == "cached veg data"
        mock_fetch.assert_not_called()

    @patch("agent.tools.satellite_activity._fetch_ndvi")
    def test_no_valid_data_points(self, mock_fetch, tool):
        mock_fetch.return_value = {
            "subset": [
                {"calendar_date": "2024-01-01", "data": []},
            ]
        }
        r = tool.execute(
            mode="vegetation",
            latitude=40,
            longitude=-90,
            start_date="2024-01-01",
            end_date="2024-06-01",
        )
        assert r.success
        assert "No valid" in r.output or "No NDVI" in r.output


# ══════════════════════════════════════════════════════════════════════════
# 4. Events mode
# ══════════════════════════════════════════════════════════════════════════


class TestEventsMode:
    def test_invalid_category(self, tool):
        r = tool.execute(mode="events", category="fake_category")
        assert not r.success
        assert "Invalid category" in r.output

    def test_valid_categories(self):
        assert "wildfires" in EONET_CATEGORIES
        assert "volcanoes" in EONET_CATEGORIES
        assert "severeStorms" in EONET_CATEGORIES

    def test_invalid_status(self, tool):
        r = tool.execute(mode="events", status="maybe")
        assert not r.success
        assert "open" in r.output or "closed" in r.output

    @patch("agent.tools.satellite_activity._fetch_eonet")
    def test_fetch_failure(self, mock_fetch, tool):
        mock_fetch.return_value = None
        r = tool.execute(mode="events")
        assert not r.success
        assert "Failed" in r.output

    @patch("agent.tools.satellite_activity._fetch_eonet")
    def test_no_events(self, mock_fetch, tool):
        mock_fetch.return_value = []
        r = tool.execute(mode="events")
        assert r.success
        assert "No open" in r.output or "No" in r.output

    @patch("agent.tools.satellite_activity._fetch_eonet")
    def test_successful_events(self, mock_fetch, tool):
        mock_fetch.return_value = [
            {
                "title": "Wildfire in California",
                "categories": [{"id": "wildfires"}],
                "geometry": [{"coordinates": [-120.5, 37.2], "date": "2024-03-15T00:00:00Z"}],
            },
            {
                "title": "Volcanic Eruption in Iceland",
                "categories": [{"id": "volcanoes"}],
                "geometry": [{"coordinates": [-21.3, 63.6], "date": "2024-03-14T00:00:00Z"}],
            },
        ]
        r = tool.execute(mode="events")
        assert r.success
        assert "EONET" in r.output
        assert "Total events: 2" in r.output
        assert "Wildfire" in r.output

    @patch("agent.tools.satellite_activity._fetch_eonet")
    def test_events_cache_hit(self, mock_fetch, tool):
        tool._cache.get.return_value = "cached events"
        r = tool.execute(mode="events")
        assert r.success
        assert r.output == "cached events"
        mock_fetch.assert_not_called()

    @patch("agent.tools.satellite_activity._fetch_eonet")
    def test_events_category_counts(self, mock_fetch, tool):
        mock_fetch.return_value = [
            {"title": "F1", "categories": [{"id": "wildfires"}], "geometry": []},
            {"title": "F2", "categories": [{"id": "wildfires"}], "geometry": []},
            {"title": "V1", "categories": [{"id": "volcanoes"}], "geometry": []},
        ]
        r = tool.execute(mode="events")
        assert r.success
        assert "wildfires" in r.output

    @patch("agent.tools.satellite_activity._fetch_eonet")
    def test_events_days_clamped(self, mock_fetch, tool):
        mock_fetch.return_value = []
        tool.execute(mode="events", days=999)
        call_args = mock_fetch.call_args
        assert call_args[0][1] == 365  # clamped to 365

    @patch("agent.tools.satellite_activity._fetch_eonet")
    def test_events_bbox_passthrough(self, mock_fetch, tool):
        mock_fetch.return_value = []
        tool.execute(mode="events", area="-130,25,-60,50")
        call_args = mock_fetch.call_args
        assert call_args[0][3] == "-130,25,-60,50"

    @patch("agent.tools.satellite_activity._fetch_eonet")
    def test_events_closed_status(self, mock_fetch, tool):
        mock_fetch.return_value = []
        tool.execute(mode="events", status="closed")
        call_args = mock_fetch.call_args
        assert call_args[0][2] == "closed"

    @patch("agent.tools.satellite_activity._fetch_eonet")
    def test_events_no_category_filter(self, mock_fetch, tool):
        mock_fetch.return_value = []
        tool.execute(mode="events")
        call_args = mock_fetch.call_args
        assert call_args[0][0] is None  # no category filter

    @patch("agent.tools.satellite_activity._fetch_eonet")
    def test_events_many_truncated(self, mock_fetch, tool):
        events = [{"title": f"Event {i}", "categories": [{"id": "wildfires"}], "geometry": []} for i in range(30)]
        mock_fetch.return_value = events
        r = tool.execute(mode="events")
        assert r.success
        assert "10 more events" in r.output


# ══════════════════════════════════════════════════════════════════════════
# 5. Helper functions
# ══════════════════════════════════════════════════════════════════════════


class TestHelpers:
    # -- _ndvi_health --
    def test_ndvi_water(self):
        assert _ndvi_health(-0.1) == "water_or_barren"

    def test_ndvi_bare(self):
        assert _ndvi_health(0.1) == "bare_soil"

    def test_ndvi_sparse(self):
        assert _ndvi_health(0.2) == "sparse"

    def test_ndvi_moderate(self):
        assert _ndvi_health(0.4) == "moderate"

    def test_ndvi_healthy(self):
        assert _ndvi_health(0.6) == "healthy"

    def test_ndvi_dense(self):
        assert _ndvi_health(0.8) == "dense"

    def test_ndvi_boundary_zero(self):
        assert _ndvi_health(0.0) == "bare_soil"

    def test_ndvi_boundary_015(self):
        assert _ndvi_health(0.15) == "sparse"

    # -- _date_to_modis --
    def test_modis_jan1(self):
        assert _date_to_modis("2024-01-01") == "A2024001"

    def test_modis_feb1(self):
        assert _date_to_modis("2024-02-01") == "A2024032"

    def test_modis_dec31(self):
        assert _date_to_modis("2024-12-31") == "A2024366"  # 2024 is leap year

    def test_modis_invalid(self):
        assert _date_to_modis("not-a-date") is None

    def test_modis_none(self):
        assert _date_to_modis(None) is None

    # -- _safe_float --
    def test_safe_float_normal(self):
        assert _safe_float("3.14") == 3.14

    def test_safe_float_int(self):
        assert _safe_float(42) == 42.0

    def test_safe_float_bad(self):
        assert _safe_float("abc") == 0.0

    def test_safe_float_none(self):
        assert _safe_float(None) == 0.0

    def test_safe_float_custom_default(self):
        assert _safe_float("abc", -1.0) == -1.0

    # -- _cluster_hotspots --
    def test_cluster_empty(self):
        assert _cluster_hotspots([]) == []

    def test_cluster_single(self):
        pts = [{"latitude": "10.0", "longitude": "20.0", "frp": "50"}]
        clusters = _cluster_hotspots(pts)
        assert len(clusters) == 1
        assert clusters[0]["count"] == 1
        assert clusters[0]["avg_frp"] == 50.0

    def test_cluster_same_cell(self):
        pts = [
            {"latitude": "10.05", "longitude": "20.05", "frp": "30"},
            {"latitude": "10.06", "longitude": "20.06", "frp": "70"},
        ]
        clusters = _cluster_hotspots(pts, cell_size_deg=0.1)
        assert len(clusters) == 1
        assert clusters[0]["count"] == 2
        assert clusters[0]["avg_frp"] == 50.0

    def test_cluster_different_cells(self):
        pts = [
            {"latitude": "10.0", "longitude": "20.0", "frp": "30"},
            {"latitude": "20.0", "longitude": "30.0", "frp": "70"},
        ]
        clusters = _cluster_hotspots(pts)
        assert len(clusters) == 2
        # Sorted by total_frp descending
        assert clusters[0]["total_frp"] == 70.0

    def test_cluster_sorted_by_total_frp(self):
        pts = [
            {"latitude": "10.0", "longitude": "20.0", "frp": "10"},
            {"latitude": "50.0", "longitude": "60.0", "frp": "200"},
        ]
        clusters = _cluster_hotspots(pts)
        assert clusters[0]["total_frp"] > clusters[1]["total_frp"]

    def test_cluster_malformed_coords(self):
        pts = [
            {"latitude": "abc", "longitude": "xyz", "frp": "10"},
        ]
        clusters = _cluster_hotspots(pts)
        assert len(clusters) == 0

    def test_cluster_zero_frp(self):
        pts = [{"latitude": "10", "longitude": "20", "frp": "0"}]
        clusters = _cluster_hotspots(pts)
        assert len(clusters) == 1
        assert clusters[0]["avg_frp"] == 0.0


# ══════════════════════════════════════════════════════════════════════════
# 6. Fetch functions (HTTP layer)
# ══════════════════════════════════════════════════════════════════════════


class TestFetchFirms:
    @patch("agent.tools.satellite_activity.httpx.get")
    def test_success(self, mock_get):
        csv = _firms_csv({"latitude": "10", "longitude": "20", "frp": "50", "confidence": "high"})
        mock_get.return_value = _mock_response(200, text=csv)
        result = _fetch_firms("USA", "VIIRS_NOAA20_NRT", 1, "key")
        assert result is not None
        assert len(result) == 1

    @patch("agent.tools.satellite_activity.httpx.get")
    def test_http_error(self, mock_get):
        mock_get.return_value = _mock_response(403)
        result = _fetch_firms("USA", "VIIRS_NOAA20_NRT", 1, "key")
        assert result is None

    @patch("agent.tools.satellite_activity.httpx.get")
    def test_empty_response(self, mock_get):
        mock_get.return_value = _mock_response(200, text="")
        result = _fetch_firms("USA", "VIIRS_NOAA20_NRT", 1, "key")
        assert result == []

    @patch("agent.tools.satellite_activity.httpx.get")
    def test_network_error(self, mock_get):
        import httpx

        mock_get.side_effect = httpx.ConnectError("timeout")
        result = _fetch_firms("USA", "VIIRS_NOAA20_NRT", 1, "key")
        assert result is None


class TestFetchNdvi:
    @patch("agent.tools.satellite_activity.httpx.get")
    def test_success(self, mock_get):
        mock_get.return_value = _mock_response(
            200, json_data={"subset": [{"calendar_date": "2024-01-01", "data": [5000]}]}
        )
        result = _fetch_ndvi(40, -90, "2024-01-01", "2024-06-01")
        assert result is not None
        assert "subset" in result

    @patch("agent.tools.satellite_activity.httpx.get")
    def test_http_error(self, mock_get):
        mock_get.return_value = _mock_response(500)
        result = _fetch_ndvi(40, -90, "2024-01-01", "2024-06-01")
        assert result is None

    def test_invalid_start_date(self):
        result = _fetch_ndvi(40, -90, "not-a-date", "2024-06-01")
        assert result is None

    def test_invalid_end_date(self):
        result = _fetch_ndvi(40, -90, "2024-01-01", "bad")
        assert result is None


class TestFetchEonet:
    @patch("agent.tools.satellite_activity.httpx.get")
    def test_success(self, mock_get):
        mock_get.return_value = _mock_response(200, json_data={"events": [{"title": "Fire 1"}]})
        result = _fetch_eonet()
        assert result is not None
        assert len(result) == 1

    @patch("agent.tools.satellite_activity.httpx.get")
    def test_http_error(self, mock_get):
        mock_get.return_value = _mock_response(500)
        result = _fetch_eonet()
        assert result is None

    @patch("agent.tools.satellite_activity.httpx.get")
    def test_network_error(self, mock_get):
        import httpx

        mock_get.side_effect = httpx.ConnectError("fail")
        result = _fetch_eonet()
        assert result is None


# ══════════════════════════════════════════════════════════════════════════
# 7. API key handling
# ══════════════════════════════════════════════════════════════════════════


class TestApiKey:
    def test_key_from_env(self):
        with patch.dict(os.environ, {"TIRRA_NASA_FIRMS_KEY": "my-firms-key"}):
            t = SatelliteActivityTool()
            assert t._firms_key == "my-firms-key"

    def test_no_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("TIRRA_NASA_FIRMS_KEY", None)
            t = SatelliteActivityTool()
            assert t._firms_key is None

    def test_empty_key(self):
        with patch.dict(os.environ, {"TIRRA_NASA_FIRMS_KEY": "  "}):
            t = SatelliteActivityTool()
            assert t._firms_key is None


# ══════════════════════════════════════════════════════════════════════════
# 8. Tool metadata
# ══════════════════════════════════════════════════════════════════════════


class TestToolMetadata:
    def test_name(self):
        t = SatelliteActivityTool()
        assert t.name == "satellite_activity"

    def test_description(self):
        t = SatelliteActivityTool()
        assert "fire" in t.description.lower()
        assert "vegetation" in t.description.lower()
        assert "events" in t.description.lower()

    def test_params_has_mode(self):
        t = SatelliteActivityTool()
        assert "mode" in t.parameters["properties"]

    def test_mode_required(self):
        t = SatelliteActivityTool()
        assert "mode" in t.parameters.get("required", [])


# ══════════════════════════════════════════════════════════════════════════
# 9. Integration counts
# ══════════════════════════════════════════════════════════════════════════


class TestIntegration:
    def test_tool_count(self):
        from agent.cli import build_tool_registry

        registry = build_tool_registry()
        names = registry.list_names()
        assert len(names) == 61, f"Expected 61 tools, got {len(names)}: {names}"

    def test_arm_count(self):
        from agent.learning.bandit import DEFAULT_ARMS

        assert len(DEFAULT_ARMS) == 48, f"Expected 48 arms, got {len(DEFAULT_ARMS)}"

    def test_satellite_registered(self):
        from agent.cli import build_tool_registry

        registry = build_tool_registry()
        assert "satellite_activity" in registry.list_names()

    def test_satellite_arm_exists(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm_names = [a.name for a in DEFAULT_ARMS]
        assert "satellite_surveillance" in arm_names
