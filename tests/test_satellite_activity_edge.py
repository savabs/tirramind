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
    COUNTRY_BBOX,
    EONET_CATEGORIES,
    FIRMS_MAX_DAYS,
    FIRMS_SOURCES,
    VALID_MODES,
    SatelliteActivityTool,
    _cache_entry,
    _cluster_hotspots,
    _date_to_modis,
    _fetch_eonet,
    _fetch_firms,
    _fetch_ndvi,
    _ndvi_health,
    _parse_bbox,
    _resolve_firms_area,
    _safe_float,
)

# ── FIXTURES ──


@pytest.fixture(autouse=True)
def _reset_firms_availability_memo():
    """The archive-window memo is module-global; never let it leak between tests."""
    import agent.tools.satellite_activity as sat

    sat._availability_memo["windows"] = None
    sat._availability_memo["fetched_at"] = 0.0
    yield
    sat._availability_memo["windows"] = None
    sat._availability_memo["fetched_at"] = 0.0


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
    def test_fire_cache_hit_still_carries_the_structured_payload(self, mock_fetch, tool):
        """A cache hit must return what the PIPELINE consumes, not just prose.

        This assertion used to be `r.output == "cached fire data"` with nothing
        said about `r.data` — a test certifying a data-loss path. ToolOperator
        persists `result.data if result.data is not None else result.output`
        (agent/pipeline/operators.py), so a hit with data=None stored the
        human-readable blob as the node's row, and
        `_extract_satellite_activity` returns [] for a non-dict. Green run,
        row written, signal gone.
        """
        cached_payload = {"mode": "fire", "hotspot_count": 479, "area": "USA"}
        tool._cache.get.return_value = _cache_entry("cached fire data", cached_payload)
        r = tool.execute(mode="fire", area="USA")
        assert r.success
        assert r.output == "cached fire data"
        assert isinstance(r.data, dict), "cache hit dropped the structured payload"
        assert r.data["hotspot_count"] == 479
        mock_fetch.assert_not_called()

    @patch("agent.tools.satellite_activity._fetch_firms")
    def test_legacy_string_cache_entry_is_a_miss_not_a_degraded_hit(self, mock_fetch, tool):
        """An entry written by the old string-only cache must refetch."""
        tool._cache.get.return_value = "cached fire data"
        mock_fetch.return_value = [{"latitude": "30", "longitude": "-90", "frp": "5", "acq_date": "2026-09-20"}]
        r = tool.execute(mode="fire", area="USA")
        assert r.success
        mock_fetch.assert_called_once()
        assert isinstance(r.data, dict)
        assert r.data["hotspot_count"] == 1

    @patch("agent.tools.satellite_activity._fetch_firms")
    def test_cold_and_cached_paths_yield_identical_evidence(self, mock_fetch, tool):
        """The assertion that would have caught the cache bug.

        Run the payload through the real convergence extractor on both paths
        and require the same evidence count. Before the fix the cached path
        handed the extractor a string and it produced zero.
        """
        from agent.convergence.extractors import extract_evidence

        mock_fetch.return_value = [
            {
                "latitude": "30",
                "longitude": "-90",
                "frp": "42.0",
                "confidence": "high",
                "daynight": "D",
                "acq_date": "2026-09-20",
            }
        ]
        cold = tool.execute(mode="fire", area="USA")
        assert cold.success
        # What the tool actually wrote to the cache is what the next run reads.
        written = tool._cache.put.call_args[0][2]
        tool._cache.get.return_value = written
        mock_fetch.reset_mock()

        cached = tool.execute(mode="fire", area="USA")
        mock_fetch.assert_not_called()

        cold_ev = extract_evidence("satellite_activity", cold.data)
        cached_ev = extract_evidence("satellite_activity", cached.data)
        assert len(cold_ev) > 0
        assert len(cached_ev) == len(cold_ev)

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
        tool._cache.get.return_value = _cache_entry(
            "cached veg data", {"mode": "vegetation", "latest_ndvi": 0.61, "anomaly_pct": -3.0}
        )
        r = tool.execute(
            mode="vegetation",
            latitude=40,
            longitude=-90,
            start_date="2024-01-01",
            end_date="2024-06-01",
        )
        assert r.success
        assert r.output == "cached veg data"
        assert isinstance(r.data, dict), "cache hit dropped the structured payload"
        assert r.data["latest_ndvi"] == 0.61
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
        tool._cache.get.return_value = _cache_entry(
            "cached events", {"mode": "events", "event_count": 7, "category_counts": {"wildfires": 7}}
        )
        r = tool.execute(mode="events")
        assert r.success
        assert r.output == "cached events"
        assert isinstance(r.data, dict), "cache hit dropped the structured payload"
        assert r.data["event_count"] == 7
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
    def test_empty_200_body_is_a_failure_not_an_empty_day(self, mock_get):
        """FIRMS always sends the CSV header. A bodyless 200 is a broken fetch.

        This used to return [], which _fire() reported as "No thermal hotspots
        detected" with success=True — a green row indistinguishable from a
        healthy quiet day.
        """
        mock_get.return_value = _mock_response(200, text="")
        result = _fetch_firms("USA", "VIIRS_NOAA20_NRT", 1, "key")
        assert result is None

    @patch("agent.tools.satellite_activity.httpx.get")
    def test_html_error_body_with_200_is_a_failure(self, mock_get):
        """A proxy/error page served as 200 parses to [] — reject it loudly."""
        mock_get.return_value = _mock_response(200, text="<html><body>Service Unavailable</body></html>")
        assert _fetch_firms("USA", "VIIRS_NOAA20_NRT", 1, "key") is None

    @patch("agent.tools.satellite_activity.httpx.get")
    def test_header_only_csv_is_a_legitimate_empty(self, mock_get):
        """A well-formed FIRMS CSV with no rows really is zero detections."""
        header = "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,frp"
        mock_get.return_value = _mock_response(200, text=header)
        assert _fetch_firms("USA", "VIIRS_NOAA20_NRT", 1, "key") == []

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


# ══════════════════════════════════════════════════════════════════════════
# 10. Regression — the fire node that never wrote a row (2026-09-23)
# ══════════════════════════════════════════════════════════════════════════
#
# Two stacked defects kept `fetch_satellite_activity` at zero rows for its
# entire life, and neither was silent-on-the-surface — the operator raised,
# the run went red, and nobody read the log for 26 days:
#
#   1. daily_collection.py registered params={"mode": "fire"} with no 'area',
#      so _fire() returned success=False and ToolOperator raised.
#   2. The error string it returned recommended a country code ("USA"), which
#      the FIRMS /api/area/ endpoint answers with HTTP 400
#      "Invalid area. Expects: [west,south,east,north]." Copying the advice
#      into the DAG would have produced a second, identical outage.
#   3. Same path: days was clamped to 10, but FIRMS answers HTTP 400
#      "Invalid day range. Expects [1..5]." above 5.
#
# These assert the *wire shape* — the URL actually built and the params dict
# actually registered — because every intermediate layer reported fine.


class TestFireWireShapeRegression:
    """The DAG's real params must produce a URL FIRMS accepts."""

    def test_dag_node_params_carry_a_usable_area(self):
        """This is the assertion that would have caught 26 days of loss."""
        from agent.pipeline.dags.daily_collection import build_daily_collection_dag

        dag = build_daily_collection_dag(db_path="/tmp/probe_satellite_activity_dagonly.db")
        node = next(n for n in dag.nodes.values() if n.id == "fetch_satellite_activity")

        assert node.params.get("mode") == "fire"
        area = node.params.get("area", "")
        assert area, "fire mode hard-fails without 'area' — the node writes zero rows"
        # And the area must be one FIRMS actually serves, not a country code.
        assert _resolve_firms_area(area) is not None, f"area {area!r} does not resolve to a FIRMS bbox"
        assert 1 <= int(node.params.get("days", 1)) <= FIRMS_MAX_DAYS

    def test_fetch_firms_url_uses_bbox_and_legal_day_range(self):
        """The URL _fetch_firms builds for the DAG's params must be FIRMS-legal."""
        from agent.pipeline.dags.daily_collection import build_daily_collection_dag

        dag = build_daily_collection_dag(db_path="/tmp/probe_satellite_activity_dagonly.db")
        node = next(n for n in dag.nodes.values() if n.id == "fetch_satellite_activity")

        tool = SatelliteActivityTool()
        tool._firms_key = "test-key"

        with patch("agent.tools.satellite_activity.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, text="latitude,longitude,frp\n1,2,3\n")
            r = tool.execute(**node.params)

        assert r.success, r.output
        url = mock_get.call_args[0][0]
        # Path tail is /<bbox>/<days>; bbox is 4 comma-separated floats.
        area_seg, days_seg = url.rsplit("/", 2)[-2:]
        assert _parse_bbox(area_seg) is not None, f"area segment {area_seg!r} is not a bbox — FIRMS 400s on this"
        assert 1 <= int(days_seg) <= FIRMS_MAX_DAYS, f"day range {days_seg} — FIRMS rejects >{FIRMS_MAX_DAYS}"

    def test_firms_max_days_is_the_probed_value(self):
        """Pin the empirically probed ceiling, independent of the call site.

        The clamp test below used to assert `clamped == FIRMS_MAX_DAYS` while
        the code computes `min(FIRMS_MAX_DAYS, days)` — a tautology. Regressing
        the constant back to the broken 10 left the whole file green (verified
        by editing a scratch copy). This is the assertion that pins the value.
        """
        assert FIRMS_MAX_DAYS == 5, (
            "FIRMS /api/area/csv answers HTTP 400 'Invalid day range. Expects [1..5]' above 5 — probed live 2026-09-23"
        )

    def test_days_clamped_to_firms_limit_not_ten(self):
        """days=99 used to clamp to 10, which FIRMS answers with HTTP 400."""
        tool = SatelliteActivityTool()
        tool._firms_key = "test-key"
        with patch("agent.tools.satellite_activity._fetch_firms", return_value=[]) as mock_fetch:
            tool.execute(mode="fire", area="-125,24,-66,50", days=99)
        # Literal, not FIRMS_MAX_DAYS: the constant is what is on trial.
        assert mock_fetch.call_args[0][2] == 5

    def test_country_code_is_resolved_to_a_bbox_before_fetch(self):
        """Callers passing 'USA' keep working, but FIRMS never sees 'USA'."""
        tool = SatelliteActivityTool()
        tool._firms_key = "test-key"
        with patch("agent.tools.satellite_activity._fetch_firms", return_value=[]) as mock_fetch:
            tool.execute(mode="fire", area="USA")
        sent_area = mock_fetch.call_args[0][0]
        assert sent_area != "USA"
        assert _parse_bbox(sent_area) is not None

    def test_missing_area_error_no_longer_teaches_a_400(self):
        """The old message said 'Use country code (USA)' — that value HTTP 400s."""
        tool = SatelliteActivityTool()
        tool._firms_key = "test-key"
        r = tool.execute(mode="fire")
        assert not r.success
        assert "west,south,east,north" in r.output

    def test_unresolvable_area_is_rejected_locally(self):
        tool = SatelliteActivityTool()
        tool._firms_key = "test-key"
        with patch("agent.tools.satellite_activity._fetch_firms") as mock_fetch:
            r = tool.execute(mode="fire", area="Atlantis")
        assert not r.success
        mock_fetch.assert_not_called()

    def test_backfill_date_reaches_the_url(self):
        """_fetch_firms had no date arg at all; the outage gap needs one."""
        tool = SatelliteActivityTool()
        tool._firms_key = "test-key"
        with (
            patch(
                "agent.tools.satellite_activity._fetch_firms_availability",
                return_value={"VIIRS_NOAA20_NRT": ("2026-07-01", "2026-09-23")},
            ),
            patch("agent.tools.satellite_activity.httpx.get") as mock_get,
        ):
            mock_get.return_value = MagicMock(
                status_code=200,
                text="latitude,longitude,frp,acq_date\n1,2,3,2026-08-20\n",
            )
            r = tool.execute(mode="fire", area="usa", days=1, date="2026-08-20")
        assert r.success, r.output
        assert mock_get.call_args[0][0].endswith("/1/2026-08-20")

    def test_every_country_bbox_shorthand_is_a_legal_bbox(self):
        """A typo in COUNTRY_BBOX would be a silent HTTP 400 at collection time."""
        for code, bbox in COUNTRY_BBOX.items():
            assert _parse_bbox(bbox) is not None, f"COUNTRY_BBOX[{code!r}] = {bbox!r} is not a legal bbox"


# ══════════════════════════════════════════════════════════════════════════
# REGRESSION: the `date` backfill parameter
#
# Three separate ways the new parameter reported success while fetching
# nothing, or fetching the wrong days:
#
#   1. The docstring and the schema both called `date` the END of the
#      days-long window. Against the live API it is the START. Probed
#      2026-09-23, bbox -125,32,-114,42, VIIRS_NOAA20_NRT:
#        days=3 date=2026-09-15 -> 200, 495 rows, acq_dates
#                                  {2026-09-15, 2026-09-16, 2026-09-17}
#        days=5 (no date)       -> 200, 667 rows, acq_dates
#                                  {2026-09-19 .. 2026-09-23}
#      i.e. the dated form is anchored at the start, the undated form ends
#      today. A gap backfill with days>1 therefore fetched the wrong days
#      and stored a green row describing a window it never saw.
#   2. A date outside the NRT rolling archive is answered HTTP 200 with a
#      header-only CSV, not an error. Probed 2026-09-23: date=2026-06-01 ->
#      200 / 0 rows; date=2026-09-01 -> 200 / 1501 rows. Every *genuine*
#      error (bad key, bad source, days=10) is a loud 400, so this is the
#      one case the status check cannot see — and it is exactly the case
#      the parameter exists to exercise.
#   3. The payload carried no observation timestamp at all, so an August
#      window landed stamped with pipeline_data.fetched_at (wall clock).
# ══════════════════════════════════════════════════════════════════════════


_AVAILABLE = {"VIIRS_NOAA20_NRT": ("2026-07-01", "2026-09-23")}


def _dated_csv(*acq_dates):
    rows = [
        {
            "latitude": "36.5",
            "longitude": "-119.5",
            "frp": "12.5",
            "confidence": "nominal",
            "daynight": "D",
            "acq_date": d,
        }
        for d in acq_dates
    ]
    return _firms_csv(*rows)


class TestBackfillDateSemantics:
    def _tool(self):
        t = SatelliteActivityTool()
        t._firms_key = "test-key"
        return t

    def test_date_is_documented_as_the_window_start(self):
        """The schema text must match the live API, not the opposite of it."""
        desc = SatelliteActivityTool.parameters["properties"]["date"]["description"].lower()
        assert "start" in desc
        assert "end date" not in desc

    def test_window_bounds_run_forward_from_date(self):
        """date..date+days-1 — the anchor the live API actually uses."""
        with (
            patch("agent.tools.satellite_activity._fetch_firms_availability", return_value=_AVAILABLE),
            patch("agent.tools.satellite_activity.httpx.get") as mock_get,
        ):
            mock_get.return_value = _mock_response(200, text=_dated_csv("2026-09-15", "2026-09-16", "2026-09-17"))
            r = self._tool().execute(mode="fire", area="usa", days=3, date="2026-09-15")
        assert r.success, r.output
        assert r.data["window_start"] == "2026-09-15"
        assert r.data["window_end"] == "2026-09-17"

    def test_observed_acq_dates_are_carried_on_the_row(self):
        """The row must describe the window it observed, not only fetched_at."""
        with (
            patch("agent.tools.satellite_activity._fetch_firms_availability", return_value=_AVAILABLE),
            patch("agent.tools.satellite_activity.httpx.get") as mock_get,
        ):
            mock_get.return_value = _mock_response(200, text=_dated_csv("2026-09-15", "2026-09-16", "2026-09-17"))
            r = self._tool().execute(mode="fire", area="usa", days=3, date="2026-09-15")
        assert r.data["acq_date_min"] == "2026-09-15"
        assert r.data["acq_date_max"] == "2026-09-17"
        # And the observed span must sit inside the window the tool claims.
        assert r.data["window_start"] <= r.data["acq_date_min"]
        assert r.data["acq_date_max"] <= r.data["window_end"]

    def test_undated_window_ends_today(self):
        """With no date FIRMS returns a TRAILING window — opposite anchor."""
        from datetime import UTC, datetime, timedelta

        with patch("agent.tools.satellite_activity._fetch_firms", return_value=[]):
            r = self._tool().execute(mode="fire", area="usa", days=5)
        today = datetime.now(tz=UTC)
        assert r.data["window_end"] == today.strftime("%Y-%m-%d")
        assert r.data["window_start"] == (today - timedelta(days=4)).strftime("%Y-%m-%d")

    def test_date_before_archive_is_rejected_not_reported_as_zero(self):
        """The whole finding: 200 + header-only CSV must not become a zero row."""
        with (
            patch("agent.tools.satellite_activity._fetch_firms_availability", return_value=_AVAILABLE),
            patch("agent.tools.satellite_activity._fetch_firms") as mock_fetch,
        ):
            r = self._tool().execute(mode="fire", area="usa", days=1, date="2020-01-01")
        assert not r.success, "an unfillable window must be a failed window, not a filled one"
        assert "archive" in r.output.lower()
        mock_fetch.assert_not_called(), "no point spending a transaction on a window FIRMS cannot serve"

    def test_date_after_archive_is_rejected(self):
        with (
            patch("agent.tools.satellite_activity._fetch_firms_availability", return_value=_AVAILABLE),
            patch("agent.tools.satellite_activity._fetch_firms") as mock_fetch,
        ):
            r = self._tool().execute(mode="fire", area="usa", days=1, date="2099-01-01")
        assert not r.success
        mock_fetch.assert_not_called()

    def test_date_inside_archive_passes_through(self):
        with (
            patch("agent.tools.satellite_activity._fetch_firms_availability", return_value=_AVAILABLE),
            patch("agent.tools.satellite_activity._fetch_firms", return_value=[]) as mock_fetch,
        ):
            r = self._tool().execute(mode="fire", area="usa", days=1, date="2026-09-01")
        assert r.success, r.output
        mock_fetch.assert_called_once()
        assert r.data["archive_checked"] is True
        assert r.data["archive_min"] == "2026-07-01"

    def test_unverifiable_archive_is_flagged_not_silently_assumed(self):
        """If the availability endpoint is unreadable, say so on the row."""
        with (
            patch("agent.tools.satellite_activity._fetch_firms_availability", return_value=None),
            patch("agent.tools.satellite_activity._fetch_firms", return_value=[]),
        ):
            r = self._tool().execute(mode="fire", area="usa", days=1, date="2026-09-01")
        assert r.success
        assert r.data["archive_checked"] is False
        assert r.data["archive_min"] is None

    def test_undated_run_never_spends_an_availability_call(self):
        with (
            patch("agent.tools.satellite_activity._fetch_firms_availability") as mock_avail,
            patch("agent.tools.satellite_activity._fetch_firms", return_value=[]),
        ):
            self._tool().execute(mode="fire", area="usa", days=1)
        mock_avail.assert_not_called()

    def test_empty_result_is_marked_on_the_payload(self):
        """A zero must be self-describing, not shaped like a healthy run."""
        with patch("agent.tools.satellite_activity._fetch_firms", return_value=[]):
            empty = self._tool().execute(mode="fire", area="usa", days=1)
        with patch(
            "agent.tools.satellite_activity._fetch_firms",
            return_value=[{"latitude": "36", "longitude": "-119", "frp": "9", "acq_date": "2026-09-22"}],
        ):
            healthy = self._tool().execute(mode="fire", area="usa", days=1)
        assert empty.data["empty_result"] is True
        assert healthy.data["empty_result"] is False
        assert empty.data["acq_date_min"] is None
        assert healthy.data["acq_date_max"] == "2026-09-22"


class TestWindowMustFitInsideTheArchive:
    """Both ends of the window, not just the start date.

    The first repair checked only `date` against the archive. A window that
    BEGINS inside the archive can still run off the end of it, and FIRMS
    answers that with HTTP 200 carrying only the days it holds. Probed live
    2026-09-23 (VIIRS_NOAA20_NRT, archive 2026-07-01..2026-09-23, bbox
    -125,32,-114,42):

        date=2026-09-23 days=5 -> HTTP 200, 76 rows, acq_dates {2026-09-23}
        date=2026-07-01 days=2 -> HTTP 200, 82 rows, acq_dates {07-01, 07-02}

    Through the tool as first repaired, the first of those returned
    success=True with window_start=2026-09-23, window_end=2026-09-27,
    archive_checked=True and empty_result=False — a green row claiming five
    days of which four were never fetched, which a backfill loop would then
    mark filled.
    """

    def _tool(self):
        t = SatelliteActivityTool()
        t._firms_key = "test-key"
        return t

    def test_window_running_past_archive_max_is_rejected(self):
        with (
            patch("agent.tools.satellite_activity._fetch_firms_availability", return_value=_AVAILABLE),
            patch("agent.tools.satellite_activity._fetch_firms") as mock_fetch,
        ):
            r = self._tool().execute(mode="fire", area="usa", days=5, date="2026-09-23")
        assert not r.success, "4 of the 5 requested days do not exist — that is an unfilled window"
        assert "2026-09-23" in r.output and "archive" in r.output.lower()
        mock_fetch.assert_not_called()

    def test_the_rejection_says_how_many_days_are_servable(self):
        """A backfill caller has to be able to act on it, not just retry."""
        with (
            patch("agent.tools.satellite_activity._fetch_firms_availability", return_value=_AVAILABLE),
            patch("agent.tools.satellite_activity._fetch_firms"),
        ):
            r = self._tool().execute(mode="fire", area="usa", days=5, date="2026-09-21")
        assert not r.success
        # 09-21, 09-22, 09-23 exist -> days<=3
        assert "days<=3" in r.output.replace(" ", "")

    def test_a_window_ending_exactly_on_archive_max_is_allowed(self):
        with (
            patch("agent.tools.satellite_activity._fetch_firms_availability", return_value=_AVAILABLE),
            patch("agent.tools.satellite_activity._fetch_firms", return_value=[]) as mock_fetch,
        ):
            r = self._tool().execute(mode="fire", area="usa", days=3, date="2026-09-21")
        assert r.success, r.output
        mock_fetch.assert_called_once()
        assert r.data["window_end"] == "2026-09-23"

    def test_unverifiable_archive_does_not_invent_a_rejection(self):
        """No availability data means 'unknown', never a guessed refusal."""
        with (
            patch("agent.tools.satellite_activity._fetch_firms_availability", return_value=None),
            patch("agent.tools.satellite_activity._fetch_firms", return_value=[]) as mock_fetch,
        ):
            r = self._tool().execute(mode="fire", area="usa", days=5, date="2026-09-23")
        assert r.success
        assert r.data["archive_checked"] is False
        mock_fetch.assert_called_once()


class TestClaimedWindowIsCheckedAgainstWhatArrived:
    """The only assertion that can catch the FIRMS anchor being wrong.

    `date` shipped documented as the window END while the live API treats it
    as the START, and every test of it asserted the URL string, which was
    identical either way. Comparing the acq_dates FIRMS actually returned
    against the window the payload claims is what closes that.
    """

    def _tool(self):
        t = SatelliteActivityTool()
        t._firms_key = "test-key"
        return t

    def test_matching_window_is_flagged_true(self):
        with (
            patch("agent.tools.satellite_activity._fetch_firms_availability", return_value=_AVAILABLE),
            patch("agent.tools.satellite_activity.httpx.get") as mock_get,
        ):
            mock_get.return_value = _mock_response(200, text=_dated_csv("2026-09-15", "2026-09-16"))
            r = self._tool().execute(mode="fire", area="usa", days=3, date="2026-09-15")
        assert r.data["window_matches_observed"] is True

    def test_rows_outside_the_claimed_window_are_flagged_on_the_row(self):
        """If the anchor were flipped back, the row says so instead of lying."""
        with (
            patch("agent.tools.satellite_activity._fetch_firms_availability", return_value=_AVAILABLE),
            patch("agent.tools.satellite_activity.httpx.get") as mock_get,
        ):
            # Claimed window is 2026-09-15..2026-09-17; FIRMS served earlier days.
            mock_get.return_value = _mock_response(200, text=_dated_csv("2026-09-13", "2026-09-14"))
            r = self._tool().execute(mode="fire", area="usa", days=3, date="2026-09-15")
        assert r.success
        assert r.data["window_matches_observed"] is False, (
            "the row claims 2026-09-15..17 while carrying 09-13..09-14 observations"
        )

    def test_undated_trailing_window_contains_its_observations(self):
        """Probed 2026-09-23: undated days=3 -> acq_dates 09-21..09-23."""
        from datetime import UTC, datetime, timedelta

        today = datetime.now(tz=UTC)
        served = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in (2, 1, 0)]
        with patch("agent.tools.satellite_activity.httpx.get") as mock_get:
            mock_get.return_value = _mock_response(200, text=_dated_csv(*served))
            r = self._tool().execute(mode="fire", area="usa", days=3)
        assert r.data["window_matches_observed"] is True, (
            "the undated anchor is a trailing window ending today — observations must sit inside it"
        )


class TestEmptyFireResultIsNeverCached:
    """A cached zero is a zero re-served for the whole TTL (6h in production)."""

    def test_zero_hotspots_is_not_written_to_the_cache(self):
        tool = SatelliteActivityTool()
        tool._firms_key = "test-key"
        tool._cache = MagicMock()
        tool._cache.get.return_value = None
        with patch("agent.tools.satellite_activity._fetch_firms", return_value=[]):
            r = tool.execute(mode="fire", area="usa", days=1)
        assert r.success
        tool._cache.put.assert_not_called(), "a transient empty window must not be pinned for the cache TTL"

    def test_a_healthy_result_is_still_cached(self):
        tool = SatelliteActivityTool()
        tool._firms_key = "test-key"
        tool._cache = MagicMock()
        tool._cache.get.return_value = None
        with patch(
            "agent.tools.satellite_activity._fetch_firms",
            return_value=[{"latitude": "36", "longitude": "-119", "frp": "9", "acq_date": "2026-09-22"}],
        ):
            r = tool.execute(mode="fire", area="usa", days=1)
        assert r.success
        tool._cache.put.assert_called_once()
        entry = tool._cache.put.call_args[0][2]
        assert isinstance(entry["data"], dict) and entry["data"]["hotspot_count"] == 1


class TestFireRowActuallyLands:
    """Row counts, not green ticks. Exercise the DAG's real params end to end.

    The prior regression class asserted URL segments only, so a FIRMS 200
    carrying a header-only CSV produced success=True, hotspot_count=0 and
    exactly one stored row — identical in every observable way to a healthy
    run. These run the tool through a ToolOperator into a THROWAWAY store and
    assert on the stored row's contents.
    """

    @staticmethod
    def _dag_params():
        from agent.pipeline.dags.daily_collection import build_daily_collection_dag

        dag = build_daily_collection_dag(db_path="/tmp/probe_satellite_activity_dagonly.db")
        node = next(n for n in dag.nodes.values() if n.id == "fetch_satellite_activity")
        return dict(node.params)

    @staticmethod
    def _store(tmp_path):
        from agent.pipeline.store import PipelineStore

        return PipelineStore(db_path=str(tmp_path / "probe_satellite.db"))

    def _run_into_store(self, tmp_path, csv_text):
        """Return (result, payload, node_status, stored_rows) for a FIRMS body.

        Mirrors production exactly: ToolOperator's payload fallback
        (agent/pipeline/operators.py:175), the executor's classification of
        that payload (agent/pipeline/executor.py:871 ``nr.status =
        classify_payload_status(result)``) and its rule that a row is stored
        only when the node is ``completed`` (executor.py:653). Using
        ``result.success`` as the store gate instead would be a kinder gate
        than the pipeline's, which is how a zero row got written.
        """
        from agent.pipeline.operators import classify_payload_status

        store = self._store(tmp_path)
        tool = SatelliteActivityTool()
        tool._firms_key = "test-key"
        params = self._dag_params()
        with patch("agent.tools.satellite_activity.httpx.get") as mock_get:
            mock_get.return_value = _mock_response(200, text=csv_text)
            result = tool.execute(**params)

        if result.success:
            payload = result.data if result.data is not None else result.output
            node_status = classify_payload_status(payload)
        else:
            payload, node_status = None, "failed"
        if node_status == "completed":
            store.store_data(source="satellite_activity", params=params, data=payload)
        rows = store.query_data(source="satellite_activity", limit=10)
        return result, payload, node_status, rows

    def test_healthy_csv_lands_a_row_with_hotspots(self, tmp_path):
        csv_text = _dated_csv("2026-09-22", "2026-09-22", "2026-09-23")
        result, payload, node_status, rows = self._run_into_store(tmp_path, csv_text)
        assert result.success, result.output
        assert isinstance(payload, dict), "ToolOperator would persist prose, not a feature dict"
        assert node_status == "completed"
        assert len(rows) == 1
        stored = rows[0]["data"]
        assert stored["hotspot_count"] == 3, f"row landed but carries no hotspots: {stored}"
        assert stored["empty_result"] is False
        assert stored["acq_date_min"] == "2026-09-22"

    def test_html_200_writes_no_row_at_all(self, tmp_path):
        """A 200 that is not a FIRMS CSV used to store a green zero row."""
        result, _payload, _status, rows = self._run_into_store(tmp_path, "<html><body>502 Bad Gateway</body></html>")
        assert not result.success
        assert rows == [], "a malformed 200 must not leave a row behind"

    def test_empty_200_writes_no_row_at_all(self, tmp_path):
        result, _payload, _status, rows = self._run_into_store(tmp_path, "")
        assert not result.success
        assert rows == []

    def test_zero_hotspot_window_is_skipped_not_counted_as_a_collection(self, tmp_path):
        """The finding: a zero used to be a 'completed' node with one row.

        A header-only CSV is a well-formed FIRMS response, so the fetch guard
        cannot reject it — and for the DAG's CONUS bbox it is never a real
        reading (probed live 2026-09-23: CONUS days=1 -> 76+ hotspots for a
        California sub-box alone). The payload therefore says ``skipped`` in
        the one word the executor reads, so no row lands and the run does not
        report a healthy collection it never made.
        """
        header = "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,frp"
        result, payload, node_status, rows = self._run_into_store(tmp_path, header)
        assert result.success, result.output
        assert payload["hotspot_count"] == 0
        assert payload["empty_result"] is True
        assert node_status == "skipped", "a zero must not be classified as a completed collection"
        assert rows == [], "the executor stores only 'completed' nodes — a zero must leave no row"
        assert payload["reason"], "a skipped node that gives no reason is under-reporting"

    def test_a_zero_never_reaches_the_convergence_extractor(self, tmp_path):
        """hotspot_count=0 used to become four real evidence items."""
        from agent.convergence.extractors import extract_evidence

        header = "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,frp"
        _result, payload, node_status, rows = self._run_into_store(tmp_path, header)
        assert node_status == "skipped"
        assert rows == []
        # Belt and braces: even if some other path did hand it over, the
        # fabricated zero is what the extractor would turn into evidence.
        fabricated = extract_evidence("satellite_activity", payload)
        assert len(fabricated) >= 1, "sanity: the extractor really does mine a zero"
        # ...which is exactly why no row may be stored for it.
