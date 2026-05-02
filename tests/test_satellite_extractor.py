"""Comprehensive edge-case test suite for satellite_activity convergence extractor.

Tests the full pipeline:
  1. Tool data= dict schemas (fire, vegetation, events)
  2. Extractor → Evidence conversion for all 3 modes
  3. Edge cases: empty data, missing fields, NaN/None values, boundary values
  4. Property checks: category validity, signal_id format, direction bounds, TTL

Per spec: docs/specs/tier2_satellite_spec.md
"""

from __future__ import annotations

import time

import pytest

from agent.convergence.evidence import Evidence
from agent.convergence.extractors import (
    _extract_satellite_activity,
    _extract_satellite_events,
    _extract_satellite_fire,
    _extract_satellite_vegetation,
    extract_evidence,
    registered_tools,
)
from agent.convergence.taxonomy import CATEGORIES

# ═══════════════════════════════════════════════════════════════
#  Fixtures — representative data dicts matching tool output
# ═══════════════════════════════════════════════════════════════


@pytest.fixture()
def fire_data_typical() -> dict:
    """Typical fire mode data= dict with moderate activity."""
    return {
        "mode": "fire",
        "area": "USA",
        "source": "VIIRS_NOAA20_NRT",
        "days": 1,
        "hotspot_count": 150,
        "frp_avg": 12.5,
        "frp_max": 85.0,
        "frp_total": 1875.0,
        "confidence_counts": {"high": 50, "nominal": 80, "low": 20},
        "daynight_counts": {"D": 100, "N": 50},
        "cluster_count": 8,
        "clusters": [
            {
                "centroid_lat": 34.05,
                "centroid_lon": -118.25,
                "count": 25,
                "avg_frp": 15.3,
                "max_frp": 85.0,
                "total_frp": 382.5,
            },
            {
                "centroid_lat": 37.77,
                "centroid_lon": -122.42,
                "count": 10,
                "avg_frp": 8.0,
                "max_frp": 20.0,
                "total_frp": 80.0,
            },
        ],
    }


@pytest.fixture()
def fire_data_empty() -> dict:
    """Fire mode with zero hotspots (no activity)."""
    return {
        "mode": "fire",
        "area": "ISL",
        "source": "VIIRS_NOAA20_NRT",
        "days": 1,
        "hotspot_count": 0,
        "frp_avg": 0.0,
        "frp_max": 0.0,
        "frp_total": 0.0,
        "confidence_counts": {},
        "daynight_counts": {},
        "cluster_count": 0,
        "clusters": [],
    }


@pytest.fixture()
def vegetation_data_typical() -> dict:
    """Typical vegetation mode with crop stress detected."""
    return {
        "mode": "vegetation",
        "latitude": 40.0,
        "longitude": -95.0,
        "start_date": "2025-01-01",
        "end_date": "2026-03-30",
        "observation_count": 8,
        "latest_ndvi": 0.32,
        "latest_date": "2026-03-17",
        "latest_health": "moderate",
        "avg_ndvi": 0.55,
        "min_ndvi": 0.25,
        "max_ndvi": 0.72,
        "anomaly_pct": -41.82,
        "series": [
            {"date": "2025-07-05", "ndvi": 0.72, "health": "dense", "pixels": 1},
            {"date": "2026-03-17", "ndvi": 0.32, "health": "moderate", "pixels": 1},
        ],
    }


@pytest.fixture()
def vegetation_data_empty() -> dict:
    """Vegetation mode with zero observations."""
    return {
        "mode": "vegetation",
        "latitude": 0.0,
        "longitude": 0.0,
        "start_date": "2026-01-01",
        "end_date": "2026-03-01",
        "observation_count": 0,
        "latest_ndvi": 0.0,
        "latest_date": "",
        "latest_health": "bare_soil",
        "avg_ndvi": 0.0,
        "min_ndvi": 0.0,
        "max_ndvi": 0.0,
        "anomaly_pct": 0.0,
        "series": [],
    }


@pytest.fixture()
def events_data_typical() -> dict:
    """Typical events mode with mixed natural events."""
    return {
        "mode": "events",
        "days": 30,
        "status": "open",
        "category_filter": None,
        "event_count": 25,
        "category_counts": {
            "wildfires": 10,
            "severeStorms": 5,
            "volcanoes": 3,
            "earthquakes": 4,
            "floods": 3,
        },
        "events": [
            {
                "title": "Wildfire in California",
                "categories": ["wildfires"],
                "lat": 34.05,
                "lon": -118.25,
                "date": "2026-04-01",
            },
        ],
    }


@pytest.fixture()
def events_data_empty() -> dict:
    """Events mode with zero events."""
    return {
        "mode": "events",
        "days": 30,
        "status": "open",
        "category_filter": None,
        "event_count": 0,
        "category_counts": {},
        "events": [],
    }


# ═══════════════════════════════════════════════════════════════
#  Registration tests
# ═══════════════════════════════════════════════════════════════


class TestRegistration:
    """Verify satellite_activity is properly registered as an extractor."""

    def test_satellite_activity_registered(self):
        assert "satellite_activity" in registered_tools()

    def test_extract_evidence_dispatches_to_satellite(self):
        data = {
            "mode": "fire",
            "hotspot_count": 10,
            "frp_total": 50.0,
            "frp_max": 10.0,
            "cluster_count": 1,
        }
        result = extract_evidence("satellite_activity", data)
        assert len(result) == 4
        assert all(isinstance(e, Evidence) for e in result)

    def test_extract_evidence_returns_empty_for_none_data(self):
        result = extract_evidence("satellite_activity", None)
        assert result == []


# ═══════════════════════════════════════════════════════════════
#  Dispatcher tests
# ═══════════════════════════════════════════════════════════════


class TestDispatcher:
    """Tests for the top-level _extract_satellite_activity dispatcher."""

    def test_non_dict_returns_empty(self):
        assert _extract_satellite_activity("satellite_activity", "text") == []
        assert _extract_satellite_activity("satellite_activity", 42) == []
        assert _extract_satellite_activity("satellite_activity", None) == []
        assert _extract_satellite_activity("satellite_activity", [1, 2]) == []

    def test_missing_mode_returns_empty(self):
        assert _extract_satellite_activity("satellite_activity", {}) == []
        assert _extract_satellite_activity("satellite_activity", {"key": "val"}) == []

    def test_unknown_mode_returns_empty(self):
        assert _extract_satellite_activity("satellite_activity", {"mode": "radar"}) == []
        assert _extract_satellite_activity("satellite_activity", {"mode": ""}) == []
        assert _extract_satellite_activity("satellite_activity", {"mode": 123}) == []

    def test_dispatches_fire(self, fire_data_typical):
        result = _extract_satellite_activity("satellite_activity", fire_data_typical)
        signal_ids = {e.signal_id for e in result}
        assert "satellite.fire.hotspot_count" in signal_ids

    def test_dispatches_vegetation(self, vegetation_data_typical):
        result = _extract_satellite_activity("satellite_activity", vegetation_data_typical)
        signal_ids = {e.signal_id for e in result}
        assert "satellite.vegetation.ndvi_latest" in signal_ids

    def test_dispatches_events(self, events_data_typical):
        result = _extract_satellite_activity("satellite_activity", events_data_typical)
        signal_ids = {e.signal_id for e in result}
        assert "satellite.events.active_count" in signal_ids


# ═══════════════════════════════════════════════════════════════
#  Fire mode tests
# ═══════════════════════════════════════════════════════════════


class TestFireExtractor:
    """Comprehensive tests for the fire mode extractor."""

    def test_typical_fire_produces_4_signals(self, fire_data_typical):
        result = _extract_satellite_fire("satellite_activity", fire_data_typical)
        assert len(result) == 4
        ids = {e.signal_id for e in result}
        assert ids == {
            "satellite.fire.hotspot_count",
            "satellite.fire.frp_total",
            "satellite.fire.frp_max",
            "satellite.fire.cluster_count",
        }

    def test_fire_all_evidence_valid(self, fire_data_typical):
        result = _extract_satellite_fire("satellite_activity", fire_data_typical)
        for ev in result:
            assert isinstance(ev, Evidence)
            assert ev.category == "physical_disruption"
            assert ev.category in CATEGORIES
            assert ev.source == "satellite_activity"
            assert isinstance(ev.value, (int, float))
            assert ev.direction in (-1, 0, 1)
            assert isinstance(ev.tags, tuple)
            assert all(isinstance(t, str) for t in ev.tags)
            assert ev.ttl == 21_600
            assert ev.confidence > 0

    def test_fire_direction_thresholds(self):
        """Verify direction triggers at documented thresholds."""
        # Below thresholds
        data = {
            "mode": "fire",
            "hotspot_count": 100,  # exactly threshold, not >
            "frp_total": 1000.0,
            "frp_max": 100.0,
            "cluster_count": 5,
        }
        result = _extract_satellite_fire("satellite_activity", data)
        for ev in result:
            assert ev.direction == 0, f"{ev.signal_id} should be 0 at threshold"

        # Above thresholds
        data_high = {
            "mode": "fire",
            "hotspot_count": 101,
            "frp_total": 1001.0,
            "frp_max": 101.0,
            "cluster_count": 6,
        }
        result_high = _extract_satellite_fire("satellite_activity", data_high)
        for ev in result_high:
            assert ev.direction == 1, f"{ev.signal_id} should be 1 above threshold"

    def test_fire_zero_hotspots_still_emits(self, fire_data_empty):
        result = _extract_satellite_fire("satellite_activity", fire_data_empty)
        assert len(result) == 4
        by_id = {e.signal_id: e for e in result}
        assert by_id["satellite.fire.hotspot_count"].value == 0.0
        assert by_id["satellite.fire.frp_total"].value == 0.0
        assert by_id["satellite.fire.frp_max"].value == 0.0
        assert by_id["satellite.fire.cluster_count"].value == 0.0
        for ev in result:
            assert ev.direction == 0

    def test_fire_missing_hotspot_count_returns_empty(self):
        """Data dict without hotspot_count key → no signals."""
        data = {"mode": "fire", "frp_total": 100.0}
        assert _extract_satellite_fire("satellite_activity", data) == []

    def test_fire_none_hotspot_count_returns_empty(self):
        data = {"mode": "fire", "hotspot_count": None}
        assert _extract_satellite_fire("satellite_activity", data) == []

    def test_fire_string_frp_values_handled(self):
        """Non-numeric FRP values default to 0.0 via _safe_float."""
        data = {
            "mode": "fire",
            "hotspot_count": 5,
            "frp_total": "not_a_number",
            "frp_max": "",
            "cluster_count": "abc",
        }
        result = _extract_satellite_fire("satellite_activity", data)
        assert len(result) == 4
        by_id = {e.signal_id: e for e in result}
        assert by_id["satellite.fire.frp_total"].value == 0.0
        assert by_id["satellite.fire.frp_max"].value == 0.0
        assert by_id["satellite.fire.cluster_count"].value == 0.0

    def test_fire_negative_frp_preserved(self):
        """Negative FRP (physically impossible but defensively handled)."""
        data = {
            "mode": "fire",
            "hotspot_count": 1,
            "frp_total": -5.0,
            "frp_max": -1.0,
            "cluster_count": 0,
        }
        result = _extract_satellite_fire("satellite_activity", data)
        by_id = {e.signal_id: e for e in result}
        assert by_id["satellite.fire.frp_total"].value == -5.0
        assert by_id["satellite.fire.frp_max"].value == -1.0

    def test_fire_very_large_values(self):
        """Extremely large fire events (e.g. massive wildfire)."""
        data = {
            "mode": "fire",
            "hotspot_count": 50000,
            "frp_total": 500000.0,
            "frp_max": 5000.0,
            "cluster_count": 200,
        }
        result = _extract_satellite_fire("satellite_activity", data)
        assert len(result) == 4
        for ev in result:
            assert ev.direction == 1

    def test_fire_float_hotspot_count(self):
        """Hotspot count as float (e.g. from JSON parsing)."""
        data = {
            "mode": "fire",
            "hotspot_count": 50.0,
            "frp_total": 100.0,
            "frp_max": 20.0,
            "cluster_count": 2.0,
        }
        result = _extract_satellite_fire("satellite_activity", data)
        assert len(result) == 4
        by_id = {e.signal_id: e for e in result}
        assert by_id["satellite.fire.hotspot_count"].value == 50.0

    def test_fire_missing_frp_fields_default_zero(self):
        """Only hotspot_count present; other fields absent → 0.0."""
        data = {"mode": "fire", "hotspot_count": 10}
        result = _extract_satellite_fire("satellite_activity", data)
        assert len(result) == 4
        by_id = {e.signal_id: e for e in result}
        assert by_id["satellite.fire.frp_total"].value == 0.0
        assert by_id["satellite.fire.frp_max"].value == 0.0
        assert by_id["satellite.fire.cluster_count"].value == 0.0

    def test_fire_timestamps_are_recent(self, fire_data_typical):
        now = time.time()
        result = _extract_satellite_fire("satellite_activity", fire_data_typical)
        for ev in result:
            assert abs(ev.timestamp - now) < 5.0


# ═══════════════════════════════════════════════════════════════
#  Vegetation mode tests
# ═══════════════════════════════════════════════════════════════


class TestVegetationExtractor:
    """Comprehensive tests for the vegetation mode extractor."""

    def test_typical_vegetation_produces_3_signals(self, vegetation_data_typical):
        result = _extract_satellite_vegetation("satellite_activity", vegetation_data_typical)
        assert len(result) == 3
        ids = {e.signal_id for e in result}
        assert ids == {
            "satellite.vegetation.ndvi_latest",
            "satellite.vegetation.anomaly_pct",
            "satellite.vegetation.health_class_ordinal",
        }

    def test_vegetation_all_evidence_valid(self, vegetation_data_typical):
        result = _extract_satellite_vegetation("satellite_activity", vegetation_data_typical)
        for ev in result:
            assert isinstance(ev, Evidence)
            assert ev.category == "supply_chain"
            assert ev.category in CATEGORIES
            assert ev.source == "satellite_activity"
            assert isinstance(ev.value, (int, float))
            assert ev.direction in (-1, 0, 1)
            assert isinstance(ev.tags, tuple)
            assert ev.ttl == 604_800

    def test_vegetation_crop_stress_direction(self, vegetation_data_typical):
        """Anomaly < -10% → direction = -1 (crop stress)."""
        result = _extract_satellite_vegetation("satellite_activity", vegetation_data_typical)
        by_id = {e.signal_id: e for e in result}
        anom = by_id["satellite.vegetation.anomaly_pct"]
        assert anom.value == -41.82
        assert anom.direction == -1

    def test_vegetation_above_average_direction(self):
        """Anomaly > +10% → direction = +1 (above-average growth)."""
        data = {
            "mode": "vegetation",
            "observation_count": 5,
            "latest_ndvi": 0.8,
            "latest_date": "2026-03-20",
            "latest_health": "dense",
            "avg_ndvi": 0.6,
            "anomaly_pct": 33.33,
        }
        result = _extract_satellite_vegetation("satellite_activity", data)
        by_id = {e.signal_id: e for e in result}
        assert by_id["satellite.vegetation.anomaly_pct"].direction == 1

    def test_vegetation_neutral_anomaly(self):
        """Anomaly between -10% and +10% → direction = 0."""
        data = {
            "mode": "vegetation",
            "observation_count": 5,
            "latest_ndvi": 0.55,
            "latest_date": "2026-03-20",
            "latest_health": "moderate",
            "avg_ndvi": 0.55,
            "anomaly_pct": 0.0,
        }
        result = _extract_satellite_vegetation("satellite_activity", data)
        by_id = {e.signal_id: e for e in result}
        assert by_id["satellite.vegetation.anomaly_pct"].direction == 0

    def test_vegetation_boundary_anomaly_minus_10(self):
        """Exactly -10.0% → direction = 0 (threshold is <, not <=)."""
        data = {
            "mode": "vegetation",
            "observation_count": 3,
            "latest_ndvi": 0.45,
            "latest_date": "2026-03-20",
            "latest_health": "moderate",
            "avg_ndvi": 0.5,
            "anomaly_pct": -10.0,
        }
        result = _extract_satellite_vegetation("satellite_activity", data)
        by_id = {e.signal_id: e for e in result}
        assert by_id["satellite.vegetation.anomaly_pct"].direction == 0

    def test_vegetation_boundary_anomaly_plus_10(self):
        """Exactly +10.0% → direction = 0."""
        data = {
            "mode": "vegetation",
            "observation_count": 3,
            "latest_ndvi": 0.55,
            "latest_date": "2026-03-20",
            "latest_health": "moderate",
            "avg_ndvi": 0.5,
            "anomaly_pct": 10.0,
        }
        result = _extract_satellite_vegetation("satellite_activity", data)
        by_id = {e.signal_id: e for e in result}
        assert by_id["satellite.vegetation.anomaly_pct"].direction == 0

    def test_vegetation_health_ordinal_mapping(self):
        """Every NDVI health class maps to the correct ordinal."""
        health_to_ordinal = {
            "water_or_barren": 0.0,
            "bare_soil": 1.0,
            "sparse": 2.0,
            "moderate": 3.0,
            "healthy": 4.0,
            "dense": 5.0,
        }
        for health, expected_ordinal in health_to_ordinal.items():
            data = {
                "mode": "vegetation",
                "observation_count": 1,
                "latest_ndvi": 0.5,
                "latest_date": "2026-01-01",
                "latest_health": health,
                "avg_ndvi": 0.5,
                "anomaly_pct": 0.0,
            }
            result = _extract_satellite_vegetation("satellite_activity", data)
            by_id = {e.signal_id: e for e in result}
            assert by_id["satellite.vegetation.health_class_ordinal"].value == expected_ordinal, (
                f"Health '{health}' → expected ordinal {expected_ordinal}"
            )

    def test_vegetation_unknown_health_defaults_to_bare_soil(self):
        """Unknown health string → ordinal 1.0 (bare_soil)."""
        data = {
            "mode": "vegetation",
            "observation_count": 1,
            "latest_ndvi": 0.5,
            "latest_date": "2026-01-01",
            "latest_health": "UNKNOWN_GARBAGE",
            "avg_ndvi": 0.5,
            "anomaly_pct": 0.0,
        }
        result = _extract_satellite_vegetation("satellite_activity", data)
        by_id = {e.signal_id: e for e in result}
        assert by_id["satellite.vegetation.health_class_ordinal"].value == 1.0

    def test_vegetation_empty_health_string(self):
        """Empty health string → ordinal 1.0 (bare_soil)."""
        data = {
            "mode": "vegetation",
            "observation_count": 1,
            "latest_ndvi": 0.5,
            "latest_date": "2026-01-01",
            "latest_health": "",
            "avg_ndvi": 0.5,
            "anomaly_pct": 0.0,
        }
        result = _extract_satellite_vegetation("satellite_activity", data)
        by_id = {e.signal_id: e for e in result}
        assert by_id["satellite.vegetation.health_class_ordinal"].value == 1.0

    def test_vegetation_zero_observations_with_ndvi_data_still_emits(self):
        """observation_count=0 but latest_ndvi present → still emit."""
        data = {
            "mode": "vegetation",
            "observation_count": 0,
            "latest_ndvi": 0.4,
            "latest_date": "2026-01-01",
            "latest_health": "moderate",
            "avg_ndvi": 0.4,
            "anomaly_pct": 0.0,
        }
        result = _extract_satellite_vegetation("satellite_activity", data)
        assert len(result) == 3

    def test_vegetation_truly_empty_returns_empty(self):
        """No observation_count, no latest_ndvi, no avg_ndvi → empty."""
        data = {"mode": "vegetation", "observation_count": 0}
        result = _extract_satellite_vegetation("satellite_activity", data)
        assert result == []

    def test_vegetation_negative_ndvi(self):
        """Negative NDVI (water/barren) is valid and preserved."""
        data = {
            "mode": "vegetation",
            "observation_count": 1,
            "latest_ndvi": -0.15,
            "latest_date": "2026-01-01",
            "latest_health": "water_or_barren",
            "avg_ndvi": -0.1,
            "anomaly_pct": -50.0,
        }
        result = _extract_satellite_vegetation("satellite_activity", data)
        by_id = {e.signal_id: e for e in result}
        assert by_id["satellite.vegetation.ndvi_latest"].value == -0.15
        assert by_id["satellite.vegetation.health_class_ordinal"].value == 0.0

    def test_vegetation_extreme_anomaly(self):
        """Very large anomaly (e.g. -90%) with extreme crop failure."""
        data = {
            "mode": "vegetation",
            "observation_count": 10,
            "latest_ndvi": 0.05,
            "latest_date": "2026-01-01",
            "latest_health": "bare_soil",
            "avg_ndvi": 0.5,
            "anomaly_pct": -90.0,
        }
        result = _extract_satellite_vegetation("satellite_activity", data)
        by_id = {e.signal_id: e for e in result}
        assert by_id["satellite.vegetation.anomaly_pct"].direction == -1
        assert by_id["satellite.vegetation.anomaly_pct"].value == -90.0

    def test_vegetation_string_ndvi_defaults_zero(self):
        """Non-numeric latest_ndvi → _safe_float defaults to 0.0."""
        data = {
            "mode": "vegetation",
            "observation_count": 1,
            "latest_ndvi": "invalid",
            "latest_date": "2026-01-01",
            "latest_health": "moderate",
            "avg_ndvi": "also_invalid",
            "anomaly_pct": "nope",
        }
        result = _extract_satellite_vegetation("satellite_activity", data)
        assert len(result) == 3
        by_id = {e.signal_id: e for e in result}
        assert by_id["satellite.vegetation.ndvi_latest"].value == 0.0
        assert by_id["satellite.vegetation.anomaly_pct"].value == 0.0

    def test_vegetation_none_health_defaults(self):
        """None health → str(None) = 'none' → unknown → ordinal 1.0."""
        data = {
            "mode": "vegetation",
            "observation_count": 1,
            "latest_ndvi": 0.5,
            "latest_date": "2026-01-01",
            "latest_health": None,
            "avg_ndvi": 0.5,
            "anomaly_pct": 0.0,
        }
        result = _extract_satellite_vegetation("satellite_activity", data)
        by_id = {e.signal_id: e for e in result}
        assert by_id["satellite.vegetation.health_class_ordinal"].value == 1.0

    def test_vegetation_timestamps_are_recent(self, vegetation_data_typical):
        now = time.time()
        result = _extract_satellite_vegetation("satellite_activity", vegetation_data_typical)
        for ev in result:
            assert abs(ev.timestamp - now) < 5.0


# ═══════════════════════════════════════════════════════════════
#  Events mode tests
# ═══════════════════════════════════════════════════════════════


class TestEventsExtractor:
    """Comprehensive tests for the events mode extractor."""

    def test_typical_events_produces_3_signals(self, events_data_typical):
        result = _extract_satellite_events("satellite_activity", events_data_typical)
        assert len(result) == 3
        ids = {e.signal_id for e in result}
        assert ids == {
            "satellite.events.active_count",
            "satellite.events.wildfire_count",
            "satellite.events.severe_storm_count",
        }

    def test_events_all_evidence_valid(self, events_data_typical):
        result = _extract_satellite_events("satellite_activity", events_data_typical)
        for ev in result:
            assert isinstance(ev, Evidence)
            assert ev.category == "physical_disruption"
            assert ev.category in CATEGORIES
            assert ev.source == "satellite_activity"
            assert isinstance(ev.value, (int, float))
            assert ev.direction in (-1, 0, 1)
            assert isinstance(ev.tags, tuple)
            assert ev.ttl == 43_200

    def test_events_direction_thresholds(self, events_data_typical):
        """25 events > 20 → direction=1 for active_count."""
        result = _extract_satellite_events("satellite_activity", events_data_typical)
        by_id = {e.signal_id: e for e in result}
        assert by_id["satellite.events.active_count"].direction == 1
        assert by_id["satellite.events.wildfire_count"].direction == 1  # 10 > 5
        assert by_id["satellite.events.severe_storm_count"].direction == 1  # 5 > 3

    def test_events_direction_at_threshold(self):
        """Exactly at threshold → direction = 0."""
        data = {
            "mode": "events",
            "event_count": 20,
            "category_counts": {"wildfires": 5, "severeStorms": 3},
        }
        result = _extract_satellite_events("satellite_activity", data)
        for ev in result:
            assert ev.direction == 0

    def test_events_below_threshold(self):
        data = {
            "mode": "events",
            "event_count": 5,
            "category_counts": {"wildfires": 2, "severeStorms": 1},
        }
        result = _extract_satellite_events("satellite_activity", data)
        for ev in result:
            assert ev.direction == 0

    def test_events_zero_events(self, events_data_empty):
        result = _extract_satellite_events("satellite_activity", events_data_empty)
        assert len(result) == 3
        by_id = {e.signal_id: e for e in result}
        assert by_id["satellite.events.active_count"].value == 0.0
        assert by_id["satellite.events.wildfire_count"].value == 0.0
        assert by_id["satellite.events.severe_storm_count"].value == 0.0

    def test_events_missing_event_count_returns_empty(self):
        data = {"mode": "events", "category_counts": {"wildfires": 5}}
        assert _extract_satellite_events("satellite_activity", data) == []

    def test_events_none_event_count_returns_empty(self):
        data = {"mode": "events", "event_count": None}
        assert _extract_satellite_events("satellite_activity", data) == []

    def test_events_missing_category_counts_defaults_zero(self):
        """No category_counts → wildfires and storms default to 0."""
        data = {"mode": "events", "event_count": 10}
        result = _extract_satellite_events("satellite_activity", data)
        assert len(result) == 3
        by_id = {e.signal_id: e for e in result}
        assert by_id["satellite.events.wildfire_count"].value == 0.0
        assert by_id["satellite.events.severe_storm_count"].value == 0.0

    def test_events_non_dict_category_counts(self):
        """category_counts is not a dict → defaults safely."""
        data = {"mode": "events", "event_count": 10, "category_counts": "invalid"}
        result = _extract_satellite_events("satellite_activity", data)
        assert len(result) == 3
        by_id = {e.signal_id: e for e in result}
        assert by_id["satellite.events.wildfire_count"].value == 0.0

    def test_events_only_volcanoes_no_fires_storms(self):
        """Event categories present but no wildfires or storms."""
        data = {
            "mode": "events",
            "event_count": 3,
            "category_counts": {"volcanoes": 3},
        }
        result = _extract_satellite_events("satellite_activity", data)
        by_id = {e.signal_id: e for e in result}
        assert by_id["satellite.events.wildfire_count"].value == 0.0
        assert by_id["satellite.events.severe_storm_count"].value == 0.0

    def test_events_very_large_counts(self):
        """Extreme event counts (e.g. during major disaster season)."""
        data = {
            "mode": "events",
            "event_count": 500,
            "category_counts": {"wildfires": 200, "severeStorms": 150},
        }
        result = _extract_satellite_events("satellite_activity", data)
        assert len(result) == 3
        for ev in result:
            assert ev.direction == 1

    def test_events_string_counts_handled(self):
        """String values in category_counts → _safe_float handles."""
        data = {
            "mode": "events",
            "event_count": 10,
            "category_counts": {"wildfires": "three", "severeStorms": ""},
        }
        result = _extract_satellite_events("satellite_activity", data)
        by_id = {e.signal_id: e for e in result}
        assert by_id["satellite.events.wildfire_count"].value == 0.0
        assert by_id["satellite.events.severe_storm_count"].value == 0.0

    def test_events_timestamps_are_recent(self, events_data_typical):
        now = time.time()
        result = _extract_satellite_events("satellite_activity", events_data_typical)
        for ev in result:
            assert abs(ev.timestamp - now) < 5.0


# ═══════════════════════════════════════════════════════════════
#  Signal property tests (all modes)
# ═══════════════════════════════════════════════════════════════


class TestSignalProperties:
    """Cross-cutting property checks for all satellite signals."""

    ALL_FIRE_IDS = {
        "satellite.fire.hotspot_count",
        "satellite.fire.frp_total",
        "satellite.fire.frp_max",
        "satellite.fire.cluster_count",
    }
    ALL_VEG_IDS = {
        "satellite.vegetation.ndvi_latest",
        "satellite.vegetation.anomaly_pct",
        "satellite.vegetation.health_class_ordinal",
    }
    ALL_EVENT_IDS = {
        "satellite.events.active_count",
        "satellite.events.wildfire_count",
        "satellite.events.severe_storm_count",
    }

    def _all_evidence(self) -> list[Evidence]:
        fire_data = {
            "mode": "fire",
            "hotspot_count": 50,
            "frp_total": 500.0,
            "frp_max": 50.0,
            "cluster_count": 4,
        }
        veg_data = {
            "mode": "vegetation",
            "observation_count": 5,
            "latest_ndvi": 0.6,
            "latest_date": "2026-01-01",
            "latest_health": "healthy",
            "avg_ndvi": 0.55,
            "anomaly_pct": 9.1,
        }
        event_data = {
            "mode": "events",
            "event_count": 15,
            "category_counts": {"wildfires": 4, "severeStorms": 2},
        }
        all_evidence = []
        all_evidence.extend(_extract_satellite_fire("satellite_activity", fire_data))
        all_evidence.extend(_extract_satellite_vegetation("satellite_activity", veg_data))
        all_evidence.extend(_extract_satellite_events("satellite_activity", event_data))
        return all_evidence

    def test_total_signal_count(self):
        """10 unique signals across all 3 modes."""
        evidence = self._all_evidence()
        assert len(evidence) == 10

    def test_all_signal_ids_follow_naming_convention(self):
        """All signal_ids start with 'satellite.' and have 3+ dot-separated parts."""
        for ev in self._all_evidence():
            assert ev.signal_id.startswith("satellite.")
            parts = ev.signal_id.split(".")
            assert len(parts) >= 3, f"Signal {ev.signal_id} has too few parts"

    def test_all_signal_ids_unique(self):
        ids = [ev.signal_id for ev in self._all_evidence()]
        assert len(ids) == len(set(ids))

    def test_all_categories_in_taxonomy(self):
        for ev in self._all_evidence():
            assert ev.category in CATEGORIES

    def test_fire_category_is_physical_disruption(self):
        fire = _extract_satellite_fire(
            "satellite_activity",
            {
                "mode": "fire",
                "hotspot_count": 1,
                "frp_total": 1,
                "frp_max": 1,
                "cluster_count": 0,
            },
        )
        for ev in fire:
            assert ev.category == "physical_disruption"

    def test_vegetation_category_is_supply_chain(self):
        veg = _extract_satellite_vegetation(
            "satellite_activity",
            {
                "mode": "vegetation",
                "observation_count": 1,
                "latest_ndvi": 0.5,
                "latest_date": "2026-01-01",
                "latest_health": "moderate",
                "avg_ndvi": 0.5,
                "anomaly_pct": 0.0,
            },
        )
        for ev in veg:
            assert ev.category == "supply_chain"

    def test_events_category_is_physical_disruption(self):
        events = _extract_satellite_events(
            "satellite_activity",
            {
                "mode": "events",
                "event_count": 3,
                "category_counts": {"wildfires": 1},
            },
        )
        for ev in events:
            assert ev.category == "physical_disruption"

    def test_all_confidence_in_valid_range(self):
        for ev in self._all_evidence():
            assert 0.0 < ev.confidence <= 1.0

    def test_all_ttl_positive(self):
        for ev in self._all_evidence():
            assert ev.ttl > 0

    def test_fire_ttl_is_6_hours(self):
        for ev in _extract_satellite_fire(
            "satellite_activity",
            {
                "mode": "fire",
                "hotspot_count": 1,
                "frp_total": 1,
                "frp_max": 1,
                "cluster_count": 0,
            },
        ):
            assert ev.ttl == 21_600

    def test_vegetation_ttl_is_7_days(self):
        for ev in _extract_satellite_vegetation(
            "satellite_activity",
            {
                "mode": "vegetation",
                "observation_count": 1,
                "latest_ndvi": 0.5,
                "latest_date": "2026-01-01",
                "latest_health": "moderate",
                "avg_ndvi": 0.5,
                "anomaly_pct": 0.0,
            },
        ):
            assert ev.ttl == 604_800

    def test_events_ttl_is_12_hours(self):
        for ev in _extract_satellite_events(
            "satellite_activity",
            {
                "mode": "events",
                "event_count": 1,
                "category_counts": {},
            },
        ):
            assert ev.ttl == 43_200

    def test_all_tags_are_string_tuples(self):
        for ev in self._all_evidence():
            assert isinstance(ev.tags, tuple)
            assert len(ev.tags) >= 2
            assert all(isinstance(t, str) for t in ev.tags)
            assert "satellite" in ev.tags

    def test_all_sources_are_satellite_activity(self):
        for ev in self._all_evidence():
            assert ev.source == "satellite_activity"


# ═══════════════════════════════════════════════════════════════
#  Tool data= dict integration tests
# ═══════════════════════════════════════════════════════════════


class TestToolDataIntegration:
    """Test that the tool's data= dicts from satellite_activity.py flow
    correctly through the extractor pipeline."""

    def test_fire_tool_data_to_evidence_round_trip(self):
        """Simulate the exact data= dict the tool would produce."""
        tool_data = {
            "mode": "fire",
            "area": "BRA",
            "source": "VIIRS_NOAA20_NRT",
            "days": 3,
            "hotspot_count": 1200,
            "frp_avg": 8.5,
            "frp_max": 250.0,
            "frp_total": 10200.0,
            "confidence_counts": {"high": 400, "nominal": 600, "low": 200},
            "daynight_counts": {"D": 800, "N": 400},
            "cluster_count": 15,
            "clusters": [
                {
                    "centroid_lat": -3.5,
                    "centroid_lon": -60.0,
                    "count": 100,
                    "avg_frp": 12.0,
                    "max_frp": 250.0,
                    "total_frp": 1200.0,
                }
            ],
        }
        result = extract_evidence("satellite_activity", tool_data)
        assert len(result) == 4
        by_id = {e.signal_id: e for e in result}
        assert by_id["satellite.fire.hotspot_count"].value == 1200.0
        assert by_id["satellite.fire.hotspot_count"].direction == 1
        assert by_id["satellite.fire.frp_total"].value == 10200.0
        assert by_id["satellite.fire.frp_total"].direction == 1
        assert by_id["satellite.fire.frp_max"].value == 250.0
        assert by_id["satellite.fire.frp_max"].direction == 1
        assert by_id["satellite.fire.cluster_count"].value == 15.0
        assert by_id["satellite.fire.cluster_count"].direction == 1

    def test_vegetation_tool_data_to_evidence_round_trip(self):
        tool_data = {
            "mode": "vegetation",
            "latitude": 38.0,
            "longitude": -100.0,
            "start_date": "2025-06-01",
            "end_date": "2026-04-01",
            "observation_count": 12,
            "latest_ndvi": 0.22,
            "latest_date": "2026-03-25",
            "latest_health": "sparse",
            "avg_ndvi": 0.48,
            "min_ndvi": 0.18,
            "max_ndvi": 0.65,
            "anomaly_pct": -54.17,
            "series": [
                {"date": "2025-06-01", "ndvi": 0.65, "health": "healthy", "pixels": 4},
                {"date": "2026-03-25", "ndvi": 0.22, "health": "sparse", "pixels": 4},
            ],
        }
        result = extract_evidence("satellite_activity", tool_data)
        assert len(result) == 3
        by_id = {e.signal_id: e for e in result}
        assert by_id["satellite.vegetation.ndvi_latest"].value == 0.22
        assert by_id["satellite.vegetation.anomaly_pct"].value == -54.17
        assert by_id["satellite.vegetation.anomaly_pct"].direction == -1
        assert by_id["satellite.vegetation.health_class_ordinal"].value == 2.0  # sparse

    def test_events_tool_data_to_evidence_round_trip(self):
        tool_data = {
            "mode": "events",
            "days": 7,
            "status": "open",
            "category_filter": None,
            "event_count": 45,
            "category_counts": {
                "wildfires": 20,
                "severeStorms": 8,
                "volcanoes": 5,
                "earthquakes": 7,
                "floods": 5,
            },
            "events": [
                {
                    "title": "Test Event",
                    "categories": ["wildfires"],
                    "lat": 36.0,
                    "lon": -120.0,
                    "date": "2026-04-05",
                },
            ],
        }
        result = extract_evidence("satellite_activity", tool_data)
        assert len(result) == 3
        by_id = {e.signal_id: e for e in result}
        assert by_id["satellite.events.active_count"].value == 45.0
        assert by_id["satellite.events.active_count"].direction == 1
        assert by_id["satellite.events.wildfire_count"].value == 20.0
        assert by_id["satellite.events.wildfire_count"].direction == 1
        assert by_id["satellite.events.severe_storm_count"].value == 8.0
        assert by_id["satellite.events.severe_storm_count"].direction == 1

    def test_fire_empty_result_tool_data(self):
        """Data dict from tool when no hotspots detected."""
        tool_data = {
            "mode": "fire",
            "area": "ISL",
            "source": "VIIRS_NOAA20_NRT",
            "days": 1,
            "hotspot_count": 0,
            "frp_avg": 0.0,
            "frp_max": 0.0,
            "frp_total": 0.0,
            "confidence_counts": {},
            "daynight_counts": {},
            "cluster_count": 0,
            "clusters": [],
        }
        result = extract_evidence("satellite_activity", tool_data)
        assert len(result) == 4
        for ev in result:
            assert ev.direction == 0

    def test_vegetation_empty_result_tool_data(self):
        """Data dict from tool when no NDVI observations."""
        tool_data = {
            "mode": "vegetation",
            "latitude": 0.0,
            "longitude": 0.0,
            "start_date": "2026-01-01",
            "end_date": "2026-03-01",
            "observation_count": 0,
            "latest_ndvi": 0.0,
            "latest_date": "",
            "latest_health": "bare_soil",
            "avg_ndvi": 0.0,
            "min_ndvi": 0.0,
            "max_ndvi": 0.0,
            "anomaly_pct": 0.0,
            "series": [],
        }
        result = extract_evidence("satellite_activity", tool_data)
        # observation_count=0 but latest_ndvi=0.0 is present → emits
        assert len(result) == 3

    def test_events_empty_result_tool_data(self):
        """Data dict from tool when no events detected."""
        tool_data = {
            "mode": "events",
            "days": 30,
            "status": "open",
            "category_filter": None,
            "event_count": 0,
            "category_counts": {},
            "events": [],
        }
        result = extract_evidence("satellite_activity", tool_data)
        assert len(result) == 3
        for ev in result:
            assert ev.value == 0.0
            assert ev.direction == 0
