"""
Edge case tests for AIS Vessel Tracking tool (7b-D).

Covers: mode routing, parameter validation, bounding box math, ship type
filtering, named areas, vessel lookup, port calls, destination flow,
cache integration, HTTP errors, schema validation, registry integration.
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from agent.tools.ais_vessel import (
    AISVesselTool,
    _in_bbox,
    _ship_type_label,
    _ship_type_matches,
    _NAMED_AREAS,
    _NAV_STATUS,
    _SHIP_TYPE_RANGES,
)
from agent.tools.base import ToolResult


# ── Helpers ─────────────────────────────────────────────────────────


def _make_feature(
    mmsi: int,
    lat: float,
    lon: float,
    sog: float = 5.0,
    cog: float = 180.0,
    nav_stat: int = 0,
    heading: int = 180,
) -> dict:
    """Build a single AIS GeoJSON feature."""
    return {
        "mmsi": mmsi,
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": {
            "mmsi": mmsi,
            "sog": sog,
            "cog": cog,
            "navStat": nav_stat,
            "rot": 0,
            "posAcc": True,
            "raim": False,
            "heading": heading,
            "timestamp": 59,
        },
    }


def _make_meta(
    mmsi: int,
    name: str = "TEST VESSEL",
    ship_type: int = 70,
    destination: str = "NLRTM",
    imo: int = 9999999,
) -> dict:
    """Build a single vessel metadata entry."""
    return {
        "callSign": "XYZQ",
        "destination": destination,
        "draught": 105,
        "eta": 416128,
        "imo": imo,
        "mmsi": mmsi,
        "name": name,
        "posType": 1,
        "referencePointA": 100,
        "referencePointB": 30,
        "referencePointC": 15,
        "referencePointD": 10,
        "shipType": ship_type,
        "timestamp": 1591521868371,
    }


def _make_port_call(
    name: str = "TEST SHIP",
    port: str = "FIHEL",
    prev_port: str = "SEGOT",
    next_port: str = "FIRAA",
    cargo: bool = True,
) -> dict:
    """Build a single port call entry."""
    return {
        "portCallId": 12345,
        "vesselName": name,
        "portToVisit": port,
        "prevPort": prev_port,
        "nextPort": next_port,
        "arrivalWithCargo": cargo,
        "domesticTrafficArrival": False,
    }


def _tool_with_mock_cache():
    """Return (tool, mock_cache) pair."""
    cache = MagicMock()
    cache.get.return_value = None  # cache miss by default
    tool = AISVesselTool(cache=cache)
    return tool, cache


# ── Unit tests: pure functions ──────────────────────────────────────


class TestInBbox:
    def test_inside(self):
        assert _in_bbox(60.0, 25.0, 59.0, 61.0, 24.0, 26.0)

    def test_outside_north(self):
        assert not _in_bbox(62.0, 25.0, 59.0, 61.0, 24.0, 26.0)

    def test_outside_south(self):
        assert not _in_bbox(58.0, 25.0, 59.0, 61.0, 24.0, 26.0)

    def test_outside_east(self):
        assert not _in_bbox(60.0, 27.0, 59.0, 61.0, 24.0, 26.0)

    def test_outside_west(self):
        assert not _in_bbox(60.0, 23.0, 59.0, 61.0, 24.0, 26.0)

    def test_on_boundary(self):
        # On the edge counts as inside
        assert _in_bbox(59.0, 24.0, 59.0, 61.0, 24.0, 26.0)
        assert _in_bbox(61.0, 26.0, 59.0, 61.0, 24.0, 26.0)

    def test_zero_area_box(self):
        # Point == box boundaries
        assert _in_bbox(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def test_negative_coordinates(self):
        assert _in_bbox(-10.0, -20.0, -15.0, -5.0, -25.0, -15.0)
        assert not _in_bbox(-10.0, -30.0, -15.0, -5.0, -25.0, -15.0)


class TestShipTypeLabel:
    def test_tanker(self):
        assert _ship_type_label(80) == "tanker"
        assert _ship_type_label(89) == "tanker"

    def test_cargo(self):
        assert _ship_type_label(70) == "cargo"
        assert _ship_type_label(79) == "cargo"

    def test_passenger(self):
        assert _ship_type_label(60) == "passenger"

    def test_fishing(self):
        assert _ship_type_label(30) == "fishing"
        assert _ship_type_label(39) == "fishing"

    def test_tug(self):
        assert _ship_type_label(50) == "tug"
        assert _ship_type_label(59) == "tug"

    def test_other(self):
        assert _ship_type_label(0) == "other"
        assert _ship_type_label(99) == "other"
        assert _ship_type_label(25) == "other"
        assert _ship_type_label(90) == "other"


class TestShipTypeMatches:
    def test_all_matches_anything(self):
        assert _ship_type_matches(80, "all")
        assert _ship_type_matches(0, "all")

    def test_tanker_range(self):
        assert _ship_type_matches(80, "tanker")
        assert _ship_type_matches(85, "tanker")
        assert _ship_type_matches(89, "tanker")
        assert not _ship_type_matches(79, "tanker")
        assert not _ship_type_matches(90, "tanker")

    def test_cargo_range(self):
        assert _ship_type_matches(70, "cargo")
        assert not _ship_type_matches(69, "cargo")

    def test_unknown_filter(self):
        # Unknown filter type matches everything (graceful)
        assert _ship_type_matches(50, "unknown_type")


# ── Mode routing ───────────────────────────────────────────────────


class TestModeRouting:
    def test_unknown_mode(self):
        tool = AISVesselTool()
        r = tool.execute(mode="invalid_mode")
        assert not r.success
        assert "Unknown mode" in r.output

    def test_empty_mode(self):
        tool = AISVesselTool()
        r = tool.execute(mode="")
        assert not r.success

    def test_none_mode(self):
        tool = AISVesselTool()
        r = tool.execute(mode=None)
        assert not r.success

    def test_valid_modes_dispatch(self):
        """All valid mode names should dispatch without raising."""
        tool = AISVesselTool()
        for mode in ["area", "vessel", "port_calls", "destination_flow"]:
            # These will fail due to missing params or HTTP, but shouldn't raise
            r = tool.execute(mode=mode)
            assert isinstance(r, ToolResult)


# ── Area mode ──────────────────────────────────────────────────────


class TestAreaMode:
    def test_named_area(self):
        tool = AISVesselTool()
        features = [
            _make_feature(111111111, 57.0, 11.0),  # inside danish_straits
            _make_feature(222222222, 65.0, 25.0),  # outside
        ]
        with patch.object(tool, "_fetch_locations", return_value=features):
            r = tool.execute(mode="area", area_name="danish_straits")
        assert r.success
        assert r.data["total_vessels"] == 1
        assert r.data["vessels"][0]["mmsi"] == 111111111

    def test_custom_bbox(self):
        tool = AISVesselTool()
        features = [
            _make_feature(111111111, 60.0, 25.0),
            _make_feature(222222222, 60.0, 30.0),
        ]
        with patch.object(tool, "_fetch_locations", return_value=features):
            r = tool.execute(
                mode="area", lat_min=59.0, lat_max=61.0, lon_min=24.0, lon_max=26.0
            )
        assert r.success
        assert r.data["total_vessels"] == 1

    def test_no_bbox_no_area(self):
        tool = AISVesselTool()
        r = tool.execute(mode="area")
        assert not r.success
        assert "area_name" in r.output

    def test_invalid_bbox_min_gte_max(self):
        tool = AISVesselTool()
        r = tool.execute(
            mode="area", lat_min=61.0, lat_max=59.0, lon_min=24.0, lon_max=26.0
        )
        assert not r.success
        assert "Invalid bounding box" in r.output

    def test_ship_type_filter_tanker(self):
        tool = AISVesselTool()
        features = [
            _make_feature(111111111, 57.0, 11.0),  # in danish_straits
            _make_feature(222222222, 57.5, 12.0),  # in danish_straits
        ]
        meta = {
            111111111: _make_meta(111111111, ship_type=80),  # tanker
            222222222: _make_meta(222222222, ship_type=70),  # cargo
        }
        with patch.object(
            tool, "_fetch_locations", return_value=features
        ), patch.object(tool, "_fetch_metadata", return_value=meta):
            r = tool.execute(
                mode="area", area_name="danish_straits", ship_type="tanker"
            )
        assert r.success
        assert r.data["total_vessels"] == 1
        assert r.data["vessels"][0]["mmsi"] == 111111111

    def test_ship_type_all_no_metadata_fetch(self):
        """ship_type=all should NOT fetch metadata (optimization)."""
        tool = AISVesselTool()
        features = [_make_feature(111111111, 57.0, 11.0)]
        with patch.object(
            tool, "_fetch_locations", return_value=features
        ) as loc_mock, patch.object(tool, "_fetch_metadata") as meta_mock:
            r = tool.execute(mode="area", area_name="danish_straits", ship_type="all")
        assert r.success
        loc_mock.assert_called_once()
        meta_mock.assert_not_called()

    def test_empty_area(self):
        tool = AISVesselTool()
        with patch.object(tool, "_fetch_locations", return_value=[]):
            r = tool.execute(mode="area", area_name="st_petersburg")
        assert r.success
        assert r.data["total_vessels"] == 0

    def test_limit_clamp(self):
        tool = AISVesselTool()
        features = [_make_feature(i, 57.0, 11.0, sog=float(i % 20)) for i in range(100)]
        with patch.object(tool, "_fetch_locations", return_value=features):
            r = tool.execute(mode="area", area_name="danish_straits", limit=5)
        assert r.success
        assert len(r.data["vessels"]) <= 5

    def test_limit_zero_clamped_to_1(self):
        tool = AISVesselTool()
        features = [_make_feature(111, 57.0, 11.0)]
        with patch.object(tool, "_fetch_locations", return_value=features):
            r = tool.execute(mode="area", area_name="danish_straits", limit=0)
        assert r.success

    def test_limit_over_500_clamped(self):
        tool = AISVesselTool()
        features = [_make_feature(111, 57.0, 11.0)]
        with patch.object(tool, "_fetch_locations", return_value=features):
            r = tool.execute(mode="area", area_name="danish_straits", limit=9999)
        assert r.success

    def test_malformed_feature_no_geometry(self):
        tool = AISVesselTool()
        features = [{"mmsi": 111, "type": "Feature"}]  # no geometry
        with patch.object(tool, "_fetch_locations", return_value=features):
            r = tool.execute(mode="area", area_name="full_baltic")
        assert r.success
        assert r.data["total_vessels"] == 0

    def test_malformed_feature_wrong_geom_type(self):
        tool = AISVesselTool()
        features = [
            {
                "mmsi": 111,
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]},
                "properties": {"mmsi": 111},
            }
        ]
        with patch.object(tool, "_fetch_locations", return_value=features):
            r = tool.execute(mode="area", area_name="full_baltic")
        assert r.success
        assert r.data["total_vessels"] == 0

    def test_malformed_feature_short_coordinates(self):
        tool = AISVesselTool()
        features = [
            {
                "mmsi": 111,
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [25.0]},  # only 1 coord
                "properties": {"mmsi": 111},
            }
        ]
        with patch.object(tool, "_fetch_locations", return_value=features):
            r = tool.execute(mode="area", area_name="full_baltic")
        assert r.success
        assert r.data["total_vessels"] == 0

    def test_no_mmsi_in_feature(self):
        tool = AISVesselTool()
        features = [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [25.0, 60.0]},
                "properties": {},
            }
        ]
        with patch.object(tool, "_fetch_locations", return_value=features):
            r = tool.execute(mode="area", area_name="full_baltic")
        assert r.success
        assert r.data["total_vessels"] == 0

    def test_http_error_locations(self):
        tool = AISVesselTool()
        import httpx

        with patch.object(
            tool, "_fetch_locations", side_effect=httpx.HTTPError("timeout")
        ):
            r = tool.execute(mode="area", area_name="danish_straits")
        assert not r.success
        assert "fetch failed" in r.output

    def test_http_error_metadata(self):
        tool = AISVesselTool()
        import httpx

        features = [_make_feature(111, 57.0, 11.0)]
        with patch.object(
            tool, "_fetch_locations", return_value=features
        ), patch.object(
            tool, "_fetch_metadata", side_effect=httpx.HTTPError("timeout")
        ):
            r = tool.execute(
                mode="area", area_name="danish_straits", ship_type="tanker"
            )
        assert not r.success
        assert "metadata fetch failed" in r.output

    def test_nav_status_counting(self):
        tool = AISVesselTool()
        features = [
            _make_feature(1, 57.0, 11.0, nav_stat=0),  # under_way_engine
            _make_feature(2, 57.5, 11.5, nav_stat=0),
            _make_feature(3, 56.5, 12.0, nav_stat=5),  # moored
            _make_feature(4, 57.0, 12.5, nav_stat=1),  # at_anchor
        ]
        with patch.object(tool, "_fetch_locations", return_value=features):
            r = tool.execute(mode="area", area_name="danish_straits")
        assert r.success
        assert r.data["status_counts"]["under_way_engine"] == 2
        assert r.data["status_counts"]["moored"] == 1
        assert r.data["status_counts"]["at_anchor"] == 1

    def test_output_sorted_by_speed(self):
        """Area mode output should list fastest-moving vessels first."""
        tool = AISVesselTool()
        features = [
            _make_feature(1, 57.0, 11.0, sog=2.0),
            _make_feature(2, 57.5, 11.5, sog=15.0),
            _make_feature(3, 56.5, 12.0, sog=8.0),
        ]
        meta = {
            1: _make_meta(1, name="SLOW BOAT"),
            2: _make_meta(2, name="FAST SHIP"),
            3: _make_meta(3, name="MED VESSEL"),
        }
        with patch.object(
            tool, "_fetch_locations", return_value=features
        ), patch.object(tool, "_fetch_metadata", return_value=meta):
            r = tool.execute(mode="area", area_name="danish_straits", ship_type="cargo")
        assert r.success
        # Output should have FAST SHIP before MED VESSEL before SLOW BOAT
        lines = r.output.split("\n")
        vessel_lines = [l for l in lines if "SOG:" in l]
        assert "FAST SHIP" in vessel_lines[0]


# ── Named areas ─────────────────────────────────────────────────────


class TestNamedAreas:
    def test_all_named_areas_exist(self):
        """All documented named areas must be in the dict."""
        expected = {
            "danish_straits",
            "gulf_of_finland",
            "st_petersburg",
            "gotland",
            "skagerrak",
            "kiel",
            "gulf_of_bothnia",
            "riga_gulf",
            "full_baltic",
        }
        assert expected == set(_NAMED_AREAS.keys())

    def test_all_named_areas_valid_bbox(self):
        """Every named area must have lat_min < lat_max, lon_min < lon_max."""
        for name, (lat_min, lat_max, lon_min, lon_max) in _NAMED_AREAS.items():
            assert lat_min < lat_max, f"{name}: lat_min >= lat_max"
            assert lon_min < lon_max, f"{name}: lon_min >= lon_max"

    def test_named_area_case_insensitive(self):
        tool = AISVesselTool()
        features = [_make_feature(111, 57.0, 11.0)]
        with patch.object(tool, "_fetch_locations", return_value=features):
            r = tool.execute(mode="area", area_name="DANISH_STRAITS")
        assert r.success

    def test_unknown_named_area_without_bbox(self):
        tool = AISVesselTool()
        r = tool.execute(mode="area", area_name="bermuda_triangle")
        assert not r.success


# ── Vessel mode ────────────────────────────────────────────────────


class TestVesselMode:
    def test_vessel_found(self):
        tool = AISVesselTool()
        meta = _make_meta(230935000, name="EIRA", ship_type=70, destination="FIRAA RR6")
        features = [_make_feature(230935000, 64.657, 24.412, sog=0.0, nav_stat=5)]

        with patch.object(
            tool, "_fetch_vessel_metadata_single", return_value=meta
        ), patch.object(tool, "_fetch_locations", return_value=features):
            r = tool.execute(mode="vessel", mmsi=230935000)
        assert r.success
        assert r.data["name"] == "EIRA"
        assert r.data["lat"] == pytest.approx(64.657)
        assert "EIRA" in r.output

    def test_vessel_no_mmsi(self):
        tool = AISVesselTool()
        r = tool.execute(mode="vessel")
        assert not r.success
        assert "mmsi" in r.output.lower()

    def test_vessel_not_found(self):
        tool = AISVesselTool()
        with patch.object(
            tool, "_fetch_vessel_metadata_single", return_value=None
        ), patch.object(tool, "_fetch_locations", return_value=[]):
            r = tool.execute(mode="vessel", mmsi=999999999)
        assert not r.success
        assert "not found" in r.output.lower()

    def test_vessel_meta_only_no_position(self):
        """Vessel has metadata but isn't in location feed (out of range)."""
        tool = AISVesselTool()
        meta = _make_meta(123456789, name="FAR AWAY SHIP", destination="SG SIN")
        with patch.object(
            tool, "_fetch_vessel_metadata_single", return_value=meta
        ), patch.object(tool, "_fetch_locations", return_value=[]):
            r = tool.execute(mode="vessel", mmsi=123456789)
        assert r.success
        assert "FAR AWAY SHIP" in r.output
        assert "not in current AIS coverage" in r.output

    def test_vessel_position_only_no_meta(self):
        """Vessel in location feed but no metadata returned."""
        tool = AISVesselTool()
        features = [_make_feature(123456789, 60.0, 25.0, sog=10.0)]
        with patch.object(
            tool, "_fetch_vessel_metadata_single", return_value=None
        ), patch.object(tool, "_fetch_locations", return_value=features):
            r = tool.execute(mode="vessel", mmsi=123456789)
        assert r.success
        assert r.data["lat"] == pytest.approx(60.0)
        assert "MMSI:123456789" in r.output  # fallback name

    def test_vessel_draught_formatting(self):
        tool = AISVesselTool()
        meta = _make_meta(111)
        meta["draught"] = 125  # 12.5m in AIS encoding (tenths of meters)
        features = [_make_feature(111, 60.0, 25.0)]
        with patch.object(
            tool, "_fetch_vessel_metadata_single", return_value=meta
        ), patch.object(tool, "_fetch_locations", return_value=features):
            r = tool.execute(mode="vessel", mmsi=111)
        assert r.success
        assert "12.5m" in r.output

    def test_vessel_http_error_meta(self):
        tool = AISVesselTool()
        import httpx

        with patch.object(
            tool, "_fetch_vessel_metadata_single", side_effect=httpx.HTTPError("err")
        ):
            r = tool.execute(mode="vessel", mmsi=111)
        assert not r.success

    def test_vessel_http_error_locations_still_works(self):
        """If location fetch fails, vessel mode still returns metadata."""
        tool = AISVesselTool()
        import httpx

        meta = _make_meta(111, name="STILL WORKS")
        with patch.object(
            tool, "_fetch_vessel_metadata_single", return_value=meta
        ), patch.object(tool, "_fetch_locations", side_effect=httpx.HTTPError("err")):
            r = tool.execute(mode="vessel", mmsi=111)
        assert r.success
        assert "STILL WORKS" in r.output

    def test_vessel_mmsi_as_string(self):
        """MMSI passed as string should still work (coerced to int)."""
        tool = AISVesselTool()
        meta = _make_meta(230935000, name="EIRA")
        with patch.object(
            tool, "_fetch_vessel_metadata_single", return_value=meta
        ), patch.object(tool, "_fetch_locations", return_value=[]):
            r = tool.execute(mode="vessel", mmsi="230935000")
        assert r.success


# ── Port calls mode ────────────────────────────────────────────────


class TestPortCallsMode:
    def test_port_calls_basic(self):
        tool = AISVesselTool()
        calls = [
            _make_port_call("SHIP A", "FIHEL", "SEGOT", "FIRAA"),
            _make_port_call("SHIP B", "FIHEL", "NLRTM", "FIKOK"),
            _make_port_call("SHIP C", "FIRAA", "EETLL", "FIHEL"),
        ]
        with patch.object(tool, "_fetch_port_calls", return_value=calls):
            r = tool.execute(mode="port_calls", from_date="2026-03-26")
        assert r.success
        assert r.data["total_calls"] == 3
        assert r.data["port_counts"]["FIHEL"] == 2
        assert r.data["port_counts"]["FIRAA"] == 1

    def test_port_calls_default_date(self):
        tool = AISVesselTool()
        with patch.object(tool, "_fetch_port_calls", return_value=[]) as mock:
            r = tool.execute(mode="port_calls")
        assert r.success
        # Should have called with yesterday's date
        called_date = mock.call_args[0][0]
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime(
            "%Y-%m-%d"
        )
        assert called_date == yesterday

    def test_port_calls_empty(self):
        tool = AISVesselTool()
        with patch.object(tool, "_fetch_port_calls", return_value=[]):
            r = tool.execute(mode="port_calls", from_date="2026-03-26")
        assert r.success
        assert r.data["total_calls"] == 0

    def test_port_calls_http_error(self):
        tool = AISVesselTool()
        import httpx

        with patch.object(
            tool, "_fetch_port_calls", side_effect=httpx.HTTPError("err")
        ):
            r = tool.execute(mode="port_calls", from_date="2026-03-26")
        assert not r.success

    def test_port_calls_limit(self):
        tool = AISVesselTool()
        calls = [_make_port_call(f"SHIP {i}", "FIHEL") for i in range(100)]
        with patch.object(tool, "_fetch_port_calls", return_value=calls):
            r = tool.execute(mode="port_calls", from_date="2026-03-26", limit=10)
        assert r.success
        assert len(r.data["calls"]) == 10

    def test_port_calls_cargo_vs_ballast(self):
        tool = AISVesselTool()
        calls = [
            _make_port_call("LADEN", "FIHEL", cargo=True),
            _make_port_call("EMPTY", "FIHEL", cargo=False),
        ]
        with patch.object(tool, "_fetch_port_calls", return_value=calls):
            r = tool.execute(mode="port_calls", from_date="2026-03-26")
        assert r.success
        assert "cargo" in r.output
        assert "ballast" in r.output


# ── Destination flow mode ──────────────────────────────────────────


class TestDestinationFlowMode:
    def test_basic_flow(self):
        tool = AISVesselTool()
        meta = {
            1: _make_meta(1, ship_type=80, destination="PORT SAID"),
            2: _make_meta(2, ship_type=80, destination="PORT SAID"),
            3: _make_meta(3, ship_type=80, destination="NLRTM"),
            4: _make_meta(4, ship_type=70, destination="RU LED"),
        }
        with patch.object(tool, "_fetch_metadata", return_value=meta):
            r = tool.execute(mode="destination_flow")
        assert r.success
        assert r.data["destinations"]["PORT SAID"] == 2
        assert r.data["destinations"]["NLRTM"] == 1
        assert r.data["destinations"]["RU LED"] == 1

    def test_flow_type_filter(self):
        tool = AISVesselTool()
        meta = {
            1: _make_meta(1, ship_type=80, destination="PORT SAID"),  # tanker
            2: _make_meta(2, ship_type=70, destination="PORT SAID"),  # cargo
            3: _make_meta(3, ship_type=80, destination="NLRTM"),  # tanker
        }
        with patch.object(tool, "_fetch_metadata", return_value=meta):
            r = tool.execute(mode="destination_flow", ship_type="tanker")
        assert r.success
        assert r.data["total_with_destination"] == 2
        assert r.data["destinations"]["PORT SAID"] == 1
        assert r.data["destinations"]["NLRTM"] == 1

    def test_flow_strategic_groups(self):
        tool = AISVesselTool()
        meta = {
            1: _make_meta(1, destination="PORT SAID"),
            2: _make_meta(2, destination="SUEZ"),
            3: _make_meta(3, destination="EGPSD"),
            4: _make_meta(4, destination="RU LED"),
            5: _make_meta(5, destination="RULED"),
            6: _make_meta(6, destination="ROTTERDAM"),
            7: _make_meta(7, destination="NLRTM"),
        }
        with patch.object(tool, "_fetch_metadata", return_value=meta):
            r = tool.execute(mode="destination_flow")
        assert r.success
        assert r.data["strategic"]["suez"] == 3  # PORT SAID + SUEZ + EGPSD
        assert r.data["strategic"]["russia"] == 2  # RU LED + RULED
        assert r.data["strategic"]["rotterdam"] == 2  # ROTTERDAM + NLRTM

    def test_flow_empty_destinations(self):
        """Vessels with no destination should be excluded."""
        tool = AISVesselTool()
        meta = {
            1: _make_meta(1, destination="PORT SAID"),
            2: _make_meta(2, destination=""),
            3: _make_meta(3, destination="   "),
        }
        with patch.object(tool, "_fetch_metadata", return_value=meta):
            r = tool.execute(mode="destination_flow")
        assert r.success
        assert r.data["total_with_destination"] == 1

    def test_flow_http_error(self):
        tool = AISVesselTool()
        import httpx

        with patch.object(tool, "_fetch_metadata", side_effect=httpx.HTTPError("err")):
            r = tool.execute(mode="destination_flow")
        assert not r.success

    def test_flow_limit(self):
        tool = AISVesselTool()
        meta = {i: _make_meta(i, destination=f"PORT_{i}") for i in range(100)}
        with patch.object(tool, "_fetch_metadata", return_value=meta):
            r = tool.execute(mode="destination_flow", limit=10)
        assert r.success
        assert len(r.data["destinations"]) == 10

    def test_flow_destination_case_normalization(self):
        """Destinations are uppercased for consistent grouping."""
        tool = AISVesselTool()
        meta = {
            1: _make_meta(1, destination="port said"),
            2: _make_meta(2, destination="Port Said"),
        }
        with patch.object(tool, "_fetch_metadata", return_value=meta):
            r = tool.execute(mode="destination_flow")
        assert r.success
        assert r.data["destinations"]["PORT SAID"] == 2


# ── Cache integration ──────────────────────────────────────────────


class TestCacheIntegration:
    def test_locations_cache_hit(self):
        tool, cache = _tool_with_mock_cache()
        cached_data = [_make_feature(111, 57.0, 11.0)]
        cache.get.return_value = cached_data

        result = tool._fetch_locations()
        assert result == cached_data
        cache.get.assert_called_once()

    def test_locations_cache_miss_then_put(self):
        tool, cache = _tool_with_mock_cache()
        features = [_make_feature(111, 57.0, 11.0)]
        resp_mock = MagicMock()
        resp_mock.json.return_value = {"features": features}
        resp_mock.raise_for_status = MagicMock()

        with patch.object(tool, "_get", return_value=resp_mock):
            result = tool._fetch_locations()

        assert result == features
        cache.put.assert_called_once()

    def test_metadata_cache_hit(self):
        tool, cache = _tool_with_mock_cache()
        cached_meta = {111: _make_meta(111)}
        cache.get.return_value = cached_meta

        result = tool._fetch_metadata()
        assert result == cached_meta

    def test_metadata_cache_miss_indexes_by_mmsi(self):
        tool, cache = _tool_with_mock_cache()
        vessels = [_make_meta(111), _make_meta(222)]
        resp_mock = MagicMock()
        resp_mock.json.return_value = vessels
        resp_mock.raise_for_status = MagicMock()

        with patch.object(tool, "_get", return_value=resp_mock):
            result = tool._fetch_metadata()

        assert 111 in result
        assert 222 in result
        cache.put.assert_called_once()

    def test_port_calls_cache_miss(self):
        tool, cache = _tool_with_mock_cache()
        calls = [_make_port_call("SHIP A")]
        resp_mock = MagicMock()
        resp_mock.json.return_value = {"portCalls": calls}
        resp_mock.raise_for_status = MagicMock()

        with patch.object(tool, "_get", return_value=resp_mock):
            result = tool._fetch_port_calls("2026-03-26")

        assert result == calls
        cache.put.assert_called_once()

    def test_no_cache_still_works(self):
        """Tool with cache=None should still function."""
        tool = AISVesselTool(cache=None)
        features = [_make_feature(111, 57.0, 11.0)]
        resp_mock = MagicMock()
        resp_mock.json.return_value = {"features": features}
        resp_mock.raise_for_status = MagicMock()

        with patch.object(tool, "_get", return_value=resp_mock):
            result = tool._fetch_locations()
        assert result == features

    def test_empty_locations_not_cached(self):
        """Empty responses should not be cached."""
        tool, cache = _tool_with_mock_cache()
        resp_mock = MagicMock()
        resp_mock.json.return_value = {"features": []}
        resp_mock.raise_for_status = MagicMock()

        with patch.object(tool, "_get", return_value=resp_mock):
            result = tool._fetch_locations()
        assert result == []
        cache.put.assert_not_called()

    def test_single_vessel_meta_cache(self):
        tool, cache = _tool_with_mock_cache()
        meta = _make_meta(111)
        resp_mock = MagicMock()
        resp_mock.json.return_value = meta
        resp_mock.raise_for_status = MagicMock()
        resp_mock.status_code = 200

        with patch.object(tool, "_get", return_value=resp_mock):
            result = tool._fetch_vessel_metadata_single(111)
        assert result == meta
        cache.put.assert_called_once()

    def test_single_vessel_meta_404(self):
        tool, cache = _tool_with_mock_cache()
        resp_mock = MagicMock()
        resp_mock.status_code = 404

        with patch.object(tool, "_get", return_value=resp_mock):
            result = tool._fetch_vessel_metadata_single(999999)
        assert result is None
        cache.put.assert_not_called()


# ── Schema validation ──────────────────────────────────────────────


class TestToolSchema:
    def test_has_required_attrs(self):
        tool = AISVesselTool()
        assert tool.name == "ais_vessel_tracking"
        assert isinstance(tool.description, str)
        assert len(tool.description) > 20
        assert isinstance(tool.parameters, dict)

    def test_parameters_has_mode(self):
        tool = AISVesselTool()
        props = tool.parameters["properties"]
        assert "mode" in props
        assert set(props["mode"]["enum"]) == {
            "area",
            "vessel",
            "port_calls",
            "destination_flow",
        }

    def test_all_ship_types_in_enum(self):
        tool = AISVesselTool()
        enum = set(tool.parameters["properties"]["ship_type"]["enum"])
        expected = {"tanker", "cargo", "passenger", "fishing", "tug", "all"}
        assert enum == expected

    def test_named_areas_match_enum(self):
        tool = AISVesselTool()
        enum = set(tool.parameters["properties"]["area_name"]["enum"])
        assert enum == set(_NAMED_AREAS.keys())

    def test_openai_tool_format(self):
        tool = AISVesselTool()
        schema = tool.to_openai_tool()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "ais_vessel_tracking"
        assert "parameters" in schema["function"]


# ── Nav status codes ───────────────────────────────────────────────


class TestNavStatus:
    def test_known_statuses(self):
        assert _NAV_STATUS[0] == "under_way_engine"
        assert _NAV_STATUS[1] == "at_anchor"
        assert _NAV_STATUS[5] == "moored"
        assert _NAV_STATUS[7] == "fishing"

    def test_all_common_codes_mapped(self):
        # AIS spec: 0-12, 14, 15
        for code in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15]:
            assert code in _NAV_STATUS


# ── Registry integration ───────────────────────────────────────────


class TestRegistryIntegration:
    def test_tool_in_registry(self):
        try:
            from agent.cli import build_tool_registry
            from agent.config.settings import AgentConfig
        except (ImportError, ModuleNotFoundError):
            pytest.skip("apscheduler or other dep not installed")

        config = AgentConfig()
        registry = build_tool_registry(config)
        names = registry.list_names()

        assert "ais_vessel_tracking" in names
        assert (
            len(names) == 47
        )  # was 27, +3 for defi/gov_contracts/academic_preprints, +1 sanctions_monitor, +1 cert_transparency, +1 sovereign_debt, +1 central_bank_balance, +1 foia_requests

    def test_tool_executes_from_registry(self):
        try:
            from agent.cli import build_tool_registry
            from agent.config.settings import AgentConfig
        except (ImportError, ModuleNotFoundError):
            pytest.skip("apscheduler or other dep not installed")

        config = AgentConfig()
        registry = build_tool_registry(config)
        tool = registry.get("ais_vessel_tracking")
        assert tool is not None

        # Should return error for missing params, not crash
        r = tool.execute(mode="vessel")
        assert not r.success


# ── Exception safety ───────────────────────────────────────────────


class TestExceptionSafety:
    def test_execute_catches_unexpected_exception(self):
        tool = AISVesselTool()
        with patch.object(tool, "_mode_area", side_effect=RuntimeError("boom")):
            r = tool.execute(mode="area", area_name="danish_straits")
        assert not r.success
        assert "AIS tool error" in r.output

    def test_execute_catches_key_error(self):
        tool = AISVesselTool()
        with patch.object(tool, "_mode_vessel", side_effect=KeyError("bad_key")):
            r = tool.execute(mode="vessel", mmsi=111)
        assert not r.success

    def test_execute_catches_type_error(self):
        tool = AISVesselTool()
        with patch.object(tool, "_mode_port_calls", side_effect=TypeError("bad")):
            r = tool.execute(mode="port_calls")
        assert not r.success


# ── Edge cases in data shapes ──────────────────────────────────────


class TestDataShapeEdgeCases:
    def test_metadata_non_list_response(self):
        """If /vessels returns a dict instead of list, handle gracefully."""
        tool, cache = _tool_with_mock_cache()
        resp_mock = MagicMock()
        resp_mock.json.return_value = {"error": "something"}  # not a list
        resp_mock.raise_for_status = MagicMock()

        with patch.object(tool, "_get", return_value=resp_mock):
            result = tool._fetch_metadata()
        assert result == {}

    def test_port_calls_list_response(self):
        """Port calls might come as plain list instead of {portCalls: [...]}."""
        tool, cache = _tool_with_mock_cache()
        calls = [_make_port_call("SHIP A")]
        resp_mock = MagicMock()
        resp_mock.json.return_value = calls  # plain list
        resp_mock.raise_for_status = MagicMock()

        with patch.object(tool, "_get", return_value=resp_mock):
            result = tool._fetch_port_calls("2026-03-26")
        assert result == calls

    def test_feature_mmsi_in_properties_only(self):
        """Some features have MMSI only in properties, not top-level."""
        tool = AISVesselTool()
        features = [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [11.0, 57.0]},
                "properties": {
                    "mmsi": 111,
                    "sog": 5.0,
                    "cog": 180.0,
                    "navStat": 0,
                    "heading": 180,
                },
            }
        ]
        with patch.object(tool, "_fetch_locations", return_value=features):
            r = tool.execute(mode="area", area_name="danish_straits")
        assert r.success
        assert r.data["total_vessels"] == 1

    def test_metadata_with_none_mmsi_skipped(self):
        """Metadata entries with null MMSI should be skipped."""
        tool, cache = _tool_with_mock_cache()
        vessels = [
            _make_meta(111),
            {"name": "GHOST SHIP", "shipType": 70},  # no mmsi key
        ]
        resp_mock = MagicMock()
        resp_mock.json.return_value = vessels
        resp_mock.raise_for_status = MagicMock()

        with patch.object(tool, "_get", return_value=resp_mock):
            result = tool._fetch_metadata()
        assert 111 in result
        assert len(result) == 1
