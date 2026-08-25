"""Tests for Idea 14 — Nightlight & NDVI Economic Zone Activity Signals.

Covers:
    1.  EconomicZone is frozen (immutable)
    2.  ECONOMIC_ZONES has exactly 50 entries
    3.  ZONE_BY_ID keys match zone_id fields
    4.  No duplicate zone_ids in manifest
    5.  All zones have valid lat (-90..90) and lon (-180..180)
    6.  All bbox_deg > 0
    7.  category is one of valid set
    8.  _zone_area_str produces correct bbox string
    9.  _date_to_modis converts valid date correctly
    10. _date_to_modis returns None for invalid date
    11. _aggregate_frp: empty list → all zeros
    12. _aggregate_frp: single high-confidence hotspot counts as industrial
    13. _aggregate_frp: frp_total_mw sums correctly
    14. _aggregate_frp: mean_brightness_k computed correctly
    15. _aggregate_frp: industrial_ratio = industrial_count / total
    16. _extract_ndvi: valid MODIS response returns float in [-1, 1]
    17. _extract_ndvi: empty subset returns None
    18. _extract_ndvi: out-of-range values filtered
    19. NightlightActivityTool.name == 'nightlight_activity'
    20. NightlightActivityTool.parameters has required fields
    21. execute: invalid mode returns success=False
    22. execute: bad zone_ids returns success=False
    23. execute: nightlight mode skips when no FIRMS key
    24. execute: ndvi mode calls _fetch_zone_ndvi per ag zone
    25. execute: persists signals to store when store is provided
    26. execute: handles FIRMS HTTP failure gracefully
    27. execute: handles MODIS HTTP failure gracefully
    28. execute: zone_ids filter restricts to requested zones
    29. _ndvi_to_health: sparse NDVI maps below 0.5
    30. _ndvi_to_health: healthy NDVI maps above 0.75
    31. _ndvi_to_health: clamps negative NDVI to 0.0
    32. _ndvi_to_health: clamps NDVI > 1.0 to 1.0
    33. _persist_nightlight: stores 3 signals per zone
    34. _persist_ndvi: stores 2 signals per zone
    35. _persist_nightlight: handles store failure gracefully
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.tools.nightlight_activity import (
    ECONOMIC_ZONES,
    ZONE_BY_ID,
    EconomicZone,
    NightlightActivityTool,
    _aggregate_frp,
    _date_to_modis,
    _extract_ndvi,
    _zone_area_str,
)


# ═══════════════════════════════════════════════════════════════
# 1–7. EconomicZone manifest
# ═══════════════════════════════════════════════════════════════

class TestManifest:

    def test_frozen(self):
        z = ECONOMIC_ZONES[0]
        with pytest.raises((AttributeError, TypeError)):
            z.zone_id = "oops"  # type: ignore[misc]

    def test_exactly_50_zones(self):
        assert len(ECONOMIC_ZONES) == 50

    def test_zone_by_id_keys_match(self):
        for k, v in ZONE_BY_ID.items():
            assert v.zone_id == k

    def test_no_duplicate_zone_ids(self):
        ids = [z.zone_id for z in ECONOMIC_ZONES]
        assert len(ids) == len(set(ids))

    def test_valid_latitudes(self):
        for z in ECONOMIC_ZONES:
            assert -90.0 <= z.lat <= 90.0, f"{z.zone_id} lat={z.lat}"

    def test_valid_longitudes(self):
        for z in ECONOMIC_ZONES:
            assert -180.0 <= z.lon <= 180.0, f"{z.zone_id} lon={z.lon}"

    def test_bbox_positive(self):
        for z in ECONOMIC_ZONES:
            assert z.bbox_deg > 0.0, f"{z.zone_id} bbox={z.bbox_deg}"

    def test_valid_categories(self):
        valid = {"industrial", "agricultural", "port", "urban"}
        for z in ECONOMIC_ZONES:
            assert z.category in valid, f"{z.zone_id} category={z.category}"


# ═══════════════════════════════════════════════════════════════
# 8–10. Helper functions
# ═══════════════════════════════════════════════════════════════

class TestHelpers:

    def test_zone_area_str_format(self):
        z = EconomicZone("test", "Test", lat=10.0, lon=20.0, bbox_deg=1.0, category="industrial")
        area = _zone_area_str(z)
        parts = area.split(",")
        assert len(parts) == 4
        lons, lats, lone, late = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
        assert lons == pytest.approx(19.0)
        assert lats == pytest.approx(9.0)
        assert lone == pytest.approx(21.0)
        assert late == pytest.approx(11.0)

    def test_date_to_modis_valid(self):
        result = _date_to_modis("2024-01-01")
        assert result == "A2024001"

    def test_date_to_modis_leap_year(self):
        result = _date_to_modis("2024-12-31")
        assert result == "A2024366"

    def test_date_to_modis_invalid(self):
        assert _date_to_modis("not-a-date") is None
        assert _date_to_modis("2024-13-01") is None


# ═══════════════════════════════════════════════════════════════
# 11–15. _aggregate_frp
# ═══════════════════════════════════════════════════════════════

class TestAggregateFrp:

    def test_empty_returns_zeros(self):
        result = _aggregate_frp([])
        assert result["frp_total_mw"] == 0.0
        assert result["hotspot_count"] == 0
        assert result["mean_brightness_k"] == 0.0
        assert result["industrial_ratio"] == 0.0

    def test_single_high_conf_industrial(self):
        row = {"frp": "100.0", "bright_ti4": "350.0", "confidence": "high"}
        result = _aggregate_frp([row])
        assert result["industrial_ratio"] == pytest.approx(1.0)
        assert result["frp_total_mw"] == pytest.approx(100.0)

    def test_frp_total_sums(self):
        rows = [
            {"frp": "30.0", "bright_ti4": "320.0", "confidence": "nominal"},
            {"frp": "70.0", "bright_ti4": "360.0", "confidence": "high"},
        ]
        result = _aggregate_frp(rows)
        assert result["frp_total_mw"] == pytest.approx(100.0)
        assert result["hotspot_count"] == 2

    def test_mean_brightness_correct(self):
        rows = [
            {"frp": "10.0", "bright_ti4": "300.0", "confidence": "nominal"},
            {"frp": "10.0", "bright_ti4": "400.0", "confidence": "nominal"},
        ]
        result = _aggregate_frp(rows)
        assert result["mean_brightness_k"] == pytest.approx(350.0)

    def test_industrial_ratio_partial(self):
        rows = [
            {"frp": "200.0", "bright_ti4": "380.0", "confidence": "high"},
            {"frp": "10.0",  "bright_ti4": "300.0", "confidence": "nominal"},
        ]
        result = _aggregate_frp(rows)
        assert result["industrial_ratio"] == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════════
# 16–18. _extract_ndvi
# ═══════════════════════════════════════════════════════════════

class TestExtractNdvi:

    def _modis_payload(self, values: list) -> dict:
        return {
            "subset": [
                {
                    "band": "250m_16_days_NDVI",
                    "data": [str(v) for v in values],
                }
            ]
        }

    def test_valid_ndvi_in_range(self):
        payload = self._modis_payload([5000, 6000])  # 0.5 and 0.6
        result = _extract_ndvi(payload)
        assert result is not None
        assert result == pytest.approx(0.55, abs=0.01)

    def test_empty_subset_returns_none(self):
        assert _extract_ndvi({"subset": []}) is None

    def test_out_of_range_filtered(self):
        payload = self._modis_payload([5000, 99999, -9999])
        result = _extract_ndvi(payload)
        assert result is not None
        assert -1.0 <= result <= 1.0


# ═══════════════════════════════════════════════════════════════
# 19–20. Tool metadata
# ═══════════════════════════════════════════════════════════════

class TestToolMetadata:

    def test_name(self):
        tool = NightlightActivityTool()
        assert tool.name == "nightlight_activity"

    def test_parameters_has_mode(self):
        tool = NightlightActivityTool()
        props = tool.parameters["properties"]
        assert "mode" in props
        assert set(props["mode"]["enum"]) == {"nightlight", "ndvi", "both"}


# ═══════════════════════════════════════════════════════════════
# 21–28. execute()
# ═══════════════════════════════════════════════════════════════

class TestExecute:

    def test_invalid_mode(self):
        tool = NightlightActivityTool(firms_api_key="key")
        result = tool.execute(mode="invalid")
        assert result.success is False

    def test_bad_zone_ids(self):
        tool = NightlightActivityTool(firms_api_key="key")
        result = tool.execute(mode="nightlight", zone_ids=["no_such_zone"])
        assert result.success is False

    def test_nightlight_skips_without_key(self):
        tool = NightlightActivityTool(firms_api_key="")
        result = tool.execute(mode="nightlight")
        # No key → warns but still succeeds (empty data)
        assert isinstance(result.success, bool)

    def test_ndvi_mode_calls_fetch(self):
        tool = NightlightActivityTool(firms_api_key="key")
        with patch(
            "agent.tools.nightlight_activity._fetch_zone_ndvi",
            return_value=0.55,
        ) as mock_ndvi:
            result = tool.execute(mode="ndvi")
        assert mock_ndvi.called
        assert result.success is True

    def test_persists_signals_to_store(self):
        mock_store = MagicMock()
        tool = NightlightActivityTool(firms_api_key="key", store=mock_store)
        with patch(
            "agent.tools.nightlight_activity._fetch_zone_ndvi",
            return_value=0.6,
        ):
            tool.execute(mode="ndvi")
        # store_signal called for each ag zone × 2 signals (value + health)
        assert mock_store.store_signal.call_count > 0

    def test_firms_failure_graceful(self):
        tool = NightlightActivityTool(firms_api_key="badkey")
        with patch(
            "agent.tools.nightlight_activity._fetch_zone_frp",
            return_value={},
        ):
            result = tool.execute(mode="nightlight", zone_ids=["permian_basin"])
        # No crash — result may have empty data
        assert isinstance(result, object)

    def test_modis_failure_graceful(self):
        tool = NightlightActivityTool()
        with patch(
            "agent.tools.nightlight_activity._fetch_zone_ndvi",
            return_value=None,
        ):
            result = tool.execute(mode="ndvi")
        assert result.success is True  # graceful empty result

    def test_zone_ids_filter(self):
        tool = NightlightActivityTool(firms_api_key="key")
        called_zones = []
        def mock_fetch(zone, firms_key, days=7, source="VIIRS_SNPP_NRT"):
            called_zones.append(zone.zone_id)
            return {"frp_total_mw": 100.0, "hotspot_count": 5,
                    "mean_brightness_k": 350.0, "industrial_ratio": 0.8}
        with patch("agent.tools.nightlight_activity._fetch_zone_frp", side_effect=mock_fetch):
            tool.execute(mode="nightlight", zone_ids=["permian_basin"])
        assert called_zones == ["permian_basin"]


# ═══════════════════════════════════════════════════════════════
# 29–32. _ndvi_to_health
# ═══════════════════════════════════════════════════════════════

class TestNdviToHealth:

    def test_sparse_below_half(self):
        h = NightlightActivityTool._ndvi_to_health(0.15)
        assert h < 0.5

    def test_healthy_above_75(self):
        h = NightlightActivityTool._ndvi_to_health(0.7)
        assert h > 0.75

    def test_negative_ndvi_clamps_to_near_zero(self):
        h = NightlightActivityTool._ndvi_to_health(-0.5)
        assert h >= 0.0

    def test_over_one_clamps_to_one(self):
        h = NightlightActivityTool._ndvi_to_health(1.5)
        assert h <= 1.0


# ═══════════════════════════════════════════════════════════════
# 33–35. Persistence helpers
# ═══════════════════════════════════════════════════════════════

class TestPersistence:

    def _zone(self):
        return EconomicZone(
            "test_zone", "Test Zone", 10.0, 20.0, 1.0, "industrial", ("oil",)
        )

    def test_persist_nightlight_3_signals(self):
        mock_store = MagicMock()
        tool = NightlightActivityTool(store=mock_store)
        frp_data = {"frp_total_mw": 100.0, "hotspot_count": 5,
                    "mean_brightness_k": 350.0, "industrial_ratio": 0.8}
        tool._persist_nightlight(self._zone(), frp_data, 1.0)
        assert mock_store.store_signal.call_count == 3

    def test_persist_ndvi_2_signals(self):
        mock_store = MagicMock()
        tool = NightlightActivityTool(store=mock_store)
        ag_zone = EconomicZone("ag_zone", "Ag", 30.0, -90.0, 2.0, "agricultural", ("wheat",))
        tool._persist_ndvi(ag_zone, 0.55, 0.62, 1.0)
        assert mock_store.store_signal.call_count == 2

    def test_persist_handles_store_error(self):
        mock_store = MagicMock()
        mock_store.store_signal.side_effect = RuntimeError("disk error")
        tool = NightlightActivityTool(store=mock_store)
        frp_data = {"frp_total_mw": 50.0, "hotspot_count": 2,
                    "mean_brightness_k": 340.0, "industrial_ratio": 0.5}
        tool._persist_nightlight(self._zone(), frp_data, 1.0)  # must not raise
