"""Tests for ais_vessel L2 upgrade — vessel entity persistence,
entity_ids in output dicts, IMO-first / MMSI-fallback identity,
port call observations, and MI integration.

Mirrors the test pattern from test_whale_alert_l2.py.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent.pipeline.entity import entity_id_from_key
from agent.tools import ais_vessel as ais_vessel_mod
from agent.tools.ais_vessel import _MAX_FEED_AGE_SECONDS, _MAX_VESSEL_OBS_PER_RUN, AISVesselTool

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_vessel(
    *,
    mmsi: int = 230000001,
    imo: int | None = 9000001,
    name: str = "FINNLADY",
    lat: float = 60.1,
    lon: float = 25.0,
    sog: float = 12.0,
    cog: float = 180.0,
    heading: int = 180,
    nav_status: str = "under way using engine",
    ship_type: str = "passenger",
    ship_type_code: int | None = 60,
    destination: str | None = "TALLINN",
    draught: float | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build a parsed vessel dict (area/vessel mode output)."""
    v: dict[str, Any] = {
        "mmsi": mmsi,
        "lat": lat,
        "lon": lon,
        "sog": sog,
        "cog": cog,
        "heading": heading,
        "nav_status": nav_status,
    }
    if name:
        v["name"] = name
    if imo is not None:
        v["imo"] = imo
    if ship_type:
        v["ship_type"] = ship_type
    if ship_type_code is not None:
        v["ship_type_code"] = ship_type_code
    if destination:
        v["destination"] = destination
    if draught is not None:
        v["draught"] = draught
    if timestamp:
        v["timestamp"] = timestamp
    return v


def _make_port_call(
    *,
    mmsi: int | None = 230000001,
    imo: int | None = 9000001,
    vessel_name: str = "FINNLADY",
    port: str = "VUOSAARI",
    prev_port: str = "TALLINN",
    next_port: str = "TALLINN",
    cargo: bool = True,
    eta: str | None = None,
    vessel_type_code: int | None = 60,
    nationality: str | None = "FI",
) -> dict[str, Any]:
    """Build a port call dict from Digitraffic format."""
    c: dict[str, Any] = {
        "vesselName": vessel_name,
        "portToVisit": port,
        "prevPort": prev_port,
        "nextPort": next_port,
        "arrivalWithCargo": cargo,
    }
    if mmsi is not None:
        c["mmsi"] = mmsi
    if imo is not None:
        c["imoLloyds"] = imo
    if eta:
        c["eta"] = eta
    if vessel_type_code is not None:
        c["vesselTypeCode"] = vessel_type_code
    if nationality:
        c["nationality"] = nationality
    return c


def _make_store() -> MagicMock:
    """Create a mock PipelineStore with the entity API surface."""
    store = MagicMock()
    store.register_entity = MagicMock(return_value="eid")
    store.add_entity_alias = MagicMock()
    store.store_entity_observation = MagicMock(return_value=1)
    store.resolve_entity = MagicMock(return_value=None)
    return store


def _vessel_register_calls(store: MagicMock) -> list:
    """Filter register_entity calls to only vessel-type registrations."""
    return [c for c in store.register_entity.call_args_list if c.kwargs.get("entity_type") == "vessel"]


# ===========================================================================
# Class: TestConstructor
# ===========================================================================


class TestConstructor:
    """Step 10b.4.2: PipelineStore kwarg in constructor."""

    def test_default_no_store(self) -> None:
        tool = AISVesselTool()
        assert tool._store is None
        assert tool._cache is None

    def test_with_store(self) -> None:
        store = _make_store()
        tool = AISVesselTool(pipeline_store=store)
        assert tool._store is store

    def test_with_cache_and_store(self) -> None:
        cache = MagicMock()
        store = _make_store()
        tool = AISVesselTool(cache, pipeline_store=store)
        assert tool._cache is cache
        assert tool._store is store

    def test_store_keyword_only(self) -> None:
        """pipeline_store must be keyword-only — positional should fail."""
        store = _make_store()
        with pytest.raises(TypeError):
            AISVesselTool(None, store)  # type: ignore[misc]


# ===========================================================================
# Class: TestVesselEntityId
# ===========================================================================


class TestVesselEntityId:
    """Step 10b.4.3: _vessel_entity_id helper — IMO-first, MMSI-fallback."""

    def test_imo_preferred(self) -> None:
        eid = AISVesselTool._vessel_entity_id(230000001, 9000001)
        expected = entity_id_from_key("vessel", "9000001")
        assert eid == expected

    def test_mmsi_fallback(self) -> None:
        eid = AISVesselTool._vessel_entity_id(230000001, None)
        expected = entity_id_from_key("vessel", "mmsi:230000001")
        assert eid == expected

    def test_imo_zero_treated_as_missing(self) -> None:
        eid = AISVesselTool._vessel_entity_id(230000001, 0)
        expected = entity_id_from_key("vessel", "mmsi:230000001")
        assert eid == expected

    def test_entity_id_from_key_none(self) -> None:
        """When entity_id_from_key is unavailable, returns None."""
        with patch("agent.tools.ais_vessel.entity_id_from_key", None):
            eid = AISVesselTool._vessel_entity_id(230000001, 9000001)
        assert eid is None

    def test_same_imo_different_mmsi(self) -> None:
        """Reflagged vessel — same IMO → same entity_id."""
        eid_a = AISVesselTool._vessel_entity_id(230000001, 9000001)
        eid_b = AISVesselTool._vessel_entity_id(230000999, 9000001)
        assert eid_a == eid_b

    def test_different_imo_different_entity(self) -> None:
        eid_a = AISVesselTool._vessel_entity_id(230000001, 9000001)
        eid_b = AISVesselTool._vessel_entity_id(230000001, 9000002)
        assert eid_a != eid_b


# ===========================================================================
# Class: TestPersistGuard
# ===========================================================================


class TestPersistGuard:
    """Step 10b.4.4: Guard methods skip when store/entities unavailable."""

    def test_no_store_noop(self) -> None:
        tool = AISVesselTool()
        tool._persist_entities([_make_vessel()])  # should not raise

    def test_empty_list_noop(self) -> None:
        store = _make_store()
        tool = AISVesselTool(pipeline_store=store)
        tool._persist_entities([])
        store.register_entity.assert_not_called()

    def test_entity_id_unavailable(self) -> None:
        store = _make_store()
        tool = AISVesselTool(pipeline_store=store)
        with patch("agent.tools.ais_vessel.entity_id_from_key", None):
            tool._persist_entities([_make_vessel()])
        store.register_entity.assert_not_called()

    def test_persist_error_is_isolated_and_counted(self) -> None:
        """A failing vessel costs one vessel, and the failure is *reported*.

        This used to assert that ``_persist_entities_inner`` raises, i.e. that
        one bad vessel aborts the rest of the batch.  The caller then swallowed
        that exception and returned success — 9 rows of 500, green.  The
        contract now: keep going, count the failure, hand the count back.
        """
        store = _make_store()
        good = _make_vessel(mmsi=230000001, imo=9000001, name="GOOD SHIP")
        bad = _make_vessel(mmsi=230000002, imo=9000002, name="BAD SHIP")
        other = _make_vessel(mmsi=230000003, imo=9000003, name="OTHER SHIP")

        def flaky(**kwargs: Any) -> str:
            if kwargs.get("entity_type") == "vessel" and kwargs.get("canonical_name") == "BAD SHIP":
                raise RuntimeError("DB locked")
            return "eid"

        store.register_entity.side_effect = flaky

        receipt = AISVesselTool(pipeline_store=store)._persist_entities([good, bad, other])

        assert receipt["unique"] == 3
        assert receipt["failed"] == 1
        # The two healthy vessels still landed — the batch was not aborted.
        assert receipt["positions"] == 2

    def test_persist_receipt_without_store(self) -> None:
        """No store: nothing is claimed as written."""
        receipt = AISVesselTool()._persist_entities([_make_vessel()])
        assert receipt["positions"] == 0
        assert receipt["received"] == 1

    def test_port_call_no_store_noop(self) -> None:
        tool = AISVesselTool()
        tool._persist_port_call_entities([_make_port_call()])  # should not raise

    def test_port_call_entity_id_unavailable(self) -> None:
        store = _make_store()
        tool = AISVesselTool(pipeline_store=store)
        with patch("agent.tools.ais_vessel.entity_id_from_key", None):
            tool._persist_port_call_entities([_make_port_call()])
        store.register_entity.assert_not_called()


# ===========================================================================
# Class: TestPersistPositionEntities
# ===========================================================================


class TestPersistPositionEntities:
    """Step 10b.4.4: Position persistence — area/vessel modes."""

    def test_registers_vessel_with_imo(self) -> None:
        store = _make_store()
        tool = AISVesselTool(pipeline_store=store)
        v = _make_vessel(mmsi=230000001, imo=9000001, name="FINNLADY")
        tool._persist_entities_inner([v])

        vcalls = _vessel_register_calls(store)
        assert len(vcalls) == 1
        call_kw = vcalls[0].kwargs
        assert call_kw["entity_type"] == "vessel"
        assert call_kw["canonical_name"] == "FINNLADY"
        eid = entity_id_from_key("vessel", "9000001")
        assert call_kw["entity_id"] == eid

    def test_dual_aliases_with_imo(self) -> None:
        store = _make_store()
        tool = AISVesselTool(pipeline_store=store)
        v = _make_vessel(mmsi=230000001, imo=9000001)
        tool._persist_entities_inner([v])

        alias_calls = store.add_entity_alias.call_args_list
        assert len(alias_calls) == 2
        sources = {c.args[1] for c in alias_calls}
        assert sources == {"mmsi", "imo"}

    def test_mmsi_only_single_alias(self) -> None:
        store = _make_store()
        tool = AISVesselTool(pipeline_store=store)
        v = _make_vessel(mmsi=230000001, imo=None)
        tool._persist_entities_inner([v])

        alias_calls = store.add_entity_alias.call_args_list
        assert len(alias_calls) == 1
        assert alias_calls[0].args[1] == "mmsi"

    def test_position_observation_stored(self) -> None:
        store = _make_store()
        tool = AISVesselTool(pipeline_store=store)
        v = _make_vessel(lat=60.1, lon=25.0, sog=12.0, cog=180.0)
        tool._persist_entities_inner([v])

        store.store_entity_observation.assert_called_once()
        call_kw = store.store_entity_observation.call_args.kwargs
        assert call_kw["source_tool"] == "ais_vessel"
        assert call_kw["observation_type"] == "vessel_position"
        assert call_kw["depth_level"] == 2
        val = call_kw["value"]
        assert val["lat"] == 60.1
        assert val["lon"] == 25.0
        assert val["sog"] == 12.0
        assert val["cog"] == 180.0

    def test_dedup_by_entity_id(self) -> None:
        """Same vessel appearing twice in area scan → registered once."""
        store = _make_store()
        tool = AISVesselTool(pipeline_store=store)
        v1 = _make_vessel(mmsi=230000001, imo=9000001)
        v2 = _make_vessel(mmsi=230000001, imo=9000001, lat=60.2, lon=25.1)
        tool._persist_entities_inner([v1, v2])

        # Vessel registered once (dedup); country may also be registered
        assert len(_vessel_register_calls(store)) == 1
        # But observation stored only for first (dedup skips second)
        assert store.store_entity_observation.call_count == 1

    def test_multiple_distinct_vessels(self) -> None:
        store = _make_store()
        tool = AISVesselTool(pipeline_store=store)
        v1 = _make_vessel(mmsi=230000001, imo=9000001, name="SHIP_A")
        v2 = _make_vessel(mmsi=230000002, imo=9000002, name="SHIP_B")
        tool._persist_entities_inner([v1, v2])

        assert len(_vessel_register_calls(store)) == 2
        assert store.store_entity_observation.call_count == 2

    def test_name_fallback_to_mmsi(self) -> None:
        store = _make_store()
        tool = AISVesselTool(pipeline_store=store)
        v = _make_vessel(mmsi=230000001, imo=None, name="")
        tool._persist_entities_inner([v])

        vcalls = _vessel_register_calls(store)
        assert vcalls[0].kwargs["canonical_name"] == "230000001"

    def test_metadata_passed(self) -> None:
        store = _make_store()
        tool = AISVesselTool(pipeline_store=store)
        v = _make_vessel(
            ship_type="tanker",
            ship_type_code=80,
            destination="ROTTERDAM",
            draught=125.0,
        )
        tool._persist_entities_inner([v])

        vcalls = _vessel_register_calls(store)
        call_kw = vcalls[0].kwargs
        meta = call_kw["metadata"]
        assert meta["ship_type"] == "tanker"
        assert meta["destination"] == "ROTTERDAM"

    def test_no_position_no_observation(self) -> None:
        """Vessel with no lat/lon → entity registered but no position obs."""
        store = _make_store()
        tool = AISVesselTool(pipeline_store=store)
        v = {"mmsi": 230000001, "imo": 9000001, "name": "GHOST"}
        tool._persist_entities_inner([v])

        store.register_entity.assert_called_once()
        store.store_entity_observation.assert_not_called()

    def test_skip_vessel_missing_both_ids(self) -> None:
        """Vessel with no MMSI and no IMO → skipped entirely."""
        store = _make_store()
        tool = AISVesselTool(pipeline_store=store)
        v = {"lat": 60.0, "lon": 25.0, "name": "UNKNOWN"}
        tool._persist_entities_inner([v])

        store.register_entity.assert_not_called()


# ===========================================================================
# Class: TestPersistPortCallEntities
# ===========================================================================


class TestPersistPortCallEntities:
    """Step 10b.4.5: Port call persistence logic."""

    def test_registers_vessel_from_port_call(self) -> None:
        store = _make_store()
        tool = AISVesselTool(pipeline_store=store)
        c = _make_port_call(mmsi=230000001, imo=9000001)
        tool._persist_port_call_entities_inner([c])

        store.register_entity.assert_called_once()
        call_kw = store.register_entity.call_args.kwargs
        assert call_kw["entity_type"] == "vessel"
        assert call_kw["canonical_name"] == "FINNLADY"

    def test_port_call_observation_type(self) -> None:
        store = _make_store()
        tool = AISVesselTool(pipeline_store=store)
        c = _make_port_call(port="VUOSAARI", prev_port="TALLINN", next_port="TALLINN")
        tool._persist_port_call_entities_inner([c])

        store.store_entity_observation.assert_called_once()
        call_kw = store.store_entity_observation.call_args.kwargs
        assert call_kw["observation_type"] == "port_call"
        assert call_kw["depth_level"] == 2
        val = call_kw["value"]
        assert val["port"] == "VUOSAARI"
        assert val["prev_port"] == "TALLINN"
        assert val["next_port"] == "TALLINN"
        assert val["arrival_with_cargo"] is True

    def test_mmsi_only_port_call(self) -> None:
        store = _make_store()
        tool = AISVesselTool(pipeline_store=store)
        c = _make_port_call(mmsi=230000001, imo=None)
        tool._persist_port_call_entities_inner([c])

        alias_calls = store.add_entity_alias.call_args_list
        assert len(alias_calls) == 1
        assert alias_calls[0].args[1] == "mmsi"

    def test_missing_both_ids_skipped(self) -> None:
        store = _make_store()
        tool = AISVesselTool(pipeline_store=store)
        c = _make_port_call(mmsi=None, imo=None)
        tool._persist_port_call_entities_inner([c])

        store.register_entity.assert_not_called()

    def test_dedup_across_port_calls(self) -> None:
        """Same vessel with multiple port calls → registered once, 2 observations."""
        store = _make_store()
        tool = AISVesselTool(pipeline_store=store)
        c1 = _make_port_call(mmsi=230000001, imo=9000001, port="VUOSAARI")
        c2 = _make_port_call(mmsi=230000001, imo=9000001, port="HELSINKI")
        tool._persist_port_call_entities_inner([c1, c2])

        # Vessel registered once (dedup); country may also be registered for HELSINKI
        assert len(_vessel_register_calls(store)) == 1
        assert store.store_entity_observation.call_count == 2

    def test_port_call_metadata(self) -> None:
        store = _make_store()
        tool = AISVesselTool(pipeline_store=store)
        c = _make_port_call(vessel_type_code=80, nationality="FI")
        tool._persist_port_call_entities_inner([c])

        call_kw = store.register_entity.call_args.kwargs
        meta = call_kw["metadata"]
        assert meta["vesselTypeCode"] == 80
        assert meta["nationality"] == "FI"

    def test_port_call_eta_as_observed_at(self) -> None:
        """When eta is available, used as observed_at instead of now()."""
        store = _make_store()
        tool = AISVesselTool(pipeline_store=store)
        c = _make_port_call(eta="2025-06-15T12:00:00Z")
        tool._persist_port_call_entities_inner([c])

        call_kw = store.store_entity_observation.call_args.kwargs
        # Was asserted as the raw ISO string. Production always converts
        # observed_at to a float Unix timestamp via _observed_at_ts (the same
        # float-only invariant enforced elsewhere in the pipeline -- storing a
        # date string was a latent bug per agent/tools/gdelt.py's own
        # comment). 1749988800.0 is the correct epoch value for
        # 2025-06-15T12:00:00Z. Fixed 2026-08-27.
        assert call_kw["observed_at"] == 1749988800.0


# ===========================================================================
# Class: TestEntityIdsOutput
# ===========================================================================


class TestEntityIdsOutput:
    """Step 10b.4.7: entity_id fields in output dicts."""

    def test_area_mode_entity_ids(self) -> None:
        """Area mode adds entity_id to each vessel dict."""
        store = _make_store()
        tool = AISVesselTool(pipeline_store=store)

        feature = {
            "mmsi": 230000001,
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [25.0, 60.1]},
            "properties": {
                "mmsi": 230000001,
                "sog": 12.0,
                "cog": 180.0,
                "heading": 180,
                "navStat": 0,
            },
        }
        meta = {
            230000001: {
                "name": "FINNLADY",
                "imo": 9000001,
                "shipType": 60,
                "destination": "TALLINN",
            },
        }

        with (
            patch.object(tool, "_fetch_locations", return_value=[feature]),
            patch.object(tool, "_fetch_metadata", return_value=meta),
        ):
            result = tool.execute(
                mode="area",
                lat_min=59.0,
                lat_max=61.0,
                lon_min=24.0,
                lon_max=26.0,
                ship_type="passenger",
            )

        assert result.success
        vessels = result.data["vessels"]
        assert len(vessels) > 0
        eid = vessels[0].get("entity_id")
        expected = entity_id_from_key("vessel", "9000001")
        assert eid == expected

    def test_vessel_mode_entity_id(self) -> None:
        """Vessel mode adds entity_id to result dict."""
        store = _make_store()
        tool = AISVesselTool(pipeline_store=store)

        meta = {
            "name": "FINNLADY",
            "imo": 9000001,
            "shipType": 60,
            "callSign": "OJAS",
            "destination": "TALLINN",
            "draught": 65,
        }
        feature = {
            "mmsi": 230000001,
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [25.0, 60.1]},
            "properties": {
                "mmsi": 230000001,
                "sog": 12.0,
                "cog": 180.0,
                "heading": 180,
                "navStat": 0,
            },
        }

        with (
            patch.object(tool, "_fetch_vessel_metadata_single", return_value=meta),
            patch.object(tool, "_fetch_locations", return_value=[feature]),
        ):
            result = tool.execute(mode="vessel", mmsi=230000001)

        assert result.success
        eid = result.data.get("entity_id")
        expected = entity_id_from_key("vessel", "9000001")
        assert eid == expected

    def test_port_calls_entity_ids(self) -> None:
        """Port calls mode adds entity_id to each call dict."""
        store = _make_store()
        tool = AISVesselTool(pipeline_store=store)

        calls = [
            _make_port_call(mmsi=230000001, imo=9000001, port="VUOSAARI"),
        ]

        with patch.object(tool, "_fetch_port_calls", return_value=calls):
            result = tool.execute(mode="port_calls", from_date="2025-01-01")

        assert result.success
        call_data = result.data["calls"]
        assert len(call_data) > 0
        eid = call_data[0].get("entity_id")
        expected = entity_id_from_key("vessel", "9000001")
        assert eid == expected

    def test_destination_flow_no_entity_ids(self) -> None:
        """destination_flow mode is aggregate — should NOT have entity_ids."""
        tool = AISVesselTool()

        feature = {
            "mmsi": 230000001,
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [25.0, 60.1]},
            "properties": {
                "mmsi": 230000001,
                "sog": 12.0,
                "cog": 180.0,
                "heading": 180,
                "navStat": 0,
            },
        }
        meta = {
            230000001: {
                "name": "FINNLADY",
                "imo": 9000001,
                "shipType": 60,
                "destination": "TALLINN",
            },
        }

        with (
            patch.object(tool, "_fetch_locations", return_value=[feature]),
            patch.object(tool, "_fetch_metadata", return_value=meta),
        ):
            result = tool.execute(mode="destination_flow")

        assert result.success
        # destination_flow data should NOT have entity_id keys
        assert "entity_id" not in result.data

    def test_entity_id_none_when_module_unavailable(self) -> None:
        """When entity_id_from_key is None, entity_id not added to vessel dicts."""
        tool = AISVesselTool()

        feature = {
            "mmsi": 230000001,
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [25.0, 60.1]},
            "properties": {
                "mmsi": 230000001,
                "sog": 12.0,
                "cog": 180.0,
                "heading": 180,
                "navStat": 0,
            },
        }

        with (
            patch.object(tool, "_fetch_locations", return_value=[feature]),
            patch("agent.tools.ais_vessel.entity_id_from_key", None),
        ):
            result = tool.execute(
                mode="area",
                lat_min=59.0,
                lat_max=61.0,
                lon_min=24.0,
                lon_max=26.0,
            )

        assert result.success
        vessels = result.data["vessels"]
        if vessels:
            assert "entity_id" not in vessels[0]


# ===========================================================================
# Class: TestIntegration
# ===========================================================================


class TestIntegration:
    """End-to-end integration scenarios."""

    def test_backward_compat_no_store(self) -> None:
        """Tool works identically without pipeline_store."""
        tool = AISVesselTool()

        feature = {
            "mmsi": 230000001,
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [25.0, 60.1]},
            "properties": {
                "mmsi": 230000001,
                "sog": 12.0,
                "cog": 180.0,
                "heading": 180,
                "navStat": 0,
            },
        }

        with patch.object(tool, "_fetch_locations", return_value=[feature]):
            result = tool.execute(
                mode="area",
                lat_min=59.0,
                lat_max=61.0,
                lon_min=24.0,
                lon_max=26.0,
            )

        assert result.success

    def test_persistence_error_non_fatal_area(self) -> None:
        """Persistence errors don't break the area mode result."""
        store = _make_store()
        store.register_entity.side_effect = RuntimeError("DB locked")
        tool = AISVesselTool(pipeline_store=store)

        feature = {
            "mmsi": 230000001,
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [25.0, 60.1]},
            "properties": {
                "mmsi": 230000001,
                "sog": 12.0,
                "cog": 180.0,
                "heading": 180,
                "navStat": 0,
            },
        }

        with patch.object(tool, "_fetch_locations", return_value=[feature]):
            result = tool.execute(
                mode="area",
                lat_min=59.0,
                lat_max=61.0,
                lon_min=24.0,
                lon_max=26.0,
            )

        assert result.success

    def test_persistence_error_non_fatal_port_calls(self) -> None:
        """Persistence errors don't break the port_calls mode result."""
        store = _make_store()
        store.register_entity.side_effect = RuntimeError("DB locked")
        tool = AISVesselTool(pipeline_store=store)

        calls = [_make_port_call()]

        with patch.object(tool, "_fetch_port_calls", return_value=calls):
            result = tool.execute(mode="port_calls", from_date="2025-01-01")

        assert result.success


# ===========================================================================
# Class: TestRealStoreIntegration
# ===========================================================================


class TestRealStoreIntegration:
    """MI measurement: real PipelineStore + entities persisted + queried back."""

    def test_l2_with_real_store(self, tmp_path: Path) -> None:
        from agent.pipeline.store import PipelineStore

        db = tmp_path / "test.db"
        ps = PipelineStore(db_path=db)
        tool = AISVesselTool(pipeline_store=ps)

        v = _make_vessel(mmsi=230000001, imo=9000001, name="FINNLADY")
        tool._persist_entities_inner([v])

        eid = entity_id_from_key("vessel", "9000001")
        obs = ps.query_entity_observations(eid)
        assert len(obs) == 1
        assert obs[0]["observation_type"] == "vessel_position"
        val = obs[0]["value"]
        assert val["lat"] == 60.1
        assert val["lon"] == 25.0

    def test_port_call_with_real_store(self, tmp_path: Path) -> None:
        from agent.pipeline.store import PipelineStore

        db = tmp_path / "test.db"
        ps = PipelineStore(db_path=db)
        tool = AISVesselTool(pipeline_store=ps)

        c = _make_port_call(mmsi=230000001, imo=9000001, port="VUOSAARI")
        tool._persist_port_call_entities_inner([c])

        eid = entity_id_from_key("vessel", "9000001")
        obs = ps.query_entity_observations(eid)
        assert len(obs) == 1
        assert obs[0]["observation_type"] == "port_call"
        val = obs[0]["value"]
        assert val["port"] == "VUOSAARI"

    def test_dual_observation_types(self, tmp_path: Path) -> None:
        """Same vessel: position + port_call → both stored."""
        from agent.pipeline.store import PipelineStore

        db = tmp_path / "test.db"
        ps = PipelineStore(db_path=db)
        tool = AISVesselTool(pipeline_store=ps)

        v = _make_vessel(mmsi=230000001, imo=9000001)
        tool._persist_entities_inner([v])

        c = _make_port_call(mmsi=230000001, imo=9000001, port="VUOSAARI")
        tool._persist_port_call_entities_inner([c])

        eid = entity_id_from_key("vessel", "9000001")
        obs = ps.query_entity_observations(eid)
        assert len(obs) == 2
        types = {o["observation_type"] for o in obs}
        assert types == {"vessel_position", "port_call"}


# ===========================================================================
# Class: TestMIMeasurement
# ===========================================================================


class TestMIMeasurement:
    """Mutual information: L2 provides strictly more entity info than L1."""

    def test_l2_more_info_than_l1(self) -> None:
        """L2 vessel list contains entity_id field; L1 does not."""
        v_l1 = _make_vessel()
        assert "entity_id" not in v_l1

        v_l2 = _make_vessel()
        eid = AISVesselTool._vessel_entity_id(v_l2["mmsi"], v_l2.get("imo"))
        v_l2["entity_id"] = eid
        assert "entity_id" in v_l2
        assert v_l2["entity_id"] is not None

    def test_mi_with_real_store(self, tmp_path: Path) -> None:
        """L2 with real store: entity queryable after persistence."""
        from agent.pipeline.store import PipelineStore

        db = tmp_path / "test.db"
        ps = PipelineStore(db_path=db)
        tool = AISVesselTool(pipeline_store=ps)

        v = _make_vessel(mmsi=230000001, imo=9000001)
        tool._persist_entities_inner([v])

        eid = entity_id_from_key("vessel", "9000001")
        obs = ps.query_entity_observations(eid)
        assert len(obs) >= 1
        # L2 provides structured entity-resolved observations
        val = obs[0]["value"]
        assert "lat" in val and "lon" in val


# ===========================================================================
# Class: TestDagSnapshotWritesVesselRows
# ===========================================================================

# Copied verbatim from agent/pipeline/dags/daily_collection.py (the
# "fetch_ais_vessel" node).  If that node's params change, this literal must be
# updated deliberately — the regression below is only meaningful when it runs
# the exact wiring production runs.
DAG_PARAMS = {"mode": "area_daily_snapshot", "area_name": "full_baltic", "ship_type": "tanker"}


def _baltic_feature(
    mmsi: int,
    lat: float,
    lon: float,
    sog: float,
    *,
    reported_at_ms: int | None = None,
) -> dict[str, Any]:
    """A location feature inside the full_baltic bbox (54-66N, 9-31E).

    ``timestampExternal`` is epoch **milliseconds** and ``timestamp`` is the AIS
    second-of-minute field, exactly as the live endpoint serves them (sampled
    2026-09-23: ``{"timestamp": 49, "timestampExternal": 1790176129011}``).
    """
    return {
        "mmsi": mmsi,
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": {
            "mmsi": mmsi,
            "sog": sog,
            "cog": 90.0,
            "heading": 91,
            "navStat": 0,
            "timestamp": 49,
            "timestampExternal": reported_at_ms if reported_at_ms is not None else int(time.time() * 1000),
        },
    }


class TestDagSnapshotWritesVesselRows:
    """Regression: the DAG's snapshot mode must write per-vessel rows.

    Between commit 48cb217 (2026-08-25) and 2026-09-23 the DAG node ran
    ``mode="area_daily_snapshot"``, which fetched every live position and then
    threw the positions away — one ``area_daily_activity`` row per run, zero
    ``vessel_position`` rows, and ``result.success`` True throughout.  AIS is
    time-gated: the API serves only "now", so each silent run was a day of moat
    data that cannot be refetched at any price.

    These assertions are on **row counts**, not on ``result.success``, because
    ``result.success`` never noticed.
    """

    def _run_dag_node(
        self,
        tmp_path: Path,
        *,
        features: list[dict[str, Any]] | None = None,
        meta: dict[int, dict[str, Any]] | None = None,
        store: Any = None,
        feed_updated_at: str | None = None,
        db_name: str = "t.db",
    ) -> tuple[Any, Any]:
        from agent.pipeline.store import PipelineStore

        ps = store or PipelineStore(db_path=tmp_path / db_name)
        tool = AISVesselTool(pipeline_store=ps)

        if features is None:
            features = [
                _baltic_feature(230000001, 60.1, 25.0, 12.0),  # tanker
                _baltic_feature(230000002, 59.0, 22.0, 0.2),  # tanker
                _baltic_feature(230000003, 58.0, 20.0, 8.0),  # cargo
            ]
        if meta is None:
            meta = {
                230000001: {"name": "NORDIC STAR", "imo": 9000001, "shipType": 80, "destination": "TALLINN"},
                230000002: {"name": "BALTIC SUN", "imo": 9000002, "shipType": 84, "destination": "ROTTERDAM"},
                230000003: {"name": "GOTLAND", "imo": 9000003, "shipType": 70, "destination": "GDANSK"},
            }

        # Patching _fetch_locations bypasses the payload envelope, so the feed's
        # publish time is injected here the way a real fetch would record it.
        tool._last_feed_updated_at = feed_updated_at

        with (
            patch.object(tool, "_fetch_locations", return_value=features),
            patch.object(tool, "_fetch_metadata", return_value=meta),
        ):
            result = tool.execute(**DAG_PARAMS)

        return ps, result

    @staticmethod
    def _row_counts(db: Path) -> dict[str, int]:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
            return dict(
                con.execute(
                    "SELECT observation_type, count(*) FROM entity_observations "
                    "WHERE source_tool='ais_vessel' GROUP BY 1"
                ).fetchall()
            )

    def test_snapshot_writes_vessel_position_rows(self, tmp_path: Path) -> None:
        """The node writes per-vessel rows AND keeps the single area row."""
        ps, result = self._run_dag_node(tmp_path)
        assert result.success

        with sqlite3.connect(f"file:{tmp_path / 't.db'}?mode=ro", uri=True) as con:
            counts = dict(
                con.execute(
                    "SELECT observation_type, count(*) FROM entity_observations "
                    "WHERE source_tool='ais_vessel' GROUP BY 1"
                ).fetchall()
            )

        # The whole point: > 0, where the unfixed code wrote exactly 0.
        assert counts.get("vessel_position", 0) > 0
        assert counts.get("vessel_position") == 3
        # ...without losing the MP-1 daily series.
        assert counts.get("area_daily_activity") == 1

    def test_snapshot_preserves_mp1_series_shape(self, tmp_path: Path) -> None:
        """``series_count`` and ``data`` stay as ghost_chains/mp1_data_health read them."""
        _, result = self._run_dag_node(tmp_path)

        assert result.data["series_count"] == 2  # two tankers
        assert result.data["tanker_count"] == 2
        assert result.data["vessel_count"] == 3
        # Per-vessel records must not bloat the persisted area row.
        assert "vessels" not in result.data

    def test_requested_ship_type_survives_the_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every tanker survives the cut — asserted by identity, not by length.

        The previous version of this test asserted
        ``len(ranked) == _MAX_VESSEL_OBS_PER_RUN``, whose only content is that
        truncation happened and that this is fine.  That is the shape of
        ``test_500_returns_partial_results`` asserting ``len(result) == 1``
        against ``total=200``: a test certifying the data loss.
        """
        monkeypatch.setattr(ais_vessel_mod, "_MAX_VESSEL_OBS_PER_RUN", 3)
        vessels = [{"mmsi": i, "ship_type": "cargo"} for i in range(10)]
        vessels += [{"mmsi": 901, "ship_type": "tanker"}, {"mmsi": 902, "ship_type": "tanker"}]

        ranked = AISVesselTool._rank_vessels_for_persist(vessels, "tanker")

        kept = {v["mmsi"] for v in ranked}
        assert {901, 902} <= kept, "the signal ships must never be the ones cut"

    def test_cap_is_above_live_feed_volume(self) -> None:
        """Tripwire: the ceiling must not bite on a normal day.

        The Digitraffic locations feed carried 1,187 vessels on 2026-09-23,
        1,185 of them inside the full_baltic bbox.  A cap below that silently
        discards positions that AIS can never serve again — which is what the
        old value of 500 did, on 58% of every run.
        """
        assert _MAX_VESSEL_OBS_PER_RUN > 1185 * 2

    def test_truncation_is_visible_when_it_bites(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If the ceiling ever fires, it must show up in the log AND in the row."""
        monkeypatch.setattr(ais_vessel_mod, "_MAX_VESSEL_OBS_PER_RUN", 2)
        caplog.set_level(logging.WARNING, logger="agent.tools.ais_vessel")

        _, result = self._run_dag_node(tmp_path)

        assert "truncated" in caplog.text.lower()
        assert result.data["vessels_available"] == 3
        assert result.data["vessels_persisted"] == 2
        assert result.data["vessels_truncated"] == 1
        # The dropped vessel is the cargo one, not a tanker.
        assert self._row_counts(tmp_path / "t.db")["vessel_position"] == 2
        # ...and the node goes RED.  A warning in a log file nobody greps is
        # how 58% of every run disappeared for four months: the only signal
        # the DAG reads is result.success (agent/pipeline/operators.py).
        assert result.success is False
        assert "dropped by" in result.output

    def test_result_data_carries_the_write_receipt(self, tmp_path: Path) -> None:
        """Downstream must be able to see rows-written, not only things-seen."""
        _, result = self._run_dag_node(tmp_path)

        assert result.data["vessels_available"] == 3
        assert result.data["vessels_persisted"] == 3
        assert result.data["vessels_failed"] == 0
        assert result.data["feed_feature_count"] == 3
        assert "3 written / 3 available" in result.output

    def test_one_failing_write_does_not_abort_the_batch_and_is_reported(
        self,
        tmp_path: Path,
    ) -> None:
        """Fault injection: the exact shape that wrote 9 rows of 500 and went green.

        A store failure on one vessel must (a) cost that one vessel only and
        (b) turn the node red — ``agent/pipeline/operators.py`` raises only when
        ``result.success`` is False, so success=True here is a silent loss.
        """
        from agent.pipeline.store import PipelineStore

        ps = PipelineStore(db_path=tmp_path / "fault.db")
        real = ps.store_entity_observation
        seen = {"n": 0}

        def flaky(**kwargs: Any) -> int:
            if kwargs.get("observation_type") == "vessel_position":
                seen["n"] += 1
                if seen["n"] == 2:
                    raise RuntimeError("DB locked")
            return real(**kwargs)

        ps.store_entity_observation = flaky  # type: ignore[method-assign]

        _, result = self._run_dag_node(tmp_path, store=ps, db_name="fault.db")

        counts = self._row_counts(tmp_path / "fault.db")
        # Isolation: the third vessel was written even though the second blew up.
        assert counts["vessel_position"] == 2
        # ...and the MP-1 area row still landed.
        assert counts["area_daily_activity"] == 1
        # Honesty: a partial write is not a success.
        assert result.success is False
        assert result.data["vessels_failed"] == 1
        assert result.data["vessels_persisted"] == 2

    def test_empty_feed_is_a_failure_not_a_zero(self, tmp_path: Path) -> None:
        """An empty upstream must not enter the z-score series as 'no ships'."""
        _, result = self._run_dag_node(tmp_path, features=[], meta={})

        assert result.success is False
        assert "0 features" in result.output
        # Nothing at all was written — no fabricated zero row.
        assert self._row_counts(tmp_path / "t.db") == {}

    def test_stale_feed_reports_degraded(self, tmp_path: Path) -> None:
        """A frozen feed is a replay, not a measurement — the node must say so."""
        _, result = self._run_dag_node(tmp_path, feed_updated_at="2026-09-01T00:00:00Z")

        assert result.success is False
        assert "stale" in result.output.lower()
        # The rows are still written (they are real positions); only the claim
        # that this is today's measurement is withdrawn.
        assert self._row_counts(tmp_path / "t.db")["vessel_position"] == 3

    def test_observed_at_is_the_ais_report_time(self, tmp_path: Path) -> None:
        """observed_at comes from timestampExternal, not from the fetch clock."""
        reported_ms = int((time.time() - 6 * 3600) * 1000)
        features = [_baltic_feature(230000001, 60.1, 25.0, 12.0, reported_at_ms=reported_ms)]
        meta = {230000001: {"name": "NORDIC STAR", "imo": 9000001, "shipType": 80}}

        self._run_dag_node(tmp_path, features=features, meta=meta)

        with sqlite3.connect(f"file:{tmp_path / 't.db'}?mode=ro", uri=True) as con:
            (observed_at,) = con.execute(
                "SELECT observed_at FROM entity_observations WHERE observation_type='vessel_position'"
            ).fetchone()

        assert abs(observed_at - reported_ms / 1000) < 1.0
        # And decisively not the fetch clock: the report is 6h old.
        assert time.time() - observed_at > 5 * 3600

    def test_rerun_collapses_instead_of_duplicating(self, tmp_path: Path) -> None:
        """A retry serves byte-identical positions; they must not double the rows.

        OBSERVATION_UNIQUE_KEY includes observed_at, so a fetch-clock timestamp
        made every retry append a fresh copy — 1,000 of the live table's 1,550
        vessel_position rows are that artifact.
        """
        from agent.pipeline.store import PipelineStore

        ps = PipelineStore(db_path=tmp_path / "t.db")
        features = [
            _baltic_feature(230000001, 60.1, 25.0, 12.0, reported_at_ms=1790176129011),
            _baltic_feature(230000002, 59.0, 22.0, 0.2, reported_at_ms=1790176130000),
        ]
        meta = {
            230000001: {"name": "NORDIC STAR", "imo": 9000001, "shipType": 80},
            230000002: {"name": "BALTIC SUN", "imo": 9000002, "shipType": 84},
        }

        for _ in range(2):
            self._run_dag_node(tmp_path, features=features, meta=meta, store=ps)

        assert self._row_counts(tmp_path / "t.db")["vessel_position"] == 2

    def test_missing_report_time_is_counted_not_hidden(self, tmp_path: Path) -> None:
        """A feed that stops publishing timestampExternal must not degrade quietly.

        The fetch clock is a legitimate fallback — the row is real and it lands
        — but it is dated by when we asked rather than by when the vessel
        reported, and observed_at is part of OBSERVATION_UNIQUE_KEY, so those
        rows stop collapsing on retry.  That is a slow-motion version of the
        duplicate footprint already in the live table, so it is counted.
        """
        clean = _baltic_feature(230000001, 60.1, 25.0, 12.0)
        naked = _baltic_feature(230000002, 59.0, 22.0, 0.2)
        del naked["properties"]["timestampExternal"]
        meta = {
            230000001: {"name": "NORDIC STAR", "imo": 9000001, "shipType": 80},
            230000002: {"name": "BALTIC SUN", "imo": 9000002, "shipType": 84},
        }

        _, result = self._run_dag_node(tmp_path, features=[clean, naked], meta=meta)

        # The row still lands — the fallback is not a drop.
        assert self._row_counts(tmp_path / "t.db")["vessel_position"] == 2
        assert result.data["vessels_persisted"] == 2
        assert result.data["vessels_without_report_time"] == 1


class TestFeedPublishTimeIsActuallyWired:
    """The staleness guard is only as good as the value feeding it.

    Every snapshot test injects ``_last_feed_updated_at`` onto the instance
    because it patches ``_fetch_locations``.  That leaves the guard's input
    untested: if the real fetch never recorded ``dataUpdatedTime``, the
    staleness check would silently never fire and the whole suite would stay
    green — the right mechanism on the wrong wiring.
    """

    class _Resp:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return self._payload

    class _Cache:
        def __init__(self, seeded: Any = None) -> None:
            self.seeded = seeded
            self.put_calls: list[Any] = []

        def get(self, namespace: str, key: dict[str, Any]) -> Any:
            return self.seeded

        def put(self, namespace: str, key: dict[str, Any], value: Any) -> None:
            self.put_calls.append(value)

    def test_real_fetch_records_the_feeds_publish_time(self) -> None:
        """``_fetch_locations`` must set what ``_feed_age_seconds`` reads."""
        tool = AISVesselTool()
        payload = {
            "type": "FeatureCollection",
            "dataUpdatedTime": "2026-09-23T16:12:20Z",
            "features": [_baltic_feature(230000001, 60.1, 25.0, 12.0)],
        }

        with patch.object(tool, "_get", return_value=self._Resp(payload)):
            features = tool._fetch_locations()

        assert len(features) == 1
        assert tool._last_feed_updated_at == "2026-09-23T16:12:20Z"

        # And the snapshot mode reads it back off the counts, not off a
        # test-injected attribute.
        with (
            patch.object(tool, "_get", return_value=self._Resp(payload)),
            patch.object(tool, "_fetch_metadata", return_value={230000001: {"shipType": 80}}),
        ):
            counts = tool._count_area_vessels("full_baltic", ship_type="all")
        assert counts["feed_updated_at"] == "2026-09-23T16:12:20Z"
        assert tool._feed_age_seconds(counts["feed_updated_at"]) is not None

    def test_cached_payload_keeps_the_publish_time(self) -> None:
        """A cache hit is not a fresh feed — it must carry its own timestamp."""
        cache = self._Cache(
            seeded={
                "dataUpdatedTime": "2026-09-01T00:00:00Z",
                "features": [_baltic_feature(230000001, 60.1, 25.0, 12.0)],
            }
        )
        tool = AISVesselTool(cache=cache)

        tool._fetch_locations()

        assert tool._last_feed_updated_at == "2026-09-01T00:00:00Z"
        age = tool._feed_age_seconds(tool._last_feed_updated_at)
        assert age is not None and age > _MAX_FEED_AGE_SECONDS

    def test_legacy_list_cache_entry_reports_unknown_not_fresh(self) -> None:
        """A pre-envelope cache entry has no publish time; it must not fake one."""
        cache = self._Cache(seeded=[_baltic_feature(230000001, 60.1, 25.0, 12.0)])
        tool = AISVesselTool(cache=cache)

        features = tool._fetch_locations()

        assert len(features) == 1
        assert tool._last_feed_updated_at is None
        assert tool._feed_age_seconds(None) is None
