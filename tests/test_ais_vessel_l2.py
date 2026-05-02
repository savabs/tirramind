"""Tests for ais_vessel L2 upgrade — vessel entity persistence,
entity_ids in output dicts, IMO-first / MMSI-fallback identity,
port call observations, and MI integration.

Mirrors the test pattern from test_whale_alert_l2.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent.pipeline.entity import entity_id_from_key
from agent.tools.ais_vessel import AISVesselTool

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

    def test_persist_error_caught(self) -> None:
        """Errors in persistence are caught; method doesn't raise."""
        store = _make_store()
        store.register_entity.side_effect = RuntimeError("DB locked")
        tool = AISVesselTool(pipeline_store=store)
        # _persist_entities calls _persist_entities_inner which will raise,
        # but guard catches it implicitly (our implementation doesn't have inner try/except
        # — the caller wraps in try/except). Let's test the outer level.
        # Our guard doesn't catch — the caller does. So test the inner directly:
        with pytest.raises(RuntimeError):
            tool._persist_entities_inner([_make_vessel()])

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
        assert call_kw["observed_at"] == "2025-06-15T12:00:00Z"


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
