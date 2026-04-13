"""Tests for supply_chain_monitor L2 entity persistence."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent.tools.supply_chain_monitor import SupplyChainMonitorTool


# ── Helpers ──────────────────────────────────────────────────────


def _make_series_data() -> dict[str, Any]:
    """Simulates BLS series data as returned by _fetch_bls_multi."""
    return {
        "PCU334413334413": {
            "label": "Semiconductors & Related Devices",
            "sector": "tech",
            "values": [
                {"period": "M01", "value": "120.5"},
                {"period": "M02", "value": "121.3"},
                {"period": "M03", "value": "122.0"},
            ],
        },
        "PCU331110331110": {
            "label": "Iron & Steel Mills",
            "sector": "materials",
            "values": [
                {"period": "M01", "value": "200.1"},
                {"period": "M02", "value": "198.7"},
            ],
        },
    }


def _make_mock_store() -> MagicMock:
    store = MagicMock()
    store.register_entity.return_value = "mock_eid"
    store.store_entity_observation.return_value = 1
    return store


# ── No pipeline_store → graceful skip ──


class TestNoPipelineStore:
    def test_no_store_is_noop(self):
        tool = SupplyChainMonitorTool(cache=None)
        tool._persist_entities(_make_series_data())

    def test_empty_data_no_calls(self):
        store = _make_mock_store()
        tool = SupplyChainMonitorTool(cache=None, pipeline_store=store)
        tool._persist_entities({})
        store.register_entity.assert_not_called()


# ── Entity registration ──


class TestEntityRegistration:
    def test_topic_entity_per_series(self):
        store = _make_mock_store()
        tool = SupplyChainMonitorTool(cache=None, pipeline_store=store)
        data = _make_series_data()
        tool._persist_entities(data)
        topic_calls = [
            c
            for c in store.register_entity.call_args_list
            if c.kwargs.get("entity_type") == "topic"
        ]
        assert len(topic_calls) == 2

    def test_entity_canonical_name_is_label(self):
        store = _make_mock_store()
        tool = SupplyChainMonitorTool(cache=None, pipeline_store=store)
        data = {
            "EIUIR": {
                "label": "All Imports",
                "sector": "imports",
                "values": [],
            }
        }
        tool._persist_entities(data)
        call = store.register_entity.call_args
        assert call.kwargs["canonical_name"] == "All Imports"

    def test_metadata_contains_series_id_and_sector(self):
        store = _make_mock_store()
        tool = SupplyChainMonitorTool(cache=None, pipeline_store=store)
        data = {
            "PCU334413334413": {
                "label": "Semiconductors",
                "sector": "tech",
                "values": [],
            }
        }
        tool._persist_entities(data)
        call = store.register_entity.call_args
        meta = call.kwargs.get("metadata", {})
        assert meta["series_id"] == "PCU334413334413"
        assert meta["sector"] == "tech"


# ── Observation storage ──


class TestObservationStorage:
    def test_observation_type_is_price_movement(self):
        store = _make_mock_store()
        tool = SupplyChainMonitorTool(cache=None, pipeline_store=store)
        tool._persist_entities(_make_series_data())
        for obs_call in store.store_entity_observation.call_args_list:
            assert obs_call.kwargs["observation_type"] == "price_movement"
            assert obs_call.kwargs["depth_level"] == 2
            assert obs_call.kwargs["source_tool"] == "supply_chain_monitor"

    def test_latest_value_captured(self):
        store = _make_mock_store()
        tool = SupplyChainMonitorTool(cache=None, pipeline_store=store)
        data = {
            "PCU334413334413": {
                "label": "Semiconductors",
                "sector": "tech",
                "values": [
                    {"period": "M01", "value": "120.5"},
                    {"period": "M02", "value": "121.3"},
                ],
            }
        }
        tool._persist_entities(data)
        obs_call = store.store_entity_observation.call_args
        val = obs_call.kwargs["value"]
        assert val["latest_value"] == 121.3
        assert val["num_periods"] == 2

    def test_empty_values_latest_is_none(self):
        store = _make_mock_store()
        tool = SupplyChainMonitorTool(cache=None, pipeline_store=store)
        data = {
            "PCU000": {
                "label": "Empty Series",
                "sector": "test",
                "values": [],
            }
        }
        tool._persist_entities(data)
        obs_call = store.store_entity_observation.call_args
        assert obs_call.kwargs["value"]["latest_value"] is None
        assert obs_call.kwargs["value"]["num_periods"] == 0

    def test_observation_count_matches_series_count(self):
        store = _make_mock_store()
        tool = SupplyChainMonitorTool(cache=None, pipeline_store=store)
        tool._persist_entities(_make_series_data())
        assert store.store_entity_observation.call_count == 2


# ── Idempotency ──


class TestIdempotency:
    def test_same_series_produces_same_entity_id(self):
        """Calling with the same series_id should use the same entity_id."""
        store = _make_mock_store()
        tool = SupplyChainMonitorTool(cache=None, pipeline_store=store)
        data1 = {
            "PCU334413334413": {
                "label": "Semiconductors",
                "sector": "tech",
                "values": [{"period": "M01", "value": "120"}],
            }
        }
        tool._persist_entities(data1)
        eid1 = store.store_entity_observation.call_args.kwargs["entity_id"]

        store.reset_mock()
        tool._persist_entities(data1)
        eid2 = store.store_entity_observation.call_args.kwargs["entity_id"]
        assert eid1 == eid2


# ── Edge cases ──


class TestEdgeCases:
    def test_persistence_exception_non_fatal(self):
        store = _make_mock_store()
        store.register_entity.side_effect = RuntimeError("DB fail")
        tool = SupplyChainMonitorTool(cache=None, pipeline_store=store)
        # Should not raise
        tool._persist_entities(_make_series_data())

    def test_non_dict_info_handled(self):
        """If info is a list (raw values), handle gracefully."""
        store = _make_mock_store()
        tool = SupplyChainMonitorTool(cache=None, pipeline_store=store)
        data = {
            "EIUIR": [
                {"period": "M01", "value": "100.0"},
                {"period": "M02", "value": "101.5"},
            ]
        }
        tool._persist_entities(data)
        assert store.register_entity.call_count == 1
        obs_call = store.store_entity_observation.call_args
        val = obs_call.kwargs["value"]
        assert val["latest_value"] == 101.5
        assert val["num_periods"] == 2

    def test_no_links_created(self):
        """Supply chain monitor should never create entity links."""
        store = _make_mock_store()
        tool = SupplyChainMonitorTool(cache=None, pipeline_store=store)
        tool._persist_entities(_make_series_data())
        store.link_entities.assert_not_called()

    def test_single_series(self):
        store = _make_mock_store()
        tool = SupplyChainMonitorTool(cache=None, pipeline_store=store)
        data = {
            "PCU324110324110": {
                "label": "Petroleum Refineries",
                "sector": "energy",
                "values": [{"period": "M06", "value": "350.2"}],
            }
        }
        tool._persist_entities(data)
        assert store.register_entity.call_count == 1
        assert store.store_entity_observation.call_count == 1
