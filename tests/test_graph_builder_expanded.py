"""Tests for Phase 13a: Graph builder expansion — new entity/observation types.

Covers:
    - New ENTITY_TYPES (domain, protocol, topic) produce correct one-hot encoding
    - New OBSERVATION_TYPES encode correctly in events
    - Unknown entity types handled gracefully (warning, default index)
    - Unknown observation types don't crash graph build
    - insider_trade (not "purchase") encodes correctly
    - Dynamic type iteration: types in store but not in ENTITY_TYPES get node features
"""

from __future__ import annotations

import logging
import time

import pytest
import torch

from agent.pipeline.entity import entity_id_from_key
from agent.pipeline.store import PipelineStore
from agent.models.gnn.graph_builder import (
    ENTITY_TYPES,
    OBSERVATION_TYPES,
    GraphBuilder,
    IDMap,
    _ENTITY_TYPE_TO_IDX,
    _OBS_TYPE_TO_IDX,
    _build_node_features,
)


# ── Fixtures ───────────────────────────────────────────────────


@pytest.fixture()
def store() -> PipelineStore:
    s = PipelineStore(db_path=":memory:")
    yield s
    s.close()


def _reg(store: PipelineStore, etype: str, key: str, name: str) -> str:
    eid = entity_id_from_key(etype, key)
    store.register_entity(etype, name, eid, metadata={etype: key})
    return eid


def _obs(store, entity_id, tool, obs_type, ts, value=None):
    return store.store_entity_observation(
        entity_id=entity_id,
        source_tool=tool,
        observed_at=ts,
        observation_type=obs_type,
        value=value or {},
    )


# ── Type Registry Tests ───────────────────────────────────────


class TestExpandedTypeRegistries:
    """Verify the expanded ENTITY_TYPES and OBSERVATION_TYPES lists."""

    def test_entity_types_contains_new_types(self):
        assert "domain" in ENTITY_TYPES
        assert "instrument" in ENTITY_TYPES
        assert "protocol" in ENTITY_TYPES
        assert "topic" in ENTITY_TYPES

    def test_entity_types_sorted_alphabetically(self):
        assert ENTITY_TYPES == sorted(ENTITY_TYPES)

    def test_entity_types_count(self):
        assert len(ENTITY_TYPES) == 11

    def test_observation_types_contains_new_types(self):
        new_obs = [
            "cert_issued",
            "dns_change",
            "lobbying_spend",
            "pageview_spike",
            "patent_filing",
            "project_status",
            "tvl_change",
        ]
        for obs in new_obs:
            assert obs in OBSERVATION_TYPES, f"{obs} missing from OBSERVATION_TYPES"

    def test_observation_types_contains_phase28_types(self):
        phase28_obs = ["capital_flow", "economic_activity", "sovereign_yield"]
        for obs in phase28_obs:
            assert obs in OBSERVATION_TYPES, f"{obs} missing from OBSERVATION_TYPES"

    def test_observation_types_contains_phase29_types(self):
        phase29_obs = ["bankruptcy_status", "investigation_signal", "research_velocity"]
        for obs in phase29_obs:
            assert obs in OBSERVATION_TYPES, f"{obs} missing from OBSERVATION_TYPES"

    def test_observation_types_contains_phase31_types(self):
        phase31_obs = [
            "consumer_confidence",
            "food_security",
            "internet_disruption",
            "migration_pressure",
        ]
        for obs in phase31_obs:
            assert obs in OBSERVATION_TYPES, f"{obs} missing from OBSERVATION_TYPES"

    def test_observation_types_sorted_alphabetically(self):
        assert OBSERVATION_TYPES == sorted(OBSERVATION_TYPES)

    def test_observation_types_count(self):
        assert len(OBSERVATION_TYPES) == 46

    def test_insider_trade_in_observation_types(self):
        """Verify 'insider_trade' exists (not 'purchase')."""
        assert "insider_trade" in OBSERVATION_TYPES
        assert "purchase" not in OBSERVATION_TYPES

    def test_entity_type_to_idx_has_all_types(self):
        for etype in ENTITY_TYPES:
            assert etype in _ENTITY_TYPE_TO_IDX

    def test_obs_type_to_idx_has_all_types(self):
        for otype in OBSERVATION_TYPES:
            assert otype in _OBS_TYPE_TO_IDX


# ── One-Hot Encoding Tests ─────────────────────────────────────


class TestNewTypeOneHotEncoding:
    """Verify new entity types produce correct one-hot positions."""

    def test_domain_one_hot_position(self):
        idx = _ENTITY_TYPE_TO_IDX["domain"]
        features = _build_node_features("domain", ["d1"], [], 0.0)
        assert features.shape == (1, len(ENTITY_TYPES) + 3)
        assert features[0, idx] == 1.0
        # All other type positions should be 0
        for i in range(len(ENTITY_TYPES)):
            if i != idx:
                assert features[0, i] == 0.0

    def test_protocol_one_hot_position(self):
        idx = _ENTITY_TYPE_TO_IDX["protocol"]
        features = _build_node_features("protocol", ["p1"], [], 0.0)
        assert features[0, idx] == 1.0

    def test_topic_one_hot_position(self):
        idx = _ENTITY_TYPE_TO_IDX["topic"]
        features = _build_node_features("topic", ["t1"], [], 0.0)
        assert features[0, idx] == 1.0

    def test_company_one_hot_still_correct(self):
        """Regression: company encoding unchanged after expansion."""
        idx = _ENTITY_TYPE_TO_IDX["company"]
        features = _build_node_features("company", ["c1"], [], 0.0)
        assert features[0, idx] == 1.0

    def test_feature_dim_is_10_plus_3(self):
        """10 entity types + 3 observation stats = 13 features."""
        features = _build_node_features("company", ["c1"], [], 0.0)
        assert features.shape[1] == 14


# ── Unknown Type Fallback Tests ────────────────────────────────


class TestUnknownTypeFallback:
    """Verify unknown entity types are handled gracefully."""

    def test_unknown_entity_type_defaults_to_index_0(self, caplog):
        with caplog.at_level(logging.WARNING):
            features = _build_node_features("alien_type", ["a1"], [], 0.0)
        assert features.shape == (1, len(ENTITY_TYPES) + 3)
        # Index 0 should be set
        assert features[0, 0] == 1.0
        # Warning logged
        assert "Unknown entity type" in caplog.text

    def test_unknown_type_in_store_still_gets_nodes(self, store):
        """Entities with types not in ENTITY_TYPES should still appear as nodes."""
        eid = entity_id_from_key("alien_type", "x1")
        store.register_entity("alien_type", "Alien Entity", eid)
        _obs(store, eid, "test_tool", "alien_obs", time.time(), {"val": 1})

        builder = GraphBuilder(store)
        data, id_map, events = builder.build()

        # alien_type should have a node
        assert id_map.num_nodes_of_type("alien_type") == 1
        assert hasattr(data["alien_type"], "x")
        assert data["alien_type"].x.shape[0] == 1

    def test_unknown_observation_type_in_events(self, store):
        """Unknown obs types should still appear in events list."""
        eid = _reg(store, "company", "C1", "TestCo")
        _obs(store, eid, "test_tool", "completely_new_obs", time.time(), {"v": 1})

        builder = GraphBuilder(store)
        _, _, events = builder.build()
        assert len(events) == 1
        assert events[0]["observation_type"] == "completely_new_obs"


# ── Full Build with New Types ──────────────────────────────────


class TestGraphBuildWithNewTypes:
    """End-to-end graph build with new entity and observation types."""

    def test_build_with_domain_entities(self, store):
        d1 = _reg(store, "domain", "example.com", "example.com")
        d2 = _reg(store, "domain", "test.io", "test.io")
        _obs(store, d1, "cert_transparency", "cert_issued", 1000.0, {"issuer": "LE"})
        _obs(store, d2, "dns_monitor", "dns_change", 1001.0, {"ttl": 300})

        builder = GraphBuilder(store)
        data, id_map, events = builder.build()

        assert id_map.num_nodes_of_type("domain") == 2
        assert data["domain"].x.shape == (2, 14)
        assert len(events) == 2

    def test_build_with_protocol_entities(self, store):
        p1 = _reg(store, "protocol", "aave", "Aave")
        _obs(store, p1, "defi_flows", "tvl_change", 2000.0, {"tvl_usd": 5e9})

        builder = GraphBuilder(store)
        data, id_map, events = builder.build()

        assert id_map.num_nodes_of_type("protocol") == 1
        assert data["protocol"].x.shape == (1, 14)

    def test_build_with_topic_entities(self, store):
        t1 = _reg(store, "topic", "Tesla,_Inc.", "Tesla")
        _obs(
            store,
            t1,
            "wikipedia_pageviews",
            "pageview_spike",
            3000.0,
            {"z_score": 4.5, "latest_views": 50000},
        )

        builder = GraphBuilder(store)
        data, id_map, events = builder.build()

        assert id_map.num_nodes_of_type("topic") == 1
        assert data["topic"].x.shape == (1, 14)

    def test_mixed_old_and_new_types(self, store):
        """Build graph with both old (company, person) and new (domain, protocol) types."""
        c1 = _reg(store, "company", "AAPL_CIK", "Apple")
        d1 = _reg(store, "domain", "apple.com", "apple.com")
        p1 = _reg(store, "protocol", "uniswap", "Uniswap")
        per1 = _reg(store, "person", "JOE_CIK", "Joe Smith")

        now = time.time()
        _obs(store, c1, "insider_filings", "insider_trade", now - 100)
        _obs(store, d1, "cert_transparency", "cert_issued", now - 50)
        _obs(store, p1, "defi_flows", "tvl_change", now - 25, {"tvl_usd": 1e9})
        _obs(store, per1, "patent_filings", "patent_filing", now - 10)

        builder = GraphBuilder(store)
        data, id_map, events = builder.build()

        assert id_map.num_nodes_of_type("company") == 1
        assert id_map.num_nodes_of_type("domain") == 1
        assert id_map.num_nodes_of_type("protocol") == 1
        assert id_map.num_nodes_of_type("person") == 1
        assert len(events) == 4

    def test_all_new_observation_types_appear_in_events(self, store):
        """Each new obs type should flow through to the events list."""
        new_obs_types = [
            "cert_issued",
            "dns_change",
            "lobbying_spend",
            "pageview_spike",
            "patent_filing",
            "project_status",
            "tvl_change",
        ]
        eid = _reg(store, "company", "C1", "TestCo")
        now = time.time()
        for i, ot in enumerate(new_obs_types):
            _obs(store, eid, "test", ot, now + i)

        builder = GraphBuilder(store)
        _, _, events = builder.build()

        event_types = {e["observation_type"] for e in events}
        for ot in new_obs_types:
            assert ot in event_types, f"{ot} not in events"

    def test_insider_trade_obs_type_encodes(self, store):
        """Verify 'insider_trade' (not 'purchase') encodes in events."""
        eid = _reg(store, "person", "INSIDER1", "Jane Doe")
        _obs(store, eid, "insider_filings", "insider_trade", time.time())

        builder = GraphBuilder(store)
        _, _, events = builder.build()
        assert events[0]["observation_type"] == "insider_trade"
