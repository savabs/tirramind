"""Phase 28 integration diagnostics — country node enrichment via
sovereign_debt, capital_flows, global_pmi L2 persistence.

Tests use real PipelineStore (:memory:) + mock tool outputs to verify
the full path: tool._persist_entities → store → graph_builder.
"""

from __future__ import annotations

import time

import pytest

from agent.models.gnn.graph_builder import (
    ENRICHMENT_DIM,
    OBSERVATION_TYPES,
    GraphBuilder,
)
from agent.pipeline.entity import entity_id_from_key
from agent.pipeline.store import PipelineStore


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture()
def store():
    s = PipelineStore(db_path=":memory:")
    yield s
    s.close()


def _reg(store, cc):
    """Register a country entity and return its entity id."""
    eid = entity_id_from_key("country", cc)
    store.register_entity("country", cc, eid, metadata={"country": cc})
    return eid


def _obs(store, eid, tool, obs_type, value=None, ts=None):
    return store.store_entity_observation(
        entity_id=eid,
        source_tool=tool,
        observed_at=ts or time.time(),
        observation_type=obs_type,
        value=value or {},
    )


# ── 1. New obs types registered in OBSERVATION_TYPES ─────────────


class TestObsTypeRegistration:
    """Phase 28 obs types exist in graph builder constants."""

    def test_capital_flow_in_obs_types(self):
        assert "capital_flow" in OBSERVATION_TYPES

    def test_economic_activity_in_obs_types(self):
        assert "economic_activity" in OBSERVATION_TYPES

    def test_sovereign_yield_in_obs_types(self):
        assert "sovereign_yield" in OBSERVATION_TYPES

    def test_enrichment_dim_is_44(self):
        assert ENRICHMENT_DIM == 44


# ── 2. Single-tool persistence flows through to graph ────────────


class TestSingleToolPersistence:
    """Each tool persists obs that appear in graph builder events."""

    def test_sovereign_yield_us(self, store):
        eid = _reg(store, "US")
        _obs(
            store,
            eid,
            "sovereign_debt",
            "sovereign_yield",
            {"source": "us_treasury", "maturity": "10y", "yield_pct": 4.44},
        )

        builder = GraphBuilder(store)
        data, id_map, events = builder.build()

        assert id_map.num_nodes_of_type("country") == 1
        assert len(events) == 1
        assert events[0]["observation_type"] == "sovereign_yield"

    def test_capital_flow_jp(self, store):
        eid = _reg(store, "JP")
        _obs(
            store,
            eid,
            "capital_flows",
            "capital_flow",
            {"flow_type": "holdings", "series": "Japan", "latest_value": 1100.0},
        )

        builder = GraphBuilder(store)
        _, id_map, events = builder.build()

        assert id_map.num_nodes_of_type("country") == 1
        assert events[0]["observation_type"] == "capital_flow"

    def test_economic_activity_de(self, store):
        eid = _reg(store, "DE")
        _obs(
            store,
            eid,
            "global_pmi",
            "economic_activity",
            {"indicator": "cli", "value": 99.0, "regime": "contracting"},
        )

        builder = GraphBuilder(store)
        _, id_map, events = builder.build()

        assert id_map.num_nodes_of_type("country") == 1
        assert events[0]["observation_type"] == "economic_activity"


# ── 3. Multi-tool enrichment on same country node ────────────────


class TestMultiToolEnrichment:
    """Multiple tools enriching the same country node."""

    def test_us_receives_all_three_obs_types(self, store):
        eid = _reg(store, "US")
        now = time.time()
        _obs(
            store,
            eid,
            "sovereign_debt",
            "sovereign_yield",
            {"source": "us_treasury", "yield_pct": 4.44},
            ts=now - 10,
        )
        _obs(
            store,
            eid,
            "capital_flows",
            "capital_flow",
            {"flow_type": "flows", "series": "net_tic"},
            ts=now - 5,
        )
        _obs(
            store,
            eid,
            "global_pmi",
            "economic_activity",
            {"indicator": "cli", "value": 101.0},
            ts=now,
        )

        builder = GraphBuilder(store)
        data, id_map, events = builder.build()

        assert id_map.num_nodes_of_type("country") == 1
        assert len(events) == 3
        obs_types = {e["observation_type"] for e in events}
        assert obs_types == {"sovereign_yield", "capital_flow", "economic_activity"}

    def test_jp_receives_sovereign_and_flows(self, store):
        eid = _reg(store, "JP")
        _obs(
            store,
            eid,
            "sovereign_debt",
            "sovereign_yield",
            {"source": "mof", "yield_pct": 2.30},
        )
        _obs(
            store,
            eid,
            "capital_flows",
            "capital_flow",
            {"flow_type": "holdings", "latest_value": 1100.0},
        )

        builder = GraphBuilder(store)
        _, _, events = builder.build()

        assert len(events) == 2

    def test_mixed_countries_all_enriched(self, store):
        us_eid = _reg(store, "US")
        de_eid = _reg(store, "DE")
        jp_eid = _reg(store, "JP")

        _obs(store, us_eid, "sovereign_debt", "sovereign_yield", {"yield_pct": 4.44})
        _obs(store, de_eid, "global_pmi", "economic_activity", {"value": 99.0})
        _obs(store, jp_eid, "capital_flows", "capital_flow", {"latest_value": 1100.0})

        builder = GraphBuilder(store)
        data, id_map, events = builder.build()

        assert id_map.num_nodes_of_type("country") == 3
        assert len(events) == 3


# ── 4. Coexistence with Phase 27 cb obs types ───────────────────


class TestCoexistenceWithPhase27:
    """Phase 28 types coexist with Phase 27 cb_balance_sheet/cb_policy_rate."""

    def test_us_has_cb_and_sovereign_and_flow(self, store):
        eid = _reg(store, "US")
        now = time.time()
        _obs(
            store,
            eid,
            "central_bank_balance",
            "cb_balance_sheet",
            {"cb_code": "fed", "usd_trillions": 7.5},
            ts=now - 30,
        )
        _obs(
            store,
            eid,
            "central_bank_balance",
            "cb_policy_rate",
            {"cb_code": "fed", "current_rate": 5.33},
            ts=now - 20,
        )
        _obs(
            store,
            eid,
            "sovereign_debt",
            "sovereign_yield",
            {"yield_pct": 4.44},
            ts=now - 10,
        )
        _obs(
            store,
            eid,
            "capital_flows",
            "capital_flow",
            {"flow_type": "flows"},
            ts=now - 5,
        )
        _obs(store, eid, "global_pmi", "economic_activity", {"value": 101.0}, ts=now)

        builder = GraphBuilder(store)
        _, _, events = builder.build()

        assert len(events) == 5
        obs_types = {e["observation_type"] for e in events}
        assert obs_types == {
            "cb_balance_sheet",
            "cb_policy_rate",
            "sovereign_yield",
            "capital_flow",
            "economic_activity",
        }


# ── 5. Entity ID determinism ────────────────────────────────────


class TestEntityIdDeterminism:
    """Country entity IDs are deterministic and consistent across tools."""

    def test_same_country_same_eid(self):
        eid1 = entity_id_from_key("country", "US")
        eid2 = entity_id_from_key("country", "US")
        assert eid1 == eid2

    def test_different_countries_different_eids(self):
        us = entity_id_from_key("country", "US")
        jp = entity_id_from_key("country", "JP")
        de = entity_id_from_key("country", "DE")
        assert len({us, jp, de}) == 3

    def test_eid_is_16_char_hex(self):
        eid = entity_id_from_key("country", "US")
        assert len(eid) == 16
        int(eid, 16)  # valid hex
