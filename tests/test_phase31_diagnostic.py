"""Phase 31 integration diagnostics — remaining country signals."""

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


@pytest.fixture()
def store():
    s = PipelineStore(db_path=":memory:")
    yield s
    s.close()


def _reg_country(store: PipelineStore, cc: str) -> str:
    eid = entity_id_from_key("country", cc)
    store.register_entity("country", cc, eid, metadata={"country": cc})
    return eid


def _obs(store: PipelineStore, eid: str, tool: str, obs_type: str, value=None, ts=None):
    return store.store_entity_observation(
        entity_id=eid,
        source_tool=tool,
        observed_at=ts or time.time(),
        observation_type=obs_type,
        value=value or {},
    )


class TestPhase31Registry:
    def test_phase31_obs_types_registered(self):
        for obs_type in [
            "consumer_confidence",
            "food_security",
            "internet_disruption",
            "migration_pressure",
        ]:
            assert obs_type in OBSERVATION_TYPES

    def test_enrichment_dim(self):
        assert ENRICHMENT_DIM == 55

    def test_obs_type_count(self):
        assert len(OBSERVATION_TYPES) == 46


class TestPhase31CountryFlow:
    def test_country_receives_all_phase31_obs(self, store):
        eid = _reg_country(store, "DE")
        now = time.time()
        _obs(
            store,
            eid,
            "consumer_sentiment",
            "consumer_confidence",
            {"latest": -4.0},
            ts=now - 4,
        )
        _obs(
            store,
            eid,
            "food_security",
            "food_security",
            {"latest_value": 99.0},
            ts=now - 3,
        )
        _obs(
            store,
            eid,
            "internet_outages",
            "internet_disruption",
            {"anomaly_rate_pct": 12.0},
            ts=now - 2,
        )
        _obs(
            store,
            eid,
            "migration_flows",
            "migration_pressure",
            {"acceptance_rate": 40.0},
            ts=now - 1,
        )

        _, id_map, events = GraphBuilder(store).build()

        assert id_map.num_nodes_of_type("country") == 1
        assert {event["observation_type"] for event in events} == {
            "consumer_confidence",
            "food_security",
            "internet_disruption",
            "migration_pressure",
        }

    def test_country_coexists_with_prior_phase_obs(self, store):
        eid = _reg_country(store, "US")
        now = time.time()
        _obs(
            store,
            eid,
            "central_bank_balance",
            "cb_balance_sheet",
            {"usd_trillions": 7.0},
            ts=now - 5,
        )
        _obs(store, eid, "global_pmi", "economic_activity", {"value": 101.0}, ts=now - 4)
        _obs(
            store,
            eid,
            "consumer_sentiment",
            "consumer_confidence",
            {"latest": 72.0},
            ts=now - 3,
        )
        _obs(
            store,
            eid,
            "food_security",
            "food_security",
            {"latest_value": 103.0},
            ts=now - 2,
        )
        _obs(
            store,
            eid,
            "internet_outages",
            "internet_disruption",
            {"disconnect_rate_pct": 2.0},
            ts=now - 1,
        )

        _, _, events = GraphBuilder(store).build()

        assert {event["observation_type"] for event in events} == {
            "cb_balance_sheet",
            "economic_activity",
            "consumer_confidence",
            "food_security",
            "internet_disruption",
        }
