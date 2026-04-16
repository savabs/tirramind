"""Phase 29 integration diagnostics — company/topic investigative L2
persistence via bankruptcy_court, foia_requests, academic_preprints.

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


def _reg(store, etype, name):
    """Register an entity and return its entity id."""
    eid = entity_id_from_key(etype, name)
    store.register_entity(etype, name, eid)
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
    """Phase 29 obs types exist in graph builder constants."""

    def test_bankruptcy_status_in_obs_types(self):
        assert "bankruptcy_status" in OBSERVATION_TYPES

    def test_investigation_signal_in_obs_types(self):
        assert "investigation_signal" in OBSERVATION_TYPES

    def test_research_velocity_in_obs_types(self):
        assert "research_velocity" in OBSERVATION_TYPES

    def test_enrichment_dim_is_44(self):
        assert ENRICHMENT_DIM == 44

    def test_obs_types_count_is_35(self):
        assert len(OBSERVATION_TYPES) == 35


# ── 2. Single-tool persistence flows through to graph ────────────


class TestSingleToolPersistence:
    """Each tool persists obs that appear in graph builder events."""

    def test_bankruptcy_status_company(self, store):
        eid = _reg(store, "company", "Acme Corp")
        _obs(
            store,
            eid,
            "bankruptcy_court",
            "bankruptcy_status",
            {"source": "pacer", "chapter": "11", "court": "S.D.N.Y."},
        )

        builder = GraphBuilder(store)
        data, id_map, events = builder.build()

        assert id_map.num_nodes_of_type("company") == 1
        assert len(events) == 1
        assert events[0]["observation_type"] == "bankruptcy_status"

    def test_investigation_signal_company(self, store):
        eid = _reg(store, "company", "EPA Target LLC")
        _obs(
            store,
            eid,
            "foia_requests",
            "investigation_signal",
            {"source": "muckrock", "agency": "EPA"},
        )

        builder = GraphBuilder(store)
        _, id_map, events = builder.build()

        assert id_map.num_nodes_of_type("company") == 1
        assert events[0]["observation_type"] == "investigation_signal"

    def test_research_velocity_company_trial(self, store):
        eid = _reg(store, "company", "Pfizer Inc")
        _obs(
            store,
            eid,
            "academic_preprints",
            "research_velocity",
            {
                "source": "clinicaltrials",
                "nct_id": "NCT00001234",
                "status": "RECRUITING",
            },
        )

        builder = GraphBuilder(store)
        _, id_map, events = builder.build()

        assert id_map.num_nodes_of_type("company") == 1
        assert events[0]["observation_type"] == "research_velocity"

    def test_research_velocity_topic_paper(self, store):
        eid = _reg(store, "topic", "cs.AI")
        _obs(
            store,
            eid,
            "academic_preprints",
            "research_velocity",
            {"source": "arxiv", "paper_id": "2401.12345", "title": "Deep RL Trading"},
        )

        builder = GraphBuilder(store)
        _, id_map, events = builder.build()

        assert id_map.num_nodes_of_type("topic") == 1
        assert events[0]["observation_type"] == "research_velocity"


# ── 3. Multi-tool enrichment on same company node ────────────────


class TestMultiToolEnrichment:
    """Multiple tools enriching the same company entity node."""

    def test_company_receives_all_three_obs_types(self, store):
        eid = _reg(store, "company", "MegaCorp")
        now = time.time()
        _obs(
            store,
            eid,
            "bankruptcy_court",
            "bankruptcy_status",
            {"source": "sec_enforcement", "type": "admin_proc"},
            ts=now - 20,
        )
        _obs(
            store,
            eid,
            "foia_requests",
            "investigation_signal",
            {"source": "muckrock", "agency": "SEC"},
            ts=now - 10,
        )
        _obs(
            store,
            eid,
            "academic_preprints",
            "research_velocity",
            {"source": "clinicaltrials", "nct_id": "NCT99"},
            ts=now,
        )

        builder = GraphBuilder(store)
        data, id_map, events = builder.build()

        assert id_map.num_nodes_of_type("company") == 1
        assert len(events) == 3
        obs_types = {e["observation_type"] for e in events}
        assert obs_types == {
            "bankruptcy_status",
            "investigation_signal",
            "research_velocity",
        }

    def test_two_companies_each_with_obs(self, store):
        eid_a = _reg(store, "company", "Alpha Inc")
        eid_b = _reg(store, "company", "Beta LLC")
        _obs(store, eid_a, "bankruptcy_court", "bankruptcy_status", {"source": "pacer"})
        _obs(
            store,
            eid_b,
            "foia_requests",
            "investigation_signal",
            {"source": "muckrock"},
        )

        builder = GraphBuilder(store)
        _, id_map, events = builder.build()

        assert id_map.num_nodes_of_type("company") == 2
        assert len(events) == 2


# ── 4. Cross-entity-type integration ────────────────────────────


class TestCrossEntityTypeIntegration:
    """research_velocity on company + topic entities coexist."""

    def test_company_and_topic_both_get_research_velocity(self, store):
        eid_co = _reg(store, "company", "BioNTech")
        eid_tp = _reg(store, "topic", "q-fin")
        now = time.time()
        _obs(
            store,
            eid_co,
            "academic_preprints",
            "research_velocity",
            {"source": "clinicaltrials"},
            ts=now - 5,
        )
        _obs(
            store,
            eid_tp,
            "academic_preprints",
            "research_velocity",
            {"source": "arxiv"},
            ts=now,
        )

        builder = GraphBuilder(store)
        _, id_map, events = builder.build()

        assert id_map.num_nodes_of_type("company") == 1
        assert id_map.num_nodes_of_type("topic") == 1
        assert len(events) == 2
        sources = {e["value"].get("source") for e in events}
        assert sources == {"clinicaltrials", "arxiv"}


# ── 5. Coexistence with Phase 28 country obs ────────────────────


class TestCoexistenceWithPhase28:
    """Phase 29 company/topic types coexist with Phase 28 country types."""

    def test_company_and_country_obs_coexist(self, store):
        co_eid = _reg(store, "company", "Acme Corp")
        us_eid = _reg(store, "country", "US")
        now = time.time()
        _obs(
            store,
            co_eid,
            "bankruptcy_court",
            "bankruptcy_status",
            {"source": "pacer"},
            ts=now - 10,
        )
        _obs(
            store,
            us_eid,
            "sovereign_debt",
            "sovereign_yield",
            {"yield_pct": 4.44},
            ts=now,
        )

        builder = GraphBuilder(store)
        _, id_map, events = builder.build()

        assert id_map.num_nodes_of_type("company") == 1
        assert id_map.num_nodes_of_type("country") == 1
        assert len(events) == 2
        obs_types = {e["observation_type"] for e in events}
        assert obs_types == {"bankruptcy_status", "sovereign_yield"}


# ── 6. Entity ID determinism ────────────────────────────────────


class TestEntityIdDeterminism:
    """Entity IDs are deterministic for company and topic types."""

    def test_same_company_same_eid(self):
        eid1 = entity_id_from_key("company", "Pfizer")
        eid2 = entity_id_from_key("company", "Pfizer")
        assert eid1 == eid2

    def test_same_topic_same_eid(self):
        eid1 = entity_id_from_key("topic", "cs.AI")
        eid2 = entity_id_from_key("topic", "cs.AI")
        assert eid1 == eid2

    def test_different_companies_different_eids(self):
        a = entity_id_from_key("company", "Pfizer")
        b = entity_id_from_key("company", "Moderna")
        c = entity_id_from_key("company", "BioNTech")
        assert len({a, b, c}) == 3

    def test_company_vs_topic_different_eids(self):
        co = entity_id_from_key("company", "cs.AI")
        tp = entity_id_from_key("topic", "cs.AI")
        assert co != tp

    def test_eid_is_16_char_hex(self):
        eid = entity_id_from_key("company", "TestCo")
        assert len(eid) == 16
        int(eid, 16)  # valid hex
