"""Tests for Phase 12a: PipelineStore bulk queries + GraphBuilder.

Covers:
    - query_all_entities / query_all_observations / query_all_entity_links
    - IDMap operations
    - GraphBuilder.build() — happy path and edge cases
    - Node feature shapes, edge_index integrity, event ordering
"""

from __future__ import annotations

import time

import pytest

from agent.models.gnn.graph_builder import (
    ENTITY_TYPES,
    GraphBuilder,
    IDMap,
    _build_edge_data,
    _build_node_features,
    _compute_obs_stats,
)
from agent.pipeline.entity import entity_id_from_key
from agent.pipeline.store import PipelineStore

# ── Fixtures ───────────────────────────────────────────────────


@pytest.fixture()
def store() -> PipelineStore:
    s = PipelineStore(db_path=":memory:")
    yield s
    s.close()


def _reg(store: PipelineStore, etype: str, key: str, name: str) -> str:
    """Register entity and return its entity_id."""
    eid = entity_id_from_key(etype, key)
    store.register_entity(etype, name, eid, metadata={etype: key})
    return eid


def _obs(
    store: PipelineStore,
    entity_id: str,
    tool: str,
    obs_type: str,
    ts: float,
    value: dict | None = None,
) -> int:
    return store.store_entity_observation(
        entity_id=entity_id,
        source_tool=tool,
        observed_at=ts,
        observation_type=obs_type,
        value=value or {},
    )


def _link(
    store: PipelineStore,
    eid_a: str,
    eid_b: str,
    ltype: str,
    conf: float = 1.0,
) -> int | None:
    return store.link_entities(eid_a, eid_b, ltype, source="test", confidence=conf)


def _seed_graph(store: PipelineStore) -> dict:
    """Seed a small but complete graph for testing.

    Graph:
        company:Exxon --headquartered_in--> country:US
        vessel:Tanker1 --port_call_to--> country:RU
        wallet:w1       --exchange_based_in--> country:US

    Observations:
        Exxon: insider_trade at t=1000, t=2000
        US: geopolitical_event at t=1500
        Tanker1: port_call at t=1800
        w1: btc_transfer at t=900
    """
    exxon = _reg(store, "company", "EXX", "Exxon")
    us = _reg(store, "country", "US", "United States")
    ru = _reg(store, "country", "RU", "Russia")
    tanker = _reg(store, "vessel", "IMO001", "Tanker1")
    w1 = _reg(store, "wallet", "bc1qtest", "Wallet1")

    _link(store, exxon, us, "headquartered_in")
    _link(store, tanker, ru, "port_call_to")
    _link(store, w1, us, "exchange_based_in")

    _obs(
        store,
        exxon,
        "insider_filings",
        "insider_trade",
        1000.0,
        {"value": 500000, "direction": "sell"},
    )
    _obs(
        store,
        exxon,
        "insider_filings",
        "insider_trade",
        2000.0,
        {"value": 200000, "direction": "buy"},
    )
    _obs(
        store,
        us,
        "gdelt",
        "geopolitical_event",
        1500.0,
        {"goldstein_scale": -5.0, "num_articles": 42},
    )
    _obs(store, tanker, "ais_vessel", "port_call", 1800.0, {"port_name": "Novorossiysk"})
    _obs(
        store,
        w1,
        "whale_alert",
        "btc_transfer",
        900.0,
        {"btc_amount": 100.0, "usd_amount": 5000000.0},
    )

    return {
        "exxon": exxon,
        "us": us,
        "ru": ru,
        "tanker": tanker,
        "w1": w1,
    }


# ═══════════════════════════════════════════════════════════════
# PipelineStore bulk query tests
# ═══════════════════════════════════════════════════════════════


class TestQueryAllEntities:
    def test_empty_store(self, store: PipelineStore):
        assert store.query_all_entities() == []

    def test_returns_all(self, store: PipelineStore):
        _seed_graph(store)
        entities = store.query_all_entities()
        assert len(entities) == 5
        types = {e["entity_type"] for e in entities}
        assert types == {"company", "country", "vessel", "wallet"}

    def test_filter_by_type(self, store: PipelineStore):
        _seed_graph(store)
        countries = store.query_all_entities(entity_type="country")
        assert len(countries) == 2
        assert all(e["entity_type"] == "country" for e in countries)

    def test_filter_nonexistent_type(self, store: PipelineStore):
        _seed_graph(store)
        assert store.query_all_entities(entity_type="spaceship") == []

    def test_metadata_deserialized(self, store: PipelineStore):
        _seed_graph(store)
        entities = store.query_all_entities(entity_type="company")
        assert len(entities) == 1
        assert entities[0]["metadata"] is not None
        assert "company" in entities[0]["metadata"]


class TestQueryAllObservations:
    def test_empty_store(self, store: PipelineStore):
        assert store.query_all_observations() == []

    def test_returns_all_sorted(self, store: PipelineStore):
        _seed_graph(store)
        obs = store.query_all_observations()
        assert len(obs) == 5
        timestamps = [o["observed_at"] for o in obs]
        assert timestamps == sorted(timestamps)

    def test_since_filter(self, store: PipelineStore):
        _seed_graph(store)
        obs = store.query_all_observations(since=1500.0)
        assert len(obs) == 3  # t=1500, t=1800, t=2000
        assert all(o["observed_at"] >= 1500.0 for o in obs)

    def test_until_filter(self, store: PipelineStore):
        _seed_graph(store)
        obs = store.query_all_observations(until=1000.0)
        assert len(obs) == 2  # t=900, t=1000
        assert all(o["observed_at"] <= 1000.0 for o in obs)

    def test_since_and_until(self, store: PipelineStore):
        _seed_graph(store)
        obs = store.query_all_observations(since=1000.0, until=1800.0)
        assert len(obs) == 3  # t=1000, t=1500, t=1800 (not 900 or 2000)
        assert all(1000.0 <= o["observed_at"] <= 1800.0 for o in obs)

    def test_value_deserialized(self, store: PipelineStore):
        _seed_graph(store)
        obs = store.query_all_observations(since=900.0, until=900.0)
        assert len(obs) == 1
        assert obs[0]["value"]["btc_amount"] == 100.0


class TestQueryAllEntityLinks:
    def test_empty_store(self, store: PipelineStore):
        assert store.query_all_entity_links() == []

    def test_returns_all(self, store: PipelineStore):
        _seed_graph(store)
        links = store.query_all_entity_links()
        assert len(links) == 3

    def test_filter_by_type(self, store: PipelineStore):
        _seed_graph(store)
        links = store.query_all_entity_links(link_type="headquartered_in")
        assert len(links) == 1
        assert links[0]["link_type"] == "headquartered_in"

    def test_min_confidence_filter(self, store: PipelineStore):
        ids = _seed_graph(store)
        # Add a low-confidence link
        _link(store, ids["exxon"], ids["ru"], "operates_in", conf=0.3)
        all_links = store.query_all_entity_links()
        assert len(all_links) == 4
        hi_conf = store.query_all_entity_links(min_confidence=0.5)
        assert len(hi_conf) == 3

    def test_metadata_deserialized(self, store: PipelineStore):
        _seed_graph(store)
        links = store.query_all_entity_links()
        # Links from _seed_graph have no metadata (None)
        assert all(link["metadata"] is None for link in links)


# ═══════════════════════════════════════════════════════════════
# IDMap tests
# ═══════════════════════════════════════════════════════════════


class TestIDMap:
    def test_add_and_lookup(self):
        m = IDMap()
        gid = m.add("company", "abc")
        assert gid == 0
        assert m.global_id("company", "abc") == 0
        assert m.local_id("company", "abc") == 0

    def test_idempotent_add(self):
        m = IDMap()
        gid1 = m.add("company", "abc")
        gid2 = m.add("company", "abc")
        assert gid1 == gid2
        assert m.num_nodes == 1

    def test_different_types_separate(self):
        m = IDMap()
        g1 = m.add("company", "abc")
        g2 = m.add("country", "abc")
        assert g1 != g2
        assert m.local_id("company", "abc") == 0
        assert m.local_id("country", "abc") == 0  # Each type starts at 0

    def test_num_nodes(self):
        m = IDMap()
        m.add("company", "a")
        m.add("company", "b")
        m.add("country", "us")
        assert m.num_nodes == 3
        assert m.num_nodes_of_type("company") == 2
        assert m.num_nodes_of_type("country") == 1
        assert m.num_nodes_of_type("vessel") == 0

    def test_lookup_missing(self):
        m = IDMap()
        assert m.global_id("company", "missing") is None
        assert m.local_id("company", "missing") is None

    def test_global_to_typed_roundtrip(self):
        m = IDMap()
        m.add("vessel", "imo1")
        m.add("wallet", "bc1q")
        assert m.global_to_typed[0] == ("vessel", "imo1")
        assert m.global_to_typed[1] == ("wallet", "bc1q")


# ═══════════════════════════════════════════════════════════════
# _compute_obs_stats tests
# ═══════════════════════════════════════════════════════════════


class TestComputeObsStats:
    def test_no_observations(self):
        stats = _compute_obs_stats([], "eid1", 5000.0)
        assert stats["count"] == 0.0
        assert stats["recency"] == 0.0
        assert stats["mean_value"] == 0.0

    def test_with_observations(self):
        obs = [
            {"entity_id": "e1", "observed_at": 1000.0, "value": {"value": 100}},
            {"entity_id": "e1", "observed_at": 2000.0, "value": {"value": 300}},
            {"entity_id": "e2", "observed_at": 1500.0, "value": {"value": 999}},
        ]
        stats = _compute_obs_stats(obs, "e1", 3000.0)
        assert stats["count"] == 2.0
        assert stats["recency"] == 1000.0  # 3000 - 2000
        assert stats["mean_value"] == 200.0  # (100+300)/2

    def test_filters_by_entity_id(self):
        obs = [
            {"entity_id": "e1", "observed_at": 100.0, "value": {}},
            {"entity_id": "e2", "observed_at": 200.0, "value": {}},
        ]
        stats = _compute_obs_stats(obs, "e1", 500.0)
        assert stats["count"] == 1.0

    def test_goldstein_value_extraction(self):
        obs = [
            {
                "entity_id": "e1",
                "observed_at": 100.0,
                "value": {"goldstein_scale": -5.0, "num_articles": 10},
            },
        ]
        stats = _compute_obs_stats(obs, "e1", 100.0)
        assert stats["mean_value"] == -5.0  # goldstein_scale is first match

    def test_btc_amount_extraction(self):
        obs = [
            {
                "entity_id": "e1",
                "observed_at": 100.0,
                "value": {"btc_amount": 42.5, "usd_amount": 2000000},
            },
        ]
        stats = _compute_obs_stats(obs, "e1", 100.0)
        # usd_amount comes first in the priority list
        assert stats["mean_value"] == 2000000.0

    def test_non_numeric_value_ignored(self):
        obs = [
            {"entity_id": "e1", "observed_at": 100.0, "value": {"direction": "sell"}},
        ]
        stats = _compute_obs_stats(obs, "e1", 100.0)
        assert stats["mean_value"] == 0.0


# ═══════════════════════════════════════════════════════════════
# _build_node_features tests
# ═══════════════════════════════════════════════════════════════


class TestBuildNodeFeatures:
    def test_empty_entities(self):
        ft = _build_node_features("company", [], [], 1000.0)
        assert ft.shape == (0, len(ENTITY_TYPES) + 3)

    def test_correct_shape(self):
        ft = _build_node_features("company", ["e1", "e2"], [], 1000.0)
        assert ft.shape == (2, len(ENTITY_TYPES) + 3)

    def test_one_hot_encoding(self):
        ft = _build_node_features("country", ["e1"], [], 1000.0)
        type_idx = ENTITY_TYPES.index("country")
        assert ft[0, type_idx].item() == 1.0
        # All other type positions should be 0
        for i in range(len(ENTITY_TYPES)):
            if i != type_idx:
                assert ft[0, i].item() == 0.0

    def test_obs_stats_populated(self):
        obs = [
            {"entity_id": "e1", "observed_at": 500.0, "value": {"value": 100.0}},
        ]
        ft = _build_node_features("company", ["e1"], obs, 1000.0)
        type_dim = len(ENTITY_TYPES)
        assert ft[0, type_dim].item() == 1.0  # count
        assert ft[0, type_dim + 1].item() == 500.0  # recency = 1000 - 500
        assert ft[0, type_dim + 2].item() == 100.0  # mean_value


# ═══════════════════════════════════════════════════════════════
# _build_edge_data tests
# ═══════════════════════════════════════════════════════════════


class TestBuildEdgeData:
    def test_empty_links(self):
        m = IDMap()
        result = _build_edge_data([], m)
        assert result == {}

    def test_single_link(self):
        m = IDMap()
        m.add("company", "c1")
        m.add("country", "us")
        links = [
            {
                "entity_id_a": "c1",
                "entity_id_b": "us",
                "link_type": "headquartered_in",
                "confidence": 0.9,
                "created_at": time.time() - 86400,  # 1 day ago
            }
        ]
        result = _build_edge_data(links, m)
        assert ("company", "headquartered_in", "country") in result
        triplet = result[("company", "headquartered_in", "country")]
        assert triplet["edge_index"].shape == (2, 1)
        assert triplet["edge_attr"].shape == (1, 2)
        assert triplet["edge_attr"][0, 0].item() == pytest.approx(0.9)
        assert triplet["edge_attr"][0, 1].item() == pytest.approx(1.0, abs=0.1)

    def test_missing_entity_skipped(self):
        m = IDMap()
        m.add("company", "c1")
        # "us" not registered
        links = [
            {
                "entity_id_a": "c1",
                "entity_id_b": "us",
                "link_type": "headquartered_in",
                "confidence": 1.0,
                "created_at": time.time(),
            }
        ]
        result = _build_edge_data(links, m)
        assert result == {}

    def test_multiple_edge_types(self):
        m = IDMap()
        m.add("company", "c1")
        m.add("country", "us")
        m.add("vessel", "v1")
        m.add("country", "ru")
        links = [
            {
                "entity_id_a": "c1",
                "entity_id_b": "us",
                "link_type": "headquartered_in",
                "confidence": 1.0,
                "created_at": time.time(),
            },
            {
                "entity_id_a": "v1",
                "entity_id_b": "ru",
                "link_type": "port_call_to",
                "confidence": 0.8,
                "created_at": time.time(),
            },
        ]
        result = _build_edge_data(links, m)
        assert len(result) == 2
        assert ("company", "headquartered_in", "country") in result
        assert ("vessel", "port_call_to", "country") in result


# ═══════════════════════════════════════════════════════════════
# GraphBuilder integration tests
# ═══════════════════════════════════════════════════════════════


class TestGraphBuilder:
    def test_empty_store(self, store: PipelineStore):
        builder = GraphBuilder(store)
        data, id_map, events = builder.build()
        assert id_map.num_nodes == 0
        assert len(events) == 0
        # HeteroData should be valid but empty
        assert len(data.node_types) == 0

    def test_full_build(self, store: PipelineStore):
        _seed_graph(store)
        builder = GraphBuilder(store)
        data, id_map, events = builder.build()

        # ID map
        assert id_map.num_nodes == 5  # exxon, us, ru, tanker, w1
        assert id_map.num_nodes_of_type("company") == 1
        assert id_map.num_nodes_of_type("country") == 2
        assert id_map.num_nodes_of_type("vessel") == 1
        assert id_map.num_nodes_of_type("wallet") == 1

        # Node features
        assert "company" in data.node_types
        assert "country" in data.node_types
        assert data["company"].x.shape[0] == 1
        assert data["country"].x.shape[0] == 2
        feat_dim = len(ENTITY_TYPES) + 3
        assert data["company"].x.shape[1] == feat_dim
        assert data["country"].x.shape[1] == feat_dim

        # Edge types
        assert len(data.edge_types) == 3  # hq_in, port_call_to, exchange_based_in

        # Events sorted by time
        assert len(events) == 5
        assert events[0]["observed_at"] <= events[-1]["observed_at"]

    def test_since_filter(self, store: PipelineStore):
        _seed_graph(store)
        builder = GraphBuilder(store)
        data, id_map, events = builder.build(since=1500.0)

        # All entities still present (since only filters observations)
        assert id_map.num_nodes == 5
        # Only 3 observations: t=1500, t=1800, t=2000
        assert len(events) == 3

    def test_until_filter(self, store: PipelineStore):
        _seed_graph(store)
        builder = GraphBuilder(store)
        data, id_map, events = builder.build(until=1000.0)
        assert len(events) == 2

    def test_node_ids_attribute(self, store: PipelineStore):
        _seed_graph(store)
        builder = GraphBuilder(store)
        data, id_map, events = builder.build()

        # Every node type should have node_ids list
        for ntype in data.node_types:
            assert hasattr(data[ntype], "node_ids")
            assert len(data[ntype].node_ids) == data[ntype].x.shape[0]

    def test_edge_index_valid_range(self, store: PipelineStore):
        _seed_graph(store)
        builder = GraphBuilder(store)
        data, id_map, events = builder.build()

        for etype in data.edge_types:
            src_type, _, dst_type = etype
            ei = data[etype].edge_index
            assert ei.min().item() >= 0
            assert ei[0].max().item() < data[src_type].x.shape[0]
            assert ei[1].max().item() < data[dst_type].x.shape[0]

    def test_single_entity_no_links(self, store: PipelineStore):
        """An entity with no links produces a graph with no edges."""
        _reg(store, "company", "solo", "Solo Inc")
        builder = GraphBuilder(store)
        data, id_map, events = builder.build()
        assert id_map.num_nodes == 1
        assert "company" in data.node_types
        assert len(data.edge_types) == 0
        assert len(events) == 0

    def test_orphan_entity_no_observations(self, store: PipelineStore):
        """Entity with no observations gets zero obs stats."""
        _reg(store, "vessel", "IMO999", "EmptyVessel")
        builder = GraphBuilder(store)
        data, id_map, events = builder.build()

        feat_dim = len(ENTITY_TYPES) + 3
        assert data["vessel"].x.shape == (1, feat_dim)
        # obs_count, recency, mean_value all zero
        type_dim = len(ENTITY_TYPES)
        assert data["vessel"].x[0, type_dim].item() == 0.0
        assert data["vessel"].x[0, type_dim + 1].item() == 0.0
        assert data["vessel"].x[0, type_dim + 2].item() == 0.0

    def test_duplicate_links_same_count(self, store: PipelineStore):
        """link_entities uses INSERT OR IGNORE, so duplicates stay at 1."""
        ids = _seed_graph(store)
        # Try to add duplicate
        _link(store, ids["exxon"], ids["us"], "headquartered_in")
        builder = GraphBuilder(store)
        data, id_map, events = builder.build()
        # Should still be 3 links total
        total_edges = sum(data[et].edge_index.shape[1] for et in data.edge_types)
        assert total_edges == 3

    def test_events_time_ordering(self, store: PipelineStore):
        _seed_graph(store)
        builder = GraphBuilder(store)
        _, _, events = builder.build()
        for i in range(1, len(events)):
            assert events[i]["observed_at"] >= events[i - 1]["observed_at"]

    def test_multiple_obs_same_entity(self, store: PipelineStore):
        """Exxon has 2 observations — feature must reflect both."""
        _seed_graph(store)
        builder = GraphBuilder(store)
        data, id_map, events = builder.build()
        type_dim = len(ENTITY_TYPES)
        # Company = Exxon with 2 insider_trade obs
        assert data["company"].x[0, type_dim].item() == 2.0  # count

    def test_heterodata_metadata(self, store: PipelineStore):
        """Verify HeteroData.metadata() returns correct structure."""
        _seed_graph(store)
        builder = GraphBuilder(store)
        data, _, _ = builder.build()
        node_types, edge_types = data.metadata()
        assert isinstance(node_types, list)
        assert isinstance(edge_types, list)
        assert all(isinstance(t, str) for t in node_types)
        assert all(isinstance(t, tuple) and len(t) == 3 for t in edge_types)


def test_reference_time_ignores_corrupt_future_timestamps():
    import time

    from agent.models.gnn.graph_builder import _reference_time

    now = time.time()
    obs = [
        {"observed_at": now - 86400},
        {"observed_at": now - 3600},
        {"observed_at": 222462959400.0},  # year ~9019 gov_contracts bug
    ]
    ref = _reference_time(obs)
    assert ref <= now + 86400
    assert abs(ref - (now - 3600)) < 5.0
