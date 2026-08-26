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
    BASE_FEAT_DIM,
    ENRICHMENT_DIM,
    ENTITY_TYPES,
    OBSERVATION_TYPES,
    GraphBuilder,
    IDMap,
    SchemaDriftError,
    _build_edge_data,
    _build_node_features,
    _compute_obs_stats,
    _links_as_of,
    validate_schema_against_store,
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


class TestUnknownEntityType:
    """An unregistered entity type must never masquerade as a registered one.

    Regression: `maritime_area` was absent from ENTITY_TYPES while present in
    pipeline.db, so `_build_node_features` fell back to index 0 and one-hot
    encoded it as `cftc_contract` behind a log warning. The GNN trained and
    scored it as the wrong entity kind for months.

    Feature building stays non-fatal (runtime discovery of new types is a
    supported feature) but must claim NO type rather than the wrong one.
    Loud detection is validate_schema_against_store()'s job.
    """

    def test_unknown_type_does_not_masquerade_as_first_type(self):
        ft = _build_node_features("definitely_not_a_real_type", ["e1"], [], 1000.0)
        assert ft[0, 0].item() != 1.0, f"unknown type was silently encoded as {ENTITY_TYPES[0]!r}"

    def test_unknown_type_one_hot_block_is_all_zero(self):
        ft = _build_node_features("definitely_not_a_real_type", ["e1"], [], 1000.0)
        one_hot = ft[0, : len(ENTITY_TYPES)]
        assert one_hot.sum().item() == 0.0, "unknown type claimed an identity"

    def test_known_type_still_encodes(self):
        """Guard the guard: the zero-block change must not break normal types."""
        ft = _build_node_features("country", ["e1"], [], 1000.0)
        assert ft[0, ENTITY_TYPES.index("country")].item() == 1.0
        assert ft[0, : len(ENTITY_TYPES)].sum().item() == 1.0


class TestLinkFutureBlindness:
    """Historical snapshots must not contain links created after the window.

    Regression (2026-08-26): `build()` filtered observations by `until` but
    fetched links with NO time filter, so every historical snapshot — including
    2023 windows — received the complete present-day link set (all 16,870).
    `build_from_cached()`, the path the TRAINING loop uses, was worse: links are
    pre-fetched once and reused across every window.

    This is LESSONS.md F-04, and it is the flattering kind of leakage: backtests
    improve and the improvement is entirely spurious.
    """

    def test_link_created_after_window_is_dropped(self):
        links = [
            {"entity_id_a": "a", "entity_id_b": "b", "created_at": 1000.0},
            {"entity_id_a": "c", "entity_id_b": "d", "created_at": 5000.0},
        ]
        kept = _links_as_of(links, until=2000.0)
        assert len(kept) == 1
        assert kept[0]["entity_id_a"] == "a", "future link leaked into the window"

    def test_link_exactly_at_boundary_is_kept(self):
        links = [{"entity_id_a": "a", "entity_id_b": "b", "created_at": 2000.0}]
        assert len(_links_as_of(links, until=2000.0)) == 1

    def test_until_none_keeps_everything(self):
        """until=None means 'live/current' — nothing is in the future."""
        links = [
            {"entity_id_a": "a", "entity_id_b": "b", "created_at": 1000.0},
            {"entity_id_a": "c", "entity_id_b": "d", "created_at": 9e9},
        ]
        assert len(_links_as_of(links, until=None)) == 2

    def test_link_without_created_at_is_kept(self):
        """Documented trade-off: pre-column links are kept, not silently dropped."""
        links = [{"entity_id_a": "a", "entity_id_b": "b"}]
        assert len(_links_as_of(links, until=1000.0)) == 1

    def test_edge_age_uses_window_clock_not_wallclock(self):
        """age_days measured against the snapshot, not against today.

        Replaying a 2023 window in 2026 previously stamped every edge as ~1000
        days old — a value the model could never have observed at that time.
        """
        id_map = IDMap()
        id_map.add("company", "a")
        id_map.add("company", "b")
        window_now = 1_000_000.0
        links = [
            {
                "entity_id_a": "a",
                "entity_id_b": "b",
                "link_type": "works_for",
                "confidence": 1.0,
                "created_at": window_now - 86400.0,  # exactly 1 day before window end
            }
        ]
        edge_data = _build_edge_data(links, id_map, reference_time=window_now)
        (attrs,) = (t["edge_attr"] for t in edge_data.values())
        age_days = attrs[0, 1].item()
        assert abs(age_days - 1.0) < 0.01, (
            f"age_days={age_days} — should be 1.0 relative to the window, not "
            "a huge number relative to wall-clock now"
        )


class TestEnrichmentDimDerivation:
    """ENRICHMENT_DIM must stay derived from OBSERVATION_TYPES.

    Regression: it was hardcoded to 55 — correct only at 46 observation types.
    When the registry grew to 48, `_build_node_features` wrote obs_type_dist at
    `offset + 9 + ot_idx`, which for ot_idx=46 addressed index 69 in a
    14+55=69-wide tensor:

        IndexError: index 69 is out of bounds for dimension 1 with size 69

    That crashed the entity_scoring DAG outright, and for instrument nodes the
    same overflow silently corrupted the price-feature block that follows.
    """

    def test_enrichment_dim_matches_formula(self):
        assert 9 + len(OBSERVATION_TYPES) == ENRICHMENT_DIM

    def test_enrichment_block_has_room_for_every_obs_type(self):
        """The exact invariant the old hardcoded value violated."""
        last_written_offset = 9 + len(OBSERVATION_TYPES) - 1
        assert last_written_offset < ENRICHMENT_DIM

    def test_enriched_features_do_not_overflow(self):
        """Build with enrichment and assert no out-of-bounds write."""
        enrichment = {
            "e1": {
                "cusum": 0.0,
                "hawkes": 0.0,
                "event_study": 0.0,
                "bocpd": 0.0,
                "variance": 0.0,
                "min": 0.0,
                "max": 0.0,
                "iqr": 0.0,
                "num_tools": 0.0,
            }
        }
        ft = _build_node_features("company", ["e1"], [], 1000.0, enrichment=enrichment)
        assert ft.shape[1] == BASE_FEAT_DIM + ENRICHMENT_DIM


class TestSchemaDriftGuard:
    """validate_schema_against_store is the guard missing when 23 → 49 drifted."""

    def test_clean_store_passes(self, store):
        report = validate_schema_against_store(store)
        assert report == {"unknown_entity_types": [], "unknown_observation_types": []}

    def test_unknown_entity_type_detected(self, store):
        eid = entity_id_from_key("alien_type", "x1")
        store.register_entity("alien_type", "Alien Entity", eid)
        with pytest.raises(SchemaDriftError, match="alien_type"):
            validate_schema_against_store(store)

    def test_non_strict_reports_without_raising(self, store):
        eid = entity_id_from_key("alien_type", "x1")
        store.register_entity("alien_type", "Alien Entity", eid)
        report = validate_schema_against_store(store, strict=False)
        assert report["unknown_entity_types"] == ["alien_type"]

    def test_error_message_is_actionable(self, store):
        eid = entity_id_from_key("alien_type", "x1")
        store.register_entity("alien_type", "Alien Entity", eid)
        with pytest.raises(SchemaDriftError, match="retrain"):
            validate_schema_against_store(store)


class TestSchemaMatchesDatabase:
    """ENTITY_TYPES / OBSERVATION_TYPES ordering is load-bearing.

    One-hot positions derive from list index, so inserting a type shifts every
    later index and invalidates existing checkpoints. Keeping both lists sorted
    makes insertions reviewable and the drift obvious.
    """

    def test_entity_types_sorted(self):
        assert sorted(ENTITY_TYPES) == ENTITY_TYPES

    def test_observation_types_sorted(self):
        assert sorted(OBSERVATION_TYPES) == OBSERVATION_TYPES

    def test_base_feat_dim_tracks_entity_types(self):
        assert len(ENTITY_TYPES) + 3 == BASE_FEAT_DIM


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
