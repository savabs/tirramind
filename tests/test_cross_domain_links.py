"""Tests for Phase 25d — cross-domain entity linking integration.

Validates that instrument → company/country links, CFTC → instrument links,
and Polymarket topic/wallet entities all survive store persistence and
produce the expected graph connectivity in the GraphBuilder output.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from agent.pipeline.entity import entity_id_from_key
from agent.pipeline.store import PipelineStore
from agent.models.gnn.graph_builder import GraphBuilder, ENTITY_TYPES, OBSERVATION_TYPES


# ── Helpers ───────────────────────────────────────────────────


def _make_store(tmp_path) -> PipelineStore:
    """Create a fresh PipelineStore with schema."""
    db = tmp_path / "test.db"
    return PipelineStore(str(db))


def _register(store, entity_type, key, name=None):
    """Register entity and return its ID."""
    eid = entity_id_from_key(entity_type, key)
    store.register_entity(entity_type, name or key, eid)
    return eid


def _observe(store, eid, obs_type, value=None, ts=None):
    """Store an observation on an entity."""
    store.store_entity_observation(
        entity_id=eid,
        observation_type=obs_type,
        value=value or {"signal": 1.0},
        source_tool="test",
        observed_at=ts or time.time(),
        depth_level=2,
    )


def _link(store, eid_a, eid_b, link_type):
    """Create a directional link between two entities."""
    store.link_entities(eid_a, eid_b, link_type, source="test")


# ── Test: Instrument → Company → Country links ───────────────


class TestInstrumentCompanyCountryGraph:
    """Validate instrument → company → country link chain in the graph."""

    @pytest.fixture
    def linked_store(self, tmp_path):
        store = _make_store(tmp_path)
        # Instrument: SPY
        spy = _register(store, "instrument", "SPY", "SPDR S&P 500 ETF")
        _observe(store, spy, "instrument_return", {"log_return": 0.01})
        # Company: BlackRock (issuer of SPY via iShares — using for test)
        blackrock = _register(
            store, "company", "State Street", "State Street Global Advisors"
        )
        # Country: US
        us = _register(store, "country", "US", "United States")
        # Links
        _link(store, spy, blackrock, "tracks_issuer")
        _link(store, blackrock, us, "located_in")
        return store

    def test_graph_has_instrument_node(self, linked_store):
        builder = GraphBuilder(linked_store)
        data, id_map, _ = builder.build()
        assert "instrument" in data.node_types

    def test_graph_has_company_node(self, linked_store):
        builder = GraphBuilder(linked_store)
        data, id_map, _ = builder.build()
        assert "company" in data.node_types

    def test_graph_has_country_node(self, linked_store):
        builder = GraphBuilder(linked_store)
        data, id_map, _ = builder.build()
        assert "country" in data.node_types

    def test_tracks_issuer_edge_exists(self, linked_store):
        builder = GraphBuilder(linked_store)
        data, id_map, _ = builder.build()
        edge_types = data.edge_types
        # Graph builder creates edges from (src_type, link_type, dst_type) triplets
        assert ("instrument", "tracks_issuer", "company") in edge_types

    def test_located_in_edge_exists(self, linked_store):
        builder = GraphBuilder(linked_store)
        data, id_map, _ = builder.build()
        edge_types = data.edge_types
        assert ("company", "located_in", "country") in edge_types

    def test_instrument_degree_nonzero(self, linked_store):
        """Instruments with issuer links should have degree > 0."""
        builder = GraphBuilder(linked_store)
        data, id_map, _ = builder.build()
        # tracks_issuer edge should have at least 1 edge
        edge_key = ("instrument", "tracks_issuer", "company")
        assert data[edge_key].edge_index.size(1) >= 1


# ── Test: CFTC contract → Instrument links ────────────────────


class TestCftcInstrumentGraph:
    """Validate CFTC contract → instrument linking in the graph."""

    @pytest.fixture
    def cftc_store(self, tmp_path):
        store = _make_store(tmp_path)
        # Instrument: CL=F (crude oil)
        cl = _register(store, "instrument", "CL=F", "Crude Oil Futures")
        _observe(store, cl, "instrument_return", {"log_return": -0.02})
        # CFTC contract
        cftc = _register(
            store,
            "cftc_contract",
            "06765A",
            "CRUDE OIL WTI - NEW YORK MERCANTILE EXCHANGE",
        )
        _observe(
            store,
            cftc,
            "futures_positioning",
            {
                "commercial_long": 350000,
                "commercial_short": 420000,
                "noncommercial_long": 500000,
                "noncommercial_short": 380000,
            },
        )
        # Link: CFTC tracks instrument
        _link(store, cftc, cl, "cftc_tracks")
        return store

    def test_cftc_contract_in_graph(self, cftc_store):
        builder = GraphBuilder(cftc_store)
        data, id_map, _ = builder.build()
        assert "cftc_contract" in data.node_types

    def test_cftc_tracks_edge(self, cftc_store):
        builder = GraphBuilder(cftc_store)
        data, id_map, _ = builder.build()
        assert ("cftc_contract", "cftc_tracks", "instrument") in data.edge_types

    def test_cftc_observation_in_events(self, cftc_store):
        builder = GraphBuilder(cftc_store)
        _, _, events = builder.build()
        futures_events = [
            e for e in events if e.get("observation_type") == "futures_positioning"
        ]
        assert len(futures_events) >= 1

    def test_instrument_reachable_from_cftc(self, cftc_store):
        """CFTC contract has edge to instrument — instrument is 1-hop neighbor."""
        builder = GraphBuilder(cftc_store)
        data, id_map, _ = builder.build()
        edge_key = ("cftc_contract", "cftc_tracks", "instrument")
        assert data[edge_key].edge_index.size(1) == 1


# ── Test: Polymarket topic + wallet entities ──────────────────


class TestPolymarketEntitiesInGraph:
    """Validate topic and wallet entities appear in the graph."""

    @pytest.fixture
    def poly_store(self, tmp_path):
        store = _make_store(tmp_path)
        # Topic entity
        topic = _register(
            store, "topic", "will-trump-win-2024", "Will Trump win the 2024 election?"
        )
        _observe(
            store,
            topic,
            "market_probability",
            {
                "yes_price": 0.52,
                "volume_24h": 1500000,
                "liquidity": 500000,
            },
        )
        # Wallet entity
        wallet = _register(
            store,
            "wallet",
            "0xabcdef1234567890abcdef1234567890abcdef12",
            "whale-0xabcd",
        )
        _observe(
            store,
            wallet,
            "whale_trade",
            {
                "composite_score": 85.3,
                "accuracy": 0.72,
                "volume": 250000,
            },
        )
        return store

    def test_topic_node_in_graph(self, poly_store):
        builder = GraphBuilder(poly_store)
        data, id_map, _ = builder.build()
        assert "topic" in data.node_types

    def test_wallet_node_in_graph(self, poly_store):
        builder = GraphBuilder(poly_store)
        data, id_map, _ = builder.build()
        assert "wallet" in data.node_types

    def test_market_probability_events(self, poly_store):
        builder = GraphBuilder(poly_store)
        _, _, events = builder.build()
        mp_events = [
            e for e in events if e.get("observation_type") == "market_probability"
        ]
        assert len(mp_events) >= 1

    def test_whale_trade_events(self, poly_store):
        builder = GraphBuilder(poly_store)
        _, _, events = builder.build()
        wt_events = [e for e in events if e.get("observation_type") == "whale_trade"]
        assert len(wt_events) >= 1


# ── Test: Multi-domain connected graph ────────────────────────


class TestMultiDomainGraph:
    """Validate a graph with entities/links across all Phase 25 domains."""

    @pytest.fixture
    def rich_store(self, tmp_path):
        store = _make_store(tmp_path)
        ts = int(time.time())

        # Instruments
        spy = _register(store, "instrument", "SPY")
        cl = _register(store, "instrument", "CL=F")
        for inst in [spy, cl]:
            _observe(store, inst, "instrument_return", {"log_return": 0.01}, ts)

        # Companies
        ssga = _register(store, "company", "State Street")

        # Countries
        us = _register(store, "country", "US")

        # CFTC
        cftc_cl = _register(store, "cftc_contract", "06765A")
        _observe(store, cftc_cl, "futures_positioning", {"commercial_long": 100}, ts)

        # Polymarket
        topic = _register(store, "topic", "fed-rate-cut")
        _observe(store, topic, "market_probability", {"yes_price": 0.65}, ts)

        wallet = _register(store, "wallet", "0xdeadbeef")
        _observe(store, wallet, "whale_trade", {"composite_score": 90.0}, ts)

        # Links
        _link(store, spy, ssga, "tracks_issuer")
        _link(store, ssga, us, "located_in")
        _link(store, cftc_cl, cl, "cftc_tracks")

        return store

    def test_all_entity_types_present(self, rich_store):
        builder = GraphBuilder(rich_store)
        data, _, _ = builder.build()
        expected = {
            "instrument",
            "company",
            "country",
            "cftc_contract",
            "topic",
            "wallet",
        }
        actual = set(data.node_types)
        assert expected.issubset(actual), f"Missing: {expected - actual}"

    def test_all_link_types_present(self, rich_store):
        builder = GraphBuilder(rich_store)
        data, _, _ = builder.build()
        edge_types = set(data.edge_types)
        assert ("instrument", "tracks_issuer", "company") in edge_types
        assert ("company", "located_in", "country") in edge_types
        assert ("cftc_contract", "cftc_tracks", "instrument") in edge_types

    def test_total_node_count(self, rich_store):
        """Should have 7 entities total."""
        builder = GraphBuilder(rich_store)
        data, _, _ = builder.build()
        total = sum(data[nt].x.size(0) for nt in data.node_types)
        assert total == 7

    def test_total_edge_count(self, rich_store):
        """Should have 3 explicit links."""
        builder = GraphBuilder(rich_store)
        data, _, _ = builder.build()
        total = sum(data[et].edge_index.size(1) for et in data.edge_types)
        assert total == 3

    def test_event_count(self, rich_store):
        """Should have 5 observations total (2 instrument + 1 CFTC + 1 poly + 1 whale)."""
        builder = GraphBuilder(rich_store)
        _, _, events = builder.build()
        assert len(events) == 5

    def test_diagnostics_instrument_nonzero_degree(self, rich_store):
        """Instruments should now have degree > 0 due to cross-domain links."""
        all_entities = rich_store.query_all_entities()
        all_links = rich_store.query_all_entity_links()

        eid_to_type = {e["entity_id"]: e["entity_type"] for e in all_entities}
        entity_degree: dict[str, int] = {}
        for lnk in all_links:
            for k in ("entity_id_a", "entity_id_b"):
                eid = lnk[k]
                entity_degree[eid] = entity_degree.get(eid, 0) + 1

        # SPY should have degree >= 1 (tracks_issuer)
        spy_eid = entity_id_from_key("instrument", "SPY")
        assert entity_degree.get(spy_eid, 0) >= 1

        # CL=F should have degree >= 1 (cftc_tracks target)
        cl_eid = entity_id_from_key("instrument", "CL=F")
        assert entity_degree.get(cl_eid, 0) >= 1

        # State Street should have degree >= 2 (tracks_issuer + located_in)
        ssga_eid = entity_id_from_key("company", "State Street")
        assert entity_degree.get(ssga_eid, 0) >= 2


# ── Test: Graph builder handles new entity/obs types ──────────


class TestNewTypeRegistrations:
    """Confirm all Phase 25 types are properly registered in graph_builder."""

    def test_cftc_contract_in_entity_types(self):
        assert "cftc_contract" in ENTITY_TYPES

    def test_futures_positioning_in_obs_types(self):
        assert "futures_positioning" in OBSERVATION_TYPES

    def test_market_probability_in_obs_types(self):
        assert "market_probability" in OBSERVATION_TYPES

    def test_whale_trade_in_obs_types(self):
        assert "whale_trade" in OBSERVATION_TYPES

    def test_entity_types_sorted(self):
        """Entity types should be alphabetically sorted for deterministic one-hot encoding."""
        assert ENTITY_TYPES == sorted(ENTITY_TYPES)

    def test_observation_types_sorted(self):
        """Obs types should be alphabetically sorted for deterministic encoding."""
        assert OBSERVATION_TYPES == sorted(OBSERVATION_TYPES)
