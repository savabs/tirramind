"""Phase 30 integration diagnostics — Crypto cross-domain linking.

Tests use real PipelineStore (:memory:) to verify:
  - instrument → protocol links (tracks_protocol) from instrument_universe
  - wallet → instrument links (trades_instrument) from whale_alert
  - Multi-hop path: wallet → BTC-USD → bitcoin protocol
  - Graph builder picks up new edge types
"""

from __future__ import annotations

import time

import pytest

from agent.models.gnn.graph_builder import GraphBuilder
from agent.pipeline.entity import entity_id_from_key
from agent.pipeline.store import PipelineStore
from agent.tools.instrument_universe import (
    _entity_id,
    _persist_instrument_links,
)

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture()
def store():
    s = PipelineStore(db_path=":memory:")
    yield s
    s.close()


def _reg(store, etype, name):
    eid = entity_id_from_key(etype, name)
    store.register_entity(etype, name, eid)
    return eid


def _link(store, eid_a, eid_b, link_type, source="test"):
    return store.link_entities(
        entity_id_a=eid_a,
        entity_id_b=eid_b,
        link_type=link_type,
        source=source,
        confidence=1.0,
        metadata={},
    )


def _obs(store, eid, tool, obs_type, value=None, ts=None):
    return store.store_entity_observation(
        entity_id=eid,
        source_tool=tool,
        observed_at=ts or time.time(),
        observation_type=obs_type,
        value=value or {},
    )


# ══════════════════════════════════════════════════════════════
# 1. Instrument → Protocol links via _persist_instrument_links
# ══════════════════════════════════════════════════════════════


class TestInstrumentProtocolLinks:
    """Real store: _persist_instrument_links creates tracks_protocol edges."""

    def test_tracks_protocol_links_created(self, store):
        counts = _persist_instrument_links(store)
        assert counts["tracks_protocol"] == 2

    def test_btc_protocol_entity_exists(self, store):
        _persist_instrument_links(store)
        proto_eid = entity_id_from_key("protocol", "bitcoin")
        entity = store.get_entity(proto_eid)
        assert entity is not None
        assert entity["entity_type"] == "protocol"

    def test_eth_protocol_entity_exists(self, store):
        _persist_instrument_links(store)
        proto_eid = entity_id_from_key("protocol", "ethereum")
        entity = store.get_entity(proto_eid)
        assert entity is not None
        assert entity["entity_type"] == "protocol"

    def test_btc_instrument_linked_to_bitcoin(self, store):
        _persist_instrument_links(store)
        btc_eid = _entity_id("BTC-USD")
        proto_eid = entity_id_from_key("protocol", "bitcoin")
        links = store.query_entity_links(btc_eid, link_type="tracks_protocol")
        assert len(links) >= 1
        assert any(l["entity_id_b"] == proto_eid for l in links)

    def test_idempotent_second_call(self, store):
        c1 = _persist_instrument_links(store)
        c2 = _persist_instrument_links(store)
        # Second call should produce 0 new tracks_protocol (INSERT OR IGNORE)
        assert c2["tracks_protocol"] == 0

    def test_non_crypto_counts_unchanged(self, store):
        counts = _persist_instrument_links(store)
        # tracks_issuer should be positive (many instruments have issuers)
        assert counts["tracks_issuer"] > 0


# ══════════════════════════════════════════════════════════════
# 2. Wallet → Instrument links (whale_alert path)
# ══════════════════════════════════════════════════════════════


class TestWalletInstrumentLinks:
    """Real store: simulate whale_alert wallet linking to BTC-USD."""

    def test_wallet_trades_instrument_link(self, store):
        # Pre-register BTC-USD instrument
        btc_eid = _entity_id("BTC-USD")
        _reg(store, "instrument", "BTC-USD")

        # Register a wallet and create trades_instrument link
        w_eid = _reg(store, "wallet", "1WalletAddr")
        _link(store, w_eid, btc_eid, "trades_instrument", source="whale_alert")

        links = store.query_entity_links(w_eid)
        link_types = [l["link_type"] for l in links]
        assert "trades_instrument" in link_types

    def test_multiple_wallets_linked(self, store):
        btc_eid = _entity_id("BTC-USD")
        _reg(store, "instrument", "BTC-USD")

        for i in range(5):
            w_eid = _reg(store, "wallet", f"wallet_{i}")
            _link(store, w_eid, btc_eid, "trades_instrument")

        # Each wallet should have a link to BTC-USD
        for i in range(5):
            w_eid = entity_id_from_key("wallet", f"wallet_{i}")
            links = store.query_entity_links(w_eid)
            assert any(l["link_type"] == "trades_instrument" for l in links)


# ══════════════════════════════════════════════════════════════
# 3. Multi-hop path: wallet → BTC-USD → bitcoin protocol
# ══════════════════════════════════════════════════════════════


class TestMultiHopPath:
    """Verify the wallet→instrument→protocol chain is traversable."""

    def test_wallet_to_protocol_via_instrument(self, store):
        # Phase 30 creates both link types
        _persist_instrument_links(store)  # creates instrument→protocol

        btc_eid = _entity_id("BTC-USD")
        w_eid = _reg(store, "wallet", "1SomeWhale")
        _link(store, w_eid, btc_eid, "trades_instrument", source="whale_alert")

        # Hop 1: wallet → BTC-USD
        hop1 = store.query_entity_links(w_eid, link_type="trades_instrument")
        assert len(hop1) == 1
        assert hop1[0]["entity_id_b"] == btc_eid

        # Hop 2: BTC-USD → bitcoin protocol
        hop2 = store.query_entity_links(btc_eid, link_type="tracks_protocol")
        assert len(hop2) == 1
        proto_eid = entity_id_from_key("protocol", "bitcoin")
        assert hop2[0]["entity_id_b"] == proto_eid


# ══════════════════════════════════════════════════════════════
# 4. Graph builder sees crypto nodes
# ══════════════════════════════════════════════════════════════


class TestGraphBuilderCryptoNodes:
    """Graph builder picks up protocol and wallet nodes with correct edges."""

    def test_protocol_node_in_graph(self, store):
        _persist_instrument_links(store)
        # Need an observation for the protocol entity to appear in graph
        proto_eid = entity_id_from_key("protocol", "bitcoin")
        _obs(store, proto_eid, "defi_flows", "protocol_tvl", {"tvl_usd": 1e9})

        builder = GraphBuilder(store)
        data, id_map, _ = builder.build()
        assert id_map.num_nodes_of_type("protocol") >= 1

    def test_instrument_node_has_tracks_protocol_edge(self, store):
        _persist_instrument_links(store)
        btc_eid = _entity_id("BTC-USD")
        proto_eid = entity_id_from_key("protocol", "bitcoin")

        # Register BTC-USD as instrument entity so graph builder sees it
        _reg(store, "instrument", "BTC-USD")
        # Need observations for both entities to appear in graph
        _obs(store, btc_eid, "instrument_universe", "price_return", {"value": 0.02})
        _obs(store, proto_eid, "defi_flows", "protocol_tvl", {"tvl_usd": 5e8})

        builder = GraphBuilder(store)
        data, id_map, _ = builder.build()

        # tracks_protocol should be an edge type triplet
        edge_triplets = list(data.edge_types)
        tracks_proto = [et for et in edge_triplets if et[1] == "tracks_protocol"]
        assert len(tracks_proto) >= 1

    def test_wallet_instrument_edge_in_graph(self, store):
        _persist_instrument_links(store)
        btc_eid = _entity_id("BTC-USD")
        # Must register as instrument type for id_map to have it
        _reg(store, "instrument", "BTC-USD")
        w_eid = _reg(store, "wallet", "1GraphWallet")
        _link(store, w_eid, btc_eid, "trades_instrument")

        # Observations needed for nodes to materialize
        _obs(store, btc_eid, "instrument_universe", "price_return", {"value": 0.01})
        _obs(store, w_eid, "whale_alert", "btc_transfer", {"value_btc": 100.0})

        builder = GraphBuilder(store)
        data, id_map, _ = builder.build()

        edge_triplets = list(data.edge_types)
        trades_inst = [et for et in edge_triplets if et[1] == "trades_instrument"]
        assert len(trades_inst) >= 1

    def test_full_crypto_subgraph(self, store):
        """Full chain: wallet → BTC-USD → bitcoin in one graph build."""
        _persist_instrument_links(store)
        btc_eid = _entity_id("BTC-USD")
        proto_eid = entity_id_from_key("protocol", "bitcoin")
        # Register BTC-USD as instrument entity
        _reg(store, "instrument", "BTC-USD")
        w_eid = _reg(store, "wallet", "1FullChain")
        _link(store, w_eid, btc_eid, "trades_instrument")

        now = time.time()
        _obs(store, w_eid, "whale_alert", "btc_transfer", {"value_btc": 50}, ts=now)
        _obs(
            store,
            btc_eid,
            "instrument_universe",
            "price_return",
            {"value": 0.01},
            ts=now,
        )
        _obs(store, proto_eid, "defi_flows", "protocol_tvl", {"tvl_usd": 1e9}, ts=now)

        builder = GraphBuilder(store)
        data, id_map, _ = builder.build()

        assert id_map.num_nodes_of_type("wallet") >= 1
        assert id_map.num_nodes_of_type("instrument") >= 1
        assert id_map.num_nodes_of_type("protocol") >= 1

        edge_names = {et[1] for et in data.edge_types}
        assert "trades_instrument" in edge_names
        assert "tracks_protocol" in edge_names
