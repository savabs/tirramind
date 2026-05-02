"""Phase 30 edge case tests — Crypto Islands + Cross-Domain Linking.

Covers:
  - InstrumentDef protocol field (construction, backward compat)
  - _persist_instrument_links tracks_protocol (creation, count, non-crypto skip)
  - Protocol entity ID consistency (instrument_universe vs defi_flows path)
  - whale_alert trades_instrument links (creation, idempotency, empty, missing addr)
  - Instrument EID consistency (_entity_id vs entity_id_from_key)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from agent.pipeline.entity import entity_id_from_key
from agent.tools.instrument_universe import (
    INSTRUMENTS,
    InstrumentDef,
    _entity_id,
    _persist_instrument_links,
)
from agent.tools.whale_alert import _BTC_INSTRUMENT_EID, WhaleAlertTool

# ── Helpers ────────────────────────────────────────────────────


def _mock_store() -> MagicMock:
    """Mock PipelineStore with idempotent link_entities."""
    store = MagicMock()
    _seen_links: set[tuple[str, str, str]] = set()

    def _link(
        entity_id_a: str,
        entity_id_b: str,
        link_type: str,
        source: str = "",
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> int | None:
        key = (entity_id_a, entity_id_b, link_type)
        if key in _seen_links:
            return None  # duplicate — idempotent
        _seen_links.add(key)
        return len(_seen_links)

    store.link_entities = MagicMock(side_effect=_link)
    store.register_entity = MagicMock(return_value="eid")
    store.add_entity_alias = MagicMock()
    store.store_entity_observation = MagicMock(return_value=1)
    return store


def _make_tx(
    *,
    hash: str = "abc123",
    time: int = 1700000000,
    inputs: list[dict[str, Any]] | None = None,
    outputs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if inputs is None:
        inputs = [{"addr": "1SenderAddr", "value_btc": 50.0}]
    if outputs is None:
        outputs = [{"addr": "1ReceiverAddr", "value_btc": 50.0}]
    return {
        "hash": hash,
        "time": time,
        "value_btc": 50.0,
        "confirmed": False,
        "inputs": inputs,
        "outputs": outputs,
    }


# ══════════════════════════════════════════════════════════════
# 1. InstrumentDef protocol field
# ══════════════════════════════════════════════════════════════


class TestInstrumentDefProtocol:
    """Step 30.1 — protocol field on InstrumentDef."""

    def test_default_protocol_is_none(self) -> None:
        inst = InstrumentDef("AAPL", "Apple", "equity_etf", "US")
        assert inst.protocol is None

    def test_crypto_with_protocol(self) -> None:
        inst = InstrumentDef("BTC-USD", "Bitcoin", "crypto", "Global", protocol="bitcoin")
        assert inst.protocol == "bitcoin"
        assert inst.ticker == "BTC-USD"

    def test_frozen_dataclass_hash(self) -> None:
        a = InstrumentDef("BTC-USD", "Bitcoin", "crypto", "Global", protocol="bitcoin")
        b = InstrumentDef("BTC-USD", "Bitcoin", "crypto", "Global", protocol="bitcoin")
        assert a == b
        assert hash(a) == hash(b)

    def test_protocol_does_not_break_existing_fields(self) -> None:
        inst = InstrumentDef(
            "SPY",
            "S&P 500",
            "equity_etf",
            "US",
            issuer="State Street",
            country="US",
        )
        assert inst.protocol is None
        assert inst.issuer == "State Street"
        assert inst.country == "US"

    def test_btc_in_universe_has_protocol(self) -> None:
        btc = next(i for i in INSTRUMENTS if i.ticker == "BTC-USD")
        assert btc.protocol == "bitcoin"

    def test_eth_in_universe_has_protocol(self) -> None:
        eth = next(i for i in INSTRUMENTS if i.ticker == "ETH-USD")
        assert eth.protocol == "ethereum"

    def test_non_crypto_instruments_have_no_protocol(self) -> None:
        for inst in INSTRUMENTS:
            if inst.asset_class != "crypto":
                assert inst.protocol is None, f"{inst.ticker} should not have protocol"

    def test_all_existing_instruments_construct(self) -> None:
        """Backward compat: every instrument in INSTRUMENTS still constructs."""
        for inst in INSTRUMENTS:
            assert isinstance(inst, InstrumentDef)
            assert inst.ticker


# ══════════════════════════════════════════════════════════════
# 2. _persist_instrument_links tracks_protocol
# ══════════════════════════════════════════════════════════════


class TestTracksProtocolLinks:
    """Step 30.2 — tracks_protocol links from crypto instruments."""

    def test_counts_include_tracks_protocol(self) -> None:
        store = _mock_store()
        counts = _persist_instrument_links(store)
        assert "tracks_protocol" in counts

    def test_two_tracks_protocol_links_created(self) -> None:
        store = _mock_store()
        counts = _persist_instrument_links(store)
        assert counts["tracks_protocol"] == 2  # BTC-USD + ETH-USD

    def test_protocol_entity_registered(self) -> None:
        store = _mock_store()
        _persist_instrument_links(store)
        # Check register_entity was called with protocol type for both
        reg_calls = store.register_entity.call_args_list
        protocol_regs = [
            c for c in reg_calls if c.kwargs.get("entity_type") == "protocol" or (c.args and c.args[0] == "protocol")
        ]
        assert len(protocol_regs) == 2

    def test_link_entity_ids_correct(self) -> None:
        store = _mock_store()
        _persist_instrument_links(store)
        link_calls = store.link_entities.call_args_list
        protocol_links = [c for c in link_calls if c.kwargs.get("link_type") == "tracks_protocol"]
        assert len(protocol_links) == 2

        btc_eid = _entity_id("BTC-USD")
        eth_eid = _entity_id("ETH-USD")
        btc_proto_eid = entity_id_from_key("protocol", "bitcoin")
        eth_proto_eid = entity_id_from_key("protocol", "ethereum")

        link_pairs = {(c.kwargs["entity_id_a"], c.kwargs["entity_id_b"]) for c in protocol_links}
        assert (btc_eid, btc_proto_eid) in link_pairs
        assert (eth_eid, eth_proto_eid) in link_pairs

    def test_idempotent_double_call(self) -> None:
        store = _mock_store()
        c1 = _persist_instrument_links(store)
        c2 = _persist_instrument_links(store)
        # Second call: all tracks_protocol links are duplicates → 0
        assert c2["tracks_protocol"] == 0

    def test_non_crypto_no_tracks_protocol(self) -> None:
        """Only crypto instruments get tracks_protocol links."""
        store = _mock_store()
        _persist_instrument_links(store)
        link_calls = store.link_entities.call_args_list
        protocol_links = [c for c in link_calls if c.kwargs.get("link_type") == "tracks_protocol"]
        protocol_tickers = {c.kwargs.get("metadata", {}).get("ticker") for c in protocol_links}
        assert protocol_tickers == {"BTC-USD", "ETH-USD"}


# ══════════════════════════════════════════════════════════════
# 3. Protocol entity ID consistency
# ══════════════════════════════════════════════════════════════


class TestProtocolEntityIDConsistency:
    """Step 30.3 — instrument_universe and defi_flows produce same IDs."""

    def test_bitcoin_protocol_id_matches(self) -> None:
        """entity_id_from_key('protocol', 'bitcoin') is deterministic."""
        eid = entity_id_from_key("protocol", "bitcoin")
        assert isinstance(eid, str)
        assert len(eid) == 16
        # Same call again should produce same result
        assert entity_id_from_key("protocol", "bitcoin") == eid

    def test_ethereum_protocol_id_matches(self) -> None:
        eid = entity_id_from_key("protocol", "ethereum")
        assert isinstance(eid, str)
        assert len(eid) == 16
        assert entity_id_from_key("protocol", "ethereum") == eid

    def test_bitcoin_ne_ethereum(self) -> None:
        assert entity_id_from_key("protocol", "bitcoin") != entity_id_from_key("protocol", "ethereum")

    def test_instrument_eid_matches_entity_id_from_key(self) -> None:
        """_entity_id(ticker) uses same hash as entity_id_from_key('instrument', ticker)."""
        for ticker in ("BTC-USD", "ETH-USD", "CL=F", "SPY"):
            assert _entity_id(ticker) == entity_id_from_key("instrument", ticker)


# ══════════════════════════════════════════════════════════════
# 4. whale_alert trades_instrument links
# ══════════════════════════════════════════════════════════════


class TestWhaleAlertTradesInstrument:
    """Step 30.4 — wallet → BTC-USD links from whale_alert."""

    def test_btc_instrument_eid_computed(self) -> None:
        assert _BTC_INSTRUMENT_EID is not None
        assert entity_id_from_key("instrument", "BTC-USD") == _BTC_INSTRUMENT_EID
        assert _entity_id("BTC-USD") == _BTC_INSTRUMENT_EID

    def test_trades_instrument_link_created(self) -> None:
        store = _mock_store()
        tool = WhaleAlertTool(pipeline_store=store)
        txs = [_make_tx()]
        tool._persist_entities_inner(txs)

        link_calls = store.link_entities.call_args_list
        ti_links = [c for c in link_calls if c.kwargs.get("link_type") == "trades_instrument"]
        assert len(ti_links) >= 1

    def test_trades_instrument_targets_btc(self) -> None:
        store = _mock_store()
        tool = WhaleAlertTool(pipeline_store=store)
        txs = [_make_tx()]
        tool._persist_entities_inner(txs)

        link_calls = store.link_entities.call_args_list
        ti_links = [c for c in link_calls if c.kwargs.get("link_type") == "trades_instrument"]
        for c in ti_links:
            assert c.kwargs["entity_id_b"] == _BTC_INSTRUMENT_EID

    def test_sender_and_receiver_both_linked(self) -> None:
        store = _mock_store()
        tool = WhaleAlertTool(pipeline_store=store)
        txs = [
            _make_tx(
                inputs=[{"addr": "sender1", "value_btc": 10.0}],
                outputs=[{"addr": "receiver1", "value_btc": 10.0}],
            )
        ]
        tool._persist_entities_inner(txs)

        link_calls = store.link_entities.call_args_list
        ti_links = [c for c in link_calls if c.kwargs.get("link_type") == "trades_instrument"]
        linked_wallets = {c.kwargs["entity_id_a"] for c in ti_links}
        assert entity_id_from_key("wallet", "sender1") in linked_wallets
        assert entity_id_from_key("wallet", "receiver1") in linked_wallets

    def test_multiple_wallets_per_tx(self) -> None:
        store = _mock_store()
        tool = WhaleAlertTool(pipeline_store=store)
        txs = [
            _make_tx(
                inputs=[
                    {"addr": "s1", "value_btc": 5.0},
                    {"addr": "s2", "value_btc": 5.0},
                ],
                outputs=[
                    {"addr": "r1", "value_btc": 7.0},
                    {"addr": "r2", "value_btc": 3.0},
                ],
            )
        ]
        tool._persist_entities_inner(txs)

        link_calls = store.link_entities.call_args_list
        ti_links = [c for c in link_calls if c.kwargs.get("link_type") == "trades_instrument"]
        linked_wallets = {c.kwargs["entity_id_a"] for c in ti_links}
        for addr in ("s1", "s2", "r1", "r2"):
            assert entity_id_from_key("wallet", addr) in linked_wallets

    def test_empty_txs_no_links(self) -> None:
        store = _mock_store()
        tool = WhaleAlertTool(pipeline_store=store)
        tool._persist_entities_inner([])
        assert store.link_entities.call_count == 0

    def test_missing_addr_skipped(self) -> None:
        store = _mock_store()
        tool = WhaleAlertTool(pipeline_store=store)
        txs = [
            _make_tx(
                inputs=[{"addr": "", "value_btc": 5.0}],
                outputs=[{"addr": "", "value_btc": 5.0}],
            )
        ]
        tool._persist_entities_inner(txs)

        link_calls = store.link_entities.call_args_list
        ti_links = [c for c in link_calls if c.kwargs.get("link_type") == "trades_instrument"]
        assert len(ti_links) == 0

    def test_idempotent_same_wallet(self) -> None:
        """Second call with same wallet should still call link_entities (store handles dedup)."""
        store = _mock_store()
        tool = WhaleAlertTool(pipeline_store=store)
        tx = _make_tx(inputs=[{"addr": "w1", "value_btc": 5.0}], outputs=[])
        tool._persist_entities_inner([tx, tx])

        link_calls = store.link_entities.call_args_list
        ti_links = [c for c in link_calls if c.kwargs.get("link_type") == "trades_instrument"]
        # Called twice (once per tx), but store idempotency handles dedup
        assert len(ti_links) == 2

    def test_no_store_skips_gracefully(self) -> None:
        tool = WhaleAlertTool()
        assert tool._store is None
        # Should not raise
        tool._persist_entities([_make_tx()])

    def test_tx_hash_in_metadata(self) -> None:
        store = _mock_store()
        tool = WhaleAlertTool(pipeline_store=store)
        txs = [_make_tx(hash="deadbeef123")]
        tool._persist_entities_inner(txs)

        link_calls = store.link_entities.call_args_list
        ti_links = [c for c in link_calls if c.kwargs.get("link_type") == "trades_instrument"]
        for c in ti_links:
            assert c.kwargs["metadata"]["tx_hash"] == "deadbeef123"
