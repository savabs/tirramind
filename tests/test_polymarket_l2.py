"""Tests for Polymarket L2 entity persistence — Phase 25c.

Tests cover:
- polymarket.py: topic entity registration + market_probability observations
- polymarket_whales.py: wallet entity registration + whale_trade observations
- Edge cases: missing slugs, missing wallets, empty data, persistence failures
- Graph builder type integration
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from agent.tools.polymarket import PolymarketTool
from agent.tools.polymarket_whales import PolymarketWhalesTool

# ── Helpers ───────────────────────────────────────────────────


def _make_store() -> MagicMock:
    store = MagicMock()
    store.register_entity = MagicMock(side_effect=lambda **kw: kw["entity_id"])
    store.store_entity_observation = MagicMock(return_value=1)
    store.link_entities = MagicMock(return_value=1)
    store.close = MagicMock()
    return store


def _make_market(
    slug: str = "will-btc-reach-100k",
    question: str = "Will BTC reach $100K?",
    category: str = "crypto",
    yes_price: float = 0.65,
    no_price: float = 0.35,
    volume_24h: float = 50000.0,
    volume_total: float = 500000.0,
    liquidity: float = 100000.0,
    spread: float = 0.02,
    price_change_24h: float = 0.05,
    price_change_1wk: float = -0.02,
    **overrides: Any,
) -> dict[str, Any]:
    mkt = {
        "slug": slug,
        "question": question,
        "category": category,
        "yes_price": yes_price,
        "no_price": no_price,
        "volume_24h": volume_24h,
        "volume_total": volume_total,
        "liquidity": liquidity,
        "spread": spread,
        "price_change_24h": price_change_24h,
        "price_change_1wk": price_change_1wk,
        "end_date": "2025-12-31",
    }
    mkt.update(overrides)
    return mkt


def _make_wallet(
    wallet: str = "0xabc123def456",
    composite: float = 0.85,
    accuracy: float = 0.72,
    total_volume: float = 150000.0,
    total_resolved: int = 50,
    markets: int = 12,
    profit_factor: float = 2.1,
    **overrides: Any,
) -> dict[str, Any]:
    w = {
        "wallet": wallet,
        "composite": composite,
        "accuracy": accuracy,
        "total_volume": total_volume,
        "total_resolved": total_resolved,
        "markets": markets,
        "profit_factor": profit_factor,
    }
    w.update(overrides)
    return w


# ── Polymarket topic entity tests ────────────────────────────


class TestPolymarketPersistEntitiesBasic:
    def test_no_store_returns_zeros(self):
        tool = PolymarketTool(cache=None, pipeline_store=None)
        result = tool._persist_entities([_make_market()])
        assert result == {"topics": 0, "observations": 0}

    def test_empty_markets_returns_zeros(self):
        store = _make_store()
        tool = PolymarketTool(cache=None, pipeline_store=store)
        result = tool._persist_entities([])
        assert result == {"topics": 0, "observations": 0}
        store.register_entity.assert_not_called()

    def test_single_market_registers_topic(self):
        store = _make_store()
        tool = PolymarketTool(cache=None, pipeline_store=store)
        tool._persist_entities([_make_market(slug="btc-100k")])

        topic_calls = [c for c in store.register_entity.call_args_list if c.kwargs.get("entity_type") == "topic"]
        assert len(topic_calls) == 1
        kw = topic_calls[0].kwargs
        assert kw["metadata"]["slug"] == "btc-100k"
        assert kw["metadata"]["source"] == "polymarket"

    def test_single_market_stores_observation(self):
        store = _make_store()
        tool = PolymarketTool(cache=None, pipeline_store=store)
        tool._persist_entities([_make_market(yes_price=0.7)])

        assert store.store_entity_observation.call_count == 1
        kw = store.store_entity_observation.call_args.kwargs
        assert kw["observation_type"] == "market_probability"
        assert kw["depth_level"] == 2
        assert kw["source_tool"] == "polymarket"
        assert kw["value"]["yes_price"] == 0.7

    def test_observation_value_contains_signals(self):
        store = _make_store()
        tool = PolymarketTool(cache=None, pipeline_store=store)
        mkt = _make_market(volume_24h=99000, liquidity=200000, spread=0.01)
        tool._persist_entities([mkt])
        val = store.store_entity_observation.call_args.kwargs["value"]
        assert val["volume_24h"] == 99000
        assert val["liquidity"] == 200000
        assert val["spread"] == 0.01


class TestPolymarketPersistEntitiesMulti:
    def test_two_markets_two_topics(self):
        store = _make_store()
        tool = PolymarketTool(cache=None, pipeline_store=store)
        markets = [
            _make_market(slug="btc-100k"),
            _make_market(slug="fed-rate-cut"),
        ]
        result = tool._persist_entities(markets)
        assert result["topics"] == 2
        assert result["observations"] == 2

    def test_duplicate_slug_deduped(self):
        store = _make_store()
        tool = PolymarketTool(cache=None, pipeline_store=store)
        markets = [
            _make_market(slug="btc-100k"),
            _make_market(slug="btc-100k"),
        ]
        result = tool._persist_entities(markets)
        assert result["topics"] == 1  # deduped
        assert result["observations"] == 2  # one per row


class TestPolymarketPersistEdgeCases:
    def test_empty_slug_skipped(self):
        store = _make_store()
        tool = PolymarketTool(cache=None, pipeline_store=store)
        result = tool._persist_entities([_make_market(slug="")])
        assert result["topics"] == 0
        assert result["observations"] == 0

    def test_whitespace_slug_skipped(self):
        store = _make_store()
        tool = PolymarketTool(cache=None, pipeline_store=store)
        result = tool._persist_entities([_make_market(slug="   ")])
        assert result["topics"] == 0

    def test_missing_slug_key(self):
        store = _make_store()
        tool = PolymarketTool(cache=None, pipeline_store=store)
        mkt = _make_market()
        del mkt["slug"]
        result = tool._persist_entities([mkt])
        assert result["topics"] == 0

    def test_persist_exception_is_nonfatal(self):
        store = _make_store()
        store.register_entity.side_effect = RuntimeError("DB error")
        tool = PolymarketTool(cache=None, pipeline_store=store)
        result = tool._persist_entities([_make_market()])
        assert result == {"topics": 0, "observations": 0}

    def test_long_question_truncated(self):
        store = _make_store()
        tool = PolymarketTool(cache=None, pipeline_store=store)
        long_q = "x" * 500
        tool._persist_entities([_make_market(question=long_q)])
        kw = store.register_entity.call_args.kwargs
        assert len(kw["canonical_name"]) <= 200


class TestPolymarketConstructor:
    def test_accepts_pipeline_store(self):
        store = _make_store()
        tool = PolymarketTool(cache=None, pipeline_store=store)
        assert tool._store is store

    def test_pipeline_store_defaults_none(self):
        tool = PolymarketTool(cache=None)
        assert tool._store is None

    def test_backward_compatible(self):
        tool = PolymarketTool(None)
        assert tool._store is None


# ── Polymarket whales wallet entity tests ────────────────────


class TestWhalePersistBasic:
    def test_registers_wallet_entity(self):
        store = _make_store()
        tool = PolymarketWhalesTool()
        result = tool._persist_wallet_entities([_make_wallet(wallet="0xabc123")], store)
        assert result["wallets"] == 1
        wallet_calls = [c for c in store.register_entity.call_args_list if c.kwargs.get("entity_type") == "wallet"]
        assert len(wallet_calls) == 1
        assert wallet_calls[0].kwargs["canonical_name"] == "0xabc123"

    def test_stores_whale_trade_observation(self):
        store = _make_store()
        tool = PolymarketWhalesTool()
        tool._persist_wallet_entities([_make_wallet(composite=0.9, accuracy=0.8)], store)
        assert store.store_entity_observation.call_count == 1
        kw = store.store_entity_observation.call_args.kwargs
        assert kw["observation_type"] == "whale_trade"
        assert kw["depth_level"] == 2
        assert kw["value"]["composite_score"] == 0.9
        assert kw["value"]["accuracy"] == 0.8


class TestWhalePersistMulti:
    def test_two_wallets(self):
        store = _make_store()
        tool = PolymarketWhalesTool()
        wallets = [
            _make_wallet(wallet="0xaaa"),
            _make_wallet(wallet="0xbbb"),
        ]
        result = tool._persist_wallet_entities(wallets, store)
        assert result["wallets"] == 2
        assert result["observations"] == 2

    def test_duplicate_wallet_deduped(self):
        store = _make_store()
        tool = PolymarketWhalesTool()
        wallets = [
            _make_wallet(wallet="0xaaa"),
            _make_wallet(wallet="0xaaa"),
        ]
        result = tool._persist_wallet_entities(wallets, store)
        assert result["wallets"] == 1
        assert result["observations"] == 1


class TestWhalePersistEdgeCases:
    def test_empty_wallet_skipped(self):
        store = _make_store()
        tool = PolymarketWhalesTool()
        result = tool._persist_wallet_entities([_make_wallet(wallet="")], store)
        assert result["wallets"] == 0

    def test_non_0x_wallet_skipped(self):
        store = _make_store()
        tool = PolymarketWhalesTool()
        result = tool._persist_wallet_entities([_make_wallet(wallet="not-a-wallet")], store)
        assert result["wallets"] == 0

    def test_empty_list(self):
        store = _make_store()
        tool = PolymarketWhalesTool()
        result = tool._persist_wallet_entities([], store)
        assert result == {"wallets": 0, "observations": 0}

    def test_no_entity_id_func(self):
        """Should return zeros if entity_id_from_key is unavailable."""
        store = _make_store()
        tool = PolymarketWhalesTool()
        with patch("agent.tools.polymarket_whales.entity_id_from_key", None):
            result = tool._persist_wallet_entities([_make_wallet()], store)
        assert result == {"wallets": 0, "observations": 0}

    def test_case_insensitive_dedup(self):
        """Wallets are lowercased for dedup."""
        store = _make_store()
        tool = PolymarketWhalesTool()
        wallets = [
            _make_wallet(wallet="0xABC"),
            _make_wallet(wallet="0xabc"),
        ]
        result = tool._persist_wallet_entities(wallets, store)
        assert result["wallets"] == 1


# ── Graph builder integration ────────────────────────────────


class TestPolymarketGraphIntegration:
    def test_market_probability_in_observation_types(self):
        from agent.models.gnn.graph_builder import OBSERVATION_TYPES

        assert "market_probability" in OBSERVATION_TYPES

    def test_whale_trade_in_observation_types(self):
        from agent.models.gnn.graph_builder import OBSERVATION_TYPES

        assert "whale_trade" in OBSERVATION_TYPES

    def test_topic_in_entity_types(self):
        from agent.models.gnn.graph_builder import ENTITY_TYPES

        assert "topic" in ENTITY_TYPES

    def test_wallet_in_entity_types(self):
        from agent.models.gnn.graph_builder import ENTITY_TYPES

        assert "wallet" in ENTITY_TYPES
