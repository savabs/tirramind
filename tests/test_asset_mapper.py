"""Tests for AssetMapper — entity-to-ticker resolution."""

import pytest

from agent.learning.policy.asset_mapper import AssetMapper
from agent.pipeline.store import PipelineStore


@pytest.fixture()
def store() -> PipelineStore:
    """In-memory PipelineStore seeded with entities + aliases."""
    s = PipelineStore(db_path=":memory:")
    # Companies with tickers
    s.register_entity("company", "Apple Inc", "ent-aapl")
    s.add_entity_alias("ent-aapl", "ticker", "AAPL")
    s.register_entity("company", "Microsoft Corp", "ent-msft")
    s.add_entity_alias("ent-msft", "ticker", "MSFT")
    # Company without ticker
    s.register_entity("company", "Private Co", "ent-priv")
    # Non-company with ticker alias (e.g., crypto)
    s.register_entity("wallet", "Big Whale Wallet", "ent-btc")
    s.add_entity_alias("ent-btc", "ticker", "BTC-USD")
    # Person (not tradeable)
    s.register_entity("person", "Tim Cook", "ent-ceo")
    return s


class TestResolve:
    def test_resolve_known_ticker(self, store: PipelineStore) -> None:
        mapper = AssetMapper(store)
        assert mapper.resolve("ent-aapl") == "AAPL"

    def test_resolve_no_ticker(self, store: PipelineStore) -> None:
        mapper = AssetMapper(store)
        assert mapper.resolve("ent-priv") is None

    def test_resolve_person(self, store: PipelineStore) -> None:
        mapper = AssetMapper(store)
        assert mapper.resolve("ent-ceo") is None

    def test_resolve_nonexistent(self, store: PipelineStore) -> None:
        mapper = AssetMapper(store)
        assert mapper.resolve("ent-nonexistent") is None

    def test_resolve_wallet_with_ticker(self, store: PipelineStore) -> None:
        """Wallets can have ticker aliases (crypto tickers)."""
        mapper = AssetMapper(store)
        assert mapper.resolve("ent-btc") == "BTC-USD"

    def test_caching(self, store: PipelineStore) -> None:
        """Second call hits cache, not store."""
        mapper = AssetMapper(store)
        assert mapper.resolve("ent-aapl") == "AAPL"
        # Mutate the store underneath — mapper should return cached value
        assert mapper.resolve("ent-aapl") == "AAPL"
        assert "ent-aapl" in mapper._cache


class TestResolveBatch:
    def test_batch_mixed(self, store: PipelineStore) -> None:
        mapper = AssetMapper(store)
        result = mapper.resolve_batch(["ent-aapl", "ent-priv", "ent-msft", "ent-ceo"])
        assert result == {"ent-aapl": "AAPL", "ent-msft": "MSFT"}

    def test_batch_empty(self, store: PipelineStore) -> None:
        mapper = AssetMapper(store)
        assert mapper.resolve_batch([]) == {}

    def test_batch_all_untradeable(self, store: PipelineStore) -> None:
        mapper = AssetMapper(store)
        assert mapper.resolve_batch(["ent-priv", "ent-ceo"]) == {}


class TestTradeableEntities:
    def test_returns_all_with_tickers(self, store: PipelineStore) -> None:
        mapper = AssetMapper(store)
        te = mapper.tradeable_entities()
        assert "ent-aapl" in te
        assert "ent-msft" in te
        assert "ent-btc" in te
        assert "ent-priv" not in te
        assert "ent-ceo" not in te

    def test_cached_after_first_call(self, store: PipelineStore) -> None:
        mapper = AssetMapper(store)
        te1 = mapper.tradeable_entities()
        te2 = mapper.tradeable_entities()
        assert te1 == te2

    def test_clear_cache(self, store: PipelineStore) -> None:
        mapper = AssetMapper(store)
        mapper.tradeable_entities()
        mapper.clear_cache()
        assert mapper._all_tradeable is None
        assert mapper._cache == {}


class TestEmptyStore:
    def test_resolve_empty(self) -> None:
        s = PipelineStore(db_path=":memory:")
        mapper = AssetMapper(s)
        assert mapper.resolve("anything") is None

    def test_tradeable_empty(self) -> None:
        s = PipelineStore(db_path=":memory:")
        mapper = AssetMapper(s)
        assert mapper.tradeable_entities() == {}
