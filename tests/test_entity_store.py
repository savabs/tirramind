"""Tests for entity registry CRUD in PipelineStore."""

from __future__ import annotations

import time

import pytest

from agent.pipeline.entity import entity_id_from_key
from agent.pipeline.store import PipelineStore


# ── Fixtures ───────────────────────────────────────────────────


@pytest.fixture
def store():
    """In-memory PipelineStore for fast tests."""
    s = PipelineStore(db_path=":memory:")
    yield s
    s.close()


# ── Entity registration ───────────────────────────────────────


class TestRegisterEntity:
    def test_register_basic(self, store: PipelineStore):
        eid = entity_id_from_key("company", "12345")
        result = store.register_entity("company", "Apple", eid)
        assert result == eid

        entity = store.get_entity(eid)
        assert entity is not None
        assert entity["entity_type"] == "company"
        assert entity["canonical_name"] == "Apple"

    def test_register_idempotent(self, store: PipelineStore):
        eid = entity_id_from_key("company", "12345")
        store.register_entity("company", "Apple", eid)
        store.register_entity("company", "Apple", eid)

        entity = store.get_entity(eid)
        assert entity is not None

    def test_register_with_metadata(self, store: PipelineStore):
        eid = entity_id_from_key("company", "99999")
        store.register_entity("company", "Tesla", eid, metadata={"sector": "auto"})
        entity = store.get_entity(eid)
        assert entity["metadata"] == {"sector": "auto"}

    def test_register_empty_name_raises(self, store: PipelineStore):
        eid = entity_id_from_key("company", "bad")
        with pytest.raises(ValueError, match="non-empty"):
            store.register_entity("company", "", eid)

    def test_register_whitespace_name_raises(self, store: PipelineStore):
        eid = entity_id_from_key("company", "bad")
        with pytest.raises(ValueError, match="non-empty"):
            store.register_entity("company", "   ", eid)

    def test_get_nonexistent_entity(self, store: PipelineStore):
        assert store.get_entity("nonexistent") is None


# ── Entity aliases ─────────────────────────────────────────────


class TestEntityAliases:
    def test_add_and_resolve(self, store: PipelineStore):
        eid = entity_id_from_key("company", "320193")
        store.register_entity("company", "apple", eid)
        store.add_entity_alias(eid, "sec_cik", "320193")

        resolved = store.resolve_entity("sec_cik", "320193")
        assert resolved == eid

    def test_resolve_nonexistent(self, store: PipelineStore):
        assert store.resolve_entity("sec_cik", "999999") is None

    def test_multiple_aliases_same_entity(self, store: PipelineStore):
        eid = entity_id_from_key("company", "320193")
        store.register_entity("company", "apple", eid)
        store.add_entity_alias(eid, "sec_cik", "320193")
        store.add_entity_alias(eid, "ticker", "AAPL")

        assert store.resolve_entity("sec_cik", "320193") == eid
        assert store.resolve_entity("ticker", "AAPL") == eid

    def test_alias_idempotent(self, store: PipelineStore):
        eid = entity_id_from_key("company", "320193")
        store.register_entity("company", "apple", eid)
        store.add_entity_alias(eid, "sec_cik", "320193")
        # Second add should be silently ignored (INSERT OR IGNORE)
        store.add_entity_alias(eid, "sec_cik", "320193")

        resolved = store.resolve_entity("sec_cik", "320193")
        assert resolved == eid

    def test_alias_unique_constraint(self, store: PipelineStore):
        """Same (source, external_id) can't map to two different entities."""
        eid1 = entity_id_from_key("company", "111")
        eid2 = entity_id_from_key("company", "222")
        store.register_entity("company", "Company A", eid1)
        store.register_entity("company", "Company B", eid2)
        store.add_entity_alias(eid1, "ticker", "AAA")
        # Second entity tries to claim the same alias — ignored
        store.add_entity_alias(eid2, "ticker", "AAA")
        # First entity keeps the alias
        assert store.resolve_entity("ticker", "AAA") == eid1


# ── Entity observations ───────────────────────────────────────


class TestEntityObservations:
    def test_store_and_query(self, store: PipelineStore):
        eid = entity_id_from_key("person", "john_doe")
        store.register_entity("person", "john doe", eid)

        row_id = store.store_entity_observation(
            entity_id=eid,
            source_tool="insider_filings",
            observed_at=1700000000.0,
            observation_type="filing",
            value={"shares": 10000, "action": "purchase"},
            depth_level=2,
        )
        assert row_id is not None

        results = store.query_entity_observations(eid)
        assert len(results) == 1
        assert results[0]["source_tool"] == "insider_filings"
        assert results[0]["value"] == {"shares": 10000, "action": "purchase"}
        assert results[0]["depth_level"] == 2

    def test_query_filters(self, store: PipelineStore):
        eid = entity_id_from_key("company", "apple")
        store.register_entity("company", "apple", eid)

        # Insert observations from different tools
        store.store_entity_observation(
            eid,
            "insider_filings",
            1700000000.0,
            "filing",
            {"shares": 100},
            depth_level=1,
        )
        store.store_entity_observation(
            eid,
            "insider_filings",
            1700001000.0,
            "filing",
            {"shares": 200},
            depth_level=2,
        )
        store.store_entity_observation(
            eid,
            "form144",
            1700002000.0,
            "sell_intent",
            {"shares": 50},
            depth_level=1,
        )

        # Filter by source_tool
        insider_only = store.query_entity_observations(
            eid, source_tool="insider_filings"
        )
        assert len(insider_only) == 2

        # Filter by depth_level
        l2_only = store.query_entity_observations(eid, depth_level=2)
        assert len(l2_only) == 1
        assert l2_only[0]["value"]["shares"] == 200

        # Filter by time range
        recent = store.query_entity_observations(eid, since=1700001500.0)
        assert len(recent) == 1
        assert recent[0]["source_tool"] == "form144"

    def test_observation_with_metadata(self, store: PipelineStore):
        eid = entity_id_from_key("vessel", "mmsi_123")
        store.register_entity("vessel", "tanker alpha", eid)

        store.store_entity_observation(
            eid,
            "ais_vessel",
            1700000000.0,
            "position",
            {"lat": 40.7, "lon": -74.0},
            metadata={"accuracy": "high"},
        )
        results = store.query_entity_observations(eid)
        assert results[0]["metadata"] == {"accuracy": "high"}

    def test_observation_default_depth_level(self, store: PipelineStore):
        eid = entity_id_from_key("wallet", "bc1q_test")
        store.register_entity("wallet", "bc1q_test", eid)

        store.store_entity_observation(
            eid,
            "whale_alert",
            1700000000.0,
            "transaction",
            {"btc": 100.5},
        )
        results = store.query_entity_observations(eid)
        assert results[0]["depth_level"] == 1

    def test_query_limit(self, store: PipelineStore):
        eid = entity_id_from_key("company", "many_obs")
        store.register_entity("company", "many obs", eid)

        for i in range(10):
            store.store_entity_observation(
                eid,
                "insider_filings",
                1700000000.0 + i,
                "filing",
                {"index": i},
            )
        results = store.query_entity_observations(eid, limit=3)
        assert len(results) == 3
        # Should be most recent first
        assert results[0]["value"]["index"] == 9


# ── Depth evaluations ─────────────────────────────────────────


class TestDepthEvaluations:
    def test_store_and_query(self, store: PipelineStore):
        row_id = store.store_depth_evaluation(
            tool_name="insider_filings",
            depth_level=2,
            target_variable="market_regime",
            sample_size=100,
            mi_gain=0.15,
            kl_divergence=0.08,
        )
        assert row_id is not None

        results = store.query_depth_evaluations("insider_filings")
        assert len(results) == 1
        assert results[0]["mi_gain"] == pytest.approx(0.15)
        assert results[0]["kl_divergence"] == pytest.approx(0.08)
        assert results[0]["sharpe_delta"] is None
        assert results[0]["sample_size"] == 100

    def test_nullable_sharpe_delta(self, store: PipelineStore):
        store.store_depth_evaluation(
            tool_name="gdelt",
            depth_level=2,
            target_variable="equity_return",
            sample_size=50,
            mi_gain=0.12,
        )
        results = store.query_depth_evaluations("gdelt")
        assert results[0]["sharpe_delta"] is None
        assert results[0]["kl_divergence"] is None

    def test_query_filters(self, store: PipelineStore):
        store.store_depth_evaluation("tool_a", 1, "target_a", 100, mi_gain=0.1)
        store.store_depth_evaluation("tool_a", 2, "target_a", 100, mi_gain=0.2)
        store.store_depth_evaluation("tool_b", 1, "target_a", 100, mi_gain=0.3)

        # Filter by tool
        assert len(store.query_depth_evaluations("tool_a")) == 2

        # Filter by depth
        assert len(store.query_depth_evaluations("tool_a", depth_level=2)) == 1

        # Filter by target
        assert len(store.query_depth_evaluations(target_variable="target_a")) == 3

    def test_query_all(self, store: PipelineStore):
        """Query with no filters returns all records."""
        store.store_depth_evaluation("tool_a", 1, "target_a", 50, mi_gain=0.1)
        store.store_depth_evaluation("tool_b", 2, "target_b", 75, mi_gain=0.2)
        results = store.query_depth_evaluations()
        assert len(results) == 2

    def test_with_metadata(self, store: PipelineStore):
        store.store_depth_evaluation(
            "insider_filings",
            2,
            "regime",
            100,
            mi_gain=0.1,
            metadata={"method": "ksg", "k": 5},
        )
        results = store.query_depth_evaluations("insider_filings")
        assert results[0]["metadata"] == {"method": "ksg", "k": 5}


# ── Schema migration (existing DBs) ───────────────────────────


class TestSchemaMigration:
    def test_entity_tables_exist(self, store: PipelineStore):
        """Verify entity tables were created by _init_schema."""
        conn = store._get_conn()
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "entities" in tables
        assert "entity_aliases" in tables
        assert "entity_observations" in tables
        assert "depth_evaluations" in tables
