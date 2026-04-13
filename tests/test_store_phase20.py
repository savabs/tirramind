"""Tests for PipelineStore entity_alerts + convergence_clusters tables (Phase 20, Step 20.10).

Covers:
    - Round-trip store + query for entity_alerts
    - Round-trip store + query for convergence_clusters
    - Query filters (entity_id, time range, min_composite, min_score)
    - Empty table queries
    - Duplicate cluster_id constraint
"""

from __future__ import annotations

import time

import pytest

from agent.pipeline.store import PipelineStore


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def store():
    """In-memory PipelineStore."""
    s = PipelineStore(":memory:")
    yield s
    s.close()


NOW = time.time()


# ═══════════════════════════════════════════════════════════════
# Entity Alerts
# ═══════════════════════════════════════════════════════════════


class TestEntityAlertStorage:
    def test_store_and_query(self, store):
        row_id = store.store_entity_alert(
            entity_id="c0",
            entity_type="company",
            entity_name="Acme Corp",
            alert_time=NOW,
            obs_type_surprise=2.5,
            temporal_surprise=1.2,
            value_surprise=0.8,
            neighborhood_surprise=0.5,
            memory_drift=0.1,
            cusum_statistic=3.0,
            hawkes_intensity=0.7,
            event_study_score=1.1,
            composite_surprise=5.1,
            observation_count=42,
            evidence_sources=("tool_a", "tool_b"),
        )
        assert isinstance(row_id, int)
        results = store.query_entity_alerts(entity_id="c0")
        assert len(results) == 1
        r = results[0]
        assert r["entity_id"] == "c0"
        assert r["entity_type"] == "company"
        assert r["entity_name"] == "Acme Corp"
        assert r["obs_type_surprise"] == pytest.approx(2.5)
        assert r["composite_surprise"] == pytest.approx(5.1)
        assert r["observation_count"] == 42
        assert r["evidence_sources"] == ("tool_a", "tool_b")

    def test_query_by_time_range(self, store):
        for i in range(5):
            store.store_entity_alert(
                entity_id=f"c{i}",
                entity_type="company",
                entity_name=f"Corp {i}",
                alert_time=NOW + i * 100,
                obs_type_surprise=1.0,
                temporal_surprise=1.0,
                value_surprise=1.0,
                neighborhood_surprise=0.5,
                memory_drift=0.1,
                cusum_statistic=1.0,
                hawkes_intensity=0.5,
                event_study_score=0.5,
                composite_surprise=float(i),
                observation_count=10,
            )
        results = store.query_entity_alerts(since=NOW + 150, until=NOW + 350)
        assert len(results) == 2  # c2 (NOW+200) and c3 (NOW+300)

    def test_query_min_composite(self, store):
        for i in range(5):
            store.store_entity_alert(
                entity_id=f"c{i}",
                entity_type="company",
                entity_name=f"Corp {i}",
                alert_time=NOW,
                obs_type_surprise=float(i),
                temporal_surprise=0.0,
                value_surprise=0.0,
                neighborhood_surprise=0.0,
                memory_drift=0.0,
                cusum_statistic=0.0,
                hawkes_intensity=0.0,
                event_study_score=0.0,
                composite_surprise=float(i),
                observation_count=1,
            )
        results = store.query_entity_alerts(min_composite=3.0)
        assert len(results) == 2  # c3 (3.0) and c4 (4.0)

    def test_empty_query(self, store):
        results = store.query_entity_alerts()
        assert results == []

    def test_metadata_roundtrip(self, store):
        store.store_entity_alert(
            entity_id="c0",
            entity_type="company",
            entity_name="Test",
            alert_time=NOW,
            obs_type_surprise=1.0,
            temporal_surprise=1.0,
            value_surprise=1.0,
            neighborhood_surprise=0.5,
            memory_drift=0.1,
            cusum_statistic=1.0,
            hawkes_intensity=0.5,
            event_study_score=0.5,
            composite_surprise=3.0,
            observation_count=5,
            metadata={"extra": "info"},
        )
        results = store.query_entity_alerts(entity_id="c0")
        assert results[0]["metadata"] == {"extra": "info"}

    def test_null_metadata(self, store):
        store.store_entity_alert(
            entity_id="c0",
            entity_type="company",
            entity_name="Test",
            alert_time=NOW,
            obs_type_surprise=1.0,
            temporal_surprise=1.0,
            value_surprise=1.0,
            neighborhood_surprise=0.5,
            memory_drift=0.1,
            cusum_statistic=1.0,
            hawkes_intensity=0.5,
            event_study_score=0.5,
            composite_surprise=3.0,
            observation_count=5,
        )
        results = store.query_entity_alerts(entity_id="c0")
        assert results[0]["metadata"] is None

    def test_limit(self, store):
        for i in range(10):
            store.store_entity_alert(
                entity_id="c0",
                entity_type="company",
                entity_name="Test",
                alert_time=NOW + i,
                obs_type_surprise=1.0,
                temporal_surprise=1.0,
                value_surprise=1.0,
                neighborhood_surprise=0.5,
                memory_drift=0.1,
                cusum_statistic=1.0,
                hawkes_intensity=0.5,
                event_study_score=0.5,
                composite_surprise=1.0,
                observation_count=1,
            )
        results = store.query_entity_alerts(entity_id="c0", limit=3)
        assert len(results) == 3


# ═══════════════════════════════════════════════════════════════
# Convergence Clusters
# ═══════════════════════════════════════════════════════════════


class TestConvergenceClusterStorage:
    def test_store_and_query(self, store):
        row_id = store.store_convergence_cluster(
            cluster_id="abc123",
            cluster_time=NOW,
            member_entity_ids=["c0", "c1", "c2"],
            correlated_surprise_score=0.85,
            temporal_span_hours=2.5,
            contributing_domains=("company", "wallet"),
            contributing_tools=("insider_filings", "blockchain"),
        )
        assert isinstance(row_id, int)
        results = store.query_convergence_clusters()
        assert len(results) == 1
        r = results[0]
        assert r["cluster_id"] == "abc123"
        assert r["member_entity_ids"] == ["c0", "c1", "c2"]
        assert r["correlated_surprise_score"] == pytest.approx(0.85)
        assert r["temporal_span_hours"] == pytest.approx(2.5)
        assert r["contributing_domains"] == ("company", "wallet")
        assert r["contributing_tools"] == ("insider_filings", "blockchain")

    def test_duplicate_cluster_id_fails(self, store):
        store.store_convergence_cluster(
            cluster_id="dup123",
            cluster_time=NOW,
            member_entity_ids=["c0", "c1"],
            correlated_surprise_score=0.5,
            temporal_span_hours=1.0,
        )
        import sqlite3

        with pytest.raises(sqlite3.IntegrityError):
            store.store_convergence_cluster(
                cluster_id="dup123",
                cluster_time=NOW + 100,
                member_entity_ids=["c2", "c3"],
                correlated_surprise_score=0.6,
                temporal_span_hours=2.0,
            )

    def test_query_by_time_range(self, store):
        for i in range(5):
            store.store_convergence_cluster(
                cluster_id=f"cl_{i}",
                cluster_time=NOW + i * 100,
                member_entity_ids=[f"c{i}", f"c{i+1}"],
                correlated_surprise_score=0.5,
                temporal_span_hours=1.0,
            )
        results = store.query_convergence_clusters(since=NOW + 150, until=NOW + 350)
        assert len(results) == 2

    def test_query_min_score(self, store):
        for i in range(5):
            store.store_convergence_cluster(
                cluster_id=f"cl_{i}",
                cluster_time=NOW,
                member_entity_ids=[f"c{i}", f"c{i+1}"],
                correlated_surprise_score=i * 0.2,
                temporal_span_hours=1.0,
            )
        results = store.query_convergence_clusters(min_score=0.5)
        assert len(results) == 2  # 0.6 and 0.8

    def test_empty_query(self, store):
        results = store.query_convergence_clusters()
        assert results == []

    def test_metadata_roundtrip(self, store):
        store.store_convergence_cluster(
            cluster_id="meta_test",
            cluster_time=NOW,
            member_entity_ids=["c0", "c1"],
            correlated_surprise_score=0.9,
            temporal_span_hours=1.0,
            metadata={"reason": "test"},
        )
        results = store.query_convergence_clusters()
        assert results[0]["metadata"] == {"reason": "test"}

    def test_limit(self, store):
        for i in range(10):
            store.store_convergence_cluster(
                cluster_id=f"cl_{i}",
                cluster_time=NOW + i,
                member_entity_ids=[f"c{i}", f"c{i+1}"],
                correlated_surprise_score=0.5,
                temporal_span_hours=1.0,
            )
        results = store.query_convergence_clusters(limit=3)
        assert len(results) == 3
