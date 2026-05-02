"""Tests for ConvergenceCluster dataclass — construction, validation, edge cases."""

from __future__ import annotations

import pytest

from agent.fusion.alert import EntityAlert
from agent.fusion.convergence import ConvergenceCluster

# ── Helpers ────────────────────────────────────────────────────

_CLUSTER_TIME = 1714000000.0


def _make_alert(**overrides) -> EntityAlert:
    defaults = dict(
        entity_id="ent_001",
        entity_type="company",
        entity_name="Acme Corp",
        alert_time=_CLUSTER_TIME,
        obs_type_surprise=1.5,
        temporal_surprise=0.8,
        value_surprise=2.1,
        neighborhood_surprise=0.3,
        memory_drift=0.05,
        cusum_statistic=3.2,
        hawkes_intensity=0.7,
        event_study_score=1.1,
        composite_surprise=1.34,
        observation_count=42,
        evidence_sources=("insider_filings",),
        metadata=None,
    )
    defaults.update(overrides)
    return EntityAlert(**defaults)


def _make_cluster(**overrides) -> ConvergenceCluster:
    alert_a = _make_alert(
        entity_id="ent_001",
        entity_type="person",
        entity_name="Jane Doe",
        alert_time=_CLUSTER_TIME - 3600,
    )
    alert_b = _make_alert(
        entity_id="ent_002",
        entity_type="company",
        entity_name="Acme Corp",
        alert_time=_CLUSTER_TIME,
    )
    defaults = dict(
        cluster_id="clust_abc123",
        cluster_time=_CLUSTER_TIME,
        member_alerts=(alert_a, alert_b),
        correlated_surprise_score=0.85,
        temporal_span_hours=1.0,
        contributing_domains=("insider_filings", "patent_filings"),
        contributing_tools=("insider_filings_tool", "patent_filings_tool"),
        metadata=None,
    )
    defaults.update(overrides)
    return ConvergenceCluster(**defaults)


# ── Basic construction ─────────────────────────────────────────


class TestConvergenceClusterConstruction:
    def test_create_valid_cluster(self) -> None:
        cluster = _make_cluster()
        assert cluster.cluster_id == "clust_abc123"
        assert cluster.cluster_time == _CLUSTER_TIME
        assert len(cluster.member_alerts) == 2
        assert cluster.correlated_surprise_score == 0.85
        assert cluster.temporal_span_hours == 1.0
        assert cluster.contributing_domains == ("insider_filings", "patent_filings")
        assert cluster.contributing_tools == (
            "insider_filings_tool",
            "patent_filings_tool",
        )
        assert cluster.metadata is None

    def test_three_members(self) -> None:
        alert_c = _make_alert(entity_id="ent_003", entity_type="wallet", entity_name="0xdeadbeef")
        cluster = _make_cluster(
            member_alerts=(
                _make_alert(entity_id="ent_001"),
                _make_alert(entity_id="ent_002"),
                alert_c,
            ),
            contributing_domains=("filings", "blockchain", "dns"),
        )
        assert len(cluster.member_alerts) == 3
        assert len(cluster.contributing_domains) == 3

    def test_large_cluster(self) -> None:
        alerts = tuple(_make_alert(entity_id=f"ent_{i:03d}", entity_name=f"Entity {i}") for i in range(10))
        cluster = _make_cluster(member_alerts=alerts)
        assert len(cluster.member_alerts) == 10

    def test_metadata_dict(self) -> None:
        cluster = _make_cluster(metadata={"reason": "test", "score_breakdown": [0.5, 0.3]})
        assert cluster.metadata["reason"] == "test"


# ── Validation: minimum 2 members ─────────────────────────────


class TestConvergenceClusterValidation:
    def test_single_member_raises(self) -> None:
        single = (_make_alert(),)
        with pytest.raises(ValueError, match="requires >= 2"):
            _make_cluster(member_alerts=single)

    def test_empty_members_raises(self) -> None:
        with pytest.raises(ValueError, match="requires >= 2"):
            _make_cluster(member_alerts=())

    def test_two_members_ok(self) -> None:
        cluster = _make_cluster()  # default has 2
        assert len(cluster.member_alerts) == 2


# ── Immutability ───────────────────────────────────────────────


class TestConvergenceClusterImmutability:
    def test_frozen_cluster_id(self) -> None:
        cluster = _make_cluster()
        with pytest.raises(AttributeError):
            cluster.cluster_id = "changed"  # type: ignore[misc]

    def test_frozen_score(self) -> None:
        cluster = _make_cluster()
        with pytest.raises(AttributeError):
            cluster.correlated_surprise_score = 0.0  # type: ignore[misc]

    def test_frozen_member_alerts(self) -> None:
        cluster = _make_cluster()
        with pytest.raises(AttributeError):
            cluster.member_alerts = ()  # type: ignore[misc]


# ── Edge cases: surprise scores ────────────────────────────────


class TestConvergenceClusterScores:
    def test_zero_correlated_surprise(self) -> None:
        """Zero correlation is valid — members surprised in orthogonal dimensions."""
        cluster = _make_cluster(correlated_surprise_score=0.0)
        assert cluster.correlated_surprise_score == 0.0

    def test_negative_correlation(self) -> None:
        """Negative cosine similarity is valid (anti-correlated surprise)."""
        cluster = _make_cluster(correlated_surprise_score=-0.3)
        assert cluster.correlated_surprise_score == -0.3

    def test_perfect_correlation(self) -> None:
        cluster = _make_cluster(correlated_surprise_score=1.0)
        assert cluster.correlated_surprise_score == 1.0

    def test_zero_temporal_span(self) -> None:
        """Zero span = all members alerted at same instant."""
        cluster = _make_cluster(temporal_span_hours=0.0)
        assert cluster.temporal_span_hours == 0.0

    def test_large_temporal_span(self) -> None:
        cluster = _make_cluster(temporal_span_hours=168.0)  # 1 week
        assert cluster.temporal_span_hours == 168.0


# ── Edge cases: domains / tools ────────────────────────────────


class TestConvergenceClusterDomains:
    def test_empty_domains(self) -> None:
        """Empty contributing_domains is valid (descriptive only)."""
        cluster = _make_cluster(contributing_domains=())
        assert cluster.contributing_domains == ()

    def test_empty_tools(self) -> None:
        cluster = _make_cluster(contributing_tools=())
        assert cluster.contributing_tools == ()

    def test_many_domains(self) -> None:
        domains = tuple(f"domain_{i}" for i in range(20))
        cluster = _make_cluster(contributing_domains=domains)
        assert len(cluster.contributing_domains) == 20

    def test_duplicate_domains_allowed(self) -> None:
        """Duplicates are allowed — caller is responsible for dedup if needed."""
        cluster = _make_cluster(contributing_domains=("gdelt", "gdelt"))
        assert len(cluster.contributing_domains) == 2


# ── Equality / hashing ─────────────────────────────────────────


class TestConvergenceClusterEquality:
    def test_equal_clusters(self) -> None:
        a = _make_cluster()
        b = _make_cluster()
        assert a == b

    def test_different_cluster_id(self) -> None:
        a = _make_cluster(cluster_id="a")
        b = _make_cluster(cluster_id="b")
        assert a != b

    def test_hashable_without_metadata(self) -> None:
        cluster = _make_cluster(metadata=None)
        assert isinstance(hash(cluster), int)


# ── Mixed entity types ─────────────────────────────────────────


class TestConvergenceClusterMixedTypes:
    def test_person_company_cluster(self) -> None:
        cluster = _make_cluster()
        types = {a.entity_type for a in cluster.member_alerts}
        assert "person" in types
        assert "company" in types

    def test_wallet_vessel_cluster(self) -> None:
        a1 = _make_alert(entity_id="w1", entity_type="wallet", entity_name="0xabc")
        a2 = _make_alert(entity_id="v1", entity_type="vessel", entity_name="EVER GIVEN")
        cluster = _make_cluster(member_alerts=(a1, a2))
        types = {a.entity_type for a in cluster.member_alerts}
        assert types == {"wallet", "vessel"}

    def test_all_same_type_valid(self) -> None:
        """Two entities of the same type can still form a cluster."""
        a1 = _make_alert(entity_id="c1", entity_type="company", entity_name="Corp A")
        a2 = _make_alert(entity_id="c2", entity_type="company", entity_name="Corp B")
        cluster = _make_cluster(member_alerts=(a1, a2))
        assert len(cluster.member_alerts) == 2
