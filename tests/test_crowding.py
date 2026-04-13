"""Tests for CrowdingEstimator — convergence cluster crowding risk."""

from __future__ import annotations

import numpy as np
import pytest

from agent.adversarial.config import CrowdingConfig
from agent.adversarial.crowding import CrowdingEstimator
from agent.fusion.alert import EntityAlert
from agent.fusion.convergence import ConvergenceCluster


def _make_alert(eid: str) -> EntityAlert:
    """Helper: minimal EntityAlert for testing."""
    return EntityAlert(
        entity_id=eid,
        entity_type="company",
        entity_name=f"Company {eid}",
        alert_time=1000.0,
        obs_type_surprise=3.0,
        temporal_surprise=1.0,
        value_surprise=2.0,
        neighborhood_surprise=1.5,
        memory_drift=0.5,
        composite_surprise=3.0,
        cusum_statistic=0.0,
        hawkes_intensity=0.0,
        event_study_score=0.0,
        observation_count=1,
        evidence_sources=("test",),
    )


def _make_cluster(
    entity_ids: list[str],
    corr: float = 0.8,
    cluster_id: str = "c1",
) -> ConvergenceCluster:
    """Helper: build a ConvergenceCluster with specified entities."""
    alerts = tuple(_make_alert(eid) for eid in entity_ids)
    return ConvergenceCluster(
        cluster_id=cluster_id,
        cluster_time=1000.0,
        member_alerts=alerts,
        correlated_surprise_score=corr,
        temporal_span_hours=1.0,
        contributing_domains=("test",),
        contributing_tools=("test",),
    )


@pytest.fixture
def estimator() -> CrowdingEstimator:
    return CrowdingEstimator(
        CrowdingConfig(
            cluster_size_threshold=3,
            correlation_threshold=0.7,
            volume_lookback=10,
        )
    )


class TestCrowdingBasic:
    """Core crowding risk tests."""

    def test_large_cluster_high_position_produces_flag(
        self, estimator: CrowdingEstimator
    ):
        cluster = _make_cluster(["a", "b", "c", "d", "e"], corr=0.9)
        weights = {"a": 0.3, "b": 0.2, "c": 0.1, "d": 0.05, "e": 0.05}
        # Volume scale must be comparable to weight scale for meaningful unwind risk
        vols = {eid: np.full(20, 1.0) for eid in weights}
        flags = estimator.assess([cluster], weights, vols, timestamp=99.0)
        assert len(flags) > 0
        for f in flags:
            assert f.flag_type == "crowding_risk"
            assert f.severity > 0
            assert f.timestamp == 99.0

    def test_small_cluster_no_flag(self, estimator: CrowdingEstimator):
        """Cluster with < threshold members → no flag."""
        cluster = _make_cluster(["a", "b"], corr=0.9)
        weights = {"a": 0.5, "b": 0.5}
        vols = {eid: np.full(20, 1.0) for eid in weights}
        flags = estimator.assess([cluster], weights, vols)
        assert flags == []

    def test_empty_clusters(self, estimator: CrowdingEstimator):
        flags = estimator.assess([], {}, {})
        assert flags == []

    def test_zero_positions_no_flag(self, estimator: CrowdingEstimator):
        """All weights = 0 → no unwind risk."""
        cluster = _make_cluster(["a", "b", "c", "d", "e"])
        weights = {eid: 0.0 for eid in ["a", "b", "c", "d", "e"]}
        vols = {eid: np.full(20, 1.0) for eid in weights}
        flags = estimator.assess([cluster], weights, vols)
        assert flags == []

    def test_no_volume_history_uses_default(self, estimator: CrowdingEstimator):
        """Missing volume history → default liquidity = 1.0."""
        cluster = _make_cluster(["a", "b", "c", "d", "e"], corr=0.9)
        weights = {"a": 0.5}  # Only 'a' has a position
        flags = estimator.assess([cluster], weights, {}, timestamp=0.0)
        # With liq=1 and high position, should flag
        assert len(flags) >= 1


class TestClusterCrowdingScore:
    """Tests for the cluster_crowding_score method."""

    def test_formula(self, estimator: CrowdingEstimator):
        cluster = _make_cluster(["a", "b", "c", "d", "e"], corr=0.8)
        score = estimator.cluster_crowding_score(cluster, mean_cluster_size=5.0)
        # (5 / 5.0) * 0.8 = 0.8
        assert abs(score - 0.8) < 1e-6

    def test_larger_than_mean(self, estimator: CrowdingEstimator):
        cluster = _make_cluster(
            ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"], corr=0.9
        )
        score = estimator.cluster_crowding_score(cluster, mean_cluster_size=5.0)
        # (10 / 5.0) * 0.9 = 1.8
        assert abs(score - 1.8) < 1e-6

    def test_mean_size_floor(self, estimator: CrowdingEstimator):
        """mean_cluster_size < 1 → clamped to 1."""
        cluster = _make_cluster(["a", "b", "c", "d", "e"], corr=0.6)
        score = estimator.cluster_crowding_score(cluster, mean_cluster_size=0.5)
        # (5 / 1.0) * 0.6 = 3.0
        assert abs(score - 3.0) < 1e-6

    def test_zero_correlation(self, estimator: CrowdingEstimator):
        cluster = _make_cluster(["a", "b", "c", "d", "e"], corr=0.0)
        score = estimator.cluster_crowding_score(cluster, mean_cluster_size=5.0)
        assert score == 0.0


class TestCrowdingEdgeCases:
    """Edge case coverage."""

    def test_multiple_clusters(self, estimator: CrowdingEstimator):
        c1 = _make_cluster(["a", "b", "c", "d", "e"], corr=0.9, cluster_id="c1")
        c2 = _make_cluster(["f", "g", "h", "i", "j"], corr=0.8, cluster_id="c2")
        weights = {eid: 0.1 for eid in "abcdefghij"}
        vols = {eid: np.full(20, 1.0) for eid in weights}
        flags = estimator.assess([c1, c2], weights, vols)
        cluster_ids = {f.evidence.get("cluster_id") for f in flags}
        # Both clusters should produce flags
        assert "c1" in cluster_ids
        assert "c2" in cluster_ids

    def test_entity_not_in_weights(self, estimator: CrowdingEstimator):
        """Entity in cluster but not in position_weights → skip (w=0)."""
        cluster = _make_cluster(["a", "b", "c", "d", "e"], corr=0.9)
        weights = {"a": 0.5}  # only 'a' has position
        vols = {"a": np.full(20, 1.0)}
        flags = estimator.assess([cluster], weights, vols)
        # Only 'a' should get flagged
        entity_ids = {f.entity_id for f in flags}
        assert entity_ids <= {"a"}

    def test_very_low_liquidity_high_severity(self, estimator: CrowdingEstimator):
        """Low liquidity amplifies unwind risk."""
        cluster = _make_cluster(["a", "b", "c", "d", "e"], corr=0.9)
        weights = {"a": 0.5}
        vols = {"a": np.full(20, 0.001)}  # very low volume
        flags = estimator.assess([cluster], weights, vols)
        assert len(flags) >= 1
        # Severity should be high (capped at 1.0)
        assert flags[0].severity == 1.0

    def test_negative_weights_use_abs(self, estimator: CrowdingEstimator):
        """Short positions (negative weights) should still trigger crowding."""
        cluster = _make_cluster(["a", "b", "c", "d", "e"], corr=0.9)
        weights = {"a": -0.5}
        vols = {"a": np.full(20, 1.0)}
        flags = estimator.assess([cluster], weights, vols)
        assert len(flags) >= 1
