"""Tests for AdversarialScanner — orchestrator of all adversarial detectors."""

from __future__ import annotations

import numpy as np
import pytest

from agent.adversarial.config import (
    AdversarialConfig,
    CrowdingConfig,
    EdgeDecayConfig,
    VPINConfig,
)
from agent.adversarial.scanner import AdversarialScanner
from agent.fusion.alert import EntityAlert
from agent.fusion.convergence import ConvergenceCluster


def _make_alert(eid: str) -> EntityAlert:
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
    corr: float = 0.9,
    cluster_id: str = "c1",
) -> ConvergenceCluster:
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
def scanner() -> AdversarialScanner:
    return AdversarialScanner(
        AdversarialConfig(
            edge_decay=EdgeDecayConfig(
                rolling_window=20,
                bocpd_hazard_lambda=50.0,
                decay_threshold=0.2,
                min_history=20,
            ),
            vpin=VPINConfig(n_buckets=10, sigma_window=5, spike_threshold=0.7),
            crowding=CrowdingConfig(cluster_size_threshold=3),
        )
    )


class TestScannerEmpty:
    """Empty / minimal inputs."""

    def test_all_empty(self, scanner: AdversarialScanner):
        flags = scanner.scan(
            signal_returns={},
            market_returns=np.array([]),
            market_volumes=np.array([]),
            clusters=[],
            position_weights={},
            volume_history={},
        )
        assert flags == []

    def test_only_short_signals(self, scanner: AdversarialScanner):
        """Signals too short for edge decay → no flags from that detector."""
        flags = scanner.scan(
            signal_returns={"sig1": np.array([0.01] * 5)},
            market_returns=np.array([]),
            market_volumes=np.array([]),
            clusters=[],
            position_weights={},
            volume_history={},
        )
        assert flags == []


class TestScannerEdgeDecay:
    """Edge decay flags flow through scanner."""

    def test_decaying_signal_flagged(self, scanner: AdversarialScanner):
        rng = np.random.default_rng(42)
        n = 200
        returns = np.empty(n)
        returns[:100] = 0.02 + 0.005 * rng.standard_normal(100)
        returns[100:] = 0.0 + 0.005 * rng.standard_normal(100)

        flags = scanner.scan(
            signal_returns={"decay_sig": returns},
            market_returns=np.array([]),
            market_volumes=np.array([]),
            clusters=[],
            position_weights={},
            volume_history={},
            timestamp=42.0,
        )
        edge_flags = [f for f in flags if f.flag_type == "edge_decay"]
        assert len(edge_flags) >= 1
        assert edge_flags[0].signal_name == "decay_sig"


class TestScannerVPIN:
    """VPIN flags flow through scanner."""

    def test_one_sided_flow_flagged(self, scanner: AdversarialScanner):
        n = 100
        returns = np.full(n, 0.05)
        volumes = np.full(n, 1e6)

        flags = scanner.scan(
            signal_returns={},
            market_returns=returns,
            market_volumes=volumes,
            clusters=[],
            position_weights={},
            volume_history={},
        )
        vpin_flags = [f for f in flags if f.flag_type == "vpin_spike"]
        assert len(vpin_flags) >= 1

    def test_invalid_volumes_no_crash(self, scanner: AdversarialScanner):
        """All-zero volumes → ValueError caught internally → no crash."""
        flags = scanner.scan(
            signal_returns={},
            market_returns=np.array([0.01, 0.02, 0.03]),
            market_volumes=np.zeros(3),
            clusters=[],
            position_weights={},
            volume_history={},
        )
        # Should not crash, VPIN section skipped
        assert isinstance(flags, list)


class TestScannerCrowding:
    """Crowding flags flow through scanner."""

    def test_crowded_cluster_flagged(self, scanner: AdversarialScanner):
        cluster = _make_cluster(["a", "b", "c", "d", "e"])
        weights = {eid: 0.2 for eid in "abcde"}
        vols = {eid: np.full(20, 1.0) for eid in weights}

        flags = scanner.scan(
            signal_returns={},
            market_returns=np.array([]),
            market_volumes=np.array([]),
            clusters=[cluster],
            position_weights=weights,
            volume_history=vols,
            timestamp=55.0,
        )
        crowd_flags = [f for f in flags if f.flag_type == "crowding_risk"]
        assert len(crowd_flags) >= 1


class TestScannerCombined:
    """All three detectors producing flags simultaneously."""

    def test_all_detectors_fire(self, scanner: AdversarialScanner):
        rng = np.random.default_rng(42)

        # Edge decay signal
        n = 200
        sig_returns = np.empty(n)
        sig_returns[:100] = 0.02 + 0.005 * rng.standard_normal(100)
        sig_returns[100:] = 0.0 + 0.005 * rng.standard_normal(100)

        # VPIN one-sided
        mkt_returns = np.full(100, 0.05)
        mkt_volumes = np.full(100, 1e6)

        # Crowding cluster
        cluster = _make_cluster(["a", "b", "c", "d", "e"])
        weights = {eid: 0.2 for eid in "abcde"}
        vols = {eid: np.full(20, 1.0) for eid in weights}

        flags = scanner.scan(
            signal_returns={"decay_sig": sig_returns},
            market_returns=mkt_returns,
            market_volumes=mkt_volumes,
            clusters=[cluster],
            position_weights=weights,
            volume_history=vols,
        )

        types = {f.flag_type for f in flags}
        assert "edge_decay" in types
        assert "vpin_spike" in types
        assert "crowding_risk" in types

    def test_timestamp_propagated(self, scanner: AdversarialScanner):
        rng = np.random.default_rng(42)
        n = 200
        sig = np.empty(n)
        sig[:100] = 0.02 + 0.005 * rng.standard_normal(100)
        sig[100:] = 0.0 + 0.005 * rng.standard_normal(100)

        flags = scanner.scan(
            signal_returns={"sig": sig},
            market_returns=np.array([]),
            market_volumes=np.array([]),
            clusters=[],
            position_weights={},
            volume_history={},
            timestamp=12345.0,
        )
        for f in flags:
            assert f.timestamp == 12345.0
