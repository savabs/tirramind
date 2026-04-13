"""Walk-forward validation for the adversarial layer.

Tests that each detector correctly identifies planted signals
on synthetic data, and does NOT flag clean data (false positive control).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from agent.adversarial.config import (
    AdversarialConfig,
    CrowdingConfig,
    EdgeDecayConfig,
    VPINConfig,
)
from agent.adversarial.crowding import CrowdingEstimator
from agent.adversarial.edge_decay import EdgeDecayMonitor
from agent.adversarial.scanner import AdversarialScanner
from agent.adversarial.vpin import VPINEstimator
from agent.fusion.alert import EntityAlert
from agent.fusion.convergence import ConvergenceCluster


def _make_alert(eid: str, composite: float = 3.0) -> EntityAlert:
    return EntityAlert(
        entity_id=eid,
        entity_type="company",
        entity_name=f"Company {eid}",
        alert_time=1000.0,
        obs_type_surprise=composite,
        temporal_surprise=1.0,
        value_surprise=2.0,
        neighborhood_surprise=1.5,
        memory_drift=0.5,
        composite_surprise=composite,
        cusum_statistic=0.0,
        hawkes_intensity=0.0,
        event_study_score=0.0,
        observation_count=1,
        evidence_sources=("test",),
    )


def _make_cluster(
    eids: list[str],
    corr: float = 0.9,
    cluster_id: str = "c1",
) -> ConvergenceCluster:
    return ConvergenceCluster(
        cluster_id=cluster_id,
        cluster_time=1000.0,
        member_alerts=tuple(_make_alert(eid) for eid in eids),
        correlated_surprise_score=corr,
        temporal_span_hours=1.0,
        contributing_domains=("test",),
        contributing_tools=("test",),
    )


# ── Scenario 1: Planted edge decay ──────────────────────────


class TestPlantedEdgeDecay:
    """Signal with known Sharpe drop → detected within k periods."""

    def test_strong_decay_detected(self):
        monitor = EdgeDecayMonitor(
            EdgeDecayConfig(
                rolling_window=20,
                bocpd_hazard_lambda=30.0,
                decay_threshold=0.1,
                min_history=20,
                periods_per_year=52,
            )
        )
        rng = np.random.default_rng(42)
        n = 300

        # Strong signal → no signal
        returns = np.empty(n)
        returns[:150] = 0.04 + 0.003 * rng.standard_normal(150)
        returns[150:] = -0.02 + 0.003 * rng.standard_normal(150)

        flags = monitor.update("planted_decay", returns, timestamp=100.0)
        assert len(flags) >= 1
        assert flags[0].flag_type == "edge_decay"
        assert flags[0].severity > 0.1

    def test_gradual_decay_detected(self):
        """Sharpe linearly decaying to zero → eventually flagged."""
        monitor = EdgeDecayMonitor(
            EdgeDecayConfig(
                rolling_window=20,
                bocpd_hazard_lambda=30.0,
                decay_threshold=0.1,
                min_history=20,
            )
        )
        rng = np.random.default_rng(42)
        n = 400
        # Signal strength decays linearly across the series
        strength = np.linspace(0.04, -0.01, n)
        returns = strength + 0.003 * rng.standard_normal(n)

        flags = monitor.update("gradual_decay", returns)
        assert len(flags) >= 1

    def test_stable_signal_no_decay(self):
        """Constant Sharpe signal → no false positive."""
        monitor = EdgeDecayMonitor(
            EdgeDecayConfig(
                rolling_window=20,
                bocpd_hazard_lambda=30.0,
                decay_threshold=0.5,  # higher threshold for strict false-positive control
                min_history=20,
            )
        )
        rng = np.random.default_rng(42)
        returns = 0.01 + 0.005 * rng.standard_normal(300)
        flags = monitor.update("stable_signal", returns)
        assert flags == []


# ── Scenario 2: Planted VPIN spike ──────────────────────────


class TestPlantedVPINSpike:
    """One-sided order flow burst → VPIN spikes above threshold."""

    def test_one_sided_buy_burst(self):
        est = VPINEstimator(
            VPINConfig(
                n_buckets=10,
                sigma_window=5,
                spike_threshold=0.6,
            )
        )
        rng = np.random.default_rng(42)
        n = 100

        # Normal period followed by all-buy burst
        returns = np.empty(n)
        returns[:70] = 0.001 * rng.standard_normal(70)
        returns[70:] = 0.05 + 0.001 * rng.standard_normal(30)
        volumes = np.full(n, 1e6)

        vpin = est.compute(returns, volumes)
        flags = est.flag_spikes(vpin, entity_id="SPY", timestamp=999.0)
        assert len(flags) >= 1
        assert flags[0].flag_type == "vpin_spike"
        assert flags[0].evidence["vpin_latest"] > 0.6

    def test_one_sided_sell_burst(self):
        est = VPINEstimator(
            VPINConfig(
                n_buckets=10,
                sigma_window=5,
                spike_threshold=0.6,
            )
        )
        rng = np.random.default_rng(42)
        n = 100

        returns = np.empty(n)
        returns[:70] = 0.001 * rng.standard_normal(70)
        returns[70:] = -0.05 + 0.001 * rng.standard_normal(30)
        volumes = np.full(n, 1e6)

        vpin = est.compute(returns, volumes)
        flags = est.flag_spikes(vpin)
        assert len(flags) >= 1

    def test_balanced_flow_no_spike(self):
        """Symmetric random walk → no VPIN spike (false positive control)."""
        est = VPINEstimator(
            VPINConfig(
                n_buckets=20,
                sigma_window=10,
                spike_threshold=0.7,
            )
        )
        rng = np.random.default_rng(42)
        returns = 0.001 * rng.standard_normal(200)
        volumes = np.full(200, 1e6)

        vpin = est.compute(returns, volumes)
        flags = est.flag_spikes(vpin)
        assert flags == []


# ── Scenario 3: Planted crowding ─────────────────────────────


class TestPlantedCrowding:
    """Large dense cluster with high positions → crowding flagged."""

    def test_dense_cluster_high_position(self):
        est = CrowdingEstimator(
            CrowdingConfig(
                cluster_size_threshold=3,
                volume_lookback=10,
            )
        )
        eids = [f"e{i}" for i in range(8)]
        cluster = _make_cluster(eids, corr=0.95)
        weights = {eid: 0.3 for eid in eids}
        vols = {eid: np.full(20, 1.0) for eid in eids}

        flags = est.assess([cluster], weights, vols)
        assert len(flags) >= 1
        for f in flags:
            assert f.flag_type == "crowding_risk"
            assert f.severity > 0

    def test_sparse_cluster_no_crowding(self):
        """Small cluster with low correlation → no flag."""
        est = CrowdingEstimator(
            CrowdingConfig(
                cluster_size_threshold=5,
            )
        )
        cluster = _make_cluster(["a", "b", "c"], corr=0.3)
        weights = {"a": 0.1, "b": 0.1, "c": 0.1}
        vols = {eid: np.full(20, 1.0) for eid in weights}

        flags = est.assess([cluster], weights, vols)
        assert flags == []


# ── Scenario 4: False positive control on clean data ────────


class TestCleanDataNoFlags:
    """Normal (non-manipulated) synthetic data → no flags."""

    def test_scanner_clean_data(self):
        scanner = AdversarialScanner(
            AdversarialConfig(
                edge_decay=EdgeDecayConfig(
                    rolling_window=20,
                    bocpd_hazard_lambda=50.0,
                    decay_threshold=0.5,
                    min_history=20,
                ),
                vpin=VPINConfig(n_buckets=20, sigma_window=10, spike_threshold=0.7),
                crowding=CrowdingConfig(cluster_size_threshold=5),
            )
        )
        rng = np.random.default_rng(42)

        # Clean signals: stable Sharpe
        n = 200
        signal_returns = {
            f"sig_{i}": 0.01 + 0.005 * rng.standard_normal(n) for i in range(3)
        }

        # Clean market: symmetric random walk
        market_returns = 0.001 * rng.standard_normal(n)
        market_volumes = np.full(n, 1e6)

        # No clusters, no positions
        flags = scanner.scan(
            signal_returns=signal_returns,
            market_returns=market_returns,
            market_volumes=market_volumes,
            clusters=[],
            position_weights={},
            volume_history={},
        )
        assert flags == []


# ── Scenario 5: Full scanner integration on synthetic data ──


class TestFullScannerIntegration:
    """End-to-end scanner with planted abnormalities in all channels."""

    def test_all_channels_fire(self):
        scanner = AdversarialScanner(
            AdversarialConfig(
                edge_decay=EdgeDecayConfig(
                    rolling_window=20,
                    bocpd_hazard_lambda=30.0,
                    decay_threshold=0.1,
                    min_history=20,
                ),
                vpin=VPINConfig(n_buckets=10, sigma_window=5, spike_threshold=0.6),
                crowding=CrowdingConfig(cluster_size_threshold=3),
            )
        )
        rng = np.random.default_rng(42)

        # Edge decay signal
        n = 300
        sig_returns = np.empty(n)
        sig_returns[:150] = 0.04 + 0.003 * rng.standard_normal(150)
        sig_returns[150:] = -0.02 + 0.003 * rng.standard_normal(150)

        # VPIN: one-sided flow
        mkt_n = 100
        mkt_returns = np.full(mkt_n, 0.05)
        mkt_volumes = np.full(mkt_n, 1e6)

        # Crowding: large cluster
        eids = [f"e{i}" for i in range(6)]
        cluster = _make_cluster(eids, corr=0.95)
        weights = {eid: 0.2 for eid in eids}
        vols = {eid: np.full(20, 1.0) for eid in eids}

        flags = scanner.scan(
            signal_returns={"decaying": sig_returns},
            market_returns=mkt_returns,
            market_volumes=mkt_volumes,
            clusters=[cluster],
            position_weights=weights,
            volume_history=vols,
            timestamp=12345.0,
        )

        types = {f.flag_type for f in flags}
        assert "edge_decay" in types, f"Missing edge_decay, got {types}"
        assert "vpin_spike" in types, f"Missing vpin_spike, got {types}"
        assert "crowding_risk" in types, f"Missing crowding_risk, got {types}"

        # All flags have the propagated timestamp
        for f in flags:
            assert f.timestamp == 12345.0

    def test_result_count_reasonable(self):
        """Ensure no flag explosion from valid inputs."""
        scanner = AdversarialScanner(
            AdversarialConfig(
                edge_decay=EdgeDecayConfig(
                    rolling_window=20,
                    min_history=20,
                    bocpd_hazard_lambda=30.0,
                ),
                crowding=CrowdingConfig(cluster_size_threshold=3),
            )
        )
        rng = np.random.default_rng(42)

        # 10 signals
        sig_returns = {
            f"sig_{i}": 0.01 + 0.01 * rng.standard_normal(200) for i in range(10)
        }

        flags = scanner.scan(
            signal_returns=sig_returns,
            market_returns=0.001 * rng.standard_normal(200),
            market_volumes=np.full(200, 1e6),
            clusters=[],
            position_weights={},
            volume_history={},
        )
        # At most 10 edge decay + 1 VPIN + 0 crowding
        assert len(flags) <= 11
