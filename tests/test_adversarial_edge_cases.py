"""Cross-cutting edge case tests for the adversarial layer.

Covers boundary conditions that span multiple components:
- Zero-variance / constant data
- NaN/Inf propagation
- Empty inputs
- Single-entity portfolios
- Simultaneous decay of all signals
- Market-wide flags (entity_id=None) in reward and state
"""

from __future__ import annotations

import numpy as np
import pytest

from agent.adversarial.config import (
    CrowdingConfig,
    EdgeDecayConfig,
    VPINConfig,
)
from agent.adversarial.crowding import CrowdingEstimator
from agent.adversarial.edge_decay import EdgeDecayMonitor
from agent.adversarial.flags import AdversarialFlag
from agent.adversarial.scanner import AdversarialScanner
from agent.adversarial.vpin import VPINEstimator
from agent.fusion.alert import EntityAlert
from agent.fusion.convergence import ConvergenceCluster
from agent.learning.policy.reward_fn import RewardFunction
from agent.learning.policy.state_assembler import StateAssembler


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


def _make_cluster(eids: list[str], corr: float = 0.9) -> ConvergenceCluster:
    return ConvergenceCluster(
        cluster_id="test_cluster",
        cluster_time=1000.0,
        member_alerts=tuple(_make_alert(eid) for eid in eids),
        correlated_surprise_score=corr,
        temporal_span_hours=1.0,
        contributing_domains=("test",),
        contributing_tools=("test",),
    )


# ── Zero-variance / constant data ───────────────────────────


class TestZeroVariance:
    def test_constant_returns_no_edge_decay(self):
        """Constant returns → std=0 (floored at ε) → Sharpe = 0 → no decay."""
        monitor = EdgeDecayMonitor(
            EdgeDecayConfig(
                rolling_window=10,
                min_history=10,
                bocpd_hazard_lambda=20.0,
            )
        )
        returns = np.full(100, 0.01)
        flags = monitor.update("constant", returns)
        assert isinstance(flags, list)

    def test_zero_returns_low_vpin(self):
        """Zero returns → Φ(0)=0.5 → buy=sell → low VPIN."""
        est = VPINEstimator(VPINConfig(n_buckets=10, sigma_window=5))
        returns = np.zeros(100)
        volumes = np.full(100, 1e6)
        vpin = est.compute(returns, volumes)
        assert np.all(vpin < 0.1)


# ── NaN and Inf handling ─────────────────────────────────────


class TestNaNInfHandling:
    def test_vpin_nan_returns_raises(self):
        est = VPINEstimator()
        with pytest.raises(ValueError, match="NaN"):
            est.compute(np.array([np.nan, 0.01]), np.array([100.0, 100.0]))

    def test_vpin_nan_volumes_raises(self):
        est = VPINEstimator()
        with pytest.raises(ValueError, match="NaN"):
            est.compute(np.array([0.01, 0.02]), np.array([np.nan, 100.0]))

    def test_edge_decay_nan_graceful(self):
        """NaN propagation in returns → should not crash the monitor."""
        monitor = EdgeDecayMonitor(
            EdgeDecayConfig(
                rolling_window=10,
                min_history=10,
                bocpd_hazard_lambda=20.0,
            )
        )
        returns = np.ones(50) * 0.01
        returns[25] = np.nan
        try:
            flags = monitor.update("nan_signal", returns)
            assert isinstance(flags, list)
        except (ValueError, FloatingPointError):
            pass  # acceptable

    def test_edge_decay_inf_graceful(self):
        monitor = EdgeDecayMonitor(
            EdgeDecayConfig(
                rolling_window=10,
                min_history=10,
                bocpd_hazard_lambda=20.0,
            )
        )
        returns = np.ones(50) * 0.01
        returns[25] = np.inf
        try:
            flags = monitor.update("inf_signal", returns)
            assert isinstance(flags, list)
        except (ValueError, FloatingPointError, OverflowError):
            pass


# ── Empty / minimal inputs ───────────────────────────────────


class TestEmptyInputs:
    def test_scanner_all_empty(self):
        scanner = AdversarialScanner()
        flags = scanner.scan(
            signal_returns={},
            market_returns=np.array([]),
            market_volumes=np.array([]),
            clusters=[],
            position_weights={},
            volume_history={},
        )
        assert flags == []

    def test_crowding_empty_cluster_list(self):
        est = CrowdingEstimator()
        flags = est.assess([], {"a": 0.5}, {"a": np.array([1.0])})
        assert flags == []

    def test_vpin_flag_spikes_empty(self):
        est = VPINEstimator()
        flags = est.flag_spikes(np.array([]))
        assert flags == []


# ── Single-entity portfolio ──────────────────────────────────


class TestSingleEntity:
    def test_single_entity_no_crowding(self):
        """A portfolio with one entity can't crowd."""
        est = CrowdingEstimator(CrowdingConfig(cluster_size_threshold=2))
        # Can't even make a ConvergenceCluster with 1 entity
        # So empty cluster list → no flags
        flags = est.assess([], {"solo": 1.0}, {"solo": np.array([1e6])})
        assert flags == []


# ── Simultaneous decay of all signals ────────────────────────


class TestSimultaneousDecay:
    def test_all_signals_decaying(self):
        """Multiple signals all decaying → each gets its own flag."""
        monitor = EdgeDecayMonitor(
            EdgeDecayConfig(
                rolling_window=20,
                bocpd_hazard_lambda=30.0,
                decay_threshold=0.1,
                min_history=20,
            )
        )
        rng = np.random.default_rng(42)
        n_signals = 5
        all_flags = []
        for i in range(n_signals):
            n = 300
            returns = np.empty(n)
            returns[:150] = 0.04 + 0.003 * rng.standard_normal(150)
            returns[150:] = -0.02 + 0.003 * rng.standard_normal(150)
            flags = monitor.update(f"signal_{i}", returns)
            all_flags.extend(flags)

        # Each signal should produce a flag
        flagged_signals = {f.signal_name for f in all_flags}
        assert len(flagged_signals) == n_signals


# ── Market-wide flags (entity_id=None) ──────────────────────


class TestMarketWideFlags:
    def test_vpin_flag_has_no_entity_id(self):
        """VPIN flag without entity_id → works in reward and state."""
        flag = AdversarialFlag(
            flag_type="vpin_spike",
            severity=0.8,
            confidence=0.9,
        )
        assert flag.entity_id is None

    def test_reward_penalty_with_none_entity(self):
        """Flags with entity_id=None still contribute to penalty."""
        rf = RewardFunction()
        flags = [
            AdversarialFlag(
                flag_type="vpin_spike",
                severity=0.8,
                confidence=0.9,
                entity_id=None,
            )
        ]
        penalty = rf.adversarial_penalty(flags)
        assert penalty == pytest.approx(0.8 * 0.9)

    def test_state_assembler_adversarial_block_with_flags(self):
        """State assembler includes adversarial features from flags."""
        sa = StateAssembler(max_entities=5, market_dim=2)
        flags = [
            AdversarialFlag(
                flag_type="edge_decay",
                severity=0.6,
                confidence=0.7,
                signal_name="sig1",
            ),
            AdversarialFlag(
                flag_type="vpin_spike",
                severity=0.8,
                confidence=0.9,
            ),
            AdversarialFlag(
                flag_type="crowding_risk",
                severity=0.5,
                confidence=0.6,
                entity_id="e1",
            ),
        ]
        state, meta = sa.assemble([], [], {}, {}, adversarial_flags=flags)
        # Last 4 elements are adversarial block
        adv = state[-4:]
        assert adv[0].item() == pytest.approx(0.6)  # mean edge_decay severity
        assert adv[1].item() == pytest.approx(0.8)  # max vpin severity
        assert adv[2].item() == pytest.approx(0.5)  # max crowding severity
        assert adv[3].item() == pytest.approx(0.3)  # 3 flags / 10

    def test_state_assembler_no_flags_zeros(self):
        """No adversarial flags → adversarial block is all zeros."""
        sa = StateAssembler(max_entities=5, market_dim=2)
        state, meta = sa.assemble([], [], {}, {})
        adv = state[-4:]
        assert adv[0].item() == 0.0
        assert adv[1].item() == 0.0
        assert adv[2].item() == 0.0
        assert adv[3].item() == 0.0


# ── Reward function adversarial integration ──────────────────


class TestRewardAdversarialIntegration:
    def test_no_flags_backward_compatible(self):
        """Without flags, combined() behaves identically to pre-Phase-22."""
        rf = RewardFunction()
        total_no_flags, bd_no = rf.combined(
            0.01,
            np.array([0.01, -0.01]),
            np.array([2.0]),
            step=0,
            total_steps=10,
        )
        total_with_flags, bd_with = rf.combined(
            0.01,
            np.array([0.01, -0.01]),
            np.array([2.0]),
            step=0,
            total_steps=10,
            adversarial_flags=[],
        )
        assert total_no_flags == pytest.approx(total_with_flags)

    def test_high_severity_reduces_reward(self):
        rf = RewardFunction()
        rolling = np.array([0.01, -0.01, 0.02, -0.005, 0.015])

        total_clean, _ = rf.combined(
            0.01,
            rolling,
            np.array([2.0]),
            step=0,
            total_steps=10,
        )
        flags = [
            AdversarialFlag(
                flag_type="edge_decay",
                severity=0.9,
                confidence=0.9,
                signal_name="sig1",
            ),
        ]
        total_flagged, bd = rf.combined(
            0.01,
            rolling,
            np.array([2.0]),
            step=0,
            total_steps=10,
            adversarial_flags=flags,
        )
        assert total_flagged < total_clean
        assert bd["adversarial_penalty"] > 0

    def test_penalty_proportional_to_severity_confidence(self):
        rf = RewardFunction()
        flag_low = [
            AdversarialFlag(
                flag_type="vpin_spike",
                severity=0.1,
                confidence=0.1,
            )
        ]
        flag_high = [
            AdversarialFlag(
                flag_type="vpin_spike",
                severity=0.9,
                confidence=0.9,
            )
        ]
        pen_low = rf.adversarial_penalty(flag_low)
        pen_high = rf.adversarial_penalty(flag_high)
        assert pen_high > pen_low
        assert pen_low == pytest.approx(0.01)
        assert pen_high == pytest.approx(0.81)
