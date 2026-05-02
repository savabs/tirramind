"""Tests for AdversarialFlag and adversarial config dataclasses."""

from __future__ import annotations

import pytest

from agent.adversarial.config import (
    AdversarialConfig,
    CrowdingConfig,
    EdgeDecayConfig,
    VPINConfig,
)
from agent.adversarial.flags import AdversarialFlag

# ── AdversarialFlag ──────────────────────────────────────────────────


class TestAdversarialFlag:
    def test_create_edge_decay(self):
        f = AdversarialFlag(
            flag_type="edge_decay",
            severity=0.7,
            confidence=0.8,
            signal_name="momentum",
        )
        assert f.flag_type == "edge_decay"
        assert f.severity == 0.7
        assert f.confidence == 0.8
        assert f.signal_name == "momentum"
        assert f.entity_id is None
        assert f.evidence == {}

    def test_create_vpin_spike(self):
        f = AdversarialFlag(
            flag_type="vpin_spike",
            severity=0.9,
            confidence=0.85,
            entity_id="AAPL",
            evidence={"vpin_latest": 0.9},
        )
        assert f.flag_type == "vpin_spike"
        assert f.entity_id == "AAPL"
        assert f.evidence["vpin_latest"] == 0.9

    def test_create_crowding_risk(self):
        f = AdversarialFlag(
            flag_type="crowding_risk",
            severity=0.5,
            confidence=0.6,
            entity_id="cluster_1",
        )
        assert f.flag_type == "crowding_risk"

    def test_immutable(self):
        f = AdversarialFlag(flag_type="edge_decay", severity=0.5, confidence=0.5)
        with pytest.raises(AttributeError):
            f.severity = 0.9  # type: ignore[misc]

    def test_invalid_flag_type(self):
        with pytest.raises(ValueError, match="Invalid flag_type"):
            AdversarialFlag(flag_type="unknown", severity=0.5, confidence=0.5)

    def test_severity_below_zero(self):
        with pytest.raises(ValueError, match="severity"):
            AdversarialFlag(flag_type="edge_decay", severity=-0.1, confidence=0.5)

    def test_severity_above_one(self):
        with pytest.raises(ValueError, match="severity"):
            AdversarialFlag(flag_type="edge_decay", severity=1.1, confidence=0.5)

    def test_confidence_below_zero(self):
        with pytest.raises(ValueError, match="confidence"):
            AdversarialFlag(flag_type="edge_decay", severity=0.5, confidence=-0.1)

    def test_confidence_above_one(self):
        with pytest.raises(ValueError, match="confidence"):
            AdversarialFlag(flag_type="edge_decay", severity=0.5, confidence=1.1)

    def test_severity_boundary_zero(self):
        f = AdversarialFlag(flag_type="edge_decay", severity=0.0, confidence=0.0)
        assert f.severity == 0.0
        assert f.confidence == 0.0

    def test_severity_boundary_one(self):
        f = AdversarialFlag(flag_type="edge_decay", severity=1.0, confidence=1.0)
        assert f.severity == 1.0
        assert f.confidence == 1.0

    def test_default_evidence(self):
        f = AdversarialFlag(flag_type="edge_decay", severity=0.5, confidence=0.5)
        assert f.evidence == {}

    def test_custom_timestamp(self):
        f = AdversarialFlag(flag_type="edge_decay", severity=0.5, confidence=0.5, timestamp=123.456)
        assert f.timestamp == 123.456

    def test_default_timestamp_is_float(self):
        f = AdversarialFlag(flag_type="edge_decay", severity=0.5, confidence=0.5)
        assert isinstance(f.timestamp, float)


# ── Config Dataclasses ───────────────────────────────────────────────


class TestEdgeDecayConfig:
    def test_defaults(self):
        c = EdgeDecayConfig()
        assert c.rolling_window == 52
        assert c.bocpd_hazard_lambda == 100.0
        assert c.decay_threshold == 0.5
        assert c.min_history == 52
        assert c.periods_per_year == 52

    def test_custom_values(self):
        c = EdgeDecayConfig(rolling_window=26, decay_threshold=0.3)
        assert c.rolling_window == 26
        assert c.decay_threshold == 0.3

    def test_frozen(self):
        c = EdgeDecayConfig()
        with pytest.raises(AttributeError):
            c.rolling_window = 10  # type: ignore[misc]


class TestVPINConfig:
    def test_defaults(self):
        c = VPINConfig()
        assert c.n_buckets == 50
        assert c.sigma_window == 20
        assert c.spike_threshold == 0.7

    def test_frozen(self):
        c = VPINConfig()
        with pytest.raises(AttributeError):
            c.n_buckets = 5  # type: ignore[misc]


class TestCrowdingConfig:
    def test_defaults(self):
        c = CrowdingConfig()
        assert c.cluster_size_threshold == 5
        assert c.correlation_threshold == 0.7
        assert c.volume_lookback == 20

    def test_frozen(self):
        c = CrowdingConfig()
        with pytest.raises(AttributeError):
            c.cluster_size_threshold = 1  # type: ignore[misc]


class TestAdversarialConfig:
    def test_defaults(self):
        c = AdversarialConfig()
        assert isinstance(c.edge_decay, EdgeDecayConfig)
        assert isinstance(c.vpin, VPINConfig)
        assert isinstance(c.crowding, CrowdingConfig)

    def test_override_nested(self):
        c = AdversarialConfig(
            edge_decay=EdgeDecayConfig(rolling_window=26),
            vpin=VPINConfig(n_buckets=30),
        )
        assert c.edge_decay.rolling_window == 26
        assert c.vpin.n_buckets == 30
        assert c.crowding == CrowdingConfig()

    def test_frozen(self):
        c = AdversarialConfig()
        with pytest.raises(AttributeError):
            c.edge_decay = EdgeDecayConfig()  # type: ignore[misc]
