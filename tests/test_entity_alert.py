"""Tests for EntityAlert dataclass — construction, immutability, edge cases."""

from __future__ import annotations

import pytest

from agent.fusion.alert import EntityAlert

# ── Helpers ────────────────────────────────────────────────────

_ALERT_TIME = 1714000000.0


def _make_alert(**overrides) -> EntityAlert:
    defaults = dict(
        entity_id="ent_001",
        entity_type="company",
        entity_name="Acme Corp",
        alert_time=_ALERT_TIME,
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
        evidence_sources=("insider_filings", "patent_filings"),
        metadata=None,
    )
    defaults.update(overrides)
    return EntityAlert(**defaults)


# ── Basic construction ─────────────────────────────────────────


class TestEntityAlertConstruction:
    def test_create_valid_alert(self) -> None:
        alert = _make_alert()
        assert alert.entity_id == "ent_001"
        assert alert.entity_type == "company"
        assert alert.entity_name == "Acme Corp"
        assert alert.alert_time == _ALERT_TIME
        assert alert.obs_type_surprise == 1.5
        assert alert.temporal_surprise == 0.8
        assert alert.value_surprise == 2.1
        assert alert.neighborhood_surprise == 0.3
        assert alert.memory_drift == 0.05
        assert alert.cusum_statistic == 3.2
        assert alert.hawkes_intensity == 0.7
        assert alert.event_study_score == 1.1
        assert alert.composite_surprise == 1.34
        assert alert.observation_count == 42
        assert alert.evidence_sources == ("insider_filings", "patent_filings")
        assert alert.metadata is None

    def test_all_entity_types(self) -> None:
        """Same dataclass handles all entity types without code branches."""
        for etype in (
            "person",
            "company",
            "wallet",
            "vessel",
            "country",
            "domain",
            "protocol",
            "facility",
        ):
            alert = _make_alert(entity_type=etype, entity_id=f"ent_{etype}")
            assert alert.entity_type == etype

    def test_metadata_dict(self) -> None:
        alert = _make_alert(metadata={"source": "test", "version": 2})
        assert alert.metadata == {"source": "test", "version": 2}

    def test_empty_evidence_sources(self) -> None:
        alert = _make_alert(evidence_sources=())
        assert alert.evidence_sources == ()

    def test_single_evidence_source(self) -> None:
        alert = _make_alert(evidence_sources=("gdelt",))
        assert len(alert.evidence_sources) == 1


# ── Immutability ───────────────────────────────────────────────


class TestEntityAlertImmutability:
    def test_frozen_field_raises(self) -> None:
        alert = _make_alert()
        with pytest.raises(AttributeError):
            alert.entity_id = "changed"  # type: ignore[misc]

    def test_frozen_surprise_field(self) -> None:
        alert = _make_alert()
        with pytest.raises(AttributeError):
            alert.obs_type_surprise = 99.0  # type: ignore[misc]

    def test_frozen_composite(self) -> None:
        alert = _make_alert()
        with pytest.raises(AttributeError):
            alert.composite_surprise = 0.0  # type: ignore[misc]

    def test_frozen_metadata(self) -> None:
        alert = _make_alert()
        with pytest.raises(AttributeError):
            alert.metadata = {"hack": True}  # type: ignore[misc]


# ── Edge cases: surprise signal values ─────────────────────────


class TestEntityAlertSurpriseValues:
    def test_zero_surprise_all_fields(self) -> None:
        """Zero surprise is valid — entity behaves exactly as predicted."""
        alert = _make_alert(
            obs_type_surprise=0.0,
            temporal_surprise=0.0,
            value_surprise=0.0,
            neighborhood_surprise=0.0,
            memory_drift=0.0,
            composite_surprise=0.0,
        )
        assert alert.composite_surprise == 0.0

    def test_negative_surprise_valid(self) -> None:
        """Negative values valid — e.g. log-prob can be negative, z-scores can be negative."""
        alert = _make_alert(
            obs_type_surprise=-0.5,
            temporal_surprise=-1.2,
            value_surprise=-0.1,
        )
        assert alert.obs_type_surprise == -0.5

    def test_large_surprise_values(self) -> None:
        """Very large surprise values are valid (outliers)."""
        alert = _make_alert(
            obs_type_surprise=100.0,
            temporal_surprise=50.0,
            value_surprise=200.0,
            neighborhood_surprise=75.0,
            memory_drift=30.0,
            composite_surprise=91.0,
        )
        assert alert.composite_surprise == 91.0

    def test_inf_surprise(self) -> None:
        """Inf is technically valid for -log(0) obs_type surprise."""
        import math

        alert = _make_alert(obs_type_surprise=math.inf)
        assert math.isinf(alert.obs_type_surprise)


# ── Edge cases: enrichment features ───────────────────────────


class TestEntityAlertEnrichment:
    def test_zero_enrichment(self) -> None:
        alert = _make_alert(
            cusum_statistic=0.0,
            hawkes_intensity=0.0,
            event_study_score=0.0,
        )
        assert alert.cusum_statistic == 0.0

    def test_high_cusum(self) -> None:
        alert = _make_alert(cusum_statistic=50.0)
        assert alert.cusum_statistic == 50.0

    def test_negative_event_study(self) -> None:
        """Negative abnormal score = below baseline is valid."""
        alert = _make_alert(event_study_score=-2.5)
        assert alert.event_study_score == -2.5


# ── Edge cases: observation count ──────────────────────────────


class TestEntityAlertObservationCount:
    def test_zero_observations(self) -> None:
        alert = _make_alert(observation_count=0)
        assert alert.observation_count == 0

    def test_one_observation(self) -> None:
        alert = _make_alert(observation_count=1)
        assert alert.observation_count == 1

    def test_large_observation_count(self) -> None:
        alert = _make_alert(observation_count=1_000_000)
        assert alert.observation_count == 1_000_000


# ── Equality / hashing ─────────────────────────────────────────


class TestEntityAlertEquality:
    def test_equal_alerts(self) -> None:
        a = _make_alert()
        b = _make_alert()
        assert a == b

    def test_different_entity_id(self) -> None:
        a = _make_alert(entity_id="a")
        b = _make_alert(entity_id="b")
        assert a != b

    def test_hashable(self) -> None:
        """Frozen dataclass should be hashable (when metadata is None)."""
        alert = _make_alert(metadata=None)
        assert isinstance(hash(alert), int)

    def test_hashable_with_metadata_fails(self) -> None:
        """Dict is unhashable so alert with metadata should not be hashable."""
        alert = _make_alert(metadata={"k": "v"})
        with pytest.raises(TypeError):
            hash(alert)

    def test_set_of_alerts(self) -> None:
        """Can use alerts (without metadata) in sets."""
        a = _make_alert(entity_id="a")
        b = _make_alert(entity_id="b")
        c = _make_alert(entity_id="a")  # same as a
        s = {a, b, c}
        assert len(s) == 2
