"""Tests for Evidence dataclass and EvidenceBus (Phase 7c-A.1)."""

from __future__ import annotations

import math

import pytest

from agent.convergence.evidence import VALID_CATEGORIES, Evidence, EvidenceBus

# ── Helpers ────────────────────────────────────────────────────


def _make_evidence(**overrides) -> Evidence:
    """Build a valid Evidence with sensible defaults, overridable."""
    defaults = dict(
        source="cftc",
        signal_id="cftc.crude_oil.mm_net_long",
        timestamp=1712000000.0,
        value=42.0,
        direction=1,
        confidence=0.8,
        category="positioning",
        tags=("commodity", "energy"),
        ttl=86400,
    )
    defaults.update(overrides)
    return Evidence(**defaults)


# ── Evidence construction ─────────────────────────────────────


class TestEvidenceConstruction:
    def test_valid_construction(self):
        e = _make_evidence()
        assert e.source == "cftc"
        assert e.signal_id == "cftc.crude_oil.mm_net_long"
        assert e.timestamp == 1712000000.0
        assert e.value == 42.0
        assert e.direction == 1
        assert e.confidence == 0.8
        assert e.category == "positioning"
        assert e.tags == ("commodity", "energy")
        assert e.ttl == 86400

    def test_all_directions(self):
        for d in (-1, 0, 1):
            e = _make_evidence(direction=d)
            assert e.direction == d

    def test_nan_value_valid(self):
        e = _make_evidence(value=float("nan"))
        assert math.isnan(e.value)

    def test_inf_value_valid(self):
        e = _make_evidence(value=float("inf"))
        assert math.isinf(e.value)

    def test_negative_value_valid(self):
        e = _make_evidence(value=-999.5)
        assert e.value == -999.5

    def test_zero_value_valid(self):
        e = _make_evidence(value=0.0)
        assert e.value == 0.0

    def test_empty_tags_valid(self):
        e = _make_evidence(tags=())
        assert e.tags == ()

    def test_all_categories_accepted(self):
        for cat in sorted(VALID_CATEGORIES):
            e = _make_evidence(category=cat)
            assert e.category == cat

    def test_confidence_boundary_zero(self):
        e = _make_evidence(confidence=0.0)
        assert e.confidence == 0.0

    def test_confidence_boundary_one(self):
        e = _make_evidence(confidence=1.0)
        assert e.confidence == 1.0

    def test_ttl_one_valid(self):
        e = _make_evidence(ttl=1)
        assert e.ttl == 1

    def test_large_timestamp_valid(self):
        # Year ~2100
        e = _make_evidence(timestamp=4102444800.0)
        assert e.timestamp == 4102444800.0


# ── Evidence validation (should reject) ───────────────────────


class TestEvidenceValidation:
    def test_empty_signal_id_rejected(self):
        with pytest.raises(ValueError, match="signal_id must be non-empty"):
            _make_evidence(signal_id="")

    def test_zero_timestamp_rejected(self):
        with pytest.raises(ValueError, match="timestamp must be positive"):
            _make_evidence(timestamp=0)

    def test_negative_timestamp_rejected(self):
        with pytest.raises(ValueError, match="timestamp must be positive"):
            _make_evidence(timestamp=-1.0)

    def test_confidence_below_zero_rejected(self):
        with pytest.raises(ValueError, match="confidence must be in"):
            _make_evidence(confidence=-0.01)

    def test_confidence_above_one_rejected(self):
        with pytest.raises(ValueError, match="confidence must be in"):
            _make_evidence(confidence=1.01)

    def test_invalid_direction_rejected(self):
        with pytest.raises(ValueError, match="direction must be"):
            _make_evidence(direction=2)

    def test_direction_float_rejected(self):
        with pytest.raises(ValueError, match="direction must be"):
            _make_evidence(direction=0.5)

    def test_invalid_category_rejected(self):
        with pytest.raises(ValueError, match="category must be one of"):
            _make_evidence(category="unknown_category")

    def test_category_case_sensitive(self):
        with pytest.raises(ValueError, match="category must be one of"):
            _make_evidence(category="Positioning")

    def test_zero_ttl_rejected(self):
        with pytest.raises(ValueError, match="ttl must be positive"):
            _make_evidence(ttl=0)

    def test_negative_ttl_rejected(self):
        with pytest.raises(ValueError, match="ttl must be positive"):
            _make_evidence(ttl=-100)


# ── Evidence immutability ─────────────────────────────────────


class TestEvidenceImmutability:
    def test_frozen(self):
        e = _make_evidence()
        with pytest.raises(AttributeError):
            e.value = 99.0  # type: ignore[misc]

    def test_frozen_tags(self):
        e = _make_evidence()
        with pytest.raises(AttributeError):
            e.tags = ("new",)  # type: ignore[misc]


# ── VALID_CATEGORIES ──────────────────────────────────────────


class TestCategories:
    def test_count(self):
        assert len(VALID_CATEGORIES) == 11

    def test_expected_members(self):
        expected = {
            "physical_flow",
            "physical_disruption",
            "financial_stress",
            "monetary_policy",
            "regulatory_action",
            "behavioral_intent",
            "positioning",
            "macro_momentum",
            "biological",
            "geopolitical",
            "supply_chain",
        }
        assert expected == VALID_CATEGORIES

    def test_frozenset(self):
        assert isinstance(VALID_CATEGORIES, frozenset)


# ── EvidenceBus ───────────────────────────────────────────────


class TestEvidenceBus:
    def test_empty_bus(self):
        bus = EvidenceBus()
        assert len(bus) == 0
        assert not bus
        assert bus.flush() == []
        assert bus.snapshot() == []

    def test_submit_and_flush(self):
        bus = EvidenceBus()
        e1 = _make_evidence(signal_id="a")
        e2 = _make_evidence(signal_id="b")
        bus.submit(e1)
        bus.submit(e2)
        assert len(bus) == 2
        assert bool(bus)

        flushed = bus.flush()
        assert len(flushed) == 2
        assert flushed[0].signal_id == "a"
        assert flushed[1].signal_id == "b"

        # Bus should be empty after flush
        assert len(bus) == 0
        assert bus.flush() == []

    def test_snapshot_does_not_clear(self):
        bus = EvidenceBus()
        bus.submit(_make_evidence(signal_id="x"))
        snap = bus.snapshot()
        assert len(snap) == 1
        assert len(bus) == 1  # Still there

    def test_flush_returns_independent_list(self):
        bus = EvidenceBus()
        bus.submit(_make_evidence())
        flushed = bus.flush()
        # Mutating flushed should not affect bus internals
        flushed.append(_make_evidence(signal_id="extra"))
        assert len(bus) == 0

    def test_snapshot_returns_independent_list(self):
        bus = EvidenceBus()
        bus.submit(_make_evidence())
        snap = bus.snapshot()
        snap.append(_make_evidence(signal_id="extra"))
        assert len(bus) == 1

    def test_submit_rejects_non_evidence(self):
        bus = EvidenceBus()
        with pytest.raises(TypeError, match="Expected Evidence"):
            bus.submit({"source": "fake"})  # type: ignore[arg-type]

    def test_submit_rejects_none(self):
        bus = EvidenceBus()
        with pytest.raises(TypeError, match="Expected Evidence"):
            bus.submit(None)  # type: ignore[arg-type]

    def test_repr(self):
        bus = EvidenceBus()
        assert "0 items" in repr(bus)
        bus.submit(_make_evidence())
        assert "1 items" in repr(bus)

    def test_large_bus_performance(self):
        """10,000 items should still work without issue."""
        bus = EvidenceBus()
        for i in range(10_000):
            bus.submit(_make_evidence(signal_id=f"sig_{i}"))
        assert len(bus) == 10_000
        flushed = bus.flush()
        assert len(flushed) == 10_000
        assert len(bus) == 0

    def test_multiple_flush_cycles(self):
        bus = EvidenceBus()
        bus.submit(_make_evidence(signal_id="cycle1"))
        first = bus.flush()
        assert len(first) == 1

        bus.submit(_make_evidence(signal_id="cycle2a"))
        bus.submit(_make_evidence(signal_id="cycle2b"))
        second = bus.flush()
        assert len(second) == 2
        assert second[0].signal_id == "cycle2a"

    def test_submit_after_flush(self):
        bus = EvidenceBus()
        bus.submit(_make_evidence(signal_id="before"))
        bus.flush()
        bus.submit(_make_evidence(signal_id="after"))
        assert len(bus) == 1
        assert bus.snapshot()[0].signal_id == "after"
