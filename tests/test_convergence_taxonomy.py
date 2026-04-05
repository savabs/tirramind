"""Tests for signal taxonomy and SignalRegistry (Phase 7c-A.2)."""

from __future__ import annotations

import pytest

from agent.convergence.taxonomy import (
    CATEGORIES,
    VALID_FREQUENCIES,
    SignalMeta,
    SignalRegistry,
)


# ── Helpers ────────────────────────────────────────────────────


def _make_meta(**overrides) -> SignalMeta:
    """Build a valid SignalMeta with sensible defaults, overridable."""
    defaults = dict(
        signal_id="cftc.crude_oil.mm_net_long",
        source="cftc",
        category="positioning",
        frequency="weekly",
        direction_semantics="higher = more speculative longs",
        flip_sign=False,
        default_ttl=604_800,  # 7 days
        min_observations=30,
    )
    defaults.update(overrides)
    return SignalMeta(**defaults)


# ── CATEGORIES constant ───────────────────────────────────────


class TestCategories:
    def test_is_frozenset(self):
        assert isinstance(CATEGORIES, frozenset)

    def test_count(self):
        assert len(CATEGORIES) == 11

    def test_all_expected_present(self):
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
        assert CATEGORIES == expected

    def test_no_whitespace_in_names(self):
        for cat in CATEGORIES:
            assert cat == cat.strip()
            assert " " not in cat


# ── VALID_FREQUENCIES constant ────────────────────────────────


class TestFrequencies:
    def test_is_frozenset(self):
        assert isinstance(VALID_FREQUENCIES, frozenset)

    def test_count(self):
        assert len(VALID_FREQUENCIES) == 5

    def test_all_expected_present(self):
        expected = {"intraday", "daily", "weekly", "monthly", "event"}
        assert VALID_FREQUENCIES == expected


# ── SignalMeta construction ───────────────────────────────────


class TestSignalMetaConstruction:
    def test_valid_construction(self):
        m = _make_meta()
        assert m.signal_id == "cftc.crude_oil.mm_net_long"
        assert m.source == "cftc"
        assert m.category == "positioning"
        assert m.frequency == "weekly"
        assert m.flip_sign is False
        assert m.default_ttl == 604_800
        assert m.min_observations == 30

    def test_all_categories_accepted(self):
        for cat in sorted(CATEGORIES):
            m = _make_meta(signal_id=f"test.{cat}", category=cat)
            assert m.category == cat

    def test_all_frequencies_accepted(self):
        for freq in sorted(VALID_FREQUENCIES):
            m = _make_meta(signal_id=f"test.{freq}", frequency=freq)
            assert m.frequency == freq

    def test_flip_sign_true(self):
        m = _make_meta(flip_sign=True)
        assert m.flip_sign is True

    def test_default_ttl_custom(self):
        m = _make_meta(default_ttl=3600)
        assert m.default_ttl == 3600

    def test_min_observations_one(self):
        m = _make_meta(min_observations=1)
        assert m.min_observations == 1

    def test_frozen(self):
        m = _make_meta()
        with pytest.raises(AttributeError):
            m.signal_id = "changed"  # type: ignore[misc]


# ── SignalMeta validation ─────────────────────────────────────


class TestSignalMetaValidation:
    def test_empty_signal_id_rejected(self):
        with pytest.raises(ValueError, match="signal_id must be non-empty"):
            _make_meta(signal_id="")

    def test_empty_source_rejected(self):
        with pytest.raises(ValueError, match="source must be non-empty"):
            _make_meta(source="")

    def test_invalid_category_rejected(self):
        with pytest.raises(ValueError, match="category must be one of"):
            _make_meta(category="invented_category")

    def test_category_case_sensitive(self):
        with pytest.raises(ValueError, match="category must be one of"):
            _make_meta(category="Positioning")

    def test_invalid_frequency_rejected(self):
        with pytest.raises(ValueError, match="frequency must be one of"):
            _make_meta(frequency="yearly")

    def test_frequency_case_sensitive(self):
        with pytest.raises(ValueError, match="frequency must be one of"):
            _make_meta(frequency="Daily")

    def test_zero_ttl_rejected(self):
        with pytest.raises(ValueError, match="default_ttl must be positive"):
            _make_meta(default_ttl=0)

    def test_negative_ttl_rejected(self):
        with pytest.raises(ValueError, match="default_ttl must be positive"):
            _make_meta(default_ttl=-1)

    def test_zero_min_observations_rejected(self):
        with pytest.raises(ValueError, match="min_observations must be >= 1"):
            _make_meta(min_observations=0)

    def test_negative_min_observations_rejected(self):
        with pytest.raises(ValueError, match="min_observations must be >= 1"):
            _make_meta(min_observations=-5)


# ── SignalRegistry — registration ─────────────────────────────


class TestSignalRegistryRegister:
    def test_register_one(self):
        reg = SignalRegistry()
        reg.register(_make_meta())
        assert len(reg) == 1

    def test_register_multiple(self):
        reg = SignalRegistry()
        reg.register(_make_meta(signal_id="a"))
        reg.register(_make_meta(signal_id="b"))
        reg.register(_make_meta(signal_id="c"))
        assert len(reg) == 3

    def test_duplicate_signal_id_rejected(self):
        reg = SignalRegistry()
        reg.register(_make_meta(signal_id="dup"))
        with pytest.raises(ValueError, match="Duplicate signal_id"):
            reg.register(_make_meta(signal_id="dup"))

    def test_same_source_different_ids_ok(self):
        reg = SignalRegistry()
        reg.register(_make_meta(signal_id="cftc.oil", source="cftc"))
        reg.register(_make_meta(signal_id="cftc.gas", source="cftc"))
        assert len(reg) == 2

    def test_rejects_non_signal_meta(self):
        reg = SignalRegistry()
        with pytest.raises(TypeError, match="Expected SignalMeta"):
            reg.register({"signal_id": "fake"})  # type: ignore[arg-type]


# ── SignalRegistry — lookup ───────────────────────────────────


class TestSignalRegistryLookup:
    @pytest.fixture
    def populated_registry(self) -> SignalRegistry:
        reg = SignalRegistry()
        reg.register(_make_meta(
            signal_id="cftc.oil.net_long", source="cftc",
            category="positioning", frequency="weekly",
        ))
        reg.register(_make_meta(
            signal_id="cftc.gas.net_long", source="cftc",
            category="positioning", frequency="weekly",
        ))
        reg.register(_make_meta(
            signal_id="weather.us.alert_count", source="weather_alerts",
            category="physical_disruption", frequency="daily",
        ))
        reg.register(_make_meta(
            signal_id="pmi.us.manufacturing", source="global_pmi",
            category="macro_momentum", frequency="monthly",
        ))
        reg.register(_make_meta(
            signal_id="earthquake.global.mag_max", source="earthquake",
            category="physical_disruption", frequency="event",
        ))
        return reg

    def test_get_existing(self, populated_registry: SignalRegistry):
        m = populated_registry.get("cftc.oil.net_long")
        assert m is not None
        assert m.source == "cftc"

    def test_get_nonexistent(self, populated_registry: SignalRegistry):
        assert populated_registry.get("nonexistent") is None

    def test_by_source_cftc(self, populated_registry: SignalRegistry):
        metas = populated_registry.by_source("cftc")
        assert len(metas) == 2
        ids = {m.signal_id for m in metas}
        assert ids == {"cftc.oil.net_long", "cftc.gas.net_long"}

    def test_by_source_unknown(self, populated_registry: SignalRegistry):
        assert populated_registry.by_source("unknown_tool") == []

    def test_by_category_physical_disruption(self, populated_registry: SignalRegistry):
        metas = populated_registry.by_category("physical_disruption")
        assert len(metas) == 2
        ids = {m.signal_id for m in metas}
        assert "weather.us.alert_count" in ids
        assert "earthquake.global.mag_max" in ids

    def test_by_category_empty(self, populated_registry: SignalRegistry):
        assert populated_registry.by_category("biological") == []

    def test_all_ids_sorted(self, populated_registry: SignalRegistry):
        ids = populated_registry.all_ids()
        assert ids == sorted(ids)
        assert len(ids) == 5

    def test_contains(self, populated_registry: SignalRegistry):
        assert "cftc.oil.net_long" in populated_registry
        assert "nonexistent" not in populated_registry

    def test_frequencies_grouping(self, populated_registry: SignalRegistry):
        freqs = populated_registry.frequencies()
        assert "weekly" in freqs
        assert len(freqs["weekly"]) == 2
        assert "daily" in freqs
        assert len(freqs["daily"]) == 1
        assert "monthly" in freqs
        assert len(freqs["monthly"]) == 1
        assert "event" in freqs
        assert len(freqs["event"]) == 1

    def test_by_source_returns_copy(self, populated_registry: SignalRegistry):
        """Mutating returned list should not affect internal state."""
        metas = populated_registry.by_source("cftc")
        metas.clear()
        assert len(populated_registry.by_source("cftc")) == 2

    def test_by_category_returns_copy(self, populated_registry: SignalRegistry):
        metas = populated_registry.by_category("positioning")
        metas.clear()
        assert len(populated_registry.by_category("positioning")) == 2


# ── SignalRegistry — repr / empty ─────────────────────────────


class TestSignalRegistryMisc:
    def test_empty_registry(self):
        reg = SignalRegistry()
        assert len(reg) == 0
        assert reg.all_ids() == []
        assert reg.frequencies() == {}

    def test_repr(self):
        reg = SignalRegistry()
        assert "0 signals" in repr(reg)
        reg.register(_make_meta(signal_id="x"))
        assert "1 signals" in repr(reg)

    def test_large_registry_performance(self):
        """500 signals should register and query without issue."""
        reg = SignalRegistry()
        cats = sorted(CATEGORIES)
        freqs = sorted(VALID_FREQUENCIES)
        for i in range(500):
            reg.register(_make_meta(
                signal_id=f"tool_{i}.sig",
                source=f"tool_{i % 20}",
                category=cats[i % len(cats)],
                frequency=freqs[i % len(freqs)],
            ))
        assert len(reg) == 500
        assert len(reg.all_ids()) == 500
        # Lookup still works
        assert reg.get("tool_0.sig") is not None
        assert reg.get("tool_499.sig") is not None
        # Grouping works
        assert sum(len(v) for v in reg.frequencies().values()) == 500
