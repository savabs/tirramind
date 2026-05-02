"""Comprehensive edge-case test suite for Sub-phase 7c-A (Evidence, Taxonomy, Extractors).

Phase 7c-A.5: Stress tests covering NaN values, boundary confidence, bus at scale,
deeply nested data, wrong types, case-sensitive categories, numeric strings,
extra/missing keys, and cross-module integration.
"""

from __future__ import annotations

import math
import time

import pytest

from agent.convergence.evidence import VALID_CATEGORIES, Evidence, EvidenceBus
from agent.convergence.extractors import (
    _REGISTRY,
    _safe_float,
    _safe_int,
    extract_evidence,
    registered_tools,
)
from agent.convergence.taxonomy import (
    CATEGORIES,
    VALID_FREQUENCIES,
    SignalMeta,
    SignalRegistry,
)

# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════


def _make_evidence(**overrides) -> Evidence:
    defaults = dict(
        source="test",
        signal_id="test.signal",
        timestamp=1712000000.0,
        value=1.0,
        direction=0,
        confidence=0.5,
        category="positioning",
        tags=(),
        ttl=3600,
    )
    defaults.update(overrides)
    return Evidence(**defaults)


def _make_meta(**overrides) -> SignalMeta:
    defaults = dict(
        signal_id="test.meta",
        source="test",
        category="positioning",
        frequency="daily",
        direction_semantics="higher = more",
    )
    defaults.update(overrides)
    return SignalMeta(**defaults)


# ══════════════════════════════════════════════════════════════
#  EVIDENCE EDGE CASES
# ══════════════════════════════════════════════════════════════


class TestEvidenceNaNAndSpecialFloats:
    """NaN/Inf/negative-zero in value field."""

    def test_nan_value(self):
        e = _make_evidence(value=float("nan"))
        assert math.isnan(e.value)

    def test_positive_inf_value(self):
        e = _make_evidence(value=float("inf"))
        assert e.value == float("inf")

    def test_negative_inf_value(self):
        e = _make_evidence(value=float("-inf"))
        assert e.value == float("-inf")

    def test_negative_zero_value(self):
        e = _make_evidence(value=-0.0)
        assert e.value == 0.0  # -0.0 == 0.0 in Python

    def test_very_small_positive_value(self):
        e = _make_evidence(value=1e-308)
        assert e.value == 1e-308

    def test_very_large_value(self):
        e = _make_evidence(value=1e308)
        assert e.value == 1e308

    def test_nan_confidence_rejected(self):
        """NaN is not in [0, 1]."""
        with pytest.raises(ValueError, match="confidence"):
            _make_evidence(confidence=float("nan"))

    def test_inf_confidence_rejected(self):
        with pytest.raises(ValueError, match="confidence"):
            _make_evidence(confidence=float("inf"))

    def test_negative_inf_confidence_rejected(self):
        with pytest.raises(ValueError, match="confidence"):
            _make_evidence(confidence=float("-inf"))


class TestEvidenceBoundaryConfidence:
    """Boundary values around 0.0 and 1.0."""

    def test_confidence_exactly_zero(self):
        e = _make_evidence(confidence=0.0)
        assert e.confidence == 0.0

    def test_confidence_exactly_one(self):
        e = _make_evidence(confidence=1.0)
        assert e.confidence == 1.0

    def test_confidence_just_above_zero(self):
        e = _make_evidence(confidence=1e-15)
        assert e.confidence > 0.0

    def test_confidence_just_below_one(self):
        e = _make_evidence(confidence=1.0 - 1e-15)
        assert e.confidence < 1.0

    def test_confidence_tiny_negative_rejected(self):
        with pytest.raises(ValueError, match="confidence"):
            _make_evidence(confidence=-1e-15)

    def test_confidence_slightly_above_one_rejected(self):
        with pytest.raises(ValueError, match="confidence"):
            _make_evidence(confidence=1.0 + 1e-10)


class TestEvidenceTimestampEdge:
    """Timestamp boundaries."""

    def test_tiny_positive_timestamp(self):
        e = _make_evidence(timestamp=0.001)
        assert e.timestamp == 0.001

    def test_epoch_one_valid(self):
        e = _make_evidence(timestamp=1.0)
        assert e.timestamp == 1.0

    def test_very_large_timestamp(self):
        """Year ~3000."""
        e = _make_evidence(timestamp=32503680000.0)
        assert e.timestamp > 0

    def test_float_zero_rejected(self):
        with pytest.raises(ValueError, match="timestamp"):
            _make_evidence(timestamp=0.0)


class TestEvidenceTagsEdge:
    """Tags tuple boundaries."""

    def test_empty_tags(self):
        e = _make_evidence(tags=())
        assert e.tags == ()

    def test_single_tag(self):
        e = _make_evidence(tags=("us",))
        assert len(e.tags) == 1

    def test_many_tags(self):
        tags = tuple(f"tag_{i}" for i in range(100))
        e = _make_evidence(tags=tags)
        assert len(e.tags) == 100

    def test_unicode_tags(self):
        e = _make_evidence(tags=("日本", "дата", "émission"))
        assert "日本" in e.tags

    def test_empty_string_tag(self):
        """Empty string tag is technically valid (tags are metadata labels)."""
        e = _make_evidence(tags=("",))
        assert e.tags == ("",)


class TestEvidenceSignalIdEdge:
    """Signal ID validation."""

    def test_whitespace_only_signal_id_not_empty(self):
        """Whitespace-only is non-empty string, so it passes validation."""
        e = _make_evidence(signal_id="   ")
        assert e.signal_id == "   "

    def test_very_long_signal_id(self):
        long_id = "a" * 1000
        e = _make_evidence(signal_id=long_id)
        assert len(e.signal_id) == 1000

    def test_special_chars_signal_id(self):
        e = _make_evidence(signal_id="tool.metric/sub:type#1")
        assert e.signal_id == "tool.metric/sub:type#1"


class TestEvidenceTtlEdge:
    """TTL validation."""

    def test_ttl_one(self):
        e = _make_evidence(ttl=1)
        assert e.ttl == 1

    def test_ttl_very_large(self):
        e = _make_evidence(ttl=10**9)
        assert e.ttl == 10**9

    def test_ttl_negative_rejected(self):
        with pytest.raises(ValueError, match="ttl"):
            _make_evidence(ttl=-1)


# ══════════════════════════════════════════════════════════════
#  EVIDENCE BUS EDGE CASES
# ══════════════════════════════════════════════════════════════


class TestEvidenceBusEdge:
    """Advanced bus behavior."""

    def test_10k_items(self):
        bus = EvidenceBus()
        for i in range(10_000):
            bus.submit(_make_evidence(signal_id=f"s{i}"))
        assert len(bus) == 10_000
        snap = bus.snapshot()
        assert len(snap) == 10_000
        flushed = bus.flush()
        assert len(flushed) == 10_000
        assert len(bus) == 0

    def test_alternating_submit_flush(self):
        """Interleaved submit/flush cycles."""
        bus = EvidenceBus()
        for cycle in range(50):
            bus.submit(_make_evidence(signal_id=f"c{cycle}"))
            items = bus.flush()
            assert len(items) == 1
            assert items[0].signal_id == f"c{cycle}"
        assert len(bus) == 0

    def test_submit_string_rejected(self):
        bus = EvidenceBus()
        with pytest.raises(TypeError, match="Expected Evidence"):
            bus.submit("not evidence")  # type: ignore[arg-type]

    def test_submit_dict_rejected(self):
        bus = EvidenceBus()
        with pytest.raises(TypeError, match="Expected Evidence"):
            bus.submit({"source": "fake"})  # type: ignore[arg-type]

    def test_submit_int_rejected(self):
        bus = EvidenceBus()
        with pytest.raises(TypeError, match="Expected Evidence"):
            bus.submit(42)  # type: ignore[arg-type]

    def test_submit_list_rejected(self):
        bus = EvidenceBus()
        with pytest.raises(TypeError, match="Expected Evidence"):
            bus.submit([_make_evidence()])  # type: ignore[arg-type]

    def test_flush_returns_order_preserved(self):
        bus = EvidenceBus()
        ids = [f"ordered_{i}" for i in range(100)]
        for sid in ids:
            bus.submit(_make_evidence(signal_id=sid))
        flushed = bus.flush()
        assert [e.signal_id for e in flushed] == ids

    def test_snapshot_independent_of_future_submits(self):
        bus = EvidenceBus()
        bus.submit(_make_evidence(signal_id="a"))
        snap = bus.snapshot()
        bus.submit(_make_evidence(signal_id="b"))
        assert len(snap) == 1  # snap not affected

    def test_empty_bus_bool_false(self):
        bus = EvidenceBus()
        assert not bus
        assert bool(bus) is False

    def test_nonempty_bus_bool_true(self):
        bus = EvidenceBus()
        bus.submit(_make_evidence())
        assert bus
        assert bool(bus) is True


# ══════════════════════════════════════════════════════════════
#  TAXONOMY EDGE CASES
# ══════════════════════════════════════════════════════════════


class TestSignalMetaEdge:
    """SignalMeta edge cases."""

    def test_flip_sign_both_values(self):
        for flip in (True, False):
            m = _make_meta(flip_sign=flip)
            assert m.flip_sign is flip

    def test_min_observations_exactly_one(self):
        m = _make_meta(min_observations=1)
        assert m.min_observations == 1

    def test_min_observations_zero_rejected(self):
        with pytest.raises(ValueError, match="min_observations"):
            _make_meta(min_observations=0)

    def test_min_observations_negative_rejected(self):
        with pytest.raises(ValueError, match="min_observations"):
            _make_meta(min_observations=-10)

    def test_very_large_ttl(self):
        m = _make_meta(default_ttl=10**9)
        assert m.default_ttl == 10**9

    def test_category_with_extra_underscore_rejected(self):
        with pytest.raises(ValueError, match="category"):
            _make_meta(category="physical_flow_")

    def test_category_with_prefix_rejected(self):
        with pytest.raises(ValueError, match="category"):
            _make_meta(category="_positioning")

    def test_frequency_uppercase_rejected(self):
        with pytest.raises(ValueError, match="frequency"):
            _make_meta(frequency="DAILY")

    def test_frequency_mixed_case_rejected(self):
        with pytest.raises(ValueError, match="frequency"):
            _make_meta(frequency="Weekly")

    def test_unicode_signal_id_valid(self):
        m = _make_meta(signal_id="日本.pmi.manufacturing")
        assert "日本" in m.signal_id

    def test_long_direction_semantics(self):
        desc = "x" * 500
        m = _make_meta(direction_semantics=desc)
        assert len(m.direction_semantics) == 500


class TestSignalRegistryEdge:
    """Registry edge cases."""

    def test_register_none_rejected(self):
        reg = SignalRegistry()
        with pytest.raises(TypeError, match="Expected SignalMeta"):
            reg.register(None)  # type: ignore[arg-type]

    def test_register_evidence_rejected(self):
        """Evidence is not SignalMeta."""
        reg = SignalRegistry()
        e = _make_evidence()
        with pytest.raises(TypeError, match="Expected SignalMeta"):
            reg.register(e)  # type: ignore[arg-type]

    def test_duplicate_after_different(self):
        reg = SignalRegistry()
        reg.register(_make_meta(signal_id="a"))
        reg.register(_make_meta(signal_id="b"))
        with pytest.raises(ValueError, match="Duplicate"):
            reg.register(_make_meta(signal_id="a"))

    def test_empty_registry_queries(self):
        reg = SignalRegistry()
        assert reg.get("anything") is None
        assert reg.by_source("x") == []
        assert reg.by_category("positioning") == []
        assert reg.all_ids() == []
        assert reg.frequencies() == {}
        assert len(reg) == 0

    def test_500_signal_registry(self):
        """Scale test: 500 signals."""
        reg = SignalRegistry()
        cats = sorted(CATEGORIES)
        freqs = sorted(VALID_FREQUENCIES)
        for i in range(500):
            reg.register(
                _make_meta(
                    signal_id=f"s{i}",
                    source=f"tool_{i % 25}",
                    category=cats[i % len(cats)],
                    frequency=freqs[i % len(freqs)],
                )
            )
        assert len(reg) == 500
        assert len(reg.all_ids()) == 500
        # by_source returns correct count
        assert len(reg.by_source("tool_0")) == 500 // 25
        # contains works
        assert "s0" in reg
        assert "s499" in reg
        assert "s500" not in reg
        # frequencies groups correctly
        freq_map = reg.frequencies()
        total = sum(len(v) for v in freq_map.values())
        assert total == 500

    def test_by_source_returns_independent_copy(self):
        reg = SignalRegistry()
        reg.register(_make_meta(signal_id="a", source="x"))
        result = reg.by_source("x")
        result.clear()
        assert len(reg.by_source("x")) == 1

    def test_by_category_returns_independent_copy(self):
        reg = SignalRegistry()
        reg.register(_make_meta(signal_id="a", category="positioning"))
        result = reg.by_category("positioning")
        result.clear()
        assert len(reg.by_category("positioning")) == 1


class TestCategoryCrossModuleConsistency:
    """Verify CATEGORIES is the same object in evidence.py and taxonomy.py."""

    def test_categories_same_contents(self):
        assert VALID_CATEGORIES == CATEGORIES

    def test_categories_count_11(self):
        assert len(CATEGORIES) == 11
        assert len(VALID_CATEGORIES) == 11

    def test_all_categories_lowercase_underscore(self):
        import re

        for cat in CATEGORIES:
            assert re.match(r"^[a-z_]+$", cat), f"Bad category format: {cat!r}"


# ══════════════════════════════════════════════════════════════
#  _safe_float / _safe_int EDGE CASES
# ══════════════════════════════════════════════════════════════


class TestSafeFloat:
    """Exhaustive edge cases for _safe_float."""

    def test_none(self):
        assert _safe_float(None) == 0.0

    def test_none_custom_default(self):
        assert _safe_float(None, -1.0) == -1.0

    def test_int(self):
        assert _safe_float(42) == 42.0

    def test_float(self):
        assert _safe_float(3.14) == 3.14

    def test_string_number(self):
        assert _safe_float("3.14") == 3.14

    def test_string_negative(self):
        assert _safe_float("-100.5") == -100.5

    def test_string_scientific(self):
        assert _safe_float("1e10") == 1e10

    def test_string_empty(self):
        assert _safe_float("") == 0.0

    def test_string_garbage(self):
        assert _safe_float("abc") == 0.0

    def test_string_nan(self):
        result = _safe_float("nan")
        assert math.isnan(result)

    def test_string_inf(self):
        assert _safe_float("inf") == float("inf")

    def test_bool_true(self):
        assert _safe_float(True) == 1.0

    def test_bool_false(self):
        assert _safe_float(False) == 0.0

    def test_list_returns_default(self):
        assert _safe_float([1, 2]) == 0.0

    def test_dict_returns_default(self):
        assert _safe_float({"a": 1}) == 0.0

    def test_zero(self):
        assert _safe_float(0) == 0.0

    def test_negative_zero(self):
        assert _safe_float(-0.0) == 0.0


class TestSafeInt:
    """Exhaustive edge cases for _safe_int."""

    def test_none(self):
        assert _safe_int(None) == 0

    def test_none_custom_default(self):
        assert _safe_int(None, -1) == -1

    def test_int(self):
        assert _safe_int(42) == 42

    def test_float_truncates(self):
        assert _safe_int(3.9) == 3

    def test_string_int(self):
        assert _safe_int("99") == 99

    def test_string_float_rejects(self):
        """int("3.14") raises ValueError."""
        assert _safe_int("3.14") == 0

    def test_string_empty(self):
        assert _safe_int("") == 0

    def test_string_garbage(self):
        assert _safe_int("abc") == 0

    def test_bool_true(self):
        assert _safe_int(True) == 1

    def test_bool_false(self):
        assert _safe_int(False) == 0

    def test_list_returns_default(self):
        assert _safe_int([1]) == 0

    def test_dict_returns_default(self):
        assert _safe_int({"a": 1}) == 0

    def test_negative(self):
        assert _safe_int(-50) == -50

    def test_zero(self):
        assert _safe_int(0) == 0


# ══════════════════════════════════════════════════════════════
#  EXTRACTOR EDGE CASES — DEEPLY NESTED / WRONG TYPES / MISSING KEYS
# ══════════════════════════════════════════════════════════════


class TestExtractorNoneAndBadTypes:
    """extract_evidence should never raise, always return []."""

    def test_none_data(self):
        for tool in registered_tools():
            result = extract_evidence(tool, None)
            assert result == [], f"{tool} returned non-empty for None"

    def test_empty_dict(self):
        for tool in registered_tools():
            result = extract_evidence(tool, {})
            assert isinstance(result, list), f"{tool} returned non-list for {{}}"

    def test_empty_list(self):
        for tool in registered_tools():
            result = extract_evidence(tool, [])
            assert isinstance(result, list)

    def test_string_data(self):
        for tool in registered_tools():
            result = extract_evidence(tool, "garbage")
            assert isinstance(result, list)

    def test_int_data(self):
        for tool in registered_tools():
            result = extract_evidence(tool, 42)
            assert isinstance(result, list)

    def test_bool_data(self):
        for tool in registered_tools():
            result = extract_evidence(tool, True)
            assert isinstance(result, list)

    def test_nested_none_values(self):
        """Dict with all keys set to None."""
        data = {"contracts": None, "alerts": None, "entries": None, "signals": None}
        for tool in registered_tools():
            result = extract_evidence(tool, data)
            assert isinstance(result, list)

    def test_unregistered_tool_returns_empty(self):
        assert extract_evidence("nonexistent_tool_xyz", {"data": 1}) == []

    def test_float_data(self):
        for tool in registered_tools():
            result = extract_evidence(tool, 3.14)
            assert isinstance(result, list)


class TestExtractorDeeplyNested:
    """Deeply nested or unusual data structures."""

    def test_deeply_nested_dict(self):
        """50-level nesting."""
        data: dict = {}
        current = data
        for i in range(50):
            current[f"level_{i}"] = {}
            current = current[f"level_{i}"]
        current["value"] = 42
        for tool in registered_tools():
            result = extract_evidence(tool, data)
            assert isinstance(result, list)

    def test_list_of_none(self):
        for tool in registered_tools():
            result = extract_evidence(tool, {"contracts": [None, None, None]})
            assert isinstance(result, list)

    def test_list_of_mixed_types(self):
        """List containing int, str, None, dict, list."""
        mixed = [1, "two", None, {"k": "v"}, [3, 4]]
        for tool in registered_tools():
            result = extract_evidence(tool, {"contracts": mixed, "entries": mixed})
            assert isinstance(result, list)


class TestExtractorMissingKeys:
    """Expected keys missing from data dicts."""

    def test_cftc_missing_market_name(self):
        data = {"contracts": [{"_mm_net_pct_oi": 5}]}
        result = extract_evidence("cftc", data)
        assert isinstance(result, list)
        # Should still produce evidence (falls back to "unknown" slug)
        if result:
            assert "cftc" in result[0].signal_id

    def test_cftc_missing_pct_oi(self):
        data = {"contracts": [{"Market_and_Exchange_Names": "GOLD"}]}
        result = extract_evidence("cftc", data)
        assert result == []  # No _mm_net_pct_oi → skips

    def test_weather_empty_alerts(self):
        data = {"alert_count": 0}
        result = extract_evidence("weather_alerts", data)
        assert isinstance(result, list)

    def test_finra_missing_signals(self):
        data = {"ticker": "AAPL"}
        result = extract_evidence("finra_short_volume", data)
        assert isinstance(result, list)

    def test_disease_empty_entries(self):
        data = {"entries": []}
        result = extract_evidence("disease_surveillance", data)
        assert isinstance(result, list)


class TestExtractorNumericStrings:
    """Values that are numeric strings instead of actual numbers."""

    def test_cftc_string_pct_oi(self):
        data = {"contracts": [{"Market_and_Exchange_Names": "X", "_mm_net_pct_oi": "15.5"}]}
        result = extract_evidence("cftc", data)
        assert isinstance(result, list)
        if result:
            assert result[0].value == 15.5

    def test_weather_string_alert_count(self):
        data = {"alert_count": "5"}
        result = extract_evidence("weather_alerts", data)
        assert isinstance(result, list)
        if result:
            assert result[0].value == 5.0

    def test_cftc_string_nan_pct_oi(self):
        """String "nan" passed as value."""
        data = {"contracts": [{"Market_and_Exchange_Names": "X", "_mm_net_pct_oi": "nan"}]}
        result = extract_evidence("cftc", data)
        assert isinstance(result, list)


class TestExtractorOutputIntegrity:
    """Every Evidence produced by any extractor must pass all validation."""

    def test_all_tools_produce_valid_evidence(self):
        """For every registered tool, feed representative data and check all outputs."""
        # Representative minimal data for several tools
        test_data = {
            "cftc": {"contracts": [{"Market_and_Exchange_Names": "OIL", "_mm_net_pct_oi": 10}]},
            "weather_alerts": {"alert_count": 3},
            "finra_short_volume": {
                "ticker": "A",
                "signals": {"latest_ratio": 0.5, "is_anomaly": True},
            },
            "earthquake_proximity": {
                "events": [
                    {
                        "properties": {
                            "mag": 5.0,
                            "place": "Alaska",
                            "sig": 400,
                            "alert": "yellow",
                        }
                    }
                ]
            },
            "disease_surveillance": {"entries": [{"title": "Bird flu - USA"}]},
            "global_pmi": {"signals": {"USA": {"latest_value": 52.0, "mom_change": 0.3}}},
            "job_postings": {
                "jolts_level": 8000,
                "quits_level": 3500,
                "layoffs_level": 1700,
            },
            "gdelt": {"events": [{"GoldsteinScale": -5.0, "NumMentions": 100, "EventCode": "190"}]},
            "sovereign_debt": {
                "yields": {"2Y": 4.5, "10Y": 4.3},
                "curve": [{"label": "2s10s", "spread_bps": -20}],
            },
        }
        for tool_name, data in test_data.items():
            results = extract_evidence(tool_name, data)
            for ev in results:
                # If we got here, Evidence.__post_init__ already validated.
                # Double-check key invariants:
                assert isinstance(ev, Evidence)
                assert ev.source == tool_name
                assert ev.signal_id  # non-empty
                assert ev.timestamp > 0
                assert 0.0 <= ev.confidence <= 1.0
                assert ev.direction in (-1, 0, 1)
                assert ev.category in CATEGORIES
                assert ev.ttl > 0
                assert isinstance(ev.tags, tuple)


class TestExtractorOutputStubsReturnEmpty:
    """Output-only stubs must return []."""

    STUBS = [
        "satellite_activity",
        "foia_requests",
        "interconnection_queue",
        "internet_infrastructure",
        "electricity_monitor",
    ]

    def test_stubs_return_empty(self):
        for tool in self.STUBS:
            assert tool in registered_tools(), f"Stub {tool} not registered"
            # Even with plausible data, stubs return []
            result = extract_evidence(tool, {"data": "anything"})
            assert result == [], f"Stub {tool} returned non-empty"


# ══════════════════════════════════════════════════════════════
#  CROSS-MODULE INTEGRATION
# ══════════════════════════════════════════════════════════════


class TestEvidenceToBusIntegration:
    """Evidence created by extractors must be submittable to EvidenceBus."""

    def test_extractor_output_submittable(self):
        bus = EvidenceBus()
        data = {"contracts": [{"Market_and_Exchange_Names": "X", "_mm_net_pct_oi": 5}]}
        evidences = extract_evidence("cftc", data)
        assert len(evidences) > 0
        for ev in evidences:
            bus.submit(ev)
        assert len(bus) == len(evidences)

    def test_multi_tool_bus_flush_cycle(self):
        """Submit evidence from multiple tools, flush, re-submit."""
        bus = EvidenceBus()
        tools_data = [
            (
                "cftc",
                {"contracts": [{"Market_and_Exchange_Names": "OIL", "_mm_net_pct_oi": 10}]},
            ),
            ("weather_alerts", {"alert_count": 2}),
            (
                "gdelt",
                {"events": [{"GoldsteinScale": -3, "NumMentions": 50, "EventCode": "170"}]},
            ),
        ]
        for tool, data in tools_data:
            for ev in extract_evidence(tool, data):
                bus.submit(ev)
        first_count = len(bus)
        assert first_count > 0
        flushed = bus.flush()
        assert len(flushed) == first_count
        assert len(bus) == 0

        # Second cycle
        for tool, data in tools_data:
            for ev in extract_evidence(tool, data):
                bus.submit(ev)
        assert len(bus) == first_count


class TestExtractorRegistryIntegrity:
    """Verify all registered extractors return consistent data."""

    def test_no_duplicate_registry_entries(self):
        """Each tool is registered exactly once."""
        tools = registered_tools()
        assert len(tools) == len(set(tools))

    def test_registry_count_is_51(self):
        """51 extractors after Phase 45.3 (internet_outages + migration_flows stubs added)."""
        assert len(registered_tools()) == 51

    def test_all_registered_tools_callable(self):
        for tool_name in registered_tools():
            fn = _REGISTRY[tool_name]
            assert callable(fn), f"{tool_name} extractor is not callable"


class TestTaxonomyRegistryWithEvidence:
    """SignalRegistry entries must use valid categories for Evidence."""

    def test_meta_category_valid_for_evidence(self):
        """Every category in SignalMeta must be acceptable in Evidence."""
        for cat in CATEGORIES:
            # Must not raise
            _make_evidence(category=cat)

    def test_meta_category_matches_evidence_categories(self):
        assert VALID_CATEGORIES == CATEGORIES


# ══════════════════════════════════════════════════════════════
#  PERFORMANCE EDGE CASES
# ══════════════════════════════════════════════════════════════


class TestPerformance:
    """Performance constraints."""

    def test_1000_extractions_under_1s(self):
        data = {"contracts": [{"Market_and_Exchange_Names": "G", "_mm_net_pct_oi": 1}]}
        start = time.monotonic()
        for _ in range(1000):
            extract_evidence("cftc", data)
        elapsed = time.monotonic() - start
        assert elapsed < 1.0, f"1000 extractions took {elapsed:.2f}s"

    def test_bus_10k_submit_flush_under_1s(self):
        bus = EvidenceBus()
        evidences = [_make_evidence(signal_id=f"p{i}") for i in range(10_000)]
        start = time.monotonic()
        for ev in evidences:
            bus.submit(ev)
        bus.flush()
        elapsed = time.monotonic() - start
        assert elapsed < 1.0, f"10K submit+flush took {elapsed:.2f}s"

    def test_registry_500_signals_lookup_under_100ms(self):
        reg = SignalRegistry()
        cats = sorted(CATEGORIES)
        freqs = sorted(VALID_FREQUENCIES)
        for i in range(500):
            reg.register(
                _make_meta(
                    signal_id=f"perf{i}",
                    source=f"t{i % 20}",
                    category=cats[i % len(cats)],
                    frequency=freqs[i % len(freqs)],
                )
            )
        start = time.monotonic()
        for i in range(500):
            reg.get(f"perf{i}")
            reg.by_source(f"t{i % 20}")
        elapsed = time.monotonic() - start
        assert elapsed < 0.1, f"Lookups took {elapsed:.3f}s"
