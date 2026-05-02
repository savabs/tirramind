"""
TirraMind — Engineered Feature Protocol Tests

Comprehensive edge-case coverage for EngineeredFeature and validate_feature.
"""

from __future__ import annotations

import time

import pytest

from agent.features.protocol import (
    FEATURE_NAME_PATTERN,
    VALID_HORIZONS,
    VALID_UNITS,
    EngineeredFeature,
    FeatureValidationError,
    validate_feature,
    validate_features,
)

# ── Helpers ────────────────────────────────────────────────────

_NOW = time.time()


def _valid_feature(**overrides: object) -> EngineeredFeature:
    """Return a fully valid EngineeredFeature, with optional field overrides."""
    defaults = dict(
        feature_name="convergence.stress_breadth.7d",
        version=1,
        effective_at=_NOW - 3600.0,
        computed_at=_NOW,
        horizon="7d",
        value=1.23,
        quality=0.85,
        missing_reason=None,
        source_signals=("convergence.event_count",),
        builder="stress_breadth_builder",
        unit="z_score",
        metadata=None,
    )
    defaults.update(overrides)
    return EngineeredFeature(**defaults)  # type: ignore[arg-type]


# ── Happy path ─────────────────────────────────────────────────


class TestValidFeature:
    """Baseline: a valid feature should pass validation with zero errors."""

    def test_valid_minimal(self) -> None:
        feat = _valid_feature()
        assert validate_feature(feat) == []

    def test_valid_with_metadata(self) -> None:
        feat = _valid_feature(metadata={"debug": True, "source_url": "https://example.com"})
        assert validate_feature(feat) == []

    def test_valid_missing_value(self) -> None:
        feat = _valid_feature(value=None, missing_reason="upstream_stale")
        assert validate_feature(feat) == []

    def test_valid_zero_value(self) -> None:
        feat = _valid_feature(value=0.0)
        assert validate_feature(feat) == []

    def test_valid_negative_value(self) -> None:
        feat = _valid_feature(value=-3.5)
        assert validate_feature(feat) == []

    def test_valid_quality_boundaries(self) -> None:
        for q in (0.0, 0.5, 1.0):
            feat = _valid_feature(quality=q)
            assert validate_feature(feat) == [], f"quality={q} should be valid"

    def test_valid_all_horizons(self) -> None:
        for h in VALID_HORIZONS:
            feat = _valid_feature(horizon=h)
            assert validate_feature(feat) == [], f"horizon='{h}' should be valid"

    def test_valid_all_units(self) -> None:
        for u in VALID_UNITS:
            feat = _valid_feature(unit=u)
            assert validate_feature(feat) == [], f"unit='{u}' should be valid"

    def test_valid_multiple_source_signals(self) -> None:
        feat = _valid_feature(source_signals=("sig_a", "sig_b", "sig_c"))
        assert validate_feature(feat) == []

    def test_valid_effective_equals_computed(self) -> None:
        t = _NOW
        feat = _valid_feature(effective_at=t, computed_at=t)
        assert validate_feature(feat) == []


# ── Feature name validation ────────────────────────────────────


class TestFeatureNameValidation:
    """Name must be lowercase dotted segments: {domain}.{metric}[.{qualifier}]."""

    def test_empty_name(self) -> None:
        feat = _valid_feature(feature_name="")
        errs = validate_feature(feat)
        assert any("non-empty" in e for e in errs)

    def test_whitespace_name(self) -> None:
        feat = _valid_feature(feature_name="   ")
        errs = validate_feature(feat)
        assert any("non-empty" in e or "pattern" in e for e in errs)

    def test_single_segment(self) -> None:
        feat = _valid_feature(feature_name="convergence")
        errs = validate_feature(feat)
        assert any("pattern" in e for e in errs)

    def test_uppercase_rejected(self) -> None:
        feat = _valid_feature(feature_name="Convergence.Stress")
        errs = validate_feature(feat)
        assert any("pattern" in e for e in errs)

    def test_spaces_in_name(self) -> None:
        feat = _valid_feature(feature_name="convergence. stress")
        errs = validate_feature(feat)
        assert any("pattern" in e for e in errs)

    def test_five_segments_rejected(self) -> None:
        feat = _valid_feature(feature_name="a.b.c.d.e")
        errs = validate_feature(feat)
        assert any("pattern" in e for e in errs)

    def test_leading_dot(self) -> None:
        feat = _valid_feature(feature_name=".convergence.stress")
        errs = validate_feature(feat)
        assert any("pattern" in e for e in errs)

    def test_trailing_dot(self) -> None:
        feat = _valid_feature(feature_name="convergence.stress.")
        errs = validate_feature(feat)
        assert any("pattern" in e for e in errs)

    def test_double_dot(self) -> None:
        feat = _valid_feature(feature_name="convergence..stress")
        errs = validate_feature(feat)
        assert any("pattern" in e for e in errs)

    def test_hyphen_rejected(self) -> None:
        feat = _valid_feature(feature_name="convergence.stress-breadth")
        errs = validate_feature(feat)
        assert any("pattern" in e for e in errs)

    def test_starts_with_digit(self) -> None:
        feat = _valid_feature(feature_name="1convergence.stress")
        errs = validate_feature(feat)
        assert any("pattern" in e for e in errs)

    def test_segment_starts_with_digit(self) -> None:
        feat = _valid_feature(feature_name="convergence.7d_stress")
        errs = validate_feature(feat)
        assert any("pattern" in e for e in errs)

    def test_valid_two_segments(self) -> None:
        assert FEATURE_NAME_PATTERN.match("convergence.stress") is not None

    def test_valid_three_segments(self) -> None:
        assert FEATURE_NAME_PATTERN.match("convergence.stress.7d") is not None

    def test_valid_four_segments(self) -> None:
        assert FEATURE_NAME_PATTERN.match("convergence.stress.7d.v2") is not None

    def test_underscores_allowed(self) -> None:
        assert FEATURE_NAME_PATTERN.match("macro.rate_of_change.30d") is not None

    def test_digits_within_segment(self) -> None:
        assert FEATURE_NAME_PATTERN.match("convergence.stress7.d30") is not None


# ── Version validation ─────────────────────────────────────────


class TestVersionValidation:
    def test_zero_version(self) -> None:
        feat = _valid_feature(version=0)
        errs = validate_feature(feat)
        assert any("positive integer" in e for e in errs)

    def test_negative_version(self) -> None:
        feat = _valid_feature(version=-1)
        errs = validate_feature(feat)
        assert any("positive integer" in e for e in errs)

    def test_valid_version(self) -> None:
        feat = _valid_feature(version=42)
        assert validate_feature(feat) == []


# ── Temporal validation ────────────────────────────────────────


class TestTemporalValidation:
    def test_effective_after_computed_is_leakage(self) -> None:
        feat = _valid_feature(effective_at=_NOW + 100, computed_at=_NOW)
        errs = validate_feature(feat)
        assert any("look-ahead" in e for e in errs)

    def test_effective_way_before_floor(self) -> None:
        feat = _valid_feature(effective_at=1_000_000_000.0)  # 2001
        errs = validate_feature(feat)
        assert any("2020" in e for e in errs)

    def test_computed_far_future(self) -> None:
        feat = _valid_feature(computed_at=_NOW + 200_000.0)
        errs = validate_feature(feat)
        assert any("future" in e for e in errs)

    def test_effective_at_floor_boundary(self) -> None:
        """Exactly at the floor should be valid."""
        feat = _valid_feature(effective_at=1_577_836_800.0, computed_at=_NOW)
        assert validate_feature(feat) == []

    def test_effective_slightly_below_floor(self) -> None:
        feat = _valid_feature(effective_at=1_577_836_799.0, computed_at=_NOW)
        errs = validate_feature(feat)
        assert any("2020" in e for e in errs)


# ── Value + missingness ────────────────────────────────────────


class TestValueMissingness:
    def test_none_without_reason(self) -> None:
        feat = _valid_feature(value=None, missing_reason=None)
        errs = validate_feature(feat)
        assert any("missing_reason is required" in e for e in errs)

    def test_none_with_empty_reason(self) -> None:
        feat = _valid_feature(value=None, missing_reason="")
        errs = validate_feature(feat)
        assert any("non-empty" in e for e in errs)

    def test_none_with_whitespace_reason(self) -> None:
        feat = _valid_feature(value=None, missing_reason="   ")
        errs = validate_feature(feat)
        assert any("non-empty" in e for e in errs)

    def test_value_with_reason_is_inconsistent(self) -> None:
        feat = _valid_feature(value=1.5, missing_reason="bug")
        errs = validate_feature(feat)
        assert any("must be None when value is present" in e for e in errs)

    def test_nan_rejected(self) -> None:
        feat = _valid_feature(value=float("nan"))
        errs = validate_feature(feat)
        assert any("finite" in e for e in errs)

    def test_positive_inf_rejected(self) -> None:
        feat = _valid_feature(value=float("inf"))
        errs = validate_feature(feat)
        assert any("finite" in e for e in errs)

    def test_negative_inf_rejected(self) -> None:
        feat = _valid_feature(value=float("-inf"))
        errs = validate_feature(feat)
        assert any("finite" in e for e in errs)

    def test_very_large_value_accepted(self) -> None:
        feat = _valid_feature(value=1e18)
        assert validate_feature(feat) == []

    def test_very_small_value_accepted(self) -> None:
        feat = _valid_feature(value=1e-18)
        assert validate_feature(feat) == []


# ── Quality validation ─────────────────────────────────────────


class TestQualityValidation:
    def test_negative_quality(self) -> None:
        feat = _valid_feature(quality=-0.01)
        errs = validate_feature(feat)
        assert any("quality" in e for e in errs)

    def test_quality_above_one(self) -> None:
        feat = _valid_feature(quality=1.01)
        errs = validate_feature(feat)
        assert any("quality" in e for e in errs)

    def test_quality_nan(self) -> None:
        feat = _valid_feature(quality=float("nan"))
        errs = validate_feature(feat)
        assert any("quality" in e for e in errs)

    def test_quality_inf(self) -> None:
        feat = _valid_feature(quality=float("inf"))
        errs = validate_feature(feat)
        assert any("quality" in e for e in errs)


# ── Horizon validation ─────────────────────────────────────────


class TestHorizonValidation:
    def test_unknown_horizon(self) -> None:
        feat = _valid_feature(horizon="2w")
        errs = validate_feature(feat)
        assert any("horizon" in e for e in errs)

    def test_empty_horizon(self) -> None:
        feat = _valid_feature(horizon="")
        errs = validate_feature(feat)
        assert any("horizon" in e for e in errs)

    def test_case_sensitive(self) -> None:
        feat = _valid_feature(horizon="7D")
        errs = validate_feature(feat)
        assert any("horizon" in e for e in errs)


# ── Unit validation ────────────────────────────────────────────


class TestUnitValidation:
    def test_unknown_unit(self) -> None:
        feat = _valid_feature(unit="dollars")
        errs = validate_feature(feat)
        assert any("unit" in e for e in errs)

    def test_empty_unit(self) -> None:
        feat = _valid_feature(unit="")
        errs = validate_feature(feat)
        assert any("unit" in e for e in errs)


# ── Source signals ─────────────────────────────────────────────


class TestSourceSignals:
    def test_empty_source_signals(self) -> None:
        feat = _valid_feature(source_signals=())
        errs = validate_feature(feat)
        assert any("source_signals" in e for e in errs)


# ── Builder ────────────────────────────────────────────────────


class TestBuilder:
    def test_empty_builder(self) -> None:
        feat = _valid_feature(builder="")
        errs = validate_feature(feat)
        assert any("builder" in e for e in errs)

    def test_whitespace_builder(self) -> None:
        feat = _valid_feature(builder="   ")
        errs = validate_feature(feat)
        assert any("builder" in e for e in errs)


# ── Metadata validation ───────────────────────────────────────


class TestMetadata:
    def test_none_metadata_accepted(self) -> None:
        feat = _valid_feature(metadata=None)
        assert validate_feature(feat) == []

    def test_dict_metadata_accepted(self) -> None:
        feat = _valid_feature(metadata={"key": "val"})
        assert validate_feature(feat) == []

    def test_non_dict_metadata_rejected(self) -> None:
        feat = _valid_feature(metadata="not a dict")  # type: ignore[arg-type]
        errs = validate_feature(feat)
        assert any("metadata" in e for e in errs)

    def test_list_metadata_rejected(self) -> None:
        feat = _valid_feature(metadata=[1, 2, 3])  # type: ignore[arg-type]
        errs = validate_feature(feat)
        assert any("metadata" in e for e in errs)


# ── Frozen semantics ───────────────────────────────────────────


class TestFrozenSemantics:
    def test_cannot_mutate_value(self) -> None:
        feat = _valid_feature()
        with pytest.raises(AttributeError):
            feat.value = 999.0  # type: ignore[misc]

    def test_cannot_mutate_name(self) -> None:
        feat = _valid_feature()
        with pytest.raises(AttributeError):
            feat.feature_name = "hacked.name"  # type: ignore[misc]

    def test_cannot_mutate_version(self) -> None:
        feat = _valid_feature()
        with pytest.raises(AttributeError):
            feat.version = 99  # type: ignore[misc]


# ── Serialization round-trip ───────────────────────────────────


class TestSerialization:
    def test_round_trip_basic(self) -> None:
        feat = _valid_feature()
        d = feat.to_dict()
        restored = EngineeredFeature.from_dict(d)
        assert restored == feat

    def test_round_trip_with_none_value(self) -> None:
        feat = _valid_feature(value=None, missing_reason="upstream_stale")
        d = feat.to_dict()
        restored = EngineeredFeature.from_dict(d)
        assert restored == feat
        assert restored.value is None
        assert restored.missing_reason == "upstream_stale"

    def test_round_trip_with_metadata(self) -> None:
        meta = {"window_size": 52, "z_threshold": 2.0, "tags": ["macro", "us"]}
        feat = _valid_feature(metadata=meta)
        d = feat.to_dict()
        restored = EngineeredFeature.from_dict(d)
        assert restored.metadata == meta

    def test_source_signals_survives_round_trip_as_tuple(self) -> None:
        feat = _valid_feature(source_signals=("a", "b"))
        d = feat.to_dict()
        assert isinstance(d["source_signals"], list)  # JSON-friendly
        restored = EngineeredFeature.from_dict(d)
        assert isinstance(restored.source_signals, tuple)
        assert restored.source_signals == ("a", "b")

    def test_to_dict_contains_all_fields(self) -> None:
        feat = _valid_feature()
        d = feat.to_dict()
        expected_keys = {
            "feature_name",
            "version",
            "effective_at",
            "computed_at",
            "horizon",
            "value",
            "quality",
            "missing_reason",
            "source_signals",
            "builder",
            "unit",
            "metadata",
        }
        assert set(d.keys()) == expected_keys

    def test_from_dict_with_missing_optional_fields(self) -> None:
        """from_dict should handle missing optional keys gracefully."""
        d = {
            "feature_name": "convergence.stress_breadth.7d",
            "version": 1,
            "effective_at": _NOW - 3600,
            "computed_at": _NOW,
            "horizon": "7d",
            "value": 1.0,
            "quality": 0.9,
        }
        feat = EngineeredFeature.from_dict(d)
        assert feat.missing_reason is None
        assert feat.source_signals == ()
        assert feat.builder == ""
        assert feat.unit == "raw"
        assert feat.metadata is None


# ── Batch validation ───────────────────────────────────────────


class TestBatchValidation:
    def test_all_valid(self) -> None:
        features = [_valid_feature(), _valid_feature(feature_name="macro.rate.30d")]
        assert validate_features(features) == {}

    def test_mixed_batch(self) -> None:
        features = [
            _valid_feature(),
            _valid_feature(feature_name=""),  # bad
            _valid_feature(value=float("nan")),  # bad
            _valid_feature(feature_name="macro.rate.30d"),  # good
        ]
        bad = validate_features(features)
        assert 0 not in bad
        assert 1 in bad
        assert 2 in bad
        assert 3 not in bad

    def test_empty_batch(self) -> None:
        assert validate_features([]) == {}


# ── FeatureValidationError ─────────────────────────────────────


class TestFeatureValidationError:
    def test_error_message(self) -> None:
        err = FeatureValidationError(["bad name", "bad version"])
        assert "2 validation error(s)" in str(err)
        assert "bad name" in str(err)
        assert err.errors == ["bad name", "bad version"]

    def test_raise_on_invalid(self) -> None:
        """Demonstrate the expected consumer pattern."""
        feat = _valid_feature(feature_name="", builder="")
        errs = validate_feature(feat)
        assert len(errs) >= 2
        with pytest.raises(FeatureValidationError):
            if errs:
                raise FeatureValidationError(errs)


# ── Multiple simultaneous errors ───────────────────────────────


class TestMultipleErrors:
    def test_maximally_invalid_feature(self) -> None:
        """A feature that violates as many rules as possible."""
        feat = EngineeredFeature(
            feature_name="",
            version=0,
            effective_at=1_000_000_000.0,  # before floor
            computed_at=_NOW - 100,  # before effective_at? no, let's make effective > computed
            horizon="invalid",
            value=float("nan"),
            quality=-0.5,
            missing_reason=None,
            source_signals=(),
            builder="",
            unit="invalid_unit",
            metadata=None,
        )
        # Fix: make effective_at > computed_at for look-ahead error
        feat2 = EngineeredFeature(
            feature_name="",
            version=0,
            effective_at=_NOW + 500,
            computed_at=_NOW,
            horizon="invalid",
            value=float("nan"),
            quality=-0.5,
            missing_reason=None,
            source_signals=(),
            builder="",
            unit="invalid_unit",
            metadata=None,
        )
        errs = validate_feature(feat2)
        # Should catch at least: name, version, temporal, horizon, value(nan),
        # quality, unit, source_signals, builder
        assert len(errs) >= 8, f"Expected >= 8 errors, got {len(errs)}: {errs}"

    def test_error_messages_are_descriptive(self) -> None:
        """Every error should be a non-empty string."""
        feat = _valid_feature(feature_name="", version=-1, builder="")
        errs = validate_feature(feat)
        for e in errs:
            assert isinstance(e, str)
            assert len(e) > 10, f"Error too short: {e!r}"


# ── Edge: equality and hashing ─────────────────────────────────


class TestEqualityAndHashing:
    def test_equal_features_are_equal(self) -> None:
        a = _valid_feature()
        b = _valid_feature()
        assert a == b

    def test_different_value_not_equal(self) -> None:
        a = _valid_feature(value=1.0)
        b = _valid_feature(value=2.0)
        assert a != b

    def test_hashable(self) -> None:
        """Frozen dataclass with hash=False on metadata still works for sets
        when metadata is None."""
        feat = _valid_feature()
        {feat}  # should not raise

    def test_hashable_with_metadata(self) -> None:
        """metadata has hash=False so it's excluded from hash. Two features
        differing only in metadata hash the same but are not equal."""
        a = _valid_feature(metadata=None)
        b = _valid_feature(metadata={"x": 1})
        # hash should work for both
        hash(a)
        hash(b)
        # But they are not equal because __eq__ still checks all fields
        assert a != b


# ── Regex pattern unit tests ──────────────────────────────────


class TestFeatureNameRegex:
    """Direct tests on the compiled pattern (separate from validation)."""

    @pytest.mark.parametrize(
        "name",
        [
            "convergence.stress",
            "macro.rate_of_change.30d",
            "convergence.stress_breadth.7d.v2",
            "liquidity.vpin.1d",
            "a.b",
            "abc.def.ghi.jkl",
        ],
    )
    def test_valid_names(self, name: str) -> None:
        assert FEATURE_NAME_PATTERN.match(name) is not None, f"'{name}' should match"

    @pytest.mark.parametrize(
        "name",
        [
            "",
            "single",
            "a.b.c.d.e",
            "A.B",
            ".a.b",
            "a.b.",
            "a..b",
            "a b.c",
            "1a.b",
            "a.1b",
            "a-b.c",
            "a.b-c",
        ],
    )
    def test_invalid_names(self, name: str) -> None:
        assert FEATURE_NAME_PATTERN.match(name) is None, f"'{name}' should NOT match"
