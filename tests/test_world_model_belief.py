"""Tests for BeliefState protocol — validation, serialization, edge cases."""

from __future__ import annotations

import math
import time

import pytest

from agent.models.belief import (
    VALID_DIST_TYPES,
    VARIABLE_NAME_PATTERN,
    BeliefState,
    BeliefValidationError,
    validate_belief,
)

# ── Helpers ────────────────────────────────────────────────────

_NOW = time.time()
_VALID_HASH = "a" * 64


def _valid_gaussian(**overrides) -> BeliefState:
    defaults = dict(
        variable_name="regime.macro",
        version=1,
        effective_at=_NOW - 100,
        computed_at=_NOW,
        dist_type="gaussian",
        mean=0.5,
        variance=0.1,
        evidence_count=3,
        model_graph_hash=_VALID_HASH,
        confidence=0.9,
        stale=False,
    )
    defaults.update(overrides)
    return BeliefState(**defaults)


def _valid_categorical(**overrides) -> BeliefState:
    defaults = dict(
        variable_name="regime.macro",
        version=1,
        effective_at=_NOW - 100,
        computed_at=_NOW,
        dist_type="categorical",
        probabilities={"expansion": 0.6, "contraction": 0.3, "crisis": 0.1},
        evidence_count=3,
        model_graph_hash=_VALID_HASH,
        confidence=0.85,
        stale=False,
    )
    defaults.update(overrides)
    return BeliefState(**defaults)


# ── Happy path ─────────────────────────────────────────────────


class TestBeliefStateHappyPath:
    def test_valid_gaussian_passes(self):
        b = _valid_gaussian()
        assert validate_belief(b) == []

    def test_valid_categorical_passes(self):
        b = _valid_categorical()
        assert validate_belief(b) == []

    def test_valid_empirical_passes(self):
        b = _valid_gaussian(dist_type="empirical", mean=1.0, variance=None)
        assert validate_belief(b) == []

    def test_frozen_immutable(self):
        b = _valid_gaussian()
        with pytest.raises(AttributeError):
            b.mean = 999  # type: ignore[misc]

    def test_to_dict_contains_all_fields(self):
        b = _valid_gaussian(metadata={"debug": True})
        d = b.to_dict()
        assert d["variable_name"] == "regime.macro"
        assert d["version"] == 1
        assert d["dist_type"] == "gaussian"
        assert d["mean"] == 0.5
        assert d["variance"] == 0.1
        assert d["confidence"] == 0.9
        assert d["stale"] is False
        assert d["metadata"] == {"debug": True}

    def test_from_dict_round_trip(self):
        b = _valid_gaussian(metadata={"k": "v"})
        d = b.to_dict()
        b2 = BeliefState.from_dict(d)
        assert b2 == b

    def test_categorical_round_trip(self):
        b = _valid_categorical()
        d = b.to_dict()
        b2 = BeliefState.from_dict(d)
        assert b2 == b

    def test_from_dict_defaults(self):
        """from_dict handles missing optional fields gracefully."""
        d = {
            "variable_name": "latent.x",
            "version": 1,
            "effective_at": _NOW - 10,
            "computed_at": _NOW,
            "dist_type": "empirical",
        }
        b = BeliefState.from_dict(d)
        assert b.evidence_count == 0
        assert b.model_graph_hash == ""
        assert b.confidence == 1.0
        assert b.stale is False
        assert b.metadata is None


# ── Variable name validation ───────────────────────────────────


class TestVariableNameValidation:
    def test_empty_name(self):
        errs = validate_belief(_valid_gaussian(variable_name=""))
        assert any("non-empty" in e for e in errs)

    @pytest.mark.parametrize(
        "name",
        [
            "REGIME.macro",  # uppercase
            "regime macro",  # space
            ".regime.macro",  # leading dot
            "regime.",  # trailing dot
            "regime",  # single segment (allowed by pattern? let's check)
        ],
    )
    def test_invalid_names(self, name):
        errs = validate_belief(_valid_gaussian(variable_name=name))
        assert len(errs) > 0

    @pytest.mark.parametrize(
        "name",
        [
            "regime.macro",
            "latent.stress_level",
            "obs.rate_momentum",
            "regime.macro.30d",
            "a.b.c.d",  # 4 segments
        ],
    )
    def test_valid_names(self, name):
        errs = validate_belief(_valid_gaussian(variable_name=name))
        assert errs == []


# ── Version validation ─────────────────────────────────────────


class TestVersionValidation:
    def test_version_zero(self):
        errs = validate_belief(_valid_gaussian(version=0))
        assert any("positive integer" in e for e in errs)

    def test_version_negative(self):
        errs = validate_belief(_valid_gaussian(version=-1))
        assert any("positive integer" in e for e in errs)

    def test_version_one(self):
        assert validate_belief(_valid_gaussian(version=1)) == []

    def test_version_large(self):
        assert validate_belief(_valid_gaussian(version=999)) == []


# ── Temporal validation ────────────────────────────────────────


class TestTemporalValidation:
    def test_effective_after_computed(self):
        errs = validate_belief(
            _valid_gaussian(effective_at=_NOW + 10, computed_at=_NOW)
        )
        assert any("leakage" in e for e in errs)

    def test_effective_before_epoch_floor(self):
        errs = validate_belief(
            _valid_gaussian(effective_at=1_500_000_000.0, computed_at=_NOW)
        )
        assert any("2020" in e for e in errs)

    def test_computed_far_future(self):
        errs = validate_belief(_valid_gaussian(computed_at=_NOW + 200_000))
        assert any("future" in e for e in errs)

    def test_same_timestamp_ok(self):
        ts = _NOW
        assert validate_belief(_valid_gaussian(effective_at=ts, computed_at=ts)) == []


# ── Distribution type validation ───────────────────────────────


class TestDistTypeValidation:
    def test_invalid_dist_type(self):
        errs = validate_belief(_valid_gaussian(dist_type="poisson"))
        assert any("not recognized" in e for e in errs)

    @pytest.mark.parametrize("dt", sorted(VALID_DIST_TYPES))
    def test_valid_dist_types_accepted(self, dt):
        if dt == "gaussian":
            b = _valid_gaussian(dist_type=dt)
        elif dt == "categorical":
            b = _valid_categorical(dist_type=dt)
        else:
            b = _valid_gaussian(dist_type=dt, mean=1.0, variance=None)
        errs = validate_belief(b)
        # Should not fail on dist_type itself
        assert not any("not recognized" in e for e in errs)


# ── Gaussian-specific validation ───────────────────────────────


class TestGaussianValidation:
    def test_missing_mean(self):
        errs = validate_belief(_valid_gaussian(mean=None))
        assert any("mean is required" in e for e in errs)

    def test_missing_variance(self):
        errs = validate_belief(_valid_gaussian(variance=None))
        assert any("variance is required" in e for e in errs)

    def test_nan_mean(self):
        errs = validate_belief(_valid_gaussian(mean=float("nan")))
        assert any("mean must be finite" in e for e in errs)

    def test_inf_mean(self):
        errs = validate_belief(_valid_gaussian(mean=float("inf")))
        assert any("mean must be finite" in e for e in errs)

    def test_nan_variance(self):
        errs = validate_belief(_valid_gaussian(variance=float("nan")))
        assert any("variance must be finite" in e for e in errs)

    def test_inf_variance(self):
        errs = validate_belief(_valid_gaussian(variance=float("inf")))
        assert any("variance must be finite" in e for e in errs)

    def test_negative_variance(self):
        errs = validate_belief(_valid_gaussian(variance=-0.01))
        assert any("variance must be >= 0" in e for e in errs)

    def test_zero_variance_ok(self):
        """Zero variance = point mass / delta distribution, valid."""
        assert validate_belief(_valid_gaussian(variance=0.0)) == []

    def test_very_large_mean_ok(self):
        assert validate_belief(_valid_gaussian(mean=1e15)) == []

    def test_very_small_variance_ok(self):
        assert validate_belief(_valid_gaussian(variance=1e-30)) == []


# ── Categorical-specific validation ────────────────────────────


class TestCategoricalValidation:
    def test_missing_probabilities(self):
        errs = validate_belief(_valid_categorical(probabilities=None))
        assert any("probabilities dict is required" in e for e in errs)

    def test_empty_probabilities(self):
        errs = validate_belief(_valid_categorical(probabilities={}))
        assert any("non-empty" in e for e in errs)

    def test_probabilities_dont_sum_to_one(self):
        errs = validate_belief(_valid_categorical(probabilities={"a": 0.3, "b": 0.2}))
        assert any("sum to 1.0" in e for e in errs)

    def test_probabilities_sum_within_tolerance(self):
        """0.999999999 should pass (within 1e-6)."""
        probs = {"a": 1.0 / 3, "b": 1.0 / 3, "c": 1.0 / 3}
        assert validate_belief(_valid_categorical(probabilities=probs)) == []

    def test_negative_probability(self):
        errs = validate_belief(_valid_categorical(probabilities={"a": -0.1, "b": 1.1}))
        assert any("must be in [0, 1]" in e for e in errs)

    def test_probability_above_one(self):
        errs = validate_belief(_valid_categorical(probabilities={"a": 1.5, "b": -0.5}))
        assert any("must be in [0, 1]" in e for e in errs)

    def test_nan_probability(self):
        errs = validate_belief(
            _valid_categorical(probabilities={"a": float("nan"), "b": 0.5, "c": 0.5})
        )
        assert any("finite" in e for e in errs)

    def test_two_state_categorical(self):
        b = _valid_categorical(probabilities={"on": 0.7, "off": 0.3})
        assert validate_belief(b) == []


# ── Evidence count validation ──────────────────────────────────


class TestEvidenceCountValidation:
    def test_negative_evidence_count(self):
        errs = validate_belief(_valid_gaussian(evidence_count=-1))
        assert any("non-negative" in e for e in errs)

    def test_zero_evidence_count_ok(self):
        assert validate_belief(_valid_gaussian(evidence_count=0)) == []


# ── Model graph hash validation ────────────────────────────────


class TestGraphHashValidation:
    def test_wrong_length(self):
        errs = validate_belief(_valid_gaussian(model_graph_hash="abc123"))
        assert any("64 hex chars" in e for e in errs)

    def test_not_hex(self):
        errs = validate_belief(_valid_gaussian(model_graph_hash="z" * 64))
        assert any("hexadecimal" in e for e in errs)

    def test_empty_hash_ok(self):
        """Empty hash is allowed (e.g. before graph is set)."""
        assert validate_belief(_valid_gaussian(model_graph_hash="")) == []

    def test_valid_hash(self):
        h = "abcdef0123456789" * 4  # 64 hex chars
        assert validate_belief(_valid_gaussian(model_graph_hash=h)) == []


# ── Confidence validation ──────────────────────────────────────


class TestConfidenceValidation:
    def test_confidence_below_zero(self):
        errs = validate_belief(_valid_gaussian(confidence=-0.1))
        assert any("confidence" in e for e in errs)

    def test_confidence_above_one(self):
        errs = validate_belief(_valid_gaussian(confidence=1.01))
        assert any("confidence" in e for e in errs)

    def test_confidence_nan(self):
        errs = validate_belief(_valid_gaussian(confidence=float("nan")))
        assert any("finite" in e for e in errs)

    def test_confidence_zero_ok(self):
        assert validate_belief(_valid_gaussian(confidence=0.0)) == []

    def test_confidence_one_ok(self):
        assert validate_belief(_valid_gaussian(confidence=1.0)) == []


# ── BeliefValidationError ──────────────────────────────────────


class TestBeliefValidationError:
    def test_error_message(self):
        err = BeliefValidationError(["bad field 1", "bad field 2"])
        assert "2 validation error(s)" in str(err)
        assert err.errors == ["bad field 1", "bad field 2"]


# ── Multiple errors ────────────────────────────────────────────


class TestMultipleErrors:
    def test_accumulates_all_errors(self):
        """A completely invalid belief should produce many errors."""
        b = BeliefState(
            variable_name="",
            version=0,
            effective_at=_NOW + 10,
            computed_at=_NOW,
            dist_type="invalid",
            mean=float("nan"),
            variance=-1.0,
            evidence_count=-5,
            model_graph_hash="short",
            confidence=2.0,
        )
        errs = validate_belief(b)
        assert (
            len(errs) >= 5
        )  # at least: name, version, temporal, dist_type, confidence
