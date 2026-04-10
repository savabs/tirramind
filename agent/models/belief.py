"""
TirraMind — Belief State Protocol

Defines the immutable belief record contract for the world model.
Every belief is a distribution (never a point estimate) over one variable.

Design principles:
    1. Immutable records — frozen dataclass, never mutated after creation.
    2. Point-in-time safety — ``effective_at`` is when evidence was knowable.
    3. Distribution-first — every belief carries full distributional params.
    4. Versioned — ``version`` tracks world model schema evolution.
    5. Validated at write boundary — ``validate_belief`` is a pure function.

Naming convention: ``{type}.{name}`` or ``{type}.{name}.{qualifier}``
    Examples: ``regime.macro``, ``latent.stress_level``, ``obs.rate_momentum``

References:
    - EngineeredFeature protocol: agent/features/protocol.py (design mirror)
    - Spec: docs/specs/world_model_spec.md (sub-phase 9.1)
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

# ── Constants ──────────────────────────────────────────────────

VARIABLE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9][a-z0-9_]*){1,3}$")

VALID_DIST_TYPES: frozenset[str] = frozenset({"gaussian", "categorical", "empirical"})

_EPOCH_FLOOR = 1_577_836_800.0  # 2020-01-01 00:00 UTC
_EPOCH_CEILING_DRIFT = 86_400.0  # allow up to 1 day into the "future"

# Tolerance for categorical probability sum.
_PROB_SUM_TOLERANCE = 1e-6

# Expected length for SHA-256 hex digest.
_HASH_HEX_LEN = 64


# ── Protocol dataclass ─────────────────────────────────────────


@dataclass(frozen=True)
class BeliefState:
    """A single posterior belief about one world-model variable.

    This is the output contract that Phase 10+ consumers depend on.
    Every belief is a distribution, never a point estimate.
    """

    # ── Identity ──
    variable_name: str
    """Node name from the world model graph, dotted hierarchy."""

    version: int
    """Schema version of the world model that produced this belief."""

    # ── Temporal ──
    effective_at: float
    """Unix epoch: when the underlying evidence was knowable."""

    computed_at: float
    """Unix epoch: when the world model computed this belief."""

    # ── Distribution ──
    dist_type: str
    """Distribution family: ``'gaussian'``, ``'categorical'``, ``'empirical'``."""

    mean: float | None = None
    """For Gaussian: E[X].  For categorical/empirical: None or summary stat."""

    variance: float | None = None
    """For Gaussian: Var[X].  Must be >= 0.  None for categorical."""

    probabilities: dict[str, float] | None = None
    """For categorical: ``{state_label: probability}``, sums to 1.0.
    None for Gaussian."""

    # ── Provenance ──
    evidence_count: int = 0
    """Number of EngineeredFeature observations consumed for this update."""

    model_graph_hash: str = ""
    """SHA-256 hex digest of the DAG structure for reproducibility."""

    # ── Quality ──
    confidence: float = 1.0
    """Overall confidence in this belief, [0.0, 1.0]."""

    stale: bool = False
    """True if no fresh evidence was available when this belief was computed."""

    # ── Optional metadata ──
    metadata: dict[str, Any] | None = field(default=None, hash=False)
    """Free-form JSON-serializable context for debugging / provenance."""

    # ── Serialization ──────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON / SQLite storage."""
        return {
            "variable_name": self.variable_name,
            "version": self.version,
            "effective_at": self.effective_at,
            "computed_at": self.computed_at,
            "dist_type": self.dist_type,
            "mean": self.mean,
            "variance": self.variance,
            "probabilities": self.probabilities,
            "evidence_count": self.evidence_count,
            "model_graph_hash": self.model_graph_hash,
            "confidence": self.confidence,
            "stale": self.stale,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BeliefState:
        """Reconstruct from a dict (inverse of ``to_dict``)."""
        return cls(
            variable_name=d["variable_name"],
            version=d["version"],
            effective_at=d["effective_at"],
            computed_at=d["computed_at"],
            dist_type=d["dist_type"],
            mean=d.get("mean"),
            variance=d.get("variance"),
            probabilities=d.get("probabilities"),
            evidence_count=d.get("evidence_count", 0),
            model_graph_hash=d.get("model_graph_hash", ""),
            confidence=d.get("confidence", 1.0),
            stale=d.get("stale", False),
            metadata=d.get("metadata"),
        )


# ── Validation ─────────────────────────────────────────────────


class BeliefValidationError(ValueError):
    """Raised when a caller converts validation errors to an exception."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(f"{len(errors)} validation error(s): {'; '.join(errors)}")


def validate_belief(belief: BeliefState) -> list[str]:
    """Validate a single belief record.

    Returns a list of human-readable error strings.  Empty list = valid.
    Pure function — never raises.
    """
    errors: list[str] = []

    # ── Variable name ──
    if not belief.variable_name:
        errors.append("variable_name must be non-empty")
    elif not VARIABLE_NAME_PATTERN.match(belief.variable_name):
        errors.append(
            f"variable_name '{belief.variable_name}' must match pattern "
            "'{{type}}.{{name}}[.{{qualifier}}]' (lowercase, 1-4 dot-segments)"
        )

    # ── Version ──
    if not isinstance(belief.version, int) or belief.version < 1:
        errors.append("version must be a positive integer (>= 1)")

    # ── Temporal: effective_at <= computed_at ──
    if belief.effective_at > belief.computed_at:
        errors.append(
            "effective_at must not be after computed_at "
            "(information-leakage violation)"
        )

    now = time.time()
    if belief.effective_at < _EPOCH_FLOOR:
        errors.append(f"effective_at ({belief.effective_at}) is before 2020-01-01")
    if belief.computed_at > now + _EPOCH_CEILING_DRIFT:
        errors.append(
            f"computed_at ({belief.computed_at}) is more than 1 day in the future"
        )

    # ── Distribution type ──
    if belief.dist_type not in VALID_DIST_TYPES:
        errors.append(
            f"dist_type '{belief.dist_type}' not recognized; "
            f"must be one of {sorted(VALID_DIST_TYPES)}"
        )

    # ── Distribution-specific validation ──
    if belief.dist_type == "gaussian":
        if belief.mean is None:
            errors.append("mean is required for dist_type='gaussian'")
        elif not isinstance(belief.mean, (int, float)):
            errors.append("mean must be a number for dist_type='gaussian'")
        elif math.isnan(belief.mean) or math.isinf(belief.mean):
            errors.append("mean must be finite")

        if belief.variance is None:
            errors.append("variance is required for dist_type='gaussian'")
        elif not isinstance(belief.variance, (int, float)):
            errors.append("variance must be a number for dist_type='gaussian'")
        elif math.isnan(belief.variance) or math.isinf(belief.variance):
            errors.append("variance must be finite")
        elif belief.variance < 0:
            errors.append("variance must be >= 0")

    elif belief.dist_type == "categorical":
        if belief.probabilities is None:
            errors.append("probabilities dict is required for dist_type='categorical'")
        elif not isinstance(belief.probabilities, dict):
            errors.append("probabilities must be a dict")
        elif len(belief.probabilities) == 0:
            errors.append("probabilities must be non-empty")
        else:
            # Check individual values
            for state, prob in belief.probabilities.items():
                if not isinstance(state, str):
                    errors.append(f"probability key '{state}' must be a string")
                if not isinstance(prob, (int, float)):
                    errors.append(f"probability for '{state}' must be a number")
                elif prob < 0 or prob > 1:
                    errors.append(
                        f"probability for '{state}' ({prob}) must be in [0, 1]"
                    )
                elif math.isnan(prob) or math.isinf(prob):
                    errors.append(f"probability for '{state}' must be finite")
            # Check sum
            prob_sum = sum(belief.probabilities.values())
            if abs(prob_sum - 1.0) > _PROB_SUM_TOLERANCE:
                errors.append(f"probabilities must sum to 1.0 (got {prob_sum})")

    # (empirical: mean may or may not be present — no strict requirement)

    # ── Evidence count ──
    if not isinstance(belief.evidence_count, int) or belief.evidence_count < 0:
        errors.append("evidence_count must be a non-negative integer")

    # ── Model graph hash ──
    if not isinstance(belief.model_graph_hash, str):
        errors.append("model_graph_hash must be a string")
    elif belief.model_graph_hash and len(belief.model_graph_hash) != _HASH_HEX_LEN:
        errors.append(
            f"model_graph_hash must be {_HASH_HEX_LEN} hex chars "
            f"(got {len(belief.model_graph_hash)})"
        )
    elif belief.model_graph_hash:
        try:
            int(belief.model_graph_hash, 16)
        except ValueError:
            errors.append("model_graph_hash must be valid hexadecimal")

    # ── Confidence ──
    if not isinstance(belief.confidence, (int, float)):
        errors.append("confidence must be a number")
    elif math.isnan(belief.confidence) or math.isinf(belief.confidence):
        errors.append("confidence must be finite")
    elif not (0.0 <= belief.confidence <= 1.0):
        errors.append("confidence must be in [0.0, 1.0]")

    return errors
