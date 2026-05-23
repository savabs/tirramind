"""
TirraMind — Engineered Feature Protocol

Defines the immutable feature record contract and validation rules for
model-ready quantitative state variables.

Design principles:
    1. Immutable records — frozen dataclass, never mutated after creation.
    2. Point-in-time safety — ``effective_at`` captures when information
       became knowable, not when the DAG ran (``computed_at``).
    3. Explicit missingness — ``value=None`` requires ``missing_reason``.
    4. Versioned — ``version`` field enables schema evolution without
       corrupting historical records.
    5. Validated at write boundary — ``validate_feature`` is a pure function
       returning an error list; callers decide whether to raise or log.

Naming convention:  ``{domain}.{metric}.{horizon}``
    Examples: ``convergence.stress_breadth.7d``, ``macro.rate_momentum.30d``

References:
    - Feast feature store: entity-keyed timestamped schema with validation
      (concept only — we don't need entity/join machinery)
    - Existing ``ConvergenceSignal`` / ``AtomicSignalResult`` in convergence pkg
    - Research: docs/research/signal_protocol_feature_engineering.md
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

# ── Constants ──────────────────────────────────────────────────

# Dotted name: 2-4 segments of word chars, e.g. "convergence.stress_breadth.7d"
FEATURE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*(\.[a-z0-9][a-z0-9_]*){0,2}$")

VALID_HORIZONS: frozenset[str] = frozenset(
    {
        "spot",  # instantaneous / no look-back
        "1d",
        "3d",
        "7d",
        "14d",
        "30d",
        "60d",
        "90d",
        "180d",
        "365d",
    }
)

VALID_UNITS: frozenset[str] = frozenset(
    {
        "z_score",  # standard deviations from mean
        "count",  # integer count
        "ratio",  # dimensionless ratio
        "flag",  # binary 0/1
        "pct",  # percentage [0, 100]
        "probability",  # [0, 1]
        "raw",  # unscaled value
        "bps",  # basis points
        "index",  # index value (100-based or arbitrary)
        "log_return",  # log price return
        "rank",  # ordinal rank or percentile [0, 1]
        "duration_days",  # time span in days
    }
)

# Reasonable timestamp range guards (epoch seconds).
_EPOCH_FLOOR = 1_577_836_800.0  # 2020-01-01 00:00 UTC
_EPOCH_CEILING_DRIFT = 86_400.0  # allow up to 1 day into the "future"


# ── Protocol dataclass ─────────────────────────────────────────


@dataclass(frozen=True)
class EngineeredFeature:
    """A single model-ready quantitative feature value.

    This is the stable downstream contract that Phase 9+ consumers depend on.
    All fields are present in every record; optional semantics are encoded via
    ``None`` values with mandatory companion reasons.
    """

    # ── Identity ──
    feature_name: str
    """Dotted hierarchy: ``{domain}.{metric}[.{horizon}][.{qualifier}]``."""

    version: int
    """Schema version (monotonically increasing). Bumped when the builder's
    output semantics change."""

    # ── Temporal ──
    effective_at: float
    """Unix epoch: when this information *became knowable* (point-in-time).
    Must be <= ``computed_at``. This is the timestamp used for look-back joins
    in training to prevent information leakage."""

    computed_at: float
    """Unix epoch: when the DAG actually produced this value."""

    horizon: str
    """Temporal grain of the feature: ``spot``, ``1d``, ``7d``, etc."""

    # ── Value ──
    value: float | None
    """The feature value, or ``None`` for explicit missingness."""

    quality: float
    """Confidence / quality score in ``[0, 1]``. 1.0 = full confidence,
    0.0 = lowest usable quality. Consumers can threshold on this."""

    missing_reason: str | None = None
    """Required when ``value is None``. Examples: ``'upstream_stale'``,
    ``'insufficient_history'``, ``'source_unavailable'``."""

    # ── Lineage ──
    source_signals: tuple[str, ...] = ()
    """Signal names (from the pipeline ``signals`` table or tool sources)
    that this feature was derived from."""

    builder: str = ""
    """Name of the ``FeatureBuilder`` that produced this record."""

    # ── Measurement ──
    unit: str = "raw"
    """Physical / statistical unit. Must be one of ``VALID_UNITS``."""

    # ── Optional metadata ──
    metadata: dict[str, Any] | None = field(default=None, hash=False)
    """Free-form JSON-serializable context.  Not part of the stable contract —
    use sparingly for debugging or provenance detail."""

    # ── Serialization ──────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for JSON / SQLite storage."""
        return {
            "feature_name": self.feature_name,
            "version": self.version,
            "effective_at": self.effective_at,
            "computed_at": self.computed_at,
            "horizon": self.horizon,
            "value": self.value,
            "quality": self.quality,
            "missing_reason": self.missing_reason,
            "source_signals": list(self.source_signals),
            "builder": self.builder,
            "unit": self.unit,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EngineeredFeature:
        """Reconstruct from a dict (inverse of ``to_dict``)."""
        return cls(
            feature_name=d["feature_name"],
            version=d["version"],
            effective_at=d["effective_at"],
            computed_at=d["computed_at"],
            horizon=d["horizon"],
            value=d["value"],
            quality=d["quality"],
            missing_reason=d.get("missing_reason"),
            source_signals=tuple(d.get("source_signals", ())),
            builder=d.get("builder", ""),
            unit=d.get("unit", "raw"),
            metadata=d.get("metadata"),
        )


# ── Validation ─────────────────────────────────────────────────


class FeatureValidationError(ValueError):
    """Raised when a caller chooses to convert validation errors to exceptions."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(f"{len(errors)} validation error(s): {'; '.join(errors)}")


def validate_feature(feature: EngineeredFeature) -> list[str]:
    """Validate a single feature record.

    Returns a list of human-readable error strings.  An empty list means
    the record is valid.  This is a **pure function** — it never raises.
    """
    errors: list[str] = []

    # ── Name ──
    if not feature.feature_name:
        errors.append("feature_name must be non-empty")
    elif not FEATURE_NAME_PATTERN.match(feature.feature_name):
        errors.append(
            f"feature_name '{feature.feature_name}' must match pattern "
            f"'{{domain}}.{{metric}}[.{{qualifier}}]' (lowercase, 2-4 dot-segments)"
        )

    # ── Version ──
    if not isinstance(feature.version, int) or feature.version < 1:
        errors.append("version must be a positive integer (>= 1)")

    # ── Temporal: effective_at <= computed_at (point-in-time safety) ──
    if feature.effective_at > feature.computed_at:
        errors.append("effective_at must not be after computed_at (look-ahead / information-leakage violation)")

    # Reasonable range guards
    now = time.time()
    if feature.effective_at < _EPOCH_FLOOR:
        errors.append(f"effective_at ({feature.effective_at}) is before 2020-01-01 — likely a bug")
    if feature.computed_at > now + _EPOCH_CEILING_DRIFT:
        errors.append(f"computed_at ({feature.computed_at}) is more than 1 day in the future — likely a bug")

    # ── Horizon ──
    if feature.horizon not in VALID_HORIZONS:
        errors.append(f"horizon '{feature.horizon}' not recognized; must be one of {sorted(VALID_HORIZONS)}")

    # ── Value + missingness consistency ──
    if feature.value is None:
        if feature.missing_reason is None:
            errors.append("missing_reason is required when value is None")
        elif not feature.missing_reason.strip():
            errors.append("missing_reason must be non-empty when value is None")
    else:
        if feature.missing_reason is not None:
            errors.append("missing_reason must be None when value is present")
        # NaN / Inf check
        if math.isnan(feature.value) or math.isinf(feature.value):
            errors.append("value must be finite (no NaN or Inf); use value=None with missing_reason for missing data")

    # ── Quality ──
    if not isinstance(feature.quality, (int, float)):
        errors.append("quality must be a number")
    elif math.isnan(feature.quality) or math.isinf(feature.quality):
        errors.append("quality must be finite")
    elif not (0.0 <= feature.quality <= 1.0):
        errors.append("quality must be in [0.0, 1.0]")

    # ── Unit ──
    if feature.unit not in VALID_UNITS:
        errors.append(f"unit '{feature.unit}' not recognized; must be one of {sorted(VALID_UNITS)}")

    # ── Source signals ──
    if not feature.source_signals:
        errors.append("source_signals must be non-empty (at least one source)")

    # ── Builder ──
    if not feature.builder or not feature.builder.strip():
        errors.append("builder must be non-empty")

    # ── Metadata JSON-safety ──
    if feature.metadata is not None:
        if not isinstance(feature.metadata, dict):
            errors.append("metadata must be a dict or None")

    return errors


def validate_features(features: list[EngineeredFeature]) -> dict[int, list[str]]:
    """Validate a batch of features.

    Returns ``{index: [errors]}`` for every feature that has errors.
    An empty dict means the entire batch is valid.
    """
    bad: dict[int, list[str]] = {}
    for idx, feat in enumerate(features):
        errs = validate_feature(feat)
        if errs:
            bad[idx] = errs
    return bad
