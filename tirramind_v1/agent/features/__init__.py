"""
TirraMind — Engineered Features Package

Model-ready quantitative state variables derived from pipeline signals and data.

Provides:
    EngineeredFeature  — immutable feature record (the protocol contract)
    validate_feature   — pure validation returning error list
    VALID_HORIZONS     — recognized temporal horizons
    VALID_UNITS        — recognized measurement units

Builders (import directly from agent.features.builders):
    FeatureBuilder     — abstract base class for builders
    ConvergenceFeatureBuilder — convergence-derived aggregate features
    MacroStateFeatureBuilder  — continuous macro state features
"""

from agent.features.protocol import (
    VALID_HORIZONS,
    VALID_UNITS,
    EngineeredFeature,
    FeatureValidationError,
    validate_feature,
    validate_features,
)

__all__ = [
    "EngineeredFeature",
    "FeatureValidationError",
    "VALID_HORIZONS",
    "VALID_UNITS",
    "validate_feature",
    "validate_features",
]
