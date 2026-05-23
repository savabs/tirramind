"""
TirraMind — EntityAlert dataclass

Immutable record of per-entity prediction-surprise anomaly signals.

The five surprise signals are extracted from the HetTGN's self-supervised
predictions vs. actual observations. Statistical monitor values (CUSUM,
Hawkes, Event Study) are stored for traceability but their primary role
is as GNN input features, not as anomaly scores.

Design principles:
    1. Immutable (frozen dataclass) — never mutated after creation.
    2. Surprise-first — the 5 prediction-surprise signals are the anomaly.
    3. Type-agnostic — same structure for person, company, wallet, vessel, etc.
    4. No archetypes — no pattern matching, no named anomaly categories.

References:
    - Spec: docs/specs/signal_fusion_spec.md (Step 20.1)
    - Research: docs/research/signal_fusion.md (Paradigm Revision)
    - BeliefState protocol: agent/models/belief.py (design mirror)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EntityAlert:
    """Per-entity prediction-surprise anomaly record.

    Five surprise signals from HetTGN self-supervised predictions:
        obs_type_surprise:       -log P(actual_obs_type | h_i)
        temporal_surprise:       |dt_pred - dt_actual|, z-scored per type
        value_surprise:          |v_pred - v_actual| / sigma_type
        neighborhood_surprise:   attention-weighted avg neighbor composite surprise
        memory_drift:            L2 norm of GRU memory state change

    Enrichment features (inputs to GNN, stored for traceability):
        cusum_statistic:   current CUSUM accumulator value
        hawkes_intensity:  Hawkes process intensity at alert_time
        event_study_score: standardized abnormal score vs. entity baseline
    """

    # Identity
    entity_id: str
    entity_type: str
    entity_name: str
    alert_time: float  # unix epoch

    # Five prediction-surprise signals (the anomaly)
    obs_type_surprise: float
    temporal_surprise: float
    value_surprise: float
    neighborhood_surprise: float
    memory_drift: float

    # Enrichment features (GNN inputs, stored for traceability)
    cusum_statistic: float
    hawkes_intensity: float
    event_study_score: float

    # Composite
    composite_surprise: float  # weighted combination of the 5 surprise signals

    # Metadata
    observation_count: int
    evidence_sources: tuple[str, ...]
    metadata: dict | None = None
