"""GNN Feature Builder — entity embedding aggregation into EngineeredFeatures.

Converts HetTGN entity embeddings into 11 scalar EngineeredFeature records
that the WorldModel Bayesian DAG and Kalman filter can consume.

Features produced (per entity type with edges):
    - gnn.{type}_anomaly.spot — centroid deviation z-score (stress signal)
    - gnn.{type}_activity.spot — mean embedding norm (activity level)

Plus one cross-entity correlation measure:
    - gnn.cross_entity.spot — mean pairwise cosine of type centroids

Mathematical basis:
    For entity type τ with n_τ entities and embeddings {h_i^τ}:
        μ_τ = (1/n_τ) Σ h_i^τ                       (centroid)
        anomaly_τ = mean(||h_i^τ - μ_τ||)            (mean deviation)
        activity_τ = mean(||h_i^τ||)                  (mean norm)
        cross_entity = (2/K(K-1)) Σ cos(μ_τ1, μ_τ2)  (type correlation)

    All values are z-scored against rolling historical baseline from store.

References:
    - Spec: docs/specs/world_model_bridge_spec.md (step 19b)
    - Research: docs/research/world_model_bridge.md
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from agent.features.builders import FeatureBuilder
from agent.features.protocol import EngineeredFeature
from agent.pipeline.store import PipelineStore

log = logging.getLogger(__name__)

# Entity types that have GNN edges (from Phase 17 entity linking)
# Tier 8 (Change 16): Use get_connected_types() for dynamic resolution
_SEED_CONNECTED_TYPES: tuple[str, ...] = (
    "person",
    "company",
    "wallet",
    "country",
    "vessel",
)

# Backward-compatible alias for code that reads the old name
_CONNECTED_TYPES = _SEED_CONNECTED_TYPES

# Minimum entities per type to produce a meaningful feature
_MIN_ENTITIES_DEFAULT = 2

# Rolling z-score lookback (how many prior feature values to use)
_ZSCORE_LOOKBACK = 30

# Minimum historical values for z-score normalization
_MIN_HISTORY_FOR_ZSCORE = 3


def get_connected_types(
    store: PipelineStore | None = None,
    registry: Any = None,
    min_entities: int = _MIN_ENTITIES_DEFAULT,
    min_links: int = 1,
) -> tuple[str, ...]:
    """Return entity types eligible for GNN embedding.

    A dynamically discovered type is included when it has ≥ *min_entities*
    entities AND ≥ *min_links* links to other connected types.

    Without a store or registry, returns the seed types.
    """
    if store is None or registry is None:
        return _SEED_CONNECTED_TYPES

    candidates = set(_SEED_CONNECTED_TYPES)

    try:
        type_rows = registry.query_entity_types(active_only=True) if hasattr(registry, "query_entity_types") else []
        known_active = registry.known_entity_types() if hasattr(registry, "known_entity_types") else set()
    except Exception:
        return _SEED_CONNECTED_TYPES

    for t in known_active:
        if t in candidates:
            continue
        try:
            entities = store.query_all_entities(entity_type=t)
            if len(entities) < min_entities:
                continue
            # Check links involving this type
            link_count = 0
            for ent in entities[:10]:  # sample up to 10
                links = store.query_entity_links(entity_id=ent["entity_id"])
                link_count += len(links)
                if link_count >= min_links:
                    break
            if link_count >= min_links:
                candidates.add(t)
        except Exception:
            continue

    return tuple(sorted(candidates))


class GNNFeatureBuilder(FeatureBuilder):
    """Aggregate GNN entity embeddings into model-ready EngineeredFeatures.

    Produces 11 features:
        5 × anomaly (centroid deviation per type)
        5 × activity (embedding norm per type)
        1 × cross_entity (pairwise type centroid cosine)

    Parameters
    ----------
    model_path : str | Path | None
        Path to a saved GNN model checkpoint. If None or not found,
        behavior depends on ``train_if_missing``.
    min_entities_per_type : int
        Minimum entities of a type for a valid feature. Types below
        this threshold get ``value=None, quality=0.0``.
    train_if_missing : bool
        If True and no model exists, train a fresh GNN before inference.
        If False and no model exists, return features from a randomly-
        initialized model (quality reduced by 0.5).
    trainer_config : dict | None
        Config overrides for TrainerConfig (hidden_dim, epochs, etc).
    """

    VERSION = 1

    def __init__(
        self,
        model_path: str | Path | None = None,
        min_entities_per_type: int = _MIN_ENTITIES_DEFAULT,
        train_if_missing: bool = True,
        trainer_config: dict[str, Any] | None = None,
    ) -> None:
        self._model_path = Path(model_path) if model_path else None
        self._min_entities = min_entities_per_type
        self._train_if_missing = train_if_missing
        self._trainer_config_overrides = trainer_config or {}

    @property
    def name(self) -> str:
        return "GNNFeatureBuilder"

    def build(
        self,
        store: PipelineStore,
        as_of: float,
    ) -> list[EngineeredFeature]:
        """Produce 11 GNN-derived features.

        Pipeline:
        1. Load or train GNN model
        2. Run inference (until=as_of for point-in-time safety)
        3. Compute per-type anomaly and activity
        4. Compute cross-entity correlation
        5. Z-score normalize against rolling history from store
        6. Return EngineeredFeature list
        """
        # Lazy imports to avoid torch at module level
        try:
            import torch

            from agent.models.gnn.trainer import Trainer, TrainerConfig
        except ImportError:
            log.warning("torch not available — returning missing features.")
            return self._all_missing(as_of, "torch_not_available")

        # Build config
        cfg = TrainerConfig(**self._trainer_config_overrides)

        # Load or create trainer
        trainer = self._get_trainer(store, cfg, Trainer)
        if trainer is None:
            return self._all_missing(as_of, "model_init_failed")

        # Run inference
        try:
            embeddings, id_map = trainer.infer(until=as_of)
        except Exception:
            log.exception("GNN inference failed.")
            return self._all_missing(as_of, "inference_failed")

        if not embeddings:
            return self._all_missing(as_of, "no_entities_in_graph")

        # Compute raw features per entity type
        features: list[EngineeredFeature] = []
        centroids: dict[str, Any] = {}  # type → centroid tensor

        for etype in _CONNECTED_TYPES:
            emb = embeddings.get(etype)

            if emb is None or emb.shape[0] < self._min_entities:
                features.extend(self._missing_type_features(as_of, etype, "insufficient_entities"))
                continue

            centroid = emb.mean(dim=0)
            centroids[etype] = centroid

            # Anomaly: mean distance from centroid
            deviations = torch.norm(emb - centroid, dim=1)
            raw_anomaly = deviations.mean().item()

            # Activity: mean embedding norm
            norms = torch.norm(emb, dim=1)
            raw_activity = norms.mean().item()

            # Quality based on entity count: min(1.0, count/10)
            entity_count = emb.shape[0]
            base_quality = min(1.0, entity_count / 10.0)

            # Z-score normalize
            anom_z, anom_quality = self._zscore_from_history(
                store, f"gnn.{etype}_anomaly.spot", raw_anomaly, base_quality
            )
            act_z, act_quality = self._zscore_from_history(
                store, f"gnn.{etype}_activity.spot", raw_activity, base_quality
            )

            features.append(
                EngineeredFeature(
                    feature_name=f"gnn.{etype}_anomaly.spot",
                    version=self.VERSION,
                    effective_at=as_of,
                    computed_at=as_of,
                    horizon="spot",
                    value=anom_z,
                    quality=anom_quality,
                    source_signals=(f"gnn.{etype}.embedding",),
                    builder=self.name,
                    unit="z_score",
                    missing_reason=None,
                )
            )
            features.append(
                EngineeredFeature(
                    feature_name=f"gnn.{etype}_activity.spot",
                    version=self.VERSION,
                    effective_at=as_of,
                    computed_at=as_of,
                    horizon="spot",
                    value=act_z,
                    quality=act_quality,
                    source_signals=(f"gnn.{etype}.embedding",),
                    builder=self.name,
                    unit="z_score",
                    missing_reason=None,
                )
            )

        # Cross-entity correlation: mean pairwise cosine of centroids
        cross_val, cross_quality = self._compute_cross_entity(store, centroids, as_of)
        features.append(
            EngineeredFeature(
                feature_name="gnn.cross_entity.spot",
                version=self.VERSION,
                effective_at=as_of,
                computed_at=as_of,
                horizon="spot",
                value=cross_val,
                quality=cross_quality,
                source_signals=tuple(f"gnn.{t}.centroid" for t in centroids),
                builder=self.name,
                unit="z_score" if len(centroids) >= 2 else "raw",
                missing_reason=("insufficient_types" if len(centroids) < 2 else None),
            )
        )

        return features

    # ── Private helpers ────────────────────────────────────────

    def _get_trainer(
        self,
        store: PipelineStore,
        cfg: Any,
        TrainerCls: type,
    ) -> Any | None:
        """Load saved model or create/train a new one."""
        try:
            if self._model_path and self._model_path.exists():
                log.debug("Loading GNN model from %s", self._model_path)
                return TrainerCls.load_model(self._model_path, store)
        except Exception:
            log.warning("Failed to load model from %s — will rebuild.", self._model_path)

        try:
            trainer = TrainerCls(store, cfg)
            trainer.build_model()

            if self._train_if_missing:
                log.info("Training GNN model (no saved checkpoint found).")
                trainer.train()
                if self._model_path:
                    try:
                        trainer.save_model(self._model_path)
                    except Exception:
                        log.warning("Failed to save model after training.")
            else:
                log.info("Using randomly-initialized GNN (train_if_missing=False).")

            return trainer
        except Exception:
            log.exception("Failed to create/train GNN model.")
            return None

    def _zscore_from_history(
        self,
        store: PipelineStore,
        feature_name: str,
        raw_value: float,
        base_quality: float,
    ) -> tuple[float, float]:
        """Z-score normalize raw_value against rolling history from store.

        Returns (z_scored_value, adjusted_quality).
        If insufficient history (<3 values), returns raw_value with quality*0.5.
        Handles NaN/inf raw values gracefully.
        """
        if not math.isfinite(raw_value):
            return 0.0, 0.0

        try:
            rows = store.query_features(
                feature_name,
                limit=_ZSCORE_LOOKBACK,
            )
        except Exception:
            rows = []

        values = []
        for r in rows:
            v = r.get("value")
            if v is not None and math.isfinite(v):
                values.append(v)

        if len(values) < _MIN_HISTORY_FOR_ZSCORE:
            # Not enough history — return raw value, reduced quality
            return raw_value, base_quality * 0.5

        arr = np.array(values)
        mu = arr.mean()
        sigma = arr.std()
        if sigma < 1e-10:
            return 0.0, base_quality

        z = (raw_value - mu) / sigma
        # Clip extreme z-scores
        z = max(-10.0, min(10.0, z))
        return z, base_quality

    def _compute_cross_entity(
        self,
        store: PipelineStore,
        centroids: dict[str, Any],
        as_of: float,
    ) -> tuple[float | None, float]:
        """Compute mean pairwise cosine similarity of type centroids.

        Returns (value, quality). If < 2 types, returns (None, 0.0).
        """
        import torch

        types_with_centroids = list(centroids.keys())
        k = len(types_with_centroids)

        if k < 2:
            return None, 0.0

        # Compute pairwise cosines
        cosines: list[float] = []
        for i in range(k):
            for j in range(i + 1, k):
                c_i = centroids[types_with_centroids[i]]
                c_j = centroids[types_with_centroids[j]]
                cos_sim = torch.nn.functional.cosine_similarity(c_i.unsqueeze(0), c_j.unsqueeze(0)).item()
                cosines.append(cos_sim)

        raw_cross = float(np.mean(cosines))
        base_quality = min(1.0, k / 5.0)  # full quality at 5 types

        z, quality = self._zscore_from_history(store, "gnn.cross_entity.spot", raw_cross, base_quality)
        return z, quality

    def _missing_type_features(
        self,
        as_of: float,
        etype: str,
        reason: str,
    ) -> list[EngineeredFeature]:
        """Emit missing anomaly + activity features for a type."""
        return [
            EngineeredFeature(
                feature_name=f"gnn.{etype}_anomaly.spot",
                version=self.VERSION,
                effective_at=as_of,
                computed_at=as_of,
                horizon="spot",
                value=None,
                quality=0.0,
                missing_reason=reason,
                source_signals=(f"gnn.{etype}.embedding",),
                builder=self.name,
                unit="z_score",
            ),
            EngineeredFeature(
                feature_name=f"gnn.{etype}_activity.spot",
                version=self.VERSION,
                effective_at=as_of,
                computed_at=as_of,
                horizon="spot",
                value=None,
                quality=0.0,
                missing_reason=reason,
                source_signals=(f"gnn.{etype}.embedding",),
                builder=self.name,
                unit="z_score",
            ),
        ]

    def _all_missing(
        self,
        as_of: float,
        reason: str,
    ) -> list[EngineeredFeature]:
        """Emit all 11 features as explicitly missing."""
        features: list[EngineeredFeature] = []
        for etype in _CONNECTED_TYPES:
            features.extend(self._missing_type_features(as_of, etype, reason))
        features.append(
            EngineeredFeature(
                feature_name="gnn.cross_entity.spot",
                version=self.VERSION,
                effective_at=as_of,
                computed_at=as_of,
                horizon="spot",
                value=None,
                quality=0.0,
                missing_reason=reason,
                source_signals=tuple(f"gnn.{t}.centroid" for t in _CONNECTED_TYPES),
                builder=self.name,
                unit="z_score",
            )
        )
        return features
