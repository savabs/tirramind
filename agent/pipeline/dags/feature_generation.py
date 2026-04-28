"""
TirraMind — Feature Generation DAG

Runs after convergence_detection completes. Reads stored signals and
pipeline data, runs all registered FeatureBuilders, and persists the
resulting EngineeredFeature records to the ``features`` table.

Schedule: weekdays at 19:00 UTC (30 min after convergence_detection).

All functions follow the FunctionOperator contract:
    fn(params: dict, upstream_results: dict) -> dict
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Any

from agent.features.builders import (
    ConvergenceFeatureBuilder,
    FeatureBuilder,
    MacroStateFeatureBuilder,
)
from agent.features.gnn_builder import GNNFeatureBuilder
from agent.features.protocol import EngineeredFeature, validate_feature
from agent.pipeline.dag import DAG
from agent.pipeline.regime_gate import feature_trust_scale, get_current_regime
from agent.pipeline.store import PipelineStore

log = logging.getLogger(__name__)

# Default set of builders — extend this list as new builders are added.
DEFAULT_BUILDERS: list[FeatureBuilder] = [
    ConvergenceFeatureBuilder(),
    MacroStateFeatureBuilder(),
    GNNFeatureBuilder(model_path=".tirra_pipeline/gnn_model.pt"),
]


# ═══════════════════════════════════════════════════════════════
#  FunctionOperator Callback
# ═══════════════════════════════════════════════════════════════


def run_feature_generation(params: dict, upstream: dict) -> dict:
    """FunctionOperator callback for the feature_generation DAG.

    1. Open PipelineStore.
    2. Run each registered FeatureBuilder.
    3. Persist all valid features via store_features_batch().
    4. Return summary dict.

    Parameters (from ``params``):
        db_path : str
            Path to the pipeline SQLite database.
        as_of : float | None
            Reference time (unix epoch). Defaults to now.
        builders : list[FeatureBuilder] | None
            Override the default builder list (for testing).
    """
    db_path: str = params.get("db_path", ".tirra_pipeline/pipeline.db")
    as_of: float = params.get("as_of") or time.time()
    builders: list[FeatureBuilder] = (
        params.get("builders")
        if params.get("builders") is not None
        else DEFAULT_BUILDERS
    )

    store = PipelineStore(db_path)
    try:
        # ── Phase 49b: feature trust scale ────────────────────────────
        # When the regime gate detects recent instability (stability < 3d),
        # GNN-derived features are less trustworthy because the GNN was
        # trained on distribution data that may no longer apply.
        # Scale their values toward zero by the trust factor so downstream
        # consumers (world model, entity scoring) de-weight them proportionally.
        # Non-GNN features are unaffected — they come from stable structural
        # sources (filings, macro) that are regime-independent.
        # trust=1.0 in stable regimes → no change.
        try:
            regime_ctx = get_current_regime(store)
            trust = feature_trust_scale(regime_ctx)
        except Exception as exc:
            log.warning(
                "Phase 49b: feature trust scale check failed — defaulting to 1.0: %s",
                exc,
            )
            trust = 1.0

        all_features: list[EngineeredFeature] = []
        builder_summaries: list[dict[str, Any]] = []

        for builder in builders:
            try:
                features = builder.build(store, as_of)
                all_features.extend(features)
                builder_summaries.append(
                    {
                        "builder": builder.name,
                        "features_produced": len(features),
                        "missing": sum(1 for f in features if f.value is None),
                    }
                )
                log.debug(
                    "Builder %s produced %d features (%d missing).",
                    builder.name,
                    len(features),
                    sum(1 for f in features if f.value is None),
                )
            except Exception:
                log.exception("Builder %s failed — skipping.", builder.name)
                builder_summaries.append(
                    {
                        "builder": builder.name,
                        "features_produced": 0,
                        "error": True,
                    }
                )

        # Persist all features in one batch
        # ── Phase 49b: apply trust scaling to GNN-derived features ────
        if trust < 1.0:
            scaled: list[EngineeredFeature] = []
            n_scaled = 0
            for feat in all_features:
                if feat.feature_name.startswith("gnn.") and feat.value is not None:
                    scaled.append(dataclasses.replace(feat, value=feat.value * trust))
                    n_scaled += 1
                else:
                    scaled.append(feat)
            all_features = scaled
            log.info(
                "Phase 49b: scaled %d GNN feature values by trust=%.2f "
                "(stability < 3d, regime=%s).",
                n_scaled,
                trust,
                regime_ctx.regime_label,
            )

        stored_count = 0
        if all_features:
            try:
                row_ids = store.store_features_batch(all_features)
                stored_count = len(row_ids)
            except ValueError:
                # Batch validation failed — fall back to one-by-one
                log.warning("Batch store failed — falling back to individual inserts.")
                for feat in all_features:
                    try:
                        store.store_feature(feat)
                        stored_count += 1
                    except ValueError:
                        log.warning("Skipping invalid feature: %s", feat.feature_name)

        log.info(
            "Feature generation complete: %d produced, %d stored.",
            len(all_features),
            stored_count,
        )

        return {
            "produced": len(all_features),
            "stored": stored_count,
            "builders": builder_summaries,
        }

    finally:
        store.close()


# ═══════════════════════════════════════════════════════════════
#  DAG Builder
# ═══════════════════════════════════════════════════════════════


def build_feature_generation_dag(
    db_path: str = ".tirra_pipeline/pipeline.db",
) -> DAG:
    """Build the feature_generation DAG.

    Single node: ``generate_features`` (FunctionOperator).
    Schedule: weekdays at 19:00 UTC, 30 min after convergence_detection.
    """
    dag = DAG(
        name="feature_generation",
        schedule="0 19 * * 1-5",
        description=(
            "Engineered feature generation: transform pipeline signals "
            "and data into model-ready quantitative state variables"
        ),
    )

    dag.add(
        "generate_features",
        operator=run_feature_generation,
        params={"db_path": db_path},
        timeout=120,
        retries=1,
        store_result=True,
    )

    return dag
