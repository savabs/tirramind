"""TirraMind — GNN Integration Layer (Phases 12f, 15d, 16a, 16b)

Provides:
    AutoPatternDetector      — Run crystallized GNN patterns through the
                               entity graph and store results as
                               cross_entity_pattern observations.
    retrain_and_discover     — End-to-end: train model → extract patterns →
                               crystallize → return production rules + diagnostics.
    compare_patterns         — Compare MI/hit-rate of auto vs hand-crafted
                               cross-entity patterns.
    compute_diagnostics      — Entity-type density, edge-type attention
                               distribution, neighborhood sparsity, and
                               supervised-head confidence by entity type
                               (Phase 15d) — feeds GNN-guided expansion.
    format_diagnostic_report — Annotate raw diagnostics with threshold
                               flags for guided tool expansion (Phase 16a).
    run_diagnostics          — CLI-callable entry point: open a real
                               PipelineStore → train → diagnose → report
                               (Phase 16b).

Auto-discovered patterns produce observations with
``metadata_json.source = "auto_gnn"`` so they can be distinguished
from hand-crafted patterns (``metadata_json.pattern_type`` in
insider_x_gdelt, vessel_x_sanctions, whale_x_geopolitical).

References:
    Spec steps: 12f.1, 12f.2, 12f.3, 15d.1, 15d.2, 16a.1, 16b.1.
"""

from __future__ import annotations

import logging
import os
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.models.gnn.graph_builder import GraphBuilder
from agent.models.gnn.het_tgn import HetTGN
from agent.models.gnn.pattern_extractor import (
    CrystallizedPattern,
    PatternExtractor,
    crystallize,
    extract_temporal_lags,
)
from agent.models.gnn.trainer import (
    FineTuner,
    Trainer,
    TrainerConfig,
    generate_outcome_labels,
)
from agent.pipeline.store import PipelineStore

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# AutoPatternDetector — run crystallized patterns
# ═══════════════════════════════════════════════════════════════


class AutoPatternDetector:
    """Detect cross-entity patterns using GNN-crystallized rules.

    Each CrystallizedPattern specifies:
        source_type, target_type, via_edge → find linked entity pairs
        obs_type_a, obs_type_b           → source and target obs types
        window_seconds                    → temporal window for co-occurrence

    For each linked (source, target) pair, check whether obs_type_a on
    the source entity is temporally close to obs_type_b on the target
    within ``window_seconds``.  If so, store a ``cross_entity_pattern``
    observation.
    """

    def __init__(self, store: PipelineStore) -> None:
        self.store = store

    def detect(
        self,
        patterns: list[CrystallizedPattern],
        since: float | None = None,
        until: float | None = None,
    ) -> list[dict[str, Any]]:
        """Run all crystallized patterns and return detected co-occurrences.

        Args:
            patterns: CrystallizedPattern rules to evaluate.
            since: Only consider observations after this timestamp.
            until: Only consider observations before this timestamp.

        Returns:
            List of detection dicts ready for store_observations().
        """
        detections: list[dict[str, Any]] = []

        # Pre-fetch all entities and build type map
        all_entities = self.store.query_all_entities()
        eid_to_type = {e["entity_id"]: e["entity_type"] for e in all_entities}

        # Pre-fetch all observations
        all_obs = self.store.query_all_observations(since=since, until=until)
        obs_by_entity: dict[str, list[dict]] = {}
        for o in all_obs:
            eid = o.get("entity_id")
            if eid:
                obs_by_entity.setdefault(eid, []).append(o)

        # Pre-fetch all links
        all_links = self.store.query_all_entity_links()
        link_index: dict[tuple[str, str], list[str]] = {}
        for lnk in all_links:
            key = (lnk["entity_id_a"], lnk["link_type"])
            link_index.setdefault(key, []).append(lnk["entity_id_b"])

        for pattern in patterns:
            hits = self._detect_one(pattern, eid_to_type, obs_by_entity, link_index)
            detections.extend(hits)

        return detections

    def _detect_one(
        self,
        pattern: CrystallizedPattern,
        eid_to_type: dict[str, str],
        obs_by_entity: dict[str, list[dict]],
        link_index: dict[tuple[str, str], list[str]],
    ) -> list[dict[str, Any]]:
        """Detect co-occurrences for a single crystallized pattern."""
        hits: list[dict[str, Any]] = []

        for (eid_a, link_type), targets in link_index.items():
            if link_type != pattern.via_edge:
                continue
            src_type = eid_to_type.get(eid_a)
            if src_type != pattern.source_type:
                continue

            src_obs = [
                o
                for o in obs_by_entity.get(eid_a, [])
                if o.get("observation_type") == pattern.obs_type_a
            ]
            if not src_obs:
                continue

            for eid_b in targets:
                dst_type = eid_to_type.get(eid_b)
                if dst_type != pattern.target_type:
                    continue

                dst_obs = [
                    o
                    for o in obs_by_entity.get(eid_b, [])
                    if o.get("observation_type") == pattern.obs_type_b
                ]
                if not dst_obs:
                    continue

                # Find temporal co-occurrences within window
                for so in src_obs:
                    st = so.get("observed_at", 0.0)
                    for do in dst_obs:
                        dt = do.get("observed_at", 0.0)
                        lag = dt - st
                        if 0 < lag <= pattern.window_seconds:
                            score = 1.0 - (lag / pattern.window_seconds)
                            hits.append(
                                {
                                    "entity_a": eid_a,
                                    "entity_b": eid_b,
                                    "source_type": pattern.source_type,
                                    "target_type": pattern.target_type,
                                    "via_edge": pattern.via_edge,
                                    "obs_type_a": pattern.obs_type_a,
                                    "obs_type_b": pattern.obs_type_b,
                                    "source_time": st,
                                    "target_time": dt,
                                    "lag": lag,
                                    "score": score,
                                    "pattern_source": "auto_gnn",
                                }
                            )

        return hits

    def store_observations(self, detections: list[dict[str, Any]]) -> int:
        """Store detected patterns as cross_entity_pattern observations.

        Args:
            detections: Output from detect().

        Returns:
            Number of observations stored.
        """
        count = 0
        for d in detections:
            self.store.store_entity_observation(
                entity_id=d["entity_a"],
                source_tool="auto_gnn",
                observed_at=d["source_time"],
                observation_type="cross_entity_pattern",
                value=d,
                depth_level=3,
                metadata={
                    "source": "auto_gnn",
                    "pattern_type": f"{d['obs_type_a']}_x_{d['obs_type_b']}",
                },
            )
            count += 1
        return count


# ═══════════════════════════════════════════════════════════════
# Comparative evaluation
# ═══════════════════════════════════════════════════════════════


def compare_patterns(store: PipelineStore) -> dict[str, Any]:
    """Compare auto-discovered vs hand-crafted cross-entity patterns.

    Groups all cross_entity_pattern observations by source and computes
    basic statistics: count, mean score, score distribution.

    Returns:
        Dict with 'auto_gnn' and 'hand_crafted' sub-dicts containing
        count, mean_score, and score_quartiles.
    """
    all_obs = store.query_all_observations()
    ce_obs = [o for o in all_obs if o.get("observation_type") == "cross_entity_pattern"]

    auto_scores: list[float] = []
    hand_scores: list[float] = []

    for o in ce_obs:
        value = o.get("value", {})
        if isinstance(value, str):
            import json

            try:
                value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                value = {}

        source = value.get("pattern_source", "hand_crafted")
        score = value.get("score", 0.0)
        if isinstance(score, (int, float)):
            if source == "auto_gnn":
                auto_scores.append(score)
            else:
                hand_scores.append(score)

    def _stats(scores: list[float]) -> dict[str, Any]:
        if not scores:
            return {"count": 0, "mean_score": 0.0}
        import torch

        t = torch.tensor(scores, dtype=torch.float)
        result: dict[str, Any] = {
            "count": len(scores),
            "mean_score": t.mean().item(),
        }
        if len(scores) >= 4:
            result["p25"] = t.quantile(0.25).item()
            result["p50"] = t.quantile(0.50).item()
            result["p75"] = t.quantile(0.75).item()
        return result

    return {
        "auto_gnn": _stats(auto_scores),
        "hand_crafted": _stats(hand_scores),
    }


# ═══════════════════════════════════════════════════════════════
# Periodic retraining
# ═══════════════════════════════════════════════════════════════


def compute_diagnostics(
    model: HetTGN,
    store: PipelineStore,
    crystallized: list[CrystallizedPattern] | None = None,
) -> dict[str, Any]:
    """Compute GNN diagnostics for guided tool expansion (Phase 15d).

    Returns a diagnostic dict with:
        entity_type_density  — {type: count} for all entity types.
        observation_density  — {obs_type: count} across all observations.
        edge_type_attention  — {edge_type: mean_attention} from HGT layers.
        neighborhood_sparsity — {entity_type: mean_degree} across entity types.
        supervised_confidence — {entity_type: mean_prob} from supervised head
                                (if crystallized patterns supplied).

    These diagnostics feed the GNN-guided expansion loop: sparse entity
    types or edge types with low attention signal where the next data
    tool should be added.

    Args:
        model: Trained HetTGN.
        store: PipelineStore.
        crystallized: Optional crystallized patterns for supervised
                      confidence computation.

    Returns:
        Diagnostic dict.
    """
    import torch

    all_entities = store.query_all_entities()
    all_links = store.query_all_entity_links()

    # 1. Entity-type density
    entity_type_density: dict[str, int] = {}
    eid_to_type: dict[str, str] = {}
    for e in all_entities:
        etype = e["entity_type"]
        entity_type_density[etype] = entity_type_density.get(etype, 0) + 1
        eid_to_type[e["entity_id"]] = etype

    # 2. Observation density — aggregate via SQL to avoid loading ~1M rows
    conn = store._get_conn()
    obs_density: dict[str, int] = {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT observation_type, COUNT(*) FROM entity_observations GROUP BY observation_type"
        ).fetchall()
    }

    # 3. Edge-type attention from HGT layers
    edge_type_attention: dict[str, float] = {}
    data = None
    id_map = None
    try:
        graph_builder = GraphBuilder(store)
        data, id_map, _ = graph_builder.build()
        if data.node_types:
            attn_weights = model.get_attention_weights(data, id_map)
            edge_type_attention = {
                ("→".join(k) if isinstance(k, tuple) else str(k)): (
                    v.item() if hasattr(v, "item") else float(v)
                )
                for k, v in attn_weights.items()
            }
    except Exception as exc:
        log.warning("Could not compute attention weights: %s", exc)

    # 4. Neighborhood sparsity — mean degree per entity type
    degree_counts: dict[str, list[int]] = {}
    entity_degree: dict[str, int] = {}
    for lnk in all_links:
        for eid_key in ("entity_id_a", "entity_id_b"):
            eid = lnk[eid_key]
            entity_degree[eid] = entity_degree.get(eid, 0) + 1

    for eid, deg in entity_degree.items():
        etype = eid_to_type.get(eid)
        if etype:
            degree_counts.setdefault(etype, []).append(deg)
    # Include zero-degree entities
    for e in all_entities:
        if e["entity_id"] not in entity_degree:
            degree_counts.setdefault(e["entity_type"], []).append(0)

    neighborhood_sparsity: dict[str, float] = {}
    for etype, degrees in degree_counts.items():
        neighborhood_sparsity[etype] = sum(degrees) / max(len(degrees), 1)

    # 5. Supervised confidence by entity type
    supervised_confidence: dict[str, float] = {}
    if crystallized and data is not None and id_map is not None:
        try:
            labels = generate_outcome_labels(crystallized, store)
            if labels and data.node_types:
                model.eval()
                with torch.no_grad():
                    embeddings = model(data, id_map)

                type_probs: dict[str, list[float]] = {}
                for lbl in labels:
                    src_local = id_map.local_id(lbl.src_type, lbl.src_entity_id)
                    dst_local = id_map.local_id(lbl.dst_type, lbl.dst_entity_id)
                    if src_local is None or dst_local is None:
                        continue
                    if lbl.src_type not in embeddings or lbl.dst_type not in embeddings:
                        continue
                    if src_local >= embeddings[lbl.src_type].size(0):
                        continue
                    if dst_local >= embeddings[lbl.dst_type].size(0):
                        continue

                    src_emb = embeddings[lbl.src_type][src_local].unsqueeze(0)
                    dst_emb = embeddings[lbl.dst_type][dst_local].unsqueeze(0)
                    prob = model.predict_outcome(src_emb, dst_emb).item()
                    type_probs.setdefault(lbl.src_type, []).append(prob)

                for etype, probs in type_probs.items():
                    supervised_confidence[etype] = sum(probs) / len(probs)
        except Exception as exc:
            log.warning("Could not compute supervised confidence: %s", exc)

    return {
        "entity_type_density": entity_type_density,
        "observation_density": obs_density,
        "edge_type_attention": edge_type_attention,
        "neighborhood_sparsity": neighborhood_sparsity,
        "supervised_confidence": supervised_confidence,
    }


# ═══════════════════════════════════════════════════════════════
# Diagnostic report formatting (Phase 16a)
# ═══════════════════════════════════════════════════════════════

# Provisional thresholds (see research doc for rationale).
_THRESH_ENTITY_DENSITY = 5
_THRESH_OBS_DENSITY = 10
_THRESH_ATTENTION = 0.05
_THRESH_MEAN_DEGREE = 1.0
_THRESH_CONFIDENCE_LO = 0.4
_THRESH_CONFIDENCE_HI = 0.6


def format_diagnostic_report(
    diagnostics: dict[str, Any],
    *,
    entity_density_min: int = _THRESH_ENTITY_DENSITY,
    obs_density_min: int = _THRESH_OBS_DENSITY,
    attention_min: float = _THRESH_ATTENTION,
    mean_degree_min: float = _THRESH_MEAN_DEGREE,
    confidence_lo: float = _THRESH_CONFIDENCE_LO,
    confidence_hi: float = _THRESH_CONFIDENCE_HI,
) -> dict[str, Any]:
    """Format raw diagnostics into a structured report with threshold flags.

    Takes the dict returned by ``compute_diagnostics()`` and annotates each
    stream with ``flagged`` entries that violate the given thresholds.

    Args:
        diagnostics: Output from ``compute_diagnostics()``.
        entity_density_min: Minimum entity count to be considered healthy.
        obs_density_min: Minimum observation count per type.
        attention_min: Minimum mean-attention for an edge type.
        mean_degree_min: Minimum mean-degree for an entity type.
        confidence_lo / confidence_hi: Supervised confidence band that
            indicates uncertainty (values within this range are flagged).

    Returns:
        Dict with sections: ``entity_density``, ``observation_density``,
        ``edge_attention``, ``neighborhood_sparsity``, ``supervised_confidence``.
        Each section has ``values`` (original data) and ``flagged`` (list of
        items that violate the threshold).  Top-level ``summary`` key gives
        counts of flagged items per section.
    """
    etd = diagnostics.get("entity_type_density", {})
    obd = diagnostics.get("observation_density", {})
    eta = diagnostics.get("edge_type_attention", {})
    nsp = diagnostics.get("neighborhood_sparsity", {})
    svc = diagnostics.get("supervised_confidence", {})

    flagged_entities = {k: v for k, v in etd.items() if v < entity_density_min}
    flagged_obs = {k: v for k, v in obd.items() if v < obs_density_min}
    flagged_attention = {k: v for k, v in eta.items() if v < attention_min}
    flagged_sparsity = {k: v for k, v in nsp.items() if v < mean_degree_min}
    flagged_confidence = {
        k: v for k, v in svc.items() if confidence_lo <= v <= confidence_hi
    }

    return {
        "entity_density": {"values": etd, "flagged": flagged_entities},
        "observation_density": {"values": obd, "flagged": flagged_obs},
        "edge_attention": {"values": eta, "flagged": flagged_attention},
        "neighborhood_sparsity": {"values": nsp, "flagged": flagged_sparsity},
        "supervised_confidence": {"values": svc, "flagged": flagged_confidence},
        "summary": {
            "flagged_entity_types": len(flagged_entities),
            "flagged_obs_types": len(flagged_obs),
            "flagged_edge_types": len(flagged_attention),
            "flagged_sparse_types": len(flagged_sparsity),
            "flagged_uncertain_types": len(flagged_confidence),
        },
    }


def retrain_and_discover(
    store: PipelineStore,
    config: TrainerConfig | None = None,
    score_threshold: float = 0.1,
    top_k: int = 10,
    finetune: bool = False,
    finetune_epochs: int = 10,
    include_diagnostics: bool = True,
) -> dict[str, Any]:
    """End-to-end: train HetTGN → extract patterns → crystallize → diagnostics.

    This is the main entry point for periodic re-training.

    Args:
        store: PipelineStore with entity data.
        config: Training configuration.
        score_threshold: Minimum score for crystallization.
        top_k: Number of patterns to analyze for lags.
        finetune: Whether to fine-tune with outcome labels.
        finetune_epochs: Number of fine-tuning epochs.
        include_diagnostics: Whether to compute GNN diagnostics.

    Returns:
        Dict with 'patterns' (list[CrystallizedPattern]),
        'diagnostics' (optional), 'finetune_history' (optional),
        'supervised_metrics' (optional).
    """
    cfg = config or TrainerConfig()
    trainer = Trainer(store, cfg)
    model = trainer.build_model()

    log.info("Training HetTGN (%d epochs)...", cfg.epochs)
    history = trainer.train()
    final_loss = history["total"][-1] if history["total"] else float("nan")
    log.info("Training complete. Final loss: %.4f", final_loss)

    log.info("Extracting meta-path patterns...")
    extractor = PatternExtractor(model, store)
    patterns = extractor.extract_metapath_importance()
    log.info("Found %d meta-path patterns.", len(patterns))

    log.info("Analyzing temporal lags for top-%d patterns...", top_k)
    patterns = extract_temporal_lags(patterns, store, top_k=top_k)

    log.info("Crystallizing patterns (threshold=%.3f)...", score_threshold)
    crystallized = crystallize(patterns, store, threshold=score_threshold)
    log.info("Crystallized %d production rules.", len(crystallized))

    result: dict[str, Any] = {"patterns": crystallized}

    # Fine-tune if requested and patterns exist
    if finetune and crystallized:
        log.info("Generating outcome labels...")
        labels = generate_outcome_labels(crystallized, store)
        log.info("Generated %d outcome labels.", len(labels))

        if labels:
            ft = FineTuner(
                model,
                store,
                labels,
                epochs=finetune_epochs,
            )
            ft_history = ft.finetune()
            result["finetune_history"] = ft_history

            # Evaluate on the same labels (in production, split temporally)
            from agent.models.gnn.trainer import evaluate_supervised

            metrics = evaluate_supervised(model, store, labels)
            result["supervised_metrics"] = metrics
            log.info(
                "Supervised eval — AUROC: %.3f, F1: %.3f",
                metrics.get("auroc", 0),
                metrics.get("f1", 0),
            )

    # Diagnostics
    if include_diagnostics:
        log.info("Computing GNN diagnostics...")
        diagnostics = compute_diagnostics(
            model,
            store,
            crystallized if crystallized else None,
        )
        result["diagnostics"] = diagnostics
        sparse_types = [
            t for t, d in diagnostics["neighborhood_sparsity"].items() if d < 1.0
        ]
        if sparse_types:
            log.info(
                "Sparse entity types (mean degree < 1): %s",
                ", ".join(sparse_types),
            )

    return result


# ═══════════════════════════════════════════════════════════════
# CLI-callable diagnostics entry point (Phase 16b)
# ═══════════════════════════════════════════════════════════════


def run_diagnostics(
    db_path: str,
    *,
    config: TrainerConfig | None = None,
    score_threshold: float = 0.0,
    finetune: bool = False,
    finetune_epochs: int = 10,
) -> dict[str, Any]:
    """Open a PipelineStore from disk, train, diagnose, and return a report.

    This is the reusable entry point for running GNN diagnostics on a
    real (or any on-disk) PipelineStore.  It validates that the DB exists
    and contains enough data before attempting training.

    Args:
        db_path: Path to the PipelineStore SQLite file.
        config: Optional TrainerConfig overrides.
        score_threshold: Crystallization threshold (0.0 = keep all).
        finetune: Whether to fine-tune with outcome labels.
        finetune_epochs: Number of fine-tuning epochs if fine-tuning.

    Returns:
        Dict with keys:
            ``status``      — "ok", "empty_graph", or "error"
            ``diagnostics`` — raw compute_diagnostics() output (if ok)
            ``report``      — format_diagnostic_report() output (if ok)
            ``patterns``    — list of CrystallizedPattern (if ok)
            ``message``     — human-readable status string
            ``entity_count``— number of entities found
            ``obs_count``   — number of observations found

    Raises:
        FileNotFoundError: If *db_path* does not point to an existing file.
    """
    # ── Validate path ──────────────────────────────────────────
    resolved = Path(db_path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"PipelineStore DB not found: {resolved}")

    # ── Open store and check contents ──────────────────────────
    store = PipelineStore(db_path=str(resolved))
    try:
        entities = store.query_all_entities()
        observations = store.query_all_observations()
        entity_count = len(entities)
        obs_count = len(observations)

        if entity_count == 0:
            log.warning(
                "PipelineStore at %s has no entities — skipping diagnostics.", resolved
            )
            return {
                "status": "empty_graph",
                "diagnostics": None,
                "report": None,
                "patterns": [],
                "message": f"PipelineStore at {resolved} has 0 entities and {obs_count} observations. "
                "Diagnostics require at least one entity.",
                "entity_count": entity_count,
                "obs_count": obs_count,
            }

        log.info(
            "PipelineStore loaded: %d entities, %d observations.",
            entity_count,
            obs_count,
        )

        # ── Run the pipeline ───────────────────────────────────
        result = retrain_and_discover(
            store,
            config=config,
            score_threshold=score_threshold,
            finetune=finetune,
            finetune_epochs=finetune_epochs,
            include_diagnostics=True,
        )

        raw_diag = result.get("diagnostics", {})
        report = format_diagnostic_report(raw_diag)
        crystallized = result.get("patterns", [])

        return {
            "status": "ok",
            "diagnostics": raw_diag,
            "report": report,
            "patterns": crystallized,
            "message": (
                f"Diagnostics complete. {entity_count} entities, {obs_count} observations. "
                f"{report['summary']['flagged_entity_types']} entity types flagged, "
                f"{report['summary']['flagged_sparse_types']} sparse types, "
                f"{report['summary']['flagged_uncertain_types']} uncertain types."
            ),
            "entity_count": entity_count,
            "obs_count": obs_count,
        }

    except Exception as exc:
        log.error("Diagnostics failed on %s: %s", resolved, exc, exc_info=True)
        return {
            "status": "error",
            "diagnostics": None,
            "report": None,
            "patterns": [],
            "message": f"Diagnostics failed: {exc}",
            "entity_count": 0,
            "obs_count": 0,
        }
    finally:
        store.close()
