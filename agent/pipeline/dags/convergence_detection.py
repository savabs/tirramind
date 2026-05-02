"""
TirraMind — Convergence Detection DAG

Runs after daily_collection completes. Reads stored tool outputs,
builds signal streams, detects multi-source convergence events,
and emits ``convergence.*`` signals to the pipeline store.

Schedule: weekdays at 18:30 UTC (30 min after daily_collection).

All functions follow the FunctionOperator contract:
    fn(params: dict, upstream_results: dict) -> dict
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from agent.convergence.detector import ConvergenceDetector, ConvergenceDetectorConfig
from agent.convergence.evidence import Evidence
from agent.convergence.extractors import extract_evidence, registered_tools
from agent.convergence.signals import emit_signals, from_detection_result
from agent.convergence.taxonomy import SignalMeta, SignalRegistry
from agent.pipeline.dag import DAG
from agent.pipeline.store import PipelineStore

log = logging.getLogger(__name__)

# Seconds per day.
_DAY = 86_400


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════


def _load_evidence_from_store(
    store: PipelineStore,
    lookback_days: int = 365,
    as_of: float | None = None,
) -> list[Evidence]:
    """Load all evidence from the pipeline store.

    Iterates every registered tool, queries stored data rows,
    and runs the appropriate extractor to produce Evidence objects.
    """
    if as_of is None:
        as_of = time.time()
    since = as_of - lookback_days * _DAY

    all_evidence: list[Evidence] = []
    for tool_name in registered_tools():
        rows = store.query_data(tool_name, since=since, until=as_of, limit=1000)
        for row in rows:
            data = row.get("data")
            if data is None:
                continue
            evidence = extract_evidence(tool_name, data)
            all_evidence.extend(evidence)

    log.debug("Pre-loaded %d evidence items from store.", len(all_evidence))
    return all_evidence


def build_registry_from_evidence(evidence: list[Evidence]) -> SignalRegistry:
    """Build a SignalRegistry from observed evidence.

    Each unique ``signal_id`` gets a :class:`SignalMeta` entry using
    the category and source from the evidence, with sensible defaults
    for frequency and direction semantics.

    If the same ``signal_id`` appears with conflicting categories,
    the first occurrence wins.
    """
    registry = SignalRegistry()
    seen: set[str] = set()

    for ev in evidence:
        if ev.signal_id in seen:
            continue
        seen.add(ev.signal_id)
        try:
            meta = SignalMeta(
                signal_id=ev.signal_id,
                source=ev.source,
                category=ev.category,
                frequency="daily",
                direction_semantics="auto-detected",
            )
            registry.register(meta)
        except (ValueError, TypeError):
            log.warning("Skipping unregisterable signal %s", ev.signal_id, exc_info=True)

    log.debug("Built registry with %d signals from evidence.", len(registry))
    return registry


def _load_convergence_thresholds(threshold_dir: Path) -> dict[str, float]:
    """Load GP-BO learned convergence thresholds if available.

    Returns a dict with keys matching ConvergenceDetectorConfig fields
    (z_threshold, p_threshold, fdr_q).  Empty dict if no trials exist.
    """
    from agent.learning.threshold_optimizer import ThresholdOptimizer

    try:
        opt = ThresholdOptimizer(persist_dir=threshold_dir)
    except Exception:
        log.warning("Could not load ThresholdOptimizer from %s", threshold_dir)
        return {}

    best = opt.current_best("convergence")
    if best is None:
        return {}

    # Map BO param names → ConvergenceDetectorConfig field names (they match)
    return {k: v for k, v in best.items() if k in ("z_threshold", "p_threshold", "fdr_q")}


# ═══════════════════════════════════════════════════════════════
#  FunctionOperator Callback
# ═══════════════════════════════════════════════════════════════


def run_convergence_detection(params: dict, upstream: dict) -> dict:
    """FunctionOperator callback for the convergence detection DAG.

    1. Open PipelineStore.
    2. Load evidence to discover known signal streams.
    3. Build SignalRegistry from observed evidence.
    4. Instantiate ConvergenceDetector.
    5. Run detection cycle.
    6. Convert results to ConvergenceSignals and emit.
    7. Return summary dict.
    """
    db_path = params.get("db_path", ".tirra_pipeline/pipeline.db")
    lookback_days = params.get("lookback_days", 365)
    as_of: float | None = params.get("as_of")  # None → now

    store = PipelineStore(db_path)
    try:
        # Build registry from whatever signals exist in the store
        evidence = _load_evidence_from_store(store, lookback_days, as_of)
        if not evidence:
            log.info("No evidence in store — skipping detection.")
            return {"detected": 0, "emitted": 0, "signals": []}

        registry = build_registry_from_evidence(evidence)

        # Configure and run detector
        # Merge learned convergence thresholds from GP-BO (Tier 3, Change 7)
        config_kwargs: dict[str, Any] = {"lookback_days": lookback_days}
        threshold_dir = params.get("threshold_dir")
        default_dir = Path(db_path).parent / "threshold_bo"
        _dir = Path(threshold_dir) if threshold_dir else (default_dir if default_dir.exists() else None)
        if _dir is not None:
            config_kwargs.update(_load_convergence_thresholds(_dir))

        config = ConvergenceDetectorConfig(**config_kwargs)
        detector = ConvergenceDetector(store, registry, config)
        results = detector.detect(as_of)

        # Convert detection results to convergence signals,
        # passing persistence count from the detector's history.
        history = detector.persistence_history
        signals = []
        for r in results:
            fp = r.clique.fingerprint()
            persist_count = history.get(fp, 0)
            signals.append(from_detection_result(r, persistence_count=persist_count, as_of=as_of))

        # Emit signals to the pipeline store
        emitted = emit_signals(signals, store)

        # Build summary for DAG output
        signal_summaries = [
            {
                "signal_name": s.signal_name,
                "event_type": s.event_type,
                "value": round(s.value, 4),
                "categories": s.categories_involved,
            }
            for s in signals
        ]

        log.info(
            "Convergence detection complete: %d detected, %d emitted.",
            len(results),
            emitted,
        )

        return {
            "detected": len(results),
            "emitted": emitted,
            "signals": signal_summaries,
        }

    finally:
        store.close()


# ═══════════════════════════════════════════════════════════════
#  DAG Builder
# ═══════════════════════════════════════════════════════════════


def build_convergence_detection_dag(
    db_path: str = ".tirra_pipeline/pipeline.db",
) -> DAG:
    """Build the convergence_detection DAG.

    Single node: ``run_detection`` (FunctionOperator).
    Schedule: weekdays at 18:30 UTC, 30 min after daily_collection.
    """
    dag = DAG(
        name="convergence_detection",
        schedule="30 18 * * 1-5",
        description="Multi-source convergence detection: find correlated anomalies across 60 data pipes",
    )

    dag.add(
        "run_detection",
        operator=run_convergence_detection,
        params={"db_path": db_path},
        timeout=300,
        retries=1,
        store_result=True,
    )

    return dag
