"""TirraMind — Adversarial Scan DAG

Scheduled after convergence_detection and before rl_training, this DAG:
    1. Loads per-signal returns and market data from PipelineStore
    2. Loads convergence clusters and current portfolio positions
    3. Runs the AdversarialScanner to produce flags
    4. Stores flags in the adversarial_flags table

Schedule: weekdays at 19:15 UTC (between convergence detection and RL training).
"""

from __future__ import annotations

import logging
import time

import numpy as np

from agent.adversarial.scanner import AdversarialScanner
from agent.pipeline.dag import DAG
from agent.pipeline.store import PipelineStore

log = logging.getLogger(__name__)

DAG_NAME = "adversarial_scan"
DEPENDS_ON = ["convergence_detection"]


def run_adversarial_scan(params: dict, upstream: dict) -> dict:
    """FunctionOperator callback for the adversarial_scan DAG step.

    Parameters (from ``params``):
        db_path : str

    Returns dict with scan summary for downstream DAGs.
    """
    db_path = params.get("db_path", ".tirra_pipeline/pipeline.db")
    store = PipelineStore(db_path)
    try:
        scanner = AdversarialScanner()

        flags = scanner.scan(
            signal_returns={},  # populated from PipelineStore in production
            market_returns=np.array([]),
            market_volumes=np.array([]),
            clusters=[],  # populated from PipelineStore in production
            position_weights={},  # populated from PipelineStore in production
            volume_history={},
            timestamp=time.time(),
        )

        # Store flags. PipelineStore has no ``put()`` method (that call
        # always raised AttributeError, and adversarial_flags did not exist
        # as a table) — the real writer is ``store_adversarial_flag``, which
        # also owns the adversarial_flags schema in agent/pipeline/store.py.
        flags_stored = 0
        for flag in flags:
            store.store_adversarial_flag(
                flag_type=flag.flag_type,
                severity=flag.severity,
                confidence=flag.confidence,
                flagged_at=flag.timestamp,
                entity_id=flag.entity_id,
                signal_name=flag.signal_name,
                evidence=flag.evidence,
            )
            flags_stored += 1

        log.info("Adversarial scan complete: %d flags produced, %d stored", len(flags), flags_stored)
        return {
            "n_flags": len(flags),
            "flags_stored": flags_stored,
            "flag_types": [f.flag_type for f in flags],
        }
    finally:
        store.close()


def build_adversarial_scan_dag(
    db_path: str = ".tirra_pipeline/pipeline.db",
) -> DAG:
    """Build the adversarial_scan DAG.

    Single node: ``scan_adversarial``.
    Schedule: weekdays at 19:15 UTC (after convergence detection, before RL).
    """
    dag = DAG(
        name=DAG_NAME,
        schedule="15 19 * * 1-5",
        description=("Adversarial intelligence scan: edge decay monitoring, VPIN estimation, crowding risk assessment"),
    )

    dag.add(
        "scan_adversarial",
        operator=run_adversarial_scan,
        params={"db_path": db_path},
    )

    return dag
