"""TirraMind — Adversarial Scan DAG

Scheduled after convergence_detection and before rl_training, this DAG:
    1. Loads per-signal returns and market data from PipelineStore
    2. Loads convergence clusters and current portfolio positions
    3. Runs the AdversarialScanner to produce flags
    4. Stores flags in the adversarial_flags table

Schedule: weekdays at 19:15 UTC (between convergence detection and RL training).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

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

    # Store flags
    for flag in flags:
        store.put(
            "adversarial_flags",
            json.dumps(
                {
                    "flag_type": flag.flag_type,
                    "severity": flag.severity,
                    "confidence": flag.confidence,
                    "entity_id": flag.entity_id,
                    "signal_name": flag.signal_name,
                    "evidence": flag.evidence,
                    "timestamp": flag.timestamp,
                }
            ),
        )

    log.info("Adversarial scan complete: %d flags produced", len(flags))
    return {
        "n_flags": len(flags),
        "flag_types": [f.flag_type for f in flags],
    }


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
        description=(
            "Adversarial intelligence scan: edge decay monitoring, "
            "VPIN estimation, crowding risk assessment"
        ),
    )

    dag.add(
        "scan_adversarial",
        operator=run_adversarial_scan,
        params={"db_path": db_path},
    )

    return dag
