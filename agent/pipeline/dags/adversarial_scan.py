"""TirraMind — Adversarial Scan DAG

Scheduled after convergence_detection and before rl_training, this DAG:
    1. Loads per-signal returns and market data from PipelineStore
    2. Loads convergence clusters and current portfolio positions
    3. Runs the AdversarialScanner to produce flags
    4. Stores flags in the adversarial_flags table

Schedule: weekdays at 19:15 UTC (between convergence detection and RL training).

**Step 1 and step 2 do not exist.** The scanner has always been called with
hardcoded empty literals — ``signal_returns={}``, ``market_returns=np.array([])``,
``clusters=[]`` — carrying a ``# populated from PipelineStore in production``
comment that was never honoured. Layer 6 is therefore structurally incapable of
producing a flag: ``adversarial_flags`` has held zero rows across six
``completed`` runs. See ``docs/publications/thirteen_ways_a_pipeline_lies.md``
Way 9 ("placeholders that outlived the promise to fill them") and the
2026-09-23 audit, item P1.8.

Until the loaders exist this node **fails loudly** rather than reporting a
successful scan of nothing. What is missing, precisely:

``signal_returns``
    Needs a per-signal return series. ``PipelineStore`` can query signals by
    name (``query_signals``) but exposes no way to *enumerate* signal names,
    so the edge-decay detector cannot be fed through the public API at all.
``market_returns`` / ``market_volumes``
    Need an aggregate market series from ``instrument_daily`` — which has no
    scheduled producer (audit P6.1), so the series would be stale even once
    wired.
``clusters``
    ``query_convergence_clusters`` returns rows carrying only
    ``member_entity_ids``; ``ConvergenceCluster`` requires the ``EntityAlert``
    objects themselves, so reconstructing one means re-reading
    ``entity_alerts`` per member.
``position_weights`` / ``volume_history``
    ``query_portfolio_weights`` exists, but ``portfolio_weights`` is empty
    because of the RL cold-start deadlock (audit C4).

None of those are decisions this DAG can make on its own, and a partial
wiring that still cannot reach a flag would reproduce the original defect with
more code. The check below is deliberately on the *outcome*: a scan that
produced nothing from inputs that contained nothing raises.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from agent.adversarial.scanner import AdversarialScanner
from agent.pipeline.dag import DAG
from agent.pipeline.operators import DOMAIN_TABLES_PARAM_KEY
from agent.pipeline.store import PipelineStore

log = logging.getLogger(__name__)

DAG_NAME = "adversarial_scan"
DEPENDS_ON = ["convergence_detection"]

#: The table this DAG exists to populate. Declared to the executor so the
#: zero-rows guard measures *this*, not the generic result envelope.
DOMAIN_TABLE = "adversarial_flags"

#: Scanner inputs with no loader yet, in the order they appear in
#: ``AdversarialScanner.scan``. Kept as data rather than prose so the error
#: message and the module docstring cannot drift apart.
UNWIRED_INPUTS: tuple[str, ...] = (
    "signal_returns",
    "market_returns",
    "market_volumes",
    "clusters",
    "position_weights",
    "volume_history",
)


def _input_sizes(scan_inputs: dict) -> dict[str, int]:
    """Per-input element counts, for logging and for the payload."""
    return {name: len(value) for name, value in scan_inputs.items()}


def run_adversarial_scan(params: dict, upstream: dict) -> dict:
    """FunctionOperator callback for the adversarial_scan DAG step.

    Parameters (from ``params``):
        db_path : str

    Returns dict with scan summary for downstream DAGs.

    Raises:
        RuntimeError: when the scan produced no flags *and* every scanner
            input was empty — i.e. the run was structurally incapable of
            producing anything. Reporting success for that is the defect
            this DAG is named in the audit for.
    """
    db_path = params.get("db_path", ".tirra_pipeline/pipeline.db")
    store = PipelineStore(db_path)
    try:
        scanner = AdversarialScanner()

        # Every one of these is empty because no loader exists yet; see the
        # module docstring for what each would need. They are named
        # explicitly (rather than inlined at the call site) so the guard
        # below reasons about the same objects the scanner receives.
        scan_inputs: dict = {
            "signal_returns": {},
            "market_returns": np.array([]),
            "market_volumes": np.array([]),
            "clusters": [],
            "position_weights": {},
            "volume_history": {},
        }
        sizes = _input_sizes(scan_inputs)
        log.info("Adversarial scan inputs: %s", sizes)

        flags = scanner.scan(timestamp=time.time(), **scan_inputs)

        if not flags and not any(sizes.values()):
            # Loud, not silent: no bare except, no empty-list return, no
            # "0 flags" success payload. The node fails and the executor
            # records a failed run, which is the honest state of layer 6.
            raise RuntimeError(
                f"{DAG_NAME} produced no flags because every scanner input was empty: "
                f"{sizes}. Inputs with no loader: {list(UNWIRED_INPUTS)}. Layer 6 "
                "cannot produce a flag until these are wired to PipelineStore (audit "
                "P1.8 / C4). Failing rather than recording another 'completed' run "
                f"over an empty {DOMAIN_TABLE}."
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

        if flags and flags_stored == 0:
            raise RuntimeError(
                f"{DAG_NAME} produced {len(flags)} flag(s) but stored none of them — "
                f"refusing to report a successful scan with an empty {DOMAIN_TABLE}."
            )

        log.info("Adversarial scan complete: %d flags produced, %d stored", len(flags), flags_stored)
        return {
            "n_flags": len(flags),
            "flags_stored": flags_stored,
            "flag_types": [f.flag_type for f in flags],
            # Carried in the payload so a genuine "scanned real inputs, found
            # nothing" run is distinguishable from "scanned nothing" by
            # anyone reading the stored envelope, not just the logs.
            "input_sizes": sizes,
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
        params={
            "db_path": db_path,
            # Measured by the executor's zero-rows guard as a real COUNT(*)
            # delta. Without this the guard counted the pipeline_data
            # envelope, which this node writes on every "success" — the
            # reason six completed runs sat on top of an empty table.
            DOMAIN_TABLES_PARAM_KEY: [DOMAIN_TABLE],
        },
    )

    return dag
