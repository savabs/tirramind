"""
TirraMind — Daily Collection DAG

Fetches data from all stateless surveillance tools once daily.
All nodes are independent → single parallel layer → maximum throughput.

Schedule: weekdays at 18:00 UTC (after US market close, CFTC/FINRA publish windows).

Nodes:
    fetch_cftc          — CFTC Commitments of Traders, latest report
    fetch_finra_scan    — FINRA Reg SHO short volume, all-ticker scan
    fetch_power_demand  — NYISO power grid actual demand by zone
    fetch_power_fuel    — NYISO generation by fuel type
    fetch_gdelt         — GDELT geopolitical events, last 24h
    fetch_polymarket    — Polymarket prediction market odds, all categories

Change 12: Optional ``tool_router`` parameter.  When provided, the bandit
decides which optional tools to enable before DAG execution.

Change 15 (Tier 8): Quarantine cycle for discovered tools.  Newly discovered
sources run in quarantine (shadowed observations) for a configurable number
of successful cycles before promotion to active.  Three consecutive failures
disable the source.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from agent.pipeline.dag import DAG

if TYPE_CHECKING:
    from agent.learning.tool_router import ToolRoutingBandit, ToolContext
    from agent.pipeline.store import PipelineStore

log = logging.getLogger(__name__)

# ── Quarantine constants ──────────────────────────────────
_QUARANTINE_CYCLES_TO_PROMOTE = 5
_MAX_CONSECUTIVE_FAILURES = 3


def run_quarantine_cycle(store: PipelineStore) -> dict[str, str]:
    """Execute one quarantine check cycle for discovered sources.

    For each source in 'quarantine' status:
    - Try loading its tool config and executing a probe
    - On success: decrement remaining quarantine cycles; promote to 'active'
      after *_QUARANTINE_CYCLES_TO_PROMOTE* consecutive successes
    - On failure: increment consecutive_failures; move to 'disabled' after
      *_MAX_CONSECUTIVE_FAILURES*

    Returns a mapping of ``{source_id: new_status}`` for each source that
    changed status, or ``"quarantine"`` if unchanged.
    """
    results: dict[str, str] = {}
    try:
        sources = store.query_discovered_sources(status="quarantine")
    except Exception:
        return results

    for src in sources:
        source_id = src["source_id"]
        try:
            from agent.discovery.tool_factory import ToolFactory

            factory = ToolFactory()
            config_path = factory._config_dir / f"discovered_{source_id[:8]}.json"
            if not config_path.exists():
                # No config on disk — can't probe, treat as failure
                raise FileNotFoundError(f"No config for {source_id}")

            tools = factory.load_all_configs()
            tool = next((t for t in tools if source_id[:8] in t.name), None)
            if tool is None:
                raise FileNotFoundError(f"Tool not found for {source_id}")

            # Execute a probe — if it doesn't raise, consider it a success
            tool.execute()

            # Success: reset failure counter
            store.reset_source_failures(source_id)

            # Track quarantine progress via metadata
            meta = src.get("metadata_json") or {}
            if isinstance(meta, str):
                import json

                meta = json.loads(meta)
            q_successes = meta.get("quarantine_successes", 0) + 1
            meta["quarantine_successes"] = q_successes

            if q_successes >= _QUARANTINE_CYCLES_TO_PROMOTE:
                store.update_source_status(source_id, "active")
                results[source_id] = "active"
                log.info("Promoted discovered source %s to active", source_id)
            else:
                results[source_id] = "quarantine"

        except Exception as exc:
            store.increment_source_failures(source_id)
            # Refresh to get updated failure count
            updated = store.query_discovered_sources(status="quarantine")
            updated_src = next((s for s in updated if s["source_id"] == source_id), src)
            if updated_src.get("consecutive_failures", 0) >= _MAX_CONSECUTIVE_FAILURES:
                store.update_source_status(source_id, "disabled")
                results[source_id] = "disabled"
                log.warning(
                    "Disabled discovered source %s after %d failures: %s",
                    source_id,
                    _MAX_CONSECUTIVE_FAILURES,
                    exc,
                )
            else:
                results[source_id] = "quarantine"
                log.debug("Quarantine failure for %s: %s", source_id, exc)

    return results


def build_daily_collection_dag(
    db_path: str = ".tirra_pipeline/pipeline.db",
    tool_router: ToolRoutingBandit | None = None,
    tool_context: ToolContext | None = None,
) -> DAG:
    """Build the daily_collection DAG. Pure data declaration, no side effects.

    Parameters
    ----------
    db_path : str
        Path to the pipeline SQLite database.
    tool_router : ToolRoutingBandit, optional
        If provided, the bandit decides which optional tools are enabled.
    tool_context : ToolContext, optional
        Context for the routing decision (regime, day, staleness).
    """
    dag = DAG(
        name="daily_collection",
        schedule="0 18 * * 1-5",
        description="Daily surveillance: fetch all stateless data sources after US market close",
    )

    # ── CFTC Commitments of Traders ────────────────────────
    dag.add(
        "fetch_cftc",
        operator="cftc",
        params={"mode": "latest"},
        timeout=120,
        retries=2,
    )

    # ── FINRA Short Volume (all-ticker scan) ───────────────
    dag.add(
        "fetch_finra_scan",
        operator="finra_short_volume",
        params={"mode": "short_volume"},  # no ticker → scan mode
        timeout=180,
        retries=2,
    )

    # ── NYISO Power Grid: Actual Demand ────────────────────
    dag.add(
        "fetch_power_demand",
        operator="power_grid",
        params={"mode": "demand"},
        timeout=60,
        retries=2,
    )

    # ── NYISO Power Grid: Fuel Mix ─────────────────────────
    dag.add(
        "fetch_power_fuel",
        operator="power_grid",
        params={"mode": "fuel_mix"},
        timeout=60,
        retries=2,
    )

    # ── GDELT Geopolitical Events (24h lookback) ───────────
    dag.add(
        "fetch_gdelt",
        operator="gdelt",
        params={"mode": "events", "hours_back": 24, "limit": 500},
        timeout=120,
        retries=2,
    )

    # ── Polymarket Prediction Markets ──────────────────────
    dag.add(
        "fetch_polymarket",
        operator="polymarket",
        params={"category": "all", "limit": 100},
        timeout=60,
        retries=2,
    )

    # ── Instrument Universe (daily prices) ─────────────────
    from agent.tools.instrument_universe import run_instrument_ingest

    dag.add(
        "fetch_instruments",
        operator=run_instrument_ingest,
        params={"db_path": db_path},
        timeout=300,
        retries=1,
    )

    # ── Change 12: Apply tool routing decisions ────────────
    if tool_router is not None:
        decisions = tool_router.decide(tool_context)
        for node_id, node in dag.nodes.items():
            if node_id in decisions:
                node.enabled = decisions[node_id]

    return dag
