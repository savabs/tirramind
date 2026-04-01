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
"""

from __future__ import annotations

from agent.pipeline.dag import DAG


def build_daily_collection_dag() -> DAG:
    """Build the daily_collection DAG. Pure data declaration, no side effects."""
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

    return dag
