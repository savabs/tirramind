#!/usr/bin/env python3
"""Playground niche competition — objective density + tool coverage per candidate.

Maps N1–N6 to entity_types and source_tools, aggregates observations from pipeline.db,
and prints a scorecard table. Optional JSON output for docs automation.

Usage:
    python scripts/niche_competition_scorecard.py
    python scripts/niche_competition_scorecard.py --db-path .tirra_pipeline/pipeline.db --json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

_DEFAULT_DB = ".tirra_pipeline/pipeline.db"

# Niche definitions: hero entity types + source tools (subset of global surface)
NICHE_FILTERS: dict[str, dict] = {
    "N1": {
        "name": "Commodity & energy macro",
        "entity_types": [
            "instrument",
            "cftc_contract",
            "country",
            "region",
            "organization",
        ],
        "source_tools": [
            "cftc",
            "instrument_universe",
            "weather_alerts",
            "ais_vessel",
            "food_security",
            "comtrade",
            "transport_throughput",
            "energy_supply",
            "satellite_activity",
        ],
        "instrument_asset_classes": ["commodity_future"],
    },
    "N2": {
        "name": "Volatility & options regime",
        "entity_types": ["instrument", "cftc_contract"],
        "source_tools": ["instrument_universe", "cftc"],
        "instrument_asset_classes": ["vol", "commodity_future", "equity_index", "fx"],
    },
    "N3": {
        "name": "Crypto on-chain + macro bridge",
        "entity_types": ["instrument", "wallet", "protocol", "company"],
        "source_tools": [
            "whale_alert",
            "instrument_universe",
            "gdelt",
            "polymarket",
        ],
        "instrument_asset_classes": ["crypto"],
    },
    "N4": {
        "name": "Geopolitical / sovereign stress",
        "entity_types": ["country", "organization", "instrument", "topic"],
        "source_tools": [
            "gdelt",
            "sovereign_debt",
            "sanctions_monitor",
            "political_risk",
            "capital_flows",
            "global_pmi",
            "central_bank_balance",
            "food_security",
            "disease_surveillance",
        ],
        "instrument_asset_classes": ["fx", "fixed_income", "commodity_future"],
    },
    "N5": {
        "name": "Cross-asset liquidity & rates",
        "entity_types": ["country", "instrument", "central_bank"],
        "source_tools": [
            "central_bank_balance",
            "global_pmi",
            "capital_flows",
            "instrument_universe",
            "sovereign_debt",
        ],
        "instrument_asset_classes": [
            "fixed_income",
            "fx",
            "equity_index",
            "commodity_future",
        ],
    },
    "N6": {
        "name": "Equity microstructure + positioning",
        "entity_types": ["instrument", "company", "person"],
        "source_tools": [
            "instrument_universe",
            "finra_short_volume",
            "insider_filings",
            "form144",
            "bankruptcy_court",
        ],
        "instrument_asset_classes": [
            "equity_etf",
            "sector_etf",
            "equity_index",
        ],
    },
}

# Density scoring thresholds (0–5) from plan
_MIN_OBS_EXCELLENT = 50_000
_MIN_OBS_GOOD = 10_000
_MIN_OBS_OK = 2_000
_MIN_SPAN_DAYS = 365
_MIN_TOOLS = 4


def _query_niche_stats(conn: sqlite3.Connection, niche_id: str) -> dict:
    spec = NICHE_FILTERS[niche_id]
    et_placeholders = ",".join("?" * len(spec["entity_types"]))
    st_placeholders = ",".join("?" * len(spec["source_tools"]))

    sql = f"""
        SELECT
            COUNT(DISTINCT e.entity_id) AS entity_count,
            COUNT(eo.id) AS obs_count,
            MIN(eo.observed_at) AS earliest_ts,
            MAX(eo.observed_at) AS latest_ts,
            COUNT(DISTINCT eo.source_tool) AS tools_hit
        FROM entity_observations eo
        JOIN entities e ON eo.entity_id = e.entity_id
        WHERE e.entity_type IN ({et_placeholders})
          AND eo.source_tool IN ({st_placeholders})
    """
    params = list(spec["entity_types"]) + list(spec["source_tools"])
    row = conn.execute(sql, params).fetchone()
    entity_count, obs_count, earliest, latest, tools_hit = row

    span_days = 0.0
    if earliest is not None and latest is not None and latest > earliest:
        span_days = (latest - earliest) / 86400.0

    obs_per_entity = round(obs_count / entity_count, 1) if entity_count else 0.0

    # Tools present in DB (any entity)
    tool_rows = conn.execute(
        """
        SELECT source_tool, COUNT(*) AS c
        FROM entity_observations
        WHERE source_tool IN ({})
        GROUP BY source_tool
        """.format(st_placeholders),
        list(spec["source_tools"]),
    ).fetchall()
    tools_detail = {r[0]: r[1] for r in tool_rows}

    return {
        "niche_id": niche_id,
        "name": spec["name"],
        "entity_count": entity_count,
        "obs_count": obs_count,
        "obs_per_entity": obs_per_entity,
        "span_days": round(span_days, 0),
        "tools_hit": tools_hit,
        "tools_expected": len(spec["source_tools"]),
        "tools_detail": tools_detail,
        "missing_tools": [t for t in spec["source_tools"] if t not in tools_detail],
    }


def _density_score(stats: dict) -> int:
    """Map objective stats to 0–5 density score."""
    obs = stats["obs_count"]
    span = stats["span_days"]
    tools = stats["tools_hit"]

    if obs >= _MIN_OBS_EXCELLENT and span >= _MIN_SPAN_DAYS and tools >= _MIN_TOOLS:
        return 5
    if obs >= _MIN_OBS_GOOD and span >= 180 and tools >= 3:
        return 4
    if obs >= _MIN_OBS_OK and span >= 90 and tools >= 2:
        return 3
    if obs >= 500 and tools >= 1:
        return 2
    if obs > 0:
        return 1
    return 0


def _integration_score(niche_id: str, stats: dict) -> int:
    """Heuristic integration score from tool diversity + checkpoint priors."""
    tools = stats["tools_hit"]
    missing = len(stats["missing_tools"])
    base = min(5, tools + 1)  # 0–5 from tool hit count
    if missing > len(NICHE_FILTERS[niche_id]["source_tools"]) // 2:
        base = max(0, base - 2)
    priors = {"N1": 4, "N2": 2, "N3": 3, "N4": 4, "N5": 3, "N6": 3}
    return min(5, max(base, priors.get(niche_id, 2) - (1 if missing > 3 else 0)))


def _agent_loop_score(niche_id: str) -> int:
    """Tie-breaker: regime/sparsity teaching value (from plan)."""
    return {"N1": 5, "N2": 4, "N3": 4, "N4": 5, "N5": 4, "N6": 3}.get(niche_id, 3)


def run_scorecard(db_path: str, emit_json: bool = False) -> int:
    path = Path(db_path)
    if not path.exists():
        print(f"ERROR: database not found: {path}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(str(path))
    try:
        results = []
        for niche_id in sorted(NICHE_FILTERS):
            stats = _query_niche_stats(conn, niche_id)
            stats["density_score"] = _density_score(stats)
            stats["integration_score"] = _integration_score(niche_id, stats)
            stats["agent_score"] = _agent_loop_score(niche_id)
            results.append(stats)
    finally:
        conn.close()

    if emit_json:
        print(json.dumps(results, indent=2))
        return 0

    width = 100
    print("=" * width)
    print(f"Niche Competition Scorecard — {db_path}")
    print("=" * width)
    hdr = (
        f"{'ID':<4} {'NICHE':<32} {'OBS':>10} {'ENT':>8} "
        f"{'O/E':>8} {'SPANd':>7} {'TOOLS':>7} {'DENS':>5} {'INT':>5} {'AGT':>5}"
    )
    print(hdr)
    print("-" * width)
    for s in results:
        print(
            f"{s['niche_id']:<4} {s['name']:<32} "
            f"{s['obs_count']:>10,} {s['entity_count']:>8,} "
            f"{s['obs_per_entity']:>8.1f} {s['span_days']:>7.0f} "
            f"{s['tools_hit']}/{s['tools_expected']:>4} "
            f"{s['density_score']:>5} {s['integration_score']:>5} {s['agent_score']:>5}"
        )
    print()
    print("DENS/INT/AGT = objective 0–5 scores (Money/Moat = founder input in session)")
    print()
    for s in results:
        if s["missing_tools"]:
            print(f"  {s['niche_id']} missing tools: {', '.join(s['missing_tools'][:8])}")
            if len(s["missing_tools"]) > 8:
                print(f"      ... +{len(s['missing_tools']) - 8} more")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Niche competition objective scorecard")
    parser.add_argument("--db-path", default=_DEFAULT_DB)
    parser.add_argument("--json", action="store_true", help="Emit JSON for doc automation")
    args = parser.parse_args()
    return run_scorecard(args.db_path, emit_json=args.json)


if __name__ == "__main__":
    sys.exit(main())
