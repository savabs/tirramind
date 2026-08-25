#!/usr/bin/env python3
"""MP-1 data health report — observation counts for ghost chain sensors.

Verifies AIS + EIA collectors have sufficient density for chain matching.

Usage:
    python scripts/mp1_data_health.py
    python scripts/mp1_data_health.py --json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DB_PATH = Path(".tirra_pipeline/pipeline.db")

# Minimum obs thresholds for Phase A chain matching (advisory)
_THRESHOLDS = {
    "cftc": 500,
    "energy_supply": 50,
    "ais_vessel": 100,
    "gdelt": 1000,
    "instrument_universe": 1000,
}

_MP1_INSTRUMENTS = ("CL=F", "BZ=F", "NG=F")


def _count(con: sqlite3.Connection, source_tool: str, obs_type: str | None = None) -> int:
    if obs_type:
        return con.execute(
            "SELECT COUNT(*) FROM entity_observations "
            "WHERE source_tool=? AND observation_type=?",
            (source_tool, obs_type),
        ).fetchone()[0]
    return con.execute(
        "SELECT COUNT(*) FROM entity_observations WHERE source_tool=?",
        (source_tool,),
    ).fetchone()[0]


def _latest(con: sqlite3.Connection, source_tool: str) -> str | None:
    row = con.execute(
        "SELECT MAX(observed_at) FROM entity_observations WHERE source_tool=?",
        (source_tool,),
    ).fetchone()
    if not row or not row[0]:
        return None
    raw = row[0]
    s = str(raw).strip()
    if s.replace(".", "", 1).isdigit():
        from datetime import datetime, timezone

        return datetime.fromtimestamp(float(s), tz=timezone.utc).isoformat()
    return s


def _cftc_wti_obs(con: sqlite3.Connection) -> int:
    from agent.quant.ghost_chains import _cftc_contract_for_ticker

    eid = _cftc_contract_for_ticker(con, "CL=F")
    if not eid:
        return 0
    return con.execute(
        "SELECT COUNT(*) FROM entity_observations WHERE entity_id=? AND source_tool='cftc'",
        (eid,),
    ).fetchone()[0]


def build_report(db_path: Path) -> dict:
    con = sqlite3.connect(str(db_path))
    sensors = {}
    for tool, min_obs in _THRESHOLDS.items():
        count = _count(con, tool)
        sensors[tool] = {
            "obs_count": count,
            "min_recommended": min_obs,
            "status": "ok" if count >= min_obs else "low",
            "latest_observed_at": _latest(con, tool),
        }

    sensors["cftc"]["wti_cl_obs"] = _cftc_wti_obs(con)

    def _ais_daily_days(obs_type: str) -> int:
        rows = con.execute(
            """
            SELECT DISTINCT substr(
                CASE
                    WHEN typeof(observed_at)='real' THEN datetime(observed_at, 'unixepoch')
                    ELSE observed_at
                END, 1, 10
            )
            FROM entity_observations
            WHERE source_tool='ais_vessel' AND observation_type=?
            """,
            (obs_type,),
        ).fetchall()
        return len(rows)

    sensors["ais_vessel"]["live_daily_days"] = _ais_daily_days("area_daily_activity")
    sensors["ais_vessel"]["proxy_daily_days"] = _ais_daily_days("baltic_activity_proxy")

    dag_nodes = []
    try:
        rows = con.execute(
            "SELECT name, last_status FROM dag_runs "
            "WHERE name IN ('fetch_ais_vessel', 'fetch_energy_supply', 'fetch_cftc') "
            "ORDER BY started_at DESC LIMIT 10"
        ).fetchall()
        seen: set[str] = set()
        for name, status in rows:
            if name not in seen:
                dag_nodes.append({"node": name, "last_status": status})
                seen.add(name)
    except sqlite3.OperationalError:
        dag_nodes = [{"note": "dag_runs table unavailable"}]

    con.close()

    blockers = [
        f"{k}: {v['obs_count']} obs (need ≥{v['min_recommended']})"
        for k, v in sensors.items()
        if v["status"] == "low"
    ]

    return {
        "micro_playground": "MP-1",
        "instruments": list(_MP1_INSTRUMENTS),
        "sensors": sensors,
        "dag_recent": dag_nodes,
        "ready_for_chain_scan": len(blockers) == 0,
        "blockers": blockers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MP-1 sensor data health")
    parser.add_argument("--db-path", default=str(DB_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"ERROR: DB not found at {db_path}")
        sys.exit(1)

    report = build_report(db_path)

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print("MP-1 Data Health Report")
    print("=" * 50)
    for tool, info in report["sensors"].items():
        flag = "OK" if info["status"] == "ok" else "LOW"
        extra = ""
        if tool == "cftc" and "wti_cl_obs" in info:
            extra = f", WTI obs={info['wti_cl_obs']}"
        if tool == "ais_vessel":
            extra += (
                f", live_days={info.get('live_daily_days', 0)}"
                f", proxy_days={info.get('proxy_daily_days', 0)}"
            )
        print(
            f"  [{flag}] {tool}: {info['obs_count']} obs{extra} "
            f"(latest {info['latest_observed_at']})"
        )

    if report["dag_recent"]:
        print("\nRecent DAG nodes:")
        for d in report["dag_recent"]:
            if "node" in d:
                print(f"  {d['node']}: {d['last_status']}")

    if report["blockers"]:
        print("\nBlockers for reliable chain scan:")
        for b in report["blockers"]:
            print(f"  — {b}")
        print("\nRun: python scripts/backfill_cftc.py")
        print("     daily_collection DAG (fetch_ais_vessel, fetch_energy_supply)")
    else:
        print("\nSensor density OK for chain scan.")


if __name__ == "__main__":
    main()
