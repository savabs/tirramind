#!/usr/bin/env python3
"""Gate A — Layer 1 data audit before GNN training (N1 commodity focus).

Checks pipeline.db coverage for the minimum sensor set:
  - instrument_daily density on commodity futures
  - produced_in producer links
  - CFTC positioning
  - GDELT geopolitical events
  - ghost-path tool sparsity (informational)
  - timestamp sanity (bad future dates)

Usage:
    python scripts/gate_a_audit.py
    python scripts/gate_a_audit.py --db-path .tirra_pipeline/pipeline.db --json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB = Path(".tirra_pipeline/pipeline.db")
MIN_DAILY_BARS = 30
GHOST_TOOLS = (
    "ais_vessel",
    "energy_supply",
    "weather_alerts",
    "satellite_activity",
    "capital_flows",
    "central_bank_balance",
)
# Unix for 2030-01-01 — flag anything beyond as corrupt clock
MAX_SANE_TS = 189_345_600_0.0


def _ts_fmt(ts: float | None) -> str:
    if ts is None:
        return "—"
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d")


def run_audit(db_path: Path) -> dict:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    out: dict = {"db_path": str(db_path), "checks": {}, "issues": [], "pass": True}

    # Commodity instruments + daily bars
    rows = con.execute(
        """
        SELECT e.entity_id, e.canonical_name, e.metadata_json,
               COUNT(eo.id) AS n_daily,
               MAX(eo.observed_at) AS t_max
        FROM entities e
        LEFT JOIN entity_observations eo ON eo.entity_id = e.entity_id
            AND eo.observation_type = 'instrument_daily'
        WHERE e.entity_type = 'instrument'
          AND (e.metadata_json LIKE '%commodity_future%' OR e.canonical_name LIKE '%=F')
        GROUP BY e.entity_id
        ORDER BY n_daily DESC
        """
    ).fetchall()
    commodities = []
    low_bars = []
    for r in rows:
        meta = {}
        if r["metadata_json"]:
            try:
                meta = json.loads(r["metadata_json"])
            except json.JSONDecodeError:
                pass
        ticker = meta.get("ticker") or r["canonical_name"]
        n = int(r["n_daily"] or 0)
        commodities.append({"ticker": ticker, "n_daily": n, "latest": _ts_fmt(r["t_max"])})
        if n < MIN_DAILY_BARS:
            low_bars.append(ticker)

    out["checks"]["commodity_instruments"] = {
        "count": len(commodities),
        "min_daily_bars": MIN_DAILY_BARS,
        "below_min": low_bars,
        "pass": len(low_bars) == 0 and len(commodities) > 0,
    }
    if low_bars:
        out["issues"].append(f"Commodities with <{MIN_DAILY_BARS} daily bars: {low_bars}")
        out["pass"] = False

    prod_links = con.execute(
        "SELECT COUNT(*) FROM entity_links WHERE link_type='produced_in'"
    ).fetchone()[0]
    prod_inst = con.execute(
        "SELECT COUNT(DISTINCT entity_id_a) FROM entity_links WHERE link_type='produced_in'"
    ).fetchone()[0]
    out["checks"]["produced_in"] = {
        "links": prod_links,
        "instruments": prod_inst,
        "pass": prod_links > 0 and prod_inst >= len(commodities) - 1,
    }
    if prod_inst < len(commodities) - 1:
        out["issues"].append(
            f"produced_in covers {prod_inst}/{len(commodities)} commodities only"
        )

    cftc_raw = con.execute(
        "SELECT COUNT(*) FROM entity_observations WHERE observation_type='futures_positioning'"
    ).fetchone()[0]
    cftc_der = con.execute(
        "SELECT COUNT(*) FROM entity_observations "
        "WHERE observation_type='futures_positioning_derived'"
    ).fetchone()[0]
    out["checks"]["cftc"] = {
        "raw_obs": cftc_raw,
        "derived_obs": cftc_der,
        "pass": cftc_raw > 0 and cftc_der > 0,
    }
    if cftc_raw == 0:
        out["issues"].append("No CFTC futures_positioning observations")
        out["pass"] = False

    geo = con.execute(
        "SELECT COUNT(*), MIN(observed_at), MAX(observed_at) "
        "FROM entity_observations WHERE observation_type='geopolitical_event'"
    ).fetchone()
    out["checks"]["gdelt"] = {
        "count": geo[0],
        "earliest": _ts_fmt(geo[1]),
        "latest": _ts_fmt(geo[2]),
        "pass": geo[0] > 1000,
    }

    max_ts = con.execute("SELECT MAX(observed_at) FROM entity_observations").fetchone()[0]
    latest_daily = con.execute(
        "SELECT MAX(observed_at) FROM entity_observations "
        "WHERE observation_type='instrument_daily'"
    ).fetchone()[0]
    out["checks"]["freshness"] = {
        "latest_any_obs": _ts_fmt(max_ts),
        "latest_instrument_daily": _ts_fmt(latest_daily),
        "note": "Compare to today; stale DB is OK for backtest if documented",
    }

    bad_ts = con.execute(
        "SELECT observation_type, source_tool, COUNT(*) n "
        "FROM entity_observations WHERE observed_at > ? "
        "GROUP BY observation_type, source_tool",
        (MAX_SANE_TS,),
    ).fetchall()
    bad_list = [{"type": r[0], "source": r[1], "count": r[2]} for r in bad_ts]
    out["checks"]["timestamp_sanity"] = {
        "corrupt_future_rows": bad_list,
        "pass": len(bad_list) == 0,
    }
    if bad_list:
        out["issues"].append(f"Corrupt future timestamps: {bad_list}")
        # Informational for commodity train — does not fail gate if only gov_contracts

    ghost = {}
    for tool in GHOST_TOOLS:
        n = con.execute(
            "SELECT COUNT(*) FROM entity_observations WHERE source_tool=?", (tool,)
        ).fetchone()[0]
        ghost[tool] = n
    out["checks"]["ghost_path_tools"] = ghost
    sparse_ghost = [t for t, n in ghost.items() if n < 10]
    if sparse_ghost:
        out["issues"].append(f"Sparse ghost-path tools (optional): {sparse_ghost}")

    out["commodities_sample"] = commodities[:10]
    con.close()
    return out


def _print_report(result: dict) -> None:
    print("=" * 72)
    print("GATE A — Layer 1 data audit (N1 commodity)")
    print("=" * 72)
    print(f"DB: {result['db_path']}")
    print(f"VERDICT: {'PASS' if result['pass'] else 'FAIL'}")
    print()
    for name, chk in result["checks"].items():
        if name == "ghost_path_tools":
            print("Ghost-path tools (informational):")
            for t, n in chk.items():
                flag = " SPARSE" if n < 10 else ""
                print(f"  {t:25} {n:6}{flag}")
            continue
        status = chk.get("pass", "—")
        print(f"  {name}: {status}  { {k: v for k, v in chk.items() if k != 'pass'} }")
    if result["issues"]:
        print("\nIssues:")
        for iss in result["issues"]:
            print(f"  - {iss}")
    print("\nTop commodities by daily bars:")
    for c in result.get("commodities_sample", []):
        print(f"  {c['ticker']:<8} days={c['n_daily']:<4} latest={c['latest']}")
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate A Layer 1 audit")
    parser.add_argument("--db-path", default=str(DEFAULT_DB))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    db = Path(args.db_path)
    if not db.exists():
        print(f"DB not found: {db}", file=sys.stderr)
        return 2
    result = run_audit(db)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_report(result)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
