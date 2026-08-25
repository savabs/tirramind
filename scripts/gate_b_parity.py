#!/usr/bin/env python3
"""Gate B — Layer 2 parity: standalone micro probe vs GNN M9 vector.

Compares compute_micro_snapshot (N1 probe path) with compute_gnn_micro_features
(graph_builder path) on the same causal cutoff (_reference_time).

Expected matches (tight tolerance):
  spread_roll, spread_cs_proxy ↔ GNN[0:2], kyle_lambda ↔ GNN[7]

Known intentional diff:
  signed_flow_z (probe) vs ofi_zscore (GNN[2]) when MicrostructureFeatureExtractor
  produces a non-zero OFI z-score from synthetic daily trades.

Usage:
    python scripts/gate_b_parity.py
    python scripts/gate_b_parity.py --db-path .tirra_pipeline/pipeline.db --json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.models.gnn.graph_builder import _reference_time
from agent.pipeline.store import PipelineStore
from agent.quant.microstructure_signals import (
    GNN_MICRO_DIM,
    compute_gnn_micro_features,
    compute_micro_snapshot,
    list_instruments_by_asset_class,
)

DEFAULT_DB = Path(".tirra_pipeline/pipeline.db")
TOL_TIGHT = 1e-4
TOL_KYLE = 5e-4


def _close(a: float, b: float, tol: float) -> bool:
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    return abs(a - b) <= tol or (abs(a) + abs(b) > 0 and abs(a - b) / max(abs(a), abs(b), 1e-12) <= tol)


def run_parity(
    db_path: Path,
    *,
    asset_class: str = "commodity_future",
    min_days: int = 30,
) -> dict:
    store = PipelineStore(str(db_path))
    try:
        entities = store.query_all_entities()
        observations = store.query_all_observations()
        current_time = _reference_time(observations)

        instruments = list_instruments_by_asset_class(entities, asset_class)
        rows: list[dict] = []
        failures: list[str] = []

        for eid, ticker in sorted(instruments, key=lambda x: x[1]):
            snap = compute_micro_snapshot(
                eid, observations, until_ts=current_time, min_days=min_days
            )
            gnn = compute_gnn_micro_features(
                eid, observations, current_time, min_days=min_days
            )
            if snap is None:
                rows.append(
                    {
                        "ticker": ticker,
                        "entity_id": eid,
                        "status": "SKIP",
                        "reason": f"<{min_days} daily bars at cutoff",
                    }
                )
                continue

            spread_cs_gnn = gnn[0]
            spread_roll_gnn = gnn[1]
            ofi_gnn = gnn[2]
            kyle_gnn = gnn[7]

            ok_roll = _close(snap.spread_roll, spread_roll_gnn, TOL_TIGHT)
            ok_cs = _close(snap.spread_cs_proxy, spread_cs_gnn, TOL_TIGHT)
            ok_kyle = _close(snap.kyle_lambda, kyle_gnn, TOL_KYLE)
            ofi_match = _close(snap.signed_flow_z, ofi_gnn, TOL_TIGHT)

            row = {
                "ticker": ticker,
                "entity_id": eid,
                "status": "PASS" if (ok_roll and ok_cs and ok_kyle) else "FAIL",
                "reference_time": current_time,
                "n_days": snap.n_days,
                "probe": {
                    "spread_roll": snap.spread_roll,
                    "spread_cs_proxy": snap.spread_cs_proxy,
                    "signed_flow_z": snap.signed_flow_z,
                    "kyle_lambda": snap.kyle_lambda,
                },
                "gnn": {
                    "spread_cs": spread_cs_gnn,
                    "spread_roll": spread_roll_gnn,
                    "ofi_zscore": ofi_gnn,
                    "kyle_lambda": kyle_gnn,
                    "l1_norm": sum(abs(x) for x in gnn),
                },
                "checks": {
                    "spread_roll": ok_roll,
                    "spread_cs": ok_cs,
                    "kyle_lambda": ok_kyle,
                    "ofi_z_intentional_diff": not ofi_match,
                },
            }
            rows.append(row)
            if row["status"] == "FAIL":
                bad = [k for k, v in row["checks"].items() if k != "ofi_z_intentional_diff" and not v]
                failures.append(f"{ticker}: failed {bad}")

        evaluated = [r for r in rows if r.get("status") in ("PASS", "FAIL")]
        passed = [r for r in evaluated if r["status"] == "PASS"]
        ofi_diffs = [
            r["ticker"]
            for r in passed
            if r.get("checks", {}).get("ofi_z_intentional_diff")
        ]

        out = {
            "db_path": str(db_path),
            "asset_class": asset_class,
            "reference_time": current_time,
            "gnn_micro_dim": GNN_MICRO_DIM,
            "instruments_total": len(instruments),
            "evaluated": len(evaluated),
            "passed": len(passed),
            "failed": len([r for r in evaluated if r["status"] == "FAIL"]),
            "skipped": len([r for r in rows if r.get("status") == "SKIP"]),
            "ofi_intentional_diff_tickers": ofi_diffs,
            "rows": rows,
            "issues": failures,
            "pass": len(failures) == 0 and len(passed) > 0,
        }
        return out
    finally:
        store.close()


def _print_report(result: dict) -> None:
    print("=" * 72)
    print("GATE B — Layer 2 parity (probe vs GNN M9 vector)")
    print("=" * 72)
    print(f"DB: {result['db_path']}")
    print(f"reference_time: {result['reference_time']:.0f}")
    print(f"VERDICT: {'PASS' if result['pass'] else 'FAIL'}")
    print(
        f"  evaluated={result['evaluated']} passed={result['passed']} "
        f"failed={result['failed']} skipped={result['skipped']}"
    )
    if result["ofi_intentional_diff_tickers"]:
        print(
            "  ofi_z intentional diffs (OK): "
            + ", ".join(result["ofi_intentional_diff_tickers"])
        )
    print()
    hdr = f"{'ticker':<8} {'status':<6} {'spr_roll':>6} {'spr_cs':>6} {'kyle':>6} {'ofi~':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in result["rows"]:
        if r.get("status") == "SKIP":
            print(f"{r['ticker']:<8} SKIP   {r.get('reason', '')}")
            continue
        chk = r["checks"]
        mark = lambda ok: "OK" if ok else "FAIL"
        print(
            f"{r['ticker']:<8} {r['status']:<6} "
            f"{mark(chk['spread_roll']):>6} {mark(chk['spread_cs']):>6} "
            f"{mark(chk['kyle_lambda']):>6} "
            f"{'diff' if chk.get('ofi_z_intentional_diff') else 'match':>6}"
        )
    if result["issues"]:
        print("\nFailures:")
        for iss in result["issues"]:
            print(f"  - {iss}")
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate B micro parity audit")
    parser.add_argument("--db-path", default=str(DEFAULT_DB))
    parser.add_argument("--asset-class", default="commodity_future")
    parser.add_argument("--min-days", type=int, default=30)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    db = Path(args.db_path)
    if not db.exists():
        print(f"DB not found: {db}", file=sys.stderr)
        return 2
    result = run_parity(db, asset_class=args.asset_class, min_days=args.min_days)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_report(result)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
