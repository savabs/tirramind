#!/usr/bin/env python3
"""Gate C — Layer 2/3 graph audit before GNN training.

Builds HeteroData from pipeline.db (no enrichment) and checks:
  - instrument node feature dim = 34 (BASE 14 + PRICE 9 + M9 11)
  - micro block L1 norm > 0 for commodity futures with enough daily bars
  - reference_time sane (not corrupt far-future timestamps)
  - no NaN/Inf in node feature tensors

Usage:
    python scripts/gate_c_graph_audit.py
    python scripts/gate_c_graph_audit.py --db-path .tirra_pipeline/pipeline.db --json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from agent.models.gnn.graph_builder import (
    BASE_FEAT_DIM,
    M15_QUANT_DIM,
    MICROSTRUCTURE_DIM,
    PRICE_FEAT_DIM,
    GraphBuilder,
    _reference_time,
)
from agent.pipeline.store import PipelineStore
from agent.quant.microstructure_signals import list_instruments_by_asset_class

DEFAULT_DB = Path(".tirra_pipeline/pipeline.db")
EXPECTED_INSTRUMENT_DIM = (
    BASE_FEAT_DIM + PRICE_FEAT_DIM + MICROSTRUCTURE_DIM + M15_QUANT_DIM
)
MIN_MICRO_L1 = 1e-6
# Same ceiling as graph_builder _reference_time slack
MAX_SANE_TS = time.time() + 86400.0


def _ts_fmt(ts: float) -> str:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def run_audit(db_path: Path, *, asset_class: str = "commodity_future") -> dict:
    store = PipelineStore(str(db_path))
    try:
        entities = store.query_all_entities()
        observations = store.query_all_observations()
        ref_time = _reference_time(observations)

        builder = GraphBuilder(store)
        data, id_map, events = builder.build()

        issues: list[str] = []
        out: dict = {"db_path": str(db_path), "checks": {}, "issues": issues, "pass": True}

        # reference_time sanity
        ref_ok = 0.0 < ref_time <= MAX_SANE_TS
        out["checks"]["reference_time"] = {
            "value": ref_time,
            "formatted": _ts_fmt(ref_time) if ref_time > 0 else "—",
            "pass": ref_ok,
        }
        if not ref_ok:
            issues.append(f"reference_time out of sane range: {ref_time}")
            out["pass"] = False

        if "instrument" not in data.node_types:
            issues.append("No instrument nodes in graph")
            out["pass"] = False
            out["checks"]["instrument_dim"] = {"pass": False, "expected": EXPECTED_INSTRUMENT_DIM}
            return out

        x = data["instrument"].x
        dim = int(x.shape[1])
        dim_ok = dim == EXPECTED_INSTRUMENT_DIM
        out["checks"]["instrument_dim"] = {
            "actual": dim,
            "expected": EXPECTED_INSTRUMENT_DIM,
            "base": BASE_FEAT_DIM,
            "price": PRICE_FEAT_DIM,
            "micro": MICROSTRUCTURE_DIM,
            "m15_quant": M15_QUANT_DIM,
            "pass": dim_ok,
        }
        if not dim_ok:
            issues.append(f"instrument dim {dim} != {EXPECTED_INSTRUMENT_DIM}")
            out["pass"] = False

        finite_ok = bool(torch.isfinite(x).all().item())
        out["checks"]["finite_features"] = {"pass": finite_ok}
        if not finite_ok:
            nan_n = int((~torch.isfinite(x)).sum().item())
            issues.append(f"Non-finite values in instrument features: {nan_n} cells")
            out["pass"] = False

        micro_offset = BASE_FEAT_DIM + PRICE_FEAT_DIM
        micro_block = x[:, micro_offset : micro_offset + MICROSTRUCTURE_DIM]
        m15_offset = micro_offset + MICROSTRUCTURE_DIM
        m15_block = x[:, m15_offset : m15_offset + M15_QUANT_DIM]

        commodity_ids = {
            eid for eid, _ in list_instruments_by_asset_class(entities, asset_class)
        }
        node_ids = data["instrument"].node_ids
        micro_rows: list[dict] = []
        zero_micro: list[str] = []

        for local_idx, eid in enumerate(node_ids):
            l1 = float(micro_block[local_idx].abs().sum().item())
            ticker = eid
            for ent in entities:
                if ent.get("entity_id") == eid:
                    meta = ent.get("metadata") or {}
                    ticker = str(meta.get("ticker") or ent.get("canonical_name", eid))
                    break
            if eid in commodity_ids:
                m15_l1 = float(m15_block[local_idx].abs().sum().item())
                micro_rows.append(
                    {
                        "ticker": ticker,
                        "entity_id": eid,
                        "micro_l1": l1,
                        "m15_l1": m15_l1,
                    }
                )
                if l1 < MIN_MICRO_L1:
                    zero_micro.append(ticker)

        commodities_with_micro = sum(1 for r in micro_rows if r["micro_l1"] >= MIN_MICRO_L1)
        micro_ok = len(zero_micro) == 0 and commodities_with_micro > 0
        out["checks"]["commodity_micro"] = {
            "commodity_count": len(micro_rows),
            "with_nonzero_micro": commodities_with_micro,
            "zero_micro_tickers": zero_micro,
            "min_l1": MIN_MICRO_L1,
            "pass": micro_ok,
        }
        if not micro_ok:
            issues.append(f"Commodities with zero micro block: {zero_micro}")
            out["pass"] = False

        out["checks"]["graph_scale"] = {
            "instrument_nodes": int(x.shape[0]),
            "events": len(events),
            "edge_types": len(data.edge_types),
            "pass": True,
        }
        out["commodity_micro_sample"] = sorted(
            micro_rows, key=lambda r: r["micro_l1"], reverse=True
        )[:10]
        return out
    finally:
        store.close()


def _print_report(result: dict) -> None:
    print("=" * 72)
    print("GATE C — Graph audit (instrument features + micro block)")
    print("=" * 72)
    print(f"DB: {result['db_path']}")
    print(f"VERDICT: {'PASS' if result['pass'] else 'FAIL'}")
    for name, chk in result["checks"].items():
        status = chk.get("pass", "—")
        detail = {k: v for k, v in chk.items() if k != "pass"}
        print(f"  {name}: {status}  {detail}")
    if result["issues"]:
        print("\nIssues:")
        for iss in result["issues"]:
            print(f"  - {iss}")
    print("\nTop commodity micro L1 norms:")
    for r in result.get("commodity_micro_sample", []):
        print(f"  {r['ticker']:<8} l1={r['micro_l1']:.6f}")
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate C graph audit")
    parser.add_argument("--db-path", default=str(DEFAULT_DB))
    parser.add_argument("--asset-class", default="commodity_future")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    db = Path(args.db_path)
    if not db.exists():
        print(f"DB not found: {db}", file=sys.stderr)
        return 2
    result = run_audit(db, asset_class=args.asset_class)
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
