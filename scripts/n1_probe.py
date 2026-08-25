#!/usr/bin/env python3
"""Combined N1 probe: micro + CFTC positioning + producer-country GDELT stress.

No GNN. Fuses three sensor layers for commodity futures (N1 playground).

Usage:
    python scripts/n1_probe.py
    python scripts/n1_probe.py --min-priority 3
    python scripts/n1_probe.py --output .tirra_pipeline/n1_probe/latest.json
    python scripts/n1_probe.py --write-default-thresholds
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.quant.n1_probe import (
    build_all_n1_probes,
    default_thresholds_dict,
    thresholds_from_dict,
)

DEFAULT_THRESHOLDS_PATH = Path(".tirra_pipeline/n1_probe_thresholds.json")
DEFAULT_OUTPUT = Path(".tirra_pipeline/n1_probe/latest.json")


def _render_table(probes: list, min_priority: int) -> None:
    shown = [p for p in probes if p.composite_priority >= min_priority]
    print(f"\nN1 COMBINED PROBE — {len(shown)}/{len(probes)} instruments "
          f"(priority ≥ {min_priority})")
    print("Layers: micro (daily) | CFTC (weekly) | supply GDELT (30d default)")
    print()
    hdr = (
        f"{'Ticker':<8} {'Pri':>3} {'Sector':<12} {'Position':<18} "
        f"{'Supply':<10} {'Flow_z':>7} {'Micro alerts'}"
    )
    print(hdr)
    print("-" * len(hdr))
    for p in shown:
        micro = p.micro
        alerts = ",".join(
            f"{a['code'][:4]}({a['severity'][:1]})" for a in p.micro_alerts
        ) or "—"
        pos = p.positioning.get("label") or "—"
        supply = p.supply.get("stress_level", "—")
        print(
            f"{p.ticker:<8} {p.composite_priority:>3} {p.sector:<12} {pos:<18} "
            f"{supply:<10} {micro.get('signed_flow_z', 0):>7.2f} {alerts}"
        )
        if p.composite_flags:
            print(f"         └ {p.chain_narrative}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="N1 combined intelligence probe")
    parser.add_argument("--db-path", default=".tirra_pipeline/pipeline.db")
    parser.add_argument("--asset-class", default="commodity_future")
    parser.add_argument("--gdelt-days", type=int, default=None)
    parser.add_argument("--min-priority", type=int, default=2)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS_PATH))
    parser.add_argument("--write-default-thresholds", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    if args.write_default_thresholds:
        path = Path(args.thresholds)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(default_thresholds_dict(), indent=2), encoding="utf-8")
        print(f"Wrote default thresholds to {path}")
        return 0

    db = Path(args.db_path)
    if not db.exists():
        print(f"Database not found: {db}", file=sys.stderr)
        return 1

    th_path = Path(args.thresholds)
    if th_path.exists():
        thresholds = thresholds_from_dict(
            json.loads(th_path.read_text(encoding="utf-8"))
        )
        gdelt_days = args.gdelt_days or thresholds.gdelt_lookback_days
    else:
        thresholds = None
        gdelt_days = args.gdelt_days or 30

    probes = build_all_n1_probes(
        str(db),
        asset_class=args.asset_class,
        gdelt_lookback_days=gdelt_days,
        thresholds=thresholds,
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "playground": "N1_commodity_futures",
        "asset_class": args.asset_class,
        "gdelt_lookback_days": gdelt_days,
        "thresholds_file": str(th_path) if th_path.exists() else "built-in defaults",
        "n_instruments": len(probes),
        "instruments": [p.to_dict() for p in probes],
    }

    _render_table(probes, args.min_priority)

    if not args.no_write:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
