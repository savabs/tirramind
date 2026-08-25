#!/usr/bin/env python3
"""N1 commodity micro + positioning alerts (standalone — no GNN).

Fuses daily micro proxies (instrument_daily) with CFTC 52w positioning rank when
available. Writes JSON report for probes and downstream consumers.

Usage:
    python scripts/microstructure_alerts.py
    python scripts/microstructure_alerts.py --output reports/micro_alerts.json
    python scripts/microstructure_alerts.py --alerts-only --min-severity STRONG
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.pipeline.store import PipelineStore
from agent.quant.microstructure_signals import (
    build_instrument_panel,
    list_instruments_by_asset_class,
    load_cftc_ranks_by_ticker,
    panels_with_alerts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="N1 microstructure alerts")
    parser.add_argument("--db-path", default=".tirra_pipeline/pipeline.db")
    parser.add_argument("--asset-class", default="commodity_future")
    parser.add_argument("--min-days", type=int, default=30)
    parser.add_argument(
        "--output",
        default=".tirra_pipeline/micro_alerts/latest.json",
        help="JSON output path (use - for stdout only)",
    )
    parser.add_argument(
        "--alerts-only",
        action="store_true",
        help="Only include instruments with at least one micro alert",
    )
    parser.add_argument(
        "--min-severity",
        choices=("WATCH", "STRONG"),
        default="WATCH",
        help="Minimum alert severity when --alerts-only",
    )
    parser.add_argument("--no-write", action="store_true", help="Skip writing output file")
    args = parser.parse_args()

    db = Path(args.db_path)
    if not db.exists():
        print(f"Database not found: {db}", file=sys.stderr)
        return 1

    store = PipelineStore(str(db))
    try:
        entities = store.query_all_entities()
        observations = store.query_all_observations()
        cftc_ranks = load_cftc_ranks_by_ticker(observations, entities)

        instruments = list_instruments_by_asset_class(entities, args.asset_class)
        panels = []
        for eid, ticker in sorted(instruments, key=lambda x: x[1]):
            panel = build_instrument_panel(
                ticker,
                eid,
                observations,
                cftc_rank=cftc_ranks.get(ticker),
                min_days=args.min_days,
            )
            if panel is not None:
                panels.append(panel)

        if args.alerts_only:
            panels = panels_with_alerts(panels, min_severity=args.min_severity)

        # Sort: STRONG alert count desc, then |flow_z|
        def _sort_key(p):
            strong = sum(1 for a in p.alerts if a.severity == "STRONG")
            return (strong, abs(p.snapshot.signed_flow_z))

        panels.sort(key=_sort_key, reverse=True)

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "asset_class": args.asset_class,
            "n_instruments": len(instruments),
            "n_panel": len(panels),
            "data_notes": {
                "micro": "daily proxies from instrument_daily (not tick OFI/VPIN)",
                "positioning": "futures_positioning_derived on cftc_contract when present",
            },
            "instruments": [p.to_dict() for p in panels],
        }

        text = json.dumps(report, indent=2)
        print(text)

        if not args.no_write and args.output != "-":
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text, encoding="utf-8")
            print(f"\nWrote {out_path}", file=sys.stderr)

        # Human summary to stderr
        alerted = sum(1 for p in panels if p.alerts)
        print(
            f"\nSummary: {alerted}/{len(panels)} instruments with micro alerts "
            f"({args.asset_class})",
            file=sys.stderr,
        )
        for p in panels[:8]:
            if not p.alerts:
                continue
            codes = ", ".join(f"{a.code}({a.severity})" for a in p.alerts)
            cot = (
                f" CFTC={p.cftc_positioning_label}"
                if p.cftc_positioning_label
                else ""
            )
            print(f"  {p.ticker}: {codes}{cot}", file=sys.stderr)

        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
