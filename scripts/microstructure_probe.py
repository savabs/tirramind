#!/usr/bin/env python3
"""Print N1 commodity microstructure snapshots from pipeline.db (no GNN).

Daily proxies from instrument_daily (close, volume, log_return).
Tick OFI/VPIN are not available until bar/tick ingest exists.

Usage:
    python scripts/microstructure_probe.py
    python scripts/microstructure_probe.py --db-path .tirra_pipeline/pipeline.db
    python scripts/microstructure_probe.py --asset-class commodity_future --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.pipeline.store import PipelineStore
from agent.quant.microstructure_signals import (
    compute_micro_snapshot,
    list_instruments_by_asset_class,
    rank_snapshots,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Microstructure probe (standalone)")
    parser.add_argument("--db-path", default=".tirra_pipeline/pipeline.db")
    parser.add_argument(
        "--asset-class",
        default="commodity_future",
        help="Filter instruments by metadata asset_class",
    )
    parser.add_argument("--min-days", type=int, default=30)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    db = Path(args.db_path)
    if not db.exists():
        print(f"Database not found: {db}", file=sys.stderr)
        return 1

    store = PipelineStore(str(db))
    try:
        entities = store.query_all_entities()
        observations = store.query_all_observations()

        instruments = list_instruments_by_asset_class(entities, args.asset_class)
        eid_to_ticker = {eid: ticker for eid, ticker in instruments}

        snapshots = []
        for eid in sorted(eid_to_ticker):
            snap = compute_micro_snapshot(
                eid,
                observations,
                min_days=args.min_days,
            )
            if snap is not None:
                snapshots.append(snap)

        snapshots = rank_snapshots(snapshots, key="signed_flow_z")

        if args.as_json:
            print(json.dumps([s.to_dict() for s in snapshots], indent=2))
            return 0

        print(f"Microstructure probe — {args.asset_class} ({len(snapshots)} instruments)")
        print("Source: instrument_daily | spread_cs = proxy H/L | flow = sign(ret)*vol")
        print()
        hdr = (
            f"{'entity':<22} {'days':>5} {'spr_roll':>9} {'spr_cs':>8} "
            f"{'flow_z':>8} {'kyle_l':>10} {'vol20':>8} {'vov':>8}"
        )
        print(hdr)
        print("-" * len(hdr))
        for s in snapshots:
            ticker = eid_to_ticker.get(s.entity_id, s.entity_id[:8])
            print(
                f"{ticker:<22} {s.n_days:>5} {s.spread_roll:>9.5f} {s.spread_cs_proxy:>8.5f} "
                f"{s.signed_flow_z:>8.2f} {s.kyle_lambda:>10.6f} {s.vol_20d:>8.4f} {s.vol_of_vol:>8.4f}"
            )
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
