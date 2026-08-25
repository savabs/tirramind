#!/usr/bin/env python3
"""Append daily micro snapshots to a CSV history file (no GNN).

Usage:
    python scripts/export_micro_history.py
    python scripts/export_micro_history.py --history .tirra_pipeline/micro_history.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.pipeline.store import PipelineStore
from agent.quant.microstructure_signals import (
    build_instrument_panel,
    list_instruments_by_asset_class,
    load_cftc_ranks_by_ticker,
)

CSV_FIELDS = [
    "exported_at",
    "ticker",
    "entity_id",
    "as_of_ts",
    "n_days",
    "spread_roll",
    "spread_cs_proxy",
    "signed_flow_z",
    "kyle_lambda",
    "vol_20d",
    "vol_of_vol",
    "cftc_mm_pct_52w_rank",
    "cftc_positioning_label",
    "alert_count",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default=".tirra_pipeline/pipeline.db")
    parser.add_argument("--asset-class", default="commodity_future")
    parser.add_argument(
        "--history",
        default=".tirra_pipeline/micro_history.csv",
    )
    args = parser.parse_args()

    history_path = Path(args.history)
    write_header = not history_path.exists()

    store = PipelineStore(args.db_path)
    try:
        entities = store.query_all_entities()
        observations = store.query_all_observations()
        cftc_ranks = load_cftc_ranks_by_ticker(observations, entities)
        instruments = list_instruments_by_asset_class(entities, args.asset_class)
        exported_at = datetime.now(timezone.utc).isoformat()

        rows = []
        for eid, ticker in instruments:
            panel = build_instrument_panel(
                ticker,
                eid,
                observations,
                cftc_rank=cftc_ranks.get(ticker),
            )
            if panel is None:
                continue
            s = panel.snapshot
            rows.append(
                {
                    "exported_at": exported_at,
                    "ticker": ticker,
                    "entity_id": eid,
                    "as_of_ts": s.as_of_ts,
                    "n_days": s.n_days,
                    "spread_roll": s.spread_roll,
                    "spread_cs_proxy": s.spread_cs_proxy,
                    "signed_flow_z": s.signed_flow_z,
                    "kyle_lambda": s.kyle_lambda,
                    "vol_20d": s.vol_20d,
                    "vol_of_vol": s.vol_of_vol,
                    "cftc_mm_pct_52w_rank": panel.cftc_mm_pct_52w_rank,
                    "cftc_positioning_label": panel.cftc_positioning_label,
                    "alert_count": len(panel.alerts),
                }
            )

        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

        print(f"Appended {len(rows)} rows to {history_path}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
