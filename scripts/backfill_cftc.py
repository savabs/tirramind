#!/usr/bin/env python3
"""CFTC 3-Year Historical Backfill

Downloads 3 years of CFTC disaggregated futures reports (2022, 2023, 2024)
and persists all mapped contract positioning observations to the pipeline DB.

The `_filter_contracts` fix (always keep mapped contracts) means all 19
instruments in `cftc_code_to_ticker()` will be populated after this runs.

Expected result after completion:
  - cftc_contract entities: ~20 contracts
  - entity_observations with source='cftc': >10,000 weekly observations
  - entity_links with link_type='cftc_tracks': 19 (one per mapped instrument)

Usage:
    python scripts/backfill_cftc.py [--db-path PATH] [--years 2022,2023,2024]
    python scripts/backfill_cftc.py --dry-run  # check counts without writing
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DB_PATH = Path(".tirra_pipeline/pipeline.db")
DEFAULT_YEARS = [2022, 2023, 2024]


def get_counts(db_path: Path) -> dict:
    conn = sqlite3.connect(str(db_path))
    cftc_obs = conn.execute(
        "SELECT count(*) FROM entity_observations WHERE source_tool='cftc'"
    ).fetchone()[0]
    cftc_contracts = conn.execute(
        "SELECT count(*) FROM entities WHERE entity_type='cftc_contract'"
    ).fetchone()[0]
    cftc_tracks = conn.execute(
        "SELECT count(*) FROM entity_links WHERE link_type='cftc_tracks'"
    ).fetchone()[0]
    conn.close()
    return {
        "cftc_observations": cftc_obs,
        "cftc_contract_entities": cftc_contracts,
        "cftc_tracks_links": cftc_tracks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill 3yr CFTC history")
    parser.add_argument("--db-path", default=str(DB_PATH))
    parser.add_argument(
        "--years",
        default=",".join(str(y) for y in DEFAULT_YEARS),
        help="Comma-separated years to backfill (default: 2022,2023,2024)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print DB counts only, no write")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds between downloads (default: 2.0)")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"ERROR: DB not found at {db_path}")
        sys.exit(1)

    years = [int(y.strip()) for y in args.years.split(",") if y.strip()]

    print("CFTC 3-Year Historical Backfill")
    print("=" * 50)
    print(f"  DB:    {db_path}")
    print(f"  Years: {years}")
    print()

    before = get_counts(db_path)
    print("  Before:")
    for k, v in before.items():
        print(f"    {k}: {v:,}")

    if args.dry_run:
        print("\n  [dry-run] No writes performed.")
        return

    print()

    from agent.data.cache import DataCache
    from agent.pipeline.store import PipelineStore
    from agent.tools.cftc import CFTCTool as CftcTool

    store = PipelineStore(str(db_path))
    cache = DataCache(str(db_path.parent / "cache.db"))
    tool = CftcTool(pipeline_store=store, cache=cache)

    for year in sorted(years):
        print(f"  Downloading CFTC {year}...")
        result = tool.execute(mode="historical", year=year, top_n=1000)
        if result.success:
            n_contracts = len(result.data.get("contracts", [])) if result.data else 0
            print(f"    ✓ year={year}  contracts_processed={n_contracts}")
        else:
            print(f"    ✗ year={year}  error={result.output[:120]}")

        if year != years[-1]:
            time.sleep(args.delay)

    print()
    after = get_counts(db_path)
    print("  After:")
    for k, v in after.items():
        delta = v - before[k]
        print(f"    {k}: {v:,}  (+{delta:,})")

    print()
    tracks = after["cftc_tracks_links"]
    obs = after["cftc_observations"]
    if tracks >= 19 and obs > 10000:
        print("  ✓ SUCCESS: CFTC backfill complete — all 19 instruments linked, >10K observations")
    elif tracks < 19:
        print(f"  ⚠ WARNING: Only {tracks}/19 cftc_tracks links. Check cftc_code_to_ticker() mappings.")
    elif obs <= 10000:
        print(f"  ⚠ WARNING: Only {obs:,} observations (target >10K). May need more years.")

    print()
    print("  Next: run 'python scripts/phase40_gnn_backtest.py' to measure IC impact")
    print("        and 'python scripts/source_ablation.py --sources cftc' for attribution")


if __name__ == "__main__":
    main()
