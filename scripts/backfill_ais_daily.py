#!/usr/bin/env python3
"""Backfill AIS daily activity for MP-1 ghost chains.

Digitraffic has no historical Baltic bbox snapshots. This script:
  1. Backfills ~90d Finnish port-call tanker counts (proxy series)
  2. Stores today's live full_baltic tanker count (area_daily_activity)

Usage:
    python scripts/backfill_ais_daily.py
    python scripts/backfill_ais_daily.py --lookback-days 120
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config.settings import AgentConfig
from agent.data.cache import DataCache
from agent.pipeline.store import PipelineStore
from agent.tools.ais_vessel import backfill_ais_area_daily, backfill_ais_port_call_proxy, ingest_area_daily_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill AIS daily activity for ghost chains")
    parser.add_argument("--lookback-days", type=int, default=60)
    parser.add_argument(
        "--skip-proxy",
        action="store_true",
        help="Only store today's live Baltic snapshot (no port-call history)",
    )
    parser.add_argument("--area", default="full_baltic")
    args = parser.parse_args()

    config = AgentConfig.from_env()
    store = PipelineStore(config.pipeline.db_path)
    cache = DataCache()
    try:
        if args.skip_proxy:
            result = {
                "proxy": {"days_stored": 0, "skipped": True},
                "live": ingest_area_daily_snapshot(
                    store, area_name=args.area, cache=cache
                ),
            }
        else:
            result = backfill_ais_area_daily(
                store,
                lookback_days=args.lookback_days,
                area_name=args.area,
                cache=cache,
            )
    finally:
        store.close()

    proxy = result.get("proxy", {})
    live = result.get("live", {})
    print(f"Proxy days stored: {proxy.get('days_stored', 0)}")
    print(f"Port calls fetched: {proxy.get('port_calls_fetched', 0)}")
    if proxy.get("error"):
        print(f"Proxy error: {proxy['error']}")
    if live.get("stored"):
        print(
            f"Live snapshot: {live.get('tanker_count', '?')} tankers / "
            f"{live.get('vessel_count', '?')} total ({live.get('observed_day', '?')})"
        )
    else:
        print(f"Live snapshot failed: {live.get('error', 'unknown')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
