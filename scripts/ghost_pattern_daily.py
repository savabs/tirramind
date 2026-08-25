#!/usr/bin/env python3
"""Daily MP-1 ghost pattern loop — refresh, scan, resolve.

Usage:
    python scripts/ghost_pattern_daily.py
    python scripts/ghost_pattern_daily.py --skip-fetch   # scan + resolve only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], label: str) -> int:
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    r = subprocess.run(cmd, cwd=str(ROOT))
    return r.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="MP-1 ghost pattern daily loop")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip sensor refresh")
    parser.add_argument("--sessions", type=int, default=5)
    args = parser.parse_args()

    py = sys.executable
    env = {"TIRRA_PIPELINE_DB": ".tirra_pipeline/pipeline.db"}

    if not args.skip_fetch:
        fetch = """
import sys
from datetime import date
sys.path.insert(0, '.')
from agent.config.settings import AgentConfig
from agent.pipeline.store import PipelineStore
from agent.data.cache import DataCache
from agent.tools.cftc import CFTCTool
from agent.tools.energy_supply import EnergySupplyTool
from agent.tools.ais_vessel import ingest_area_daily_snapshot
from agent.tools.instrument_universe import ingest_daily_prices, backfill_recent_readout_prices

config = AgentConfig.from_env()
store = PipelineStore(config.pipeline.db_path)
cache = DataCache()
for tool, kw in [
    (CFTCTool(cache=cache, pipeline_store=store), {"mode": "latest"}),
    (EnergySupplyTool(cache=cache, pipeline_store=store), {"mode": "petroleum_stocks"}),
]:
    tool.execute(**kw)
ingest_area_daily_snapshot(store, cache=cache)
ingest_daily_prices(store, as_of=date.today())
backfill_recent_readout_prices(store, lookback_days=120)
print("fetch done")
"""
        subprocess.run([py, "-c", fetch], cwd=str(ROOT), check=False, env={**dict(__import__("os").environ), **env})

    _run([py, "scripts/mp1_data_health.py"], "MP-1 data health")
    _run([py, "scripts/ghost_pattern_scan.py"], "Chain scan (writes new auto-matches)")
    _run(
        [py, "scripts/resolve_ghost_alert.py", "--all", "--sessions", str(args.sessions)],
        "Resolve pending alerts",
    )

    alerts = sorted((ROOT / "ghost_archive" / "alerts").glob("*.json"))
    pending = 0
    resolved = 0
    for p in alerts:
        import json

        a = json.loads(p.read_text())
        if a.get("outcome"):
            resolved += 1
        else:
            pending += 1
    print(f"\nArchive: {len(alerts)} alerts | {resolved} resolved | {pending} pending")


if __name__ == "__main__":
    main()
