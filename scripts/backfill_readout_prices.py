#!/usr/bin/env python3
"""Backfill per-trading-day closes for MP-1 readout instruments.

Ghost alert resolution needs contiguous sessions after the chain anchor.
``ingest_daily_prices`` only stores one snapshot per ingest run.

Usage:
    python scripts/backfill_readout_prices.py
    python scripts/backfill_readout_prices.py --lookback-days 90 --ticker CL=F
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config.settings import AgentConfig
from agent.pipeline.store import PipelineStore
from agent.tools.instrument_universe import MP1_READOUT_TICKERS, backfill_recent_readout_prices


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill readout instrument daily bars")
    parser.add_argument("--lookback-days", type=int, default=120)
    parser.add_argument("--ticker", action="append", help="Ticker (repeatable; default: MP-1 readouts)")
    args = parser.parse_args()

    tickers = args.ticker or list(MP1_READOUT_TICKERS)
    config = AgentConfig.from_env()
    store = PipelineStore(config.pipeline.db_path)
    try:
        result = backfill_recent_readout_prices(
            store, tickers=tickers, lookback_days=args.lookback_days
        )
    finally:
        store.close()

    print(f"Filled: {', '.join(result['tickers_filled']) or 'none'}")
    print(f"Observations stored: {result['observations_stored']}")
    if result["tickers_failed"]:
        print(f"Failed: {', '.join(result['tickers_failed'])}")
        sys.exit(1)


if __name__ == "__main__":
    main()
