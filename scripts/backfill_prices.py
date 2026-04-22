#!/usr/bin/env python3
"""Backfill historical daily prices for all instruments.

Usage:
    python scripts/backfill_prices.py [--db-path PATH] [--years N] [--batch-size N]

Fetches N years of daily price data from yfinance for all 90 instruments and
stores each day as a separate observation in the PipelineStore. Supports resume
via --skip-existing (default: on).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from agent.pipeline.store import PipelineStore
from agent.tools.instrument_universe import backfill_historical_prices

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill historical instrument prices")
    parser.add_argument(
        "--db-path",
        default=".tirra_pipeline/pipeline.db",
        help="Path to PipelineStore SQLite DB (default: .tirra_pipeline/pipeline.db)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=3,
        help="Years of history to fetch (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Tickers per yfinance batch (default: 20)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-download even if instrument already has data",
    )
    args = parser.parse_args()

    store = PipelineStore(db_path=args.db_path)

    console.print(
        f"[bold cyan]Backfilling {args.years}y of prices[/] "
        f"[dim]DB: {args.db_path} | batch: {args.batch_size}[/]"
    )

    try:
        result = backfill_historical_prices(
            store=store,
            lookback_years=args.years,
            batch_size=args.batch_size,
            skip_existing=not args.no_skip_existing,
        )
    finally:
        store.close()

    console.print(f"\n[green]Backfill complete![/]")
    console.print(f"  Instruments filled: {result['instruments_backfilled']}")
    console.print(f"  Instruments skipped: {result['instruments_skipped']}")
    console.print(f"  Instruments failed: {len(result['instruments_failed'])}")
    console.print(f"  Total observations: {result['total_observations']}")

    if result["instruments_failed"]:
        console.print(
            f"  [yellow]Failed tickers: {', '.join(result['instruments_failed'][:20])}[/]"
        )

    sys.exit(0 if not result["instruments_failed"] else 1)


if __name__ == "__main__":
    main()
