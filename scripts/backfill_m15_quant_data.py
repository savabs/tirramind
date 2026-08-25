#!/usr/bin/env python3
"""Backfill M15 quant data: US yield curve, options chains, dividends.

Usage:
    python scripts/backfill_m15_quant_data.py
    python scripts/backfill_m15_quant_data.py --db-path .tirra_pipeline/pipeline.db
    python scripts/backfill_m15_quant_data.py --only rates
    python scripts/backfill_m15_quant_data.py --only options --only dividends
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.pipeline.store import PipelineStore
from agent.tools.dividend_data import ingest_dividends
from agent.tools.options_chain import ingest_options_chains
from agent.tools.sovereign_debt import SovereignDebtTool

log = logging.getLogger(__name__)
UTC = timezone.utc


def backfill_eu_jp_rates(store: PipelineStore, months: int) -> dict:
    """Backfill EU (DE, FR, IT) and Japan sovereign_yield rows."""
    tool = SovereignDebtTool(pipeline_store=store)
    eu_countries = ["DE", "FR", "IT"]
    total = 0
    now = datetime.now(tz=UTC)
    for i in range(months):
        year = now.year
        month = now.month - i
        while month <= 0:
            month += 12
            year -= 1
        yyyymm = f"{year:04d}-{month:02d}"
        for mode, extra in (
            ("eu_yields", {"countries": eu_countries, "month": yyyymm}),
            ("jp_yields", {}),
        ):
            result = tool.execute(mode=mode, **extra)
            if result.success and result.data:
                n = len(result.data.get("records", [])) or 1
                total += n
                log.info("%s %s: ok", mode, yyyymm if mode == "eu_yields" else "latest")
            else:
                log.warning("%s failed for %s: %s", mode, yyyymm, result.output)
    return {"months": months, "eu_jp_obs_approx": total}


def backfill_us_rates(store: PipelineStore, months: int) -> dict:
    tool = SovereignDebtTool(pipeline_store=store)
    total = 0
    now = datetime.now(tz=UTC)
    for i in range(months):
        year = now.year
        month = now.month - i
        while month <= 0:
            month += 12
            year -= 1
        yyyymm = f"{year:04d}-{month:02d}"
        result = tool.execute(mode="us_yields", month=yyyymm)
        if result.success and result.data:
            n = len(result.data.get("records", []))
            total += n
            log.info("US yields %s: %d days", yyyymm, n)
        else:
            log.warning("US yields failed for %s: %s", yyyymm, result.output)
    return {"months": months, "yield_obs_approx": total}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Backfill M15 quant data")
    parser.add_argument("--db-path", default=".tirra_pipeline/pipeline.db")
    parser.add_argument("--months", type=int, default=24, help="US Treasury months")
    parser.add_argument(
        "--only",
        action="append",
        choices=["rates", "eu_jp", "options", "dividends"],
    )
    parser.add_argument("--must-only", action="store_true", help="Skip should-add tickers")
    args = parser.parse_args()

    db = Path(args.db_path)
    if not db.exists():
        log.error("DB not found: %s", db)
        return 2

    only = set(args.only or ["rates", "options", "dividends"])
    include_should = not args.must_only
    store = PipelineStore(str(db))
    summary: dict = {}

    try:
        if "rates" in only:
            summary["rates"] = backfill_us_rates(store, args.months)
        if "eu_jp" in only:
            summary["eu_jp"] = backfill_eu_jp_rates(store, min(args.months, 24))
        if "options" in only:
            summary["options"] = ingest_options_chains(
                store, include_should=include_should
            )
        if "dividends" in only:
            summary["dividends"] = ingest_dividends(
                store, include_should=include_should
            )
    finally:
        store.close()

    print("M15 backfill complete:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
