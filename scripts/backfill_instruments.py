#!/usr/bin/env python3
"""Backfill 2 years of daily instrument prices into PipelineStore.

Usage:
    python scripts/backfill_instruments.py --db .tirra_pipeline/pipeline.db
    python scripts/backfill_instruments.py --db pipeline.db --years 1
    python scripts/backfill_instruments.py --db pipeline.db --dry-run

Design:
    Downloads 2yr of daily OHLCV data for all 89 tradeable instruments in one
    yfinance batch call, then iterates over each trading day and stores
    per-instrument observations (return, volume, volatility) into PipelineStore.

    Idempotency: For each instrument, queries existing observations from the store
    and skips dates that are already present. Safe to re-run after partial failure.

    Rolling windows:
        - log_return: requires previous close → computed from day 1 onward
        - realized_vol_20d: rolling 20-day std of log returns × sqrt(252)
        - avg_volume_20d: rolling 20-day mean volume
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

# Add project root to path when run as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.pipeline.entity import entity_id_from_key
from agent.pipeline.store import PipelineStore
from agent.tools.instrument_universe import (
    INSTRUMENTS,
    tradeable_instruments,
)

log = logging.getLogger("backfill_instruments")

_ENTITY_TYPE = "instrument"
_SOURCE_TOOL = "instrument_universe"


def _date_to_ts(d: date) -> float:
    """Convert a date to midnight UTC timestamp."""
    return datetime(d.year, d.month, d.day).timestamp()


def _get_existing_dates(store: PipelineStore, entity_id: str) -> set[date]:
    """Query all dates we already have observations for this instrument.

    Returns a set of date objects (observed_at truncated to day).
    """
    obs = store.query_entity_observations(
        entity_id,
        source_tool=_SOURCE_TOOL,
        limit=10_000,
    )
    dates = set()
    for o in obs:
        ts = o.get("observed_at", 0.0)
        if ts > 0:
            dt = datetime.fromtimestamp(ts)
            dates.add(dt.date())
    return dates


def backfill(
    db_path: str,
    years: int = 2,
    dry_run: bool = False,
) -> dict[str, int]:
    """Download and store historical instrument data.

    Parameters
    ----------
    db_path : Path to the PipelineStore SQLite database.
    years   : How many years of history to download (default 2).
    dry_run : If True, download data but don't write to store.

    Returns
    -------
    Summary dict with keys: instruments, days_total, observations_stored, days_skipped.
    """
    import yfinance as yf

    instruments = tradeable_instruments()
    tickers = [i.ticker for i in instruments]
    ticker_map = {i.ticker: i for i in instruments}

    # ── Download batch ─────────────────────────────────────
    end_date = date.today()
    start_date = end_date - timedelta(days=years * 365)

    log.info(
        "Downloading %d instruments from %s to %s",
        len(tickers),
        start_date.isoformat(),
        end_date.isoformat(),
    )

    raw = yf.download(
        tickers,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        interval="1d",
        group_by="ticker",
        progress=True,
        threads=True,
    )

    if raw.empty:
        log.error("yfinance returned empty DataFrame — aborting")
        return {"instruments": 0, "days_total": 0, "observations_stored": 0, "days_skipped": 0}

    log.info("Downloaded %d rows × %d columns", len(raw), len(raw.columns))

    # ── Open store ─────────────────────────────────────────
    store = PipelineStore(db_path) if not dry_run else None

    total_obs = 0
    total_days = 0
    total_skipped = 0
    failed_tickers: list[str] = []

    for i, inst in enumerate(instruments):
        ticker = inst.ticker
        eid = entity_id_from_key(_ENTITY_TYPE, ticker)

        # Extract per-ticker DataFrame
        try:
            if len(tickers) == 1:
                df = raw.copy()
            else:
                if ticker not in raw.columns.get_level_values(0):
                    log.warning("Ticker %s not in download results", ticker)
                    failed_tickers.append(ticker)
                    continue
                df = raw[ticker].copy()

            df = df.dropna(subset=["Close"])
            if df.empty:
                log.warning("Ticker %s: no data after NaN drop", ticker)
                failed_tickers.append(ticker)
                continue
        except Exception:
            log.warning("Failed to extract %s from batch", ticker, exc_info=True)
            failed_tickers.append(ticker)
            continue

        closes = df["Close"].values.astype(float)
        volumes = df["Volume"].values.astype(float) if "Volume" in df.columns else np.zeros(len(closes))
        highs = df["High"].values.astype(float) if "High" in df.columns else closes
        lows = df["Low"].values.astype(float) if "Low" in df.columns else closes
        dates = df.index

        # Compute full series of log returns
        log_returns = np.full(len(closes), np.nan)
        if len(closes) >= 2:
            log_returns[1:] = np.diff(np.log(closes))

        # ── Register entity once ──────────────────────────
        if store is not None:
            store.register_entity(
                entity_type=_ENTITY_TYPE,
                canonical_name=inst.name,
                entity_id=eid,
                metadata={
                    "ticker": ticker,
                    "asset_class": inst.asset_class,
                    "region": inst.region,
                },
            )

        # ── Check existing dates for idempotency ──────────
        existing_dates: set[date] = set()
        if store is not None:
            existing_dates = _get_existing_dates(store, eid)

        # ── Store day-by-day observations ─────────────────
        ticker_obs = 0
        ticker_skipped = 0

        for day_idx in range(len(closes)):
            obs_date = dates[day_idx]
            # Convert pandas Timestamp to date
            if hasattr(obs_date, "date"):
                obs_d = obs_date.date()
            else:
                obs_d = obs_date

            if obs_d in existing_dates:
                ticker_skipped += 1
                continue

            observed_at = _date_to_ts(obs_d)
            latest_close = float(closes[day_idx])
            latest_volume = float(volumes[day_idx])

            # Log return
            lr = float(log_returns[day_idx])

            # Rolling 20d realized vol (annualized)
            if day_idx >= 20:
                window_returns = log_returns[day_idx - 19 : day_idx + 1]
                valid_returns = window_returns[~np.isnan(window_returns)]
                if len(valid_returns) >= 2:
                    realized_vol = float(np.std(valid_returns) * math.sqrt(252))
                else:
                    realized_vol = float("nan")
            elif day_idx >= 2:
                window_returns = log_returns[1 : day_idx + 1]
                valid_returns = window_returns[~np.isnan(window_returns)]
                if len(valid_returns) >= 2:
                    realized_vol = float(np.std(valid_returns) * math.sqrt(252))
                else:
                    realized_vol = float("nan")
            else:
                realized_vol = float("nan")

            # Rolling 20d avg volume
            if day_idx >= 19:
                avg_vol = float(np.mean(volumes[day_idx - 19 : day_idx + 1]))
            elif day_idx >= 1:
                avg_vol = float(np.mean(volumes[: day_idx + 1]))
            else:
                avg_vol = float(volumes[day_idx])

            # Intraday range
            intraday_range = float(highs[day_idx] - lows[day_idx])

            if store is None:
                ticker_obs += 3 if not math.isnan(lr) else 2
                continue

            # ── Store observations ────────────────────────
            if not math.isnan(lr):
                store.store_entity_observation(
                    entity_id=eid,
                    source_tool=_SOURCE_TOOL,
                    observed_at=observed_at,
                    observation_type="instrument_return",
                    value={"log_return": lr, "close": latest_close},
                    depth_level=1,
                )
                ticker_obs += 1

            store.store_entity_observation(
                entity_id=eid,
                source_tool=_SOURCE_TOOL,
                observed_at=observed_at,
                observation_type="instrument_volume",
                value={"volume": latest_volume, "avg_volume_20d": avg_vol},
                depth_level=1,
            )
            ticker_obs += 1

            if not math.isnan(realized_vol):
                store.store_entity_observation(
                    entity_id=eid,
                    source_tool=_SOURCE_TOOL,
                    observed_at=observed_at,
                    observation_type="instrument_volatility",
                    value={
                        "realized_vol_20d": realized_vol,
                        "intraday_range": intraday_range,
                    },
                    depth_level=1,
                )
                ticker_obs += 1

        total_obs += ticker_obs
        total_days += len(closes) - ticker_skipped
        total_skipped += ticker_skipped

        if (i + 1) % 10 == 0:
            log.info(
                "Progress: %d/%d tickers, %d obs stored so far",
                i + 1,
                len(instruments),
                total_obs,
            )

    if store is not None:
        store.close()

    summary = {
        "instruments": len(instruments) - len(failed_tickers),
        "instruments_failed": len(failed_tickers),
        "days_total": total_days,
        "observations_stored": total_obs,
        "days_skipped": total_skipped,
    }

    log.info(
        "Backfill complete: %d instruments, %d days, %d observations stored, %d days skipped",
        summary["instruments"],
        summary["days_total"],
        summary["observations_stored"],
        summary["days_skipped"],
    )

    if failed_tickers:
        log.warning("Failed tickers (%d): %s", len(failed_tickers), failed_tickers)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill instrument prices into PipelineStore",
    )
    parser.add_argument(
        "--db",
        default=".tirra_pipeline/pipeline.db",
        help="Path to PipelineStore SQLite database",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=2,
        help="Years of history to download (default: 2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download data but don't write to store",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    result = backfill(
        db_path=args.db,
        years=args.years,
        dry_run=args.dry_run,
    )

    print(f"\nBackfill summary:")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
