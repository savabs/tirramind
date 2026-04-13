"""TirraMind — Instrument Universe

Defines the global tradeable instrument universe and daily price ingest.

Design:
    - InstrumentDef: frozen dataclass describing one instrument.
    - INSTRUMENTS: canonical tuple of ~90 instruments (all yfinance-verified).
    - ingest_daily_prices(): batch-fetch prices, register entities, store observations.

Instruments are first-class GNN entity nodes (type="instrument").
The GNN's cross-type attention discovers which entity patterns predict
instrument behavior. No hand-coded entity→instrument mapping.

Verified 2026-04-13: 89/90 tickers returned data from yfinance (period='5d').
LBS=F (lumber) was delisted — replaced with OJ=F (orange juice).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from agent.pipeline.store import PipelineStore

log = logging.getLogger(__name__)


# ── Instrument definition ──────────────────────────────────────


@dataclass(frozen=True)
class InstrumentDef:
    """A single tradeable instrument."""

    ticker: str
    name: str
    asset_class: str  # commodity_future | fx | equity_index | equity_etf | sector_etf | fixed_income | vol | crypto
    region: str  # US | Europe | Asia | LatAm | Pacific | Global | EM
    is_tradeable: bool = True


# ── Instrument universe ────────────────────────────────────────

INSTRUMENTS: tuple[InstrumentDef, ...] = (
    # ── Commodity Futures (20) ─────────────────────────────
    InstrumentDef("CL=F", "WTI Crude Oil", "commodity_future", "Global"),
    InstrumentDef("BZ=F", "Brent Crude Oil", "commodity_future", "Global"),
    InstrumentDef("NG=F", "Natural Gas", "commodity_future", "US"),
    InstrumentDef("RB=F", "RBOB Gasoline", "commodity_future", "US"),
    InstrumentDef("GC=F", "Gold", "commodity_future", "Global"),
    InstrumentDef("SI=F", "Silver", "commodity_future", "Global"),
    InstrumentDef("PL=F", "Platinum", "commodity_future", "Global"),
    InstrumentDef("PA=F", "Palladium", "commodity_future", "Global"),
    InstrumentDef("HG=F", "Copper", "commodity_future", "Global"),
    InstrumentDef("ZW=F", "Wheat", "commodity_future", "US"),
    InstrumentDef("ZC=F", "Corn", "commodity_future", "US"),
    InstrumentDef("ZS=F", "Soybeans", "commodity_future", "US"),
    InstrumentDef("KC=F", "Coffee", "commodity_future", "Global"),
    InstrumentDef("CC=F", "Cocoa", "commodity_future", "Global"),
    InstrumentDef("CT=F", "Cotton", "commodity_future", "US"),
    InstrumentDef("SB=F", "Sugar", "commodity_future", "Global"),
    InstrumentDef("ZO=F", "Oats", "commodity_future", "US"),
    InstrumentDef("OJ=F", "Orange Juice", "commodity_future", "US"),
    InstrumentDef("LE=F", "Live Cattle", "commodity_future", "US"),
    InstrumentDef("HE=F", "Lean Hogs", "commodity_future", "US"),
    # ── FX Pairs (15) ─────────────────────────────────────
    InstrumentDef("EURUSD=X", "EUR/USD", "fx", "Global"),
    InstrumentDef("USDJPY=X", "USD/JPY", "fx", "Global"),
    InstrumentDef("GBPUSD=X", "GBP/USD", "fx", "Global"),
    InstrumentDef("USDCHF=X", "USD/CHF", "fx", "Global"),
    InstrumentDef("AUDUSD=X", "AUD/USD", "fx", "Global"),
    InstrumentDef("USDCAD=X", "USD/CAD", "fx", "Global"),
    InstrumentDef("NZDUSD=X", "NZD/USD", "fx", "Global"),
    InstrumentDef("EURGBP=X", "EUR/GBP", "fx", "Global"),
    InstrumentDef("EURJPY=X", "EUR/JPY", "fx", "Global"),
    InstrumentDef("GBPJPY=X", "GBP/JPY", "fx", "Global"),
    InstrumentDef("USDMXN=X", "USD/MXN", "fx", "EM"),
    InstrumentDef("USDBRL=X", "USD/BRL", "fx", "EM"),
    InstrumentDef("USDINR=X", "USD/INR", "fx", "EM"),
    InstrumentDef("USDCNY=X", "USD/CNY", "fx", "EM"),
    InstrumentDef("USDZAR=X", "USD/ZAR", "fx", "EM"),
    # ── Equity Index Futures (4) ───────────────────────────
    InstrumentDef("ES=F", "S&P 500 Futures", "equity_index", "US"),
    InstrumentDef("NQ=F", "Nasdaq 100 Futures", "equity_index", "US"),
    InstrumentDef("YM=F", "Dow Futures", "equity_index", "US"),
    InstrumentDef("RTY=F", "Russell 2000 Futures", "equity_index", "US"),
    # ── Equity ETFs (21) ──────────────────────────────────
    InstrumentDef("SPY", "S&P 500 ETF", "equity_etf", "US"),
    InstrumentDef("QQQ", "Nasdaq 100 ETF", "equity_etf", "US"),
    InstrumentDef("IWM", "Russell 2000 ETF", "equity_etf", "US"),
    InstrumentDef("DIA", "Dow Jones ETF", "equity_etf", "US"),
    InstrumentDef("EWZ", "Brazil ETF", "equity_etf", "LatAm"),
    InstrumentDef("EWG", "Germany ETF", "equity_etf", "Europe"),
    InstrumentDef("FXI", "China ETF", "equity_etf", "Asia"),
    InstrumentDef("EWJ", "Japan ETF", "equity_etf", "Asia"),
    InstrumentDef("EWY", "South Korea ETF", "equity_etf", "Asia"),
    InstrumentDef("EWA", "Australia ETF", "equity_etf", "Pacific"),
    InstrumentDef("EWC", "Canada ETF", "equity_etf", "US"),
    InstrumentDef("EWU", "United Kingdom ETF", "equity_etf", "Europe"),
    InstrumentDef("EWQ", "France ETF", "equity_etf", "Europe"),
    InstrumentDef("EWP", "Spain ETF", "equity_etf", "Europe"),
    InstrumentDef("EWI", "Italy ETF", "equity_etf", "Europe"),
    InstrumentDef("INDA", "India ETF", "equity_etf", "Asia"),
    InstrumentDef("EWT", "Taiwan ETF", "equity_etf", "Asia"),
    InstrumentDef("EWH", "Hong Kong ETF", "equity_etf", "Asia"),
    InstrumentDef("THD", "Thailand ETF", "equity_etf", "Asia"),
    InstrumentDef("EWW", "Mexico ETF", "equity_etf", "LatAm"),
    InstrumentDef("VGK", "FTSE Europe ETF", "equity_etf", "Europe"),
    # ── Sector ETFs (15) ──────────────────────────────────
    InstrumentDef("XLE", "Energy Select", "sector_etf", "US"),
    InstrumentDef("XLF", "Financials Select", "sector_etf", "US"),
    InstrumentDef("XLK", "Technology Select", "sector_etf", "US"),
    InstrumentDef("XLV", "Healthcare Select", "sector_etf", "US"),
    InstrumentDef("XLI", "Industrials Select", "sector_etf", "US"),
    InstrumentDef("XLP", "Consumer Staples Select", "sector_etf", "US"),
    InstrumentDef("XLY", "Consumer Discretionary Select", "sector_etf", "US"),
    InstrumentDef("XLB", "Materials Select", "sector_etf", "US"),
    InstrumentDef("XLU", "Utilities Select", "sector_etf", "US"),
    InstrumentDef("XLRE", "Real Estate Select", "sector_etf", "US"),
    InstrumentDef("XLC", "Communication Services Select", "sector_etf", "US"),
    InstrumentDef("GDX", "Gold Miners ETF", "sector_etf", "Global"),
    InstrumentDef("SLV", "Silver ETF", "sector_etf", "Global"),
    InstrumentDef("USO", "US Oil Fund", "sector_etf", "US"),
    InstrumentDef("UNG", "US Natural Gas Fund", "sector_etf", "US"),
    # ── Fixed Income (10) ─────────────────────────────────
    InstrumentDef("ZN=F", "10-Year T-Note Futures", "fixed_income", "US"),
    InstrumentDef("ZB=F", "30-Year T-Bond Futures", "fixed_income", "US"),
    InstrumentDef("ZF=F", "5-Year T-Note Futures", "fixed_income", "US"),
    InstrumentDef("TLT", "20+ Year Treasury ETF", "fixed_income", "US"),
    InstrumentDef("IEF", "7-10 Year Treasury ETF", "fixed_income", "US"),
    InstrumentDef("SHY", "1-3 Year Treasury ETF", "fixed_income", "US"),
    InstrumentDef("HYG", "High Yield Corporate ETF", "fixed_income", "US"),
    InstrumentDef("LQD", "Investment Grade Corporate ETF", "fixed_income", "US"),
    InstrumentDef("EMB", "EM Bond ETF", "fixed_income", "EM"),
    InstrumentDef("AGG", "US Aggregate Bond ETF", "fixed_income", "US"),
    # ── Volatility (3) ────────────────────────────────────
    InstrumentDef("^VIX", "VIX Index", "vol", "US", is_tradeable=False),
    InstrumentDef("VIXY", "VIX Short-Term Futures ETF", "vol", "US"),
    InstrumentDef("UVXY", "Ultra VIX Short-Term Futures ETF", "vol", "US"),
    # ── Crypto (2) ────────────────────────────────────────
    InstrumentDef("BTC-USD", "Bitcoin", "crypto", "Global"),
    InstrumentDef("ETH-USD", "Ethereum", "crypto", "Global"),
)


# ── Helpers ────────────────────────────────────────────────────


def tradeable_instruments() -> list[InstrumentDef]:
    """Return only instruments that are tradeable (excludes e.g. ^VIX)."""
    return [i for i in INSTRUMENTS if i.is_tradeable]


def instruments_by_class(asset_class: str) -> list[InstrumentDef]:
    """Return instruments matching the given asset class."""
    return [i for i in INSTRUMENTS if i.asset_class == asset_class]


def ticker_to_instrument() -> dict[str, InstrumentDef]:
    """Return {ticker: InstrumentDef} lookup for all instruments."""
    return {i.ticker: i for i in INSTRUMENTS}


# ── Daily price ingest ─────────────────────────────────────────

# Entity ID helper (inline to avoid circular import at module level)
_ENTITY_TYPE = "instrument"
_SOURCE_TOOL = "instrument_universe"


def _entity_id(ticker: str) -> str:
    """Deterministic entity ID for an instrument ticker."""
    import hashlib

    raw = f"{_ENTITY_TYPE}:{ticker}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def ingest_daily_prices(
    store: PipelineStore,
    as_of: date | None = None,
    lookback_days: int = 30,
) -> dict[str, Any]:
    """Fetch daily prices for all tradeable instruments and store as entity observations.

    Parameters
    ----------
    store : PipelineStore to write entities/observations.
    as_of : Reference date (default: today). Only the most recent row is stored as
            today's observation, but we fetch ``lookback_days`` to compute vol/avg_volume.
    lookback_days : How many days of history to fetch for vol calculation.

    Returns
    -------
    Summary dict with keys: instruments_fetched, instruments_failed, observations_stored.
    """
    import yfinance as yf

    if as_of is None:
        as_of = date.today()

    instruments = tradeable_instruments()
    tickers = [i.ticker for i in instruments]
    ticker_map = {i.ticker: i for i in instruments}

    # ── Batch download ─────────────────────────────────────
    log.info("Downloading %d instruments, lookback=%dd", len(tickers), lookback_days)
    try:
        # yfinance batch download: returns MultiIndex columns (ticker, field)
        # when multiple tickers. Returns single-level when 1 ticker.
        raw = yf.download(
            tickers,
            period=f"{lookback_days}d",
            interval="1d",
            group_by="ticker",
            progress=False,
            threads=True,
        )
    except Exception:
        log.exception("Batch yfinance download failed")
        raise RuntimeError("Instrument price download failed entirely")

    # ── Process per ticker ─────────────────────────────────
    fetched = 0
    failed: list[str] = []
    obs_stored = 0
    observed_at = time.mktime(as_of.timetuple())

    for ticker in tickers:
        try:
            # Extract this ticker's data from the batch result
            if len(tickers) == 1:
                df = raw  # single ticker: flat columns
            else:
                if ticker not in raw.columns.get_level_values(0):
                    failed.append(ticker)
                    continue
                df = raw[ticker]

            # Drop rows with NaN close
            df = df.dropna(subset=["Close"])
            if df.empty:
                failed.append(ticker)
                continue

            # ── Compute signals from history ───────────────
            closes = df["Close"].values.astype(float)
            volumes = (
                df["Volume"].values.astype(float)
                if "Volume" in df.columns
                else np.zeros(len(closes))
            )
            highs = df["High"].values.astype(float) if "High" in df.columns else closes
            lows = df["Low"].values.astype(float) if "Low" in df.columns else closes

            # Log returns
            if len(closes) >= 2:
                log_returns = np.diff(np.log(closes))
                latest_return = float(log_returns[-1])
            else:
                log_returns = np.array([])
                latest_return = float("nan")

            latest_close = float(closes[-1])
            latest_volume = float(volumes[-1]) if len(volumes) > 0 else 0.0

            # 20d realized vol (annualised)
            if len(log_returns) >= 20:
                realized_vol = float(np.std(log_returns[-20:]) * math.sqrt(252))
            elif len(log_returns) >= 2:
                realized_vol = float(np.std(log_returns) * math.sqrt(252))
            else:
                realized_vol = float("nan")

            # 20d average volume
            if len(volumes) >= 20:
                avg_volume = float(np.mean(volumes[-20:]))
            elif len(volumes) >= 1:
                avg_volume = float(np.mean(volumes))
            else:
                avg_volume = 0.0

            # Intraday range (latest bar)
            if len(highs) > 0 and len(lows) > 0:
                intraday_range = float(highs[-1] - lows[-1])
            else:
                intraday_range = 0.0

            # ── Register entity ────────────────────────────
            inst = ticker_map[ticker]
            eid = _entity_id(ticker)
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

            # ── Store observations ─────────────────────────
            if not math.isnan(latest_return):
                store.store_entity_observation(
                    entity_id=eid,
                    source_tool=_SOURCE_TOOL,
                    observed_at=observed_at,
                    observation_type="instrument_return",
                    value={"log_return": latest_return, "close": latest_close},
                    depth_level=1,
                )
                obs_stored += 1

            store.store_entity_observation(
                entity_id=eid,
                source_tool=_SOURCE_TOOL,
                observed_at=observed_at,
                observation_type="instrument_volume",
                value={"volume": latest_volume, "avg_volume_20d": avg_volume},
                depth_level=1,
            )
            obs_stored += 1

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
                obs_stored += 1

            fetched += 1

        except Exception:
            log.warning("Failed to process instrument %s", ticker, exc_info=True)
            failed.append(ticker)

    # ── Check failure rate ─────────────────────────────────
    total = len(tickers)
    if len(failed) > total / 2:
        raise RuntimeError(
            f"Instrument ingest: {len(failed)}/{total} tickers failed "
            f"(>50%). Likely API issue. Failed: {failed[:10]}..."
        )

    summary = {
        "instruments_fetched": fetched,
        "instruments_failed": failed,
        "observations_stored": obs_stored,
        "as_of": as_of.isoformat(),
    }
    log.info(
        "Instrument ingest complete: %d/%d fetched, %d observations stored",
        fetched,
        total,
        obs_stored,
    )
    return summary


# ── DAG callback ───────────────────────────────────────────────


def run_instrument_ingest(
    params: dict[str, Any],
    upstream_results: dict[str, Any],
) -> dict[str, Any]:
    """FunctionOperator callback for the daily_collection DAG.

    params:
        db_path: str — PipelineStore database path (injected by DAG builder)
    """
    from agent.pipeline.store import PipelineStore

    db_path = params.get("db_path", ".tirra_pipeline/pipeline.db")
    store = PipelineStore(db_path)
    try:
        return ingest_daily_prices(store)
    finally:
        store.close()
