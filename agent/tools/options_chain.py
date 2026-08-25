"""
Tool: Options chain EOD snapshots (M15.1) via yfinance.

Stores aggregate chain statistics on instrument entities — not full chain dumps
(to keep pipeline.db bounded). Full chains are reconstructed at feature time
from stored ATM/skew/term summaries + optional re-fetch.

Verified: SPY, QQQ, ^VIX, ^SPX, CL return option expiries; CL=F / GC=F do not.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

UTC = timezone.utc

from agent.tools.base import Tool, ToolResult
from agent.tools.m15_universe import all_options_tickers, instrument_def_for_ticker

if TYPE_CHECKING:
    from agent.pipeline.store import PipelineStore

log = logging.getLogger(__name__)

_SOURCE = "options_chain"


def _date_to_ts(date_str: str) -> float:
    dt = datetime.strptime(date_str[:10], "%Y-%m-%d").replace(tzinfo=UTC)
    return dt.timestamp()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def summarize_chain(
    underlying: str,
    expiry: str,
    calls,
    puts,
    spot: float | None,
) -> dict[str, Any]:
    """Aggregate one expiry's chain into a compact M15 observation payload."""
    spot_f = _safe_float(spot, 0.0)
    if spot_f <= 0 and len(calls) > 0:
        spot_f = _safe_float(calls["strike"].median())

    def _atm_iv(df, side: str) -> float | None:
        if df is None or len(df) == 0 or spot_f <= 0:
            return None
        idx = (df["strike"] - spot_f).abs().idxmin()
        row = df.loc[idx]
        iv = _safe_float(row.get("impliedVolatility"), -1.0)
        return iv if iv > 0 else None

    call_iv = _atm_iv(calls, "call")
    put_iv = _atm_iv(puts, "put")

    call_oi = (
        float(calls["openInterest"].fillna(0).sum()) if len(calls) else 0.0
    )
    put_oi = float(puts["openInterest"].fillna(0).sum()) if len(puts) else 0.0
    pc_oi = (put_oi / call_oi) if call_oi > 1e-6 else None

    return {
        "underlying": underlying,
        "expiry": expiry,
        "spot": spot_f,
        "n_calls": int(len(calls)),
        "n_puts": int(len(puts)),
        "atm_call_iv": call_iv,
        "atm_put_iv": put_iv,
        "put_call_oi_ratio": pc_oi,
        "total_open_interest": call_oi + put_oi,
    }


def fetch_chain_snapshot(
    ticker: str,
    *,
    expiry_index: int = 0,
) -> dict[str, Any] | None:
    """Fetch one expiry snapshot for underlying. Returns None if no chain."""
    import yfinance as yf

    t = yf.Ticker(ticker)
    expiries = t.options
    if not expiries or expiry_index >= len(expiries):
        return None
    expiry = expiries[expiry_index]
    chain = t.option_chain(expiry)
    spot = None
    try:
        hist = t.history(period="5d")
        if hist is not None and len(hist) > 0:
            spot = float(hist["Close"].iloc[-1])
    except Exception:
        pass
    summary = summarize_chain(ticker, expiry, chain.calls, chain.puts, spot)
    summary["fetched_at"] = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    return summary


def persist_options_snapshot(
    store: PipelineStore,
    ticker: str,
    summary: dict[str, Any],
    *,
    observed_at: float | None = None,
) -> bool:
    from agent.pipeline.entity import entity_id_from_key

    eid = entity_id_from_key("instrument", ticker)
    meta = {"ticker": ticker, "asset_class": "options_underlying"}
    inst = instrument_def_for_ticker(ticker)
    if inst is not None:
        meta["asset_class"] = inst.asset_class
    store.register_entity(
        entity_type="instrument",
        canonical_name=ticker,
        entity_id=eid,
        metadata=meta,
    )
    ts = observed_at if observed_at is not None else time.time()
    store.store_entity_observation(
        entity_id=eid,
        source_tool=_SOURCE,
        observed_at=ts,
        observation_type="options_chain_eod",
        value=summary,
        depth_level=2,
    )
    return True


class OptionsChainTool(Tool):
    """Agent tool: fetch options chain summary for one underlying."""

    def __init__(self, pipeline_store: PipelineStore | None = None) -> None:
        self._store = pipeline_store

    @property
    def name(self) -> str:
        return "options_chain"

    @property
    def description(self) -> str:
        return (
            "Fetch EOD options chain summary (ATM IV, put/call OI) for a US "
            "equity/ETF/index via yfinance. Persists options_chain_eod observations."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Underlying symbol (e.g. SPY, ^VIX, CL).",
                },
                "expiry_index": {
                    "type": "integer",
                    "description": "Index into listed expiries (0 = nearest).",
                    "default": 0,
                },
            },
            "required": ["ticker"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        ticker = str(kwargs.get("ticker", "SPY")).strip()
        expiry_index = int(kwargs.get("expiry_index", 0))
        summary = fetch_chain_snapshot(ticker, expiry_index=expiry_index)
        if summary is None:
            return ToolResult(
                success=False,
                output=f"No options chain for {ticker}",
                data=None,
            )
        if self._store is not None:
            persist_options_snapshot(self._store, ticker, summary)
        return ToolResult(
            success=True,
            output=f"Options chain {ticker} expiry={summary.get('expiry')}",
            data=summary,
        )


def ingest_options_chains(
    store: PipelineStore,
    tickers: list[str] | None = None,
    *,
    include_should: bool = True,
) -> dict[str, Any]:
    """Batch ingest options_chain_eod for M15 universe."""
    symbols = tickers or all_options_tickers(include_should=include_should)
    ok: list[str] = []
    skipped: list[str] = []
    for ticker in symbols:
        summary = fetch_chain_snapshot(ticker, expiry_index=0)
        if summary is None:
            skipped.append(ticker)
            continue
        persist_options_snapshot(store, ticker, summary)
        ok.append(ticker)
        time.sleep(0.35)
    return {
        "stored": len(ok),
        "tickers_ok": ok,
        "tickers_skipped": skipped,
    }


def run_options_chain_ingest(
    params: dict[str, Any],
    upstream_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """DAG callable: ingest M15 options snapshots."""
    from agent.pipeline.store import PipelineStore

    db_path = params.get("db_path", ".tirra_pipeline/pipeline.db")
    include_should = bool(params.get("include_should", True))
    store = PipelineStore(db_path)
    try:
        return ingest_options_chains(store, include_should=include_should)
    finally:
        store.close()
