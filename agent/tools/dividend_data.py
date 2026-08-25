"""
Tool: Dividend history (M15.4) via yfinance.

Persists dividend observations on instrument entities for BSM dividend yield q.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

UTC = timezone.utc

from agent.tools.base import Tool, ToolResult
from agent.tools.m15_universe import all_dividend_tickers, instrument_def_for_ticker

if TYPE_CHECKING:
    from agent.pipeline.store import PipelineStore

log = logging.getLogger(__name__)

_SOURCE = "dividend_data"


def _pandas_ts_to_epoch(ts: Any) -> float:
    if hasattr(ts, "timestamp"):
        return float(ts.timestamp())
    return time.time()


def ingest_dividends(
    store: PipelineStore,
    tickers: list[str] | None = None,
    *,
    include_should: bool = True,
    max_per_ticker: int = 24,
) -> dict[str, Any]:
    """Store recent dividend rows per ticker (observation_type=dividend)."""
    import yfinance as yf

    from agent.pipeline.entity import entity_id_from_key

    symbols = tickers or all_dividend_tickers(include_should=include_should)
    stored = 0
    tickers_ok: list[str] = []
    tickers_empty: list[str] = []

    for ticker in symbols:
        try:
            div = yf.Ticker(ticker).dividends
        except Exception:
            log.warning("Dividend fetch failed for %s", ticker)
            tickers_empty.append(ticker)
            continue
        if div is None or len(div) == 0:
            tickers_empty.append(ticker)
            continue

        inst = instrument_def_for_ticker(ticker)
        meta = {"ticker": ticker}
        if inst is not None:
            meta["asset_class"] = inst.asset_class
        eid = entity_id_from_key("instrument", ticker)
        store.register_entity(
            entity_type="instrument",
            canonical_name=ticker,
            entity_id=eid,
            metadata=meta,
        )

        tail = div.tail(max_per_ticker)
        for ex_ts, amount in tail.items():
            store.store_entity_observation(
                entity_id=eid,
                source_tool=_SOURCE,
                observed_at=_pandas_ts_to_epoch(ex_ts),
                observation_type="dividend",
                value={
                    "ticker": ticker,
                    "amount": float(amount),
                    "ex_date": str(ex_ts)[:10],
                    "currency": "USD",
                },
                depth_level=2,
            )
            stored += 1
        tickers_ok.append(ticker)
        time.sleep(0.25)

    return {
        "observations_stored": stored,
        "tickers_ok": tickers_ok,
        "tickers_empty": tickers_empty,
    }


class DividendDataTool(Tool):
    def __init__(self, pipeline_store: PipelineStore | None = None) -> None:
        self._store = pipeline_store

    @property
    def name(self) -> str:
        return "dividend_data"

    @property
    def description(self) -> str:
        return "Fetch dividend history for an equity/ETF and persist dividend observations."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "max_rows": {"type": "integer", "default": 24},
            },
            "required": ["ticker"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        ticker = str(kwargs.get("ticker", "SPY"))
        if self._store is None:
            return ToolResult(success=False, output="No pipeline store", data=None)
        out = ingest_dividends(
            self._store,
            [ticker],
            include_should=False,
            max_per_ticker=int(kwargs.get("max_rows", 24)),
        )
        return ToolResult(
            success=out["observations_stored"] > 0,
            output=f"Dividends {ticker}: {out['observations_stored']} rows",
            data=out,
        )


def run_dividend_ingest(
    params: dict[str, Any],
    upstream_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from agent.pipeline.store import PipelineStore

    db_path = params.get("db_path", ".tirra_pipeline/pipeline.db")
    store = PipelineStore(db_path)
    try:
        return ingest_dividends(
            store,
            include_should=bool(params.get("include_should", True)),
        )
    finally:
        store.close()
