"""
Tool: Market Data

Fetches historical OHLCV (Open, High, Low, Close, Volume) market data
using yfinance. Supports single or multiple tickers, configurable
period and interval.
"""

from __future__ import annotations

import logging
from typing import Any

import yfinance as yf

from agent.data.cache import DataCache
from agent.tools.base import Tool, ToolResult

log = logging.getLogger(__name__)


class MarketDataTool(Tool):
    name = "market_data"
    description = (
        "Fetch historical OHLCV price data for one or more stock/ETF/index tickers. "
        "Returns daily bars by default. Use this to get price history, compare assets, "
        "or gather data for quantitative analysis."
    )
    parameters = {
        "type": "object",
        "properties": {
            "tickers": {
                "type": "string",
                "description": (
                    "Ticker symbol(s). Single ticker like 'AAPL' or comma-separated "
                    "like 'AAPL,MSFT,GOOGL'."
                ),
            },
            "period": {
                "type": "string",
                "description": (
                    "How far back to fetch. Options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, "
                    "5y, 10y, ytd, max. Default: 1mo."
                ),
                "default": "1mo",
            },
            "interval": {
                "type": "string",
                "description": (
                    "Bar size. Options: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, "
                    "5d, 1wk, 1mo, 3mo. Default: 1d. "
                    "Intraday intervals only available for recent periods."
                ),
                "default": "1d",
            },
        },
        "required": ["tickers"],
    }

    _VALID_PERIODS = {
        "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max",
    }
    _VALID_INTERVALS = {
        "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h",
        "1d", "5d", "1wk", "1mo", "3mo",
    }

    def __init__(self, cache: DataCache | None = None) -> None:
        self._cache = cache

    def execute(
        self,
        *,
        tickers: str,
        period: str = "1mo",
        interval: str = "1d",
        **_: Any,
    ) -> ToolResult:
        # Validate inputs
        if period not in self._VALID_PERIODS:
            return ToolResult(
                success=False,
                output=f"Invalid period '{period}'. Must be one of: {', '.join(sorted(self._VALID_PERIODS))}",
            )
        if interval not in self._VALID_INTERVALS:
            return ToolResult(
                success=False,
                output=f"Invalid interval '{interval}'. Must be one of: {', '.join(sorted(self._VALID_INTERVALS))}",
            )

        # Parse ticker list
        ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        if not ticker_list:
            return ToolResult(success=False, output="No valid tickers provided.")

        try:
            results: list[str] = []
            all_data: dict[str, Any] = {}

            for ticker in ticker_list:
                cache_params = {"ticker": ticker, "period": period, "interval": interval}
                cached = self._cache.get("market_data", cache_params) if self._cache else None

                if cached is not None:
                    log.debug("Cache hit for %s", ticker)
                    results.append(f"[{ticker}] (cached) {cached['summary']}")
                    all_data[ticker] = cached["data"]
                    continue

                data = self._fetch_single(ticker, period, interval)
                if data is None:
                    results.append(f"[{ticker}] No data returned (invalid ticker or no history).")
                    continue

                rows = len(data)
                first = data.iloc[0]
                last = data.iloc[-1]
                summary = (
                    f"{rows} bars ({period}, {interval})\n"
                    f"  First: {data.index[0]} — O:{first['Open']:.2f} H:{first['High']:.2f} "
                    f"L:{first['Low']:.2f} C:{first['Close']:.2f} V:{int(first['Volume'])}\n"
                    f"  Last:  {data.index[-1]} — O:{last['Open']:.2f} H:{last['High']:.2f} "
                    f"L:{last['Low']:.2f} C:{last['Close']:.2f} V:{int(last['Volume'])}"
                )
                results.append(f"[{ticker}] {summary}")
                # Convert index to strings for JSON serialization
                data_dict = data.reset_index().to_dict(orient="records")
                all_data[ticker] = data_dict

                if self._cache:
                    self._cache.put("market_data", cache_params, {"summary": summary, "data": data_dict})

            output = "\n\n".join(results) if results else "No data returned for any ticker."
            return ToolResult(success=True, output=output, data=all_data)

        except Exception as exc:
            log.exception("Market data fetch failed")
            return ToolResult(success=False, output=f"Market data error: {exc}")

    def _fetch_single(self, ticker: str, period: str, interval: str):
        """Fetch OHLCV for a single ticker. Returns DataFrame or None."""
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval)
        if df is None or df.empty:
            return None
        return df
