---
title: "Research: Market Data / yfinance (Phase 0)"
tags:
  - doc/research
  - layer/world-model
  - topic/market-data
---

# Research: Market Data / yfinance (Phase 0)

**Date:** 2026-03-28 (retroactive documentation)
**Tool:** `agent/tools/market_data.py` → `MarketDataTool`
**Status:** IMPLEMENTED, TESTED

## APIs Used

### yfinance (Yahoo Finance wrapper) ✅
- **URL:** None direct — uses `yfinance` Python library which wraps Yahoo Finance (IB backend)
- **Auth:** None
- **Format:** pandas DataFrame → dict/JSON
- **Rate limits:** Yahoo Finance has undocumented limits, yfinance handles internally
- **Coverage:** **Global** — all exchanges worldwide (NYSE, NASDAQ, LSE, TSE, HKEX, BSE, Euronext, etc.)

## Geographic Coverage
- Any ticker on any exchange that Yahoo Finance covers
- Stocks, ETFs, indices, futures, forex, crypto
- **Verdict:** `[G:GLOBAL]`

## Implementation Details
- Single mode
- Takes `tickers` (comma-separated), `period` (1d to max), `interval` (1m to 3mo)
- Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
- Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
- Cache key: `market_data`

## Signal Value
- **NOTE:** Per project architecture, market_data is a TARGET variable source, not a predictive feature source
- Used as the dependent variable in backtesting, scoring, and validation
- Price data is Layer 3 (consequence) — used to measure prediction accuracy, never as input features
- Provides OHLCV, volume, dividends, splits

## Role in Architecture
- Layer 3 consequence data — what we predict, not what we use to predict
- Used by: backtest.py, scoring.py, regime.py (as validation target)
- Referenced in copilot-instructions.md: "Prices are needed only as TARGET variables (what we predict), never as input features"

## Risks
- yfinance is unofficial — could break if Yahoo Finance changes their backend
- No SLA, no support
- Occasional data gaps for non-US tickers
- Rate limiting behavior is opaque

## Related

- [[project_memory]]
