---
title: "Spec: finra_short_volume"
tags:
  - doc/spec
  - topic/finra
---

# Spec: finra_short_volume

## Goal

Build a tool that fetches FINRA Reg SHO daily short volume and consolidated short interest data. Two modes: `short_volume` (daily, near-real-time) and `short_interest` (bi-monthly, accumulated positions). Expose cross-signal metadata for downstream fusion with insider filings and CFTC positioning.

## Files Affected

| File | Action |
|------|--------|
| `agent/tools/finra_short_volume.py` | CREATE — FinraShortVolumeTool |
| `agent/cli.py` | MODIFY — import + register |
| `agent/learning/bandit.py` | MODIFY — add `institutional_flow` arm |
| `tests/test_finra_short_volume_edge.py` | CREATE — edge case tests |

## Implementation Steps

### Step 1: Skeleton + Parameter Schema

Create `agent/tools/finra_short_volume.py` with:
- `FinraShortVolumeTool(Tool)` class
- `name = "finra_short_volume"`
- Parameters:
  - `mode`: enum `"short_volume"` | `"short_interest"` (required)
  - `ticker`: string, optional (if omitted in short_volume mode, return top anomalies)
  - `date`: string YYYY-MM-DD, optional (default: most recent trading day)
  - `days_back`: int 1-20, optional, default 5 (for short_volume multi-day trend)
  - `min_total_volume`: int, optional, default 100000 (filter noise in scan mode)
  - `limit`: int, optional, default 20 (top N results in scan mode)

### Step 2: Implement `_fetch_short_volume()`

- POST to `https://api.finra.org/data/group/otcMarket/name/regShoDaily`
- Required compareFilter: `tradeReportDate` = date, `securitiesInformationProcessorSymbolIdentifier` = ticker
- Fields: `securitiesInformationProcessorSymbolIdentifier`, `totalParQuantity`, `shortParQuantity`, `shortExemptParQuantity`, `reportingFacilityCode`
- For single-ticker: one request, aggregate across 3 facilities
- For scan mode (no ticker): pagination via offset (5000 per page), aggregate all facilities per ticker, return sorted by short ratio
- Cache: 24hr TTL per date+ticker key
- Handle: 200 (data), 204 (no data), 400 (bad request), 429 (rate limit), 500+ (server error)

### Step 3: Implement `_fetch_short_interest()`

- POST to `https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest`
- Required compareFilter: `symbolCode` = ticker, `settlementDate` = date
- Fields: `symbolCode`, `settlementDate`, `currentShortPositionQuantity`, `previousShortPositionQuantity`, `changePercent`, `daysToCoverQuantity`, `averageDailyVolumeQuantity`, `marketClassCode`
- Settlement dates: mid-month (15th) and end-of-month (~28-31). Search backwards from target date across recent settlement dates.
- Cache: 7-day TTL
- Handle: 200 (data), 204 (no data), 400 (bad params)

### Step 4: Implement `_compute_signals()`

For short_volume mode with days_back > 1:
- Fetch each trading day's data (skip weekends/holidays via 204 detection)
- Compute per-ticker: `short_ratio_current`, `short_ratio_avg`, `short_ratio_zscore`, `trend_direction` (rising/falling/flat)
- `is_anomaly` flag: z-score > 1.5 or < -1.5
- For scan mode: rank by |z-score| descending

For short_interest mode:
- `squeeze_risk`: days_to_cover > 5.0
- `building_short`: changePercent > 15%
- `covering`: changePercent < -15%

### Step 5: Implement `execute()`

Full pipeline:
1. Validate params (mode, date format, ticker format)
2. Route to `_fetch_short_volume()` or `_fetch_short_interest()`
3. Apply `_compute_signals()` if multi-day data available
4. Format output with key metrics + signal flags

### Step 6: Register in CLI + Bandit

- `agent/cli.py`: Add `from agent.tools.finra_short_volume import FinraShortVolumeTool`, register with cache
- `agent/learning/bandit.py`: Add `institutional_flow` arm with tools `["finra_short_volume", "market_data", "cftc"]`

### Step 7: Live Test

- Single-ticker short_volume: NVDA, TSLA, AAPL for today
- Scan mode: top 10 short ratio anomalies
- Short interest: AAPL for most recent settlement date
- Verify output format, signal flags, cache behavior

### Step 8: Edge Case Tests

- Input validation: invalid mode, bad ticker format, future date, invalid days_back
- Short volume: normal response, empty/204, multi-facility aggregation, scan with pagination mock, scan with min_volume filter
- Short interest: normal response, no settlement date found, squeeze risk flag, covering flag
- Signal computation: z-score calculation, trend direction, anomaly flagging
- Error handling: 400, 429, 500, timeout, malformed JSON
- Integration: CLI registration, bandit arm, openai_tool schema

## Edge Cases

1. Weekend/holiday dates → 204, skip gracefully
2. Ticker not found → empty result, not error
3. Fractional par quantities (e.g., 109,953.7638) → the API returns floats, not ints
4. Zero total volume → avoid division by zero in ratio calc
5. Short interest settlement date not found for recent dates → search backwards
6. API returns > 5000 records in scan → paginate up to 30,000 max (6 pages)
7. All facilities return 0 for a ticker → report but flag as "no off-exchange volume"
8. ETF vs equity short ratio baselines differ (ETFs structurally higher ~50-65%)

## Testing Plan

Unit tests (mocked): ~50+ tests covering all modes, edge cases, signal computation
Live tests: 3-4 tests behind TIRRA_LIVE_TESTS=1 flag
Integration tests: CLI registration, bandit arm, tool schema

## Related

- [[finra_short_data|Research: FINRA Short Data]]
