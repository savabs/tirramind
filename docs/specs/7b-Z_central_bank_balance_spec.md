---
title: "Spec: 7b-Z — Central Bank Balance Sheets Tool"
tags:
  - doc/spec
  - layer/surveillance
  - phase/7b
  - topic/central-bank
---

# Spec: 7b-Z — Central Bank Balance Sheets Tool

## Goal

Provide cross-central-bank balance sheet analytics: global liquidity index, policy divergence detection, and rate monitoring. The unique edge is computing **cross-CB relative positioning** — not just individual series (which macro_data.py already handles).

## Files Affected

| Action | Path |
|--------|------|
| Create | `agent/tools/central_bank_balance.py` |
| Create | `tests/test_central_bank_balance_edge.py` |
| Modify | `agent/cli.py` (register tool) |
| Modify | `agent/learning/bandit.py` (add arm) |
| Create | `[[7b-Z_central_bank_balance]]` |

## Implementation Steps

### 2.1: Create tool skeleton
- Class `CentralBankBalanceTool(Tool)` with name `central_bank_balance`
- Constructor takes `fred_api_key: str`, `cache: DataCache | None`
- Parameters schema with `mode` enum: `balance_sheets`, `liquidity_index`, `policy_divergence`, `rate_monitor`
- Optional params: `period` (1m/3m/6m/1y/2y/5y), `banks` (filter to specific CBs)

### 2.2: Implement CB registry and FRED fetcher
- Define `CB_REGISTRY`: dict mapping CB name → {fred_series, currency, fx_series, rate_series, name}
- Central banks: Fed, ECB, BOJ, BOE, SNB, BOC, RBA (7 banks)
- Implement `_fetch_fred_series(series_id, start_date, end_date)` — reuse FRED API pattern from macro_data.py
- Implement `_fetch_ecb_series(sdmx_key, last_n)` — direct ECB SDW call

### 2.3: Implement mode `balance_sheets`
- Fetch latest balance sheet data for all (or filtered) CBs from FRED
- Fetch FX rates to normalize to USD
- For ECB: prefer ECB SDW direct (more current) with FRED as fallback
- Compute WoW, MoM, YoY percentage changes
- Output: table of CBs with level (native + USD), changes, direction arrows

### 2.4: Implement mode `liquidity_index`
- Fetch time series for all CB assets over specified period
- Convert all to USD using period FX rates
- Compute: Gross = Sum(CB assets in USD)
- Compute: Net = Gross - Fed_RRP(RRPONTSYD) - Fed_TGA(WDTGAL)
- Return time series + current level + trend metrics (WoW/MoM/YoY change)

### 2.5: Implement mode `policy_divergence`
- Compute balance sheet growth rates (3m, 6m, 12m annualized) for each CB
- Classify each: expanding / stable / contracting (thresholds: >2%/yr expanding, <-2%/yr contracting)
- Identify divergent pairs (one expanding, one contracting)
- Fetch policy rates: Fed (DFF), ECB (deposit rate), BOJ rate
- Compute rate differentials between pairs
- Flag synchronized moves (all tightening or all easing)

### 2.6: Implement mode `rate_monitor`
- Fetch current policy rate for each CB
- Detect last change date (scan for level changes in recent history)
- Compute days since last change
- Flag recent changes (< 30 days) as potential market movers

### 2.7: Register tool in cli.py + bandit arm
- Import and register `CentralBankBalanceTool(fred_api_key=fred_key, cache=cache)`
- Add GoalArm `global_liquidity` with tools `["central_bank_balance", "macro_data"]`
- Update tool count: 35→36, arm count: 23→24

### 2.8: Edge case tests
- Full test suite in `tests/test_central_bank_balance_edge.py`
- Mock all HTTP (FRED + ECB SDW)
- Test: missing API key, invalid mode, HTTP errors, empty responses, partial CB data, FX conversion failures, discontinued series, mixed frequencies

## Edge Cases
- One CB's data unavailable → continue with others, note in output
- FX rate fetch fails → exclude that CB from USD normalization, note it
- FRED API key missing → return error for FRED-dependent modes (all of them currently)
- All data stale → warn user about data freshness
- Negative balance sheet change rates > 20% → flag as potential data error

## Testing Plan
- Mock FRED JSON API responses for each series
- Mock ECB SDW JSON responses
- Verify USD normalization math
- Verify liquidity index computation (sum - drains)
- Verify divergence classification logic
- Verify rate change detection
- All edge cases from above

---

## Related

- [[7b-Z_central_bank_balance|Task: 7B-Z Central Bank Balance]]
- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]
