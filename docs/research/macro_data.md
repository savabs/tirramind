---
title: "Research: Macro Data / FRED (Phase 1)"
tags:
  - doc/research
  - topic/macro
---

# Research: Macro Data / FRED (Phase 1)

**Date:** 2026-03-28 (retroactive documentation)
**Tool:** `agent/tools/macro_data.py` → `MacroDataTool`
**Status:** IMPLEMENTED, TESTED — **US-CENTRIC, NEEDS EXPANSION**

## APIs Used

### FRED (Federal Reserve Economic Data) ✅ SELECTED
- **URL:** `https://api.stlouisfed.org/fred/series/observations`
- **Method:** GET
- **Auth:** **API key required** — `TIRRA_FRED_API_KEY` env var. Free at fred.stlouisfed.org.
- **Format:** JSON
- **Rate limits:** 120 req/min with key
- **Coverage:** **Primarily US** — 800K+ data series, ~90% US-centric. Some international series exist (exchange rates, global indices).

## Geographic Coverage
- FRED is maintained by the St. Louis Federal Reserve — inherently US-focused
- Core series used: GDP, CPI, Fed Funds Rate, Unemployment, Fed Balance Sheet, Treasury General Account, Reverse Repo, M2
- Some global series available (e.g., DEXJPUS for USD/JPY) but tool doesn't guide users to them
- **Verdict:** `[G:US-ONLY]` `[G:NEEDS-EXPANSION]`

## Implementation Details
- Single mode
- Takes `series_id` (comma-separated for multiple), `start_date`, `end_date`
- Missing values (FRED uses `"."`) filtered out
- Cache key: `macro_data`
- Timeout: 15s

## Signal Value
- Fed balance sheet changes = liquidity regime shifts
- Treasury General Account drains/fills = fiscal stimulus/tightening
- M2 velocity changes = money multiplier signal
- Yield curve inversions (via spread series) = recession predictor

## Global Expansion — International Sources
| Source | Coverage | URL | Auth | Status |
|--------|----------|-----|------|--------|
| ECB Statistical Data Warehouse | EU / Eurozone | `https://data-api.ecb.europa.eu/` | None | **NOT PROBED** |
| BOJ Time Series Search | Japan | `https://www.stat-search.boj.or.jp/ssi/` | None | **NOT PROBED** |
| BOE Interactive Database | UK | `https://www.bankofengland.co.uk/boeapps/database/` | None | **NOT PROBED** |
| BIS Statistics | Global (all central banks) | `https://data.bis.org/` | None | **NOT PROBED** |
| OECD Main Economic Indicators | 38 OECD countries | `https://stats.oecd.org/` | None | **NOT PROBED** |
| RBI Database on Indian Economy | India | `https://dbie.rbi.org.in/` | None | **NOT PROBED** |
| PBOC Statistics | China | `http://www.pbc.gov.cn/` | None | **NOT PROBED** — likely Chinese-only |

## Globalization Priority: MEDIUM
- ECB + BOJ + BIS would cover the most important central banks
- OECD API gives harmonized cross-country data
- Cross-CB relative positioning (BOJ expanding while Fed tightening) = carry trade signal
- Planned as 7b-Z in task file

## Risks
- Only tool requiring an API key (TIRRA_FRED_API_KEY)
- FRED API version changes could break parsing
- International CB APIs have varying data models — normalization required

## Related

- [[project_memory]]
