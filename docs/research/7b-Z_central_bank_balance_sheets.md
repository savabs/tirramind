---
title: "Feature: 7b-Z — Global Central Bank Balance Sheets & Rate Decisions"
tags:
  - doc/research
  - layer/surveillance
  - phase/7b
  - topic/central-bank
---

# Feature: 7b-Z — Global Central Bank Balance Sheets & Rate Decisions

## Current Architecture

- `macro_data.py` already fetches individual FRED series (WALCL), ECB balance sheet, World Bank data
- The new tool's value is **cross-CB derived analytics** — not raw data fetching
- Existing cache/tool pattern in `sovereign_debt.py`, `macro_data.py`

## Data Sources (All Free)

### FRED API (requires API key, existing infra)
| Series | Central Bank | Frequency | Unit | Description |
|--------|-------------|-----------|------|-------------|
| WALCL | Fed | Weekly (Wed) | Millions USD | Fed Total Assets |
| WTREGEN | Fed | Weekly | Millions USD | Fed Treasury Securities Held |
| WSHOMCB | Fed | Weekly | Millions USD | Fed MBS Holdings |
| RRPONTSYD | Fed | Daily | Billions USD | Reverse Repo (liability = drains liquidity) |
| WDTGAL | Fed | Weekly | Millions USD | Treasury General Account (drains liquidity) |
| JPNASSETS | BOJ | Monthly | 100M Yen | BOJ Total Assets |
| ECBASSETSW | ECB | Weekly | Millions EUR | ECB Total Assets (FRED mirror) |
| GBCBBS | BOE | Monthly (disc.) | Millions GBP | BOE Total Assets (may be discontinued) |
| SNBASSETSM | SNB | Monthly | Millions CHF | Swiss National Bank Total Assets |
| BCBASSETSM | BOC | Monthly | Millions CAD | Bank of Canada Total Assets |
| RBASSETSM | RBA | Monthly | Millions AUD | Reserve Bank of Australia Total Assets |

### ECB Statistical Data Warehouse (free, no key — CONFIRMED WORKING)
- `ILM/W.U2.C.T000000.Z5.Z01` — ECB Total Assets (weekly, EUR millions)
- `FM/B.U2.EUR.4F.KR.DFR.LEV` — ECB Deposit Facility Rate
- `FM/B.U2.EUR.4F.KR.MFI.NWT` — ECB Main Refinancing Rate
- Returns SDMX JSON, well-structured

### Rate Decision Sources
- FRED: DFF (Fed Funds effective), DFEDTAR/DFEDTARU/DFEDTARL (target range)
- ECB SDW: deposit rate, main refi rate
- BOJ: FRED series or BOJ website (harder to parse)
- These can be tracked for changes → "surprise" detection

## Tool Design: 4 Modes

### Mode 1: `balance_sheets` — Cross-CB Balance Sheet Snapshot
- Fetch latest data for all major CBs (Fed, ECB, BOJ, BOE, SNB, BOC, RBA)
- Normalize all to USD using latest FX rates
- Show absolute levels + percentage change (WoW, MoM, YoY)
- Highlight who's expanding vs contracting

### Mode 2: `liquidity_index` — Global Liquidity Index
- Sum of major CB balance sheets (normalized to USD)
- Subtract Fed RRP (RRPONTSYD) and TGA (WDTGAL) — these drain liquidity
- Net Global Liquidity = Sum(CB assets) - Fed_RRP - Fed_TGA
- Time series: show trend over user-specified period
- WoW/MoM/YoY change rates
- This is the metric that drives crypto/risk assets

### Mode 3: `policy_divergence` — Relative Policy Positioning
- Compare balance sheet growth rates across CBs
- Identify divergence: e.g., "BOJ expanding while Fed contracting"
- Rate differentials: Fed funds vs ECB deposit rate vs BOJ rate
- Carry trade signals: which CB pairs have widening/narrowing differentials
- Synchronized tightening/easing detection

### Mode 4: `rate_monitor` — Rate Decision Tracking
- Current policy rates for all major CBs
- Last change date and direction for each
- Rate change history (last N changes per CB)
- Highlight recent changes (within last 30 days) as potential surprises

## Observations

- FRED mirrors most CB data so we can use a single API for most balance sheets
- ECB SDW provides higher-frequency ECB data than FRED's mirror
- BOJ data on FRED (JPNASSETS) is monthly with ~1 month lag — acceptable
- PBOC does not publish easily parseable balance sheet data; skip for v1
- RBI weekly supplement is PDF-based; skip for v1
- Net liquidity (assets minus RRP/TGA) is the actual market-moving metric

## Risks

- FRED series for some CBs may be discontinued (GBCBBS has been spotty)
- FX conversion introduces noise — use weekly average, not spot
- BOJ data is monthly while Fed/ECB are weekly → mixed frequencies
- API key dependency for FRED (but already established in project)

## Edge Cases
- Missing/delayed data for one CB shouldn't break the whole tool
- Handle discontinued FRED series gracefully
- FX rate unavailable → skip that CB from USD normalization
- Empty observation windows → meaningful error message

---

## Related

- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]
