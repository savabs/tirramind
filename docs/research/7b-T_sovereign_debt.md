---
title: "Feature: 7b-T Government Bond / Sovereign Debt Markets"
tags:
  - doc/research
  - layer/surveillance
  - phase/7b
  - topic/sovereign-debt
---

# Feature: 7b-T Government Bond / Sovereign Debt Markets

## Current Architecture
- Tool base: `agent/tools/base.py` → `Tool` ABC with `execute(**kwargs) → ToolResult`
- Data cache: `agent/data/cache.py` → `DataCache.get(source, params)` / `put(source, params, data, ttl)`
- Registration: `agent/cli.py` → `build_tool_registry()`, import + `registry.register()`
- Bandit arm: `agent/learning/bandit.py` → `GoalArm` dataclass, `DEFAULT_ARMS` list
- HTTP: `httpx` (sync), standard `_UA`, `_TIMEOUT` pattern
- Current count: 34 tools, 22 arms

## Signal Theory
Bond markets are smarter than equity markets. They price fiscal stress months/years before equity investors notice. Detroit munis screamed 18 months before bankruptcy filled. Greek 10Y yields blew out months before the equity crash.

Key signals:
- **Yield curve shape** — flattening/inversion → recession signal (2s10s, 3m10y)
- **Cross-country spreads** — IT-DE, GR-DE widening = eurozone fiscal stress
- **Yield level changes** — sudden moves = market stress or central bank surprise
- **Term premium** — long-end vs short-end decomposition
- **Japan yield** — BOJ yield curve control makes JGB breakouts extremely significant

## API Endpoints — Confirmed Working

### 1. US Treasury Daily Yield Curve (FREE, no auth)
- **URL:** `https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value_month={YYYYMM}`
- **Format:** Atom XML feed with `<m:properties>` entries
- **Maturities:** BC_1MONTH, BC_2MONTH, BC_3MONTH, BC_4MONTH, BC_6MONTH, BC_1YEAR, BC_2YEAR, BC_3YEAR, BC_5YEAR, BC_7YEAR, BC_10YEAR, BC_20YEAR, BC_30YEAR
- **Date field:** `d:NEW_DATE`
- **Frequency:** Daily (business days)
- **Verified:** 20 entries for March 2026, latest 2026-03-27: 2Y=3.88%, 10Y=4.44%, 30Y=4.98%

### 2. ECB IRS — Per-Country Government Bond Yields (FREE, no auth)
- **URL:** `https://data-api.ecb.europa.eu/service/data/IRS/M.{CC}.L.L40.CI.0000.EUR.N.Z?startPeriod={YYYY-MM}&format=csvdata`
- **Format:** CSV with KEY, FREQ, REF_AREA, ..., TIME_PERIOD, OBS_VALUE
- **Countries (24):** AT, BE, BG, CY, DE, EE, ES, FI, FR, GR, HR, IE, IT, LT, LU, LV, MT, NL, PT, SI, SK (+ I9, I10, U2 = aggregates)
- **Frequency:** Monthly
- **Data:** Long-term (close to 10Y) government bond yield
- **Verified:** DE=2.744%, ES=3.174%, FR=3.4%, GR=3.39%, IT=3.388% (Feb 2026)
- **Key for spreads:** IT-DE, GR-DE, ES-DE = fiscal stress indicators

### 3. ECB FM — Euro Area Aggregate Benchmark Yields (FREE, no auth)
- **URL:** `https://data-api.ecb.europa.eu/service/data/FM/M.U2.EUR.4F.BB.U2_{tenor}Y.YLD?startPeriod={YYYY-MM}&format=csvdata`
- **Tenors:** 2Y, 3Y, 5Y, 7Y, 10Y
- **Frequency:** Monthly
- **Use:** Euro area yield curve shape (aggregate)

### 4. Japan MOF — JGB Yield Curve (FREE, no auth)
- **Current month:** `https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/jgbcme.csv`
- **Full history:** `https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/historical/jgbcme_all.csv`
- **Format:** CSV with header row 2: `Date,1Y,2Y,3Y,4Y,5Y,6Y,7Y,8Y,9Y,10Y,15Y,20Y,25Y,30Y,40Y`
- **Date format:** `YYYY/M/D`
- **Frequency:** Daily
- **History:** 13,168 rows from 1974 to present
- **Verified:** 2026-03-26: 1Y=1.084, 10Y=2.286, 30Y=3.489, 40Y=3.556

### 5. UK DMO — Gilt Issuance History (FREE, no auth)
- **URL:** `https://www.dmo.gov.uk/data/XmlDataReport?reportCode=D2.1E`
- **Format:** XML with `<View_Gilt_Issuance_History>` elements
- **Fields:** INSTRUMENT_NAME, ISIN_CODE, ACTUAL_DATE, ISSUANCE_TYPE, NOMINAL_ISSUED, ISSUE_CLEAN_PRICE, ISSUE_YIELD, INDEXATION_LAG
- **Use:** Auction pricing signals; new issuance yield levels

### 6. UK DMO — Gilts in Issue (FREE, no auth)
- **URL:** `https://www.dmo.gov.uk/data/XmlDataReport?reportCode=D1A`
- **Format:** XML with `<View_GILTS_IN_ISSUE>` elements
- **Fields:** ISIN, redemption date, coupon, amount in issue
- **Use:** Supply data — total outstanding by maturity

## Dead Endpoints
- **EMMA/MSRB** — TOS explicitly prohibits automated access/scraping
- **Japan MOF auction_result.htm** — 404
- **ECB FM per-country** (FM.M.{CC}...) — 404 for individual countries, only U2 aggregate
- **UK DMO D4H export** — Returns HTML page, not CSV

## Risks
- **US Treasury XML namespace:** Atom XML with `d:` and `m:` prefixes — need careful XML parsing
- **Japan MOF encoding:** CSV has Japanese UTF-8 characters in footer note
- **ECB rate limiting:** No documented rate limits, but be polite (0.5s between requests)
- **UK DMO availability:** XML endpoint may go down; no documented SLA
- **Missing entries:** Some dates have `-` for certain maturities (Japan pre-1980s, newer tenors)

## Tool Design
- **Modes:** `us_yields`, `eu_yields`, `jp_yields`, `uk_gilts`, `spreads`
- `us_yields` — US Treasury daily yield curve for a given month
- `eu_yields` — ECB per-country government bond yields (multiple countries in one call)
- `jp_yields` — Japan MOF daily JGB yields
- `uk_gilts` — UK DMO gilt issuance data
- `spreads` — Computed cross-country spreads (IT-DE, GR-DE, ES-DE, etc.)
- **Bandit arm:** `sovereign_stress` — tools: `sovereign_debt`, `macro_data`, `market_data`

---

## Related

- [[7b-T_sovereign_debt_spec|Spec: 7B-T Sovereign Debt]]
- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]
