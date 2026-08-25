---
title: "Batch 8: Labor Disruptions, Migration Flows, Energy Supply — Research"
tags:
  - doc/research
  - layer/surveillance
  - phase/7b
---

# Batch 8: Labor Disruptions, Migration Flows, Energy Supply — Research

**Date:** 2026-04-01
**Tools:** 7b-AJ (Labor Disruptions), 7b-AK (Migration & Refugee Flows), 7b-AL (Energy Supply Side)
**Status:** Research complete, implementation in progress

## Goal

Build three surveillance tools covering:
1. US labor disruptions (strikes, lockouts) — physical production halts
2. Global migration/refugee flows — physical human displacement
3. Energy supply side (petroleum stocks, rig counts) — physical energy production

All three are L0-L1 cause data: physical reality, not opinions or derived signals.

## Sources Probed (2026-04-01)

### 7b-AJ: Labor Disruptions & Strike Activity

#### BLS Public Data API v2 — ★★★★★ PRIMARY SOURCE

**URL:** `https://api.bls.gov/publicAPI/v2/timeseries/data/`
**Method:** POST with JSON body `{"seriesid": [...], "startyear": "YYYY", "endyear": "YYYY"}`
**Auth:** None required (public tier: 25 queries/day, no registration needed)
**Rate limit:** 25 requests/day (unregistered), 500/day with free API key

**Series confirmed working (2026-04-01):**

| Series ID | Description | Records (2022-2025) | Status |
|-----------|-------------|---------------------|--------|
| `WSU001` | Workers involved in major stoppages (thousands) | 48 monthly records | ✅ Active, current through Feb 2026 |
| `WSU002` | Days idle during month (thousands) | 28 records | ✅ Active, current through Apr 2024 |
| `WSU003` | — | 0 records | ❌ Empty |
| `WSU004` | — | 0 records | ❌ Empty |
| `WSU005` | — | 0 records | ❌ Empty |

**Response schema:**
```json
{
  "status": "REQUEST_SUCCEEDED",
  "Results": {
    "series": [{
      "seriesID": "WSU001",
      "data": [{
        "year": "2026",
        "period": "M02",
        "periodName": "February",
        "latest": "true",
        "value": "674.6",
        "footnotes": [{"code": "P", "text": "Preliminary"}]
      }]
    }]
  }
}
```

**Key observations:**
- Data arrives newest-first; must sort chronologically for trend analysis
- Footnote code "P" = preliminary data (may be revised)
- Values are in thousands (WSU001: thousands of workers; WSU002: thousands of days)
- Coverage: US only, major stoppages (1,000+ workers)
- Historical data available back to ~1993

#### NLRB — ❌ DEAD (timed out)
- URL `https://www.nlrb.gov/api/search` timed out after 10s
- Not viable as a primary source

#### Cornell ILR Labor Action Tracker — ❌ DEAD (404 GitHub Pages)
- URL `https://striketracker.ilr.cornell.edu/api/v1/stoppages` returned 404
- Appears to be a static site, no REST API

#### FRED — ⚠️ POSSIBLE SUPPLEMENT
- FRED requires a registered API key (DEMO_KEY rejected with 400)
- Already wired into the project via `TIRRA_FRED_API_KEY`
- Could add FRED series as supplement later; not needed for initial build

**Decision:** BLS API is the sole primary source. WSU001 (workers) and WSU002 (idle days) cover the core signals. No auth needed.

---

### 7b-AK: Migration & Refugee Flows

#### UNHCR Refugee Statistics API v1 — ★★★★★ PRIMARY SOURCE

**Base URL:** `https://api.unhcr.org/population/v1/`
**Auth:** None required
**Rate limit:** Not documented; appears generous

**Endpoints confirmed working (2026-04-01):**

| Endpoint | Description | Response |
|----------|-------------|----------|
| `/population/` | Refugee/IDP/stateless populations by country/year | ✅ 200, JSON, paginated |
| `/demographics/` | Age/sex breakdown of displaced populations | ✅ 200, JSON |
| `/asylum-decisions/` | Asylum decision outcomes by country | ✅ 200, JSON, paginated (5 pages for 2023) |

**Query parameters:** `year`, `coa` (country of asylum, ISO3), `coo` (country of origin, ISO3), `limit`, `page`

**Response schema (population endpoint):**
```json
{
  "page": 1,
  "maxPages": 1,
  "items": [{
    "year": 2023,
    "coo_id": "-", "coo_name": "-", "coo": "-", "coo_iso": "-",
    "coa_id": 196, "coa_name": "Türkiye", "coa": "TUR", "coa_iso": "TUR",
    "refugees": 3251127,
    "asylum_seekers": 222069,
    "returned_refugees": 21865,
    "idps": "0",
    "returned_idps": "0",
    "stateless": 415,
    "ooc": "0",
    "oip": "-",
    "hst": "0"
  }]
}
```

**Key observations:**
- Fields may be integers OR string "0" or "-" — must handle mixed types
- `coo` = country of origin (3-letter), `coa` = country of asylum (3-letter)
- Global aggregates returned when no country filter (all countries summed)
- Categories: refugees, asylum_seekers, returned_refugees, idps (internally displaced), returned_idps, stateless, ooc (others of concern), oip (other people in need of intl protection), hst (host community)
- Data annual, typically available with ~6-12 month lag

#### World Bank Remittances — ★★★★ SECONDARY SOURCE

**URL:** `https://api.worldbank.org/v2/country/{iso2}/indicator/BX.TRF.PWKR.CD.DT`
**Auth:** None required
**Already used:** Same API pattern as food_security.py (World Bank Open Data)

**Indicator:** `BX.TRF.PWKR.CD.DT` — Personal remittances received (current US$)

**Confirmed working:** Philippines 2018-2023, values like $39.1B (2023). Annual data.

**Key observations:**
- Remittance flows track diaspora economics and labor migration
- Sudden drops = sanctions/banking disruption in corridor
- Surges = crisis-driven displacement or new migration wave
- Standard World Bank JSON response `[metadata, data_array]`

#### UNHCR Operational Data Portal (data.unhcr.org) — ❌ DEAD (404 with obfuscated JS)
- URL `https://data.unhcr.org/api/population` returned 404 with antibot JS
- Not a usable REST API

#### IDMC (Internal Displacement Monitoring Centre) — ❌ DEAD (404)
- URL `https://api.internal-displacement.org/data` returned 404 HTML page
- No usable REST API found

**Decision:** UNHCR Refugee Statistics API (population + asylum-decisions) as primary. World Bank remittances as secondary for economic impact. Both free, no auth.

---

### 7b-AL: Energy Supply Side

#### EIA API v2 — ★★★★★ PRIMARY SOURCE (all three modes)

**Base URL:** `https://api.eia.gov/v2/`
**Auth:** `DEMO_KEY` accepted (or `TIRRA_EIA_API_KEY` env var)
**Rate limit:** 1000 requests/hour with registered key; DEMO_KEY more limited

**Endpoints confirmed working (2026-04-01):**

| Endpoint | Description | Status |
|----------|-------------|--------|
| `petroleum/sum/sndw/data/` | Weekly petroleum supply & disposition | ✅ 200, 2269 total records |
| `petroleum/stoc/wstk/data/` | Weekly petroleum stocks | ✅ 200, 265730 total records |
| `natural-gas/enr/drill/data/` | Monthly rig counts (rotary rigs) | ✅ 200, 3822 total records |
| `drilling/data/` | Drilling productivity | ❌ 404 |

**Query parameters (EIA v2 standard):**
- `api_key` — required
- `frequency` — `weekly` or `monthly`
- `data[0]` — `value`
- `facets[series][]` — filter by series code
- `length` — number of records to return
- `sort[0][column]` / `sort[0][direction]` — sorting
- `start` / `end` — date range filters

**Key series codes confirmed:**

| Series | Description | Units | Frequency |
|--------|-------------|-------|-----------|
| `WCESTUS1` | US crude oil ending stocks excl. SPR | Thousand barrels (MBBL) | Weekly |
| `WGTSTUS1` | US total motor gasoline stocks | MBBL | Weekly |
| `WDISTUS1` | US distillate fuel oil stocks | MBBL | Weekly |
| `WPRSTUS1` | US strategic petroleum reserve | MBBL | Weekly |
| `E_ERTRR0_XR0_NUS_M` | US rotary rigs in operation | Count | Monthly |

**Response schema (petroleum):**
```json
{
  "response": {
    "total": "2269",
    "frequency": "weekly",
    "data": [{
      "period": "2026-03-20",
      "duoarea": "NUS",
      "area-name": "U.S.",
      "product": "EPC0",
      "product-name": "Crude Oil",
      "process": "SAX",
      "process-name": "Ending Stocks Excluding SPR",
      "series": "WCESTUS1",
      "series-description": "U.S. Ending Stocks excluding SPR of Crude Oil (Thousand Barrels)",
      "value": "456185",
      "units": "MBBL"
    }]
  }
}
```

#### NRC Power Reactor Status — ❌ DEAD
- `powerreactorstatus.txt` returned 404
- `PowerReactorStatusReport.txt` returned 404
- `power-reactor-status.html` timed out
- NRC website seems to have restructured; not viable without HTML scraping

**Decision:** EIA API v2 as sole source. Three modes: petroleum_stocks (weekly crude/gasoline/distillate), rig_count (monthly rotary rigs), petroleum_supply (weekly supply & disposition). Uses DEMO_KEY fallback or TIRRA_EIA_API_KEY env var.

---

## Current Architecture

- **Tool pattern:** `Tool(ABC)` from `agent/tools/base.py`, `ToolResult(success, output, data)`
- **Registration:** Import in `agent/cli.py`, `registry.register(XxxTool(cache=cache))`
- **Bandit:** `GoalArm` in `agent/learning/bandit.py` DEFAULT_ARMS
- **Cache:** `DataCache` with configurable TTL per tool
- **Recent references:** `food_security.py` (World Bank pattern), `political_risk.py` (API key pattern), `internet_outages.py` (multi-source pattern)

## Risks

- **BLS rate limit:** 25 req/day without key. Overview mode fetches 2 series = 2 requests. Cache TTL of 6hr mitigates.
- **UNHCR mixed types:** Fields can be int, string "0", or string "-". Must handle all gracefully.
- **EIA DEMO_KEY:** Works but may have stricter rate limits. Real key via env var is preferred.
- **NRC dead:** Nuclear reactor status not available via API. Could add later if NRC restores endpoint.

## Implementation Intent

### 7b-AJ: Labor Disruptions
- **File:** `agent/tools/labor_disruptions.py`
- **Modes:** `work_stoppages` (WSU001), `idle_days` (WSU002), `overview` (both + derived signals)
- **Signals:** alert thresholds, trend (6mo vs prior 6mo), intensity ratio, consecutive active months
- **Cache TTL:** 6 hours

### 7b-AK: Migration & Refugee Flows
- **File:** `agent/tools/migration_flows.py`
- **Modes:** `displacement` (UNHCR population), `asylum` (UNHCR asylum-decisions), `remittances` (World Bank)
- **Signals:** displacement velocity (YoY change), refugee concentration, asylum acceptance rate, remittance anomaly
- **Cache TTL:** 12 hours (annual data)

### 7b-AL: Energy Supply Side
- **File:** `agent/tools/energy_supply.py`
- **Modes:** `petroleum_stocks` (weekly crude/gasoline/distillate), `rig_count` (monthly), `petroleum_supply` (weekly S&D)
- **Signals:** inventory change (week-over-week), rig count trend, stock vs 5-year average proxy
- **Cache TTL:** 2 hours (weekly data updates Wednesdays)

## Related

- [[project_memory]]
