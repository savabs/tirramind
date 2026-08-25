---
title: "Data Tool Audit: Research & Globalization Status"
tags:
  - doc/research
---

# Data Tool Audit: Research & Globalization Status

**Date:** 2026-03-28
**Purpose:** Track whether each data tool was properly researched before coding, and whether its geographic coverage is global or needs expansion. This is the source of truth for research/globalization compliance.

**Tags:**
- `[R:FULL]` — Dedicated research doc exists in `docs/research/`
- `[R:IMPLICIT]` — Research documented only in task file notes, no standalone doc
- `[R:NONE]` — No documented research found
- `[G:GLOBAL]` — Data source is inherently global or covers all major economies
- `[G:US-ONLY]` — Currently US-only, needs globalization
- `[G:REGIONAL]` — Covers specific region(s), may or may not need expansion
- `[G:NEEDS-EXPANSION]` — Could be global but currently region-locked by implementation choice
- `[G:INHERENT]` — Data source is region-locked by nature (e.g., CFTC = US futures regulator)

---

## Tool Registry (30 tools, 20 data tools)

### 1. academic_preprints.py — AcademicPreprintsTool
- **Tags:** `[R:IMPLICIT]` `[G:GLOBAL]`
- **APIs:** arXiv (export.arxiv.org, Atom XML, no auth), ClinicalTrials.gov v2 (JSON, no auth)
- **Coverage:** Global — arXiv covers all fields worldwide; ClinicalTrials.gov indexes 220+ countries
- **Research doc:** None standalone — documented in task file 7b-M notes
- **Tests:** test_academic_preprints_edge.py (55+ tests)
- **Action needed:** Write standalone research doc

### 2. ais_vessel.py — AISVesselTool
- **Tags:** `[R:FULL]` `[G:REGIONAL]`
- **APIs:** Finland Digitraffic (meri.digitraffic.fi, REST, no auth)
- **Coverage:** Baltic Sea + approaches (Lat 54.9–65.2°N, Lon 11.5–37.5°E). Finland, Sweden, Estonia, Latvia, Lithuania, Poland, Denmark, Norway, NW Russia. BUT captures global destination intent (Suez, Russia routing).
- **Research doc:** `[[7b-D_ais_vessel_tracking]]`
- **Tests:** test_ais_vessel_edge.py (89 tests)
- **Action needed:** None — regional by sensor placement, but shows global trade routing. Could add MarineTraffic/AISHub if paid tier opens.

### 3. cftc.py — CFTCTool
- **Tags:** `[R:FULL]` `[G:INHERENT]`
- **APIs:** CFTC.gov (f_disagg.txt flat file + historical ZIPs, no auth)
- **Coverage:** US-only — CFTC regulates US futures exchanges
- **Research doc:** `[[cftc]]`
- **Tests:** Phase 6 pipeline tests (65 tests)
- **Action needed:** Region-locked by nature (US regulatory body). Global expansion = separate tools for ICE London, Eurex, SGX, TOCOM, MCX, B3. Not a globalization of this tool — it's new tools per exchange.

### 4. defi_flows.py — DefiFlowsTool
- **Tags:** `[R:IMPLICIT]` `[G:GLOBAL]`
- **APIs:** DefiLlama (api.llama.fi — protocols, stablecoins, dexs; no auth)
- **Coverage:** Global — blockchain is borderless. 7000+ protocols, 350+ stablecoins, 1000+ DEXes.
- **Research doc:** None standalone — documented in task file 7b-L notes
- **Tests:** test_defi_flows_edge.py (70+ tests)
- **Action needed:** Write standalone research doc

### 5. earthquake_proximity.py — EarthquakeProximityTool
- **Tags:** `[R:IMPLICIT]` `[G:GLOBAL]`
- **APIs:** USGS Earthquake Hazards (earthquake.usgs.gov, GeoJSON, no auth)
- **Coverage:** Global — USGS monitors worldwide. 19 infrastructure zones across 8 sectors spanning all continents (Taiwan, Chile, Japan, US, Indonesia, NZ, etc.)
- **Research doc:** `docs/research/earthquake_proximity.py` exists (misnamed — should be .md)
- **Tests:** test_earthquake_proximity_edge.py (83 tests)
- **Action needed:** Clean up research doc filename (.py → .md)

### 6. finra_short_volume.py — FINRAShortVolumeTool
- **Tags:** `[R:FULL]` `[G:INHERENT]`
- **APIs:** FINRA (api.finra.org — Reg SHO Daily + Consolidated Short Interest, no auth)
- **Coverage:** US-only — FINRA regulates US equities/OTC
- **Research doc:** `[[finra_short_data]]`
- **Tests:** test_finra_short_volume_edge.py (97 tests)
- **Action needed:** Region-locked by nature (US regulatory body). Global expansion = separate tools per exchange.
- **International alternatives probed (2026-03-28):** See `[[international_api_alternatives]]`
  - ⚠️ **HKEX** — 200 but 1MB HTML (needs scraping)
  - ❌ ESMA Short Selling Register — DEAD (all 404)
  - ❌ FCA UK — BLOCKED (403, S3 access denied)
  - ❌ ASX Short Positions — NOT FOUND (404, URL changed)
  - **Verdict:** No working free API found for non-US short selling data

### 7. form144.py — Form144Tool
- **Tags:** `[R:FULL]` `[G:INHERENT]`
- **APIs:** SEC EFTS (efts.sec.gov, XML, 10 req/sec) + EDGAR Archives
- **Coverage:** US-only — SEC Form 144 is US insider selling disclosure
- **Research doc:** `[[form144]]`
- **Tests:** test_form144_edge.py (57 tests)
- **Action needed:** Region-locked by nature. Global expansion = separate tools for Companies House (UK), SEBI (India), EDINET (Japan), etc.

### 8. gdelt.py — GDELTTool
- **Tags:** `[R:FULL]` `[G:GLOBAL]`
- **APIs:** GDELT (data.gdeltproject.org raw + api.gdeltproject.org DOC API, no auth)
- **Coverage:** Global — CAMEO taxonomy covers all countries
- **Research doc:** `[[gdelt]]`
- **Tests:** Phase 6 pipeline tests
- **Action needed:** None — fully global

### 9. gov_contracts.py — GovContractsTool
- **Tags:** `[R:FULL]` `[G:US-ONLY]` `[G:NEEDS-EXPANSION]`
- **APIs:** USASpending.gov (api.usaspending.gov, POST, no auth)
- **Coverage:** US federal contracts only
- **Research doc:** `[[7b-G_gov_contracts]]`
- **Tests:** test_gov_contracts_edge.py (60+ tests)
- **International alternatives probed (2026-03-28):** See `[[international_api_alternatives]]`
  - ✅ **UK Contracts Finder** — WORKING, OCDS JSON, no auth, rich data (HIGHEST PRIORITY)
  - ✅ **EU Open Data Portal** — WORKING, bulk CSV of TED procurement data
  - ❌ EU TED API v3 — DEAD (404)
  - ❌ AusTender — BLOCKED (403)
  - ❌ India GeM — NO API (404)
  - ❌ Canada BuyAndSell — DOWN (no route to host)
  - 🔑 Korea KONEPS — needs Korean API key

### 10. insider_filings.py — InsiderFilingsTool
- **Tags:** `[R:IMPLICIT]` `[G:INHERENT]`
- **APIs:** SEC EFTS + EDGAR Archives (no auth, 10 req/sec)
- **Coverage:** US-only — SEC Form 4
- **Research doc:** None standalone — early phase tool
- **Tests:** Phase 3-7 embedded tests
- **Action needed:** Region-locked by nature (SEC Form 4). Global expansion = separate tools per jurisdiction.
- **International alternatives probed (2026-03-28):** See `[[international_api_alternatives]]`
  - 🔑 **UK Companies House** — AUTH-GATED, free API key registration
  - 🔑 **EDINET Japan** — AUTH-GATED, needs Japanese FSA subscription key
  - ❌ SEDAR+ Canada — NOT FOUND (404)

### 11. macro_data.py — MacroDataTool
- **Tags:** `[R:FULL]` `[G:US-ONLY]` `[G:NEEDS-EXPANSION]`
- **APIs:** FRED (api.stlouisfed.org, requires TIRRA_FRED_API_KEY)
- **Coverage:** Primarily US Fed data (Fed balance sheet, treasury, M2, unemployment). Some global series exist but tool is Fed-centric.
- **Research doc:** `[[macro_data]]`
- **Tests:** Phase 7 pipeline tests
- **International alternatives probed (2026-03-28):** See `[[international_api_alternatives]]`
  - ✅ **ECB Data API** — WORKING, no auth, interest rates + exchange rates + balance sheet
  - ✅ **OECD SDMX** — WORKING, no auth, CLI (38 countries) + GDP. Complex 9-12 dim keys.
  - ✅ **World Bank API** — WORKING, no auth, 200+ countries. Lower frequency.
  - ❌ BOE — BLOCKED (403, bot protection)
  - ❌ BOJ — NOT FOUND (404)
  - ❌ BIS — DEAD (404, API doesn't exist)

### 12. market_data.py — MarketDataTool
- **Tags:** `[R:NONE]` `[G:GLOBAL]`
- **APIs:** yfinance (wraps Yahoo Finance)
- **Coverage:** Global — all major exchanges worldwide
- **Research doc:** None — early phase tool, no documentation
- **Tests:** Phase 3-7 embedded tests
- **Action needed:** Write research doc. Tool itself is global.

### 13. polymarket.py — PolymarketTool
- **Tags:** `[R:IMPLICIT]` `[G:GLOBAL]`
- **APIs:** Gamma API (gamma-api.polymarket.com, REST, no auth)
- **Coverage:** Global — prediction markets on global topics
- **Research doc:** Implicit in polymarket_whale.md
- **Tests:** Phase 7b embedded tests
- **Action needed:** Write standalone research doc

### 14. polymarket_whales.py — PolymarketWhalesTool
- **Tags:** `[R:FULL]` `[G:GLOBAL]`
- **APIs:** Polymarket Data API (data-api.polymarket.com, REST, no auth)
- **Coverage:** Global — on-chain, borderless wallets
- **Research doc:** `[[polymarket_whale]]`
- **Tests:** test_polymarket_whales_edge.py (100 tests)
- **Action needed:** None — fully researched and global

### 15. power_grid.py — PowerGridTool
- **Tags:** `[R:FULL]` `[G:REGIONAL]` `[G:NEEDS-EXPANSION]`
- **APIs:** NYISO MIS CSV (mis.nyiso.com, no auth)
- **Coverage:** New York state only (11 zones)
- **Research doc:** `[[power_grid]]`
- **Tests:** test_power_grid_edge.py (98 tests)
- **International alternatives probed (2026-03-28):** See `[[international_api_alternatives]]`
  - 🔑 **ENTSO-E Transparency** — AUTH-GATED, free registration, 36 EU countries. API confirmed working (proper XML error at 401).
  - ❌ Other US ISOs (CAISO, PJM, ERCOT) — blocked in 6g research, not re-probed

### 16. regulatory_gazette.py — RegulatoryGazetteTool
- **Tags:** `[R:FULL]` `[G:US-ONLY]` `[G:NEEDS-EXPANSION]`
- **APIs:** Federal Register API (federalregister.gov, JSON, no auth)
- **Coverage:** US Federal Register only.
- **Research doc:** `[[7b-Q_regulatory_gazette]]`
- **Tests:** test_regulatory_gazette_edge.py (144 tests)
- **International alternatives probed (2026-03-28):** See `[[international_api_alternatives]]`
  - ✅ **UK legislation.gov.uk** — WORKING, Atom XML, no auth. Statutory Instruments with titles, dates, links.
  - ⚠️ **EU Publications SPARQL** — PARTIALLY WORKING, Virtuoso RDF store responds. OJ entries found with dates but title retrieval needs correct CDM ontology paths.
  - ❌ EUR-Lex Search/RSS — DEAD (all 404)
  - ❌ EU Official Journal CELLAR REST — 500 error

### 17. transport_throughput.py — TransportThroughputTool
- **Tags:** `[R:FULL]` `[G:US-ONLY]` `[G:NEEDS-EXPANSION]`
- **APIs:** BTS Border Crossings via Socrata (data.transportation.gov, SoQL, no auth)
- **Coverage:** US-Canada + US-Mexico land borders only
- **Research doc:** `[[7b-R_transport_throughput]]`
- **Tests:** test_transport_throughput_edge.py (62 tests)
- **International alternatives probed (2026-03-28):** See `[[international_api_alternatives]]`
  - ✅ **Eurostat Transport** — WORKING, SDMX JSON, no auth, 27 EU countries (air passengers + goods transport)

### 18. weather_alerts.py — WeatherAlertsTool
- **Tags:** `[R:FULL]` `[G:REGIONAL]`
- **APIs:** NOAA NWS (api.weather.gov, GeoJSON, no auth), NASA FIRMS MODIS (CSV, no auth)
- **Coverage:** Mixed — NWS is US-only; FIRMS is global but filtered to 12 US infrastructure zones
- **Research doc:** `[[7b-C_weather_alerts]]`
- **Tests:** test_weather_alerts_edge.py (92 tests)
- **International alternatives probed (2026-03-28):** See `[[international_api_alternatives]]`
  - ✅ **DWD (Germany)** — WORKING, CAP 1.2 XML in ZIP, no auth. Multi-language (EN/DE/FR/ES).
  - ✅ **JMA (Japan)** — WORKING, JSON, no auth. Warnings with area codes (Tokyo: 130000).
  - ❌ Meteoalarm EU — TIMEOUT (hangs indefinitely)
  - ❌ PAGASA Philippines — NOT FOUND (404)

### 19. whale_alert.py — WhaleAlertTool
- **Tags:** `[R:FULL]` `[G:GLOBAL]`
- **APIs:** Blockchain.info (mempool + blocks, no auth)
- **Coverage:** Global — Bitcoin is borderless
- **Research doc:** `[[whale_alert]]`
- **Tests:** test_whale_alert_edge.py (36 tests)
- **Action needed:** None — fully researched and global

### 20. wikipedia_pageviews.py — WikipediaPageviewsTool
- **Tags:** `[R:IMPLICIT]` `[G:GLOBAL]`
- **APIs:** Wikimedia REST API (wikimedia.org, JSON, no auth)
- **Coverage:** Global — 300+ language editions
- **Research doc:** None standalone — documented in task file 7b-O notes
- **Tests:** test_wikipedia_pageviews_edge.py (77 tests)
- **Action needed:** Write standalone research doc

---

## Summary Statistics

### Research Status
| Status | Count | Tools |
|--------|-------|-------|
| `[R:FULL]` | 9 | ais_vessel, cftc, finra_short_volume, form144, gdelt, polymarket_whales, power_grid, regulatory_gazette, whale_alert |
| `[R:IMPLICIT]` | 10 | academic_preprints, defi_flows, earthquake_proximity, gov_contracts, insider_filings, macro_data, polymarket, transport_throughput, weather_alerts, wikipedia_pageviews |
| `[R:NONE]` | 1 | market_data |

### Geographic Coverage
| Status | Count | Tools |
|--------|-------|-------|
| `[G:GLOBAL]` | 9 | academic_preprints, defi_flows, earthquake_proximity, gdelt, market_data, polymarket, polymarket_whales, whale_alert, wikipedia_pageviews |
| `[G:INHERENT]` (region-locked by data source nature) | 4 | cftc, finra_short_volume, form144, insider_filings |
| `[G:NEEDS-EXPANSION]` (currently regional, should be global) | 5 | gov_contracts, macro_data, power_grid, regulatory_gazette, transport_throughput |
| `[G:REGIONAL]` (regional by sensor/source, acceptable) | 2 | ais_vessel, weather_alerts |

### Globalization Priority (Updated 2026-03-28 with live probe results)

**Full findings:** `[[international_api_alternatives]]`

| Priority | Tool | Current | Verified Working Alternatives |
|----------|------|---------|-------------------------------|
| **HIGH** | gov_contracts | US only | ✅ UK Contracts Finder (OCDS JSON, no auth) |
| **HIGH** | macro_data | US/FRED only | ✅ ECB Data API + OECD SDMX + World Bank (all no auth) |
| **HIGH** | power_grid | NY state only | 🔑 ENTSO-E (36 EU countries, free registration) |
| **MEDIUM** | transport_throughput | US borders only | ✅ Eurostat Transport (27 EU countries, no auth) |
| **MEDIUM** | regulatory_gazette | US only | ✅ UK legislation.gov.uk (Atom XML, no auth) |
| **LOW** | weather_alerts | US NWS + global FIRMS | ✅ DWD Germany + JMA Japan (both no auth) |
| **LOW** | insider_filings | US SEC | 🔑 UK Companies House + EDINET Japan (both free key) |
| **BLOCKED** | finra_short_volume | US only | ❌ No working free API found globally |

### Research Docs Status (Updated 2026-03-28)
All 20 tools now have standalone research docs (`[R:FULL]`). 11 docs created on 2026-03-28.
International API alternatives documented in `[[international_api_alternatives]]`.

---

## Workflow Mandate (going forward)

For EVERY new data tool, before ANY code is written:

1. **Research** → Create `docs/research/<tool_name>.md` with:
   - API endpoints probed (URLs, methods, auth requirements)
   - Geographic coverage (which countries/regions)
   - Data format (JSON/XML/CSV/ZIP)
   - Rate limits and auth requirements
   - Data freshness (real-time, daily, weekly, monthly)
   - Signal value assessment
   - Risks (API stability, data quality, legal)
2. **Spec** → Create `docs/specs/<tool_name>_spec.md` with atomic steps
3. **Implement** → Code the tool
4. **Test** → Write edge case tests
5. **Tag** → Update this document with research + globalization tags

## Related

- [[tool_globalization_track1_spec|Spec: Tool Globalization Track1]]
