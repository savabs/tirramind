---
title: International API Alternatives — Live Probe Results
tags:
  - doc/research
---

# International API Alternatives — Live Probe Results

**Date:** 2026-03-28
**Purpose:** Document live-probed international API alternatives for every region-locked tool. This is the result of 60+ HTTP endpoint probes across 15+ countries.

**Methodology:** Every endpoint below was probed live with `httpx` — status codes, response bodies, data structures, and auth requirements verified empirically.

---

## TIER 1: CONFIRMED WORKING — NO AUTH

These APIs returned real data with zero authentication. Ready for tool implementation.

### 1. UK Contracts Finder (gov_contracts alternative)
- **URL:** `https://www.contractsfinder.service.gov.uk/Published/Notices/OCDS/Search?publishedFrom=YYYY-MM-DD&publishedTo=YYYY-MM-DD`
- **Method:** GET
- **Format:** JSON (OCDS — Open Contracting Data Standard)
- **Auth:** None
- **Quality:** **EXCELLENT** — Rich structured JSON with:
  - Tender details: title, description, status
  - Buyer: organization name, identifier
  - Awards: amount (e.g., £1,296,000), currency (GBP), supplier name
  - Parties: roles (buyer, tenderer), addresses
  - Contract period, dates
- **Coverage:** All UK public sector procurement
- **Sample:** "The provision of Police Constable Degree Apprenticeships", Northamptonshire Police, £1.3M
- **Implementation priority:** **HIGHEST** — cleanest data, standard format, no auth, rich signal

### 2. ECB Data API (macro_data alternative)
- **URL:** `https://data-api.ecb.europa.eu/service/data/{flowRef}/{key}?format=jsondata`
- **Method:** GET
- **Format:** SDMX JSON
- **Auth:** None
- **Verified datasets:**
  - `EXR/D.USD.EUR.SP00.A` — Exchange rates (EUR/USD, daily)
  - `FM/B.U2.EUR.4F.KR.MFI.NWT` — Interest rates (ECB MRR: 2.65, 2.4, 2.15)
  - `ILM/W.U2.C.T000000.Z5.Z01` — ECB Total Assets (balance sheet: 6,176,464 / 6,168,261 / 6,155,306 in millions EUR)
- **Coverage:** Eurozone monetary policy, exchange rates, balance sheet
- **Implementation priority:** **HIGH** — direct FRED equivalent for Europe

### 3. Eurostat Transport API (transport_throughput alternative)
- **URL:** `https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/{dataset}/{key}?format=SDMX-JSON`
- **Method:** GET
- **Format:** SDMX JSON
- **Auth:** None
- **Verified datasets:**
  - `ttr00012` — Air transport of passengers (27 EU countries + Euro area)
  - `ttr00007` — Goods transport (similar coverage)
  - Dimensions: freq, unit, vessel, tra_cov, geo (AT, BE, BG, CY, CZ, DE, DK, EE, EL, ES, FI, FR, HR, HU, IE, IT, LT, LU, LV, MT, NL, PL, PT, RO, SE, SI, SK + EA, EU27_2020)
- **Coverage:** 27 EU countries
- **Implementation priority:** **HIGH** — direct transport throughput equivalent for EU

### 4. World Bank API (macro_data alternative)
- **URL:** `https://api.worldbank.org/v2/country/all/indicator/{indicator}?format=json&per_page=300`
- **Method:** GET
- **Format:** JSON
- **Auth:** None
- **Verified indicators:**
  - GDP, industrial production %, agriculture value added
  - 266 records (multi-country)
  - Sample: Brazil industrial 20.9%, China 36.5%
- **Coverage:** 200+ countries
- **Implementation priority:** **MEDIUM** — broad but lower frequency (annual/quarterly). Complement to ECB/OECD.

### 5. OECD SDMX API (macro_data alternative)
- **URL:** `https://sdmx.oecd.org/public/rest/data/{agency},{dataflow},{version}/{key}?lastNObservations={n}`
- **Method:** GET
- **Format:** SDMX XML (generic)
- **Auth:** None
- **Verified datasets:**
  - CLI (Composite Leading Indicators): `OECD.SDD.STES,DSD_STES@DF_CLI,4.1/{FREQ}.{REF_AREA}.{MEASURE}.{UNIT_MEASURE}.{ACTIVITY}.{ADJUSTMENT}.{TRANSFORMATION}.{TIME_HORIZ}.{METHODOLOGY}`
    - Working example: `M.USA+GBR+DEU.LI......?lastNObservations=2`
    - Values: USA 100.82, GBR 128.62, DEU 108.93
  - GDP (National Accounts): `OECD.SDD.NAD,DSD_NAMAIN10@DF_TABLE1,/{12 dimensions}`
    - Working example: `A.USA...........?lastNObservations=1`
    - 684 values returned (720KB)
- **Complexity:** High — 9-12 dimension SDMX keys, needs careful documentation
- **Coverage:** 38 OECD member countries + partners
- **Implementation priority:** **MEDIUM** — powerful but complex API format

### 6. UK Legislation (regulatory_gazette alternative)
- **URL:** `https://www.legislation.gov.uk/{type}?_page={n}&_pageSize={n}`
- **Method:** GET
- **Format:** Atom XML
- **Auth:** None
- **Verified content types:**
  - `/uksi` — UK Statutory Instruments (20 entries per page)
  - Sample: "The Human Medicines (Amendment) Regulations 2026" (2026-03-27), "The Renewables Obligation (Amendment) Order 2026"
  - Each entry: title, updated timestamp, link to full text
- **Coverage:** All UK primary and secondary legislation
- **Implementation priority:** **MEDIUM** — Atom XML needs parsing, decent regulatory signal

### 7. DWD German Weather Service (weather_alerts alternative)
- **URL:** `https://opendata.dwd.de/weather/alerts/cap/COMMUNEUNION_DWD_STAT/`
- **Method:** GET (directory listing → download ZIP → extract XML)
- **Format:** CAP 1.2 XML (Common Alerting Protocol) in ZIP files
- **Auth:** None
- **Verified content:**
  - ZIP file (170KB) containing multiple CAP XML alerts
  - Each alert: identifier, sender (opendata@dwd.de), sent datetime, status (Actual), msgType (Alert), scope (Public)
  - Available in EN, DE, ES, FR, MUL translations
- **Coverage:** All of Germany
- **Implementation priority:** **LOW** — ZIP→XML pipeline is more complex, but CAP standard is universal

### 8. JMA Japan Meteorological Agency (weather_alerts alternative)
- **URL:** `https://www.jma.go.jp/bosai/warning/data/warning/{areacode}.json`
- **Method:** GET
- **Format:** JSON
- **Auth:** None
- **Verified content:**
  - Tokyo (130000): reportDatetime, publishingOffice (気象庁), headlineText, areaTypes, timeSeries
  - Also: `https://www.jma.go.jp/bosai/forecast/data/overview_forecast/{areacode}.json` — weather overview
- **Coverage:** All of Japan (region code based)
- **Implementation priority:** **LOW** — JSON in Japanese, needs translation mapping. Useful for Japan-specific infrastructure monitoring.

### 9. EU Open Data Portal — TED Bulk (gov_contracts alternative)
- **URL:** `https://data.europa.eu/api/hub/search/search?q=ted+tenders&limit={n}`
- **Method:** GET
- **Format:** JSON (catalog search, links to CSV bulk downloads)
- **Auth:** None
- **Verified content:**
  - 10,539 datasets related to "TED procurement"
  - Includes "Tenders Electronic Daily (TED) (csv subset)" with 48 CSV distributions
  - Bulk download route — not real-time API
- **Coverage:** All 27 EU member states
- **Implementation priority:** **MEDIUM** — Bulk CSV, good for historical analysis. Not real-time.

---

## TIER 2: AUTH-GATED — FREE REGISTRATION

These APIs are confirmed functional but require free API key registration.

### 10. ENTSO-E Transparency Platform (power_grid alternative)
- **URL:** `https://web-api.tp.entsoe.eu/api?securityToken={token}&...`
- **Method:** GET
- **Format:** XML (Acknowledgement_MarketDocument schema)
- **Auth:** Free security token — register at transparency.entsoe.eu
- **Verification:** Returns 401 with proper `Acknowledgement_MarketDocument` XML error (confirms API is functional, just needs auth)
- **Coverage:** 36 European countries — generation, load, prices, cross-border flows
- **Implementation priority:** **HIGH** — Best power grid API for Europe. Registration is free.

### 11. EDINET Japan (insider_filings alternative)
- **URL:** `https://api.edinet-fsa.go.jp/api/v2/documents.json?date=YYYY-MM-DD&Subscription-Key={key}`
- **Method:** GET
- **Format:** JSON
- **Auth:** Subscription key — register with Japanese FSA (Financial Services Agency)
- **Verification:** Returns 401 with "Access denied due to invalid subscription key"
- **Coverage:** All Japanese public company filings (equivalent to SEC EDGAR)
- **Implementation priority:** **MEDIUM** — key registration may require Japanese address/residency

### 12. UK Companies House (insider_filings alternative)
- **URL:** `https://api.company-information.service.gov.uk/search/officers?q={name}`
- **Method:** GET
- **Format:** JSON
- **Auth:** Free API key — register at developer.company-information.service.gov.uk
- **Verification:** Returns 401 with proper `ch:service` error type
- **Coverage:** All UK companies — officers, filings, charges, insolvency
- **Implementation priority:** **MEDIUM** — easy registration, UK-focused

### 13. Korea KONEPS (gov_contracts alternative)
- **URL:** `https://apis.data.go.kr/1230000/HrcspSsstndrdInfoService/...`
- **Method:** GET
- **Format:** JSON/XML
- **Auth:** Korean Data Portal API key (data.go.kr)
- **Verification:** Returns 500 (server error) without valid key — but API exists and accepts requests
- **Coverage:** Korean government procurement
- **Implementation priority:** **LOW** — Korean API key registration process

---

## TIER 3: CONFIRMED BROKEN / DEAD / BLOCKED

These endpoints were probed and confirmed non-functional.

| Source | URL Tested | Status | Reason |
|--------|-----------|--------|--------|
| EU TED API v3 | `ted.europa.eu/api/v3.0/...` | 404 | Old API decommissioned |
| EUR-Lex Search | `eur-lex.europa.eu/eurlex-ws/...` | 404 | All REST endpoints dead |
| EUR-Lex RSS | `eur-lex.europa.eu/oj/browse-oj-rss.html` | 404 | Feed removed |
| BOE Database | `bankofengland.co.uk/boeapps/...` | 403 | Bot protection |
| BIS Statistics | `data.bis.org/api/v1,v2/...` | 404 | API not found |
| FCA UK Short Selling | `data.fca.org.uk/...` | 403 | S3 access denied |
| ESMA Short Selling Register | `registers.esma.europa.eu/...` | 404 | All endpoints dead |
| AusTender | `www.tenders.gov.au/...` | 403 | Blocked |
| India GeM | `gem.gov.in/...` | 404 | No public API |
| Canada BuyAndSell | `buyandsell.gc.ca/...` | Error | No route to host |
| SEDAR+ Canada | `www.sedarplus.ca/...` | 404 | No API |
| Meteoalarm EU | `feeds.meteoalarm.org/...` | Timeout | Hangs indefinitely |
| Japan TFX | `www.tfx.co.jp/en/historical/positions/` | 404 | Not found |
| BOJ Statistics | `stat-search.boj.or.jp/ssi/...` | 404 | Not found |
| JMA base | `jma.go.jp/bosai/warning/data/warning/010000.json` | 404 | Wrong area code |
| PAGASA Philippines | `pubfiles.pagasa.dost.gov.ph/...` | 404 | Not found |
| ASX Short Positions | `www2.asx.com.au/.../shortpositions.csv` | 404 | URL changed/removed |

---

## TIER 4: PARTIALLY WORKING — NEEDS MORE INVESTIGATION

### HKEX Short Positions (finra_short_volume alternative)
- **URL:** `https://www.hkexnews.hk/sdw/search/searchsdw.aspx`
- **Status:** 200 but returns 1MB HTML page
- **Issue:** Data embedded in HTML table, needs scraping
- **Priority:** LOW — fragile, HTML scraping

### TSX Canada Company Directory
- **URL:** `https://www.tsx.com/json/company-directory/search/tsx/%5E*`
- **Status:** 200, returns 304KB JSON (2144 companies)
- **Issue:** Company directory only, not short positions
- **Priority:** LOW — not the right data type

### EU Publications SPARQL (regulatory_gazette alternative)
- **URL:** `https://publications.europa.eu/webapi/rdf/sparql`
- **Status:** 200, Virtuoso RDF store works
- **Issue:** Found OJ entries (dates work) but title retrieval needs correct ontology path. Basic queries work, complex CDM queries return 0 results.
- **Verified:** Named graphs exist (`cellar/...`), OJ type works, date sorting works
- **Priority:** MEDIUM — needs SPARQL expertise to construct correct queries

---

## Recommended Implementation Order

Based on data quality, ease of implementation, and signal value:

| Priority | Action | Tool to Modify | New Source | Effort |
|----------|--------|---------------|------------|--------|
| 1 | Add UK procurement | gov_contracts | UK Contracts Finder OCDS | **EASY** — JSON, no auth, OCDS standard |
| 2 | Add EU macro data | macro_data | ECB Data API | **EASY** — JSON, no auth, SDMX |
| 3 | Add EU transport | transport_throughput | Eurostat Transport | **MEDIUM** — SDMX JSON format |
| 4 | Register ENTSO-E key | power_grid | ENTSO-E Transparency | **MEDIUM** — needs free registration |
| 5 | Add UK legislation | regulatory_gazette | legislation.gov.uk | **MEDIUM** — Atom XML parsing |
| 6 | Add OECD indicators | macro_data | OECD SDMX | **HARD** — complex 9-12 dimension keys |
| 7 | Add World Bank data | macro_data | World Bank API | **EASY** — JSON, no auth. Low freq. |
| 8 | Add EU TED bulk | gov_contracts | EU Open Data Portal CSV | **MEDIUM** — bulk download, not real-time |
| 9 | Register Companies House key | insider_filings | UK Companies House | **MEDIUM** — needs free registration |
| 10 | Add DWD weather | weather_alerts | DWD Open Data | **HARD** — ZIP→XML pipeline |

---

## Globalization Coverage Matrix (Post-Expansion)

| Tool | Current | + Tier 1 (no auth) | + Tier 2 (free key) | Total Regions |
|------|---------|-------------------|---------------------|---------------|
| gov_contracts | US | + UK, EU (bulk) | + KR | US, UK, EU (4) |
| macro_data | US (FRED) | + EU (ECB), OECD (38), World (200+) | — | Global (5 sources) |
| transport_throughput | US borders | + EU (27 countries) | — | US + EU (2) |
| power_grid | NY state | — | + EU (ENTSO-E, 36 countries) | US + EU (2) |
| regulatory_gazette | US | + UK | — | US + UK (2) |
| weather_alerts | US (NWS) | + DE (DWD), JP (JMA) | — | US + DE + JP (3) |
| insider_filings | US (SEC) | — | + JP (EDINET), UK (Companies House) | US + UK + JP (3) |
| finra_short_volume | US | — | — | US only (no working alt found) |

---

## Key Findings

1. **UK has the best international APIs** — Contracts Finder (OCDS standard), legislation.gov.uk (Atom), Companies House (JSON). UK open data strategy is mature.
2. **ECB is the FRED equivalent for Europe** — clean JSON, no auth, comprehensive monetary data.
3. **OECD is powerful but painful** — 9-12 dimension SDMX keys. Once cracked, covers 38+ countries. CLI and GDP both confirmed working.
4. **EU procurement (TED)** — old API dead, but bulk CSV available via Open Data Portal. Not real-time.
5. **Short selling data is the hardest to globalize** — ESMA, FCA, ASX all broken/blocked. Only HKEX works (HTML scraping).
6. **Many Asian APIs are gated** — EDINET (Japanese FSA key), KONEPS (Korean data portal key). Not insurmountable but adds registration friction.
7. **Weather is globally available** — DWD (Germany), JMA (Japan) both work. CAP standard is universal.
8. **Central bank APIs vary wildly** — ECB excellent, BOE blocked, BOJ not found. BIS API doesn't exist.

## Related

- [[project_memory]]
