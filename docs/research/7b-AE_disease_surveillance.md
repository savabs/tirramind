---
title: "7b-AE: Disease & Pandemic Surveillance — Research"
tags:
  - doc/research
  - layer/surveillance
  - phase/7b
  - topic/disease-surveillance
---

# 7b-AE: Disease & Pandemic Surveillance — Research

**Date:** 2026-03-31
**Layer:** L0 (wastewater = physical measurement) + L1 (outbreak declarations = committed expert assessment)
**Status:** Research complete, ready for spec

## Thesis

Pathogen concentrations in sewage are physics — you can't spin, delay, or retract them. Wastewater data detected COVID waves 2-3 weeks before hospital admissions in every study. The same principle applies to influenza, RSV, avian flu, measles, and mpox. If we see wastewater spikes in 10+ US states simultaneously, or a novel pathogen appearing in sewage, we know before anyone has been hospitalized. Combine this with WHO global outbreak declarations and ECDC European surveillance for worldwide coverage — a pandemic signal crosses our detection surface months before it becomes a headline.

**Market relevance:** Pandemics → pharma demand (vaccine makers, PPE), travel disruptions (airlines, hotels, cruise), labor shortages (absenteeism), supply chain disruption (factory shutdowns), and healthcare sector revenue. H5N1 avian influenza in wastewater + poultry culling data = food price spike signal.

## Sources Probed (16 endpoints tested live, 2026-03-31)

### Tier 1: CONFIRMED WORKING — BUILD THESE

#### 1. CDC NWSS Wastewater (Socrata API) — ★★★★★ HIGHEST PRIORITY

**The single most valuable L0 disease data source in existence for free.**

CDC runs the National Wastewater Surveillance System (NWSS). They have **8 separate datasets**, each tracking a specific pathogen via PCR concentrations in raw wastewater at treatment plants across the US.

| Dataset ID | Pathogen | Records | Updated |
|-----------|----------|---------|---------|
| `j9g8-acpt` | **SARS-CoV-2** | 555,242 | 2026-03-27 |
| `ymmh-divb` | **Influenza A** | 264,181 | 2026-03-27 |
| `45cq-cw4i` | **RSV** | 243,611 | 2026-03-27 |
| `xpxn-rzgz` | **Mpox** | 252,103 | 2026-03-27 |
| `akvg-8vrb` | **Measles** | 38,996 | 2026-03-27 |
| `mtpu-urpp` | **Avian Influenza H5** | 93,369 | 2026-03-27 |
| `2ew6-ywp6` | **Aggregate metrics** | 837,382 | 2025-09-12 |
| `g653-rqe2` | **SARS-CoV-2 concentrations** | (legacy) | 2025-09-12 |

**Total: ~2.28 million records across 8 datasets.**

**API details:**
- **URL pattern:** `https://data.cdc.gov/resource/{dataset_id}.json`
- **Auth:** None required (Socrata open data)
- **Rate limit:** None enforced (standard Socrata — 1000 records/request with `$limit`, paginate with `$offset`)
- **Query:** Full SoQL support — `$where`, `$select`, `$group`, `$order`, aggregations, date filtering
- **Pagination:** `$limit` (max 50000 per page) + `$offset`
- **Freshness:** Updated weekly (lag ~7-10 days from sample collection to publication)
- **Geographic granularity:** Individual treatment plants → county FIPS → state/jurisdiction
- **Coverage:** 51 jurisdictions (50 states + DC), 1000+ wastewater treatment plants

**Schema (shared across all pathogen datasets):**
```
record_id             — unique hash
site                  — treatment plant ID
state_territory       — 2-letter state code
source                — data provider (CDC_Verily, State_Territory)
county_fips           — 5-digit FIPS code
counties_served       — county name(s)
population_served     — integer, population feeding this plant
sample_collect_date   — YYYY-MM-DD
sample_type           — "grab" | "24-hr time-weighted composite" | etc.
sample_matrix         — "raw wastewater"
pcr_target            — pathogen name (sars-cov-2, fluav, rsv, etc.)
pcr_target_avg_conc   — raw concentration (copies/L wastewater) ← THE SIGNAL
pcr_target_units      — always "copies/l wastewater"
pcr_target_detect     — "yes" | "no" (detected above LOD?)
pcr_target_flowpop_lin — flow-population normalized concentration
pcr_target_mic_lin    — microbial-indicator-corrected concentration
lod_sewage            — limit of detection (copies/L)
date_updated          — last data refresh timestamp
```

**Aggregated metrics endpoint (2ew6-ywp6) schema:**
```
wwtp_jurisdiction     — state name
wwtp_id               — plant ID
county_fips           — FIPS
population_served     — integer
date_start / date_end — 15-day window
ptc_15d               — percent change in 15-day average (THE TREND SIGNAL)
detect_prop_15d       — detection proportion in 15-day window
percentile            — percentile rank within site's historical distribution
```

**Key signals derivable:**
1. **Outbreak velocity** — `ptc_15d` > 100 = concentration doubled in 2 weeks
2. **Geographic spread** — count of states with `detect_prop_15d` > 50% trending up
3. **Novel pathogen alert** — Avian H5 dataset moving from 0% to >0% detection signals spillover
4. **Regional hotspot** — cluster of high-percentile sites within same FIPS region
5. **Seasonal comparison** — current percentile vs historical for same-week-of-year
6. **Multi-pathogen co-circulation** — flu + RSV + COVID all rising = healthcare strain

#### 2. WHO Disease Outbreak News (OData API) — ★★★★ GLOBAL OFFICIAL DECLARATIONS

**URL:** `https://www.who.int/api/hubs/diseaseoutbreaknews`

**API details:**
- **Auth:** None
- **Protocol:** OData v4
- **Pagination:** `$top` (page size, default 50), `@odata.nextLink` for next page
- **Filtering:** `$filter`, `$orderby`, `$select` supported
- **Records:** 3,175 total outbreak entries
- **Freshness:** New entries as outbreaks are declared; most recent = 2026-03-13
- **Coverage:** Global — all WHO member states

**Fields per entry:**
```
DonId, Title, PublicationDate, PublicationDateAndTime
Overview, Assessment, Advice, Response, Epidemiology  (HTML body text)
Summary, FurtherInformation
UrlName, ItemDefaultUrl
DateCreated, LastModified, FormattedDate
```

**Limitation:** No structured country/disease/pathogen fields — must parse from Title + body text. Titles follow pattern like "Nipah virus infection - Bangladesh" or "Mpox: recombinant virus ... – Global situation".

**Key signals:**
1. **New outbreak detection** — new DonId entries not seen before
2. **Disease clustering** — multiple DON entries for same pathogen across countries = spreading
3. **Escalation velocity** — DON frequency for same disease increasing = WHO is worried
4. **Novel pathogen** — title contains "unknown", "novel", "undiagnosed" = highest alert

#### 3. ECDC Open Data Portal — ★★★ EU SURVEILLANCE

Three working datasets:

| Endpoint | Path | Records | Coverage |
|----------|------|---------|----------|
| COVID cases/deaths | `/covid19/nationalcasedeath/json` | 12,648 | EU/EEA by country/week |
| Virus variants | `/covid19/virusvariant/json` | 148,258 | EU/EEA sequencing data |
| Hospital occupancy | `/covid19/hospitalicuadmissionrates/json` | 28,337 | EU/EEA admission rates |

**API details:**
- **Base URL:** `https://opendata.ecdc.europa.eu`                            
- **Auth:** None
- **Format:** JSON array (entire dataset returned at once — no pagination needed)
- **Fields:** country, country_code, year_week, indicator, source + dataset-specific fields
- **Freshness:** Varies; COVID cases updated through recent weeks

**Limitation:** COVID-focused. Broader surveillance (TESSy Atlas) returns HTML, not usable JSON API. Limited to EU/EEA countries (~30).

**Key signals:**
1. **EU wave detection** — cases rising across 5+ EU countries simultaneously
2. **Variant takeover velocity** — new variant going from 1% to 50% share in N weeks
3. **Hospital strain** — ICU admission rates crossing capacity thresholds

#### 4. NCBI E-utilities (GenBank) — ★★★ GENOMIC SEQUENCE COUNTS

**URL:** `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi`

- **Auth:** None (rate limit: 3 req/sec without API key, 10 req/sec with free key)
- **Database:** `nucleotide` (GenBank sequences)
- **Query format:** Entrez term syntax — organism names, date ranges, sequence types
- **Returns:** Sequence count + IDs

**Sample counts (probed 2026-03-31):**
- SARS-CoV-2 sequences from 2026: **12,897**
- Measles sequences from 2025: **2,264**
- H5N1 sequences from 2026: **2** (low but will spike if outbreak expands)

**Key signals:**
1. **Submission velocity** — accelerating sequence submissions = outbreak expanding, labs are actively characterizing
2. **Novel organism emergence** — new organism + country combination appearing for first time
3. **Geographic spread of variant** — same lineage appearing in sequences from new countries

#### 5. disease.sh — ★★ COVID AGGREGATOR (SUPPLEMENTARY)

**URL:** `https://disease.sh/v3/covid-19/`

- **Auth:** None
- **Endpoints:** `/all` (global), `/countries` (by country, 231 countries), `/historical` (time series)
- **Data:** Cases, deaths, recovered, active, tests — all COVID-19
- **Freshness:** Automated from JHU/Worldometers (may have stopped updating)

**Limitation:** COVID-only. Likely stale (no flu check failed with 404). Useful as a supplementary global snapshot but not a primary source.

### Tier 2: DEAD / AUTH-GATED / UNREACHABLE

| Source | Status | Notes |
|--------|--------|-------|
| **ProMED-mail** | ❌ DEAD | All URLs return 404 (promedmail.org, isid.org/promedmail). RSS feed, WP API, promed-posts — all dead. ProMED was the gold standard for early outbreak detection (first to report SARS, MERS, COVID). Appears to have been taken offline or restructured. Major loss. |
| **GISAID** | ❌ AUTH-GATED | Requires academic registration + institutional affiliation. No public API. The EpiCoV portal (epicov.org) returns 200 but requires login. Cannot use programmatically without credentials. |
| **India IDSP** | ❌ NO API | `idsp.nic.in` DNS fails. `idsp.mohfw.gov.in` returns HTML page (no API). Outbreak reports are PDFs, not machine-readable. |
| **Brazil InfoGripe** | ❌ TIMEOUT | `info.gripe.fiocruz.br` times out consistently. May be geo-restricted or down. |
| **HealthMap** | ❌ DEAD | healthmap.org was shut down. No replacement API found. |
| **Global.health** | ❌ DEAD | `data.covid-19.global.health` DNS fails. |
| **outbreak.info** | ❌ AUTH-GATED | Genomics API returns 401 Unauthorized. |
| **WHO GHO OData** | ❌ DNS FAIL | `ghoapi.who.int` resolves but fails to connect (possibly ISP DNS issue from India). |
| **WHO EIOS** | ⚠️ PORTAL ONLY | `portal.who.int/eios` returns 200 (HTML portal). No public API — requires WHO account. |
| **CDC FluView** | ⚠️ BINARY | POST endpoint returns ZIP file (PK header). Would need to download + unzip + parse CSV. Complex but doable as a fallback. |

## Architecture Decision

### What to build (4 sources, 3 tiers of reliability):

**Primary (L0 physics — highest value):**
1. **CDC NWSS Wastewater** — 6 pathogen datasets via Socrata. This IS the tool.

**Secondary (L1-L2 global official):**
2. **WHO DON** — Global outbreak declarations via OData. Parse disease/country from title.
3. **ECDC Open Data** — EU surveillance (cases, variants, hospitalizations).

**Tertiary (supplementary):**
4. **NCBI E-utilities** — Genomic sequence submission velocity as an outbreak proxy.

### What NOT to build:
- ProMED: dead
- GISAID: auth-gated, can't automate
- India/Brazil regional: no machine-readable API
- disease.sh: COVID-only, likely stale, low edge

### Tool modes:

| Mode | Source | What it detects | Layer |
|------|--------|-----------------|-------|
| `wastewater` | CDC NWSS (6 datasets) | Pathogen concentration trends, multi-pathogen co-circulation, geographic spread | L0 |
| `outbreaks` | WHO DON | Global outbreak declarations, novel pathogen alerts, event velocity | L1-L2 |
| `eu_surveillance` | ECDC | EU cases/deaths/variants/hospitalizations trends | L1-L2 |
| `genomics` | NCBI E-utilities | Sequence submission velocity, novel organism detection | L0 |

### Signal taxonomy:

1. **wastewater_surge** — pathogen concentration rising >100% in 15 days in a state/region
2. **multi_state_wave** — same pathogen surging in 5+ states simultaneously
3. **novel_pathogen** — new pathogen detected in wastewater (e.g., H5 avian flu going from 0→positive)
4. **outbreak_declared** — new WHO DON entry (global scope)
5. **variant_takeover** — ECDC variant share crossing 20% threshold in EU
6. **genomic_surge** — sequence submission rate for a pathogen accelerating (>2x 30-day average)
7. **cross_source_corroboration** — wastewater spike + WHO DON + ECDC all flagging same pathogen = highest confidence

## Risks

1. **CDC data is US-only.** Wastewater surveillance in other countries exists (Netherlands, Australia, UK have programs) but no free global API discovered. WHO DON + ECDC provide the global / EU layer.
2. **WHO DON lacks structured metadata.** Must parse disease name and country from title text + body HTML. Regex + keyword matching needed.
3. **NCBI sequence counts are noisy.** Submission patterns depend on lab capacity and funding, not just outbreak severity. Best as a confirming signal, not standalone.
4. **CDC Socrata has no explicit rate limit** but abusive queries could get throttled. Use caching (1-2 hour TTL for wastewater, 6hr for genomics).
5. **Avian H5 dataset (93K records) is the live pandemic watchlist item.** H5N1 spillover to humans is the single highest-impact pandemic risk as of 2026. This dataset detects it in sewage before any hospital reports.

## Data Requirements

- **Pathogen list:** SARS-CoV-2, Influenza A, RSV, Mpox, Measles, Avian Influenza H5 (all from CDC). Additional pathogens from WHO DON (Nipah, Ebola, Marburg, cholera, etc.).
- **Geographic:** US state-level (CDC), global country-level (WHO), EU country-level (ECDC)
- **Temporal:** Weekly (CDC wastewater), as-published (WHO DON), weekly (ECDC)
- **Cache TTL:** Wastewater 2hr (semi-real-time importance), WHO DON 6hr, ECDC 12hr, NCBI 24hr

## Math/Algorithm Survey

- **Trend detection:** 15-day percent change already computed by CDC (`ptc_15d`). For raw concentrations, use rolling z-score vs 90-day baseline.
- **Multi-state wave detection:** Count states with `detect_prop_15d` > 50% AND `ptc_15d` > 50. If count > threshold → alert.
- **Outbreak velocity (WHO):** DON publication frequency for same disease. Hawkes process intensity fitting would be ideal for detecting acceleration.
- **Cross-source corroboration:** When 2+ independent sources flag the same pathogen within the same 7-day window, confidence multiplier applies.
- **Seasonal adjustment:** Compare current percentile rank against same epidemiological week in prior years. Easier for flu/RSV which have strong seasonality.

## Pipeline Integration

Daily DAG node:
1. `fetch_wastewater` — pull latest 7 days from all 6 CDC datasets, store in PipelineStore
2. `fetch_who_don` — pull latest WHO DON entries, check for new DonIds
3. `fetch_ecdc` — pull ECDC COVID/variant/hospital data
4. `check_genomic_velocity` — query NCBI for current submission counts vs 30-day baseline
5. `detect_signals` — run surge/wave/novel/corroboration detection on stored data

Schedule: `0 14 * * *` (2 PM UTC daily — after CDC typically publishes updates)

---

## Related

- [[7b-AE_disease_surveillance_spec|Spec: 7B-Ae Disease Surveillance]]
- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]
