---
title: "Spec: 7b-AE Disease & Pandemic Surveillance"
tags:
  - doc/spec
  - layer/surveillance
  - phase/7b
  - topic/disease-surveillance
---

# Spec: 7b-AE Disease & Pandemic Surveillance

## Goal

Build `DiseaseSurveillanceTool` — 4-mode tool covering wastewater pathogen concentrations (CDC NWSS), global outbreak declarations (WHO DON), EU surveillance (ECDC), and genomic sequence velocity (NCBI). Primary L0 source is CDC wastewater: physics that can't be faked, delayed, or retracted.

## Files Affected

1. **Create:** `agent/tools/disease_surveillance.py` — main tool (~500 lines)
2. **Modify:** `agent/cli.py` — register tool
3. **Modify:** `agent/learning/bandit.py` — add `pandemic_surveillance` arm
4. **Create:** `tests/test_disease_surveillance_edge.py` — comprehensive edge case tests

## Implementation Steps

### 2.1: Create tool skeleton
- Class `DiseaseSurveillanceTool(Tool)` with name/description/parameters
- 4 modes: `wastewater`, `outbreaks`, `eu_surveillance`, `genomics`
- Constructor takes `cache: DataCache | None = None`
- `execute()` dispatches to `_execute_wastewater()`, `_execute_outbreaks()`, `_execute_eu_surveillance()`, `_execute_genomics()`

### 2.2: Implement `wastewater` mode (CDC NWSS Socrata)
- 6 pathogen dataset IDs: j9g8-acpt (SARS-CoV-2), ymmh-divb (Flu A), 45cq-cw4i (RSV), xpxn-rzgz (Mpox), akvg-8vrb (Measles), mtpu-urpp (Avian H5)
- Aggregated metrics: 2ew6-ywp6
- URL: `https://data.cdc.gov/resource/{id}.json`
- Params: `pathogen` (default all), `state` filter, `days_back` (default 30), `limit` (default 100, max 1000)
- Fetch aggregate metrics (ptc_15d, detect_prop_15d, percentile) for trend signal
- For specific pathogen: fetch raw concentration data, compute summary stats per state
- Cache TTL: 7200s (2hr)

### 2.3: Implement `outbreaks` mode (WHO DON)
- URL: `https://www.who.int/api/hubs/diseaseoutbreaknews`
- OData v4: `$top`, `$orderby=PublicationDate desc`, `$filter`
- Parse disease/country from Title (regex: "Disease - Country" pattern)
- Params: `disease` filter (keyword match in title), `limit` (default 25)
- Cache TTL: 21600s (6hr)

### 2.4: Implement `eu_surveillance` mode (ECDC)
- 3 datasets: nationalcasedeath, virusvariant, hospitalicuadmissionrates
- URL: `https://opendata.ecdc.europa.eu/covid19/{dataset}/json`
- Params: `dataset` (default nationalcasedeath), `country` filter, `weeks` (default 12)
- Cache TTL: 43200s (12hr)

### 2.5: Implement `genomics` mode (NCBI E-utilities) 
- URL: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi`
- Params: organism (default SARS-CoV-2), period (current year vs prior), retmode=json
- Compare current count vs baseline → velocity signal
- Cache TTL: 86400s (24hr)

### 2.6: Register + bandit arm
- Import + register in cli.py: `registry.register(DiseaseSurveillanceTool(cache=cache))`
- Add GoalArm `pandemic_surveillance` with tools: [disease_surveillance, weather_alerts, web_search]

### 2.7: Edge case tests
- All 4 modes: normal, empty, HTTP errors, malformed JSON, cache hits/misses
- Parameter validation: invalid mode, bad pathogen, bad state code, limits
- Parsing: WHO title regex, ECDC filters, NCBI response format
- Tool metadata, registry integration, bandit arm

## Edge Cases

- CDC Socrata returns empty for future dates or non-existent pathogen
- WHO DON title parsing: some titles don't match "Disease - Country" pattern
- ECDC dataset returns entire dataset at once (could be large) — slice to recent weeks
- NCBI rate limit (3/sec) — single call per invocation, cache aggressively
- HTTP 429/500 from any source — graceful failure with informative message
- Cache miss on first call — must handle None from cache.get()

## Testing Plan

Mock all HTTP calls via `unittest.mock.patch` on `httpx.Client.get`. Test each mode independently. Verify parameter clamping, error messages, data formatting, cache key construction, signal computation.

---

## Related

- [[7b-AE_disease_surveillance|Research: 7B-Ae Disease Surveillance]]
- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]
