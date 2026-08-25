---
title: "Research: Academic Preprints (7b-M)"
tags:
  - doc/research
  - layer/surveillance
  - phase/7b
---

# Research: Academic Preprints (7b-M)

**Date:** 2026-03-28 (retroactive documentation)
**Tool:** `agent/tools/academic_preprints.py` → `AcademicPreprintsTool`
**Status:** IMPLEMENTED, TESTED

## APIs Probed

### arXiv API ✅ SELECTED
- **URL:** `https://export.arxiv.org/api/query`
- **Method:** GET (requires `follow_redirects=True` — HTTP→HTTPS redirect)
- **Auth:** None
- **Format:** Atom XML (namespace `http://www.w3.org/2005/Atom`)
- **Rate limits:** Undocumented, polite use expected
- **Coverage:** **Global** — all physics, CS, math, biology, economics, finance papers worldwide
- **Features:** Full-text search, category filters, date sorting, pagination via `start` + `max_results`

### ClinicalTrials.gov v2 API ✅ SELECTED
- **URL:** `https://clinicaltrials.gov/api/v2/studies`
- **Method:** GET
- **Auth:** None
- **Format:** JSON (nested `protocolSection` structure)
- **Rate limits:** Undocumented
- **Coverage:** **Global** — 220+ countries, though US-centric by volume. Indexes trials from all WHO ICTRP registries.
- **Features:** Keyword search, status filter, pageSize up to 100, pageToken pagination

## Geographic Coverage
- arXiv: Fully global. 300+ institutions. Language: primarily English but all-country submissions.
- ClinicalTrials: Global registry (WHO ICTRP partner). ~450K studies. US-heavy (~60%) but international trials indexed.
- **Verdict:** `[G:GLOBAL]`

## Modes Implemented
1. `papers` — arXiv keyword search with market-relevant category filtering
2. `trending` — recent arXiv papers in market-relevant categories (q-fin, cs.AI, cs.LG, cs.CR, cs.CL, econ, stat.ML, physics.soc-ph)
3. `trials` — ClinicalTrials.gov search with status filtering

## Signal Value
- Research preprints precede product launches by 6-24 months
- Clinical trial registrations precede FDA decisions by years
- Surge in papers on a topic = emerging technology trend
- Trial phase transitions = pharma pipeline signal

## Risks
- arXiv XML parsing fragile (namespace-dependent)
- ClinicalTrials JSON structure may change between API versions
- No rate limit info — could be silently throttled

---

## Related

- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]
