---
title: "Task: Phase 29 — Company + Investigative L2"
tags:
  - doc/task
  - status/done
  - phase/29
  - topic/entity-linking
  - topic/bankruptcy
  - topic/foia
  - topic/academic-preprints
  - layer/surveillance
  - layer/world-model
---

# Task: Phase 29 — Company + Investigative L2

Status: completed
Research: [[phase29_company_investigative_l2]]
Spec: [[phase29_company_investigative_l2_spec]]

## Steps

- [x] 29.1: Add L2 persistence to `bankruptcy_court` tool + 23 edge case tests
- [x] 29.2: Add L2 persistence to `foia_requests` tool + 14 edge case tests
- [x] 29.3: Add L2 persistence to `academic_preprints` tool + 21 edge case tests
- [x] 29.4: Add `bankruptcy_status`, `investigation_signal`, `research_velocity` to graph_builder OBSERVATION_TYPES
- [x] 29.5: Integration diagnostic tests (18 cross-tool tests)
- [x] 29.6: Full regression (4483/4484 pass, 1 pre-existing), stale ENRICHMENT_DIM fixed

## Notes

- L2 pattern reference: `insider_filings.py`, `consumer_sentiment.py` (recent Phase 27–28 upgrades)
- OBSERVATION_TYPES: 32 → 35
- ENRICHMENT_DIM: 41 → 44
- Entity types touched: company, person, topic
- Each tool's `_persist_entities()` uses None-guard + try/except non-fatal pattern

## Related

- [[phase29_company_investigative_l2]]
- [[phase29_company_investigative_l2_spec]]
- [[phase28_country_macro_enrichment]]
- [[7b-E_bankruptcy_court]]
- [[7b-S_foia_logs]]
- [[7b-M_academic_preprints]]
