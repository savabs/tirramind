---
title: "Task: Phase 28 — Country Node Macro Enrichment"
tags:
  - doc/task
  - status/done
  - phase/28
  - topic/entity-linking
  - topic/graph-connectivity
  - layer/surveillance
  - layer/world-model
---

# Task: Phase 28 — Country Node Macro Enrichment

Status: completed
Research: [[phase28_country_macro_enrichment]]
Spec: [[phase28_country_macro_enrichment_spec]]

## Goal

Add PipelineStore L2 persistence to `sovereign_debt`, `capital_flows`, and `global_pmi`. Country nodes go from 3 → 6 observation types.

## Steps

- [x] 28.1: Add PipelineStore to `sovereign_debt`, persist `sovereign_yield` on country entities
- [x] 28.2: Add PipelineStore to `capital_flows`, persist `capital_flow` on country entities
- [x] 28.3: Add PipelineStore to `global_pmi`, persist `economic_activity` on country entities
- [x] 28.4: Register 3 new obs types in graph builder, update ENRICHMENT_DIM 38→41
- [x] 28.5: Edge case tests for all 3 tools + graph builder
- [x] 28.6: Integration diagnostics
- [x] 28.7: Regression + checkpoint

## Related

- [[phase28_country_macro_enrichment]]
- [[phase28_country_macro_enrichment_spec]]
- [[starved_class_audit]]
- [[l2_expansion_roadmap]]
- [[phase27_fx_country_monetary_linking]]
