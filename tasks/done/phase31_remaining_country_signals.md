---
title: "Task: Phase 31 — Remaining Country Signals"
tags:
  - doc/task
  - status/done
  - phase/31
  - topic/entity-linking
  - topic/consumer-sentiment
  - topic/food-security
  - topic/internet-infrastructure
  - topic/migration
  - layer/surveillance
  - layer/world-model
---

# Task: Phase 31 — Remaining Country Signals

Status: completed
Research: [[phase31_remaining_country_signals]]
Spec: [[phase31_remaining_country_signals_spec]]

## Goal

Add PipelineStore L2 persistence to `consumer_sentiment`, `food_security`, `internet_outages`, and `migration_flows`. Country nodes gain four new observation types.

## Steps

- [x] 31.1: Add L2 persistence to `consumer_sentiment`
- [x] 31.2: Add L2 persistence to `food_security`
- [x] 31.3: Add L2 persistence to `internet_outages`
- [x] 31.4: Add L2 persistence to `migration_flows`
- [x] 31.5: Register 4 new obs types in graph builder, update `ENRICHMENT_DIM` 44→48
- [x] 31.6: Add Phase 31 edge-case tests
- [x] 31.7: Add Phase 31 integration diagnostics and targeted regression

## Related

- [[phase31_remaining_country_signals]]
- [[phase31_remaining_country_signals_spec]]
- [[phase28_country_macro_enrichment]]
- [[starved_class_audit]]
- [[l2_expansion_roadmap]]
- [[chat_checkpoint_2026-04-17_phase31_complete]]