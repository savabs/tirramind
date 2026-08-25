---
title: "Checkpoint: Phase 31 Complete — Remaining Country Signals"
tags:
  - doc/checkpoint
  - phase/31
  - topic/entity-linking
  - topic/consumer-sentiment
  - topic/food-security
  - topic/internet-infrastructure
  - topic/migration
  - layer/surveillance
  - layer/world-model
---

# Checkpoint: Phase 31 Complete

## Summary

Phase 31 added L2 country-node persistence to four existing surveillance tools:

- `consumer_sentiment` → `consumer_confidence`
- `food_security` → `food_security`
- `internet_outages` → `internet_disruption`
- `migration_flows` → `migration_pressure`

Country nodes now absorb household demand, agricultural stress, network reliability, and migration-pressure evidence directly in the entity graph.

Graph-builder updates:

- `OBSERVATION_TYPES`: 35 → 39
- `ENRICHMENT_DIM`: 44 → 48

## Files Modified

### Tool files

- `agent/tools/consumer_sentiment.py`
- `agent/tools/food_security.py`
- `agent/tools/internet_outages.py`
- `agent/tools/migration_flows.py`

### Graph builder

- `agent/models/gnn/graph_builder.py`

### Tests

- `tests/test_phase31_country_signal_l2.py` (new)
- `tests/test_phase31_diagnostic.py` (new)
- `tests/test_graph_builder_expanded.py`
- `tests/test_phase28_diagnostic.py`
- `tests/test_phase29_diagnostic.py`

## Validation

Focused regression suite:

- `tests/test_phase31_country_signal_l2.py`
- `tests/test_phase31_diagnostic.py`
- `tests/test_graph_builder_expanded.py`
- `tests/test_phase28_diagnostic.py`
- `tests/test_phase29_diagnostic.py`
- `tests/test_consumer_sentiment_edge.py`
- `tests/test_food_security_edge.py`
- `tests/test_internet_outages_edge.py`
- `tests/test_migration_flows_edge.py`

Result: **398 passed**.

## Notes

- EU / global aggregate pseudo-countries are intentionally skipped (`EU27_2020`, `EA20`, `WLD`, `ALL`).
- Migration persistence normalizes ISO-3 inputs to ISO-2 before country-entity registration.
- Unrelated uncommitted Phase 30 files remained in the working tree during this session and were left untouched.

## Related

- [[phase31_remaining_country_signals]]
- [[phase31_remaining_country_signals_spec]]
- [[quant_training_ground]]
- [[chat_checkpoint_2026-04-16_phase29_complete]]