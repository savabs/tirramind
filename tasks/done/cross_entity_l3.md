---
title: "Task: L3 Cross-Entity Pattern Detection — Phase 11"
tags:
  - doc/task
  - status/done
  - phase/11
  - topic/surveillance
  - topic/convergence
  - layer/feature-engineering
---

# Task: L3 Cross-Entity Pattern Detection — Phase 11

Status: completed
Research: [[cross_entity_l3]]
Spec: [[cross_entity_l3_spec]]

## Phase 11a: Cross-Entity Infrastructure

- [x] 11a.1: Add `entity_links` table to PipelineStore schema (DDL + migration)
- [x] 11a.2: Add `link_entities()` method to PipelineStore (INSERT OR IGNORE, idempotent)
- [x] 11a.3: Add `query_entity_links()` method (direction filter, confidence threshold)
- [x] 11a.4: Add `query_co_occurrences()` method (temporal join across entity observation streams)
- [x] 11a.5: Edge case test suite for entity_links + co-occurrences (32 tests)

## Phase 11b: First L3 Pattern (Insider × GDELT)

- [x] 11b.1: Create `agent/pipeline/cross_entity.py` — CrossEntityDetector class
- [x] 11b.2: Implement company→country link seeder (SEC ticker data → `headquartered_in` links)
- [x] 11b.3: Implement Insider × GDELT co-occurrence detector
- [x] 11b.4: Implement L3 observation storage (depth_level=3, `cross_entity_pattern` type)
- [x] 11b.5: Edge case test suite for cross_entity module (26 tests)

## Notes

- L3 = cross-domain entity combinations. The signal is in the junction between data sources, not in any single source.
- Three concrete patterns identified in research: Insider×GDELT, Vessel×Sanctions, Crypto×Geopolitical. Starting with Insider×GDELT because company→country links have highest coverage.
- Key infrastructure gap: no `entity_links` table exists yet. Must build that first.
- Co-occurrence query is a temporal self-join across entity_observations. Window default: 72h (accommodates SEC T+2 disclosure lag).
- Scoring uses existing conditional MI estimator in `depth_eval.py`.
- This phase does NOT add new data tools — it mines cross-domain patterns from data already being collected by L2 tools.

## Related

- [[cross_entity_l3]]
- [[cross_entity_l3_spec]]
- [[gdelt_l2]]
- [[ais_vessel_l2]]
- [[whale_alert_l2]]
- [[deep_surveillance_10b]]
