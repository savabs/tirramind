---
title: "Task: Phase 43 — High-Volume DAG Wiring"
tags:
  - doc/task
  - status/done
  - phase/43
  - topic/pipeline
  - topic/diversity
  - layer/surveillance
---

# Task: Phase 43 — High-Volume DAG Wiring

Status: completed
Research: [[phase43_high_volume_dag_wiring]]
Spec: [[phase43_high_volume_dag_wiring_spec]]

## Steps

- [x] 43.1: Add `fetch_ais_vessel` node to `daily_collection.py`
- [x] 43.2: Add `fetch_gov_contracts` node to `daily_collection.py`
- [x] 43.3: Add `fetch_sanctions_monitor` node to `daily_collection.py`
- [x] 43.4: Add `fetch_patent_filings` node to `daily_collection.py`
- [x] 43.5: Update `test_pipeline_registry.py` (3 count assertions + 4 node config tests) — 49/49 PASS
- [x] 43.6: Full regression — 0 new failures (pre-existing feature_generation_dag/entity_linking excluded per Phase 29 doc)
- [x] Bonus: Fix `test_defi_flows_edge.py` cache.set→cache.put (Phase 42 artifact)

## Related

- [[phase43_high_volume_dag_wiring]]
- [[phase43_high_volume_dag_wiring_spec]]
- [[chat_checkpoint_2026-04-21_phase42_complete]]
