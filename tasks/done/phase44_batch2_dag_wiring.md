---
title: "Task: Phase 44 — Batch 2 DAG Wiring"
tags:
  - doc/task
  - status/done
  - phase/44
  - topic/pipeline
  - topic/diversity
  - layer/surveillance
---

# Task: Phase 44 — Batch 2 DAG Wiring

Status: completed
Research: [[phase44_batch2_dag_wiring]]
Spec: [[phase44_batch2_dag_wiring_spec]]

## Steps

- [x] 44.1: Add 5 Phase 44 nodes to `agent/pipeline/dags/daily_collection.py`
- [x] 44.2: Update 3 count assertions in `tests/test_pipeline_registry.py` (22→27)
- [x] 44.3: Add 5 per-node config tests to `tests/test_pipeline_registry.py`
- [x] 44.4: Run `pytest tests/test_pipeline_registry.py -v` — 53 tests pass (48→53)
- [x] 44.5: Full regression — 0 new failures; comtrade.py bug (`None[:50]`) fixed as bonus (22 baseline failures, down from 23)
- [x] bonus: Fix pre-existing `comtrade.py:407` TypeError (`r.get('commodity', 'N/A')[:50]` → `(r.get('commodity') or 'N/A')[:50]`)

## Related

- [[phase44_batch2_dag_wiring]]
- [[phase44_batch2_dag_wiring_spec]]
- [[phase43_high_volume_dag_wiring]]
