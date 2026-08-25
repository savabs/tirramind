---
title: "Task: Phase 38 — Downstream Pipeline Integration"
tags:
  - doc/task
  - status/done
  - phase/38
  - topic/pipeline
  - topic/convergence
  - layer/feature-engineering
  - layer/surveillance
---

# Task: Phase 38 — Downstream Pipeline Integration

Status: completed
Research: [[phase38_downstream_pipeline_integration]]
Spec: [[phase38_downstream_pipeline_integration_spec]]

## Steps

- [x] 38.1: Add `table_name` to all tool nodes in `build_daily_collection_dag()` so source names match convergence extractor registry
- [x] 38.2: Add `fetch_macro` node to daily_collection DAG (macro_data tool, DFF/GS10/GS2/WALCL series)
- [x] 38.3: Write `tests/test_phase38_pipeline_integration.py` — source name validation, mock convergence evidence, mock feature generation
- [x] 38.4: Fix any test regressions from DAG structure changes
- [x] 38.5: Run full test suite, verify green

## Related

- [[phase38_downstream_pipeline_integration]]
- [[phase38_downstream_pipeline_integration_spec]]
- [[phase37_first_live_pipeline]]
- [[quant_training_ground]]
- [[chat_checkpoint_2026-04-19_phase37_complete]]
