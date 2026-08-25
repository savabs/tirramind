---
title: "Task: Deep Surveillance Phase 10a"
tags:
  - doc/task
  - status/done
  - phase/10
  - topic/surveillance
  - layer/surveillance
---

# Task: Deep Surveillance Phase 10a — Depth Evaluation Framework + Entity Registry

Status: completed
Research: [[deep_surveillance_tools]]
Spec: [[deep_surveillance_tools_spec]]

## Steps

- [x] 10a.1: Add entity tables (entities, entity_aliases, entity_observations) to PipelineStore schema + CRUD methods
- [x] 10a.2: Add depth_evaluations table + store/query methods to PipelineStore
- [x] 10a.3: Create entity name normalization utilities in `agent/pipeline/entity.py`
- [x] 10a.4: Implement SEC company_tickers seed loader in `agent/pipeline/entity.py`
- [x] 10a.5: Implement MI computation module (KSG estimator) in `agent/pipeline/depth_eval.py`
- [x] 10a.6: Implement KL divergence measurement in `agent/pipeline/depth_eval.py`
- [x] 10a.7: Integration test — full loop (seed entities → observations → MI → depth_evaluation)
- [x] 10a.8: Edge case test suite for all Phase 10a code

## Related

- [[deep_surveillance_tools]]
- [[deep_surveillance_tools_spec]]
- [[project_memory]]
