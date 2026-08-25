---
title: "Task: Signal Fusion — Self-Supervised Entity Micro-Alpha"
tags:
  - doc/task
  - status/done
  - phase/20
  - topic/signal-fusion
  - topic/entity-anomaly
  - topic/micro-alpha
  - layer/fusion
  - layer/feature-engineering
---

# Task: Signal Fusion — Phase 20

Status: completed
Research: [[signal_fusion]]
Spec: [[signal_fusion_spec]]

## Steps

- [x] **20.1:** Create fusion module skeleton + EntityAlert + ConvergenceCluster dataclasses + tests
- [x] **20.2:** Implement CUSUM sequential monitor (node feature enrichment role) + edge case tests
- [x] **20.3:** Implement Hawkes process intensity estimator (node feature enrichment role) + edge case tests
- [x] **20.4:** Implement per-entity rolling baseline / Event Study (node feature enrichment role) + edge case tests
- [x] **20.5:** Enrich GraphBuilder node features (12d → ~39d) + update HetTGN in_channels + tests
- [x] **20.6:** Add value_pred_head to HetTGN + value prediction loss in trainer + tests
- [x] **20.7:** Implement SurpriseExtractor (5 prediction-surprise signals) + tests
- [x] **20.8:** Implement ConvergenceDetector (correlated neighborhood surprise) + tests
- [x] **20.9:** Implement EntityAnomalyScorer orchestrator + integration tests
- [x] **20.10:** Add entity_alerts + convergence_clusters storage to PipelineStore + round-trip tests
- [x] **20.11:** Add optional entity_id to BeliefState (backward compat) + tests
- [x] **20.12:** Implement entity scoring DAG + pipeline integration tests
- [x] **20.13:** Full integration test + comprehensive edge case suite (anti-bias tests)

## Notes

- **Paradigm: GNN prediction surprise IS the anomaly signal.** Statistical monitors (CUSUM, Hawkes, EventStudy) are node feature enrichment inputs, NOT anomaly outputs.
- Five surprise signals: obs_type_surprise, temporal_surprise, value_surprise, neighborhood_surprise, memory_drift
- Convergence = correlated prediction surprise across graph neighborhoods (via GNN attention weights + embedding similarity), NOT hand-coded graph traversal
- **No archetypes anywhere** — removed from code, tests, and evaluation permanently
- Existing macro pipeline (aggregates → DAG → Kalman) is NOT modified
- BOCPD runs conditionally (only when CUSUM triggers) to save computation
- Composite surprise weights start equal; learned in Phase 21 via RL
- Hawkes parameters (μ, α, β) are per entity TYPE, not per individual entity
- Minimum 10 observations before entity baseline scoring activates
- All per-entity scores are z-scored for cross-entity comparability

## Related

- [[signal_fusion]] — Research note
- [[signal_fusion_spec]] — Spec (atomic steps)
- [[quant_training_ground]] — Master task file
