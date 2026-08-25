---
title: "Checkpoint: Phase 20 Signal Fusion Complete"
tags:
  - doc/checkpoint
  - phase/20
  - topic/signal-fusion
  - topic/entity-anomaly
  - topic/micro-alpha
  - layer/fusion
  - layer/feature-engineering
---

# Checkpoint: 2026-04-12 — Phase 20 Signal Fusion Complete

**Session scope**: Completed Steps 20.11–20.13 (final 3 steps of Phase 20). All 13 steps now implemented, tested, and passing.

**Prior context**: Steps 20.1–20.10 were completed across 3 earlier sessions (see conversation summary). This session verified 20.11 and implemented 20.12–20.13.

**Prior checkpoint**: [[chat_checkpoint_2026-04-12_phase19_complete]]

---

## Paradigm (Critical Context for Future Work)

**GNN prediction surprise IS the anomaly signal.** The HetTGN already predicts what happens next to each entity via self-supervised training. Phase 20 extracts that prediction error as the primary anomaly signal. Statistical monitors (CUSUM, Hawkes, EventStudy) are node feature enrichment inputs to the GNN, NOT anomaly detection outputs. **No archetypes anywhere** — permanently removed.

**Five Surprise Signals:**
1. `obs_type_surprise`: -log P(actual_obs_type | h_i) — from existing classification head
2. `temporal_surprise`: |dt_pred - dt_actual| — from existing time-delta head
3. `value_surprise`: |v_pred - v_actual| / σ_type — from NEW value prediction head (added in 20.6)
4. `neighborhood_surprise`: attention-weighted avg of neighbor composite surprise — computed from graph structure
5. `memory_drift`: ||m_t - m_{t-1}||_2 — from HeteroMemory GRU state diff

**Convergence** = correlated prediction surprise across graph neighborhoods (cosine similarity of surprise vectors), NOT hand-coded graph traversal or archetype matching.

---

## What Was Done This Session

### Step 20.11: BeliefState entity_id (verified)
- Added `entity_id: str | None = None` field to `BeliefState` dataclass in `agent/models/belief.py`
- Updated `to_dict()` and `from_dict()` serialization — backward compatible
- **123 existing BeliefState tests pass** (test_world_model_belief: 62, test_world_model_state_filter + test_world_model_edge_cases: 61)

### Step 20.12: Entity Scoring DAG (new)
- **Created**: `agent/pipeline/dags/entity_scoring.py`
  - `DAG_NAME = "entity_scoring"`, `DEPENDS_ON = ["gnn_inference"]`
  - `run_entity_scoring(params, upstream)` — FunctionOperator callback
  - `build_entity_scoring_dag()` — single-node DAG
  - Schedule: weekdays 18:45 UTC (after gnn_inference at 18:30, before feature_generation at 19:00)
  - Graceful skip for: insufficient entities (<5), missing torch, missing model checkpoint, corrupt model
  - Persists `EntityAlert`s and `ConvergenceCluster`s to PipelineStore
  - Individual alert/cluster store failures don't crash the whole run
- **Modified**: `agent/pipeline/dags/__init__.py` — registered `build_entity_scoring_dag` in `get_default_dags()`
- **Created**: `tests/test_entity_scoring_dag.py` — 28 tests
  - TestConstants (2): DAG_NAME, DEPENDS_ON
  - TestDAGStructure (8): builds, name, schedule, node, validates, single node, operator callable, custom params
  - TestPipelineOrdering (2): after gnn_inference, before feature_generation
  - TestSkipBehavior (4): empty store, insufficient entities, no model file, torch unavailable
  - TestModelLoadFailure (1): corrupt model returns error
  - TestSuccessfulScoring (8): alerts stored, clusters stored, empty results, as_of forwarded, config forwarded, result includes as_of, alert store failure continues, cluster store failure continues
  - TestDAGRegistry (2): included in default dags, ordering after gnn_inference
- **28/28 pass, 124 existing DAG tests pass**

### Step 20.13: Integration Tests + Edge Cases (new)
- **Created**: `tests/test_signal_fusion_integration.py` — 28 tests
  - TestFullPipeline (6): basic pipeline, correct fields, linked entities → clusters, no observations, multiple obs types, finite enrichment values
  - TestAntiBias (5): generic convergence (cosine similarity, not archetypes), novel tool combinations detected, cold-start entity scored, no archetype labels on EntityAlert, no archetype on ConvergenceCluster
  - TestEdgeCases (14): zero observations, NaN embeddings handled, stale entity, CUSUM accumulation, CUSUM separate entities, Hawkes decay, Hawkes burst, large entity count (100), single entity no cluster, disconnected entities separate clusters, baseline cold start, baseline detects anomaly, surprise extractor empty IDMap, multiple obs types per entity
  - TestPersistenceRoundTrip (3): alert round-trip, cluster round-trip, alert query filters
- **28/28 pass**

---

## Complete Phase 20 File Inventory

### New Files Created (Phase 20)

| File | Purpose | Tests |
|------|---------|-------|
| `agent/fusion/__init__.py` | Module init + public exports | — |
| `agent/fusion/alert.py` | EntityAlert frozen dataclass (13 data fields) | in convergence cluster tests |
| `agent/fusion/convergence.py` | ConvergenceCluster dataclass + ConvergenceDetector (BFS + cosine similarity) | 25 + 28 tests |
| `agent/fusion/cusum.py` | CUSUM sequential monitor (node feature enrichment) | 20 tests |
| `agent/fusion/hawkes.py` | Hawkes process intensity (node feature enrichment) | 20 tests |
| `agent/fusion/entity_baseline.py` | Event Study baseline (node feature enrichment) | 29 tests |
| `agent/fusion/surprise.py` | SurpriseExtractor — 5 prediction-surprise signals | 40 tests |
| `agent/fusion/entity_scorer.py` | EntityAnomalyScorer orchestrator (full pipeline) | 17 tests |
| `agent/pipeline/dags/entity_scoring.py` | Entity scoring DAG (runs after gnn_inference) | 28 tests |
| `tests/test_surprise.py` | SurpriseExtractor tests | 40 |
| `tests/test_fusion_convergence_detector.py` | ConvergenceDetector tests | 28 |
| `tests/test_entity_scorer.py` | EntityAnomalyScorer tests | 17 |
| `tests/test_store_phase20.py` | PipelineStore entity_alerts + convergence_clusters | 14 |
| `tests/test_entity_scoring_dag.py` | Entity scoring DAG tests | 28 |
| `tests/test_signal_fusion_integration.py` | Full integration + anti-bias + edge cases | 28 |

### Modified Files (Phase 20)

| File | Change |
|------|--------|
| `agent/models/gnn/graph_builder.py` | ENRICHMENT_DIM=27, BASE_FEAT_DIM=12, enrichment-aware `build()` |
| `agent/models/gnn/het_tgn.py` | Added `value_pred_head` MLP + `predict_value()` method |
| `agent/models/gnn/trainer.py` | 4-loss training: obs_type CE (1.0), time_delta MSE (0.1), contrastive (0.5), value Huber (0.3) |
| `agent/pipeline/store.py` | Added `entity_alerts` + `convergence_clusters` tables, 4 new methods |
| `agent/models/belief.py` | Added `entity_id: str | None = None` field + serialization |
| `agent/pipeline/dags/__init__.py` | Registered `build_entity_scoring_dag` |

### Test Counts (Phase 20 total)

| Test File | Count |
|-----------|-------|
| test_convergence_cluster.py | 25 |
| test_surprise.py | 40 |
| test_fusion_convergence_detector.py | 28 |
| test_entity_scorer.py | 17 |
| test_store_phase20.py | 14 |
| test_entity_scoring_dag.py | 28 |
| test_signal_fusion_integration.py | 28 |
| CUSUM tests (earlier session) | 20 |
| Hawkes tests (earlier session) | 20 |
| Entity baseline tests (earlier session) | 29 |
| **TOTAL Phase 20** | **249** |

### Regression Tests Verified

| Test Suite | Count | Status |
|------------|-------|--------|
| test_het_tgn.py + test_graph_builder.py | 110 | PASS |
| test_trainer.py | 27 | PASS |
| test_pipeline_store.py | 56 | PASS |
| test_world_model_belief.py | 62 | PASS |
| test_world_model_state_filter.py + test_world_model_edge_cases.py | 61 | PASS |
| Existing DAG tests (gnn_inference, convergence, expanded, pipeline) | 124 | PASS |

---

## Technical Architecture Summary

### Node Features: 12d → 39d
- Base (12d): entity_type_embed(6) + obs_type_dist(5) + obs_count(1)
- Enrichment (27d): CUSUM(1) + Hawkes(1) + EventStudy(1) + BOCPD(1) + variance(1) + min(1) + max(1) + IQR(1) + num_tools(1) + obs_type_distribution(18)

### HetTGN Architecture
- Per-type projection → HGT convolution (AttentionCapturingHGTConv, 2 layers, 2 heads) → HeteroMemory (GRU + Time2Vec) → 5 prediction heads:
  - `obs_type_head`: Linear → 18 (observation type prediction)
  - `time_delta_head`: MLP → Softplus → 1 (next-event timing)
  - `value_pred_head`: MLP → 1 (observation value prediction) — NEW
  - `link_weight`: bilinear (link prediction)
  - `supervised_head`: bilinear sigmoid (CrystallizedPattern)

### Pipeline DAG Order (weekdays)
```
18:00  daily_collection
18:30  gnn_inference (train/load HetTGN)
18:45  entity_scoring (surprise extraction + convergence detection)  ← NEW
19:00  feature_generation
19:30  world_model_update
```

### Key APIs

- **EntityAnomalyScorer**: `scorer = EntityAnomalyScorer(store, model, config=ScorerConfig()); alerts, clusters = scorer.score_entities(as_of)`
- **SurpriseExtractor**: `extractor.extract(model, data, id_map, observations, memory_before=tensor) → dict[str, EntitySurprise]`
- **ConvergenceDetector**: `detector.detect(entity_surprises, links, surprise_threshold=2.0) → list[ConvergenceCluster]`
- **PipelineStore**: `store.store_entity_alert(...)`, `store.query_entity_alerts(...)`, `store.store_convergence_cluster(...)`, `store.query_convergence_clusters(...)`

---

## Bugs Fixed During Implementation

1. **EntityBaseline.abnormal_score() signature**: Required `(eid, current_value)` not just `(eid)`. Fixed by passing `last_val` from observation loop.
2. **EntityAlert field name mismatch**: ConvergenceDetector used `cusum_enrichment` but EntityAlert has `cusum_statistic`. Fixed immediately.
3. **Mock model __call__ vs side_effect**: `MagicMock.__call__ = MagicMock(side_effect=fn)` doesn't work — must use `model.side_effect = fn`. Fixed in integration tests.
4. **DAG test patch targets**: `run_entity_scoring` does lazy `from X import Y` inside the function body. Must patch `agent.models.gnn.trainer.Trainer` not `agent.pipeline.dags.entity_scoring.Trainer`. Fixed in DAG tests.
5. **Store instance vs class patching**: `run_entity_scoring` creates its own `PipelineStore(db_path)` from params. `patch.object(fixture_store, ...)` doesn't affect the new instance. Fix: `patch.object(PipelineStore, method_name, ...)` patches at class level.
6. **EntityBaseline constant values**: `EntityBaseline` with all identical observations returns `sigma=0 → score=0.0`. Not a bug — correct behavior. Fixed test to use varied baseline values.

---

## What Comes Next

Phase 20 is **complete**. The task file at `[[signal_fusion]]` has `Status: completed` and `status/done` tag. It should be moved to `tasks/done/` and the master tracker `[[quant_training_ground]]` updated.

**Phase 21** (per the project roadmap) is typically the RL policy layer — composite surprise weight learning, action selection, and portfolio optimization. The Phase 20 composite surprise currently uses **equal weights** (configurable via ScorerConfig) — Phase 21's RL policy would learn optimal weights from outcomes.

Before starting Phase 21:
1. Move `[[signal_fusion]]` to `[[signal_fusion]]`
2. Update `[[quant_training_ground]]` master tracker to mark Phase 20 complete
3. Create Phase 21 research/spec/task triad

---

## Session Plan

The full paradigm rationale and step plan are persisted at `/memories/session/plan.md`.

## Related

- [[signal_fusion]] — Research note
- [[signal_fusion_spec]] — Spec (13 atomic steps)
- [[signal_fusion|Task: Signal Fusion]] — Task file (all 13 steps checked off)
- [[quant_training_ground]] — Master task tracker
- [[chat_checkpoint_2026-04-12_phase19_complete]] — Previous checkpoint
