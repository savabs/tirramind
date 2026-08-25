---
title: "Task: GNN ↔ World Model Bridge"
tags:
  - doc/task
  - status/done
  - phase/19
  - topic/world-model
  - topic/surveillance
  - layer/world-model
  - layer/feature-engineering
---

# Task: GNN ↔ World Model Bridge

Status: completed
Research: [[world_model_bridge]]
Spec: [[world_model_bridge_spec]]

## Steps

### Phase 19a: GNN Training on Real Data
- [x] 19a.1: Add Trainer.infer() method returning (embeddings, id_map)
- [x] 19a.2: Add Trainer.save_model() / load_model() persistence
- [x] 19a.3: Write + run GNN real-data training tests (smoke test + persistence round-trip)

### Phase 19b: GNN → Feature Bridge
- [x] 19b.1: Implement GNNFeatureBuilder (anomaly + activity + cross-entity features)
- [x] 19b.2: Implement rolling z-score normalization with store-backed history
- [x] 19b.3: Write + run GNNFeatureBuilder edge case tests (25 tests, all pass)
- [x] 19b.4: Register GNNFeatureBuilder in DEFAULT_BUILDERS

### Phase 19c: Expanded World Model DAG
- [x] 19c.1: Add 11 observed nodes + 8 causal edges + CPDs to initial_graph.py
- [x] 19c.2: Expand Kalman filter dimensions in world_model_update.py (obs_dim 6→17)
- [x] 19c.3: Write + run expanded DAG tests (35 tests, all pass)

### Phase 19d: End-to-End Pipeline
- [x] 19d.1: Create gnn_inference DAG step (train/load + save model)
- [x] 19d.2: Wire GNNFeatureBuilder to use persisted model path
- [x] 19d.3: Write + run pipeline DAG tests (12 tests, all pass)

### Phase 19e: Walk-Forward Backtest
- [x] 19e.1: Implement RegimeStrategy (beliefs → position weights)
- [x] 19e.2: Write + run backtest integration tests (28 tests, all pass)
- [x] 19e.3: Build E2E backtest harness (27 tests, all pass)

## Related

- [[world_model_bridge]]
- [[world_model_bridge_spec]]
- [[entity_linking_layer]]
- [[tier1_tool_expansion]]
- [[gnn_guided_tool_expansion]]
