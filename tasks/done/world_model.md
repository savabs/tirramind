---
title: "Task: World Model (Phase 9)"
tags:
  - doc/task
  - layer/fusion
  - layer/world-model
  - phase/9
  - status/active
  - topic/world-model
---

# Task: World Model (Phase 9)

Status: completed
Research: [[world_model]]
Spec: [[world_model_spec]]

## Overview

Build the probabilistic causal engine: Bayesian network DAG (pgmpy) + Kalman filter (filterpy) to maintain belief state over hidden economic variables. Outputs posterior distributions for Phase 10 signal fusion.

Hybrid architecture: discrete causal DAG for regime/latent variables + continuous state filter for quantitative dynamics. Expert-specified initial graph with 9 nodes (3 latent/regime + 6 observed mapped to EngineeredFeatures).

## Steps

### Sub-phase 9.1: Belief Protocol + Persistence

- [x] **9.1.1: Define BeliefState dataclass + validate_belief** — 62 tests ✅
- [x] **9.1.2: Add beliefs table to PipelineStore** — integration test ✅

### Sub-phase 9.2: Graph Structure + Expert DAG

- [x] **9.2.1: Implement WorldModelGraph + NodeSpec** — 29 tests ✅
- [x] **9.2.2: Define initial expert DAG** — 52 tests ✅

### Sub-phase 9.3: Belief Propagation Engine

- [x] **9.3.1: Implement BeliefPropagator** — 28 tests ✅

### Sub-phase 9.4: Continuous State Filter

- [x] **9.4.1: Implement ContinuousStateFilter** — 28 tests ✅

### Sub-phase 9.5: World Model Orchestrator

- [x] **9.5.1: Implement WorldModel class** — 17 tests ✅

### Sub-phase 9.6: Intervention Engine

- [x] **9.6.1: Implement InterventionEngine** — 19 tests ✅

### Sub-phase 9.7: Causal Structure Discovery

- [x] **9.7.1: Implement CausalStructureDiscovery** — 13 tests ✅

### Sub-phase 9.8: Pipeline DAG Integration

- [x] **9.8.1: Create world model update DAG** — 8 tests ✅
- [x] **9.8.2: Add dependencies** — tigramite optional; pgmpy+filterpy already in quant ✅

### Sub-phase 9.9: Edge Case Test Suite

- [x] **9.9.1: Comprehensive edge case tests** — 33 tests ✅ (graph cycles, disconnected components, hash stability, quality weighting, latent evidence, precision, large/small obs, covariance PD over 1000 steps, dimension mismatch, mixed quality, old features, None values, interventions on every node)

---

## Related

- [[world_model|Research: World Model]]
- [[world_model_spec|Spec: World Model]]
- [[convergence_detection]]
- [[signal_protocol_feature_engineering]]
- [[rl_layer]]
