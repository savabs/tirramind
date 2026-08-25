---
title: "Task: Tier 6 — Learned Feature Selection & Tool Routing"
tags:
  - doc/task
  - status/done
  - phase/25
  - topic/self-improving
  - topic/feature-selection
  - topic/tool-routing
  - layer/learning
  - layer/surveillance
---

# Task: Tier 6 — Learned Feature Selection & Tool Routing

Status: completed
Research: [[tier6_learned_observation]]
Spec: [[tier6_learned_observation_spec]]

## Goal

Move from 75% → 82% learned by implementing Changes 11 (learned feature selection) and 12 (learned tool routing).

## Steps

### Change 11: FeatureGate nn.Module
- [x] 11.1: Create `agent/learning/policy/feature_gate.py` — `FeatureGateConfig` + `FeatureGate(nn.Module)` with regime-conditioned soft gating, gate floor, entropy regularization
- [x] 11.2: Add `FeatureGateConfig` to `PolicyConfig` in `config.py` (default None, backward compat)
- [x] 11.3: Wire FeatureGate into `LearnedStateEncoder.forward()` — gate before entity parsing
- [x] 11.4: Wire into `rl_training.py` — create gate, add entropy loss to actor loss, checkpoint
- [x] 11.5: Add diagnostic output method `gate_diagnostics()`

### Change 11: Tests
- [x] T11.1: Edge case tests for FeatureGate (shape, gradients, entropy, floor, NaN, save/load) — 36 tests passing

### Change 12: ToolRoutingBandit
- [x] 12.1: Create `agent/learning/tool_router.py` — `ToolRoutingBandit` with contextual Thompson Sampling
- [x] 12.2: Integrate with `daily_collection.py` — conditional tool execution via `enabled` flag
- [x] 12.3: Add `enabled` flag to `Node` in `dag.py`, skip disabled nodes in `DAGExecutor`
- [x] 12.4: Record tool outcomes after DAG completion (signal contribution metric)
- [x] 12.5: Persistence (save/load to JSON) and cold-start behavior

### Change 12: Tests
- [x] T12.1: Edge case tests for ToolRoutingBandit (cold start, convergence, always-on, persistence, exploration) — 25 tests passing

### Integration
- [x] INT.1: Regression — 156 tests pass across feature_gate, tool_router, sac, learned_architecture, rl_edge_cases

## Related

- [[tier6_learned_observation]]
- [[tier6_learned_observation_spec]]
- [[learned_vs_handcoded_architecture_spec]]
- [[learned_architecture_impl]]
- [[tier5_differentiable_kalman]]
