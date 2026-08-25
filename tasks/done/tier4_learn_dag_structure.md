---
title: "Task: Implement Causal DAG Structure Learning (Change 3)"
tags:
  - doc/task
  - status/done
  - phase/25
  - topic/self-improving
  - topic/structure-learning
  - layer/world-model
---

# Task: Implement Causal DAG Structure Learning (Change 3)

Status: completed
Research: [[causal_dag_structure_learning]]
Spec: [[learned_vs_handcoded_architecture_spec]]

## Steps

### 3.1: Add `remove_edge()` to WorldModelGraph
- [x] 3.1.1: Add `remove_edge(parent, child)` method with validation
- [x] 3.1.2: Add `has_edge(parent, child)` convenience method

### 3.2: Implement `refine_structure()` on WorldModel
- [x] 3.2.1: Build discretized DataFrame (reuse `_discretize` logic from `fit_cpds`)
- [x] 3.2.2: Extract current graph into pgmpy `DAG` for warm start
- [x] 3.2.3: Build `ExpertKnowledge` constraints (forbidden edges, temporal order)
- [x] 3.2.4: Run `HillClimbSearch.fit(df)` with `bic-d` scoring
- [x] 3.2.5: Diff learned edges vs current → compute added/removed
- [x] 3.2.6: Apply changes to `WorldModelGraph` (add/remove edges)
- [x] 3.2.7: Return audit dict with change details + BIC scores

### 3.3: Wire into world_model_update DAG
- [x] 3.3.1: Add `_maybe_refine_structure()` helper (quarterly schedule, 90-day interval)
- [x] 3.3.2: Call before `_maybe_fit_params()` in `run_world_model_update()`
- [x] 3.3.3: Add `structure_fit_enabled` and `structure_fit_interval_days` params

### 3.4: Edge-case test suite
- [x] 3.4.1: Synthetic data with known structure → verify edge recovery (F1 > 80%)
- [x] 3.4.2: Constraint enforcement tests (regime roots, max indegree, forbidden edges)
- [x] 3.4.3: BIC improvement validation (reject spurious edges)
- [x] 3.4.4: remove_edge / has_edge tests
- [x] 3.4.5: Minimum sample guard (skip if < min_samples)
- [x] 3.4.6: Empty/degenerate data handling
- [x] 3.4.7: Integration test with world_model_update DAG wiring

## Related

- [[causal_dag_structure_learning]]
- [[learned_vs_handcoded_architecture_spec]]
- [[learned_architecture_impl]]
