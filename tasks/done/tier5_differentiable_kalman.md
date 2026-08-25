---
title: "Task: Implement Differentiable Kalman Filter (Change 10c)"
tags:
  - doc/task
  - status/done
  - phase/25
  - topic/self-improving
  - topic/differentiable
  - layer/fusion
---

# Task: Implement Differentiable Kalman Filter (Change 10c)

Status: completed
Research: [[differentiable_kalman]]
Spec: [[learned_vs_handcoded_architecture_spec]]

## Steps

### 10c.1: Implement DifferentiableKalmanFilter nn.Module
- [x] 10c.1.1: Create `agent/models/diff_kalman.py` with `DifferentiableKalmanFilter(nn.Module)`
- [x] 10c.1.2: Cholesky PSD parameterization for Q and R (L @ L^T + ε·I) with inverse_softplus import
- [x] 10c.1.3: Per-regime F, L_Q as nn.ParameterDict
- [x] 10c.1.4: Shared H, L_R as nn.Parameter
- [x] 10c.1.5: predict(regime) in torch preserving autograd
- [x] 10c.1.6: update(observations, quality) with NaN masking in torch + numpy auto-conversion
- [x] 10c.1.7: get_beliefs() converting torch state → BeliefState list
- [x] 10c.1.8: reset(x0, P0) with proper detach for multi-iteration optimization

### 10c.2: Conversion and integration
- [x] 10c.2.1: from_numpy_filter(ContinuousStateFilter) class method — import params via _psd_to_cholesky_param
- [x] 10c.2.2: to_numpy_params() — export back to numpy for EM fitting
- [x] 10c.2.3: Wire into _build_world_model() as optional backend (use_differentiable_filter flag)
- [x] 10c.2.4: WorldModel accepts Union[ContinuousStateFilter, DifferentiableKalmanFilter]
- [x] 10c.2.5: _regime_configs property for WorldModel compatibility
- [x] 10c.2.6: EM fitting with diff filter (temporary numpy filter → EM → transfer back)

### 10c.3: Edge-case test suite (66 tests)
- [x] 10c.3.1: Construction, dims, parameter count, buffer/param separation (6 tests)
- [x] 10c.3.2: PSD enforcement — random init, post-gradient, zero input (5 tests)
- [x] 10c.3.3: Forward shapes — predict/update with torch/numpy, custom dims (5 tests)
- [x] 10c.3.4: Gradient flow — predict, update, chain, regime isolation, numpy update (5 tests)
- [x] 10c.3.5: NaN masking — all NaN, partial, single valid, zero quality (4 tests)
- [x] 10c.3.6: Numerical equivalence with numpy filter (3 tests)
- [x] 10c.3.7: Regime switching — different states, invalid regime, mid-sequence (3 tests)
- [x] 10c.3.8: Save/load state_dict round-trip (2 tests)
- [x] 10c.3.9: from_numpy_filter conversion — dims, names, F, H, Q, R, state (7 tests)
- [x] 10c.3.10: to_numpy_params export (2 tests)
- [x] 10c.3.11: Reset with numpy/torch/defaults (3 tests)
- [x] 10c.3.12: get_beliefs — count, names, dist_type, values, wrong count (5 tests)
- [x] 10c.3.13: Parameter optimization — SGD loss reduction, PSD after many steps (2 tests)
- [x] 10c.3.14: WorldModel integration — update with diff filter, regime fallback (2 tests)
- [x] 10c.3.15: DAG _build_world_model — numpy default, diff flag, learned edges, param match (4 tests)
- [x] 10c.3.16: Edge cases — wrong shape, double predict, update w/o predict, large obs, detached clone, single regime, negative quality, symmetric P (8 tests)

## Related

- [[differentiable_kalman]]
- [[learned_vs_handcoded_architecture_spec]]
- [[tier4_learned_state_encoder]]
- [[learned_architecture_impl]]
