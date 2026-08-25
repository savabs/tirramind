---
title: "Task: Tier 3 — Learn Meta-Parameters"
tags:
  - doc/task
  - status/done
  - phase/25
  - topic/learning-agent
  - topic/self-improving
  - layer/learning
  - layer/fusion
---

# Task: Tier 3 — Learn Meta-Parameters

Status: completed
Research: [[learned_vs_handcoded_audit]]
Spec: [[learned_vs_handcoded_architecture_spec]]

## Goal

Move from 45% → 55% learned by replacing three sets of hand-coded meta-parameters with Bayesian optimization + hierarchical bandit:

1. **Change 5** — Reward function weights (5 dims) optimized via GP-BO on portfolio Sharpe
2. **Change 7** — Detector thresholds (CUSUM/Hawkes/convergence) optimized via GP-BO on signal F1
3. **Change 8** — Dynamic goal arm discovery via novel exploration arm + promotion mechanism

## Steps

### Infrastructure: BayesianParamOptimizer (shared by Changes 5 & 7)
- [x] I.1: Create `agent/learning/param_optimizer.py` — RBF kernel GP + Expected Improvement acquisition + trial persistence
- [x] I.2: No extra dependency needed — implemented with numpy/scipy only

### Change 5: Reward weight optimization
- [x] 5.1: Add `RewardWeightOptimizer` class to `agent/learning/reward.py` wrapping BayesianParamOptimizer
- [x] 5.2: `suggest_weights()` → samples from GP posterior, `record_trial(weights, sharpe)` → updates GP
- [x] 5.3: `current_best() -> RewardWeights` for hot-loading into compute_reward

### Change 7: Detector threshold optimization
- [x] 7.1: Create `agent/learning/threshold_optimizer.py` — wraps BayesianParamOptimizer for detector param spaces
- [x] 7.2: Register CUSUM (k, h), Hawkes (μ, α, β), convergence (z, p, fdr_q) as named param spaces
- [x] 7.3: `suggest(detector_name) -> dict`, `record(detector_name, params, metric)`, `current_best(detector_name) -> dict`

### Change 8: Dynamic goal arm discovery
- [x] 8.1: Add `NOVEL_EXPLORATION_ARM` to DEFAULT_ARMS in `agent/learning/bandit.py`
- [x] 8.2: Add `add_arm(GoalArm)` and `record_novel_pull(tools, reward, desc)` to StrategyBandit
- [x] 8.3: Track novel arm success history; after 3+ high-reward pulls with similar tool signature, auto-promote

### Testing
- [x] T.1: Edge case tests for BayesianParamOptimizer (GP math, EI, boundary, persistence) — 14 passed
- [x] T.2: Edge case tests for RewardWeightOptimizer + ThresholdOptimizer — 14 passed
- [x] T.3: Edge case tests for novel arm discovery + promotion — 16 passed
- [x] T.4: Regression: all 73 existing Tier 1+2 tests still pass

### Integration: Wire Optimizers into Live Runtime
- [x] INT.1: Wire `RewardWeightOptimizer` into `AutonomousRunner.__init__` + `run()` — suggest weights before loop, pass to `compute_reward`, record trial after loop
- [x] INT.2: Wire novel arm recording into `AutonomousRunner.run()` — extract `tools_used` from `result.episodes`, call `record_novel_pull()` for `novel_exploration` arm
- [x] INT.3: Wire `ThresholdOptimizer` into `entity_scoring.py` — `_merge_learned_thresholds()` loads CUSUM/Hawkes best params into `ScorerConfig`
- [x] INT.4: Wire `ThresholdOptimizer` into `convergence_detection.py` — `_load_convergence_thresholds()` loads z/p/fdr_q into `ConvergenceDetectorConfig`
- [x] INT.5: Fix `GoalGenerator._arm_fallback_goal()` to handle novel arm (empty tools list)

### Integration Testing
- [x] IT.1: 41 integration tests covering all wiring surfaces — `tests/test_tier3_integration.py`
- [x] IT.2: Regression: all 134 Tier 1+2+3 tests pass (0 failures)

## Related

- [[learned_vs_handcoded_architecture_spec]]
- [[learned_vs_handcoded_audit]]
- [[learned_architecture_impl]]
