---
title: "Task: convergence_backtest_score_cache"
tags:
  - doc/task
  - phase/7c
  - status/active
  - topic/backtest
  - topic/convergence
---

# Task: convergence_backtest_score_cache

Status: completed
Research: [[convergence_backtest_score_cache]]
Spec: [[convergence_backtest_score_cache_spec]]

## Goal
Speed up macro backtests by reusing convergence step scores across targets when timestamps overlap.

## Steps

- [x] 1.1: Add shared timestamp cache to `precompute_convergence_scores()`
- [x] 1.2: Thread one cache through `run_macro_backtest()`
- [x] 1.3: Add focused cache regression tests
- [x] 1.4: Run focused validation and record result

## Verification

- `python -m pytest tests/test_convergence_backtest.py -q` — 64 passed

## Notes

- `precompute_convergence_scores()` now accepts an optional `step_score_cache` keyed by timestamp.
- `run_macro_backtest()` creates one shared cache per run and passes it across target assets.
- The optimization is correctness-preserving because convergence step scores depend on macro history and timestamp, not on target returns.

---

## Related

- [[convergence_backtest_score_cache|Research: Convergence Backtest Score Cache]]
- [[convergence_backtest_score_cache_spec|Spec: Convergence Backtest Score Cache]]
- [[convergence_detection]]
- [[convergence_backtest]]
- [[convergence_template_expansion]]
- [[convergence_signal_expansion]]
- [[convergence_napm_refresh]]
- [[convergence_backtest_fast_mode]]
- [[convergence_template_batch2]]
- [[convergence_audit_pre_worldmodel]]
