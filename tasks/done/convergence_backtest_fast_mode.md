---
title: "Task: convergence_backtest_fast_mode"
tags:
  - doc/task
  - phase/7c
  - status/active
  - topic/backtest
  - topic/convergence
---

# Task: convergence_backtest_fast_mode

Status: completed
Research: [[convergence_backtest_fast_mode]]
Spec: [[convergence_backtest_fast_mode_spec]]

## Goal
Add a reduced-cost fast mode for convergence macro backtests so iteration is cheaper during development.

## Steps

- [x] 1.1: Add bootstrap-count runtime knob to `run_macro_backtest()`
- [x] 1.2: Add CLI fast preset for macro runs
- [x] 1.3: Add focused regression coverage
- [x] 1.4: Run focused validation and record result

## Verification

- `python -m pytest tests/test_convergence_backtest.py -q` — 62 passed

## Notes

- Fast mode is opt-in via `--fast`.
- Full-mode defaults remain unchanged.
- Fast mode currently reduces the default macro run to `SPY`, narrows the default start year from 2010 to 2018, and drops bootstrap resamples from 1000 to 200.
- Explicit user-provided `--targets`, `--start-year`, `--end-year`, and `--bootstrap-count` values are preserved.

---

## Related

- [[convergence_backtest_fast_mode|Research: Convergence Backtest Fast Mode]]
- [[convergence_backtest_fast_mode_spec|Spec: Convergence Backtest Fast Mode]]
- [[convergence_detection]]
- [[convergence_backtest]]
- [[convergence_template_expansion]]
- [[convergence_signal_expansion]]
- [[convergence_napm_refresh]]
- [[convergence_backtest_score_cache]]
- [[convergence_template_batch2]]
- [[convergence_audit_pre_worldmodel]]
