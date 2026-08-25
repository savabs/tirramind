---
title: "Spec: Convergence Backtest Fast Mode"
tags:
  - doc/spec
  - phase/7c
  - topic/backtest
  - topic/convergence
---

# Spec: Convergence Backtest Fast Mode

## Goal
Add a reduced-cost fast mode to the convergence backtest CLI so development runs are cheaper, while keeping full-mode defaults unchanged.

## Files Affected
- `agent/convergence/backtest.py` — add runtime knobs and CLI fast preset
- `tests/test_convergence_backtest.py` — add focused regression coverage
- `[[convergence_backtest_fast_mode]]` — track progress

## Implementation Steps
1. Add a `bootstrap_count` parameter to `run_macro_backtest()` and use it where bootstrap CI is computed.
2. Add a CLI `--fast` flag that applies a reduced-cost preset for macro runs without changing default behavior.
3. Add targeted tests covering the bootstrap parameter and CLI fast preset behavior.
4. Run focused validation and record results.

## Edge Cases
- `bootstrap_count` should not break existing callers when omitted.
- Fast mode should only affect macro/validate/save-baseline execution paths.
- Explicit user-provided targets/date range should remain usable alongside fast mode.

## Testing Plan
- Add unit tests for `run_macro_backtest()` bootstrap parameter plumb-through using mocks.
- Add CLI tests that patch `run_macro_backtest()` and verify fast-mode kwargs.
- Run the convergence backtest test module.

---

## Related

- [[convergence_backtest_fast_mode|Research: Convergence Backtest Fast Mode]]
- [[convergence_detection]]
- [[convergence_backtest]]
- [[convergence_template_expansion]]
- [[convergence_signal_expansion]]
- [[convergence_napm_refresh]]
- [[convergence_backtest_score_cache]]
- [[convergence_template_batch2]]
- [[convergence_audit_pre_worldmodel]]
