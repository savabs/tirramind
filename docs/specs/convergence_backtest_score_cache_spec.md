---
title: "Spec: Convergence Backtest Score Cache"
tags:
  - doc/spec
  - phase/7c
  - topic/backtest
  - topic/convergence
---

# Spec: Convergence Backtest Score Cache

## Goal
Eliminate redundant convergence detector work in the macro backtest by caching step scores across targets within a single run.

## Files Affected
- `agent/convergence/backtest.py` — add shared timestamp cache plumbing
- `tests/test_convergence_backtest.py` — add focused cache regression tests
- `[[convergence_backtest_score_cache]]` — track progress

## Implementation Steps
1. Add an optional `step_score_cache` parameter to `precompute_convergence_scores()` keyed by timestamp.
2. Reuse cached `StepScore` values for repeated timestamps before rebuilding evidence or running the detector.
3. Create a shared cache in `run_macro_backtest()` and pass it to each target's score precompute call.
4. Add focused regression tests:
   - repeated timestamps reuse cached results inside `precompute_convergence_scores()`
   - `run_macro_backtest()` passes one shared cache across targets
5. Run focused validation.

## Edge Cases
- Empty or non-overlapping timestamp arrays should still work normally.
- Cache hits must preserve output ordering relative to the requested timestamp array.
- The optimization must not alter default backtest results.

## Testing Plan
- Patch detector calls to prove cached timestamps avoid re-execution.
- Patch `precompute_convergence_scores()` at the `run_macro_backtest()` layer to verify one cache object is shared across assets.
- Run the convergence backtest test module.

---

## Related

- [[convergence_backtest_score_cache|Research: Convergence Backtest Score Cache]]
- [[convergence_detection]]
- [[convergence_backtest]]
- [[convergence_template_expansion]]
- [[convergence_signal_expansion]]
- [[convergence_napm_refresh]]
- [[convergence_backtest_fast_mode]]
- [[convergence_template_batch2]]
- [[convergence_audit_pre_worldmodel]]
