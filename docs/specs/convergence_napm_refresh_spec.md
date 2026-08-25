---
title: "Spec: Convergence Backtest NAPM Refresh"
tags:
  - doc/spec
  - phase/7c
  - topic/convergence
---

# Spec: Convergence Backtest NAPM Refresh

## Goal
Replace the fragile NAPM series in the convergence macro backtest with a maintained free historical proxy while preserving the backtest's macro-momentum role and keeping the change narrowly scoped.

## Files Affected
- `agent/convergence/backtest.py` — modify FRED series mapping and any related comments/constants
- `tests/test_convergence_backtest.py` — add or update targeted regression coverage
- `[[convergence_napm_refresh]]` — track progress

## Implementation Steps
1. Replace the `NAPM` config entry in `FRED_SERIES` with a maintained monthly FRED proxy and document the reason in code-facing comments if needed.
2. Add focused regression tests that pin the replacement series id, signal id, frequency, and direction rule semantics.
3. Run focused convergence backtest tests and fix any regressions.
4. Update the task file with verification results.

## Edge Cases
- Replacement series missing from fetched payload should not break registry construction.
- Direction rule must remain threshold-based and deterministic.
- Existing signal id should remain stable unless tests prove a taxonomy change is required.

## Testing Plan
- Run targeted tests for `agent/convergence/backtest.py`.
- Run the existing convergence backtest test module.
- If the module passes, record the exact verification command in the task file.

---

## Related

- [[convergence_napm_refresh|Research: Convergence Napm Refresh]]
- [[convergence_detection]]
- [[convergence_backtest]]
- [[convergence_template_expansion]]
- [[convergence_signal_expansion]]
- [[convergence_backtest_fast_mode]]
- [[convergence_backtest_score_cache]]
- [[convergence_template_batch2]]
- [[convergence_audit_pre_worldmodel]]
