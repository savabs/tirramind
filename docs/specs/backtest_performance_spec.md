---
title: "Spec: backtest_performance"
tags:
  - doc/spec
  - topic/backtest
---

# Spec: backtest_performance

## Goal
Reduce `precompute_convergence_scores()` runtime by 3-5× through eliminating redundant computation: pre-build evidence once, vectorize z-scores, remove per-step SQLite overhead, reuse detector.

## Files Affected
- `agent/convergence/backtest.py` — main changes (evidence pre-build, score loop rewrite)
- `tests/test_convergence_backtest.py` — regression tests for equivalence

## Implementation Steps

### Step 1: Vectorize confidence z-score in `build_evidence()`
Replace the O(k²) inner loop with cumulative mean/std. Evidence output must be numerically equivalent.

### Step 2: Add `build_all_evidence()` method to `HistoricalEvidenceBuilder`
Pre-builds Evidence for the full date range in one pass. Returns evidence sorted by timestamp.

### Step 3: Rewrite scoring loop to pre-build + slice
- Call `build_all_evidence()` once before the loop
- At each step: bisect for as_of cutoff + lookback start
- Pass evidence slice directly to detector

### Step 4: Eliminate per-step PipelineStore
Create one dummy store outside the loop. Reuse single detector instance. Remove monkey-patch in favor of direct evidence injection.

### Step 5: Regression test
Run both old and new paths on synthetic data. Verify identical StepScore output. Ensure no look-ahead leakage.

## Edge Cases
- Empty evidence at early timestamps (handled by existing `len(evidence) < 4` guard)
- Floating-point z-score differences between old/new path (accept ε < 1e-10)
- Persistence history now carries across steps (intentional behavior change / bug fix)

## Testing Plan
- Unit test: vectorized z-scores match loop-based z-scores on known data
- Unit test: `build_all_evidence()` produces same evidence as `build_evidence()` called N times
- Regression: full `precompute_convergence_scores()` output matches before/after

---

## Related

- [[backtest_performance|Research: Backtest Performance]]
- [[convergence_backtest]]
- [[scoring_validation]]
