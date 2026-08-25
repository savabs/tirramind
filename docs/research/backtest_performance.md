---
title: "Feature: Convergence Backtest Performance Optimization"
tags:
  - doc/research
  - topic/backtest
---

# Feature: Convergence Backtest Performance Optimization

## Current Architecture

The inner loop of `precompute_convergence_scores()` runs once per weekly timestamp (~800 steps for 2010–2025). At each step:

1. `build_evidence(as_of=ts)` — rebuilds ALL Evidence from FRED data
2. Creates a fresh `PipelineStore(":memory:")` — SQLite init + schema
3. Stores all evidence into the SQLite store
4. Creates a new `ConvergenceDetector` instance
5. Monkey-patches `_load_evidence` to bypass the store (!)
6. Calls `detector.detect(as_of=ts)`
7. Closes the store

## Identified Bottlenecks (ranked by impact)

### 1. O(k²) confidence z-score in `build_evidence()` — HIGH IMPACT

```python
for ts, value in relevant:                    # k iterations
    vals = [v for _, v in relevant if _ <= ts] # k inner filter
    mean_v = np.mean(vals[:-1])
    std_v = np.std(vals[:-1], ddof=1)
```

For each observation in the lookback window, a list comprehension re-scans all previous observations. This is O(k²) per series per step.

**Fix:** Replace with vectorized cumulative mean/std using `np.cumsum`. One pass, O(k) per series.

### 2. Evidence rebuilt from scratch every step — HIGH IMPACT

`build_evidence(as_of=ts)` re-filters, recomputes directions, and recomputes z-scores for the ENTIRE lookback window at every timestep. But from step t to step t+1 (one week later), only ~1 new observation per series enters the window and ~1 exits.

**Fix:** Pre-build ALL Evidence for the full date range once. At each step, use `bisect` on pre-sorted timestamps to slice `evidence[:cutoff_idx]` for the lookback window. No recomputation.

### 3. PipelineStore created + populated + closed every step — MEDIUM IMPACT

Each step creates an in-memory SQLite database (3 tables, 2 indices, WAL mode init, schema creation), serializes all evidence to JSON rows, inserts them, then the detector's `_load_evidence` is immediately monkey-patched to not even use the store. The store is pure waste.

**Fix:** Create ONE dummy store outside the loop. Or better: pass evidence directly to the detector, skip the store entirely. The monkey-patch already shows the store is unnecessary.

### 4. New ConvergenceDetector per step — LOW-MEDIUM IMPACT

Persistence history (consecutive detection counts) resets every step because a new detector is created. This also means the persistence filter never fires (needs ≥ min_persist consecutive detections).

**Fix:** Reuse a single detector instance across all steps. This also fixes the semantic bug where persistence tracking was silently broken.

### 5. Cross-target timestamp misalignment — LOW IMPACT (already mitigated)

Different yfinance downloads for SPY/TLT/GLD produce slightly different weekly timestamps. The shared `step_score_cache` can miss due to float key mismatches.

**Fix (future):** Canonical Monday-based weekly grid. Already discussed, lower priority now.

## Estimated Speedup

| Bottleneck | Est. fraction of runtime | Estimated speedup |
|---|---|---|
| O(k²) confidence z-score | ~30% | 5-10× on that path |
| Evidence rebuild per step | ~40% | 3-5× total |
| PipelineStore per step | ~15% | eliminate entirely |
| Detector re-init per step | ~5% | small + fixes bug |

Combined estimate: **3-5× overall speedup** on `precompute_convergence_scores()`.

## Implementation Approach

**Strategy: Pre-build + slice.**

1. Build ALL evidence for the full date range once (with vectorized z-scores)
2. Sort once by timestamp
3. At each backtest step: binary-search to find cutoff index, slice `evidence[:idx]`
4. Filter to lookback window using another bisect
5. Pass directly to detector (no store, no monkey-patch)
6. Reuse single detector instance across steps

## Risks

- Evidence semantics must remain identical (same z-scores, directions, confidence values). Need regression test.
- Persistence history reuse changes detector behavior (now it actually works correctly). This is a bug fix, not a regression, but scores may shift slightly.
- Must preserve no-look-ahead guarantee (binary search ensures this naturally).

## Data Requirements
None — pure compute optimization, no new data sources.

## Math/Algorithm Notes
The only mathematical change is replacing the O(k²) z-score loop with Welford-style running statistics or cumulative sums. This is numerically equivalent for the same data; the only difference is order of operations in floating point, which is negligible for these magnitudes.

---

## Related

- [[backtest_performance_spec|Spec: Backtest Performance]]
- [[convergence_backtest]]
- [[scoring_validation]]
