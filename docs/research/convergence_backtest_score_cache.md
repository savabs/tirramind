---
title: "Research: Convergence Backtest Score Cache"
tags:
  - doc/research
  - phase/7c
  - topic/backtest
  - topic/convergence
---

# Research: Convergence Backtest Score Cache

## Goal
Reduce macro backtest runtime by reusing convergence step scores across targets when they share weekly timestamps.

## Current Architecture
- `run_macro_backtest()` loops over target assets and calls `_fetch_target_returns()` per ticker.
- For each target it calls `precompute_convergence_scores()` using the same `fred_data` but that target's weekly timestamps.
- `precompute_convergence_scores()` rebuilds evidence and runs the convergence detector independently for every timestamp.
- Step scores depend only on:
  - `fred_data`
  - detector configuration
  - series configuration
  - timestamp
- Step scores do **not** depend on the target asset's returns.

## Observations
- This means overlapping timestamps across SPY, TLT, and GLD are redundant work today.
- The cheapest safe optimization is a shared in-memory cache keyed by timestamp, scoped to one backtest run.
- Exact timestamp reuse is sufficient for the first pass because weekly yfinance bars for broad US assets typically align closely, and the cache remains correct even when overlap is partial.
- This avoids broader architectural changes like introducing a canonical calendar or persistent score cache.

## Risks
- Caching must remain scoped to a single `run_macro_backtest()` call; otherwise stale scores could leak across different detector configs or input histories.
- `StepScore` is a mutable dataclass, so cached objects should be treated as immutable once created.
- If timestamps differ slightly across assets, hit rate will be lower, but correctness is unaffected.

## Data Requirements
- No new data sources.
- The cache key should be the point-in-time weekly timestamp already used by the detector.

## Math/Algorithm Survey
- This is a pure computational reuse optimization. The detector output at time $t$ is deterministic for fixed historical inputs.
- Reusing the same `StepScore` for the same timestamp preserves numerical output exactly; it only removes duplicate execution.

## Step-Local References
- `agent/convergence/backtest.py`
- `tests/test_convergence_backtest.py`
- `[[checkpoint_archive_2026]]` (archived entry for `chat_checkpoint_2026-04-06_session8`)

---

## Related

- [[convergence_backtest_score_cache_spec|Spec: Convergence Backtest Score Cache]]
- [[convergence_detection]]
- [[convergence_backtest]]
- [[convergence_template_expansion]]
- [[convergence_signal_expansion]]
- [[convergence_napm_refresh]]
- [[convergence_backtest_fast_mode]]
- [[convergence_template_batch2]]
- [[convergence_audit_pre_worldmodel]]
