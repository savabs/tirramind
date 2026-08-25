---
title: "Research: Convergence Backtest Fast Mode"
tags:
  - doc/research
  - phase/7c
  - topic/backtest
  - topic/convergence
---

# Research: Convergence Backtest Fast Mode

## Goal
Reduce iteration latency for convergence macro backtest work by adding an explicitly reduced-cost fast mode while preserving the current full-run defaults.

## Current Architecture
- `agent/convergence/backtest.py` exposes `run_macro_backtest()` and a small CLI in `main()`.
- The expensive parts are:
  - per-target `precompute_convergence_scores()` over long weekly histories
  - bootstrap CI calls using `n_bootstrap=1000` for each non-benchmark strategy
- Current CLI only exposes `--macro`, `--validate`, `--save-baseline`, `--start-year`, `--end-year`, and `--targets`.

## Observations
- Existing checkpoint notes already identify a fast mode as a useful next engineering step.
- The safest place to reduce work is at the CLI/preset layer, not by changing the default algorithm.
- `n_bootstrap=1000` is hardcoded inside the backtest loop, so there is no way to reduce confidence-interval cost during development.
- Asset count and date range also dominate runtime because convergence scoring runs once per target.

## Risks
- A fast mode can accidentally become the default if the API contract is not explicit.
- Validation against the saved baseline should continue to use the full mode unless the user intentionally asks for a reduced-cost check.
- Changing walk-forward defaults would alter result comparability more than reducing bootstrap count; prefer minimal preset changes.

## Data Requirements
- No new data sources.
- Must remain compatible with existing FRED cache and yfinance target fetch.

## Math/Algorithm Survey
- The bootstrap CI is estimation overhead, not core signal generation.
- Reducing `n_bootstrap` speeds development but widens the CI; this is acceptable for a clearly labeled fast mode.
- Restricting targets or shortening the date range also lowers compute, but those should be optional CLI presets rather than silent default behavior changes.

## Step-Local References
- `agent/convergence/backtest.py`
- `tests/test_convergence_backtest.py`
- `[[checkpoint_archive_2026]]` (archived entry for `chat_checkpoint_2026-04-06_session5`)

---

## Related

- [[convergence_backtest_fast_mode_spec|Spec: Convergence Backtest Fast Mode]]
- [[convergence_detection]]
- [[convergence_backtest]]
- [[convergence_template_expansion]]
- [[convergence_signal_expansion]]
- [[convergence_napm_refresh]]
- [[convergence_backtest_score_cache]]
- [[convergence_template_batch2]]
- [[convergence_audit_pre_worldmodel]]
