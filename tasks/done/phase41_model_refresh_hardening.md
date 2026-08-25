---
title: "Task: Phase 41 — Model Refresh Hardening"
tags:
  - doc/task
  - status/done
  - phase/41
  - topic/gnn
  - topic/backtest
  - topic/pipeline
  - layer/world-model
  - layer/learning
  - layer/surveillance
---

# Task: Phase 41 — Model Refresh Hardening

Status: completed
Research: [[phase41_model_refresh_hardening]]
Spec: [[phase41_model_refresh_hardening_spec]]

## Steps

- [x] 41.1: Clamp Kendall log-variance in `Trainer` + add `log_var_min`/`log_var_max` to `TrainerConfig`.
- [x] 41.2: Add edge-case tests in `tests/test_trainer.py` covering the clamp (4 tests, all passing).
- [x] 41.3: Restore a healthy checkpoint as the live model (file-level swap, with architecture smoke check).
- [x] 41.4: Fresh retrain with clamp (5 epochs, 48h window, `--since 2023-01-01`); regen features; rerun backtest; record metrics.
- [x] 41.5: Wire `whale_alert` into `daily_collection` + DAG test.
- [x] 41.6: Update project memory + write Phase 41 checkpoint.

## Final Metrics (41.4 retrain)

- Loss curve: 33.91 → 28.40 → 7.81 → 5.73 → **2.13** (monotonic, no negative total)
- Effective weights: obs=1.035, dt=0.086, contr=3.640, val=0.507 (all ∈ [0.05, 20])
- Val: top-1 = 100.0%, time_delta MAE = 0.1s
- Test: top-1 = **87.0%**, time_delta MAE = **1.9s**
- Backtest baselines (40 folds): EqualWeight Sharpe 0.995, BuyHold SPY 0.900, 60/40 0.991
- `daily_collection` DAG: 8 → **9 nodes** (added `fetch_whale_alert`)

## Acceptance

- All new trainer tests pass.
- Retrained model: monotonic loss, effective weights in `[0.05, 20]`, test top-1 ≥ 50%, test time_delta MAE ≤ 60s.
- `daily_collection` DAG has 9 nodes (was 8); DAG validates.

## Related

- [[phase41_model_refresh_hardening]]
- [[phase41_model_refresh_hardening_spec]]
- [[real_data_model_refresh]]
- [[chat_checkpoint_2026-04-20_phase40_full]]
