---
title: "Checkpoint: Phase 41 Complete — Model Refresh Hardening"
tags:
  - doc/checkpoint
  - phase/41
  - topic/gnn
  - topic/backtest
  - topic/pipeline
  - layer/world-model
  - layer/learning
  - layer/surveillance
---

# Phase 41 Complete — Model Refresh Hardening

Closes the two loose ends from Phase 40:
1. Broken 10-epoch GNN checkpoint (loss diverged to −23.58, effective weights up to 2006).
2. Auto-tune (Kendall uncertainty-weighted loss) runaway when component losses approach zero.

## Root Cause (math recap)

Kendall et al. (2018) homoscedastic multi-task loss:

$$\mathcal{L} = \sum_k e^{-s_k}\mathcal{L}_k + s_k,\quad s_k = \ln\sigma_k^2$$

Gradient on $s_k$: $\partial\mathcal{L}/\partial s_k = 1 - e^{-s_k}\mathcal{L}_k$.
When $\mathcal{L}_k \to 0$, gradient $\to +1$ is bounded but total loss
$s_k \to -\infty$ is unbounded below and effective weight $e^{-s_k} \to +\infty$
pushes other task gradients to explode, breaking the trade-off. Liebel & Körner (2018)
addresses the same instability for regression+classification mixes with the same
bounding remedy.

## Fix

Clamp $s_k \in [-3, 3]$ (i.e. $\sigma_k^2 \in [e^{-3}, e^3] \approx [0.05, 20]$) at loss
composition time via `torch.clamp`. Non-inplace, so gradients still flow within the
valid interval; saturates at bounds so runaway is impossible.

Configurable via `TrainerConfig.log_var_min` / `log_var_max`.

## Changes Landed

### Step 41.1 / 41.2 — Trainer hardening
- [agent/models/gnn/trainer.py](agent/models/gnn/trainer.py): added `log_var_min=-3.0`, `log_var_max=3.0` to `TrainerConfig`; applied `torch.clamp` in loss composition and in `effective_loss_weights()` reporting.
- [tests/test_trainer.py](tests/test_trainer.py): appended `TestLogVarClamp` (4 tests). Full suite: **31/31 passing**.

### Step 41.3 — Model restoration
- Archived broken 10-epoch as `.tirra_pipeline/gnn_model_broken_10ep.pt`.
- Discovered `gnn_model_live.pt` was stale (119 nodes vs current 929 graph) — unusable.
- Promoted `gnn_model_pre_phase40.pt` (2-epoch, 929 nodes) as the live model.
- Smoke-loaded through `Trainer.load_model` — confirmed 929 nodes, 6 node types, 339k params.

### Step 41.4 — Fresh retrain with clamp
Command: `python scripts/retrain_gnn.py --db-path .tirra_pipeline/pipeline.db --epochs 5 --auto-tune --since 2023-01-01 --window-size 172800 --backup`

| Epoch | Total | obs_type | time_delta | contrastive | value |
|---|---|---|---|---|---|
| 1 | 33.9081 | 1.4389 | 294.0420 | 0.0000 | 6.2344 |
| 2 | 28.3956 | 0.4908 | 273.3463 | 0.0000 | 0.7451 |
| 3 |  7.8088 | 3.0156 |  16.7150 | 0.0000 | 0.1552 |
| 4 |  5.7274 | 0.0476 |  34.2975 | 0.0000 | 0.0079 |
| 5 | **2.1294** | 0.0004 |   0.0344 | 0.0000 | 0.0001 |

- Final effective weights: obs=1.035, dt=0.086, contr=3.640, val=0.507 — all ∈ [0.05, 20] ✓
- Val: top-1 = 100.0%, time_delta MAE = 0.1s
- Test: top-1 = **87.0%** (random baseline 2.17%), time_delta MAE = **1.9s**
- Training time: 597s (~10 min)

Acceptance: **all criteria exceeded by large margins.**

### Step 41.4 — Backtest (40 folds, min_train=252, test=21)
| Strategy | Total Return | Sharpe | Max DD |
|---|---|---|---|
| equal_weight | 23.14% | 0.995 | -8.20% |
| buy_hold SPY | 48.84% | 0.900 | -18.76% |
| buy_hold 60/40 | 31.75% | 0.991 | -11.41% |

Baselines look healthy — matches Phase 40 Run 1 profile.

### Step 41.5 — Whale Alert wired into daily_collection
- [agent/pipeline/dags/daily_collection.py](agent/pipeline/dags/daily_collection.py): added `fetch_whale_alert` node (operator=`whale_alert`, mode=confirmed, min_btc=10.0, limit=100, timeout=60, retries=2).
- [tests/test_pipeline_registry.py](tests/test_pipeline_registry.py): node count 8→9, expected set expanded, new `test_whale_alert_node_config`. All **40 registry tests passing**.

`whale_alert` is already registered in [agent/cli.py](agent/cli.py) with the required pipeline_store wiring — no additional glue needed. Tool is free (blockchain.info), no auth, L2-ready (persists wallet entities + wallet→BTC-USD links).

## Invariants / Acceptance Log

- [x] All new trainer tests pass (4/4).
- [x] Full trainer suite: 31/31.
- [x] Full registry suite: 40/40.
- [x] Retrain: monotonic loss, no negative total.
- [x] Effective weights ∈ [0.05, 20].
- [x] Test top-1 ≥ 50% (achieved 87.0%).
- [x] Test time_delta MAE ≤ 60s (achieved 1.9s).
- [x] `daily_collection` validates clean with 9 nodes, single parallel layer.

## File State

- Live model: `.tirra_pipeline/gnn_model.pt` (5-epoch, clamp-hardened, 929 nodes).
- Backups: `gnn_model_broken_10ep.pt` (archived), `gnn_model_pre_phase40.pt` (2-epoch safety), `gnn_model_synthetic_backup.pt`, auto-backup from `--backup` flag on this run.
- Features: regenerated via `feature_generation` DAG (929 entities, 71123 observations, 272 links).

## Next Candidates (not started)

1. **GNN-guided expansion audit** — now that the model is healthy and 87% accurate on observation-type prediction, run the evaluation described in copilot-instructions.md §"Signal Depth Doctrine" rule 6: which entity neighborhoods are sparse, which attention heads are starved, where are L2 gaps that would benefit from new tools or upgrades.
2. **Clamp bounds sweep** — current `[-3, 3]` is Kendall's standard guidance; measure whether `[-2, 2]` or `[-4, 4]` changes final test MAE materially on future refreshes.
3. **Longer training** — 5 epochs already hits 87% test top-1; 8-10 epochs should be safe with the clamp and may close the val(100%)/test(87%) gap if it's not overfitting. Phase 42 candidate.
4. **Replace 2-epoch fallback** — `gnn_model_pre_phase40.pt` can now be retired/replaced by the hardened 5-epoch as the safety checkpoint.

## Related

- [[phase41_model_refresh_hardening]]
- [[phase41_model_refresh_hardening_spec]]
- [[chat_checkpoint_2026-04-20_phase40_full]]
- [[real_data_model_refresh]]
