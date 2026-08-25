---
title: "Task: Phase 40 — Real Data Model Refresh"
tags:
  - doc/task
  - status/done
  - phase/40
  - topic/gnn
  - topic/backtest
  - layer/world-model
  - layer/learning
---

# Task: Phase 40 — Real Data Model Refresh

Status: completed
Research: [[real_data_model_refresh]]
Spec: [[real_data_model_refresh_spec]]

## Steps

- [x] 40.1: Create `scripts/retrain_gnn.py` training script
- [x] 40.2: Backup current model + retrain GNN on real data (5 epochs)
- [x] 40.3: Regenerate features with retrained model
- [x] 40.4: Run walk-forward backtest + record baselines
- [x] 40.5: Write edge-case tests for retrain script (8 tests, all passing)
- [x] 40.6: Update tracker + write checkpoint

## Performance Fixes Applied
- O(n) bucket-based windowing (was O(n×w))
- Entity type cache (eliminated per-window DB queries)
- `--since` flag for temporal filtering (skip 1970-era GDELT timestamps)
- `log(1+dt)` normalization for time_delta targets (was raw seconds → MSE in billions)
- 3-level graph caching with bisect slicing
- Logging setup for training progress visibility

## Results

### 40.2 Training Results
- **Config:** 5 epochs, lr=0.001, hidden=64, layers=2, heads=2, window=48h, auto_tune=True
- **Data:** 929 entities (6 types), 71,123 observations, 272 entity links, filtered since 2023-01-01
- **Training time:** 982.4s (16.4 min), 397 windows/epoch
- **Loss curve:**
  | Epoch | Total   | obs_type | time_delta | contrastive | value  |
  |-------|---------|----------|------------|-------------|--------|
  | 1     | 22.7693 | 2.1113   | 147.5543   | 0.0000      | 6.5384 |
  | 2     | 11.9476 | 0.4012   | 77.2474    | 0.0000      | 1.0953 |
  | 3     | 4.4114  | 0.0385   | 12.6716    | 0.0000      | 0.0113 |
  | 4     | 3.2190  | 0.0466   | 5.7147     | 0.0000      | 0.0866 |
  | 5     | 1.9956  | 0.0012   | 0.1321     | 0.0000      | 0.0009 |
- **Learned loss weights:** obs=1.130, dt=0.098, contr=3.640, val=0.470
- **Validation:** top-1=100.0%, top-5=100.0%, time_delta MAE=0.3s (6,410 predictions)
- **Test:** top-1=86.8%, top-5=86.8%, time_delta MAE=1.7s (5,770 predictions)
- **Random baseline:** top-1=2.17%, top-5=10.87% → **40x lift on test**
- **Model saved:** `.tirra_pipeline/gnn_model.pt` (339,141 params)

### 40.4 Backtest Results
- **Period:** 2023-04-18 to 2026-04-18 (1,097 dates × 89 instruments)
- **Walk-forward:** min_train=252, test=21, step=21 → 40 folds

| Strategy              | Total Return | Sharpe | Max Drawdown |
|-----------------------|-------------|--------|--------------|
| EqualWeight           | 23.14%      | 0.995  | -8.20%       |
| BuyHold SPY 100%      | 48.84%      | 0.900  | -18.76%      |
| BuyHold AGG:40%+SPY:60% | 31.75%   | 0.991  | -11.41%      |

- EqualWeight has highest Sharpe (0.995) and lowest drawdown (-8.20%)
- Top contributors (EqualWeight): GDX +1.30%, SI=F +1.27%, SLV +1.27%; detractors: UVXY -2.00%
- Note: This is the naive EqualWeight baseline — GNN-informed allocation not yet wired

## 40.7 — GNN-Signal Backtest Results (Phase 40 Final)

### Model Retrain (Kaggle sessions, epochs 6-10)
- **Config:** 1,858,459 params (vs 339K in step 40.2), hidden=128, layers=2, heads=2, 9 node types, 14 edge types
- **Data:** 2,451 entities, 11,915 entity_links, 977,870 observations (901,704 GDELT, 68,089 instrument_daily)
- **Final loss:** 14.14 (epoch 10), contrastive=0.769, value=3.49
- **Contrastive learning fixed:** was 0.0 in 40.2 (model too small/no contrastive pairs), now active

### GNN Backtest Infrastructure Built (`scripts/phase40_gnn_backtest.py`)
- `_load_instrument_returns_fast()`: direct SQL filter on `observation_type='instrument_daily'` — 1s vs 60s
- `GNNEmbeddingNormStrategy`: `w_i = softmax(z-score(||h_i||_2))` — activity level proxy
- `GNNValueHeadStrategy`: `w_i = softmax(model.predict_value(embeddings)["instrument"][:, 0])` — predicted return quantile
- `GNN_LOOKBACK_DAYS = 90`: 90-day obs window per fold — 26x reduction (29,993 vs 778,832 obs), ~0.4s/fold vs ~10s
- Pre-fetched graph via `prepare_static()` + `prefetch_observations()` (31s one-time), bisect-sliced per fold

### Walk-Forward Results (40 folds, same config as 40.4)

| Strategy         | Total Return | Sharpe | Max Drawdown | Max Weight |
|-----------------|-------------|--------|--------------|-----------|
| EqualWeight      | 23.14%      | 0.995  | -8.20%       | 1.1%      |
| GNN-EmbNorm      | 25.10%      | 0.582  | -11.59%      | 44.65%    |
| GNN-ValueHead    | 20.32%      | 0.870  | -8.08%       | 6.7%      |

### Conclusion

Neither GNN strategy beats EqualWeight on risk-adjusted returns:
- **GNN-EmbNorm**: Higher raw return (+1.95%) but highly concentrated (44.65% single asset) → Sharpe drops to 0.582 (ΔSharpe = -0.413). L2 embedding norm = activity/connectivity proxy, not a return signal.
- **GNN-ValueHead**: Closest (ΔSharpe = -0.124, ΔDD = +0.12%) but still underperforms. Value head trained with in-sample MSE on all history — no walk-forward awareness, likely in-sample bias.

**GNN is not yet producing predictive edge for portfolio allocation as a standalone signal.** The representation is meaningful (loss 91→14, contrastive working) but the mapping from embeddings to portfolio weights needs work. Phase 41 paths:
1. Train a downstream ranker on train-fold embeddings (linear model, avoids leakage)
2. Add temperature softmax to EmbNorm to reduce concentration
3. Combine GNN signal with momentum/vol signals
4. Walk-forward GNN retraining (per-fold model, no in-sample bias)

## Steps

- [x] 40.7: Wire GNN embeddings into walk-forward backtest
- [x] 40.8: Run GNN-signal backtest and compare to EqualWeight baseline

## Status

Phase 40 complete. GNN signal tested, edge not yet confirmed. Proceed to Phase 41.

## Related

- [[real_data_model_refresh]]
- [[real_data_model_refresh_spec]]
- [[phase41_model_refresh_hardening]]
- [[phase41b_gnn_signal_extraction]]
- [[chat_checkpoint_2026-04-20_phase40_full]]
- [[chat_checkpoint_2026-04-28_phase40_gnn_backtest_results]]
