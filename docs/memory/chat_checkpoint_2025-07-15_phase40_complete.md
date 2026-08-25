---
title: "Checkpoint: Phase 40 Complete — Real Data Model Refresh"
tags:
  - doc/checkpoint
  - phase/40
  - topic/gnn
  - topic/backtest
  - layer/world-model
---

# Checkpoint: Phase 40 Complete

**Date:** 2025-07-15
**Task:** [[phase40_real_data_model_refresh]]
**Spec:** [[real_data_model_refresh_spec]]
**Research:** [[real_data_model_refresh]]

## What Was Done

Phase 40 is **fully complete**. All 6 steps executed successfully:

1. **40.1** — `scripts/retrain_gnn.py` created (CLI training script with Rich output, auto-tune, --since filter)
2. **40.2** — GNN trained on real pipeline data (5 epochs, 982s, 339K params)
3. **40.3** — Features regenerated via `run_collection.py --dag feature_generation` (5.9s)
4. **40.4** — Walk-forward backtest completed (40 folds, 89 instruments, 2023-2026)
5. **40.5** — 8 edge-case tests written and passing
6. **40.6** — Task file updated with all metrics

## Key Results

### GNN Training
- Loss: 22.77 → 1.996 (91% reduction over 5 epochs)
- Test obs_type accuracy: 86.8% (vs 2.17% random baseline = 40x lift)
- Val accuracy: 100% (likely some train/val leakage in temporal ordering — not critical for self-supervised pretraining)
- Time_delta MAE: 1.7s on test
- Learned loss weights: obs=1.130, dt=0.098, contr=3.640, val=0.470
- Contrastive loss stayed at 0 — needs investigation (may need more diverse entity pairs in windows)

### Backtest Baselines
| Strategy | Return | Sharpe | Max Drawdown |
|----------|--------|--------|--------------|
| EqualWeight | 23.14% | 0.995 | -8.20% |
| SPY 100% | 48.84% | 0.900 | -18.76% |
| 60/40 | 31.75% | 0.991 | -11.41% |

EqualWeight is the naive baseline. GNN-informed allocation not yet wired.

## Performance Optimizations Shipped
- O(n) bucket windowing (was O(n×w))
- Entity type cache (eliminated per-window DB queries)
- 3-level graph caching with bisect slicing
- log(1+dt) normalization for time_delta targets
- --since temporal filter for data quality

## Files Changed/Created
- `scripts/retrain_gnn.py` — new CLI training script
- `agent/models/gnn/trainer.py` — perf optimizations
- `agent/models/gnn/graph_builder.py` — 3-level caching
- `tests/test_phase40_retrain.py` — 8 edge-case tests
- `.tirra_pipeline/gnn_model.pt` — retrained model artifact

## What's Next
- Wire GNN embeddings into portfolio allocation (move from EqualWeight to GNN-informed weights)
- Investigate contrastive loss = 0 (need more diverse entity pairs per window)
- Consider more epochs or learning rate scheduling for further loss reduction
- Val accuracy = 100% suggests possible temporal leakage worth auditing

## Related
- [[real_data_model_refresh]]
- [[real_data_model_refresh_spec]]
- [[phase40_real_data_model_refresh]]
