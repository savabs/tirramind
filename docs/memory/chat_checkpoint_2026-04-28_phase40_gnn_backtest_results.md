---
title: "Checkpoint: Phase 40 GNN Backtest Results — 2026-04-28"
tags:
  - doc/checkpoint
  - phase/40
  - topic/gnn
  - topic/backtest
  - status/done
---

# Checkpoint: Phase 40 GNN Backtest Results — 2026-04-28

## Session Summary

This session completed Phase 40 by running the GNN-signal walk-forward backtest and comparing two GNN allocation strategies against the EqualWeight baseline.

## What Was Done

1. **Applied 90-day lookback window** to `GNNEmbeddingNormStrategy` and `GNNValueHeadStrategy`
   - 26x obs reduction per fold: 29,993 obs vs 778,832 (full history)
   - ~0.4s/fold vs ~10s/fold without the window
   - Uses `bisect.bisect_left()` on prefetched obs timestamps (in-memory, no DB per fold)

2. **Fixed `KeyError: 'EqualWeight'`** in comparison table — `EqualWeightStrategy.name` returns `"equal_weight"` not `"EqualWeight"`.

3. **Ran full backtest to completion** (~190s: 11s startup + 1s price load + 31s obs prefetch + ~56s GNN-EmbNorm + ~90s GNN-ValueHead)

## Phase 40 Final Results

### Model
- **HetTGN**: 1,858,459 params, hidden_dim=128, num_layers=2, heads=2
- **9 node types**: topic, person, company, country, instrument, wallet, protocol, cftc_contract, organization
- **14 edge types**, 2,451 nodes, 11,915 links
- **Trained**: 10 epochs (5 local + 5 Kaggle), final loss=14.14, contrastive=0.769, value=3.49
- **Checkpoint**: `.tirra_pipeline/gnn_model.pt`

### Walk-Forward Config
- Period: 2023-04-18 → 2026-04-18 (1,097 dates × 89 instruments)
- min_train=252, test_size=21, step_size=21 → 40 folds

### Results Table

| Strategy         | Total Return | Sharpe | Max Drawdown | Max Weight |
|-----------------|-------------|--------|--------------|-----------|
| EqualWeight      | 23.14%      | **0.995** | -8.20%  | 1.1%      |
| GNN-EmbNorm      | 25.10%      | 0.582  | -11.59%      | **44.65%** |
| GNN-ValueHead    | 20.32%      | 0.870  | -8.08%       | 6.7%      |

### Interpretation

**GNN does NOT yet produce predictive edge over EqualWeight.**

- **GNN-EmbNorm (ΔSharpe = -0.413)**: L2 embedding norm = activity/connectivity, not return predictability. Softmax concentrates 44.65% in one high-norm asset. More raw return (+1.95%) but far worse risk-adjusted performance.
- **GNN-ValueHead (ΔSharpe = -0.124)**: Value head trained with in-sample MSE on full history — no walk-forward splits during training → in-sample bias. Closest to EqualWeight in behavior (6.7% max weight) but still underperforms.
- **Win Rate = 0.000** in output: cosmetic bug — `score_returns()` has no `win_rate`/`volatility` keys, `.get('win_rate', 0)` defaults to 0. The Sharpe/Return/MaxDD figures are correct.

## Infrastructure Built

### `scripts/phase40_gnn_backtest.py`
- **`_load_instrument_returns_fast(db_path, entity_ids)`**: Direct SQL `WHERE observation_type='instrument_daily' AND entity_id IN (...)` → 1s vs 60s via store
- **90-day lookback constant**: `GNN_LOOKBACK_DAYS = 90`
- **`GNNEmbeddingNormStrategy`**: `w_i = softmax(z-score(||h_i||_2))` where `h_i` = 128-dim embedding
- **`GNNValueHeadStrategy`**: `w_i = softmax(model.predict_value(embeddings)["instrument"][:, 0])`
- **Pre-fetch design**: `prepare_static()` + `prefetch_observations()` once at startup → bisect-slice per fold
- **Fold-level cache**: `self._cache[fold_date]` — prevents redundant GNN forward passes

### Key API References (confirmed working)
```python
trainer._graph_builder.prepare_static()              # → (id_map, entities, links)
trainer._graph_builder.prefetch_observations(since, until)  # → sorted obs list
trainer._graph_builder.build_from_cached(id_map, links, observations=obs_window)  # → (HeteroData, IDMap, events)
model.forward(data, id_map)                          # → dict[str, tensor[N_type, 128]]
model.predict_value(embeddings)                      # → dict[str, tensor[N_type, 1]]
id_map.local_id("instrument", eid)                   # → int or None
```

## Critical Operational Notes

- **Python path**: Always `/home/becmachlean/anaconda3/bin/python` — NOT `conda run`, NOT `python3`. A search engine venv in base terminal breaks imports.
- **Background run**: Use `> /tmp/out.txt 2>&1 &` + `tail -f /tmp/out.txt` for monitoring.
- **GNN `model.forward()` only READS TGN memory** (doesn't update it) — safe to call per fold on same loaded model.

## Phase 41 Paths (not yet started)

Per task file, four options ranked by expected impact:
1. **Downstream ranker on train-fold embeddings** — train a simple linear ranker on train-fold GNN embeddings to predict next-period returns (no lookahead). Cleanest fix for in-sample bias.
2. **Temperature softmax on EmbNorm** — add `temperature` param to reduce concentration. Quick sanity check.
3. **Combine GNN + momentum/vol signals** — ensemble; momentum signal already present in feature set.
4. **Walk-forward GNN retraining** — retrain GNN on each train fold. Expensive but eliminates all in-sample bias.

## Source of Truth

- Task file: [[phase40_real_data_model_refresh]]
- Backtest script: `scripts/phase40_gnn_backtest.py`
- Model: `.tirra_pipeline/gnn_model.pt`
- DB: `.tirra_pipeline/pipeline.db`

## Related
- [[real_data_model_refresh]]
- [[real_data_model_refresh_spec]]
- [[phase40_real_data_model_refresh]]
