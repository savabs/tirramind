---
title: "Spec: Real Data Model Refresh"
tags:
  - doc/spec
  - phase/40
  - topic/gnn
  - topic/backtest
  - layer/world-model
  - layer/learning
---

# Spec: Real Data Model Refresh

## Goal

Retrain the HetTGN on real pipeline observations, regenerate GNN features, run the first real walk-forward backtest, and create a reusable training script. Establish performance baselines on 3 years of real multi-asset returns.

## Files Affected

| File | Action | Description |
|------|--------|-------------|
| `scripts/retrain_gnn.py` | CREATE | GNN training + evaluation script |
| `.tirra_pipeline/gnn_model_pre_phase40.pt` | CREATE | Backup of current synthetic checkpoint |
| `.tirra_pipeline/gnn_model.pt` | OVERWRITE | Retrained model on real data |

## Implementation Steps

### 40.1: Create `scripts/retrain_gnn.py`

Create a training script that:

1. Accepts CLI args: `--db-path`, `--epochs`, `--lr`, `--auto-tune`, `--model-out`, `--backup`
2. Opens PipelineStore, prints data summary (entity counts, observation counts, link counts, date range)
3. Instantiates `Trainer(store, config)` with `TrainerConfig` from CLI args
4. Calls `trainer.build_model()` — prints model architecture summary (params, node types, edge types)
5. Calls `trainer.train()` — prints per-epoch loss breakdown
6. Calls `evaluate(model, store, split="val")` — prints top-1/5 accuracy, time_delta MAE
7. Calls `evaluate(model, store, split="test")` — prints held-out metrics
8. Calls `trainer.save_model(model_out)` — persists checkpoint
9. Reports total training time

**Config defaults:**
- epochs=20
- learning_rate=1e-3
- window_size=86400
- auto_tune_loss_weights=True
- hidden_dim=64, memory_dim=64, num_heads=2, num_layers=2

**Test**: Run `python scripts/retrain_gnn.py --db-path .tirra_pipeline/pipeline.db --epochs 2` — should complete without error, print loss curves and evaluation metrics.

### 40.2: Backup current model + retrain on real data

1. Copy `.tirra_pipeline/gnn_model.pt` → `.tirra_pipeline/gnn_model_pre_phase40.pt`
2. Run: `python scripts/retrain_gnn.py --db-path .tirra_pipeline/pipeline.db --epochs 20 --auto-tune`
3. Verify: model saved, val metrics printed, loss decreasing across epochs

**Success criteria**: 
- Val obs_type top-1 accuracy > 30% (random baseline = 1/46 ≈ 2.2%)
- Loss monotonically decreasing for first 10 epochs
- Training completes in < 60 minutes

### 40.3: Regenerate features with retrained model

1. Run feature generation: `python scripts/run_collection.py --dag feature_generation`
   - Or manually: instantiate FeatureBuilder trio, call build(store, as_of=now)
2. Verify: GNN features produced (11 features), convergence features produced, macro features (0 unless FRED key set)
3. Compare feature values to pre-retrain: GNN anomaly/activity z-scores should differ from Phase 39 values

**Test**: Query `features` table, confirm new rows with timestamps after retrain.

### 40.4: Run walk-forward backtest

1. Run: `python scripts/run_backtest.py`
2. Verify output: Sharpe, total return, max drawdown for each strategy
3. Record baseline metrics:
   - EqualWeight: Sharpe, return, drawdown
   - BuyAndHold(SPY): Sharpe, return, drawdown
   - BuyAndHold(60/40): Sharpe, return, drawdown

**Success criteria**:
- Backtest completes without error
- Reports ≥20 monthly folds (3yr data ÷ 21-day test windows - 252 training minimum)
- All strategies produce valid metrics (no NaN Sharpe)

### 40.5: Write edge-case tests for retrain script

Test the training script's robustness:

1. Empty store → graceful error
2. Store with only 1 observation type → trains without crash
3. Store with no entity links → contrastive loss = 0, other losses train normally
4. Epochs=0 → returns empty history
5. Model save/load round-trip → weights match
6. Evaluation on empty test split → returns zero metrics gracefully

### 40.6: Update task file + record results

1. Record actual metrics from 40.2-40.4 in the task file
2. Update quant_training_ground.md
3. Write checkpoint

## Edge Cases

- **Sparse entity types**: 5 of 11 entity types have 0 real entities. Graph builder handles this by creating empty tensors. HGT layers skip types with 0 nodes. No crash expected.
- **Timestamp edge cases**: GDELT country observations have suspicious 1970-epoch timestamps. These will be included in training — the trainer handles arbitrary timestamp ranges via windowing. The windows will separate them from 2023+ instrument data.
- **Memory growth**: If new entities appear between build_model() and infer(), memory.resize() handles it (Phase 39 fix).
- **Backtest with NaN returns**: load_instrument_returns fills missing values with 0.0. Instruments with no data get zero weight from EqualWeight.

## Testing Plan

| Test | Verification |
|------|-------------|
| 40.1 | Script runs with `--epochs 2`, produces loss output |
| 40.2 | Backup exists, retrained model saved, val accuracy > random |
| 40.3 | Features table has new rows post-retrain |
| 40.4 | Backtest prints valid metrics for all strategies |
| 40.5 | 6 edge-case unit tests pass |
| 40.6 | Task file complete, checkpoint written |

## Related

- [[real_data_model_refresh]]
- [[phase40_real_data_model_refresh]]
- [[quant_training_ground]]
- [[phase35_gnn_retrain_expanded_graph]]
