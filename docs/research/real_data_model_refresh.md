---
title: "Research: Real Data Model Refresh"
tags:
  - doc/research
  - phase/40
  - topic/gnn
  - topic/backtest
  - layer/world-model
  - layer/learning
---

# Research: Real Data Model Refresh

## Problem Statement

The GNN model (HetTGN) was trained on **synthetic data** in Phase 35. The live pipeline has accumulated **real observations** (68K+ instrument_daily, 1.4K Polymarket, 735 GDELT, 40 CFTC) but the model has never seen any of them. The current checkpoint has `num_nodes=918` while the live graph has 929 entities. The model produces embeddings from random initialization against real data — no learned structure.

Phase 40 retrains the GNN on real observations, regenerates features, runs the first real walk-forward backtest, and creates the `scripts/retrain_gnn.py` training script for future retraining.

## Current State

### Live Pipeline Database

| Metric | Value |
|--------|-------|
| Total entities | 929 (89 instrument, 82 country, 729 topic, 20 CFTC, 7 company, 2 protocol) |
| Entity links | 272 across 10 link types |
| Observations | 71,162 total |
| Price data | 68,089 `instrument_daily` obs (89 instruments, 2023-04-18 to 2026-04-18) |
| Polymarket | 1,458 `market_probability` obs on topic entities |
| GDELT | 735 `geopolitical_event` obs on country entities |
| CFTC | 40 `futures_positioning` obs on cftc_contract entities |
| Features | 48 rows (14 feature names × ~3 runs) |
| Portfolio weights | 0 rows |
| DAG runs | 2 daily_collection (1 ok, 1 fail), 2 gnn_inference (1 ok, 1 fail) |

### Model Checkpoint

| Field | Value |
|-------|-------|
| File | `.tirra_pipeline/gnn_model.pt` (1599 KB) |
| Hidden dim | 64 |
| Layers | 2 HGT |
| Epochs trained | 5 |
| Node types | 6 (cftc_contract, company, country, instrument, protocol, topic) |
| Edge types | 10 |
| `in_channels` | 14 per type (BASE_FEAT_DIM, no enrichment) |
| `num_nodes` | 918 (stale — live has 929) |
| Training data | Synthetic (Phase 35 SyntheticGraphGenerator) |
| Backups | `gnn_model_synthetic_backup.pt` (214 KB), `gnn_model_live.pt` (1023 KB) |

### Key Architecture: Trainer.train() Flow

1. `_split_observations()` — chronological 70/15/15 split of all store observations
2. `_make_windows(train_obs)` — 86400s (1-day) fixed windows
3. Per epoch, per consecutive window pair (W_i, W_{i+1}):
   - `build(until=t_end)` — graph snapshot up to window end
   - Forward pass → embeddings
   - `_compute_targets()` from W_{i+1}
   - 4 losses: obs_type CE, time_delta MSE, contrastive margin, value Huber
   - Backward + clip grads + optimizer step
   - Update memory from W_i events

### Data Density Analysis

**Critical question:** Is the real observation density sufficient for walk-forward training with 1-day windows?

- **Total observations**: 71,162 over ~1095 days (3yr) = ~65 obs/day
- **Effective entities with observations**: Only ~91 of 929 entities have any observations
- **instrument_daily**: 68,089 / 1095 days / 89 instruments ≈ 0.7 obs/instrument/day — good density
- **market_probability**: 1,458 obs concentrated in a few days — sparse
- **geopolitical_event**: 735 obs — sparse (≈0.7/day)
- **futures_positioning**: 40 obs — very sparse (monthly)

**Window population**: With 65 obs/day, most 1-day windows will have observations, primarily from instruments. This is sufficient for training but the learned signal will be dominated by `instrument_daily` patterns.

**Training viability**: The 68K instrument observations alone create ~760 training windows (1095 days × 70%) with ~44 obs per window on average. This is a reasonable training signal.

### Gaps Identified

1. **No retrain script**: No `scripts/retrain_gnn.py` exists. Training was done ad-hoc via Python in Phase 35.
2. **Entity types with 0 observations**: domain, organization, person, vessel, wallet — 5 of 11 types have zero real observations. These types exist in the ENTITY_TYPES list but have no entries in the entities table.
3. **Observation type coverage**: Only 7 of 46 observation types have real data. Model predicts over 46 types but will only see 7 during training.
4. **GDELT timestamp issue**: Country observations show 1970-epoch dates — appears to be a data quality bug where GDELT event timestamps were stored incorrectly. Needs investigation but is not a Phase 40 blocker.
5. **Model metadata mismatch**: Checkpoint has 6 node types; live graph has 6 types with data. The full schema defines 11 types but the 5 empty types don't appear in the built graph, so metadata stays at 6. This is consistent.
6. **Enrichment not used**: Current `in_channels=14` (BASE_FEAT_DIM) — enrichment features (ENRICHMENT_DIM=55) are not being passed to `build()`. This means cusum, hawkes, BOCPD, and obs_type distribution features are not utilized. Phase 40 should use them if feasible.
7. **No evaluation script**: No standalone script to evaluate model quality (top-1/5 accuracy, time_delta MAE) on held-out data.

## Design Decisions

### D1: Train on real data with current density

**Decision**: Train the GNN on all 71K real observations using the existing Trainer API.

**Rationale**: 68K instrument_daily observations provide dense temporal signal. Other observation types add cross-entity learning (Polymarket topics → instruments via `topic_relates_to_instrument` edges, GDELT countries → instruments via `located_in`/`exchange_country`/`fx_*` edges). Even with sparse non-instrument observations, the contrastive loss on 272 real entity links provides meaningful structural signal.

### D2: Increase epochs from 5 to 20

**Rationale**: Phase 35 used 5 epochs on synthetic data (small graph, clean patterns). Real data is noisier with 71K observations — more epochs needed for convergence. 20 epochs with ~760 training windows = ~15,200 forward passes. Estimated time: 10-30 minutes on CPU.

### D3: Keep window_size=86400 (1 day)

**Rationale**: Instrument daily observations are naturally daily-aligned. 1-day windows match the data cadence. Larger windows would reduce the number of training steps. Smaller windows would create many empty windows.

### D4: Enable auto_tune_loss_weights

**Rationale**: With imbalanced observation types (97% instrument_daily), fixed loss weights will bias the model toward instrument predictions. Kendall et al. 2018 uncertainty weighting automatically adjusts task weights based on learned uncertainty, which should help the model learn from sparse observation types.

### D5: Create `scripts/retrain_gnn.py` for reproducible training

**Rationale**: No training script exists. All prior training was ad-hoc. A proper script enables: scheduled retraining, hyperparameter tuning, model comparison, and CI/CD integration.

### D6: Run walk-forward backtest after retrain

**Rationale**: With real price data (3 years, 89 instruments) and a retrained GNN, we can run the first meaningful baseline backtest. This validates the full pipeline: observations → GNN → features → strategy → P&L. Even with only baseline strategies (EqualWeight, BuyAndHold), this establishes performance baselines.

## Files Affected

| File | Action |
|------|--------|
| `scripts/retrain_gnn.py` | CREATE — GNN training script |
| `agent/models/gnn/trainer.py` | No changes needed — API is ready |
| `agent/features/builders.py` | No changes needed |
| `agent/pipeline/dags/feature_generation.py` | No changes needed |
| `.tirra_pipeline/gnn_model.pt` | OVERWRITE — retrained checkpoint |
| `.tirra_pipeline/gnn_model_pre_phase40.pt` | CREATE — backup of current model |

## Risk Assessment

1. **Training time**: 20 epochs on 71K obs with CPU could be slow. Mitigation: start with 10 epochs, increase if loss still decreasing.
2. **Overfitting to instrument_daily**: 97% of observations are one type. Mitigation: auto_tune_loss_weights + contrastive loss on link structure.
3. **Memory**: 929 nodes × 64-dim memory is trivial (~240KB). No memory concern.
4. **Backtest overfitting**: 3 years of data with monthly folds may show deceptive results. Mitigation: use conservative interpretation, focus on Sharpe vs benchmarks not absolute returns.

## Related

- [[chat_checkpoint_2026-04-20_phase39_complete]]
- [[real_data_model_refresh_spec]]
- [[phase40_real_data_model_refresh]]
- [[phase35_gnn_retrain_expanded_graph]]
- [[quant_training_ground]]
