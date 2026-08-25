---
title: "Checkpoint: Phase 41 IC Diagnostic + Contrarian Test — 2026-04-29"
tags:
  - doc/checkpoint
  - phase/41
  - topic/gnn
  - topic/backtest
  - topic/evaluation
  - status/done
---

# Checkpoint: Phase 41 IC Diagnostic + Contrarian Test — 2026-04-29

> **Next session:** Read this file first, then [[quant_training_ground]] for roadmap.
> Immediate next step: **Option 2 — return-prediction auxiliary loss on GNN training.**

---

## 1. What Was Done This Session

### 1.1 Added IC Diagnostic to backtest script

File: `scripts/phase40_gnn_backtest.py`

Changes:
- `_compute_ic_report()` function: computes per-fold Spearman IC between raw GNN scores and 21d forward returns, prints mean IC, IC std, t-stat, IC IR, fraction of positive-IC folds, signal quality label.
- `GNNEmbeddingNormStrategy._raw_scores`: stores raw norms (pre-softmax) per fold for IC computation.
- `GNNValueHeadStrategy._raw_scores`: stores raw value predictions per fold for IC computation.
- New section `── 10. IC Diagnostic ──` in `main()` calls `_compute_ic_report` for all GNN strategies.
- `TEMPERATURE_EMBNORM_LOW = 0.3` constant added — second EmbNorm instance with lower temperature to reduce concentration.
- `GNNEmbeddingNormStrategy.name` property is now dynamic: `f"GNN-EmbNorm{tag}-T{self._temperature}"`.

### 1.2 Added invert flag to GNNEmbeddingNormStrategy (Option 1)

- `invert: bool = False` parameter on `__init__`.
- When `invert=True`: `z = -z` before softmax → long low-norm instruments (contrarian).
- Added `GNN-EmbNormInv-T0.3` strategy instance in `main()`.

---

## 2. Phase 41 Results (Ground Truth)

### 2a — Temperature Fix (from previous session result, confirmed again):

| Strategy | Sharpe | Total Return | Max Drawdown | Max Weight |
|---|---|---|---|---|
| EqualWeight | 0.995 | 23.14% | -8.20% | 1.12% |
| GNN-EmbNorm-T1.0 | 0.582 | 25.10% | -11.59% | **44.65%** |
| GNN-EmbNorm-T0.3 | 0.983 | 23.05% | -7.93% | 5.89% |
| GNN-EmbNormInv-T0.3 | 0.973 | 23.09% | -7.51% | — |
| GNN-ValueHead | 0.870 | 20.32% | -8.08% | 6.68% |

T=0.3 fixes the 44.65% concentration → max weight drops to 5.89%. Sharpe 0.983 ≈ EqualWeight (neutral, not hurting).

### 2b — IC Diagnostic Results:

| Strategy | Mean IC | IC t-stat | IC > 0 folds | Signal Quality |
|---|---|---|---|---|
| GNN-EmbNorm-T1.0 | -0.0325 | -1.26 | 13/40 | WEAK, NOT SIG |
| GNN-EmbNorm-T0.3 | -0.0325 | -1.26 | 13/40 | WEAK, NOT SIG |
| GNN-EmbNormInv-T0.3 | -0.0325 | -1.26 | 13/40 | WEAK, NOT SIG |
| GNN-ValueHead | -0.0165 | -0.96 | 19/40 | NOISE, NOT SIG |

**Key finding:** IC is slightly negative for EmbNorm (mean IC = -0.0325, t = -1.26). Not statistically significant.
The GNN embedding is primarily a *geopolitical activity proxy* (92.2% GDELT inputs), not a return-predictive financial state.

### 2c — Option 1 (Contrarian Inversion) Result:

ΔSharpe vs EqualWeight: -0.022. **No improvement.** IC = -0.033 with IC Std = 0.163 — too noisy for contrarian to work. Neither direction is consistent enough across 40 folds.

---

## 3. Root Cause Analysis (Confirmed)

**Why the GNN has no return signal:**
1. 92.2% of 977,870 observations are `geopolitical_event` (GDELT backfill).
2. GNN embedding encodes *GDELT activity level* around an entity, not return-predictive financial state.
3. The GNN architecture is correct. The input data is unbalanced.
4. Training objective: contrastive + value quantile head. Neither explicitly optimises for return prediction.

**The two paths to fix this:**
- **Option 2 (code fix, fast):** Add return-prediction auxiliary loss to GNN training → push embeddings to encode return-relevant features without waiting for more data.
- **Option 3 (data fix, slow):** Balance observation types by backfilling physical sensors so GDELT < 50% of total obs. Data-gated, ~weeks.

---

## 4. Next Step: Option 2 — Return Auxiliary Loss

**What to build:**
1. Add `return_pred_head` MLP to GNN model: `instrument_emb → [128→64→1]`, outputs scalar return prediction.
2. During training, for instrument nodes with known next-period log-returns in `entity_observations`, compute MSE loss: `L_ret = MSE(return_pred_head(h_i), actual_log_return_i)`.
3. Combined loss: `L_total = L_contrastive + λ_val × L_value + λ_ret × L_ret` where `λ_ret` starts at 1.0.
4. Retrain GNN with this objective.
5. Re-run `scripts/phase40_gnn_backtest.py` — expect IC to improve because embedding now explicitly encodes return signal.

**Files to modify:**
- `agent/models/gnn/model.py` — add `return_pred_head` MLP, expose it in `forward()` output or via a dedicated method.
- `agent/models/gnn/trainer.py` — add `_compute_return_loss()`, add `λ_ret` hyperparameter, wire into training loop.
- `scripts/retrain_gnn.py` (or equivalent retrain script) — to trigger a fresh retrain with the new loss.

**Key constraint:** Walk-forward correctness. The return labels for training must come only from observations that precede the current training window — no leakage. The existing instrument_daily observations in the DB are timestamped, so this is achievable by filtering `observed_at < fold_end_ts` during the loss computation. But for the initial retrain (not walk-forward GNN retraining), train on all available instrument_daily observations in the DB.

**Expected IC target:** If return aux loss works, Mean IC should move from -0.033 → +0.03 to +0.07 range. If it stays near 0, the embedding capacity is the bottleneck (hidden_dim=128 too small for dual task) and we need Option 3 or a larger model.

---

## 5. Operational Rules (Never Forget)

1. **Python path:** Always `/home/becmachlean/anaconda3/bin/python` — NOT `conda run`, NOT `python3`.
2. **Run long scripts in background:** `> /tmp/out.txt 2>&1 &` then `tail -f /tmp/out.txt`. Piping to grep buffers output.
3. **DB path:** `.tirra_pipeline/pipeline.db` (relative to project root).
4. **Model path:** `.tirra_pipeline/gnn_model.pt` (23.7 MB, 1,858,459 params, 10 epochs, EWC-enabled).
5. **`EqualWeightStrategy.name` returns `"equal_weight"`** (snake_case) — comparison table key is `results["equal_weight"]`.
6. **GNN forward does not mutate TGN memory.** Multiple fold calls on the same loaded model are safe.
7. **`model.predict_value()` requires the embeddings dict** — call `model.forward()` first, then pass result to `predict_value()`.

---

## 6. Git State

```
HEAD → main (a2daf7b) — unchanged from last session
```

Files modified this session (not committed):
- `scripts/phase40_gnn_backtest.py` — IC diagnostic, temperature variant, invert flag

---

## Related

- [[quant_training_ground]] — roadmap and phase ordering
- [[world_state_prediction_methodology]] — four-tier evaluation framework
- [[chat_checkpoint_2026-04-28_session_end_full]] — previous session (Phase 40 full results + methodology)
