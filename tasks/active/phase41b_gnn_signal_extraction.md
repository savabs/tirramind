---
title: "Task: Phase 41b — GNN Signal Extraction"
tags:
  - doc/task
  - phase/41b
  - topic/world-model
  - layer/feature-engineering
  - status/active
---

# Task: Phase 41b — GNN Signal Extraction

Status: active
Research: [[phase41b_gnn_signal_extraction]]
Spec: [[phase41b_gnn_signal_extraction_spec]]

## Exit Condition

Mean IC > 0.03 and t-stat > 2.0 for GNN-ReturnHead strategy in `scripts/phase40_gnn_backtest.py`.

## Steps

- [x] 41b.0 — Research complete (`[[phase41b_gnn_signal_extraction]]`)
- [x] 41b.0b — Spec written (`[[phase41b_gnn_signal_extraction_spec]]`)
- [x] 41b.1 — Add `_listnet_loss()` helper function to `agent/models/gnn/trainer.py`
- [x] 41b.2 — Add `use_listnet_return_loss` + `listnet_temperature` fields to `TrainerConfig`
- [x] 41b.3 — Replace Huber return loss with conditional ListNet/Huber + update minimum guard
- [x] 41b.4 — Add `--listnet` flag to `scripts/retrain_gnn.py`
- [x] 41b.5 — Run unit tests: ListNet helper, minimum-2 guard, integration 1-epoch run
- [x] 41b.6 — Build `scripts/phase41b_propagation_diagnostic.py` (Granger causality North Star)
- [x] 41b.7 — Push to GitHub (`savabs/tirramind`), upload to Kaggle dataset, retrain epochs 21–30
- [ ] 41b.8 — Run `scripts/phase40_gnn_backtest.py` → check IC
- [ ] 41b.9 — If IC < target: run `scripts/phase41b_propagation_diagnostic.py` to diagnose

## Kaggle Retrain Command

```bash
python scripts/retrain_gnn.py \
  --auto-tune --listnet \
  --gdelt-frac 0.05 \
  --epochs 30 --resume 20 \
  --backup
```

## Notes

- `auto_tune_loss_weights` and `gdelt_subsample_frac=0.05` already exist — just need the flags passed
- ListNet requires ≥2 instruments per window; windows with 0-1 instruments skip return loss silently
- If IC diagnostic still fails post-retrain, run the Granger diagnostic to confirm signal path
- Current epoch 20 loss = 5.26 (best checkpoint). Retrain from there, don't start from scratch.

## Related

- [[phase41b_gnn_signal_extraction]] — research
- [[phase41b_gnn_signal_extraction_spec]] — spec
