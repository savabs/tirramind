---
title: "Checkpoint 2026-05-26: EWC sidecar fix landed; kaggle_watch built"
tags:
  - doc/checkpoint
  - phase/46
  - phase/47
  - topic/training
  - topic/gnn
  - topic/kaggle
date: 2026-05-26
---

# Checkpoint — 2026-05-26

## Session Summary

Diagnosed and fixed the 5-epoch loss spike bug in GNN training, built the live Kaggle monitoring dashboard, and confirmed all 16 EWC tests pass.

## What Was Accomplished

### Fix: EWC sidecar persistence (commit `5ba8b2c`)
- **Bug**: `pipeline_orchestrator.py` runs `retrain_gnn.py` as sub-processes in 5-epoch blocks. `--resume` loaded `epoch_NNN.pt` which does NOT contain EWC state → `self._ewc_state = None` → unregularised Adam overshot → spike (ep34: loss 7013, ep39: 7899).
- **Fix**: After Fisher computation, `train()` now writes `{checkpoint_dir}/ewc_state.pt` sidecar (Fisher diagonal + anchor). On resume, `train()` checks for the sidecar and loads it before training begins. EWC is now active from epoch 1 of every block.
- **Tests**: All 16 EWC tests pass via `PYTHONPATH=. ~/.local/bin/pytest tests/test_ewc.py`
- **Confirmed in v15 log**: "EWC state serialised" at t=16728s (end of block 1) and t=33291s (end of block 2) — these were `save_model()` (old path). "EWC sidecar saved" will appear for first time in v16.

### New: `scripts/kaggle_watch.py`
Self-refreshing terminal dashboard — shows Kaggle kernel status + W&B training history in one view. Usage: `python3.11 scripts/kaggle_watch.py` (60s refresh), `--once` for snapshot. No W&B package needed (uses GraphQL directly via urllib).

### v15 analysis
- Kernel ran epochs 28–40 (3 blocks), cancelled at 12hr limit
- Loss pattern: ep29=41895→ep33=150 (Block 1 OK, no prior EWC), ep34=7013→ep38=116 (spike: Block 2 had no sidecar), ep39=7899→ep40=3865 (spike: Block 3 had no sidecar)
- `time_delta` loss = NaN every epoch — known issue, not yet investigated
- W&B NOT connected (WANDB_API_KEY secret failed in Kaggle) — old stale run visible, ignore
- epoch_040 checkpoint at `/tmp/hg_v15_out/tirramind_v1/.tirra_pipeline/checkpoints/h_g/epoch_040.pt`
- `gnn_model_h_g.pt` at `/tmp/hg_v15_out/tirramind_v1/.tirra_pipeline/gnn_model_h_g.pt` (contains EWC from ep40: 173 Fisher params, lambda=1000.0)

## Current State

- GitHub: `savabs/tirramind` main @ `5ba8b2c` — EWC fix + kaggle_watch
- v16 needs: epoch_040.pt uploaded to tirramind-data, code re-zipped + uploaded to tirramind-code, W&B secret fixed, Cell 11 → `--epochs 50 --resume 40`
- See [[kaggle_runbook]] "V16 Upload Checklist" for exact steps

## Immediate Next Steps

1. **Stage + upload epoch_040.pt** to tirramind-data dataset:
   ```bash
   mkdir -p /tmp/staging_v16/.tirra_pipeline/checkpoints
   cp /tmp/hg_v15_out/tirramind_v1/.tirra_pipeline/checkpoints/h_g/epoch_040.pt \
      /tmp/staging_v16/.tirra_pipeline/checkpoints/epoch_040.pt
   cp /home/becmachlean/2024/projects/tirramind_v1/.tirra_pipeline/pipeline.db \
      /tmp/staging_v16/.tirra_pipeline/pipeline.db
   cd /tmp/staging_v16 && zip -r /tmp/tirramind_data_v16.zip .tirra_pipeline/
   kaggle datasets version -p /tmp/tirramind_upload_data -m "v16: epoch_040 + EWC sidecar era"
   ```
2. **Re-upload tirramind-code** (includes sidecar fix + kaggle_watch.py)
3. **Fix W&B secret** in Kaggle Settings → Secrets
4. **Push kernel v16**: Cell 11 `--epochs 50 --resume 40`, Cell 13 `if epoch_num > 40`
5. **Run backtest** on epoch_040 after v16 completes: `python3.11 scripts/phase40_gnn_backtest.py`

## Key Facts Table

| Fact | Value |
|---|---|
| Best checkpoint | `epoch_040.pt` at `/tmp/hg_v15_out/...` |
| EWC Fisher params | 173, lambda=1000.0 |
| Return loss (stable from ep34) | ~91.4 |
| `dt` loss | NaN every epoch (unresolved) |
| Working pytest | `PYTHONPATH=. ~/.local/bin/pytest` (Python 3.10.12 — NOT python3.11) |
| System pytest broken | `AttributeError: __spec__` from `py._vendored_packages.apipkg` |
| EWC sidecar file | `{checkpoint_dir}/ewc_state.pt` (will be created by v16, not present yet) |

## Related

- [[kaggle_runbook]]
- [[tirramind_structure]]
