---
title: "Checkpoint 2026-05-04"
tags:
  - doc/checkpoint
  - phase/42
  - topic/gnn
  - topic/training
---

# Checkpoint — 2026-05-04

## What Was Accomplished This Session

1. **H-G run already completed on Kaggle** — epochs 31–40 trained successfully.
2. **Downloaded H-G artifacts** — user ran the cp commands:
   - `.tirra_pipeline/gnn_model_h_g.pt` (24 MB)
   - `.tirra_pipeline/checkpoints/h_g/epoch_031.pt` … `epoch_040.pt` (10 × 22 MB)
3. **Built new upload zip** — `tirramind_data_upload.zip` (376 MB) contains `pipeline.db` + `epoch_030.pt`. This is ready for future Kaggle uploads if needed.
4. **Diagnosed broken `.venv`** — the project venv symlinks to `/home/becmachlean/anaconda3/bin/python3.11` which no longer exists (Anaconda uninstalled). Rebuilt venv with system python3.10.
5. **Started torch install** — `pip install torch==2.2.2` was running when session ended. It was **killed before completing**. Venv is now rebuilt but torch is NOT installed.

## Immediate Next Step (top priority on resume)

**Finish installing torch + torch-geometric, then run the H-G backtest:**

```bash
cd /home/becmachlean/2024/projects/tirramind_v1

# Install deps (torch download is ~800MB, takes a few min)
.venv/bin/pip install --quiet torch==2.2.2
.venv/bin/pip install --quiet torch-geometric

# Run the H-G evaluation
.venv/bin/python scripts/phase40_gnn_backtest.py --model .tirra_pipeline/gnn_model_h_g.pt
```

**What to watch for:**
- ICIR > 0.25 = directional signal exists
- ICIR > 0.40 = strong signal
- Compare vs H-A baseline (epochs 1–30, shared checkpoints folder)

## Current State of All Files

| File | State |
|---|---|
| `agent/models/gnn/trainer.py` | ✅ 4 structural fixes committed (`1ffd453`) |
| `scripts/retrain_gnn.py` | ✅ wandb CLI flags complete |
| `scripts/wandb_monitor.py` | ✅ agent-callable live monitor |
| All 4 Kaggle notebooks | ✅ wandb pip + Kaggle secret + flags |
| `.tirra_pipeline/gnn_model_h_g.pt` | ✅ Downloaded (epoch 40) |
| `.tirra_pipeline/checkpoints/h_g/` | ✅ epoch_031–040 present |
| `.venv` | ⚠️ Rebuilt (python 3.10) but torch NOT installed yet |
| `tirramind_data_upload.zip` | ✅ Built (376 MB, contains db + epoch_030) |

## Key Facts

- **Python env**: `.venv` rebuilt from system python3.10 (`/usr/bin/python3`). Old one was broken (linked to missing anaconda python3.11).
- **Git HEAD**: `1ffd453` — 4 structural fixes committed and pushed to `origin/main`
- **GitHub remote**: `git@github.com:savabs/tirramind.git`
- **H-G trained epochs**: 31–40 (started from epoch_030.pt seed, the CFTC derived features hypothesis)
- **H-A baseline**: epochs 1–30 in `.tirra_pipeline/checkpoints/` (no hypothesis subfolder)
- **wandb**: Not yet active — Kaggle Secret `WANDB_API_KEY` still needs to be added by user. Will stream on next training run.
- **Next hypothesis runs pending**: H-A epochs 31–40, H-D fresh retrain (lr=3e-4), H-H (needs vessel data first)

## Related

- [[kaggle_runbook]]
- [[quant_training_ground]]
