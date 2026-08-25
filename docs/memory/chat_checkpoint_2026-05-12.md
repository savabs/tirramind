---
title: "Checkpoint: 2026-05-12 — v15 Kernel Training (epoch 33/38)"
tags:
  - doc/checkpoint
  - phase/47
  - topic/gnn-training
  - topic/kaggle
  - status/active
date: 2026-05-12
---

# Checkpoint: 2026-05-12

## Session Summary

Fixed the recurring `--wandb-run-name` crash that had killed every Kaggle kernel (v11–v14). Root cause: git clone always fails (Kaggle secrets service returns Connection error for `tirramind_token`), so the kernel falls back to the `tirramind-code` dataset, which still had the old `pipeline_orchestrator.py` with `--wandb-run-name` (argparser only accepts `--wandb-run` → exit 2 → crash at ~60s).

Two zips in the dataset needed to be fixed — `tirramind_code.zip` was rebuilt in a prior session but `tirramind_code_p47.zip` still had the old flag. This session fixed both, uploaded dataset v13, waited 3 minutes, pushed kernel v15.

**v15 is actively training: confirmed at epoch 33/38 as of end of session.**

---

## Current State

### Kaggle Kernel
- **Kernel**: `deeperisbetter/tirramind-h-g` — **v15 RUNNING**
- **Progress**: epoch 33 of 38 at session end
- **W&B**: entity `999-sbpatel`, project `tirramind` — wired, WANDB_API_KEY loaded from Kaggle secret
- **GNN**: HetTGN, `hidden_dim=128`, `num_layers=2`, started from epoch 28 checkpoint

### Datasets
- `tirramind-code` v13: both `tirramind_code.zip` and `tirramind_code_p47.zip` fixed — `--wandb-run` (correct)
- `tirramind-data` v12: flat checkpoint at `checkpoints/epoch_028.pt` — correct

### Local Files
- `/tmp/tirramind_code_upload/` — both zips fixed, already uploaded
- `/tmp/hg_kernel/` — kernel metadata, v15 already pushed

---

## What Happened (Chronological)

1. Discovered `tirramind_code_p47.zip` still had old `--wandb-run-name` flag
2. Replaced it: `cp tirramind_code.zip tirramind_code_p47.zip` (both now identical, fixed)
3. Uploaded dataset v13: both zips upload successful
4. Waited 3 minutes for Kaggle dataset processing
5. Pushed kernel v15
6. Confirmed RUNNING at 90s mark (past old crash point)
7. Confirmed RUNNING at 7min mark
8. User reported epoch 33/38 visible in Kaggle UI — training confirmed

---

## Immediate Next Steps (Next Session)

### 1. Confirm v15 COMPLETE and download output
```bash
export KAGGLE_API_TOKEN=$(python3.11 -c "import json; print(json.load(open('/home/becmachlean/.kaggle/kaggle.json'))['key'])") && \
kaggle kernels status deeperisbetter/tirramind-h-g
```

### 2. Download log + checkpoints
```bash
rm -rf /tmp/hg_v15_out && \
kaggle kernels output deeperisbetter/tirramind-h-g -p /tmp/hg_v15_out --force && \
find /tmp/hg_v15_out -name "epoch_*.pt" | sort && \
grep -a "epoch\|IC\|loss\|Block\|wandb\|WANDB" /tmp/hg_v15_out/tirramind-h-g.log | tail -30
```

### 3. Upload new checkpoint to tirramind-data dataset
```bash
LATEST=$(find /tmp/hg_v15_out -name "epoch_*.pt" | sort | tail -1 | xargs basename)
mkdir -p /tmp/staging_v15/.tirra_pipeline/checkpoints
cp /tmp/hg_v15_out/.../$LATEST /tmp/staging_v15/.tirra_pipeline/checkpoints/$LATEST
cp /home/becmachlean/2024/projects/tirramind_v1/.tirra_pipeline/pipeline.db /tmp/staging_v15/.tirra_pipeline/pipeline.db
cd /tmp/staging_v15 && zip -r /tmp/tirramind_data_v15.zip .tirra_pipeline/ && cd -
kaggle datasets version -p /tmp/tirramind_upload_data -m "v13: $LATEST checkpoint"
```

### 4. Push v16 kernel to continue training (epoch 38 → 50+)
```bash
kaggle kernels push -p /tmp/hg_kernel
```

### 5. After epoch 50: run backtest on Kaggle
- Script: `scripts/phase40_gnn_backtest.py`
- Target: IC > 0.03, |t| > 2.0 (currently IC = -0.033 at epoch 28)

---

## Key Facts (DO NOT DERIVE AGAIN)

| Fact | Value |
|---|---|
| Kaggle auth | `export KAGGLE_API_TOKEN=$(python3.11 -c "import json; print(json.load(open('/home/becmachlean/.kaggle/kaggle.json'))['key'])")` |
| Kernel slug | `deeperisbetter/tirramind-h-g` |
| Checkpoint format | FLAT: `checkpoints/epoch_NNN.pt` — NOT `checkpoints/h_g/epoch_NNN.pt` |
| Python locally | Always `python3.11` — never `python3` |
| W&B entity/project | `999-sbpatel` / `tirramind` |
| Dataset processing wait | ≥3 minutes before pushing kernel |
| Root cause of v11–v14 crashes | Both zips in tirramind-code dataset must have `--wandb-run` (not `--wandb-run-name`) |

---

## Related

- [[kaggle_runbook]]
- [[quant_training_ground]]
- [[tirramind_structure]]
