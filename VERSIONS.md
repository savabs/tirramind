---
title: VERSIONS — TirraMind GNN Phase & Kernel Version Log
tags:
  - doc/wiki
  - topic/gnn
  - status/active
---

# VERSIONS — TirraMind GNN: Phase & Kernel Version Log

> **Update this file EVERY time a new version is pushed to Kaggle or a phase completes.**
> Key IDs: Kaggle kernel slug `deeperisbetter/tirramind-phase50`, notebook fingerprint, checkpoint epoch.

### Trust cutoff (standing rule)

**Only runs on or after `2026-05-28` are trustworthy** for Phase 50 decisions. Earlier kernel versions (V38–V39 and pre-cutoff artifacts) had broken alignment, contrastive loss, and/or blocked gradients — **do not use for IC comparison or resume**.

### Timestamp discipline (standing rule)

Every new row MUST include:
- **`pushed_at`** — ISO-8601 UTC when `kaggle kernels push` ran (also written to `.tirra_pipeline/kaggle_state.json`)
- **`completed_at`** — UTC when training finished or kernel terminal status was recorded
- **`kernel_status`** — last Kaggle status string (`COMPLETE`, `ERROR`, `CANCEL_ACKNOWLEDGED`, …)
- Local artifact paths with **mtime** when downloaded (e.g. `epoch_090.pt @ 2026-05-30T12:53:00Z`)

Status checks: `python3 scripts/kaggle_launch.py --status` and `kaggle kernels status deeperisbetter/tirramind-phase50`.

### Version naming (one ID — no drift)

**`V{N}` is the only version number we use in repo, logs, and conversation.**

| Place | Field | Must equal |
|-------|--------|--------------|
| Notebook | `_NOTEBOOK_CONFIG["kernel_version"]` | **N** |
| Launcher | `CANONICAL_KERNEL_VERSION` in `scripts/kaggle_launch.py` | **N** |
| Log | Banner `Kernel vN` | **N** |
| Kaggle website | **Version N** on [tirramind-phase50](https://www.kaggle.com/code/deeperisbetter/tirramind-phase50) | **N** |
| This file | Row **VN** + “Current notebook target: **VN**” | **N** |
| State | `.tirra_pipeline/kaggle_state.json` → `kernel_version` | **N** (last push) |

**Before each new push:** increment **N** in all four repo places, then push. Do not push with a stale `kernel_version` in the notebook — Kaggle will still increment its counter and you will get “Kaggle Version 50 / repo V47” confusion again.

**Version sync (mandatory before push):** `python3 scripts/kaggle_launch.py --verify-version`

**Current notebook target: V73** (`kernel_version: 73` in `tirramind_kaggle_phase50.ipynb`, `CANONICAL_KERNEL_VERSION = 73` in `kaggle_launch.py`)

**Last status check:** `2026-06-08` — **V73 prep** — Phase B Stage-1 SSL-only HetTGN (`--preset phase50_stage1_ssl`). Post-train: PurgedRanker + Stage2-RidgeRanker. See [[gnn_two_stage_spec]].

---

## ACTIVE KERNEL VERSIONS (Phase 50 — Kaggle: `deeperisbetter/tirramind-phase50`)

Each push increments **VN** everywhere first, then `kaggle kernels push` (Kaggle UI shows the same **Version N**).

| Ver | pushed_at (UTC) | completed_at (UTC) | kernel_status | Fingerprint | Epochs | IC Result | Status |
|-----|-----------------|--------------------|---------------|-------------|--------|-----------|--------|
| V40 | 2026-05-28 | — | — | ? | 1→75 | raw head only (pre-concat) | done |
| V41 | 2026-05-29 | — | — | `5a2b8f3c9d1e` | 63→75 | history-alignment fix | done |
| V42 | 2026-05-29 | — | — | pending | 75→90 | +nan (bad ckpt load) | dead |
| V43 | 2026-05-29 ~13:26 | — | COMPLETE (quickrun) | `d4e5f6a7b8c9` | 0→10 | **+0.3622** (1-week snap; not walk-forward) | done |
| V44 | 2026-05-30 | — | CANCELLED | `f8b2c1d4a3e5` | 0→90 | — | cancelled (CPU: `enable_gpu=False`) |
| V45 | 2026-05-30 | 2026-05-30 | COMPLETE | `e15903bd5462` | 0→90 | **+0.0369** (ICIR +0.236) | **done — best full run** |
| V46 | 2026-05-31 | 2026-05-31 | COMPLETE | `a1b2c3d4e5f6` | 0→90 | +0.0250 (ICIR +0.134) | done |
| V47a | 2026-05-30T13:22:44Z | 2026-05-30T~13:30Z | **ERROR** | — | — | — | **dead** (notebook still v46; numpy crash; never trained) |
| V47b | 2026-06-01T09:04:26Z | 2026-06-01T~09:15Z | **ERROR** | `06bed9202806` | 0→90 | — | **dead** (numpy==1.26.4 pin → `numpy.rec`; never trained) |
| V47c | 2026-06-01T09:12:52Z | — | — | `06bed9202806` | — | — | **renumbered → V49** (banner still said v47 on Kaggle; see V49) |
| **V49** | **2026-06-01T09:12:52Z** | **2026-06-01T~10:33Z** | **ERROR**† | `06bed9202806` | 0→90 | **+0.0378** ValueHead (t=1.43); PurgedRanker **−0.034** (fail) | **do not promote** — see V49 notes |
| **V50** | 2026-06-02T04:34:20Z | 2026-06-02T~04:38Z | **ERROR** | `65d7fae8a2a1` | — | — | **dead** — `NameError: _cfg` in train cell In [6] (never trained) |
| **V51** | 2026-06-02T05:15:21Z | 2026-06-02T~08:16Z | **ERROR**† | `a653121dc22c` | 0→90 | PurgedRanker **−0.031** (eval-only) | **trained; do not promote** — post-train eval bug (`self` in `evaluate`) |
| **V52** | 2026-06-02T08:19:44Z | 2026-06-02T13:59Z | **COMPLETE** (eval kernel) | `594cc5dd3d5a` | 0→90 | **+0.0474** PurgedRanker (t=1.95, ICIR +0.335) | **promoted** — beats V45 baseline (+0.029) |
| **V53** | 2026-06-05T04:54:24Z | 2026-06-05T05:04Z | **ERROR**† | `ad9b1e17bd58` | 0→2 | — (smoke) | **smoke PASS** — N1 doctrine trained; post-train scipy crash only |
| **V54** | 2026-06-05T06:02:39Z | 2026-06-05T06:19Z | **ERROR**† | `c432203d1609` | 0→10 | — (mid-run) | **trained 10/10** — post-train numpy/scipy import error only |
| **V55** | 2026-06-05T08:13:13Z | 2026-06-05T09:43Z | **COMPLETE** | `65f113ef7127` | 0→90 | PurgedRanker **+0.027** (t=0.87) | **do not promote** — embedding-only, return loss flat |
| **V56** | 2026-06-05T10:51:04Z | 2026-06-05T11:00Z | **ERROR**† | `5090a7146ca3` | 0→2 | — | trained 2/2; eval cwd bug (`scripts/` not found) |
| **V57** | 2026-06-05T12:44:18Z | 2026-06-05T14:00Z | **ERROR**† | `d1dbeedfd155` | 0→2 | — | trained 2/2; 64min on retrain val/test eval; diagnostics `build_graph` bug |
| **V58** | 2026-06-05T14:11:34Z | 2026-06-05T15:10Z | **COMPLETE** | `99094463a28b` | 0→2 | smoke +0.032 / full +0.001 | **GPU** — 59min; duplicate full backtest cell |
| **V59** | 2026-06-05T14:20:10Z | 2026-06-05T15:37Z | **COMPLETE** | `7742f1ec3d10` | 0→2 | smoke +0.005 / full +0.029 | **CPU** — 75min; duplicate full backtest cell |
| **V60** | 2026-06-05T16:50:37Z | 2026-06-05T17:20Z | **COMPLETE** | `74e185daa6ed` | 0→10 | smoke -0.044 (PurgedRanker) | **concat-head + VICReg** — 50win; embedding collapse 48.4%; return loss flat |
| **V61** | 2026-06-06T04:49:46Z | 2026-06-06T05:13Z | **COMPLETE** | `734f23a6a500` | 0→10 | smoke PurgedRanker **-0.069** / EmbNorm **+0.058** | **ContraNorm + log loss** — collapse 24.7% (↓48%); return loss flat; log_loss was no-op (auto_tune) |
| **V62** | 2026-06-06T07:29:26Z | 2026-06-06T07:48Z | **COMPLETE** | `2925444fea1f` | 0→10 | smoke PurgedRanker **-0.072** / EmbNorm **+0.030** | **return isolated** — return flat @209.39 ep3; collapse 14.1%; eff rank 1.0/93 |
| **V63** | 2026-06-06T08:25:24Z | 2026-06-06T09:20Z | **COMPLETE** | `7c689eabb6d4` | 0→10 | smoke PurgedRanker **-0.119** | **V52-exact smoke** — plateau@ep7; gate FAIL |
| **V64** | 2026-06-06T09:45:29Z | 2026-06-06T10:26Z | **COMPLETE** | `e74b4f330dee` | 0→10 | smoke PurgedRanker **-0.095** | **grad-flow diag** — clamp(+5) saturation pred_std=0; plateau@ep5; gate FAIL |
| **V65** | 2026-06-07T06:13:56Z | — | pending | pending | 0→10 | — | **clamp50+layernorm+diag** — H2 fix + in-sample IC |
| **V66** | 2026-06-07T06:14:05Z | — | pending | pending | 0→10 | — | **no CSRC + vicreg0.1** — H4 ablation |
| **V67** | 2026-06-07T06:16:12Z | — | **ERROR**† | `e1308431eff7` | 0→10 | — | **PCGrad** — ep1 tensor size crash (fixed: zero-pad grads) |
| **V68** | 2026-06-07T08:22:15Z | — | pending | pending | 0→10 | — | **eval concat gate** — V65 train + GNN-ConcatReturnHead primary IC |
| **V69** | 2026-06-07T08:22:26Z | — | pending | pending | 0→10 | — | **listnet τ=0.5** + concat eval gate |
| **V70** | 2026-06-07T08:22:45Z | — | pending | pending | 0→10 | — | **PCGrad safe** + concat eval gate |
| **V71** | 2026-06-07T08:23:04Z | — | COMPLETE | pending | 0→10 | ConcatReturnHead **not run** (load_model bug) | **τ=0.5 + PCGrad** — eval misaligned; gate FAIL |
| **V72** | 2026-06-07T13:38:36Z | 2026-06-07T~14:20Z | **ERROR**† | `06dcbc9c1b78` | 0→10 | ConcatReturnHead **+0.0587** (t=1.03) **FAIL** | **load_model concat fix** — eval aligned; gate FAIL; post-eval JSON bug |
| **V73** | 2026-06-08T09:13:02Z | — | pending | pending | 0→90 | — | **Stage-1 SSL** — `phase50_stage1_ssl`; no return/CSRC/concat; post-train PurgedRanker + Stage2 |

**V49 notes:** F-11 GRU chronological, V45-style sampling (`gdelt_frac=0.05`, `max_windows=200`), `return_concat_head`, `--skip-eval`. Training ~76 min; ~46 s/epoch; 16× embedding collapse warnings. †Kernel ERROR on post-train `scipy.stats.spearmanr` cell. **Kaggle output API lists `epoch_090.pt` / `gnn_model_phase50.pt` as ~875 B stubs only** — local walk-forward used `epoch_086.pt` (20 MB) + May `gnn_model_phase50.pt` shell; results in `.tirra_pipeline/ic_results_v49_epoch086.json`. Phase 41b gate **not met** (need IC>0.03 **and** t>2). **Keep V45 as reference** until V50.

**Push checklist:** `smoke_test_gnn.py` 9/9 → `--verify-version` → `--push-only --epochs 90` → record `pushed_at` / terminal status / Kaggle Version N here.

### Trusted local artifacts (mtime ≥ 2026-05-28)

| Path | mtime (local) | Notes |
|------|---------------|-------|
| `.tirra_pipeline/checkpoints/phase50/epoch_071.pt` … `epoch_090.pt` | 2026-05-30 18:21–18:23 +0530 | **V45/V46 only** — not V49 until re-downloaded |
| `.tirra_pipeline/kaggle_logs_full_latest.txt` | 2026-06-01 | **V49** full training log (90/90) |
| `.tirra_pipeline/ic_results_v45_epoch090.json` | 2026-06-01 | **V45** walk-forward IC baseline (epoch 90, honest PurgedRanker +0.029) |
| `.tirra_pipeline/ic_results_v49_epoch086.json` | 2026-06-01 | **V49** walk-forward IC (epoch 86; PurgedRanker −0.034) |
| `.tirra_pipeline/kaggle_downloads_v49/phase50_ckpts/epoch_001.pt`–`epoch_086.pt` | 2026-06-01 | **V49** checkpoints (~20 MB each); 087+ not recoverable from Kaggle |
| `.tirra_pipeline/gnn_model_phase50.pt` | 2026-05-24 | **Before cutoff — do not trust** |
| `.tirra_pipeline/kaggle_state.json` | `kernel_version: 51`, fp `a653121dc22c` | **V51** push metadata |
| `.tirra_pipeline/kaggle_logs_v50_latest.txt` | 2026-06-02 | **V50** failure log (`NameError: _cfg`) |
| `.tirra_pipeline/ic_results_v52_epoch090_kaggle_eval.json` | 2026-06-02 | **V52** Kaggle eval result (PurgedRanker mean IC +0.0474; best strategy) |

### Pre-cutoff archive (do not use)

V38–V39 and any checkpoint/model dated **before 2026-05-28** are retained for history only.


### V42 Planned Command Diff vs V41

```diff
- (no --use-concat-head flag)
+ --use-concat-head
```

### V43 Research Candidates

- **VICReg** (`arxiv:2105.04906`): 3 regularizers — variance threshold, invariance (MSE), covariance decorrelation. Prevents dimensional collapse without negative samples.
- **GradNorm** (`arxiv:1711.02257`): Auto-balances multi-task loss weights based on gradient magnitudes. Addresses obs_type CE dominating return loss 154:1.
- **HGT type-specific projections** (`arxiv:2003.01332`): Per-relation-type Q/K/V matrices for heterogeneous attention. Better ghost pattern discrimination.

---

## PROJECT PHASES (Pipeline + Architecture)

| Phase | Date | What | Key Files | Status |
|-------|------|------|-----------|--------|
| 38 | 2026-04-18 | Pipeline plumbing — real tool output → convergence detection → feature generation | `agent/pipeline/dags/` | done |
| 39 | 2026-04-19 | Pipeline robustness — NaN handling, missing entity guards, feature-generation hardening | `agent/pipeline/dags/feature_generation.py` | done |
| 40 | 2026-04-20 | **First real GNN training** on pipeline data. Created `scripts/retrain_gnn.py`. 6 steps: CLI, auto-tune, `--since`, `--backup`, Rich output, W&B. | `scripts/retrain_gnn.py`, `agent/models/gnn/trainer.py` | done |
| 41 | 2026-04-21 | GNN signal extraction, IC diagnostic, model refresh hardening | `scripts/phase40_gnn_backtest.py` | done |
| 42 | 2026-04-21 | Entity diversity expansion — more instrument types, richer graph | `agent/models/gnn/graph_builder.py` | done |
| 43 | 2026-04-22 | High-volume DAG wiring — batch processing, rate limiting | `agent/pipeline/dags/` | done |
| 44 | 2026-04-22 | Batch 2 DAG wiring — additional data sources | `agent/pipeline/dags/` | done |
| 45 | 2026-04-22 | Strategy transformer integration | `agent/models/` | done |
| 46 | ? | ? | ? | ? |
| 47 | 2026-04-23 | Extended backfill, doctrine complete. Data leakage discovered (IC=0.48 fake). | `[[data_strategy_doctrine]]` | done |
| 48 | ? | ? | ? | ? |
| 49 | 2026-04-24 | Phase 49 complete | ? | done |
| **50** | **2026-05-26+** | **Price features + residual returns. CSRC loss. Embedding collapse fix. Ghost pattern mechanism.** | `trainer.py`, `retrain_gnn.py`, `het_tgn.py`, notebook | **active** |

---

## NOTEBOOK CONFIG FINGERPRINT REFERENCE

The notebook fingerprint is `sha256(json.dumps(config, sort_keys=True))[:12]`.
It uniquely identifies the exact configuration. If two runs have the same fingerprint, they are identical configs.

**Known fingerprints:**
| `5a1971b5b8e6` | V73 | v73_stage1_ssl |
| `52c98aa03b15` | V73 | v73_stage1_ssl |
| `06dcbc9c1b78` | V72 | v72_load_model_concat_fix |
| `7f95e4013ab1` | V71 | v68d_tau05_pcgrad |
| `d212968e44df` | V70 | v68c_pcgrad_safe |
| `4e5e6af43f43` | V69 | v68b_listnet_tau05 |
| `333c47f9a412` | V68 | v68a_eval_concat_gate |
| `e1308431eff7` | V67 | v67c_pcgrad |
| `e20f94a324c8` | V66 | v66b_no_csrc_vicreg |
| `b16ec8e1e583` | V65 | v65a_clamp_batchnorm_diag |
| `e74b4f330dee` | V64 | grad_flow_diag_v64_import_fix |
| `fad544f95e77` | V64 | grad_flow_diag_v64 |
| `7c689eabb6d4` | V63 | v52_exact_smoke_ep3_gate_v63 |
| `2925444fea1f` | V62 | return_isolated_v62_no_autotune |
| `734f23a6a500` | V61 | train_efficiency_v61_contranorm_logloss |
| `74e185daa6ed` | V60 | train_efficiency_v60_concat_vicreg |
| `7742f1ec3d10` | V59 | n1_eval_smoke_v59_cpu |
| `99094463a28b` | V58 | n1_eval_smoke_v58 |
| `d1dbeedfd155` | V57 | n1_eval_smoke_v57_cwdfix |
| `5090a7146ca3` | V56 | n1_eval_smoke_v56 |
| `65f113ef7127` | V55 | n1_doctrine_full_v55 |
| `c432203d1609` | V54 | n1_doctrine_midrun_v54 |
| `ad9b1e17bd58` | V53 | n1_doctrine_smoke_v53 |
| `594cc5dd3d5a` | V52 | gru_chronological_f11_v52_contrastive_cap1x |
| `a653121dc22c` | V51 | gru_chronological_f11_v51_cfg_namefix |
| Fingerprint | Version | Config highlights |
|-------------|---------|-------------------|
| `5a2b8f3c9d1e` | V41 | CSRC, history fix, resume ep63, no concat head |
| `06bed9202806` | V49 | F-11 GRU, gdelt_frac=0.05, skip-eval, collapse run |
| `65d7fae8a2a1` | V50 | contrastive_log_var_min=-1.0, scipy upgrade cell |

---

## UPDATE INSTRUCTIONS

When pushing a new version:
1. Increment `kernel_version` in notebook `_NOTEBOOK_CONFIG`
2. Set `CANONICAL_KERNEL_VERSION` in `scripts/kaggle_launch.py` to the same integer
3. Set **Current notebook target: V{N}** line in this file to the same integer
4. Run `python3 scripts/kaggle_launch.py --verify-version` (must pass)
5. Add row with **`pushed_at`**, **`completed_at`**, **`kernel_status`**, fingerprint, IC, local download mtimes
6. Run `python3 scripts/smoke_test_gnn.py` (9/9) before push
7. After run: `kaggle kernels status …` and update row terminal status + timestamps
8. Update status of previous version (running → done/dead/error)
9. If new code changes: update LESSONS.md with any new fuckups
10. Commit VERSIONS.md and LESSONS.md
