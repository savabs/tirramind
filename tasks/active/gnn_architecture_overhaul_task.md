---
title: "Task: GNN Architecture Overhaul"
tags:
  - doc/task
  - phase/50
  - topic/training
  - topic/gnn
  - topic/architecture
  - layer/learning
  - status/active
---

> Primary document: [[gnn_architecture_overhaul_task.html]]

Research: [[gnn_architecture_overhaul]]
Spec: [[gnn_architecture_overhaul_spec]] | Phase B: [[gnn_two_stage_spec]]

## Phase A — Honest baselines

- [x] A.1 — Implement `scripts/honest_baseline_audit.py`
- [x] A.2 — Run smoke audit locally, record results in JSON
- [x] A.3 — Run full audit (optional Kaggle checkpoint compare)
- [x] A.4 — Write checkpoint with gate verdict + recommendation

## Phase A-data — DATA_FIX (label pipeline)

- [x] D.1 — Canonical `agent/quant/forward_returns.py` (single label owner)
- [x] D.2 — `scripts/data_label_audit.py` (backfill shift, leakage, shuffled-label)
- [x] D.3 — Align `honest_baseline_audit` IC targets to trainer forward returns
- [x] D.4 — Fix calendar-date ↔ observation-timestamp label alignment
- [x] D.5 — Align `phase40_gnn_backtest.py` IC targets to canonical forward returns
- [x] D.6 — Re-run honest baseline + data audit after fixes; write checkpoint

## Phase B — Decoupled two-stage (gated on A)

- [x] B.0 — Research + spec for two-stage pipeline (only if Phase A shows signal)
- [x] B.1 — Stage 1: self-supervised-only HetTGN training config (`--preset phase50_stage1_ssl`)
- [x] B.2 — Stage 2: embedding export + separate ranker (`export_gnn_embeddings.py`)
- [x] B.3 — Walk-forward eval on Stage 2 ranker (`stage2_ranker_eval.py`)
- [ ] B.4 — Train Stage-1 SSL on Kaggle V73 + full Stage2 eval on SSL checkpoint (prepped; fingerprint `52c98aa03b15`)

## Related

- [[gnn_architecture_overhaul]]
- [[gnn_architecture_overhaul_spec]]
- [[checkpoint_2026-06-07_v72_architecture_handoff]]
- [[quant_training_ground]]
- [[phase41b_gnn_signal_extraction]]
