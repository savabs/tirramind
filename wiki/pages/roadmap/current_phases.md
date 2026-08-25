---
title: Current Phases
tags:
  - doc/wiki
  - topic/pipeline
type: roadmap
summary: Snapshot of completed and upcoming TirraMind phases. Updated 2026-05-27.
status: active
source_docs:
  - [[quant_training_ground]]
  - [[kaggle_runbook]]
updated_on: 2026-05-27
---

# Current Phases

## Completed (as of 2026-05-27)

All phases 0–47, 49, 49b are complete. Key milestones:

- **Phases 0–36** — Full observational surface (51 tools), pipeline DAG, GNN architecture, entity linking, signal fusion, RL policy, adversarial layer, world model bridge.
- **Phase 37–38** — First live pipeline run + downstream integration fix.
- **Phase 46** — EWC continuous learning (online GNN, CPU, $0). 16 EWC tests pass.
- **Phase 47** — Historical backfill runner + density audit script.
- **Phase 49** — GNN downstream alignment (belief log-likelihood delta).
- **Phase 49b** — Convergence detection as control signal (`regime_gate.py`).

## Current Focus (2026-05-27)

**GNN training — `tirramind-phase50` Kaggle kernel**

- Active checkpoint: `epoch_021.pt` (phase50 run). Also available: `epoch_040.pt` (H-G run).
- GNN full-pipeline IC = −0.033 at epoch 40 (WEAK — target: >+0.03).
- `return_raw_head` (price feature MLP): ICIR = **+0.467** with correct 26-week history evaluation (TRADEABLE floor, not the GNN's contribution).
- Fixes applied 2026-05-27: `time_delta NaN` guard in both training and EWC paths; `xsnorm_price_feats` cross-sectional normalisation; `freeze_backbone` flag for raw head isolation.
- Next Kaggle push: **V34** — resume from ep40, `use_listnet_return_loss=True`, target IC → +0.03–+0.07.

## Pending / Gated

- **Phase 40** — Real data GNN retrain walk-forward (gated on 3–4 weeks live data accumulation).
- **Phase 48** — Transformer world model + Dreamer RL (hard-gated: density audit pass + Phase 40 results showing ceiling hit).
- **Phase 50** — Regime-stratified replay buffer (after Phase 40 confirms IC baseline).

## Related Pages

- [[pages/architecture/system_overview]]
- [[pages/architecture/execution_engines]]
- [[kaggle_runbook]]
- [[pages/analysis/convergence_signal_priorities]]