---
title: "Training Stagnation Root Cause Analysis — TirraMind Phase 50"
tags:
  - doc/research
  - phase/50
  - topic/training
  - topic/gnn
  - topic/loss
  - topic/embedding-collapse
  - topic/backfill
  - layer/learning
  - status/active
---

> Primary document: [[training_stagnation_root_cause.html]]

## Summary

Six hypotheses explaining why return loss freezes and IC is negative on every post-V52 run (V55–V64), regardless of config. Root causes span data (backfill distribution shift), architecture (hard clamp before ListNet, dead PCGrad, concat-head bypass), and training dynamics (CSRC collapse, TGN memory staleness, MTL gradient dominance).

## Key Findings

- **H2 (clamp saturation)** is mechanistically confirmed by V64 [GRAD_FLOW] — pred_std → 0.00 by ep5, all predictions → +5.0. Gradient of clamp at boundary is zero → ListNet receives no ranking gradient.
- **H1 (backfill label shift)** has highest prior probability (~70%) — V63 ran exact V52 config post-backfill and produced IC −0.119 vs +0.0474. Label distribution change is the only explanation.
- **H3 (MTL dominance)** — obs_type sees ~2,145 entities vs return's ~53; PCGrad is implemented (trainer.py:168) but never called.
- **H4 (CSRC collapse)** — InfoNCE on noisy decile pairs in non-stationary financial returns drives dimensional collapse.

## Recommended Experiment Order

1. **V65-A** (immediate): remove clamp(-5,5) + add BatchNorm + log in-sample IC per epoch
2. **V65-B** (if collapse > 20% persists): disable CSRC, enable VICReg=0.1
3. **V65-C** (if gradient conflict confirmed): wire PCGrad into backward loop

## Related

- [[checkpoint_2026-06-07_v72_architecture_handoff]]
- [[chat_checkpoint_2026-06-06_v63_v64_training]]
- [[phase41b_gnn_signal_extraction]]
- [[quant_training_ground]]
- [[VERSIONS]]
- [[training_efficiency_v61_solution]]
