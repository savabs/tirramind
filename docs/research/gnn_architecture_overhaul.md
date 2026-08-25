---
title: "Research: GNN Architecture Overhaul — Post-V72 Pivot"
tags:
  - doc/research
  - phase/50
  - topic/training
  - topic/gnn
  - topic/architecture
  - layer/learning
  - status/active
---

> Primary document: [[gnn_architecture_overhaul.html]]

## Summary

After V72 confirmatory FAIL (IC +0.059, t=1.03), stop tuning HetTGN+ListNet+concat MTL. Phase A: honest baselines to test label learnability. Phase B: decoupled two-stage if signal exists.

## Related

- [[gnn_architecture_overhaul_spec]]
- [[gnn_architecture_overhaul_task]]
- [[training_stagnation_root_cause]]
- [[checkpoint_2026-06-07_v72_architecture_handoff]]
- [[phase41b_gnn_signal_extraction]]
- [[quant_training_ground]]
