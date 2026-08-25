---
title: "Checkpoint: 2026-05-05"
tags:
  - doc/checkpoint
---

# Checkpoint: 2026-05-05

**Generated:** 2026-05-05T16:30
**Summary:** Research synthesis: machine intelligence architecture vs SOTA 2024-2026. Key findings: GNNs beat LLMs on geopolitical forecasting (HTKGH paper, DARPA); TDC-AE (2502.19307) validates ghost pattern via embedding trajectory anomaly; cross-domain GAD validates multi-domain approach; PIGDreamer validates POMDP framing; EWC confirmed correct for continual KGE. Priority: (1) fix GDELT dominance, (2) build embedding_snapshots table, (3) build TDC-AE ghost pattern detector, (4) motif library. H-D training running on Kaggle (epochs 19-40).

> **Single-owner rule:** Do NOT copy raw metric values (test counts, node counts,
> ENRICHMENT_DIM, DAG size, failure counts) into this checkpoint.
> Reference the canonical owners instead:
> - Current metrics → [[tirramind_structure]] (`memories/repo/tirramind_structure.md`)
> - Roadmap / next phases → [[quant_training_ground]] (`[[quant_training_ground]]`)
> This checkpoint is an **append-only historical record**. Never edit it after the session ends.

---

## Active Tasks

- [[autonomic_workflow_system]]
- [[database_architecture_strategy]]
- [[phase26_mcp_agent_upgrade]]
- [[phase40_real_data_model_refresh]]
- [[phase41b_gnn_signal_extraction]]
- [[phase42_ghost_pattern_activation]]
- [[quant_training_ground]]

## Recent Commits

```
6edbfb6 fix: raise FileNotFoundError if resume checkpoint missing (no silent fallback to epoch 0)
2371f36 fix: relax find_data_root to only require pipeline.db not checkpoints/
34b5950 fix: simplify epoch_018.pt discovery to rglob /kaggle/input
3fb86f0 feat: resume H-D training from epoch 18 on Kaggle
b33ced7 feat: add backtest cell to H-A, H-D, H-H notebooks (auto-runs after training)
```

## Files Changed (last commit)

```
agent/models/gnn/trainer.py
```

## Canonical State References

- Current metrics: see [[tirramind_structure]]
- Roadmap / next phases: see [[quant_training_ground]]
- Architecture decisions: see `docs/adr/`

## Related

- [[autonomic_workflow_system]]
- [[database_architecture_strategy]]
- [[phase26_mcp_agent_upgrade]]
- [[phase40_real_data_model_refresh]]
- [[phase41b_gnn_signal_extraction]]
- [[phase42_ghost_pattern_activation]]
- [[quant_training_ground]]
