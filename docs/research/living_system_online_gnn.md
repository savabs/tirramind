---
title: "Research: Phase 46 — Living System: Online GNN with EWC"
tags:
  - doc/research
  - phase/46
  - topic/living-system
  - layer/world-model
---

# Research: Phase 46 — Living System: Online GNN with EWC Continuous Learning

## Problem Statement

The HetTGN GNN weights are currently **frozen between full retrains**. Full retrain runs once daily
(or on-demand). Between runs, the model cannot adapt to new entity observations, regime shifts, or
emerging patterns. This makes the GNN a static snapshot rather than a living system.

Everything else is already live:
- HeteroMemory GRU: entity hidden state updates on every new event
- Bayesian world model: belief propagation updates per DAG run
- Thompson Sampling bandit: alpha/beta params update per run
- Reviewed Memory: candidate promotion after each learning entry

The missing piece: **GNN weight updates between full retrains**.

## Goal

After each batch of ≥N new observations (collected since last update), run 1 gradient step on
those new events only — while penalising large changes to weights that mattered for prior events
(Elastic Weight Consolidation). This keeps the model plastic (adapts) without catastrophic forgetting
(old patterns remain).

---

## Algorithm: Elastic Weight Consolidation (EWC)

### Source

Kirkpatrick et al. 2017, "Overcoming Catastrophic Forgetting in Neural Networks"  
arXiv:1612.00796 — the canonical EWC paper. Used in production continual learning systems.

### Loss formulation

$$\mathcal{L}_\text{total} = \mathcal{L}_\text{new} + \lambda \sum_i F_i (\theta_i - \theta_i^*)^2$$

Where:
- $\mathcal{L}_\text{new}$ = standard training loss on the new event batch
- $\theta_i^*$ = parameter values after the last full retrain ("anchor")
- $F_i$ = Fisher Information diagonal for parameter $i$ (measures importance)
- $\lambda$ = EWC strength hyperparameter (higher = more conservative)

The Fisher diagonal is computed once after each full retrain:

$$F_i = \mathbb{E}\left[\left(\frac{\partial \log p(y|x)}{\partial \theta_i}\right)^2\right]$$

Approximated empirically: run the training batch, sum squared gradients per parameter.

### Why EWC over alternatives

| Method | Pros | Cons | Decision |
|--------|------|------|----------|
| EWC (Kirkpatrick 2017) | Principled, well-studied, $O(n_\theta)$ storage, CPU-only | One Fisher matrix per task | **CHOSEN** |
| Online EWC (Schwarz 2018) | Accumulates Fisher across tasks | More complex, higher memory | Defer to Phase 49 if needed |
| ER Replay (Rolnick 2019) | Simple, no Fisher | Requires replay buffer | Already partially available via EpisodicMemory |
| PackNet (Mallya 2018) | Zero forgetting | Requires architecture changes | Incompatible with current HetTGN |
| Progressive Nets (Rusu 2016) | Zero forgetting | 4x param growth | Incompatible |

EWC is the correct first choice: it adds ~1.7 MB storage (weights + Fisher diagonal for 220K params),
runs on CPU, adds <1 second per online update, requires no architecture changes.

---

## Current Architecture — Integration Points

### Trainer (`agent/models/gnn/trainer.py`)

- `TrainerConfig` — frozen dataclass. Need to add `ewc_lambda: float` and `online_batch_threshold: int`
- `Trainer.save_model()` — saves `model_state_dict` + config to checkpoint dict via `torch.save`.
  Need to also save `fisher_diagonal` and `anchor_state_dict` into the same checkpoint dict.
- `Trainer.load_model()` — reconstructs model from checkpoint. Need to load Fisher + anchor if present.
- `Trainer.train()` — full retrain loop. After completion, compute Fisher diagonal and store anchor.
- New method: `Trainer.online_update(new_events)` — 1 gradient step on new_events + EWC penalty.

### GNN Inference DAG (`agent/pipeline/dags/gnn_inference.py`)

- `run_gnn_inference()` — current entry point. Calls `Trainer.train()` on schedule.
- Need to add: after training, query "observations since last_online_update timestamp" and run
  `online_update()` if count ≥ threshold. On subsequent DAG runs (between full retrains), check
  new obs count and run online_update if threshold crossed.

### Checkpoint file (`gnn_model.pt`)

- Currently stores: `model_state_dict`, `config`, `metadata_node_types`, `metadata_edge_types`,
  `in_channels`, `num_nodes`
- Will add (backward compatible, presence-checked on load):
  - `ewc_fisher`: dict[str, Tensor] — Fisher diagonal per named parameter
  - `ewc_anchor`: dict[str, Tensor] — model weights at last full retrain
  - `ewc_last_online_update_ts`: float — Unix timestamp of last online update
  - `ewc_obs_count_at_last_update`: int — obs count when last update ran

---

## Storage Overhead

HetTGN has ~220K parameters. Fisher diagonal = 1 tensor per parameter = same shape as weights.

| Item | Size |
|------|------|
| Current weights (model_state_dict) | ~860 KB |
| Fisher diagonal (ewc_fisher) | ~860 KB |
| Anchor weights (ewc_anchor) | ~860 KB |
| **Total overhead** | **~1.7 MB extra** |

Total checkpoint: ~2.6 MB. Negligible.

---

## Compute Overhead

Fisher computation: one forward + backward pass over the training batch after full retrain.
Same cost as one training epoch. Runs once after each full retrain.

Online update: one forward + backward pass over new events + EWC penalty computation.
With 100–500 new obs: <1 second on CPU. Runs at most once per DAG run.

---

## Risks

1. **Lambda tuning** — too low → forgetting; too high → no adaptation. Start with λ=1000.
   Observable: monitor training loss on new events. If loss stops decreasing, λ is too high.
2. **Small new-event batches** — gradient from 10 obs is noisy. Threshold must be ≥50 to be useful.
   Default: 100 new obs per online update.
3. **Checkpoint backward compatibility** — `load_model` must handle checkpoints without Fisher fields
   (old format). Presence-check all new fields before loading.
4. **Fisher computed on stale data** — if the graph schema changes (new entity types added), the
   Fisher diagonal shape may not match the new model. Handle: if shape mismatch, skip EWC penalty
   and recompute Fisher after next full retrain.

---

## Files to Create / Modify

| File | Change |
|------|--------|
| `agent/models/gnn/ewc.py` | NEW — `EWCState` dataclass + `compute_fisher()` + `ewc_penalty()` |
| `agent/models/gnn/trainer.py` | Modify `TrainerConfig`, `save_model`, `load_model`, `train`, add `online_update` |
| `agent/pipeline/dags/gnn_inference.py` | Modify `run_gnn_inference` to call `online_update` after training and on subsequent runs |
| `tests/test_ewc.py` | NEW — unit tests for EWC class and integration |

---

## References

1. Kirkpatrick et al. 2017 — "Overcoming Catastrophic Forgetting in Neural Networks" arXiv:1612.00796
2. Schwarz et al. 2018 — "Progress & Compress: A scalable framework for continual learning" (Online EWC) arXiv:1805.06370
3. TirraMind `agent/models/gnn/trainer.py` — existing Trainer/TrainerConfig architecture
4. TirraMind `agent/pipeline/dags/gnn_inference.py` — existing DAG entry point

## Related

- [[living_system_online_gnn_spec]]
- [[quant_training_ground]]
- [[chat_checkpoint_2026-04-22_final_session]]
