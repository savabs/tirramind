---
title: "Research: Phase 41 — Model Refresh Hardening"
tags:
  - doc/research
  - phase/41
  - topic/gnn
  - topic/backtest
  - topic/pipeline
  - layer/world-model
  - layer/learning
  - layer/surveillance
---

# Research: Phase 41 — Model Refresh Hardening

## Problem Statement

Phase 40 completed its checklist but left the system in a degraded state:

1. The live checkpoint on disk (`.tirra_pipeline/gnn_model.pt`) is the **10-epoch overtrained model** whose auto-tuned loss weights diverged and whose temporal head collapsed to predicting a constant 24h delta (see [[chat_checkpoint_2026-04-20_phase40_full]] §4.2).
2. The Kendall-style uncertainty-weighted multi-task loss in [agent/models/gnn/trainer.py](agent/models/gnn/trainer.py) has no regularizer on the log-variance parameters. When component losses approach 0, the log-variances drift to $-\infty$, effective weights explode, and the total loss becomes unboundedly negative while actual learning stalls.
3. Feature generation and the walk-forward backtest currently consume the broken model, so any downstream analysis is contaminated.
4. Observation diversity is pathological: 95.7% of observations are `instrument_daily`. Several L2 tools are built but not attached to any scheduled DAG, so their entities never receive observations and the GNN has no cross-domain signal to learn.

## Phase 40 Context (source of truth)

Full debrief: [[chat_checkpoint_2026-04-20_phase40_full]]. Key extracts:

- Run 1 (5 epochs, 48h window) was healthy: test top-1 ≈ 86.8%, time_delta MAE ≈ 1.7s, learned weights stable in `[0.1, 3.6]`.
- Run 2 (10 epochs, 24h window) overfit catastrophically: total loss → $-23.58$, effective weights up to 2006.68, time_delta MAE collapsed to 83,517s ≈ 24h (the model learned "always predict one day").
- The 5-epoch good model was overwritten. The 5-epoch checkpoint that still exists on disk is `gnn_model_live.pt` (from 2026-04-19 pipeline runs) — compatible architecture, real pipeline data.
- Checkpoint explicitly recommends clamping the log-variance parameters as the minimal-risk fix (Option A).

## Math of the Bug

The Kendall et al. 2018 uncertainty-weighted loss is:

$$
\mathcal{L}_\text{total} = \sum_k \frac{1}{2\sigma_k^2}\,\mathcal{L}_k + \ln \sigma_k
$$

With the implementation using $s_k = \ln \sigma_k^2$ as a free parameter:

$$
\mathcal{L}_\text{total} = \sum_k e^{-s_k}\,\mathcal{L}_k + s_k
$$

The gradient with respect to $s_k$ is $1 - e^{-s_k}\,\mathcal{L}_k$. When $\mathcal{L}_k \to 0$, the gradient is $+1$, i.e. the optimizer keeps decreasing $s_k$ without bound. The first term vanishes, the second term $s_k$ goes to $-\infty$, and $\mathcal{L}_\text{total} \to -\infty$. Effective weight $e^{-s_k}$ explodes, but because $\mathcal{L}_k \approx 0$, actual parameter updates on the shared backbone stall.

**Trusted sources:**
- Kendall, Gal, Cipolla 2018, *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics*, arXiv:1705.07115 — the original formulation.
- Liebel & Körner 2018, *Auxiliary Tasks in Multi-task Learning*, arXiv:1805.06334 — documents the collapse mode and proposes a $+1$ offset (equivalent to clamping away from $-\infty$).
- Practitioner consensus (e.g. `pytorch-forecasting`, `torchmultimodal`): clamp $s_k$ (or equivalently the effective weight) to a bounded range, usually $[-3, 3]$ for $s_k$ ⇒ effective weight $\in [e^{-3}, e^{3}] \approx [0.05, 20]$.

Clamping $s_k \in [s_\text{min}, s_\text{max}]$ is the standard, low-risk fix. It preserves the uncertainty-weighting behavior in the regime where component losses are non-trivial and only prevents runaway when one component is ~0.

## Options Considered for Fix

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| A. Clamp log-variance each step | Trivial, local, matches literature. Keeps gradient on `s_k` for mid-range values. | Introduces two hyperparameters. | **Chosen.** |
| B. Early-stop on auto-tune once all $\mathcal{L}_k < \epsilon$ | Addresses the root cause (nothing more to learn). | Requires per-task threshold. Partial: if only one task saturates, the rest are still useful. | Secondary. Could be added later. |
| C. Loss floor $\max(\mathcal{L}_k, \epsilon)$ | Keeps the uncertainty term grounded. | Distorts the loss shape near the minimum. | Reject. |
| D. Swap to Liebel–Körner formulation $\sum_k \tfrac{1}{2\sigma_k^2}\mathcal{L}_k + \ln(1+\sigma_k^2)$ | Cannot diverge. | Larger code change, different dynamics, need to reverify. | Reject for now. |

Chosen: **Option A with defaults $s_\text{min}=-3.0$, $s_\text{max}=3.0$**. Applied inside both the training step and `effective_loss_weights()` so serialized weights are also sane.

## Observation Diversity Gap

Checkpoint §8 lists built-but-not-scheduled Tier-1 tools. Of those, these are already L2-capable (persist entity observations when a `pipeline_store` is passed) and do not require paid API keys:

| Tool | L2 since | Entity types written | Auth |
|---|---|---|---|
| [agent/tools/whale_alert.py](agent/tools/whale_alert.py) | Phase 10b | `wallet`, links to `instrument` (BTC-USD) | Free — `blockchain.info` |
| [agent/tools/sanctions_monitor.py](agent/tools/sanctions_monitor.py) | Phase 18 | `organization`, `country` links | Free — OFAC |
| [agent/tools/insider_filings.py](agent/tools/insider_filings.py) | Phase 10b | `person`, `company` | Free — SEC EDGAR |

`whale_alert` is the cheapest, fastest, zero-auth, and adds the `wallet` entity type that currently has zero nodes in the graph. Wiring it into `daily_collection` adds a genuinely new entity type and new cross-domain links (`wallet → instrument`) — high expected GNN-diagnostic impact.

## Files Affected

- [agent/models/gnn/trainer.py](agent/models/gnn/trainer.py) — add clamp fields + enforce in training step and `effective_loss_weights`.
- [tests/test_trainer.py](tests/test_trainer.py) — new tests for clamp behaviour.
- `.tirra_pipeline/gnn_model.pt` / `gnn_model_pre_phase40.pt` / `gnn_model_broken_10ep.pt` — model file swap (data files, not code).
- [agent/pipeline/dags/daily_collection.py](agent/pipeline/dags/daily_collection.py) — add `fetch_whale_alert` node.
- [tests/test_daily_collection_dag.py](tests/test_daily_collection_dag.py) (or equivalent) — assert the new node exists and validates.

## Risks

- Retraining is expensive (~16 min for 5 epochs, 48h window). Keep the good-model swap as a safety net so the pipeline is never held hostage by a training failure.
- `whale_alert` is stateless mempool data — there is no historical backfill. Observations will only accumulate going forward. This is fine: the goal is ongoing diversity, not a retroactive fix.
- Clamp bounds are hyperparameters. Defaults `[-3, 3]` are literature-standard but should be overridable per training run.

## Related

- [[phase41_model_refresh_hardening_spec]]
- [[phase41_model_refresh_hardening]]
- [[real_data_model_refresh]]
- [[chat_checkpoint_2026-04-20_phase40_full]]
- [[temporal_het_gnn]]
- [[whale_alert_l2]]
- [[starved_class_audit]]
