---
title: "Research: GNN Downstream Alignment (Phase 49)"
tags:
  - doc/research
  - phase/49
  - topic/gnn-alignment
  - layer/world-model
  - layer/feature-engineering
  - status/active
---

# Research: GNN Downstream Alignment (Phase 49)

## Problem Statement

The HetTGN is trained purely self-supervised:
- Next observation type (cross-entropy)
- Time delta prediction (MSE)
- Value prediction (Huber)
- Contrastive loss (same-entity pairs)

None of these objectives are the downstream decision objective. The GNN optimises for
"predict the next observation," not "produce embeddings that improve world-model likelihood
or RL advantage estimates."

This is the self-supervision alignment gap: embeddings can be locally valid (predict next
obs accurately) while being globally useless (carry no signal for the POMDP's value function).

**The core principle (filed 2026-04-23):** The real join is the feature space that remains
stable across regime shifts while still being informative. EWC addresses parameter stability.
It does not address representational alignment. These are distinct problems.

## Current Architecture

- `agent/models/gnn/trainer.py` — `Trainer.train()` runs self-supervised loss only
- `agent/models/gnn/gnn_inference.py` — DAG operator: trains, saves, does EWC online update
- Downstream consumers of GNN embeddings:
  - `agent/pipeline/dags/feature_generation.py` — reads entity embeddings from PipelineStore, assembles an `ENRICHMENT_DIM`-wide feature matrix (value was 41 at Phase 49; current value is derived — see `[[project_metrics]]`)
  - `agent/pipeline/dags/world_model_update.py` — injects features into pgmpy Bayesian DAG as evidence
  - `agent/pipeline/dags/rl_training.py` — SAC consumes belief states from world model (not raw embeddings)

## Observations

1. The world model and RL layer are separated from the GNN by two hops (GNN → features → world model → RL). Gradient from RL back to GNN is architecturally absent.
2. The world model does produce a measurable quantity: **belief log-likelihood** improvement after evidence injection. This can serve as a weak alignment signal.
3. The RL layer produces **advantage estimates** per entity after each SAC update. These are also measurable.
4. Both signals are already computed — they are just not fed back to the GNN.

## Risks

- Full end-to-end gradient (RL → GNN) is expensive and unstable. Not recommended without extensive research.
- Auxiliary loss that is too strong may override the self-supervised structure that makes the GNN data-efficient.
- Signal must be computed at DAG time (after world_model_update and rl_training), not during GNN training — requires a delayed feedback loop across DAG runs.

## Candidate Approaches

### Option A: World-model likelihood as auxiliary loss (recommended first)
After each DAG run, compute delta in Bayesian DAG log-likelihood (belief before evidence vs after evidence injection using GNN features). Accumulate per entity. On next GNN training window, add a weak auxiliary loss term: entities with low likelihood improvement get higher gradient weight.
- Cost: one extra belief-state read per DAG run
- Benefit: direct signal from the part of the system that uses GNN embeddings

### Option B: RL advantage-weighted replay
After SAC update, identify entity states that contributed to high-advantage transitions. Upweight those entities in the GNN's next training window (importance sampling over the event replay buffer).
- Cost: requires cross-DAG state passing (rl_training → gnn_inference)
- Benefit: directly aligns embeddings with the RL objective

### Option C: Dual-memory / slow-fast weights
Keep a "slow" frozen copy of the GNN weights (last full retrain) and a "fast" copy (online updates only). Periodically align the fast copy toward downstream performance, not just toward the anchor. Complements EWC.
- Cost: 2× model storage, more complex training loop
- Research needed: prior art in dual-memory continual learning

## Data Requirements

- `beliefs` table in PipelineStore: before/after log-likelihood per belief node per DAG run
- `rl_transitions` table: advantage values per state (already stored)

## Math / Algorithm Survey

- Kirkpatrick et al. (2017) EWC — addresses stability, not alignment; already implemented
- Auxiliary task learning: Jaderberg et al. (2017) UNREAL — auxiliary losses improve representation quality; concept applicable here
- Experience replay with prioritised sampling: Schaul et al. (2016) PER — priority by TD error; analogous priority-by-likelihood-delta for GNN

## Priority Ranking (from architecture review 2026-04-23)

Rank 1 of 5 gaps identified. Implement after Phase 47 (backfill) and Phase 40 (real retrain),
since the alignment signal is only meaningful once GNN embeddings are trained on real data.
Do not implement before Phase 40 — there is nothing to align against.

## Files to Affect (when implemented)

- `agent/pipeline/dags/gnn_inference.py` — add auxiliary loss computation using `beliefs` delta
- `agent/models/gnn/trainer.py` — `train()` accepts optional `alignment_weights` per entity
- `agent/pipeline/store.py` — query for belief log-likelihood delta (already stored in `beliefs` table)
- New: `agent/models/gnn/alignment.py` — compute_alignment_weights(store) → dict[entity_id, float]

## Related

- [[living_system_online_gnn]] — Phase 46 research (EWC, stability)
- [[gnn_downstream_alignment_spec]] — spec (not yet created)
- [[quant_training_ground]] — task file, Phase 49 entry
- [[transformer_world_model]] — Phase 48 research (downstream alignment becomes less critical once transformer learns causal structure jointly)
