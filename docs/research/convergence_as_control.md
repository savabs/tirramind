---
title: "Research: Convergence Detection as Control Signal (Phase 49b)"
tags:
  - doc/research
  - phase/49b
  - topic/convergence
  - layer/fusion
  - layer/world-model
  - layer/learning
  - status/active
---

# Research: Convergence Detection as Control Signal (Phase 49b)

## Problem Statement

The `convergence_detection` DAG currently outputs:
- `convergence_score` — scalar, written to `signals` table
- `regime_label` — HMM state label, written to `convergence_clusters` table
- BOCPD changepoint posterior — written to `signals` table

These are consumed downstream as features only. Nothing reads them to modulate system behaviour.

**The gap:** Convergence detection is a regime sensor. A regime sensor that doesn't change how
the system behaves in different regimes is a passive feature generator, not a control signal.
It is leaving value on the table.

**What it should do:** Gate and modulate the rest of the system dynamically. When a regime shift
is detected, the system should behave differently — not just emit a number.

## Current Architecture

- `agent/convergence/` — 10 modules (BOCPD, HMM, Fisher combined test, BH FDR, 12 causal templates)
- `agent/pipeline/dags/convergence_detection.py` — DAG runs at 18:30, writes to `signals` + `convergence_clusters`
- Consumers (current): `feature_generation.py` reads convergence signals as input features only
- Consumers (missing): `gnn_inference.py`, `rl_training.py`, `world_model_update.py` — none read regime_label

## Observations

1. BOCPD changepoint detection is already real-time and accurate. It produces a posterior over "did the regime change?" — this is exactly the kind of signal that should modulate downstream behaviour.
2. HMM regime_label is a discrete state (e.g. "bull," "bear," "crisis"). Different regimes warrant different: RL exploration rates, world model priors, trust in GNN embeddings (fresh vs stale regime), and GNN retrain urgency.
3. The `signals` table stores this information — nothing is missing architecturally. The gap is wiring.

## What "Control Signal" Means Concretely

### Control target 1: SAC exploration rate
In high-uncertainty regimes (BOCPD changepoint probability > threshold), raise SAC entropy coefficient α. In stable regimes, lower it toward exploitation. This is a wiring change to `rl_training.py` — read `regime_label` from `signals` table before SAC update, adjust `target_entropy`.

### Control target 2: World model priors
When regime_label changes, reset or soften the pgmpy Bayesian DAG priors (reduce confidence in beliefs accumulated under the previous regime). Currently the world model accumulates beliefs indefinitely regardless of regime. A regime shift should partially decay stale beliefs.

### Control target 3: GNN retrain trigger
When BOCPD detects a changepoint with posterior > threshold (e.g. 0.9), flag for a full GNN retrain on the next DAG run rather than the weekly scheduled retrain. This is the EWC interaction: EWC protects against forgetting; a deliberate retrain trigger handles genuine regime shift where old embeddings are wrong rather than just stale.

### Control target 4: Embedding trust / modulation
When regime_label changes, reduce the confidence weight of GNN embeddings in the feature matrix. Concretely: scale ENRICHMENT_DIM features by a trust factor (0.5–1.0) proportional to how long the current regime has been stable. Fresh regime = lower trust = downstream is more prior-weighted.

## Risks

- Over-reacting to false BOCPD positives: add minimum observation count before triggering retrain or prior decay
- SAC entropy oscillation if regime flips frequently: apply exponential moving average to entropy coefficient
- World model prior reset is irreversible in current pgmpy architecture: need a snapshot/restore mechanism

## Data Requirements

- `signals` table: `regime_label`, `changepoint_posterior`, `convergence_score` per run (already written)
- `convergence_clusters` table: HMM state sequence (already written)
- New: a "regime confidence" or "stability duration" field, computed from the signal sequence

## Priority Ranking (from architecture review 2026-04-23)

Rank 2 of 5 gaps. This is a wiring change, not a new model. Cheapest high-value upgrade
available. Can be implemented during the Phase 47 backfill window with minimal risk.
Unlike Phase 49 (downstream alignment), this does NOT need Phase 40 to be meaningful —
convergence detection runs on live data and produces real regime labels today.

## Implementation Sequence (when ready)

1. Add `read_latest_regime(store) -> tuple[str, float]` helper — reads most recent (regime_label, changepoint_posterior) from signals table
2. Wire into `rl_training.py`: adjust `target_entropy` based on changepoint_posterior
3. Wire into `world_model_update.py`: apply belief decay when regime_label differs from previous run
4. Wire into `gnn_inference.py`: set `force_retrain=True` flag when changepoint_posterior > 0.9
5. Wire into `feature_generation.py`: apply stability trust factor to ENRICHMENT_DIM features

Each step is independently testable. Do not bundle them.

## Files to Affect (when implemented)

- `agent/pipeline/dags/rl_training.py` — entropy modulation
- `agent/pipeline/dags/world_model_update.py` — belief decay on regime shift
- `agent/pipeline/dags/gnn_inference.py` — changepoint-triggered retrain flag
- `agent/pipeline/dags/feature_generation.py` — stability trust factor
- New: `agent/pipeline/regime_gate.py` — `read_latest_regime(store)` helper, shared by all DAGs above

## Related

- [[convergence_detection]] — Phase 7c research (BOCPD + HMM implementation)
- [[living_system_online_gnn]] — Phase 46 (EWC — the mitigation for routine drift; this phase handles deliberate regime shifts)
- [[convergence_as_control_spec]] — spec (not yet created)
- [[quant_training_ground]] — task file, Phase 49b entry
