---
title: "Task: Implement Self-Improving Architecture (Tier 1+2)"
tags:
  - doc/task
  - status/done
  - phase/25
  - topic/learning-agent
  - topic/self-improving
  - layer/world-model
  - layer/fusion
  - layer/learning
---

# Task: Implement Self-Improving Architecture (Tier 1+2)

Status: completed
Research: [[learned_vs_handcoded_audit]]
Spec: [[learned_vs_handcoded_architecture_spec]]

## Steps

### Change 1: Wire beliefs → SAC policy (Tier 1)
- [x] 1.1: Add `query_all_latest_beliefs()` to PipelineStore — returns latest BeliefState per variable_name
- [x] 1.2: In `_sac_inference()`, query beliefs from store, convert to BeliefState, pass to assembler
- [x] 1.3: Pack global belief means (latent.stress_level, etc.) into market_features so SAC gets world model state immediately

### Change 4: Adaptive surprise fusion weights (Tier 2)
- [x] 4.1: Add `AdaptiveSurpriseWeights` class with EG-on-simplex update rule
- [x] 4.2: Integrate into `SurpriseExtractor.__init__` as optional adaptive mode

### Change 9: GNN loss auto-tuning (Tier 2)
- [x] 9.1: Add 4 learnable `nn.Parameter` log-variances to `Trainer` for uncertainty-weighted multi-task loss (Kendall et al. 2018)
- [x] 9.2: Replace fixed `cfg.*_weight * loss` with `exp(-log_var) * loss + log_var`

### Change 2a: CPD learning via MLE (Tier 2)
- [x] 2a.1: Add `fit_cpds()` method to WorldModel using pgmpy BayesianEstimator with BDeu priors
- [x] 2a.2: Wire into world_model_update DAG as periodic re-fit step

### Change 2b: Kalman EM parameter fitting (Tier 2)
- [x] 2b.1: Add `fit_filter_params()` to ContinuousStateFilter implementing Shumway-Stoffer EM
- [x] 2b.2: Wire into world_model_update DAG after CPD fitting

### Testing
- [x] T.1: Edge case tests for all 5 changes (37/37 passing)
- [x] T.2: Fixed fit_cpds() bug — skip nodes with no observed data to prevent malformed CPDs
- [x] T.3: DAG integration tests for fitting wiring (36/36 passing)

## Related

- [[learned_vs_handcoded_audit]]
- [[learned_vs_handcoded_architecture_spec]]
