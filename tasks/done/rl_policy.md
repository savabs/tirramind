---
title: "Task: RL Policy — Surprise-Driven Portfolio Allocation"
tags:
  - doc/task
  - status/done
  - phase/21
  - topic/rl-policy
  - topic/portfolio-optimization
  - topic/surprise-weighting
  - layer/learning
---

# Task: RL Policy — Surprise-Driven Portfolio Allocation

Status: completed
Research: [[rl_policy]]
Spec: [[rl_policy_spec]]

---

## Phase 21a: Surprise Weight Learning (MVP)

- [x] 21a.1: Create policy module + config dataclasses (`agent/learning/policy/__init__.py`, `config.py`)
- [x] 21a.2: Implement symlog transforms (`symlog.py`, `tests/test_symlog.py`)
- [x] 21a.3: Implement entity-to-asset mapper (`asset_mapper.py`, `tests/test_asset_mapper.py`)
- [x] 21a.4: Implement reward function (`reward_fn.py`, `tests/test_reward_fn.py`)
- [x] 21a.5: Implement differentiable backtest / weight learner (`weight_learner.py`, `tests/test_weight_learner.py`)
- [x] 21a.6: Modify SurpriseExtractor to accept learned weights (`surprise.py`)
- [x] 21a.7: Add PipelineStore tables for RL (`store.py`)
- [x] 21a.8: Edge case test suite for Phase 21a (`tests/test_rl_edge_cases.py`)

## Phase 21b: SAC Actor-Critic

- [x] 21b.1: Implement state assembler (`state_assembler.py`, `tests/test_state_assembler.py`)
- [x] 21b.2: Implement replay buffer (`replay_buffer.py`, `tests/test_replay_buffer.py`)
- [x] 21b.3: Implement SAC networks — actor, twin critics, temperature (`sac.py`, `tests/test_sac.py`)
- [x] 21b.4: Implement SAC training loop (`sac.py` — SACTrainer.update(), `tests/test_sac.py`)
- [x] 21b.5: Implement portfolio strategy adapter (`portfolio_strategy.py`, `tests/test_portfolio_strategy.py`)
- [x] 21b.6: Historical data loading handled by RL training DAG (`dags/rl_training.py`)
- [x] 21b.7: Implement RL training DAG (`dags/rl_training.py`, `tests/test_rl_training_dag.py`)
- [x] 21b.8: Edge case test suite for Phase 21b (`tests/test_rl_edge_cases.py`)

## Integration & Validation

- [x] 21.9: Wire RL training DAG into DAG registry (`dags/__init__.py`)
- [x] 21.10: Walk-forward validation of full pipeline (`tests/test_rl_validation.py`)
- [x] 21.11: Update learning module exports + documentation (`agent/learning/policy/__init__.py`)

---

## Related

- [[rl_policy]] — Research doc
- [[rl_policy_spec]] — Spec doc
- [[signal_fusion|Signal Fusion task]] — Phase 20 (upstream)
- [[quant_training_ground]] — Master phase tracker
