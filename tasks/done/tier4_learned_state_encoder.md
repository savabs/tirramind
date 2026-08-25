---
title: "Task: Implement Learned State Encoder (Change 6)"
tags:
  - doc/task
  - status/done
  - phase/25
  - topic/self-improving
  - topic/state-representation
  - layer/learning
---

# Task: Implement Learned State Encoder (Change 6)

Status: completed
Research: [[learned_state_encoder]]
Spec: [[learned_vs_handcoded_architecture_spec]]

## Steps

### 6.1: Implement LearnedStateEncoder nn.Module
- [x] 6.1.1: Create `agent/learning/policy/state_encoder.py` with `LearnedStateEncoder(nn.Module)`
- [x] 6.1.2: Entity embedding MLP: raw 9-dim (5 surprise + 4 belief) per entity → entity_embed_dim
- [x] 6.1.3: Reshape 463-dim flat state into entity block (E×9) + global block (M+1+4)
- [x] 6.1.4: Learnable [CLS] token + MultiheadAttention over entity tokens
- [x] 6.1.5: Output: [z_CLS ; global_features] = compact state
- [x] 6.1.6: `forward(state_flat)` — takes assembler output, returns compact state
- [x] 6.1.7: Add `StateEncoderConfig` dataclass in config.py

### 6.2: Integrate encoder into SACTrainer
- [x] 6.2.1: Add optional `encoder` param to SACTrainer.__init__
- [x] 6.2.2: Include encoder params in actor optimizer
- [x] 6.2.3: In update(): encode states, re-encode for actor (gradient flow)
- [x] 6.2.4: In select_action(): run state through encoder
- [x] 6.2.5: In save()/load(): include encoder state dict + config, backward compat

### 6.3: Wire into DAGs
- [x] 6.3.1: In rl_training.py: create encoder from PolicyConfig.state_encoder
- [x] 6.3.2: In inference.py: SACTrainer.load auto-reconstructs encoder from checkpoint
- [x] 6.3.3: Cold-start fallback: no encoder in checkpoint → skip encoding (has_encoder flag)

### 6.4: Edge-case test suite (36/36 passing)
- [x] 6.4.1: Encoder forward shape tests (6 tests: single/batch/custom/multi-layer)
- [x] 6.4.2: Gradient flow (3 tests: direct/MLP/CLS token)
- [x] 6.4.3: Padding mask (4 tests: zero-pad/all-zero/single/max entities)
- [x] 6.4.4: Save/load round-trip (4 tests: with/without encoder, config preserved)
- [x] 6.4.5: Backward compat (2 tests: old format loads, old format selects action)
- [x] 6.4.6: SACTrainer with/without encoder (6 tests: select_action/update/grads/optim)
- [x] 6.4.7: Determinism (2 tests: encoder/trainer)
- [x] 6.4.8: Edge cases (6 tests: batch=1/64, same entities, large/negative values, minimal config)
- [x] 6.4.9: Integration (3 tests: multi-update, weight changes, save-load after training)

## Related

- [[learned_state_encoder]]
- [[learned_vs_handcoded_architecture_spec]]
- [[learned_architecture_impl]]
- [[tier4_learn_dag_structure]]
