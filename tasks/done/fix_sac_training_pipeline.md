---
title: "Task: Fix SAC Training Pipeline & Wire Transition Loop"
tags:
  - doc/task
  - status/done
  - phase/25
  - topic/self-improving
  - topic/training-loop
  - layer/learning
---

# Task: Fix SAC Training Pipeline & Wire Transition Loop

Status: completed
Research: [[tier5_gradient_bypass]]
Spec: [[fix_sac_training_pipeline_spec]]

## Steps

### A.1: Fix transition key mismatch
- [x] A.1.1: Change `t["state_json"]` → `t["state"]` in `_train_sac()` (rl_training.py)
- [x] A.1.2: Change `t["action_json"]` → `t["action"]` in `_train_sac()`
- [x] A.1.3: Change `t["next_state_json"]` → `t["next_state"]` in `_train_sac()`
- [x] A.1.4: Remove `json.loads()` wrapping (already parsed by query method)

### A.2: Fix checkpoint load key
- [x] A.2.1: Change `checkpoint["state_dict_blob"]` → `checkpoint["state_dict_bytes"]`

### A.3: Fix checkpoint save kwargs
- [x] A.3.1: Change `config_json=json.dumps({...})` → `config={...}` (dict, not string)
- [x] A.3.2: Change `state_dict_blob=` → `state_dict_bytes=`
- [x] A.3.3: Change `metrics_json=json.dumps({...})` → `metrics={...}` (dict, not string)

### A.4: Align assembler
- [x] A.4.1: Replace `StateAssembler()` import/usage with `InstrumentStateAssembler`
- [x] A.4.2: Import `tradeable_instruments` and build ticker list
- [x] A.4.3: Verify `state_dim` matches inference path

### A.5: Wire transition loop
- [x] A.5.1: Add pending transition table + store methods (store.py)
- [x] A.5.2: In `_sac_inference()`: store pending transition after action selection
- [x] A.5.3: In `_emit_portfolio()`: complete yesterday's pending transition with reward

### A.6: Edge-case test suite
- [x] A.6.1: Transition key deserialization tests (9 tests)
- [x] A.6.2: Checkpoint save/load round-trip test (8 tests)
- [x] A.6.3: Assembler dimension consistency test (3 tests)
- [x] A.6.4: Pending transition lifecycle test (16 tests)
- [x] A.6.5: Full two-day integration test (3 tests)
- [x] A.6.6: Edge cases (9 tests)

## Related

- [[tier5_gradient_bypass]]
- [[fix_sac_training_pipeline_spec]]
- [[learned_vs_handcoded_architecture_spec]]
- [[tier5_differentiable_kalman]]
