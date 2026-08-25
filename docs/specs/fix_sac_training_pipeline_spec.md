---
title: "Spec: Fix SAC Training Pipeline & Wire Transition Loop"
tags:
  - doc/spec
  - phase/25
  - topic/self-improving
  - topic/training-loop
  - layer/learning
---

# Spec: Fix SAC Training Pipeline & Wire Transition Loop

**Date:** 2026-04-14
**Research:** [[tier5_gradient_bypass]]
**Goal:** Fix the 5 latent bugs in `_train_sac()` that prevent SAC from ever training, then implement the missing transition writer that closes the online learning loop.

---

## Goal

Make the SAC training pipeline functional end-to-end: inference stores transitions → rl_training reads them → SAC learns from real experience → checkpoint updates.

## Files Affected

| File | Action |
|------|--------|
| `agent/pipeline/dags/rl_training.py` | Fix 5 bugs in `_train_sac()` |
| `agent/pipeline/dags/inference.py` | Add transition writer to `_sac_inference()` and complete in `_emit_portfolio()` |
| `agent/pipeline/store.py` | Add `store_pending_transition()`, `complete_pending_transition()`, `query_pending_transition()` |
| `tests/test_rl_training_dag.py` | Rewrite: transition deserialization, checkpoint round-trip, assembler alignment |
| `tests/test_training_pipeline_e2e.py` | New: full loop integration test |

## Implementation Steps

### Step A.1: Fix transition key mismatch in `_train_sac()` (rl_training.py:224-228)

`query_rl_transitions()` renames keys during JSON parse:
- `state_json` → `state`, `action_json` → `action`, `next_state_json` → `next_state`

But `_train_sac()` reads the old names. Fix:
```python
# Before (broken):
state = np.array(json.loads(t["state_json"]), ...)
# After (correct — already parsed by query method):
state = np.array(t["state"], ...)
```

Note: `query_rl_transitions()` already calls `json.loads()`, so the training code must NOT call it again. The values are already Python lists/dicts.

### Step A.2: Fix checkpoint load key mismatch (rl_training.py:208)

`_parse_checkpoint_row()` renames `state_dict_blob` → `state_dict_bytes`. But `_train_sac()` reads `checkpoint["state_dict_blob"]`.

Fix: `checkpoint["state_dict_blob"]` → `checkpoint["state_dict_bytes"]`

### Step A.3: Fix checkpoint save kwargs (rl_training.py:254-265)

The `store_rl_checkpoint()` signature is:
```python
def store_rl_checkpoint(self, policy_type, config, state_dict_bytes, metrics=None, is_best=False)
```

But `_train_sac()` passes wrong kwarg names AND pre-serializes:
```python
store.store_rl_checkpoint(
    policy_type="sac",
    config_json=json.dumps({...}),      # wrong name, wrong type
    state_dict_blob=state_blob,           # wrong name
    metrics_json=json.dumps({...}),       # wrong name, wrong type
    is_best=False,
)
```

Fix: pass dicts (not pre-serialized strings) with correct kwarg names:
```python
store.store_rl_checkpoint(
    policy_type="sac",
    config={...},
    state_dict_bytes=state_blob,
    metrics={...},
    is_best=False,
)
```

### Step A.4: Align assembler to InstrumentStateAssembler (rl_training.py:188)

Training uses `StateAssembler()` (generic) but inference uses `InstrumentStateAssembler(instrument_tickers=tickers)`. The state dimensions differ. Fix: use the same assembler in both paths.

This requires importing `tradeable_instruments` and building the same ticker list that `_sac_inference()` uses.

### Step A.5: Implement transition writer (inference.py)

Two-phase approach matching the natural pipeline flow:

**Phase 1 — `_sac_inference()`: Store pending transition**
After SAC produces `(state, action)`, store them as a pending transition with today's date. The reward is unknown yet (it requires tomorrow's P&L).

**Phase 2 — `_emit_portfolio()`: Complete previous transition**
After computing yesterday's P&L (steps 3-4), look up yesterday's pending transition, set reward = `portfolio_return`, assemble today's state as `next_state`, and store the completed transition.

Store API additions:
- `store_pending_transition(date, state_list, action_list, metadata)` → row ID
- `query_pending_transition(date)` → row or None
- `complete_pending_transition(date, reward, next_state_list, done)` → updates the row in `rl_transitions`

### Step A.6: Edge-case test suite

Cover:
- Transition deserialization with already-parsed keys
- Checkpoint save/load round-trip with correct kwargs
- Assembler dimension consistency between training and inference
- Pending transition create → complete → query in rl_training
- Transition writer stores correct state/action format
- Replay buffer loads from completed transitions
- Missing pending transition (first day) handled gracefully
- Double-completion prevented (idempotent)
- P&L of zero (flat day) stores correct reward
- Empty weights → no pending transition stored

## Edge Cases

- First day: no yesterday pending → skip transition completion, just store today's pending
- Inference skipped (no SAC model) → no pending transition stored
- P&L unavailable (no returns data) → don't complete the pending transition (wait for next day with data)
- Transition with NaN/Inf in state → reject, log warning

## Testing Plan

1. Unit tests for each bug fix (key names, checkpoint kwargs, assembler dims)
2. Unit test for pending transition store/query/complete cycle
3. Integration test: mock full pipeline day 1 → day 2 → verify transition in DB
4. Regression: existing rl_training tests still pass

---

## Related

- [[tier5_gradient_bypass]]
- [[learned_vs_handcoded_architecture_spec]]
- [[tier5_differentiable_kalman]]
