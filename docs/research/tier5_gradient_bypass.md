---
title: "Research: Tier 5 — Differentiable Belief Bypass & Training Pipeline Audit"
tags:
  - doc/research
  - phase/25
  - topic/self-improving
  - topic/differentiable
  - layer/fusion
  - layer/learning
---

# Research: Tier 5 — Differentiable Belief Bypass & Training Pipeline Audit

**Date:** 2025-07-21
**Spec:** [[learned_vs_handcoded_architecture_spec]]
**Task:** [[tier5_differentiable_kalman]] (Change 10c complete), Change 10 A/B next
**Audit:** [[learned_vs_handcoded_audit]]

---

## 1. Current State

Change 10c is complete: `DifferentiableKalmanFilter` is an `nn.Module` with autograd-preserving `predict()` and `update()`, Cholesky PSD parameterization, and integration into the world model DAG. 66 tests pass.

The original plan for the next step was **"Differentiable Belief Bypass"** — a training-time shortcut where Kalman state tensors flow directly to the `LearnedStateEncoder` with gradients intact, bypassing the SQLite belief store.

**This research reveals that a prerequisite infrastructure fix is needed first.**

---

## 2. Gradient Domain Analysis

A comprehensive trace of the full pipeline identified **two disconnected gradient domains**:

### Domain A: World Model (Kalman)

```
DifferentiableKalmanFilter
  predict(regime) → _x, _P tensors (autograd ✓)
  update(obs)     → _x, _P tensors (autograd ✓)
  get_beliefs()   → .detach() + float() → BeliefState (autograd ✗)
  → SQLite         → row dicts with floats (autograd ✗)
```

Filter params (F, Q, H, R) are `nn.Parameter` and can be optimized via EM or gradient descent, but **downstream reward signal never reaches them** because `get_beliefs()` calls `.detach()`.

### Domain B: Policy (SAC)

```
ReplayBuffer → numpy states
  → torch.from_numpy() (new graph)
  → LearnedStateEncoder (attention over entities)
  → GaussianActor → log_prob → actor_loss (autograd ✓)
  → TwinCritic → Q values → critic_loss (autograd ✓)
```

Clean gradient chain from encoder through to policy losses. Encoder params learn from actor loss. This works correctly today.

### Gradient Breaks (ranked by impact)

| # | Break | Location | Impact | Fix LOC |
|---|-------|----------|--------|---------|
| 1 | `get_beliefs()` detach + float() | `diff_kalman.py:280` | Disconnects Kalman from all downstream losses | ~50 |
| 2 | SQLite belief store round-trip | `store.py` + `world_model_update.py` | Serializes to floats, destroys computation graph | ~200 (bypass) |
| 3 | numpy state assembly | `state_assembler.py:330` | `np.zeros()` + `np.concatenate()` + `torch.from_numpy()` | ~150 |
| 4 | JSON transition serialization | `rl_training.py:224` | Inherent to off-policy RL — **DON'T FIX** |
| 5 | ReplayBuffer numpy storage | `replay_buffer.py` | Inherent to off-policy RL — **DON'T FIX** |
| 6 | Critic `.detach()` on target | `sac.py` | Correct by design (standard SAC) — **DON'T FIX** |

Breaks 4-6 are architecturally correct. Off-policy RL requires storing transitions and replaying them; that inherently breaks the computation graph from collection time. The SAC target network detach is standard (Haarnoja et al., 2018).

---

## 3. CRITICAL FINDING: Training Pipeline Gap

**The SAC training pipeline has never produced or consumed real transitions.**

### 3a. No Transition Writer

`store_rl_transition()` exists in `store.py:1675` but is **never called anywhere in the codebase**. The `rl_transitions` table is always empty. This means:
- `_train_sac()` in `rl_training.py` loads zero transitions
- The `len(buffer) < cfg.batch_size` check at line 237 always fires
- SAC training returns `{"status": "insufficient_data", "buffer_size": 0}`
- **SAC has never actually trained on real policy rollouts**

### 3b. Key Name Mismatch Bug

Even if transitions existed, `_train_sac()` would crash:
```python
# rl_training.py:224 reads:
state = np.array(json.loads(t["state_json"]), dtype=np.float32)

# But query_rl_transitions() at store.py:1727 renames keys:
d["state"] = json.loads(d.pop("state_json", "{}"))
```
The training code reads `t["state_json"]` but the query returns `t["state"]` → `KeyError`.

### 3c. Assembler Dimension Mismatch

`rl_training.py:200` uses `StateAssembler()` (generic) to compute `state_dim`. But `inference.py:266` uses `InstrumentStateAssembler(instrument_tickers=tickers)` which has a **different state layout** (includes instrument surprise block). If a checkpoint were saved from inference using `InstrumentStateAssembler` and loaded during training with `StateAssembler.state_dim`, dimensions would not match.

### 3d. Missing Transition Loop

The intended flow should be:
1. **Day T inference**: assemble state `s_t`, SAC selects action `a_t`, store `s_t` and `a_t`
2. **Day T+1**: compute reward `r_t = realized P&L(a_t)`, assemble new state `s_{t+1}`
3. Store `(s_t, a_t, r_t, s_{t+1}, done=False)` as a transition
4. Periodically: `rl_training.py` reads accumulated transitions and trains SAC

Nobody implements steps 1-3. The `emit_portfolio` node stores weights and P&L, but never creates the RL transition linking yesterday's state-action to today's reward and state.

---

## 4. Impact on the Differentiable Bypass

The "Differentiable Belief Bypass" (connecting Domain A to Domain B during training) was the planned next step. But building it on top of a training pipeline that has no data flowing through it creates three problems:

1. **Can't test end-to-end gradient flow** without actual transitions to train on
2. **Can't measure improvement** from the bypass (no baseline SAC training works)
3. **Architecture decisions** for the bypass depend on understanding the actual transition format — but no transitions exist yet

**Recommendation: Fix the training pipeline first, THEN implement the differentiable bypass.**

This isn't a detour — it's a prerequisite. The bypass is meaningless if SAC never trains.

---

## 5. Proposed Implementation Plan

### Phase A: Fix Training Pipeline Infrastructure (prerequisite)

**A.1: Fix key name mismatch in `_train_sac()`**
- Change `t["state_json"]` → `t["state"]` etc. in `rl_training.py:224-228`
- ~5 LOC

**A.2: Align assembler usage**
- `_train_sac()` should use `InstrumentStateAssembler` (same as inference)
- Pass the same `instrument_tickers` list
- ~20 LOC

**A.3: Implement transition writer in `emit_portfolio`**
- After SAC inference produces `(state, action)` and P&L is computed:
  - Store `state_t` and `action_t` in a "pending transition" row
  - On the next run, compute reward from realized P&L, assemble `state_{t+1}`, complete the transition
- ~100 LOC

**A.4: Validate the full training loop end-to-end**
- Integration test: inference stores transitions → rl_training reads and trains → checkpoint updates
- ~80 LOC tests

### Phase B: Differentiable Belief Bypass (after A works)

**B.1: Add `get_beliefs_differentiable()` to DifferentiableKalmanFilter**
- Returns `(belief_means: Tensor, belief_variances: Tensor)` **without** `.detach()`
- Alongside the existing `get_beliefs()` (still needed for SQLite persistence)
- ~30 LOC

**B.2: Create `DifferentiableStateAssembler`**
- Accepts torch Tensors for belief features instead of `BeliefState` objects
- Uses `torch.cat()` instead of `np.concatenate()`
- Produces state tensor with gradients intact
- ~100 LOC

**B.3: Training-time bypass path in `_train_sac()`**
- During training, if `DifferentiableKalmanFilter` is available:
  - Run Kalman predict/update in torch (preserving gradients)
  - Build state via `DifferentiableStateAssembler`
  - Feed directly to SAC actor (no replay buffer for this gradient path)
  - This is a **model-based RL augmentation**: real transitions from replay buffer (off-policy, no Kalman gradients) + synthetic rollouts through the differentiable Kalman (on-policy, with gradients)
- ~200 LOC

**B.4: End-to-end gradient test**
- Verify loss.backward() on SAC actor loss produces non-zero gradients in Kalman F, Q, H, R parameters
- ~50 LOC tests

### Total effort: Phase A ~200 LOC, Phase B ~380 LOC

---

## 6. Design Decisions for Phase B

### Why model-based augmentation, not pure differentiable training?

Off-policy replay is correct and necessary for SAC (sample efficiency). Throwing it away to get end-to-end gradients would be wrong. Instead:
- **Replay buffer path** (existing): transitions from real experience → SAC update. No Kalman gradients. This is the primary learning signal.
- **Model-based path** (new): Kalman runs forward on current observations → differentiable state → SAC actor loss → backward through Kalman. This is an auxiliary gradient that fine-tunes Kalman params toward what SAC finds useful.

This is analogous to how Dreamer (Hafner et al., 2020) augments real experience with imagined rollouts through a learned world model.

### Why not make the whole assembler differentiable?

Most of the state vector (instrument surprises, entity surprises, market features, adversarial flags) comes from non-differentiable sources (GNN embeddings stored as numpy, aggregated alert stats). Making those differentiable requires rearchitecting the GNN→feature pipeline — that's a separate (much larger) change.

The belief block (E×4 floats per entity) is the specific segment that benefits from differentiation because it comes from the Kalman filter's `_x` and `_P` tensors. A targeted bypass for just this block is the right scope.

### What about the Joseph form numerical stability?

The `update()` method uses Joseph form: `P' = (I - KH)P(I - KH)^T + KR'K^T`. This is more numerically stable than the standard form but produces longer computation graphs. For the small state dimensions we use (3D state, 17D obs), this is not a memory concern. For larger state spaces (future), consider using the square-root form (Cholesky Kalman) instead.

---

## 7. Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Phase A changes break inference | Medium | Inference path uses InstrumentStateAssembler separately; fix only training path |
| Transition writer creates DB bloat | Low | Set retention policy (e.g., last 10K transitions) |
| Differentiable bypass makes training unstable | Medium | Auxiliary gradient weight as hyperparameter (start at 0.01, tune) |
| Kalman gradients are noisy/uninformative | Low | Monitor gradient magnitude vs replay gradients; disable if harmful |
| Assembler dimension conflict between old/new checkpoints | Medium | Checkpoint stores state_dim; validate on load |

---

## 8. References

- Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning." — SAC algorithm, target network detach
- Hafner, D., et al. (2020). "Dream to Control: Learning Behaviors by Latent Imagination." — Model-based augmentation of off-policy RL
- Sarkka, S. (2013). "Bayesian Filtering and Smoothing", Ch. 4. — Kalman filter mathematics
- The spec: [[learned_vs_handcoded_architecture_spec]], Change 10 (Options A/B/C)

---

## Related

- [[learned_vs_handcoded_architecture_spec]]
- [[learned_vs_handcoded_audit]]
- [[tier5_differentiable_kalman]]
- [[tier4_learned_state_encoder]]
- [[differentiable_kalman]]
