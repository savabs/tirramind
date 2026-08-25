---
title: "Spec: Differentiable Belief Bypass (Phase B)"
tags:
  - doc/spec
  - phase/25
  - topic/self-improving
  - topic/differentiable
  - layer/fusion
  - layer/learning
---

# Spec: Differentiable Belief Bypass (Phase B)

## Goal

Connect the Kalman filter's learnable parameters (F, Q, H, R) to the SAC policy loss via a differentiable bypass path, enabling end-to-end gradient flow from RL reward signal through to the world model's state-space dynamics.

This is **model-based RL augmentation** (analogous to Dreamer, Hafner et al. 2020): the primary SAC learning signal remains off-policy replay, but an auxiliary gradient fine-tunes Kalman parameters toward states that the policy finds informative.

## Mathematical Rationale

Currently, two gradient domains are disconnected:
- **Domain A (Kalman)**: F, Q, H, R are nn.Parameter but `get_beliefs()` calls `.detach()`, killing gradients.
- **Domain B (SAC)**: LearnedStateEncoder → GaussianActor → TwinCritic — clean gradient chain.

The bypass creates a direct computational path:

```
DiffKalman.predict(regime) → x_pred (autograd ✓)
DiffKalman.update(obs) → x_updated (autograd ✓)
  → DifferentiableStateAssembler.assemble_beliefs(x, P) → state_tensor (autograd ✓)
    → Actor(state_tensor) → log_prob → auxiliary_actor_loss
      → .backward() → ∂loss/∂F, ∂loss/∂Q, ∂loss/∂H, ∂loss/∂R
```

The auxiliary gradient weight (`aux_kalman_weight`) starts at 0.01 to prevent destabilizing the primary RL signal.

## Files Affected

| File | Change |
|------|--------|
| `agent/models/diff_kalman.py` | Add `get_beliefs_differentiable()` method |
| `agent/learning/policy/state_assembler.py` | Add `DifferentiableStateAssembler` class |
| `agent/pipeline/dags/rl_training.py` | Add model-based augmentation path in `_train_sac()` |
| `tests/test_differentiable_bypass.py` | End-to-end gradient + edge-case tests |

## Implementation Steps

### B.1: `get_beliefs_differentiable()` (~30 LOC)

Add to `DifferentiableKalmanFilter`:
```python
def get_beliefs_differentiable(self) -> tuple[Tensor, Tensor]:
    """Return (belief_means, belief_variances) WITHOUT detaching.
    
    belief_means: shape (state_dim,) — current state estimate
    belief_variances: shape (state_dim,) — diagonal of covariance
    
    Used ONLY during training for model-based gradient augmentation.
    """
    return self._x, torch.diagonal(self._P)
```

This is a 2-line method that simply exposes the autograd-connected internal state. The existing `get_beliefs()` remains unchanged for inference-time SQLite persistence.

### B.2: `DifferentiableStateAssembler` (~100 LOC)

New class in `state_assembler.py` that builds a state tensor with gradient-connected belief features:

- Accepts `belief_means: Tensor` and `belief_variances: Tensor` (from B.1)
- Non-differentiable components (instrument surprises, entity surprises, market features, adversarial) are still numpy → `torch.from_numpy()` (detached)
- Belief block uses `torch.stack()` to preserve gradients through belief_mean, belief_var
- Layout matches `InstrumentStateAssembler` exactly (same state_dim, same block ordering)
- Returns `(state_tensor, metadata)` — tensor has `requires_grad=True` via belief features

Key design: only the belief block (E×4: mean, var, confidence, stale) carries gradients. Confidence and stale are set to constant 1.0/0.0 tensors with `requires_grad=False`. This matches the architecture: only Kalman parameters should receive the auxiliary gradient.

### B.3: Model-based augmentation in `_train_sac()` (~200 LOC)

Add to the training function after the existing SAC update:

1. Check if `DifferentiableKalmanFilter` is available (via checkpoint or config flag)
2. If available, perform a **model-based forward pass**:
   - Load recent observations from store
   - Run `kalman.predict(regime)` → `kalman.update(obs)` (preserving autograd)
   - Get differentiable beliefs: `means, vars = kalman.get_beliefs_differentiable()`
   - Assemble state via `DifferentiableStateAssembler`
   - Run through actor: `action, log_prob = actor.sample(diff_state)`
   - Compute auxiliary loss: `aux_loss = (alpha * log_prob - q_min).mean()` (same as actor loss)
   - Scale: `total_aux = config.aux_kalman_weight * aux_loss`
   - `total_aux.backward()` — gradients now flow into Kalman params
   - Clip Kalman gradients separately (max norm 0.1)
   - Step a separate Kalman optimizer
3. Report `aux_actor_loss` and `kalman_grad_norm` in metrics

The Kalman optimizer is Adam with a lower learning rate (1e-4) to prevent the auxiliary signal from overwhelming the EM-fitted initial values.

### B.4: End-to-end gradient tests (~80 LOC)

1. **Gradient existence**: Build DiffKalman → predict → update → get_beliefs_differentiable → DifferentiableStateAssembler → GaussianActor.sample → loss.backward() → verify F, Q, H, R have non-zero .grad
2. **Gradient magnitude**: Verify gradients are reasonable (not exploding, not vanishing to 1e-30)
3. **Layout consistency**: DifferentiableStateAssembler.state_dim == InstrumentStateAssembler.state_dim for same config
4. **Detach isolation**: Verify that the auxiliary backward does NOT affect SAC actor/critic parameters beyond what the SAC optimizer did
5. **NaN robustness**: Feed NaN observations → verify no NaN in gradients
6. **Zero observations**: All-zero obs → gradients still finite
7. **Multiple regimes**: Predict with different regimes → verify correct F gets gradient

## Edge Cases

- DiffKalman not available (no checkpoint) → skip augmentation, return metrics without aux fields
- Zero valid observations → Kalman update is no-op; gradient path through predict() only
- Very large state/obs (stress test memory with state_dim=50)
- Kalman divergence (P goes to infinity) → gradient clipping prevents propagation

## Testing Plan

- 48 existing tests (Phase A) must remain green
- 66 DifferentiableKalmanFilter tests must remain green
- New tests in `tests/test_differentiable_bypass.py` covering B.1-B.4

## Related

- [[tier5_gradient_bypass]]
- [[fix_sac_training_pipeline_spec]]
- [[learned_vs_handcoded_architecture_spec]]
- [[tier5_differentiable_kalman]]
- [[tier4_learned_state_encoder]]
