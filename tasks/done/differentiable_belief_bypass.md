---
title: "Task: Differentiable Belief Bypass (Phase B)"
tags:
  - doc/task
  - status/done
  - phase/25
  - topic/self-improving
  - topic/differentiable
  - layer/fusion
  - layer/learning
---

# Task: Differentiable Belief Bypass (Phase B)

Status: completed
Research: [[tier5_gradient_bypass]]
Spec: [[differentiable_belief_bypass_spec]]

## Steps

### B.1: get_beliefs_differentiable()
- [x] B.1.1: Add `get_beliefs_differentiable()` method to DifferentiableKalmanFilter
- [x] B.1.2: Returns (means, variances) tensors without `.detach()`

### B.2: DifferentiableStateAssembler
- [x] B.2.1: Create `DifferentiableStateAssembler` class in `state_assembler.py`
- [x] B.2.2: Accept tensor beliefs + numpy non-differentiable blocks
- [x] B.2.3: Layout matches `InstrumentStateAssembler` exactly
- [x] B.2.4: Only belief block carries gradients

### B.3: Model-based augmentation in _train_sac()
- [x] B.3.1: Add `aux_kalman_weight` to SACConfig / PolicyConfig
- [x] B.3.2: Load or build DiffKalman in `_train_sac()`
- [x] B.3.3: Implement forward pass: predict → update → diff beliefs → diff assemble → actor
- [x] B.3.4: Compute and backward auxiliary loss scaled by weight
- [x] B.3.5: Separate Kalman optimizer with gradient clipping
- [x] B.3.6: Report aux metrics

### B.4: Edge-case test suite
- [x] B.4.1: Gradient existence test (F, Q, H, R get non-zero .grad)
- [x] B.4.2: Gradient magnitude test (not exploding, not vanishing)
- [x] B.4.3: Layout consistency (DiffAssembler.state_dim == InstrumentAssembler.state_dim)
- [x] B.4.4: Detach isolation (aux backward doesn't corrupt SAC params)
- [x] B.4.5: NaN/zero observation robustness
- [x] B.4.6: Multi-regime gradient routing
- [x] B.4.7: DiffKalman unavailable → graceful skip
- [x] B.4.8: Full integration: replay update + aux update on same batch

## Related

- [[tier5_gradient_bypass]]
- [[differentiable_belief_bypass_spec]]
- [[fix_sac_training_pipeline]]
- [[learned_vs_handcoded_architecture_spec]]
- [[tier5_differentiable_kalman]]
