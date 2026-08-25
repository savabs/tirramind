---
title: "Research: Differentiable Kalman Filter (Change 10c)"
tags:
  - doc/research
  - phase/25
  - topic/self-improving
  - topic/differentiable
  - layer/fusion
---

# Research: Differentiable Kalman Filter (Change 10, Option C)

## Problem Statement

The current pipeline has **two hard gradient breaks** between the world model and SAC policy:

1. **ContinuousStateFilter** (`agent/models/state_filter.py`) — pure numpy predict/update. No torch tensors, no autograd.
2. **SQLite round-trip** — beliefs serialized to JSON, stored in DB, queried back as dicts. All differentiability lost.

The spec identifies Option C (Differentiable Kalman) as the smallest-scope change toward end-to-end gradient flow. This replaces the numpy Kalman with a PyTorch `nn.Module`.

## Current Architecture

### ContinuousStateFilter (numpy)
- **state_dim=3**: latent stress_level, macro_momentum, liquidity_state
- **obs_dim=17**: 6 macro/convergence + 5 GNN anomaly + 5 GNN activity + 1 cross-entity
- **Regime-conditioned**: F, Q per regime (expansion/contraction/crisis); H, R shared
- **EM fitting** (Change 2b): `fit_filter_params()` updates F/Q/H/R in-place via Shumway-Stoffer EM
- **No serialization**: fitted params live only in the filter instance per run

### Gradient Break Analysis
```
Features → ContinuousStateFilter.update() [NUMPY — no gradients]
         → get_beliefs() → Python floats
         → store.store_beliefs_batch() [SQLITE — serialized]
         → store.query_all_latest_beliefs() [DESERIALIZED]
         → assembler → torch.from_numpy() [LEAF TENSOR — no grad]
         → encoder → SAC
```

## Design: DifferentiableKalmanFilter(nn.Module)

### Core Architecture
Same linear-Gaussian state-space model, reimplemented in PyTorch:

$$x_t = F_r \cdot x_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, Q_r)$$
$$y_t = H \cdot x_t + v_t, \quad v_t \sim \mathcal{N}(0, R)$$

### Learnable Parameters
- **F_r** per regime: `nn.Parameter` (state_dim × state_dim)
- **L_Q_r** per regime: Cholesky factor, `nn.Parameter` (state_dim × state_dim lower triangular). Q_r = L_Q @ L_Q^T ensures PSD.
- **H**: `nn.Parameter` (obs_dim × state_dim)
- **L_R**: Cholesky factor, `nn.Parameter` (obs_dim × obs_dim lower triangular). R = L_R @ L_R^T ensures PSD.

### PSD Parameterization (Cholesky)
Covariance matrices must be symmetric positive-definite. Unconstrained optimization of all entries can produce non-PSD matrices. Standard approach:

$$Q = L_Q L_Q^T + \epsilon I$$

where $L_Q$ is lower triangular with positive diagonal (enforced via softplus). $\epsilon$ = 1e-6 floor for numerical stability.

### Missing Data Masking
Same approach as numpy filter — reduce H, R, y to valid rows. In torch: use boolean indexing on observation vector.

### Regime Switching
Store per-regime (F_r, L_Q_r) as `nn.ParameterDict` or `nn.ModuleDict`. Select by regime name at predict time.

### What This Does NOT Do Yet
- Does NOT wire gradients through the full pipeline to SAC (that requires removing the SQLite break — a separate future change)
- Does NOT replace the EM fitting (EM can run on numpy, then transfer params to torch via `load_numpy_params()`)
- The primary value is: **filter operations preserve gradients**, so when the belief→policy path is later made differentiable, the filter is ready

### What This DOES Do
1. Filter parameters are `nn.Parameter` — they can be optimized by any torch optimizer
2. `predict()`/`update()` preserve autograd graph — `loss.backward()` flows through filter ops
3. `state_dict()`/`load_state_dict()` come free from `nn.Module` — parameters can be checkpointed
4. `from_numpy_filter()` method imports params from existing ContinuousStateFilter (initialize from expert or EM-fitted values)
5. Foundation for future end-to-end training (Change 10 full)

## Risks

1. **Numerical stability** — Kalman gain involves matrix inverse via `torch.linalg.solve`. Joseph form + Cholesky parameterization mitigate.
2. **No immediate SAC gradient connection** — this change is foundational, not end-to-end. Value comes from checkpointing and future wiring.
3. **Regime transitions are discrete** — selecting F_r by regime name is not differentiable. Future: soft attention over regimes.

## Trusted References

- Sarkka S. (2013). "Bayesian Filtering and Smoothing", Cambridge University Press, Ch. 4
- Haarnoja et al. (2018). arXiv:1801.01290 — SAC (downstream consumer)
- Shumway & Stoffer (2017). "Time Series Analysis and Its Applications", Ch. 6 — EM for state-space models

## Related

- [[learned_vs_handcoded_architecture_spec]]
- [[learned_state_encoder]]
- [[tier4_learned_state_encoder]]
