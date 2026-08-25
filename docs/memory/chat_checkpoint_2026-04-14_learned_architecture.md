---
title: "Checkpoint: Self-Improving Architecture — Tier 1+2 Implementation"
tags:
  - doc/checkpoint
  - phase/25
  - topic/learning-agent
  - topic/self-improving
  - layer/world-model
  - layer/fusion
  - layer/learning
---

# Checkpoint: Self-Improving Architecture — Tier 1+2 Implementation

**Date:** 2026-04-14
**Session purpose:** Audit learned vs hand-coded ratio, document architectural changes, implement Tier 1 + Tier 2 changes to make TirraMind more end-to-end learned.
**Task file:** [[learned_architecture_impl]]
**Research:** [[learned_vs_handcoded_audit]]
**Spec:** [[learned_vs_handcoded_architecture_spec]]

---

## Session Narrative

The session had three distinct phases:

1. **Audit & Discussion** — Deep codebase audit of all 7 layers, concluding TirraMind is ~25% learned / ~75% hand-coded. Key finding: the learned components (GNN, SAC, Thompson bandit, Kalman, HMM) exist but are disconnected — beliefs never reach the policy, fusion weights are static, loss weights are fixed, world model parameters never update from data.

2. **Documentation** — Created two formal artifacts:
   - `[[learned_vs_handcoded_audit]]` — Layer-by-layer scorecard of what's learned vs hand-coded in each subsystem
   - `[[learned_vs_handcoded_architecture_spec]]` — 10 prioritized architectural changes across 5 tiers to reach 75% learned / 25% hand-coded

3. **Implementation** — Coded 5 changes (Tier 1 + Tier 2), created 37 tests, fixed 4 test failures including one real production bug. All 37 tests passing.

---

## What Was Implemented (5 Changes)

### Change 1: Wire Beliefs → SAC Policy (Tier 1) ✅

**Problem:** In `inference.py:299-302`, world model beliefs were stubbed (`beliefs = []`). SAC got zero-padded belief features. Two learned components (world model + RL policy) existed but weren't connected.

**Solution:**
- **`agent/pipeline/store.py` L828-850**: Added `query_all_latest_beliefs()` method. SQL uses `GROUP BY variable_name` with `MAX(effective_at)` subquery to get latest belief per variable. Returns `list[dict]`.
- **`agent/pipeline/dags/inference.py` L299-315**: Replaced stubbed `beliefs = []` with:
  1. `store.query_all_latest_beliefs()` → convert via `BeliefState.from_dict()`
  2. Pack global beliefs (where `entity_id is None`) into `market_features` as `belief.{variable_name}` → `mean`
  3. Pass both `beliefs` and `market_features` to `assembler.assemble()`

**Design decision:** Belief table schema does NOT include `entity_id` column (it's in the BeliefState dataclass but not persisted to SQL). Current world model beliefs are per-variable (e.g., `latent.stress_level`), not per-entity. So we pack them as global market features now; per-entity beliefs will work when the schema supports them.

**Net effect:** SAC now receives world model state (latent stress, macro momentum, liquidity state) as part of its observation — the belief→policy loop is closed.

---

### Change 4: Adaptive Surprise Fusion Weights (Tier 2) ✅

**Problem:** 5 surprise channels fused with static weights `(0.30, 0.15, 0.25, 0.20, 0.10)` — never adapt to which channels actually predict anomalous outcomes.

**Solution:**
- **`agent/fusion/surprise.py` L375-507**: Added `AdaptiveSurpriseWeights` class (~130 LOC):
  - `__init__`: initial_weights tuple, learning_rate (default 0.01), min_weight (default 0.02)
  - `update(gradients)`: Exponentiated Gradient on simplex — `w_k *= exp(-eta * grad_k)`, enforce min_weight floor, renormalize to sum=1
  - `compute_gradients(surprise_vectors, outcomes)`: MSE-based gradient per channel
  - `to_dict()` / `from_dict()`: serialization for persistence
  - Properties: `weights`, `weights_tuple`, `n_updates`
- **`agent/fusion/surprise.py` L115-126**: Added `sync_adaptive_weights()` to `SurpriseExtractor` — if adaptive weights attached, copies their current weights into the extractor's active weight tuple. Called at start of `extract()`.
- **`SurpriseExtractor.__init__`**: New `adaptive_weights: AdaptiveSurpriseWeights | None = None` parameter.

**Math:** EG on simplex (Kivinen & Warmuth 1997). Guarantees weights stay on probability simplex without projection. min_weight floor prevents any channel from collapsing to zero (preserves exploration).

**What's NOT wired yet:** The gradient computation from downstream outcomes. The `compute_gradients()` method exists but nothing calls it in the pipeline yet — that requires a retrospective evaluation step (compare surprise predictions to actual anomalous outcomes after they're confirmed).

---

### Change 9: GNN Loss Auto-Tuning (Tier 2) ✅

**Problem:** GNN trainer uses fixed loss weights: `obs_type_weight=1.0`, `time_delta_weight=0.1`, `contrastive_weight=0.5`, `value_weight=0.3`. Early training should emphasize easier tasks; later training should shift to harder ones.

**Solution:**
- **`agent/models/gnn/trainer.py` L310**: Added `auto_tune_loss_weights: bool = False` to `TrainerConfig`
- **L386-406**: When enabled, `build_model()` creates `self._log_vars` dict with 4 `nn.Parameter` log-variances, initialized from config weights as `-log(config_weight)`. Added to optimizer param groups alongside model params.
- **L712-725**: Modified loss computation: when `_log_vars is not None`, uses Kendall et al. 2018 uncertainty weighting: `loss_total = sum(exp(-log_var_k) * loss_k + log_var_k)` instead of `sum(fixed_weight_k * loss_k)`.
- **L768-778**: Epoch-level logging of effective weights when auto-tuning.
- **L780-793**: Added `effective_loss_weights()` method returning `{task: exp(-log_var)}` when auto-tuning, else config weights.

**Math:** Kendall, Gal & Cipolla (2018) "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics." The `log_var` parameterization ensures precision (`1/sigma^2`) stays positive without constrained optimization. The `+ log_var` term acts as regularizer preventing any task from being silenced.

**Backward compatible:** `auto_tune_loss_weights=False` (default) preserves existing fixed-weight behavior exactly.

---

### Change 2a: CPD Learning via Bayesian Estimation (Tier 2) ✅

**Problem:** Expert causal DAG has 20 nodes with hand-set CPD probability tables ("weakly informative priors") that never update from observed data.

**Solution:**
- **`agent/models/world_model.py` L201-314**: Added `fit_cpds()` method (~120 LOC):
  - Takes `feature_history: list[list[EngineeredFeature]]`, `equivalent_sample_size` (BDeu hyperparameter, default 10.0), `min_samples` (default 100)
  - Builds pandas DataFrame from features, discretizes continuous values via `_discretize()` using each node's `bin_edges`
  - **Guard (bug fix):** Skips nodes where the data column is entirely NaN or missing — prevents pgmpy BDeu from producing malformed CPDs for latent/regime nodes with no feature observations
  - Uses `pgmpy.estimators.BayesianEstimator` with BDeu priors to fit each node's CPD
  - Validates graph post-fit, logs warnings on errors
  - Returns `{"fitted": bool, "n_samples": int, "nodes_fitted": list[str]}`
- **`agent/models/world_model.py` L316-333**: Added `_discretize()` static method — maps continuous value to state label using node's `bin_edges`.

**Math:** BDeu (Bayesian Dirichlet equivalent uniform) prior from Heckerman et al. (1995). The `equivalent_sample_size` controls how quickly data overwhelms the expert priors — with ESS=10, roughly 50 observations per parent configuration will dominate.

**Bug found & fixed:** pgmpy's `BayesianEstimator` with BDeu produces divide-by-zero warnings and malformed CPDs (probabilities don't sum to 1) for root nodes that have no observed data column. Root cause: `regime.macro` is a latent node with `feature_name=None`, so no data column exists. Fix: skip fitting for any node where `df[node_name].notna().sum() == 0`.

---

### Change 2b: Kalman EM Parameter Fitting (Tier 2) ✅

**Problem:** Kalman filter matrices F (3×3 per regime), Q (3×3 per regime), H (17×3), R (17×17) are all hand-set constants. The filter can't learn from experience.

**Solution:**
- **`agent/models/state_filter.py` L294-~490**: Added `fit_filter_params()` method (~200 LOC) implementing full Shumway-Stoffer EM:
  - **E-step forward:** Standard Kalman filter with log-likelihood accumulation, handles missing observations (NaN masking of H/R rows)
  - **E-step backward:** RTS (Rauch-Tung-Striebel) smoother computes smoothed state estimates and lag-one cross-covariances
  - **M-step:** Closed-form MLE updates for F, Q per regime and shared H, R:
    - $\hat{F}_r = (\sum_t x_t x_{t-1}^T)(\sum_t x_{t-1} x_{t-1}^T)^{-1}$ (regime-conditioned)
    - $\hat{Q}_r = \frac{1}{T_r}\sum_t (P_t^s + x_t^s {x_t^s}^T - \hat{F}_r(P_{t,t-1}^s + x_{t-1}^s {x_t^s}^T)^T)$
    - $\hat{H} = (\sum_t y_t {x_t^s}^T)(\sum_t P_t^s + x_t^s {x_t^s}^T)^{-1}$
    - $\hat{R} = \frac{1}{T}\sum_t (y_t y_t^T - \hat{H}(P_t^s + x_t^s {x_t^s}^T)\hat{H}^T)$
  - **PSD enforcement:** After each M-step, checks eigenvalues of Q and R; if any negative, clamps to ε=1e-8 and reconstructs via eigendecomposition
  - **Convergence:** Monitors relative log-likelihood change; stops when `|ΔLL/LL| < tol` or `max_iter` reached
  - Applies fitted params back to `self._regime_configs`, `self._H`, `self._R`
  - Returns `{"fitted": bool, "n_samples": int, "iterations": int, "log_likelihoods": list[float]}`

**Math:** Shumway & Stoffer (2017) "Time Series Analysis and Its Applications", Chapter 6. The EM algorithm is guaranteed to monotonically increase the log-likelihood (modulo numerical issues). RTS smoother is the standard backward pass for linear Gaussian state-space models.

**Key numerical decision:** PSD enforcement via eigenvalue clamping rather than adding εI. Eigenvalue clamping is more surgical — it fixes only the problematic directions instead of inflating all variances.

---

## Test Suite: 37/37 Passing ✅

**File:** `tests/test_learned_architecture.py` (666 LOC)

| Test Class | Tests | What's Covered |
|-----------|-------|---------------|
| `TestQueryAllLatestBeliefs` | 4 | Empty store, latest-per-variable selection, single belief, many variables (50) |
| `TestAdaptiveSurpriseWeights` | 12 | Default weights, custom weights, wrong-length raises, EG update moves weights, min_weight prevents collapse, zero gradients no-op, gradient computation (empty, mismatched, basic), serialization roundtrip, integration with SurpriseExtractor |
| `TestGNNLossAutoTuning` | 6 | Config default off, config on, log_vars creation, log_vars=None when off, effective_weights fixed, effective_weights auto-tuned |
| `TestCPDLearning` | 7 | Insufficient samples, fitting with enough data, graph structure preservation post-fit, _discretize basic/boundary/low/None-states |
| `TestKalmanEM` | 8 | Insufficient samples, mismatched labels raises, EM convergence on synthetic, approximate param recovery, missing observations, multi-regime, PSD guarantee, 1D case |

**Bugs found by tests:**
1. Belief timestamps < 2020-01-01 epoch floor (test-only — fixed with `_BASE_TS = 1_700_000_000.0`)
2. `fit_cpds()` produces malformed CPDs for latent nodes with no observations (production bug — fixed with NaN guard)

---

## Files Modified — Complete Inventory

| File | Lines | What Changed |
|------|-------|-------------|
| `agent/pipeline/store.py` | 1957 | Added `query_all_latest_beliefs()` at L828-850 |
| `agent/pipeline/dags/inference.py` | 782 | Replaced stubbed beliefs (L299-315) with real query + market_features packing |
| `agent/fusion/surprise.py` | 500 | Added `AdaptiveSurpriseWeights` class (L375-507), `sync_adaptive_weights()` (L115-126), adaptive mode in `__init__` |
| `agent/models/gnn/trainer.py` | 1469 | Added `auto_tune_loss_weights` config (L310), `_log_vars` params (L386-406), uncertainty-weighted loss (L712-725), `effective_loss_weights()` (L780-793), epoch logging (L768-778) |
| `agent/models/world_model.py` | 333 | Added `fit_cpds()` (L201-314) with NaN guard, `_discretize()` (L316-333) |
| `agent/models/state_filter.py` | 546 | Added `fit_filter_params()` (L294-~490) with Shumway-Stoffer EM + RTS smoother + PSD enforcement |
| `tests/test_learned_architecture.py` | 666 | **Created** — 37 tests across 5 classes |
| `[[learned_architecture_impl]]` | ~70 | **Created** — task file tracking all steps |
| `[[learned_vs_handcoded_audit]]` | — | **Created** — layer-by-layer learned vs hand-coded scorecard |
| `[[learned_vs_handcoded_architecture_spec]]` | — | **Created** — 10 changes across 5 tiers with full math |

---

## What's Left on the Active Task

Two integration steps remain open:

### 2a.2: Wire `fit_cpds()` into world_model_update DAG
- **File:** `agent/pipeline/dags/world_model_update.py`
- **What:** In `run_world_model_update()`, before `wm.update(features, as_of)`, load historical feature window from PipelineStore, call `wm.fit_cpds(feature_history)` periodically (e.g., weekly, or when sample count exceeds threshold)
- **Guard:** If `fit_cpds()` returns `fitted=False`, proceed with expert CPDs (no-regression fallback)
- **Persistence:** Store fitted CPD parameters to PipelineStore for auditing

### 2b.2: Wire `fit_filter_params()` into world_model_update DAG
- **File:** `agent/pipeline/dags/world_model_update.py`
- **What:** After CPD fitting, load historical (features, regime_labels) from PipelineStore, call `wm._state_filter.fit_filter_params(observations, labels)`
- **Guard:** If EM fails or data insufficient, keep current params (warm-start behavior)
- **Ordering:** Must run after CPD fitting because updated beliefs from fitted CPDs provide better regime labels for Kalman EM

### Context for implementation:
- `world_model_update.py` is a single-node DAG: `run_world_model_update()` (FunctionOperator)
- Currently: builds model → loads latest features → runs `wm.update()` → stores beliefs
- Need to add: loads feature *history* → periodic `wm.fit_cpds()` → periodic `wm._state_filter.fit_filter_params()` → then existing update
- The store has `query_features()` which can retrieve historical feature rows
- Regime labels come from beliefs (discrete variable `regime.macro`)

---

## Full Roadmap — Remaining Tiers

### Tier 3: Learn Meta-Parameters (2-4 weeks)
- **Change 5:** Reward weight optimization — Bayesian optimization over 5-dim reward weight space, quarterly, using rolling Sharpe as objective. Needs `RewardWeightOptimizer` in `agent/learning/reward.py`.
- **Change 7:** Detector threshold optimization — GP-BO over CUSUM k/h, Hawkes μ/α/β, convergence z/p. Monthly, evaluated by F1 of entity alerts vs confirmed anomalies. Needs `ThresholdOptimizer`.
- **Change 8:** Dynamic goal arm discovery — Hierarchical bandit with "novel" meta-arm. LLM generates unconstrained goals; successful ones get promoted to permanent arms after N=3 successes. Needs `NovelExplorationArm` in bandit.

### Tier 4: Learn Representations (1-2 months)
- **Change 6:** Learned state encoder — Replace hand-designed `InstrumentStateAssembler` with attention-based (Set Transformer) encoder. Train end-to-end with SAC. ~400 LOC.
- **Change 3:** Causal graph structure learning — Hill-climb search from expert graph with constraints (regime roots, acyclicity, max in-degree 4). BIC/BDeu scoring. Quarterly. ~150 LOC.

### Tier 5: End-to-End Differentiable (2-3 months)
- **Change 10:** Differentiable Kalman (Option C) or variational world model (Option B/Dreamer V3). Enables gradient flow from portfolio loss → world model → GNN. The ultimate target.

### Expected progression:

| After Tier | % Learned | % Hand-Coded |
|-----------|-----------|-------------|
| Current (pre-session) | 25% | 75% |
| Tier 1+2 (this session) | ~45% | ~55% |
| Tier 3 | ~55% | ~45% |
| Tier 4 | ~65% | ~35% |
| Tier 5 | ~75% | ~25% |

---

## Key Architectural Decisions Made

1. **Global beliefs → market_features (not per-entity).** Current belief schema doesn't persist `entity_id` to SQL. Rather than refactoring the schema, we pack global beliefs into `market_features` dict as `belief.{variable_name}`. SAC gets world model state immediately. Per-entity beliefs can be added later when the schema supports it.

2. **EG on simplex (not softmax reparameterization) for surprise weights.** EG preserves the simplex constraint naturally without projection. `min_weight` floor prevents channel collapse while maintaining exploration.

3. **Kendall et al. uncertainty weighting (not gradient normalization) for GNN losses.** Uncertainty weighting has a clean probabilistic interpretation and is a single-line change to loss computation. GradNorm requires a separate loss for the normalizer which adds complexity.

4. **BDeu priors (not K2 or flat) for CPD learning.** BDeu's `equivalent_sample_size` gives explicit control over prior strength. At ESS=10, ~50 observations per parent config will dominate the expert prior. Compatible with pgmpy's `BayesianEstimator` out of the box.

5. **Skip nodes with no observations in fit_cpds().** Rather than trying to infer CPDs for latent/regime nodes from downstream evidence (which would require a different estimator), we simply skip them. Expert priors remain for unobserved nodes. This is the conservative choice.

6. **PSD enforcement via eigenvalue clamping.** More surgical than adding εI — fixes only problematic eigenvalue directions without inflating well-conditioned ones.

7. **auto_tune_loss_weights defaults to False.** Existing behavior is preserved exactly. Opt-in by setting the config flag. No surprise regressions.

---

## Regression Risk Assessment

| Change | Risk | Mitigation |
|--------|------|-----------|
| Beliefs → SAC | Low — additive only, SAC already handles zero beliefs | If beliefs are stale/corrupt, market_features dict is empty; SAC falls back to previous behavior |
| Adaptive surprise | Zero — opt-in only via `adaptive_weights` param | SurpriseExtractor default construction is unchanged |
| GNN auto-tune | Zero — opt-in via `auto_tune_loss_weights=False` default | Fixed-weight path unchanged |
| CPD fitting | Low — guard skips nodes with no data | Expert CPDs preserved for all unobserved nodes; `min_samples` prevents fitting on tiny datasets |
| Kalman EM | Low — warm-start from expert params | If EM diverges, current params persist (EM only applies when it converges); PSD enforcement prevents numerical blowup |

---

## How to Resume

1. Read this checkpoint
2. Read [[learned_architecture_impl]] for step-by-step status
3. The immediate next work is steps 2a.2 and 2b.2: wiring `fit_cpds()` and `fit_filter_params()` into `agent/pipeline/dags/world_model_update.py`
4. After that, write DAG-level integration tests (fit skipped on insufficient data, fit + update in sequence, EM failure fallback)
5. Then run broader regression tests across the pipeline

---

## Related

- [[learned_architecture_impl]] — Active task file
- [[learned_vs_handcoded_audit]] — Research: full audit
- [[learned_vs_handcoded_architecture_spec]] — Spec: 10 changes across 5 tiers
- [[e2e_global_integration]] — Phase 24 context
- [[rl_policy]] — SAC implementation (Phase 21)
- [[world_model_bridge]] — GNN → world model bridge (Phase 19)
- [[signal_fusion]] — Surprise extraction (Phase 20)
- [[project_memory]] — Persistent architectural memory
