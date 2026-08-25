---
title: "Checkpoint: Self-Improving Architecture — Tier 1+2 Fully Complete"
tags:
  - doc/checkpoint
  - phase/25
  - topic/learning-agent
  - topic/self-improving
  - topic/world-model
  - topic/meta-learning
  - layer/world-model
  - layer/fusion
  - layer/learning
---

# Checkpoint: Self-Improving Architecture — Tier 1+2 Fully Complete

**Date:** 2026-04-14 (end of session)
**Task file:** [[learned_architecture_impl]]
**Research:** [[learned_vs_handcoded_audit]]
**Spec:** [[learned_vs_handcoded_architecture_spec]]

---

## One-Paragraph Summary

Two sessions on 2026-04-14 completed the full Tier 1 + Tier 2 self-improving architecture work:

- **Session 1** (earlier checkpoint: `chat_checkpoint_2026-04-14_learned_architecture.md`): audited the entire codebase (25% learned / 75% hand-coded), wrote a 10-change spec across 5 tiers, implemented 5 core changes (belief→policy wiring, adaptive surprise weights, GNN loss auto-tuning, CPD learning, Kalman EM fitting), created 37 edge case tests, and fixed a production bug in fit_cpds().

- **Session 2** (this checkpoint): wired the CPD and Kalman fitting methods into the world_model_update DAG as a periodic fitting step (steps 2a.2 and 2b.2 — the final unchecked items), created 36 integration tests covering the full DAG fitting lifecycle.

**All 14 task steps are now checked off. All 73 tests pass. Tier 1+2 work is done.**

---

## What Session 2 Added

### The Problem

Session 1 implemented `WorldModel.fit_cpds()` and `ContinuousStateFilter.fit_filter_params()` as standalone methods. But they weren't called anywhere in the production pipeline — the world_model_update DAG still ran with expert (hard-coded) parameters every time. The last two unchecked steps in the task file were:

- 2a.2: Wire `fit_cpds()` into world_model_update DAG as periodic re-fit step
- 2b.2: Wire `fit_filter_params()` into world_model_update DAG after CPD fitting

### The Solution: `_maybe_fit_params()` Orchestrator

Added ~200 LOC of integration code to `agent/pipeline/dags/world_model_update.py`:

**5 new helper functions:**

1. **`_should_fit(store, as_of, fit_interval_days)`** — Periodicity control. Uses `pipeline_data` table with `source="world_model_fit"` to track when the last fit occurred. Critically, it uses the *logical* `as_of` timestamp from the stored marker params — not the wall-clock `fetched_at` — so backtesting with historical `as_of` values also respects the interval correctly. Returns `(should_fit: bool, reason: str)`.

2. **`_load_feature_history(store, since, until)`** — Queries all feature names from the store, groups results into daily UTC buckets using `int(effective_at // 86400)` as day keys. Returns `list[list[EngineeredFeature]]` sorted oldest-first. Each inner list is one day's snapshot, suitable for passing to `wm.fit_cpds()`.

3. **`_load_regime_labels(store, day_keys, default_regime)`** — Queries stored `regime.macro` beliefs, extracts the MAP (most probable) state from each belief's `probabilities` dict, and maps them to the corresponding day key. Days without stored beliefs fall back to `default_regime` (first regime config key, or "expansion"). Returns one label per day key — exactly the shape `fit_filter_params()` expects.

4. **`_build_observation_sequence(snapshots)`** — Converts daily feature snapshots into `(_OBS_DIM,)` numpy arrays using the existing `_FEATURE_TO_OBS_INDEX` mapping. Features missing from a snapshot or with `value is None` are left as NaN. The Kalman filter already handles NaN rows as missing observations in its E-step (valid mask on H/R rows).

5. **`_maybe_fit_params(store, wm, as_of, ...)`** — The orchestrator. Full flow:
   - Check periodicity → skip if too recent
   - Load 90 days of feature history, group into daily snapshots
   - Skip if < 10 daily snapshots (not enough data)
   - **Step 1:** Call `wm.fit_cpds(snapshots)` in try/except → CPD failure doesn't block anything
   - **Step 2:** Build observation sequences + load regime labels
   - Call `wm._filter.fit_filter_params(obs_seq, regime_labels)` in try/except → Kalman failure doesn't block anything
   - Need ≥ 30 observations for Kalman EM (its own `min_samples` guard also enforces this)
   - Store fit marker to `pipeline_data` for periodicity control
   - Return full result dict with `cpd_result`, `kalman_result`, `n_snapshots`

**Modified `run_world_model_update()`:**
- Now accepts 3 new params: `fit_enabled` (default True), `fit_interval_days` (default 7), `history_window_days` (default 90)
- Calls `_maybe_fit_params()` **before** `wm.update()` so newly fitted params are used in the same run's belief propagation + Kalman step
- Adds `fit_result` to the returned dict

### Key Design Decisions in Session 2

1. **Fit before update, not after.** Fitting runs before the normal `wm.update()` call so the just-fitted CPDs and Kalman matrices are used immediately in this run's belief generation. The alternative (fit after) would waste 7 days of stale params.

2. **Periodicity via `pipeline_data` stored markers, not config-file timestamps.** The `pipeline_data` table is the store's general-purpose key-value store. We use `source="world_model_fit"` and store `as_of` in the params dict so the periodicity check works correctly even in backtesting scenarios (where wall-clock time is irrelevant).

3. **Daily buckets via `int(effective_at // 86400)`.** Simple UTC-day bucketing. No timezone handling needed — features use Unix epoch timestamps and the DAG runs on UTC schedule. One snapshot per day keeps the CPD fitting from being overwhelmed by sub-daily observations that are essentially the same market state.

4. **Default regime fallback for missing belief days.** When there's no stored `regime.macro` belief for a given day, we use the first regime config key (typically "expansion"). This is safe because Kalman EM will simply lump those observations under the default regime's F/Q — a minor sub-optimality, not a correctness issue.

5. **Independent failure modes.** CPD fitting and Kalman EM are wrapped in independent try/except blocks. CPD failure doesn't prevent Kalman fitting (they're truly independent — CPDs are for the discrete DAG, Kalman is for the continuous state filter). Either failing still stores a fit marker, since partial fitting is better than no fitting.

6. **min 10 snapshots for CPD, min 30 for Kalman.** CPD fitting with BDeu priors is robust with less data (the prior provides regularization), but Kalman EM needs more observations for the M-step covariance estimates to be reasonable. The 30 threshold matches the `min_samples` default in `fit_filter_params()` itself.

---

## Complete Files Modified (Cumulative, Both Sessions)

| File | Lines | What Changed |
|------|-------|-------------|
| `agent/pipeline/store.py` | 1957 | Added `query_all_latest_beliefs()` at L828-850 |
| `agent/pipeline/dags/inference.py` | 782 | Replaced stubbed beliefs (L299-315) with real query + market_features packing |
| `agent/fusion/surprise.py` | 500 | Added `AdaptiveSurpriseWeights` class (L375-507), `sync_adaptive_weights()` (L115-126), adaptive mode in `__init__` |
| `agent/models/gnn/trainer.py` | 1469 | Added `auto_tune_loss_weights` config (L310), `_log_vars` params (L386-406), uncertainty-weighted loss (L712-725), `effective_loss_weights()` (L780-793) |
| `agent/models/world_model.py` | 333 | Added `fit_cpds()` (L201-314) with NaN guard, `_discretize()` (L316-333) |
| `agent/models/state_filter.py` | 546 | Added `fit_filter_params()` (L294-~490) with Shumway-Stoffer EM + RTS smoother + PSD enforcement |
| `agent/pipeline/dags/world_model_update.py` | ~370 | **Session 2:** Added `_should_fit()`, `_load_feature_history()`, `_load_regime_labels()`, `_build_observation_sequence()`, `_maybe_fit_params()`, modified `run_world_model_update()` to call fitting before update |
| `tests/test_learned_architecture.py` | 666 | **Session 1 created** — 37 tests across 5 classes |
| `tests/test_world_model_update_fitting.py` | ~550 | **Session 2 created** — 36 tests across 6 classes |
| `[[learned_architecture_impl]]` | ~70 | Created in Session 1, updated in Session 2 (all 14 steps checked) |
| `[[learned_vs_handcoded_audit]]` | — | **Session 1 created** — layer-by-layer scorecard |
| `[[learned_vs_handcoded_architecture_spec]]` | — | **Session 1 created** — 10 changes across 5 tiers with full math |

---

## Full Test Suite: 73/73 Passing

### Session 1 tests: `tests/test_learned_architecture.py` (37 tests)

| Test Class | Tests | What's Covered |
|-----------|-------|---------------|
| `TestQueryAllLatestBeliefs` | 4 | Empty store, latest-per-variable, single belief, 50 variables |
| `TestAdaptiveSurpriseWeights` | 12 | Default/custom weights, EG update, min_weight floor, gradient computation, serialization, SurpriseExtractor integration |
| `TestGNNLossAutoTuning` | 6 | Config on/off, log_vars creation, effective_weights fixed vs auto-tuned |
| `TestCPDLearning` | 7 | min_samples guard, fitting, graph preservation, _discretize edge cases |
| `TestKalmanEM` | 8 | min_samples, mismatched labels, EM convergence on synthetic, param recovery, missing obs, multi-regime, PSD guarantee, 1D case |

### Session 2 tests: `tests/test_world_model_update_fitting.py` (36 tests)

| Test Class | Tests | What's Covered |
|-----------|-------|---------------|
| `TestShouldFit` | 6 | No prior marker, recent skip, old trigger, exact boundary, as_of vs fetched_at, 1-day interval |
| `TestLoadFeatureHistory` | 4 | Empty store, daily grouping + feature count, oldest-first order, time window bounds, malformed row graceful skip |
| `TestLoadRegimeLabels` | 5 | No beliefs → default, correct MAP extraction, mixed regimes + missing days, empty keys, gaussian belief ignored |
| `TestBuildObservationSequence` | 7 | Known feature mapped, unknown ignored, None stays NaN, full 17-position snapshot, multi-snapshot, empty snapshot, output shape |
| `TestMaybeFitParams` | 9 | Disabled, recent skip, insufficient snapshots, full CPD+Kalman flow, CPD failure doesn't block Kalman, Kalman failure still stores marker, <30 obs skips Kalman, obs sequence shape verification, marker prevents double-fit |
| `TestRunWorldModelUpdateWithFitting` | 4 | fit_result in output, params forwarded, fit runs before update (call ordering), default param values |

---

## All 7 Architectural Decisions (Cumulative)

These are the design choices that a future session should not re-derive:

1. **Global beliefs → market_features (not per-entity).** Belief schema doesn't persist `entity_id` to SQL. Pack global beliefs into `market_features` dict as `belief.{variable_name}`. Per-entity beliefs deferred to future schema work.

2. **EG on simplex (not softmax reparameterization) for surprise weights.** Exponentiated Gradient naturally preserves simplex constraint. `min_weight` floor prevents channel collapse.

3. **Kendall et al. uncertainty weighting (not GradNorm) for GNN losses.** Clean probabilistic interpretation, single-line change. GradNorm requires a separate normalizer loss.

4. **BDeu priors (not K2 or flat) for CPD learning.** `equivalent_sample_size` gives explicit prior-strength control. ESS=10 → ~50 observations per parent config dominates.

5. **Skip nodes with no observations in fit_cpds().** Latent/regime nodes have no feature data. Expert priors remain for those nodes — conservative choice.

6. **PSD enforcement via eigenvalue clamping.** More surgical than εI addition — fixes only problematic directions.

7. **Fit before update in the DAG, not after.** Newly-fitted params are used immediately in the same run's belief generation.

8. **Periodicity via `pipeline_data` markers using logical `as_of`.** Backtesting-safe — doesn't rely on wall-clock time.

9. **Independent failure modes for CPD and Kalman fitting.** They're truly independent subsystems; one failing shouldn't block the other or the normal update.

---

## Regression Risk Still Low

| Change | Risk | Mitigation |
|--------|------|-----------|
| Beliefs → SAC | Low | If beliefs stale/corrupt, market_features dict empty → SAC falls back |
| Adaptive surprise | Zero | Opt-in only via `adaptive_weights` param |
| GNN auto-tune | Zero | `auto_tune_loss_weights=False` default |
| CPD fitting | Low | NaN guard, `min_samples`, expert CPDs preserved for latent nodes |
| Kalman EM | Low | Warm-start from expert params, PSD enforcement, try/except around fitting |
| **DAG fitting wiring** | **Low** | `fit_enabled=True` default but fitting is fully guarded: periodicity check, min snapshot count, min observation count, independent try/excepts, always stores marker. If everything fails, `wm.update()` still runs with expert params. |

---

## What's NOT Done Yet (Future Tiers)

The spec documents 10 changes across 5 tiers. Tiers 1+2 (5 changes) are complete.

### Tier 3: Learn Meta-Parameters (2-4 weeks estimated)

- **Change 5: Reward weight optimization.** Currently: `reward.py` has `eval_weight=0.4, sharpe_weight=0.3, facts_weight=0.2, novelty_bonus=0.1, dead_end_penalty=0.3` — all hand-coded. Need: Bayesian optimization over 5-dim reward weight space using rolling 90-day Sharpe as objective. `RewardWeightOptimizer` in `agent/learning/reward.py`. Quarterly schedule, ~20 evaluations per quarter.

- **Change 7: Detector threshold optimization.** Currently: CUSUM (k=0.5, h=5.0), Hawkes (μ=0.1, α=0.5, β=1.0), convergence (z=2.0, p=0.05, fdr_q=0.05) — all hand-tuned. Need: GP-BO over each detector's 2-5 dim param space, evaluated by F1 of entity alerts vs confirmed anomalous events. `ThresholdOptimizer` in `agent/convergence/` or `agent/fusion/`. Monthly schedule.

- **Change 8: Dynamic goal arm discovery.** Currently: 45 GoalArms in `bandit.py` are a fixed list. Need: Hierarchical Thompson Sampling with a "novel" meta-arm. When selected, LLM generates unconstrained goal. Successful novel goals get promoted to permanent arms after N=3 successes. `NovelExplorationArm` in bandit.

### Tier 4: Learn Representations (1-2 months)

- **Change 6: Learned state encoder.** Replace `InstrumentStateAssembler` (hand-designed tensor layout, top-K truncation) with Set Transformer (Lee et al. ICML 2019) that attends over variable-length entity sets. Train end-to-end with SAC. ~400 LOC.

- **Change 3: Causal graph structure learning.** Use constrained hill-climb search from expert graph (pgmpy `HillClimbSearch` + `BicScore`). Enforce: regime nodes are roots, observed nodes can't parent latent nodes, acyclicity, max in-degree 4. Quarterly schedule, accept edges only if BIC improvement > threshold. ~150 LOC.

### Tier 5: End-to-End Differentiable (2-3 months)

- **Change 10: Differentiable world model.** Option A: differentiable Kalman (torch-based E/M steps). Option B: Dreamer V3-style variational world model. Option C: full end-to-end where gradient flows from portfolio loss → world model → GNN. This is the ultimate target for self-improving TirraMind.

### Expected Learned vs Hand-Coded Progression

| After Tier | % Learned | % Hand-Coded |
|-----------|-----------|-------------|
| Pre-session (baseline) | 25% | 75% |
| Tier 1+2 (completed) | ~45% | ~55% |
| Tier 3 | ~55% | ~45% |
| Tier 4 | ~65% | ~35% |
| Tier 5 | ~75% | ~25% |

---

## Partially Wired Components (Won't Break, But Need Pipeline Connection)

1. **`AdaptiveSurpriseWeights.compute_gradients()`** — The method exists and is tested, but nothing in the pipeline *calls* it yet. It needs a retrospective evaluation step: after entity alerts are confirmed/denied by subsequent data, compute how well each surprise channel predicted the outcome, then call `adaptive_weights.update(gradients)`. This is a pipeline orchestration task, not a math task.

2. **`auto_tune_loss_weights=False`** — The GNN trainer has auto-tuning ready but it's off by default. To activate: set the config flag when training. No code change needed, just a config update in the training config.

3. **Per-entity beliefs** — The `BeliefState` dataclass has an `entity_id` field but the SQL beliefs table doesn't persist it. Currently all beliefs are global (per-variable). When entity-level beliefs are needed, the schema needs a migration.

---

## Parallel Active Work Streams

This self-improving architecture work (Phase 24-derived) runs in parallel with other active tasks:

| Task | Status | File |
|------|--------|------|
| Self-Improving Architecture (Tier 1+2) | **COMPLETE** ✅ | [[learned_architecture_impl]] |
| Phase 25: Cross-Domain Entity Linking | Active, step 25.1 not started | [[phase25_cross_domain_entity_linking]] |
| GNN-Guided Expansion R2 | Active, paused | [[gnn_guided_expansion_r2]] |
| L2 Tool Expansion | Active, paused | [[l2_tool_expansion]] |
| Quant Training Ground | Active, paused | [[quant_training_ground]] |

The **recommended next work** depends on priority:

- **If maximizing self-improvement trajectory:** Start Tier 3 (Changes 5, 7, 8). This continues the "make TirraMind learn its own parameters" thread. Highest impact is probably Change 5 (reward weight optimization) because it closes the outer loop that determines *what the agent values*.

- **If maximizing data surface:** Resume Phase 25 step 25.1 (cross-domain entity linking). This expands the surveillance surface — more entities, more links, more signal.

- **If maximizing training quality:** Resume the quant training ground work. More backtesting infrastructure.

The self-improving architecture and entity linking are largely independent code paths. They can be worked on in alternating sessions without interference.

---

## How to Cold-Start the Next Session

1. Read **this checkpoint**.
2. Read [[learned_architecture_impl]] (the task file) — it's the canonical progress tracker. All 14 steps are checked.
3. If continuing self-improving work → read [[learned_vs_handcoded_architecture_spec]] starting from "Change 5: Learn Reward Function Weights" for the Tier 3 spec.
4. If switching to entity linking → read [[phase25_cross_domain_entity_linking]] and [[phase25_cross_domain_entity_linking_spec]].
5. Look at `agent/pipeline/dags/world_model_update.py` to see the fitting integration pattern — it's a good template for future periodic-fitting DAG steps.
6. **Do NOT re-audit the codebase.** The [[learned_vs_handcoded_audit]] is current as of today. Don't re-derive what's learned vs hand-coded. Trust the audit and spec.

---

## Mathematical Methods Used (Quick Reference for Continuity)

| Method | Reference | Where | Parameters |
|--------|-----------|-------|-----------|
| BDeu Bayesian estimation | Heckerman et al. 1995 | `world_model.py:fit_cpds()` | ESS=10, min_samples=50 |
| Shumway-Stoffer EM for linear Gaussian SSM | Shumway & Stoffer 2017, Ch. 6 | `state_filter.py:fit_filter_params()` | max_iter=20, tol=1e-4, min_samples=30 |
| RTS smoother (backward pass) | Rauch, Tung, Striebel 1965 | Inside `fit_filter_params()` E-step | — |
| Exponentiated Gradient on simplex | Kivinen & Warmuth 1997 | `surprise.py:AdaptiveSurpriseWeights` | lr=0.01, min_weight=0.02 |
| Kendall uncertainty-weighted multi-task loss | Kendall, Gal & Cipolla 2018 | `trainer.py` (GNN) | log_var parameterization |
| PSD enforcement via eigenvalue clamping | Standard numerical linear algebra | `fit_filter_params()` M-step | ε=1e-8 |

---

## Store/DB Schema Notes for Future Sessions

- **Fit markers:** Stored in `pipeline_data` table with `source="world_model_fit"`. Params contain `as_of` (logical time of fit), `cpd_fitted`, `kalman_fitted`, `n_snapshots`. Data contains full `cpd_result` and `kalman_result` dicts from the fitting methods.
- **Features:** `features` table, queried by `feature_name` + `effective_at` range. Unique on `(feature_name, version, effective_at)`.
- **Beliefs:** `beliefs` table, queried by `variable_name` + `effective_at` range. Has `probabilities_json` for categorical beliefs (used for regime labels).
- **Feature validation:** `store_features_batch()` and `store_feature()` validate `source_signals` must be non-empty. Test fixtures need `source_signals=("test_signal",)` not `()`.

---

## Related

- [[learned_architecture_impl]] — Active task file (all steps complete)
- [[learned_vs_handcoded_audit]] — Research: full layer-by-layer audit
- [[learned_vs_handcoded_architecture_spec]] — Spec: 10 changes across 5 tiers
- [[chat_checkpoint_2026-04-14_learned_architecture]] — Session 1 checkpoint (more detailed per-change implementation notes)
- [[phase25_cross_domain_entity_linking]] — Parallel active task
- [[project_memory]] — Persistent architectural memory
- [[e2e_global_integration]] — Phase 24 context
- [[rl_policy]] — SAC implementation (Phase 21)
- [[world_model_bridge]] — GNN → world model bridge (Phase 19)
- [[signal_fusion]] — Surprise extraction (Phase 20)
