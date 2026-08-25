---
title: "Checkpoint: Master Session — Tier 6 Complete, Full Self-Improving Architecture Status"
tags:
  - doc/checkpoint
  - phase/25
  - topic/self-improving
  - topic/feature-selection
  - topic/tool-routing
  - layer/learning
  - layer/surveillance
  - layer/fusion
---

# Checkpoint: 2026-04-15 — Tier 6 Complete

**Date**: 2026-04-15
**Previous checkpoint**: [[chat_checkpoint_2026-04-14_learned_architecture]]
**Master spec**: [[learned_vs_handcoded_architecture_spec]]
**Active track**: Self-Improving Architecture (Phase 25)

---

## Session Summary

This session completed the remaining work for Tier 6 (Changes 11 + 12) of the self-improving architecture. The system has moved from **75% → 82% learned**.

### Work Done This Session

1. **Created `tests/test_tool_router.py`** — 25 edge case tests for `ToolRoutingBandit` (Change 12):
   - 10 test classes: ColdStart(3), Convergence(3), AlwaysOn(2), MinExploration(2), Persistence(4), ErrorHandling(2), AddTool(2), DAGIntegration(4), Stats(3)
   - Covers: cold start with uniform priors, convergence to high/low-reward arms, always-on enforcement, min exploration rate, JSON persistence with corruption/missing-tool handling, unknown tool errors, dynamic tool addition, full DAG integration (router → node.enabled → executor skip), diagnostics

2. **Wired FeatureGate into `rl_training.py` (step 11.4)**:
   - In `_train_sac()`: when `PolicyConfig.feature_gate` is set, creates `FeatureGate` and attaches via `encoder.set_feature_gate(gate)`
   - Sets regime context (zero vector for cold start; future: load HMM posterior from store)
   - Added `FeatureGate, FeatureGateConfig` imports to rl_training.py

3. **Modified `SACTrainer` for gate integration**:
   - Added `set_regime_context(tensor)` method — stores regime context for encoder
   - Modified `_encode_state()` to pass `regime_context` to encoder's `forward()`
   - Added gate entropy loss to actor loss: `if encoder.feature_gate is not None: actor_loss += gate.entropy_loss()`
   - Added `gate_entropy` to returned metrics dict
   - Gate params auto-included in actor optimizer (already registered as encoder submodule)

4. **Fixed `test_metric_keys` in `tests/test_sac.py`** — added `gate_entropy` to expected keys

5. **Regression verified**: 156 tests pass across feature_gate(36), tool_router(25), sac(27), learned_architecture(37), rl_edge_cases(32). Full architecture suite: 527/528 pass (1 flaky: `test_cls_token_updates` — MHA non-determinism in parallel runs, passes in isolation).

6. **Updated task file** `[[tier6_learned_observation]]` — all checkboxes marked, status set to `completed`, tag changed to `status/done`.
7. **Cleaned task directories** — moved 10 completed task files from `tasks/active/` to `tasks/done/`; only genuinely unfinished tasks remain active.

---

## Complete Self-Improving Architecture Progress

### Tier-by-Tier Status

| Tier | Changes | Status | Tests | % Learned |
|------|---------|--------|-------|-----------|
| 1 | Change 1: Wire beliefs → SAC | **DONE** | Part of 37 | 28% |
| 2 | Changes 2a, 2b, 4, 9: CPD learning, Kalman EM, adaptive surprise, GNN loss auto-tune | **DONE** | 37 + 36 | 45% |
| 3 | Changes 3, 5: Learned reward shaping, detector self-calibration | **DONE** | Tier 3 tests | 55% |
| 4 | Changes 6, 7, 8: Learned state encoder, DAG structure learning | **DONE** | Encoder + DAG tests | 65% |
| 5 | Change 10c + Phase A + Phase B: DiffKalman, SAC pipeline fix, differentiable belief bypass | **DONE** | 66 + 48 + 44 | 75% |
| **6** | **Changes 11, 12: Feature gate, tool routing** | **DONE** | **36 + 25** | **82%** |
| 7 | Changes 13, 14: Self-modifying graph, meta-learned scheduling | Not started | — | 90% |
| 8 | Changes 15, 16: Autonomous data discovery, self-extending ontology | Not started | — | 95% |

### Total Test Count (Self-Improving Architecture Only)
- **528 tests** across 13 test files
- 527/528 pass consistently (1 flaky MHA test)

---

## File Inventory — All Uncommitted Changes

### New Files (Untracked) — 17 files
| File | Purpose | Tests |
|------|---------|-------|
| `agent/learning/policy/feature_gate.py` | FeatureGate nn.Module (Change 11) | 36 |
| `agent/learning/policy/state_encoder.py` | LearnedStateEncoder MHA (Change 6) | ~30 |
| `agent/learning/tool_router.py` | ToolRoutingBandit Thompson Sampling (Change 12) | 25 |
| `agent/models/diff_kalman.py` | DifferentiableKalmanFilter nn.Module (Change 10c) | 66 |
| `agent/learning/param_optimizer.py` | Parameter/threshold optimization (Changes 3, 5) | — |
| `agent/learning/threshold_optimizer.py` | Detector self-calibration (Change 5) | — |
| `tests/test_feature_gate.py` | 36 tests for FeatureGate | — |
| `tests/test_tool_router.py` | 25 tests for ToolRoutingBandit | — |
| `tests/test_learned_architecture.py` | 37 tests for Tier 1+2 changes | — |
| `tests/test_tier3_meta_params.py` | Tier 3 meta-param tests | — |
| `tests/test_tier3_integration.py` | Tier 3 integration tests | — |
| `tests/test_tier4_state_encoder.py` | ~30 tests for LearnedStateEncoder | — |
| `tests/test_tier4_dag_structure.py` | DAG structure learning tests | — |
| `tests/test_tier5_diff_kalman.py` | 66 tests for DiffKalman | — |
| `tests/test_differentiable_bypass.py` | 44 tests for Phase B bypass | — |
| `tests/test_training_pipeline_fix.py` | 48 tests for Phase A SAC fix | — |
| `tests/test_world_model_update_fitting.py` | 36 tests for CPD/Kalman EM fitting | — |

### Modified Files (Tracked) — 22 files, +2610/-120 lines
| File | Changes |
|------|---------|
| `agent/learning/policy/config.py` | +FeatureGateConfig, +SACConfig.aux_kalman_weight, +StateEncoderConfig, +WeightLearnerConfig |
| `agent/learning/policy/sac.py` | +set_regime_context, +gate entropy in actor loss, +gate_entropy metric, +_encode_state regime routing, +SACTrainer.load, encoder integration |
| `agent/learning/policy/state_assembler.py` | +DifferentiableStateAssembler (Phase B) |
| `agent/pipeline/dags/rl_training.py` | +FeatureGate creation, +Kalman augmentation (Phase B), +regime context, +SAC pipeline fixes (Phase A) |
| `agent/pipeline/dags/daily_collection.py` | +tool_router integration, +TYPE_CHECKING imports |
| `agent/pipeline/dag.py` | +enabled: bool = True to Node |
| `agent/pipeline/executor.py` | +skip disabled nodes ("Skipped: disabled by tool router") |
| `agent/pipeline/dags/world_model_update.py` | +CPD fitting, +Kalman EM fitting, +DiffKalman backend option |
| `agent/pipeline/dags/inference.py` | +belief→policy wiring (Change 1), +pending transitions |
| `agent/pipeline/dags/entity_scoring.py` | +entity scoring enhancements |
| `agent/pipeline/store.py` | +query_all_latest_beliefs, +pending transitions, +rl_checkpoint methods |
| `agent/models/world_model.py` | +fit_cpds(), +Union[ContinuousStateFilter, DiffKalman], +_regime_configs |
| `agent/models/state_filter.py` | +fit_filter_params() Shumway-Stoffer EM |
| `agent/models/gnn/trainer.py` | +learnable log-variance loss weights (Kendall et al. 2018) |
| `agent/models/graph.py` | +graph structure additions |
| `agent/fusion/surprise.py` | +AdaptiveSurpriseWeights EG on simplex |
| `agent/learning/bandit.py` | +expanded bandit features |
| `agent/learning/reward.py` | +learned reward shaping |
| `agent/learning/goal_generator.py` | +goal generation enhancements |
| `agent/core/autonomous.py` | +autonomous loop enhancements |
| `agent/pipeline/dags/convergence_detection.py` | +convergence detection enhancements |
| `tests/test_sac.py` | +gate_entropy to expected metric keys |

---

## Key Architecture Concepts Implemented

### Change 11: FeatureGate (Regime-Conditioned Soft Gating)
- **Location**: `agent/learning/policy/feature_gate.py`
- **Math**: $g_k = (1-f) \cdot \sigma(\text{MLP}(r))_k + f$ where $r$ is regime context, $f$ = gate floor
- **Groups**: 5 groups matching state layout: surprise(250), belief(200), market(8), entity_count(1), adversarial(4) = 463 total
- **Config**: `FeatureGateConfig(n_feature_groups=5, regime_dim=4, gate_hidden_dim=16, gate_floor=0.05, entropy_weight=0.01)`
- **Integration path**: `rl_training._train_sac()` → creates gate → `encoder.set_feature_gate(gate)` → `sac.set_regime_context(zeros)` → gate auto-applied in `encoder.forward(state, regime_context)` → entropy loss added to actor loss
- **Gate params in actor optimizer**: Yes, via `add_module("feat_gate", gate)` on encoder

### Change 12: ToolRoutingBandit (Contextual Thompson Sampling)
- **Location**: `agent/learning/tool_router.py`
- **Math**: Per-tool Beta(α,β) posterior. Sample θ~Beta(α,β), run tool if θ > threshold.
- **Always-on**: `fetch_instruments` (never skipped)
- **Optional tools**: fetch_cftc, fetch_finra_scan, fetch_power_demand, fetch_power_fuel, fetch_gdelt, fetch_polymarket
- **Integration path**: `build_daily_collection_dag(tool_router=r, tool_context=ctx)` → `r.decide(ctx)` → sets `node.enabled` → executor skips disabled nodes
- **Persistence**: JSON to `.tirra_pipeline/tool_router.json`
- **Cold start**: Beta(1,1) uniform prior → all tools run → equivalent to current static schedule

### SACTrainer Enhancement
- `set_regime_context(tensor)` — stores regime context, passed through `_encode_state()` to encoder
- `_encode_state` now calls `self._encoder(state, regime_context=self._regime_context)`
- Actor update: checks `encoder.feature_gate`, adds `gate.entropy_loss()` to actor loss
- Returns `gate_entropy` in metrics dict

---

## Known Issues

1. **Flaky test**: `test_tier4_state_encoder.py::TestGradientFlow::test_cls_token_updates` — MHA non-determinism causes CLS token to occasionally get zero gradient when run in full suite. Passes in isolation. Pre-existing, not caused by Tier 6 changes.

2. **Uncommitted work**: All changes since last commit (`1c3ae41 cross domain entity linked`) are uncommitted. This spans Tiers 1-6 (all 12 changes + Phase A + Phase B). Recommend committing after verifying full test suite.

3. **Task directory state**: Cleaned. The following completed files were normalized and moved to `tasks/done/`: `differentiable_belief_bypass.md`, `fix_sac_training_pipeline.md`, `gnn_guided_expansion_r2.md`, `l2_tool_expansion.md`, `learned_architecture_impl.md`, `tier3_learn_meta_params.md`, `tier4_learn_dag_structure.md`, `tier4_learned_state_encoder.md`, `tier5_differentiable_kalman.md`, `tier6_learned_observation.md`.

   Remaining active tasks are now limited to:
   - `phase25_cross_domain_entity_linking.md`
   - `predictive_platform_positioning_task.md`
   - `quant_training_ground.md`

4. **Regime context cold start**: Currently uses `torch.zeros(regime_dim)` as regime context in `_train_sac()`. Future: load actual HMM posterior from store when beliefs are available.

---

## What's Next: Tier 7 (Changes 13 + 14)

Per [[learned_vs_handcoded_architecture_spec]], Tier 7 targets **90% learned**:

### Change 13: Self-Modifying Graph Schema
- Currently: Expert causal DAG with 20 fixed nodes and 19 fixed edges (structure is hand-coded, only CPD values learned via Tier 2)
- Target: Structure learning on live data adds/removes/merges nodes in the causal DAG; graph topology becomes a learned object
- This requires: pgmpy structure learning (HC, PC, or MMHC algorithm), edge scoring, graph edit operations, validation that learned structure maintains key invariants

### Change 14: Meta-Learned Scheduling
- Currently: Fixed intervals for all pipeline operations (daily collection at 18:00, weekly CPD refit, etc.)
- Target: System learns when to re-fit parameters, how much history to use, which components need retraining
- This requires: Scheduling policy (likely bandit-based like Change 12 but over fit/retrain actions), staleness metrics, data sufficiency detection

### Recommended Approach
1. Research phase: read pgmpy structure learning docs, survey meta-learning scheduling papers
2. Spec: define what "learned structure" means operationally (what can change, what invariants remain)
3. Implement Change 13 first (higher impact — moves from learned-params to learned-structure)
4. Then Change 14 (scheduling is lower risk and partially covered by tool routing bandit pattern)

---

## Quick Resume Instructions

1. Read this checkpoint
2. Read [[learned_vs_handcoded_architecture_spec]] lines 340-380 for Tier 7+8 description
3. Confirm test suite passes: `python -m pytest tests/test_feature_gate.py tests/test_tool_router.py tests/test_sac.py tests/test_learned_architecture.py tests/test_rl_edge_cases.py --tb=short -q`
7. Begin research phase for Tier 7 in `[[tier7_self_modifying_structure]]`

---

## Related

- [[tier6_learned_observation]]
- [[tier6_learned_observation_spec]]
- [[tier6_learned_observation|Task: Tier 6]]
- [[learned_vs_handcoded_architecture_spec]]
- [[chat_checkpoint_2026-04-14_learned_architecture]]
- [[chat_checkpoint_2026-04-14_learned_architecture]]
- [[chat_checkpoint_2026-04-14_tier1_tier2_complete]]
- [[learned_architecture_impl]]
- [[differentiable_belief_bypass]]
- [[fix_sac_training_pipeline]]
- [[tier5_differentiable_kalman]]
- [[tier4_learned_state_encoder]]
- [[tier4_learn_dag_structure]]
- [[tier3_learn_meta_params]]
- [[project_memory]]
