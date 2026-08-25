---
title: "Spec: Tier 7 — Self-Modifying Structure"
tags:
  - doc/spec
  - phase/25
  - topic/self-improving
  - topic/world-model
  - topic/meta-learning
  - layer/world-model
  - layer/learning
---

# Spec: Tier 7 — Self-Modifying Structure

**Date:** 2026-04-15
**Research:** [[tier7_self_modifying_structure]]
**Master Spec:** [[learned_vs_handcoded_architecture_spec]]
**Goal:** Implement Changes 13 + 14. Move from 82% → 90% learned.

---

## Goal

1. **Change 13 — Self-modifying graph**: Track edge confidence via BIC-δ scores on rolling windows. Use hysteresis-based decisions to add/remove edges. Version the DAG and mark beliefs stale on structure changes.
2. **Change 14 — Meta-learned scheduling**: Replace all fixed refit intervals with per-component Thompson Sampling bandits that learn optimal intervals from information-gain rewards.

---

## Files Affected

### New Files
| File | Purpose |
|------|---------|
| `agent/models/edge_tracker.py` | EdgeConfidenceTracker — BIC-δ scoring, rolling windows, hysteresis |
| `agent/learning/meta_scheduler.py` | MetaScheduler — per-component Thompson Sampling over interval arms |
| `tests/test_tier7_edge_tracker.py` | Edge confidence tracker tests |
| `tests/test_tier7_meta_scheduler.py` | Meta-scheduler tests |

### Modified Files
| File | Change |
|------|--------|
| `agent/models/world_model.py` | Add `edge_confidence_scores()` method, DAG version hash |
| `agent/pipeline/dags/world_model_update.py` | Wire EdgeConfidenceTracker into `_maybe_refine_structure()`, replace fixed intervals with MetaScheduler |
| `agent/pipeline/store.py` | Add `edge_confidence_history` and `component_performance` storage helpers |
| `agent/pipeline/dags/gnn_inference.py` | Accept dynamic epochs from MetaScheduler |
| `agent/pipeline/dags/rl_training.py` | Accept dynamic SAC training interval from MetaScheduler |

---

## Implementation Steps

### Change 13: Self-Modifying Graph (Steps 13.1–13.6)

#### 13.1: Create EdgeConfidenceTracker with BIC-δ scoring

**File:** `agent/models/edge_tracker.py` (new)

Create `EdgeConfidenceTracker` class:
- `__init__(node_specs, windows=(30, 60, 90))` — stores windows (in days) for rolling BIC evaluation
- `compute_edge_contributions(edges, feature_df, scoring_method)` → dict mapping `(parent, child)` → `float` BIC-δ
  - For each edge: `delta = scorer.local_score(child, parents_with) - scorer.local_score(child, parents_without)`
  - Uses pgmpy `structure_score.BIC` with `local_score(variable, parents)` API
- `evaluate(edges, feature_history, as_of)` → dict mapping edge → `EdgeConfidence(confidence, stability, n_windows)`
  - Computes BIC-δ on each rolling window
  - `confidence = sigmoid(-mean(deltas))` — high when edge helps fit
  - `stability = 1 - std(deltas) / (|mean(deltas)| + eps)` — high when consistent across windows
- Pure computation, no DB access, no side effects.

**Test:** BIC-δ is negative for true causal edges, positive for spurious edges on synthetic data.

#### 13.2: Add hysteresis decision logic

**File:** `agent/models/edge_tracker.py`

Add `suggest_changes(edge_confidences, current_edges, *, add_threshold=0.7, remove_threshold=0.3, stability_min=0.5, consecutive_required=2)`:
- Maintains internal `_consecutive_counts: dict[edge, int]` tracking how many consecutive evaluations an edge has been above/below threshold.
- Returns `EdgeSuggestion(edges_to_add, edges_to_remove)` — only edges meeting both confidence AND consecutive criteria.
- Respects structural constraints: never suggest adding edge that creates a cycle, never suggest removing expert-anchored edges (regime→observed).

**Test:** Hysteresis prevents flip-flopping: edge oscillating around threshold is not modified until it stabilizes.

#### 13.3: Add DAG versioning to WorldModel

**File:** `agent/models/world_model.py`

- Add `dag_version` property → SHA256 of sorted `(parent, child)` edge tuples. Deterministic hash.
- Add `_dag_version_history: list[tuple[float, str]]` — `(timestamp, hash)` pairs.
- On any `refine_structure()` call that changes edges, append to version history.

**Test:** Hash changes when edges change, remains stable when they don't.

#### 13.4: Wire EdgeConfidenceTracker into world_model_update DAG

**File:** `agent/pipeline/dags/world_model_update.py`

Modify `_maybe_refine_structure()`:
1. After loading feature history, create `EdgeConfidenceTracker` and call `evaluate()` on current edges.
2. Call `suggest_changes()` with the confidences.
3. If suggestions exist, apply them to the graph (using existing `add_edge`/`remove_edge`).
4. Store edge confidence scores in the pipeline store (via new helper).
5. If structure changed, log the new `dag_version`.

The existing HillClimbSearch in `refine_structure()` remains as the primary structure learning mechanism. The EdgeConfidenceTracker acts as a gating/validation layer: HillClimb proposes changes, EdgeConfidenceTracker must confirm them via hysteresis before they're applied.

**Test:** Integration test: synthetic data with known structure → tracker converges to correct edges over multiple evaluation cycles.

#### 13.5: Store edge confidence history in PipelineStore

**File:** `agent/pipeline/store.py`

Add helper methods:
- `store_edge_confidences(as_of, dag_version, confidences: dict)` — stores via `store_data()` with source=`"edge_confidence"`.
- `query_edge_confidence_history(edge, limit)` → list of `(as_of, confidence, stability)` tuples.

**Test:** Round-trip store/query with edge confidence data.

#### 13.6: Mark beliefs stale on structure change

**File:** `agent/pipeline/dags/world_model_update.py`

After a structure change is applied:
- Call `store.mark_beliefs_stale(reason="structure_change", dag_version=new_hash)`.
- This forces the next world model update cycle to re-run full inference rather than incremental update.

Use existing `PipelineStore` stale-marking capability (the `stale` column on beliefs table already exists per the checkpoint).

**Test:** After structure change, beliefs are marked stale; next update re-infers.

---

### Change 14: Meta-Learned Scheduling (Steps 14.1–14.5)

#### 14.1: Create MetaScheduler with per-component bandits

**File:** `agent/learning/meta_scheduler.py` (new)

Create `MetaScheduler` class:
- `__init__(components, persist_path)` where `components` is a dict like:
  ```python
  {
      "cpd_fit": {"arms": [3, 5, 7, 14, 30], "default": 7},
      "structure_refine": {"arms": [30, 60, 90, 180], "default": 90},
      "gnn_epochs": {"arms": [5, 10, 20, 40], "default": 10},
      "history_window": {"arms": [30, 60, 90, 180], "default": 90},
  }
  ```
- Each component gets a Thompson Sampling bandit with Beta(α=1, β=1) per arm (uniform prior).
- `suggest(component) → int` — Thompson sample from each arm's Beta posterior, return the arm with highest sample.
- `record_outcome(component, arm, reward: float)` — update Beta posterior. Reward in [0, 1].
- `save()` / `load()` — JSON persistence (same pattern as `ToolRoutingBandit`).
- `diagnostics() → dict` — per-component arm statistics (pulls, mean reward, posterior params).

**Test:** With fixed rewards, bandit converges to best arm. Cold start returns default arm with high probability.

#### 14.2: Define reward functions for each component

**File:** `agent/learning/meta_scheduler.py`

Add `compute_refit_reward(component, before_metrics, after_metrics) → float`:
- **cpd_fit**: `reward = sigmoid(ΔBIC)` where ΔBIC = `sum(local_scores_after) - sum(local_scores_before)`. Positive BIC improvement → reward > 0.5.
- **structure_refine**: `reward = sigmoid(n_confident_changes)` where confident changes = edges crossing confidence threshold with stability. More meaningful changes → higher reward.
- **gnn_epochs**: `reward = sigmoid(-Δval_loss)` — validation loss decreased → good.
- **history_window**: `reward = sigmoid(ΔBIC)` — same as cpd_fit but evaluated on a held-out week of data.

All rewards are [0, 1] via sigmoid. Conservative: the sigmoid is centered so that "no change" gives reward ≈ 0.5, improvements > 0.5, degradations < 0.5.

**Test:** Known improvements yield reward > 0.5; degradations < 0.5; no change ≈ 0.5.

#### 14.3: Wire MetaScheduler into world_model_update

**File:** `agent/pipeline/dags/world_model_update.py`

Modify `run_world_model_update()`:
1. Load or create MetaScheduler (persisted at `{db_dir}/meta_scheduler.json`).
2. Replace `fit_interval_days` with `scheduler.suggest("cpd_fit")`.
3. Replace `structure_fit_interval_days` with `scheduler.suggest("structure_refine")`.
4. Replace `history_window_days` with `scheduler.suggest("history_window")`.
5. After CPD fit, compute reward and call `scheduler.record_outcome("cpd_fit", arm, reward)`.
6. After structure refine, compute reward and record.
7. Save scheduler state.

The `params` dict passed to `run_world_model_update()` still accepts explicit overrides (for backtesting), but defaults change from hardcoded to scheduler-suggested.

**Test:** Integration test: scheduler suggestions are used in place of hardcoded values; outcomes are recorded.

#### 14.4: Wire MetaScheduler into GNN training

**File:** `agent/pipeline/dags/gnn_inference.py`

- Load MetaScheduler.
- Use `scheduler.suggest("gnn_epochs")` instead of `TrainerConfig.epochs = 10`.
- After training, compute reward from validation loss improvement and record.
- Persist scheduler state.

**Test:** GNN trainer uses scheduler-suggested epochs; different arms produce different training lengths.

#### 14.5: Store component performance history

**File:** `agent/pipeline/store.py`

Add helpers:
- `store_component_performance(component, as_of, arm, reward, metrics)` — stores via `store_data()` with source=`"component_perf_{component}"`.
- `query_component_history(component, limit)` → recent performance records.

**Test:** Round-trip store/query for component performance data.

---

## Edge Cases

1. **No feature history available**: EdgeConfidenceTracker returns empty confidences, no changes suggested.
2. **All edges low confidence on first run**: Hysteresis prevents removal (need 2+ consecutive evaluations).
3. **MetaScheduler missing persist file**: Falls back to default intervals (current hardcoded values).
4. **Scheduler suggests very short interval**: Arms are bounded (minimum 3 days for CPD). Cannot suggest 0.
5. **Structure change + CPD fit in same cycle**: Structure change runs first, CPDs are invalidated, CPD fit runs on new structure.
6. **GNN epochs arm = 5 produces underfitting**: Low validation improvement → low reward → arm is downweighted over time.

## Testing Plan

- **`tests/test_tier7_edge_tracker.py`**: ~35 tests covering BIC-δ computation, confidence/stability math, hysteresis decisions, structural constraint enforcement, rolling window edge cases, serialization.
- **`tests/test_tier7_meta_scheduler.py`**: ~30 tests covering Thompson Sampling convergence, reward computation, persistence, cold start defaults, explicit overrides, diagnostics, integration with world_model_update parameters.

---

## Related

- [[tier7_self_modifying_structure]] — Research
- [[learned_vs_handcoded_architecture_spec]] — Master spec
- [[learned_vs_handcoded_audit]] — Original audit
- [[chat_checkpoint_2026-04-15_tier6_complete]] — Previous checkpoint
