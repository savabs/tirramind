---
title: "Spec: Phase 39 — Pipeline Robustness"
tags:
  - doc/spec
  - phase/39
  - topic/pipeline
  - layer/feature-engineering
  - layer/world-model
---

# Spec: Phase 39 — Pipeline Robustness

## Goal

Make the pipeline produce real (non-None) feature values on every run, even when entity count grows between GNN training and inference, FRED API key is missing, or convergence evidence is sparse.

## Files Affected

| File | Change |
|------|--------|
| `agent/models/gnn/het_tgn.py` | Add `HeteroMemory.resize(new_num_nodes)` method |
| `agent/models/gnn/trainer.py` | Call `resize()` in `infer()` after graph build when entity count exceeds buffer |
| `agent/features/gnn_builder.py` | No change (already has try/except fallback) |
| `agent/features/builders.py` | `ConvergenceFeatureBuilder`: return `value=0.0` when evidence exists but no convergence; `MacroStateFeatureBuilder`: graceful skip when FRED key missing |
| `tests/test_phase39_pipeline_robustness.py` | New — tests for all three fixes |

## Implementation Steps

### 39.1: HeteroMemory.resize() method

Add `resize(new_num_nodes: int)` to `HeteroMemory`:
- If `new_num_nodes <= self.num_nodes`: no-op
- If `new_num_nodes > self.num_nodes`: create new buffers of size `new_num_nodes`, copy old data, register new buffers
- Update `self.num_nodes`

### 39.2: Trainer.infer() resize call

In `trainer.infer()`, after `data, id_map, _ = self._graph_builder.build(until=until)`:
- Compute `current_num_nodes` from the built graph
- If `current_num_nodes > model.memory.num_nodes`: call `model.memory.resize(current_num_nodes)`
- Log a warning about entity count growth

### 39.3: MacroStateFeatureBuilder graceful degradation

In `MacroStateFeatureBuilder.build()`:
- Catch the FRED API key error when querying pipeline_data
- Return empty list (no features) instead of features with `value=None`
- Log info-level message about missing macro data

### 39.4: ConvergenceFeatureBuilder zero-vs-None semantics

In `ConvergenceFeatureBuilder.build()`:
- When pipeline has evidence (signals table may be empty but pipeline_data has convergence-relevant sources): return `value=0.0` instead of `value=None`
- When no evidence exists at all: return `value=None` (or skip feature)
- Add docstring clarifying the semantics

### 39.5: Test suite

Write `tests/test_phase39_pipeline_robustness.py`:
- `TestHeteroMemoryResize`: resize up preserves old memory, new rows are zero, resize down is no-op
- `TestTrainerInferResize`: mock graph with more entities than checkpoint → no crash
- `TestMacroBuilderGraceful`: missing FRED key → empty features, no exception
- `TestConvergenceFeatureSemantics`: evidence present + 0 convergence → value=0.0; no evidence → value=None or empty
- `TestEndToEndFeatureGeneration`: mock pipeline with entity growth → all builders produce features without crash

### 39.6: Live pipeline re-run

Re-run `daily_collection → convergence_detection → feature_generation` and verify:
- GNN features produce real values (not None)
- Macro features gracefully skipped (or produce values if FRED key is set)
- Convergence features are 0.0 (not None)

## Edge Cases

- Entity count shrinks between runs (entities deleted) — resize should handle both directions
- GNN checkpoint file missing entirely — existing fallback in gnn_builder handles this
- FRED API key present but invalid (401 response) — should degrade same as missing key
- Extremely large entity growth (918 → 10000) — resize should still work efficiently

## Testing Plan

All tests must pass in isolation and as part of the full regression suite. Target: 15-25 new tests covering all three fixes plus integration.

## Related

- [[phase39_pipeline_robustness]]
- [[phase39_pipeline_robustness]]
- [[phase38_downstream_pipeline_integration]]
- [[quant_training_ground]]
