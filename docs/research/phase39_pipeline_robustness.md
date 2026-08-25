---
title: "Feature: Phase 39 — Pipeline Robustness"
tags:
  - doc/research
  - phase/39
  - topic/pipeline
  - layer/feature-engineering
  - layer/world-model
---

# Feature: Phase 39 — Pipeline Robustness

## Goal

Phase 37/38 proved the pipeline runs end-to-end with real data but revealed three gaps that prevent non-trivial feature output:

1. **GNN OOB crash on entity growth.** The HetTGN memory buffer is sized to the checkpoint's `num_nodes` (918). After fresh collection the DB grew to 929 entities → `IndexError: index 918 is out of bounds for dimension 0 with size 918`. The model must tolerate entity count changes between training and inference.
2. **fetch_macro fails without FRED API key.** `MacroStateFeatureBuilder` produces all-None features because the macro_data tool raises on missing `TIRRA_FRED_API_KEY`. Need graceful degradation (skip macro features when no key) plus documentation for key setup.
3. **Convergence features empty (expected).** With only 118 evidence items from 5 sources, the detector correctly finds no statistically significant convergence. This is not a bug — features will populate as data accumulates over daily runs. However, the `ConvergenceFeatureBuilder` should not emit `missing` features when evidence exists but no convergence was detected; it should emit zero-valued features to distinguish "no data" from "data present, no convergence."

## Current Architecture

### GNN Inference Path

- `agent/features/gnn_builder.py:196` calls `trainer.infer(until=as_of)`
- `agent/models/gnn/trainer.py:1082` calls `model(data, id_map)` where `data` comes from current DB
- `agent/models/gnn/het_tgn.py:427` calls `self.memory.get_memory(global_ids)` where `global_ids` can exceed buffer size
- Memory buffer allocated in `HeteroMemory.__init__` (line 159): `torch.zeros(num_nodes, memory_dim)` — fixed size, never resized
- `load_model()` (trainer.py:1175) uses `checkpoint["num_nodes"]` to size the model — stale count

### Feature Generation

- `ConvergenceFeatureBuilder.build()` queries signals table → 0 signals → returns features with `value=None`
- `MacroStateFeatureBuilder.build()` queries pipeline_data for `macro_data` source → 0 rows → returns `value=None`
- `GNNFeatureBuilder.build()` calls trainer.infer() → crashes → fallback returns `value=None`

## Observations

- Evidence extraction works (118 items from 5 sources) — Phase 38 fix confirmed
- Convergence detector correctly returns 0 when evidence is sparse — not a bug
- GNN OOB is a straightforward buffer-sizing problem, not architectural
- FRED API key is a configuration gap, not a code bug
- All 17 features currently show `missing=True` → no feature has ever had a real value in production

## Risks

- **GNN weight transfer:** When resizing memory buffer, old weights (918 rows) must be preserved, new rows (919-928) should be zero-initialized. A simple `resize_` or copy is sufficient since memory is per-entity state, not learned weights.
- **Convergence false precision:** Emitting `value=0.0` when evidence exists but no convergence was detected could be misleading if downstream consumers interpret 0.0 differently from None. Document the semantics.
- **FRED rate limits:** Free FRED API key has 120 requests/min. Not a concern for daily pipeline.

## Fix Analysis

### GNN OOB Fix

**Option A (simplest):** In `HeteroMemory.get_memory()`, clamp or expand the buffer dynamically when `max(node_ids) >= self.memory.size(0)`. Expand with zero rows.

**Option B (cleanest):** In `trainer.load_model()`, after loading checkpoint, compare `checkpoint["num_nodes"]` with current graph `num_nodes`. If current > checkpoint, resize the memory buffer.

**Option C (recommended):** In `trainer.infer()`, after building the graph, check `id_map.num_nodes > model.memory.num_nodes` and call a `resize_memory(new_count)` method. This catches every case at the point of use.

Recommendation: **Option C** — add `HeteroMemory.resize(new_num_nodes)` method that preserves existing memory rows and adds zero-initialized rows. Call it in `infer()` after graph build.

### FRED API Key

Add graceful degradation: `MacroStateFeatureBuilder.build()` should catch the missing-key error and return empty features instead of propagating. Document key setup in `.env.example`.

### Convergence Feature Semantics

`ConvergenceFeatureBuilder` should return `value=0.0` (not `None`) when evidence exists but detector found no convergence. `None` should mean "no data available at all."

## Related

- [[phase39_pipeline_robustness_spec]]
- [[phase39_pipeline_robustness]]
- [[phase38_downstream_pipeline_integration]]
- [[phase37_first_live_pipeline]]
- [[quant_training_ground]]
