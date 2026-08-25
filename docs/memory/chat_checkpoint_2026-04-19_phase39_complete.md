---
title: "Checkpoint: Phase 39 — Pipeline Robustness Complete"
tags:
  - doc/checkpoint
  - phase/39
  - topic/pipeline
  - topic/gnn
  - topic/feature-engineering
  - layer/feature-engineering
  - layer/world-model
---

# Checkpoint: Phase 39 Complete (2026-04-19)

## Executive State

Phase 39 is complete in the implementation and validation sense.

The code changes for Phase 39 were implemented, the dedicated robustness suite passed, the relevant regression suites were updated to the new semantics and passed, and the live feature-generation path was re-run successfully against the real pipeline database.

The remaining work is administrative rather than functional:

- `[[quant_training_ground]]` is stale and still names Phase 38 as the latest completed phase.
- `[[phase39_pipeline_robustness]]` is marked `status/done` and `Status: completed`, but at the time of this checkpoint it still lives under `tasks/active/` rather than `tasks/done/`.
- This checkpoint file did not exist before this write, so the top-level tracker also still pointed at the Phase 38 checkpoint.

## Why Phase 39 Existed

Phase 38 fixed the downstream pipeline plumbing so real tool output flowed into convergence detection and feature generation. After that, a live verification pass exposed three robustness issues that were not wiring problems anymore.

### 1. GNN inference crashed after entity growth

The live graph had grown beyond the entity count represented in the saved HetTGN checkpoint memory buffer.

Observed failure mode:

- checkpoint memory size: 918 entities
- live graph size during inference: 929 entities
- result: out-of-bounds access in GNN memory lookup during feature generation

This was not a graph-builder bug and not a bad checkpoint file. The root cause was that inference assumed the memory buffer size remained aligned with the current entity graph indefinitely.

### 2. Macro features degraded poorly when no FRED-backed data existed

The live environment did not have macro data available, and the relevant missing configuration gap was the absent `TIRRA_FRED_API_KEY`.

The problem was not that macro logic computed the wrong values. The problem was that the builder produced missing-valued features in the complete absence of source data, which was the wrong semantic representation for “the pipeline never had macro rows to begin with.”

### 3. Convergence empty-output semantics were ambiguous

The convergence builder treated these two situations too similarly:

- there is no upstream pipeline data at all
- there is upstream pipeline data, but no convergence signals were detected

Those are materially different states.

For downstream consumers:

- “no data exists” should mean no convergence features are emitted
- “data exists but no convergence was found” should mean emit zero-valued convergence features

Phase 39 formalized that distinction.

## Phase 39 Research / Spec / Task Artifacts

The full triad exists and governed the implementation:

- Research: [[phase39_pipeline_robustness]]
- Spec: [[phase39_pipeline_robustness_spec]]
- Task: [[phase39_pipeline_robustness]]

The Phase 39 task file currently shows all six steps completed and includes the final validation notes.

## What Changed

### 39.1 — Dynamic HetTGN memory resizing

File changed:

- `agent/models/gnn/het_tgn.py`

Change:

- Added `HeteroMemory.resize(new_num_nodes)`.

Behavior:

- no-op when `new_num_nodes <= current_num_nodes`
- allocates larger memory and `last_update` buffers when the graph grows
- preserves existing rows
- zero-initializes newly added rows
- updates `self.num_nodes`

Why it matters:

- The live graph can expand between training time and inference time.
- Without this, any newly discovered entity whose node index exceeds the stored memory size can crash inference.

### 39.2 — Infer-time entity-growth guard

File changed:

- `agent/models/gnn/trainer.py`

Change inside `Trainer.infer()`:

- after `GraphBuilder.build(until=until)` returns `data` and `id_map`, compare `id_map.num_nodes` against `model.memory.num_nodes`
- if the live graph is larger, log a warning and resize the model memory before calling the model

Observed live log after the fix:

- `Entity count grew 918 → 929 since last training; resizing GNN memory buffer.`

Why this is the correct insertion point:

- graph size is only known after the current graph snapshot is built
- resizing here protects the production inference path without forcing retraining first

Important note:

- This file also has unrelated later edits beyond the core Phase 39 change. Future work in this area should re-read the current file before modifying it again.

### 39.3 — Macro builder graceful degradation

File changed:

- `agent/features/builders.py`

Change:

- `MacroStateFeatureBuilder.build()` now returns `[]` immediately when `store.query_data("macro_data")` yields no rows.

Semantic meaning:

- no macro source data in pipeline storage means no macro features should be emitted
- this is different from emitting features with `None` values and missing reasons

Why this matters:

- downstream counts and quality metrics now reflect absence of source data honestly
- macro output no longer pollutes feature storage with placeholder records when the pipeline never ingested macro rows

### 39.4 — Convergence builder zero-versus-empty semantics

File changed:

- `agent/features/builders.py`

Change:

- when no convergence rows are found, `ConvergenceFeatureBuilder.build()` now checks whether relevant upstream pipeline data exists from convergence-capable sources
- if upstream data exists, emit 3 zero-valued convergence features via `_zero_features(as_of)`
- if upstream data does not exist, return `[]`

New semantic contract:

- upstream data absent → `[]`
- upstream data present, no convergence detected → 3 zero-valued convergence features
- convergence detected → real-valued convergence features as before

The zero-valued convergence feature set is:

- `convergence.stress_breadth.7d = 0.0`
- `convergence.stress_intensity.7d = 0.0`
- `convergence.regime_persistence.7d = 0.0`

With:

- `quality = 1.0`
- `missing_reason = None`

Why this matters:

- “nothing is converging” is a valid observation, not missing data
- downstream consumers can distinguish quiet conditions from missing pipeline coverage

### 39.5 — Dedicated robustness suite

File created:

- `tests/test_phase39_pipeline_robustness.py`

Coverage areas:

- `HeteroMemory.resize()`
- infer-time resizing behavior
- macro graceful degradation with no FRED data
- convergence zero-vs-empty semantics
- integration sanity checks around feature generation behavior

Important correction made during implementation:

- the first version of the test used an incorrect `HetTGN` constructor shape
- this was fixed to use `metadata` and `in_channels`, matching the real model API

Important note:

- This test file may also have later edits beyond the initial Phase 39 change. Re-read before touching it in future work.

### 39.6 — Regression updates for intentional semantic changes

Files changed:

- `tests/test_feature_builders.py`
- `tests/test_phase38_pipeline_integration.py`

What changed:

- tests that expected missing-valued macro/convergence features under complete source-data absence were updated to expect `[]`
- the stale Phase 38 macro integration expectation was corrected to the new Phase 39 behavior

This was not papering over broken behavior. These were stale tests reflecting the old semantic contract.

## Test and Validation Timeline

### Dedicated Phase 39 suite

Initial run:

- result: 17 passed, 1 failed
- failure cause: incorrect `HetTGN` constructor usage inside the new test suite

After fixing the test:

- result: 18/18 passed

### Targeted regressions

There was one mistaken regression command using the wrong test filename first. After correcting that, the meaningful regression results were:

- `tests/test_feature_builders.py` surfaced 4 stale expectations tied to the old semantics
- those expectations were updated
- rerun result: 88 passed across the targeted regression set

### Additional stale regression uncovered

One more stale expectation remained in:

- `tests/test_phase38_pipeline_integration.py`

Specifically:

- `TestMacroFeatureBuilderIntegration::test_macro_builder_missing_when_no_data`

This was updated from “three missing-valued macro features” to `[]`.

After the fix, that targeted test passed.

## Live Pipeline Verification Results

After Phase 39 code changes and test corrections, the live feature-generation path was re-run successfully.

Observed runtime behavior:

- the GNN memory buffer resized automatically for the larger live graph
- feature generation completed rather than crashing

Observed live feature-generation summary:

- `produced: 14`
- `stored: 14`

Per-builder summary:

- `ConvergenceFeatureBuilder`: 3 produced, 0 missing
- `MacroStateFeatureBuilder`: 0 produced, 0 missing
- `GNNFeatureBuilder`: 11 produced, 6 missing

Interpretation:

- convergence features were emitted cleanly under the new semantics
- macro features were skipped entirely because there was no macro source data in the live pipeline state
- GNN features were produced successfully, and the previous inference crash was eliminated

Most important operational outcome:

- live feature generation no longer dies when the entity graph outgrows the checkpoint memory buffer

## Source Data Context Confirmed During This Work

Earlier live verification during this session confirmed that the pipeline was already ingesting real upstream evidence again after Phase 38, including data from:

- `cftc`
- `gdelt`
- `polymarket`
- `power_grid`
- `finra_short_volume`

Evidence extraction previously confirmed:

- total evidence items found: 118

This context matters because it proves Phase 39 was fixing robustness on a live, functioning downstream path, not on an empty or synthetic pipeline.

## Files Touched During Phase 39 Work

Primary implementation / validation files:

- `agent/models/gnn/het_tgn.py`
- `agent/models/gnn/trainer.py`
- `agent/features/builders.py`
- `tests/test_phase39_pipeline_robustness.py`
- `tests/test_feature_builders.py`
- `tests/test_phase38_pipeline_integration.py`
- `[[phase39_pipeline_robustness]]`
- `[[phase39_pipeline_robustness]]`
- `[[phase39_pipeline_robustness_spec]]`

## Current Canonical Task State

The Phase 39 task file currently shows:

- frontmatter tag `status/done`
- `Status: completed`
- all steps 39.1 through 39.6 checked

The step notes capture the validation summary, including:

- dedicated Phase 39 suite passing
- regression pass summary
- successful live rerun with GNN resize working

## Current Repo-State Mismatches

These are the important state mismatches still visible after the coding work completed.

### 1. Top-level training tracker is stale

`[[quant_training_ground]]` still says:

- latest completed phase: Phase 38
- active phase task: [[phase39_pipeline_robustness]]
- latest checkpoint: [[chat_checkpoint_2026-04-19_phase38_complete]]

That is no longer true.

The real state at checkpoint time is:

- latest completed implementation phase: Phase 39
- no new Phase 40 task triad created yet
- latest checkpoint should now be this Phase 39 checkpoint

### 2. Phase 39 task file location is stale relative to its status

`[[phase39_pipeline_robustness]]` is already marked done/completed but remains in `tasks/active/`.

To fully match repository conventions, it should be moved to `tasks/done/` after tracker reconciliation.

## What Was Not Completed Yet

The following recommended closeout actions were identified but not performed before this checkpoint write:

- update `[[quant_training_ground]]` to reflect Phase 39 completion
- move `phase39_pipeline_robustness.md` from `tasks/active/` to `tasks/done/`
- create the next triad for Phase 40, if the next work item is approved
- configure `TIRRA_FRED_API_KEY` if live macro features are desired rather than skipped

## Recommended Immediate Next Step

The immediate next step is not another robustness fix.

The immediate next step is closeout and handoff consistency:

1. update `quant_training_ground.md` so the repo’s main tracker matches reality
2. move the completed Phase 39 task file into `tasks/done/`
3. then decide Phase 40 scope

## Likely Phase 40 Direction

Based on the state reached after Phase 39, the next substantive engineering phase is likely to be one of these:

- live-model refresh / retraining workflow now that the graph grows in production
- historical backfill and backtest work on real feature streams
- operationalizing macro ingestion by supplying the missing FRED configuration and validating live macro feature output

The most natural sequencing is:

- administrative closeout first
- then choose whether Phase 40 is about model refresh, data backfill, or broader production hardening

## Practical Lessons Captured

### 1. Dynamic graph systems need inference-time growth guards

Training-time graph shape is not a safe assumption in production. Any temporal memory attached to node indices must either be rebuilt or resized when the graph expands.

### 2. “No data” and “zero signal” are different states

This distinction affected both convergence and macro behavior.

Bad pattern:

- using missing-valued features to represent every empty output condition

Better pattern:

- `[]` for absent upstream data
- zero-valued features for observed quiet conditions

### 3. Semantic changes require regression expectation rewrites

When the meaning of output states changes, test failures are often stale expectations rather than code defects. The right response is to revalidate the contract and then update the tests deliberately.

## Resume Instructions

If resuming from this checkpoint in a fresh session, read in this order:

1. [[chat_checkpoint_2026-04-19_phase39_complete]]
2. [[quant_training_ground]]
3. [[phase39_pipeline_robustness]]
4. [[phase39_pipeline_robustness_spec]]

Then decide whether the next action is:

- repo closeout only
- Phase 40 planning
- live macro configuration and validation

## Related

- [[phase39_pipeline_robustness]]
- [[phase39_pipeline_robustness_spec]]
- [[phase38_downstream_pipeline_integration]]
- [[quant_training_ground]]
- [[chat_checkpoint_2026-04-19_phase38_complete]]