---
title: "Checkpoint: Phase 38 — Downstream Pipeline Integration Complete"
tags:
  - doc/checkpoint
  - phase/38
  - topic/pipeline
  - topic/convergence
  - layer/feature-engineering
  - layer/surveillance
---

# Checkpoint: Phase 38 Complete (2026-04-19)

## What Was Done

Phase 37 (first live pipeline run) revealed that convergence_detection produced **0 evidence** and feature_generation produced **17 features all with None values**. Phase 38 diagnosed and fixed the root cause.

### Root Cause

DAGExecutor stores tool results using `source = node.table_name or node.id`. Since no daily_collection node set `table_name`, results were stored as `"fetch_cftc"` but convergence extractors query for `"cftc"` → 0 rows found.

### Changes Made

1. **`agent/pipeline/dags/daily_collection.py`** — Added `table_name` to all 7 tool nodes matching extractor registry names. Added `fetch_macro` node (macro_data tool, DFF/GS10/GS2/WALCL series). DAG now has 8 nodes total.

2. **`tests/test_phase38_pipeline_integration.py`** — 20 tests across 6 test classes:
   - TestSourceNameAlignment (4 tests) — validates table_name matches extractor registry
   - TestMacroDataNode (4 tests) — validates fetch_macro configuration
   - TestConvergenceEvidenceLoading (3 tests) — proves correct source names yield evidence, wrong names yield 0
   - TestMacroFeatureBuilderIntegration (2 tests) — proves macro features produce values with data
   - TestPipelineSmokeTest (1 test) — full convergence→features pipeline
   - TestDagStructureUpdated (6 tests) — validates 8-node DAG structure

3. **`tests/test_pipeline_registry.py`** — Fixed stale assertions: node count 6→8, added fetch_macro/fetch_instruments to expected IDs, updated operator type assertion for callable nodes.

### Test Results

- Phase 38 tests: 20/20 pass
- Pipeline registry tests: 39/39 pass
- Combined: 59/59 pass

### Node ID → table_name Mapping

| Node ID | table_name | Operator |
|---------|-----------|----------|
| fetch_cftc | cftc | cftc (str) |
| fetch_finra_scan | finra_short_volume | finra_short_volume (str) |
| fetch_power_demand | power_grid | power_grid (str) |
| fetch_power_fuel | power_grid | power_grid (str) |
| fetch_gdelt | gdelt | gdelt (str) |
| fetch_polymarket | polymarket | polymarket (str) |
| fetch_macro | macro_data | macro_data (str) |
| fetch_instruments | (none) | callable |

## What's Next

Phase 38 fixed the plumbing. The downstream pipeline should now produce real evidence and features. Suggested next steps:

1. **Re-run the live pipeline** to confirm non-zero evidence and non-None features in production
2. **GNN inference DAG integration** — ensure GNN features also flow correctly
3. **Phase 39 candidates:**
   - Backtest with real pipeline data (walk-forward with live features)
   - Additional convergence extractor coverage (some extractors still need field-name alignment with actual tool output formats)
   - Dashboard/monitoring for pipeline health

## Files Modified

- `agent/pipeline/dags/daily_collection.py`
- `tests/test_phase38_pipeline_integration.py` (created)
- `tests/test_pipeline_registry.py` (updated)
- `[[phase38_downstream_pipeline_integration]]` (created, completed)
- `[[quant_training_ground]]` (updated)
- `[[phase38_downstream_pipeline_integration]]` (created)
- `[[phase38_downstream_pipeline_integration_spec]]` (created)

## Related

- [[phase38_downstream_pipeline_integration]]
- [[phase38_downstream_pipeline_integration_spec]]
- [[phase37_first_live_pipeline]]
- [[quant_training_ground]]
- [[chat_checkpoint_2026-04-19_phase37_complete]]
