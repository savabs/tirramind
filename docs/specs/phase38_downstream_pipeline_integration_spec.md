---
title: "Spec: Phase 38 — Downstream Pipeline Integration"
tags:
  - doc/spec
  - phase/38
  - topic/pipeline
  - topic/convergence
  - layer/feature-engineering
  - layer/surveillance
---

# Spec: Phase 38 — Downstream Pipeline Integration

## Goal

Fix the source name mismatch in daily_collection that prevents convergence_detection
from finding any evidence. Add macro_data to the DAG so MacroStateFeatureBuilder
gets input. Result: the full collection → convergence → features pipeline produces
real, non-empty output.

## Files Affected

### Modified Files

| File | Change |
|------|--------|
| `agent/pipeline/dags/daily_collection.py` | Add `table_name` to all tool nodes; add `fetch_macro` node |
| `tests/test_pipeline_registry.py` | Update any assertions about node structure |

### New Files

| File | Purpose |
|------|---------|
| `tests/test_phase38_pipeline_integration.py` | Verify source names match, convergence produces evidence, features are non-empty |

## Implementation Steps

### 38.1: Fix source names on existing DAG nodes

Add `table_name=<tool_name>` to every tool-backed node in
`build_daily_collection_dag()`:

```python
dag.add("fetch_cftc", operator="cftc",
        table_name="cftc",                      # ← ADD
        params={"mode": "latest"}, ...)

dag.add("fetch_finra_scan", operator="finra_short_volume",
        table_name="finra_short_volume",         # ← ADD
        params={"mode": "short_volume"}, ...)

dag.add("fetch_power_demand", operator="power_grid",
        table_name="power_grid",                 # ← ADD
        params={"mode": "demand"}, ...)

dag.add("fetch_power_fuel", operator="power_grid",
        table_name="power_grid",                 # ← ADD
        params={"mode": "fuel_mix"}, ...)

dag.add("fetch_gdelt", operator="gdelt",
        table_name="gdelt",                      # ← ADD
        params={"mode": "events", ...}, ...)

dag.add("fetch_polymarket", operator="polymarket",
        table_name="polymarket",                 # ← ADD
        params={"category": "all", ...}, ...)
```

The `fetch_instruments` node remains unchanged (no convergence extractor).

**Test:** Build DAG, assert every tool node has `table_name` set, and that
`table_name` matches the registered extractor name for that operator.

### 38.2: Add macro_data fetch node

Add a new node to the daily_collection DAG:

```python
dag.add(
    "fetch_macro",
    operator="macro_data",
    table_name="macro_data",
    params={"series_ids": ["DFF", "GS10", "GS2", "WALCL"]},
    timeout=120,
    retries=2,
)
```

This fetches the 4 FRED series needed by MacroStateFeatureBuilder.

**Test:** Build DAG, assert `fetch_macro` node exists with correct table_name.

### 38.3: Write integration tests

Create `tests/test_phase38_pipeline_integration.py` with:

1. **test_dag_source_names_match_extractors** — Build DAG, for each tool node
   verify `node.table_name` is in the convergence extractor registry.
2. **test_convergence_finds_evidence_after_fix** — Seed `pipeline_data` with
   correct source names, run `_load_evidence_from_store()`, assert evidence > 0.
3. **test_macro_builder_finds_data** — Seed `pipeline_data` with
   `source="macro_data"`, run MacroStateFeatureBuilder, assert non-None values.
4. **test_full_pipeline_smoke** — Seed pipeline_data with mock data for all
   active tools, run convergence_detection + feature_generation, assert
   non-zero features with values.

### 38.4: Fix any test regressions

Update existing tests in `test_pipeline_registry.py` or `test_e2e_integration.py`
if they assert on DAG node counts or structure.

## Edge Cases

1. **Two nodes with same table_name** — `fetch_power_demand` and `fetch_power_fuel`
   both use `table_name="power_grid"`. This is fine — `pipeline_data` allows
   multiple rows with the same source, and `query_data` returns all of them.
2. **Missing FRED API key** — `fetch_macro` node will fail if `TIRRA_FRED_API_KEY`
   is not set. The DAG executor handles this via retries + skip. Convergence
   still works from the other 5 tools.
3. **Old pipeline_data rows** — Rows written before this fix have wrong source
   names. They are invisible to convergence but harmless. No migration needed.

## Testing Plan

- Unit: test_phase38 verifies source name correctness and mock data flow
- Regression: full test suite passes
- Manual: re-run `scripts/run_collection.py` + convergence + features on live DB
  to verify non-zero evidence and features

## Related

- [[phase38_downstream_pipeline_integration]]
- [[phase38_downstream_pipeline_integration_task]]
- [[phase37_first_live_pipeline]]
- [[chat_checkpoint_2026-04-19_phase37_complete]]
