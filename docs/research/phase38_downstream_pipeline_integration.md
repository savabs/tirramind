---
title: "Research: Phase 38 — Downstream Pipeline Integration"
tags:
  - doc/research
  - phase/38
  - topic/pipeline
  - topic/convergence
  - layer/feature-engineering
  - layer/surveillance
---

# Research: Phase 38 — Downstream Pipeline Integration

## Problem Statement

Phase 37 demonstrated the full pipeline runs end-to-end on real data, but
convergence_detection produced **0 evidence** and feature_generation produced
**17 features with all values missing**. The pipeline's downstream stages
are disconnected from real data. This phase fixes the integration gaps so
the full chain (collection → convergence → features) produces real output.

## Root Cause Analysis

### Issue 1: Source Name Mismatch (Critical)

**The bug:** The DAGExecutor stores tool results to `pipeline_data` using:

```python
source = node.table_name or node.id  # executor.py line 210
```

Since no daily_collection node sets `table_name`, all results are stored
with the **node ID** as source (e.g., `"fetch_cftc"`). But convergence
extractors query by **tool name** (e.g., `"cftc"`):

```python
for tool_name in registered_tools():        # returns ["cftc", "gdelt", ...]
    rows = store.query_data(tool_name, ...)  # queries source="cftc" → 0 rows
```

**Current mismatch table:**

| DAG Node ID | Operator (Tool) | Stored as Source | Extractor Expects |
|-------------|-----------------|------------------|-------------------|
| `fetch_cftc` | `cftc` | `fetch_cftc` | `cftc` |
| `fetch_finra_scan` | `finra_short_volume` | `fetch_finra_scan` | `finra_short_volume` |
| `fetch_power_demand` | `power_grid` | `fetch_power_demand` | `power_grid` |
| `fetch_power_fuel` | `power_grid` | `fetch_power_fuel` | `power_grid` |
| `fetch_gdelt` | `gdelt` | `fetch_gdelt` | `gdelt` |
| `fetch_polymarket` | `polymarket` | `fetch_polymarket` | `polymarket` |
| `fetch_instruments` | callable | `fetch_instruments` | (none) |

Every convergence extractor query returns 0 rows → 0 evidence → 0 detection.

**Fix:** Add `table_name` to each DAG node, matching the registered tool name.

### Issue 2: No macro_data in DAG

`MacroStateFeatureBuilder` reads `pipeline_data` with `source="macro_data"`.
But `macro_data` is not a node in the daily_collection DAG — it's only
registered as a tool in `cli.py`. No pipeline_data rows exist for it.

**Fix:** Add a `fetch_macro` node to the daily_collection DAG that calls
the `macro_data` tool with FRED series DFF, GS10, GS2, WALCL.

### Issue 3: Feature Generation Cascade Failure

With convergence producing 0 evidence → 0 signals, the ConvergenceFeatureBuilder
finds nothing in the `signals` table. With no macro_data rows, the
MacroStateFeatureBuilder finds nothing. GNNFeatureBuilder works (it reads
entity_observations directly), but its 11 features are a minority of the
17 total. All 6 convergence + macro features report `value=None`.

**Fix:** Issues 1 and 2 unblock this automatically.

## Current Architecture (Context)

### Data Flow

```
daily_collection → pipeline_data (source names wrong)
                 → entity_observations (L2 tools, correct names)
                 → entities, entity_links (L2 tools)

convergence_detection → reads pipeline_data (by tool name → 0 rows)
                      → produces signals → signals table

feature_generation → ConvergenceFeatureBuilder: reads signals table
                   → MacroStateFeatureBuilder: reads pipeline_data("macro_data")
                   → GNNFeatureBuilder: reads entity_observations (works)
```

### Extractor Registry

49 extractors registered in `agent/convergence/extractors.py`. Only 5 tools
are currently in the daily_collection DAG (cftc, finra_short_volume,
power_grid, gdelt, polymarket). The remaining 44 tools have extractors
but no DAG nodes — they would need new nodes to produce convergence evidence.

For Phase 38, fixing the 5 existing tools + adding macro_data is sufficient
to prove the pipeline works. Expanding to more tools is Phase 39+ scope.

## Observations

1. The `table_name` field already exists on the `Node` dataclass — it was
   designed for exactly this use case but never populated.
2. The `power_grid` tool runs as two nodes (demand + fuel_mix). Both should
   use `table_name="power_grid"` since the extractor expects that source name.
   Multiple rows with the same source is fine — `query_data` returns all
   matching rows ordered by `fetched_at DESC`.
3. The `fetch_instruments` node doesn't need a table_name fix — there's no
   convergence extractor for instruments (prices go to entity_observations).
4. The MacroDataTool requires a FRED API key. The DAG node should use
   `params={"series_ids": ["DFF", "GS10", "GS2", "WALCL"]}` to fetch
   the specific series needed by MacroStateFeatureBuilder.

## Risks

1. **Stale pipeline_data rows from Phase 37 run** — The existing `pipeline_data`
   table has rows with wrong source names (`fetch_cftc` etc.). After the fix,
   convergence detection will only find rows written by future DAG runs. Old
   rows won't be visible. This is acceptable — the old data is still there
   for forensics if needed.
2. **macro_data FRED API key** — MacroDataTool requires `TIRRA_FRED_API_KEY`.
   If not set, the node will fail gracefully (retries=2, then skip).
3. **Test breakage** — Tests that assert on DAG node structure may need
   updates if they check for the absence of `table_name`.

## Data Requirements

- FRED API key for macro_data (env var `TIRRA_FRED_API_KEY`)
- Existing pipeline DB with entity observations from Phase 37

## Related

- [[phase37_first_live_pipeline]]
- [[phase38_downstream_pipeline_integration_spec]]
- [[phase38_downstream_pipeline_integration]]
- [[chat_checkpoint_2026-04-19_phase37_complete]]
