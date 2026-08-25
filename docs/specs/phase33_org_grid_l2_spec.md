---
title: "Spec: Phase 33 — Organization + Grid Enrichment L2"
tags:
  - doc/spec
  - phase/33
  - topic/l2-expansion
  - layer/surveillance
---

# Spec: Phase 33 — Organization + Grid Enrichment L2

## Goal

Add L2 entity persistence to `regulatory_gazette` and `electricity_monitor`, populating `organization` entities with `regulatory_velocity` and `grid_demand` observations. Update graph_builder constants.

## Files Affected

| File | Change |
|------|--------|
| `agent/tools/regulatory_gazette.py` | Add `pipeline_store`, `_persist_entities`, `_persist_entities_inner` |
| `agent/tools/electricity_monitor.py` | Add `pipeline_store`, `_persist_entities`, `_persist_entities_inner` |
| `agent/models/gnn/graph_builder.py` | Add 2 obs types, ENRICHMENT_DIM 52→54 |
| `tests/test_phase33_l2.py` | Edge case tests |

## Implementation Steps

### 33.1: `regulatory_gazette` L2

1. Add `TYPE_CHECKING` import for `PipelineStore`, try-import `entity_id_from_key`.
2. Add `pipeline_store` parameter to `__init__`, store as `self._store`.
3. Restructure `execute()` to capture result before returning, call `_persist_entities(result.data, mode)` when `result.success and result.data`.
4. Implement `_persist_entities(data, mode)` → guard wrapper.
5. Implement `_persist_entities_inner(data, mode)`:
   - Extract `documents` list from `data`.
   - Collect unique agencies: for each doc, iterate `doc["agencies"]`, resolve to `MARKET_AGENCIES` key if possible, else lowercase+underscore.
   - For each unique agency: `register_entity("organization", agency_key, eid)`, `store_entity_observation(observation_type="regulatory_velocity", value={mode, doc_count, significant_count, ...})`.
   - Return `{"regulatory_velocity_obs": N}`.

### 33.2: `electricity_monitor` L2

1. Add `TYPE_CHECKING` import for `PipelineStore`, try-import `entity_id_from_key`.
2. Add `pipeline_store` parameter to `__init__` (must remain keyword-only: `*, cache, pipeline_store`).
3. Restructure `execute()` to capture result, call `_persist_entities(region, mode)` when `result.success`.
4. Implement `_persist_entities(region, mode)` → guard wrapper.
5. Implement `_persist_entities_inner(region, mode)`:
   - Register BA as `organization` entity with key = region code (e.g., `PJM`).
   - Store `grid_demand` obs with `{mode, region, region_name}`.
   - Return `{"grid_demand_obs": 1}`.

NOTE: Unlike other tools, electricity_monitor persists based on the input parameters (region, mode), not the result data, because mode handlers return formatted text strings not structured dicts. This keeps it simple and reliable.

### 33.3: `graph_builder` update

- Add `grid_demand` after `geopolitical_event` (alphabetical)
- Add `regulatory_velocity` after `project_status`
- OBSERVATION_TYPES: 43 → 45
- ENRICHMENT_DIM: 52 → 54

## Edge Cases

### regulatory_gazette
- No documents → 0 obs
- `_list_agencies()` mode has no documents → 0 obs
- Document with empty agencies list → skip
- Unknown agency (not in MARKET_AGENCIES) → use best-effort key
- Multiple docs from same agency → count aggregation
- Exception in persistence → non-fatal, return 0

### electricity_monitor
- Empty/missing region → not reached (execute validates before)
- Unknown BA code (not in KNOWN_REGIONS) → still persisted (EIA has more BAs than our dict)
- No API key → execute returns error before persistence
- Exception in persistence → non-fatal, return 0

## Testing Plan

- Guard tests: no store → 0, no entity_id_from_key → 0
- Exception safety tests
- Per-tool mode-specific tests with mock data
- Graph builder: 45 obs types, ENRICHMENT_DIM == 54, sorted order

## Related

- [[phase33_org_grid_l2]]
- [[l2_expansion_roadmap]]
- [[quant_training_ground]]
