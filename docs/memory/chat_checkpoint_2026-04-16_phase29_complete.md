---
title: "Checkpoint: Phase 29 Complete — Company + Investigative L2"
tags:
  - doc/checkpoint
  - phase/29
  - topic/entity-linking
  - topic/bankruptcy
  - topic/foia
  - topic/academic-preprints
  - layer/surveillance
  - layer/world-model
---

# Checkpoint: Phase 29 Complete

## Summary

Phase 29 added L2 entity persistence to three investigative/company tools:
- **bankruptcy_court** → `bankruptcy_status` obs on company entities (23 L2 tests)
- **foia_requests** → `investigation_signal` obs on company entities (14 L2 tests)
- **academic_preprints** → `research_velocity` obs on company/topic entities (21 L2 tests)

OBSERVATION_TYPES: 32 → 35. ENRICHMENT_DIM: 41 → 44.
Phase 29 diagnostic tests: 18 integration tests.
Full regression: 4483/4484 pass (1 pre-existing: `test_ross_stores_xml` shares count mismatch).

## Files Modified

### Tool files
- `agent/tools/bankruptcy_court.py` — L2 imports, constructor, execute refactor, _persist_entities/_persist_entities_inner
- `agent/tools/foia_requests.py` — same pattern + added `data=` to 3 ToolResult returns
- `agent/tools/academic_preprints.py` — same pattern; trials→company, papers/trending→topic

### Graph builder
- `agent/models/gnn/graph_builder.py` — 3 new obs types, ENRICHMENT_DIM 41→44

### Test files
- `tests/test_bankruptcy_court_edge.py` — 23 L2 tests
- `tests/test_foia_requests_edge.py` — 14 L2 tests
- `tests/test_academic_preprints_edge.py` — 21 L2 tests
- `tests/test_graph_builder_expanded.py` — count 32→35, phase29 types assertion
- `tests/test_phase28_diagnostic.py` — ENRICHMENT_DIM 41→44
- `tests/test_phase29_diagnostic.py` — 18 integration tests (NEW)

## Pre-existing Issues

- `test_feature_generation_dag.py` — stale feature count (17 vs 6), excluded from regression
- `test_form144_edge.py::TestXMLParser::test_ross_stores_xml` — shares_to_sell 4154 vs 4454

## Entity Persistence Pattern (L2)

All three tools follow the same pattern:
1. Imports: `time`, `TYPE_CHECKING`, `PipelineStore` guard, `_entity_id_from_key` try/except
2. Constructor: `pipeline_store: "PipelineStore | None" = None` → `self._store`
3. Execute: capture result → call `_persist_entities(result.data, mode)` after success
4. `_persist_entities()`: None-guard outer wrapper + try/except non-fatal
5. `_persist_entities_inner()`: register_entity + store_entity_observation(depth_level=2)

## What's Next

Phase 29 task file should be moved to `tasks/done/`.
Next phase TBD — consider: more L2 upgrades on remaining tools, GNN training pipeline, or cross-entity linking.

## Related

- [[phase29_company_investigative_l2]]
- [[phase29_company_investigative_l2_spec]]
- [[chat_checkpoint_2026-04-16_phase29_ready]]
