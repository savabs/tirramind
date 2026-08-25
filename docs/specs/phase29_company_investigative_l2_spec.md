---
title: "Spec: Phase 29 — Company + Investigative L2"
tags:
  - doc/spec
  - phase/29
  - topic/entity-linking
  - topic/bankruptcy
  - topic/foia
  - topic/academic-preprints
  - layer/surveillance
  - layer/world-model
---

# Spec: Phase 29 — Company + Investigative L2

## Goal

Upgrade `bankruptcy_court`, `foia_requests`, and `academic_preprints` to L2 entity-resolved persistence. Company, person, and topic nodes gain three new observation types. OBSERVATION_TYPES grows 32 → 35, ENRICHMENT_DIM 41 → 44.

## Files Affected

| File | Change |
|------|--------|
| `agent/tools/bankruptcy_court.py` | Add `_persist_entities()` + `_persist_entities_inner()`, call from `execute()` |
| `agent/tools/foia_requests.py` | Add `_persist_entities()` + `_persist_entities_inner()`, call from `execute()` |
| `agent/tools/academic_preprints.py` | Add `_persist_entities()` + `_persist_entities_inner()`, call from `execute()` |
| `agent/models/gnn/graph_builder.py` | Add `bankruptcy_status`, `investigation_signal`, `research_velocity` to `OBSERVATION_TYPES` |
| `tests/test_bankruptcy_court_edge.py` | Add L2 persistence tests |
| `tests/test_foia_requests_edge.py` | Add L2 persistence tests |
| `tests/test_academic_preprints_edge.py` | Add L2 persistence tests |
| `tests/test_graph_builder_expanded.py` | Update obs type count 32→35, membership assertions |
| `tests/test_phase29_diagnostic.py` | NEW — integration diagnostic tests |

## Implementation Steps

### 29.1: bankruptcy_court L2 persistence

Add `_persist_entities()` / `_persist_entities_inner()` to `BankruptcyCourtTool`.

- Extract debtor/respondent/filer name from each result record
- Normalize name: strip "In re:", "Debtor", common legal suffixes
- `register_entity("company", normalized_name, entity_id)`
- `store_entity_observation(entity_id, "bankruptcy_court", ts, "bankruptcy_status", value_dict, depth_level=2)`
- Call from `execute()` after successful fetch, non-fatal try/except
- Modes: `us_bankruptcy`, `sec_enforcement`, `sec_bankruptcy`, `uk_insolvency` all persist
- Verification: 12+ tests — each mode persists correct entities, None store is safe, exception non-fatal, idempotent, entity IDs deterministic

### 29.2: foia_requests L2 persistence

Add `_persist_entities()` / `_persist_entities_inner()` to `FOIARequestsTool`.

- Extract subject entities from request titles (company names, person names)
- Use simple heuristic: capitalize words in title, match against known patterns
- `register_entity("company"|"person", name, entity_id)`
- `store_entity_observation(entity_id, "foia_requests", ts, "investigation_signal", value_dict, depth_level=2)`
- Call from `execute()` after successful fetch
- Modes: `muckrock`, `whatdotheyknow` both persist
- Verification: 12+ tests — both modes persist, entity type selection correct, None store safe, exception non-fatal

### 29.3: academic_preprints L2 persistence

Add `_persist_entities()` / `_persist_entities_inner()` to `AcademicPreprintsTool`.

- `trials` mode: extract sponsor company → company entity, trial phase → obs value
- `papers`/`trending` mode: extract arXiv category → topic entity
- `register_entity("company"|"topic", name, entity_id)`
- `store_entity_observation(entity_id, "academic_preprints", ts, "research_velocity", value_dict, depth_level=2)`
- Verification: 12+ tests — trials sponsor extraction, papers category extraction, trending mode, None store, exception handling

### 29.4: Graph builder update

- Add `bankruptcy_status`, `investigation_signal`, `research_velocity` to `OBSERVATION_TYPES` (alphabetical sort)
- Update `ENRICHMENT_DIM = 9 + len(OBSERVATION_TYPES)` (41 → 44)
- Verification: graph_builder_expanded tests pass — obs type count 32→35, new types in sorted list, ENRICHMENT_DIM=44

### 29.5: Integration diagnostics

Create `tests/test_phase29_diagnostic.py`:
- bankruptcy_court L2 → store → graph builder flow (company node has bankruptcy_status)
- foia_requests L2 → store → graph builder flow (company + person nodes have investigation_signal)
- academic_preprints L2 → store → graph builder flow (company + topic nodes have research_velocity)
- Cross-tool: company entity receives observations from all 3 tools
- Verification: 10+ integration tests pass

### 29.6: Regression + stale count fixes

- Run full test suite
- Fix any stale count assertions (tool count, bandit arms, obs type count, ENRICHMENT_DIM)
- Write checkpoint
- Verification: all tests pass, no Phase 29 regressions

## Edge Cases

- Debtor name normalization: "In re: Foo Corp" vs "Foo Corporation" → both create same entity? No — keep simple, let graph learn. Accept minor duplication initially.
- Empty result sets: mode returns 0 records → `_persist_entities` returns `{obs_type: 0}` gracefully
- None pipeline_store: skip silently, return zero counts
- Exception in persistence: non-fatal, log and return zero counts
- Missing fields: entity extraction from free-text may yield None → skip that record

## Testing Plan

- Per-tool edge case suites: 12+ tests each (36+ total)
- Graph builder assertions: updated counts
- Integration diagnostics: 10+ cross-tool tests
- Full regression: all ~9054+ tests pass
- Target: ~50 new tests across 4 test files

## Related

- [[phase29_company_investigative_l2]]
- [[phase29_company_investigative_l2_spec]]
- [[phase28_country_macro_enrichment_spec]]
- [[7b-E_bankruptcy_court]]
- [[7b-S_foia_logs]]
- [[7b-M_academic_preprints]]
