---
title: "Spec: Phase 18 — Tier 1 L2 Tool Expansion"
tags:
  - doc/spec
  - phase/18
  - topic/surveillance
  - topic/world-model
  - layer/surveillance
  - layer/world-model
---

# Spec: Tier 1 L2 Tool Expansion

## Goal

Upgrade sanctions_monitor, gov_contracts, and supply_chain_monitor from
L1 (aggregate returns) to L2 (entity-resolved, GNN-integrated). This fills
the critical `organization` entity gap (0 tools → 2 tools) and adds 3 new
observation types + 5 new edge types to the world model graph.

## Files Affected

| File | Action |
|------|--------|
| `agent/models/gnn/graph_builder.py` | Add 3 observation types to OBSERVATION_TYPES |
| `agent/tools/sanctions_monitor.py` | Add pipeline_store, _persist_entities, entity links |
| `agent/tools/gov_contracts.py` | Add pipeline_store, _persist_entities, entity links |
| `agent/tools/supply_chain_monitor.py` | Add pipeline_store, _persist_entities (no links) |
| `tests/test_sanctions_monitor_l2.py` | New test file |
| `tests/test_gov_contracts_l2.py` | New test file |
| `tests/test_supply_chain_monitor_l2.py` | New test file |

## Implementation Steps

### Step 18.1: Expand graph_builder observation types

Add `"sanctions_listing"`, `"contract_award"`, `"price_movement"` to
`OBSERVATION_TYPES` in `agent/models/gnn/graph_builder.py`. Run existing
graph_builder tests to confirm no regression.

### Step 18.2: sanctions_monitor L2

1. Add `pipeline_store: PipelineStore | None = None` kwarg to `__init__`.
2. Add `_PROGRAM_COUNTRY` mapping dict (program → ISO 3166-1 alpha-2).
3. Add `_persist_entities(self, results: list[dict])` guard method.
4. Add `_persist_entities_inner(self, results)`:
   - For each result:
     - Map SDN_Type ("Individual" → person, "Entity" → organization,
       "Vessel" → vessel, default → organization)
     - Generate entity_id via `entity_id_from_key(type, normalized_name)`
     - `register_entity(entity_id, type, {"name": ..., "source": ...})`
     - `store_entity_observation(entity_id, "sanctions_listing", depth_level=2, ...)`
     - For each program in programs[]:
       - Look up country in `_PROGRAM_COUNTRY`
       - If country found: create country entity, `link_entities(entity_id, country_id, "sanctioned_under", confidence, metadata)`
5. Call `_persist_entities(results)` at end of each mode handler before return.

### Step 18.3: sanctions_monitor L2 tests

Test file: `tests/test_sanctions_monitor_l2.py`

Required test cases:
- Entity registration for each type (individual→person, entity→organization, vessel→vessel)
- Observation storage with correct depth_level=2
- Program → country link creation (known program)
- Skipped links for multi-country programs (SDGT, ISIL)
- Empty results → no persistence calls
- No pipeline_store → graceful skip (L1 fallback)
- Deduplication across results (same entity appears twice)
- Unicode entity names (Cyrillic, Arabic transliterations)

### Step 18.4: gov_contracts L2

1. Add `pipeline_store: PipelineStore | None = None` kwarg to `__init__`.
2. Add `_persist_entities(self, results: list[dict], country_code: str)` guard.
3. Add `_persist_entities_inner(self, results, country_code)`:
   - For each contract:
     - Company (recipient): `entity_id_from_key("company", normalize_company_name(recipient))`
     - Agency: `entity_id_from_key("organization", normalize_company_name(agency))`
     - `register_entity(company_id, "company", {...})`
     - `register_entity(agency_id, "organization", {...})`
     - `store_entity_observation(company_id, "contract_award", depth_level=2, ...)`
     - `link_entities(company_id, agency_id, "awarded_by", confidence=1.0, ...)`
     - Country entity: `entity_id_from_key("country", country_code)`
     - `register_entity(country_id, "country", ...)`
     - `link_entities(company_id, country_id, "operates_in", confidence=0.9, ...)`
4. Call `_persist_entities(results, "US")` for USASpending results,
   `_persist_entities(results, "GB")` for UK Contracts Finder results.

### Step 18.5: gov_contracts L2 tests

Test file: `tests/test_gov_contracts_l2.py`

Required test cases:
- Company + agency entity registration
- Observation storage for company
- awarded_by link creation (company → organization)
- operates_in link creation (company → country)
- US vs UK country code handling
- Empty results → no persistence calls
- No pipeline_store → graceful skip
- Company name normalization (suffixes stripped, Unicode handled)
- Deduplication across contracts (same company, different contracts)

### Step 18.6: supply_chain_monitor L2

1. Add `pipeline_store: PipelineStore | None = None` kwarg to `__init__`.
2. Add `_persist_entities(self, results: dict)` guard method.
3. Add `_persist_entities_inner(self, results)`:
   - For each series in results:
     - Entity: `entity_id_from_key("topic", series_id)`
     - `register_entity(entity_id, "topic", {"name": label, "sector": sector})`
     - `store_entity_observation(entity_id, "price_movement", depth_level=2, ...)`
     - No links — sector-level data has no natural entity connections
4. Call `_persist_entities(results)` at end of each mode handler.

### Step 18.7: supply_chain_monitor L2 tests

Test file: `tests/test_supply_chain_monitor_l2.py`

Required test cases:
- Topic entity registration for each series
- Observation storage with correct metadata
- Empty results → no persistence calls
- No pipeline_store → graceful skip
- Multiple series → multiple entities
- Same series across calls → same entity_id (idempotent)

### Step 18.8: Integration validation

- Run all existing tests to confirm no regressions
- Run all new L2 tests
- Verify OBSERVATION_TYPES count is now 18

## Edge Cases

- Sanctioned entity with no programs → still register entity, skip link creation
- Contract with empty/None recipient → skip that record
- BLS series with no values → register entity, skip observation
- SDN entity with type not in {Individual, Entity, Vessel} → default to organization
- Agency name that's just an abbreviation (e.g., "DOD") → normalize but don't strip
- Same entity registered by multiple tools → entity_id collision is correct (same
  entity should merge via same key), observations are additive

## Testing Plan

Each tool gets a dedicated test file testing L2 behavior with mocked
PipelineStore. Tests verify:
1. Correct entity types registered
2. Correct observation types stored
3. Correct links created with correct confidence
4. Graceful degradation when no pipeline_store
5. Edge cases per tool (listed above)

Integration test at Step 18.8 verifies no regressions.

## Related

- [[tier1_tool_expansion]] — research doc
- [[tier1_tool_expansion|task]] — tracked in active task
- [[l2_tool_expansion_spec]] — Phase 13 L2 pattern reference
- [[entity_linking_layer_spec]] — Phase 17 entity linking spec
