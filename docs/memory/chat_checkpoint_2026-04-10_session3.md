---
title: "Checkpoint: Phase 18 Complete — Tier 1 L2 Tool Expansion"
tags:
  - doc/checkpoint
  - phase/18
  - topic/surveillance
  - topic/world-model
  - layer/surveillance
  - layer/world-model
---

# Checkpoint: 2026-04-10 Session 3 — Phase 18 Complete

**Session scope**: Completed the full Phase 18 pipeline for Tier 1 tool expansion: research, spec, task decomposition, implementation, and edge-case validation for `sanctions_monitor`, `gov_contracts`, and `supply_chain_monitor` L2 upgrades.

**Prior checkpoint**: [[chat_checkpoint_2026-04-10_session2]]

---

## What Was Completed

### Workflow artifacts created

- Research: [[tier1_tool_expansion]]
- Spec: [[tier1_tool_expansion_spec]]
- Task: [[tier1_tool_expansion]]

### Implementation completed

#### 18.1: Graph builder registry expansion

- Added 3 new observation types to `OBSERVATION_TYPES` in `agent/models/gnn/graph_builder.py`:
  - `sanctions_listing`
  - `contract_award`
  - `price_movement`
- Observation type count changed from 15 to 18.

#### 18.2–18.3: sanctions_monitor upgraded to L2

- Added `pipeline_store` support.
- Added `_persist_entities()` and `_persist_entities_inner()`.
- Added entity registration for:
  - `person` from sanctioned individuals
  - `organization` from sanctioned entities
  - `vessel` from sanctioned vessels
- Added `sanctions_listing` observations with `depth_level=2`.
- Added `_PROGRAM_COUNTRY` mapping for sanctions programs to ISO country codes.
- Added `sanctioned_under` links from sanctioned entity -> country when the program maps to a single country.
- This is one of the first tools in the repo to materially populate the previously empty `organization` entity type.

#### 18.4–18.5: gov_contracts upgraded to L2

- Added `pipeline_store` support.
- Added `_persist_entities()` and `_persist_entities_inner()`.
- Added entity registration for:
  - `company` from contract recipients
  - `organization` from awarding agencies
  - `country` from region context (`US`, `GB`)
- Added `contract_award` observations on recipient companies.
- Added `awarded_by` links from company -> organization.
- Added `operates_in` links from company -> country.
- This is the second tool in the phase to populate `organization` nodes.

#### 18.6–18.7: supply_chain_monitor upgraded to L2

- Added `pipeline_store` support.
- Added `_persist_entities()` and `_persist_entities_inner()`.
- Added `topic` entity registration for BLS series / sector channels.
- Added `price_movement` observations with latest value metadata.
- No entity links were added here because the source is sector-level aggregate time series rather than explicit entity-to-entity records.

### Tests completed

- `tests/test_sanctions_monitor_l2.py`: 27 passed
- `tests/test_gov_contracts_l2.py`: 21 passed
- `tests/test_supply_chain_monitor_l2.py`: 14 passed

### Regression / integration validation

- Ran the new L2 tests plus existing graph/GNN regression tests.
- One expected failure appeared in `tests/test_graph_builder_expanded.py` because the observation count assertion still expected 15.
- Updated the test to expect 18.
- Final integration result: **167 passed, 0 failed**.

---

## Files Created Or Updated

### New docs

- `[[tier1_tool_expansion]]`
- `[[tier1_tool_expansion_spec]]`
- `[[tier1_tool_expansion]]`
- `[[chat_checkpoint_2026-04-10_session3]]`

### Implementation files

- `agent/models/gnn/graph_builder.py`
- `agent/tools/sanctions_monitor.py`
- `agent/tools/gov_contracts.py`
- `agent/tools/supply_chain_monitor.py`

### New test files

- `tests/test_sanctions_monitor_l2.py`
- `tests/test_gov_contracts_l2.py`
- `tests/test_supply_chain_monitor_l2.py`

### Updated existing test files

- `tests/test_graph_builder_expanded.py`

---

## Architectural Outcome

### Critical graph improvement

Phase 16 diagnostics identified a severe gap: **`organization` entity type had zero producing tools**.

After Phase 18:

- `sanctions_monitor` now produces `organization` entities for sanctioned entities.
- `gov_contracts` now produces `organization` entities for awarding agencies.

This materially improves heterogeneity in the entity graph and gives the GNN more cross-type structure to learn from.

### New edge families introduced

- `organization/person/vessel -> country` via `sanctioned_under`
- `company -> organization` via `awarded_by`
- `company -> country` via `operates_in`

### New observation channels introduced

- `sanctions_listing`
- `contract_award`
- `price_movement`

---

## Current State

- Phase 18 implementation is complete.
- Phase 18 edge-case validation is complete.
- The task checklist in [[tier1_tool_expansion]] has been fully checked off.
- The task file is still located in `tasks/active/` and has not yet been moved to `tasks/done/`.

---

## Natural Next Step

Close Phase 18 housekeeping:

1. mark the task as completed in metadata/status
2. move it from `tasks/active/` to `tasks/done/`
3. update the master roadmap / active-phase tracker to point to the next phase

---

## Related

- [[chat_checkpoint_2026-04-10_session2]]
- [[tier1_tool_expansion]]
- [[tier1_tool_expansion_spec]]
- [[entity_linking_layer]]
- [[gnn_guided_tool_expansion]]