---
title: "Spec: Phase 32 — Trade + Disease + Political L2"
tags:
  - doc/spec
  - phase/32
  - topic/l2-expansion
  - layer/surveillance
---

# Spec: Phase 32 — Trade + Disease + Political L2

## Goal

Add L2 entity persistence to `comtrade`, `transport_throughput`,
`disease_surveillance`, `political_risk`.  Register 4 new observation types
in `graph_builder.py` and update `ENRICHMENT_DIM`.

## Files Affected

| File | Change |
|------|--------|
| `agent/tools/comtrade.py` | Add `pipeline_store`, entity import, `_persist_entities`, `_persist_entities_inner`, ISO3→ISO2 map |
| `agent/tools/transport_throughput.py` | Add `pipeline_store`, entity import, `_persist_entities`, `_persist_entities_inner` |
| `agent/tools/disease_surveillance.py` | Add `pipeline_store`, entity import, `_persist_entities`, `_persist_entities_inner` (multi-mode) |
| `agent/tools/political_risk.py` | Add `pipeline_store`, entity import, `_persist_entities`, `_persist_entities_inner` |
| `agent/models/gnn/graph_builder.py` | Add 4 obs types, update ENRICHMENT_DIM 48→52 |
| `tests/test_phase32_l2.py` | Edge case tests for all 4 tools + graph builder |

## Implementation Steps

### 32.1: comtrade L2

1. Add `pipeline_store: "PipelineStore | None" = None` to `__init__`.
2. Add `TYPE_CHECKING` import for `PipelineStore`, try-import for `entity_id_from_key`.
3. Add ISO3→ISO2 mapping dict `_ISO3_TO_ISO2` (cover all countries in `M49_CODES`).
4. Add `_persist_entities(self, data: dict, mode: str) -> dict[str, int]` — non-fatal wrapper.
5. Add `_persist_entities_inner`:
   - Extract reporter ISO-3 from `data["reporter"]`, convert to ISO-2.
   - Skip if reporter is empty or "World" or not in mapping.
   - Register country entity, store `trade_flow` observation with value containing:
     `mode`, `partner`, `flow`, `commodity_code`, `trade_value_usd` (top record),
     `record_count`, `period`.
   - Return `{"trade_flow_obs": 1}`.
6. Call `self._persist_entities(result.data, mode)` after successful execute in each mode handler's result return. Actually: add a single call in `execute()` after mode dispatch, similar to food_security pattern.

**Note:** comtrade `execute()` dispatches to `_flows`/`_commodity`/`_partners` which return ToolResult directly. Must call _persist after getting the result. Restructure `execute()` to capture result, persist, then return.

### 32.2: transport_throughput L2

1. Add `pipeline_store` to `__init__`.
2. Add TYPE_CHECKING + try-import.
3. Add `_persist_entities(self, data: dict, mode: str) -> dict[str, int]`.
4. Add `_persist_entities_inner`:
   - Extract border info from records.  Map borders to country codes:
     `"US-Canada Border" → ["US", "CA"]`, `"US-Mexico Border" → ["US", "MX"]`.
   - For each unique country found, register entity, store `border_throughput` obs.
   - Value: `mode`, `measure`, `total_volume` (sum), `record_count`, `period`.
   - Return `{"border_throughput_obs": <count>}`.
5. Call `_persist_entities()` after successful execute.

### 32.3: disease_surveillance L2

1. Add `pipeline_store` to `__init__`.
2. Add TYPE_CHECKING + try-import.
3. Add `_persist_entities(self, data: dict, mode: str) -> dict[str, int]`.
4. Add `_persist_entities_inner` — mode-specific logic:
   - **wastewater:** Persist on US country entity.  Value: `hot_states`,
     `total_samples`, `states_count`, `pathogen` (if specific).
   - **outbreaks:** Extract unique countries from `entries[].country_parsed`.
     Persist one `pathogen_level` obs per country.  Value: `entry_count`,
     `disease_frequency` (top diseases for that country).
   - **eu_surveillance:** Extract unique countries from `records[].country_code`.
     Persist per country.  Value: `dataset`, `latest_week`, `indicator`, `value`.
   - **genomics:** Skip (no country dimension).
5. Return `{"pathogen_level_obs": <count>}`.

### 32.4: political_risk L2

1. Add `pipeline_store` to `__init__`.
2. Add TYPE_CHECKING + try-import.
3. Add `_persist_entities(self, data: dict, mode: str) -> dict[str, int]`.
4. Add `_persist_entities_inner`:
   - **candidates:** Persist per candidate as `person` entity keyed by
     `candidate_id`.  Value: `name`, `party`, `office`, `state`,
     `has_raised_funds`, `candidate_status`.
   - **expenditures:** Persist per unique `candidate_id` found in records as
     `person` entity.  Value: `total_support`, `total_oppose`, `total_spent`.
   - **filings:** Skip (committee-level, not person entity).
5. Return `{"campaign_finance_obs": <count>}`.

### 32.5: graph_builder update

1. Add to OBSERVATION_TYPES (maintaining alpha sort):
   `border_throughput`, `campaign_finance`, `pathogen_level`, `trade_flow`.
2. Update ENRICHMENT_DIM: `9 + len(OBSERVATION_TYPES)` = 9 + 43 = 52.
3. Remove duplicate `"instrument"` from ENTITY_TYPES if still present (seen in audit).

### 32.6: edge case tests

Cover per tool:
- No store → returns zero counts
- `_entity_id_from_key` is None → returns zero counts
- Store raises exception → caught, returns zero counts
- Empty/missing data → zero obs, no crash
- Tool-specific edge cases:
  - comtrade: unknown ISO-3 code, "World" reporter
  - transport_throughput: empty records, single border only
  - disease_surveillance: wastewater aggregates to US, outbreaks with unparseable country, genomics skipped
  - political_risk: filings skipped, candidate with no candidate_id, expenditures with missing amounts

### 32.7: validation

1. Run all tests, confirm Phase 32 tests pass.
2. Check obs type count = 43, ENRICHMENT_DIM = 52.
3. Write checkpoint.

## Edge Cases

- Comtrade ISO-3 code not in mapping → skip (log warning, return 0).
- Comtrade "World" (M49=0) partner → don't persist partner entity.
- Transport throughput records with zero volume → skip.
- Disease WHO DON entries with no parseable country → skip that entry.
- Disease ECDC records with no country_code → skip.
- Political risk candidate with empty candidate_id → skip.
- All tools: `_store is None` → return 0.  `_entity_id_from_key is None` → return 0.
- All tools: inner function throws → outer catches, returns 0, logs.

## Testing Plan

- Unit tests for `_persist_entities_inner` with mock store.
- Guard tests (no store, no entity_id_from_key).
- Exception safety tests.
- Edge case data (empty, malformed, missing fields).
- Graph builder: verify obs type count and ENRICHMENT_DIM.

## Related

- [[phase32_trade_disease_political_l2|Research]]
- [[phase32_trade_disease_political_l2|Task]]
- [[l2_expansion_roadmap]]
