---
title: "Spec: Phase 28 — Country Node Macro Enrichment"
tags:
  - doc/spec
  - phase/28
  - topic/entity-linking
  - topic/graph-connectivity
  - layer/surveillance
  - layer/world-model
---

# Spec: Phase 28 — Country Node Macro Enrichment

## Goal

Add PipelineStore support to `sovereign_debt`, `capital_flows`, and `global_pmi` so they persist country-level observations onto country entity nodes. After this phase, country nodes receive 6 observation types total (up from 3 after Phase 27).

## Files Affected

| File | Change |
|------|--------|
| `agent/tools/sovereign_debt.py` | Add PipelineStore init, country mapping, `_persist_entities`/`_persist_entities_inner` |
| `agent/tools/capital_flows.py` | Add PipelineStore init, country mapping, `_persist_entities`/`_persist_entities_inner` |
| `agent/tools/global_pmi.py` | Add PipelineStore init, ISO-3→ISO-2 mapping, `_persist_entities`/`_persist_entities_inner` |
| `agent/models/gnn/graph_builder.py` | Add 3 obs types to `OBSERVATION_TYPES`, update `ENRICHMENT_DIM` 38→41 |
| `tests/test_sovereign_debt_edge.py` | Add L2 persistence tests |
| `tests/test_capital_flows_edge.py` | Add L2 persistence tests |
| `tests/test_global_pmi_edge.py` | Add L2 persistence tests |
| `tests/test_graph_builder_expanded.py` | Update obs type count assertion |
| `tests/test_phase28_diagnostic.py` | NEW — integration diagnostics |

## Implementation Steps

### 28.1: L2 upgrade `sovereign_debt`
- Add `TYPE_CHECKING` import for PipelineStore
- Add `_entity_id_from_key` try/except import
- Add `pipeline_store` kwarg to `__init__`
- Add `SOVEREIGN_COUNTRY_MAP` for mode→country resolution:
  - `us_yields` → `{"US": "US"}`
  - `eu_yields` → identity (ECB codes are ISO-2)
  - `jp_yields` → `{"JP": "JP"}`
  - `uk_gilts` → `{"GB": "GB"}`
  - `spreads` → skip (derived mode, would double-persist)
- Add `_persist_entities(data, mode)` and `_persist_entities_inner(data, mode)`
- Obs type: `sovereign_yield`
- Value schema: `{source: "us_treasury"|"ecb"|"mof"|"dmo", maturity: str|None, yield_pct: float, curve_2s10s: float|None, date: str}`
- For `us_yields`: persist ONE observation per call with latest 10Y yield + curve metrics
- For `eu_yields`: persist ONE observation per country with latest yield
- For `jp_yields`: persist ONE observation with latest 10Y yield
- For `uk_gilts`: persist ONE observation with latest auction yield
- Call `_persist_entities()` after successful execute in the dispatcher

### 28.2: L2 upgrade `capital_flows`
- Same import pattern as 28.1
- Add `pipeline_store` kwarg to `__init__`
- Add `HOLDINGS_COUNTRY_MAP`: `{"japan": "JP", "china": "CN", "uk": "GB"}` (skip `total`)
- Add `RESERVES_COUNTRY_MAP`: `{"china_reserves": "CN", "japan_reserves": "JP", "saudi_reserves": "SA", "india_reserves": "IN"}` (skip `total_reserves_ex_gold`)
- Flows mode: all map to `US` (aggregate US capital flows)
- Obs type: `capital_flow`
- Value schema: `{flow_type: "holdings"|"flows"|"reserves", series: str, latest_value: float, mom_change_pct: float|None, stress: bool|None}`
- Call `_persist_entities()` after successful execute

### 28.3: L2 upgrade `global_pmi`
- Same import pattern
- Add `pipeline_store` kwarg to `__init__`
- Add `ISO3_TO_ISO2` mapping (comprehensive for all `_KNOWN_CODES`)
- Skip aggregate codes (OECD, G-7, EA19, G-20) — no country entity for these
- Obs type: `economic_activity`
- Value schema: `{indicator: "cli"|"bci"|"cci", value: float, period: str, regime: str|None, momentum_6m: float|None}`
- Persist ONE observation per country per call with latest values
- Call `_persist_entities()` after successful execute

### 28.4: Register obs types in graph builder
- Add `capital_flow`, `economic_activity`, `sovereign_yield` (alphabetical) to `OBSERVATION_TYPES`
- Update `ENRICHMENT_DIM`: 9 base stats + 32 obs_type_dist = 41

### 28.5: Edge case tests
- Per tool: no-store guard, exception non-fatal, country mapping correctness, depth_level=2, obs type correctness, idempotent entity registration, skip aggregates, mode-specific dispatch
- Graph builder: obs type count = 32, ENRICHMENT_DIM = 41

### 28.6: Integration diagnostics
- Verify country nodes receive observations from all 3 new tools
- Verify path: USDJPY → fx_base_country → US ← sovereign_yield(us_treasury) confirmed
- Verify country node obs type count ≥ 6

### 28.7: Regression + checkpoint

## Edge Cases

- Tool returns no data → `_persist_entities` returns 0 counts, no error
- PipelineStore is None → silent skip
- Entity module unavailable → silent skip
- Partial country data (some countries fail) → persist what succeeded
- Global PMI aggregate codes → must be skipped, not persisted
- Capital flows `total` series → must be skipped
- Sovereign debt `spreads` mode → must be skipped (derived from eu_yields)

## Testing Plan

Each tool gets ~10 L2-specific tests covering the cases above. Graph builder gets count assertion update. Integration diagnostic file (~8 tests) verifies end-to-end flow.

## Related

- [[phase28_country_macro_enrichment]]
- [[phase28_country_macro_enrichment]]
- [[starved_class_audit]]
- [[l2_expansion_roadmap]]
- [[phase27_fx_country_monetary_linking_spec]]
