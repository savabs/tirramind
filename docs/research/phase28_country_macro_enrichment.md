---
title: "Research: Phase 28 — Country Node Macro Enrichment"
tags:
  - doc/research
  - phase/28
  - topic/entity-linking
  - topic/graph-connectivity
  - topic/sovereign-debt
  - topic/capital-flows
  - topic/pmi
  - layer/surveillance
  - layer/world-model
---

# Research: Phase 28 — Country Node Macro Enrichment

## Purpose

After Phase 27 delivered `cb_balance_sheet` and `cb_policy_rate` on country nodes (2 obs types), country is still under-observed. This phase adds 3 more country observation types from existing L1 tools: bond yields, capital flows, and economic activity indicators.

## Current Architecture

### Tools to Upgrade

| Tool | File | Current State | Country Data Available |
|------|------|---------------|----------------------|
| `sovereign_debt` | `agent/tools/sovereign_debt.py` | L1, no PipelineStore | US/EU/JP/UK bond yields by maturity |
| `capital_flows` | `agent/tools/capital_flows.py` | L1, no PipelineStore, FRED-based | Holdings (JP/CN/UK), flows, reserves (US/CN/JP/SA/IN) |
| `global_pmi` | `agent/tools/global_pmi.py` | L1, no PipelineStore, OECD SDMX | CLI/BCI/CCI for 40+ countries (ISO-3 codes) |

### L2 Pattern (from Phase 27 central_bank_balance)

The proven pattern for L2 country-node persistence:
1. Add `TYPE_CHECKING` import for `PipelineStore`
2. Add `_entity_id_from_key` import with try/except
3. Accept `pipeline_store: PipelineStore | None = None` in `__init__`
4. Call `_persist_entities()` after successful `execute()` with `result.data`
5. `_persist_entities()` guards on `self._store is None` and wraps inner in try/except
6. `_persist_entities_inner()` does the actual work:
   - Map tool-specific keys → ISO-2 country codes
   - `entity_id_from_key("country", code)` → deterministic SHA-256[:16]
   - `store.register_entity(entity_type="country", canonical_name=code, entity_id=eid)`
   - `store.store_entity_observation(entity_id=eid, source_tool=..., observed_at=time.time(), observation_type=..., value={...}, depth_level=2)`
7. Return counts dict

### Country Code Mapping Requirements

**sovereign_debt:** Multi-source. Needs per-mode mapping:
- `us_yields` → `US`
- `eu_yields` → ECB country codes are ISO-2 already (DE, FR, IT, ES, GR, PT, NL, BE, AT, IE, FI, etc.)
- `jp_yields` → `JP`
- `uk_gilts` → `GB`
- `spreads` → same as eu_yields keys + US

**capital_flows:** FRED series keys need mapping:
- Holdings: `japan→JP`, `china→CN`, `uk→GB` (total is aggregate — skip for L2)
- Reserves: `china_reserves→CN`, `japan_reserves→JP`, `saudi_reserves→SA`, `india_reserves→IN`, `total_reserves_ex_gold` → skip (aggregate)
- Flows: aggregate US-level — all map to `US`

**global_pmi:** OECD uses ISO-3 codes. Need ISO-3 → ISO-2 mapping:
- `USA→US`, `GBR→GB`, `DEU→DE`, `FRA→FR`, `JPN→JP`, `CHN→CN`, `KOR→KR`, `AUS→AU`, `CAN→CA`, `ITA→IT`, `ESP→ES`, `BRA→BR`, `IND→IN`, `MEX→MX`, `TUR→TR`, `ZAF→ZA`, etc.
- Aggregates (`OECD`, `G-7`, `EA19`, `G-20`) → skip for L2 (no country entity)

### New Observation Types

| Obs Type | Placed On | Source Tool | Value Fields |
|----------|-----------|-------------|-------------|
| `sovereign_yield` | country | sovereign_debt | `{source, maturity, yield_pct, curve_2s10s, date}` |
| `capital_flow` | country | capital_flows | `{series, latest_value, mom_change_pct, flow_type}` |
| `economic_activity` | country | global_pmi | `{indicator, value, period, regime, momentum_6m}` |

### Graph Builder Impact

- OBSERVATION_TYPES: 29 → 32 (+3)
- ENRICHMENT_DIM: 38 → 41 (9 base stats + 32 obs_type_dist)

## Risks

- **sovereign_debt has 5 modes with different country granularity.** Need per-mode dispatch in `_persist_entities`.
- **capital_flows uses FRED series names, not country codes.** Need explicit mapping dict.
- **global_pmi uses ISO-3 codes.** Need ISO-3→ISO-2 mapping (this is well-defined).
- **capital_flows holdings keys may map multiple series to same entity.** `total` should be skipped.
- **sovereign_debt spreads mode calls eu_yields internally.** Don't double-persist.

## Related

- [[phase28_country_macro_enrichment_spec]]
- [[phase28_country_macro_enrichment]]
- [[starved_class_audit]]
- [[l2_expansion_roadmap]]
- [[phase27_fx_country_monetary_linking]]
