---
title: "Spec: Phase 31 — Remaining Country Signals"
tags:
  - doc/spec
  - phase/31
  - topic/entity-linking
  - topic/consumer-sentiment
  - topic/food-security
  - topic/internet-infrastructure
  - topic/migration
  - layer/surveillance
  - layer/world-model
---

# Spec: Phase 31 — Remaining Country Signals

## Goal

Add L2 country-node persistence to `consumer_sentiment`, `food_security`, `internet_outages`, and `migration_flows`. Country nodes gain four new observation types. `OBSERVATION_TYPES` grows 35 → 39 and `ENRICHMENT_DIM` grows 44 → 48.

## Files Affected

| File | Change |
|------|--------|
| `agent/tools/consumer_sentiment.py` | Add PipelineStore wiring + `consumer_confidence` persistence |
| `agent/tools/food_security.py` | Add PipelineStore wiring + `food_security` persistence |
| `agent/tools/internet_outages.py` | Add PipelineStore wiring + `internet_disruption` persistence |
| `agent/tools/migration_flows.py` | Add PipelineStore wiring + `migration_pressure` persistence |
| `agent/models/gnn/graph_builder.py` | Add 4 obs types to `OBSERVATION_TYPES`, update `ENRICHMENT_DIM` |
| `tests/test_graph_builder_expanded.py` | Update observation-type membership and count assertions |
| `tests/test_phase28_diagnostic.py` | Update stale `ENRICHMENT_DIM` assertion |
| `tests/test_phase29_diagnostic.py` | Update stale count and `ENRICHMENT_DIM` assertions |
| `tests/test_phase31_country_signal_l2.py` | NEW — L2 edge-case tests for all four tools |
| `tests/test_phase31_diagnostic.py` | NEW — integration diagnostics |

## Implementation Steps

### 31.1: L2 upgrade `consumer_sentiment`

- Add `pipeline_store` to constructor and `entity_id_from_key` guard imports
- Persist `consumer_confidence` observations
- `eu_confidence`: persist one observation per ISO-2 country, skip `EU27_2020` and `EA20`
- `us_sentiment`: persist one observation on `US`
- `inflation_reality`: persist one observation on `US`
- Value payload keeps mode-specific fields (`latest`, `mom_change`, `trend`, `expectation_gap`, etc.)

### 31.2: L2 upgrade `food_security`

- Add the standard L2 persistence pattern
- Persist `food_security` on the requested country node
- Skip aggregate `WLD`
- Carry through mode, indicator, latest value/year, YoY change, trend, stress, and vulnerability fields

### 31.3: L2 upgrade `internet_outages`

- Add the standard L2 persistence pattern
- Persist `internet_disruption` when a concrete ISO-2 country is present
- `censorship`: anomaly/confirmed/failure metrics
- `network_health`: disconnect-rate and ASN metrics
- `outage_detection`: aggregate anomaly/failure rates
- Skip `ALL` / empty-country global scans

### 31.4: L2 upgrade `migration_flows`

- Add the standard L2 persistence pattern
- Add ISO-3 → ISO-2 normalization helper for UNHCR modes
- Persist `migration_pressure` for:
  - `displacement`
  - `asylum`
  - `remittances`
- Skip global aggregate runs with no country

### 31.5: Graph builder update

- Add these observation types in alphabetical order:
  - `consumer_confidence`
  - `food_security`
  - `internet_disruption`
  - `migration_pressure`
- Update `ENRICHMENT_DIM` from 44 → 48

### 31.6: Edge-case test suite

Create `tests/test_phase31_country_signal_l2.py` covering:

- no-store guard
- non-fatal persistence exception path
- aggregate skipping (`EU27_2020`, `EA20`, `WLD`, `ALL`, missing country)
- deterministic country-ID mapping
- mode-specific payload persistence for all four tools

### 31.7: Integration diagnostics

Create `tests/test_phase31_diagnostic.py` covering:

- new observation types present in graph builder constants
- country node receives each new Phase 31 observation family
- country node can hold Phase 27/28/29/31 observations together
- `ENRICHMENT_DIM == 48`, `len(OBSERVATION_TYPES) == 39`

## Edge Cases

- EU aggregate geo codes must not create country entities
- `food_security(country="WLD")` must not persist
- `internet_outages(mode="censorship")` without country must not persist
- unknown ISO-3 codes in migration payloads must be skipped, not crash
- empty tool results must produce zero observation counts
- persistence failures remain non-fatal to tool execution

## Testing Plan

- Focused unit/edge tests for the new L2 hooks
- Graph-builder constant updates
- Phase 31 integration diagnostics
- Targeted regression across touched test files

## Related

- [[phase31_remaining_country_signals]]
- [[phase31_remaining_country_signals_spec]]
- [[phase28_country_macro_enrichment_spec]]
- [[starved_class_audit]]
- [[l2_expansion_roadmap]]