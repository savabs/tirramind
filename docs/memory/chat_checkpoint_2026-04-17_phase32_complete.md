---
title: "Checkpoint: Phase 32 Complete — Trade + Disease + Political L2"
tags:
  - doc/checkpoint
  - phase/32
  - topic/l2-expansion
  - layer/surveillance
---

# Checkpoint: Phase 32 Complete

**Date:** 2026-04-17
**Task:** [[phase32_trade_disease_political_l2]]
**Spec:** [[phase32_trade_disease_political_l2_spec]]
**Research:** [[phase32_trade_disease_political_l2]]

## What Was Done

Phase 32 added L2 entity persistence to 4 tools, expanding the GNN entity graph:

| Tool | Obs Type | Entity Type | Key Details |
|------|----------|-------------|-------------|
| `comtrade` | `trade_flow` | `country` | ISO3→ISO2 mapping (34 countries), persists on reporter side |
| `transport_throughput` | `border_throughput` | `country` | Border→country mapping (US-Canada→US/CA, US-Mexico→US/MX) |
| `disease_surveillance` | `pathogen_level` | `country` | 3 modes: wastewater→US, outbreaks→country_parsed, eu→country_code; genomics skipped |
| `political_risk` | `campaign_finance` | `person` | candidates→per candidate_id, expenditures→aggregated per candidate_id, filings skipped |

### Graph Builder State

- **OBSERVATION_TYPES:** 43 (was 39)
- **ENRICHMENT_DIM:** 52 (was 48)
- **ENTITY_TYPES:** 11 (unchanged)

### Test Results

- Phase 32 edge cases: **48/48 pass**
- Full regression: **3772 pass**, 1 pre-existing fail (`test_entity_linking.py::TestWhaleAlertTransactsWith::test_no_inputs_no_links`)

## Files Modified

- `agent/tools/comtrade.py` — L2 persistence added
- `agent/tools/transport_throughput.py` — L2 persistence added
- `agent/tools/disease_surveillance.py` — L2 persistence added
- `agent/tools/political_risk.py` — L2 persistence added
- `agent/models/gnn/graph_builder.py` — 4 new obs types, ENRICHMENT_DIM 48→52
- `tests/test_phase32_l2.py` — 48 edge case tests

## What's Next

Per [[l2_expansion_roadmap]], the next phases are:

- **Phase 33:** Real Estate + Weather + Grid L2 (property_tax, building_permit, real_estate_listing, weather_event, grid_status)
- **Phase 34:** Government + Compliance + Lobbying L2 (contract_award, regulatory_filing, lobbying_activity)
- **Phase 35:** Ship Tracking L2 (already deep entity coverage, may need port_call refinement)

## Known Issues

- Pre-existing test failure in `test_entity_linking.py` unrelated to Phase 32
- `disease_surveillance` outbreaks mode uses first 2 chars of `country_parsed` as ISO-2 key — imperfect but functional approximation
- `comtrade` persists only reporter side to avoid double-counting bilateral trade

## Related

- [[l2_expansion_roadmap]]
- [[quant_training_ground]]
- [[phase31_country_signal_l2_spec]]
