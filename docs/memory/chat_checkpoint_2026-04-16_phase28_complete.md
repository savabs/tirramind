---
title: "Checkpoint: Phase 28 Complete — Country Node Macro Enrichment"
tags:
  - doc/checkpoint
  - phase/28
  - topic/entity-linking
  - topic/graph-connectivity
  - layer/surveillance
  - layer/world-model
---

# Checkpoint: Phase 28 Complete

**Date:** 2026-04-16
**Task:** [[phase28_country_macro_enrichment]]
**Spec:** [[phase28_country_macro_enrichment_spec]]

## What Was Done

Phase 28 added L2 persistence to three macro tools — sovereign_debt, capital_flows, and global_pmi — so country entity nodes now receive 6 observation types total (up from 3 after Phase 27).

### 1. sovereign_debt L2 (28.1)
- `pipeline_store` kwarg added to `__init__` (line ~341)
- `_persist_entities()` / `_persist_entities_inner()` pattern with None guard + try/except
- Observation type: `sovereign_yield`
- Country mapping from `_COUNTRY_ENDPOINTS` keys (US, DE, GB, JP, AU, CA)
- Entity ID via `entity_id_from_key("country", iso2)`

### 2. capital_flows L2 (28.2)
- `pipeline_store` kwarg added to `__init__` (line ~302)
- Same persistence pattern
- Observation type: `capital_flow`
- Country mapping via `HOLDINGS_COUNTRY_MAP` and `RESERVES_COUNTRY_MAP`
- Covers US, JP, CN, GB, CA, CH, EU, KR, BR holders + US, CN, JP, EU, CH, RU, IN, KR, BR reserve holders

### 3. global_pmi L2 (28.3)
- `pipeline_store` kwarg added to `__init__` (line ~206)
- Same persistence pattern
- Observation type: `economic_activity`
- `ISO3_TO_ISO2` mapping covers 44 countries (line ~127)
- Three OECD indicator families: CLI, BCI, CCI

### 4. Graph Builder (28.4)
- 3 new entries in `OBSERVATION_TYPES`: `capital_flow`, `economic_activity`, `sovereign_yield`
- Total obs types: 29 → 32
- `ENRICHMENT_DIM`: 38 → 41 (9 base stats + 32 obs_type_dist)

### 5. Tests (28.5 + 28.6)
- `tests/test_sovereign_debt_edge.py`: 18 L2 persistence tests added
- `tests/test_capital_flows_edge.py`: 14 L2 persistence tests added
- `tests/test_global_pmi_edge.py`: 12 L2 persistence tests added
- `tests/test_graph_builder_expanded.py`: obs type count 29→32, Phase 28 membership check
- `tests/test_phase28_diagnostic.py`: NEW — 14 integration tests

### 6. Stale Count Fixes (28.7)
During regression, several tests had hardcoded counts that became stale:
- `tests/test_finra_short_volume_edge.py`: tool count 47→60, arm count 35→48
- `tests/test_sovereign_debt_edge.py`: tool count 47→60, arm count 35→48
- `tests/test_surprise.py`: obs type tensor dimension 18→32 (4 tests in TestObsTypeSurprise + 2 in TestCompositeSurprise). Changed to use `len(OBSERVATION_TYPES)` to be resilient to future additions.

### 7. Regression (28.7)
- **9054 passed, 0 failed, 4 skipped** (excluding known pre-existing failures)
- Runtime: ~15 minutes

## Files Modified (Phase 28 Only)

| File | Change |
|------|--------|
| `agent/tools/sovereign_debt.py` | Added `PipelineStore` support, `_persist_entities()`, obs type `sovereign_yield` |
| `agent/tools/capital_flows.py` | Added `PipelineStore` support, `_persist_entities()`, obs type `capital_flow`, country maps |
| `agent/tools/global_pmi.py` | Added `PipelineStore` support, `_persist_entities()`, obs type `economic_activity`, `ISO3_TO_ISO2` |
| `agent/models/gnn/graph_builder.py` | 3 new obs types, `ENRICHMENT_DIM` 38→41 |
| `tests/test_sovereign_debt_edge.py` | 18 new L2 tests + stale count fixes |
| `tests/test_capital_flows_edge.py` | 14 new L2 tests |
| `tests/test_global_pmi_edge.py` | 12 new L2 tests |
| `tests/test_graph_builder_expanded.py` | Updated obs type count 29→32 |
| `tests/test_phase28_diagnostic.py` | NEW — 14 integration tests |
| `tests/test_finra_short_volume_edge.py` | Stale count fixes (47→60, 35→48) |
| `tests/test_surprise.py` | Obs type dim 18→`len(OBSERVATION_TYPES)`, all 6 affected tests fixed |

## Known Pre-Existing Failures (Not Phase 28)

All confirmed pre-existing via `git log` — last modified in "Initial code import" or "Add GNN fusion":

| Test File | Failure | Last Modified |
|-----------|---------|---------------|
| `tests/test_feature_generation_dag.py` | expects 6 features, gets 17 | Initial code import |
| `tests/test_form144_edge.py` | shares_to_sell 4154≠4454 | Initial code import |
| `tests/test_pipeline_registry.py` (5 tests) | DAG node count 6≠7 + related | Initial code import |
| `tests/test_gnn_trainer_19a.py` | model persistence | Add GNN fusion |
| `tests/test_world_model_discovery.py` (2 tests) | summary counts | Add GNN fusion |
| `tests/test_world_model_update_fitting.py` | default params | Add GNN fusion |

## Current State of the Entity Graph

**Country nodes now have 6 observation types:**
1. `cb_balance_sheet` (Phase 27 — central bank)
2. `cb_policy_rate` (Phase 27 — central bank)
3. `sovereign_yield` (Phase 28 — sovereign_debt)
4. `capital_flow` (Phase 28 — capital_flows)
5. `economic_activity` (Phase 28 — global_pmi)
6. Plus any cross-links from FX instruments (Phase 27)

**Key constants:**
- `OBSERVATION_TYPES`: 32 entries
- `ENRICHMENT_DIM`: 41 (9 base stats + 32 obs_type_dist)
- Tool registry: 60 tools, 48 bandit arms

## What Remains Starved

Per [[starved_class_audit]] (from Phase 27 checkpoint), remaining priorities:

1. **Commodity futures** — have CFTC L2 but lack physical signals (vessel, weather, satellite)
2. **Equity ETFs / sector ETFs** — linked to issuers/countries but lack earnings/fund-flow obs
3. **Fixed income** — treasury_receipts is L1 aggregate; no entity-level bond obs
4. **Vol instruments** — VIX/VVIX/MOVE are index-level; no entity-level positioning
5. **Crypto** — BTC whale_alert is L2; no ETH/alt entity observations

**Country nodes are no longer starved** — they now have 6 observation types + FX two-country wiring.

## Resume Instructions

> Phase 28 is complete. All steps done, all regression green (9054 passed).
> Task file at `[[phase28_country_macro_enrichment]]` marked `status/done`.
>
> **To continue:** Consult [[starved_class_audit]] and the Phase 27 checkpoint's starved-class ranking to pick Phase 29. The most natural next move is either:
> - Commodity futures physical-signal enrichment (vessel tracking, weather, satellite)
> - Equity ETF/sector fund-flow observations
> - Or run the GNN-guided evaluation to let attention weights identify the real gap
>
> **Uncommitted changes:** All Phase 28 work is in the working tree, not yet committed. Consider committing before starting Phase 29.

## Related

- [[phase28_country_macro_enrichment]]
- [[phase28_country_macro_enrichment_spec]]
- [[phase27_fx_country_monetary_linking]]
- [[chat_checkpoint_2026-04-16_phase27_complete]]
- [[starved_class_audit]]
