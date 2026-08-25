---
title: "Checkpoint: Phase 27 Complete — FX Country Wiring + Central Bank L2"
tags:
  - doc/checkpoint
  - phase/27
  - topic/fx
  - topic/central-bank
  - topic/entity-linking
  - layer/surveillance
  - layer/world-model
---

# Checkpoint: Phase 27 Complete

**Date:** 2026-04-16
**Task:** [[phase27_fx_country_monetary_linking]]
**Spec:** [[phase27_fx_country_monetary_linking_spec]]

## What Was Done

Phase 27 connected FX instruments and country-sensitive assets to causal monetary-state signal through two mechanisms:

### 1. FX Two-Country Wiring (27.1 + 27.2)
- Extended `InstrumentDef` with `base_country` / `quote_country` fields
- All 15 FX pairs now carry explicit two-country metadata
- `_persist_instrument_links()` creates `fx_base_country` + `fx_quote_country` link types
- 26 tests verify metadata correctness and link persistence

### 2. Central Bank L2 Observations (27.3 + 27.4)
- `central_bank_balance` tool upgraded with `PipelineStore` support
- Two observation families: `cb_balance_sheet` and `cb_policy_rate`
- Observations land on country entity nodes via deterministic `CB_TO_COUNTRY` mapping (7 CBs → 7 countries)
- `OBSERVATION_TYPES` in graph builder updated (27 → 29), `ENRICHMENT_DIM` updated (36 → 38)
- 26 L2 persistence tests + 4 graph builder tests added

### 3. Integration Diagnostics (27.5)
- 11 end-to-end integration tests verify:
  - FX instruments gain dual-country links (EURUSD→EU+US, EURGBP→EU+GB)
  - CB observations survive store → graph builder flow
  - Full path: USDJPY → fx_base_country → US ← cb_balance_sheet(fed) confirmed

### 4. Regression (27.6)
- 292/292 tests pass across all touched suites
- Two stale count assertions fixed (tool registry: 60, bandit arms: 48)

## Files Modified

| File | Change |
|------|--------|
| `agent/tools/instrument_universe.py` | Added `base_country`/`quote_country` to `InstrumentDef`, FX pair metadata, `_persist_instrument_links()` FX country logic |
| `agent/tools/central_bank_balance.py` | Added `PipelineStore` support, `CB_TO_COUNTRY`, `_persist_entities()`/`_persist_entities_inner()` |
| `agent/models/gnn/graph_builder.py` | Added `cb_balance_sheet` + `cb_policy_rate` to `OBSERVATION_TYPES`, `ENRICHMENT_DIM` 36→38 |
| `tests/test_instrument_universe.py` | 26 new Phase 27 tests |
| `tests/test_central_bank_balance_edge.py` | 26 L2 persistence tests + fixed stale count assertions |
| `tests/test_graph_builder_expanded.py` | Updated obs type count assertion (27→29) |
| `tests/test_phase27_diagnostic.py` | NEW — 11 integration diagnostic tests |

## What Remains Starved

Per [[starved_class_audit]], the remaining starved instrument classes after Phase 27:

1. **Commodity futures** — have CFTC L2 but lack physical-world observation channels (vessel tracking, weather, satellite)
2. **Equity ETFs / sector ETFs** — linked to issuers/countries but lack earnings-level or fund-flow observations
3. **Fixed income** — treasury_receipts is L1 aggregate; no entity-level bond observations
4. **Vol instruments** — VIX/VVIX/MOVE are index-level; no entity-level positioning data
5. **Crypto** — BTC whale_alert is L2; no ETH/alt entity observations yet

FX instruments are no longer starved — they now have two-country graph connectivity + CB monetary-state observations on country nodes.

## Resume Instructions

> Phase 27 is complete. Consult [[starved_class_audit]] and [[l2_expansion_roadmap]] for the next expansion priority. The next natural phase would address the most-starved instrument class from the audit.

## Related

- [[phase27_fx_country_monetary_linking]]
- [[phase27_fx_country_monetary_linking_spec]]
- [[starved_class_audit]]
- [[l2_expansion_roadmap]]
- [[chat_checkpoint_2026-04-15_full_picture_logout]]
- [[quant_training_ground]]
