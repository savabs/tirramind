---
title: "Checkpoint: Phase 34 Complete — Commodity Country Links + Diagnostic Sweep"
tags:
  - doc/checkpoint
  - phase/34
  - topic/l2-expansion
  - layer/surveillance
  - topic/entity-linking
---

# Checkpoint: Phase 34 Complete

**Date:** 2026-04-17
**Task:** [[phase34_commodity_links_diagnostic]]
**Spec:** [[phase34_commodity_links_diagnostic_spec]]
**Research:** [[phase34_commodity_links_diagnostic]]

## What Was Done

Phase 34 closed the last graph connectivity gap — 20 commodity futures that were completely isolated nodes — and added a reusable graph diagnostic utility.

### 1. InstrumentDef: `primary_exchange_country` Field

Added `primary_exchange_country: str | None = None` to the frozen `InstrumentDef` dataclass. Set `primary_exchange_country="US"` on all 20 commodity futures (all trade on CME Group / ICE Futures US exchanges).

**Design rationale:** Used a new field rather than setting `country` because `country` semantically means "underlying domicile" (e.g., the issuer's HQ). Exchange venue is a different relationship. The GNN can now learn separate weights for `located_in` (domicile) vs `exchange_country` (exchange venue).

### 2. `_persist_instrument_links`: 7th Link Type

Extended with `exchange_country` link type: instrument → country. This is the 7th link type in the instrument link persistence system:

| Link Type | Count | Description |
|-----------|-------|-------------|
| `tracks_issuer` | ~36 | instrument → company (ETF/stock → issuer) |
| `located_in` (inst) | ~67 | instrument → country (domicile) |
| `located_in` (issuer) | ~9 | company → country (issuer HQ, deduped) |
| `fx_base_country` | 15 | instrument → country (FX base) |
| `fx_quote_country` | 15 | instrument → country (FX quote) |
| `tracks_protocol` | 2 | instrument → protocol (crypto) |
| **`exchange_country`** | **20** | **instrument → country (exchange venue)** |

### 3. Graph Diagnostic Utility

Created `agent/models/gnn/graph_diagnostics.py` with `diagnose_graph(store)` function. Reports:
- Entity counts per type
- Observation counts per type
- Link counts per link type
- Orphan entities (zero links on either side)
- Entity types with zero observations
- Observation types with zero stored instances

### Graph State (Final L2 Expansion)

- **ENTITY_TYPES:** 11 (unchanged)
- **OBSERVATION_TYPES:** 45 (unchanged — no new obs types, only new link type)
- **ENRICHMENT_DIM:** 54 (unchanged)
- **Link types:** 7 distinct types in instrument_universe (+ all L2 tool links)
- **Commodity futures:** All 20 now connected to US country node via `exchange_country`
- **Remaining unlinked to country:** EMB (EM Bond, has issuer link), BTC-USD/ETH-USD (have protocol links)

### Test Results

- Phase 34 edge cases: **34/34 pass**
- Full regression: **3772 pass**, 1 pre-existing fail (`test_entity_linking.py::TestWhaleAlertTransactsWith::test_no_inputs_no_links`)

## L2 Expansion Roadmap: COMPLETE

All phases (27-34) are now done:

| Phase | Status | What |
|-------|--------|------|
| 27 | Done | Central bank L2 + FX two-country links |
| 28 | Done | Sovereign yield, capital flow, economic activity |
| 29 | Done | Bankruptcy, investigation, research velocity |
| 30 | Done | Crypto protocol links |
| 31 | Done | Consumer confidence, food security, internet health, migration |
| 32 | Done | Trade flow, border throughput, pathogen, campaign finance |
| 33 | Done | Regulatory velocity, grid demand |
| **34** | **Done** | **Commodity country links + diagnostic utility** |

## Files Modified

- `agent/tools/instrument_universe.py` — `primary_exchange_country` field + `exchange_country` link
- `agent/models/gnn/graph_diagnostics.py` — **NEW** diagnostic utility

## Files Created

- `[[phase34_commodity_links_diagnostic]]`
- `[[phase34_commodity_links_diagnostic_spec]]`
- `[[phase34_commodity_links_diagnostic]]`
- `tests/test_phase34_commodity_links.py` (34 tests)

## Next Steps

The L2 expansion roadmap is fully complete. Potential next directions:
- GNN retrain with the expanded graph (all commodity futures now connected)
- Full graph diagnostic sweep on a live pipeline database
- Phase 35+ planning based on GNN attention analysis
- Edge decay monitoring for the new link types

## Related

- [[phase34_commodity_links_diagnostic]] — task
- [[phase34_commodity_links_diagnostic_spec]] — spec
- [[phase33_org_grid_l2]] — previous phase
