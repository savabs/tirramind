---
title: "Checkpoint: Phase 25 Cross-Domain Entity Linking Complete"
tags:
  - doc/checkpoint
  - phase/25
  - topic/gnn-expansion
  - topic/entity-linking
  - layer/world-model
  - layer/surveillance
---

# Checkpoint: Phase 25 Cross-Domain Entity Linking Complete

**Date:** 2025-01-22
**Task:** [[phase25_cross_domain_entity_linking]]
**Spec:** [[phase25_cross_domain_entity_linking_spec]]
**Research:** [[phase25_gnn_diagnostic]]

## Summary

Phase 25 connected the previously-isolated instrument subgraph (90 instruments from Phase 24) to the broader entity graph through three upgrade paths: instrument metadata enrichment with explicit issuer/country/CFTC links, CFTC L2 entity persistence, and Polymarket L2 topic/wallet persistence. Instruments now have non-zero graph degree and the HetTGN can propagate attention across instrument → company → country, cftc_contract → instrument, and topic/wallet domains.

## What Changed

### Step 25.1: InstrumentDef Metadata Enrichment
- `InstrumentDef` dataclass extended with `issuer`, `country`, `cftc_code` optional fields
- All 90 instruments enriched: ETFs have issuers (BlackRock, Vanguard, State Street, Invesco, VanEck, USCF, ProShares), futures/FX have ISO country codes
- 18 CFTC contract market codes verified against live cftc.gov data (e.g., CL=F→06765A, GC=F→088691)
- New helper: `cftc_code_to_ticker()` maps CFTC codes back to instrument tickers

### Step 25.2: Link Persistence
- `_persist_instrument_links(store)` creates `tracks_issuer` (instrument→company) and `located_in` (company→country) links
- Wired into `ingest_daily_prices()` — links created on every daily run, idempotent via INSERT OR IGNORE
- 27 new tests, 73 total instrument tests green

### Step 25.3: CFTC L2 Entity Upgrade
- `cftc.py` now accepts optional `PipelineStore` and persists `cftc_contract` entities with `futures_positioning` observations
- Links CFTC contracts to instruments via `cftc_tracks` where mapping exists
- `_report_date_to_ts()` uses `calendar.timegm()` for UTC correctness
- 33 new tests covering persistence, dedup, edge cases, DB errors

### Step 25.4: Polymarket L2 Entity Upgrade
- `polymarket.py` persists `topic` entities with `market_probability` observations (yes_price, volume, liquidity, spread)
- `polymarket_whales.py` persists `wallet` entities with `whale_trade` observations (composite_score, accuracy, volume)
- Case-insensitive wallet dedup, empty/whitespace slug filtering
- 28 new tests covering both modules

### Step 25.5: Graph Diagnostics + Integration
- 26 new cross-domain integration tests validating full store→graph builder flow
- Confirmed: instrument→company→country chain, cftc_contract→instrument edges, topic/wallet nodes in graph
- Instruments now have **non-zero degree** (was 0.0 in Phase 24 diagnostic baseline)
- All 20 GNN training/diagnostic tests pass including the new entity types

### Graph Builder Updates
- ENTITY_TYPES: 11 (added `cftc_contract`)
- OBSERVATION_TYPES: 27 (added `futures_positioning`, `market_probability`, `whale_trade`)
- ENRICHMENT_DIM: 36 (= 9 + 27)
- BASE_FEAT_DIM: 14 (= 11 + 3)
- SEED_ENTITY_TYPES in `entity.py`: 11 (added `cftc_contract`)

## Files Modified

| File | Change |
|------|--------|
| `agent/tools/instrument_universe.py` | InstrumentDef enrichment, `cftc_code_to_ticker()`, `_persist_instrument_links()` |
| `agent/tools/cftc.py` | PipelineStore support, L2 entity persistence, `_report_date_to_ts()` |
| `agent/tools/polymarket.py` | PipelineStore support, topic entity persistence |
| `agent/tools/polymarket_whales.py` | Wallet entity persistence |
| `agent/models/gnn/graph_builder.py` | New entity/observation types, updated dimensions |
| `agent/pipeline/entity.py` | Added `cftc_contract` to SEED_ENTITY_TYPES |

## Files Created

| File | Tests |
|------|-------|
| `tests/test_cftc_l2.py` | 33 tests |
| `tests/test_polymarket_l2.py` | 28 tests |
| `tests/test_cross_domain_links.py` | 26 tests |

## Test Results

- **203 tests** across Phase 25 test files: all green
- **371 tests** across all graph/entity test files: all green
- **20 tests** GNN training/diagnostic integration: all green

## Remaining Starved Instrument Classes

Based on the current link coverage, these instrument classes still have limited cross-domain connectivity:
- **FX pairs** (EURUSD=X, GBPUSD=X, etc.): Have country but no issuer entity. Could benefit from central bank entities.
- **Crypto ETFs** (IBIT, BITO, etc.): Have issuer but no direct on-chain entity links. Future: link to protocol/wallet entities.
- **Broad market ETFs** (VTI, VXUS, etc.): Have issuer/country but no sector or component links.
- **Instruments without CFTC mapping**: Only 18 of ~30 futures have verified CFTC codes. Remaining commodities (LH, FC, HO) need mapping.

## Next Phase Candidates

1. **Insider filings L2 → instrument links** — SEC EDGAR insider trades can link person entities directly to instrument entities via `insider_trade_target` edges. High-value because it creates person→instrument paths.
2. **GNN-guided evaluation** — Train GNN on the Phase 25 graph, measure attention weights and pattern recovery by entity type to determine which connections are most valuable.
3. **Cross-entity pattern enrichment** — Now that multiple entity types are connected, the cross-entity pattern crystallizer can find more meaningful patterns across domains.

## Related

- [[phase25_cross_domain_entity_linking]]
- [[phase25_cross_domain_entity_linking_spec]]
- [[phase25_gnn_diagnostic]]
- [[e2e_global_integration]]
- [[entity_linking_layer]]
