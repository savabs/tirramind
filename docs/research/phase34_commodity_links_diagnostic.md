---
title: "Research: Phase 34 — Commodity Country Links + Diagnostic Sweep"
tags:
  - doc/research
  - phase/34
  - topic/l2-expansion
  - layer/surveillance
  - topic/entity-linking
---

# Feature: Phase 34 — Commodity Country Links + Diagnostic Sweep

## Context

Phases 27-33 added L2 entity persistence to all 42 data tools. Phase 34 is the
final phase of the L2 expansion roadmap. It addresses the last graph connectivity
gap — commodity futures have zero entity links — and performs a diagnostic sweep
to verify the complete entity graph is healthy.

## Current Architecture

### InstrumentDef Dataclass
```
ticker, name, asset_class, region, is_tradeable,
issuer, country, cftc_code, base_country, quote_country, protocol
```

### _persist_instrument_links: 6 Link Types
1. `tracks_issuer` — instrument → company (ETFs/stocks → issuer org)
2. `located_in` — instrument → country (when `inst.country` is set)
3. `located_in` — company → country (issuer HQ → country, deduped)
4. `fx_base_country` — instrument → country (FX base currency side)
5. `fx_quote_country` — instrument → country (FX quote currency side)
6. `tracks_protocol` — instrument → protocol (crypto → blockchain)

### Graph Builder State (Post-Phase 33)
- ENTITY_TYPES: 11
- OBSERVATION_TYPES: 45
- ENRICHMENT_DIM: 54
- Edge types are dynamic — inferred from `link_type` in DB, not hardcoded

## Observations

### The Gap: 20 Commodity Futures Have Zero Entity Links

All 20 commodity futures have `country=None` because commodities don't have a
single "issuer country" in the same sense as equities. They also have no issuer,
base_country, quote_country, or protocol fields. Result: **completely isolated
nodes** in the entity graph — the GNN cannot propagate any information to/from
these instruments via edges.

Breakdown of the 23 instruments with `country=None`:
- **20 commodity futures** — zero links of any kind (CRITICAL)
- **EMB** (EM Bond ETF) — has `issuer="BlackRock"` → gets `tracks_issuer` link
- **BTC-USD** — has `protocol="bitcoin"` → gets `tracks_protocol` link
- **ETH-USD** — has `protocol="ethereum"` → gets `tracks_protocol` link

### Exchange Location Is Factual and Unambiguous

Every commodity future in our universe trades on a US exchange:

| Ticker | Exchange | Exchange Group |
|--------|----------|---------------|
| CL=F, NG=F, RB=F, PL=F, PA=F | NYMEX | CME Group |
| GC=F, SI=F, HG=F | COMEX | CME Group |
| ZW=F, ZC=F, ZS=F, ZO=F | CBOT | CME Group |
| LE=F, HE=F | CME | CME Group |
| BZ=F | ICE Futures US | ICE |
| KC=F, CC=F, CT=F, SB=F, OJ=F | ICE Futures US | ICE |

All yfinance tickers (`=F` suffix) specifically reference the US-listed contracts.

### Design Decision: New Field vs Reusing `country`

**Option A: Set `country="US"` on commodity futures.**
- Pro: Simple, uses existing link code.
- Con: Semantically wrong — `country` means "primary market/domicile" for equities.
  Gold isn't a US product.  Conflates exchange venue with economic linkage.

**Option B: Add `primary_exchange_country` field.**
- Pro: Semantically correct, preserves `country` for actual domicile.
- Pro: Creates a distinct `exchange_country` link type — the GNN can learn
  different weights for "domiciled in" vs. "traded on exchange in".
- Con: Adds one dataclass field and ~15 lines of link code.

**Chosen: Option B.** The GNN benefits from distinguishing exchange-venue links
from domicile links.  The cost is trivial.

### Link Type: `exchange_country`

New link type `exchange_country`: instrument → country. Semantics: "this
instrument trades on an exchange in this country." Distinct from `located_in`
(domicile) and `fx_base_country`/`fx_quote_country` (currency exposure).

Naming follows the existing convention (`fx_base_country`, `fx_quote_country`).

## Risks

1. **Frozen dataclass change.** Adding a field to `InstrumentDef` with a default
   (`None`) is backwards-compatible — no constructor calls break.
2. **Test coverage.** Existing `test_instrument_universe.py` tests
   `_persist_instrument_links` — new link type needs new test cases.
3. **ENRICHMENT_DIM unchanged.** No new observation types added.
   `exchange_country` is a link, not an observation.
4. **Graph builder unchanged.** Edge types are dynamic — `exchange_country`
   links will auto-create `(instrument, exchange_country, country)` edges.

## Data Requirements

- No external data needed. Exchange locations are static metadata.
- All 20 tickers verified against CME Group / ICE Futures US product listings.

## Diagnostic Sweep Scope

A diagnostic function should report:
1. Entity count per type
2. Observation count per type
3. Link count per link type
4. Orphan entities (entities with zero links)
5. Entity types with zero observations
6. Observation types with zero instances

This enables health-checking the graph after any phase and catching regressions.

## Related

- [[phase33_org_grid_l2]] — previous phase
- [[phase34_commodity_links_diagnostic_spec]] — spec
- [[phase34_commodity_links_diagnostic]] — task
