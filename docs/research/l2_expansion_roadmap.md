---
title: "Roadmap: L2 Expansion — Starved Class Remediation"
tags:
  - doc/research
  - phase/27
  - phase/28
  - phase/29
  - phase/30
  - phase/31
  - phase/32
  - topic/entity-linking
  - topic/graph-connectivity
  - layer/surveillance
  - layer/world-model
---

# L2 Expansion Roadmap — Starved Class Remediation

**Source:** [[starved_class_audit]]
**Governing umbrella:** [[quant_training_ground]]

This document is the single authoritative phase-by-phase plan for closing every identified gap in the entity graph. Each phase is scoped to ~2-4 hours of focused work with a clear verification gate.

---

## Phase Map Overview

| Phase | Name | Target Class | Tools | New Obs Types | Effort |
|---|---|---|---|---|---|
| **27** | FX Country Wiring + Central Bank L2 | country, fx | instrument_universe, central_bank_balance | monetary_balance, policy_rate | small |
| **28** | Country Node Enrichment (Macro) | country | sovereign_debt, capital_flows, global_pmi | sovereign_yield, capital_flow, economic_activity | small |
| **29** | Company + Investigative L2 | company, person | bankruptcy_court, foia_requests, academic_preprints | bankruptcy_status, investigation_signal, research_velocity | small-medium |
| **30** | Crypto Islands + Cross-Domain Linking | protocol, wallet, instrument | whale_alert, instrument_universe | (links only, no new obs types) | small |
| **31** | Remaining Country Signals | country | consumer_sentiment, food_security, internet_outages, migration_flows | consumer_confidence, food_security_index, internet_health, migration_flow | small |
| **32** | Trade + Disease + Political L2 | country, company, person | comtrade, transport_throughput, disease_surveillance, political_risk | trade_flow, border_throughput, pathogen_level, campaign_finance | medium |
| **33** | Organization + Grid Enrichment | organization, region | regulatory_gazette, electricity_monitor | regulatory_velocity, grid_demand | medium |
| **34** | Commodity Country Links + Full Diagnostic | instrument | instrument_universe (commodity metadata) | (links only) | small |

---

## Phase 27: FX Country Wiring + Central Bank L2 ← CURRENT

**Status:** Task exists at [[phase27_fx_country_monetary_linking]]
**Spec:** [[phase27_fx_country_monetary_linking_spec]]

**Starved class addressed:** Country nodes (only 1 obs source), FX pairs (1 country link)

### Steps

- [ ] 27.1: Add `quote_country` field to `InstrumentDef` for FX pairs
- [ ] 27.2: Persist FX instrument `located_in` links to BOTH country nodes
- [ ] 27.3: Add PipelineStore to `central_bank_balance`, persist `monetary_balance` + `policy_rate` on country entities
- [ ] 27.4: Register 2 new obs types in graph builder, update ENRICHMENT_DIM
- [ ] 27.5: Diagnostic — FX instruments have 2 country links, country nodes have monetary obs
- [ ] 27.6: Regression + checkpoint

**Outcome:** Country nodes go from 1→3 obs types. FX pairs go from 1→2 country links.

---

## Phase 28: Country Node Enrichment (Macro Signals)

**Starved class addressed:** Country nodes still under-observed after Phase 27. Adds 3 more country obs types covering bonds, capital flows, and economic activity.

### Tools to L2-upgrade

| Tool | Current State | Obs Type to Add | Entity Target | Data Source |
|---|---|---|---|---|
| `sovereign_debt` | L1 aggregate, no PipelineStore | `sovereign_yield` | country | US/EU/JP/UK/etc. bond yields by maturity |
| `capital_flows` | L1 aggregate, no PipelineStore | `capital_flow` | country | TIC foreign Treasury holdings by country |
| `global_pmi` | L1 aggregate, no PipelineStore | `economic_activity` | country | OECD CLI/BCI/CCI per country |

### Steps

- [ ] 28.1: Research — read each tool's current source coverage and entity mapping requirements
- [ ] 28.2: Add PipelineStore to `sovereign_debt`, persist `sovereign_yield` on country entities
- [ ] 28.3: Add PipelineStore to `capital_flows`, persist `capital_flow` on country entities
- [ ] 28.4: Add PipelineStore to `global_pmi`, persist `economic_activity` on country entities
- [ ] 28.5: Register 3 new obs types in graph builder, update ENRICHMENT_DIM
- [ ] 28.6: Edge case tests for all 3 tools (partial data, missing countries, stale data, DB errors)
- [ ] 28.7: Diagnostics — country nodes should now have 6 obs types total
- [ ] 28.8: Regression + checkpoint

**Outcome:** Country nodes go from 3→6 obs types. All major economic dimensions covered (monetary, bond, capital, activity, geopolitical).

---

## Phase 29: Company + Investigative L2

**Starved class addressed:** Company entity gaps (no bankruptcy, no FOIA investigation signal). Academic preprint velocity for pharma companies.

### Tools to L2-upgrade

| Tool | Obs Type to Add | Entity Target | Signal Value |
|---|---|---|---|
| `bankruptcy_court` | `bankruptcy_status` | company | Credit lifecycle — Ch.7/11/15 status per company |
| `foia_requests` | `investigation_signal` | company, person | Investigative convergence — multi-agency FOIA clustering per entity |
| `academic_preprints` | `research_velocity` | company (pharma) | Drug pipeline — publication/trial velocity per institution |

### Steps

- [ ] 29.1: Research — read each tool, define entity extraction schema
- [ ] 29.2: Add PipelineStore to `bankruptcy_court`, persist `bankruptcy_status` on company entities
- [ ] 29.3: Add PipelineStore to `foia_requests`, persist `investigation_signal` on company/person entities
- [ ] 29.4: Add PipelineStore to `academic_preprints`, persist `research_velocity` on company entities
- [ ] 29.5: Register 3 new obs types in graph builder, update ENRICHMENT_DIM
- [ ] 29.6: Edge case tests (invalid entities, missing data, deduplication)
- [ ] 29.7: Diagnostics + regression + checkpoint

**Outcome:** Company entity enrichment deepens. Investigation/bankruptcy signals provide unique edge no other system has.

---

## Phase 30: Crypto Islands + Cross-Domain Linking

**Starved class addressed:** BTC-USD and ETH-USD are completely isolated graph islands (zero entity links). Wallet entities exist but don't connect to instruments.

### Changes

| Change | What | Why |
|---|---|---|
| Crypto → protocol links | Link BTC-USD to `protocol:bitcoin`, ETH-USD to `protocol:ethereum` | Protocol entities already exist from `defi_flows`. Creates first cross-type path for crypto. |
| wallet → instrument links | When `whale_alert` detects BTC/ETH whale trades, link the wallet to the relevant crypto instrument | Wallets currently have `transacts_with` (wallet↔wallet) but no path to instruments. |
| Verify `defi_flows` protocol entities | Ensure protocol entities from `defi_flows` are consistently named with what we link to | Avoid duplicate entities for the same protocol. |

### Steps

- [ ] 30.1: Add `protocol` field to `InstrumentDef` for crypto instruments (e.g., `protocol="bitcoin"`)
- [ ] 30.2: Extend `_persist_instrument_links` to create `tracks_protocol` links for crypto
- [ ] 30.3: Extend `whale_alert` to create `trades_instrument` links from whale wallets to BTC-USD/ETH-USD
- [ ] 30.4: Verify protocol entity naming consistency with `defi_flows`
- [ ] 30.5: Edge case tests (missing protocol, unknown crypto, link idempotency)
- [ ] 30.6: Diagnostics — BTC/ETH no longer degree-0 + wallet paths to instruments
- [ ] 30.7: Regression + checkpoint

**Outcome:** Crypto instruments connected to protocol + wallet graph. No longer isolated.

---

## Phase 31: Remaining Country Signals

**Starved class addressed:** Country nodes after 28 still missing demand-side, stability, health, and infrastructure signals.

### Tools to L2-upgrade

| Tool | Obs Type to Add | Entity Target | Signal |
|---|---|---|---|
| `consumer_sentiment` | `consumer_confidence` | country | Demand-side economic signal per country |
| `food_security` | `food_security_index` | country | EM agricultural/food stability |
| `internet_outages` | `internet_health` | country | Infrastructure/censorship signal |
| `migration_flows` | `migration_flow` | country | EM stability, refugee/remittance signal |

### Steps

- [ ] 31.1: Research — read each tool's country-level data availability
- [ ] 31.2: Add PipelineStore to `consumer_sentiment`, persist `consumer_confidence` on country entities
- [ ] 31.3: Add PipelineStore to `food_security`, persist `food_security_index` on country entities
- [ ] 31.4: Add PipelineStore to `internet_outages`, persist `internet_health` on country entities
- [ ] 31.5: Add PipelineStore to `migration_flows`, persist `migration_flow` on country entities
- [ ] 31.6: Register 4 new obs types in graph builder, update ENRICHMENT_DIM
- [ ] 31.7: Edge case tests for all 4 tools
- [ ] 31.8: Diagnostics — country nodes should now have 10 obs types total
- [ ] 31.9: Regression + checkpoint

**Outcome:** Country nodes become the richest entity type in the graph (10 obs types), rivaling company nodes.

---

## Phase 32: Trade + Disease + Political L2

**Starved class addressed:** Cross-domain entity gaps that require medium effort. Bilateral trade flows, physical border throughput, disease surveillance, and campaign finance.

### Tools to L2-upgrade

| Tool | Obs Type to Add | Entity Target | Effort | Signal |
|---|---|---|---|---|
| `comtrade` | `trade_flow` | country (bilateral pairs) | medium | Cross-country commodity flow patterns |
| `transport_throughput` | `border_throughput` | company (border port) | small | Physical trade volume per port |
| `disease_surveillance` | `pathogen_level` | country | medium | Pandemic/health signal per country |
| `political_risk` | `campaign_finance` | person (candidate) | medium | Election spending patterns |

### Steps

- [ ] 32.1: Research — define entity schema for each tool (trade pairs, ports, candidates)
- [ ] 32.2: Add PipelineStore to `comtrade`, persist `trade_flow` on country entity pairs
- [ ] 32.3: Add PipelineStore to `transport_throughput`, persist `border_throughput` entities
- [ ] 32.4: Add PipelineStore to `disease_surveillance`, persist `pathogen_level` on country entities
- [ ] 32.5: Add PipelineStore to `political_risk`, persist `campaign_finance` on person entities
- [ ] 32.6: Register 4 new obs types in graph builder, update ENRICHMENT_DIM
- [ ] 32.7: Edge case tests for all 4 tools
- [ ] 32.8: Diagnostics + regression + checkpoint

**Outcome:** Cross-domain entity linking reaches L2/L3 depth. Bilateral trade patterns, physical supply chain, disease, and political signals all feed the entity graph.

---

## Phase 33: Organization + Grid Enrichment

**Starved class addressed:** Organization entity type (nearly empty) and regional grid entities.

### Tools to L2-upgrade

| Tool | Obs Type to Add | Entity Target | Effort | Signal |
|---|---|---|---|---|
| `regulatory_gazette` | `regulatory_velocity` | organization (agency) | medium | Per-agency rule velocity, regulatory attention |
| `electricity_monitor` | `grid_demand` | region (balancing authority) | small | Per-BA demand anomalies, economic proxy |

### Steps

- [ ] 33.1: Research — define entity extraction for regulatory targets and BA regions
- [ ] 33.2: Add PipelineStore to `regulatory_gazette`, persist `regulatory_velocity` on organization entities
- [ ] 33.3: Add PipelineStore to `electricity_monitor`, persist `grid_demand` on region entities
- [ ] 33.4: Decide if `region` needs to be a new entity type or maps to existing `country`/`organization`
- [ ] 33.5: Register new obs types in graph builder
- [ ] 33.6: Edge case tests
- [ ] 33.7: Diagnostics + regression + checkpoint

**Outcome:** Organization entities gain persistent observations for the first time. Grid demand becomes a structured signal.

---

## Phase 34: Commodity Country Links + Full Diagnostic Sweep

**Starved class addressed:** Commodity futures (no country links). Plus final diagnostic to verify everything.

### Changes

| Change | What | Why |
|---|---|---|
| Domestic commodity links | Add `primary_exchange_country` for US-anchored commodities (nat gas, grain, cattle, hogs) | 8-10 commodities are meaningfully US-domestic |
| Global commodity non-link | Leave truly global commodities (gold, oil, coffee) without country links | Architecturally correct — no one country owns them |
| Full diagnostic sweep | Run GNN diagnostics across ALL entity types with the full expanded graph | Find any remaining starved classes |
| GNN retrain baseline | Retrain the GNN with the expanded graph and establish new performance baseline | Measure impact of all L2 upgrades |

### Steps

- [ ] 34.1: Add `primary_exchange_country` to `InstrumentDef` for domestic commodities
- [ ] 34.2: Extend `_persist_instrument_links` for commodity → country links
- [ ] 34.3: Full diagnostic sweep — every entity type degree, obs count, obs type coverage
- [ ] 34.4: GNN retrain with expanded graph
- [ ] 34.5: Performance comparison vs Phase 25 baseline
- [ ] 34.6: Document remaining gaps for future phases
- [ ] 34.7: Final regression + checkpoint

**Outcome:** No instrument class has zero entity links. Full baseline established for the GNN on the expanded graph.

---

## Cumulative Impact Tracker

| Metric | Phase 25 (current) | After 27 | After 28 | After 29 | After 30 | After 31 | After 32 | After 33 | After 34 |
|---|---|---|---|---|---|---|---|---|---|
| Country obs types | 1 | 3 | 6 | 6 | 6 | 10 | 12 | 12 | 12 |
| Company obs types | 6 | 6 | 6 | 9 | 9 | 9 | 9 | 9 | 9 |
| Person obs types | 3 | 3 | 3 | 4 | 4 | 4 | 5 | 5 | 5 |
| Organization obs types | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 2 |
| FX country links/pair | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| Crypto entity links | 0 | 0 | 0 | 0 | 2+ | 2+ | 2+ | 2+ | 2+ |
| Registered obs types | 27 | 29 | 32 | 35 | 35 | 39 | 43 | 45 | 45 |
| Non-L2 tools | 38 | 36 | 33 | 30 | 29 | 25 | 21 | 19 | 19 |
| L2-persisting tools | 22 | 24 | 27 | 30 | 31 | 35 | 39 | 41 | 42 |

---

## Tools NOT included in any phase (and why)

These were explicitly evaluated and excluded:

| Tool | Reason for exclusion |
|---|---|
| `energy_supply` | National aggregate only (EIA weekly inventory). No entity resolution possible. |
| `treasury_receipts` | National daily fiscal pulse. No entity decomposition. |
| `building_permits` | Regional FRED series. Country-conditional at best. |
| `labor_disruptions` | National stoppage tally. Would need NLP for company/union extraction. |
| `macro_data` | Wrapper tool — individual source tools are L2'd instead. |
| `market_data` | Already covered by `instrument_universe` price ingest. |
| `job_postings` | BLS JOLTS = sector-level. Sector entities don't exist yet. |
| `earthquake_proximity` | Events are ephemeral — no persistent entity identity. |
| `satellite_activity` | Fire/drought events don't have persistent IDs. Better as triggers. |
| `weather_alerts` | NWS alerts are ephemeral. Better as event-driven overlay. |
| `liquidity_regime` | Computation tool (HMM/BOCPD), not data source. |
| `backtest` | Utility, not data source. |
| `code_executor` | Utility. |
| `file_manager` | Utility. |
| `shell_runner` | Utility. |
| `web_browse` | Utility. |
| `web_search` | Utility. |
| `pipeline_query` | Bridge tool, not data source. |
| `base` | Abstract base class. |

---

## Deferred Work (Post-Phase 34)

| Item | When | Why Deferred |
|---|---|---|
| Phase 26.4: Custom TirraMind MCP server | After Phase 34 if tooling bottleneck | Developer velocity, not model quality |
| `central_bank` as separate entity type | After graph builder type system is fully dynamic | Currently would create encoding debt |
| Sector entities (`job_postings` L2) | After sector taxonomy is defined | Need sector ontology first |
| NLP entity extraction from `labor_disruptions` | After NER pipeline exists | Too brittle without proper NER |
| `internet_infrastructure` ASN-level | After GNN evaluation shows ASN gap | Country-level sufficient for now |

## Related

- [[starved_class_audit]]
- [[phase27_fx_country_monetary_linking]]
- [[phase27_fx_country_monetary_linking_spec]]
- [[phase25_cross_domain_entity_linking]]
- [[crypto_islands_cross_domain_linking]]
- [[crypto_islands_cross_domain_linking_spec]]
- [[phase30_crypto_islands]]
- [[quant_training_ground]]
