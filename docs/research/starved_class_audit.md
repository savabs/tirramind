---
title: "Research: Starved Class Audit — Entity Graph Connectivity Gaps"
tags:
  - doc/research
  - phase/27
  - topic/entity-linking
  - topic/graph-connectivity
  - layer/surveillance
  - layer/world-model
---

# Starved Class Audit — Entity Graph Connectivity Gaps

## Purpose

Comprehensive audit of what's connected, what's starved, and what to improve in the GNN entity graph. This drives the prioritized L2 upgrade roadmap.

## Current Graph State

### Entity Types (11 seeded)

| Entity Type | Direct Observation Sources | # Obs Types Received | Link Target Only? |
|---|---|---|---|
| **company** | creditor_filings, finra_short_volume, gov_contracts, interconnection_queue, lobbying, patent_filings | 6 (creditor_filing, short_interest, contract_award, project_status, lobbying_spend, patent_filing) | No — richest |
| **person** | form144, insider_filings, sanctions_monitor | 3 (sell_intent, insider_trade, sanctions_listing) | No |
| **instrument** | instrument_universe | 3 (instrument_return, instrument_volume, instrument_volatility) | No |
| **vessel** | ais_vessel | 2 (vessel_position, port_call) | No |
| **wallet** | whale_alert, polymarket_whales | 2 (btc_transfer, whale_trade) | No |
| **topic** | polymarket, supply_chain_monitor, wikipedia_pageviews | 3 (market_probability, price_movement, pageview_spike) | No |
| **domain** | cert_transparency, dns_monitor | 2 (cert_issued, dns_change) | No |
| **cftc_contract** | cftc | 1 (futures_positioning) | No |
| **protocol** | defi_flows | 1 (tvl_change) | No |
| **country** | gdelt ONLY | 1 (geopolitical_event) | **MOSTLY link target** |
| **organization** | sanctions_monitor (rare) | 1 (sanctions_listing, rare) | **MOSTLY link target** |

### Observation Types (27 registered)

All 27 observation types are produced by at least one tool. No orphaned observation types.

### Link Types in Production

| Link Type | Source→Target | Created By |
|---|---|---|
| `tracks_issuer` | instrument→company | instrument_universe |
| `located_in` | instrument→country | instrument_universe |
| `located_in` | company→country | instrument_universe, interconnection_queue |
| `cftc_tracks` | cftc_contract→instrument | cftc |
| `port_call_to` | vessel→country | ais_vessel |
| `works_for` | person→company | form144, insider_filings |
| `owes_to` | company→company | creditor_filings |
| `market_authorized_in` | company→country | drug_regulatory |
| `operates_in` | company→country | gov_contracts |
| `awarded_by` | company→organization | gov_contracts |
| `lobbies_for` | company→company | lobbying |
| `patents_in` | company→country | patent_filings |
| `sanctioned_under` | person/org/company→country | sanctions_monitor |
| `transacts_with` | wallet→wallet | whale_alert |
| `event_involves` | country↔country | gdelt |

### Feature Dimensions

- **BASE_FEAT_DIM = 14**: entity_type one-hot (11) + obs_count (1) + recency (1) + mean_value (1)
- **ENRICHMENT_DIM = 36**: cusum(1) + hawkes(1) + event_study(1) + bocpd(1) + variance(1) + min(1) + max(1) + iqr(1) + num_source_tools(1) + obs_type_dist(27)
- **Total: 50 dims per node**

---

## Starved Classes — Ranked by Severity

### CRITICAL: Country Nodes (Most Starved Entity Type)

**Problem:** Country is used as a link target by 8+ tools but receives observations from only GDELT (geopolitical_event). Country nodes are in the graph, instruments/companies/vessels link to them, but the GNN has nearly no signal on these nodes. They're "dark" relay hubs.

**Impact:** FX pairs, country ETFs, and all instruments with `located_in` links to countries can't benefit from country-level intelligence because country nodes have almost empty feature vectors. The message-passing substrate is structurally handicapped.

**What country nodes SHOULD know (but don't):**
- Monetary policy state (central bank balance, policy rate) — **38 non-L2 tools have this data**
- Sovereign debt yields/spreads — `sovereign_debt.py` has this
- Capital flows / foreign Treasury holdings — `capital_flows.py` has this
- PMI / business confidence — `global_pmi.py` has this
- Consumer sentiment — `consumer_sentiment.py` has this
- Food security indicators — `food_security.py` has this
- Internet health / censorship — `internet_outages.py` has this
- Migration/refugee flows — `migration_flows.py` has this
- Disease prevalence — `disease_surveillance.py` has this

**Country is the #1 upgrade priority.** Every country-signal tool that persists observations onto country nodes directly improves 15 FX pairs + 21 country ETFs + 4 equity index futures.

### HIGH: FX Pair Connectivity Gap

**Problem:** FX instruments only link to ONE country (base currency country). EURUSD=X → EU only. Missing: EURUSD=X → US. The GNN can't learn carry/divergence signals because it only sees one side.

**Impact:** 15 FX pairs lose half their structural information. FX is the most globally important instrument class and it has the weakest graph connectivity.

**Fix:** Add `quote_country` to InstrumentDef for FX. Create `located_in` links to both countries. Already spec'd in Phase 27.

### HIGH: Organization Entity Type (Nearly Empty)

**Problem:** `organization` entities exist in the seed types and are created by `gov_contracts` (as awarding agencies) and `sanctions_monitor` (rarely, SDN list), but they receive almost no observations. They're structural placeholders.

**Impact:** Government agencies that award contracts are in the graph but have no observable state. The GNN can't learn about agency behavior or regulatory patterns.

**Potential fix:** `regulatory_gazette.py` L2 upgrade (federal register rule velocity per agency). Medium effort due to NLP for entity extraction.

### MEDIUM: Commodity Futures — No Country Links

**Problem:** 20 commodity futures have `country=None` by design (global commodities). They link to CFTC contracts but NOT to any country. They have no issuer either.

**Impact:** Commodity futures are isolated instrument nodes with only CFTC positioning signal and their own price observations. No geographic/monetary links.

**Potential fix:** This is architecturally correct — coffee doesn't "belong to" a country. But some commodities have dominant production/pricing venues (US nat gas, US grain). Consider a `primary_exchange_country` field for domestically-anchored commodities.

### MEDIUM: Crypto — No Entity Links

**Problem:** BTC-USD and ETH-USD have no issuer, no country, no CFTC code. Zero entity links. They receive price observations only.

**Impact:** Crypto instruments are graph islands. The only cross-type signal comes from `wallet` entities (whale_alert, polymarket_whales) which don't link to crypto instruments directly.

**Potential fix:** Add `protocol` link (BTC→bitcoin protocol, ETH→ethereum protocol). The protocol entities already exist from `defi_flows`. Or add explicit `wallet→instrument` links from whale_alert when whale trades involve BTC/ETH.

### LOW: EMB (EM Bond ETF) — No Country

**Problem:** EMB has no country field (`country=None`) because it tracks a basket. It has an issuer (BlackRock) but no geographic link.

**Impact:** Minor — single instrument.

---

## Non-L2 Tools Audit — Upgrade Roadmap

### Tier 1: Country-Signal Tools (High Priority, Persist Onto Country Nodes)

These tools already have country-level data. Upgrading them would directly fix the #1 starved class.

| Tool | Obs Type to Add | Target Entity | Effort | Signal |
|---|---|---|---|---|
| **central_bank_balance** | `monetary_balance`, `policy_rate` | country | small | **CRITICAL** — causal driver of FX + carry |
| **sovereign_debt** | `sovereign_yield` | country | small | **HIGH** — bond/credit signal per country |
| **capital_flows** | `capital_flow` | country | small | **HIGH** — TIC foreign holdings per country |
| **global_pmi** | `economic_activity` | country | small | **HIGH** — leading recession indicator |
| **consumer_sentiment** | `consumer_confidence` | country | tiny | MEDIUM — demand-side signal |
| **food_security** | `food_security_index` | country | tiny | MEDIUM — EM/agricultural signal |
| **internet_outages** | `internet_health` | country | tiny | MEDIUM — censorship/infrastructure |
| **migration_flows** | `migration_flow` | country | tiny | MEDIUM — EM stability signal |

### Tier 2: L2-Ready Entity Tools (Add New Entity Observations)

| Tool | Obs Type | Target Entity | Effort | Signal |
|---|---|---|---|---|
| **bankruptcy_court** | `bankruptcy_status` | company | small | **HIGH** — credit lifecycle |
| **foia_requests** | `investigation_signal` | company/person | small | **VERY HIGH** — investigative edge |
| **political_risk** | `campaign_finance` | person (candidate) | medium | HIGH — election spending patterns |
| **comtrade** | `trade_flow` | country (bilateral) | medium | HIGH — cross-country commodity flows |
| **transport_throughput** | `border_throughput` | company (port) | small | HIGH — physical trade signal |
| **academic_preprints** | `research_velocity` | company (pharma) | small | HIGH — drug pipeline signal |
| **disease_surveillance** | `pathogen_level` | country/state | medium | HIGH — pandemic/health signal |

### Tier 3: Medium-Effort Entity Tools

| Tool | Obs Type | Target Entity | Effort | Signal |
|---|---|---|---|---|
| **regulatory_gazette** | `regulatory_velocity` | organization (agency) | medium | **HIGH** — regulatory attention signal |
| **electricity_monitor** | `grid_demand` | region (BA) | small | MEDIUM — regional economic proxy |

### Not Worth L2 Upgrade (Keep as Aggregate)

| Tool | Reason |
|---|---|
| **energy_supply** | National aggregates only (EIA weekly). No entity resolution. |
| **treasury_receipts** | National daily fiscal pulse. No entity decomposition possible. |
| **building_permits** | Regional FRED series. Country-conditional at best. |
| **labor_disruptions** | National stoppage tally. No company/union resolution without NLP. |
| **macro_data** | Wrapper tool — individual source tools should be L2'd instead. |
| **market_data** | Covered by instrument_universe price ingest. |
| **job_postings** | BLS JOLTS = sector level, not company level. Sector entities don't exist yet. |

### Not Worth L2 Upgrade (Ephemeral Events)

| Tool | Reason |
|---|---|
| **earthquake_proximity** | Events are ephemeral, no persistent entity identity. |
| **satellite_activity** | Fire/drought events don't have persistent IDs. |
| **weather_alerts** | NWS alerts are ephemeral. Better as event triggers. |

---

## Instrument Class Connectivity Summary

| Asset Class | # Instruments | Links to Company | Links to Country | Links to CFTC | Other Links | Obs Sources |
|---|---|---|---|---|---|---|
| **equity_etf** | 21 | Yes (issuer) | Yes (1 country) | No | — | price only |
| **sector_etf** | 15 | Yes (issuer) | Yes (US) | No | — | price only |
| **commodity_future** | 20 | No | **No** | Yes (18/20) | — | price + CFTC |
| **fx** | 15 | No | **1 country only** | No | — | price only |
| **equity_index** | 4 | No | Yes (US) | No | — | price only |
| **fixed_income** | 10 | Some (issuer) | Yes (mostly US) | No | — | price only |
| **vol** | 3 | Some (issuer) | Yes (US) | No | — | price only |
| **crypto** | 2 | **No** | **No** | **No** | **None** | **price only — island** |

### Most Starved Instrument Classes (Priority Order)

1. **Crypto** (2 instruments) — completely isolated graph islands
2. **FX** (15 instruments) — only 1 country link, country nodes have no signal
3. **Commodity Futures** (20 instruments) — no country links, only CFTC
4. **Equity Index Futures** (4 instruments) — only US country link, no issuer

---

## Recommended Improvement Sequence

### Wave 1: Fix Country Nodes (Biggest Bang for Buck)

**Goal:** Transform country nodes from dark relay hubs into signal-rich nodes. This instantly improves connectivity for 15 FX + 21 equity ETFs + 4 equity index + anything linked to countries.

**Work:**
1. ✅ FX two-country wiring (Phase 27 step 27.1-27.2)
2. ✅ Central bank L2 → country observations (Phase 27 step 27.3)
3. **NEW** `sovereign_debt` L2 → country observations
4. **NEW** `capital_flows` L2 → country observations
5. **NEW** `global_pmi` L2 → country observations

**Observation types to add (compact):**
- `monetary_balance` — central bank balance sheet state
- `policy_rate` — interest rate level
- `sovereign_yield` — bond yield per maturity
- `capital_flow` — foreign holdings changes
- `economic_activity` — PMI/CLI composite

### Wave 2: Fix Company L2 Gaps

**Work:**
1. `bankruptcy_court` L2 → company observations (`bankruptcy_status`)
2. `foia_requests` L2 → company/person observations (`investigation_signal`)

### Wave 3: Fix Crypto Islands + Cross-Domain Links

**Work:**
1. Link BTC-USD/ETH-USD to protocol entities (bitcoin/ethereum)
2. Add `wallet→instrument` links from whale_alert

### Wave 4: Remaining Country Signals (Lower Priority)

**Work:**
1. `consumer_sentiment` L2 → country observations
2. `food_security` L2 → country observations
3. `internet_outages` L2 → country observations
4. `migration_flows` L2 → country observations

---

## Key Numbers

| Metric | Current | After Wave 1 |
|---|---|---|
| Entity types with ≥3 obs sources | 4 (company, person, instrument, topic) | 5 (+country) |
| Country obs types | 1 (geopolitical_event) | 6 (+monetary_balance, policy_rate, sovereign_yield, capital_flow, economic_activity) |
| FX country links per pair | 1 | 2 |
| Registered observation types | 27 | 32 |
| Non-L2 tools | 38 | 33 |
| Crypto entity links | 0 | 0 (Wave 3) |

## Related

- [[phase27_fx_country_monetary_linking]]
- [[phase27_fx_country_monetary_linking_spec]]
- [[phase25_cross_domain_entity_linking]]
- [[crypto_islands_cross_domain_linking]]
- [[crypto_islands_cross_domain_linking_spec]]
- [[phase30_crypto_islands]]
- [[quant_training_ground]]
