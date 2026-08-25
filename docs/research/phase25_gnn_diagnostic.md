---
title: "GNN Diagnostic: Phase 25 Input"
tags:
  - doc/research
  - phase/25
  - topic/gnn-expansion
  - topic/production-hardening
  - layer/world-model
  - layer/surveillance
---

# GNN Diagnostic: Phase 25 Input

Deferred from Phase 24b.5.2. This document captures the current state of the entity graph, identifies structural gaps, and feeds the Phase 25 specification.

---

## 1. Current Entity Graph State

### Entity Types (10 canonical)

| Entity Type | L2 Tools Feeding It | Observation Types | Status |
|---|---|---|---|
| **company** | insider_filings, form144, lobbying, patent_filings, gov_contracts, drug_regulatory, finra_short_volume, creditor_filings, supply_chain_monitor, sanctions_monitor | insider_trade, form144_filing, lobbying_spend, patent_filing, contract_award, drug_approval, short_interest, creditor_filing, sanctions_listing | **Dense** — 10 tools |
| **person** | insider_filings, form144 | insider_trade, form144_filing | Moderate — 2 tools |
| **instrument** | instrument_universe, finra_short_volume | instrument_return, instrument_volume, instrument_volatility, short_interest | **Dense** — 90 instruments × daily obs |
| **domain** | cert_transparency, dns_monitor | cert_issued, dns_change | Moderate — 2 tools |
| **vessel** | ais_vessel | vessel_position, port_call | Light — 1 tool |
| **wallet** | whale_alert, defi_flows | btc_transfer | Light — 2 tools (same domain) |
| **protocol** | defi_flows | tvl_change | Light — 1 tool |
| **organization** | gov_contracts, ais_vessel, sanctions_monitor, supply_chain_monitor, creditor_filings | contract_award, sanctions_listing, creditor_filing | Moderate — 5 tools |
| **topic** | wikipedia_pageviews | pageview_spike | Light — 1 tool |
| **country** | gdelt | geopolitical_event | Light — 1 tool (entity registration) |

### Observation Types (24 total)

3 instrument-specific: instrument_return, instrument_volume, instrument_volatility
21 entity-specific: insider_trade, form144_filing, lobbying_spend, patent_filing, contract_award, drug_approval, short_interest, creditor_filing, sanctions_listing, btc_transfer, tvl_change, cert_issued, dns_change, vessel_position, port_call, geopolitical_event, pageview_spike, cross_entity_pattern, sell_intent, price_movement, project_status

### Link Types (established)

| Link Type | Source → Target | Tool |
|---|---|---|
| insider_trade | person → company | insider_filings |
| lobbies_for | company → company | lobbying |
| holds_certificate | domain → issuer | cert_transparency |
| owns_domain | organization → domain | dns_monitor |
| manages_vessel | organization → vessel | ais_vessel |
| whale_wallet | wallet → wallet | whale_alert |
| trades_on | wallet → protocol | defi_flows |
| debtor_of | company → company | creditor_filings |
| market_authorized_in | company → country | drug_regulatory |

### ENRICHMENT_DIM = 33

Base feature dim = 13, 20 enrichment dims (cusum, hawkes, event_study, bocpd, variance, min, max, iqr, num_tools, + 11 obs_type distribution slots).

---

## 2. Structural Gap Analysis

### High-Value Gaps (GNN attention would be starved)

1. **country entity type** — only 1 tool (GDELT) registers country entities. No macro tools feed this. Treasury, central bank, sovereign debt, global PMI, consumer sentiment, capital flows, etc. all produce country-level data but remain L1-only. Country nodes are sparse and weakly connected.

2. **instrument ↔ company link** — instruments are GNN entities (Phase 24a) but there's no link between `instrument` and `company` entities. SPY tracks S&P 500 companies, CL=F relates to energy companies, etc. The GNN cannot propagate insider filing signals to the instrument that would be affected.

3. **Cross-domain entity links (L3)** — the entity linking layer (Phase 17) established the infrastructure but few cross-domain links exist in practice. company→instrument, company→country (HQ location), person→company (beyond insider filings), vessel→company (operator) connections would enable the GNN's cross-type attention to discover predictive patterns.

4. **Temporal density** — instruments have daily observations. Most other entity types have sporadic observations (event-driven). The GNN's temporal attention mechanism has unbalanced resolution across types.

### L1-Only Tools That Could Feed Entity Types

| Tool (L1 only) | Potential Entity Type | Observation Type | Priority |
|---|---|---|---|
| macro_data (FRED) | country | macro_release | Medium (global conditioning) |
| central_bank_balance | country | balance_sheet_change | Medium |
| sovereign_debt | country | debt_change | Medium |
| treasury_receipts | country | fiscal_receipt | Low (US-only) |
| consumer_sentiment | country | sentiment_reading | Low |
| global_pmi | country | pmi_reading | Medium |
| capital_flows | country | capital_flow | High |
| cftc | company/instrument | futures_positioning | **High** |
| polymarket | topic/person | prediction_market_trade | **High** |
| polymarket_whales | wallet/person | whale_bet | **High** |
| market_data | instrument | (already covered by instrument_universe) | Skip |
| electricity_monitor | country/company | grid_load | Low |
| power_grid | country | frequency_deviation | Low |
| satellite_activity | company/country | activity_change | Medium |
| comtrade | country | trade_flow | Medium |
| transport_throughput | country | throughput | Low |
| job_postings | company | hiring_signal | Medium |
| political_risk | country | risk_event | Medium |
| earthquake_proximity | country | seismic_event | Low |
| weather_alerts | country | weather_alert | Low |

### What Phase 24 Revealed

1. **Walk-forward works end-to-end** — the pipeline from data ingestion through GNN → SAC → portfolio weights → P&L is functional. The bottleneck is now **signal quality**, not infrastructure.
2. **Instrument entities are dense but isolated** — 90 instruments with daily return/volume/volatility create a rich sub-graph, but with no links to company/country/person entities, the GNN cannot propagate information across domains.
3. **Alert conditions work** — drawdown, concentration, Sharpe, and edge decay alerts are in place for paper trading.

---

## 3. Phase 25 Recommendation

### Option A: Cross-Domain Entity Linking (Recommended)

**Goal:** Wire the instrument sub-graph to the entity graph so cross-type GNN attention can propagate signals.

Key deliverables:
- instrument→company links (ETF/stock → issuer, futures → producer companies)
- company→country links (HQ, primary market)
- CFTC L2 (futures positioning → instrument entities)
- Polymarket L2 (prediction market → topic/person entities)
- GNN retraining with new links, attention weight evaluation

**Why:** This directly attacks the highest-value gap. The GNN can already do cross-type attention, but instrument nodes are an island. Connecting them to the company/person graph unlocks L3-style cross-domain signals — the stated moat.

### Option B: Production Hardening

**Goal:** Make the nightly pipeline run reliably on real data.

Key deliverables:
- Real data validation (backfill all L2 tools, not just instruments)
- Pipeline monitoring + alerting
- Error recovery / retry logic
- Config management for deployment

**Why:** Less exciting but necessary for paper trading to produce meaningful results.

### Option C: GNN-Guided Expansion R3

**Goal:** More L2 tool upgrades based on diagnostic output.

Key deliverables:
- Upgrade 3-5 more L1 tools to L2 (polymarket_whales, cftc, capital_flows, job_postings)
- Run fresh GNN diagnostics to confirm attention improvement

**Why:** Incremental improvement to surveillance surface. Lower impact than Option A because adding more isolated L2 tools without cross-domain links doesn't improve the GNN's ability to fuse signals.

### Recommendation

**Phase 25 = Option A (Cross-Domain Entity Linking) + selective pieces of Option C.**

The highest-impact work is linking instruments to the rest of the entity graph. While doing that, also upgrade CFTC and Polymarket to L2 since they're high-priority data sources that would feed the new links. This combines L3 architecture (the moat) with L2 tool expansion (GNN signal).

Production hardening (Option B) should be Phase 26, after the graph is connected.

---

## Related

- [[e2e_global_integration]] — Phase 24 research
- [[e2e_global_integration_spec]] — Phase 24 spec
- [[e2e_global_integration|Phase 24 Task]] — Completed
- [[phase25_cross_domain_entity_linking_spec]] — Phase 25 implementation spec
- [[phase25_cross_domain_entity_linking]] — Active Phase 25 task
- [[gnn_guided_expansion_r2]] — Phase 23 (prior GNN expansion)
- [[tool_priority_ranking]] — GNN-guided tool priority
- [[temporal_het_gnn]] — HetTGN architecture
- [[entity_linking_layer]] — Phase 17 entity linking
