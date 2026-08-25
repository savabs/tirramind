---
title: "Feature: GNN-Guided Expansion Round 2"
tags:
  - doc/research
  - phase/23
  - topic/surveillance
  - topic/gnn-expansion
  - layer/surveillance
---

# Feature: GNN-Guided Expansion Round 2

## Goal

Re-evaluate the entity graph after Phase 17 (entity linking) and Phases 18-22 (world model, fusion, RL, adversarial) to identify which remaining L1 tools should be upgraded to L2. The Signal Depth Doctrine requires this be GNN-driven, not checklist-driven.

**Context:** Phase 16 performed the first GNN-guided evaluation but found an empty real store — all ranking was architecture-driven. Since then:
- 3 more tools upgraded to L2 (sanctions_monitor, gov_contracts, supply_chain_monitor → Phase 18)
- Entity linking added (8 link types, Phase 17)
- Full quant stack completed (Phases 19-22)
- Still no live API runs → real store still empty

**Decision:** Use architecture-driven + synthetic evaluation to rank remaining L1 tools. The same rubric from Phase 16 applies, updated for the current graph topology.

## Current Architecture

### L2 Tools (15 — entity persistence active)

| Tool | Entity Types | Observation Types | Link Types |
|------|-------------|-------------------|------------|
| insider_filings | company, person | insider_trade | works_for |
| form144 | company, person | form144_filing, sell_intent | works_for |
| whale_alert | wallet | btc_transfer | transacts_with |
| ais_vessel | vessel, country | vessel_position, port_call | port_call_to |
| gdelt | company, person, country, org | geopolitical_event | event_involves |
| cert_transparency | domain, company | cert_issued | — |
| dns_monitor | domain | dns_change | — |
| wikipedia_pageviews | topic | pageview_spike | — |
| lobbying | company, organization | lobbying_spend | lobbies_for |
| patent_filings | company, person | patent_filing | patents_in |
| defi_flows | protocol | tvl_change | — |
| interconnection_queue | company | project_status | located_in |
| sanctions_monitor | company, org, person, vessel | sanctions_listing | sanctioned_under |
| gov_contracts | company, organization | contract_award | operates_in, awarded_by |
| supply_chain_monitor | company | supply disruption | — |

### Entity Types (9 canonical in graph_builder)

company, country, domain, organization, person, protocol, topic, vessel, wallet

### Observation Types (18 canonical in graph_builder)

btc_transfer, cert_issued, contract_award, cross_entity_pattern, dns_change, form144_filing, geopolitical_event, insider_trade, lobbying_spend, pageview_spike, patent_filing, port_call, price_movement, project_status, sanctions_listing, sell_intent, tvl_change, vessel_position

### Entity Link Types (8+ in use)

works_for, transacts_with, event_involves, port_call_to, lobbies_for, patents_in, located_in, sanctioned_under, headquartered_in, exchange_based_in, operates_in, awarded_by

### Graph Topology Analysis

**Entity types with strong coverage (≥3 obs types):**
- company: insider_trade, form144_filing, sell_intent, cert_issued, lobbying_spend, patent_filing, project_status, sanctions_listing, contract_award, supply_disruption → **10 obs types**
- person: insider_trade, form144_filing, sell_intent, patent_filing, sanctions_listing → **5 obs types**
- country: vessel_position, port_call, geopolitical_event → **3 obs types**
- organization: geopolitical_event, lobbying_spend, sanctions_listing, contract_award → **4 obs types**

**Entity types with weak coverage (<3 obs types):**
- vessel: vessel_position, port_call, sanctions_listing → 3 (borderline)
- domain: cert_issued, dns_change → **2 obs types**
- wallet: btc_transfer → **1 obs type**
- protocol: tvl_change → **1 obs type**
- topic: pageview_spike → **1 obs type**

**Entity types with no cross-domain links:**
- domain (cert_transparency, dns_monitor produce no entity links)
- protocol (defi_flows produces no entity links)
- topic (wikipedia_pageviews produces no entity links)

**Conclusion:** company and person are saturated. domain, wallet, protocol, topic are sparsest. But wallet/protocol/topic are niche by design — upgrading generic tools won't help. The biggest ROI is in tools that either (a) add new observation types to *country* or *organization* (moderate coverage, high cross-domain value) or (b) create new cross-domain links between currently-disconnected entity clusters.

## Remaining L1 Tools — Upgrade Candidates

### Category A: Entity-Level Potential (could produce entity observations)

| Tool | Potential Entity Type | Potential Obs Type | Cross-Domain Link Potential |
|------|----------------------|-------------------|----------------------------|
| **finra_short_volume** | company | short_interest | company ← market signal |
| **creditor_filings** | company | creditor_filing | company → company (debtor-creditor) |
| **drug_regulatory** | company | drug_approval | company → country (market) |
| **bankruptcy_court** | company | bankruptcy_event | company → company (creditor chain) |
| **job_postings** | company | hiring_signal | company → country (expansion) |
| **regulatory_gazette** | company, organization | regulatory_action | org → company (regulator-target) |
| **political_risk** | country | political_event | country → country (contagion) |
| **disease_surveillance** | country | disease_outbreak | country → country (spread) |
| **foia_requests** | organization | foia_release | org → company (investigative) |

### Category B: Aggregate/Global (should stay L1)

These produce country-level or market-level numbers without entity resolution. Per the Signal Depth Doctrine, they are better consumed as global conditioning variables:

treasury_receipts, consumer_sentiment, central_bank_balance, global_pmi, energy_supply, food_security, macro_data, capital_flows, liquidity_regime, building_permits, comtrade, power_grid, electricity_monitor, cftc, market_data

### Category C: Physical/Geo (niche, lower priority)

weather_alerts, earthquake_proximity, satellite_activity, transport_throughput, migration_flows, internet_infrastructure, internet_outages, labor_disruptions, academic_preprints

## Updated Ranking (Phase 16 Rubric, Current Topology)

Scoring: graph_connectivity_gain (0.30) + signal_uniqueness (0.25) + implementation_effort (0.20) + data_quality_risk (0.15) + overlap_penalty (0.10)

| Rank | Tool | Connectivity | Uniqueness | Effort | Quality | Overlap | **Score** |
|------|------|-------------|-----------|--------|---------|---------|----------|
| 1 | **finra_short_volume** | 0.20 | 0.25 | 0.18 | 0.15 | 0.08 | **0.86** |
| 2 | **creditor_filings** | 0.25 | 0.22 | 0.14 | 0.12 | 0.08 | **0.81** |
| 3 | **drug_regulatory** | 0.22 | 0.22 | 0.16 | 0.12 | 0.06 | **0.78** |
| 4 | **bankruptcy_court** | 0.20 | 0.20 | 0.14 | 0.12 | 0.06 | **0.72** |
| 5 | **regulatory_gazette** | 0.18 | 0.20 | 0.12 | 0.12 | 0.08 | **0.70** |
| 6 | **job_postings** | 0.15 | 0.18 | 0.16 | 0.12 | 0.06 | **0.67** |
| 7 | **political_risk** | 0.20 | 0.18 | 0.12 | 0.10 | 0.06 | **0.66** |
| 8 | **disease_surveillance** | 0.18 | 0.18 | 0.12 | 0.10 | 0.06 | **0.64** |

### Tier 1 (This Phase): finra_short_volume, creditor_filings, drug_regulatory

**Why finra_short_volume (#1):**
- Adds unique short_interest observation to company — only real-time market-microstructure signal in the entity graph
- High data quality (FINRA public API, daily updates, well-structured)
- Low implementation effort (tool already exists, just needs L2 persistence)
- Creates signal that the adversarial layer's VPIN can cross-reference

**Why creditor_filings (#2):**
- Creates company→company debtor-creditor links — only inter-company link type besides supply_chain
- Unique bankruptcy/distress signal not available from any other tool
- PACER/RECAP are free public data

**Why drug_regulatory (#3):**
- FDA approval/rejection is a high-signal event with entity-level resolution
- Creates company→country market links (which markets approved)
- ClinicalTrials.gov is $0, high quality, well-structured API

### Tier 2 (Next Evaluation): bankruptcy_court, regulatory_gazette, job_postings

### Deferred: political_risk, disease_surveillance (country-level, moderate ROI)

## Implementation Pattern (Established)

Each L2 upgrade follows the same pattern (from 15 existing tools):

1. Accept `pipeline_store: PipelineStore | None = None` in constructor
2. Add `_persist_entities()` wrapper with try/except non-fatal
3. Add `_persist_entities_inner()` with:
   - Entity registration via `store.register_entity(entity_type, canonical_name, entity_id, metadata)`
   - Observation storage via `store.store_entity_observation(entity_id, source_tool, observed_at, observation_type, depth_level, value)`
   - Entity linking via `store.link_entities(entity_id_a, entity_id_b, link_type, source, confidence, metadata)` where applicable
4. Call `_persist_entities()` from the main `_run()` method
5. Add new observation types and entity link types to graph_builder constants

**Effort estimate:** ~2-4 hours per tool (they already exist, just need persistence layer).

## Graph Builder Updates Required

New observation types to add to `OBSERVATION_TYPES`:
- `short_interest` (finra_short_volume)
- `creditor_filing` (creditor_filings)
- `drug_approval` (drug_regulatory)

No new entity types needed — all three tools produce `company` entities, which already exist.

New link types:
- `debtor_of` / `creditor_of` (creditor_filings: company → company)
- `market_authorized_in` (drug_regulatory: company → country)

## Risks

1. **Graph builder constant expansion** — adding 3 new obs types increases `OBSERVATION_TYPES` from 18→21, affecting `ENRICHMENT_DIM` (obs_type distribution). Must update node feature logic.
2. **Company saturation** — company already has 10 obs types. Overlap penalty is real. But these are unique *signal types* (market microstructure, distress, regulatory) not covered by existing tools.
3. **FINRA API rate limits** — documented in existing tool, 0.12s delay between requests. L2 persistence adds negligible overhead.
4. **PACER costs** — creditor_filings uses RECAP (free mirror). If RECAP coverage is incomplete, this degrades gracefully.
5. **ClinicalTrials.gov API changes** — stable public API, low risk.

## Related

- [[gnn_guided_expansion_r2_spec]]
- [[gnn_guided_tool_expansion]] — Phase 16 first-round research
- [[l2_tool_expansion]] — Phase 13 L2 audit
- [[tool_priority_ranking]] — Phase 16 ranking artifact
- [[quant_training_ground]] — Master tracker
