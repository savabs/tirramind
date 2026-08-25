---
title: "Research: L2 Tool Expansion — Entity Persistence for GNN"
tags:
  - doc/research
  - phase/13
  - topic/surveillance
  - topic/world-model
  - layer/surveillance
  - layer/world-model
---

# Research: L2 Tool Expansion — Entity Persistence for GNN

## Problem

Phase 12 delivered a Temporal Heterogeneous GNN (172/172 tests passing). But only 5 of ~57 tools feed it via L2 entity persistence (insider_filings, form144, whale_alert, gdelt, ais_vessel). The remaining tools are invisible to the GNN. Cross-entity pattern discovery requires broad surveillance surface coverage.

## Current L2 Pattern

Established by Phase 7b–7c tools:
1. `TYPE_CHECKING` import → `PipelineStore`
2. `pipeline_store: PipelineStore | None = None` kwarg in `__init__`
3. `_persist_entities(results)` called in `execute()` after fetching data
4. `_persist_entities_inner(results)` does the actual work:
   - `entity_id_from_key(type, key)` → deterministic SHA-256 ID
   - `store.register_entity(entity_type, canonical_name, entity_id)`
   - `store.add_entity_alias(entity_id, source, external_id)`
   - `store.store_entity_observation(entity_id, source_tool, observed_at, observation_type, depth_level, value)`
5. Any persistence error is caught and logged — never blocks tool execution

## Graph Builder Constraints

`ENTITY_TYPES` (6 hardcoded types) and `OBSERVATION_TYPES` (8 hardcoded types) determine:
- One-hot encoding dimensions in node features
- Model output head size (obs_type prediction)
- Temporal encoder vocabulary

Entities/observations with unknown types are either silently dropped or default to index 0.

**Critical bug**: `insider_filings.py` writes `observation_type="purchase"` but `OBSERVATION_TYPES` has `"insider_trade"`. These observations get misencoded.

## Tool Audit — Upgrade Priority

### VERY HIGH (7 tools) — Clear entities, strong temporal signal, cross-link potential

| Tool | Entity Types | Observation Types | Key Fields |
|------|-------------|-------------------|------------|
| cert_transparency | domain, company | cert_issued | domain, entry_timestamp, is_expired |
| dns_monitor | domain, company | dns_change | domain, records, cloud_providers, min_ttl |
| wikipedia_pageviews | topic | pageview_spike | article, z_score, latest_views, date |
| lobbying | company, organization | lobbying_spend | registrant_name, client_name, income, issues |
| patent_filings | company, person | patent_filing | assignee_organization, patent_date, cpc_subgroup_id |
| defi_flows | protocol | tvl_change | name, tvl_usd, chain, change_1d_pct |
| interconnection_queue | company | project_status | entity_name, nameplate_capacity_mw, energy_source_code, status |

### LOW/SKIP (~17 tools) — Aggregate data, no entity resolution

building_permits, capital_flows, cftc, comtrade, consumer_sentiment, energy_supply, finra_short_volume, food_security, global_pmi, job_postings, labor_disruptions, liquidity_regime, macro_data, market_data, power_grid, sovereign_debt, supply_chain_monitor, treasury_receipts

### MEDIUM (~10 tools) — Deferred to Phase 14

bankruptcy_court, central_bank_balance, creditor_filings, disease_surveillance, drug_regulatory, earthquake_proximity, election/political_risk, foia_requests, internet_infrastructure, internet_outages, migration_flows, regulatory_gazette, sanctions_monitor, satellite_activity, transport_throughput, weather_alerts

## New Types Required

**Entity types to add:** `domain`, `protocol`, `topic`
**Observation types to add:** `cert_issued`, `dns_change`, `pageview_spike`, `lobbying_spend`, `patent_filing`, `tvl_change`, `project_status`
**Fix:** `insider_filings` observation_type `"purchase"` → `"insider_trade"`

## Risks

1. Model architecture depends on type list sizes for output head dimensions. Expanding lists changes model shape — existing checkpoints incompatible (acceptable: no production checkpoints yet).
2. interconnection_queue returns text-only ToolResult (no `data` dict). Must extract structured data before text formatting.
3. Entity deduplication across tools (e.g., company "Tesla" from lobbying vs patent_filings) relies on `normalize_company_name` which only handles SEC-style names. Lobbying registrant names may differ.
4. wikipedia_pageviews articles don't have entity type classification. Using generic `topic` type; cross-entity linking deferred to L3.

## Related

- [[temporal_het_gnn]] — Phase 12 research (GNN architecture)
- [[temporal_het_gnn_spec]] — Phase 12 spec
- [[l2_tool_expansion_spec]] — Phase 13 spec
- [[l2_tool_expansion]] — Phase 13 task
