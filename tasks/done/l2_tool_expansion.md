---
title: "Task: L2 Tool Expansion"
tags:
  - doc/task
  - status/done
  - phase/13
  - topic/surveillance
  - topic/world-model
  - layer/surveillance
  - layer/world-model
---

# Task: L2 Tool Expansion

Status: completed
Research: [[l2_tool_expansion]]
Spec: [[l2_tool_expansion_spec]]

---

## Steps

### Phase 13a: Graph Builder + Entity Module Expansion
- [x] 13a.1: Fix insider_filings observation_type "purchase" → "insider_trade"
- [x] 13a.2: Expand EntityType Literal in entity.py (+domain, protocol, topic)
- [x] 13a.3: Expand ENTITY_TYPES list in graph_builder.py (6 → 9 types)
- [x] 13a.4: Expand OBSERVATION_TYPES list in graph_builder.py (8 → 15 types)
- [x] 13a.5: Add unknown-type fallback in _build_node_features
- [x] 13a.6: Build nodes for all types in id_map, not just hardcoded list
- [x] 13a.7: Write graph builder expansion tests (42 tests)

### Phase 13b: L2 Digital Infrastructure
- [x] 13b.1: Add L2 persistence to cert_transparency (domain entities, cert_issued obs)
- [x] 13b.2: Add L2 persistence to dns_monitor (domain entities, dns_change obs)
- [x] 13b.3: Add L2 persistence to wikipedia_pageviews (topic entities, pageview_spike obs)
- [x] 13b.4: Write L2 edge case tests for all 3 digital infra tools (34 tests)

### Phase 13c: L2 Corporate Intelligence
- [x] 13c.1: Add L2 persistence to lobbying (company entities, lobbying_spend obs)
- [x] 13c.2: Add L2 persistence to patent_filings (company entities, patent_filing obs)
- [x] 13c.3: Write L2 edge case tests for both corporate intel tools (part of 45-test suite)

### Phase 13d: L2 Energy + DeFi
- [x] 13d.1: Add L2 persistence to defi_flows (protocol entities, tvl_change obs)
- [x] 13d.2: Add L2 persistence to interconnection_queue (company entities, project_status obs)
- [x] 13d.3: Write L2 edge case tests for both energy/DeFi tools (part of 45-test suite)

### Phase 13e: Integration Verification
- [x] 13e.1: Write integration test (all L2 tools → store → graph_builder → HeteroData) (26 tests)
- [x] 13e.2: Run full GNN test suite (242 tests passing)
- [x] 13e.3: Run all new L2 tests (all passing)

---

## Future Phases (queued)

### Phase 14: Pattern Recovery Improvement
- Enhanced self-supervised meta-path learning
- Attention-weighted path scoring improvements
- Crystallization threshold tuning

### Phase 15: Outcome-Labeled Fine-Tuning
- Define outcome labels (market moves, event confirmations)
- Supervised fine-tuning on labeled entity-outcome pairs
- Walk-forward evaluation with outcome-based metrics

### Phase 16: GNN-Guided Tool Expansion
**Do not expand blindly.** After Phase 14/15, evaluate the trained GNN:
- Which entity types have sparse neighborhoods? → Upgrade those tools to L2
- Which attention heads are starved for signal? → Add observation channels there
- Which entity clusters are disconnected? → Build cross-domain linking tools
- Which aggregate tools (macro, sentiment, etc.) add value as conditioning vs. entity nodes?

Candidate tools for evaluation-driven expansion:
- **Company-entity tools** (high potential): gov_contracts, creditor_filings, bankruptcy_court, sanctions_monitor, drug_regulatory, job_postings, finra_short_volume, supply_chain_monitor
- **Aggregate tools** (likely stay L1, use as global features): treasury_receipts, consumer_sentiment, central_bank_balance, global_pmi, energy_supply, food_security, macro_data, capital_flows
- **Physical/geo tools** (potential new entity links): weather_alerts, disease_surveillance, earthquake_proximity, satellite_activity, transport_throughput
- **New tools to consider**: only if GNN evaluation reveals a specific entity-type gap or missing cross-domain edge that no existing tool can fill

---

## Phase 16 Results (GNN-Guided Expansion — 2026-04-10)

Full analysis in [[tool_priority_ranking]].

### Critical Finding: No Entity Links Exist

All 12 L2 tools persist entities and observations but **none call `store_entity_link()`**. The GNN has 9 entity types and 15 obs types but zero edges between entities. This must be fixed before further tool expansion has value.

### Diagnostic-Driven Priority Order

**Prerequisite**: Entity Linking Layer (~8h) — add `store_entity_link()` calls to existing L2 tools.

| Priority | Tool | Score | Primary Contribution |
|---|---|---|---|
| Tier 1 | sanctions_monitor L2 | 0.87 | Fills organization gap (0→1+ tools), cross-domain links |
| Tier 1 | gov_contracts L2 | 0.82 | Reinforces organization, adds procurement signal |
| Tier 1 | supply_chain_monitor L2 | 0.76 | Creates company→company edges |
| Tier 2 | finra_short_volume L2 | 0.72 | Microstructure signal for company |
| Tier 2 | creditor_filings L2 | 0.69 | Distress signal for company |
| Tier 2 | drug_regulatory L2 | 0.68 | Sector event signal for company |
| Tier 2 | disease_surveillance L2 | 0.63 | Diversify country obs |
| Tier 2 | bankruptcy_court L2 | 0.60 | Extreme distress signal |

Aggregate tools (treasury_receipts, consumer_sentiment, etc.) → stay L1, use as global conditioning.
Tier 3 tools → defer until re-evaluation after Tier 1 is live.

### Starved Entity Types

| Entity Type | Current Tools | Status |
|---|---|---|
| organization | 0 | **Critical — no tool produces this type** |
| country | 1 (gdelt) | High — needs diversification |
| protocol | 1 (defi_flows) | High — single obs channel |
| wallet | 1 (whale_alert) | High — BTC only |
| topic | 1 (wikipedia_pageviews) | High — single source |

## Related

- [[l2_tool_expansion]] — Phase 13 research
- [[l2_tool_expansion_spec]] — Phase 13 spec
- [[temporal_het_gnn]] — Phase 12 (GNN architecture)
- [[tool_priority_ranking]] — Phase 16 ranking artifact
- [[gnn_guided_tool_expansion]] — Phase 16 research
