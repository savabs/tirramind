---
title: "Tool Priority Ranking: GNN-Guided Expansion"
tags:
  - doc/research
  - phase/16
  - topic/surveillance
  - topic/world-model
  - layer/surveillance
  - layer/world-model
---

# Tool Priority Ranking: GNN-Guided Expansion

**Generated**: Phase 16c — 2026-04-10
**Method**: 5-dimension weighted rubric from [[gnn_guided_tool_expansion]]
**Evidence basis**: Synthetic diagnostic baseline ([[gnn_diagnostics_synthetic]]) + architectural coverage analysis. Real PipelineStore was empty ([[gnn_diagnostics_real]]), so ranking is architecture-driven.

---

## Executive Summary

The single highest-leverage action is **not** adding more tools — it's **adding entity links**. The 12 existing L2 tools produce 9 entity types and 15 observation types, but **zero entity links**. The GNN cannot propagate signal across isolated nodes. Until `company→headquartered_in→country`, `person→works_for→company`, and similar edges exist, adding more observation channels has diminishing returns.

After entity linking, the priority order is: fill starved entity types (organization, country, protocol, wallet, topic) with new observation channels, then add cross-domain links.

---

## Gap Analysis (16c.1)

### Structural Gap: No Entity Links

The synthetic diagnostic showed `port_call_to` and `exchange_based_in` edges receiving **zero attention** — but that's because these edges only exist in the synthetic generator, not in the real tool layer. **No L2 tool calls `store_entity_link()`**. Every entity is an isolated node.

**Impact**: The GNN's heterogeneous message-passing and attention mechanisms are designed to learn from edge-connected neighborhoods. Without edges, the model degrades to per-type autoencoders with no cross-entity signal propagation. This blocks the entire Phase 14/15/16 value chain.

**Fix**: Build an entity linking layer (or add link persistence to existing tools) before expanding the tool set further.

### Entity-Type Coverage Gaps

| Entity Type | Tools | Obs Types | Gap Severity | Notes |
|---|---|---|---|---|
| organization | 0 | 0 | **Critical** | No tool produces organization entities. Sanctions, GDELT org actors, UN agencies — all unmapped. |
| country | 1 | 1 | **High** | Only GDELT feeds country entities. No economic, trade, or regulatory signal per country. |
| protocol | 1 | 1 | **High** | Only defi_flows. No governance, exploit, or bridge-flow signal. |
| wallet | 1 | 1 | **High** | Only whale_alert BTC. No multi-chain, DeFi wallet, or exchange flow data. |
| topic | 1 | 1 | **High** | Only Wikipedia pageviews. No Reddit, news sentiment, or search trend signal. |
| vessel | 1 | 2 | **Medium** | AIS provides position + port_call. Could add sanctions/flag-state/cargo data. |
| person | 2 | 2 | **Medium** | Insider filings + Form 144 both from SEC. Could add political contributions, board memberships. |
| domain | 2 | 2 | **Medium** | Cert + DNS. Two tools already, limited cross-domain value. |
| company | 5 | 5 | **Low** | Already dense. Adding more has diminishing returns (overlap penalty). |

### Edge-Type Attention Gaps (from synthetic)

| Edge Type | Synthetic Attention | Interpretation |
|---|---|---|
| `headquartered_in` | 0.500 | Sole signal carrier — model ignores other edges |
| `port_call_to` | 0.000 | Zero attention — endpoint types (vessel, country) both sparse |
| `exchange_based_in` | 0.000 | Zero attention — endpoint types (wallet, country) both sparse |

**Implication**: Fixing vessel and wallet sparsity (more observations) AND adding entity links in the real tool layer may revive attention on `port_call_to` and `exchange_based_in`.

---

## Candidate Scoring (16c.2)

### Rubric

| Dimension | Weight | Scoring Guide |
|---|---|---|
| Graph connectivity gain | 0.30 | How much does it increase edges or mean_degree for sparse types? |
| Signal uniqueness | 0.25 | New obs type, or density boost for a starved one? |
| Implementation effort | 0.20 | L2 upgrade ≈ 2–4h (score 0.8); new tool ≈ 8–16h (score 0.3–0.5) |
| Data quality risk | 0.15 | Free API stability, update frequency, historical depth |
| Overlap penalty | 0.10 | Penalty if target entity type already has ≥ 3 obs types |

Scores are 0.0–1.0 per dimension. Final = weighted sum.

### Pre-Candidate: Entity Linking Layer (not scored — prerequisite)

This is not a "tool" but an infrastructure step. It must happen first:
- Parse SEC EDGAR data for `person→works_for→company` links
- Use GDELT actor metadata for `company→headquartered_in→country`
- Use whale_alert exchange labels for `wallet→exchange_based_in→country`
- Use AIS vessel registry for `vessel→flagged_in→country`
- **Estimated effort**: 4–8h (link extraction from existing L2 data, no new API calls)

### Scored Candidates

| # | Tool | Target Entity | Connectivity | Uniqueness | Effort | Quality | Overlap | **Score** |
|---|---|---|---|---|---|---|---|---|
| 1 | **sanctions_monitor** L2 | organization, company, vessel, person | 0.95 | 0.90 | 0.70 | 0.80 | 0.90 | **0.87** |
| 2 | **gov_contracts** L2 | company, organization | 0.80 | 0.85 | 0.75 | 0.85 | 0.80 | **0.82** |
| 3 | **supply_chain_monitor** L2 | company (→company links) | 0.90 | 0.80 | 0.50 | 0.70 | 0.70 | **0.76** |
| 4 | **finra_short_volume** L2 | company | 0.50 | 0.85 | 0.80 | 0.90 | 0.60 | **0.72** |
| 5 | **creditor_filings** L2 | company | 0.55 | 0.80 | 0.75 | 0.75 | 0.60 | **0.69** |
| 6 | **drug_regulatory** L2 | company | 0.50 | 0.80 | 0.70 | 0.85 | 0.60 | **0.68** |
| 7 | **job_postings** L2 | company | 0.40 | 0.70 | 0.60 | 0.60 | 0.50 | **0.56** |
| 8 | **bankruptcy_court** L2 | company | 0.45 | 0.75 | 0.60 | 0.70 | 0.55 | **0.60** |
| 9 | **transport_throughput** L2 | country, vessel | 0.60 | 0.65 | 0.40 | 0.65 | 0.80 | **0.60** |
| 10 | **disease_surveillance** L2 | country | 0.55 | 0.70 | 0.50 | 0.75 | 0.85 | **0.63** |
| 11 | **satellite_activity** L2 | vessel, country | 0.60 | 0.75 | 0.30 | 0.50 | 0.80 | **0.59** |
| 12 | **weather_alerts** L2 | country | 0.45 | 0.55 | 0.50 | 0.70 | 0.85 | **0.55** |
| 13 | **earthquake_proximity** L2 | country | 0.40 | 0.50 | 0.50 | 0.75 | 0.85 | **0.53** |

**Aggregate tools (not scored — stay L1)**: treasury_receipts, consumer_sentiment, central_bank_balance, global_pmi, energy_supply, food_security, macro_data, capital_flows. These produce country-level or market-level numbers. They are better consumed as **global conditioning variables** (time-varying bias in the GNN) rather than entity nodes. Forcing them into the entity graph would inflate country-node density without adding discriminative signal.

---

## Scoring Rationale

### sanctions_monitor (0.87) — Tier 1
**Why top-ranked**: Only candidate that touches 4 entity types simultaneously (organization, company, vessel, person). The OFAC/SDN list is free, stable, and deeply historical. Sanctions create cross-domain links (company→sanctioned_with→person, vessel→sanctioned_by→organization) that no other tool provides. Fills the **organization** gap entirely (currently 0 tools). New obs type: `sanctions_designation`.

### gov_contracts (0.82) — Tier 1
**Why high**: Creates company+organization entities via USAspending.gov (free API, excellent data quality). Government contracts reveal company dependencies on federal agencies (organization entities). Adds `contract_award` obs type — unique signal about revenue concentration and political exposure. Low overlap with existing company obs types (none are procurement-related).

### supply_chain_monitor (0.76) — Tier 1
**Why high**: Unique in that it creates **company→supplies→company** edges — the only candidate that adds intra-type entity links. Supply chain disruptions are predictive of earnings surprises. However, reliable free supply chain APIs are limited (score penalized on quality/effort). Possible sources: SEC 10-K supplier mentions, import/export records.

### finra_short_volume (0.72) — Tier 2
**Why mid-high**: Adds `short_volume` obs type to company entities — unique market-microstructure signal not covered by any existing tool. FINRA data is free and daily. Scored down because it only targets the already-dense company type (overlap penalty) and adds no new entity links.

### creditor_filings (0.69) — Tier 2
Company-focused (UCC filings). Adds `lien_filing` obs type — unique distress signal. Free via state-level APIs but fragmented (quality risk).

### drug_regulatory (0.68) — Tier 2
Company-focused (FDA/EMA approvals). Adds `drug_approval` obs type. Free openFDA API. Narrow sector coverage (pharma/biotech only).

### disease_surveillance (0.63) — Tier 2
Country-focused (WHO/CDC). Adds `disease_outbreak` obs type. Helps diversify country node beyond GDELT geopolitical events.

### bankruptcy_court (0.60) — Tier 2
Company-focused (PACER). Adds `bankruptcy_filing` obs type. Strong distress signal but PACER access is not fully free (quality risk scored down).

### Others (≤ 0.60) — Tier 3
Remaining candidates either target already-dense types, require expensive new data sources, or add marginal signal. They should wait until the GNN is re-evaluated on real data after Tier 1 + entity linking are complete.

---

## Tier Summary (16c.3)

### Prerequisite: Entity Linking Layer
| Action | Estimated Effort | Entity Types Affected |
|---|---|---|
| Build `store_entity_link()` calls in existing L2 tools | 4–8h | all (connects isolated nodes) |
| Extract person→works_for→company from SEC data | 2h | person, company |
| Extract company→headquartered_in→country from GDELT/SEC | 2h | company, country |
| Extract wallet→exchange_based_in→country from whale_alert | 1h | wallet, country |
| Extract vessel→flagged_in→country from AIS registry | 1h | vessel, country |

### Tier 1 (build next — highest ROI)

| Rank | Tool | Score | Primary Gap Filled | New Entity Types | New Obs Types |
|---|---|---|---|---|---|
| 1 | sanctions_monitor L2 | 0.87 | organization (0→1+), cross-domain links | organization | sanctions_designation |
| 2 | gov_contracts L2 | 0.82 | organization density, company→org links | organization | contract_award |
| 3 | supply_chain_monitor L2 | 0.76 | company→company edges (intra-type links) | — | supply_disruption |

### Tier 2 (build after Tier 1 + re-evaluate)

| Rank | Tool | Score | Primary Gap Filled |
|---|---|---|---|
| 4 | finra_short_volume L2 | 0.72 | unique microstructure signal for company |
| 5 | creditor_filings L2 | 0.69 | distress signal for company |
| 6 | drug_regulatory L2 | 0.68 | sector event signal for company |
| 7 | disease_surveillance L2 | 0.63 | diversify country obs beyond GDELT |
| 8 | bankruptcy_court L2 | 0.60 | extreme distress signal for company |

### Tier 3 (defer — reassess after real-data diagnostics)

| Rank | Tool | Score | Reason Deferred |
|---|---|---|---|
| 9 | transport_throughput | 0.60 | Moderate effort, limited free data |
| 10 | satellite_activity | 0.59 | High effort, data quality uncertain |
| 11 | job_postings | 0.56 | Company-dense, free APIs unreliable |
| 12 | weather_alerts | 0.55 | Country-focused, weak predictive link |
| 13 | earthquake_proximity | 0.53 | Country-focused, rare events |

### Aggregate tools (stay L1 — global conditioning)

treasury_receipts, consumer_sentiment, central_bank_balance, global_pmi, energy_supply, food_security, macro_data, capital_flows — feed as time-varying global features, not entity nodes.

---

## Recommended Execution Order

1. **Entity Linking Layer** (prerequisite, ~8h) — add `store_entity_link()` to existing L2 tools so the GNN has edges to learn from.
2. **sanctions_monitor L2** — fills the organization gap, creates cross-domain links.
3. **gov_contracts L2** — reinforces organization type, adds procurement signal.
4. **supply_chain_monitor L2** — creates company→company edges (unique structural contribution).
5. **Re-run diagnostics** on real PipelineStore after tools 1–4 are live. Use results to decide whether Tier 2 tools are needed or if the graph is healthy enough for Phase 14/15 improvements.

---

## Re-Evaluation Triggers

Re-run `run_diagnostics()` and revisit this ranking when:
- Tier 1 tools are implemented and have populated real data
- Entity linking layer is live and edges exist in the graph
- A new data source becomes available that changes the candidate set
- The GNN attention distribution shifts significantly after retraining on real data

---

## Related

- [[gnn_guided_tool_expansion]] — Phase 16 research (rubric definition)
- [[gnn_guided_tool_expansion_spec]] — Phase 16 spec
- [[gnn_diagnostics_synthetic]] — synthetic baseline
- [[gnn_diagnostics_real]] — real store diagnostic (empty)
- [[l2_tool_expansion]] — Phase 13 candidate catalog
- [[temporal_het_gnn]] — Phase 12 GNN architecture
