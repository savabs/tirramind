---
title: "Research: Phase 35 — GNN Retrain on Expanded Entity Graph"
tags:
  - doc/research
  - phase/35
  - topic/gnn
  - layer/world-model
---

# Research: Phase 35 — GNN Retrain on Expanded Entity Graph

## Context

The L2 expansion roadmap (Phases 27–34) massively expanded the entity graph schema:
- Entity types: 6 → 11 (added cftc_contract, domain, instrument, organization, protocol, topic)
- Observation types: 18 → 45
- Link types: 3 → 18+ (7 instrument link types + 11 tool-level link types)
- ENRICHMENT_DIM: 18 → 54
- BASE_FEAT_DIM: 9 → 14

However, the GNN training infrastructure (`SyntheticGraphGenerator` in `trainer.py`) still only covers:
- 4/11 entity types (company, country, vessel, wallet)
- 7/45 observation types
- 3/18 link types (headquartered_in, port_call_to, exchange_based_in)

**The GNN has never been trained on the expanded schema.** Per the project methodology: *"After each batch of L2 upgrades, train the GNN on the current entity graph and evaluate."*

## Current Architecture

### SyntheticGraphGenerator (trainer.py)

Creates entities of 4 types → links them with 3 link types → generates Poisson-arrival observations → injects causal patterns.

**Entity type → obs_type mapping (current):**
```
company → [insider_trade, form144_filing, sell_intent]
country → [geopolitical_event]
vessel  → [port_call, vessel_position]
wallet  → [btc_transfer]
fallback → [cross_entity_pattern]
```

### Full Schema (Required)

**Entity type → obs_type mapping (expanded):**
```
cftc_contract → [futures_positioning]
company       → [insider_trade, form144_filing, sell_intent, patent_filing,
                  contract_award, creditor_filing, lobbying_spend, drug_approval,
                  short_interest, bankruptcy_status, investigation_signal]
country       → [geopolitical_event, sanctions_listing, cb_balance_sheet,
                  cb_policy_rate, economic_activity, capital_flow, sovereign_yield,
                  consumer_confidence, food_security, internet_disruption,
                  migration_pressure, trade_flow, border_throughput, pathogen_level,
                  campaign_finance, grid_demand]
domain        → [cert_issued, dns_change]
instrument    → [instrument_return, instrument_volatility, instrument_volume,
                  price_movement]
organization  → [regulatory_velocity]
person        → [insider_trade, sell_intent, campaign_finance]
protocol      → [tvl_change]
topic         → [pageview_spike, market_probability, research_velocity]
vessel        → [port_call, vessel_position]
wallet        → [btc_transfer, whale_trade]
```

**Link types (expanded):**
```
# Instrument links (from instrument_universe.py)
tracks_issuer     : instrument → company
located_in        : instrument/company → country
fx_base_country   : instrument → country
fx_quote_country  : instrument → country
tracks_protocol   : instrument → protocol
exchange_country  : instrument → country

# Tool-level links
cftc_tracks       : cftc_contract → instrument
works_for         : person → company
transacts_with    : wallet → wallet
trades_instrument : wallet → instrument
port_call_to      : vessel → country
lobbies_for       : company → company
operates_in       : company → country
awarded_by        : company → organization
debtor_of         : company → company
sanctioned_under  : country → country
market_authorized_in : company → country
```

### Injected Patterns (current vs expanded)

Currently only one pattern: company insider_trade → country geopolitical_event via headquartered_in.

**Expanded patterns should test cross-domain causal chains:**
1. `person.insider_trade → company.sell_intent via works_for` (insider activity → corporate stress)
2. `country.sanctions_listing → company.creditor_filing via operates_in` (sanctions → financial stress)
3. `vessel.port_call → country.trade_flow via port_call_to` (shipping → trade)
4. `wallet.btc_transfer → instrument.price_movement via trades_instrument` (whale flow → price)
5. `cftc_contract.futures_positioning → instrument.instrument_volatility via cftc_tracks` (positioning → vol)
6. `country.pathogen_level → country.economic_activity via sanctioned_under` (disease → economy, self-link)

## Observations

1. **No architectural changes needed.** HetTGN accepts configurable `in_channels` per type and `metadata` edge types. GraphBuilder already has the full schema constants. Only the SyntheticGraphGenerator needs expansion.

2. **Instrument entities are special.** They're created by `instrument_universe.py` (~90 instruments), not by the store's seed entity types. The generator should create a representative set of instruments with proper link types.

3. **ENRICHMENT_DIM alignment.** The obs_type distribution feature vector is len(OBSERVATION_TYPES) = 45. This is already correct in graph_builder.py. The generator just needs to produce observations of all types so the distribution features are non-trivial.

4. **Pattern recovery metrics.** The existing `compute_diagnostics()` → `format_diagnostic_report()` pipeline provides entity_type_density, obs_density, edge_type_attention, neighborhood_sparsity. These should pick up starved neighborhoods in the expanded graph.

5. **Backward compatibility.** The expanded generator should be a superset — old tests using the 4-type generator should still pass. Add new parameters with defaults that enable the expanded schema.

## Risks

1. **Training time increase.** 11 entity types × more observations = larger graph. Monitor epoch time and adjust `num_entities` per type to keep training practical.
2. **Sparse entity types.** Some types (domain, protocol, organization) may have very few entities in practice. The generator should reflect this with smaller counts.
3. **Obs type coverage.** With 45 obs types, some may be very rare. The obs_type prediction head needs enough examples of each type for the CE loss to be meaningful.
4. **Edge type sparsity.** Some link types connect only a few entities. HGT needs enough edges per type for attention to be meaningful (min ~10 edges per type).

## Data Requirements

- No external data needed. Synthetic generation covers the expanded schema.
- The PipelineStore API is unchanged — `register_entity()`, `link_entities()`, `store_entity_observation()`.
- HetTGN architecture unchanged — just different `in_channels` dimensions and `metadata`.

## Testing Plan

1. **Schema coverage test:** Every entity type, obs type, and link type appears in generated data
2. **Training convergence:** Loss decreases over 10 epochs
3. **Pattern recovery:** Injected cross-domain patterns are recovered by attention analysis
4. **Diagnostic coverage:** All entity types appear in diagnostic report
5. **Backward compatibility:** Old 4-type tests still pass
6. **Attention analysis:** Edge type attention weights are non-degenerate (no single edge type dominates 100%)

## Related

- [[phase34_commodity_links_diagnostic]] — previous phase
- [[quant_training_ground]] — master task
- [[gnn_guided_tool_expansion]] — original GNN diagnostic methodology
- [[l2_expansion_roadmap]] — completed L2 roadmap
