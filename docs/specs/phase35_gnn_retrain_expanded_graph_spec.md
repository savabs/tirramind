---
title: "Spec: Phase 35 — GNN Retrain on Expanded Entity Graph"
tags:
  - doc/spec
  - phase/35
  - topic/gnn
  - layer/world-model
---

# Spec: Phase 35 — GNN Retrain on Expanded Entity Graph

## Goal

Update the GNN training infrastructure to cover the full 11-entity-type / 45-obs-type / 18-link-type schema, retrain the HetTGN, and run attention analysis to identify starved neighborhoods and guide future tool priorities.

## Files Affected

| File | Action |
|------|--------|
| `agent/models/gnn/trainer.py` | Modify — expand SyntheticGraphGenerator |
| `tests/test_phase35_gnn_retrain.py` | Create — edge case tests |
| `[[quant_training_ground]]` | Modify — add Phase 35 entry |

## Implementation Steps

### 35.1: Expand SyntheticGraphGenerator entity types

Add all 11 entity types to the generator with configurable counts:

```python
# New default entity counts (reflecting real-world proportions)
num_companies: int = 8,     # existing
num_countries: int = 5,     # expanded (was 3)
num_vessels: int = 4,       # existing
num_wallets: int = 4,       # existing
num_instruments: int = 10,  # NEW — representative set
num_persons: int = 6,       # NEW — insiders, execs
num_cftc_contracts: int = 4, # NEW
num_organizations: int = 3, # NEW — regulators, exchanges
num_protocols: int = 2,     # NEW — DeFi protocols
num_topics: int = 5,        # NEW — prediction markets, events
num_domains: int = 3,       # NEW — certificates
```

### 35.2: Expand `_obs_types_for()` mapping

Update to cover all 45 observation types across all 11 entity types. Uses the schema mapping from [[phase35_gnn_retrain_expanded_graph]].

### 35.3: Expand link type generation

Add all 18 link types with realistic connectivity patterns:
- instrument → company (tracks_issuer)
- instrument → country (located_in, fx_base, fx_quote, exchange_country)
- instrument → protocol (tracks_protocol)
- cftc_contract → instrument (cftc_tracks)
- person → company (works_for)
- wallet → wallet (transacts_with)
- wallet → instrument (trades_instrument)
- vessel → country (port_call_to) — existing
- company → country (headquartered_in) — existing + operates_in
- company → company (lobbies_for, debtor_of)
- company → organization (awarded_by)
- company → country (market_authorized_in)
- country → country (sanctioned_under)

### 35.4: Add 6 cross-domain injected patterns

Add patterns that test the new cross-domain causal chains:
1. `person.insider_trade → company.sell_intent via works_for`
2. `country.sanctions_listing → company.creditor_filing via operates_in` (reverse lookup)
3. `vessel.port_call → country.trade_flow via port_call_to`
4. `wallet.btc_transfer → instrument.price_movement via trades_instrument`
5. `cftc_contract.futures_positioning → instrument.instrument_volatility via cftc_tracks`
6. `country.pathogen_level → country.economic_activity via sanctioned_under`

### 35.5: Retrain HetTGN + verify convergence

- Build model with expanded graph
- Train for 10 epochs
- Verify loss decreases monotonically (no divergence)
- Verify obs_type prediction accuracy ≥ 15% top-1 (45 classes vs ~2.2% random)

### 35.6: Run attention analysis

- Use `compute_diagnostics()` on the retrained model
- Extract per-edge-type attention weights
- Identify starved entity neighborhoods
- Produce diagnostic report

### 35.7: Edge case tests

Cover: schema completeness, empty entity types, single-entity types, all obs types appear, all link types appear, training convergence, pattern injection across new types, backward compat with 4-type generator.

## Edge Cases

- Entity types with zero entities (should not crash graph builder)
- Obs types with zero observations (should still appear in enrichment dim)
- Link types connecting entity types that have only 1 entity (edge case for attention)
- Very asymmetric entity counts (e.g., 10 instruments, 2 protocols)

## Testing Plan

1. Schema coverage — assert all 11 entity types, 45 obs types, 18 link types present
2. Convergence — loss decreases over epochs
3. Pattern recovery — injected patterns detected in attention
4. Diagnostic completeness — all entity types in diagnostic report
5. Backward compat — old Trainer tests still pass
6. Edge cases — empty types, single entities, asymmetric counts

## Related

- [[phase35_gnn_retrain_expanded_graph]] — research doc
- [[quant_training_ground]] — master task
- [[gnn_guided_tool_expansion]] — Phase 16 diagnostics
- [[l2_expansion_roadmap]] — completed L2 roadmap
