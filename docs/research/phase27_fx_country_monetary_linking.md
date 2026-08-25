---
title: "Feature: Phase 27 — FX Country Wiring + Central Bank L2 Connectivity"
tags:
  - doc/research
  - phase/27
  - topic/fx
  - topic/central-bank
  - topic/entity-linking
  - layer/surveillance
  - layer/world-model
---

# Feature: Phase 27 — FX Country Wiring + Central Bank L2 Connectivity

## Current Architecture

- [[phase25_cross_domain_entity_linking]] connected instruments to company, country, CFTC contract, topic, and wallet structure.
- FX instruments in `agent/tools/instrument_universe.py` currently carry only a single `country` field, which captures one side of the pair.
- `agent/tools/central_bank_balance.py` is an aggregate analytics tool from [[7b-Z_central_bank_balance]]; it does not yet persist L2 observations into the entity graph.
- Country entities already exist and Phase 25 added deterministic `instrument -> country` links via `located_in`.
- `agent/models/gnn/graph_builder.py` still uses seeded one-hot entity typing. It can ingest unknown entity types structurally, but unknown types fall back to the default index in base features, so introducing a brand-new `central_bank` type now would add avoidable encoding debt.

## Observations

- The clearest remaining starved instrument class after Phase 25 is FX pairs. Phase 25's checkpoint explicitly identified FX as under-connected because the graph lacks monetary-policy state tied into country nodes.
- A direct L2 upgrade of [[7b-Z_central_bank_balance]] is the highest-signal next step because central-bank balance sheet and policy-rate state are causal drivers of FX regimes, carry, liquidity, and country ETF behavior.
- The cleanest representation with the current graph stack is not a new central-bank entity type. The cleaner immediate move is:
  1. enrich FX instruments with explicit two-country structure where deterministic;
  2. persist central-bank observations onto country entities already present in the graph.
- This choice avoids a graph-builder refactor while still giving the GNN new monetary-state signal on nodes that instruments already connect to.
- ECB should map to `EU` rather than decomposing to member states in this phase. That is explicit, deterministic, and consistent with the current instrument registry.

## Phase Choice Rationale

### Option A: New `central_bank` entities
- Pros: semantically precise, future-ready for institution-level reasoning.
- Cons: current graph builder still privileges seeded entity types; adding a new type now would create feature-encoding debt and broaden scope beyond the immediate signal need.

### Option B: Country-level monetary observations + explicit FX pair country wiring
- Pros: minimal architectural disruption, deterministic mapping, directly improves FX and country-sensitive instrument connectivity, reuses existing country entities.
- Cons: less institutionally expressive than dedicated central-bank entities.

### Recommendation
- Choose **Option B** for Phase 27.
- Defer dedicated `central_bank` entities until the graph-builder/type-encoding path is made fully dynamic.

## Data Requirements

- Existing `central_bank_balance` source coverage: Fed, ECB, BOJ, BOE, SNB, BOC, RBA.
- Deterministic country mappings:
  - Fed -> US
  - ECB -> EU
  - BOJ -> JP
  - BOE -> GB
  - SNB -> CH
  - BOC -> CA
  - RBA -> AU
- Deterministic FX pair mappings for the tracked universe, using explicit base/quote country metadata where the pair composition is unambiguous.

## Math/Algorithm Survey

- This phase does not introduce a new estimator; it improves the graph's causal evidence surface.
- The mathematical effect is on the message-passing substrate:
  - country nodes gain monetary-state observations,
  - FX instruments gain more complete country connectivity,
  - the HetTGN can learn cross-type attention over monetary-state signals instead of relying on sparse price-only consequences.
- Minimal high-signal observation set is preferable here:
  - one balance-sheet/liquidity observation family,
  - one policy-rate observation family.
- Avoid overloading the graph with every derived central-bank metric from the aggregate tool. The goal is causal state exposure, not duplicating the tool's formatted analytics output.

## Risks

- ECB -> `EU` is a useful but coarse mapping; eurozone country differentiation remains unresolved.
- Some FX pairs currently point only to the non-USD side; Phase 27 must fix this without guessing for synthetic or basket instruments.
- Central-bank source frequencies are mixed (weekly/monthly/daily). Observation timestamps must preserve source truth and not fabricate synchronized cadence.
- If country observations are too aggregated or too many observation types are added, enrichment dims will bloat without adding learnable signal.

## Implementation Constraints

- Reuse existing entity types where possible; do not introduce a new entity type in this phase.
- Prefer explicit metadata in the instrument registry over ticker-string parsing during ingest.
- Persist only source-backed relationships and observations.
- Keep the observation type set small and interpretable.

## Step-Local References

- [[phase25_cross_domain_entity_linking]] — completed instrument/country graph wiring baseline.
- [[phase25_gnn_diagnostic]] — prior diagnosis driving cross-domain expansion.
- [[7b-Z_central_bank_balance_sheets]] — source/tool semantics for central-bank balance state.
- [[entity_linking_layer]] — prior link persistence rules and explicit-relationship guardrails.
- [[e2e_global_integration]] — instrument registry and GNN instrument pipeline baseline.

## Related

- [[phase27_fx_country_monetary_linking_spec]]
- [[phase27_fx_country_monetary_linking]]
- [[phase25_cross_domain_entity_linking]]
- [[phase25_gnn_diagnostic]]
- [[7b-Z_central_bank_balance_sheets]]
- [[quant_training_ground]]
