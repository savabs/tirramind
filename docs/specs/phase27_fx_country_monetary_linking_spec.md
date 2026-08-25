---
title: "Spec: Phase 27 — FX Country Wiring + Central Bank L2 Connectivity"
tags:
  - doc/spec
  - phase/27
  - topic/fx
  - topic/central-bank
  - topic/entity-linking
  - layer/surveillance
  - layer/world-model
---

# Spec: Phase 27 — FX Country Wiring + Central Bank L2 Connectivity

## Goal

Connect starved FX instruments to causal monetary-state signal by:
1. making FX pair country structure explicit in the instrument registry, and
2. upgrading `central_bank_balance` to persist compact L2 monetary observations onto country entities.

## Trusted Inputs

- [[phase27_fx_country_monetary_linking]]
- [[phase25_cross_domain_entity_linking]]
- [[7b-Z_central_bank_balance_sheets]]
- [[entity_linking_layer]]
- [[e2e_global_integration]]

## Files Affected

Expected primary files:
- `agent/tools/instrument_universe.py`
- `agent/tools/central_bank_balance.py`
- `agent/models/gnn/graph_builder.py`
- `tests/test_instrument_universe.py`
- `tests/test_central_bank_balance_edge.py`
- `tests/test_graph_builder_expanded.py`
- `tests/test_backfill_instruments.py` and/or a dedicated Phase 27 integration test file

## Implementation Steps

### 27.1: Add explicit FX pair country metadata
- Extend the instrument registry so FX pairs can carry deterministic two-country structure.
- Prefer explicit metadata fields in `InstrumentDef` over ticker parsing.
- Only encode pair sides where deterministic from the maintained universe.

### 27.2: Persist multi-country FX links
- Extend instrument-link persistence so FX instruments can link to both relevant country entities.
- Preserve idempotency and explicit-only rules.
- Do not create guessed company/issuer links for FX instruments.

### 27.3: Upgrade `central_bank_balance` to L2 country observations
- Add optional `PipelineStore` support.
- Persist compact monetary-state observations onto country entities for covered central banks.
- Keep the observation families minimal, likely:
  - central-bank balance/liquidity state
  - policy-rate state
- Use deterministic bank -> country mapping only.

### 27.4: Update graph builder type registries and integration tests
- Add the new observation types needed for monetary-state persistence.
- Confirm feature dimensions update cleanly.
- Ensure graph/integration tests prove the observations survive store -> graph builder flow.

### 27.5: Re-run diagnostics with FX focus
- Validate that FX instruments gain richer country connectivity.
- Confirm country nodes now carry monetary-state observations visible to the graph.
- Record which FX pairs or country-sensitive instruments remain starved after this phase.

## Edge Cases

- ECB maps to `EU`, not individual eurozone members, in this phase.
- Pairs involving USD must gain explicit US connectivity where the pair definition is deterministic.
- Synthetic or ambiguous instruments must remain partially linked rather than guessed.
- Missing central-bank data for one bank must not block persistence for others.
- Observation timestamps must preserve the source series timing even when banks publish at different frequencies.

## Testing Plan

- FX metadata tests for explicit two-country coverage and no guessed mappings.
- Link-persistence tests for one-country and two-country instrument cases.
- Central-bank L2 tests for partial data, missing mappings, stale data, idempotency, and DB failures.
- Graph-builder integration tests asserting the new observation types are present and reach country nodes.
- Diagnostic rerun proving improved FX-country connectivity relative to the Phase 25 baseline.

## Related

- [[phase27_fx_country_monetary_linking]]
- [[phase25_cross_domain_entity_linking]]
- [[phase25_gnn_diagnostic]]
- [[7b-Z_central_bank_balance_sheets]]
- [[quant_training_ground]]
