---
title: "Task: Phase 27 — FX Country Wiring + Central Bank L2 Connectivity"
tags:
  - doc/task
  - status/done
  - phase/27
  - topic/fx
  - topic/central-bank
  - topic/entity-linking
  - layer/surveillance
  - layer/world-model
---

# Task: Phase 27 — FX Country Wiring + Central Bank L2 Connectivity

Status: completed
Research: [[phase27_fx_country_monetary_linking]]
Spec: [[phase27_fx_country_monetary_linking_spec]]

## Goal

Connect FX instruments and country-sensitive assets to causal monetary-state signal through deterministic country wiring and country-level central-bank observations.

## Scope Notes

- Layer: surveillance + world-model
- Primary target: FX instruments identified as starved after [[phase25_cross_domain_entity_linking]]
- Secondary benefit: country ETFs and other country-linked instruments gain monetary-state evidence on country nodes
- Non-goals: new `central_bank` entity type, graph-builder type-system refactor, full macro-data L2 expansion

## Steps

- [x] 27.1: Add deterministic two-country metadata for FX pairs in the instrument registry
  Verification: 12 tests pass — all 15 FX pairs have explicit base_country/quote_country, non-FX have None, legacy country preserved, codes are ISO, fields are frozen
- [x] 27.2: Persist FX instrument links to both relevant country nodes
  Verification: 14 tests pass — 15 fx_base_country + 15 fx_quote_country links created, correct targets (EURUSD→EU+US, GBPJPY→GB+JP), idempotent, legacy located_in unchanged, non-FX unaffected
- [x] 27.3: Upgrade `central_bank_balance` to persist compact monetary-state observations onto country entities
  Verification: 26 L2 tests pass — balance_sheets→cb_balance_sheet, policy_divergence→both families, rate_monitor→cb_policy_rate, liquidity_index→no obs, no-store guard, exception non-fatal, country entity IDs deterministic, depth_level=2, CB_TO_COUNTRY complete and ISO, obs value fields correct
- [x] 27.4: Update graph registries and integration tests for the new monetary observation types
  Verification: 71 graph-builder tests pass — cb_balance_sheet + cb_policy_rate in OBSERVATION_TYPES, ENRICHMENT_DIM=38 (9+29), obs types sorted alphabetically, count assertion updated
- [x] 27.5: Re-run diagnostics and confirm improved FX-country connectivity
  Verification: 11 integration tests pass — FX instruments gain dual-country links (EURUSD→EU+US, EURGBP→EU+GB no US), CB observations land on country nodes and survive store→graph flow, full path USDJPY→US←cb_balance_sheet(fed) verified end-to-end
- [x] 27.6: Run regression and write checkpoint
  Verification: 292/292 tests pass across instrument_universe (82), central_bank_balance (110), graph_builder (71), graph_builder_expanded (18), phase27_diagnostic (11). Two previously-stale count assertions fixed. No Phase 27 regressions.

## Completion Checklist

- [x] Research note exists and is current
- [x] Spec matches the intended implementation plan
- [x] Each completed step has a verification result
- [x] Edge-case testing was added and run for code changes
- [x] Checkpoint written at the end of the phase
- [x] Frontmatter tags and `## Related` section are current

## Related

- [[phase27_fx_country_monetary_linking]]
- [[phase27_fx_country_monetary_linking_spec]]
- [[quant_training_ground]]
- [[phase25_cross_domain_entity_linking]]
- [[7b-Z_central_bank_balance_sheets]]
