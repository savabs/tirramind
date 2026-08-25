---
title: "Task: Phase 25 Cross-Domain Entity Linking"
tags:
  - doc/task
  - status/done
  - phase/25
  - topic/gnn-expansion
  - topic/entity-linking
  - topic/instruments
  - layer/world-model
  - layer/surveillance
---

# Task: Phase 25 Cross-Domain Entity Linking

Status: completed
Research: [[phase25_gnn_diagnostic]]
Spec: [[phase25_cross_domain_entity_linking_spec]]

## Goal

Connect instrument entities to the broader entity graph, then upgrade the highest-value remaining L1 tools that can feed those new cross-domain edges.

## Scope Notes

- Layer: surveillance + world-model
- Main files expected to change: `agent/tools/instrument_universe.py`, `agent/tools/cftc.py`, `agent/tools/polymarket.py`, `agent/tools/polymarket_whales.py`, `agent/models/gnn/graph_builder.py`, targeted tests
- Non-goals: production deployment hardening, portfolio-policy changes, inference DAG redesign

## Steps

- [x] 25.1: Define deterministic instrument issuer and country metadata in the instrument registry
  Verification: ✅ All 90 instruments enriched with issuer/country/cftc_code. ETFs have issuers, futures have CFTC codes verified against live cftc.gov data. 18 CFTC codes validated.
- [x] 25.2: Persist instrument-to-company and company-to-country links and verify graph-builder integration
  Verification: ✅ `_persist_instrument_links()` creates `tracks_issuer` and `located_in` links. 27 new tests + 3 fixed assertions = 73 total instrument tests green.
- [x] 25.3: Upgrade `cftc.py` to persist L2 entities/observations that connect to instrument nodes
  Verification: ✅ CFTC L2 persists `cftc_contract` entities with `futures_positioning` observations, linked to instruments via `cftc_tracks`. 33 tests covering normal, missing-map, malformed, duplicate, empty-result, and DB-error cases. All green.
- [x] 25.4: Upgrade `polymarket.py` and `polymarket_whales.py` to persist L2 topic/participant structure
  Verification: ✅ Polymarket persists `topic` entities with `market_probability` observations. Whales persist `wallet` entities with `whale_trade` observations. 28 tests covering dedup, edge cases, case-insensitive wallets, missing slugs, DB errors. All green.
- [x] 25.5: Re-run graph diagnostics and confirm instrument neighborhoods are less isolated
  Verification: ✅ 26 cross-domain integration tests confirm: instrument→company→country chain, cftc_contract→instrument links, topic/wallet nodes in graph, instruments now have non-zero degree. All GNN training/diagnostic tests pass (20/20).
- [x] 25.6: Run regression and edge-case suites, then write a checkpoint
  Verification: ✅ 203 tests across all Phase 25 test files green. 371 graph/entity tests green. Checkpoint written.

## Completion Checklist

- [x] Research note exists and is current
- [x] Spec matches the actual implementation plan
- [x] Each completed step has a verification result
- [x] Edge-case testing was added and run for code changes
- [x] Checkpoint written at the end of the session or sub-phase
- [x] Frontmatter tags and `## Related` section are current

## Related

- [[phase25_gnn_diagnostic]]
- [[phase25_cross_domain_entity_linking_spec]]
- [[quant_training_ground]]
- [[entity_linking_layer]]
- [[e2e_global_integration]]

## Notes

- Prefer explicit, source-backed structural edges over inferred relationships.
- If an instrument cannot be linked deterministically, leave it unlinked and document the gap.
- Run the edge-case suite after each completed sub-phase, not only at the end.