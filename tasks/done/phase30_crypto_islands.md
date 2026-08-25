---
title: "Task: Phase 30 — Crypto Islands + Cross-Domain Linking"
tags:
  - doc/task
  - status/done
  - phase/30
  - topic/entity-linking
  - topic/crypto
  - layer/surveillance
  - layer/world-model
---

# Task: Phase 30 — Crypto Islands + Cross-Domain Linking

Status: completed
Research: [[crypto_islands_cross_domain_linking]]
Spec: [[crypto_islands_cross_domain_linking_spec]]

## Goal

Connect BTC-USD and ETH-USD to the entity graph via protocol and wallet links. After this phase, no crypto instrument is a graph island.

## Steps

- [x] 30.1: Add `protocol` field to `InstrumentDef`; update BTC-USD and ETH-USD entries
- [x] 30.2: Extend `_persist_instrument_links` to create `tracks_protocol` links for crypto
- [x] 30.3: Verify protocol entity ID consistency between instrument_universe and defi_flows
- [x] 30.4: Add `trades_instrument` links from whale wallets to BTC-USD in `whale_alert._persist_entities_inner`
- [x] 30.5: Edge case tests (`tests/test_phase30_crypto_links.py`) — 28 tests
- [x] 30.6: Integration diagnostics (`tests/test_phase30_diagnostic.py`) — 13 tests
- [x] 30.7: Targeted regression (261 tests pass) + checkpoint

## Files to Modify

- `agent/tools/instrument_universe.py` — InstrumentDef field + crypto entries + link function
- `agent/tools/whale_alert.py` — wallet → instrument links
- `tests/test_phase30_crypto_links.py` — CREATE
- `tests/test_phase30_diagnostic.py` — CREATE

## Related

- [[crypto_islands_cross_domain_linking]]
- [[crypto_islands_cross_domain_linking_spec]]
- [[starved_class_audit]]
- [[l2_expansion_roadmap]]
- [[quant_training_ground]]
