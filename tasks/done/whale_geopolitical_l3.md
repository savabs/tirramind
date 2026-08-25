---
title: "Task: Whale × Geopolitical L3 Pattern — Phase 11d"
tags:
  - doc/task
  - status/done
  - phase/11
  - topic/surveillance
  - topic/convergence
  - layer/feature-engineering
---

# Task: Whale × Geopolitical L3 Pattern — Phase 11d

Status: completed
Research: [[whale_geopolitical_l3]]
Spec: [[whale_geopolitical_l3_spec]]

## Steps

- [x] 11d.1: Constants + resolve_wallet_exchange() — wallet address → (exchange, country_fips)
- [x] 11d.2: seed_whale_country_links() — scan wallet entities → create `exchange_based_in` links
- [x] 11d.3: detect_whale_geopolitical() — co-occurrence detector for whale × GDELT events
- [x] 11d.4: Edge case test suite (24 tests — 5 resolve, 6 seed, 11 detect, 2 integration)

## Notes

- Extends `cross_entity.py` — no new files needed.
- Wallet→country links via known exchange address matching.
- KNOWN_EXCHANGE_WALLETS dict is empty by default — populated via parameter or external config.
- Goldstein threshold: -5.0 (stricter than other patterns — high-impact events only).
- Window: 24h (crypto is near-instant, tightest of all three patterns).
- Scoring includes value_btc weight: `min(btc/100, 1) × severity × proximity`.
- Same store_l3_observations() for persistence (pattern_type="whale_x_geopolitical").

## Related

- [[whale_geopolitical_l3]]
- [[whale_geopolitical_l3_spec]]
- [[cross_entity_l3]]
- [[vessel_sanctions_l3]]
