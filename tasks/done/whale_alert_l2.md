---
title: "Task: Deep Surveillance Phase 10b.3 — whale_alert L2"
tags:
  - doc/task
  - status/done
  - phase/10
  - topic/surveillance
  - layer/surveillance
---

# Task: Deep Surveillance Phase 10b.3 — whale_alert L2 Upgrade

Status: completed
Research: [[whale_alert_l2]]
Spec: [[whale_alert_l2_spec]]

## Steps

- [x] 10b.3.1: Add TYPE_CHECKING + entity imports to whale_alert.py
- [x] 10b.3.2: Accept optional PipelineStore in constructor (keyword-only)
- [x] 10b.3.3: Implement `_persist_entities()` + `_persist_entities_inner()` — register wallets, store btc_transfer observations
- [x] 10b.3.4: Add `entity_ids` mapping to parsed transaction dicts
- [x] 10b.3.5: Edge case test suite
- [x] 10b.3.6: MI measurement integration test

## Notes

- Same L2 wiring pattern as [[deep_surveillance_10b|insider_filings (10b.1)]] and [[deep_surveillance_10b2|form144 (10b.2)]]
- Entity type = "wallet", alias source = "btc_address"
- Both sender and receiver are entities (unlike SEC filings where insider is the primary entity)
- observation_type = "btc_transfer" with direction field (in/out)
- Only addresses from filtered txs (above min_btc) are persisted

## Related

- [[whale_alert_l2]]
- [[whale_alert_l2_spec]]
- [[deep_surveillance_tools]]
- [[deep_surveillance_10b]]
- [[deep_surveillance_10b2]]
