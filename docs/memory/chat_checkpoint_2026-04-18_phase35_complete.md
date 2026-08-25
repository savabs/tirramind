---
title: "Checkpoint: Phase 35 — GNN Retrain on Expanded Entity Graph"
tags:
  - doc/checkpoint
  - phase/35
  - topic/gnn
  - layer/world-model
---

# Phase 35 Complete — GNN Retrain on Expanded Entity Graph

**Date:** 2026-04-18
**Task:** [[quant_training_ground]]
**Research:** [[phase35_gnn_retrain_expanded_graph]]
**Spec:** [[phase35_gnn_retrain_expanded_graph_spec]]

## What Was Done

Expanded `SyntheticGraphGenerator` in `agent/models/gnn/trainer.py` from 4 entity types / 7 obs types / 3 link types to the full schema:
- **11 entity types** (company, country, vessel, wallet, instrument, person, cftc_contract, organization, protocol, topic, domain)
- **45 observation types** (all OBSERVATION_TYPES mapped to their entity types)
- **19 link types** (headquartered_in, operates_in, market_authorized_in, lobbies_for, debtor_of, awarded_by, works_for, port_call_to, exchange_based_in, transacts_with, trades_instrument, tracks_issuer, located_in, fx_base_country, fx_quote_country, exchange_country, tracks_protocol, cftc_tracks, sanctioned_under)

New entity type count parameters default to 0 for backward compatibility. Existing 27 tests pass unchanged (link count assertion updated from 8→18 to reflect new intra-type links).

6 cross-domain injected patterns added for testing:
1. person.insider_trade → company.sell_intent via works_for
2. country.sanctions_listing → company.creditor_filing via headquartered_in
3. vessel.port_call → country.trade_flow via port_call_to
4. wallet.btc_transfer → instrument.price_movement via trades_instrument
5. cftc_contract.futures_positioning → instrument.instrument_volatility via cftc_tracks
6. country.pathogen_level → country.economic_activity via sanctioned_under

## Training Results (10 epochs, 54 entities, 30-day synthetic)

| Metric | Value |
|--------|-------|
| Total loss | 144k → 92k (36% decrease) |
| obs_type CE | 39.4 → 9.1 |
| obs_type top-1 accuracy | 32.9% (vs 2.2% random = 15x) |
| obs_type top-5 accuracy | 80.1% |
| time_delta MAE | 866 seconds |
| Entities | 54 |
| Links | 77 |
| Observations | 166,547 |
| Pattern instances | 26,324 |

## Attention Analysis — Key Findings

### High-Attention Edges (learning signal)
| Edge | Attention |
|------|-----------|
| wallet→transacts_with→wallet | 1.000 |
| instrument→tracks_protocol→protocol | 1.000 |
| cftc_contract→cftc_tracks→instrument | 0.938 |
| company→lobbies_for→company | 0.833 |
| company→debtor_of→company | 0.750 |

### Starved Edges (zero or near-zero attention)
| Edge | Attention |
|------|-----------|
| company→market_authorized_in→country | 0.000 |
| country→sanctioned_under→country | 0.000 |
| instrument→exchange_country→country | 0.000 |
| instrument→located_in→country | 0.000 |
| wallet→exchange_based_in→country | 0.000 |
| company→operates_in→country | 0.063 |

### Disconnected Entity Types (0 degree)
- **domain** — 0 mean degree (no inbound links, only cert_issued/dns_change obs)
- **topic** — 0 mean degree (no inbound links, only pageview/market_probability/research/price obs)

### Implications for Phase 36+
1. **domain** and **topic** nodes have zero graph connectivity — they contribute observations but the GNN cannot propagate signal to/from them. Need link types: topic→instrument, topic→company, domain→company.
2. Several country-facing link types carry zero attention — may need denser evidence or weight rebalancing.
3. Intra-type edges (wallet↔wallet, company↔company) show very strong signal capture — the GNN effectively learns entity-to-entity dynamics.
4. Cross-domain causal chains (person→company, cftc→instrument, wallet→instrument) fire correctly and produce injected pattern instances.

## Files Modified
- `agent/models/gnn/trainer.py` — SyntheticGraphGenerator expanded
- `tests/test_trainer.py` — link count assertion updated (8→18)
- `tests/test_phase35_gnn_retrain.py` — **CREATED** (33 tests)
- `[[quant_training_ground]]` — Phase 35 marked complete

## Test Results
- **27/27** existing trainer tests pass
- **33/33** Phase 35 tests pass
- **60 total** tests for GNN training infrastructure

## Related

- [[quant_training_ground]]
- [[phase35_gnn_retrain_expanded_graph]]
- [[phase35_gnn_retrain_expanded_graph_spec]]
- [[chat_checkpoint_2026-04-17_phase34_complete]]
- [[l2_expansion_roadmap]]
