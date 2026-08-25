---
title: "Checkpoint: Phase 36 Complete — Connect Disconnected Entity Types"
tags:
  - doc/checkpoint
  - phase/36
  - topic/entity-linking
  - topic/gnn
  - layer/world-model
  - layer/surveillance
---

# Checkpoint — 2025-06-17

## Session Summary

**Phase 36 — Connect Disconnected Entity Types** is complete.

GNN diagnostics showed domain and topic entities had 0.0 mean degree because tools registered them but never called `link_entities()`. Phase 36 added two new link types:

- **`domain_owned_by`** (domain → company): cert_transparency and dns_monitor now look up the domain's base name in a company map derived from instrument_universe issuers.
- **`topic_relates_to_instrument`** (topic → instrument): polymarket maps topic categories (crypto, finance) to their natural instrument tickers.

## Files Modified

| File | Change |
|------|--------|
| `agent/tools/polymarket.py` | Added `_TOPIC_INSTRUMENT_MAP`, topic→instrument linking in `_persist_entities_inner()` |
| `agent/tools/instrument_universe.py` | Added `build_domain_company_map()` function |
| `agent/tools/cert_transparency.py` | Added `_link_domain_to_company()` method |
| `agent/tools/dns_monitor.py` | Added `_link_domain_to_company()` method |
| `agent/models/gnn/trainer.py` | Updated defaults (`num_topics=3, num_domains=3`), added domain/topic link generation |
| `tests/test_phase36_entity_linking.py` | NEW — 33 edge case tests |
| `tests/test_entity_linking.py` | Fixed `test_no_inputs_no_links` / `test_no_outputs_no_links` (Phase 30 whale→BTC trades_instrument side effect) |
| `tests/test_trainer.py` | Updated entity count (16) and link count (21) for new defaults |
| `tests/test_phase35_gnn_retrain.py` | Fixed `test_backward_compat_4_types` (explicit `num_topics=0, num_domains=0`) |

## Test Status

- 33/33 Phase 36 edge case tests pass
- 188/188 affected tests pass (Phase 36 + entity_linking + trainer + Phase 35)
- Pre-existing failures: `test_feature_generation_dag` (17 vs 6 produced) and ~38 others — all pre-existing from uncommitted Phase 30–35 changes

## Deferred Items

- Wallet → topic linking: requires structural changes to whale_alert persistence
- academic_preprints topic linking: arXiv categories too coarse for meaningful instrument mapping
- Country edges (starved): data-diversity issue (US-centric), not a missing-link bug

## Next Steps

- Move `[[phase36_connect_disconnected_entities]]` to `tasks/done/`
- Move stale `[[phase30_crypto_islands]]` to `tasks/done/`
- Commit all uncommitted phases (30–36) as a batch or individually
- Next phase: check [[quant_training_ground]] for the next queued phase

## Related

- [[phase36_connect_disconnected_entities]]
- [[phase36_connect_disconnected_entities_spec]]
- [[quant_training_ground]]
- [[phase35_gnn_retrain_expanded_graph]]
