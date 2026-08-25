---
title: "Checkpoint: Phase 30 Complete — Crypto Islands Linked"
tags:
  - doc/checkpoint
  - phase/30
  - topic/entity-linking
  - topic/crypto
---

# Checkpoint: Phase 30 Complete — Crypto Islands Linked

**Date:** 2025-07-25
**Previous checkpoint:** [[chat_checkpoint_2026-04-16_extensive_phase29_handoff]]

## What Was Done

Phase 30 connected BTC-USD and ETH-USD (previously graph islands) to the entity graph via two new edge types:

1. **`tracks_protocol`** — instrument → protocol (BTC-USD→bitcoin, ETH-USD→ethereum)
   - Added `protocol: str | None` field to `InstrumentDef` dataclass
   - Extended `_persist_instrument_links` to register protocol entities and create links
   - Protocol entity IDs match those created by `defi_flows` tool (both use `entity_id_from_key("protocol", name.lower())`)

2. **`trades_instrument`** — wallet → instrument (whale wallets → BTC-USD)
   - Added `_BTC_INSTRUMENT_EID` module-level constant in `whale_alert.py`
   - Extended `_persist_entities_inner` to link every sender/receiver wallet to BTC-USD

## Files Modified

- `agent/tools/instrument_universe.py` — `protocol` field, crypto entries, `_persist_instrument_links` extension
- `agent/tools/whale_alert.py` — `_BTC_INSTRUMENT_EID`, `trades_instrument` links in `_persist_entities_inner`
- `tests/test_phase30_crypto_links.py` — 28 edge case tests (mock-based)
- `tests/test_phase30_diagnostic.py` — 13 integration tests (real PipelineStore :memory:)

## Test Results

- Phase 30 edge case tests: **28 passed**
- Phase 30 diagnostics: **13 passed**
- Targeted regression (instrument_universe + whale_alert + graph_builder): **261 passed, 0 failed**

## Graph Impact

Before Phase 30:
- BTC-USD and ETH-USD were isolated nodes (no edges)
- Whale wallets connected only to each other via `transacts_with`

After Phase 30:
- BTC-USD → bitcoin protocol (via `tracks_protocol`)
- ETH-USD → ethereum protocol (via `tracks_protocol`)
- Whale wallets → BTC-USD (via `trades_instrument`)
- Multi-hop path now exists: wallet → BTC-USD → bitcoin protocol → (any defi_flows observations)

## Obsidian Lint

10 findings remain — all advisory (3 LK02 orphans + 7 ST01 long files). Zero LK01 broken links, zero FM01/FM02 frontmatter issues.

## Next Work

Per [[quant_training_ground]] and [[l2_expansion_roadmap]], the GNN-guided evaluation should next determine:
- Which entity neighborhoods are still sparse
- Whether the new crypto edges improve pattern recovery
- What the next priority tool/link type should be

## Related

- [[crypto_islands_cross_domain_linking]]
- [[crypto_islands_cross_domain_linking_spec]]
- [[phase30_crypto_islands]]
- [[starved_class_audit]]
- [[l2_expansion_roadmap]]
- [[quant_training_ground]]
