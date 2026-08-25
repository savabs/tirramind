---
title: "Spec: Crypto Islands + Cross-Domain Linking"
tags:
  - doc/spec
  - phase/30
  - topic/entity-linking
  - topic/crypto
  - layer/surveillance
  - layer/world-model
---

# Spec: Crypto Islands + Cross-Domain Linking

## Goal

Connect BTC-USD and ETH-USD instruments to the entity graph so the HetTGN can propagate attention from on-chain wallet activity and protocol TVL into crypto instrument predictions. After this phase, no crypto instrument has zero entity links.

## Files Affected

| File | Action | What |
|---|---|---|
| `agent/tools/instrument_universe.py` | MODIFY | Add `protocol` field to `InstrumentDef`; update 2 crypto entries; extend `_persist_instrument_links` |
| `agent/tools/whale_alert.py` | MODIFY | Add `trades_instrument` links from wallets to BTC-USD in `_persist_entities_inner` |
| `tests/test_phase30_crypto_links.py` | CREATE | Edge case tests for all new links |
| `tests/test_phase30_diagnostic.py` | CREATE | Integration diagnostics — BTC/ETH degree > 0 |

## Implementation Steps

### 30.1: Add `protocol` field to InstrumentDef

Add `protocol: str | None = None` after the FX fields in `InstrumentDef`. Update the two crypto entries:

```python
InstrumentDef("BTC-USD", "Bitcoin", "crypto", "Global", protocol="bitcoin"),
InstrumentDef("ETH-USD", "Ethereum", "crypto", "Global", protocol="ethereum"),
```

No other instruments get a `protocol` value.

**Test**: `InstrumentDef` construction with and without `protocol`; verify existing instruments unchanged.

### 30.2: Extend `_persist_instrument_links` for `tracks_protocol`

In `_persist_instrument_links`, after the country-linking block, add a protocol block:

```python
if inst.protocol:
    protocol_eid = entity_id_from_key("protocol", inst.protocol)
    store.register_entity(
        entity_type="protocol",
        canonical_name=inst.protocol,
        entity_id=protocol_eid,
    )
    link_id = store.link_entities(
        entity_id_a=inst_eid,
        entity_id_b=protocol_eid,
        link_type="tracks_protocol",
        source="instrument_universe",
        confidence=1.0,
        metadata={"ticker": inst.ticker},
    )
    if link_id:
        counts["tracks_protocol"] += 1
```

Add `"tracks_protocol": 0` to the `counts` dict.

**Test**: Run `_persist_instrument_links` with a mock store; verify 2 `tracks_protocol` links created for BTC-USD and ETH-USD; verify no `tracks_protocol` for non-crypto instruments.

### 30.3: Verify protocol naming consistency with defi_flows

`defi_flows` registers protocol entities with `entity_id_from_key("protocol", name.lower())`. Bitcoin chain → key `"bitcoin"`, Ethereum chain → key `"ethereum"`. Our `InstrumentDef.protocol` values are `"bitcoin"` and `"ethereum"` (already lowercase).

Write a test that computes `entity_id_from_key("protocol", "bitcoin")` and `entity_id_from_key("protocol", "ethereum")` and verifies they match what `_persist_instrument_links` would produce.

**Test**: Deterministic entity ID match between instrument_universe and defi_flows naming paths.

### 30.4: Add `trades_instrument` links in whale_alert

In `whale_alert._persist_entities_inner`, after the wallet observation block, add:

```python
BTC_INSTRUMENT_EID = _entity_id_from_key("instrument", "BTC-USD")

# Link wallet to BTC-USD instrument (idempotent)
store.link_entities(
    entity_id_a=wallet_eid,
    entity_id_b=BTC_INSTRUMENT_EID,
    link_type="trades_instrument",
    source="whale_alert",
    confidence=1.0,
    metadata={"tx_hash": tx_hash},
)
```

Compute `BTC_INSTRUMENT_EID` once at module level or as a class attribute to avoid recomputation per transaction.

**Important**: The instrument entity ID must use the same derivation as `instrument_universe._entity_id()`. Verify this is `entity_id_from_key("instrument", "BTC-USD")` — check that `_entity_id` in instrument_universe uses `hashlib.sha256(f"instrument:{ticker}")`.

**Test**: Mock store; run `_persist_entities_inner` with sample txs; verify `link_entities` called with correct BTC instrument EID and `trades_instrument` type.

### 30.5: Edge case tests

Create `tests/test_phase30_crypto_links.py` covering:

1. **InstrumentDef with protocol**: construction, repr, hash (frozen dataclass)
2. **InstrumentDef without protocol**: backward compat — all existing instruments still construct
3. **_persist_instrument_links tracks_protocol**: 2 links created for crypto, 0 for non-crypto
4. **Protocol entity ID consistency**: instrument_universe and defi_flows produce same ID for "bitcoin"/"ethereum"
5. **whale_alert trades_instrument**: links created per wallet; idempotent on duplicate wallet
6. **whale_alert empty txs**: no links created, no errors
7. **whale_alert missing addr**: skipped gracefully
8. **Instrument EID consistency**: `_entity_id("BTC-USD")` in instrument_universe equals `entity_id_from_key("instrument", "BTC-USD")` in whale_alert
9. **Link idempotency**: calling `_persist_instrument_links` twice doesn't duplicate links
10. **Self-link protection**: wallet → wallet link prevention (already handled by PipelineStore, but verify)

### 30.6: Diagnostics

Create `tests/test_phase30_diagnostic.py`:

1. Build a mock pipeline store, run `_persist_instrument_links`, run whale_alert `_persist_entities_inner` with sample data
2. Query entity_links for BTC-USD entity ID → assert degree > 0
3. Query entity_links for ETH-USD entity ID → assert degree > 0
4. Verify path exists: wallet → BTC-USD → protocol:bitcoin
5. Verify path exists: ETH-USD → protocol:ethereum
6. Count total links per crypto instrument (BTC should have tracks_protocol + N trades_instrument)

### 30.7: Regression + checkpoint

Run full test suite. Update:
- `[[quant_training_ground]]` — mark Phase 30 complete
- `[[phase30_crypto_islands]]` → move to `tasks/done/`
- Repo memory counts if changed
- Write checkpoint

## Edge Cases

- **Unknown crypto ticker in whale_alert**: Won't happen — whale_alert hardcodes BTC from blockchain.com
- **defi_flows not yet run**: Protocol entities may not exist when `_persist_instrument_links` runs. That's fine — `register_entity` is idempotent and creates the entity on first encounter
- **Protocol naming mismatch**: Guarded by test in 30.3
- **Large wallet volume**: INSERT OR IGNORE handles duplicates; no unbounded growth

## Testing Plan

| Test File | Coverage |
|---|---|
| `tests/test_phase30_crypto_links.py` | Steps 30.1–30.5 edge cases |
| `tests/test_phase30_diagnostic.py` | Step 30.6 integration diagnostics |
| Full regression | Step 30.7 |

## Related

- [[crypto_islands_cross_domain_linking]]
- [[phase30_crypto_islands]]
- [[starved_class_audit]]
- [[l2_expansion_roadmap]]
- [[phase25_cross_domain_entity_linking_spec]]
