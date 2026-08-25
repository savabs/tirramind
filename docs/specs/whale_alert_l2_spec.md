---
title: "Spec: whale_alert L2 Upgrade"
tags:
  - doc/spec
  - phase/10
  - topic/surveillance
  - layer/surveillance
---

# Spec: whale_alert L2 — Wallet Entity Registration

## Goal

Upgrade `WhaleAlertTool` to register BTC wallet entities and store per-wallet transfer observations at depth_level=2 in the PipelineStore entity registry.

## Files Affected

| File | Action |
|------|--------|
| `agent/tools/whale_alert.py` | **Modify** — add TYPE_CHECKING import, entity imports, PipelineStore constructor kwarg, `_persist_entities()`, `_persist_entities_inner()`, `entity_ids` in tx dicts |
| `tests/test_whale_alert_l2.py` | **Create** — full L2 edge case + MI integration test suite |

## Implementation Steps

### Step 10b.3.1: Add TYPE_CHECKING + entity imports

Add at the top of `whale_alert.py`:
```python
from typing import TYPE_CHECKING
# ... existing imports ...

if TYPE_CHECKING:
    from agent.pipeline.store import PipelineStore

try:
    from agent.pipeline.entity import entity_id_from_key
except ImportError:
    entity_id_from_key = None
```

### Step 10b.3.2: Accept optional PipelineStore in constructor

Change constructor signature to:
```python
def __init__(
    self,
    cache: DataCache | None = None,
    *,
    pipeline_store: PipelineStore | None = None,
) -> None:
    self._cache = cache
    self._store = pipeline_store
```

### Step 10b.3.3: Implement `_persist_entities()` + `_persist_entities_inner()`

After `_filter_txs()`, before `_compute_summary()`, call `self._persist_entities(txs)` in `execute()`.

Guard method pattern (identical to insider_filings):
```python
def _persist_entities(self, txs: list[dict[str, Any]]) -> None:
    if self._store is None or entity_id_from_key is None:
        return
    if not txs:
        return
    try:
        self._persist_entities_inner(txs)
    except Exception:
        log.exception("Entity persistence failed (non-fatal)")
```

Inner method:
- For each tx in `txs`:
  - For each input address: register wallet entity (dedup by addr), store observation (direction="out")
  - For each output address: register wallet entity (dedup by addr), store observation (direction="in")
- Dedup: use `seen_wallets: set[str]` to avoid re-registering the same address across txs
- `register_entity(entity_type="wallet", canonical_name=addr, entity_id=entity_id_from_key("wallet", addr))`
- `add_entity_alias(entity_id, "btc_address", addr)`
- `store_entity_observation(entity_id, "whale_alert", observed_at=tx["time"], observation_type="btc_transfer", depth_level=2, value={...})`

Observation value:
```python
{
    "tx_hash": tx["hash"],
    "value_btc": addr_entry["value_btc"],
    "direction": "in" | "out",
    "counterparty_count": len(other_side),  # how many addrs on the other side
    "confirmed": tx["confirmed"],
    "block_height": tx.get("block_height"),
}
```

### Step 10b.3.4: Add `entity_ids` mapping to transaction dicts

In `_parse_blockchain_txs()`, after building each tx entry, add an `entity_ids` dict mapping address → entity_id for all input + output addresses (when `entity_id_from_key` is available).

```python
if entity_id_from_key is not None:
    eid_map = {}
    for inp in inputs:
        eid_map[inp["addr"]] = entity_id_from_key("wallet", inp["addr"])
    for out in outputs:
        eid_map[out["addr"]] = entity_id_from_key("wallet", out["addr"])
    entry["entity_ids"] = eid_map
else:
    entry["entity_ids"] = {}
```

### Step 10b.3.5: Edge case test suite

Create `tests/test_whale_alert_l2.py` covering:

**Constructor tests:**
- Default (no store) — `_store` is None
- With PipelineStore — `_store` is set

**Persistence tests:**
- `_persist_entities_inner()` registers wallet entities + aliases for all input/output addrs
- Observations stored with correct fields (direction, tx_hash, value_btc, confirmed)
- Dedup: same address in multiple txs → registered once, observation per tx
- Empty tx list → no persistence calls
- No store → `_persist_entities()` is no-op
- `entity_id_from_key` unavailable → no-op
- Persistence error → caught, logged, tool still returns results

**entity_ids tests:**
- entity_ids dict present in parsed tx output
- entity_ids maps all input + output addresses
- entity_ids empty when `entity_id_from_key` is None

**Integration tests:**
- Full execute() with PipelineStore → entities + observations persisted
- Full execute() without PipelineStore → backward-compatible results

### Step 10b.3.6: MI measurement integration test

Same pattern as insider_filings/form144:
- Create PipelineStore in `:memory:`
- Simulate L1 observations (aggregate: total BTC moved) and L2 observations (per-wallet transfers)
- Compute MI gain from L2 vs L1 against a synthetic target
- Assert MI(L2) > MI(L1) (more granular = more information)

## Edge Cases

1. **No addresses in tx** — inputs or outputs with missing `addr` field → skip that address, don't crash
2. **Zero-value outputs** — `value_btc == 0` addresses are already filtered by parsing; persistence should still handle them gracefully
3. **Extremely long address** — BTC addresses are 25-34 chars (base58) or 42-62 chars (bech32); no truncation needed for entity_id (SHA-256 hash handles any length)
4. **Same address as both input and output** — possible in self-transfers; register once, store two observations (direction=in and direction=out)
5. **Persistence failure mid-batch** — exception in `_persist_entities_inner()` caught by outer guard; tool returns normal results
6. **Large batch (3000+ txs)** — only filtered txs (above min_btc) are persisted; shouldn't be more than ~50-100 whale txs per block

## Testing Plan

| Category | Count | Description |
|----------|-------|-------------|
| Constructor | 2 | Default + with store |
| Persistence | 7 | Inner logic, dedup, empty, no store, no entity_id, error, both sides |
| entity_ids | 3 | Present, correct mapping, unavailable fallback |
| Integration | 2 | With store, without store |
| MI measurement | 1 | L2 > L1 mutual information |
| **Total** | ~15+ | (will likely be ~30-40 after edge case expansion) |

## Related

- [[whale_alert_l2]]
- [[deep_surveillance_tools]]
- [[deep_surveillance_10b]]
- [[deep_surveillance_10b2]]
- [[project_memory]]
