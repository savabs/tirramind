---
title: "Research: whale_alert L2 Upgrade"
tags:
  - doc/research
  - phase/10
  - topic/surveillance
  - layer/surveillance
---

# Research: whale_alert L2 Upgrade

## Goal

Upgrade `WhaleAlertTool` from L1 (aggregate large-BTC-transfer lists) to L2 (entity-resolved wallet observations in the PipelineStore entity registry), following the proven pattern from insider_filings (10b.1) and form144 (10b.2).

---

## Current Architecture

### whale_alert.py (~281 lines)

- **Data sources:** blockchain.info free API (no key needed)
  - Mempool: `/unconfirmed-transactions` — leading indicator, ~200–600 txs per fetch
  - Confirmed: `/latestblock` + `/rawblock/{hash}` — latest confirmed block, 3000+ txs
- **Constructor:** `__init__(self, cache: DataCache | None = None)` — no PipelineStore
- **Parsing:** `_parse_blockchain_txs()` extracts per-tx:
  - `hash`, `time`, `value_btc`, `confirmed`, `block_height`
  - `inputs`: list of `{addr, value_btc}` — source wallets
  - `outputs`: list of `{addr, value_btc}` — destination wallets
- **Filtering:** `_filter_txs()` applies min_btc threshold, sorts by value descending
- **Summary:** `_compute_summary()` gives count, total_value, largest, avg_size
- **Output:** `_format_output()` gives text formatted line-per-tx with hash prefix + BTC amount

### Key Observations

1. **Addresses are already extracted.** Both input and output addresses are parsed into structured dicts. This is the natural entity key — no additional extraction work needed.
2. **No clustering.** Unlike insider_filings (which groups by company), whale_alert returns a flat sorted list of transactions. L2 adds wallet-level grouping.
3. **Entity type = "wallet".** Canonical key is the BTC address string. Deterministic: `entity_id_from_key("wallet", addr)`.
4. **Observation type = "btc_transfer".** Each output address receiving ≥ min_btc is one L2 observation.
5. **Both sender and receiver are entities.** Unlike SEC filings (reporter CIK is the main entity), BTC transactions have two sides. Both sides should be registered as wallet entities with observations.

### Differences from insider_filings/form144 L2 Pattern

| Aspect | insider_filings/form144 | whale_alert |
|--------|------------------------|-------------|
| Entity type | person (CIK) + company (CIK) | wallet (BTC address) |
| Alias source | `sec_cik`, `ticker` | `btc_address` |
| Observation type | `purchase` / `sell_intent` | `btc_transfer` |
| Dedup key | reporter_cik | BTC address |
| Two-sided? | No (insider is the actor) | Yes (sender + receiver) |
| Cluster detection | Yes (insider buying clusters) | No existing clustering |
| Observation value fields | shares, price, role | value_btc, direction, counterparty, tx_hash |

### What L2 Adds

1. **Wallet entity registration** — every address above threshold gets a canonical entity_id
2. **Transfer observations** — each large transfer is an observation on sender (direction=out) and receiver (direction=in) entities
3. **entity_ids mapping** — transaction output dicts include `entity_ids` mapping addr → entity_id for downstream traceability
4. **Backward compatible** — all changes are no-op when `pipeline_store=None`

---

## Risks

1. **Address volume** — a single block can have 3000+ txs × multiple addresses each. Must filter to min_btc threshold entities only, not register every dust output.
2. **No "company" equivalent** — BTC addresses don't have names. canonical_name will be the address itself (truncated for display). That's fine; the entity registry handles this.
3. **Address reuse** — BTC best practice is single-use addresses. Many whale wallets do reuse though (exchange hot wallets, custodial services). The entity registry naturally handles this — same address = same entity_id.
4. **Change outputs** — BTC transactions send change back to a sender-controlled address. These are indistinguishable from "real" outputs without heuristic analysis. For L2, we register all outputs above min_btc — heuristic change detection is an L3 concern.

---

## Implementation Design

Follow the exact same 7-step pattern as 10b.1/10b.2:

1. **PipelineStore kwarg** in constructor (keyword-only, optional)
2. **`_persist_entities()`** — guard method (skip if no store, catch errors)
3. **`_persist_entities_inner()`** — actual logic:
   - For each tx in the filtered whale list:
     - Register sender addresses as wallet entities (dedup by address)
     - Register receiver addresses as wallet entities (dedup by address)
     - Store observation on each address: type=`btc_transfer`, depth_level=2
4. **`entity_ids` mapping** in transaction output dicts
5. **Address-based dedup** in existing `_filter_txs` — already deduped by tx hash; wallet dedup is in persistence only

### Entity Observation Value Schema

```python
{
    "tx_hash": "abc123...",
    "value_btc": 50.0,
    "direction": "in" | "out",
    "counterparty": "1OtherAddr...",  # the other side
    "confirmed": True | False,
    "block_height": 900000,  # only if confirmed
}
```

---

## Step-Local References

- **L2 pattern template:** [[deep_surveillance_10b|insider_filings L2]] and [[deep_surveillance_10b2|form144 L2]]
- **Entity utilities:** `entity_id_from_key("wallet", addr)` in `agent/pipeline/entity.py`
- **PipelineStore API:** `register_entity()`, `add_entity_alias()`, `store_entity_observation()` in `agent/pipeline/store.py`
- **Depth evaluation:** `agent/pipeline/depth_eval.py` for MI measurement integration test

---

## Related

- [[deep_surveillance_tools]]
- [[whale_alert_l2_spec]]
- [[l2_tool_expansion]]
- [[deep_surveillance_10b]]
- [[deep_surveillance_10b2]]
- [[project_memory]]
