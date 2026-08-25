---
title: "Spec: Crypto Whale Transfer Monitoring Tool"
tags:
  - doc/spec
  - topic/whale-tracking
---

# Spec: Crypto Whale Transfer Monitoring Tool

## Goal
Monitor large cryptocurrency transfers in real-time. Two sources: blockchain.com (free, no key, BTC) and Whale Alert (free tier, multi-chain, exchange labels). Detect exchange inflows (sell pressure) and outflows (accumulation).

## Files Affected
- CREATE: `agent/tools/whale_alert.py`
- MODIFY: `agent/cli.py` (register WhaleAlertTool)
- MODIFY: `agent/learning/bandit.py` (add `crypto_whale_flows` arm)
- MODIFY: `agent/config/settings.py` (add whale_alert_key config)

## Implementation Steps

### Step 6c.1: Add whale_alert_key to AgentConfig
- New field: `whale_alert_key: str = ""` 
- Env var: `TIRRA_WHALE_ALERT_KEY`

### Step 6c.2: Create `agent/tools/whale_alert.py` skeleton
- Class `WhaleAlertTool(Tool)` with name="whale_alert"
- Parameters:
  - `mode`: "blockchain" | "whale_alert" (default: auto — uses whale_alert if key available, else blockchain)
  - `min_btc`: float (minimum BTC value for blockchain mode, default 10.0)
  - `min_usd`: int (minimum USD value for whale_alert mode, default 1_000_000)
  - `limit`: int (max transactions to return, default 20)
- Constructor: `__init__(self, api_key: str = "", cache: DataCache | None = None)`

### Step 6c.3: Implement `_fetch_blockchain()` 
- GET `https://blockchain.info/unconfirmed-transactions?format=json`
- Parse `txs[]` → compute total output value per tx in BTC
- Filter by min_btc threshold
- Sort by value descending
- Return list of dicts: {hash, time, value_btc, inputs: [{addr, value}], outputs: [{addr, value}]}

### Step 6c.4: Implement `_fetch_whale_alert()`
- GET `https://api.whale-alert.io/v1/transactions` with api_key, min_value, start (1 hour ago)
- Parse response.transactions[]
- Extract: blockchain, symbol, hash, timestamp, amount, amount_usd, from.owner, from.owner_type, to.owner, to.owner_type
- Classify each tx: "exchange_inflow" (to.owner_type == "exchange"), "exchange_outflow" (from.owner_type == "exchange"), "exchange_to_exchange", "unknown"
- Return list of dicts with classification

### Step 6c.5: Implement `_compute_summary(txs, mode)`
- Total volume (BTC or USD)
- Count of transactions
- For whale_alert mode: breakdown by flow direction (inflow/outflow/exchange-to-exchange/unknown)
- Largest single transfer
- Average transfer size

### Step 6c.6: Implement `execute()`
- Auto-detect mode: if whale_alert key available and mode not forced → use whale_alert
- Fetch → filter → compute summary → format output
- ToolResult.data: {transactions: [...], summary: {...}}

### Step 6c.7: Register + bandit arm
- Add `WhaleAlertTool(api_key=..., cache=cache)` to cli.py
- Add `crypto_whale_flows` GoalArm to bandit.py

### Step 6c.8: Edge case tests
- Blockchain.com: live fetch, filter by min_btc, empty result with high threshold
- Whale Alert: skip if no API key (graceful), test with mock response
- Malformed data, missing fields, empty txs list
- Summary computation with zero transactions
- Mode auto-detection logic

## Edge Cases
- No API key → blockchain mode only, whale_alert mode returns helpful error
- Blockchain.com returns 0 large txs during quiet periods → success with empty list
- Whale Alert rate limit (429) → graceful error message
- Transaction with no outputs → skip
- Very large values (overflow) → safe float parsing

---

## Related

- [[whale_alert|Research: Whale Alert]]
