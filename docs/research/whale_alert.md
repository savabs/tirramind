---
title: "Feature: Crypto Whale Transfer Monitoring"
tags:
  - doc/research
  - topic/whale-tracking
---

# Feature: Crypto Whale Transfer Monitoring

## Current Architecture
- Tool ABC pattern (base.py), registration in cli.py, bandit arm in bandit.py
- DataCache for caching, jsonschema validation on args

## Data Sources (verified live)

### Primary: Blockchain.com Unconfirmed Transactions
- **URL**: `https://blockchain.info/unconfirmed-transactions?format=json`
- **Cost**: Free, no API key
- **Coverage**: BTC only, returns ~100 most recent unconfirmed txs
- **Fields**: hash, time, inputs (prev_out.addr, prev_out.value), out (addr, value)
- **Rate limit**: ~1 req/10s (undocumented, gentle polling needed)
- **Signal**: large pending BTC txs, volume of pending transfers
- Live test: found 79.67 BTC and 41.40 BTC whale txs in single fetch

### Secondary: Whale Alert API (opt-in, requires free key)
- **URL**: `https://api.whale-alert.io/v1/transactions`
- **Params**: api_key, min_value (USD), start (unix timestamp), cursor
- **Cost**: Free tier — 10 req/min, 100 req/day, last ~1 hour window
- **Coverage**: Multi-chain (BTC, ETH, XRP, USDT, EOS, TRX, etc.)
- **Killer feature**: Exchange address labels (from/to owner: "Binance", "Coinbase", "unknown")
- **Fields**: blockchain, symbol, id, transaction_type, hash, from.address, from.owner, from.owner_type, to.address, to.owner, to.owner_type, timestamp, amount, amount_usd
- **Signal**: Exchange inflow (sell pressure) vs outflow (accumulation), cross-chain flows

### Also Tested (not used)
- **Mempool.space**: Free, no key, BTC mempool. But requires per-tx API calls to get values — too slow.
- **Etherscan**: Requires API key, ETH only, no exchange labels
- **Blockchair**: Returns 430 on filtered queries (paid feature)

## Quant Signal Value
1. **Exchange inflow surge** (Whale Alert): Large transfers TO exchanges = imminent selling pressure
2. **Exchange outflow surge** (Whale Alert): Large transfers FROM exchanges = accumulation/cold storage
3. **Large BTC mempool spikes** (Blockchain.com): Whale activity volume indicator
4. **Transfer size distribution**: Mean transfer size increasing = institutional activity
5. **Velocity**: Number of large txs per hour trending up/down

## Risks
- Blockchain.com returns only ~100 txs, may miss large ones in busy periods
- Whale Alert free tier: 100 req/day limits to ~4 polls/hour for 24 hours
- Whale Alert exchange labels can be stale (address re-labeling lag)
- BTC-only from blockchain.com misses ETH/stablecoin whale flows (often more signal-rich)
- Rate limits need careful handling — exponential backoff

## Implementation Approach
- **Two sources**: `blockchain` mode (default, no key) + `whale_alert` mode (opt-in)
- `blockchain` mode: fetch unconfirmed txs, filter by min_btc threshold, compute summary stats
- `whale_alert` mode: fetch recent transactions above min_value, group by direction (exchange in/out/unknown)
- Config: `TIRRA_WHALE_ALERT_KEY` env var for Whale Alert API key
- Both modes return: list of whale txs, summary stats, exchange flow direction (whale_alert only)

---

## Related

- [[whale_alert_spec|Spec: Whale Alert]]
