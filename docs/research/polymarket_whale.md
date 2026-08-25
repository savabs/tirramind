---
title: "Feature: Polymarket Whale Tracker (7b-A)"
tags:
  - doc/research
  - topic/polymarket
  - topic/whale-tracking
---

# Feature: Polymarket Whale Tracker (7b-A)

## Goal

Track wallet-level trading behavior on Polymarket to identify "smart money" — wallets with consistently high accuracy and large size — and generate signals when these wallets take new positions.

## Current Architecture

- **Existing tool:** `agent/tools/polymarket.py` — uses Gamma API (`gamma-api.polymarket.com`) to fetch active events with prices, volume, and price changes. No wallet-level data.
- **Pipeline Layer:** Phase 7 complete — SQLite persistence, DAG scheduler, 356 tests. Whale tracker will use Pipeline for scheduled polling.
- **Cache:** `agent/data/cache.py` standard tool cache available.

## API Surface — Probed 2026-03-27

### data-api.polymarket.com (PRIMARY — fully public, no auth, no key)

This is the goldmine. Every endpoint below was live-tested and returns data.

#### `GET /trades`
- **Params:** `limit` (max 1000), `offset`, `market` (conditionId), `asset_id` (token_id), `user` (proxyWallet)
- **Response (per trade):**
  ```
  proxyWallet     — wallet address (0x...)
  side            — "BUY" or "SELL"
  asset           — token ID (long numeric string)
  conditionId     — market condition ID (0x... hex)
  size            — share quantity (float)
  price           — execution price (float, 0-1 = implied probability)
  timestamp       — unix epoch seconds
  title           — market question text
  slug            — market slug for URL
  icon            — image URL
  eventSlug       — parent event slug
  outcome         — "Yes"/"No"/outcome name
  outcomeIndex    — 0 or 1
  name            — user display name (can be empty)
  pseudonym       — auto-generated pseudonym
  bio             — user bio (usually empty)
  profileImage    — avatar URL
  transactionHash — Polygon tx hash (0x...)
  ```
- **Rate limit:** No 429 observed. 12 rapid-fire requests all returned 200. Avg latency ~215ms.
- **Pagination:** `offset` param works. Max `limit=1000` (returns exactly 1000, capped).
- **Filtering:** `?user=<wallet>` for wallet history. `?market=<conditionId>` for market trades. Both work.

#### `GET /positions`
- **Params:** `user` (required — proxyWallet address)
- **Response (per position):**
  ```
  proxyWallet, asset, conditionId, size, avgPrice, initialValue, currentValue,
  cashPnl, percentPnl, totalBought, realizedPnl, percentRealizedPnl, curPrice,
  redeemable, mergeable, title, slug, icon, eventId, eventSlug, outcome,
  outcomeIndex, oppositeOutcome, oppositeAsset, endDate, negativeRisk
  ```
- **Critical fields for scoring:** `cashPnl`, `realizedPnl`, `percentPnl`, `avgPrice`, `currentValue`, `size`
- **Max return:** Observed up to 100 positions per wallet. Likely paginated (need to test `offset`).
- **Note:** `user` param must be full hex address from trades. Truncated addresses return `{"error":"required query param 'user' not provided"}`.

#### `GET /activity`
- **Params:** `user` (required), `limit`
- **Response (per activity):**
  ```
  proxyWallet, timestamp, conditionId, type, size, usdcSize, transactionHash,
  price, asset, side, outcomeIndex, title, slug, icon, eventSlug, outcome,
  name, pseudonym, bio, profileImage, profileImageOptimized
  ```
- **Key difference from /trades:** Has `usdcSize` (actual USD value) and `type` field.

### gamma-api.polymarket.com (SUPPLEMENTARY — market metadata)

- `GET /events` — active & closed events with full market detail (82 fields per market). Already used by existing tool.
  - `?closed=true` returns resolved markets (prices near 0/1 = outcome determined).
  - `outcomes` field = ["Yes", "No"] or multi-outcome.
  - `outcomePrices` on resolved markets → prices near 0.0000001 or 0.9999999 = the resolution.
- `GET /markets` — individual market lookup by slug.
- No `/leaderboard`, `/profiles`, `/users` endpoints (all 404 or 401).

### clob.polymarket.com (AUTH REQUIRED for trades — limited public surface)

- **`/trades` → 401** ("Unauthorized/Invalid api key"). Requires CLOB API key (tied to Polymarket account).
- **`/book` → 404** for tested token (market may not have active orderbook).
- **`/sampling-markets` → 200** (public, 1000 markets, full CLOB market details including tokens, fees, rewards).
- **`/sampling-simplified-markets` → 200** (public, 1000 markets, minimal fields).
- **`/prices`, `/midpoints` → 401** (auth required).
- **Verdict:** CLOB is mostly auth-gated. The data-api provides everything we need without CLOB auth.

### Polygon RPC (ON-CHAIN — available but not needed)

- CTF contract: `0x4D97DCd97eC945f40cF65F87097ACe5EA0476045`
- Free RPCs tested: `polygon-rpc.com` (disabled/broken), `rpc.ankr.com/polygon` (needs free key), `polygon-bor-rpc.publicnode.com` (works but slow, timeouts on `eth_getLogs`).
- TransferSingle events available via `eth_getLogs` but block range limitations on free RPCs.
- **Verdict:** The data-api already provides `transactionHash` per trade, which is the on-chain proof. We don't need to parse raw blockchain events — the data-api is already indexing them for us. If we ever want to verify trades on-chain (anti-spoofing), we have the tx hashes.

## Observations

### What Works
1. **Full trade history per wallet** — `/trades?user=<wallet>` returns all trades with sizes, prices, timestamps, outcomes.
2. **Open positions with P&L** — `/positions?user=<wallet>` returns cashPnl, realizedPnl, percentPnl, avgPrice per position.
3. **Market-level trade aggregation** — `/trades?market=<conditionId>` returns all trades on a specific market → whale detection per market.
4. **Resolved market outcomes** — Gamma `/events?closed=true` returns resolved markets with outcome prices near 0/1 → can determine winning outcome.
5. **Transaction hashes** — every trade includes Polygon tx hash for on-chain verification.
6. **User identity** — `name`, `pseudonym`, `bio`, `profileImage` fields available (many wallets have public names).
7. **No rate limiting observed** — 12 rapid requests, all 200. Conservative polling should be safe.

### What's Missing / Limitations
1. **No leaderboard API** — must build our own wallet scoring from trade/position data.
2. **No bulk positions export** — must query per-wallet. For initial index, need to discover wallets from trades first.
3. **Max 1000 per request** — pagination via `offset` needed for heavy wallets.
4. **`/positions` caps at ~100** — may need pagination for wallets with many open positions.
5. **No historical trade lookback beyond what pagination gives** — no `before`/`after` timestamp filters observed to work (returned same data regardless). Offset is the only pagination.
6. **Proxy wallets, not EOAs** — `proxyWallet` is Polymarket's smart contract wallet, not the user's actual Polygon address. This is fine for tracking within Polymarket but can't correlate to external on-chain activity.

### What's Cheap / Expensive
- **$0 total** — all data from public data-api with no API key.
- **Compute cost:** Polling 1000 trades every 15 min = ~96 requests/day = trivial.
- **Storage cost:** Each trade ~300 bytes. 100K trades/day = ~30MB/day = trivial for SQLite.

## Data Requirements

### For Trade Indexer (pipeline job)
- Poll `GET /trades?limit=1000` every 15 min.
- Deduplicate by `transactionHash` (unique per trade).
- Store in Pipeline DB: `(tx_hash, wallet, market_condition_id, side, size, price, timestamp, outcome, outcome_index, title, slug)`.
- Track `last_seen_timestamp` to avoid re-processing.

### For Position Tracker (pipeline job, daily)
- For each tracked whale wallet, fetch `GET /positions?user=<wallet>`.
- Store snapshots: `(wallet, condition_id, size, avg_price, cash_pnl, realized_pnl, current_value, snapshot_date)`.
- Delta detection: compare today's snapshot with yesterday's → new positions = signals.

### For Resolution Tracker (pipeline job, daily)
- Fetch `GET /events?closed=true&limit=100` from Gamma.
- Match condition_ids to stored trades → compute per-wallet accuracy on resolved markets.
- Mark trades as `resolved_correct` or `resolved_incorrect`.

### For Wallet Scorer
- Inputs: trade history + resolved outcomes + position P&L.
- Outputs: per-wallet composite score = f(accuracy_pct, avg_size, profit_factor, recency_weight, markets_participated).
- Score decay: recent accuracy weighted more heavily (exponential decay, half-life ~30 days).

## Signal Design

### Signal 1: Smart Money Consensus
- **Trigger:** 3+ top-scored wallets take same side on same market within 24h.
- **Strength:** Weighted by wallet scores and position sizes.
- **Output:** `{market, side, consensus_count, weighted_confidence, avg_entry_price}`

### Signal 2: Whale Alert
- **Trigger:** Top-scored wallet opens position >$X (threshold configurable, default $5K).
- **Output:** `{wallet, market, side, size_usd, wallet_score, wallet_accuracy}`

### Signal 3: Contrarian Smart Money
- **Trigger:** Top-scored wallet takes opposite side from market consensus (price > 0.7 but whale buys NO, or price < 0.3 but whale buys YES).
- **High alpha:** Smart money disagreeing with the crowd on a market they've researched.

### Signal 4: Position Accumulation
- **Trigger:** Wallet adds to existing position across multiple trades (DCA pattern).
- **Output:** `{wallet, market, total_accumulated, num_trades, avg_price, time_span}`

## Risks

1. **Wallet gaming** — sophisticated actors could create multiple wallets to hide size. Mitigated by tracking wallet clusters (same trading patterns, timing correlation).
2. **Market manipulation signal** — large trades could be designed to move prices temporarily. Mitigated by requiring resolved accuracy history, not just position size.
3. **API stability** — data-api is undocumented/unofficial. Could change without notice. Mitigated by: defensive parsing, fallback to Polygon RPC for on-chain verification.
4. **Cold start** — need ~2 weeks of trade indexing before wallet scores are meaningful. Mitigated by: initial backfill via large offset pagination.
5. **Sports/micro markets noise** — many BTC 15-min up/down markets pollute trade stream. Need to filter by market category or minimum end date to focus on high-signal political/crypto/macro markets.

## Architecture Decision

### Approach: data-api Only (no CLOB auth, no Polygon RPC)

The `data-api.polymarket.com` provides everything needed:
- All trades with wallet addresses, sizes, prices, outcomes, tx hashes
- All positions with P&L per wallet
- All activity with USDC values

No need for:
- CLOB API key (trades available via data-api for free)
- Polygon RPC (tx hashes available in trade data, on-chain verification can be added later if needed)
- Polygonscan (same — tx hashes already in data)

### Implementation Layers

1. **Trade Indexer** — pipeline job, polls every 15 min, deduplicates, stores in DB
2. **Resolution Tracker** — pipeline job, runs daily, matches resolved markets to trades
3. **Wallet Scorer** — computes composite scores from trade history + resolutions
4. **Signal Detector** — runs after each indexer poll, checks for consensus/whale/contrarian signals
5. **Agent Tool** — `polymarket_whales` tool for agent queries (top wallets, alerts, market whale activity)

### Pipeline DAG
```
trade_indexer (every 15 min)
    ↓
wallet_scorer (after indexer, or daily)
    ↓
signal_detector (after scorer)
    ↓
[signals stored in pipeline_data / pipeline_signals tables]
```

Resolution tracker runs independently on daily schedule.

## Math/Algorithm Survey

### Wallet Scoring
- **Accuracy:** `correct_calls / total_resolved_calls` (Bayesian smoothed with Beta prior to handle small sample sizes)
- **Profit Factor:** `sum(winning_pnl) / abs(sum(losing_pnl))`
- **Size-Weighted Accuracy:** Weight each call by its USDC size
- **Recency:** Exponential decay, half-life 30 days: `weight = exp(-λ * days_ago)` where `λ = ln(2)/30`
- **Composite Score:** `score = accuracy_bayesian * log(1 + total_volume) * recency_weight * sqrt(markets_participated)`
  - Log-volume: diminishes returns from pure size, rewards consistent accuracy
  - Sqrt-markets: rewards diversification across market types

### Consensus Detection
- **Threshold:** Configurable N wallets (default 3) from top-K scored wallets (default 100) on same side within T hours (default 24)
- **Weighted confidence:** `sum(wallet_score_i * size_i) / sum(size_i)` for consensus wallets

### Anomaly Detection  
- **Unusual activity:** z-score on wallet's trade frequency vs. historical baseline
- **Unusual size:** z-score on trade size vs. wallet's historical distribution
- **Both above threshold:** → high-conviction signal

---

## Related

- [[polymarket_whale_spec|Spec: Polymarket Whale]]
