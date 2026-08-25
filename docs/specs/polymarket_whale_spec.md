---
title: "Spec: polymarket_whale"
tags:
  - doc/spec
  - topic/polymarket
  - topic/whale-tracking
---

# Spec: polymarket_whale

## Goal

Build a whale-tracking system for Polymarket that:
1. Indexes all trades with wallet addresses from the public data-api
2. Scores wallets by accuracy on resolved markets, profit, and size
3. Detects smart-money consensus and whale alerts as pipeline signals
4. Exposes a `polymarket_whales` agent tool for querying whale activity

Uses Pipeline Layer (Phase 7) for scheduled polling and SQLite persistence. $0 cost — data-api.polymarket.com is fully public, no auth.

## Files Affected

| File | Action |
|------|--------|
| `agent/tools/polymarket_whales.py` | CREATE — PolymarketWhalesTool (agent-facing query tool) |
| `agent/pipeline/operators.py` | MODIFY — no change needed (FunctionOperator handles custom functions) |
| `agent/pipeline/dags/whale_tracking.py` | CREATE — whale_tracking DAG (trade indexer + scorer + signal detector) |
| `agent/pipeline/dags/__init__.py` | MODIFY — register whale_tracking DAG |
| `agent/cli.py` | MODIFY — import + register PolymarketWhalesTool |
| `tests/test_polymarket_whales_edge.py` | CREATE — edge case tests |

## Data Model

### New tables in Pipeline DB (via PipelineStore)

We do NOT modify PipelineStore schema. Instead, we use the existing `pipeline_data` and `signals` tables:

- **`pipeline_data`** with `source="pm_trades"` — deduplicated trade records (params_json has tx_hash as key, data_json has the trade)
- **`pipeline_data`** with `source="pm_wallet_scores"` — wallet score snapshots
- **`pipeline_data`** with `source="pm_resolutions"` — resolved market outcomes
- **`signals`** with `signal_name` in `{"pm_whale_alert", "pm_consensus", "pm_contrarian", "pm_accumulation"}` — detected signals

Trade deduplication: before inserting, query `pipeline_data` for `source="pm_trades"` + matching tx_hash in params_json. This is O(1) via the source+fetched_at index.

## Implementation Steps

### Step 1: Trade Fetcher Function

Create `agent/pipeline/dags/whale_tracking.py` with:

**`fetch_recent_trades(context: dict) -> dict`**
- Calls `GET data-api.polymarket.com/trades?limit=1000`
- Returns `{"trades": [...], "count": N}` where each trade has: `tx_hash, wallet, condition_id, side, size, price, timestamp, outcome, outcome_index, title, slug, usdc_value`
- `usdc_value = size * price` (approximate USDC spent)
- Filter out micro-markets: skip trades where `title` contains "Up or Down" (15-min BTC/ETH noise markets) — configurable via `TIRRA_PM_SKIP_MICRO` env var
- Error handling: HTTP errors → return `{"trades": [], "error": str(e)}`
- **Test:** mock httpx, verify parsing of all fields, verify micro-market filtering

### Step 2: Trade Indexer Operator

**`index_trades(context: dict) -> dict`**
- Takes `context["fetch_recent_trades"]` (upstream results from Step 1)
- Opens PipelineStore, inserts each trade into `pipeline_data`:
  - `source="pm_trades"`
  - `params_json=json.dumps({"tx_hash": tx_hash})`
  - `data_json=json.dumps(trade_dict)`
  - `fetched_at=trade["timestamp"]`
- Deduplication: before insert, check if tx_hash already exists via `store.query_data("pm_trades")` scan OR keep a set of recently-seen hashes in memory (cheaper)
- Returns `{"indexed": N, "duplicates": M, "total_seen": N+M}`
- **Test:** mock store, verify dedup, verify correct source/params

### Step 3: Wallet Scorer Function

**`score_wallets(context: dict) -> dict`**
- Query all `pm_trades` from PipelineStore
- Group trades by wallet
- For each wallet with ≥5 resolved trades:
  - **Accuracy (Bayesian):** `(correct + 1) / (total + 2)` (Beta(1,1) prior — Laplace smoothing)
  - **Profit factor:** `sum(winning_pnl) / max(abs(sum(losing_pnl)), 0.01)` (avoid div/0)
  - **Volume:** `sum(usdc_value)` across all trades
  - **Markets participated:** count distinct `condition_id`
  - **Recency weight:** `exp(-ln(2)/30 * days_since_last_trade)` (half-life 30 days)
  - **Composite score:** `accuracy * log(1 + volume) * recency * sqrt(markets)`
- To determine "correct": match trade's `condition_id` to resolved markets in `pm_resolutions`.
  - Resolved market has winning `outcomeIndex`. Trade is correct if `side="BUY"` and `outcome_index == winning_index`, or `side="SELL"` and `outcome_index != winning_index`.
- Store top-500 wallet scores in `pipeline_data` with `source="pm_wallet_scores"`
- Returns `{"scored": N, "top_10": [{"wallet": ..., "score": ..., "accuracy": ..., "volume": ...}]}`
- **Test:** synthetic trades + resolutions, verify Bayesian accuracy, verify composite formula, edge cases (0 trades, all wins, all losses, single trade)

### Step 4: Resolution Tracker Function

**`track_resolutions(context: dict) -> dict`**
- Fetch `GET gamma-api.polymarket.com/events?closed=true&limit=100`
- For each resolved event's markets:
  - Parse `outcomePrices` — the outcome with price nearest 1.0 is the winner
  - Extract `conditionId` and `winning_outcome_index`
- Store in `pipeline_data` with `source="pm_resolutions"`, `params_json={"condition_id": cid}`, `data_json={"winning_index": idx, "title": ...}`
- Deduplicate by condition_id (skip if already stored)
- Returns `{"resolved": N, "new": M}`
- **Test:** mock Gamma API, verify outcome parsing (prices near 0/1), verify dedup

### Step 5: Signal Detector Function

**`detect_signals(context: dict) -> dict`**
- Reads recent trades (from Step 2 output or from DB) + wallet scores (from DB)
- **Smart Money Consensus:** Group recent trades (last 24h) by condition_id+side. If 3+ wallets from top-100 scored wallets on same side → emit signal.
  - `signal_name="pm_consensus"`, `value=weighted_confidence`, `metadata_json={"market", "side", "wallets", "avg_price"}`
- **Whale Alert:** Any trade by top-50 scored wallet with usdc_value > $1000 (configurable via metadata).
  - `signal_name="pm_whale_alert"`, `value=usdc_value`, `metadata_json={"wallet", "market", "side", "score", "accuracy"}`
- **Contrarian Smart Money:** Top-50 wallet buys YES on market with current price < 0.3, or buys NO on market with price > 0.7.
  - Needs current price from trade's `price` field (approximation) or from Gamma API.
  - `signal_name="pm_contrarian"`, `value=wallet_score`, `metadata_json={"wallet", "market", "side", "market_price"}`
- Store signals via `store.record_signal(name, value, metadata)`
- Returns `{"signals_emitted": N, "consensus": [...], "whale_alerts": [...], "contrarian": [...]}`
- **Test:** synthetic scored wallets + trades, verify each signal type triggers correctly, verify thresholds

### Step 6: Whale Tracking DAG

Build `whale_tracking` DAG in `agent/pipeline/dags/whale_tracking.py`:

```
DAG: whale_tracking
Schedule: */15 * * * * (every 15 min)

fetch_recent_trades  (FunctionOperator → fetch_recent_trades)
        ↓
index_trades         (FunctionOperator → index_trades, depends on fetch_recent_trades)
        ↓
detect_signals       (FunctionOperator → detect_signals, depends on index_trades)
```

Separate DAG for daily scoring:
```
DAG: whale_scoring
Schedule: 0 6 * * * (daily at 06:00 UTC)

track_resolutions    (FunctionOperator → track_resolutions)
        ↓
score_wallets        (FunctionOperator → score_wallets, depends on track_resolutions)
```

Register both in `agent/pipeline/dags/__init__.py` via `build_whale_tracking_dag()` and `build_whale_scoring_dag()`.

- **Test:** verify DAG structure (no cycles, correct deps), verify schedule strings

### Step 7: Agent Tool — PolymarketWhalesTool

Create `agent/tools/polymarket_whales.py`:
- `name = "polymarket_whales"`
- **Parameters:**
  - `mode`: enum `"top_wallets"` | `"wallet_detail"` | `"market_whales"` | `"recent_signals"` (required)
  - `wallet`: string, optional (for wallet_detail mode)
  - `market`: string, optional (conditionId or search term for market_whales mode)
  - `signal_type`: string, optional (filter signals: "consensus", "whale_alert", "contrarian", "all")
  - `limit`: int, default 10
- **Modes:**
  - `top_wallets` — return top-N scored wallets with accuracy, volume, recent activity
  - `wallet_detail` — return specific wallet's positions, P&L, trade history, score
  - `market_whales` — return whale activity on a specific market (top wallets, their sides, sizes)
  - `recent_signals` — return recent signals from pipeline DB (consensus, alerts, contrarian)
- Reads from PipelineStore (`pipeline_data` for scores/trades, `signals` for signals)
- Falls back to live API call if DB is empty (cold start): `GET data-api.polymarket.com/trades?limit=1000` → aggregate on the fly
- **Test:** mock PipelineStore responses for each mode, verify formatting, verify cold-start fallback

### Step 8: CLI Registration

- `agent/cli.py`: import `PolymarketWhalesTool`, register with `cache`
- Add to existing `prediction_market` bandit arm (alongside existing `polymarket` tool)
- **Test:** verify tool count, verify bandit arm update

### Step 9: Live Validation

- Run `fetch_recent_trades` against live data-api → verify 1000 trades parsed
- Run `track_resolutions` against live Gamma → verify resolved markets found
- Run full `whale_tracking` DAG manually → verify trades indexed, signals checked
- Query `polymarket_whales` tool in `top_wallets` mode → verify output
- **Test**: 4 live tests behind `TIRRA_LIVE_TESTS=1` flag

### Step 10: Edge Case Test Suite

Comprehensive tests covering:
- **Input validation:** invalid mode, missing required params, bad wallet format, bad limit
- **Trade fetcher:** normal 1000 trades, empty response, HTTP error, timeout, malformed JSON, micro-market filtering, missing fields in trade
- **Deduplication:** duplicate tx_hash, same trade different timestamp, empty DB
- **Wallet scorer:** 0 trades, 1 trade (below threshold), all wins, all losses, no resolved markets, Bayesian accuracy with small N, composite score formula verification, recency decay at boundary (30 days exactly)
- **Resolution tracker:** no resolved events, already-tracked resolutions (dedup), ambiguous outcome prices (not near 0/1), multi-outcome markets
- **Signal detector:** no scored wallets (cold start), exactly threshold (3 wallets), below threshold (2), whale alert at exact $1000 boundary, contrarian with price at 0.3/0.7 boundary, no recent trades
- **Agent tool:** each mode with empty DB, each mode with populated DB, cold-start fallback, wallet not found, market not found
- **Pipeline DAG:** structure validation, schedule format, node dependencies, timeout values
- **Integration:** CLI registration, bandit arm, tool schema (openai format)

## Edge Cases

1. **Cold start (empty DB)** — tool falls back to live API aggregation, scorer returns empty
2. **Micro-market noise** — 15-min BTC/ETH markets filtered by title pattern
3. **Wallet with 0 resolved trades** — excluded from scoring (min threshold = 5)
4. **Division by zero** — profit factor denominator clamped to 0.01
5. **API returns less than 1000 trades** — not an error, just fewer trades in the period
6. **Duplicate tx_hash across polls** — deduplicated on insert
7. **Proxy wallet address format** — must be full 42-char hex (0x + 40 hex digits)
8. **Resolved market with ambiguous outcome** — prices not near 0 or 1 → skip (unresolved)
9. **data-api schema change** — defensive `.get()` for all fields, graceful degradation
10. **Very large wallets (>100 positions)** — positions endpoint pagination if needed

## Testing Plan

- Unit tests (mocked): ~80+ tests covering all functions, edge cases, signal detection
- Live tests: 4 tests behind `TIRRA_LIVE_TESTS=1` flag
- Integration tests: CLI registration, bandit arm, DAG structure, tool schema
- Total target: 90+ tests

---

## Related

- [[polymarket_whale|Research: Polymarket Whale]]
