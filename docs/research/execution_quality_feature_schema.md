---
title: "Feature: Execution-Quality Feature Schema"
tags:
  - doc/research
---

# Feature: Execution-Quality Feature Schema

## Purpose
- Define a machine-readable feature contract for execution quality in thin, event-driven, or fragmented markets.
- This is the non-commodity lesson worth extracting from Bloomberg TRA, adapted to TirraMind's target use cases.
- The main objective is to estimate whether predicted edge survives actual execution.

## Scope
- Primary target: prediction markets and thin books
- Secondary target: low-liquidity event-driven instruments and fragmented venues
- Non-goal: generic institutional TCA reports for conventional equity execution workflows

## Design Goals
- Venue-agnostic core fields
- Simple enough for daily collection and backtest replay
- Explicit handling of stressed conditions and poor liquidity
- Compatible with both pre-trade and post-trade analysis

## Top-Level Schema

```json
{
  "schema_version": "1.0",
  "timestamp": "2026-03-31T14:05:00Z",
  "venue": "prediction_market_x",
  "instrument_id": "market-123-yes",
  "side": "buy",
  "order_context": {},
  "book_state": {},
  "cost_estimates": {},
  "fill_estimates": {},
  "stress_flags": {},
  "lineage": []
}
```

## Required Sections

### 1. Order context

```json
{
  "decision_price": 0.54,
  "target_size": 1200.0,
  "order_type": "limit",
  "time_horizon_sec": 300,
  "urgency": "medium",
  "signal_strength": 0.73
}
```

Fields:
- `decision_price`: reference price at model decision time
- `target_size`: intended quantity
- `order_type`: `market`, `limit`, `passive_limit`, `slice`
- `time_horizon_sec`: time allowed for completion
- `urgency`: `low`, `medium`, `high`
- `signal_strength`: optional normalized conviction score

### 2. Book state

```json
{
  "best_bid": 0.53,
  "best_ask": 0.55,
  "mid_price": 0.54,
  "spread_abs": 0.02,
  "spread_bps_mid": 370.37,
  "top_bid_size": 400.0,
  "top_ask_size": 250.0,
  "depth_1pct": 900.0,
  "depth_5pct": 2400.0,
  "book_imbalance": -0.18,
  "trade_rate_1m": 17,
  "cancel_rate_1m": 9
}
```

Core derived quantities:
- `spread_abs = best_ask - best_bid`
- `mid_price = (best_bid + best_ask) / 2`
- `book_imbalance = (bid_size - ask_size) / (bid_size + ask_size)`

### 3. Cost estimates

```json
{
  "slippage_expected_abs": 0.008,
  "slippage_expected_bps": 148.15,
  "impact_proxy_abs": 0.011,
  "impact_proxy_bps": 203.70,
  "implementation_shortfall_expected": 0.014,
  "fees_abs": 3.2,
  "all_in_cost_abs": 0.017
}
```

Suggested meanings:
- `slippage_expected_abs`: expected movement between decision price and average fill price from urgency plus liquidity conditions
- `impact_proxy_abs`: expected price concession required to complete target size
- `implementation_shortfall_expected`: expected loss relative to decision price after price impact and fees
- `all_in_cost_abs`: normalized total expected cost for the intended trade

### 4. Fill estimates

```json
{
  "fill_probability_horizon": 0.61,
  "expected_fill_size": 780.0,
  "expected_fill_ratio": 0.65,
  "expected_time_to_fill_sec": 410,
  "queue_ahead_estimate": 530.0,
  "partial_fill_risk": 0.72
}
```

Interpretation:
- `fill_probability_horizon`: probability target order gets meaningfully filled within time horizon
- `queue_ahead_estimate`: estimated resting size ahead of the order at the relevant level
- `partial_fill_risk`: risk that incomplete fills leave the strategy exposed or under-hedged

### 5. Stress flags

```json
{
  "thin_book": true,
  "wide_spread": true,
  "adverse_imbalance": true,
  "event_spike": false,
  "stale_quotes": false,
  "execution_regime": "adverse"
}
```

Suggested flag logic:
- `thin_book`: depth insufficient for target size
- `wide_spread`: spread above historical threshold for venue / instrument
- `adverse_imbalance`: imbalance points against intended side
- `event_spike`: quote/trade behavior indicates event-driven instability
- `stale_quotes`: quote age exceeds acceptable threshold

## Derived Feature Recommendations
- `size_to_top_ask = target_size / top_ask_size` for buy orders
- `size_to_depth_1pct = target_size / depth_1pct`
- `depth_sufficiency_score = min(depth_1pct / target_size, 1.0)`
- `spread_zscore` relative to instrument history
- `impact_to_edge_ratio = all_in_cost_abs / expected_signal_edge_abs`

This last feature is strategically important:
- if `impact_to_edge_ratio >= 1`, the trade is probably not worth taking

## Pre-Trade vs Post-Trade Usage

### Pre-trade
- estimate cost before submitting the order
- decide whether to trade now, wait, slice, or skip

### Post-trade
- compare realized fill to estimate
- calibrate model error
- identify recurring venue- or event-specific friction

## Validation Rules
- all probabilities must be in `[0, 1]`
- `best_ask >= best_bid`
- all sizes must be non-negative
- `expected_fill_size <= target_size`
- `expected_fill_ratio = expected_fill_size / target_size` when target size > 0
- `execution_regime` must be from controlled enum: `benign`, `normal`, `fragile`, `adverse`

## Why This Is Non-Commodity
- Generic TCA dashboards are commoditized.
- Explicit execution-friction modeling for thin and event-driven markets is not.
- In markets with unstable depth, execution quality is part of the edge, not an afterthought.
- This matters more for TirraMind than polished views of common macro data.

## Recommended Future Implementation
- map venue-specific raw order book data into this common schema
- store feature rows alongside decisions and realized fills
- backtest expected vs realized execution cost
- make trade selection conditional on cost-to-edge ratio, not signal alone

## Related

- [[project_memory]]
