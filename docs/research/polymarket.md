---
title: "Research: Polymarket Events (Phase 5)"
tags:
  - doc/research
  - topic/polymarket
---

# Research: Polymarket Events (Phase 5)

**Date:** 2026-03-28 (retroactive documentation)
**Tool:** `agent/tools/polymarket.py` → `PolymarketTool`
**Status:** IMPLEMENTED, TESTED

## APIs Used

### Gamma API ✅ SELECTED
- **URL:** `https://gamma-api.polymarket.com/events`
- **Method:** GET
- **Auth:** None (public REST API, no key)
- **Format:** JSON
- **Rate limits:** None detected
- **Coverage:** **Global** — prediction markets on worldwide topics

## Geographic Coverage
- Polymarket hosts markets on: elections (global), crypto, geopolitics, finance, tech, science, sports, climate
- No geographic restriction — events span all countries
- **Verdict:** `[G:GLOBAL]`

## Implementation Details
- Single mode with filters
- Parameters: `category` (politics/crypto/finance/geopolitics/tech/science/sports/all), `limit` (1-100), `search` (keyword)
- Category mapping via tag slugs → 7 normalized categories
- Cache key: `polymarket_events`
- Timeout: 15s

## Signal Value
- Prediction market prices = aggregate probability estimates
- Political event odds = policy risk pricing
- Crypto event markets = sentiment for digital assets
- Geopolitical markets = risk assessment
- Works best combined with polymarket_whales.py (wallet-level intelligence)

## Relationship to polymarket_whales.py
- This tool fetches market-level data (events, odds, liquidity)
- polymarket_whales.py fetches wallet-level data (trades, positions, scoring)
- Combined: market-level signal + who is betting = conviction-weighted probability

## Risks
- Gamma API could change without notice
- Market resolution can be disputed
- Thin markets may have unreliable odds

## Related

- [[project_memory]]
