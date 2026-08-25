---
title: "Research: Wikipedia Pageviews (7b-O)"
tags:
  - doc/research
  - layer/surveillance
  - phase/7b
  - topic/wiki
---

# Research: Wikipedia Pageviews (7b-O)

**Date:** 2026-03-28 (retroactive documentation)
**Tool:** `agent/tools/wikipedia_pageviews.py` → `WikipediaPageviewsTool`
**Status:** IMPLEMENTED, TESTED

## APIs Probed

### Wikimedia REST API ✅ SELECTED
- **URL:** `https://wikimedia.org/api/rest_v1`
- **Method:** GET
- **Auth:** None (User-Agent required: `TirraMind/0.1`)
- **Format:** JSON
- **Rate limits:** ~200 req/s (undocumented). Politeness delay enforced: 50ms between requests.
- **Coverage:** **Global** — 300+ language editions (en, ja, de, zh, fr, es, ar, hi, ru, etc.)
- **Features:**
  - Per-article daily/monthly pageviews
  - Top 1000 articles per day
  - Top articles by country
  - Historical data back to 2015
  - All access types (desktop, mobile-web, mobile-app)

## Geographic Coverage
- Wikipedia exists in 300+ languages — inherently global
- Default project: `en.wikipedia`, supports any language edition
- **Verdict:** `[G:GLOBAL]`

## Modes Implemented
1. `spike` — z-score anomaly detection across 41-article watchlist. 30-day trailing baseline. Flags articles with z > 2.0.
2. `top` — top trending articles for a given date, with evergreen page filtering (excluding Main_Page, Special:Search, etc.)
3. `series` — raw daily pageview timeseries for any article, with inline spike flagging

## Default Watchlist (41 articles)
- US mega-cap: Apple, Microsoft, Alphabet, Amazon, Meta, Nvidia, Tesla, Berkshire
- Global mega-cap: TSMC, Samsung, Alibaba, Tencent, ASML, Toyota, LVMH, Novo Nordisk
- Finance/crypto: Bitcoin, Ethereum, JPMorgan, Goldman Sachs, BlackRock, Binance
- Defense/geopolitical: Lockheed Martin, Raytheon, BAE Systems, NATO, BRICS
- Energy/commodities: Saudi Aramco, ExxonMobil, Chevron, Lithium, Uranium, Rare-earth element
- Pharma: Pfizer, Moderna, Eli Lilly
- Systemic risk: SVB, Credit Suisse, Deutsche Bank

## Signal Value
- "Somebody knows something" — research precedes action
- Japanese Wikipedia spikes on chemical compounds = Japanese research interest
- Arabic Wikipedia spikes on political figures = regional instability forming
- Cross-language simultaneous spikes = global awareness shift
- Company page surges before earnings/M&A = informed attention
- Near-zero-variance handling prevents false z-score spikes

## Risks
- Wikimedia API occasionally returns partial data for recent dates
- Bot traffic can inflate pageview counts (filtered by `user` access type)
- Evergreen filtering needs maintenance as new high-traffic pages emerge

---

## Related

- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]
