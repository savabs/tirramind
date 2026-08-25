---
title: "Feature: Phase 5 — Observational Surface (Smart Money Signals)"
tags:
  - doc/research
---

# Feature: Phase 5 — Observational Surface (Smart Money Signals)

## Goal

Build two high-signal, zero-cost data tools: **Polymarket whale tracking** and **SEC insider filing cluster detection**. These are the cheapest edges available — informed money leaves traces in public data that nobody systematically harvests.

## Core Thesis

Markets are outputs. Reality (and insider knowledge) is the input. When informed actors — prediction market whales, corporate insiders — take positions, they leave observable traces *before* prices move. The data is free. The edge is in systematically collecting and scoring it.

---

## Data Source 1: Polymarket Smart Money Tracker

### What It Is

Polymarket is a prediction market on Polygon (CTF — Conditional Token Framework). ~$1B+ monthly volume. Markets cover politics, crypto, macro, geopolitics. All trades are on-chain (Polygon PoS), but most volume flows through the CLOB (Central Limit Order Book) API.

### Why It's Edge

- **Informed money leaks into bet sizing.** A wallet that's been 80%+ accurate over 50+ resolved markets is likely connected to real information sources.
- **Nobody else is systematically scoring wallets and tracking their new positions.** Traditional finance ignores prediction markets.
- **The signal leads mainstream markets.** E.g., Polymarket election odds move before polling averages; crypto event markets move before spot prices.

### Available APIs (All Free, No Auth Required for Read)

1. **Gamma API** (market metadata + prices):
   - `GET https://gamma-api.polymarket.com/events?closed=false&limit=100` — active events with nested markets
   - `GET https://gamma-api.polymarket.com/markets?id={market_id}` — single market detail
   - Fields: `outcomePrices`, `volume`, `liquidity`, `bestBid`, `bestAsk`, `clobTokenIds`, resolution status
   - Rate limit: ~60 req/min (undocumented but generous)

2. **CLOB API** (order book + trades):
   - `GET https://clob.polymarket.com/prices?token_id={clob_token_id}` — current price
   - `GET https://clob.polymarket.com/book?token_id={clob_token_id}` — full order book
   - `GET https://clob.polymarket.com/trades?market={condition_id}` — recent trades with maker/taker addresses
   - No auth for reads. Auth (API key + L1/L2 headers) only needed for placing orders.

3. **On-chain (Polygon RPC)**:
   - CTF contract: `0x4D97DCd97eC945f40cF65F87097ACe5EA0476045` (Conditional Token Framework)
   - Track `TransferSingle` / `TransferBatch` events for position changes
   - Free via public Polygon RPCs or Alchemy/Infura free tier (~300M compute units/month)

### Implementation Approach

**Phase 5a: Market Snapshot Tool** (lightweight, immediate value)
- Fetch active markets from Gamma API
- Return: market question, current prices, volume, liquidity, 24h/1wk change
- Filter by category (politics, crypto, finance, geopolitics)
- Cache with 15-min TTL (prediction markets move fast)

**Phase 5b: Whale Tracker** (the real edge — future phase)
- Index resolved markets + their CLOB trades
- Score each wallet by: accuracy (% correct), profit, avg position size, timing (early vs late)
- Track high-score wallets' current positions across all active markets
- Generate "smart money consensus" signals when 3+ top wallets align
- This requires more infra (DB, indexing) — defer to after 5a works

### Data Shape

```python
# Phase 5a output
{
    "markets": [
        {
            "question": "Fed rate cut in June 2026?",
            "slug": "fed-rate-cut-june-2026",
            "yes_price": 0.42,
            "no_price": 0.58,
            "volume_24h": 125000,
            "volume_total": 2500000,
            "liquidity": 50000,
            "spread": 0.02,
            "category": "finance",
            "end_date": "2026-07-01",
            "price_change_24h": +0.03,
            "price_change_1wk": -0.05,
        }
    ],
    "timestamp": "2026-03-24T12:00:00Z",
    "total_markets": 42,
}
```

---

## Data Source 2: SEC Insider Filing Clusters (Form 4)

### What It Is

SEC Form 4 filings report insider transactions (buys/sells by officers, directors, 10%+ holders). Filed within 2 business days of transaction. Public, structured, machine-readable.

### Why It's Edge

- **Individual insider buys are noise.** But when 3+ insiders at the same company buy within a 14-day window, it's one of the strongest documented equity signals.
- **Cluster detection is pure computation** — no expensive data feed needed. EDGAR is free.
- **Academic evidence:** Lakonishok & Lee (2001), Jeng et al. (2003) — insider purchase clusters predict 30-day abnormal returns of 3-8%.

### Available APIs (All Free, Rate-Limited)

1. **EDGAR Full-Text Search (EFTS)**:
   - `GET https://efts.sec.gov/LATEST/search-index?forms=4&dateRange=custom&startdt={start}&enddt={end}`
   - Returns filing metadata: CIKs, display_names, file_date, period_ending, accession numbers
   - Rate limit: 10 req/sec per IP (SEC fair access policy, User-Agent required)

2. **EDGAR Submissions API**:
   - `GET https://data.sec.gov/submissions/CIK{cik_padded}.json` — all filings for a company
   - Filter by `form: "4"` to get insider filings
   - Returns accession numbers, filing dates, primary document URLs

3. **Individual Filing XML**:
   - `GET https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{primary_doc}`
   - XML schema `X0508` — structured: `transactionType`, `sharesTraded`, `pricePerShare`, `directOrIndirect`, `ownershipNature`
   - Parse with standard XML — no special libraries needed

4. **SEC Company Tickers (mapping)**:
   - `GET https://www.sec.gov/files/company_tickers.json` — CIK → ticker mapping
   - Cache indefinitely (changes rarely)

### Implementation Approach

**Phase 5c: Insider Activity Tool**
- Given a date range (default: last 30 days), fetch all Form 4 filings
- Parse each filing XML to extract: company ticker, insider name, transaction type (P=purchase, S=sale), shares, price, date
- Group by company, detect clusters: 3+ *distinct* insiders buying within 14-day window
- Return: cluster list sorted by conviction (# insiders, total $ value, insider roles)
- SEC requires: `User-Agent: TirraMind/1.0 (contact@email.com)` header
- Cache filings permanently (they never change), cache search results with 6hr TTL

### Data Shape

```python
# Phase 5c output
{
    "clusters": [
        {
            "ticker": "AAPL",
            "company": "Apple Inc.",
            "cik": "0000320193",
            "insiders": [
                {"name": "Tim Cook", "role": "CEO", "shares": 50000, "price": 185.50, "date": "2026-03-15", "type": "P"},
                {"name": "Luca Maestri", "role": "CFO", "shares": 25000, "price": 184.20, "date": "2026-03-17", "type": "P"},
                {"name": "Jeff Williams", "role": "COO", "shares": 30000, "price": 186.00, "date": "2026-03-20", "type": "P"},
            ],
            "cluster_start": "2026-03-15",
            "cluster_end": "2026-03-20",
            "total_value": 19_527_500,
            "insider_count": 3,
            "conviction": "high",  # 3+ C-suite buyers
        }
    ],
    "scan_range": {"start": "2026-02-24", "end": "2026-03-24"},
    "total_filings_scanned": 8500,
    "clusters_found": 7,
}
```

---

## Current Architecture Fit

Both tools follow the existing pattern exactly:

1. Inherit from `Tool` base class
2. Accept `DataCache` in `__init__`
3. Use `httpx.Client` for HTTP (already a dep)
4. Cache responses via `cache.get()`/`cache.put()`
5. Register in `build_tool_registry()` in `cli.py`
6. Return `ToolResult(success, output, data)`

### New Dependencies

**None required.** All parsing (JSON for Polymarket, XML for EDGAR) is stdlib. HTTP via `httpx` (already installed).

### New Files

- `agent/tools/polymarket.py` — `PolymarketTool`
- `agent/tools/insider_filings.py` — `InsiderFilingsTool`

### Modified Files

- `agent/cli.py` — register both new tools in `build_tool_registry()`

---

## Risks

1. **SEC rate limiting:** 10 req/sec. Need respectful throttling + `User-Agent` header. Violation = IP ban.
2. **Polymarket API stability:** Gamma API is unofficial (no SLA). Could change endpoints. Mitigation: cache aggressively, fail gracefully.
3. **Form 4 XML parsing edge cases:** Amended filings (4/A), derivative transactions, indirect ownership. Start with direct purchases only, expand later.
4. **Polymarket CLOB trade data volume:** High-volume markets have millions of trades. Phase 5a avoids this by only fetching market-level data.
5. **False positive clusters:** Routine equity compensation vesting looks like "buying." Filter: only count open-market purchases (`transactionCode == "P"`), exclude grants/options/gifts.

---

## Phase Ordering

| Sub-phase | Tool | Complexity | Value |
|-----------|------|-----------|-------|
| 5a | PolymarketTool (market snapshot) | Low (~200 lines) | Immediate — agent can observe prediction market state |
| 5b | Polymarket whale tracking | High (needs indexing infra) | Deferred — requires resolved market history + wallet scoring |
| 5c | InsiderFilingsTool (cluster detection) | Medium (~300 lines) | High — strongest free equity signal known |

**Build order: 5a → 5c → 5b**

5a is fast to build and gives the agent a new observation surface immediately. 5c is the highest-value signal but requires more XML parsing work. 5b is deferred because it needs persistent storage beyond our current cache.

---

## Related

- [[observational_surface_spec|Spec: Observational Surface]]
