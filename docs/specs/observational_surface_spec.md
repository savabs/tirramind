---
title: "Spec: Phase 5 — Observational Surface (Smart Money Signals)"
tags:
  - doc/spec
---

# Spec: Phase 5 — Observational Surface (Smart Money Signals)

## Goal

Give TirraMind two new data tools that let the agent observe informed-money behavior in real time — prediction market prices/flows (Polymarket) and corporate insider buying clusters (SEC Form 4). Zero data cost. Both follow the existing Tool pattern exactly.

## Research

See: `[[observational_surface]]`

## Files Affected

### New files
- `agent/tools/polymarket.py` — PolymarketTool class
- `agent/tools/insider_filings.py` — InsiderFilingsTool class

### Modified files
- `agent/cli.py` — register both tools in `build_tool_registry()`

### No new dependencies

---

## Implementation Steps

### Sub-phase 5a: PolymarketTool (market snapshot)

**5a.1: Create `agent/tools/polymarket.py` skeleton**
- Class `PolymarketTool(Tool)` with name `"polymarket"`
- Accept `cache: DataCache | None = None` in `__init__`
- Parameters schema: `category` (optional, string: politics|crypto|finance|geopolitics|all), `limit` (optional, int, default 20), `active_only` (optional, bool, default True)
- Description for LLM: explains prediction market data, what's available
- Test: import succeeds, `to_openai_tool()` returns valid schema

**5a.2: Implement `_fetch_events()` helper**
- `GET https://gamma-api.polymarket.com/events` with params: `closed=false`, `limit=limit`, `offset=0`
- Use `httpx.Client(timeout=15)` (matches existing tools)
- Cache key: `{"source": "polymarket_events", "params": {"closed": False, "limit": N}}`
- Cache TTL: use default (6hr), but Polymarket data is more time-sensitive — document this tradeoff
- Return raw JSON list of events
- Test: mock httpx response, verify cache hit/miss behavior

**5a.3: Implement `_parse_markets()` helper**
- Extract from each event → each nested market:
  - `question`, `slug`, `yes_price` / `no_price` (from `outcomePrices` JSON string)
  - `volume_24h`, `volume_total`, `liquidity`
  - `spread` (bestAsk - bestBid if available, else None)
  - `end_date`, `price_change_24h`, `price_change_1wk`
  - `category` (from event tags — map tag slugs to our categories)
- Handle edge cases: missing prices (market not yet deployed), zero liquidity, closed markets sneaking through
- Test: parse a known Gamma API response fixture → verify all fields extracted correctly

**5a.4: Implement `execute()` method**
- Check cache → fetch if miss → parse → filter by category if specified → sort by volume_24h desc → limit
- Format human-readable output: table of top markets with question, yes/no price, volume, spread
- Return `ToolResult(success=True, output=formatted_str, data=structured_dict)`
- Handle: network errors → `ToolResult(success=False, output=error_msg)`
- Test: full execute() with mocked HTTP → verify output format and data structure

**5a.5: Register PolymarketTool in `cli.py`**
- Add `from agent.tools.polymarket import PolymarketTool` 
- Add `registry.register(PolymarketTool(cache=cache))` in `build_tool_registry()`
- Test: `build_tool_registry()` includes `"polymarket"` in `list_names()`

### Sub-phase 5c: InsiderFilingsTool (cluster detection)

**5c.1: Create `agent/tools/insider_filings.py` skeleton**
- Class `InsiderFilingsTool(Tool)` with name `"insider_filings"`
- Accept `cache: DataCache | None = None` in `__init__`
- Parameters schema: `ticker` (optional, filter to specific company), `days_back` (optional, int, default 30), `min_cluster_size` (optional, int, default 3)
- Test: import succeeds, `to_openai_tool()` returns valid schema

**5c.2: Implement `_fetch_recent_filings()` helper**
- `GET https://efts.sec.gov/LATEST/search-index?forms=4&dateRange=custom&startdt={start}&enddt={end}`
- Headers: `User-Agent: TirraMind/1.0 (research@tirramind.com)` (SEC requires this)
- Paginate: EFTS returns max 100 per page, use `from` param to paginate
- Rate limit: sleep 0.15s between requests (< 10 req/sec)
- Cache search results with 6hr TTL
- Return list of filing metadata dicts
- Test: mock response, verify pagination logic, verify User-Agent header

**5c.3: Implement `_fetch_filing_detail()` helper**
- Given accession number + CIK, fetch the Form 4 XML:
  `GET https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{primary_doc}`
- Parse XML with `xml.etree.ElementTree` (stdlib)
- Extract: `issuerTradingSymbol`, `rptOwnerName`, `officerTitle`, `isOfficer`/`isDirector`/`isTenPercentOwner`
- Extract each `nonDerivativeTransaction`: `transactionCode` (P=purchase, S=sale, A=award, etc), `transactionShares`, `transactionPricePerShare`, `transactionDate`
- Filter: only `transactionCode == "P"` (open-market purchases) for buy clusters
- Cache individual filings permanently (they never change)
- Test: parse a real Form 4 XML fixture → verify all fields extracted

**5c.4: Implement `_detect_clusters()` helper**
- Input: list of parsed filings
- Group by company (issuerTradingSymbol)
- Within each company: sort buys by date, use sliding 14-day window
- Cluster = 3+ *distinct* insider names buying within 14-day window
- Score conviction: count insiders, sum dollar value, check if C-suite (officer titles containing CEO/CFO/COO/President)
- Return clusters sorted by conviction score desc
- Test: synthetic filing data → verify cluster detection (3 insiders in 10 days = cluster, 2 insiders = no cluster, same insider 3 times = no cluster)

**5c.5: Implement `execute()` method**
- Compute date range from `days_back`
- Fetch recent filings → fetch details for each → detect clusters
- If `ticker` specified: filter to that company only
- Format output: table of clusters with ticker, insider count, total $, date range
- Handle: network errors, XML parse errors, no clusters found
- Test: full execute() with mocked HTTP → verify output

**5c.6: Register InsiderFilingsTool in `cli.py`**
- Add `from agent.tools.insider_filings import InsiderFilingsTool`
- Add `registry.register(InsiderFilingsTool(cache=cache))` in `build_tool_registry()`
- Test: `build_tool_registry()` includes `"insider_filings"` in `list_names()`

### Integration

**5.7: Add polymarket arm to bandit DEFAULT_ARMS**
- New arm in `agent/learning/bandit.py`: `GoalArm(name="prediction_markets", tools=["polymarket", "web_search", "web_browse"], description="Analyze prediction market signals...")`
- New arm: `GoalArm(name="insider_flow", tools=["insider_filings", "market_data", "web_search"], description="Detect insider buying clusters and cross-reference with price action...")`
- Test: DEFAULT_ARMS has 7 arms (was 5), new arms importable

**5.8: Update task file**
- Mark Phase 5a + 5c complete in `[[quant_training_ground]]`

---

## Edge Cases

1. **Polymarket returns empty events** — return empty list, not error
2. **Gamma API down** — ToolResult(success=False) with clear message
3. **SEC filing has no transactions** (e.g., just an amendment) — skip silently
4. **Form 4 XML uses different namespace version** — handle both X0407 and X0508 schemas
5. **Insider name variations** (e.g., "COOK TIMOTHY D" vs "Tim Cook") — normalize to uppercase, strip suffixes
6. **Company has >100 filings in 30 days** — pagination handles this (step 5c.2)
7. **Rate limit hit (SEC 429)** — retry with exponential backoff, max 3 retries

---

## Testing Plan

Each step has its own test(s). All tests use mocked HTTP (no live API calls in tests).

Key test scenarios:
- Polymarket: parse real Gamma API response, verify price extraction from JSON string
- Insider: parse real Form 4 XML, verify transaction extraction
- Cluster detection: synthetic data with known clusters → verify detection
- Cache behavior: verify cache hit skips HTTP, cache miss triggers fetch
- Error handling: HTTP 500, timeout, malformed XML → graceful degradation
- CLI registration: both tools appear in registry

---

## Related

- [[observational_surface|Research: Observational Surface]]
