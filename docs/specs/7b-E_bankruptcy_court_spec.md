---
title: "Spec: 7b-E — Bankruptcy, Court Filings & Regulatory Actions"
tags:
  - doc/spec
  - layer/surveillance
  - phase/7b
  - topic/bankruptcy
---

# Spec: 7b-E — Bankruptcy, Court Filings & Regulatory Actions

## Goal

Build `BankruptcyCourtTool` — a multi-mode tool that aggregates real-time bankruptcy filings,
SEC enforcement actions, and UK insolvency notices from free public feeds.

## Files Affected

- **CREATE:** `agent/tools/bankruptcy_court.py`
- **CREATE:** `tests/test_bankruptcy_court_edge.py`
- **MODIFY:** `agent/cli.py` (add import + register)
- **MODIFY:** `[[quant_training_ground]]` (mark steps done)

## Implementation Steps

### 2.1: Create tool skeleton with parameters and mode routing

Class `BankruptcyCourtTool(Tool)` with name `"bankruptcy_court"`.

**4 modes:**
- `us_bankruptcy` — PACER RSS (6 courts)
- `sec_enforcement` — SEC Admin Proceedings RSS + Litigation Releases RSS
- `sec_bankruptcy` — SEC EFTS 8-K Item 1.03
- `uk_insolvency` — UK Gazette Atom feed + GOV.UK SFO

**Parameters:**
```
mode: str (enum) — required
court: str — PACER court filter (sdny, del, sdtx, cdca, ndil, nj, all). Default "all"
keyword: str — text filter for sec_enforcement, uk_insolvency. Default ""
days_back: int — lookback window for sec_bankruptcy, uk_insolvency. Default 7, max 90
limit: int — max results. Default 25, max 100
```

### 2.2: Implement `us_bankruptcy` mode (PACER RSS)

6 courts, fetched in parallel with `httpx.AsyncClient` (but execute is sync, so use `asyncio.run`).
Actually — use sync httpx with ThreadPoolExecutor for parallel fetches (simpler, matches other tools).

**Courts dict:**
```python
PACER_COURTS = {
    "sdny": ("S.D. New York", "ecf.nysb.uscourts.gov"),
    "del":  ("Delaware", "ecf.deb.uscourts.gov"),
    "sdtx": ("S.D. Texas", "ecf.txsb.uscourts.gov"),
    "cdca": ("C.D. California", "ecf.cacb.uscourts.gov"),
    "ndil": ("N.D. Illinois", "ecf.ilnb.uscourts.gov"),
    "nj":   ("D. New Jersey", "ecf.njb.uscourts.gov"),
}
```

URL pattern: `https://{domain}/cgi-bin/rss_outside.pl`

Parse XML: `<item><title>`, `<link>`, `<description>`, `<pubDate>`
- Title: `{case_number} {debtor_name}` — parse with regex `^(\S+)\s+(.+)$`
- Description: parse chapter type (7, 11, 13, 15) from text

Output: list of `{case_number, debtor_name, chapter, court, link, pub_date, docket_type}`
Filter by `court` param (or "all"). Limit.

**User-Agent:** Browser-like (SEC and some PACER courts block custom UAs).

**Cache:** 10min TTL keyed on `("pacer_rss", court_code)`.

### 2.3: Implement `sec_enforcement` mode

Two sources combined:
1. SEC Admin Proceedings RSS: `https://www.sec.gov/rss/litigation/admin.xml`
2. SEC Litigation Releases RSS: `https://www.sec.gov/rss/litigation/litreleases.xml`

Parse RSS XML. Extract `<item>` elements: title, link (to PDF), description, pubDate.
Merge both feeds, sort by pubDate descending. Optional keyword filter.

Output: list of `{title, type (admin|litigation), link, pub_date, description}`
Cache: 30min TTL keyed on `("sec_enforcement",)`.

**Critical:** Must use browser-like User-Agent (`Mozilla/5.0 ...`). SEC returns 403 for custom UAs.

### 2.4: Implement `sec_bankruptcy` mode

SEC EFTS search-index: `https://efts.sec.gov/LATEST/search-index`
Query params: `q="1.03"&forms=8-K&dateRange=custom&startdt=<start>&enddt=<end>&from=0&size=<limit>&_source=form,file_date,display_names,items,ciks`

8-K Item 1.03 = Entry into Bankruptcy/Receivership.

Parse JSON. Extract company names, CIKs, filing dates, items.
Filter: `days_back` parameter controls date range.

Output: list of `{company_name, cik, file_date, items, form}`
Cache: 1hr TTL keyed on `("sec_efts_bankruptcy", days_back)`.

### 2.5: Implement `uk_insolvency` mode

Two sources:
1. UK Gazette Atom feed: `https://www.thegazette.co.uk/insolvency/data.feed`
   Parse Atom entries. Paginated (200+ pages). Only fetch page 1 (most recent).
2. GOV.UK SFO enforcement: `https://www.gov.uk/api/search.json?filter_organisations=serious-fraud-office&count=<limit>&order=-public_timestamp`

Parse Atom XML for Gazette, JSON for GOV.UK.
Combine results, sort by date. Optional keyword filter.

Output: list of `{title, source (gazette|sfo), link, pub_date, description}`
Cache: 1hr TTL keyed on `("uk_insolvency",)`.

### 2.6: Result formatting and summary statistics

Each mode returns ToolResult with:
- `output`: Human-readable summary (header, count, first N entries formatted as text)
- `data`: Machine-readable dict with `{mode, count, total, entries: [...]}` for the world model

Add a summary line showing:
- Total filings count
- Breakdown by type (chapter 11 vs 7 for PACER, admin vs lit for SEC)
- Date range of results

### 2.7: Register in cli.py

```python
from agent.tools.bankruptcy_court import BankruptcyCourtTool
registry.register(BankruptcyCourtTool(cache=cache))
```

### 2.8: Edge case test suite

See Testing Plan below.

## Edge Cases

1. PACER court returns empty RSS / no items
2. PACER court is unreachable / timeout
3. XML parse errors (malformed RSS)
4. SEC EFTS returns 0 hits for date range
5. SEC EFTS returns 403 (rate limit) — handle gracefully
6. UK Gazette returns paginated Atom — ensure only page 1 parsed
7. GOV.UK API returns empty results
8. Invalid mode string
9. Invalid court code
10. `days_back=0` should be clamped to 1
11. `limit=0` should be clamped to 1
12. `limit=999` should be clamped to 100
13. Concurrent PACER fetches — one fails, others succeed (partial results OK)
14. PACER title parsing — unusual case number formats
15. Empty/missing descriptions in RSS items
16. Unicode in debtor names
17. Keyword filter is case-insensitive
18. Very large PACER feed (900KB) — ensure proper handling

## Testing Plan

Tests in `tests/test_bankruptcy_court_edge.py`. All network calls mocked.

**Test classes (target ~80 tests):**
1. `TestModeRouting` — valid/invalid mode dispatch
2. `TestParameterValidation` — clamping, defaults, bad inputs
3. `TestPACERParsing` — XML parsing of PACER RSS items, title regex, chapter detection
4. `TestPACERCourtFiltering` — single court, all courts, invalid court
5. `TestPACERErrorHandling` — timeout, malformed XML, partial failure
6. `TestSECEnforcementParsing` — Admin + Litigation RSS merge, sort, keyword filter
7. `TestSECEnforcementErrors` — 403, timeout, malformed XML
8. `TestSECBankruptcyParsing` — EFTS JSON parse, date range, company extraction
9. `TestSECBankruptcyErrors` — 403, empty results, malformed JSON
10. `TestUKInsolvencyParsing` — Gazette Atom + GOV.UK JSON merge
11. `TestUKInsolvencyErrors` — timeout, empty, malformed
12. `TestCacheIntegration` — cache hit/miss across modes
13. `TestToolSchema` — JSON schema validation
14. `TestRegistryIntegration` — registration in ToolRegistry
15. `TestResultFormat` — output and data structure for each mode

---

## Related

- [[7b-E_bankruptcy_court|Research: 7B-E Bankruptcy Court]]
- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]
