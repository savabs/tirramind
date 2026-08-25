---
title: "Spec: Form 144 Tool — Insider Sell Intent Detection"
tags:
  - doc/spec
  - topic/form144
---

# Spec: Form 144 Tool — Insider Sell Intent Detection

## Goal

Detect insider sell intent clusters from SEC Form 144 filings. Form 144 is filed
BEFORE the sell order (T+0), giving 2+ days lead time over Form 4 (T+2). Filter
RSU/PSU noise to isolate voluntary, high-conviction sell signals.

## Files Affected

| File | Action |
|------|--------|
| `agent/tools/form144.py` | CREATE — Form144Tool class |
| `agent/cli.py` | MODIFY — register tool, extend `insider_flow` bandit arm |
| `agent/learning/bandit.py` | MODIFY — add form144 to `insider_flow` arm tools list |
| `tests/test_form144_edge.py` | CREATE — edge case test suite |

## Implementation Steps

### Step 1: Skeleton + EFTS fetch

Create `agent/tools/form144.py` with:
- `Form144Tool(Tool)` class, `name = "form144"`
- Parameters: `days_back` (int, 1-60, default 14), `ticker` (str, optional), `min_cluster_size` (int, default 2)
- `_fetch_recent_144s(start_dt, end_dt)` — EFTS search with `forms=144`, pagination, rate limiting, cache
- Same EFTS infrastructure as InsiderFilingsTool but searching form 144

### Step 2: XML parser

`_parse_form144_xml(xml_text, fallback_ticker, fallback_company, fallback_name, fallback_date)`:
- Extract from `issuerInfo`: issuerName, nameOfPerson...ToBeSold, relationshipToIssuer
- Extract from `securitiesInformation`: noOfUnitsSold, aggregateMarketValue, approxSaleDate, noOfUnitsOutstanding, securitiesExchangeName, brokerName
- Extract from `securitiesToBeSold[]`: acquiredDate, natureOfAcquisitionTransaction, amountOfSecuritiesAcquired, isGiftTransaction
- Handle namespace dynamically (ns2: / com: variations)
- Parse MM/DD/YYYY dates (Form 144 format) to YYYY-MM-DD

Returns dict:
```python
{
    "ticker": "ROST",
    "company": "ROSS STORES, INC.",
    "insider_name": "STEPHEN BRINKLEY",
    "relationship": "Officer",
    "shares_to_sell": 4154,
    "dollar_value": 884428.56,
    "shares_outstanding": 323444928,
    "approx_sale_date": "2026-03-24",
    "filing_date": "2026-03-24",
    "exchange": "NASDAQ",
    "broker": "Morgan Stanley Smith Barney LLC",
    "acquisition_type": "vesting",      # classified
    "is_gift": False,
    "acquisition_details": [...]        # raw list from securitiesToBeSold
}
```

### Step 3: Acquisition type classifier

`_classify_acquisition(nature_text)` → `"open_market" | "private_placement" | "vesting" | "gift" | "other"`

Mapping:
- Contains "open market" → "open_market" (signal weight 3.0)
- Contains "private" or "placement" → "private_placement" (2.0)
- Contains "stock unit" or "RSU" or "PSU" or "restricted" or "performance" or "option" or "incentive" → "vesting" (0.5)
- isGiftTransaction == "Y" → "gift" (0.0)
- Otherwise → "other" (1.0)

### Step 4: Parse filings pipeline

`_parse_filings(raw_hits)`:
- For each EFTS hit: extract CIK, accession, display_name → ticker
- Fetch XML, parse, yield filing dict
- Skip gifts (signal weight 0)
- Add urgency classification based on filing_date vs approx_sale_date

### Step 5: Sell intent cluster detection

`_detect_sell_clusters(filings, min_size)`:
- Group by ticker
- Sliding 14-day window on filing_date
- Count distinct insiders (by normalized name)
- Score: `distinct_count × total_dollar_value × max(acquisition_weights)`
- Return clusters ranked by score

Cluster output:
```python
{
    "ticker": "ROST",
    "company": "ROSS STORES, INC.",
    "insider_count": 3,
    "total_value": 2500000.00,
    "pct_of_outstanding": 0.08,
    "cluster_start": "2026-03-18",
    "cluster_end": "2026-03-24",
    "urgency": "immediate",
    "conviction": "high",
    "has_voluntary_sells": True,
    "filings": [...]
}
```

### Step 6: execute() method

Full pipeline:
1. Validate inputs (clamp days_back, min_cluster_size)
2. Fetch EFTS hits
3. Parse XML filings
4. Filter by ticker if specified
5. Detect clusters
6. Format human-readable output + structured data
7. Return ToolResult

### Step 7: Register in cli.py

- Import and register `Form144Tool(cache=cache)`
- Add `"form144"` to `insider_flow` arm's tools list in DEFAULT_ARMS

### Step 8: Edge case tests

36+ tests covering:
- Input validation (days_back clamping, min_cluster_size, ticker normalization)
- EFTS: normal, empty, pagination, HTTP errors, 429 retry, malformed response
- XML parser: normal, missing elements, namespace variations, empty XML, parse error
- Acquisition classifier: all types + edge cases
- Date parsing: MM/DD/YYYY, invalid dates, missing dates
- Cluster detection: normal, no clusters, single filing, entity filers, dedup
- Gift filtering, urgency classification
- Cache integration, bandit arm check
- 2 live network tests (EFTS search + XML fetch)

## Edge Cases

1. No ticker in EFTS display_name (rare) → use issuerName as fallback key → skip filing
2. XML parse error → skip filing, log warning
3. Multiple `securitiesToBeSold` entries in one filing → use highest-signal acquisition type
4. Entity name vs person name for dedup → normalize: strip "L.P.", "Inc.", "LLC" etc.
5. `approxSaleDate` missing or unparseable → urgency = "unknown"
6. `noOfUnitsSold` = 0 or missing → skip filing
7. Form 144/A amendments → treat as separate filings (dedup by accession)
8. `aggregateMarketValue` = 0 but shares > 0 → compute from shares × approx price

## Testing Plan

1. Unit tests with mocked EFTS + XML responses
2. Live integration test (2 filings minimum)
3. Cluster detection with synthetic data (known clusters)
4. All acquisition type classifications
5. All urgency classifications
6. Boundary: exactly min_cluster_size insiders

---

## Related

- [[form144|Research: Form144]]
