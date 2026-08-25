---
title: "Spec: CFTC Commitments of Traders Tool"
tags:
  - doc/spec
  - topic/cftc
---

# Spec: CFTC Commitments of Traders Tool

## Goal
Fetch and parse CFTC Commitments of Traders data (disaggregated futures report). Provide latest positioning, weekly changes, and computed signals (COT index, net positioning, concentration). Zero cost — direct CFTC.gov download.

## Files Affected
- CREATE: `agent/tools/cftc.py`
- MODIFY: `agent/cli.py` (register CFTCTool)
- MODIFY: `agent/learning/bandit.py` (add `futures_positioning` arm)

## Implementation Steps

### Step 6b.1: Create `agent/tools/cftc.py` skeleton
- Class `CFTCTool(Tool)` with name="cftc", description, parameters schema
- Parameters:
  - `mode`: "latest" | "historical" (default: "latest")
  - `contract_filter`: str (substring match on contract name, e.g. "crude", "wheat", "gold")
  - `code_filter`: str (exact CFTC contract code, e.g. "006765")
  - `top_n`: int (return top N by open interest, default 20)
  - `year`: int (for historical mode, default current year)
- Verify: `to_openai_tool()` parses, schema validates

### Step 6b.2: Implement `_fetch_latest()` 
- GET `https://www.cftc.gov/dea/newcot/f_disagg.txt`
- httpx, User-Agent header, timeout=15
- Cache key: `{"source": "cftc_weekly"}`
- Return raw CSV text
- Error handling: HTTP errors, empty response

### Step 6b.3: Implement `_fetch_historical(year: int)`
- GET `https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip`
- Extract f_year.txt from ZIP
- Cache key: `{"source": "cftc_historical", "year": year}`
- Return raw CSV text (with headers)

### Step 6b.4: Implement `_parse_rows(csv_text, has_headers)`
- The 191 column header names (from historical ZIP) stored as class constant `_HEADERS`
- Weekly file: no headers → apply `_HEADERS` as column names
- Historical file: has headers → use them directly
- Parse with csv.reader, handle "." as None, strip whitespace
- Return list of dicts (one per contract row)
- Safe int/float parsing (reuse pattern from GDELT)

### Step 6b.5: Implement `_filter_contracts(rows, contract_filter, code_filter, top_n)`
- If contract_filter: case-insensitive substring match on Market_and_Exchange_Names
- If code_filter: exact match on CFTC_Contract_Market_Code
- Sort by Open_Interest_All descending
- Return top_n rows

### Step 6b.6: Implement `_compute_signals(rows)` 
- For each contract, compute:
  - `managed_money_net` = M_Money_Long - M_Money_Short
  - `producer_merchant_net` = Prod_Merc_Long - Prod_Merc_Short  
  - `swap_dealer_net` = Swap_Long - Swap_Short
  - `mm_net_pct_oi` = managed_money_net / Open_Interest × 100
  - `concentration_top4_net_long` = Conc_Net_LE_4_TDR_Long_All
  - `weekly_oi_change` = Change_in_Open_Interest_All
  - `weekly_mm_change` = Change_in_M_Money_Long - Change_in_M_Money_Short (net flow)
- Return enriched rows with signal fields appended

### Step 6b.7: Implement `execute()` 
- Mode "latest": _fetch_latest → _parse_rows(has_headers=False) → _filter → _compute_signals → format output
- Mode "historical": _fetch_historical(year) → _parse_rows(has_headers=True) → _filter → _compute_signals → format output
- Output format: table-style text summary (contract name, OI, MM net, PM net, changes, signals)
- ToolResult.data: list of dicts for downstream quant processing

### Step 6b.8: Register + bandit arm
- Add `CFTCTool(cache=cache)` to cli.py build_tool_registry()
- Add `futures_positioning` GoalArm to bandit.py DEFAULT_ARMS
  - tools: ["cftc", "market_data"]
  - examples: ["Analyze crude oil futures positioning", "Check managed money crowding in gold"]

### Step 6b.9: Live test + edge case suite
- Live fetch latest, verify parsing, filter by "crude", verify signals computed
- Edge cases: empty filter (all contracts), bad year, missing values, contract not found, HTTP error, malformed CSV, "." handling

## Edge Cases
- Weekly file has no headers — must apply header mapping correctly
- "." values in fields (missing trader counts) → None/0
- Large contracts vs tiny contracts — top_n sorts by OI
- Year out of range for historical (before 2006)
- Empty contract_filter returns all (sorted by OI)

## Testing Plan
- Verify fetch from both URLs
- Parse at least 200+ rows from weekly
- Filter chains: by name, by code, combined
- Signal computation: spot-check managed_money_net against manual calc
- Edge case: contract with "." values doesn't crash
- Edge case: non-existent contract_filter returns empty gracefully

---

## Related

- [[cftc|Research: Cftc]]
