---
title: "Spec: Tool Globalization — Track 1 (gov_contracts + macro_data)"
tags:
  - doc/spec
---

# Spec: Tool Globalization — Track 1 (gov_contracts + macro_data)

## Goal
Expand 2 existing tools with verified international data sources. No new tools — extend existing architecture with new regional backends.

## Research
- `[[international_api_alternatives]]` — live probe results
- `[[tool_audit_research_globalization]]` — audit tags

## Design Decision: `region` Parameter

Add a `region` parameter to each tool. Default = existing behavior (US). New regions add new backend fetchers but return the same ToolResult schema.

Each tool's `execute()` dispatches to the right backend based on `region`. This keeps the interface stable for the orchestrator/bandit while expanding coverage.

---

## Files Affected

### gov_contracts.py
- Add `region` param: `"us"` (default, existing USASpending), `"uk"` (new, Contracts Finder OCDS)
- Add `_fetch_uk_contracts()` method
- Update `description` and `parameters` schema
- Keep all existing US behavior unchanged

### macro_data.py
- Add `source` param: `"fred"` (default, existing), `"ecb"` (new), `"world_bank"` (new)
- Add `_fetch_ecb()` and `_fetch_world_bank()` methods
- Update `description` and `parameters` schema
- Keep all existing FRED behavior unchanged

### Tests
- `tests/test_gov_contracts_edge.py` — add UK region tests
- `tests/test_macro_data_intl_edge.py` — new file for ECB + World Bank tests

---

## Implementation Steps (Atomic)

### Step 1: gov_contracts — UK Contracts Finder
1.1: Add `region` param to `parameters` schema (enum: ["us", "uk"])
1.2: Add `_fetch_uk_contracts()` method — GET to Contracts Finder OCDS endpoint, parse releases
1.3: Update `execute()` to dispatch on `region` — "uk" → `_fetch_uk_contracts()`, "us" → existing logic
1.4: Update `description` to mention UK coverage
1.5: Write edge case tests for UK region

### Step 2: macro_data — ECB Data API
2.1: Add `source` param to `parameters` schema (enum: ["fred", "ecb", "world_bank"])
2.2: Add ECB series mapping (common series IDs → ECB SDMX keys)
2.3: Add `_fetch_ecb()` method — GET to ECB data-api, parse SDMX JSON response
2.4: Update `execute()` to dispatch on `source` — "ecb" → `_fetch_ecb()`
2.5: Write edge case tests for ECB source

### Step 3: macro_data — World Bank API
3.1: Add `_fetch_world_bank()` method — GET to api.worldbank.org, parse JSON
3.2: Update `execute()` to dispatch on `source` — "world_bank" → `_fetch_world_bank()`
3.3: Write edge case tests for World Bank source

---

## Edge Cases

- Unknown region/source → clear error message
- UK API returns empty releases → graceful empty result
- ECB SDMX response has no datasets → clear error
- World Bank returns paginated results → handle page metadata
- Network timeout on any backend → same timeout handling as existing
- UK Contracts Finder date format differences (ISO vs US date strings)
- ECB series key not found → informative error with available series hint

## Testing Plan
- All existing US tests must continue passing (zero regression)
- Each new region/source gets: valid response, empty response, HTTP error, timeout, malformed data, param validation, output format verification
- Per workflow: mandatory edge case suite before marking complete

## Related

- [[tool_audit_research_globalization|Research: Tool Audit Globalization]]
