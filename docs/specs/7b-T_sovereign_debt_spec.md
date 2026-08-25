---
title: "Spec: 7b-T Sovereign Debt Tool"
tags:
  - doc/spec
  - layer/surveillance
  - phase/7b
  - topic/sovereign-debt
---

# Spec: 7b-T Sovereign Debt Tool

## Goal
Build `sovereign_debt` tool (#35) that fetches government bond yield data from 4 countries/regions (US, EU, JP, UK) via free public APIs, and computes cross-country spreads as fiscal stress signals.

## Files Affected
1. **Create** `agent/tools/sovereign_debt.py` — main tool implementation
2. **Edit** `agent/cli.py` — import + register tool
3. **Edit** `agent/learning/bandit.py` — add `sovereign_stress` arm (#23)
4. **Create** `tests/test_sovereign_debt_edge.py` — edge case test suite

## Implementation Steps

### 3.1: Create sovereign_debt.py skeleton
- Class `SovereignDebtTool(Tool)` with name, description, parameters
- 5 modes: `us_yields`, `eu_yields`, `jp_yields`, `uk_gilts`, `spreads`
- Parameter schema: `mode` (required), `month` (YYYY-MM, optional, default=current), `countries` (list, for eu_yields), `days` (int, for recent data window)
- Constants: URLs, timeouts, cache TTLs, valid modes

### 3.2: Implement `_fetch_us_yields(month)`
- Fetch US Treasury Atom XML for given YYYYMM
- Parse XML with namespaces (`d:`, `m:`)
- Extract date + all maturity yields (1mo through 30yr)
- Handle missing values (`-` entries)
- Return list of dicts: `{date, yields: {1m, 2m, ..., 30y}, curve_2s10s, curve_3m10y}`
- Compute 2s10s and 3m10y spreads inline

### 3.3: Implement `_fetch_eu_yields(countries, start_period)`
- Fetch ECB IRS CSV for each country code
- Parse CSV: extract TIME_PERIOD and OBS_VALUE
- Default countries: DE, FR, IT, ES, GR, PT, NL, BE, AT, IE
- Return: `{country: [{period, yield_pct}]}`
- Compute spreads vs DE (risk-free proxy) inline

### 3.4: Implement `_fetch_jp_yields()`
- Fetch Japan MOF current-month CSV
- Parse: skip header lines (row 0 = title, row 1 = column headers)
- Handle `-` values (missing maturities)
- Return list of dicts: `{date, yields: {1y, 2y, ..., 40y}}`

### 3.5: Implement `_fetch_uk_gilts()`
- Fetch UK DMO XML (D2.1E = issuance history)
- Parse XML: extract INSTRUMENT_NAME, ISIN_CODE, ACTUAL_DATE, ISSUE_YIELD, NOMINAL_ISSUED
- Filter to recent N months of issuance
- Return list of auction records sorted by date desc

### 3.6: Implement `spreads` mode
- Fetch latest EU yields for requested countries
- Compute spread vs DE for each country
- Also fetch latest US 2s10s spread
- Return structured spread data with change direction

### 3.7: Wire up execute() dispatcher
- Mode validation → dispatch to _fetch_* methods
- Cache integration: us_yields 1800s, eu_yields 3600s, jp_yields 1800s, uk_gilts 7200s
- Error handling: network errors → ToolResult(success=False)

### 3.8: Register in cli.py
- Import SovereignDebtTool
- `registry.register(SovereignDebtTool(cache=cache))`

### 3.9: Add bandit arm
- Name: `sovereign_stress`
- Tools: `["sovereign_debt", "macro_data", "market_data"]`
- Examples: yield curve inversion check, eurozone spread widening, JGB breakout

### 3.10: Write edge case tests
- XML parsing (malformed, empty, missing namespaces)
- CSV parsing (empty, missing values, non-numeric)
- Mode routing (invalid mode, missing mode)
- Parameter validation (bad month format, invalid country codes)
- Cache integration
- Network errors (timeout, 404, 500)
- Spread computation (missing DE baseline, single country)
- Tool schema validation
- Registry integration
- Bandit arm verification (count = 23)

## Edge Cases
- US Treasury XML may have entries where some maturities are empty (newly introduced tenors)
- Japan MOF CSV footer has Japanese characters — must stop parsing before that
- ECB may return 404 for small countries with no bond market (EE, LV, etc.)
- Month parameter validation: must be YYYY-MM format
- UK DMO XML may have thousands of entries — limit to recent N months
- Empty response body from any API

## Testing Plan
- Unit tests with mocked HTTP responses
- All 5 modes tested with valid mock data
- Error paths: network timeout, malformed XML/CSV, empty responses
- Spread computation math verified
- Full registry integration test
- Bandit arm count = 23

---

## Related

- [[7b-T_sovereign_debt|Research: 7B-T Sovereign Debt]]
- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]
