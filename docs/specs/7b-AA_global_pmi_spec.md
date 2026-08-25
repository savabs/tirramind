---
title: "Spec: 7b-AA — Global PMI / Leading Indicators Tool (OECD CLI)"
tags:
  - doc/spec
  - layer/surveillance
  - phase/7b
  - topic/global-pmi
---

# Spec: 7b-AA — Global PMI / Leading Indicators Tool (OECD CLI)

## Goal
Fetch OECD Composite Leading Indicators (CLI), Business Confidence Indicators (BCI), and Consumer Confidence Indicators (CCI) via the OECD SDMX API. Provide multi-country leading indicator data with quant signals: turning point detection, cross-country momentum spreads, regime classification.

## Files Affected
- **Create:** `agent/tools/global_pmi.py`
- **Modify:** `agent/cli.py` — add import + registration
- **Modify:** `agent/learning/bandit.py` — add GoalArm
- **Create:** `tests/test_global_pmi_edge.py`

## Implementation Steps

### 2.1: Create `agent/tools/global_pmi.py`
- Class `GlobalPmiTool(Tool)`
- `name = "global_pmi"`
- 3 modes: `cli` (Composite Leading Indicators), `bci` (Business Confidence), `cci` (Consumer Confidence)
- Parameters: `mode`, `countries` (comma-separated ISO codes, default "USA,GBR,DEU,FRA,JPN,CHN"), `start_period` (YYYY-MM), `end_period` (YYYY-MM), `include_signals` (bool, default true)
- `__init__(self, cache: DataCache | None = None)`
- Use `httpx.Client` with `timeout=30` (OECD can be slow), User-Agent header
- Cache with `DataCache` (key by mode + countries + period range)
- Parse CSV response (csvfilewithlabels format)
- Compute signals: momentum (6m rate of change), regime (>100/expanding vs <100/contracting), cross-country spreads
- Return `ToolResult` with formatted output + structured `data`

### 2.2: API Implementation Details
- Base URL: `https://sdmx.oecd.org/public/rest/data/`
- Dataflows:
  - CLI: `OECD.SDD.STES,DSD_STES@DF_CLI`
  - BCI: `OECD.SDD.STES,DSD_STES@DF_BCI`
  - CCI: `OECD.SDD.STES,DSD_STES@DF_CCI`
- Dimension selection: `{countries}.M.LI...AA...H` (amplitude-adjusted, monthly)
  - For BCI/CCI: dimension path may differ — need to probe, build with flexibility
- Params: `startPeriod={YYYY-MM}`, `endPeriod={YYYY-MM}`, `dimensionAtObservation=AllDimensions`, `format=csvfilewithlabels`
- Country codes: ISO 3-letter (USA, GBR, DEU, FRA, JPN, CHN, KOR, AUS, CAN, ITA, ESP, BRA, IND, MEX, OECD, G-7, EA19)
- Response: CSV with columns including REF_AREA, TIME_PERIOD, OBS_VALUE

### 2.3: Register in `agent/cli.py`
- Import `GlobalPmiTool` from `agent.tools.global_pmi`
- Add `registry.register(GlobalPmiTool(cache=cache))` after drug_regulatory

### 2.4: Add GoalArm in `agent/learning/bandit.py`
- `name="global_pmi_monitor"`
- `tools=["global_pmi", "web_search"]`
- Examples: CLI turning points for G7, US vs China momentum spread, business confidence divergence, synchronized downturn detection

### 2.5: Write edge-case tests
- Cover: all 3 modes, invalid mode, invalid country codes, empty CSV response, HTTP errors, malformed CSV, cache, missing OBS_VALUE entries, signal computation with edge values, large multi-country queries, date range validation, tool schema validation

## Edge Cases
- Country code not in OECD database → endpoint returns empty or error XML
- Some countries have discontinued/sparse series → missing values in CSV
- Rate limiting → retry or graceful failure message
- SDMX API sometimes returns XML error messages even when CSV format requested
- `OBS_VALUE` can be missing (NaN) for some periods → skip in signal computation
- BCI/CCI dimension paths may differ from CLI → need adaptive path building
- Large country lists (>10) may cause slow responses → warn in output

## Testing Plan
- Mock `httpx.Client.get` responses with synthetic CSV matching OECD format
- Test each mode independently (CLI, BCI, CCI)
- Test signal computation: momentum, regime, cross-country spread with known values
- Test country parsing (comma-separated, single, empty → default)
- Test error paths: 404 (bad dataflow), 500, timeout, rate limit
- Test empty/sparse CSV (missing values, single row)
- Validate `ToolResult` structure
- Test cache integration

---

## Related

- [[7b-AA_global_pmi|Research: 7B-Aa Global Pmi]]
- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]
