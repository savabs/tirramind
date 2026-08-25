---
title: "Spec: 7b-AO — Supply Chain Price Pressure Monitor"
tags:
  - doc/spec
  - layer/surveillance
  - phase/7b
  - topic/supply-chain
---

# Spec: 7b-AO — Supply Chain Price Pressure Monitor

## Goal

Track upstream producer price pressure across key manufacturing sectors via BLS
PPI and Import Price data. Detect broad-based vs concentrated price acceleration,
sector-level inflation signals, and import/tariff pressure.

## Files Affected

| File | Action |
|---|---|
| `agent/tools/supply_chain_monitor.py` | CREATE — new tool |
| `agent/cli.py` | MODIFY — register tool |
| `agent/learning/bandit.py` | MODIFY — add bandit arm |
| `tests/test_supply_chain_monitor_edge.py` | CREATE — edge case tests |

## Implementation Steps

### 7b-AO.1: Create tool skeleton with param validation
- Class `SupplyChainMonitorTool(Tool)`
- name: `supply_chain_monitor`
- modes: `producer_prices`, `import_prices`, `pressure_index`
- Parameters: `mode` (required), `sectors` (comma-sep, default "all"),
  `months` (int, default 12, max 24)
- Tracked PPI sectors dict with series IDs, human names, and signal categories:
  semiconductors, computers, construction_machinery, steel, petroleum, chemicals,
  cement, paperboard
- Validate mode, sectors, clamp months
- **Verification:** test metadata, parameter validation, invalid mode

### 7b-AO.2: Implement producer_prices mode
- Source: BLS PPI via POST `https://api.bls.gov/publicAPI/v2/timeseries/data/`
- Batch all requested sector series into one request (max 50 per call)
- Parse BLS response: year-period → date, value → float, handle missing
- Compute per-sector: latest value, MoM %, YoY % (same month prior year),
  3-month trend (accelerating/decelerating/stable),
  signal: PRICE_SPIKE (MoM > 3%), DEFLATION (MoM < -2%), NORMAL
- Output: sector-by-sector table + summary (how many accelerating vs decelerating)
- Cache TTL: 21600s (6hr — monthly data)
- **Verification:** mock BLS responses, math edge cases

### 7b-AO.3: Implement import_prices mode
- Source: BLS Import Price Program
- Series: EIUIR (all imports), EIUCOMP (computers), EIUIR334413 (electronics components)
- Same BLS POST endpoint, same parsing
- Compute: latest import price index, MoM %, YoY %, import vs domestic PPI spread
- Signal: TARIFF_PRESSURE (imports rising faster than domestic),
  FX_PASS_THROUGH (import prices spiking), SUPPLY_EASING (imports deflating)
- Cache TTL: 21600s
- **Verification:** fixture tests for normal, empty, error responses

### 7b-AO.4: Implement pressure_index mode
- Combines producer_prices + import_prices data
- Compute: count of sectors with MoM > 0 (rising) vs MoM < 0 (falling)
- Broad-based indicator: if >60% of sectors rising → BROAD_PRESSURE
- Concentrated indicator: if only 1-2 sectors spiking → SECTOR_SPECIFIC
- Import-domestic spread: average import MoM - average domestic MoM
- Output: pressure classification, sector breakdown, import vs domestic summary
- **Verification:** synthetic data tests for all classification branches

### 7b-AO.5: Register in cli.py + bandit arm
- Register `SupplyChainMonitorTool` in cli.py
- Add bandit arm: `supply_chain_prices` (tools: supply_chain_monitor, energy_supply, market_data)
- **Verification:** registration count assertion, bandit arm exists

### 7b-AO.6: Edge case test suite
- Input validation: invalid mode, bad sectors string, months range
- BLS: rate limited (429), empty series, series not found, malformed JSON,
  non-200 responses, timeout
- Math: zero-value latest (div by zero in MoM), single data point,
  all series return empty, negative PPI values (shouldn't happen but guard)
- Sectors: "all" keyword, individual sector, unknown sector name
- Cache: hit/miss/stale
- **Verification:** comprehensive test file

## Edge Cases

- BLS allows max 2-year span per request — for >24 months, need multiple calls
- Some PPI series are discontinued → handle gracefully, mark as unavailable
- BLS returns data ordered by year DESC, period DESC — preserve that or re-sort
- Period format "M01"-"M13" (M13 = annual avg) — filter out M13
- BLS might return REQUEST_NOT_PROCESSED error — surface clearly
- 25 req/day limit without BLS key — aggressive caching is critical

---

## Related

- [[7b-AO_supply_chain_monitor|Research: 7B-Ao Supply Chain Monitor]]
- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]
