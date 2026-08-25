---
title: "Spec: Power Grid Demand Tool (NYISO)"
tags:
  - doc/spec
  - topic/power-grid
---

# Spec: Power Grid Demand Tool (NYISO)

## Goal
Build a tool that fetches power grid data from NYISO's free public CSV files, providing real-time demand, fuel mix, pricing (LBMPs), and load forecasts across New York's 11 zones. Computes derived signals: demand-forecast deviation, DA-RT price spread, fuel mix proportions, and zone-level anomalies.

## Files Affected
- **Create**: `agent/tools/power_grid.py`
- **Modify**: `agent/cli.py` (register tool)
- **Modify**: `agent/learning/bandit.py` (add `energy_demand` arm)
- **Create**: `tests/test_power_grid_edge.py`

## Data Source
- **URL pattern**: `http://mis.nyiso.com/public/csv/{dataset}/{YYYYMMDD}{dataset}.csv`
- **Archive pattern**: `http://mis.nyiso.com/public/csv/{dataset}/{YYYYMM}01{dataset}_csv.zip`
- **Datasets**: `pal` (actual load), `rtfuelmix` (fuel mix), `damlbmp` (DA LBMPs), `realtime` (RT LBMPs), `isolf` (load forecast)
- **No auth, no key, no rate limit observed**

## Implementation Steps

### Step 1: Tool skeleton
Create `agent/tools/power_grid.py` with class `PowerGridTool(Tool)`, properties (`name`, `description`, `parameters`), and empty `execute()`.

**Parameters schema:**
```json
{
  "mode": {"type": "string", "enum": ["demand", "fuel_mix", "pricing", "forecast"]},
  "zone": {"type": "string", "description": "NYISO zone (e.g. N.Y.C., CAPITL). Omit for all zones."},
  "date": {"type": "string", "description": "Date YYYY-MM-DD. Default: today."}
}
```

### Step 2: CSV fetch helper
Implement `_fetch_csv(dataset, date_str)`:
- Build URL from dataset + date
- httpx GET with timeout, User-Agent
- Check cache first (source=`nyiso`, params={dataset, date})
- Parse CSV text → list of dicts
- Handle 404 (daily CSV expired → fall back to monthly ZIP)
- Return parsed rows

### Step 3: Monthly ZIP fallback
Implement `_fetch_from_archive(dataset, date_str)`:
- Build monthly ZIP URL from dataset + YYYYMM
- httpx GET, check cache
- Unzip in memory (zipfile + io.BytesIO)
- Find the specific day's CSV inside the ZIP
- Parse and return rows
- Cache the full month's data for subsequent calls

### Step 4: Demand mode
Implement `_demand(date_str, zone)`:
- Fetch `pal` dataset
- CSV columns: Time Stamp, Time Zone, Name (zone), PTID, Load (MW)
- Filter by zone if specified
- Compute: total load per zone, peak load, peak time, off-peak load
- Return zone summaries with load values

### Step 5: Fuel mix mode
Implement `_fuel_mix(date_str, zone)`:
- Fetch `rtfuelmix` dataset
- CSV columns: Time Stamp, Time Zone, Fuel Category, Gen MWh (actually MW)
- Latest snapshot (most recent timestamp) fuel breakdown
- Compute proportions: each fuel type's MW / total MW × 100
- Return fuel breakdown with MW and percentages

### Step 6: Pricing mode
Implement `_pricing(date_str, zone)`:
- Fetch `damlbmp` and `realtime` datasets
- DA CSV columns: Time Stamp, Name, PTID, LBMP ($/MWh), Marginal Cost Losses, Marginal Cost Congestion
- RT CSV columns: same but 5-min intervals
- Compute DA-RT spread per zone (latest available intervals)
- Flag zones where |spread| > threshold (e.g., $5/MWh)
- Return zone pricing with DA, RT, spread, congestion component

### Step 7: Forecast mode
Implement `_forecast(date_str, zone)`:
- Fetch `isolf` (forecast) and `pal` (actual) datasets
- Compute deviation: `(actual - forecast) / forecast × 100` per zone per hour
- Flag significant deviations (> ±5%)
- Return forecast vs actual comparison with deviation percentages

### Step 8: Execute dispatch + registration
Wire `execute()` to dispatch by mode, register in `cli.py` and `bandit.py`.

## Edge Cases
- Date in future → error message
- Invalid zone name → error with valid zone list
- Invalid mode → error with valid modes
- Daily CSV 404 → graceful fallback to monthly ZIP
- Monthly ZIP 404 → error "data not available for this date"
- Network timeout → ToolResult(success=False)
- Empty CSV (header only) → "no data for this date/zone"
- Malformed CSV rows → skip and log warning
- Zone filter with different casing → normalize (upper)
- Date older than archive availability → appropriate error

## Testing Plan
- Unit tests with mocked HTTP responses for all 4 modes
- CSV parsing edge cases (empty, malformed, header-only, extra columns)
- ZIP fallback logic (daily 404 → archive)
- Zone filtering and normalization
- Signal computations (deviation, spread, proportions)
- Error handling (network, invalid params, missing data)
- Live integration test (1 per mode, skip if network unavailable)

---

## Related

- [[power_grid|Research: Power Grid]]
