---
title: "Feature: Power Grid Demand (ISO/RTO Energy Data)"
tags:
  - doc/research
  - topic/power-grid
---

# Feature: Power Grid Demand (ISO/RTO Energy Data)

## Current Architecture
- Tool ABC pattern (base.py), registration in cli.py, bandit arm in bandit.py
- DataCache for caching, httpx for HTTP, jsonschema validation on args

## Data Sources — Live Probing Results

### PRIMARY: NYISO MIS (New York ISO) ✅ FREE, NO AUTH
- **Base URL**: `http://mis.nyiso.com/public/csv/{dataset}/{YYYYMMDD}{dataset}.csv`
- **Archive ZIPs**: `http://mis.nyiso.com/public/csv/{dataset}/{YYYYMM}01{dataset}_csv.zip`
- **Cost**: $0, no API key, no authentication, no rate limit observed
- **Coverage**: New York state, 11 load zones, 15 pricing zones
- **Zones**: CAPITL, CENTRL, DUNWOD, GENESE, HUD VL, LONGIL, MHK VL, MILLWD, N.Y.C., NORTH, WEST

#### Datasets (all verified live):
| Dataset | Key | Resolution | Fields | Response |
|---------|-----|-----------|--------|----------|
| Actual Load | `pal` | 5-min | Time, Zone, PTID, Load (MW) | ~3300 lines/day |
| Fuel Mix | `rtfuelmix` | 5-min | Time, Zone, Gen Type, Gen (MW) | ~330 lines/current |
| DA LBMP | `damlbmp` | Hourly | Time, Zone, PTID, LBMP, Congestion, Losses ($/MWh) | ~361 lines |
| RT LBMP | `realtime` | 5-min | Time, Zone, PTID, LBMP, Congestion, Losses ($/MWh) | ~901 lines |
| Load Forecast | `isolf` | Hourly | Time, Zone, Forecast (MW) | ~145 lines |

#### Historical Depth:
- **Daily CSVs**: Rolling ~11 days (day 12 returns 404)
- **Monthly ZIPs**: Available from at least 2023-01 through current month (~780KB each)
- Each monthly ZIP contains all daily CSVs for that month (e.g., Feb 2026 → 28 CSV files)
- **Effective**: 3+ years of history via monthly archives

#### Sample Data:
- Load: NYC zone 4,361 MW at 4:00 AM EDT, total NYISO ~15,119 MW
- DA LBMP: CAPITL $42.02, CENTRL $41.04, DUNWOD $43.29, N.Y.C. $44.78 $/MWh
- RT LBMP: CAPITL $37.23 $/MWh (5-min)
- Fuel mix: Gas 3,626 MW, Hydro 3,004 MW, Dual Fuel 3,194 MW, Nuclear 1,992 MW, Wind 932 MW

### SECONDARY: CAISO OASIS (California ISO) ✅ FREE, NO AUTH (but XML/ZIP)
- **URL**: `http://oasis.caiso.com/oasisapi/SingleZip`
- **Requires**: `follow_redirects=True` (302 on first request)
- **Format**: ZIP containing XML (not CSV) — additional parsing complexity
- **Datasets verified**: SLD_FCST (demand actual/forecast), SLD_REN_FCST (renewables), PRC_LMP (day-ahead LMP)
- **Resolution**: Hourly
- **Historical**: Unclear — 429 rate limiting on multiple historical requests (Dec 2025 worked, others 429)
- **Sample**: Demand 1,342 MW (AVA TAC area), LMP $0.77/MWh (SP15 node)
- **Verdict**: Works but XML parsing adds complexity. NYISO covers our needs in CSV format.

### BLOCKED: Other ISOs
| ISO | URL Tested | Status | Issue |
|-----|-----------|--------|-------|
| EIA v2 API | `api.eia.gov/v2/electricity/rto/` | 403 | Requires free API key (register at eia.gov) — covers ALL ISOs |
| PJM | `dataminer2.pjm.com/feed/` | 401 | API subscription key required (not free) |
| ERCOT | `data.ercot.com`, `www.ercot.com` | 403 | Incapsula bot protection on all endpoints |
| MISO | `api.misoenergy.org/` | 200 (empty) | Returns `{"error": "no data"}` — deprecated/restructured |
| ISO-NE | `webservices.iso-ne.com/` | 401 | Authentication required |
| SPP | `portal.spp.org/` | timeout | Connection timeout |

### FUTURE EXPANSION: EIA API
- `https://api.eia.gov/v2/electricity/rto/` — covers all 7 US ISOs with uniform API
- Free key registration: `https://www.eia.gov/opendata/register.php`
- Datasets: daily-region-data, region-data (hourly), fuel-type-data, interchange-data
- Would enable national coverage. Defer to Phase 6g-b or later.

## Quant Signal Value

### Tier 1: Direct Economic Activity Proxy
1. **Demand vs Forecast Deviation**: Actual load exceeding forecast = unexpected economic activity. Persistent overforecast = economic softening. This is a real-time GDP nowcasting signal.
2. **Year-over-Year Demand Growth**: Zone-level demand YoY changes reveal regional economic acceleration/deceleration faster than BLS/Census data (T+0 vs T+30).
3. **Weekday vs Weekend Ratio**: Rising ratio = industrial expansion. Falling = contraction. Baseload-normalized.

### Tier 2: Energy Market Stress
4. **DA vs RT LBMP Spread**: Large deviations between day-ahead and real-time prices = congestion or supply stress. Persistent positive RT spread = unexpected demand / supply shortage.
5. **Zonal Price Divergence**: Normally correlated zones decoupling = transmission constraints or localized events.
6. **Fuel Mix Shifts**: Sudden gas→renewables shift = price signal (gas expensive). Nuclear curtailment = extreme oversupply.

### Tier 3: Sector-Specific Signals
7. **Datacenter Zones**: N.Y.C., CENTRL (where datacenters cluster) demand growth = AI capex proxy.
8. **Industrial Zones**: WEST, GENESE (manufacturing) demand patterns = industrial production proxy.
9. **Cross-signal**: Power demand spike + natural gas price spike = utility cost pressure → affects energy sector earnings.

## Implementation Approach

### Tool Design: NYISO-focused, multi-mode
- **`demand` mode**: Actual load by zone — single day or date range
- **`fuel_mix` mode**: Generation by fuel type — current or historical
- **`pricing` mode**: DA and RT LBMPs by zone — spread computation
- **`forecast` mode**: Load forecast with actual deviation computation

### Data Flow
1. Check cache (source: `nyiso`, params include dataset + date)
2. Fetch daily CSV from NYISO MIS (or monthly ZIP for historical)
3. Parse CSV → structured dict
4. Compute derived signals (demand deviation, price spread, fuel mix proportions)
5. Return ToolResult with raw data + computed signals

### Signal Computation
- **Demand deviation**: `(actual - forecast) / forecast × 100` — requires joining `pal` + `isolf`
- **Price spread**: `RT_LBMP - DA_LBMP` per zone per interval
- **Fuel proportions**: MW per fuel type / total MW — percentage of generation
- **Zone anomaly**: Current load vs historical mean ± 2σ (requires archive fetch for baseline)

## Risks
- NYISO daily CSV rolling window is ~11 days. Historical requires monthly ZIP download + unzip.
- Monthly ZIPs are ~780KB each — fetching 12 months = ~9MB. Need efficient caching.
- CSV format may change without notice (public, undocumented API).
- Time zones: NYISO uses EDT/EST — must normalize to UTC for cross-signal fusion.
- No guaranteed SLA — it's a public file server, not an enterprise API.

## Dependencies
- httpx (existing)
- csv (stdlib)
- zipfile (stdlib)
- io (stdlib)
- No new external dependencies required

---

## Related

- [[power_grid_spec|Spec: Power Grid]]
