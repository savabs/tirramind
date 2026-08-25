---
title: "Spec: Batch 7 — Satellite Activity, Electricity Monitor, Interconnection Queue"
tags:
  - doc/spec
  - layer/surveillance
  - phase/7b
  - topic/satellite
---

# Spec: Batch 7 — Satellite Activity, Electricity Monitor, Interconnection Queue

Research: [[batch7_satellite_electricity_queue]]

## Goal
Build 3 Layer-0/Layer-1 raw observation tools that provide physical-world ground truth.

## Files to Create
1. `agent/tools/satellite_activity.py` — SatelliteActivityTool
2. `agent/tools/electricity_monitor.py` — ElectricityMonitorTool
3. `agent/tools/interconnection_queue.py` — InterconnectionQueueTool
4. `tests/test_satellite_activity_edge.py` — edge case suite
5. `tests/test_electricity_monitor_edge.py` — edge case suite
6. `tests/test_interconnection_queue_edge.py` — edge case suite

## Files to Modify
1. `agent/cli.py` — import + register 3 tools (count: 44 → 47)
2. `agent/learning/bandit.py` — add 3 GoalArm entries (count: 32 → 35)

---

## Tool 1: SatelliteActivityTool (7b-I)

**Name:** `satellite_activity`
**Modes:** `fire`, `vegetation`, `events`

### Mode: fire
- Params: `area` (country code or bbox "W,S,E,N"), `source` (VIIRS_NOAA20_NRT default), `days` (1-10, default 1)
- API: NASA FIRMS area/csv endpoint
- Env: `TIRRA_NASA_FIRMS_KEY`
- Returns: hotspot count, avg/max FRP, top clusters by proximity, confidence distribution
- Signal: FRP near industrial zones = operational intensity

### Mode: vegetation
- Params: `latitude`, `longitude`, `start_date` (YYYY-MM-DD), `end_date` (YYYY-MM-DD), `km_radius` (0-100, default 0)
- API: MODIS Web Service MOD13Q1 subset
- No auth required
- Returns: NDVI time series, current health assessment, anomaly flag vs historical mean
- Signal: Crop stress → commodity price driver

### Mode: events
- Params: `category` (wildfires/volcanoes/severeStorms/seaLakeIce/etc., optional), `days` (1-365, default 30), `status` (open/closed, default open), `bbox` (W,S,E,N optional)
- API: NASA EONET v3
- No auth required
- Returns: event list with title, category, coordinates, date, magnitude
- Signal: Natural disasters disrupting supply chains

### Helpers
- `_fetch_firms(area, source, days, api_key)` → list[dict] | None
- `_fetch_ndvi(lat, lon, start, end, km)` → dict | None
- `_fetch_eonet(category, days, status, bbox)` → list[dict] | None
- `_cluster_hotspots(points, radius_km)` → list[dict]  (simple grid-based clustering)
- `_ndvi_health(value)` → str  (bare/sparse/moderate/healthy/dense)

---

## Tool 2: ElectricityMonitorTool (7b-AD)

**Name:** `electricity_monitor`
**Modes:** `demand`, `generation`, `interchange`

### Mode: demand
- Params: `region` (BA code: PJM, CAISO, ERCO, MISO, etc.), `days` (1-7, default 1)
- API: EIA electricity/rto/region-data/data/
- Env: `TIRRA_EIA_API_KEY`
- Returns: hourly demand series, peak/trough/avg MW, demand vs forecast deviation %

### Mode: generation
- Params: `region` (BA code), `days` (1-7, default 1)
- API: EIA electricity/rto/fuel-type-data/data/
- Returns: generation by fuel type (MW), fuel mix proportions, renewable share %, fossil share %

### Mode: interchange
- Params: `region` (BA code), `days` (1-7, default 1)
- API: EIA electricity/rto/interchange-data/data/
- Returns: interchange flows to/from neighboring BAs, net import/export, trading partners

### Helpers
- `_fetch_eia(endpoint, facets, api_key)` → list[dict] | None
- `_aggregate_hourly(records)` → dict with peak/trough/avg
- `_fuel_mix_proportions(records)` → dict with fuel percentages

### US Balancing Authorities (subset)
CAISO, ERCO, ISNE, MISO, NYIS, PJM, SWPP, SOCO, TVA, DUKE, FPL, CPLE, etc.

---

## Tool 3: InterconnectionQueueTool (7b-K)

**Name:** `interconnection_queue`
**Modes:** `queue`, `summary`, `datacenter`

### Mode: queue
- Params: `state` (2-letter, optional), `fuel` (SUN/WND/NG/NUC/WAT/MWH/etc., optional), `status` (planned/construction, default planned), `min_mw` (optional)
- API: EIA electricity/operating-generator-capacity/data/ with status=PL or U
- Env: `TIRRA_EIA_API_KEY`
- Returns: list of planned/under-construction generators with capacity, technology, location, expected date

### Mode: summary
- Params: `state` (optional), `status` (planned/construction/both, default both)
- Same API endpoint
- Returns: aggregate MW by fuel type, by state, by year; total pipeline; technology breakdown

### Mode: datacenter
- Params: `state` (optional), `min_mw` (optional, default 50)
- Same API, filters for likely data center projects
- Detection: entity name matching against hyperscaler patterns (Amazon, AWS, Microsoft, Google, Meta, Apple, QTS, Equinix, Digital Realty, CyrusOne, etc.)
- Returns: suspected data center projects with capacity, location, operator, status

### Helpers
- `_fetch_generators(facets, api_key)` → list[dict] | None
- `_is_datacenter(entity_name, plant_name)` → bool
- `_status_to_eia(status)` → str  (planned→PL, construction→U)
- `_summarize_pipeline(records)` → dict

---

## Bandit Arms

### `satellite_surveillance`
- Tools: satellite_activity, weather_alerts, web_search
- Examples: fire hotspot monitoring near oil refineries, crop NDVI assessment, natural disaster tracking

### `electricity_demand`
- Tools: electricity_monitor, power_grid, macro_data
- Examples: cross-region demand anomaly, fuel mix shift, inter-regional flow analysis

### `energy_infrastructure_pipeline`
- Tools: interconnection_queue, electricity_monitor, web_search
- Examples: solar/wind pipeline growth, data center capacity buildout, queue withdrawal analysis

---

## Testing Plan
Per-tool edge case suite covering:
- Mode/param validation (invalid mode, missing required params, out-of-range values)
- API key handling (missing key, present key, env var)
- HTTP error paths (timeout, 4xx, 5xx, non-JSON)
- Cache hit/miss paths
- Empty/malformed API responses
- Helper function unit tests
- Integration counts (47 tools, 35 arms)

## Count Targets After Batch
- Tools: 44 → 47
- Arms: 32 → 35
- Test files referencing counts: ~20 files need updating

---

## Related

- [[batch7_satellite_electricity_queue|Research: Batch7 Satellite Electricity Queue]]
