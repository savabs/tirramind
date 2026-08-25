---
title: "Research: Batch 7 — Satellite Activity, Electricity Monitor, Interconnection Queue"
tags:
  - doc/research
  - layer/surveillance
  - phase/7b
  - topic/satellite
---

# Research: Batch 7 — Satellite Activity, Electricity Monitor, Interconnection Queue

## 7b-I: Satellite-Derived Physical Activity

### Current Architecture
- No satellite tools exist yet.
- power_grid.py covers NYISO only (11 NY zones, free CSV, no auth).
- All tools follow: `Tool` base class, `DataCache`, `ToolResult` pattern.

### API Survey

**NASA FIRMS (Fire Information for Resource Management System)**
- URL: `https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{source}/{area}/{day_range}`
- Auth: MAP key required (free — register at firms.modaps.eosdis.nasa.gov)
- Env var: `TIRRA_NASA_FIRMS_KEY`
- Sources: `VIIRS_NOAA20_NRT`, `VIIRS_SNPP_NRT`, `MODIS_NRT`
- Area: country code (e.g. `USA`) or bounding box `west,south,east,north`
- Day range: 1–10 (recent data only)
- Returns: CSV with latitude, longitude, brightness, scan, track, acq_date, acq_time, satellite, instrument, confidence, bright_t31, frp (fire radiative power), daynight, type
- Rate limit: 10 requests/minute (free tier)
- Signal: FRP near known industrial sites = operational intensity. Wildfire near infrastructure = supply chain disruption.
- Country endpoint: `https://firms.modaps.eosdis.nasa.gov/api/country/csv/{MAP_KEY}/{source}/{country}/{days}`

**MODIS Web Service (ORNL DAAC)**
- URL: `https://modis.ornl.gov/rst/api/v1/{product}/subset`
- Auth: None required
- Product: MOD13Q1 (MODIS NDVI 250m resolution, 16-day composite)
- Params: `latitude`, `longitude`, `band=250m_16_days_NDVI`, `startDate=A{YYYYDDD}`, `endDate=A{YYYYDDD}`, `kmAboveBelow`, `kmLeftRight`
- Available dates: `https://modis.ornl.gov/rst/api/v1/{product}/dates?latitude={lat}&longitude={lon}`
- Returns: JSON with calendar_date, band, scale, value, pixel arrays
- NDVI range: -2000 to 10000 (scale 0.0001, so real range -0.2 to 1.0; healthy vegetation > 0.3)
- Signal: Crop health, drought stress, deforestation. Agricultural commodity price driver.

**NASA EONET (Earth Observatory Natural Events Tracker)**
- URL: `https://eonet.gsfc.nasa.gov/api/v3/events`
- Auth: None required
- Params: `status` (open/closed), `limit`, `category` (wildfires, volcanoes, severeStorms, seaLakeIce, etc.), `days`, `bbox`
- Returns: JSON with events (id, title, categories, sources, geometry with coordinates and dates)
- Signal: Natural disasters that disrupt supply chains, infrastructure, agriculture.

### Design Decision
Focus on REST-API-accessible data with <30s latency. No rasterio/geopandas (heavy C deps).
Three modes: `fire` (FIRMS), `vegetation` (MODIS NDVI), `events` (EONET).
Nighttime light quantitative analysis deferred to when geospatial stack is added.

---

## 7b-AD: Electricity Monitor (Global)

### Current Architecture
- power_grid.py covers NYISO only (demand, fuel_mix, pricing, forecast).
- 7b-AD should expand to US-wide + international.

### API Survey

**EIA API v2 (Energy Information Administration)**
- Base: `https://api.eia.gov/v2/`
- Auth: API key required (free — register at eia.gov/opendata)
- Env var: `TIRRA_EIA_API_KEY`
- Rate limit: 1000 requests/hour (free tier)
- Endpoints:
  - Hourly demand: `electricity/rto/region-data/data/`
    - Facets: `respondent` (BA code), `type` (demand, day-ahead demand forecast, demand forecast, net generation)
    - Value: MW
    - Period: hourly (recent 3 days), daily aggregation available
    - Covers all US RTOs: CAISO, ERCO, ISNE, MISO, NYIS, PJM, SWPP, etc.
  - Generation by fuel: `electricity/rto/fuel-type-data/data/`
    - Facets: `respondent` (BA code), `fueltype` (COL, NG, NUC, OIL, OTH, SUN, WAT, WND)
    - Value: MW (hourly generation)
  - Interchange: `electricity/rto/interchange-data/data/`
    - Facets: `fromba` (from BA), `toba` (to BA)
    - Value: MW (hourly interchange)
  - All endpoints return JSON: `response.data` array with period, value, facet values

### Design Decision
Expand beyond NYISO to all US balancing authorities using EIA API.
Three modes: `demand`, `generation`, `interchange`.
Provides: cross-region demand anomalies, fuel mix shifts, inter-regional flow patterns.
International (ENTSO-E, AEMO) deferred — requires separate API tokens and different auth patterns.
Nightlight cross-reference deferred — requires geospatial processing.

---

## 7b-K: Interconnection Queue (US Generator Pipeline)

### Current Architecture
- No queue/pipeline tool exists.
- power_grid.py covers real-time data, not future build pipeline.

### API Survey

**EIA API v2 — Planned & Under-Construction Generators**
- Endpoint: `electricity/operating-generator-capacity/data/`
- Auth: Same EIA API key (`TIRRA_EIA_API_KEY`)
- Facets:
  - `status`: OP (operating), PL (planned), U (under construction), TS (testing), RE (retired), etc.
  - `stateid`: state FIPS code
  - `balession_authority_code`: BA code
  - `sector`: utility, IPP, commercial, industrial
  - `energy_source_code`: SUN (solar), WND (wind), NG (gas), NUC (nuclear), WAT (hydro), MWH (battery), etc.
- Data fields: entityid, entityName, plantid, plantName, generatorid, nameplate_capacity_mw, operating_year_month, technology, energy_source_desc, balancing_authority_code, status, stateid, county, latitude, longitude
- Signal: MW in pipeline by technology = energy transition speed. Data center detection via entity/plant name matching. Queue concentration by region = grid stress.

### Individual ISO Queues (backup/future)
- PJM: downloadable Excel at pjm.com — public but not REST API
- CAISO: requires login
- MISO: download page at misoenergy.org
- ERCOT: downloadable reports

### Design Decision
Use EIA API for planned/under-construction generators (comprehensive US coverage via single API).
Three modes: `queue` (search), `summary` (aggregate stats), `datacenter` (hyperscaler detection).
Data center detection: match entity names against patterns (Amazon, Microsoft, Google, Meta, Apple, etc.).

---

## Risks
- NASA FIRMS MAP key is free but requires registration — tool gracefully degrades without key.
- EIA API key is free but required — tool returns error with clear message if missing.
- MODIS Web Service has no auth but may be slow for large area requests.
- EONET is free/no-auth but event count varies (may return 0 events in quiet periods).
- EIA generator data is updated monthly, not daily — queue changes are slow-moving signals.

## Dependencies
- httpx (already in project)
- csv, io (stdlib)
- No new pip dependencies required.

---

## Related

- [[batch7_satellite_electricity_queue_spec|Spec: Batch7 Satellite Electricity Queue]]
