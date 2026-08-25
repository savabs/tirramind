---
title: "Research: Weather & Climate Alerts (7b-C)"
tags:
  - doc/research
  - layer/surveillance
  - phase/7b
  - topic/weather
---

# Research: Weather & Climate Alerts (7b-C)

**Date:** 2026-03-28 (retroactive documentation)
**Tool:** `agent/tools/weather_alerts.py` → `WeatherAlertsTool`
**Status:** IMPLEMENTED, TESTED

## APIs Probed

### NOAA National Weather Service ✅ SELECTED (US alerts)
- **URL:** `https://api.weather.gov`
- **Method:** GET
- **Auth:** None (User-Agent required: `TirraMind/0.1`)
- **Format:** GeoJSON FeatureCollection
- **Rate limits:** Undocumented, generous
- **Coverage:** **US + territories** — 50 states, DC, PR, VI, GU, AS, MP
- **Features:** Active alerts with severity filter, 20 market-relevant event types

### NASA FIRMS MODIS ✅ SELECTED (Global fires)
- **URL:** `https://firms.modaps.eosdis.nasa.gov/data/active_fire/c6.1/csv/MODIS_C6_1_Global_24h.csv`
- **Method:** GET
- **Auth:** None
- **Format:** CSV (15-30K fires/day)
- **Rate limits:** Undocumented
- **Coverage:** **Global** — MODIS satellite covers entire Earth surface
- **Features:** 24-hour rolling window, lat/lon/brightness/confidence/frp

### APIs Probed But Failed
| Source | Status | Reason |
|--------|--------|--------|
| TSA | ❌ | DNS resolution failure |
| Eurocontrol | ❌ | HTTP 404 |
| Port of LA | ❌ | SSL certificate error |

## Geographic Coverage
- NWS alerts: US-only (inherent to data source)
- NASA FIRMS: Global data, but currently filtered to 12 US infrastructure zones
- **Verdict:** `[G:REGIONAL]` — FIRMS *could* be global with international infrastructure zones
- **Potential expansion:** Add international infrastructure zones (Strait of Malacca refineries, Rhine industrial corridor, Japan Pacific Coast, Australian mining belt)
- **Potential expansion:** Add Meteoalarm (EU weather), JMA (Japan), BOM (Australia)

## Modes Implemented
1. `alerts` — active NOAA NWS severe weather alerts, filtered by severity + state
2. `fires` — NASA FIRMS fire detections near 12 infrastructure zones
3. `summary` — combined NWS + FIRMS overview

## 12 Infrastructure Zones (fires mode)
1. Permian Basin TX — oil/gas
2. Gulf Coast Refineries — refining
3. California Refineries — refining
4. Powder River Basin WY — coal
5. Appalachian Gas Fields — natural gas
6. Bakken Oil Field ND — oil
7. ERCOT Texas Grid — power
8. PJM East Grid — power
9. Corn Belt Central IL — agriculture
10. California Central Valley — agriculture
11. Pacific NW Timber OR — forestry
12. Colorado River Basin — water

## Signal Value
- Severe weather → supply chain disruption, energy demand spike, crop damage
- Wildfires near infrastructure → utility liability, supply interruption
- Hurricane forecasts → insurance sector, offshore drilling, shipping

## Risks
- NWS API occasionally returns empty during low-activity periods
- FIRMS CSV is large (15-30K rows) — parsing overhead
- No international weather alert integration yet

---

## Related

- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]
