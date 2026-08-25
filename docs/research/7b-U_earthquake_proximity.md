---
title: "Research: Earthquake Proximity (7b-U)"
tags:
  - doc/research
  - layer/surveillance
  - phase/7b
  - topic/earthquake
---

# Research: Earthquake Proximity (7b-U)

**Date:** 2026-03-28 (retroactive documentation)
**Tool:** `agent/tools/earthquake_proximity.py` → `EarthquakeProximityTool`
**Status:** IMPLEMENTED, TESTED

## APIs Probed

### USGS Earthquake Hazards Program ✅ SELECTED
- **URL:** `https://earthquake.usgs.gov/fdsnws/event/1`
- **Method:** GET
- **Auth:** None (free, public)
- **Format:** GeoJSON
- **Rate limits:** None documented
- **Coverage:** **Global** — USGS monitors earthquakes worldwide via global seismograph network
- **Key params:** `minmagnitude`, `starttime`, `endtime`, `limit` (max 500), `orderby` (magnitude/time)
- **Properties:** mag, place, time (epoch ms), alert (green/yellow/orange/red), tsunami (0/1), sig (significance score)

## Geographic Coverage
- USGS seismic network covers entire planet
- 19 critical infrastructure zones span 7 countries + 2 maritime chokepoints
- **Verdict:** `[G:GLOBAL]`

## 19 Infrastructure Zones (8 sectors)
| Sector | Zone | Location | Radius |
|--------|------|----------|--------|
| Semiconductor | TSMC Hsinchu | Taiwan | 100km |
| Semiconductor | TSMC Tainan | Taiwan | 80km |
| Semiconductor | Samsung Pyeongtaek | South Korea | 80km |
| Mining | Escondida | Chile | 100km |
| Mining | Chuquicamata | Chile | 80km |
| Mining | Grasberg | Indonesia | 80km |
| Mining | Indonesia Nickel Belt | Indonesia | 200km |
| Nuclear | Fukushima | Japan | 80km |
| Nuclear | Kashiwazaki-Kariwa | Japan | 80km |
| Nuclear | Turkey Point | Florida, US | 80km |
| Energy | BTC Pipeline | Turkey | 150km |
| Energy | Permian Basin | Texas, US | 200km |
| Energy | Oklahoma Injection | Oklahoma, US | 150km |
| Logistics | Port of LA | California, US | 60km |
| Logistics | Port of Shanghai | China | 80km |
| Logistics | Strait of Hormuz | Middle East | 150km |
| Agriculture | Waikato Dairy | New Zealand | 100km |
| Industrial | Japan Pacific Coast | Japan | 200km |
| Tech | Northern Virginia DCs | Virginia, US | 60km |

## Signal Value
- Earthquake near semiconductor fabs (TSMC, Samsung) → chip supply disruption
- Seismic activity near mines → commodity supply shock
- Nuclear plant proximity → energy market + safety concerns
- Port/logistics disruption → supply chain impact
- Infrastructure cross-reference is the unique edge — everyone sees the earthquake, nobody auto-maps to supply chain impact

## Risks
- USGS data may lag by minutes for small events
- Infrastructure zone definitions are static — need updates as new facilities built
- Equirectangular distance approximation less accurate at high latitudes

---

## Related

- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]
