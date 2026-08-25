---
title: "Research: Transport Throughput (7b-R)"
tags:
  - doc/research
  - layer/surveillance
  - phase/7b
---

# Research: Transport Throughput (7b-R)

**Date:** 2026-03-28 (retroactive documentation)
**Tool:** `agent/tools/transport_throughput.py` → `TransportThroughputTool`
**Status:** IMPLEMENTED, TESTED — **US-ONLY, NEEDS EXPANSION**

## APIs Probed

### BTS Border Crossings (Socrata) ✅ SELECTED
- **URL:** `https://data.transportation.gov/resource/keg4-3bc2.json`
- **Method:** GET (SoQL queries)
- **Auth:** None
- **Format:** JSON
- **Rate limits:** Socrata default (1000 req/rolling hour, or unlimited with app token)
- **Coverage:** US-Canada + US-Mexico land border crossings only
- **Data depth:** 333K+ records since 1996. Monthly granularity.
- **Measures:** Trucks, Trains, Rail Containers Loaded/Empty, Personal Vehicles, Buses, Pedestrians, Passengers

### APIs Probed But Failed
| Source | Status | Reason |
|--------|--------|--------|
| TSA Checkpoint | ❌ | DNS resolution failure |
| Eurocontrol | ❌ | HTTP 404 |
| Port of LA | ❌ | SSL certificate error |
| BTS Airline On-Time | ❌ | HTTP 404 |
| USACE Lock Performance | ❌ | HTTP 404 |
| AAR Rail Traffic | ❌ | HTML only, no API |

### International Sources — NOT YET PROBED
| Source | Coverage | URL | Status |
|--------|----------|-----|--------|
| UN Comtrade | Global bilateral trade | `https://comtradeapi.un.org/` | **NOT PROBED** — planned as 7b-Y |
| Eurostat Transport | EU 27 countries | `https://ec.europa.eu/eurostat/api/` | **NOT PROBED** |
| Shanghai/Singapore Port | Asia major ports | Various | **NOT PROBED** |
| Panama Canal Authority | Global shipping | `https://www.pancanal.com/` | **NOT PROBED** |
| Suez Canal Authority | Global shipping | | **NOT PROBED** |

## Geographic Coverage
- Currently: **US land borders only** — `[G:US-ONLY]` `[G:NEEDS-EXPANSION]`
- Target: + Eurostat (EU), Asian port throughput, canal transit data

## Modes Implemented
1. `recent` — latest month aggregate by border + measure
2. `trend` — monthly time series with MoM % change
3. `port` — port-level detail by state
4. `compare` — US-Canada vs US-Mexico side-by-side with ratio

## Signal Value
- Truck volume = real-time trade proxy (precedes GDP reports)
- Rail containers loaded/empty ratio = trade balance direction
- Cross-border divergence (Canada vs Mexico) = NAFTA/USMCA health indicator
- Month-over-month trends = leading economic indicator

## Globalization Priority: MEDIUM
- UN Comtrade (7b-Y) covers global bilateral trade — separate tool planned
- Eurostat transport would add EU dimension
- Asian port data (Shanghai, Singapore, Busan) would complete the picture

## Risks
- BTS data is monthly — relatively slow for real-time signal
- Socrata API occasionally returns stale data
- International port APIs are fragmented (no single API for all ports)

---

## Related

- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]
