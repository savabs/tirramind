---
title: "Research: 7b-D AIS Vessel Tracking"
tags:
  - doc/research
  - layer/surveillance
  - phase/7b
  - topic/vessel-tracking
---

# Research: 7b-D AIS Vessel Tracking

## API Probe Results (2026-03-27)

### Probed 6 AIS APIs:
1. **MarineTraffic** — 401; paid only
2. **AISHub** — 200 but empty; needs community registration
3. **BarentsWatch (Norway)** — 401; needs free registration
4. **NOAA MarineCadastre** — 301 redirect; historical bulk data only
5. **Datalastic** — 401; paid/freemium
6. **Finland Digitraffic** — **200 SUCCESS**, zero auth, rich data

### Finland Digitraffic API — Deep Probe

**Base URL:** `https://meri.digitraffic.fi/api/ais/v1`

**Endpoints:**
- `GET /locations` — all vessel positions, GeoJSON FeatureCollection (7.6MB, 18,204 vessels)
- `GET /vessels` — all vessel metadata (6MB, 18,225 vessels)
- `GET /vessels/{mmsi}` — single vessel metadata
- `GET /locations/{mmsi}` — 404 (NOT SUPPORTED — must use bulk /locations)
- `GET /api/port-call/v1/port-calls?from=ISO_DATE` — Finnish port calls (565/day)

**Coverage:** Baltic Sea + Northern Europe
- Lat: 54.95° — 65.13°
- Lon: 11.59° — 37.40°
- Covers: Finland, Sweden, Estonia, Latvia, Lithuania, Poland, Denmark, Norway, NW Russia

**Ship Types:**
- Tankers (80-89): 5,193
- Cargo (70-79): 9,406
- Passenger (60-69): 659
- Fishing (30-39): 846
- Tugs (50-59): 1,088
- Other: 1,033

**Top Destinations (ships currently in Baltic heading to):**
- Suez/Port Said: ~580+ ships — global trade chokepoint signal
- Russia (St. Petersburg): ~280 ships — sanctions monitoring
- Rotterdam/Gothenburg/Riga/Klaipeda — major EU ports
- Danish Straits (Skagen/Skaw): ~870 ships — Baltic exit point

**Rate Limits:** None detected. No rate limit headers. Two rapid requests succeed in 0.21s.

**Port Call Data:** Rich — port ID, prev/next port, cargo status, vessel name, IMO, timestamps.

## Key Insight

The Baltic API captures **destination intent** for ships heading globally. We don't need to track ships AT Suez — we detect changes in how many ships are heading TOWARD Suez from the Baltic. This is predictive: destination changes show up before the ship arrives.

Signal types extractable:
1. Suez-bound traffic volume → global trade health
2. Russia-bound traffic → sanctions impact
3. Tanker density changes → energy market signals
4. Destination distribution shifts → early warning of trade route changes
5. Port call frequency → Nordic economic activity

## Design Decisions

- **Backend:** Digitraffic only (extensible to AISHub/BarentsWatch later)
- **Caching:** Locations 5min; Metadata 6hr; Port calls 1hr
- **No per-vessel location endpoint** — must fetch all 18K and filter client-side
- **Modes:** area, vessel, port_calls, destination_flow

---

## Related

- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]
