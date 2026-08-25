---
title: "Spec: 7b-AI Internet Infrastructure Monitoring"
tags:
  - doc/spec
  - layer/surveillance
  - phase/7b
  - topic/internet-infrastructure
---

# Spec: 7b-AI Internet Infrastructure Monitoring

## Goal
Create a global internet-infrastructure surveillance tool using two commercially safe providers (IODA + OONI) to detect country-level outages, routing instability, and censorship escalation.

## Providers
- **IODA** (Georgia Tech, public API, no auth) — outage detection via BGP, active probing, Google Transparency
- **OONI** (CC BY 4.0, no auth) — censorship measurement via web_connectivity, messaging, circumvention tests

## Files Affected
- `agent/tools/internet_infrastructure.py` — new tool module
- `agent/cli.py` — tool registration
- `agent/learning/bandit.py` — bandit arm
- `tests/test_internet_infrastructure_edge.py` — edge-case suite

## Tool Modes

### Mode 1: `outages` (IODA)
Country-level internet outage alerts and events.
- Params: `country` (ISO-2, optional — blank = global), `hours_back` (default 24, max 168)
- Sources: IODA `/outages/alerts` + `/outages/events`
- Output: list of outage events with `{country, country_name, datasource, level, score, start_time, duration_minutes, value, baseline}`

### Mode 2: `censorship` (OONI)
Daily censorship/blocking measurement rates by country.
- Params: `country` (ISO-2, required), `test` (web_connectivity|telegram|whatsapp|signal|tor|facebook_messenger, default web_connectivity), `days_back` (default 30, max 90)
- Source: OONI `/aggregation` endpoint
- Output: daily `{date, ok_count, anomaly_count, confirmed_count, anomaly_rate}` + summary statistics + trend (rising/falling/stable)

### Mode 3: `signals` (IODA)
Normalized connectivity timeseries for a country.
- Params: `country` (ISO-2, required), `hours_back` (default 24, max 168)
- Source: IODA `/signals/raw/country/{CC}` with `datasource=gtr-norm`
- Output: timeseries of normalized values (0-1 scale where 1=normal), min/max/avg, alert if any point < 0.8

### Mode 4: `incidents` (OONI)
Major ongoing censorship/blocking events worldwide.
- Params: `limit` (default 20, max 100)
- Source: OONI `/incidents/search?only_ongoing=true`
- Output: list of `{title, countries, published}` events

## Normalized Evidence Schema
```
{
  "country": "CC",           # ISO-2
  "country_name": "...",     # human-readable
  "signal_type": "outage|censorship|connectivity_drop",
  "severity": "critical|warning|normal",
  "datasource": "bgp|ping|gtr|ooni_web|ooni_tor|...",
  "timestamp": 1234567890,   # unix epoch
  "value": 0.72,             # current measurement
  "baseline": 1.0,           # expected normal
  "confidence": "high|medium|low",
  "provider": "ioda|ooni"
}
```

## Cache TTLs
- IODA alerts/events: 600s (outages are time-critical)
- IODA signals: 1800s (30min resolution data)
- OONI aggregation: 3600s (daily data, non-urgent)
- OONI incidents: 3600s

## Implementation Steps
1. Implement `InternetInfrastructureTool` skeleton with parameter validation + mode routing
2. Implement `_execute_outages()` — IODA alerts + events fetch/merge
3. Implement `_execute_censorship()` — OONI aggregation fetch + anomaly rate + trend
4. Implement `_execute_signals()` — IODA gtr-norm timeseries + scoring
5. Implement `_execute_incidents()` — OONI incidents fetch
6. Register tool in cli.py, add `internet_infrastructure` bandit arm
7. Write exhaustive edge-case test suite + live smoke tests

---

## Related

- [[7b-AI_internet_infrastructure|Research: 7B-Ai Internet Infrastructure]]
- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]
