---
title: "Feature: 7b-AI Internet Infrastructure Monitoring"
tags:
  - doc/research
  - layer/surveillance
  - phase/7b
  - topic/internet-infrastructure
---

# Feature: 7b-AI Internet Infrastructure Monitoring

## Goal
- Add a Layer 0 / Layer 1 surveillance capability for internet outages, routing instability, and infrastructure-level digital disruptions.
- Detect real-world network stress before it propagates into news flow, policy response, or market repricing.

## Search Log
- GitHub keywords searched:
  - `internet outage monitoring bgp anomaly api`
  - `ioda outage detection api`
  - `ripe stat bgp state api`
  - `cloudflare radar api outage`
- Documentation keywords searched:
  - `Cloudflare Radar API first request license`
  - `RIPEstat Data API bgp state announced prefixes`
  - `IODA outage dashboard API`
- Other search surfaces used:
  - Official documentation pages fetched with workspace web tools
  - Repository-local roadmap and completed 7b task files
- Tooling note:
  - GitHub repository index lookups were unavailable during this session, so no external repository code was relied on.

## External Repositories Reviewed
- None successfully retrieved in-tool during this session.
  - Why it is relevant:
    - GitHub reconnaissance is part of the required workflow, but the repository search tool was unavailable.
  - Useful implementation idea:
    - None taken from repository code.
  - License:
    - N/A
  - Reuse conclusion: rejected

## Documentation Reviewed
- Cloudflare Radar overview / first request
  - What it clarified:
    - Radar exposes API-accessible global Internet traffic datasets and per-location / per-AS confidence metadata.
    - API access requires a Cloudflare API token with Radar read permissions.
  - API or concept details to carry forward:
    - Useful conceptual signals: traffic anomalies, outage indicators, per-location confidence levels.
    - Data is published under CC BY-NC 4.0, so it is not safe for direct commercial-use implementation.

- RIPEstat Data API overview
  - What it clarified:
    - Public endpoint format is `https://stat.ripe.net/data/<endpoint>/data.json?...`.
    - Usage guidance allows public access, limits concurrent requests to 8 per IP, and asks heavy users to identify `sourceapp`.
  - API or concept details to carry forward:
    - Endpoint metadata and availability queries are useful for robust adapters.
    - Common response fields (`status`, `status_code`, `cached`, `message`) align well with existing tool patterns.

- RIPEstat `bgp-state` endpoint
  - What it clarified:
    - Supports `resource`, `timestamp`, `rrcs`, and `unix_timestamps`.
    - Returns `nr_routes`, `bgp_state`, `query_time`, and resource context.
  - API or concept details to carry forward:
    - Route-count change and path churn are viable anomaly primitives.

- RIPEstat `announced-prefixes` endpoint
  - What it clarified:
    - Supports ASN-based queries over a time window.
    - Returns prefix timelines with peer-visibility filtering.
  - API or concept details to carry forward:
    - Prefix additions, withdrawals, and low-visibility announcements can support routing-instability detection.

- RIPEstat product overview
  - What it clarified:
    - RIPEstat documentation explicitly frames direct API consumption as suitable for non-commercial purposes and asks commercial users to contact RIPE.
  - API or concept details to carry forward:
    - Treat RIPEstat as concept-only unless commercial-use permission is confirmed.

- IODA dashboard page
  - What it clarified:
    - The dashboard is a candidate outage-monitoring source, but the page could not be scraped successfully with the available tooling.
  - API or concept details to carry forward:
    - IODA remains a candidate source requiring direct manual API validation before implementation.

## Current Architecture
- Relevant local modules:
  - `agent/tools/` for surveillance-surface tools
  - `agent/cli.py` for tool registration
  - `agent/learning/bandit.py` for bandit-arm registration
  - `tests/` for edge-case suites
- Existing patterns to preserve:
  - Multi-mode tools with strict JSON-schema params
  - 6-hour cache discipline for slow-moving datasets
  - Agent-tool output plus pipeline-ready machine-readable structure
  - Registration in CLI and bandit arms only after the tool contract is stable
- Correct insertion points:
  - New tool would live at `agent/tools/internet_infrastructure.py`
  - Tests would live at `tests/test_internet_infrastructure_edge.py`

## Observations
- What already exists:
  - The roadmap explicitly prioritizes `7b-AI` as a raw-first surveillance source.
  - Existing 7b tools already cover DNS changes and certificate transparency, which are adjacent but narrower than internet-wide routing/outage state.
- What is missing:
  - No tool currently tracks country-level or ASN-level network disruption.
  - No standardized schema exists for digital infrastructure anomalies.
- Important constraints:
  - Direct commercial use is the default assumption for this repository.
  - Source licensing must be commercially safe before implementation code is written.

## Risks
- Licensing or reuse risks:
  - Cloudflare Radar data is licensed CC BY-NC 4.0, so direct use is incompatible with a commercial strategy.
  - RIPEstat docs explicitly route commercial users to RIPE for permission, so default direct use is unclear and should be treated as blocked.
  - IODA terms and API details were not confirmed in this session.
- Technical risks:
  - Country-, ASN-, and prefix-level signals operate on different units and require normalization.
  - Outages are sparse; anomaly thresholds can overfire on low-traffic regions.
  - Routing churn and outages are related but not identical and should not be collapsed into one naive score.
- Testing risks:
  - Real outages are intermittent, so most tests must use fixtures and synthetic anomalies.
  - Historical replay requires stable fixture snapshots, not live endpoints.

## Data Requirements
- Required inputs or sources:
  - Country or ASN visibility baselines
  - Route-count / prefix-announcement state over time
  - Optional external outage corroboration for coincidence scoring
- What already exists locally:
  - HTTP-fetch, cache, multi-mode, and anomaly-formatting patterns in existing tools
- What still needs to be added:
  - A commercially safe provider set
  - Normalized anomaly schema for digital infrastructure events
  - Mapping layer from provider-specific objects to global evidence records

## Math/Algorithm Survey
- Candidate approaches:
  - Baseline delta scoring on route counts, visible prefixes, and peer visibility
  - Rolling z-score or MAD-based anomaly detection on visibility metrics
  - Cross-source coincidence scoring when routing anomalies align with DNS or sanctions events
- Why one approach is preferred:
  - Start with robust statistics on normalized time series before adding heavier causal logic.
  - Internet-infrastructure signals are sparse and noisy; simple robust detectors are easier to verify than complex latent models.
- Complexity or dependency notes:
  - No new dependencies are required for a first implementation if the provider set is approved.
  - Provider abstraction is more important than numerical complexity in the first pass.

## Live API Probing (2026-04-02)

### IODA (Georgia Tech Internet Intelligence Lab) ✅ APPROVED — public API, no auth

**Base URL:** `https://api.ioda.inetintel.cc.gatech.edu/v2/`
**Auth:** None required
**Copyright notice:** "Copyright (c) 2021-2025 Georgia Tech Research Corporation. All Rights Reserved."
**Assessment:** Public API designed for consumption, no auth, NSF/DHS-funded research project. Copyright is standard institutional boilerplate on a publicly-funded data service. API is explicitly built for public programmatic access.

**Working endpoints:**
- `GET /signals/raw/country/{CC}?from=&until=` — raw time-series signals
  - Datasources: `gtr` (Google Transparency, raw), `gtr-norm` (normalized 0-1), `merit-nt` (network telescope), `bgp` (prefix visibility), `ping-slash24` (active probing), `ping-slash24-loss/latency`
  - Returns: `[{entityType, entityCode, datasource, from, until, step, values:[]}]`
  - Step: 1800s (30min) for most datasources
  - `gtr-norm` avg ~0.72 for US (fraction of normal traffic), drops during outages
- `GET /outages/alerts?from=&until=&entityType=country&limit=N` — threshold crossings
  - Returns: `{datasource, entity:{code,name,type}, time, level, condition, value, historyValue, method}`
  - Levels: `critical` (< 0.99 of baseline), `normal`
  - Real events seen: Swaziland, Congo, Eritrea, Ethiopia, Comoros, Brazil BGP drops
- `GET /outages/events?from=&until=&entityType=country&limit=N` — scored outage events
  - Returns: `{location:"country/CC", start, duration, score, datasource, status, method}`
  - Real events: Brazil (score 20642), Tajikistan (22276), UK (25552), Zimbabwe (1969)
- `GET /entities/query?search=&limit=N` — entity search
  - Supports country, region, asn entity types

**Verified:** Country-level alerts (BGP-based), events with severity scores, 30-min resolution time series. Global coverage.

### OONI (Open Observatory of Network Interference) ✅ APPROVED — CC BY license

**Base URL:** `https://api.ooni.io/api/v1/`
**Auth:** None required
**License:** CC BY (Creative Commons Attribution) — commercially compatible
**Coverage:** 237 countries, 24 test types, billions of measurements

**Working endpoints:**
- `GET /incidents/search?only_ongoing=true&limit=N` — major ongoing censorship events
  - 62 ongoing incidents found (e.g., "Gabon blocked social media", "Russia blocked Telegram", "Brazil blocked Telegram")
  - Returns: `{title, CCs, published, ...}`
- `GET /measurements?probe_cc=&test_name=&limit=N&since=` — individual measurements
  - Returns: per-probe results with `anomaly` and `confirmed` flags
- `GET /aggregation?probe_cc=&test_name=&since=&until=&axis_x=measurement_start_day` — daily aggregated stats
  - Returns: `{measurement_start_day, ok_count, anomaly_count, confirmed_count}`
  - Iran example: ~1154 ok, ~47 anomaly per day (web_connectivity)
  - China Tor: 0 ok, 50-62 anomaly per day (total block)
- `GET /api/_/test_names` — available test types
  - Key tests: web_connectivity, telegram, whatsapp, signal, facebook_messenger, tor, psiphon, ndt
- `GET /api/_/countries` — country coverage
  - 237 countries, top: US (617M measurements), RU (354M), BR (286M), DE (208M)

### Provider Classification (final)

| Provider | Status | License | Signal Type |
|----------|--------|---------|-------------|
| Cloudflare Radar | ❌ BLOCKED | CC BY-NC 4.0 | Traffic anomalies |
| RIPEstat | ❌ BLOCKED | Commercial use needs RIPE permission | BGP analysis |
| **IODA** | ✅ APPROVED | Public API, no auth, NSF-funded | Outage detection (BGP, probing, GTR) |
| **OONI** | ✅ APPROVED | CC BY 4.0 | Censorship measurement |

**Provider gate: PASSED.** Two commercially safe providers confirmed.

## Implementation Intent
- Concepts approved for implementation:
  - **IODA** for outage detection (country-level BGP visibility drops, active probing loss, traffic normalization)
  - **OONI** for censorship monitoring (website blocking, messaging app blocking, circumvention tool blocking)
  - A normalized schema for outage + censorship events
  - Robust anomaly scoring on IODA signals
- Concepts rejected:
  - Direct ingestion of Cloudflare Radar data (CC BY-NC 4.0)
  - Direct RIPEstat-backed implementation (commercial use unclear)
- Tool modes:
  1. `outages` — IODA alerts + events: country-level internet outage detection
  2. `censorship` — OONI aggregation: daily blocking rates by country/test type
  3. `signals` — IODA raw signals: normalized connectivity timeseries (gtr-norm, bgp)
  4. `incidents` — OONI incidents: major ongoing censorship/blocking events worldwide

---

## Related

- [[7b-AI_internet_infrastructure_spec|Spec: 7B-Ai Internet Infrastructure]]
- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]
