---
title: "Feature: Tier 2 Signal Expansion — Evaluating 4 Remaining Stub Extractors"
tags:
  - doc/research
  - layer/world-model
---

# Feature: Tier 2 Signal Expansion — Evaluating 4 Remaining Stub Extractors

## Goal

Decide which (if any) of the 4 remaining stub extractors are worth converting
into real extractors. Quality gate: if a stub doesn't clearly add orthogonal,
high-cadence, quantitative signal to an under-served taxonomy category, kill it.

## Current Category Coverage (46 real extractors)

| Category | Signal count | Assessment |
|---|---:|---|
| financial_stress | 23 | Saturated |
| positioning | 19 | Well-covered |
| physical_disruption | 17 | Well-covered |
| macro_momentum | 14 | Adequate |
| geopolitical | 14 | Adequate |
| physical_flow | 13 | Adequate (power_grid Tier 1 just added here) |
| behavioral_intent | 13 | Adequate |
| regulatory_action | 9 | Thin |
| monetary_policy | 9 | Thin |
| biological | 5 | Thin |
| supply_chain | 4 | **Very thin — biggest gap** |

**Thinnest categories**: supply_chain (4), biological (5), regulatory_action (9), monetary_policy (9).

## The 4 Stubs Under Evaluation

All 4 tools currently return **text-only** (no `data=` dict). Converting any of
them requires two steps: (1) restructure the tool to emit a `data=` dict, then
(2) write the extractor.

---

### 1. `electricity_monitor` — EIA hourly demand/generation/interchange

**Data source**: EIA API v2 `/electricity/rto/` endpoints (requires `TIRRA_EIA_API_KEY`).

**Modes**:
- `demand`: hourly peak/trough/avg MW by balancing authority
- `generation`: fuel mix (MWh + share %) — total, renewable %, fossil %
- `interchange`: inter-BA power flows (exports/imports/net MWh)

**Category mapping**: `physical_flow` (already 13 signals including power_grid Tier 1).

**Evaluation**:

| Dimension | Score | Rationale |
|---|---:|---|
| Cadence/Latency | 5 | Hourly — among fastest physical feeds |
| Causal breadth | 4 | Industrial activity, weather stress, energy cost, grid congestion |
| Orthogonality | **2** | Overlaps heavily with `power_grid` (NYISO) already in Tier 1 — same modality (generation, demand, fuel mix), just wider geographic scope |
| Safety | 4 | Free with API key, stable, well-documented |
| Leverage | 2 | Text-only, needs data= restructure; 3 modes to parse; physical_flow already has 13 signals |
| **Total** | **17/25** | |

**Verdict: MARGINAL.** Adds geographic breadth to an already-covered modality.
Not filling a thin category. The signals it would add (demand MW, fuel mix %,
interchange flows) are the same *kind* of signal as power_grid — just from
different balancing authorities. This is duplication within physical_flow, not
cross-category expansion.

**Recommendation: DEFER.** Not worth the engineering cost now. Revisit when
physical_flow needs multi-BA breadth for the world model.

---

### 2. `satellite_activity` — NASA FIRMS fires, MODIS vegetation, EONET events

**Data sources**:
- NASA FIRMS (NRT thermal hotspots) — real-time
- ORNL MODIS NDVI (vegetation health) — 16-day cycle
- NASA EONET (natural events) — event-driven

**Modes**:
- `fire`: hotspot count, FRP stats, confidence, cluster centroids
- `vegetation`: NDVI time series, health classification, anomaly %
- `events`: natural event list with category/coordinates/date

**Category mapping**: Primary `physical_disruption` (17 signals, well-covered).
Secondary `geopolitical` (14), `supply_chain` (4 — thin!).

**Evaluation**:

| Dimension | Score | Rationale |
|---|---:|---|
| Cadence/Latency | 4 | Fire data is real-time; vegetation is 16-day (slow); events are irregular |
| Causal breadth | 4 | Agricultural disruption → commodity prices → supply chain. Wildfire → infrastructure/insurance. Vegetation anomaly → drought/food stress |
| Orthogonality | **4** | No existing extractor observes physical-world remote sensing. Completely different observation modality from all 46 current extractors |
| Safety | 5 | NASA/ORNL — public domain, no auth for FIRMS/EONET, ORNL open |
| Leverage | 3 | Text-only, needs restructure; fire mode is most structured; vegetation NDVI is cleanly numeric; EONET is event-driven (harder to quantify) |
| **Total** | **20/25** | |

**Key insight**: Satellite data is the **only remote-sensing modality** in the
entire extractor set. Every other extractor observes human-generated data
(filings, prices, surveys, network events). Satellite observes the physical
planet. That orthogonality is real.

**Supply chain gap**: NDVI vegetation anomalies directly signal crop stress →
food prices → supply chain pressure. This could populate the thinnest category
(supply_chain: 4 signals) with a genuinely new observation type.

**Fire/natural-event convergence**: Wildfire hotspot surges near infrastructure
or crop regions → physical_disruption AND supply_chain simultaneously. This is
exactly the cross-category evidence that makes convergence detection powerful.

**Verdict: ACCEPT — HIGH PRIORITY.**

**Recommended signals**:
- `satellite.fire.hotspot_count` (physical_disruption)
- `satellite.fire.frp_total` (physical_disruption)
- `satellite.fire.cluster_count` (physical_disruption)
- `satellite.vegetation.ndvi_anomaly_pct` (supply_chain)
- `satellite.vegetation.health_class` (supply_chain)
- `satellite.events.active_count` (physical_disruption)

---

### 3. `foia_requests` — MuckRock/WDTK FOIA activity

**Data sources**:
- MuckRock API (US FOIA requests)
- WhatDoTheyKnow Alaveteli API (UK FOI requests)

**Modes**:
- `search`: request list (title, agency, status, date)
- `agency_activity`: request count, surge ratio (≥2.0× baseline = surge flag)
- `entity_cluster`: multi-agency convergence flag (≥3 agencies OR ≥2 jurisdictions)

**Category mapping**: Primary `regulatory_action` (9 signals — thin).

**Evaluation**:

| Dimension | Score | Rationale |
|---|---:|---|
| Cadence/Latency | 3 | Near-real-time filing, but signal develops slowly (request clusters take weeks to form) |
| Causal breadth | 2 | Narrow: investigative/regulatory attention → one entity/sector. Not broad macro signal |
| Orthogonality | 3 | Unique observation type (government transparency requests), but signal-to-noise is low — most FOIA requests are routine/journalistic |
| Safety | 4 | Free, open APIs, well-documented |
| Leverage | **1** | Text-only, mostly qualitative text. The "surge ratio" is the only clearly quantitative signal. Entity clusters are count-based but very noisy. Converting text descriptions into reliable numeric evidence is unreliable |
| **Total** | **13/25** | |

**Critical problem**: FOIA requests are overwhelmingly noise. Most requests are
routine journalism, academic research, or citizens requesting personal records.
A "surge" in FOIA requests to an agency frequently indicates a news cycle or
viral social media moment, not genuine regulatory action. The signal-to-noise
ratio is terrible for convergence detection.

**The entity_cluster mode is the most promising** — when multiple agencies in
multiple jurisdictions receive requests about the same entity, that's
potentially meaningful. But parsing this reliably from text descriptions requires
NLP that doesn't belong in a deterministic convergence extractor.

**Verdict: REJECT.**

**Reason**: Too qualitative, too noisy, low causal breadth. The regulatory_action
category is better served by existing extractors (sanctions, drug_regulatory,
gazette) that observe actual government actions, not requests for information
about government actions. FOIA activity is a derivative signal that lags the
events it references.

---

### 4. `interconnection_queue` — EIA generator pipeline

**Data sources**: EIA API v2 `/electricity/operating-generator-capacity/` (requires `TIRRA_EIA_API_KEY`).

**Modes**:
- `queue`: individual generator projects (plant, entity, MW, fuel, state, tech)
- `summary`: aggregated MW by fuel type, project count, top states, status distribution
- `datacenter`: suspected data center projects matched by entity name regex

**Category mapping**: Primary `supply_chain` (4 signals — **thinnest category**).

**Evaluation**:

| Dimension | Score | Rationale |
|---|---:|---|
| Cadence/Latency | **1** | Monthly to quarterly updates at best. Generator queue status changes slowly |
| Causal breadth | 3 | Energy infrastructure buildout → future capacity → supply chain bottlenecks. Datacenter concentration → tech capex signal |
| Orthogonality | 4 | No existing extractor observes energy infrastructure pipeline. Datacenter-tracking mode is unique |
| Safety | 4 | EIA API, free with key, well-documented |
| Leverage | 2 | Text-only, needs restructure. Summary mode is most structured (MW aggregates). Datacenter mode depends on entity-name regex matching (fragile). Queue mode is just a list |
| **Total** | **14/25** | |

**The cadence problem is fatal for convergence detection.** Interconnection
queue data updates monthly at best. The convergence detection system runs daily
and looks for signals that change fast enough to detect emerging patterns. A
signal that updates once a month is essentially static relative to the detection
window.

**However**: The summary aggregates (total MW by fuel, project count by state)
would be useful as **slow structural priors** for the world model, not for
real-time convergence detection.

**Verdict: REJECT for convergence extractors. NOTE for future world model.**

**Reason**: Update cadence too slow for convergence detection. Monthly/quarterly
pipeline data doesn't converge with daily signals — it's background context, not
evidence. The supply_chain category is legitimately thin, but the fix should be
higher-cadence supply chain observations (shipping, commodity flow, trade data),
not infrastructure queue data.

---

## Summary Ranking

| Rank | Tool | Score | Category gap? | Verdict |
|---:|---|---:|---|---|
| 1 | satellite_activity | 20/25 | supply_chain (4 signals — thinnest) | **ACCEPT** |
| 2 | electricity_monitor | 17/25 | physical_flow (13 — adequate) | DEFER |
| 3 | interconnection_queue | 14/25 | supply_chain (4 — thinnest, but cadence too slow) | REJECT |
| 4 | foia_requests | 13/25 | regulatory_action (9 — thin but better served) | REJECT |

## Decision

**Convert 1 of 4 stubs: `satellite_activity` only.**

Rationale:
1. **Only remote-sensing modality** in entire system — genuinely new observation type
2. **Fills thinnest category** — supply_chain (4 signals) via vegetation/crop stress
3. **Cross-category reach** — fire → physical_disruption AND supply_chain simultaneously
4. **High cadence where it matters** — fire data is real-time (NRT), vegetation is 16-day (acceptable for structural)
5. **Perfect operational safety** — NASA/ORNL public domain, no API keys needed for FIRMS/EONET

The other 3 stubs fail on orthogonality (electricity_monitor duplicates power_grid modality),
signal quality (foia_requests too noisy/qualitative), or cadence (interconnection_queue
too slow for convergence detection).

## Implementation Prerequisites

1. Restructure `satellite_activity` tool to return `data=` dicts alongside text
2. Write the extractor with 3 modes (fire, vegetation, events)
3. Register signals in taxonomy with appropriate metadata
4. Edge-case test suite covering: empty API responses, malformed coordinates,
   zero-FRP hotspots, NDVI out of [-1, 1] range, EONET events with no geometry

## Risks

- **NDVI 16-day cadence**: Vegetation signals update slowly. They're structural
  context, not fast-moving convergence triggers. Must set appropriate TTL (≥7 days).
- **EONET event quantification**: Natural events are categorical (type + location).
  Converting to numeric evidence requires a clear schema — count by type/region
  over a time window, not individual event parsing.
- **Fire false positives**: Agricultural burns, gas flares, and volcanic activity
  create FIRMS hotspots. FRP (fire radiative power) filtering and cluster-size
  thresholds are needed to distinguish significant from routine fire activity.

## References

- NASA FIRMS: https://firms.modaps.eosdis.nasa.gov/
- ORNL MODIS NDVI: https://modis.ornl.gov/
- NASA EONET: https://eonet.gsfc.nasa.gov/
- Existing power_grid extractor: `agent/convergence/extractors.py` L3568–3780
- Tier 1 research: `[[convergence_signal_expansion]]`
- Taxonomy: `agent/convergence/taxonomy.py`

## Related

- [[tier2_satellite_spec|Spec: Tier2 Satellite]]
- [[tier2_satellite_activity|Task: Tier2 Satellite Activity]]
- [[convergence_detection]]
- [[convergence_signal_expansion]]
