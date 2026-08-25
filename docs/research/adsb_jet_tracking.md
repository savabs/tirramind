---
title: "Research: ADS-B Jet Tracking (Phase 6f)"
tags:
  - doc/research
  - topic/adsb-flight
---

# Research: ADS-B Jet Tracking (Phase 6f)

## Thesis

Track private/corporate jet movements as leading indicators: M&A activity, executive travel to regulators (SEC/FDA/DOJ), PE firm convergence on acquisition targets, government VIP aircraft anomalies. Physical-world observable (Layer 0) — actions, not disclosures.

## Data Source: OpenSky Network

The only free ADS-B API. Relies on volunteer ground receivers. https://opensky-network.org/api

### What's Available (Free, Unauthenticated)

| Endpoint | Works? | Notes |
|----------|--------|-------|
| `/states/all` (real-time positions) | ✅ | Bbox filter. 10s rate limit. |
| `/metadata/aircraft/icao/{hex}` | ✅ | Registration, model, operator. Good. |
| `/tracks/all?icao24=X` | ✅ | Current flight path. |
| `/flights/departure?airport=X` | ⚠️ | Only last ~2h window. 403 for historical. |
| `/flights/arrival?airport=X` | ⚠️ | Same — 2h window only. |
| `/flights/aircraft?icao24=X` | ❌ | 403 "cannot access historical flights" |
| `/flights/all` | ⚠️ | 2h window. Max 1h per query. |

### Empirical Coverage Assessment (2026-03-25, live probing)

**Global real-time:**
- **4,864 aircraft visible** globally at snapshot time
- Real global traffic: ~100,000-150,000 aircraft in flight
- **Coverage: ~3-5%**

**US:**
- **782 aircraft visible** in United States
- Real US traffic at peak: ~7,000-10,000
- **Coverage: ~8-10%**

**NYC (busiest US airspace):**
- **4 aircraft visible** in the entire NYC metro bounding box
- All 4 were Delta commercial flights. Zero private jets.
- Real NYC traffic: hundreds at any moment

**Teterboro (KTEB — primary NYC private jet airport):**
- **2 departures in 2-hour window**
- Both Pilatus PC-12s (one Quest Diagnostics corporate turboprop)
- Real Teterboro traffic: 30-50 movements/hour
- **Coverage: ~2-3%**

**Global flights API (last hour):**
- **16 flights** visible worldwide in 1 hour
- Only 7 had both departure AND arrival airports identified
- 4 had neither airport identified
- Real global flights/hour: thousands

## Why This Kills the Signal

### 1. Statistics require samples, not noise
At 3-5% coverage, you'd need massive baseline windows just to reach statistical significance. A "3σ anomaly in private jet traffic to Bentonville" requires a baseline where you reliably see N flights/week. At 2-3% coverage of Teterboro, you see 0-2 flights per 2h window. You can't build a distribution from that.

### 2. No historical lookback (free tier)
The flights-by-airport and flights-by-aircraft APIs return 403 for anything beyond ~2 hours. Without historical data you cannot:
- Compute normal baselines
- Detect deviations
- Backtest the signal
- Build distributional features

This is the single most fatal limitation. Anomaly detection without baselines is guessing.

### 3. Airport identification is incomplete
Only 44% of flights (7/16) had both origin AND destination identified. Without destination, there's no "CEO flew to FDA headquarters" signal — just "a plane took off."

### 4. Rate limits prevent persistent monitoring
10 seconds between state requests, 2-hour maximum time windows. To build a surveillance system you'd need persistent polling (every minute for key airports) — which would hit the rate limit in 6 queries.

### 5. The narrative is cooler than the data
Tracking Berkshire's jet is a great retrospective story. In practice:
- Companies use charter services, commercial flights, or shell LLCs for tail numbers
- FAA LADD (Limiting Aircraft Data Displayed) blocks many corporate jets from ADS-B sites
- Even when visible, a jet flying to a city tells you nothing without context about who's on it and what meeting they're going to

## What Would Make This Work (But Costs Money)

| Source | Coverage | Historical | Cost |
|--------|----------|------------|------|
| ADS-B Exchange (JetNet) | ~90% | Full | $100+/month |
| FlightAware Firehose | ~95% | Full | $500+/month |
| Flightradar24 API | ~95% | Full | $500+/month |
| OpenSky Academic | Better | 30 days | Free but requires institutional affiliation/application |

None of these meet our $0-until-proven-edge constraint.

## Comparison to Remaining Phase 6 Tools

| Tool | Data Completeness | Cost | Signal Quality | Effort |
|------|------------------|------|---------------|--------|
| **6f ADS-B** | **3-5%** | $0 | Insufficient | High |
| **6g Power Grid** | **100%** (ISO/RTO) | $0 | High | Medium |
| **6h ClinicalTrials.gov** | **100%** (full DB) | $0 | High | Medium |

Power grid data (ERCOT, PJM, CAISO, MISO) is:
- 100% complete (operator-reported, not sampled)
- Real-time (5-minute intervals)
- Direct proxy for industrial production and economic activity
- Datacenter demand = AI/cloud capex proxy
- Free public APIs from every major ISO/RTO
- Almost nobody in quantitative finance monitors this systematically

ClinicalTrials.gov is:
- 100% complete (FDA mandate to register)
- Phase transition dates = biotech catalyst detection
- Cross-references with Form 144 insider activity near pharma companies
- Structured, queryable, no coverage gaps

## Recommendation

**Skip Phase 6f (ADS-B).** The free data is fundamentally insufficient — 3-5% coverage with no historical baseline makes anomaly detection impossible. This is below the RenTech bar. Building a tool on this data would produce noise dressed as signal.

**Proceed to 6g (Power Grid) or 6h (ClinicalTrials.gov).** Both offer 100% complete data, free, with genuine signal potential that nobody else is systematically harvesting.

If ADS-B data becomes available through a paid source after the system has demonstrated alpha, we can revisit. The architecture (Layer 0 physical observables → feature extraction) doesn't change — only the data feed.

## Related

- [[deep_surveillance_tools]]
- [[tier2_signal_expansion]]
