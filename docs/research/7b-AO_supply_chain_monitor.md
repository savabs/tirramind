---
title: "Research: 7b-AO — Supply Chain Price Pressure Monitor"
tags:
  - doc/research
  - layer/surveillance
  - phase/7b
  - topic/supply-chain
---

# Research: 7b-AO — Supply Chain Price Pressure Monitor

**Date reviewed:** 2026-04-02
**Layer:** Layer 1 — committed producer pricing (L0-L1 mix)

## Original Scope vs. Reality

The original 7b-AO envisioned *commerce inventory monitoring* — tracking
stockouts, lead times, seller counts, and availability on distributor sites
(DigiKey, Mouser, RS Components, Octopart). After live probing:

| Source | Status | Notes |
|---|---|---|
| Octopart/Nexar | 404 | Requires API key (GraphQL, paid) |
| DigiKey API | Auth-gated | OAuth2, requires registered app |
| Mouser API | Auth-gated | API key required |
| RS Components | No API | HTML only |
| SIA (semiconductor sales) | 404 | No public data endpoint |
| Census M3 (Manufacturing) | 204 No Content | API works but returns empty for recent periods |
| Census MARTS (Retail) | 204 No Content | Same — no data for queried period |
| Census MTIS (Inventories) | 204 No Content | Same |

**No free, no-auth APIs exist for real-time hardware availability monitoring.**

## Pivot: BLS Producer Price Index + Import Prices

The strongest free signal for supply chain pressure comes from the Bureau of
Labor Statistics:

### BLS PPI (Producer Price Index) ✅ FREE, NO AUTH

**URL:** `https://api.bls.gov/publicAPI/v2/timeseries/data/` (POST)
**Rate limit:** 25 queries/day (no key), 500/day (with key), max 50 series/request

**Verified working series (with data through 2026-M02):**

| Series ID | Description | Latest Value | YoY Change |
|---|---|---|---|
| PCU334413334413 | Semiconductor & electronic components | 30.228 | +0.3% |
| PCU334111334111 | Electronic computers | 100.225 | +1.0% |
| PCU333120333120 | Construction machinery & equipment | 365.974 | +5.0% |
| PCU331110331110 | Iron & steel mills | 283.745 | — |
| PCU324110324110 | Petroleum refineries | 300.458 | — |
| PCU325130325130 | Synthetic dyes & pigments | 140.586 | — |
| EIUIR | All imports price index | 144.0 | — |

**Series that returned no data (possibly discontinued/renamed):**
PCU334200334200 (comms equipment), PCU335911335911 (storage batteries),
PCU336111336111 (automobiles), PCU212234212234 (copper ore)

### Signal Theory

Producer prices are upstream of consumer prices. When PPI accelerates:
- **Semiconductors PPI rising** → component cost pressure → tech margins squeeze
- **Steel PPI spiking** → construction/auto cost pressure
- **Petroleum PPI jumping** → energy input cost → everything gets more expensive
- **Cross-sector PPI acceleration** → broad-based inflation forming
- **Import prices rising faster than domestic PPI** → tariff/FX pressure
- **PPI-CPI spread widening** → margin compression (producers can't pass through)

### Additional Series Worth Tracking

| Series ID | Description | Signal |
|---|---|---|
| PCU327310327310 | Cement manufacturing | Construction demand |
| PCU322130322130 | Paperboard mills | Packaging/e-commerce proxy |
| PCU325110325110 | Petrochemicals | Chemical feedstock costs |
| WPU10170901 | Electrical power | Energy input costs |
| EIUCOMP | Import price: computers & peripherals | Tech supply chain |

## Architecture Decision

Build `supply_chain_monitor.py` with 3 modes:
1. **`producer_prices`** — BLS PPI for tracked sectors (semiconductors, computers,
   steel, machinery, petroleum, chemicals). Monthly series with MoM/YoY/trend.
2. **`import_prices`** — BLS International Price Program. Import price index
   overall + key categories. Tariff/FX pressure signal.
3. **`pressure_index`** — Cross-sector dashboard: which sectors are accelerating,
   which are deflating. PPI-CPI spread computation. Broad-based vs concentrated
   price pressure classification.

All modes: BLS only, free, no auth, 25 req/day (cache aggressively).

## Risks

- BLS rate limit (25/day without key) — batch series into single requests (max 50),
  cache 6+ hours (monthly data)
- Some series return empty — handle gracefully, flag as discontinued
- BLS 2-year window per request — for longer history, make multiple requests

---

## Related

- [[7b-AO_supply_chain_monitor_spec|Spec: 7B-Ao Supply Chain Monitor]]
- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]
