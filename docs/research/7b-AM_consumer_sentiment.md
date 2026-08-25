---
title: "Research: 7b-AM — Consumer Confidence & Sentiment Surveys"
tags:
  - doc/research
  - layer/surveillance
  - phase/7b
  - topic/consumer-sentiment
---

# Research: 7b-AM — Consumer Confidence & Sentiment Surveys

**Date reviewed:** 2026-04-02
**Layer:** Layer 1 — forward-looking committed survey data

## Current Coverage Gap

`global_pmi.py` already provides OECD Consumer Confidence Index (CCI) for 40+
countries via SDMX. However it only returns a single amplitude-adjusted index
per country. The incremental value of a dedicated consumer sentiment tool:

1. **Eurostat granular breakdown** — financial situation, economic situation,
   major purchases, saving intentions — by EU country
2. **UMichigan survey** — US-specific: headline sentiment, 1yr/5yr inflation
   expectations, buying conditions (requires FRED API key)
3. **BLS CPI** — actual consumer price index (free, no auth) — not sentiment
   but the reality check against expectations
4. **Cross-source divergence detection** — OECD CCI vs Eurostat vs UMich vs
   actual CPI = where perception diverges from reality

## Endpoints Probed

### Eurostat Consumer Confidence (ei_bsco_m) ✅ WINNER

**URL:** `https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/ei_bsco_m`
**Auth:** None required
**Format:** JSON-stat 2.0
**Rate limit:** Unknown, appears generous
**Coverage:** EU27 + individual countries (DE, FR, IT, ES, NL, BE, AT, etc.)
**Frequency:** Monthly
**Indicators:**
- `BS-CSMCI` — Consumer confidence indicator (composite)
- Additional sub-indicators available via different `indic` values

**Verified query:**
```
?s_adj=SA&indic=BS-CSMCI&geo=DE&geo=FR&geo=IT&geo=ES&geo=EU27_2020&lastTimePeriod=3
```
Returns multi-country confidence values (balance %), e.g.: EU27 -11.8, DE -9.9, FR -15.6, IT -1.7, ES -15.6 (2026-03).

**Business Surveys also available:**
- `ei_bssi_m_r2` — Industrial confidence
- `ei_bsrt_m_r2` — Retail confidence  
- `ei_bsse_m_r2` — Services confidence
- `ei_bsbu_m_r2` — Building confidence

### FRED (UMichigan Consumer Sentiment) — needs API key

**Series available:**
- `UMCSENT` — UMichigan Consumer Sentiment Index (headline)
- `MICH` — UMichigan 1-year inflation expectations
- `UMCSENT5` — UMichigan 5-year inflation expectations
- `CSCICP03USM665S` — OECD Consumer Confidence US (duplicates global_pmi)

**Status:** TIRRA_FRED_API_KEY configured as placeholder (`your-key-here`).
Tool should gracefully degrade: FRED modes return "requires API key" message.

### BLS CPI ✅ FREE, NO AUTH

**URL:** `https://api.bls.gov/publicAPI/v2/timeseries/data/` (POST)
**Series:**
- `CUUR0000SA0` — CPI All Urban Consumers (not seasonally adjusted)
- `CUSR0000SA0` — CPI All Urban Consumers (seasonally adjusted)

**Verified:** Returns monthly data, latest 2026-M02 value 326.785.
**Rate limit:** 25 queries/day without key, 500/day with BLS key.

### OECD SDMX — already covered by global_pmi.py

CCI is available via `DSD_STES@DF_CCI` dataflow but the dimension structure is
complex (9 dimensions). `global_pmi.py` handles this via CSV download.
**No need to duplicate in this tool.**

## Architecture Decision

Build `consumer_sentiment.py` with 3 modes:
1. **`eu_confidence`** — Eurostat ei_bsco_m consumer + optional business surveys,
   multi-country. Free, no auth.
2. **`us_sentiment`** — FRED UMichigan (UMCSENT, MICH inflation expectations).
   Requires FRED key → graceful degradation.
3. **`inflation_reality`** — BLS CPI actual vs survey expectations.
   Cross-references CPI changes against UMich inflation expectations.
   Free, no auth (BLS).

Signal value:
- EU confidence dropping while US sentiment stable → divergence = FX signal
- UMich inflation expectations unanchored (>4%) while CPI falling → CB credibility signal
- All-country confidence plunging simultaneously → synchronized recession signal

## Risks

- BLS rate limit (25/day) — cache aggressively (6hr TTL, monthly data)
- FRED key not configured — us_sentiment mode returns helpful error
- Eurostat occasionally returns 202/SPA for some datasets — ei_bsco_m is confirmed working

---

## Related

- [[7b-AM_consumer_sentiment_spec|Spec: 7B-Am Consumer Sentiment]]
- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]
