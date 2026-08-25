---
title: "Spec: 7b-AM — Consumer Sentiment Monitor"
tags:
  - doc/spec
  - layer/surveillance
  - phase/7b
  - topic/consumer-sentiment
---

# Spec: 7b-AM — Consumer Sentiment Monitor

## Goal

Track consumer confidence and inflation expectations across US + EU with
divergence detection. Complement (not duplicate) the existing `global_pmi.py`
CCI mode which covers OECD-level aggregates.

## Files Affected

| File | Action |
|---|---|
| `agent/tools/consumer_sentiment.py` | CREATE — new tool |
| `agent/cli.py` | MODIFY — register tool |
| `agent/learning/bandit.py` | MODIFY — add bandit arm |
| `tests/test_consumer_sentiment_edge.py` | CREATE — edge case tests |

## Implementation Steps

### 7b-AM.1: Create tool skeleton with param validation
- Class `ConsumerSentimentTool(Tool)`
- name: `consumer_sentiment`
- modes: `eu_confidence`, `us_sentiment`, `inflation_reality`
- Parameters: `mode` (required), `countries` (for EU, default "EU27_2020,DE,FR,IT,ES"),
  `months` (int, default 6, max 24)
- Validate mode, clamp months
- **Verification:** test metadata, parameter validation, invalid mode rejection

### 7b-AM.2: Implement eu_confidence mode
- Source: Eurostat `ei_bsco_m` (JSON-stat 2.0)
- URL: `https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/ei_bsco_m`
- Params: `s_adj=SA`, `indic=BS-CSMCI`, `geo=<countries>`, `lastTimePeriod=<months>`
- Parse JSON-stat dimension/value structure → per-country time series
- Compute: MoM change, 3-month trend (improving/deteriorating/stable),
  cross-country divergence (max-min spread), EU27 vs country deviation
- Cache TTL: 21600s (6hr — monthly data)
- **Verification:** fixture-backed tests for normal, empty, malformed payloads

### 7b-AM.3: Implement us_sentiment mode
- Source: FRED API (requires TIRRA_FRED_API_KEY)
- Series: UMCSENT (headline), MICH (1yr inflation exp)
- URL: `https://api.stlouisfed.org/fred/series/observations`
- Params: `series_id`, `api_key`, `file_type=json`, `sort_order=desc`, `limit=<months>`
- If no FRED key: return `ToolResult(success=False, output="Requires TIRRA_FRED_API_KEY...")`
- Compute: current vs 12-month average, directional trend, inflation expectations anchor check
  (>4% = unanchored warning)
- Cache TTL: 21600s
- **Verification:** mock FRED responses + graceful degradation without key

### 7b-AM.4: Implement inflation_reality mode
- Source: BLS CPI (POST, free, no auth)
- Series: CUUR0000SA0 (CPI-U all items NSA), CUSR0000SA0 (SA)
- URL: `https://api.bls.gov/publicAPI/v2/timeseries/data/` (POST)
- Compute: latest CPI, MoM %, annualized rate, YoY %
- If FRED key available: cross-reference CPI actual MoM vs MICH inflation expectations →
  expectation_gap signal (expectations above/below/aligned with reality)
- Cache TTL: 21600s
- **Verification:** mock BLS responses, CPI math edge cases

### 7b-AM.5: Register in cli.py + bandit arm
- Register `ConsumerSentimentTool` in cli.py
- Add bandit arm: `consumer_sentiment` (tools: consumer_sentiment, global_pmi, macro_data)
- **Verification:** registration count assertion, bandit arm exists

### 7b-AM.6: Edge case test suite
- Input validation: invalid mode, bad countries string, months out of range
- Eurostat: empty data, missing countries, malformed JSON-stat, HTTP errors
- FRED: missing key, invalid key, empty response, series not found
- BLS: rate limited (429), empty series, malformed JSON
- Math: zero-variance series, single data point, all-NaN values
- Cache: hit/miss/empty/stale
- **Verification:** comprehensive test file, all pass

## Edge Cases

- Eurostat API returns 400 for unknown geo codes → skip unknown + warn
- BLS 25 req/day limit → aggressive caching, batch series
- FRED key placeholder (`your-key-here`) → treat as missing
- JSON-stat value map has gaps (missing months) → interpolate or mark NaN
- Division by zero in MoM computation → guard with fallback

---

## Related

- [[7b-AM_consumer_sentiment|Research: 7B-Am Consumer Sentiment]]
- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]
