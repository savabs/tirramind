---
title: "Feature: Daily Context Schema"
tags:
  - doc/research
---

# Feature: Daily Context Schema

## Purpose
- Define a compact, deterministic, machine-readable object that summarizes the global state before downstream scoring, fusion, or execution decisions.
- This is TirraMind's answer to the useful part of GMM + TOP + BTMM without copying Bloomberg UI.
- The object exists to compress context for models, not to serve as a terminal screen.

## Design Goals
- Machine-first, not UI-first
- Deterministic and versioned
- Robust to partial source outages
- Compact enough for daily storage and replay
- Rich enough to support world-model evidence injection

## Top-Level Schema

```json
{
  "schema_version": "1.0",
  "as_of": "2026-03-31T13:30:00Z",
  "window": {
    "start": "2026-03-30T20:00:00Z",
    "end": "2026-03-31T13:30:00Z"
  },
  "sources": {
    "market_data": true,
    "macro_data": true,
    "news_events": true,
    "liquidity_features": true
  },
  "cross_asset": {},
  "rates_liquidity": {},
  "event_clusters": [],
  "regime_flags": {},
  "anomalies": [],
  "lineage": []
}
```

## Required Sections

### 1. Identity and timing
- `schema_version`: explicit compatibility tag
- `as_of`: timestamp the object was finalized
- `window.start` / `window.end`: data interval covered by the compression step

### 2. Source availability
- `sources`: boolean availability map by logical source family
- Purpose: distinguish true calm from missing input data

### 3. Cross-asset summary
- `cross_asset` is a fixed structure of standardized market state features

Suggested fields:

```json
{
  "equities": {
    "spx_return_z": -1.2,
    "ndx_return_z": -1.8,
    "world_breadth": 0.34,
    "dispersion": 0.71
  },
  "rates": {
    "us10y_change_bp": 7.4,
    "ust_2s10s_bp": -41.0,
    "global_dm_yield_median_change_bp": 5.1
  },
  "fx": {
    "dxy_return_z": 1.0,
    "em_fx_stress_score": 0.66
  },
  "commodities": {
    "oil_return_z": 1.4,
    "gold_return_z": -0.3,
    "agri_stress_score": 0.22
  },
  "volatility": {
    "vix_change_z": 2.1,
    "cross_asset_vol_regime": "elevated"
  }
}
```

Notes:
- Prefer normalized or z-scored values over raw returns when possible.
- Keep bucket count fixed so downstream models receive consistent vectors.

### 4. Rates and liquidity state
- `rates_liquidity` captures the macro funding environment that conditions valuation and execution.

Suggested fields:

```json
{
  "policy_state": {
    "fed_upper_bound": 5.50,
    "ecb_deposit_rate": 4.00,
    "boj_policy_rate": 0.10
  },
  "funding": {
    "sofr": 5.31,
    "repo_stress_score": 0.14,
    "usd_funding_tightness": 0.41
  },
  "curve": {
    "ust_3m10y_bp": -97.0,
    "curve_regime": "inverted"
  },
  "liquidity": {
    "credit_spread_stress": 0.29,
    "market_depth_score": 0.61,
    "risk_off_score": 0.58
  }
}
```

### 5. Event clusters
- `event_clusters` is a list of compressed catalyst groups rather than raw headlines.
- Each cluster should represent a distinct event family with evidence count and severity.

Suggested item schema:

```json
{
  "cluster_id": "cbank-2026-03-31-001",
  "type": "central_bank",
  "region": "US",
  "severity": 0.82,
  "novelty": 0.47,
  "evidence_count": 6,
  "affected_assets": ["rates", "usd", "equities"],
  "summary": "hawkish policy guidance cluster",
  "primary_sources": ["macro_data", "news_events"]
}
```

Allowed cluster families:
- `central_bank`
- `geopolitical`
- `supply_disruption`
- `regulatory`
- `corporate_systemic`
- `energy_grid`
- `weather_disaster`
- `market_structure`

### 6. Regime flags
- `regime_flags` should be low-cardinality discrete states, not prose

Suggested fields:

```json
{
  "macro_regime": "tightening",
  "risk_regime": "risk_off",
  "liquidity_regime": "fragile",
  "event_density_regime": "high",
  "execution_regime": "adverse"
}
```

### 7. Anomalies
- `anomalies` is reserved for high-salience deviations worth explicit model attention

Suggested item schema:

```json
{
  "name": "rates_equity_divergence",
  "score": 0.91,
  "direction": "unexpected",
  "description": "equities up despite large upward rates shock"
}
```

### 8. Lineage
- `lineage` records source provenance for replay and auditability

Suggested item schema:

```json
{
  "source": "macro_data",
  "retrieved_at": "2026-03-31T13:20:04Z",
  "version": "tool-v1",
  "coverage": "full"
}
```

## Validation Rules
- `schema_version`, `as_of`, `window`, `sources`, `cross_asset`, `rates_liquidity`, `regime_flags`, and `lineage` are required
- missing values should be represented as `null`, not omitted, inside fixed substructures
- event cluster and anomaly lists may be empty
- all regime flags should be chosen from controlled enumerations
- all timestamps should be UTC ISO-8601

## Why This Is Non-Commodity
- The schema itself is not the moat.
- The moat comes from what gets compressed into it and how that evidence is fused downstream.
- The point is not to create a prettier market dashboard.
- The point is to create a stable evidence object that lets unique data and mathematical layers interact consistently.

## Recommended Future Implementation
- Daily pipeline DAG emits one context object per run
- context object stored in pipeline persistence layer
- downstream world-model code consumes this object as structured evidence
- execution layer can use `execution_regime` to condition expected friction

## Related

- [[project_memory]]
