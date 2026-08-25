---
title: "Spec: Tier 1 Signal Expansion"
tags:
  - doc/spec
---

# Spec: Tier 1 Signal Expansion

## Goal

Convert three high-value data surfaces into real convergence evidence:
1. `internet_infrastructure` → outage, censorship, connectivity signals
2. `power_grid` → demand, pricing spread, fuel mix, forecast deviation signals
3. `defi_flows` → TVL rotation, stablecoin supply, DEX stress signals

Research: `[[convergence_signal_expansion]]`

## Prerequisites

- `InternetInfrastructureTool` currently returns text-only (`ToolResult(output=...)` with no `data=`). Must add structured `data=` dicts to each mode before the extractor can work.
- `PowerGridTool` and `DefiFlowsTool` already return structured `data=` dicts.

## Files Affected

| File | Action |
|------|--------|
| `agent/tools/internet_infrastructure.py` | Modify: add `data=` dict to each mode's ToolResult |
| `agent/convergence/extractors.py` | Modify: replace internet_infrastructure stub, add power_grid + defi_flows extractors |
| `tests/convergence/test_tier1_extractors.py` | Create: edge-case tests for all 3 extractors |

## Implementation Steps

### Step 1: Add `data=` dicts to InternetInfrastructureTool

The tool already builds structured internal dicts (alerts, events, rows, etc.) — it just doesn't pass them through. Add `data=` to each success ToolResult:

**outages mode:**
```python
data={
    "mode": "outages",
    "alerts": alerts,       # list of alert dicts already built
    "events": events,       # list of event dicts already built
    "country": country,
}
```

**censorship mode:**
```python
data={
    "mode": "censorship",
    "country": country,
    "test": test,
    "rows": rows,           # daily breakdown dicts already built
    "trend": trend,         # "rising" | "falling" | "stable"
    "avg_rate": avg_rate,
    "max_rate": max_rate,
}
```

**signals mode:**
```python
data={
    "mode": "signals",
    "country": country,
    "current": current,     # gtr-norm float
    "avg": avg_val,
    "min": min_val,
    "max": max_val,
    "severity": severity,   # "normal" | "warning" | "critical"
    "drops": drops,         # list of drop dicts
    "data_points": len(valid_values),
}
```

**incidents mode:**
```python
data={
    "mode": "incidents",
    "incidents": [           # structured subset of each incident
        {"title": str, "countries": list[str], "start": str}
        ...
    ],
    "country_frequency": dict[str, int],  # from Counter
}
```

### Step 2: Write `_extract_internet_infrastructure` extractor

Replace the stub. Consume the `data=` dict from Step 1. Produce Evidence objects:

| signal_id | Source mode | Category | What it measures |
|-----------|-----------|----------|-----------------|
| `internet.outage.critical_count` | outages | physical_disruption | Number of countries with critical alerts |
| `internet.outage.event_max_score` | outages | physical_disruption | Highest event severity score |
| `internet.outage.event_breadth` | outages | physical_disruption | Number of distinct countries with events |
| `internet.censorship.anomaly_rate` | censorship | geopolitical | Average anomaly rate |
| `internet.censorship.trend_rising` | censorship | geopolitical | 1 if trend is rising, 0 otherwise |
| `internet.censorship.confirmed_total` | censorship | geopolitical | Total confirmed blocks in window |
| `internet.signals.connectivity_level` | signals | physical_disruption | Current gtr-norm value (inverted: lower = worse) |
| `internet.signals.drop_count` | signals | physical_disruption | Number of drops below warning threshold |
| `internet.incidents.active_count` | incidents | geopolitical | Number of ongoing censorship incidents |
| `internet.incidents.country_breadth` | incidents | geopolitical | Number of distinct affected countries |

### Step 3: Write `_extract_power_grid` extractor

Consume existing `data=` dicts. Produce Evidence:

| signal_id | Source mode | Category | What it measures |
|-----------|-----------|----------|-----------------|
| `power_grid.demand.total_peak_mw` | demand | physical_flow | Total system peak demand |
| `power_grid.demand.zone_count` | demand | physical_flow | Number of reporting zones |
| `power_grid.fuel.gas_share_pct` | fuel_mix | physical_flow | Natural gas as % of total generation |
| `power_grid.fuel.renewable_share_pct` | fuel_mix | physical_flow | Wind + solar + hydro as % of total |
| `power_grid.pricing.stressed_zone_count` | pricing | financial_stress | Zones where |DA-RT spread| > $5 |
| `power_grid.pricing.max_spread` | pricing | financial_stress | Largest absolute DA-RT spread |
| `power_grid.pricing.avg_da_lbmp` | pricing | financial_stress | Average day-ahead price across zones |
| `power_grid.forecast.persistent_deviation_count` | forecast | physical_disruption | Zones with |avg_dev| > 3% |
| `power_grid.forecast.max_significant_deviations` | forecast | physical_disruption | Max deviation count across zones |

### Step 4: Write `_extract_defi_flows` extractor

Consume existing `data=` dicts. Produce Evidence:

| signal_id | Source mode | Category | What it measures |
|-----------|-----------|----------|-----------------|
| `defi.tvl.total_usd` | tvl | financial_stress | Total TVL across protocols |
| `defi.tvl.drawdown_breadth` | tvl | financial_stress | Count of protocols with 1d change < -5% |
| `defi.tvl.top_concentration_pct` | tvl | positioning | Top protocol's share of total TVL |
| `defi.stablecoin.total_supply` | stablecoins | financial_stress | Total stablecoin supply |
| `defi.stablecoin.top_share_pct` | stablecoins | positioning | Top stablecoin's share of total |
| `defi.dex.total_volume_24h` | dex_volume | positioning | Total 24h DEX volume |
| `defi.dex.panic_breadth` | dex_volume | financial_stress | Count of DEXes with 1d change > +50% |
| `defi.chain.total_tvl` | chain | financial_stress | Total chain TVL |
| `defi.chain.top_concentration_pct` | chain | positioning | Top chain's share of total |

### Step 5: Edge-case tests

Test all three extractors with:
- None / empty / wrong-type data
- Missing keys, partial dicts
- Zero-length lists
- Modes with no matching data
- Boundary float values (0, NaN, very large)
- Each mode individually
- Multi-mode composite extraction
- Direction and confidence values are sane

## Edge Cases

- Tool returns `data=None` → extractor returns `[]`
- Tool returns `data={}` (no mode key) → extractor returns `[]`
- `alerts`/`events`/`rows`/etc. is not a list → skip gracefully
- Division by zero (e.g., total_tvl=0 for concentration) → return 0.0
- All modes use the existing `_safe_float` helper and defensive patterns

## Testing Plan

All tests go in `tests/convergence/test_tier1_extractors.py`. Must pass before marking a step done. Existing 883 tests must continue passing.

---

## Related

- [[tier1_signal_expansion|Task: Tier1 Signal Expansion]]
