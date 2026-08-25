---
title: "Spec: Satellite Activity Convergence Integration"
tags:
  - doc/spec
  - topic/satellite
---

# Spec: Satellite Activity Convergence Integration

## Goal

Convert the `satellite_activity` tool from text-only output to structured
`data=` dicts, and write a convergence extractor that produces `Evidence`
objects for fire hotspots, vegetation stress, and natural events.

This fills the two thinnest taxonomy categories:
- `supply_chain` (currently 4 signals) via vegetation/crop stress
- `physical_disruption` (17 signals, but no remote-sensing modality)

## Files Affected

| File | Action |
|------|--------|
| `agent/tools/satellite_activity.py` | Modify — add `data=` dicts to all 3 success paths |
| `agent/convergence/extractors.py` | Modify — replace stub with real extractor |
| `tests/test_satellite_extractor.py` | Create — comprehensive edge-case test suite |

## Implementation Steps

### Step 1: Add `data=` dicts to `satellite_activity.py`

Each mode's success `ToolResult` gets a `data=` dict containing the computed
aggregates already available in the tool. No new fetching or computation — just
surface the values that are currently formatted into text strings.

**Fire mode** `data=` schema:
```python
{
    "mode": "fire",
    "area": str,                  # country code or bbox
    "source": str,                # FIRMS source name
    "days": int,                  # lookback days
    "hotspot_count": int,         # len(hotspots)
    "frp_avg": float,             # avg FRP (MW), 0.0 if no valid FRP
    "frp_max": float,             # max FRP (MW)
    "frp_total": float,           # sum of all FRP
    "confidence_counts": dict,    # {"high": N, "nominal": N, ...}
    "daynight_counts": dict,      # {"D": N, "N": N}
    "cluster_count": int,         # len(clusters)
    "clusters": list[dict],       # top 10 clusters (already computed)
}
```

**Vegetation mode** `data=` schema:
```python
{
    "mode": "vegetation",
    "latitude": float,
    "longitude": float,
    "start_date": str,
    "end_date": str,
    "observation_count": int,
    "latest_ndvi": float,
    "latest_date": str,
    "latest_health": str,
    "avg_ndvi": float,
    "min_ndvi": float,
    "max_ndvi": float,
    "anomaly_pct": float,         # latest vs historical mean
    "series": list[dict],         # [{date, ndvi, health, pixels}, ...]
}
```

**Events mode** `data=` schema:
```python
{
    "mode": "events",
    "days": int,
    "status": str,
    "category_filter": str | None,
    "event_count": int,
    "category_counts": dict,      # {"wildfires": N, "volcanoes": N, ...}
    "events": list[dict],         # [{title, categories, lat, lon, date}, ...]
}
```

### Step 2: Write convergence extractor in `extractors.py`

Replace the `_stub_extractor` registration for `satellite_activity` with
`_extract_satellite_activity`. The extractor dispatches on `data["mode"]`.

**Signals emitted per mode:**

Fire mode (category=`physical_disruption`):
- `satellite.fire.hotspot_count` — total detections (direction=1 if >100)
- `satellite.fire.frp_total` — aggregate fire radiative power MW
- `satellite.fire.frp_max` — peak single-point FRP
- `satellite.fire.cluster_count` — distinct spatial clusters

Vegetation mode:
- `satellite.vegetation.ndvi_latest` (category=`supply_chain`) — current NDVI
- `satellite.vegetation.anomaly_pct` (category=`supply_chain`) — % deviation from historical mean; negative = crop stress
- `satellite.vegetation.health_class_ordinal` (category=`supply_chain`) — 0-5 ordinal from health classification

Events mode (category=`physical_disruption`):
- `satellite.events.active_count` — total active natural events
- `satellite.events.wildfire_count` — wildfires specifically (highest supply chain impact)
- `satellite.events.severe_storm_count` — storms (shipping/logistics disruption)

Total: 10 new signals (4 physical_disruption + 3 supply_chain + 3 event-driven physical_disruption).

### Step 3: Edge-case test suite

Cover:
- Empty data / not-a-dict → []
- Unknown mode → []
- Fire with 0 hotspots → still emits count=0 signals
- Fire with NaN/None FRP values → safe defaults
- Fire with no clusters → cluster_count=0
- Vegetation with invalid NDVI (>1.0, <-1.0) → clamped health ordinal
- Vegetation with anomaly_pct=0 (avg=0 edge case) → no division by zero
- Vegetation with empty series → []
- Events with 0 events → count=0 signals
- Events with no geometry → still counted
- Events with mixed categories → correct per-category counts
- Confidence/direction values are in valid ranges
- TTL values match expected cadences
- Tags are tuples of strings
- All signal_ids follow naming convention

## Edge Cases

- FIRMS returns hotspots with `frp=0` or `frp=""` → `_safe_float` handles, treated as 0.0
- NDVI anomaly when `avg_ndvi == 0.0` → avoid division by zero, emit anomaly=0.0
- EONET events with empty geometry array → emit count but skip coordinate extraction
- Health class ordinal mapping must handle NDVI outside [-0.2, 1.0] gracefully
- All `data=` dicts must be non-None only on success paths

## Testing Plan

Single test file `tests/test_satellite_extractor.py` with:
1. Parametrized tests for each mode's happy path
2. Parametrized tests for each mode's edge cases
3. Integration test: tool data dict → extractor → Evidence list validation
4. Property checks: all Evidence objects have valid category, non-empty signal_id, numeric value

## Related

- [[tier2_signal_expansion|Research: Tier2 Signal Expansion]]
- [[tier2_satellite_activity|Task: Tier2 Satellite Activity]]
