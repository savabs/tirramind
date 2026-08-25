---
title: "Research: GDELT L2 — Actor Entity Registration"
tags:
  - doc/research
  - phase/10
  - topic/surveillance
  - layer/surveillance
---

# Research: GDELT L2 — Actor Entity Registration

## Current Architecture

`GDELTTool` (`agent/tools/gdelt.py`, ~643 lines) has two modes:

1. **events** — Fetches 15-min GDELT export CSVs, parses 61-column tab-separated rows into structured event dicts. Each event has `actor1` and `actor2` sub-dicts containing `name`, `country`, and `type`. Events are filtered by quad_class, country, goldstein, event_codes, then sorted by impact. Returns events in `data["events"]`.

2. **articles** — Keyword search via GDELT DOC API. Returns article dicts with title, url, tone, domain, etc. No structured actor data — pure text search.

Constructor is `__init__(self, cache: DataCache | None = None)`. No PipelineStore wiring yet. No edge tests exist.

## Entity Model Analysis

### What Are GDELT Actors?

Each GDELT event has two actor slots:
- `actor1` → the initiator (e.g., "UNITED STATES", "RUSSIAN MILITARY")
- `actor2` → the target (e.g., "CHINA", "IRAN GOVERNMENT")

Raw CAMEO fields available per actor:
- `actor_code` — CAMEO code (e.g., "USA", "RUSGOV", "ISRMIL"). Compound: country + type + role.
- `actor_name` — Human label from GDELT's entity extraction (e.g., "UNITED STATES", "VLADIMIR PUTIN")
- `actor_country` — 2-letter FIPS country code
- `actor_type` — CAMEO actor type code (GOV, MIL, BUS, COP, etc.)

### Identity Strategy

The natural entity key should be the **actor country code** for the country-level entity. This is the most stable identifier across GDELT events — actor names vary widely ("United States", "US", "UNITED STATES", etc.) but the FIPS country code is consistent.

**Entity type:** `"country"` — using the FIPS country code as the key. This covers both state and non-state actors since the country code is always present. The actor type (GOV, MIL, BUS) becomes metadata, not a separate entity.

**Why not actor_code (e.g., "RUSGOV") as key?** The actor_code is often blank or inconsistent. Country code is far more reliable and still provides the strategic intelligence value: which countries are in conflict/cooperation dyads. The actor_type enriches the entity as metadata.

**Why not individual person entities?** GDELT actor names are unreliable — entity resolution quality is low. Country-level aggregation is the appropriate L2 grain. Person-level would be L3 (cross-referencing with other sources).

### What Gets Persisted

**events mode:** Both actor1 and actor2 as country entities. Each event generates 1-2 observations (one per actor with a valid country code). Observation stores the dyad: which counterpart, event type, goldstein score, quad class.

**articles mode:** No persistence. Articles have no structured actor data — they're text search results. L1 only.

### Observation Schema

```python
observation_type = "geopolitical_event"
value = {
    "event_id": str,           # GDELT global event ID
    "counterpart_country": str, # the other actor's country code
    "event_root": str,         # CAMEO root code (e.g., "18" = Assault)
    "event_description": str,  # Human label
    "goldstein": float | None, # Impact score (-10 to +10)
    "quad_class": int,         # 1-4 (cooperation→conflict)
    "role": str,               # "initiator" or "target"
    "num_mentions": int,
    "location": str,           # action_geo_country
}
depth_level = 2
```

## Differences from Prior L2 Tools

| Aspect | insider_filings / whale_alert / ais_vessel | gdelt |
|--------|---------------------------------------------|-------|
| Entity granularity | Individual (person, wallet, vessel) | Aggregate (country) |
| Events per entity | 1 entity per filing/tx | 2 entities per event (dyad) |
| Identity stability | CIK / address / IMO: very stable | FIPS country code: very stable |
| Name variation | Moderate | High (ignored — use code only) |
| Observation schema | Domain-specific | Geopolitical dyad observation |
| Articles mode | N/A | L1 only — no structured actors |

## Risks

1. **High event volume** — 1 hour = ~4,800 events. Persisting all would create many observations. Mitigation: only persist events that pass the user's filters (events in the returned `data["events"]` list), not all raw events.
2. **Actor country code blank** — ~10-15% of GDELT events have empty actor codes. Skip these.
3. **Dedup** — Same country appears in many events. Register entity once (by country code), store each event as a separate observation. Use `seen` set for entity registration dedup.
4. **No edge tests exist** — Need to create both L2 tests and basic edge tests.

## Data Requirements

- No new data sources needed. Same GDELT CSV + DOC API.
- Entity type `"country"` already declared in `entity.py` (`EntityType` literal).
- Alias source: `"fips"` for the FIPS country code.

## Related

- [[ais_vessel_l2]]
- [[whale_alert_l2]]
- [[gdelt_l2_spec]]
- [[project_memory]]
