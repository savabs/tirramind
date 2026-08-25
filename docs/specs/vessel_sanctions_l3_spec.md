---
title: "Spec: Vessel × Sanctions L3 Pattern (Phase 11c)"
tags:
  - doc/spec
  - phase/11
  - topic/surveillance
  - topic/convergence
  - layer/feature-engineering
---

# Spec: Vessel × Sanctions L3 Pattern (Phase 11c)

## Goal

Detect temporal co-occurrences between vessel movements and GDELT sanctions
events in the same country. Ships stopping or continuing to visit sanctioned
countries is T0 physical intelligence.

## Files Affected

| File | Action |
|------|--------|
| `agent/pipeline/cross_entity.py` | **Modify** — add port→country mapping, `seed_vessel_country_links()`, `detect_vessel_sanctions()` |
| `tests/test_cross_entity.py` | **Modify** — add Phase 11c test classes |

## Implementation Steps

### Step 11c.1: Port → Country Mapping Utility

Add to `cross_entity.py`:

1. `ISO_TO_FIPS` dict — ISO 3166 alpha-2 → FIPS 10-4 for countries where they differ.
2. `BALTIC_PORT_TO_FIPS` dict — known Baltic port names → FIPS codes.
3. `resolve_port_country(port_name: str) -> str | None` — tries:
   a. UN LOCODE prefix (first 2 chars if uppercase alpha) → ISO → FIPS
   b. BALTIC_PORT_TO_FIPS lookup (case-insensitive substring)
   c. Returns None if unresolvable.

### Step 11c.2: `seed_vessel_country_links()` Seeder

Scan existing `port_call` observations for all vessel entities:
1. For each vessel entity, query its port_call observations.
2. Extract port names from `value.port`, `value.prev_port`, `value.next_port`.
3. Resolve each port name to FIPS country code via `resolve_port_country()`.
4. Register country entity if not present.
5. Create `port_call_to` link from vessel → country.
6. Return count of new links created.

### Step 11c.3: `detect_vessel_sanctions()` Detector

Add to `CrossEntityDetector`:
1. For a given vessel entity, find `port_call_to` links → country entities.
2. For each linked country, query co-occurrences between the vessel's
   observations and the country's geopolitical_event observations.
3. Filter GDELT events to sanctions-relevant: `event_root_code in {"16", "17"}`
   OR `quad_class == 4`.
4. Score using the same formula as Insider×GDELT (severity × proximity).
5. Return pattern dicts with `pattern_type="vessel_x_sanctions"`.

Parameters:
- `vessel_entity_id: str`
- `window_seconds: float = 48 * 3600` (tighter than Insider×GDELT)
- `sanctions_root_codes: set[str] = {"16", "17"}`
- `goldstein_threshold: float = -2.0`
- `since: float | None`
- `limit: int = 200`

### Step 11c.4: Edge Case Test Suite

Add test classes to `tests/test_cross_entity.py`:

| Class | Tests | Coverage |
|-------|-------|----------|
| `TestResolvePortCountry` | 8 | UN LOCODE, Baltic names, unknown, empty, case-insensitive |
| `TestSeedVesselCountryLinks` | 5 | Basic seeding, no obs, idempotent, multiple ports, unresolvable |
| `TestDetectVesselSanctions` | 8 | Basic hit, no links, no obs, root code filter, goldstein filter, multiple, scoring, since filter |

~21 new tests.

## Edge Cases

1. Port name is empty or None → skip, no error
2. Port name matches no known country → skip
3. Vessel has no port_call observations → no links, no patterns
4. GDELT event has no event_root_code field → skip
5. Same vessel visits multiple countries → separate links and checks per country
6. Self-link prevention: already handled by store.link_entities validation
7. Idempotent seeding: INSERT OR IGNORE in link_entities

## Related

- [[vessel_sanctions_l3]]
- [[cross_entity_l3]]
- [[cross_entity_l3_spec]]
- [[ais_vessel_l2]]
