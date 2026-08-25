---
title: "Spec: L3 Cross-Entity Infrastructure + First Pattern"
tags:
  - doc/spec
  - phase/11
  - topic/surveillance
  - topic/convergence
  - layer/feature-engineering
---

# Spec: L3 Cross-Entity Infrastructure + First Pattern

## Goal

Build the entity-linking infrastructure that L3 patterns require, then implement the first cross-domain pattern (Insider × GDELT) as proof-of-concept to validate whether cross-entity co-occurrences actually contain hidden signal.

## Files Affected

### Phase 11a: Infrastructure

| File | Action |
|------|--------|
| `agent/pipeline/store.py` | **Modify** — add `entity_links` table schema, `link_entities()`, `query_entity_links()`, `query_co_occurrences()` |
| `tests/test_entity_links.py` | **Create** — edge case + integration tests for new store methods |

### Phase 11b: First L3 Pattern

| File | Action |
|------|--------|
| `agent/pipeline/cross_entity.py` | **Create** — co-occurrence detector, pattern scorer, L3 observation writer |
| `tests/test_cross_entity.py` | **Create** — edge case + integration tests |

## Implementation Steps

### Phase 11a: Cross-Entity Infrastructure

#### Step 11a.1: Add `entity_links` table to PipelineStore schema

```sql
CREATE TABLE IF NOT EXISTS entity_links (
    link_id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id_a TEXT NOT NULL REFERENCES entities(entity_id),
    entity_id_b TEXT NOT NULL REFERENCES entities(entity_id),
    link_type TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    source TEXT NOT NULL,
    created_at REAL NOT NULL,
    metadata_json TEXT,
    UNIQUE(entity_id_a, entity_id_b, link_type)
);

CREATE INDEX IF NOT EXISTS idx_entity_links_a
    ON entity_links(entity_id_a, link_type);
CREATE INDEX IF NOT EXISTS idx_entity_links_b
    ON entity_links(entity_id_b, link_type);
```

#### Step 11a.2: Add `link_entities()` method to PipelineStore

```python
def link_entities(
    self,
    entity_id_a: str,
    entity_id_b: str,
    link_type: str,
    source: str,
    confidence: float = 1.0,
    metadata: dict[str, Any] | None = None,
) -> int:
    """Create a typed link between two entities. Idempotent (INSERT OR IGNORE)."""
```

#### Step 11a.3: Add `query_entity_links()` method

```python
def query_entity_links(
    self,
    entity_id: str,
    *,
    link_type: str | None = None,
    direction: str = "both",  # "outgoing", "incoming", "both"
    min_confidence: float = 0.0,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Query links for an entity, optionally filtered by type and direction."""
```

#### Step 11a.4: Add `query_co_occurrences()` method

```python
def query_co_occurrences(
    self,
    entity_id_a: str,
    entity_id_b: str,
    *,
    window_seconds: float = 72 * 3600,
    source_tool_a: str | None = None,
    source_tool_b: str | None = None,
    since: float | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Find temporal co-occurrences: observation pairs from two entities
    where abs(obs_a.observed_at - obs_b.observed_at) <= window_seconds.
    
    Returns list of dicts with obs_a, obs_b, time_delta_seconds.
    """
```

#### Step 11a.5: Edge case test suite for entity_links + co-occurrences

Cover: duplicate links (idempotent), missing entities, direction filtering, confidence thresholds, empty results, window boundary precision, large time gaps, multiple co-occurrences per pair.

### Phase 11b: First L3 Pattern (Insider × GDELT)

#### Step 11b.1: Create `agent/pipeline/cross_entity.py` module

Core class: `CrossEntityDetector` that:
- Takes a PipelineStore
- Finds linked entity pairs across domains
- Runs co-occurrence queries within configurable time windows
- Scores co-occurrence patterns using conditional MI from `depth_eval.py`
- Stores L3 observations at depth_level=3

#### Step 11b.2: Implement company→country link seeder

Use SEC ticker data (already in entity.py) to create `headquartered_in` links from company entities to country entities. This connects the insider_filings domain to the GDELT domain.

#### Step 11b.3: Implement Insider × GDELT co-occurrence detector

Specific detector that:
1. For each company with insider_trade observations
2. Finds the linked country (via `headquartered_in`)
3. Queries co-occurrences between that company's insider_trade obs and that country's geopolitical_event obs
4. Filters to negative Goldstein events (conflict)
5. Returns scored co-occurrence list

#### Step 11b.4: Implement L3 observation storage

When a significant co-occurrence is found, store it as:
```python
observation_type = "cross_entity_pattern"
depth_level = 3
value = {
    "pattern_type": "insider_x_gdelt",
    "entity_a": company_entity_id,
    "entity_b": country_entity_id,
    "insider_event": {...},
    "gdelt_event": {...},
    "time_delta_hours": float,
    "score": float,  # conditional MI or significance
}
```

#### Step 11b.5: Edge case test suite for cross_entity module

Cover: no linked entities, no co-occurrences, multiple co-occurrences, scoring with real store, empty observation streams, boundary time windows.

## Edge Cases

1. **No entity links exist** — detector returns empty results gracefully
2. **Entity exists but has no observations** — skip, no error
3. **Co-occurrence window too narrow** — returns empty, not error
4. **Self-links** — entity linked to itself, should be rejected
5. **Duplicate links** — idempotent INSERT OR IGNORE
6. **Very large observation sets** — limit + pagination
7. **Time zone misalignment** — all timestamps are UTC epoch floats
8. **MI estimation with too few samples** — existing `_MIN_SAMPLES = 30` check applies

## Testing Plan

| Phase | Category | Count | Description |
|-------|----------|-------|-------------|
| 11a | entity_links CRUD | 6 | Create, query, dedup, direction, confidence filter, self-link |
| 11a | co-occurrence query | 6 | Basic pair, window boundary, empty, single-tool filter, since filter, multiple matches |
| 11a | integration | 2 | Real store round-trip, backward compat |
| 11b | link seeder | 3 | Company→country, missing data, duplicate |
| 11b | co-occurrence detector | 5 | Insider×GDELT hit, no-hit, multiple, filter, scoring |
| 11b | L3 storage | 3 | Store, query back, depth_level=3 |
| 11b | MI measurement | 2 | L3 > L2 conditional MI, insufficient samples |
| **Total** | | ~27+ | |

## Related

- [[cross_entity_l3]]
- [[gdelt_l2]]
- [[ais_vessel_l2]]
- [[whale_alert_l2]]
- [[deep_surveillance_tools]]
