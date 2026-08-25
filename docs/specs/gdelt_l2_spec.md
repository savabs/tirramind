---
title: "Spec: GDELT L2 — Actor Entity Registration"
tags:
  - doc/spec
  - phase/10
  - topic/surveillance
  - layer/surveillance
---

# Spec: GDELT L2 — Actor Entity Registration

## Goal

Upgrade `GDELTTool` to register country entities from GDELT event actor pairs and store per-country geopolitical event observations at depth_level=2 in the PipelineStore entity registry. Articles mode remains L1 (no persistence).

## Files Affected

| File | Action |
|------|--------|
| `agent/tools/gdelt.py` | **Modify** — add TYPE_CHECKING import, entity imports, PipelineStore constructor kwarg, `_persist_entities()`, `_persist_entities_inner()`, `entity_ids` in output events |
| `tests/test_gdelt_l2.py` | **Create** — L2 edge case + MI integration test suite |

## Implementation Steps

### Step 10b.5.1: Add TYPE_CHECKING + entity imports

At the top of `gdelt.py`, add:
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.pipeline.store import PipelineStore

try:
    from agent.pipeline.entity import entity_id_from_key
except ImportError:
    entity_id_from_key = None
```

### Step 10b.5.2: Accept optional PipelineStore in constructor

Change constructor:
```python
def __init__(
    self,
    cache: DataCache | None = None,
    *,
    pipeline_store: PipelineStore | None = None,
) -> None:
    self._cache = cache
    self._store = pipeline_store
```

### Step 10b.5.3: Implement `_persist_entities()` + `_persist_entities_inner()`

Guard method (same pattern):
```python
def _persist_entities(self, events: list[dict[str, Any]]) -> None:
    if self._store is None or entity_id_from_key is None:
        return
    self._persist_entities_inner(events)
```

Inner method:
- For each event:
  - Extract actor1 country code, actor2 country code
  - For each valid (non-empty) country code:
    - Compute `eid = entity_id_from_key("country", country_code)`
    - If not in `seen` set: register entity, add FIPS alias
    - Store observation: `observation_type="geopolitical_event"`, role="initiator"/"target", counterpart, goldstein, etc.

### Step 10b.5.4: Add `entity_id` fields to event dicts

In `_execute_events`, after persistence, add entity_id fields to each event's actor1 and actor2 sub-dicts:
```python
if entity_id_from_key is not None:
    for e in events:
        a1c = e["actor1"]["country"]
        a2c = e["actor2"]["country"]
        if a1c:
            e["actor1"]["entity_id"] = entity_id_from_key("country", a1c)
        if a2c:
            e["actor2"]["entity_id"] = entity_id_from_key("country", a2c)
```

### Step 10b.5.5: Wire persistence into `_execute_events`

After sorting/limiting events but before the return:
```python
try:
    self._persist_entities(events)
except Exception:
    log.exception("Entity persistence failed in events mode (non-fatal)")
```

No persistence in `_execute_articles` — articles have no structured actor data.

### Step 10b.5.6: Edge case test suite + MI integration test

Create `tests/test_gdelt_l2.py`.

## Edge Cases

1. **Empty actor country code** — ~10-15% of events. Skip entity registration for that actor.
2. **Both actors same country** — domestic event. Both observations stored, same entity.
3. **No store** — all persistence is no-op.
4. **entity_id_from_key unavailable** — no-op.
5. **Persistence error** — caught, logged, tool returns normal results.
6. **Empty events list** — no persistence calls.
7. **Articles mode** — no persistence at all.
8. **Actor name None** — use country code as canonical name fallback.
9. **Dedup** — same country across many events → registered once, many observations.

## Testing Plan

| Category | Count | Description |
|----------|-------|-------------|
| Constructor | 3 | Default, with store, keyword-only enforcement |
| Persist guard | 4 | No store, empty list, no entity_id, error in inner |
| Entity persistence | 8 | Both actors persisted, empty country skip, dedup, same country dyad, name fallback, actor type metadata, observation schema, multiple events |
| entity_ids output | 4 | Both actors get entity_id, empty country excluded, articles mode absent, entity module unavailable |
| Integration | 3 | With real store, backward compat, persistence error non-fatal |
| MI measurement | 2 | L2 > L1, with real store |
| **Total** | ~24+ | |

## Related

- [[gdelt_l2]]
- [[ais_vessel_l2]]
- [[whale_alert_l2]]
- [[project_memory]]
