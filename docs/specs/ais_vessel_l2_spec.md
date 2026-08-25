---
title: "Spec: ais_vessel L2 Upgrade"
tags:
  - doc/spec
  - phase/10
  - topic/surveillance
  - layer/surveillance
---

# Spec: ais_vessel L2 — Vessel Entity Registration

## Goal

Upgrade `AISVesselTool` to register vessel entities (by IMO or MMSI) and store per-vessel position and port-call observations at depth_level=2 in the PipelineStore entity registry.

## Files Affected

| File | Action |
|------|--------|
| `agent/tools/ais_vessel.py` | **Modify** — add TYPE_CHECKING import, entity imports, PipelineStore constructor kwarg, entity_id helper, `_persist_entities()`, `_persist_entities_inner()`, `_persist_port_call_entities()`, `entity_ids` in output dicts for area/vessel/port_calls modes |
| `tests/test_ais_vessel_l2.py` | **Create** — L2 edge case + MI integration test suite |

## Implementation Steps

### Step 10b.4.1: Add TYPE_CHECKING + entity imports

At the top of `ais_vessel.py`, add:
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.pipeline.store import PipelineStore

try:
    from agent.pipeline.entity import entity_id_from_key
except ImportError:
    entity_id_from_key = None
```

### Step 10b.4.2: Accept optional PipelineStore in constructor

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

### Step 10b.4.3: Add `_vessel_entity_id()` helper

A private helper encapsulating the IMO-first, MMSI-fallback logic:

```python
@staticmethod
def _vessel_entity_id(mmsi: int, imo: int | None = None) -> str | None:
    if entity_id_from_key is None:
        return None
    if imo:
        return entity_id_from_key("vessel", str(imo))
    return entity_id_from_key("vessel", f"mmsi:{mmsi}")
```

### Step 10b.4.4: Implement `_persist_entities()` + `_persist_entities_inner()`

Guard method (same pattern as other tools):
```python
def _persist_entities(self, vessels: list[dict[str, Any]]) -> None:
    if self._store is None or entity_id_from_key is None or not vessels:
        return
    try:
        self._persist_entities_inner(vessels)
    except Exception:
        log.exception("Entity persistence failed (non-fatal)")
```

Inner method for position-based observations (area + vessel modes):
- For each vessel dict:
  - Compute `vessel_eid = self._vessel_entity_id(mmsi, imo)`
  - Register entity: `entity_type="vessel"`, `canonical_name = name or f"MMSI:{mmsi}"`
  - Add MMSI alias: `add_entity_alias(vessel_eid, "mmsi", str(mmsi))`
  - If IMO present: `add_entity_alias(vessel_eid, "imo", str(imo))`
  - Store observation: `observation_type="vessel_position"`, `depth_level=2`
- Dedup: `seen_vessels: set[str]` keyed by entity_id to avoid re-registering same vessel from duplicate records

### Step 10b.4.5: Implement `_persist_port_call_entities()`

Separate method for port_call observations (different schema):
```python
def _persist_port_call_entities(self, calls: list[dict[str, Any]]) -> None:
    if self._store is None or entity_id_from_key is None or not calls:
        return
    try:
        self._persist_port_call_entities_inner(calls)
    except Exception:
        log.exception("Port call entity persistence failed (non-fatal)")
```

Inner method:
- For each port call dict:
  - Extract `imo = call.get("imoLloyds")`, `mmsi = call.get("mmsi")`
  - If neither: skip
  - Compute entity_id (IMO-first)
  - Register entity, add aliases (mmsi + imo)
  - Store observation: `observation_type="port_call"`, `depth_level=2`

### Step 10b.4.6: Wire persistence into mode handlers

**`_mode_area()`:** After building the `matched` list (after filtering), call:
```python
try:
    self._persist_entities(matched[:limit])
except Exception:
    log.exception("Entity persistence failed in area mode (non-fatal)")
```

**`_mode_vessel()`:** After building the `result` dict, call:
```python
try:
    self._persist_entities([result])
except Exception:
    log.exception("Entity persistence failed in vessel mode (non-fatal)")
```

**`_mode_port_calls()`:** After fetching calls, call:
```python
try:
    self._persist_port_call_entities(calls[:limit])
except Exception:
    log.exception("Entity persistence failed in port_calls mode (non-fatal)")
```

**`_mode_destination_flow()`:** No persistence (aggregate only).

### Step 10b.4.7: Add `entity_ids` to output dicts

**Area mode:** Add `entity_ids` dict mapping MMSI → entity_id for each vessel in `matched`:
```python
if entity_id_from_key is not None:
    for v in matched:
        v["entity_id"] = self._vessel_entity_id(v["mmsi"], v.get("imo"))
```

**Vessel mode:** Add to the result dict:
```python
result["entity_id"] = self._vessel_entity_id(mmsi, result.get("imo"))
```

**Port calls mode:** Add to each call dict:
```python
if entity_id_from_key is not None:
    for c in calls:
        imo_val = c.get("imoLloyds")
        mmsi_val = c.get("mmsi")
        if imo_val or mmsi_val:
            c["entity_id"] = self._vessel_entity_id(mmsi_val or 0, imo_val)
```

### Step 10b.4.8: Edge case test suite + MI integration test

Create `tests/test_ais_vessel_l2.py`.

## Edge Cases

1. **No IMO** — small vessels have MMSI but no IMO. Must fall back to MMSI-keyed entity_id.
2. **IMO is None or 0** — treat as missing. Use MMSI fallback.
3. **Same IMO, different MMSI** — reflagged vessel. Both resolve to same entity_id.
4. **area mode without metadata** — when `ship_type="all"`, metadata is not fetched. No IMO/name available. Persist with MMSI-only entity.
5. **Port call missing both IMO and MMSI** — skip entity registration.
6. **No store** — all persistence is no-op.
7. **entity_id_from_key unavailable** — no-op.
8. **Persistence error** — caught, logged, tool returns normal results.
9. **Empty vessel list** — no persistence calls.
10. **destination_flow mode** — no persistence at all.

## Testing Plan

| Category | Count | Description |
|----------|-------|-------------|
| Constructor | 3 | Default, with store, keyword-only enforcement |
| _vessel_entity_id | 4 | IMO-first, MMSI-fallback, no IMO, entity_id_from_key=None |
| Persist guard | 4 | No store, empty list, no entity_id, error caught |
| Position persistence | 6 | Sender/receiver analog: area vessels, vessel mode, dual alias, dedup, name fallback, no metadata |
| Port call persistence | 4 | With IMO, with MMSI only, missing both, dedup |
| entity_ids output | 5 | Area mode, vessel mode, port calls, no entity module, destination_flow (absent) |
| Integration | 3 | With real store, backward compat, persistence error |
| MI measurement | 2 | L2 > L1, with real store |
| **Total** | ~31+ | |

## Related

- [[ais_vessel_l2]]
- [[deep_surveillance_tools]]
- [[deep_surveillance_10b]]
- [[deep_surveillance_10b2]]
- [[project_memory]]
