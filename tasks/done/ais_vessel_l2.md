---
title: "Task: Deep Surveillance Phase 10b.4 — ais_vessel L2"
tags:
  - doc/task
  - status/done
  - phase/10
  - topic/surveillance
  - layer/surveillance
---

# Task: Deep Surveillance Phase 10b.4 — ais_vessel L2 Upgrade

Status: completed
Research: [[ais_vessel_l2]]
Spec: [[ais_vessel_l2_spec]]

## Steps

- [x] 10b.4.1: Add TYPE_CHECKING + entity imports to ais_vessel.py
- [x] 10b.4.2: Accept optional PipelineStore in constructor (keyword-only)
- [x] 10b.4.3: Add `_vessel_entity_id()` helper (IMO-first, MMSI-fallback)
- [x] 10b.4.4: Implement `_persist_entities()` + `_persist_entities_inner()` — vessel registration + position observations
- [x] 10b.4.5: Implement `_persist_port_call_entities()` + inner — port call observations
- [x] 10b.4.6: Wire persistence into area, vessel, port_calls mode handlers
- [x] 10b.4.7: Add `entity_id`/`entity_ids` to output dicts (area, vessel, port_calls)
- [x] 10b.4.8: Edge case test suite + MI integration test (46 tests passing)

## Notes

- Same L2 wiring pattern as prior tools but with key differences:
  - **Dual identity:** IMO (permanent hull ID) preferred over MMSI (radio ID that changes on reflagging)
  - **Three persist hooks:** area, vessel, port_calls modes each persist entities
  - **Two observation types:** `vessel_position` and `port_call`
  - **destination_flow is L1 only** — aggregate view, no entity persistence
- Metadata not always loaded in area mode (when ship_type=all) → MMSI-only entity, IMO enriched later

## Related

- [[ais_vessel_l2]]
- [[ais_vessel_l2_spec]]
- [[deep_surveillance_tools]]
- [[deep_surveillance_10b]]
- [[deep_surveillance_10b2]]
