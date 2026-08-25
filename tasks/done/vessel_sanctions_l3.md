---
title: "Task: Vessel × Sanctions L3 Pattern — Phase 11c"
tags:
  - doc/task
  - status/done
  - phase/11
  - topic/surveillance
  - topic/convergence
  - layer/feature-engineering
---

# Task: Vessel × Sanctions L3 Pattern — Phase 11c

Status: completed
Research: [[vessel_sanctions_l3]]
Spec: [[vessel_sanctions_l3_spec]]

## Steps

- [x] 11c.1: Port → Country mapping utility (ISO→FIPS + Baltic port lookup + resolve_port_country)
- [x] 11c.2: `seed_vessel_country_links()` — scan port_call observations → create `port_call_to` links
- [x] 11c.3: `detect_vessel_sanctions()` — co-occurrence detector for vessel × GDELT sanctions events
- [x] 11c.4: Edge case test suite (30 tests — 11 resolve, 6 seed, 11 detect, 2 integration)

## Notes

- Extends `cross_entity.py` — no new files needed.
- Vessel→country links derived from port_call observation data (port, prev_port, next_port fields).
- UN LOCODE prefix (2-char ISO code) is the primary country resolution method.
- CAMEO root codes 16 (Reduce Relations) and 17 (Coerce) filter for sanctions-specific events.
- Window: 48h (tighter than Insider×GDELT 72h because AIS is T0 real-time).
- Uses same scoring formula (severity × proximity) and store_l3_observations for persistence.

## Related

- [[vessel_sanctions_l3]]
- [[vessel_sanctions_l3_spec]]
- [[cross_entity_l3]]
- [[ais_vessel_l2]]
