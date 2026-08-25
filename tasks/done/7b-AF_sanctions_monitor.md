---
title: "Task: 7b-AF Sanctions Monitor Tool"
tags:
  - doc/task
  - layer/surveillance
  - phase/7b
  - status/done
  - topic/sanctions
---

# Task: 7b-AF Sanctions Monitor Tool

Status: completed
Research: [[7b-AF_sanctions_monitor]]
Spec: [[7b-AF_sanctions_spec]]

## Summary
Built `sanctions_monitor` tool (Layer 1: Surveillance Surface) for monitoring OFAC SDN + UN Security Council sanctions lists.

## What was built
- **agent/tools/sanctions_monitor.py** — Full tool with 3 modes:
  - `search` — Name/alias search across OFAC + UN (case-insensitive substring match)
  - `recent` — Recently listed/updated entities (UN has per-entry dates; OFAC does not)
  - `programs` — Active sanctions programs with entity counts + examples
- **Registered** in `agent/cli.py` (tool #31)
- **Bandit arm** `sanctions_screening` added to `agent/learning/bandit.py` (arm #20)
- **130 edge case tests** in `tests/test_sanctions_monitor_edge.py`

## Data sources
- OFAC SDN CSV: treasury.gov, ~5.5MB, ~18,708 entries, no auth
- UN SC XML: scsanctions.un.org, ~2MB, ~900 entries, no auth
- Both cached with 6h TTL (lists change at most weekly)

## Test results
- 130 new tests: all pass
- Full suite: 1742 passed, 0 failed, 6 skipped (pre-existing skips)

---

## Related

- [[7b-AF_sanctions_monitor|Research: 7B-Af Sanctions Monitor]]
- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]
