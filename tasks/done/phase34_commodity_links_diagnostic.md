---
title: "Task: Phase 34 — Commodity Country Links + Diagnostic Sweep"
tags:
  - doc/task
  - status/done
  - phase/34
  - topic/l2-expansion
  - layer/surveillance
---

# Task: Phase 34 — Commodity Country Links + Diagnostic Sweep

Status: completed
Research: [[phase34_commodity_links_diagnostic]]
Spec: [[phase34_commodity_links_diagnostic_spec]]

## Steps

- [x] 34.1: Add `primary_exchange_country` field to InstrumentDef, set "US" on all 20 commodities
- [x] 34.2: Extend `_persist_instrument_links` with `exchange_country` link type
- [x] 34.3: Create `agent/models/gnn/graph_diagnostics.py` diagnostic utility
- [x] 34.4: Write comprehensive tests (`tests/test_phase34_commodity_links.py`)
- [x] 34.5: Full regression + checkpoint

## Related

- [[phase34_commodity_links_diagnostic]] — research
- [[phase34_commodity_links_diagnostic_spec]] — spec
- [[phase33_org_grid_l2]] — previous phase
