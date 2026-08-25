---
title: "Task: Deep Surveillance Phase 10b — insider_filings L2"
tags:
  - doc/task
  - status/done
  - phase/10
  - topic/surveillance
  - layer/surveillance
---

# Task: Deep Surveillance Phase 10b.1 — insider_filings L2 Upgrade

Status: completed
Research: [[deep_surveillance_tools]]
Spec: [[deep_surveillance_tools_10b_spec]]

## Steps

- [x] 10b.1.1: Add reporter_cik and issuer_cik to transaction dicts in _parse_filings / _parse_form4_xml
- [x] 10b.1.2: Accept optional PipelineStore in InsiderFilingsTool constructor
- [x] 10b.1.3: Implement _persist_entities() — entity registration + observation storage
- [x] 10b.1.4: CIK-based dedup in _find_best_cluster() with name fallback
- [x] 10b.1.5: Add entity_ids mapping to cluster data
- [x] 10b.1.6: Edge case test suite for all L2 changes
- [x] 10b.1.7: MI measurement integration test (L2 vs L1 depth eval)

## Notes

- Phase 10a infrastructure (entity tables, MI/KL computation) is complete and tested (83 tests passing).
- insider_filings is the first tool upgrade — establishes the L2 wiring pattern for subsequent tools.
- All changes are backward compatible — existing behavior preserved when pipeline_store=None.

## Related

- [[deep_surveillance_tools]]
- [[deep_surveillance_tools_10b_spec]]
- [[deep_surveillance_tools_spec]]
- [[deep_surveillance]]
- [[project_memory]]
