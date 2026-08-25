---
title: "Checkpoint: 2026-04-15 — Phase 27 Preflight"
tags:
  - doc/checkpoint
  - phase/27
  - topic/fx
  - topic/central-bank
  - topic/entity-linking
  - layer/surveillance
  - layer/world-model
---

# Checkpoint: 2026-04-15 — Phase 27 Preflight

## Summary

The project path has been explicitly reset to prioritize the core model track after Phase 25. Phase 25 bookkeeping is closed, and the next implementation target is now formalized as **Phase 27 — FX Country Wiring + Central Bank L2 Connectivity**.

## What Changed

- Removed the completed active copy of [[phase25_cross_domain_entity_linking]] from `tasks/active/`.
- Updated [[quant_training_ground]] so the current phase points to Phase 27 instead of Phase 25.
- Created the new research note [[phase27_fx_country_monetary_linking]].
- Created the new spec [[phase27_fx_country_monetary_linking_spec]].
- Created the new active task [[phase27_fx_country_monetary_linking]].

## Why This Phase Was Chosen

Phase 25's checkpoint identified FX pairs as the clearest remaining starved instrument class. The highest-leverage next step is to expose monetary-state signal to country nodes and improve FX pair country connectivity, rather than spending the next core-model cycle on infrastructure.

The chosen design is deliberately conservative:
- reuse **country** entities rather than add a new `central_bank` entity type,
- upgrade `central_bank_balance` to persist compact country-level monetary observations,
- enrich FX instruments with explicit two-country structure where deterministic.

This keeps the phase aligned with the current graph builder and avoids premature type-system refactors.

## Immediate Next Execution Target

Begin **Phase 27 step 27.1**: define deterministic two-country metadata for tracked FX pairs and document the exact mapping rules that Phase 27 will persist.

## Deferred Tracks

- [[phase26_mcp_agent_upgrade]] remains active as a tooling/infrastructure track, but it is not the immediate core-model priority.
- Productization/positioning remains secondary to expanding the predictive engine.

## Related

- [[phase27_fx_country_monetary_linking]]
- [[phase27_fx_country_monetary_linking_spec]]
- [[quant_training_ground]]
- [[phase25_cross_domain_entity_linking]]
- [[phase25_gnn_diagnostic]]
- [[7b-Z_central_bank_balance_sheets]]
