---
title: "Task: database_architecture_strategy"
tags:
  - doc/task
  - status/active
  - phase/38
  - topic/database
  - topic/pipeline
  - topic/productization
  - layer/surveillance
  - layer/world-model
---

# Task: database_architecture_strategy

Status: active
Research: [[database_architecture_strategy]]
Spec: [[database_architecture_strategy_spec]]

## Goal

Establish a deliberate database roadmap for TirraMind so storage evolves from local experimentation to shared production infrastructure without premature migration or SQLite-specific lock-in.

## Atomic Steps

- [x] 1.1: Research the current store design and live-pipeline pressure profile.
- [x] 1.2: Verify external SQLite and PostgreSQL documentation on concurrency, JSON storage, and migration thresholds.
- [x] 1.3: Evaluate the main option set: stay on SQLite, dual-backend path, distributed SQLite variants, immediate PostgreSQL.
- [x] 1.4: Record the decision rule and migration triggers in the research note.
- [x] 2.1: Audit `PipelineStore` for SQLite-specific assumptions that would block a backend boundary.
- [x] 2.2: Identify schema and query patterns that should be normalized before any PostgreSQL backend exists.
- [x] 2.3: Define the storage protocol / abstraction boundary for dual-backend support.
- [x] 3.1: Specify the first PostgreSQL-compatible schema version and migration rules.
- [x] 3.2: Define parity tests for SQLite and PostgreSQL backends.
- [x] 3.3: Implement PostgresBackend with connection adapter that translates SQLite SQL dialect transparently.
- [x] 3.4: Write unit tests for the SQL translation layer (DDL + DML, 33 tests).
- [x] 3.5: Add postgres factory to backend-parity contract tests (auto-skips without `TIRRA_TEST_PG_DSN`).
- [ ] 4.1: Decide the cutover point for production shared state.
- [ ] 4.2: Design delivery read-models for customer-facing queries after the canonical store exists.

## Notes

- This task is architectural and planning-oriented first. It does not authorize immediate backend migration.
- The default current decision is: SQLite for local and single-host execution, PostgreSQL for future shared deployment when workload triggers are met.

## Related

- [[database_architecture_strategy]]
- [[database_architecture_strategy_spec]]
- [[quant_training_ground]]
- [[project_memory]]
- [[chat_checkpoint_2026-04-19_phase37_complete]]