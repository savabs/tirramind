---
title: "Spec: Database Architecture Strategy"
tags:
  - doc/spec
  - phase/38
  - topic/database
  - topic/pipeline
  - topic/productization
  - layer/surveillance
  - layer/world-model
---

# Spec: Database Architecture Strategy

## Goal

Decide and document the long-term database direction for TirraMind so future implementation steps are governed by a clear rule: SQLite remains the default for local and single-host execution until the workload becomes a genuinely shared, concurrent, product-facing system-of-record problem, at which point PostgreSQL becomes the canonical deployment target. The spec should turn that high-level rule into ordered implementation phases and migration triggers.

## Files Affected

- `[[database_architecture_strategy]]`
- `[[database_architecture_strategy_spec]]`
- `[[database_architecture_strategy]]`
- `agent/pipeline/storage_backend.py` — new file: `StorageBackend` ABC + `SQLiteBackend`
- `agent/pipeline/store.py` — refactored to delegate to `StorageBackend`
- `agent/pipeline/__init__.py` — exports `StorageBackend`, `SQLiteBackend`
- `tests/test_pipeline_store.py` — covers schema migration ledger and version idempotence
- `tests/test_pipeline_store_backend_contract.py` — backend-parity contract for SQLite today, PostgreSQL later

## Implementation Steps

1. Create the research note capturing the current SQLite-centered architecture, verified external docs, considered options, and recommended direction.
2. Create this spec and an active task file so the database track is represented in the vault and can be executed incrementally.
3. Define the architecture decision explicitly:
   - SQLite remains the default local development and single-host execution store.
   - PostgreSQL is the planned shared production system of record.
4. Define concrete migration triggers so the future cutover is driven by workload facts rather than fashion.
5. Define a phased implementation sequence:
   - backend boundary / storage audit
   - PostgreSQL-ready schema discipline
   - dual-backend store interface
   - production cutover
   - delivery read-model optimization
6. Keep the scope architectural for now. Do not implement the backend migration in this phase.

## Edge Cases

- Do not treat database choice as the product thesis; the product remains predictive intelligence.
- Do not migrate immediately just because PostgreSQL is more capable in the abstract.
- Do not assume SQLite is “toy” infrastructure; it is still valid for local, serialized, single-host execution.
- Do not overfit the design to vector search, full-text search, or other optional database features that are not yet central to the predictive engine.
- Do not let JSON flexibility erode the typed relational core.

## Testing Plan

- Verify the new markdown files have valid frontmatter and `## Related` sections.
- Verify the research note includes official documentation sources and concrete migration triggers.
- Verify the task file decomposes the database work into atomic, independently executable phases.
- Verify the baseline schema version is recorded exactly once in `schema_migrations` even after repeated schema initialization.
- Verify `PipelineStore.get_schema_version()` returns the canonical baseline version.
- Verify backend-contract tests pass for the current SQLite backend and can be re-used unchanged for a future PostgreSQL backend.
- Run the Obsidian lint script after creating the new docs and fix any frontmatter or broken-link issues if they appear.

## Related

- [[database_architecture_strategy]]
- [[database_architecture_strategy_spec]]
- [[quant_training_ground]]
- [[project_memory]]