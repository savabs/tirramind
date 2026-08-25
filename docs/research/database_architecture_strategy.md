---
title: Database Architecture Strategy
tags:
  - doc/research
  - phase/38
  - topic/database
  - topic/pipeline
  - topic/productization
  - layer/surveillance
  - layer/world-model
---

# Feature: Database Architecture Strategy

## Current Architecture

- TirraMind currently uses a single SQLite-backed `PipelineStore` in [agent/pipeline/store.py](/home/becmachlean/2024/projects/tirramind_v1/agent/pipeline/store.py) as the central persistent store for runs, raw pipeline data, signals, features, beliefs, entities, observations, links, alerts, RL transitions, and paper-trade outputs.
- The store is explicitly documented as WAL-backed and local-file oriented. The current default database path is `.tirra_pipeline/pipeline.db`.
- The completed real pipeline run in [[chat_checkpoint_2026-04-19_phase37_complete]] confirms the store is now carrying real workloads: 918 entities, 69k+ observations, real GNN training, and walk-forward backtests.
- Current pressure is computational and graph-building heavy, but still appears single-host and application-server mediated rather than many independent network clients issuing SQL directly.

## What Problem We Are Actually Solving

- The database is not the product. It is the durable evidence-and-state backbone for the predictive engine.
- The long-term store must support:
  - canonical entities and aliases
  - timestamped observations with lineage
  - graph edges between entities and instruments
  - model outputs and posterior states
  - reproducible replay of what the system knew at time $t$
  - product delivery read-models later (alerts, probability surfaces, monitored coverage)
- The architecture should optimize for mathematical auditability, schema stability, and cheap evolution of the surveillance surface, not generic SaaS convenience.

## Internal Observations

- The schema already reflects the right long-term shape: typed relational tables for stable core objects plus JSON payload fields for flexible metadata.
- This is already closer to a future PostgreSQL schema than to a document-store or key-value-store design.
- SQLite is currently good enough because:
  - the database is local to the application process
  - the pipeline appears to serialize writes through the application layer
  - current scale is moderate in database terms even if training is computationally expensive
- SQLite becomes the wrong default when TirraMind needs true shared-state serving across multiple writers, multiple services, or multiple machines.

## External Sources Consulted

### Official documentation

- SQLite WAL: https://www.sqlite.org/wal.html
- SQLite appropriate uses: https://www.sqlite.org/whentouse.html
- PostgreSQL MVCC intro: https://www.postgresql.org/docs/current/mvcc-intro.html
- PostgreSQL JSON types: https://www.postgresql.org/docs/current/datatype-json.html
- Context7: `/websites/sqlite_docs`
- Context7: `/websites/postgresql_18`

### GitHub / OSS reconnaissance

- `prefix-dev/siglog` surfaced as an example of a system supporting both SQLite and PostgreSQL backends, including SQLite + LiteFS distribution. Useful as proof that dual-backend support is feasible, but not a direct design source for TirraMind.
- No OSS result changed the central conclusion: the decision boundary is workload shape, not trend-following around a specific stack.

## Verified Facts From External Docs

### SQLite

- SQLite WAL improves concurrency, allowing readers and a writer to proceed concurrently, but there is still only one writer at a time.
- SQLite documentation explicitly frames SQLite as ideal when storage is local to the application, writer concurrency is low, and the system is not a shared network repository.
- SQLite documentation explicitly recommends a client/server database when there are many concurrent writers, multiple machines directly sharing the same database, or the workload becomes a centralized shared repository.
- WAL mode requires all processes to be on the same host and does not work well as a general network-shared database filesystem model.
- WAL introduces checkpoint behavior that applications need to be aware of, and long-running readers can delay checkpoint completion and allow WAL growth.

### PostgreSQL

- PostgreSQL uses MVCC so reads do not conflict with writes in the same way as single-writer SQLite workloads; this is materially better for multi-user, multi-service shared-state systems.
- PostgreSQL `jsonb` is the preferred JSON storage type for most applications because it is binary-processed, indexable, and queryable.
- PostgreSQL documentation recommends mixed relational + JSON design rather than turning everything into unconstrained documents.
- PostgreSQL documentation also notes that large JSON updates still lock the row being updated, so JSONB is best for bounded flexible payloads, not giant mutable documents.

## Option Set Considered

### Option A: Stay on SQLite indefinitely

**Pros**
- Cheapest operational path
- Fast local development
- Zero extra infrastructure
- Already integrated and tested

**Cons**
- One-writer ceiling remains structural
- Awkward path for multi-process shared serving
- Weak fit for long-term customer-facing shared state
- More operational footguns once multiple services or hosts need the same database

**Assessment**
- Good for current local experimentation and controlled single-host runs.
- Not the right forever architecture if TirraMind becomes a shared predictive platform.

### Option B: SQLite for dev and local single-node runs, PostgreSQL for shared deployment

**Pros**
- Preserves cheap local iteration
- Matches the actual migration threshold from official docs
- Keeps the current store useful while acknowledging the eventual shared-state future
- Clean separation between local research workflow and deployed system-of-record

**Cons**
- Requires backend abstraction or migration strategy
- Schema and query differences must be managed deliberately
- Testing matrix gets larger

**Assessment**
- Best fit for TirraMind.
- Supports the current phase without pretending the current store is the final deployment shape.

### Option C: Distributed SQLite variants such as LiteFS instead of PostgreSQL

**Pros**
- Preserves SQLite ergonomics
- Can extend single-writer local-first workflows into certain deployment patterns

**Cons**
- Adds deployment complexity while preserving the underlying SQLite write model
- Solves a narrower problem than TirraMind actually has
- Less aligned with a future shared analytics and product-delivery store than PostgreSQL

**Assessment**
- Worth knowing exists.
- Not the preferred primary path for TirraMind.

### Option D: Migrate immediately to PostgreSQL everywhere

**Pros**
- Future-proofs shared-state deployment early
- Strong concurrency model
- Good fit for relational core + JSONB metadata

**Cons**
- Adds operational burden before the current bottleneck demands it
- Risks spending time on infrastructure before wiring more predictive value
- Violates the repo doctrine of not building commodity layers too early

**Assessment**
- Premature right now.
- Strong future target, weak immediate priority.

## Recommended Architecture

### Core recommendation

- **Short term:** keep SQLite as the default local development and single-host experimentation database.
- **Medium term:** define a storage abstraction and schema discipline so the relational core is portable.
- **Migration target:** PostgreSQL becomes the canonical shared system of record once TirraMind requires multi-service or multi-host concurrent writes, customer-facing shared persistence, or operational workloads that make the SQLite single-writer model a real constraint.

### Data model doctrine

- Use **typed relational tables** for stable concepts:
  - entities
  - aliases
  - observations
  - links
  - runs
  - features
  - beliefs
  - alerts
  - policy outputs
- Use **JSON/JSONB metadata** only for flexible per-tool payloads, source-specific extras, and bounded evidence details.
- Avoid giant mutable document blobs. The unit of reasoning should remain an atomic observation or state record.

### What is worth adopting from the “Postgres replaces the stack” idea

- Worth adopting:
  - one serious relational backbone
  - mixed relational + JSONB storage model
  - indexable flexible metadata
  - future geospatial support when physical-world sensing expands
  - materialized read models later for customer delivery
- Not worth adopting as doctrine:
  - “Postgres should replace everything”
  - early vector/search/auth/realtime infrastructure just because it exists
  - turning database convenience into the product thesis

## Migration Triggers

The architecture should move from SQLite-default to PostgreSQL-system-of-record when any of these become true:

1. Multiple independent services need to write to the same live store concurrently.
2. The production system runs on multiple hosts or requires remote shared persistence.
3. Customer-facing APIs, dashboards, or alerting workflows need reliable shared reads and writes against one canonical store.
4. WAL checkpoint behavior, `SQLITE_BUSY`, or one-writer contention becomes a recurring operational issue rather than a hypothetical one.
5. The evidence graph and model-output store need stronger operational guarantees for live serving, retention, and concurrent access than local-file SQLite is comfortable carrying.

## PipelineStore Backend-Boundary Audit

The first implementation step was to audit [agent/pipeline/store.py](/home/becmachlean/2024/projects/tirramind_v1/agent/pipeline/store.py) for assumptions that would block a clean backend boundary.

### Category 1: Direct SQLite driver coupling

- The store is typed directly against `sqlite3.Connection` and `sqlite3.Row`.
- Connection creation is hardcoded via `sqlite3.connect(...)` with SQLite-only arguments such as `check_same_thread=False`.
- Row parsing helpers accept `sqlite3.Row` specifically rather than a backend-agnostic row mapping protocol.

**Implication:** the current class is both the storage interface and the SQLite implementation. Those concerns need to be separated before a PostgreSQL backend can exist cleanly.

### Category 2: SQLite-only connection/session behavior

- `_get_conn()` executes `PRAGMA journal_mode=WAL` and `PRAGMA busy_timeout=5000`.
- The design assumes one lazily created long-lived local connection on a file path.
- Path creation and `:memory:` handling are SQLite-local behaviors.

**Implication:** connection lifecycle, session configuration, and local-file concerns should move behind a backend-specific adapter.

### Category 3: SQLite-specific DML semantics

- Many write paths depend on `INSERT OR REPLACE`.
- Other idempotent paths depend on `INSERT OR IGNORE`.
- Link insertion infers whether a row was newly inserted using `cursor.lastrowid` and `cursor.rowcount`.

**Implication:** these need explicit backend-neutral semantics. In PostgreSQL the equivalent behavior is usually `INSERT ... ON CONFLICT ... DO NOTHING/DO UPDATE ... RETURNING ...`, but `REPLACE` semantics are not identical and should not be hand-waved.

### Category 4: Identity retrieval assumptions

- Many methods return `cursor.lastrowid`.
- The schema uses `INTEGER PRIMARY KEY AUTOINCREMENT` in many tables.

**Implication:** row identity creation and return behavior need a backend-neutral contract. PostgreSQL will want `GENERATED ... AS IDENTITY` or `BIGSERIAL` plus `RETURNING`, not `lastrowid`.

### Category 5: SQLite-flavored schema choices that need normalization

- Time columns are uniformly stored as `REAL` Unix timestamps.
- Boolean-like values are stored as integers in several places.
- JSON payloads are stored as `TEXT` columns with manual `json.dumps` / `json.loads`.
- The schema is initialized through one large `executescript(_SCHEMA_SQL)` blob.

**Implication:** these are not fatal, but they should be made explicit as cross-backend schema policy decisions:
- whether timestamps remain epoch-float or become database-native timestamps
- whether booleans remain encoded or become actual boolean types in PostgreSQL
- which JSON fields should become `jsonb`
- how migrations will be versioned outside a monolithic bootstrap script

### Category 6: Query construction style

- Many read methods build SQL strings dynamically from safe clause fragments and placeholders.
- The current style is likely portable, but it is embedded directly in the SQLite implementation class.
- Some query shapes, such as unioned directional link reads and latest-belief joins, should be treated as canonical storage queries and tested for backend parity.

**Implication:** query text itself is not the main problem. The problem is that query behavior is not yet separated into a backend-neutral contract with parity tests.

### Category 7: Transaction model assumptions

- The store commits after most operations.
- Batch methods use manual `try/except` with `rollback()` and `commit()`.
- The design assumes a single connection and simple transaction boundaries rather than pooled sessions or explicit units of work.

**Implication:** a future PostgreSQL backend should define transaction boundaries deliberately, especially for batch writes and multi-step state transitions.

## Audit Outcome

The audit shows that the main blocker is **not SQL complexity**. The blocker is that `PipelineStore` currently mixes four roles in one class:

1. storage API
2. SQLite connection management
3. SQLite schema bootstrap
4. row parsing / serialization details

That means the first real implementation step should be:

### DB-1 recommendation

- Define a backend-neutral storage protocol around the operations TirraMind actually needs.
- Keep the current class as the initial SQLite implementation of that protocol.
- Move SQLite-specific connection/session/bootstrap behavior out of the protocol surface.
- Preserve current method semantics first; do not redesign the product data model during backend extraction.

### Minimum protocol surface suggested by the audit

- run records
- raw pipeline data
- signals
- features
- beliefs
- entity registry + aliases
- entity observations
- entity links
- alerts / convergence clusters
- RL transitions / checkpoints
- portfolio outputs / paper PnL
- discovered sources / unresolved entities / entity type registry

This is the correct next step because it reduces lock-in without paying the cost of a premature PostgreSQL migration.

## Implemented Schema-Version Baseline

The first concrete PostgreSQL-compatibility step is now implemented in [agent/pipeline/store.py](/home/becmachlean/2024/projects/tirramind_v1/agent/pipeline/store.py):

- `PipelineStore` now maintains a `schema_migrations` ledger table.
- The canonical schema name is `pipeline_store`.
- The baseline portable schema version is `1`.
- Re-running schema initialization is idempotent and does not duplicate the baseline migration row.

### Migration rules for schema version 1

- Timestamps remain epoch `REAL` values for now so SQLite and a first PostgreSQL backend can share the same application semantics during transition.
- Boolean-like fields remain integer-encoded in the canonical contract for version 1; a future PostgreSQL backend may map these to native booleans internally only when parity tests prove behavior is unchanged.
- Flexible payloads remain JSON serialized to text in version 1; PostgreSQL may later store these as `jsonb`, but only behind the same API contract.
- Schema evolution must be additive first: create the migration entry, preserve existing read/write semantics, and avoid silent type reinterpretation across backends.

### Why this matters

This gives TirraMind a real migration anchor instead of a single unversioned bootstrap blob. PostgreSQL work can now target a named schema contract and explicit baseline behavior rather than reverse-engineering the current SQLite state from scratch.

## Implemented Backend-Parity Contract

The first reusable backend contract suite now exists in [tests/test_pipeline_store_backend_contract.py](/home/becmachlean/2024/projects/tirramind_v1/tests/test_pipeline_store_backend_contract.py).

- The contract is parameterized by backend factory.
- SQLite is the first backend exercising the suite.
- A future PostgreSQL backend should be added by extending the factory map, not by rewriting the assertions.

### Current contract coverage

- schema version baseline
- DAG run round-trip semantics
- pipeline data round-trip semantics
- entity / alias / observation / link round-trip semantics

This is enough to force a first PostgreSQL backend to match the current store on core persistence behavior before any cutover is attempted.

## Proposed Phase Order

1. **Phase DB-1: Storage audit and backend boundary**
   - isolate SQLite-specific assumptions in `PipelineStore`
   - document schema invariants and query patterns
2. **Phase DB-2: PostgreSQL-ready schema discipline**
   - normalize timestamp, JSON field, and index expectations
   - remove accidental SQLite-specific behavior
3. **Phase DB-3: Dual-backend store interface**
   - keep SQLite for dev
   - add PostgreSQL backend behind a shared protocol
4. **Phase DB-4: Shared deployment cutover**
   - PostgreSQL as canonical production store
   - migration scripts and parity tests
5. **Phase DB-5: Delivery read-models**
   - materialized views / precomputed tables for alerts, probability surfaces, and customer-facing query paths

## Risks

- Migrating too early burns time on infrastructure before predictive output quality is strong enough to justify it.
- Migrating too late risks baking SQLite-specific assumptions into more of the codebase.
- Overusing JSONB can quietly degrade schema clarity and update behavior.
- Underusing JSONB can create needless schema churn for source-specific metadata.

## Conclusion

- TirraMind should not adopt “all-in-one Postgres” as ideology.
- TirraMind should adopt **PostgreSQL as the planned shared evidence backbone** once the system genuinely becomes shared, concurrent, and product-facing.
- Until then, SQLite remains valid for local, single-host, application-mediated execution.
- The right long-term architecture is **relational core + flexible metadata + explicit migration trigger**, not “replace the stack because a video said so”.

## Related

- [[database_architecture_strategy_spec]]
- [[database_architecture_strategy]]
- [[quant_training_ground]]
- [[project_memory]]
- [[chat_checkpoint_2026-04-19_phase37_complete]]