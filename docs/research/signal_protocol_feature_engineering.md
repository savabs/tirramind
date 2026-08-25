---
title: "Feature: Signal Protocol + Feature Engineering"
tags:
  - doc/research
  - layer/feature-engineering
  - layer/surveillance
  - layer/world-model
  - phase/9
  - topic/signal-protocol
---

# Feature: Signal Protocol + Feature Engineering

## Current Architecture
- Layer 1 tools write raw structured outputs into `PipelineStore.pipeline_data`.
- Convergence currently normalizes tool outputs into `Evidence` and emits scalar `ConvergenceSignal` rows into `PipelineStore.signals`.
- The pipeline store `signals` table is minimal: `signal_name`, `computed_at`, `value`, `metadata_json`.
- Existing quantitative infrastructure already includes walk-forward scoring and convergence detection, but there is no repo-wide protocol for reusable downstream features.
- Canonical phase order in `[[quant_training_ground]]` is:
  - Phase 8: Signal Protocol + Feature Engineering
  - Phase 9: World Model
- Older checkpoint/audit notes sometimes call the next phase “World Model”; that is a numbering mismatch, not the current canonical order.

## Observations
- `Evidence` is good for detection-time normalization, but it is not yet a stable downstream feature contract for Phase 9+ consumers.
- `ConvergenceSignal` is a single scalar event record, not a generalized feature vector protocol.
- There is no standard representation for:
  - feature horizon
  - units / scaling
  - source lineage
  - staleness / validity windows
  - feature versioning
  - missingness semantics
  - confidence / quality fields at the reusable feature layer
- `PipelineStore` currently has durable storage for raw data and scalar signals, but not an explicit feature table or a typed protocol for model-ready vectors.
- Phase 8 should not mix in Bayesian inference or filtering yet. Its job is to turn raw and detected signals into stable, model-ready quantitative state variables.

## Risks
- If Phase 9 starts without a strict signal protocol, the world model will consume ad hoc JSON metadata and silently accumulate schema drift.
- Overbuilding a generalized feature platform too early will slow momentum; the first version should cover the smallest high-signal subset needed for the world model.
- Reusing tool-specific naming directly as model inputs risks coupling Phase 9 to extractor details.
- Feature engineering can easily leak future information unless horizon, publication timing, and effective timestamps are first-class protocol fields.

## Data Requirements
- Read from `PipelineStore`, not tool cache, per pipeline-layer design.
- Preserve provenance from tool output → evidence → engineered feature.
- Support both event-style signals (convergence) and continuous state features (rates, spreads, breadth, momentum, stress counts).
- Include point-in-time timestamps sufficient to prevent look-ahead.

## Math/Algorithm Survey
- Phase 8 is still Layer 2. The math focus is deterministic transformation, normalization, aggregation, and state summarization rather than probabilistic inference.
- Candidate feature families for the first slice:
  - z-scored anomalies
  - rate-of-change / momentum
  - breadth / concentration
  - persistence / decay
  - cross-sectional spreads
  - regime flags
- The right first deliverable is not “all possible features”; it is a small protocol plus 1-2 production feature builders that prove the contract.

## Recommended Scope for First Atomic Slice
1. Define a reusable feature record protocol.
2. Extend storage to persist engineered features separately from raw signals.
3. Implement one deterministic feature builder over convergence output.
4. Implement one deterministic feature builder over an existing continuous macro/liquidity source.
5. Add DAG wiring and edge-case tests.

## Step-Local References
- `[[quant_training_ground]]`
- `agent/pipeline/store.py`
- `agent/convergence/evidence.py`
- `agent/convergence/signals.py`
- `[[pipeline_layer]]`
- `[[convergence_detection]]`
- `[[project_memory]]`

### Step 8.1 — Feature Protocol Design References
- **Feast feature store** (https://docs.feast.dev/getting-started/concepts/feature-view):
  entity-keyed, timestamped schema with TTL, schema validation, explicit Field types.
  Takeaway for TirraMind: timestamp is mandatory, schema validation at write time,
  TTL controls freshness. We do NOT need entity/join machinery — our features are
  global state variables, not per-entity.
- **Existing `ConvergenceSignal`** (`agent/convergence/signals.py`):
  Already has `signal_name`, `computed_at`, `value`, `event_type`, `direction`,
  `p_value`, `persistence_days`. Rich metadata but event-scoped, not a general
  feature contract.
- **Existing `AtomicSignalResult`** (`agent/convergence/atomic_signals.py`):
  Has `signal_id`, `timestamp`, `raw_value`, `z_score`, `percentile`, `is_anomaly`,
  `direction`. Good precedent for z-score normalization as a standard field.
- **Existing `signals` table** (`agent/pipeline/store.py`):
  Minimal: `signal_name`, `computed_at`, `value`, `metadata_json`. Step 8.2 will
  add a separate `features` table — 8.1 only defines the in-memory protocol.
- **Design decisions for 8.1**:
  - Frozen dataclass (immutable records)
  - `effective_at` (point-in-time knowability) separate from `computed_at` (DAG run time)
  - Explicit missingness: `value=None` requires `missing_reason`
  - Dotted name convention: `{domain}.{metric}.{horizon}`
  - `version` field for schema evolution without corrupting history
  - Validation as a pure function returning error list (not exceptions)
  - Serialization: `to_dict()` / `from_dict()` for SQLite JSON round-tripping

### Step 8.2 — Feature Persistence References
- **Existing `PipelineStore`** (`agent/pipeline/store.py`):
  SQLite WAL, tables: `dag_runs`, `pipeline_data`, `signals`. Pattern: each domain
  has its own table, store/query methods, and a static `_*_row_to_dict` helper.
- **`EngineeredFeature.to_dict()` / `from_dict()`** (`agent/features/protocol.py`):
  Full round-trip serialization already implemented. Storage layer can JSON-dump
  `source_signals` tuple and `metadata` dict into TEXT columns.
- **`validate_feature()`** (`agent/features/protocol.py`):
  Must be called at write boundary before INSERT. Invalid features rejected.
- **Design decisions for 8.2**:
  - New `features` table — do NOT overload existing `signals` table.
  - Columns: `id`, `feature_name`, `version`, `effective_at`, `computed_at`,
    `horizon`, `value` (REAL nullable), `quality`, `missing_reason`,
    `source_signals_json`, `builder`, `unit`, `metadata_json`.
  - Unique constraint on `(feature_name, version, effective_at)` to make
    duplicate recomputation idempotent (INSERT OR REPLACE).
  - Index on `(feature_name, effective_at)` for point-in-time lookups.
  - `store_feature()` validates before write; returns row id.
  - `store_features_batch()` for efficient multi-insert inside one transaction.
  - `query_features()` by name, time range, version, with limit.
  - `get_latest_feature()` convenience for single most-recent value.
  - All methods go on `PipelineStore` so the connection is shared.

### Step 8.3 — Convergence-Derived Feature Builder References
- **`ConvergenceSignal`** (`agent/convergence/signals.py`):
  Fields: `signal_name`, `computed_at`, `value` (score 0-1), `event_type`,
  `signals_involved`, `categories_involved`, `cross_category_count`,
  `p_value`, `persistence_days`, `template_match`, `direction`, `lead_signal`,
  `lag_signals`.  Stored in `signals` table with metadata JSON.
- **`DetectionResult`** (`agent/convergence/detector.py`):
  `clique` (ConvergenceClique), `event_type`, `template_match`,
  `boosted_score`, `lead_signal`, `lag_signals`, `direction`.
- **`ConvergenceClique`** (`agent/convergence/graph.py`):
  `signals`, `categories`, `edges`, `score`, `p_values`.
- **What convergence-derived features make sense**:
  The convergence system already produces per-event scalars. What Phase 9
  needs is *aggregate state variables*, not per-event records:
  1. **stress_breadth** — how many distinct signals are currently anomalous
     (count of active convergence signals). High breadth = broad stress.
  2. **stress_intensity** — max boosted_score across active events.
     High intensity = at least one strong convergence cluster.
  3. **regime_persistence** — max persistence_days across active events.
     Long persistence = regime is entrenched, not a transient spike.
  These three are complementary: breadth × intensity × persistence gives a
  multi-dimensional view of convergence state. All derived deterministically
  from the `signals` table — no new data sources needed.
- **Design decisions for 8.3**:
  - Builders read from `PipelineStore.query_signals()` for convergence signals.
  - Output: three `EngineeredFeature` records per build cycle.
  - Horizon: `7d` (aggregate over 7 days of convergence activity).
  - Staleness: if no convergence signals in the window, emit explicit
    missingness (`value=None, missing_reason="no_convergence_activity"`).
  - Each builder is a class with a `build(store, as_of) -> list[EngineeredFeature]`
    API.

### Step 8.4 — Continuous-State Feature Builder References
- **Macro data in PipelineStore**: source=`"macro_data"`, data shape is
  `{series_id: [{date, value}, ...]}`.
- **Useful continuous-state features for Phase 9**:
  1. **rate_momentum** — 30-day rate-of-change of the Federal Funds Rate
     proxy (DFF from FRED). Rising rate momentum = tightening regime.
  2. **yield_curve_slope** — spread between 10Y and 2Y Treasury yields
     (GS10, GS2 from FRED). Inverted curve = recession signal.
  3. **liquidity_pressure** — z-score of Fed balance sheet change (WALCL).
     Shrinking balance sheet = quantitative tightening.
  These three capture the macro backdrop that convergence events occur in.
  All from FRED via `pipeline_data` source=`"macro_data"`.
- **Design decisions for 8.4**:
  - Reads from `PipelineStore.query_data("macro_data")`.
  - Parses the stored JSON to extract series values.
  - Rolling statistics computed in-builder (no dependency on convergence).
  - Horizon: `30d` for rate_momentum + liquidity_pressure, `spot` for
    yield_curve_slope.
  - Staleness: if no macro data in the window, emit missingness.
  - Window: 90 days of history for z-score baseline.

### Shared Design Decisions (8.3 + 8.4)
- **FeatureBuilder abstract base** in `agent/features/builders.py`:
  `name` property + `build(store, as_of) -> list[EngineeredFeature]`.
- Both builders live in the same module.
- Builders are pure functions of (store state, as_of) — no side effects,
  no LLM calls, deterministic.

### Step 8.5 — DAG Integration References
- **DAG pattern** (`agent/pipeline/dag.py`):
  `DAG(name, schedule)` + `dag.add(node_id, operator, params, depends_on)`.
  Operators: string (tool name) or callable (FunctionOperator).
  FunctionOperator signature: `fn(params: dict, upstream: dict) -> dict`.
- **Convergence DAG as precedent** (`agent/pipeline/dags/convergence_detection.py`):
  Single `run_detection` node, FunctionOperator callback that opens
  PipelineStore, runs detection, emits signals, returns summary dict.
  Scheduled 30 min after daily_collection.
- **DAG registry** (`agent/pipeline/dags/__init__.py`):
  `get_default_dags(tool_registry)` returns all built-in DAGs.
  New DAG module must be imported and appended there.
- **Design decisions for 8.5**:
  - New module: `agent/pipeline/dags/feature_generation.py`.
  - Single `generate_features` node (FunctionOperator).
  - Depends on: nothing (reads from store; convergence_detection DAG
    runs first by schedule, but features are decoupled — they just read
    whatever is in the store).
  - Schedule: weekdays at 19:00 UTC (30 min after convergence_detection).
  - The callback instantiates both builders, calls `build()`, persists
    results via `store.store_features_batch()`, returns summary.
  - Deterministic, idempotent: re-running for the same `as_of` overwrites
    features via the unique constraint.

### Step 8.6 — Final Edge-Case Test Suite
- Integration test: build DAG, validate, execute callback with mock store.
- Test: feature generation with no convergence signals and no macro data.
- Test: feature generation with partial data (convergence only, macro only).
- Test: idempotent re-run produces same results.
- Test: features emitted pass protocol validation and persist correctly.
- Test: DAG is registered in get_default_dags.

---

## Related

- [[signal_protocol_feature_engineering_spec|Spec: Signal Protocol Feature Engineering]]
- [[convergence_detection]]
- [[world_model]]
- [[backtest_performance]]
