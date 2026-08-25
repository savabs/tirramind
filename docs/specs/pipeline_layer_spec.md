---
title: "Spec: Pipeline Layer (Deterministic DAG Scheduler)"
tags:
  - doc/spec
  - topic/pipeline
---

# Spec: Pipeline Layer (Deterministic DAG Scheduler)

## Goal

Build a second execution engine alongside the Agent Layer. The Pipeline Layer is deterministic (no LLM), scheduled (cron triggers), parallel (independent nodes execute concurrently), and persistent (results stored in SQLite). It shares the same Tool implementations with the Agent Layer.

## Files Affected

### Create
| File | Purpose |
|------|---------|
| `agent/pipeline/__init__.py` | Package exports |
| `agent/pipeline/dag.py` | DAG, Node dataclasses + validation + topo sort |
| `agent/pipeline/executor.py` | DAGExecutor — runs a DAG, parallel layers, retry, timeout |
| `agent/pipeline/store.py` | PipelineStore — SQLite wrapper for structured data + run metadata |
| `agent/pipeline/scheduler.py` | PipelineScheduler — wraps APScheduler, registers DAGs, cron triggers |
| `agent/pipeline/operators.py` | ToolOperator + FunctionOperator — bridge between DAG nodes and tools/functions |
| `agent/pipeline/registry.py` | DAGRegistry — static registry of all defined DAGs |
| `agent/pipeline/dags/__init__.py` | DAG definition package |
| `agent/pipeline/dags/daily_collection.py` | First real DAG: daily data collection across all existing tools |
| `agent/tools/pipeline_query.py` | Agent-side tool to query Pipeline Store |
| `tests/test_pipeline_dag.py` | DAG unit tests (validation, topo sort, cycles) |
| `tests/test_pipeline_executor.py` | Executor tests (parallel, retry, timeout, failure) |
| `tests/test_pipeline_store.py` | Store tests (CRUD, query, schema, concurrent access) |
| `tests/test_pipeline_scheduler.py` | Scheduler tests (registration, trigger, integration) |
| `tests/test_pipeline_operators.py` | Operator tests (tool wrapping, function wrapping, error handling) |
| `tests/test_pipeline_query_tool.py` | Pipeline query tool tests |

### Modify
| File | Change |
|------|--------|
| `agent/cli.py` | Add `--pipeline` subcommand (run/start/status), register pipeline_query tool |
| `agent/config/settings.py` | Add `PipelineConfig` dataclass |
| `pyproject.toml` | Add `apscheduler>=3.10` dependency |

## Implementation Steps

### Step 7.1: PipelineStore (SQLite persistence)
**Create `agent/pipeline/__init__.py` and `agent/pipeline/store.py`**

The foundation — everything else writes to or reads from this.

- PipelineStore class with `__init__(db_path: Path)`
- `_init_schema()` — creates tables if not exist (dag_runs, pipeline_data, signals)  
- WAL mode enabled for concurrent read/write
- `record_run_start(run_id, dag_name, trigger) -> None`
- `record_run_end(run_id, status, node_results) -> None`
- `store_data(source, params, data) -> int` — insert into pipeline_data
- `query_data(source, since=None, until=None, limit=100) -> list[dict]`
- `store_signal(signal_name, value, metadata=None) -> int`
- `query_signals(signal_name, since=None, limit=100) -> list[dict]`
- `get_runs(dag_name=None, limit=20) -> list[dict]`
- `get_run(run_id) -> dict | None`
- Context manager support (`__enter__`/`__exit__`)

**Test:** Create in-memory SQLite, store data, query it back, verify schema.

### Step 7.2: DAG + Node data model
**Create `agent/pipeline/dag.py`**

- `Node` dataclass: `id, operator, params, depends_on, retries, timeout, store_result, table_name`
- `DAG` dataclass: `name, schedule, nodes, description`
- `DAG.add(id, operator, params, depends_on, **kwargs) -> str` — register node
- `DAG.validate() -> list[str]` — check: no cycles (Kahn's algorithm), all deps exist, no self-deps, no duplicate IDs, at least one node
- `DAG.topo_sort() -> list[list[str]]` — return execution layers (each layer = set of parallelizable nodes)
- `DAG.roots() -> list[str]` — nodes with no dependencies

**Test:** Build DAGs, validate cycles/missing deps, verify topo sort layers.

### Step 7.3: Operators (Tool + Function wrappers)
**Create `agent/pipeline/operators.py`**

- `Operator` ABC: `execute(params, upstream_results) -> Any`
- `ToolOperator(tool_registry: ToolRegistry)`: looks up tool by name, calls `tool.execute(**params)`, returns `ToolResult.data`
- `FunctionOperator(fn: Callable)`: calls `fn(params, upstream_results)`, returns result
- Both handle exceptions → return error info without crashing

**Test:** Wrap a mock tool, verify delegation. Wrap a function, verify call. Test error handling.

### Step 7.4: DAGExecutor (parallel execution engine)
**Create `agent/pipeline/executor.py`**

- `DAGExecutor(tool_registry, store, max_workers=4)`
- `execute(dag: DAG, trigger="manual") -> DagRun`
  1. Validate DAG
  2. Create DagRun, record start in store
  3. Topo sort → layers
  4. For each layer: submit all nodes to ThreadPoolExecutor
  5. For each node: resolve operator (ToolOperator or FunctionOperator), call with params + upstream results
  6. Retry on failure (up to `node.retries`, exponential backoff)
  7. Timeout enforcement per node
  8. If `node.store_result`: write to PipelineStore
  9. Collect NodeResult per node
  10. Record run end in store
  11. Return DagRun
- `_execute_node(node, upstream_results) -> NodeResult`
- Upstream result injection: `params` can reference `$upstream.node_id` to receive output of a dependency

**Test:** Execute simple DAGs (linear, diamond, wide-parallel). Test retry. Test timeout. Test failure propagation (node fails → dependents skip).

### Step 7.5: PipelineConfig + CLI integration
**Modify `agent/config/settings.py`** — add PipelineConfig
**Modify `agent/cli.py`** — add `--pipeline` subcommand

PipelineConfig:
- `db_path: str` (default `.tirra_pipeline/pipeline.db`)
- `max_workers: int` (default 4)
- `log_level: str` (default "INFO")

CLI additions:
- `tirra-agent --pipeline run <dag_name>` — execute a DAG manually
- `tirra-agent --pipeline list` — show registered DAGs
- `tirra-agent --pipeline status [run_id]` — show run history or specific run
- `tirra-agent --pipeline start` — start scheduler (blocks, Ctrl+C to stop)

**Test:** CLI arg parsing, config from env vars.

### Step 7.6: PipelineScheduler (cron triggers)
**Create `agent/pipeline/scheduler.py`**

- `PipelineScheduler(executor, registry)`
- `start()` — register all DAGs with schedules, start APScheduler
- `stop()` — shutdown gracefully
- `trigger(dag_name)` — manually trigger a DAG run
- Uses APScheduler's `CronTrigger` for cron expressions
- Logs each trigger + completion

**Modify `pyproject.toml`** — add `apscheduler>=3.10` dependency

**Test:** Register DAG with schedule, verify trigger fires, verify graceful shutdown.

### Step 7.7: DAGRegistry + first real DAG
**Create `agent/pipeline/registry.py`** and **`agent/pipeline/dags/daily_collection.py`**

DAGRegistry:
- `register(dag: DAG) -> None`
- `get(name: str) -> DAG | None`
- `list_all() -> list[DAG]`
- `load_defaults(tool_registry)` — register built-in DAGs

daily_collection DAG — fetches data from all existing stateless tools:
- `fetch_cftc` (cftc tool, mode=latest)
- `fetch_finra_scan` (finra_short_volume, mode=scan)
- `fetch_power_demand` (power_grid, mode=demand)
- `fetch_power_fuel` (power_grid, mode=fuel_mix)
- `fetch_gdelt` (gdelt, mode=events)
- `fetch_polymarket` (polymarket, category=all)
- All independent (no deps between them) — execute in parallel
- Schedule: weekdays at 18:00 UTC

**Test:** Registry CRUD, daily_collection DAG validates and topo-sorts correctly.

### Step 7.8: pipeline_query tool (Agent ↔ Pipeline bridge)
**Create `agent/tools/pipeline_query.py`**
**Modify `agent/cli.py`** — register the tool

Tool that lets the Agent Layer query Pipeline Store:
- Modes: `data` (query pipeline_data), `signals` (query signals), `runs` (query dag_runs)
- Params: `source`, `signal_name`, `since`, `until`, `limit`
- Returns ToolResult with formatted output + structured data

**Test:** Store data via PipelineStore, query via tool, verify output.

### Step 7.9: Edge case test suite
**Comprehensive tests covering:**
- DAG: empty dag, single node, 100+ nodes, diamond deps, deep chains, disconnected subgraphs, duplicate node IDs, cycle detection (simple, complex, self-loop)
- Executor: all nodes succeed, all fail, partial failure, timeout per node, retry exhaustion, retry success, upstream failure cascading, empty DAG, concurrent write safety
- Store: concurrent reads/writes (threading), large data blobs, SQL injection prevention (parameterized queries), missing tables recovery, corrupt DB handling, query with no results, query across date ranges, signal storage precision
- Scheduler: invalid cron expressions, DAG not found, scheduler start/stop lifecycle, multiple DAGs same schedule
- Operators: tool not found, tool throws exception, function returns None, function throws, upstream results not available
- CLI: invalid subcommand, missing dag name, pipeline not initialized
- Integration: full flow (define DAG → register → execute → store → query via tool)

## Edge Cases
- SQLite WAL mode must be set on every connection (not just first)
- ThreadPoolExecutor futures must be collected with timeout to prevent hanging
- APScheduler timezone handling (use UTC throughout)
- Node params containing `$upstream.X` syntax must be resolved before execution
- Empty DAG (no nodes) should be a validation error, not a runtime crash
- DAG with all nodes failed should still record the run as "failed" in store
- Pipeline Store path directory must be created if it doesn't exist

## Testing Plan
- Unit tests per component (Steps 7.1-7.8 each have their own tests)
- Step 7.9 is the comprehensive edge case suite
- Integration test: define DAG programmatically → execute → verify data in SQLite → query via pipeline_query tool
- No live network tests required (tools are mocked in DAG tests)
- SQLite tests use `:memory:` for speed, one test with file-based DB for persistence verification

---

## Related

- [[pipeline_layer|Research: Pipeline Layer]]
- [[convergence_detection]]
- [[world_model]]
