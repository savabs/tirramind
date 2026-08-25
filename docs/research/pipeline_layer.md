---
title: "Feature: Pipeline Layer (Deterministic DAG Scheduler)"
tags:
  - doc/research
  - topic/pipeline
---

# Feature: Pipeline Layer (Deterministic DAG Scheduler)

## Problem Statement

TirraMind currently has one execution engine: the **Agent Layer** (Orchestrator). It's LLM-driven, adaptive, exploratory — great for research and hypothesis testing. But it's completely wrong for production data collection and signal processing:

1. **No scheduling** — every run requires a human to type a goal or the autonomous loop to generate one via LLM
2. **No persistent state** — tools are stateless, results stored only in cache (6hr TTL) and memory (text-based JSONL)
3. **No dependency management** — tasks execute depth-first, no parallel execution, no explicit DAG
4. **LLM in the loop** — every tool call goes through LLM planning, which is non-deterministic, slow (~1-3s per call), and unnecessary for routine data collection
5. **No structured storage** — all data lives as JSON files in `.tirra_cache/` with SHA256 keys, no queryable schema

Phase 7b data sources (Polymarket whale tracker, on-chain indexing, congressional trading) require:
- Scheduled polling (every 15 min, every hour, every Friday)
- Persistent accumulation (wallet scores built over weeks of observation)
- Deterministic execution (same DAG runs same way every time)
- Structured storage (queryable tables, not flat JSON files)

## Current Architecture

### Agent Layer (what exists)
```
User/Bandit → Goal → LLM Research → LLM Plan → Sequential Tool Execution → LLM Synthesize
```
- Entry: `agent/cli.py` → `agent/core/orchestrator.py`
- Tools: `agent/tools/*.py` (19 tools, all stateless, all return `ToolResult(success, output, data)`)
- Cache: `agent/data/cache.py` (file-based, SHA256 keys, 6hr TTL)
- Memory: `agent/memory/store.py` (episodic JSONL + semantic JSONL + working memory in-RAM)
- Config: `agent/config/settings.py` (env vars, `TIRRA_` prefix, frozen dataclasses)

### What Pipeline Layer adds (second engine)
```
Scheduler → DAG → Deterministic Tool Execution → Structured Storage → Signal Output
```
No LLM anywhere. No adaptive replanning. Pure code, pure math, pure data.

## Design Decisions

### D1: No external workflow engine

Could use Airflow, Prefect, Dagster, Luigi, etc. **Rejected.**

Reasons:
- We need ~200 lines of DAG execution logic, not a distributed workflow platform
- External deps add complexity, deployment burden, version conflicts
- Our DAGs are small (10-50 nodes), not 10,000-node ETL pipelines
- We need tight integration with existing Tool ABC, DataCache, and config patterns
- Single-machine execution (for now); can add Ray later for parallelism

**Decision: Build a minimal DAG executor from scratch.** Python stdlib + existing deps only.

### D2: SQLite for persistent storage

Need structured, queryable storage for:
- Accumulated wallet scores (Phase 7b-A)
- Historical data series (fetched incrementally, appended over time)
- Pipeline run metadata (what ran, when, success/failure, duration)
- Signal outputs (features ready for downstream model consumption)

Options considered:
- **PostgreSQL**: Too heavy for single-machine, requires separate server process
- **DuckDB**: Great for analytics but adds dep, less common for transactional writes
- **SQLite**: Built into Python stdlib, zero config, single file, ACID, up to ~1TB
- **Parquet files**: Good for columnar reads but no transactional writes, no schema enforcement

**Decision: SQLite.** Zero dependencies, ACID, queryable, single file at `.tirra_pipeline/pipeline.db`. Can migrate to Postgres when/if we need multi-machine.

### D3: DAG defined in Python, not YAML/JSON

DAGs are code. They need:
- Type safety (what params does each node take?)
- Conditional logic (skip if already fetched today)
- Computed parameters (yesterday's date, dynamic ticker lists)

```python
# Example: daily data collection DAG
dag = DAG("daily_collection", schedule="0 18 * * 1-5")  # 6pm weekday

fetch_cftc = dag.add("fetch_cftc", tool="cftc", params={"mode": "latest"})
fetch_finra = dag.add("fetch_finra", tool="finra_short_volume", params={"mode": "scan", "date": "latest"})
fetch_grid = dag.add("fetch_grid", tool="power_grid", params={"mode": "demand"})

# These three run in parallel (no deps)
# Then downstream:
compute_signals = dag.add("compute_signals", operator=compute_daily_signals, depends_on=[fetch_cftc, fetch_finra, fetch_grid])
```

**Decision: Python-native DAG definition.** Decorator/builder pattern, not config files.

### D4: Operators vs. Tools

The Agent Layer calls `Tool.execute(**kwargs) → ToolResult`. The Pipeline Layer needs the same tools but also needs:
- **Operators** that are not tools (pure Python functions for signal computation, data transformation)
- **Storage writes** (tool results → SQLite, not just cache)
- **Richer return types** (DataFrames, numpy arrays, not just `ToolResult.output` strings)

**Decision: Operators wrap tools.** An operator is either:
1. A `ToolOperator` that delegates to an existing `Tool` and stores the `ToolResult.data` in SQLite
2. A `FunctionOperator` that runs a pure Python function (signal computation, aggregation)

Both produce typed outputs that downstream operators consume via the DAG.

### D5: Scheduler — cron-like, in-process

Options:
- **System crontab**: Not portable, no visibility, can't query status
- **APScheduler**: Mature, in-process, cron syntax. Python library. ~500 lines.
- **Custom scheduler**: Simple loop + cron expression parser

**Decision: APScheduler.** One new dependency, well-maintained, handles cron parsing, timezone, job persistence. Run in background thread or as a service.

Fallback: if we want zero deps, stdlib `sched` + a simple cron parser (< 100 lines). But APScheduler is proven.

### D6: Shared state between Agent Layer and Pipeline Layer

Both engines need to read the same data. The Pipeline Layer writes structured data to SQLite. The Agent Layer needs to query it.

**Decision: Add a `PipelineStore` class** that both engines can import. The Agent Layer gets a new tool (`pipeline_query`) that queries Pipeline Store. The Pipeline Layer writes directly.

```
Pipeline Layer → writes → SQLite (.tirra_pipeline/pipeline.db)
Agent Layer → reads (via pipeline_query tool) → same SQLite
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     TirraMind System                        │
│                                                             │
│  ┌──────────────────────┐    ┌──────────────────────────┐  │
│  │    Agent Layer        │    │    Pipeline Layer         │  │
│  │  (LLM-driven)        │    │  (deterministic)          │  │
│  │                       │    │                           │  │
│  │  Orchestrator         │    │  Scheduler (APScheduler)  │  │
│  │  ├─ Research (LLM)    │    │  ├─ DAG Registry          │  │
│  │  ├─ Plan (LLM)        │    │  ├─ DAG Executor          │  │
│  │  ├─ Execute (Tools)   │    │  │  ├─ Topo sort          │  │
│  │  └─ Synthesize (LLM)  │    │  │  ├─ Parallel exec      │  │
│  │                       │    │  │  └─ Retry + backoff     │  │
│  │  Memory Store         │    │  ├─ Pipeline Store (SQL)   │  │
│  │  ├─ Episodic          │    │  └─ Run Tracker           │  │
│  │  ├─ Semantic          │    │                           │  │
│  │  └─ Working           │    │                           │  │
│  └────────┬──────────────┘    └────────┬──────────────────┘  │
│           │                            │                     │
│           │    ┌──────────────┐        │                     │
│           └────┤  Shared      ├────────┘                     │
│                │              │                               │
│                │  Tool ABC    │  (same tool implementations)  │
│                │  DataCache   │  (same file cache)            │
│                │  Config      │  (same env vars)              │
│                │  SQLite DB   │  (pipeline writes, agent reads)│
│                └──────────────┘                               │
└─────────────────────────────────────────────────────────────┘
```

## Component Inventory

### New files to create

| File | Purpose |
|------|---------|
| `agent/pipeline/__init__.py` | Package init |
| `agent/pipeline/dag.py` | DAG, Node, DagBuilder classes |
| `agent/pipeline/executor.py` | DAGExecutor — topo sort, parallel execution, retry |
| `agent/pipeline/scheduler.py` | Scheduler — wraps APScheduler, cron triggers, DAG registry |
| `agent/pipeline/store.py` | PipelineStore — SQLite interface for structured data |
| `agent/pipeline/operators.py` | ToolOperator, FunctionOperator base classes |
| `agent/pipeline/registry.py` | DAG registry — all defined DAGs, lookup by name |
| `agent/pipeline/dags/` | Directory for DAG definitions |
| `agent/pipeline/dags/__init__.py` | Package init |
| `agent/pipeline/dags/daily_collection.py` | Example: daily data fetch DAG |
| `agent/tools/pipeline_query.py` | Agent-side tool to query Pipeline Store |

### Files to modify

| File | Change |
|------|--------|
| `agent/cli.py` | Add `--pipeline` mode, register `pipeline_query` tool |
| `agent/config/settings.py` | Add `PipelineConfig` (db_path, scheduler settings) |
| `pyproject.toml` | Add `apscheduler>=3.10` dependency |

## Core Data Model

### DAG Node
```python
@dataclass
class Node:
    id: str                          # Unique within DAG (e.g., "fetch_cftc")
    operator: str                    # Tool name or function reference
    params: dict                     # Static params passed to operator
    depends_on: list[str] = []       # Node IDs this depends on
    retries: int = 1                 # Max retry count on failure
    timeout: int = 60                # Per-node timeout (seconds)
    store_result: bool = True        # Write result to Pipeline Store
    table_name: str | None = None    # SQLite table for result storage
```

### DAG
```python
@dataclass
class DAG:
    name: str                        # Unique identifier (e.g., "daily_collection")
    schedule: str | None = None      # Cron expression (None = manual trigger only)
    nodes: dict[str, Node] = {}      # Node registry
    description: str = ""
    
    def add(self, id, operator, params=None, depends_on=None, **kwargs) -> str:
        # Create node, register, return id
    
    def validate(self) -> list[str]:
        # Check: no cycles, all depends_on exist, no self-deps
    
    def topo_sort(self) -> list[list[str]]:
        # Return execution layers: [[roots], [layer1], [layer2], ...]
        # Each layer can execute in parallel
```

### DAG Run (execution record)
```python
@dataclass
class DagRun:
    run_id: str                      # UUID
    dag_name: str
    started_at: float                # Unix timestamp
    finished_at: float | None
    status: str                      # "running" | "completed" | "failed"
    node_results: dict[str, NodeResult]  # Per-node outcomes
    trigger: str                     # "scheduled" | "manual"

@dataclass
class NodeResult:
    node_id: str
    status: str                      # "pending" | "running" | "completed" | "failed" | "skipped"
    started_at: float | None
    finished_at: float | None
    output: Any                      # ToolResult.data or function return value
    error: str | None                # Error message if failed
    retries_used: int
```

### Pipeline Store (SQLite schema)
```sql
-- Run metadata
CREATE TABLE dag_runs (
    run_id TEXT PRIMARY KEY,
    dag_name TEXT NOT NULL,
    started_at REAL NOT NULL,
    finished_at REAL,
    status TEXT NOT NULL,
    trigger TEXT NOT NULL,
    node_results_json TEXT    -- JSON blob of all NodeResults
);

-- Generic data storage (tool results → rows)
-- Each tool writes to its own table. Schema per tool:
CREATE TABLE pipeline_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,          -- tool name
    fetched_at REAL NOT NULL,      -- unix timestamp
    params_json TEXT NOT NULL,     -- JSON of fetch params
    data_json TEXT NOT NULL        -- JSON of ToolResult.data
);
CREATE INDEX idx_pipeline_data_source ON pipeline_data(source, fetched_at);

-- Signal outputs (features ready for downstream consumption)
CREATE TABLE signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_name TEXT NOT NULL,
    computed_at REAL NOT NULL,
    value REAL NOT NULL,
    metadata_json TEXT             -- JSON with context
);
CREATE INDEX idx_signals_name ON signals(signal_name, computed_at);
```

## Execution Flow

### Manual run
```
$ tirra-agent --pipeline run daily_collection
```
1. Load DAG from registry
2. Validate (no cycles, deps exist)
3. Topo sort → execution layers
4. For each layer: execute all nodes in parallel (ThreadPoolExecutor)
5. For each node: call operator, store result, update run status
6. Return DagRun with all results

### Scheduled run
```
$ tirra-agent --pipeline start
```
1. Load all DAGs from registry
2. Register each scheduled DAG with APScheduler
3. Scheduler triggers DAG runs at cron times
4. Same execution flow as manual
5. Ctrl+C to stop

### Agent queries pipeline data
```python
# New tool: pipeline_query
ToolResult = pipeline_query_tool.execute(
    source="cftc",
    since="2026-03-20",
    limit=100
)
# Returns structured data from SQLite
```

## Risks

1. **SQLite write contention**: If scheduler + agent both write, SQLite handles with WAL mode (concurrent reads, serialized writes). Acceptable for our scale.
2. **Thread safety**: APScheduler + ThreadPoolExecutor both use threads. SQLite with WAL + `check_same_thread=False` handles this. Operators must not share mutable state.
3. **Schema evolution**: Tool result schemas will change. Use JSON blobs (flexible) for data, typed columns only for metadata (source, timestamp).
4. **APScheduler dep risk**: Well-maintained (10yr+ history), 13K+ GitHub stars, actively developed. Low risk.
5. **Scope creep**: Keep the executor minimal. No distributed execution, no container scheduling, no UI. Just: DAG → topo sort → parallel execute → store results.

## Dependencies

- **APScheduler >=3.10**: Scheduler with cron support. ~5MB installed.
- **sqlite3**: Python stdlib. No additional dep.
- **concurrent.futures**: Python stdlib. ThreadPoolExecutor for parallel node execution.
- All existing deps (httpx, jsonschema, etc.) unchanged.

## What This Enables (Phase 7b+)

Once Pipeline Layer exists:
- **7b-A Polymarket whale tracker**: Pipeline DAG polls CLOB every 15 min, accumulates wallet scores in SQLite
- **7b-B Congressional trading**: Pipeline DAG fetches new disclosures daily
- **7b-C Weather/climate**: Pipeline DAG fetches NOAA alerts, FIRMS fire data hourly
- **7b-D AIS vessels**: Pipeline DAG polls vessel positions
- **All Phase 8+ work**: Signal Protocol reads from Pipeline Store, not cache

---

## Related

- [[pipeline_layer_spec|Spec: Pipeline Layer]]
- [[convergence_detection]]
- [[world_model]]
