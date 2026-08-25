---
title: "Spec: Tier 8 — Autonomous Discovery & Self-Extending Ontology"
tags:
  - doc/spec
  - phase/25
  - topic/autonomous-discovery
  - topic/entity-ontology
  - topic/self-improving
  - layer/surveillance
  - layer/feature-engineering
  - layer/world-model
---

# Spec: Tier 8 — Autonomous Discovery & Self-Extending Ontology

**Date:** 2026-04-16
**Research:** [[tier8_autonomous_discovery]]
**Master Spec:** [[learned_vs_handcoded_architecture_spec]]
**Goal:** Implement Changes 15 + 16. Move from 90% → 95% learned.

---

## Goal

1. **Change 15 — Autonomous data source discovery**: The system finds new free
   APIs and data feeds, evaluates their signal content via mutual information,
   and wires them into the pipeline as template-based tools — without human
   intervention.

2. **Change 16 — Self-extending entity ontology**: New entity types and
   relationship types emerge from observation patterns. The fixed `Literal`
   type vocabulary becomes a dynamic runtime registry that grows as the system
   encounters novel structure.

---

## Files Affected

### New Files
| File | Purpose |
|------|---------|
| `agent/discovery/__init__.py` | Package init |
| `agent/discovery/source_scout.py` | Change 15: Catalog search + source probing |
| `agent/discovery/signal_evaluator.py` | Change 15: MI-based signal assessment |
| `agent/discovery/tool_factory.py` | Change 15: Template-based tool generation |
| `agent/discovery/ontology_registry.py` | Change 16: Dynamic entity type + link type registry |
| `agent/discovery/type_inducer.py` | Change 16: Entity type & relationship discovery |
| `tests/test_tier8_source_discovery.py` | Change 15 tests |
| `tests/test_tier8_ontology.py` | Change 16 tests |

### Modified Files
| File | Change |
|------|--------|
| `agent/pipeline/store.py` | Add `entity_type_registry`, `unresolved_entities`, `discovered_sources` tables |
| `agent/pipeline/entity.py` | Replace `Literal` EntityType with runtime validation via OntologyRegistry |
| `agent/features/gnn_builder.py` | Make `_CONNECTED_TYPES` dynamic from OntologyRegistry |
| `agent/learning/tool_router.py` | Support dynamic arm addition/removal |
| `agent/cli.py` | Load discovered tools on startup |
| `agent/pipeline/dags/daily_collection.py` | Add quarantine check for newly discovered tools |

---

## Implementation Steps

### Change 15: Autonomous Data Source Discovery (Steps 15.1–15.7)

#### 15.1: Add discovery tables to PipelineStore

**File:** `agent/pipeline/store.py`

Add three new tables to `_SCHEMA`:

```sql
CREATE TABLE IF NOT EXISTS discovered_sources (
    source_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    url TEXT NOT NULL,
    description TEXT,
    format TEXT NOT NULL,       -- 'json_api', 'csv_feed', 'html_table'
    update_frequency TEXT,      -- 'daily', 'weekly', 'monthly', 'unknown'
    topic_tags_json TEXT,
    probe_result_json TEXT,     -- sample data from probe
    mi_score REAL,              -- mutual information score
    status TEXT NOT NULL DEFAULT 'discovered',  -- discovered|quarantine|active|disabled
    discovered_at REAL NOT NULL,
    promoted_at REAL,
    tool_config_json TEXT,      -- ToolFactory config for regenerating the tool
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS unresolved_entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_text TEXT NOT NULL,
    source_tool TEXT NOT NULL,
    context_snippet TEXT,
    observed_at REAL NOT NULL,
    cluster_id INTEGER,
    resolved_type TEXT,
    resolved_at REAL
);

CREATE INDEX IF NOT EXISTS idx_unresolved_cluster
    ON unresolved_entities(cluster_id);

CREATE TABLE IF NOT EXISTS entity_type_registry (
    type_name TEXT PRIMARY KEY,
    parent_type TEXT,
    discovered_at REAL NOT NULL,
    source TEXT NOT NULL,         -- 'seed', 'induced', 'manual'
    confidence REAL NOT NULL DEFAULT 1.0,
    active INTEGER NOT NULL DEFAULT 1,
    metadata_json TEXT
);
```

Add helper methods:
- `store_discovered_source(source_id, name, url, ...)` — INSERT OR IGNORE
- `update_source_status(source_id, status)` — UPDATE status
- `query_discovered_sources(status=None)` → list of dicts
- `store_unresolved_entity(raw_text, source_tool, context, observed_at)`
- `query_unresolved_entities(cluster_id=None, resolved=False)` → list of dicts
- `register_entity_type(type_name, parent_type, source, confidence)` — INSERT OR IGNORE
- `query_entity_types(active_only=True)` → list of dicts

**Test:** Table creation, CRUD operations, status transitions.

---

#### 15.2: Create DataSourceCandidate and SourceScout

**File:** `agent/discovery/source_scout.py` (new)

```python
@dataclass
class DataSourceCandidate:
    source_id: str       # SHA-256[:16] of url
    name: str
    url: str
    description: str
    format: str          # 'json_api' | 'csv_feed'
    update_frequency: str
    topic_tags: list[str]
    probe_sample: dict | list | None = None  # Sample data from probe
    relevance_score: float = 0.0
```

`SourceScout` class:
- `__init__(catalog_urls, existing_tool_urls, topic_vocabulary)`
  - `catalog_urls`: list of API catalog endpoints to search
    - Default: `["https://catalog.data.gov/api/3/action/package_search"]`
  - `existing_tool_urls`: set of URLs already registered (to skip duplicates)
  - `topic_vocabulary`: set of strings (entity types + feature name prefixes)
    used for relevance scoring
- `search(query_terms, max_results=20) → list[DataSourceCandidate]`
  - Query each catalog API with the given terms
  - Parse response into `DataSourceCandidate` objects
  - Filter: format must be JSON or CSV; must have update_frequency
  - Score relevance via TF-IDF cosine similarity of description vs topic_vocabulary
  - Sort by relevance, return top `max_results`
- `probe(candidate) → DataSourceCandidate` (mutates probe_sample)
  - HTTP GET the URL with timeout=30s, max response 1MB
  - Parse based on format (JSON → dict, CSV → first 100 rows as dicts)
  - If parse fails, set `probe_sample = None` (source is invalid)
  - Return updated candidate
- `search_and_probe(query_terms, max_results=10) → list[DataSourceCandidate]`
  - Convenience: search → filter relevance ≥ 0.1 → probe top N → return probed

**Test:** Mock catalog responses, probe success/failure, relevance scoring.

---

#### 15.3: Create SignalEvaluator

**File:** `agent/discovery/signal_evaluator.py` (new)

`SignalEvaluator` class:
- `__init__(store: PipelineStore, min_samples=50, mi_threshold=0.05)`
- `evaluate(candidate: DataSourceCandidate) → SignalReport`
  - Extract time series from `candidate.probe_sample` (detect timestamp
    column + numeric columns heuristically)
  - For each numeric column in the probe sample and each feature series
    in store (query recent features via `store.query_features()`):
    - Align by timestamp (nearest-neighbor join within 24h tolerance)
    - If aligned length < `min_samples`: skip (insufficient data)
    - Compute MI using `sklearn.feature_selection.mutual_info_regression`
    - Normalize: `mi_normalized = mi / max(entropy_source, eps)`
  - Return `SignalReport(max_mi, mean_mi, best_feature_pair, n_aligned, pass_threshold)`

```python
@dataclass
class SignalReport:
    max_mi: float           # Best MI across all feature pairs
    mean_mi: float          # Mean MI
    best_pair: tuple[str, str]  # (source_column, feature_name)
    n_aligned: int          # Number of aligned samples
    passes_threshold: bool  # max_mi > mi_threshold
    details: dict           # Per-pair MI scores
```

**Math:**
- $I(X;Y) = H(X) + H(Y) - H(X,Y)$
- Estimator: KSG (k=3 neighbors, default in sklearn)
- Normalization: $I_{\text{norm}} = I(X;Y) / H(X)$ where $H(X)$ estimated
  via `mutual_info_regression(X, X)` (self-information ≈ entropy)
- Threshold 0.05 = 5% of source entropy is predictive → worth wiring

**Test:** Synthetic correlated time series → positive MI. Random → MI ≈ 0.

---

#### 15.4: Create ToolFactory

**File:** `agent/discovery/tool_factory.py` (new)

`ToolFactory` class:
- `__init__(config_dir=".tirra_pipeline/discovered_tools/")`
- `create_tool(candidate: DataSourceCandidate, signal_report: SignalReport) → Tool`
  - Based on `candidate.format`:
    - `'json_api'` → create `DiscoveredJsonApiTool(url, response_path, field_mapping)`
    - `'csv_feed'` → create `DiscoveredCsvFeedTool(url, delimiter, field_mapping)`
  - Auto-detect `response_path` (JSONPath to data array) from probe_sample structure
  - Auto-detect `field_mapping` from column names → observation schema

`DiscoveredJsonApiTool(Tool)`:
- `name` = `f"discovered_{candidate.source_id[:8]}"`
- `description` = candidate.description (truncated to 200 chars)
- `parameters` = minimal JSON Schema (query params if detected, else empty)
- `execute(**kwargs) → ToolResult`:
  - HTTP GET `self._url` with timeout=30s, max_response=1MB
  - Navigate response via `self._response_path` (list of keys)
  - Map fields via `self._field_mapping`
  - Return `ToolResult(success=True, data=mapped_rows)`
- `to_config() → dict` — serializable config for persistence
- `from_config(config) → DiscoveredJsonApiTool` — class method for reload

`DiscoveredCsvFeedTool(Tool)`: Same pattern for CSV sources.

Config persistence:
- `save_config(tool) → Path`: Write JSON to `config_dir/{tool.name}.json`
- `load_all_configs() → list[Tool]`: Reload all persisted discovered tools

**Test:** Create tool from mock candidate, execute with mock HTTP, serialize/deserialize config.

---

#### 15.5: Add dynamic arm support to ToolRoutingBandit

**File:** `agent/learning/tool_router.py`

Add method:
- `add_arm(tool_name: str) → None`
  - Add new tool with uniform prior Beta(1,1)
  - Update `_tool_names`, `_alpha`, `_beta`, `_pulls`, `_total_reward`
  - Persist state if `_persist_path` is set
- `remove_arm(tool_name: str) → None`
  - Remove tool from all tracking dicts
  - Persist state

**Test:** Add arm, verify it appears in `decide()`. Remove arm, verify it's gone.

---

#### 15.6: Wire discovery into pipeline

**File:** `agent/cli.py`

In `build_tool_registry()`, after registering all hardcoded tools:
- Check for `.tirra_pipeline/discovered_tools/` directory
- If exists, load all tool configs via `ToolFactory.load_all_configs()`
- Register each discovered tool that has `status='active'` in store
- Log count of loaded discovered tools

**File:** `agent/pipeline/dags/daily_collection.py`

Add quarantine logic: when DAG runs, for each discovered tool in `quarantine`
status:
- Execute tool
- Store observations as shadowed (not flowing to features/models)
- After N successful quarantine cycles (default 5), promote to `active`
- If consecutive_failures > 3, move to `disabled`

No new DAG node for discovery itself — discovery runs on a separate schedule
(weekly) and can be triggered as a standalone function.

**Test:** Quarantine promotion flow, failure disabling.

---

#### 15.7: Create discovery orchestration function

**File:** `agent/discovery/source_scout.py`

Add top-level function:
```python
def run_source_discovery(
    store: PipelineStore,
    registry: ToolRegistry,
    bandit: ToolRoutingBandit,
    *,
    query_terms: list[str] | None = None,
    max_new_tools: int = 1,
) -> list[str]:
    """Run one discovery cycle. Returns list of newly created tool names."""
```

Pipeline:
1. Build query terms from existing entity types + feature name prefixes if not provided
2. Build existing_tool_urls from `query_discovered_sources()`
3. SourceScout.search_and_probe(query_terms)
4. For each probed candidate:
   - Skip if URL already known
   - SignalEvaluator.evaluate(candidate)
   - If passes threshold: ToolFactory.create_tool(), save config,
     store_discovered_source() with status='quarantine',
     bandit.add_arm(tool_name)
   - Stop after `max_new_tools` new tools
5. Return created tool names

**Test:** End-to-end with mocked HTTP. Discovery → evaluation → creation.

---

### Change 16: Self-Extending Entity Ontology (Steps 16.1–16.6)

#### 16.1: Create OntologyRegistry

**File:** `agent/discovery/ontology_registry.py` (new)

`OntologyRegistry` class:
- `__init__(store: PipelineStore)`
  - Load seed types from DB: `SELECT DISTINCT entity_type FROM entities`
  - Load registered types from `entity_type_registry` table
  - Merge into `_known_types: dict[str, TypeInfo]`
- `known_entity_types() → frozenset[str]`
- `known_link_types() → frozenset[str]` (from `SELECT DISTINCT link_type FROM entity_links`)
- `is_valid_type(type_name: str) → bool`
- `register_type(type_name, parent_type=None, source='induced', confidence=1.0) → bool`
  - Validate: non-empty, lowercase, alphanumeric + underscore
  - Insert into `entity_type_registry` table
  - Add to `_known_types`
  - Return True if new, False if already existed
- `register_link_type(link_type: str) → bool` — same pattern
- `deactivate_type(type_name) → None` — set active=0 (never delete)
- `type_hierarchy() → dict[str, str | None]` — type_name → parent_type mapping

```python
@dataclass(frozen=True)
class TypeInfo:
    name: str
    parent_type: str | None
    source: str  # 'seed' | 'induced' | 'manual'
    confidence: float
    active: bool
```

**Seed initialization:** On first run, register the 9 existing types with
`source='seed', confidence=1.0`.

**Test:** CRUD, seed initialization, validation rules, hierarchy queries.

---

#### 16.2: Replace Literal EntityType with runtime validation

**File:** `agent/pipeline/entity.py`

Replace:
```python
EntityType = Literal[
    "company", "country", "domain", "organization",
    "person", "protocol", "topic", "vessel", "wallet",
]
```

With:
```python
# Seed entity types — always valid even without DB access
SEED_ENTITY_TYPES: frozenset[str] = frozenset({
    "company", "country", "domain", "organization",
    "person", "protocol", "topic", "vessel", "wallet",
})

# Runtime type alias for annotation — accepts any string
EntityType = str

def validate_entity_type(
    entity_type: str,
    registry: OntologyRegistry | None = None,
) -> bool:
    """Check if entity_type is a known type.

    Without a registry, falls back to SEED_ENTITY_TYPES.
    """
    if entity_type in SEED_ENTITY_TYPES:
        return True
    if registry is not None:
        return registry.is_valid_type(entity_type)
    return False
```

This is backward compatible: all existing code that passes one of the 9
seed types continues to work. New dynamically-discovered types require a
registry reference to validate.

**Test:** Seed types always valid. Dynamic types valid with registry, invalid without.

---

#### 16.3: Create TypeInducer

**File:** `agent/discovery/type_inducer.py` (new)

`TypeInducer` class:
- `__init__(store, registry, *, min_cluster_size=5, cohesion_threshold=0.6)`
- `ingest_unresolved(raw_text, source_tool, context, observed_at) → None`
  - Store in `unresolved_entities` table
- `run_induction() → list[str]` (returns newly created type names)
  - **Step 1: Cluster unresolved entities**
    - Group by `source_tool` first (entities from same source likely share type)
    - Within each source group, cluster by observation field similarity:
      - Extract field keys from `context_snippet` (parsed JSON keys or text patterns)
      - Jaccard similarity between field key sets
      - Agglomerative clustering with distance threshold
    - Assign `cluster_id` to each unresolved entity
  - **Step 2: Evaluate clusters**
    - For each cluster with ≥ `min_cluster_size` entities:
      - Compute silhouette score vs. other clusters
      - If cohesion > threshold:
        - Derive type name: `f"{source_tool}_{dominant_field}"` (most common
          distinguishing field name), cleaned to `[a-z][a-z0-9_]*`
        - Check overlap with existing types (>50% entity overlap → merge, not new type)
        - Propose type
  - **Step 3: Register approved types**
    - Register in OntologyRegistry
    - Reclassify: update `resolved_type` and `resolved_at` for all entities
      in the cluster
    - Re-register entities in main `entities` table with new type

**Test:** Synthetic unresolved entities → clustering → type proposal. Overlap
detection blocks duplicate types.

---

#### 16.4: Add relationship induction to TypeInducer

**File:** `agent/discovery/type_inducer.py`

Add method:
- `induce_relationships() → list[tuple[str, str, str]]` (returns (type_a, type_b, link_type) triples)
  - Query entity co-occurrences: entities appearing in same source_tool
    observations within a time window (default 24h)
  - Count co-occurrence frequency per (entity_type_a, entity_type_b) pair
  - For pairs exceeding frequency threshold (default 10):
    - Compute pointwise MI of co-occurrence vs independent occurrence
    - If MI > threshold: propose link_type = `f"{source_tool}_cooccurrence"`
    - Validate via BIC scoring (reuse EdgeConfidenceTracker logic):
      add proposed edge to world model DAG, check BIC improvement
  - Register validated link types in OntologyRegistry

**Test:** Synthetic co-occurring entities → relationship detection. Random
co-occurrence → no relationship.

---

#### 16.5: Make GNN _CONNECTED_TYPES dynamic

**File:** `agent/features/gnn_builder.py`

Replace:
```python
_CONNECTED_TYPES: tuple[str, ...] = (
    "person", "company", "wallet", "country", "vessel",
)
```

With dynamic resolution:
```python
_SEED_CONNECTED_TYPES: tuple[str, ...] = (
    "person", "company", "wallet", "country", "vessel",
)

def get_connected_types(
    store: PipelineStore,
    registry: OntologyRegistry | None = None,
    min_entities: int = _MIN_ENTITIES_DEFAULT,
    min_links: int = 1,
) -> tuple[str, ...]:
    """Return entity types eligible for GNN embedding.

    A type is eligible if it has ≥ min_entities entities AND ≥ min_links
    links to other connected types.
    """
    if registry is None:
        return _SEED_CONNECTED_TYPES
    # Start with seed types
    candidates = set(_SEED_CONNECTED_TYPES)
    # Add dynamically discovered types that meet criteria
    for type_info in registry.query_entity_types(active_only=True):
        t = type_info["type_name"]
        if t in candidates:
            continue
        entity_count = len(store.query_all_entities(entity_type=t))
        if entity_count < min_entities:
            continue
        # Check for links to existing connected types
        # (any link involving this type and a type already in candidates)
        has_link = False
        for existing in candidates:
            links = store.query_entity_links(entity_type_a=t)
            links += store.query_entity_links(entity_type_b=t)
            if len(links) >= min_links:
                has_link = True
                break
        if has_link:
            candidates.add(t)
    return tuple(sorted(candidates))
```

Update `GNNFeatureBuilder.build()` to call `get_connected_types()` instead
of referencing the static tuple.

**Test:** Default returns seed types. With registry + entities + links, returns
expanded set.

---

#### 16.6: Wire ontology into pipeline startup

**File:** `agent/cli.py`

In `build_tool_registry()` (or a new `build_pipeline_context()` function):
- Create `OntologyRegistry(store)` early
- Pass registry to any component that needs type validation
- On first run, seed the `entity_type_registry` table with the 9 known types

**File:** `agent/pipeline/entity.py`

Add module-level convenience:
```python
_GLOBAL_REGISTRY: OntologyRegistry | None = None

def set_ontology_registry(registry: OntologyRegistry) -> None:
    global _GLOBAL_REGISTRY
    _GLOBAL_REGISTRY = registry

def get_ontology_registry() -> OntologyRegistry | None:
    return _GLOBAL_REGISTRY
```

This allows `validate_entity_type()` to access the registry without threading
it through every call site.

**Test:** Startup seeds registry. Global accessor works.

---

## Edge Cases

### Change 15
1. **Catalog API down**: SourceScout.search() returns empty list, no crash
2. **Probe returns non-parseable data**: probe_sample = None, candidate filtered out
3. **Probe returns huge response**: Enforce 1MB cap, truncate
4. **All candidates below MI threshold**: No new tools created, log summary
5. **Duplicate URL discovered**: Skip if URL exists in discovered_sources
6. **Discovered tool fails at runtime**: Increment consecutive_failures, auto-disable at 3
7. **Max tool count reached**: Refuse new discovery, log warning
8. **Empty feature store** (no existing features for MI): Skip MI evaluation, defer source

### Change 16
1. **No unresolved entities**: run_induction() returns empty list
2. **Cluster too small**: Ignored until more entities accumulate
3. **Type name collision with seed type**: Merge into existing type, don't create new
4. **Type name collision with previously induced type**: Increment suffix
5. **Parent type not found**: Register with parent_type=None
6. **Co-occurrence is spurious**: BIC scoring filters it out
7. **Entity reassignment after type creation**: Handle concurrent observations gracefully
8. **GNN schema change mid-inference**: Snapshot approach — only update connected types
   at next GNN retrain, not mid-cycle
9. **Deactivated type reactivated**: Registry supports reactivation

---

## Testing Plan

### Unit Tests (`tests/test_tier8_source_discovery.py`)
1. PipelineStore: discovered_sources CRUD
2. PipelineStore: unresolved_entities CRUD
3. PipelineStore: entity_type_registry CRUD
4. SourceScout: search with mock catalog, relevance scoring
5. SourceScout: probe success + failure
6. SignalEvaluator: synthetic correlated data → positive MI
7. SignalEvaluator: random data → MI ≈ 0, fails threshold
8. SignalEvaluator: insufficient samples → skip
9. ToolFactory: create JSON API tool from candidate
10. ToolFactory: create CSV feed tool from candidate
11. ToolFactory: config serialization round-trip
12. ToolRoutingBandit: add_arm / remove_arm
13. Discovery orchestration: end-to-end mocked flow
14. Quarantine: promotion + failure disabling

### Unit Tests (`tests/test_tier8_ontology.py`)
1. OntologyRegistry: seed initialization
2. OntologyRegistry: register + query + deactivate
3. OntologyRegistry: type hierarchy
4. OntologyRegistry: validation rules
5. validate_entity_type: seed types always valid
6. validate_entity_type: dynamic types with/without registry
7. TypeInducer: clustering unresolved entities
8. TypeInducer: type proposal from cluster
9. TypeInducer: overlap detection prevents duplicate types
10. TypeInducer: relationship induction from co-occurrence
11. get_connected_types: default seed types
12. get_connected_types: dynamic expansion
13. get_connected_types: type below entity threshold excluded
14. Global registry accessor: set/get pattern

---

## Related

- [[tier8_autonomous_discovery]] — Research
- [[learned_vs_handcoded_architecture_spec]] — Master spec
- [[tier7_self_modifying_structure_spec]] — Previous tier spec
- [[tier7_self_modifying_structure]] — Previous tier research
- [[gnn_guided_tool_expansion]] — GNN-guided expansion context
- [[entity_linking_layer]] — Entity infrastructure
- [[temporal_het_gnn]] — GNN architecture
