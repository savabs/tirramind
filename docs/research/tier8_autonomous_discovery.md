---
title: "Feature: Tier 8 — Autonomous Discovery & Self-Extending Ontology"
tags:
  - doc/research
  - phase/25
  - topic/autonomous-discovery
  - topic/entity-ontology
  - topic/self-improving
  - layer/surveillance
  - layer/feature-engineering
  - layer/world-model
---

# Feature: Tier 8 — Autonomous Discovery & Self-Extending Ontology

Tier 8 is the final tier of the [[learned_vs_handcoded_architecture_spec]], moving
from 90% learned → 95% learned. It contains two changes:

- **Change 15** — Autonomous data source discovery: the agent finds new free APIs,
  scrape targets, and data feeds on its own, evaluates their signal content, and
  wires them into the pipeline without human intervention.
- **Change 16** — Self-extending entity ontology: new entity types and relationship
  types emerge from data rather than being predefined in code; the taxonomy grows
  as the system encounters novel structure.

## Goal

- Remove the last major hand-coded bottleneck: the fixed list of data tools and
  the fixed entity type taxonomy.
- The system should autonomously grow its surveillance surface and its ontological
  vocabulary as the world changes.
- After Tier 8, the only hand-coded residuals are safety constraints, schema
  invariants, API plumbing, textbook equations, and ethical boundaries (the
  "irreducible 5%").

---

## Search Log

### GitHub / OSS Search
- "autonomous API discovery agent" → DataAgents framework (arXiv 2509.18710),
  LLM-based tool-calling agents for data collection
- "ontology learning dynamic taxonomy" → Dynamic heterogeneous GNNs (DHGNN),
  LLM-based ontology induction from text
- "automated data source signal evaluation" → Online feature selection
  (OFFESEL, OSSFS), mutual information estimators
- "LLM agent autonomous web scraping" → Firecrawl, ScrapeGraphAI for
  structured extraction from arbitrary pages
- "graph neural network dynamic node type" → Dynamic heterogeneous graph
  neural networks (Kazemi 2022, GNN Book ch.15)
- "public API catalog registry data.gov" → data.gov CKAN API, Nasdaq Data
  Link, IMF/Census/BLS API portals

### Academic References
- Thompson 1933 / Agrawal & Goyal 2013 — Thompson Sampling (already used
  in ToolRoutingBandit, meta-scheduler)
- Kazemi 2022, "Dynamic Graph Neural Networks" — GNN Book ch.15: framework
  for DTDGs/CTDGs where node/edge types evolve
- Kendall et al. 2018 — Uncertainty-weighted multi-task loss (already used
  in GNN trainer for loss auto-tuning)
- Kraskov et al. 2004 — k-NN mutual information estimator for continuous
  variables (sklearn.feature_selection.mutual_info_regression)
- arXiv 2509.18710 — DataAgents: LLM reasoning + tool calling for
  autonomous data collection, integration, preprocessing

---

## External Repositories & Documentation Reviewed

### DataAgents (arXiv 2509.18710)
- **Why relevant:** Framework for LLM agents that autonomously find, fetch,
  and integrate new data sources using tool-calling and task decomposition.
- **Useful concept:** Discovery → Evaluate → Wire pipeline. The "3-phase
  data agent" pattern: (1) search for data sources, (2) probe/validate schema,
  (3) generate adapter code.
- **License:** Academic paper, concept only.
- **Reuse conclusion:** Concept only — implement independently.

### Firecrawl / ScrapeGraphAI
- **Why relevant:** LLM-powered structured extraction from arbitrary web
  pages. Could be used to extract data from discovered web sources.
- **Useful concept:** LLM generates extraction schema on first visit, then
  applies deterministically on subsequent visits.
- **License:** Apache-2.0 (Firecrawl), MIT (ScrapeGraphAI).
- **Reuse conclusion:** Concept only for now — we don't want runtime LLM
  dependency for data extraction. But the pattern of schema-generation →
  deterministic-extraction is sound.

### data.gov CKAN API
- **Why relevant:** Programmatic catalog of 300k+ US government datasets.
  Machine-readable metadata (title, description, format, update_frequency).
- **API endpoint:** `https://catalog.data.gov/api/3/action/package_search`
- **Useful concept:** Searchable catalog with structured metadata for
  automated discovery. Can filter by format (CSV, JSON, API), topic,
  update frequency.
- **Reuse conclusion:** Direct API use for source discovery.

### Dynamic Heterogeneous GNNs (Kazemi 2022, GNN Book ch.15)
- **Why relevant:** Framework for GNNs on graphs where node types and edge
  types evolve over time.
- **Key insight:** Two approaches: (1) snapshot-based DTDGs that retrain
  on each new graph snapshot, (2) continuous CTDGs that incrementally
  update. For our use case, snapshot-based (periodic re-embedding when
  ontology changes) is simpler and sufficient.
- **Reuse conclusion:** Concept only — our HetTGN already handles
  heterogeneous types; we just need to make the type set dynamic.

---

## Current Architecture

### Data Tool Layer (Layer 1: Surveillance)
- **61 tools** registered in `build_tool_registry()` at `agent/cli.py:90-165`
- All tools inherit from `Tool` (abstract base at `agent/tools/base.py:33`)
- Interface: `name`, `description`, `parameters` (JSON Schema), `execute(**kwargs) → ToolResult`
- `ToolRegistry`: `register()`, `get()`, `list_names()`, `execute()`, `validate_args()`, `to_openai_tools()`
- `ToolRoutingBandit` (Change 12): Thompson Sampling over 6 optional tools;
  decides which to run each DAG cycle
- Daily collection DAG (`agent/pipeline/dags/daily_collection.py`): 7 nodes
  with bandit-controlled execution

### Entity Layer
- **9 entity types** hardcoded as `Literal` in `agent/pipeline/entity.py:27-36`:
  `company, country, domain, organization, person, protocol, topic, vessel, wallet`
- `entity_id_from_key(entity_type, key) → SHA-256[:16]` deterministic ID
- **4 tables** in `agent/pipeline/store.py`:
  - `entities` (entity_id PK, entity_type TEXT, canonical_name, created_at, metadata_json)
  - `entity_aliases` (source, external_id → entity_id)
  - `entity_observations` (entity_id, source_tool, observed_at, observation_type, depth_level, value_json)
  - `entity_links` (entity_id_a, entity_id_b, link_type TEXT, confidence, source)
- **Note:** `entity_type` in the DB is already free-text TEXT — the `Literal`
  constraint is only at the Python type-checker level. No schema migration needed
  for new entity types at the storage layer.
- Same for `link_type` — already free-text in the DB.

### GNN Layer
- `_CONNECTED_TYPES` in `agent/features/gnn_builder.py:44`: tuple of 5 types
  (`person, company, wallet, country, vessel`)
- HetTGN model uses `entity_type` to build heterogeneous node features
- Adding new types requires: (a) entities of that type exist in DB, (b) links
  exist connecting them, (c) type is in `_CONNECTED_TYPES`

### Feature Layer
- `FeatureBuilder` abstract base (`agent/features/builders.py:45`): `name` + `build(store, as_of)`
- 3 default builders: Convergence, MacroState, GNN
- Feature naming: `^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*(\.[a-z0-9][a-z0-9_]*){0,2}$`

---

## Observations

### What's hand-coded that shouldn't be (Tier 8 targets)
1. **The 61-tool list** — every tool was manually researched, coded, and registered.
   Expanding the surveillance surface requires human effort per tool.
2. **The 9 entity types** — `EntityType = Literal[...]` is a fixed vocabulary.
   If the system encounters a new kind of entity (e.g., "mine", "pipeline",
   "port", "fund"), it can't represent it without code changes.
3. **The link_type vocabulary** — while free-text in the DB, link types used
   in practice are determined by the tools that create them.
4. **`_CONNECTED_TYPES`** — GNN only embeds 5 of 9 entity types. New types
   can't participate in GNN inference without modifying this tuple.

### What's already dynamic (good foundations)
1. **ToolRoutingBandit** — already learns which tools are valuable. Just
   needs expansion to handle a growing tool set.
2. **Entity tables** — `entity_type` and `link_type` are free-text. No
   schema migration needed for new types.
3. **`entity_id_from_key()`** — works with any string entity_type.
4. **EdgeConfidenceTracker** (Tier 7) — already adds/removes edges based on
   BIC scoring. New entity types just need to appear in the DAG.
5. **MetaScheduler** (Tier 7) — Thompson Sampling over components. Can
   naturally extend to new components.

---

## Design Decisions

### Change 15: Autonomous Data Source Discovery

#### Architecture: SourceScout + SignalEvaluator + ToolFactory

Three components, cleanly separated by the 7-layer stack:

**1. SourceScout** (`agent/discovery/source_scout.py` — Layer 1 extension)
- Discovers candidate data sources from known API catalogs and web search
- Catalog sources:
  - data.gov CKAN API (300k+ government datasets)
  - Public API directories (manually seeded list of catalog URLs)
  - SEC EDGAR full-text search for new filing types
- Each candidate is a `DataSourceCandidate` dataclass: url, name, description,
  format, update_frequency, topic_tags, probe_result
- **Probe step**: Fetch a sample (first 100 rows / first JSON response) to
  validate the source actually returns parseable data
- **No LLM in the hot path**: Discovery uses keyword search + structured catalog
  APIs, not LLM queries. LLM is too expensive and too slow for autonomous
  surveillance expansion. Catalog search is free and deterministic.
- **Topic relevance scoring**: TF-IDF cosine similarity between source
  description/tags and existing entity types + feature names. Sources scoring
  below a threshold are discarded before evaluation.

**2. SignalEvaluator** (`agent/discovery/signal_evaluator.py` — Layer 2 extension)
- Given a probed data source sample, evaluates its predictive signal content
- **Method: Mutual Information estimation** (Kraskov et al. 2004 k-NN estimator)
  - Compute MI(source_series, existing_feature) for each feature in store
  - If max MI > threshold, the source has novel signal
  - Normalize MI by entropy of source to get fraction of source information
    that's predictive
- **Why MI?** Non-parametric, handles nonlinear relationships, works with
  small samples (k-NN estimator needs ~50+ points). Superior to linear
  correlation for detecting regime-dependent signals.
- **Alternative considered: Granger causality** — requires stationarity
  assumption and longer time series. MI is more robust for small initial
  samples from new sources.
- **Alternative considered: Transfer entropy** — equivalent to conditional
  MI, more expensive, overkill for initial screening. Can upgrade later.
- **Minimum sample size guard**: Need ≥50 aligned time-series points to
  produce a reliable MI estimate. Sources with fewer are logged but deferred.

**3. ToolFactory** (`agent/discovery/tool_factory.py` — Layer 1 extension)
- Given a validated source with positive signal, generates a new Tool class
- **Template-based generation**, not LLM code generation:
  - `JsonApiTool`: template for REST APIs returning JSON
  - `CsvFeedTool`: template for CSV/TSV download endpoints
  - Configurable: url_template, headers, response_path (JSONPath to data array),
    field_mapping (source field → entity observation schema)
- Each generated tool inherits from `Tool`, has proper JSON Schema parameters,
  and serializes to a JSON config file for persistence across restarts
- **No runtime code generation**: The tool is configured by data (URL patterns,
  field mappings), not by generating Python code. This avoids security risks
  of arbitrary code execution.
- **Registration**: New tool is registered in ToolRegistry and added to
  ToolRoutingBandit's arm set with uniform prior Beta(1,1)
- **Persistence**: Tool configs stored as JSON in `.tirra_pipeline/discovered_tools/`
  and reloaded on startup

**Safety & Human-in-the-Loop**
- All discovered sources are **logged before activation** (write-ahead log
  in `discovered_sources` table)
- **Quarantine period**: New tools run in "shadow mode" for N cycles — data
  is collected and evaluated but doesn't flow to downstream models
- After quarantine, if signal remains positive, tool is promoted to active
- **Rate limit**: Max 1 new tool discovery per DAG cycle to prevent
  runaway expansion
- **Max tool count guard**: Cap at configurable limit (default 100) to
  prevent resource exhaustion

#### Discovery DAG Integration
- New DAG node: `discover_sources` runs weekly (lower frequency than daily
  collection to save API quota)
- Pipeline: SourceScout.search() → SourceScout.probe() → SignalEvaluator.evaluate()
  → ToolFactory.create() → register + quarantine

### Change 16: Self-Extending Entity Ontology

#### Architecture: TypeInducer + OntologyRegistry + GNN Schema Adapter

**1. OntologyRegistry** (`agent/discovery/ontology_registry.py` — Layer 1/2 bridge)
- Runtime registry of known entity types and relationship types
- Replaces the hardcoded `Literal` in `entity.py` with a dynamic set
- Initialized from DB: `SELECT DISTINCT entity_type FROM entities`
- Provides: `known_entity_types()`, `known_link_types()`,
  `register_type(name, parent_type=None)`, `register_link_type(name)`
- **Type hierarchy**: Optional parent_type for ISA relationships
  (e.g., "mine" ISA "facility", "pipeline" ISA "infrastructure")
- Persistence: `entity_type_registry` table in PipelineStore
  (type_name PK, parent_type, discovered_at, source, confidence, active)

**2. TypeInducer** (`agent/discovery/type_inducer.py` — Layer 2)
- Discovers new entity types from observation patterns
- **Method: Frequency + clustering approach**

  **Step 1 — NER-like extraction**: When a tool ingests data, it may encounter
  entities that don't match any known type. These go into an `unresolved_entities`
  staging table with raw text, source_tool, and context snippet.

  **Step 2 — Clustering**: Periodically, cluster unresolved entities by:
  - Source tool (entities from same source likely have same type)
  - Observation schema similarity (what fields appear in their value_json)
  - Co-occurrence patterns (which known entity types do they appear alongside)

  **Step 3 — Type proposal**: When a cluster reaches minimum size (default 5
  unique entities), propose a new entity type:
  - Name: most common noun extracted from entity descriptions, or source_tool
    prefix + semantic label
  - Parent_type: nearest existing type by observation schema similarity
  - Confidence: cluster cohesion score (0-1)

  **Step 4 — Validation**: New type must meet:
  - ≥ 5 distinct entities
  - Cluster cohesion > threshold (default 0.7)
  - Not a subset of an existing type (check entity overlap)

  **Step 5 — Promotion**: Once validated, register type in OntologyRegistry,
  re-classify unresolved entities, and update entity_type in DB.

- **No LLM in type induction**: Uses frequency, schema similarity, and
  co-occurrence — all computable without LLM. The type naming step could
  optionally use LLM for human-readable names, but defaults to source-derived
  names.

**3. RelationshipInducer** (within TypeInducer — Layer 2)
- Discovers new link types from entity co-occurrence
- **Method**: Entities that appear in the same observation (same source_tool,
  overlapping time window) with frequency above threshold get a candidate
  link. The link_type name derives from the source_tool + observation_type.
- Uses Edge Confidence Tracker (Tier 7) logic: BIC-based scoring to determine
  if the relationship is statistically significant vs. spurious co-occurrence.

**4. GNN Schema Adapter** (`agent/features/gnn_schema_adapter.py` — Layer 2/3 bridge)
- Dynamically updates `_CONNECTED_TYPES` based on OntologyRegistry
- **Criterion**: A type is added to GNN when it has:
  - ≥ `_MIN_ENTITIES_DEFAULT` (2) entities
  - ≥ 1 link to an existing connected type
- When types are added, GNN re-initializes embedding layers for new types
  while preserving trained weights for existing types
- Uses snapshot-based approach (Kazemi 2022): full GNN retrain on new schema
  rather than incremental, because type additions are rare events (weekly/monthly)

#### EntityType Migration
- `EntityType = Literal[...]` in `entity.py` becomes a runtime validation
  against `OntologyRegistry.known_entity_types()`
- The DB layer already handles any string — no schema migration needed
- Type-checking via `validate_entity_type(t: str) → bool` function that
  queries the registry
- Backward compatible: the 9 seed types are always present in the registry

---

## Risks

1. **Runaway tool proliferation** — Auto-discovered tools that add noise, not
   signal. Mitigated by: quarantine period, MI threshold, max tool count cap,
   and existing ToolRoutingBandit that will naturally down-weight useless tools.
2. **Data quality of discovered sources** — Free APIs may be unreliable,
   rate-limited, or change schema. Mitigated by: probe step validates data
   parsability, quarantine period catches flaky sources, tools track
   consecutive failure count and auto-disable.
3. **Entity type explosion** — Too many fine-grained types that fragment the
   graph. Mitigated by: minimum cluster size, cohesion threshold, periodic
   merge pass that collapses similar types.
4. **GNN instability on schema change** — Adding new node types may degrade
   existing embeddings. Mitigated by: snapshot retrain with warm-start (preserve
   existing type embeddings), and EdgeConfidenceTracker monitoring edge quality.
5. **Security** — Auto-wired HTTP endpoints could be malicious. Mitigated by:
   template-based tool generation (no arbitrary code), configurable URL whitelist,
   timeout enforcement, response size caps.
6. **Circular dependencies** — TypeInducer needs observations, but observations
   need entity types. Mitigated by: unresolved_entities staging table decouples
   the two — observations can be stored before types are finalized.

---

## Data Requirements

### For Source Discovery (Change 15)
- Access to data.gov CKAN API (free, no key needed)
- Existing feature time series in PipelineStore for MI computation
- Probed source data samples (fetched during discovery)

### For Type Induction (Change 16)
- Entity observations from all tools (already in store)
- Unresolved entity mentions from tool output parsing
- Entity co-occurrence data (derivable from observations)

---

## Math/Algorithm Survey

### Mutual Information Estimation (Change 15)
- **Statistic**: $I(X;Y) = H(X) + H(Y) - H(X,Y)$
- **Estimator**: KSG estimator (Kraskov, Stögbauer, Grassberger 2004) using
  k-nearest-neighbor distances in joint space
- **Implementation**: `sklearn.feature_selection.mutual_info_regression` wraps KSG
- **Assumptions**: Continuous/mixed variables, ≥50 samples for reliable estimate
- **Numerical stability**: KSG is consistent and low-bias even at small sample sizes,
  converges as $O(1/\sqrt{n})$
- **Normalization**: Use $I(X;Y) / H(X)$ to get fraction of source entropy
  that's predictive. Requires source entropy estimate (also via KSG).

### Cluster Cohesion (Change 16)
- **Method**: Silhouette score for cluster validity
- **Statistic**: $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$ where $a(i)$ is
  mean intra-cluster distance, $b(i)$ is mean nearest-cluster distance
- **Implementation**: `sklearn.metrics.silhouette_score`
- **Usage**: Validate that proposed entity type cluster is genuinely distinct
  from existing types

### BIC-based Relationship Validation (Change 16, reuse from Tier 7)
- Already implemented in `EdgeConfidenceTracker`
- $\Delta BIC = BIC(\text{model without link}) - BIC(\text{model with link})$
- Positive ΔBIC → link improves model fit

---

## Implementation Notes

### Module Organization
All new code for Tier 8 goes in `agent/discovery/`:
```
agent/discovery/
    __init__.py
    source_scout.py       # Change 15: data source discovery
    signal_evaluator.py   # Change 15: MI-based signal assessment
    tool_factory.py       # Change 15: template-based tool generation
    ontology_registry.py  # Change 16: dynamic entity type registry
    type_inducer.py       # Change 16: entity type & relationship discovery
```

Plus modifications to:
- `agent/features/gnn_builder.py` — make `_CONNECTED_TYPES` dynamic
- `agent/pipeline/entity.py` — replace `Literal` with runtime validation
- `agent/pipeline/store.py` — add `entity_type_registry`, `unresolved_entities`,
  `discovered_sources` tables
- `agent/pipeline/dags/daily_collection.py` — add quarantine check for new tools
- `agent/cli.py` — load discovered tools on startup
- `agent/learning/tool_router.py` — support dynamic arm addition

### Testing Strategy
- Unit tests per component with synthetic data
- Integration test: end-to-end discovery → evaluation → registration
- Edge cases: empty catalogs, unparseable sources, duplicate types,
  merge conflicts, GNN schema change

---

## Depth Roadmap

### L1 (Current): Static tool list + fixed entity types
- 61 hardcoded tools, 9 hardcoded entity types
- All human-configured, no autonomous expansion

### L2 (Tier 8): Autonomous discovery + dynamic ontology
- System discovers new data sources from API catalogs
- System proposes new entity types from observation patterns
- Human-verified quarantine before activation

### L3 (Future): Cross-domain emergent structure
- Entity types that only exist as the intersection of two data domains
  (e.g., "energy-political-actor" = person ∩ CFTC ∩ GDELT)
- Relationship types that emerge from cross-source temporal correlation
- Fully autonomous knowledge graph expansion

---

## Related

- [[learned_vs_handcoded_architecture_spec]] — Master spec defining Tier 8
- [[tier7_self_modifying_structure]] — Tier 7 research (edge tracker, meta-scheduler)
- [[tier7_self_modifying_structure_spec]] — Tier 7 spec
- [[tier6_learned_observation]] — Tier 6 research (feature selection, tool routing)
- [[gnn_guided_tool_expansion]] — GNN-guided tool depth decisions
- [[deep_surveillance_tools]] — Surveillance tool inventory
- [[entity_linking_layer]] — Entity linking infrastructure
- [[temporal_het_gnn]] — Heterogeneous temporal GNN architecture
