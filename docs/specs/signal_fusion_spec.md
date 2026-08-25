---
title: "Spec: Signal Fusion — Self-Supervised Entity Micro-Alpha"
tags:
  - doc/spec
  - phase/20
  - topic/signal-fusion
  - topic/entity-anomaly
  - topic/micro-alpha
  - layer/fusion
  - layer/feature-engineering
---

# Spec: Signal Fusion — Self-Supervised Entity Micro-Alpha

## Goal

Build a parallel entity-level scoring pipeline where the **GNN's own prediction surprise is the primary anomaly signal**. Statistical monitors (CUSUM, Hawkes, Event Study) serve as **node feature enrichment** — they go into the GNN as additional features, not as anomaly outputs. Convergence = **correlated prediction surprise** across graph neighborhoods, detected via GNN attention weights and embedding similarity — no hand-coded graph traversal or archetype matching.

The scorer must be **type-agnostic**: the same math handles person, company, wallet, vessel, country, domain, protocol, or any future entity type.

## Paradigm

| Old Plan | New Paradigm |
|----------|-------------|
| CUSUM output → EntityAlert | CUSUM state → GNN input feature; GNN prediction surprise → EntityAlert |
| Hand-coded convergence pass (graph traversal + domain diversity) | Correlated neighborhood surprise via GNN attention |
| Composite score: sigmoid(w₁·cusum + w₂·hawkes + ...) | Composite surprise: obs_type + temporal + value + neighborhood + memory |
| Archetype matching + cluster_level | Removed entirely from code/tests |
| GNN sees anomaly features but doesn't produce anomaly | GNN IS the anomaly detector |

## Files Affected

### New Files
| File | Purpose |
|------|---------|
| `agent/fusion/__init__.py` | Module init, public exports |
| `agent/fusion/alert.py` | EntityAlert dataclass |
| `agent/fusion/convergence.py` | ConvergenceCluster dataclass + ConvergenceDetector |
| `agent/fusion/surprise.py` | SurpriseExtractor — extracts prediction surprise from GNN |
| `agent/fusion/cusum.py` | CUSUM sequential monitor (feature enrichment role) |
| `agent/fusion/hawkes.py` | Hawkes process intensity estimator (feature enrichment role) |
| `agent/fusion/entity_baseline.py` | Per-entity rolling baseline (feature enrichment role) |
| `agent/fusion/entity_scorer.py` | EntityAnomalyScorer — orchestrates full pipeline |
| `agent/pipeline/dags/entity_scoring.py` | Entity scoring DAG |
| `tests/test_cusum.py` | CUSUM unit tests |
| `tests/test_hawkes.py` | Hawkes process unit tests |
| `tests/test_entity_baseline.py` | Entity baseline unit tests |
| `tests/test_entity_alert.py` | EntityAlert dataclass tests |
| `tests/test_convergence.py` | ConvergenceCluster + detector tests |
| `tests/test_surprise.py` | SurpriseExtractor tests |
| `tests/test_entity_scorer.py` | EntityAnomalyScorer integration tests |
| `tests/test_entity_scoring_dag.py` | DAG integration tests |

### Modified Files
| File | Change |
|------|--------|
| `agent/models/gnn/graph_builder.py` | `_build_node_features()` accepts enrichment dict; feat_dim configurable |
| `agent/models/gnn/het_tgn.py` | Add `value_pred_head`; update `in_channels` |
| `agent/models/gnn/trainer.py` | Add value prediction loss (4th self-supervised component) |
| `agent/pipeline/store.py` | Add `entity_alerts` + `convergence_clusters` tables |
| `agent/models/belief.py` | Add optional `entity_id` field |

## Implementation Steps

### Step 20.1: Create fusion module + EntityAlert + ConvergenceCluster dataclasses

**Files:** `agent/fusion/__init__.py`, `agent/fusion/alert.py`, `agent/fusion/convergence.py`, `tests/test_entity_alert.py`, `tests/test_convergence.py`

```python
@dataclass(frozen=True)
class EntityAlert:
    entity_id: str
    entity_type: str
    entity_name: str
    alert_time: float
    # Five prediction-surprise signals
    obs_type_surprise: float      # -log P(actual_obs_type | h_i)
    temporal_surprise: float      # |dt_pred - dt_actual|, z-scored
    value_surprise: float         # |v_pred - v_actual| / sigma_type
    neighborhood_surprise: float  # attention-weighted neighbor surprise
    memory_drift: float           # ||m_t - m_{t-1}||_2
    # Enrichment features (input to GNN, not anomaly output)
    cusum_statistic: float
    hawkes_intensity: float
    event_study_score: float
    # Composite
    composite_surprise: float     # weighted combination of 5 surprise signals
    # Metadata
    observation_count: int
    evidence_sources: tuple[str, ...]
    metadata: dict | None = None
```

```python
@dataclass(frozen=True)
class ConvergenceCluster:
    cluster_id: str
    cluster_time: float
    member_alerts: tuple[EntityAlert, ...]
    correlated_surprise_score: float   # mean pairwise cosine sim of surprise vectors
    temporal_span_hours: float
    contributing_domains: tuple[str, ...]   # descriptive only
    contributing_tools: tuple[str, ...]     # descriptive only
    metadata: dict | None = None
```

No `matched_archetype`, no `cluster_level`, no `convergence_score` formula. The `correlated_surprise_score` IS the native GNN-derived signal.

**Tests:** Construct valid alerts/clusters. Frozen immutability. Edge: 0.0 values, negative surprise (should be valid — low surprise). Multiple entity types. ConvergenceCluster with 1 member (invalid — need 2+).

---

### Step 20.2: Implement CUSUM sequential monitor

**Files:** `agent/fusion/cusum.py`, `tests/test_cusum.py`

Same implementation as before — CUSUM is unchanged. Its ROLE changes: output feeds into GNN as a node feature, not as an anomaly signal.

```python
class CUSUMMonitor:
    """Per-entity CUSUM for detecting persistent mean shifts.
    Role: node feature enrichment (not anomaly output).
    Reference: Page (1954), Biometrika.
    """
    def update(self, entity_id: str, z_score: float) -> tuple[float, bool]: ...
    def get_state(self, entity_id: str) -> float: ...
```

**Tests:** Synthetic mean shift detection, no false alarm on noise, reset behavior, extreme values.

---

### Step 20.3: Implement Hawkes process intensity estimator

**Files:** `agent/fusion/hawkes.py`, `tests/test_hawkes.py`

Same implementation — Hawkes intensity is a node feature enrichment input. O(1) recursive update.

```python
class HawkesIntensity:
    """Per-entity Hawkes self-exciting point process.
    Role: node feature enrichment (not anomaly output).
    Reference: Hawkes (1971), Biometrika.
    """
    def update(self, entity_id: str, event_time: float) -> float: ...
    def intensity_at(self, entity_id: str, query_time: float) -> float: ...
```

**Tests:** Intensity spike after event, exponential decay, burst detection, baseline return after gap.

---

### Step 20.4: Implement per-entity rolling baseline (Event Study)

**Files:** `agent/fusion/entity_baseline.py`, `tests/test_entity_baseline.py`

Same implementation — abnormal score is a node feature enrichment input.

```python
class EntityBaseline:
    """Per-entity rolling baseline for standardized abnormal scoring.
    Role: node feature enrichment (not anomaly output).
    Reference: MacKinlay (1997), J. Financial Economics.
    """
    def add_observation(self, entity_id: str, value: float) -> None: ...
    def abnormal_score(self, entity_id: str, current_value: float) -> float | None: ...
```

**Tests:** Known distribution, insufficient history, gap window, sliding window.

---

### Step 20.5: Enrich GraphBuilder node features (12d → ~39d)

**Files:** `agent/models/gnn/graph_builder.py` (MODIFY), `agent/models/gnn/het_tgn.py` (MODIFY)

Modify `_build_node_features()` to accept optional `enrichment: dict[str, dict[str, float]]` mapping entity_id → feature dict. Concatenate enrichment features after existing 12d.

Enrichment features per entity (27d):
- CUSUM state (1d)
- Hawkes intensity (1d)
- Event Study abnormal score (1d)
- BOCPD changepoint probability (1d)
- Observation value variance (1d)
- Observation value min (1d)
- Observation value max (1d)
- Observation IQR (1d)
- Number of distinct source tools (1d)
- Observation type distribution (18d) — frequency of each of the 18 obs types

Update HetTGN `in_channels` from 12 to 39 per type. Type projection layers handle the new dim.

**Tests:** Backward compat (no enrichment → 12d as before). With enrichment → 39d. Missing entity in enrichment → zeros for enrichment dims. Verify HetTGN forward pass with new dim.

---

### Step 20.6: Add value prediction head to HetTGN + training loss

**Files:** `agent/models/gnn/het_tgn.py` (MODIFY), `agent/models/gnn/trainer.py` (MODIFY)

Add `value_pred_head` to HetTGN:
```python
self.value_pred_head = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim // 2),
    nn.ReLU(),
    nn.Linear(hidden_dim // 2, 1),
)
```

Add to trainer `_compute_targets()`: extract observation value from next-window obs (using same 6 value fields as `_compute_obs_stats()`).

Add 4th loss component in training loop:
```
value_loss = F.huber_loss(value_pred, value_actual)  # robust to outliers
total = obs_weight*obs + dt_weight*dt + contrastive_weight*c + value_weight*value
```

TrainerConfig addition: `value_weight: float = 0.3`

**Tests:** Value prediction head produces scalar output. Training loop includes value loss. Value loss decreases over epochs on synthetic data. Backward compat: training without value targets still works (skip value loss).

---

### Step 20.7: Implement SurpriseExtractor

**Files:** `agent/fusion/surprise.py`, `tests/test_surprise.py`

**Core intelligence component.** Takes a trained HetTGN + observations + graph data, extracts per-entity prediction surprise.

```python
class SurpriseExtractor:
    """Extract prediction surprise from trained HetTGN.
    
    Five signals per entity:
    1. obs_type_surprise: -log P(actual | h_i) from obs_type_head
    2. temporal_surprise: |dt_pred - dt_actual|, z-scored per type
    3. value_surprise: |v_pred - v_actual| / sigma_type from value_pred_head
    4. neighborhood_surprise: attention-weighted avg of neighbor composite surprise
    5. memory_drift: L2 norm of memory change
    """
    def extract(self, model, data, id_map, observations, 
                memory_before) -> dict[str, EntitySurprise]: ...
```

Adaptive thresholding: z-score surprise against per-entity-type rolling history.

**Tests:** Synthetic entity with predictable behavior → low surprise. Inject anomalous obs_type → high obs_type_surprise. Inject anomalous timing → high temporal_surprise. Inject anomalous value → high value_surprise. Neighborhood: entity with surprised neighbors → high neighborhood_surprise.

---

### Step 20.8: Implement ConvergenceDetector

**Files:** `agent/fusion/convergence.py` (ADD), `tests/test_convergence.py`

Takes per-entity surprise scores + entity_links + attention weights. Groups high-surprise entities by correlated surprise in graph neighborhoods.

```python
class ConvergenceDetector:
    """Detect convergence as correlated prediction surprise in graph neighborhoods."""
    def detect(self, entity_surprises: dict[str, EntitySurprise],
               entity_links: list[dict],
               surprise_threshold: float = 2.0) -> list[ConvergenceCluster]: ...
```

Algorithm:
1. Filter to entities with composite_surprise > threshold (z-scored)
2. Build subgraph of elevated entities connected by entity_links
3. Find connected components in subgraph
4. For each component with 2+ entities → ConvergenceCluster
5. correlated_surprise_score = mean pairwise cosine similarity of surprise vectors

No domain diversity counting. No temporal window. No archetype matching.

**Tests:** Two linked entities both surprised → cluster. One entity surprised, neighbor normal → no cluster. Chain of 3 surprised entities → one cluster. Disconnected surprised entities → no cluster. Novel tool combinations → clusters detected.

---

### Step 20.9: Implement EntityAnomalyScorer orchestrator

**Files:** `agent/fusion/entity_scorer.py`, `tests/test_entity_scorer.py`

Orchestrates the full pipeline:
1. Compute statistical enrichment features (CUSUM, Hawkes, EventStudy)
2. Build enriched graph via modified GraphBuilder
3. GNN forward pass
4. Extract prediction surprise via SurpriseExtractor
5. Detect convergence via ConvergenceDetector
6. Emit EntityAlerts + ConvergenceClusters

```python
class EntityAnomalyScorer:
    def score_entities(self, as_of: float) -> tuple[list[EntityAlert], list[ConvergenceCluster]]: ...
```

**Tests:** Full pipeline with mock store + synthetic entities. Normal entity → low surprise. Shifted entity → high surprise. Entity with no history → graceful degradation. 0 entities → empty results.

---

### Step 20.10: Add entity_alerts + convergence_clusters storage to PipelineStore

**Files:** `agent/pipeline/store.py` (MODIFY)

```sql
CREATE TABLE IF NOT EXISTS entity_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_name TEXT NOT NULL,
    alert_time REAL NOT NULL,
    obs_type_surprise REAL NOT NULL,
    temporal_surprise REAL NOT NULL,
    value_surprise REAL NOT NULL,
    neighborhood_surprise REAL NOT NULL,
    memory_drift REAL NOT NULL,
    cusum_statistic REAL NOT NULL,
    hawkes_intensity REAL NOT NULL,
    event_study_score REAL NOT NULL,
    composite_surprise REAL NOT NULL,
    observation_count INTEGER NOT NULL,
    evidence_sources_json TEXT NOT NULL,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS convergence_clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id TEXT NOT NULL UNIQUE,
    cluster_time REAL NOT NULL,
    member_entity_ids_json TEXT NOT NULL,
    correlated_surprise_score REAL NOT NULL,
    temporal_span_hours REAL NOT NULL,
    contributing_domains_json TEXT NOT NULL,
    contributing_tools_json TEXT NOT NULL,
    metadata_json TEXT
);
```

**Tests:** Round-trip store + query. Query by entity_id, by time range. Empty table queries. Duplicate cluster_id → UNIQUE constraint failure.

---

### Step 20.11: Add optional entity_id to BeliefState

**Files:** `agent/models/belief.py` (MODIFY)

```python
entity_id: str | None = None
```

**Tests:** Existing tests pass. New BeliefState with entity_id roundtrips correctly.

---

### Step 20.12: Entity scoring DAG

**Files:** `agent/pipeline/dags/entity_scoring.py`, `tests/test_entity_scoring_dag.py`

```python
DAG_NAME = "entity_scoring"
DEPENDS_ON = ["gnn_inference"]

def run(pipeline, as_of: float) -> dict:
    """Score all entities via prediction surprise, detect convergence, store results."""
```

**Tests:** DAG runs with mock pipeline. Alerts stored. Clusters stored. No GNN → graceful skip.

---

### Step 20.13: Integration tests + edge cases

**Files:** `tests/test_signal_fusion_integration.py`

Full end-to-end: mock observations → enrichment → GNN → surprise → convergence → stored.

Anti-bias tests:
- Generic convergence: entity with surprised linked neighbors → ConvergenceCluster
- Novel patterns: unseen tool combinations → clusters detected
- Cold-start: entity with no history → falls back to statistical features
- Do NOT test for specific named archetype patterns

Edge cases: 0 observations, NaN embeddings, stale entities, CUSUM accumulation across runs, Hawkes decay between runs, very large entity count.

---

## Related

- [[signal_fusion]] — Research note
- [[quant_training_ground]] — Master task file
- [[world_model_bridge_spec]] — Phase 19 spec
- [[convergence_detection_spec]] — Phase 7c
- [[signal_protocol_feature_engineering_spec]] — Phase 8
- [[entity_linking_layer_spec]] — Phase 17
- [[temporal_het_gnn_spec]] — Phase 12
