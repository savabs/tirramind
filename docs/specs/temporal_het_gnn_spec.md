---
title: "Spec: Temporal Heterogeneous GNN for Automated Pattern Discovery"
tags:
  - doc/spec
  - phase/12
  - topic/world-model
  - topic/surveillance
  - layer/world-model
  - layer/feature-engineering
---

# Spec: Temporal Heterogeneous GNN

## Goal

Build a Temporal Heterogeneous Graph Network that:
1. Ingests the full TirraMind entity graph (6 node types, 3+ edge types, 7+ observation types).
2. Learns which cross-entity temporal patterns carry predictive signal via self-supervised pre-training.
3. Extracts discovered patterns and crystallizes them into production rules.

Research: [[temporal_het_gnn]]

---

## Files Affected

### New files (all in `agent/models/gnn/`)
| File | Phase | Purpose |
|------|-------|---------|
| `__init__.py` | 12a | Package init |
| `graph_builder.py` | 12a | PipelineStore → PyG HeteroData conversion |
| `temporal.py` | 12b | Time2Vec encoding + temporal feature vectors |
| `het_tgn.py` | 12c | Model: HGT backbone + custom heterogeneous memory |
| `trainer.py` | 12d | Self-supervised training loop (next-event prediction) |
| `pattern_extractor.py` | 12e | Attention analysis → crystallized production rules |

### Modified files
| File | Phase | Change |
|------|-------|--------|
| `agent/pipeline/store.py` | 12a | Add `query_all_entities()`, `query_all_observations()`, `query_all_entity_links()` |
| `agent/models/__init__.py` | 12f | Export GNN sub-package |
| `requirements.txt` / `pyproject.toml` | 12a | Add torch, torch-geometric dependencies |

---

## Implementation Steps

### Phase 12a: Graph Construction Layer

**12a.1: Add PipelineStore query methods**
- Add `query_all_entities() → List[Dict]` — returns all entities with type + metadata
- Add `query_all_observations(since: Optional[float] = None) → List[Dict]` — returns all observations ordered by timestamp
- Add `query_all_entity_links() → List[Dict]` — returns all entity links with types + confidence
- Test: verify against existing test data

**12a.2: Install PyTorch + PyG dependencies**
- `pip install torch --index-url https://download.pytorch.org/whl/cpu` (CPU-only to keep small)
- `pip install torch-geometric`
- Verify import works

**12a.3: Implement `graph_builder.py`**
- `GraphBuilder(store: PipelineStore)` class
- `build_hetero_data(since: Optional[float] = None) → HeteroData`
  - Query all entities → create node features per type
  - Node features: one-hot entity_type (6-dim) + observation stats (count, recency, mean_value) → ~10-dim per node type
  - Query all entity_links → create edge_index per (src_type, relation, dst_type) triplet
  - Edge features: confidence + age_days
  - Query all observations → attach as temporal events (not graph edges initially)
- `build_id_maps() → (type_to_ids: Dict, global_to_typed: Dict, typed_to_global: Dict)`
  - Bidirectional mapping between (type, entity_id) and flat global integer ID
- Test: build graph from synthetic PipelineStore, verify node/edge counts, feature shapes, metadata

**12a.4: Edge case tests for graph builder**
- Empty store → valid empty HeteroData
- Single entity, no observations
- Orphan entities (no links)
- Duplicate entity_links
- Missing entity referenced by link
- Observation with no matching entity
- Timestamp ordering verification

### Phase 12b: Temporal Feature Encoding

**12b.1: Implement Time2Vec**
- `Time2Vec(in_features: int, out_features: int)` — `nn.Module`
- Forward: $\mathbf{t2v}(\tau)[0] = \omega_0 \tau + \phi_0$ (linear), $\mathbf{t2v}(\tau)[i] = \sin(\omega_i \tau + \phi_i)$ for $i > 0$ (periodic)
- Learnable parameters: `omega` (frequencies), `phi` (phase shifts)
- Input: scalar timestamps → Output: `out_features`-dim vector
- Test: output shape, gradient flow, periodic component behavior

**12b.2: Implement `TemporalEncoder`**
- `TemporalEncoder(time_dim: int, obs_types: List[str], max_history: int = 32)` — `nn.Module`
- Per-entity observation history → fixed-length feature vector
  - Last K observations: type one-hot + value + Time2Vec(timestamp)
  - Aggregated: count per obs_type in recent window, mean inter-event time
- `encode_entity_history(entity_obs: List[Dict], current_time: float) → Tensor`
- Test: varying history lengths, empty history, single observation

**12b.3: Edge case tests for temporal encoding**
- Time2Vec with zero timestamp, negative timestamp, very large timestamp
- TemporalEncoder with history exceeding max_history (truncation)
- Missing value fields in observations
- All observations of same type
- Numerical stability: very close timestamps, very spread timestamps

### Phase 12c: Model Architecture (HetTGN)

**12c.1: Implement `HeteroMemory`**
- Custom module replacing TGNMemory for heterogeneous graphs
- `HeteroMemory(num_nodes: int, memory_dim: int, time_dim: int)`
- Internal: memory tensor `[num_nodes, memory_dim]`, last_update tensor `[num_nodes]`
- `get_memory(node_ids: Tensor) → (Tensor, Tensor)` — returns memory + last_update
- `update_memory(node_ids: Tensor, messages: Tensor, timestamps: Tensor)` — GRU-based update
- Uses Time2Vec for time delta encoding: `time_enc(t_current - t_last_update)`
- Test: memory initialization, update, retrieval, multiple updates

**12c.2: Implement `HetTGN` model**
- `HetTGN(metadata, hidden_dim, time_dim, num_heads, num_layers, num_node_types, num_obs_types)`
- Components:
  - `type_projections: Dict[str, nn.Linear]` — per-type input projection to `hidden_dim`
  - `memory: HeteroMemory` — per-node temporal state
  - `time_enc: Time2Vec` — continuous time encoding
  - `hgt_layers: nn.ModuleList[HGTConv]` — `num_layers` HGT convolution layers
  - `event_predictor: nn.Linear` — predict (entity, obs_type, time_delta) for next event
- Forward pass:
  1. Project per-type node features to common hidden_dim
  2. Concatenate memory state for each node
  3. Run HGT layers on HeteroData
  4. Return updated node embeddings
- `predict_next_event(embeddings, candidate_ids) → (entity_logits, type_logits, time_delta)`
- Test: forward pass shapes, gradient flow through all components

**12c.3: Edge case tests for model**
- Graph with single node type
- Graph with no edges (isolated nodes)
- Memory for unseen node IDs
- Zero-length observation sequence
- Model save/load roundtrip

### Phase 12d: Self-Supervised Training

**12d.1: Implement `SyntheticGraphGenerator`**
- Generate realistic synthetic entity graphs for training loop development
- Configurable: num_entities per type, num_observations, num_links, time_span
- Inject known patterns (e.g., "entity A observation always precedes entity B observation within 24h")
- Return PipelineStore with seeded data
- Test: verify generated data has expected structure and injected patterns

**12d.2: Implement training loop**
- `Trainer(model: HetTGN, store: PipelineStore, config: TrainerConfig)`
- `TrainerConfig`: learning_rate, epochs, window_size, batch_size, contrastive_weight
- Walk-forward split: chronological train/val/test (70/15/15 by time)
- Per-window training step:
  1. Build HeteroData snapshot for window
  2. Forward pass → embeddings
  3. Predict next event in next window
  4. Compute loss: CE(entity) + CE(obs_type) + MSE(time_delta) + contrastive
- Contrastive loss: linked entity pairs should be closer in embedding space than random pairs
- `train() → Dict[str, List[float]]` (loss curves)
- Test: loss decreases on synthetic data, no NaN gradients

**12d.3: Implement walk-forward evaluation**
- `evaluate(model, store, split='val') → Dict[str, float]`
- Metrics: entity prediction accuracy (top-1, top-5), obs_type accuracy, time_delta MAE
- No future leakage: val/test windows strictly after training windows
- Test: metrics are computable, no data leakage

**12d.4: Edge case tests for training**
- Empty training window
- Window with single observation
- All observations of same entity (degenerate case)
- NaN in loss (numerical stability)
- Walk-forward split with insufficient data
- Contrastive loss with no entity links

### Phase 12e: Pattern Extraction

**12e.1: Implement attention extraction**
- `PatternExtractor(model: HetTGN, store: PipelineStore)`
- `extract_metapath_importance() → List[MetaPathPattern]`
  - For each (src_type, edge_type, dst_type) in trained model:
    - Mean attention weight across all instances
    - Frequency (how often this metapath appears)
    - Score = mean_attention × log(frequency)
- `MetaPathPattern`: src_type, edge_type, dst_type, score, mean_lag, lag_std
- Test: patterns extracted from model with known injected patterns

**12e.2: Implement temporal lag extraction**
- For top-K meta-paths, analyze temporal lag distributions
- For each (src_type → dst_type) with high attention:
  - Collect all (src_obs_time, dst_obs_time) pairs within training data
  - Compute lag distribution: mean, std, percentiles
- Test: lag statistics match injected patterns in synthetic data

**12e.3: Implement crystallization**
- Convert top-K discovered patterns to production rule configs
- Output format compatible with cross_entity.py:
  ```python
  CrystallizedPattern(
      source_type="vessel",
      target_type="country",
      via_edge="port_call_to",
      obs_type_a="port_call",
      obs_type_b="geopolitical_event",
      window_seconds=86400,
      min_score=0.5,
  )
  ```
- `crystallize(patterns: List[MetaPathPattern], threshold: float) → List[CrystallizedPattern]`
- Test: crystallization produces valid configs, threshold filtering works

**12e.4: Edge case tests for pattern extraction**
- Model with uniform attention (no patterns learned)
- Single meta-path dominates
- Very sparse attention (most weights near zero)
- Crystallization with threshold that filters everything

### Phase 12f: Integration + Evaluation

**12f.1: Integration with cross_entity.py**
- Add `AutoPatternDetector` that loads crystallized patterns and runs them alongside hand-crafted ones
- Both auto and hand-crafted produce `cross_entity_pattern` observations in the same format
- Distinguish via `metadata_json.source: "auto_gnn"` vs `"hand_crafted"`

**12f.2: Comparative evaluation**
- Run depth_eval.py conditional MI on auto-discovered patterns
- Compare MI of auto patterns vs hand-crafted patterns
- Log results to pipeline store

**12f.3: Periodic retraining hook**
- `retrain_and_discover(store, config) → List[CrystallizedPattern]`
- Entry point for periodic re-training to discover new patterns
- Retire patterns whose MI has decayed below threshold

---

## Edge Cases (Cross-Phase)

- **Cold start:** Very few entities → model learns nothing useful. Mitigation: synthetic augmentation + minimum entity threshold.
- **Type imbalance:** Some types have many entities, others few. Mitigation: type-balanced sampling in contrastive loss.
- **Temporal sparsity:** Some entities rarely have observations. Mitigation: memory staleness detection + default embedding for inactive nodes.
- **Graph disconnection:** Some entity types may not link to others. Mitigation: HGTConv handles this gracefully (returns None for unreachable types).

---

## Testing Plan

Each sub-phase has its own edge case test suite (see steps above). Additionally:

1. **Synthetic pattern recovery test:** Inject 3 known temporal patterns into synthetic graph. Train model. Verify pattern_extractor discovers all 3 with correct temporal lags.
2. **Walk-forward integrity test:** Verify no observation from val/test period appears in training data.
3. **Regression test:** Hand-crafted patterns still work identically after integration.
4. **Performance test:** Full pipeline (build graph → train → extract) completes in <5 min on synthetic data.

---

## Related

- [[temporal_het_gnn]]
- [[cross_entity_l3]]
- [[cross_entity_l3_spec]]
- [[world_model_spec]]
- [[pipeline_layer]]
- [[project_memory]]
