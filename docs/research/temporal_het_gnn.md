---
title: "Research: Temporal Heterogeneous GNN for Automated Pattern Discovery"
tags:
  - doc/research
  - phase/12
  - topic/world-model
  - topic/surveillance
  - layer/world-model
  - layer/feature-engineering
---

# Research: Temporal Heterogeneous GNN for Automated Pattern Discovery

## Goal

Replace hardcoded L3 cross-entity patterns with a learned model that automatically discovers which cross-entity temporal co-occurrences carry signal. The three hand-crafted patterns (Insider×GDELT, Vessel×Sanctions, Whale×Geopolitical) prove the entity graph has structure. The GNN learns what that structure is without being told.

**Business problem:** Human-specified patterns are limited by the analyst's imagination. The entity graph has 6 node types × 7+ observation types × 3+ edge types = combinatorial space too large to explore manually. A learned model scans the full space and surfaces patterns we wouldn't think to look for.

**Signal depth doctrine:** This is the ultimate L3 play — cross-entity combinations discovered by machine, not human.

---

## Search Log

- **arXiv keywords:** "heterogeneous graph transformer", "temporal graph network", "time2vec", "self-supervised graph neural network", "dynamic graph learning"
- **PyG documentation:** `HGTConv`, `HeteroData`, `TGNMemory` class API references
- **GitHub:** `pyg-team/pytorch_geometric` examples (`hgt_dblp.py`, `tgn.py`)

---

## Primary References (Verified)

### 1. HGT — Heterogeneous Graph Transformer
- **Paper:** Hu, Dong, Wang, Sun. "Heterogeneous Graph Transformer." WWW 2020.
- **arXiv:** [2003.01332](https://arxiv.org/abs/2003.01332)
- **Key contribution:** Node- and edge-type-dependent attention parameters. Each (src_type, edge_type, dst_type) triplet gets its own attention head parameterization. Includes relative temporal encoding.
- **Verified claim:** Tested on Open Academic Graph (179M nodes, 2B edges). Outperforms baselines by 9–21%.
- **Relevance to TirraMind:** Our entity graph is inherently heterogeneous (company, person, vessel, wallet, country, organization). HGT's type-specific attention learns which cross-type connections matter.
- **PyG implementation:** `torch_geometric.nn.conv.HGTConv(in_channels, out_channels, metadata, heads)`. Takes `x_dict: Dict[str, Tensor]` and `edge_index_dict: Dict[Tuple[str,str,str], Tensor]`. Returns `Dict[str, Optional[Tensor]]` per-node-type embeddings. Example: `examples/hetero/hgt_dblp.py`.

### 2. TGN — Temporal Graph Networks
- **Paper:** Rossi, Chamberlain, Frasca, Eynard, Monti, Bronstein. "Temporal Graph Networks for Deep Learning on Dynamic Graphs." 2020.
- **arXiv:** [2006.10637](https://arxiv.org/abs/2006.10637)
- **Key contribution:** Generic framework for dynamic graphs as sequences of timed events. Novel combination of memory modules (per-node hidden state updated on each interaction) and graph-based operators (attention over neighborhoods). SOTA on both transductive and inductive tasks.
- **Relevance to TirraMind:** Our entity graph is dynamic — observations arrive as a stream of timed events. TGN's memory module maintains per-entity state that evolves with each new observation.
- **PyG implementation:** `torch_geometric.nn.models.TGNMemory(num_nodes, raw_msg_dim, memory_dim, time_dim, message_module, aggregator_module)`. Methods: `forward(n_id)` → `(memory, last_update)`, `update_state(src, dst, t, raw_msg)`, `reset_state()`, `detach()`. Example: `examples/tgn.py`.

### 3. Time2Vec — Learnable Time Representation
- **Paper:** Kazemi, Goel, Jain, Kobyzev, Sethi, Forsyth, Poupart. "Time2Vec: Learning a Model-Agnostic Representation of Time." 2019.
- **arXiv:** [1907.05321](https://arxiv.org/abs/1907.05321)
- **Key contribution:** Model-agnostic vector representation of time using learnable periodic + linear components: $\mathbf{t2v}(\tau)[i] = \omega_i \tau + \phi_i$ (linear for $i=0$), $\sin(\omega_i \tau + \phi_i)$ (periodic for $i > 0$). Captures both periodic patterns (day-of-week, market hours) and trends.
- **Relevance to TirraMind:** Our events span multiple timescales (minutes for BTC transfers, days for vessel movements, weeks for filing cycles). Time2Vec learns the relevant periodicities from data.

### 4. Self-Supervised Learning on Graphs
- **Paper:** Jin, Derr, Liu, Wang, Wang, Liu, Tang. "Self-supervised Learning on Graphs: Deep Insights and New Direction." 2020.
- **arXiv:** [2006.10141](https://arxiv.org/abs/2006.10141)
- **Key contribution:** Systematic study of SSL pretext tasks for GNNs. SelfTask framework achieves SOTA without labels. Tasks include: node attribute prediction, edge existence prediction, graph partitioning, distance-based methods.
- **Relevance to TirraMind:** We have no supervised labels (no market outcomes yet). SSL pretext tasks (next-event prediction, temporal link prediction) let us learn meaningful representations from the observation stream alone.

---

## PyG Library Verification

### HeteroData (Confirmed)
- **Class:** `torch_geometric.data.HeteroData`
- **API:** `data['paper'].x = tensor`, `data['author', 'writes', 'paper'].edge_index = tensor`
- **Key properties:** `.node_types`, `.edge_types`, `.metadata()` → `(List[str], List[Tuple[str,str,str]])`
- **Temporal support:** `.snapshot(start_time, end_time)`, `.sort_by_time()`, `.up_to(end_time)`
- **Graph operations:** `.subgraph()`, `.to_homogeneous()`, `.connected_components()`
- **GPU support:** `.to('cuda:0')`, `.pin_memory()`

### HGTConv (Confirmed)
- **Class:** `torch_geometric.nn.conv.HGTConv`
- **Signature:** `HGTConv(in_channels, out_channels, metadata, heads=1)`
- **Forward:** `forward(x_dict, edge_index_dict)` → `Dict[str, Optional[Tensor]]`
- **Accepts:** metadata from `HeteroData.metadata()` directly
- **Multi-head attention:** `heads` parameter controls number of attention heads

### TGNMemory (Confirmed — with constraint)
- **Class:** `torch_geometric.nn.models.TGNMemory`
- **Signature:** `TGNMemory(num_nodes, raw_msg_dim, memory_dim, time_dim, message_module, aggregator_module)`
- **Forward:** `forward(n_id)` → `(memory_tensor, last_update_tensor)`
- **Update:** `update_state(src, dst, t, raw_msg)`
- **⚠ CRITICAL CONSTRAINT:** TGNMemory is **homogeneous** — operates on flat integer node IDs, not typed nodes. HGTConv needs typed node/edge dictionaries. **Need a mapping layer:** `(type, local_id) → global_id` for memory, inverse for convolution.

---

## Current Architecture

### Existing Entity Graph (PipelineStore)

| Node Type | Example | Source Tool |
|-----------|---------|-------------|
| company | Exxon (CIK) | insider_filings, form144 |
| person | CEO (CIK) | insider_filings |
| vessel | IMO 9000001 | ais_vessel |
| wallet | bc1q... | whale_alert |
| country | US (FIPS) | gdelt |
| organization | EU | gdelt |

| Edge Type | Source → Target | Source |
|-----------|----------------|--------|
| headquartered_in | company → country | seed_company_country_links |
| port_call_to | vessel → country | seed_vessel_country_links |
| exchange_based_in | wallet → country | seed_whale_country_links |

| Observation Type | Entity Type | Key Features |
|-----------------|-------------|--------------|
| insider_trade | person/company | value, shares, direction |
| form144_filing | person/company | estimated_value |
| btc_transfer | wallet | btc_amount, usd_amount |
| port_call | vessel | port_name |
| vessel_position | vessel | lat, lon |
| geopolitical_event | country | goldstein_scale, num_articles |
| cross_entity_pattern | varies | pattern_type, score |

### Relevant Local Modules
- **agent/pipeline/store.py** — PipelineStore with entity/observation/link schema
- **agent/pipeline/cross_entity.py** — Hardcoded L3 patterns (to be augmented, not replaced)
- **agent/pipeline/depth_eval.py** — Conditional MI evaluator (KSG estimator)
- **agent/models/** — World Model (Bayesian network, pgmpy). GNN goes in `agent/models/gnn/` sub-package.
- **agent/quant/** — Scoring, changepoint, regime, spectral, backtest modules

### PipelineStore Gaps
- No `query_all_entities()` — need raw SQL or new method
- No `query_all_observations()` — same
- No global enumeration APIs — `graph_builder.py` will need direct SQL queries via `store._get_conn()`

---

## Observations

### What exists
- Complete entity/observation/link schema in PipelineStore (168 tests passing)
- Three working L3 patterns proving cross-entity signal exists
- depth_eval.py for measuring conditional mutual information
- Bayesian world model in agent/models/ (pgmpy)
- networkx already installed (useful for graph analysis)

### What is missing
- **PyTorch + PyG** — torch, torch-geometric, torch-scatter, torch-sparse not installed
- **No graph construction from PipelineStore** — need SQLite → HeteroData converter
- **No temporal encoding layer** — Time2Vec not implemented
- **No training infrastructure** — no self-supervised objectives defined
- **No pattern extraction** — attention → production rule pipeline

### Important constraints
1. **TGNMemory is homogeneous** — needs a type-aware wrapper or flat ID mapping
2. **Entity graph is sparse** — small number of entities early on, model must handle cold-start
3. **No target variable** — self-supervised only for now
4. **Local compute only** — no GPU assumed, must be CPU-tractable for small graphs
5. **Walk-forward mandatory** — chronological splits, no future leakage in training

---

## Risks

### Technical
- **Heterogeneous + Temporal gap:** No off-the-shelf model does both natively. HGT handles heterogeneous but is static. TGN handles temporal but is homogeneous. We must compose them.
- **Small graph problem:** With ~100s of entities initially, overfitting is a real risk. Need strong regularization or synthetic data augmentation.
- **Attention ≠ causation:** High attention weight on a meta-path means the model uses it for prediction, not that it represents a real causal relationship. Need depth_eval.py MI validation post-extraction.
- **PyTorch dependency size:** Adding torch + PyG is a significant dependency footprint (~2GB).

### Licensing
- PyTorch: BSD-3 ✓
- PyTorch Geometric: MIT ✓
- All referenced papers: academic, concept-only extraction. No code ported.

### Testing
- Synthetic graph data for unit tests (deterministic, small)
- Walk-forward evaluation prevents overfitting validation
- Compare auto-discovered patterns vs hand-crafted via depth_eval conditional MI

---

## Data Requirements

### Required inputs
- All entities from PipelineStore (`entities` table)
- All observations with timestamps (`entity_observations` table)
- All entity links with types and confidence (`entity_links` table)
- Entity aliases for cross-type resolution (`entity_aliases` table)

### What exists locally
- Full schema with data from 5 L2 tools (insider, whale, vessel, GDELT, CFTC)
- 3 L3 patterns seeding entity_links

### What still needs to be added
- `store.query_all_entities()` or equivalent graph export API
- `store.query_all_observations()` with time ordering
- `store.query_all_entity_links()` for edge construction

---

## Math / Architecture Survey

### Approach: HGT + Custom Temporal Memory + Time2Vec

**Why this composition:**

| Component | Role | Source | Why Chosen |
|-----------|------|--------|------------|
| HGTConv | Heterogeneous message passing | Hu et al. 2020 | Only PyG conv that handles typed nodes + edges natively |
| Custom memory | Per-entity temporal state | Inspired by TGN (Rossi et al. 2020) | TGNMemory is homogeneous; we need type-aware variant |
| Time2Vec | Continuous time encoding | Kazemi et al. 2019 | Learnable, handles multiple periodicities |
| Next-event prediction | Self-supervised objective | Standard in temporal graph lit | No labels needed, learns dynamics |
| Contrastive loss | Auxiliary objective | Jin et al. 2020 | Prevents embedding collapse |

### Alternatives Considered

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| Pure HGT (static) | Simple, well-tested | Loses temporal dynamics entirely | Rejected — temporal is critical |
| Pure TGN (homogeneous) | Strong temporal | Loses type-specific attention, treats all edges same | Rejected — heterogeneity is our moat |
| R-GCN + temporal | Handles relation types | Less expressive than attention, no temporal memory | Rejected — weaker than HGT |
| GNN on homogeneous (type as feature) | Uses standard PyG | Type info is a feature not structural, weaker inductive bias | Rejected — loses graph structure |
| TGAT (Xu et al. 2020) | Temporal attention | Homogeneous only, no HeteroData support in PyG | Rejected — same limitation as TGN |

### Heterogeneous + Temporal Integration Design

Since no single off-the-shelf model handles both, we compose:

1. **Flat ID mapping:** Assign each `(entity_type, entity_id)` a unique global integer ID. Maintain bidirectional lookup dicts.
2. **Custom HeteroMemory:** Wraps a memory tensor indexed by global ID but grouped by type. On event: update memory for involved nodes. Before HGT forward pass: inject current memory into node features.
3. **Forward pass per time window:**
   - Query observations in window
   - Update memory for active nodes
   - Build HeteroData snapshot (nodes = entities with features + memory, edges = links)
   - Run HGTConv layers → updated embeddings
   - Predict next event (which entity, what type, when)
4. **Loss:** Cross-entropy on entity prediction + cross-entropy on obs_type prediction + MSE on time delta + contrastive loss on embeddings

### Complexity Notes
- HGTConv: $O(|E| \cdot d \cdot H)$ per layer, where $|E|$ = edges, $d$ = hidden dim, $H$ = heads
- Memory update: $O(|V_{active}| \cdot d)$ per window
- Time2Vec: $O(d_{time})$ per event — negligible
- Overall: dominated by HGT forward pass, linear in graph size. CPU-tractable for ~1000 nodes

---

## Implementation Intent

### Concepts approved for implementation
- HGT backbone for heterogeneous message passing (HGTConv from PyG)
- Custom temporal memory module (inspired by TGN, adapted for heterogeneous types)
- Time2Vec for continuous time encoding
- Self-supervised: next-event prediction + contrastive loss
- Walk-forward training with chronological splits
- Pattern extraction via attention weight analysis
- Crystallization as production rules for cross_entity.py

### Concepts rejected
- Pure homogeneous approach (loses type structure)
- R-GCN (weaker attention mechanism)
- Supervised training (no labels available)
- Graph-level objectives (we need node/event-level predictions)

### Notes for the spec
- Put GNN code in `agent/models/gnn/` sub-package to separate from existing Bayesian world model
- Phase 12a (graph builder) is prerequisite for everything — write and test first
- Synthetic data generation is critical for testing since real entity count is small
- TGNMemory homogeneous constraint requires custom wrapper — don't use TGNMemory directly
- Install torch + torch-geometric as new dependencies (both have compatible licenses)
- Pattern extraction must validate via depth_eval.py conditional MI before crystallization

---

## Related

- [[temporal_het_gnn_spec]]
- [[cross_entity_l3]]
- [[cross_entity_l3_spec]]
- [[whale_geopolitical_l3]]
- [[vessel_sanctions_l3]]
- [[world_model]]
- [[world_model_spec]]
- [[pipeline_layer]]
- [[project_memory]]
