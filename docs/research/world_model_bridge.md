---
title: "Feature: GNN ↔ World Model Bridge (Phase 19)"
tags:
  - doc/research
  - phase/19
  - topic/world-model
  - topic/surveillance
  - layer/world-model
  - layer/feature-engineering
---

# Feature: GNN ↔ World Model Bridge (Phase 19)

## Goal

Connect the bottom-up learned layer (HetTGN entity graph) to the top-down reasoning layer (Bayesian DAG + Kalman filter) and produce the system's first real-data predictions. Today every component works in isolation: the GNN trains on synthetic data, the Bayesian DAG infers from macro + convergence features only, and neither informs the other. Phase 19 closes the loop:

```
PipelineStore entities → GNN embeddings → new EngineeredFeatures →
expanded WorldModel DAG → posterior beliefs → walk-forward backtest → validation
```

**Business answer:** Does any of this math produce predictive signal? Until this phase completes, we don't know.

## Search Log

- GitHub keywords: "temporal graph network embedding extraction", "GNN feature extraction pipeline", "heterogeneous graph embedding aggregation", "TGN inference pipeline production"
- Documentation keywords: "PyG HeteroData forward inference", "pgmpy add_node dynamic", "filterpy expand state dimension"
- Arxiv: Rossi et al. 2020 (TGN, arXiv:2006.10637), Hu et al. 2020 (HGT, arXiv:2003.01332), Hu et al. ICLR 2020 (pre-training GNNs, arXiv:1905.12265)

## External Repositories Reviewed

- **TGN (twitter-research/tgn)**: Reference implementation of the TGN paper. Shows embedding extraction via `model.compute_embedding()` after memory update. License: MIT. Reuse conclusion: concept only — our HetTGN already implements the architecture differently.
- **PyG examples (pyg-team/pytorch_geometric)**: Heterogeneous graph examples show `model(data.x_dict, data.edge_index_dict)` → per-type embeddings. License: MIT. Reuse conclusion: we already follow this pattern.
- **pgmpy (pgmpy/pgmpy)**: Documentation on dynamic node addition via `BayesianNetwork.add_node()` + `add_cpds()`. License: MIT. Reuse conclusion: reusable API calls.

## Current Architecture

### Components that exist and work

| Component | Location | What it does | Interface |
|-----------|----------|--------------|-----------|
| HetTGN | `agent/models/gnn/het_tgn.py` | Heterogeneous temporal GNN | `forward(data, id_map) → dict[type, Tensor[N,hidden_dim]]` |
| Trainer | `agent/models/gnn/trainer.py` | Self-supervised training loop | `train() → loss_dict`, `.model → HetTGN` |
| GraphBuilder | `agent/models/gnn/graph_builder.py` | Store → HeteroData | `build() → (HeteroData, IDMap, events)` |
| FeatureBuilder ABC | `agent/features/builders.py` | Store → EngineeredFeatures | `build(store, as_of) → list[EngineeredFeature]` |
| 2 builders | `agent/features/builders.py` | Convergence (3) + Macro (3) features | 6 features total |
| WorldModel | `agent/models/world_model.py` | DAG + Kalman orchestrator | `update(features, as_of) → list[BeliefState]` |
| Initial DAG | `agent/models/initial_graph.py` | 9-node expert causal graph | 3 regime/latent + 6 observed nodes |
| Feature gen DAG | `agent/pipeline/dags/feature_generation.py` | Pipeline stage | Schedule: 19:00 UTC weekdays |
| World model DAG | `agent/pipeline/dags/world_model_update.py` | Pipeline stage | Schedule: 19:30 UTC weekdays |
| WalkForward | `agent/quant/backtest.py` | Expanding-window backtester | `run(strategy, returns) → BacktestResult` |
| score_returns | `agent/quant/scoring.py` | Sharpe, Sortino, drawdown, etc. | `score_returns(returns) → dict` |
| PipelineStore | `agent/pipeline/store.py` | SQLite entity/feature/belief storage | Full CRUD API |

### What is missing (the gaps Phase 19 fills)

1. **GNN has never trained on real data.** `Trainer` exists but only `SyntheticGraphGenerator` has been used. Need a real-data training path.
2. **No GNN → Feature bridge.** After `forward()`, embeddings are `dict[str, Tensor]` but nothing converts them into `EngineeredFeature` records for the world model.
3. **World model DAG is too small.** Only 6 observed features (3 macro, 3 convergence). Entity-level signals from 12+ L2 tools are invisible to the DAG.
4. **No end-to-end pipeline DAG step.** GNN training + inference is not part of the scheduled pipeline.
5. **No validation.** World model beliefs have never been tested against market outcomes.

### Insertion points

- New `GNNFeatureBuilder` class in `agent/features/builders.py` (or new file `agent/features/gnn_builder.py`)
- New observed nodes + CPDs in `agent/models/initial_graph.py` (or override method)
- New DAG step in `agent/pipeline/dags/` for GNN training + inference
- New `Strategy` subclass in `agent/quant/` for belief-conditioned backtesting
- Updated `DEFAULT_BUILDERS` list in `feature_generation.py`
- Updated `_FEATURE_NAMES`, `_FEATURE_TO_OBS_INDEX`, Kalman dimensions in `world_model_update.py`

## Observations

### GNN embedding semantics

`HetTGN.forward(data, id_map) → dict[str, Tensor[N_type, hidden_dim]]` returns per-entity-type embedding matrices. Each row is a 64-dim vector encoding:
- Static features (entity type one-hot, obs count, recency, mean value)
- Learned message-passing context (neighbor information via HGT layers)
- Temporal memory (GRU state tracking historical event patterns)

These embeddings carry cross-entity structural signal that doesn't exist in the current feature set. For example:
- A company embedding encodes not just its own filings but also the behavior of its linked insider persons, lobbying clients, and patent portfolio
- A country embedding encodes bilateral event patterns (GDELT), vessel traffic, and company presence

### Aggregation strategy

Entity embeddings are per-entity. The world model operates at regime/macro level. Need an aggregation function:
- **Option A: Mean pooling per entity type** — simple, interpretable, might wash out signal
- **Option B: Attention-weighted pooling** — learnable attention over entities, more expressive
- **Option C: Anomaly statistics** — compute mean + std across entities, use z-score outliers as the signal
- **Option D: Top-k entity statistics** — for each type, take the most anomalous k entities' embedding norms, distances from mean, etc.

**Decision: Option C (anomaly statistics) for Phase 19.** Rationale:
1. The world model's DAG uses categorical discretization (3 bins per node). We need scalar features that are meaningful in a "calm / elevated / extreme" framework.
2. Anomaly z-scores map naturally to this: z < -1 = calm, -1 < z < 1 = normal, z > 1 = elevated.
3. No additional learned parameters — keeps the GNN → feature bridge deterministic.
4. Phase 20+ can upgrade to attention-weighted pooling if simple stats prove insufficient.

### New features from GNN (proposed)

For each entity type with edges (person, company, wallet, country, vessel):
1. `gnn.{type}_anomaly.spot` — mean L2 distance of entity embeddings from their type centroid (z-scored)
2. `gnn.{type}_activity.spot` — mean embedding norm (proxy for "how active is this entity cluster")
3. `gnn.cross_entropy.spot` — cross-type embedding distribution divergence (entity types becoming more/less correlated)

That's 2×5 + 1 = **11 new features**. Combined with existing 6, total = 17 features for the expanded DAG.

### Expanded DAG design

Current: 9 nodes (3 regime/latent, 6 observed)
New: add ~11 observed nodes for GNN features, plus potentially 1-2 latent nodes

**Key causal assumptions (expert-specified, learnable later):**
- Entity anomaly signals are driven by regime.stress (not regime.macro directly)
  - Rationale: entity-level anomalies (insider trading spikes, wallet movements) are stress indicators
- Cross-entropy feature is driven by both regime.macro and regime.stress
  - Rationale: entity correlations shift during both macro regime changes and stress events
- latent.risk_appetite influences entity activity levels
  - Rationale: risk-on → more trading, more filings, more activity

This preserves the existing DAG structure and extends it — no edges removed.

### Kalman filter expansion

Current: state_dim=3, obs_dim=6
New: state_dim stays 3 (same latent states), obs_dim=17 (all features)

The Kalman filter H matrix maps 17 observations to 3 latent states. New entity features map to the existing `stress_level`, `macro_momentum`, `liquidity_state` latents with initial H entries based on causal semantics:
- Entity anomaly → stress_level (primary)
- Entity activity → macro_momentum (secondary)
- Cross-entropy → liquidity_state (tertiary)

## Risks

### Technical risks
1. **GNN training instability on real data.** Synthetic data is clean; real CRT/AIS/lobbying data may have heavy tails. Mitigation: gradient clipping, learning rate warmup, robust loss variants.
2. **Feature sparsity.** If the pipeline hasn't been running long enough, few entity observations exist. GNN training may underfit. Mitigation: synthetic augmentation fallback if entity count < threshold.
3. **DAG validation failure.** pgmpy requires all CPDs to be consistent with the graph structure. Adding nodes with incorrect CPD dimensions will throw. Mitigation: validate after every node addition in tests.
4. **Embedding instability across training runs.** Different training seeds → different embedding geometry → different features. Mitigation: fix seed, use canonical centroid computation.

### Licensing risks
- All libraries (PyTorch, PyG, pgmpy, filterpy, numpy) are MIT/Apache/BSD. No risk.

### Testing risks
- Edge case: empty entity graph → GNN builder should return NaN-valued features with quality=0
- Edge case: single entity of a type → no meaningful anomaly score → quality=0
- Edge case: no links → embeddings degrade to per-type autoencoders (acceptable but lower signal)

## Data Requirements

### Required inputs
- PipelineStore with entity observations + links (from L2 tools)
- At minimum: entities from ≥3 tools with ≥10 observations each for meaningful GNN training
- Historical EngineeredFeatures (macro + convergence) for world model update

### What already exists locally
- All 15 L2 tools persist entities and 8 link types
- PipelineStore has full API for querying entities/observations/links
- `GraphBuilder.build()` converts store → HeteroData

### What still needs to be added
- GNN inference entry point (not just training)
- Feature builder for GNN embeddings
- Expanded DAG configuration
- Backtest entry point with beliefs-conditioned strategy

## Math/Algorithm Survey

### GNN training on real data
**Method:** Self-supervised TGN (Rossi et al. 2020) — predict next observation type + time delta for each entity. Already implemented in `Trainer`.
**Trusted source:** arXiv:2006.10637, our implementation in `trainer.py`.
**Key formula:** $L = L_{CE}(\hat{o}, o) + \lambda_t L_{MSE}(\hat{\Delta t}, \Delta t) + \lambda_c L_{contrastive}$
**No new math needed** — the trainer exists and is tested. We just need to run it on real data.

### Entity embedding aggregation
**Method:** Per-type centroid deviation (z-score anomaly).
For entity type $\tau$ with $n_\tau$ entities and embeddings $\{h_i^\tau\}_{i=1}^{n_\tau}$:

$$\mu_\tau = \frac{1}{n_\tau}\sum_{i=1}^{n_\tau} h_i^\tau, \quad \sigma_\tau = \sqrt{\frac{1}{n_\tau}\sum_{i=1}^{n_\tau} \|h_i^\tau - \mu_\tau\|^2}$$

$$\text{anomaly}_\tau = \frac{\|h_i^\tau - \mu_\tau\| - \mu_{\|h-\mu\|}}{\sigma_{\|h-\mu\|}}$$

The feature is the **mean anomaly z-score** across all entities of that type. A high value means entity behavior is diverging from normal — a stress signal.

**Activity feature:** $\text{activity}_\tau = \frac{1}{n_\tau}\sum \|h_i^\tau\|_2$ — average embedding norm, z-scored over time.

**Cross-entropy:** Wasserstein distance between pairwise type embedding distributions (expensive for large N — use sliced Wasserstein for efficiency). Or simpler: correlation between mean type embeddings vs historical baseline.

**Decision:** Use simple correlation-based cross-entity feature:

$$\text{cross\_entropy} = \frac{2}{K(K-1)} \sum_{\tau_1 < \tau_2} \cos(\mu_{\tau_1}, \mu_{\tau_2})$$

where $K$ = number of entity types with ≥2 entities. This measures how aligned entity types are — during crises, everything correlates.

**Trusted source:** Embedding distance metrics are standard (Mikolov et al. 2013 for cosine similarity of learned embeddings). Z-scoring is standard normalization.

### DAG expansion
**Method:** Expert-specified edges with weakly informative CPDs (same pattern as initial_graph.py). CPDs will be learned from data in a later phase via MLE fitting.
**Trusted source:** Koller & Friedman (2009) Ch. 17 (parameter learning in Bayesian networks).

### Walk-forward validation
**Method:** Expanding-window backtest with regime-conditional strategy. The `WalkForward` class already implements this. We need a `Strategy` subclass that converts beliefs → position weights.
**Trusted source:** de Prado (2018) "Advances in Financial Machine Learning" Ch. 12 (walk-forward cross-validation). Our `WalkForward` follows this design.

## Implementation Intent

### Concepts approved for implementation

1. **GNN training on real PipelineStore data** — `Trainer` already works, just need to run it against real store
2. **GNNFeatureBuilder** — new `FeatureBuilder` subclass converting embeddings → 11 EngineeredFeatures via centroid anomaly + activity + cross-entity correlation
3. **Expanded initial DAG** — add 11 observed nodes, ~8 new causal edges from regime/latent nodes
4. **Expanded Kalman** — increase obs_dim from 6 → 17, update H matrix
5. **GNN inference DAG step** — new pipeline stage between feature_generation and world_model_update
6. **Expanded world_model_update** — reads 17 features instead of 6
7. **RegimeStrategy** — Strategy subclass using beliefs from world model
8. **Backtest harness** — walk-forward test against market data (Yahoo Finance returns)

### Concepts rejected

- **Attention-weighted entity pooling** — adds learnable parameters at the bridge level; premature for Phase 19. Revisit in Phase 20.
- **Automatic DAG structure learning** — `discovery.py` exists but needs historical data; run manually later, not in the automated pipeline.
- **Multi-asset portfolio optimization** — Phase 21 (RL Policy). Phase 19 uses simple long/short SPY.

### Notes for the spec

- Break into 5 sub-phases: 19a (GNN real training), 19b (GNN→Feature), 19c (expanded DAG), 19d (E2E pipeline), 19e (backtest)
- Each sub-phase independently testable
- 19a can start immediately — no new code structure, just real data
- 19b is the hardest design work — the aggregation function is the core intellectual contribution
- 19c is mechanical — extend initial_graph.py following existing patterns
- 19d is glue — new DAG step wiring existing components
- 19e validates everything — the moment of truth

## Related

- [[world_model_bridge_spec]]
- [[entity_linking_layer]]
- [[tier1_tool_expansion]]
- [[gnn_guided_tool_expansion]]
- [[signal_protocol_feature_engineering]]
- [[convergence_detection]]
