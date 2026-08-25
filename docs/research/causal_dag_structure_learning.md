---
title: "Research: Causal DAG Structure Learning (Change 3)"
tags:
  - doc/research
  - phase/25
  - topic/self-improving
  - topic/world-model
  - topic/structure-learning
  - layer/world-model
---

# Research: Causal DAG Structure Learning (Change 3)

## Current Architecture

- **Expert graph:** 20 ternary nodes (2 regime, 1 latent, 17 observed), 19 hand-authored edges in `agent/models/initial_graph.py`
- **WorldModel** (`agent/models/world_model.py`): orchestrates DAG propagation + Kalman. Has `fit_cpds()` (Tier 2) but no structure learning.
- **WorldModelGraph** (`agent/models/graph.py`): wraps pgmpy `BayesianNetwork`. Has `add_edge()` (with cycle check), `to_dict()`/`from_dict()`, `graph_hash()`. Missing: `remove_edge()`.
- **world_model_update DAG** (`agent/pipeline/dags/world_model_update.py`): rebuilds graph from `build_initial_graph()` every run. Has `_maybe_fit_params()` on 7-day schedule for CPD/Kalman fitting.
- **No structure persistence**: graph always rebuilt from hardcoded `ALL_EDGES`.

## Observations

1. `fit_cpds()` builds a discretized DataFrame using `_discretize()` — same data format needed for HillClimbSearch.
2. `WorldModelGraph.add_edge()` already has cycle checking via networkx.
3. `WorldModelGraph` has no `remove_edge()` — need to add it.
4. The world_model_update DAG rebuilds the graph from scratch each run, so any learned structure changes are lost unless persisted.
5. `PipelineStore.store_data()` / `query_data()` can persist arbitrary JSON, suitable for storing learned edge sets.

## pgmpy API (Verified on v1.1.0)

### New API (preferred, sklearn-style)
```python
from pgmpy.causal_discovery import HillClimbSearch, ExpertKnowledge
from pgmpy.base import DAG

hc = HillClimbSearch(
    scoring_method='bic-d',      # discrete BIC
    start_dag=DAG(...),           # warm start from expert graph
    max_indegree=4,               # prevent overfit
    expert_knowledge=ExpertKnowledge(
        forbidden_edges=[...],
        required_edges=[...],
        temporal_order=[...],
    ),
    return_type='dag',
    show_progress=False,
    epsilon=1e-4,                 # min score improvement to accept move
    max_iter=1000000,
)
hc.fit(df)  # returns self
learned_dag = hc.causal_graph_   # type: pgmpy.base.DAG
```

### Key findings
- Old `pgmpy.estimators.HillClimbSearch` works but emits FutureWarning (deprecated)
- New API: constructor takes all params, `.fit(df)` returns self, result in `.causal_graph_` attribute
- `ExpertKnowledge` accepts `forbidden_edges`, `required_edges`, `temporal_order`, `search_space`
- `'bic-d'` scoring confirmed working on ternary data
- Successfully recovered known A→B→C structure from 500 synthetic samples with constraints + warm-start

## Structural Constraints

Per spec and domain knowledge:
1. **Regime nodes must be roots** — no parents allowed for `regime.macro`, `regime.stress`
   - Forbidden: any `(X, regime.macro)` or `(X, regime.stress)` where X is non-regime
   - Exception: `regime.macro → regime.stress` is allowed (inter-regime edge)
2. **Observed → latent forbidden** — observed nodes cannot parent latent nodes
3. **Max in-degree = 4** — prevents overfitting on small data
4. **Acyclicity** — enforced by pgmpy HillClimbSearch inherently
5. **Temporal ordering** — `[regime_nodes] → [latent_nodes] → [observed_nodes]`
6. **BIC improvement threshold** — only accept changes with BIC improvement > epsilon (conservative)

### Forbidden edges (concrete)
- All `(obs.*, regime.macro)`, `(obs.*, regime.stress)` — observed can't cause regime
- All `(latent.*, regime.macro)`, `(latent.*, regime.stress)` — latent can't cause regime
- All `(obs.*, latent.risk_appetite)` — observed can't directly cause latent

### Required edges (none initially)
- Could require `regime.macro → regime.stress` as structural prior, but letting data decide is better

## Risks

1. **Sparse data:** With 20 ternary nodes and potentially limited samples, structure learning may not have enough statistical power to discover subtle edges. BIC's complexity penalty helps prevent spurious edges.
2. **Missing data:** Latent/regime nodes have no direct observations. HillClimbSearch on a DataFrame with those columns as NaN — structure learning will work on observed nodes only (latent/regime columns excluded from data but present in start_dag).
3. **CPD invalidation:** Changing edges invalidates existing CPDs — must re-fit CPDs after structure changes.
4. **Warm-start DAG format:** pgmpy `DAG` (from `pgmpy.base`) vs `BayesianNetwork` (from `pgmpy.models`) — HillClimbSearch expects `DAG`, not `BayesianNetwork`. Need to extract nodes+edges into a fresh `DAG` for warm start.

## Implementation Plan

### Core method: `WorldModel.refine_structure()`
1. Build discretized DataFrame from feature_history (reuse `_discretize()` logic from `fit_cpds()`)
2. Extract current graph edges into a pgmpy `DAG` for warm start
3. Build ExpertKnowledge constraints (forbidden_edges, temporal_order)
4. Run `HillClimbSearch.fit(df)` with bic-d scoring
5. Diff learned edges vs current edges → compute added/removed/reversed
6. Apply changes to `WorldModelGraph` (add_edge + remove_edge)
7. CPDs for changed neighborhoods are invalidated — mark for re-fit
8. Return audit dict: {edges_added, edges_removed, bic_before, bic_after, n_samples}

### Persistence
- Store learned edge set via `PipelineStore.store_data("learned_structure", ...)`
- world_model_update.py loads persisted structure instead of always using `ALL_EDGES`

### Scheduling
- Quarterly (per spec) — much less frequent than CPD fitting (7d)
- Source: `_STRUCTURE_FIT_SOURCE = "world_model_structure_fit"`

## Data Requirements

- Minimum ~200 daily snapshots (per node cardinality 3, ~20 observed columns)
- Same feature history used by `fit_cpds()` — daily discretized snapshots of EngineeredFeature values

## Math/Algorithm Survey

**Score-based structure learning with BIC:**
$$\text{BIC}(G) = \sum_i \left[\log P(X_i | \text{Pa}_G(X_i)) - \frac{d_i}{2} \log N\right]$$

BIC naturally penalizes model complexity ($d_i$ = free parameters per CPD), preventing spurious edge addition when data is limited. Conservative choice — BDeu has stronger Bayesian flavor but requires ESS tuning for structure search; BIC is parameter-free.

**Hill-climbing search** iterates: try all single-edge add/remove/reverse → accept the move with the best score improvement → stop when no improvement exceeds epsilon. Warm-starting from the expert graph means the search only needs to find local improvements, not discover structure from scratch.

**Trusted source:** Koller & Friedman, "Probabilistic Graphical Models" (2009), Ch. 18 (Structure Learning). pgmpy's implementation follows this standard approach.

## Related

- [[learned_vs_handcoded_audit]]
- [[learned_vs_handcoded_architecture_spec]]
- [[learned_architecture_impl]]
- [[world_model]]
- [[world_model_bridge]]
