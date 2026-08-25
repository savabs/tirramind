---
title: "Spec: Phase 9 — World Model"
tags:
  - doc/spec
  - layer/feature-engineering
  - layer/fusion
  - layer/learning
  - layer/surveillance
  - layer/world-model
  - phase/9
  - topic/world-model
---

# Spec: Phase 9 — World Model

## Goal

Build the probabilistic causal engine at the core of TirraMind's intelligence stack. The world model maintains a machine-readable belief state over hidden economic variables, updates beliefs as EngineeredFeature observations arrive, supports causal interventions for counterfactual reasoning, and outputs posterior distributions — never point estimates — for downstream signal fusion (Phase 10) and RL policy (Phase 11).

The world model must:

1. Represent causal relationships between economic variables as a directed acyclic graph (DAG)
2. Propagate observed features as evidence to update beliefs over latent variables
3. Track continuous latent states via Kalman filtering, conditioned on discrete regime
4. Support causal interventions via do-calculus for counterfactual queries
5. Learn and validate causal structure from historical data (when sample size permits)
6. Produce full distributions (mean + variance minimum, posterior samples when feasible)
7. Persist beliefs to a dedicated PipelineStore table
8. Run deterministically on the Pipeline layer — no LLM calls, no randomness at inference time (PRNG seeded)
9. Operate on numpy arrays as native compute format

**Layer placement:** Layer 3 (World Model) in the 7-layer stack. Reads Layer 2 features (EngineeredFeature values from the `features` table). Writes beliefs to a new `beliefs` table. No data fetching (Layer 1), no signal fusion (Layer 4), no policy (Layer 5).

**Research:** [[world_model|world_model]]

---

## Architecture Overview

The world model is a **hybrid system** combining two mathematically distinct frameworks:

### Component 1: Causal DAG (pgmpy)

Encodes the qualitative causal structure — *which variables cause which* — as a directed acyclic graph. Nodes are economic state variables (observed features, discrete regimes, latent causes). Edges are causal relationships. Conditional probability distributions (CPDs) quantify the strength and shape of each relationship. Belief propagation computes posteriors given observed evidence. The do() operator enables causal interventions.

**Why not pure BN for everything?** Discretizing continuous financial features into categorical bins destroys information. A yield curve slope of -0.15 vs -0.17 matters; binning both as "inverted" loses the signal. The DAG handles discrete variables (regimes, stress categories) where discrete CPDs are natural.

### Component 2: Continuous State-Space Filter (filterpy)

Tracks the quantitative dynamics — *how continuous hidden states evolve* — as a vector of latent economic variables (true stress level, growth momentum, liquidity conditions). A Kalman filter maintains a Gaussian belief state (mean vector + covariance matrix) and updates it as new EngineeredFeature values arrive. The transition model is conditioned on the active regime from the DAG, so the dynamics change when the economy shifts state.

**Why not pure state-space?** Kalman assumes linear-Gaussian dynamics with no structural breaks. Financial regimes violate both assumptions. The DAG handles the discrete regime variable; the Kalman filter handles the continuous dynamics within each regime. Together they model the full process.

### How they connect

```
EngineeredFeatures (features table)
        │
        ├─→ Discrete features ──→ [CausalDAG] ──→ regime posteriors
        │                              │
        │                              ▼
        └─→ Continuous features ──→ [StateFilter] ──→ continuous state posteriors
                                   (regime-conditioned)
        
Both outputs ──→ [BeliefState] ──→ beliefs table ──→ Phase 10 (Signal Fusion)
```

---

## Data Structures

### BeliefState (core output protocol)

```python
@dataclass(frozen=True)
class BeliefState:
    """A single posterior belief about one world-model variable.
    
    This is the output contract that Phase 10+ consumers depend on.
    Every belief is a distribution, never a point estimate.
    """
    
    # Identity
    variable_name: str        # From the graph node name, dotted: "regime.macro", "latent.stress_level"
    version: int              # Schema version of the world model that produced this
    
    # Temporal
    effective_at: float       # Unix epoch: when the underlying evidence was knowable
    computed_at: float        # Unix epoch: when the world model computed this belief
    
    # Distribution (Gaussian parameterization — extensible)
    dist_type: str            # "gaussian", "categorical", "empirical"
    mean: float | None        # For Gaussian: E[X]. For categorical: None
    variance: float | None    # For Gaussian: Var[X]. For categorical: None
    probabilities: dict[str, float] | None  # For categorical: {state: prob}. For Gaussian: None
    
    # Provenance
    evidence_count: int       # Number of features consumed for this update
    model_graph_hash: str     # SHA256 of the DAG structure (for reproducibility)
    
    # Quality
    confidence: float         # [0, 1]: overall confidence in this belief
    stale: bool               # True if no fresh evidence was available
```

### WorldModelGraph (DAG representation)

```python
@dataclass
class WorldModelGraph:
    """Wraps a pgmpy BayesianNetwork with expert-specified or learned structure."""
    
    nodes: list[NodeSpec]          # Node definitions with metadata
    edges: list[tuple[str, str]]   # (parent, child) directed edges
    cpds: dict[str, Any]           # Node name → fitted CPD
    graph_hash: str                # SHA256 of (sorted edges + node names)
    created_at: float
    
@dataclass(frozen=True)
class NodeSpec:
    """Metadata for a single node in the world model graph."""
    name: str                  # Unique node name
    node_type: str             # "observed", "latent", "regime"
    domain: str                # Feature domain ("convergence", "macro", "market")
    cardinality: int | None    # For discrete: number of states. None for continuous
    states: tuple[str, ...] | None  # For discrete: state labels
    feature_name: str | None   # For observed nodes: maps to EngineeredFeature.feature_name
```

---

## Files Affected

### Create

| File | Purpose | Layer |
|------|---------|-------|
| `agent/models/__init__.py` | Package exports | — |
| `agent/models/graph.py` | WorldModelGraph: DAG structure, NodeSpec, CPD management, pgmpy wrapper | L3 core |
| `agent/models/belief.py` | BeliefState dataclass + validation + serialization | L3 protocol |
| `agent/models/propagator.py` | BeliefPropagator: evidence injection → posterior computation via pgmpy | L3 inference |
| `agent/models/state_filter.py` | ContinuousStateFilter: Kalman filter wrapper, regime-conditioned dynamics | L3 state |
| `agent/models/initial_graph.py` | Expert-specified initial DAG: nodes, edges, prior CPDs for 6 features | L3 config |
| `agent/models/intervention.py` | InterventionEngine: do() queries, counterfactual estimation | L3 causal |
| `agent/models/discovery.py` | CausalStructureDiscovery: tigramite PCMCI wrapper for offline structure learning | L3 learning |
| `agent/models/world_model.py` | WorldModel: top-level orchestrator combining graph + filter + propagator | L3 top |
| `agent/pipeline/dags/world_model_update.py` | DAG definition for world model update cycle | pipeline |
| `tests/test_world_model_belief.py` | BeliefState protocol, validation, serialization | — |
| `tests/test_world_model_graph.py` | Graph structure, node management, CPD operations, hashing | — |
| `tests/test_world_model_propagator.py` | Evidence injection, posterior computation, missing evidence handling | — |
| `tests/test_world_model_state_filter.py` | Kalman filter, regime conditioning, predict/update cycle | — |
| `tests/test_world_model_initial_graph.py` | Expert graph correctness, CPD consistency, known-structure validation | — |
| `tests/test_world_model_intervention.py` | do() queries, counterfactual consistency, intervention on observed vs latent | — |
| `tests/test_world_model_discovery.py` | PCMCI wrapper, synthetic recovery, small-sample guard | — |
| `tests/test_world_model_integration.py` | End-to-end: features → world model → beliefs table | — |
| `tests/test_world_model_dag.py` | Pipeline DAG wiring + execution | — |

### Modify

| File | Change |
|------|--------|
| `agent/pipeline/store.py` | Add `beliefs` table to schema, add `store_belief()`, `store_beliefs_batch()`, `query_beliefs()`, `get_latest_belief()` methods |
| `agent/pipeline/dags/__init__.py` | Register `world_model_update` DAG |
| `pyproject.toml` | Add `pgmpy>=0.0.21` dependency (tigramite deferred to 9b) |

---

## Implementation Steps

### Sub-phase 9.1: Belief Protocol + Persistence

**Goal:** Define the output contract and storage before building any model logic. Downstream phases depend on the belief format, not the model internals. Getting the protocol right first prevents rework.

**Step 9.1.1: Define BeliefState dataclass + validation**
- Create `agent/models/__init__.py` (package init, exports)
- Create `agent/models/belief.py` with `BeliefState` frozen dataclass
- Implement `validate_belief(belief: BeliefState) -> list[str]` pure function
- Validation rules:
  - `variable_name` matches pattern `^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*){0,3}$`
  - `version >= 1`
  - `effective_at <= computed_at` (no information leakage)
  - Timestamps in valid range (2020-01-01 to now + 1 day)
  - `dist_type` in `{"gaussian", "categorical", "empirical"}`
  - Gaussian: `mean` and `variance` required, `variance >= 0`, both finite
  - Categorical: `probabilities` required, sums to 1.0 (within tolerance), all values ∈ [0, 1]
  - Empirical: `mean` may be present, `probabilities` may be present
  - `evidence_count >= 0`
  - `confidence` ∈ [0.0, 1.0]
  - `model_graph_hash` is 64-char hex string
- Implement `to_dict()` and `from_dict()` for serialization
- Test: `tests/test_world_model_belief.py` — happy path, each validation rule individually, round-trip serialization, invalid inputs for every field

**Step 9.1.2: Add beliefs table to PipelineStore**
- Add `beliefs` table to `_SCHEMA_SQL` in `agent/pipeline/store.py`:
  ```sql
  CREATE TABLE IF NOT EXISTS beliefs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      variable_name TEXT NOT NULL,
      version INTEGER NOT NULL,
      effective_at REAL NOT NULL,
      computed_at REAL NOT NULL,
      dist_type TEXT NOT NULL,
      mean REAL,
      variance REAL,
      probabilities_json TEXT,
      evidence_count INTEGER NOT NULL,
      model_graph_hash TEXT NOT NULL,
      confidence REAL NOT NULL,
      stale INTEGER NOT NULL DEFAULT 0
  );
  
  CREATE UNIQUE INDEX IF NOT EXISTS idx_beliefs_unique
      ON beliefs(variable_name, version, effective_at);
  
  CREATE INDEX IF NOT EXISTS idx_beliefs_lookup
      ON beliefs(variable_name, effective_at);
  ```
- Implement `store_belief(belief: BeliefState) -> int` — validate, serialize, insert
- Implement `store_beliefs_batch(beliefs: list[BeliefState]) -> list[int]` — atomic batch
- Implement `query_beliefs(variable_name, since, until, version, limit) -> list[dict]`
- Implement `get_latest_belief(variable_name, version) -> dict | None`
- Follow identical patterns as `store_feature()` / `query_features()` — same row-to-dict conversion, same error handling
- Test: storage round-trip, batch atomicity, query filtering, unique constraint on duplicate (variable_name, version, effective_at), empty result handling

---

### Sub-phase 9.2: Graph Structure + Expert DAG

**Goal:** Build the DAG representation and populate it with an initial expert-specified structure. This is the qualitative foundation — *what causes what* — before any inference machinery.

**Step 9.2.1: Implement WorldModelGraph + NodeSpec**
- Create `agent/models/graph.py`
- `NodeSpec` frozen dataclass: name, node_type ("observed" | "latent" | "regime"), domain, cardinality, states, feature_name
- `WorldModelGraph` class:
  - Constructor from list of NodeSpec + list of edge tuples
  - Internally creates a `pgmpy.models.BayesianNetwork` from the edge list
  - `add_node(spec: NodeSpec)`, `add_edge(parent, child)` with cycle detection
  - `set_cpd(node_name, cpd)` — wraps pgmpy TabularCPD or LinearGaussianCPD
  - `get_cpd(node_name)` → CPD
  - `validate() -> list[str]` — checks: DAG is acyclic, all nodes have CPDs, CPDs are consistent with parent cardinalities, all observed nodes have feature_name set
  - `graph_hash() -> str` — SHA256 of sorted(edges) + sorted(node_names), deterministic
  - `to_dict()` / `from_dict()` for persistence (so the graph structure can be stored/loaded)
  - `get_observed_nodes()`, `get_latent_nodes()`, `get_regime_nodes()` — filtered views
  - `get_parents(node)`, `get_children(node)` — delegates to pgmpy
- Test: creation, node/edge operations, cycle rejection, hash determinism, validation failures, serialization round-trip

**Step 9.2.2: Define the initial expert DAG**
- Create `agent/models/initial_graph.py`
- Function `build_initial_graph() -> WorldModelGraph` that constructs the first expert-specified causal graph

**The initial graph structure:**

This encodes the simplest defensible causal model over TirraMind's current 6 features + 3 latent variables. The causal semantics are:
- Macro regime and stress regime are latent discrete variables that govern the observable features
- Macro features (rate_momentum, yield_curve_slope, liquidity_pressure) are caused by macro regime
- Convergence features (stress_breadth, stress_intensity, regime_persistence) are caused by stress regime
- Macro regime and stress regime interact (macro stress can cause convergence stress and vice versa)

```
Nodes (9 total):
├── Latent/Regime:
│   ├── regime.macro          (discrete: expansion / contraction / crisis, card=3)
│   ├── regime.stress         (discrete: calm / elevated / extreme, card=3)
│   └── latent.risk_appetite  (discrete: risk_on / neutral / risk_off, card=3)
│
└── Observed (mapped to EngineeredFeatures):
    ├── obs.rate_momentum         → macro.rate_momentum.30d
    ├── obs.yield_curve_slope     → macro.yield_curve_slope.spot
    ├── obs.liquidity_pressure    → macro.liquidity_pressure.30d
    ├── obs.stress_breadth        → convergence.stress_breadth.7d
    ├── obs.stress_intensity      → convergence.stress_intensity.7d
    └── obs.regime_persistence    → convergence.regime_persistence.7d

Edges (11 total):
    regime.macro → obs.rate_momentum
    regime.macro → obs.yield_curve_slope
    regime.macro → obs.liquidity_pressure
    regime.macro → latent.risk_appetite
    regime.stress → obs.stress_breadth
    regime.stress → obs.stress_intensity
    regime.stress → obs.regime_persistence
    regime.stress → latent.risk_appetite
    latent.risk_appetite → obs.liquidity_pressure   (risk appetite affects liquidity)
    latent.risk_appetite → obs.stress_intensity     (risk appetite modulates stress perception)
    regime.macro → regime.stress                    (macro state influences stress)
```

- Define prior CPDs for all nodes:
  - Regime nodes: weakly informative priors (roughly uniform with slight center bias)
  - Observed nodes: CPDs conditioned on parents, initialized with broad distributions
  - Use `pgmpy.factors.discrete.TabularCPD` for discrete nodes
- These priors will be updated by MLE fitting once enough historical feature data accumulates
- Prior CPDs encode domain knowledge but are intentionally vague — the model should work even if the priors are wrong, as evidence will dominate after a few update cycles
- Test: graph validates, hash is stable, CPDs sum to 1 per parent config, observed nodes map correctly to feature names, no orphan nodes, no isolated components

---

### Sub-phase 9.3: Belief Propagation Engine

**Goal:** Given observed EngineeredFeature values, compute posterior beliefs over all nodes in the graph.

**Step 9.3.1: Implement BeliefPropagator**
- Create `agent/models/propagator.py`
- `BeliefPropagator` class:
  - Constructor takes `WorldModelGraph`
  - `propagate(evidence: dict[str, Any], as_of: float) -> list[BeliefState]`
    - `evidence`: maps observed node names to their observed values
    - Converts evidence into pgmpy format
    - For continuous observed features: discretize into the node's state space using configurable bin edges. The bin edges for each observed node are stored in NodeSpec or in a companion config.
    - Uses `pgmpy.inference.VariableElimination` (exact, appropriate for our 9-node graph)
    - Queries posterior for every non-evidence node
    - Returns `BeliefState` for each node:
      - Regime/latent nodes → `dist_type="categorical"`, `probabilities={state: prob}`
      - Observed nodes (non-evidence) → if inference produces marginal, include it; otherwise skip
    - Also returns BeliefState for evidence nodes with `dist_type="categorical"`, probabilities concentrated on observed state
  - `propagate_incremental(new_evidence: dict[str, Any], prior_beliefs: list[BeliefState]) -> list[BeliefState]`
    - Handles partial evidence updates without full re-inference
    - For small graph (9 nodes), this can simply run full propagation; optimization deferred
  - Quality weighting: if an EngineeredFeature has `quality < 1.0`, use virtual evidence (likelihood weighting) instead of hard evidence. pgmpy supports `virtual_evidence` via `TabularCPD` with likelihood ratios.
  - Missing evidence: features with `value=None` are simply not included in the evidence dict — the model marginalizes them out (this is how BNs naturally handle missing data)

- **Discretization strategy for continuous features:**
  - Each observed node has `bin_edges: list[float]` in its NodeSpec (e.g., for 3 states: `[-inf, -0.5, 0.5, inf]`)
  - `value_to_state(value: float, bin_edges: list[float]) -> str` maps continuous EngineeredFeature values to discrete states
  - Bin edges are initially set from domain knowledge (e.g., z-score boundaries) and can be updated, but they're not learned automatically in Phase 9a
  - This is an acknowledged information bottleneck — the research doc notes continuous features lose precision when discretized. Phase 9's state filter (9.4) preserves the continuous dynamics separately

- Test: synthetic graph with known CPDs → verify posteriors match hand-computed values. Evidence on one node updates its children's beliefs. Missing evidence marginalizes correctly. Virtual evidence produces intermediate posteriors. Empty evidence returns priors.

---

### Sub-phase 9.4: Continuous State Filter

**Goal:** Track continuous latent state variables via Kalman filtering, conditioned on the active discrete regime from the DAG.

**Step 9.4.1: Implement ContinuousStateFilter**
- Create `agent/models/state_filter.py`
- `ContinuousStateFilter` class:
  - Constructor takes:
    - `state_dim: int` — dimension of hidden state vector
    - `obs_dim: int` — dimension of observation vector (number of continuous EngineeredFeatures)
    - `regime_configs: dict[str, RegimeConfig]` — per-regime transition + noise parameters
  - `RegimeConfig` frozen dataclass:
    - `name: str` — regime label (matches DAG regime node states)
    - `F: np.ndarray` — state transition matrix (state_dim × state_dim)
    - `Q: np.ndarray` — process noise covariance (state_dim × state_dim)
    - `H: np.ndarray` — observation matrix (obs_dim × state_dim)
    - `R: np.ndarray` — observation noise covariance (obs_dim × obs_dim)
  - Internal state:
    - `x: np.ndarray` — state estimate vector (state_dim,)
    - `P: np.ndarray` — state covariance matrix (state_dim × state_dim)
    - `_filter: filterpy.kalman.KalmanFilter`
  - Methods:
    - `predict(regime: str) -> tuple[np.ndarray, np.ndarray]`
      - Sets filter F, Q from `regime_configs[regime]`
      - Calls `_filter.predict()`
      - Returns (predicted_state, predicted_covariance)
    - `update(observations: np.ndarray, quality: np.ndarray) -> tuple[np.ndarray, np.ndarray]`
      - `observations`: vector of continuous feature values
      - `quality`: per-feature quality weights ∈ [0, 1]
      - For quality < 1.0: inflate R by `1/quality` to reduce confidence in noisy observations
      - For missing features (NaN in observations): mask the missing entries in H and R, so the Kalman filter marginalizes over them. Practically: create a reduced observation by dropping missing rows from H, z, and R
      - Uses Joseph form for covariance update: $P = (I-KH)P(I-KH)^T + KRK^T$ for numerical stability
      - Returns (updated_state, updated_covariance)
    - `get_beliefs(node_names: list[str], as_of: float, graph_hash: str) -> list[BeliefState]`
      - Converts internal (x, P) into list of BeliefState records
      - Each state dimension maps to one variable_name
      - `dist_type="gaussian"`, `mean=x[i]`, `variance=P[i,i]`
    - `reset(x0: np.ndarray, P0: np.ndarray)` — reinitialize state

- **Mathematical specification:**
  - Transition model (per regime $r$): $\mathbf{x}_t = F_r \mathbf{x}_{t-1} + \mathbf{w}_t$, $\mathbf{w}_t \sim \mathcal{N}(0, Q_r)$
  - Observation model: $\mathbf{y}_t = H \mathbf{x}_t + \mathbf{v}_t$, $\mathbf{v}_t \sim \mathcal{N}(0, R)$
  - Predict: $\hat{\mathbf{x}}_{t|t-1} = F_r \hat{\mathbf{x}}_{t-1|t-1}$, $P_{t|t-1} = F_r P_{t-1|t-1} F_r^T + Q_r$
  - Update: standard Kalman gain $K = P_{t|t-1} H^T (H P_{t|t-1} H^T + R')^{-1}$ where $R' = R \cdot \text{diag}(1/\text{quality})$
  - References: Sarkka, "Bayesian Filtering and Smoothing" (2013), Ch. 4

- **Initial configuration for 6 features:**
  - State dim = 3: `[stress_level, macro_momentum, liquidity_state]`
  - Obs dim = 6: all 6 EngineeredFeatures
  - 3 regimes: "expansion", "contraction", "crisis" (matching DAG regime.macro states)
  - F matrices: near-identity with regime-specific drift (expansion: slight positive drift on momentum, crisis: negative drift, larger innovations)
  - Q matrices: small for expansion (stable dynamics), larger for crisis (volatile)
  - H matrix: linear mapping from 3 latent states to 6 observed features (initially identity-like, 6×3)
  - R matrix: observation noise estimated from historical feature variance

- Test: synthetic state-space model with known parameters → verify filter recovers true state. Regime switching changes dynamics. Missing observations handled by masking. Quality weighting inflates noise. Joseph form maintains P positive-definite over 1000 steps. Reset works.

---

### Sub-phase 9.5: World Model Orchestrator

**Goal:** Combine the graph-based propagator and the continuous state filter into a single coherent update cycle.

**Step 9.5.1: Implement WorldModel class**
- Create `agent/models/world_model.py`
- `WorldModel` class:
  - Constructor takes:
    - `graph: WorldModelGraph`
    - `propagator: BeliefPropagator`
    - `state_filter: ContinuousStateFilter`
  - `update(features: list[EngineeredFeature], as_of: float) -> list[BeliefState]`
    - The core update cycle, executed once per pipeline tick:
    1. **Map features to nodes**: match each `EngineeredFeature.feature_name` to the corresponding observed node in the graph via `NodeSpec.feature_name`
    2. **Discretize continuous features**: convert float values to discrete states for DAG evidence using bin edges
    3. **Propagate DAG evidence**: call `propagator.propagate(evidence, as_of)` → get posteriors for regime/latent nodes
    4. **Extract active regime**: from DAG posterior of `regime.macro`, take the MAP (most probable) state → this selects the regime for the Kalman filter
    5. **Kalman predict**: call `state_filter.predict(regime=active_regime)`
    6. **Kalman update**: construct observations vector + quality weights from continuous features, call `state_filter.update(obs, quality)`
    7. **Collect all beliefs**: combine categorical beliefs (from DAG) + Gaussian beliefs (from Kalman) into unified list[BeliefState]
    8. **Return** all beliefs for persistence
  - `query(variable_name: str) -> BeliefState | None`
    - Returns the most recent in-memory belief for a variable (before persistence)
  - `intervene(do_variable: str, do_value: Any) -> list[BeliefState]`
    - Delegates to InterventionEngine
    - Returns beliefs under the intervention, without modifying the model state
  - `get_graph_hash() -> str` — delegates to graph

- **Reconciliation between DAG and Kalman:**
  - The DAG and Kalman operate on overlapping but different perspectives
  - The DAG's categorical posteriors provide the regime context for the Kalman
  - The Kalman's continuous state estimates could in principle feed back into the DAG as soft evidence, but in Phase 9a this feedback loop is one-directional (DAG → Kalman) to avoid circular coupling. Bidirectional feedback is a Phase 9c concern.

- Test: end-to-end with synthetic features, verify DAG beliefs + Kalman beliefs are both produced, regime switching propagates to Kalman, empty features produce stale beliefs, ordering of update steps is correct

---

### Sub-phase 9.6: Intervention Engine

**Goal:** Support "what if" queries — if we force a variable to a specific value (do-calculus), what are the resulting beliefs?

**Step 9.6.1: Implement InterventionEngine**
- Create `agent/models/intervention.py`
- `InterventionEngine` class:
  - Constructor takes `WorldModelGraph`
  - `intervene(do_variable: str, do_value: Any, evidence: dict[str, Any] | None = None) -> list[BeliefState]`
    - Uses `pgmpy.BayesianNetwork.do([(do_variable, do_value)])` to construct the interventional distribution
    - Runs VariableElimination on the mutilated graph
    - Returns posterior beliefs under the intervention
    - Does NOT modify the original model's state — creates a temporary mutilated copy
  - `compare_intervention(do_variable: str, do_value: Any, evidence: dict[str, Any] | None = None) -> dict[str, dict]`
    - Returns dict: for each variable, `{"observational": BeliefState, "interventional": BeliefState, "causal_effect": float}`
    - `causal_effect` = difference in expected value (for Gaussian) or KL divergence (for categorical)
  - Input validation: do_variable must exist in graph, do_value must be a valid state for that node's cardinality
  - Latent/regime nodes can be intervened on (counterfactual: "what if we were in crisis?")

- Test: known graph with hand-computable interventional distributions, verify do() severs incoming edges, compare with observational posterior, intervene on regime node verifies downstream changes, intervene on leaf node has no causal effect on other nodes

---

### Sub-phase 9.7: Causal Structure Discovery (offline)

**Goal:** Provide infrastructure for data-driven causal structure learning from historical feature data. This is gated — it requires sufficient sample size (>200 observations minimum) and produces suggestions for graph updates, not automatic changes.

**Step 9.7.1: Implement CausalStructureDiscovery**
- Create `agent/models/discovery.py`
- `CausalStructureDiscovery` class:
  - Constructor takes:
    - `significance_level: float = 0.05`
    - `max_lag: int = 5` (tau_max for PCMCI)
    - `min_samples: int = 200` (guard: refuse to run with fewer)
  - `discover(data: np.ndarray, variable_names: list[str]) -> DiscoveryResult`
    - `data`: shape (T, N) — T timesteps, N variables
    - Guards: check T >= min_samples, N >= 2, no all-NaN columns
    - Import tigramite lazily (GPL dependency)
    - Initialize `tigramite.data_processing.DataFrame(data, var_names=variable_names)`
    - Use `ParCorr` as default CI test (linear, fast, sufficient for initial use)
    - Run `PCMCI.run_pcmciplus()` for contemporaneous + lagged discovery
    - Return `DiscoveryResult`:
      - `edges: list[DiscoveredEdge]` — each with source, target, lag, pvalue, strength, direction ("-->", "o-o", "x->", etc.)
      - `graph_array: np.ndarray` — shape (N, N, tau_max+1), raw PCMCI output
      - `summary: dict` — metadata (T, N, test used, alpha, runtime)
  - `compare_with_expert(discovered: DiscoveryResult, expert: WorldModelGraph) -> ComparisonReport`
    - Identifies edges present in expert but missing in discovery (and vice versa)
    - Reports p-values for expert edges that were not confirmed
    - Does NOT automatically modify the expert graph — returns a report for human review
  - `DiscoveredEdge` frozen dataclass: source, target, lag, pvalue, strength, link_type
  - `ComparisonReport` frozen dataclass: confirmed_edges, missing_edges, novel_edges, summary

- **Tigramite is GPL-3.0:** We call it as an external tool. No tigramite code is copied. Our own code remains independent. tigramite is an optional dependency — import only when `discover()` is called, with a clear error message if not installed.

- Test: synthetic data with known causal structure → verify discovery recovers true edges. Verify min_samples guard rejects small data. Verify lazy import and clear error when tigramite not installed. Comparison report correctly classifies confirmed/missing/novel.

---

### Sub-phase 9.8: Pipeline DAG Integration

**Goal:** Wire the world model into the Pipeline layer so it runs automatically after feature generation.

**Step 9.8.1: Create world model DAG**
- Create `agent/pipeline/dags/world_model_update.py`
- `build_world_model_dag(db_path: str) -> DAG`:
  - DAG name: `"world_model_update"`
  - Schedule: `"0 19 * * 1-5"` (same as feature_generation — runs on weekdays 19:00 UTC)
  - Dependency: should run after feature_generation completes (if DAG dependencies are supported, add explicit dependency; otherwise rely on schedule offset)
  - Single node: `"update_beliefs"`
    - Operator: `run_world_model_update(params, upstream) -> dict`
    - params: `{"db_path": db_path, "as_of": None}` (defaults to now)
- `run_world_model_update(params: dict, upstream: dict) -> dict`:
  1. Open PipelineStore
  2. Load or build WorldModelGraph (from initial_graph for now)
  3. Construct BeliefPropagator and ContinuousStateFilter
  4. Query latest features from store: `store.get_latest_feature(name)` for each observed node
  5. Construct list of EngineeredFeature from dicts
  6. Call `world_model.update(features, as_of)`
  7. Persist beliefs via `store.store_beliefs_batch(beliefs)`
  8. Return summary dict: `{"beliefs_count": N, "stale": M, "graph_hash": hash, "as_of": as_of}`

- Register in `agent/pipeline/dags/__init__.py`
- Test: DAG wiring, mock store with synthetic features → verify beliefs are persisted, verify stale beliefs when no features available, verify idempotent re-runs

**Step 9.8.2: Add pgmpy dependency**
- Add `pgmpy>=0.0.21` to `pyproject.toml` dependencies
- filterpy should already be available or add if missing
- tigramite is optional: add to `[project.optional-dependencies]` section as `discovery = ["tigramite>=5.2"]`

---

### Sub-phase 9.9: Edge Case Test Suite

**Goal:** Comprehensive edge case coverage across all modules. This is mandatory per workflow rules.

**Step 9.9.1: Comprehensive edge case tests**

Test categories (minimum, distributed across test files):

**BeliefState edge cases:**
- Gaussian with negative variance → validation error
- Categorical probabilities sum to 0.99 → passes within tolerance (1e-6)
- Categorical probabilities sum to 0.5 → validation error
- NaN/Inf in mean or variance → validation error
- Empty variable_name → validation error
- Invalid dist_type → validation error
- Future effective_at → validation error
- computed_at before effective_at → validation error
- graph_hash not 64 hex chars → validation error

**Graph edge cases:**
- Add edge that creates cycle → rejected
- Add duplicate node → idempotent or error
- Node with cardinality=0 → validation error
- CPD dimensions don't match parent cardinalities → validation error
- Observed node without feature_name → validation error
- Graph with disconnected components → validation warning
- Graph with only one node → valid (trivial graph)
- Hash stability across serialization round-trip

**Propagator edge cases:**
- Evidence on non-existent node → clear error
- Evidence on latent node → should work (hard evidence on hidden variable)
- All evidence missing → returns priors
- Quality = 0.0 → evidence is ignored (infinite noise)
- Quality = 1.0 → hard evidence
- Quality in (0, 1) → virtual evidence (intermediate posteriors)
- Numeric precision: probabilities should sum to 1.0 within tolerance after propagation

**State filter edge cases:**
- All observations NaN → predict only, no update
- Single observation available → partial update
- Regime not in configs → clear error
- Covariance matrix loses positive-definiteness → Joseph form prevents this (verify over 1000+ steps)
- Very large observation (1e10) → filter doesn't diverge
- Very small observation (1e-10) → still produces valid state
- Quality = 0.0 for all → R inflated to infinity, effectively no update
- Dimension mismatch between obs vector and obs_dim → error

**Integration edge cases:**
- Features table empty → stale beliefs with priors
- Features from one builder missing, other present → partial evidence
- Features with mixed quality levels → proper weighting
- Same world model update run twice → beliefs deduplicated by unique index (upsert or skip)
- Very old features (effective_at > 30 days ago) → beliefs marked stale
- Feature value = None (explicit missingness) → excluded from evidence

**Intervention edge cases:**
- Intervene on non-existent variable → clear error
- Intervene on observed variable → severs incoming edges
- Intervene on every variable → all beliefs are delta distributions
- Invalid do_value for node cardinality → error

---

## Edge Cases (summary)

- Missing features produce beliefs marked `stale=True`, not errors
- Features with `value=None` are excluded from evidence — the model marginalizes
- Quality < 1.0 uses virtual/soft evidence — not hard assignments
- Regime changes alter Kalman dynamics mid-stream without resetting state
- Graph structure changes (adding/removing edges) require model reconstruction
- Concurrent writes to beliefs table handled by SQLite WAL + unique index
- Tigramite not installed → `discover()` raises ImportError with helpful message, all other functionality works
- Very small graphs (< 5 nodes) are fine — VariableElimination is exact and fast
- Large graphs (> 50 nodes) would need approximate inference — out of scope for Phase 9a

---

## Testing Plan

### Unit tests per module
Each module has its own test file (listed in Files Affected). Each test file covers:
1. Happy path — normal operation
2. Validation failures — every rule individually
3. Missing/None inputs — proper handling
4. Numerical edge cases — NaN, Inf, very large/small values
5. Serialization — round-trip to_dict/from_dict

### Integration tests
- `test_world_model_integration.py`: full pipeline from EngineeredFeature → WorldModel.update() → BeliefState → store → query back
- `test_world_model_dag.py`: DAG creation, execution with mock store, verification of outputs

### Synthetic verification
- Create a synthetic ground-truth Bayesian network with known CPDs
- Generate data from it, feed through world model, verify posteriors converge to true values
- This validates both the graph + propagator + filter work correctly end-to-end

### Regression
- After each sub-phase, run all prior tests to ensure nothing broke
- Total expected test count: ~300-400 across 9 test files

---

## Dependencies

| Library | Version | License | Purpose | Required? |
|---------|---------|---------|---------|-----------|
| pgmpy | >=0.0.21 | MIT | BayesianNetwork, CPDs, VariableElimination, do-calculus | Yes |
| filterpy | >=1.4.5 | MIT | KalmanFilter for continuous state tracking | Yes |
| numpy | >=1.24 | BSD | Array operations, state vectors, covariance matrices | Yes (existing) |
| tigramite | >=5.2 | GPL-3.0 | PCMCI causal discovery (offline, optional) | Optional |

---

## Sequencing Rationale

The sub-phases are ordered so that each builds on the previous one and is independently testable:

1. **9.1 (Belief Protocol + Persistence)** — Define the output format first. Everything else produces BeliefState. Getting this wrong forces rework in everything downstream.
2. **9.2 (Graph + Expert DAG)** — Build the DAG structure before inference. You can't propagate beliefs without a graph. Expert specification before discovery because we don't have enough data for reliable discovery yet.
3. **9.3 (Belief Propagation)** — First useful output. Given features + graph → beliefs. This is the minimum viable world model.
4. **9.4 (Continuous State Filter)** — Adds the continuous dynamics that discrete BN alone can't capture. Depends on regime posteriors from 9.3.
5. **9.5 (World Model Orchestrator)** — Combines 9.3 + 9.4 into a single update call. Depends on both being complete.
6. **9.6 (Intervention Engine)** — Adds causal queries. Depends on the graph (9.2) but not on the filter (9.4). Placed here because it's less critical than the core update loop.
7. **9.7 (Causal Discovery)** — Offline structure learning. Fully independent of online inference. Placed late because it requires historical data and is not needed for the world model to function.
8. **9.8 (Pipeline DAG)** — Wiring. Depends on everything except 9.7. Placed second-to-last because it's integration, not core logic.
9. **9.9 (Edge Case Tests)** — Full sweep after everything works. Some edge case tests will be written within each sub-phase too, but 9.9 is the comprehensive audit.

---

## What This Phase Does NOT Do

- **Signal fusion** (Phase 10) — combining multiple beliefs into trading signals. World model outputs raw beliefs.
- **Portfolio optimization** (Phase 11) — converting beliefs into positions. World model has no concept of assets or trades.
- **Feature expansion** — adding more FeatureBuilders beyond the existing 6. This would strengthen the world model but is orthogonal work.
- **Automatic graph updates** — discovery (9.7) proposes changes but a human reviews them. No auto-mutation of the live graph.
- **Nonlinear filtering** — UKF/particle filter deferred. Kalman is sufficient for the initial 3-state model. Can upgrade later.
- **Bidirectional DAG↔Kalman coupling** — the Kalman feeds on regime from DAG, but doesn't feed back. Deferred to Phase 9c.
- **Time-varying graph structure** — the DAG is static per model version. Regime-varying structure via RPCMCI is future work.

---

## Related

- [[world_model|Research: World Model]]
- [[convergence_detection]]
- [[signal_protocol_feature_engineering]]
- [[rl_layer]]
