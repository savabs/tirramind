---
title: "Feature: Tier 7 — Self-Modifying Structure"
tags:
  - doc/research
  - phase/25
  - topic/self-improving
  - topic/world-model
  - topic/meta-learning
  - layer/world-model
  - layer/learning
---

# Feature: Tier 7 — Self-Modifying Structure

## Goal

Implement Changes 13 (self-modifying graph schema) and 14 (meta-learned scheduling) from [[learned_vs_handcoded_architecture_spec]]. After this tier, the system moves from 82% → 90% learned.

**Change 13:** The DAG topology itself is a learned object. Structure learning on live data adds/removes/merges nodes. Edge confidence is tracked over time.

**Change 14:** Fixed intervals (7d CPD fit, 90d structure refinement, daily GNN train, etc.) are replaced by a meta-learner that decides *when* each component should retrain, *how much history* to use, and *which components* are stale.

## Search Log

- GitHub: "online Bayesian structure learning DAG", "edge posterior probability pgmpy", "meta-learning retraining schedule bandit"
- Documentation: pgmpy structure_score API (BIC, BDeu, K2), pgmpy HillClimbSearch, pgmpy structure_prior_ratio
- Papers: Kendall et al 2018 (uncertainty-weighted multi-task), Wang et al 2016 "Learning to reinforcement learn", Donancio et al 2024 "Dynamic Learning Rate: A Bandit Approach"

## Documentation Reviewed

- pgmpy `structure_score.BIC`: provides `local_score(variable, parents)` decomposable scoring. Edge contribution = `local_score(child, parents_with) - local_score(child, parents_without)`. Available in pgmpy 1.1.0 (installed).
- pgmpy `HillClimbSearch`: already used in Tier 4's `refine_structure()`. Supports warm-start from existing DAG, max_indegree, fixed_edges, white/black-listed edges.
- pgmpy `structure_prior_ratio`: can incorporate structural priors (expert edge probability) into scoring.

## Current Architecture

### Graph Schema (Change 13 surface)
- **`agent/models/initial_graph.py`**: 20 nodes (3 regime + 17 observed), 19 expert edges, NodeSpec dataclass with name/type/domain/cardinality/states/bin_edges
- **`agent/models/graph.py`**: `WorldModelGraph` class, `add_edge()`, `remove_edge()` methods, constraint validation (acyclicity, regime-root)
- **`agent/models/world_model.py:340-533`**: `refine_structure()` — constrained hill-climb with BIC, warm-start from current graph, max_indegree=4, enforces regime-root and observed-cannot-parent-latent constraints
- **`agent/pipeline/dags/world_model_update.py:537-595`**: `_maybe_refine_structure()` — runs every 90 days, stores learned edges via `_persist_learned_edges()`

### Scheduling (Change 14 surface)
- **CPD fit interval**: 7 days (hardcoded in `world_model_update.py:631`)
- **Structure refinement interval**: 90 days (hardcoded in `world_model_update.py:631`)
- **GNN training**: daily, always trains, 10 epochs, batch_size=64 (hardcoded in `gnn/trainer.py:299-300`)
- **RL training**: whenever 500+ alerts available (`rl_training.py:45`)
- **Feature history window**: 90 days (`world_model_update.py:544`)
- **Fit marker storage**: `PipelineStore.store_data()` with source keys `world_model_cpd_fit`, `world_model_structure_fit`, `world_model_learned_edges`

### Existing Learning Components (Tier 3-6 foundation)
- `agent/learning/param_optimizer.py`: BayesianParamOptimizer (GP-BO) — already knows how to optimize continuous parameters with Gaussian processes
- `agent/learning/threshold_optimizer.py`: Detector self-calibration
- `agent/learning/tool_router.py`: ToolRoutingBandit (Thompson Sampling Beta posteriors) — already implements the bandit pattern we'll reuse for scheduling

## Observations

### What already exists
1. **Structure learning works**: `refine_structure()` uses HillClimbSearch with BIC, constraint enforcement. Edge add/remove operations are clean.
2. **Edge persistence is in place**: learned edges are stored/queried via `PipelineStore`.
3. **Fit scheduling is centralized**: `_should_fit()` function checks interval-based markers stored in DB.
4. **Thompson Sampling bandit is proven**: `ToolRoutingBandit` handles the explore/exploit pattern cleanly.

### What is missing
1. **No edge confidence tracking**: edges are binary (present/absent). No posterior probability, stability metric, or contribution score.
2. **No DAG versioning**: when structure changes, old beliefs are not marked as stale relative to the new structure.
3. **No component performance history**: no table tracking the information gain from CPD refits, structure changes, or GNN retraining.
4. **No dynamic intervals**: all intervals are hardcoded constants, not learned.
5. **No node addition/removal/merge**: structure learning only adds/removes edges between existing nodes. It cannot create new latent nodes or merge redundant observed nodes.

### Important constraints
- pgmpy structure learning (HillClimbSearch) operates on a fixed node set. Node addition/removal requires creating a new BayesianNetwork object — we cannot modify the node set in-place.
- Node merge/creation is a fundamentally different operation from edge learning and adds significant complexity. The spec says "adds/removes/merges nodes" but the edge-level self-modification + dynamic node features via GNN attention already captures most of the self-modification value. We should defer full node creation/merge to Tier 8 where it aligns with "self-extending entity ontology."

## Risks

1. **Edge instability**: With limited financial data (daily snapshots), BIC scores may oscillate — edges added one quarter may be removed the next. Mitigation: edge confidence tracking with hysteresis (only modify edges with confidence above/below threshold for N consecutive windows).
2. **CPD cold-start on edge changes**: When an edge is added/removed, the child's CPD resets to uniform. Brief degradation until next fit. Mitigation: keep old CPD as fallback and blend during transition.
3. **Stale beliefs**: Structure changes invalidate beliefs computed under the old graph. Mitigation: mark beliefs stale on structure change, force full re-inference.
4. **Scheduling over-exploration**: Meta-scheduler might try too many refit combinations on limited compute budget. Mitigation: conservative priors (centered on current hardcoded intervals), bounded action space.
5. **Insufficient signal for scheduling reward**: We need enough DAG cycles to observe the effect of refit decisions on downstream metrics. Mitigation: use information-theoretic metrics (log-likelihood improvement, BIC δ) as fast proxy rewards rather than waiting for portfolio Sharpe.

## Math/Algorithm Survey

### Change 13: Edge Confidence Scoring

**Approach: BIC-δ edge contribution with rolling stability.**

For each edge (parent → child), the marginal contribution is:

$$\Delta\text{BIC}_{e} = \text{BIC}(X_i \mid \text{Pa}(X_i) \cup \{e\}) - \text{BIC}(X_i \mid \text{Pa}(X_i) \setminus \{e\})$$

Positive $\Delta\text{BIC}$ → edge hurts fit (should remove). Negative → edge helps. We compute this on rolling windows (last 30d, 60d, 90d) and track:

- **Edge confidence**: sigmoid of -mean(ΔBICs across windows). Near 1 = strong support, near 0 = weak.
- **Edge stability**: 1 - std(ΔBICs)/|mean(ΔBICs)|. High = consistent, low = oscillating.

**Decision rule (hysteresis)**:
- Add edge: confidence > 0.7 AND stability > 0.5 for 2+ consecutive evaluations
- Remove edge: confidence < 0.3 AND stability > 0.5 for 2+ consecutive evaluations
- Otherwise: keep current state (hysteresis prevents flip-flopping)

**Source:** Decomposable BIC scoring is standard (Schwarz 1978, Chickering 2002 "Optimal Structure Identification with Greedy Search"). The hysteresis window is repo-specific engineering on top of standard scoring because financial data is noisy and regime-dependent.

**Implementation:** pgmpy `BIC.local_score(variable, parents)` gives per-node decomposable scores. Verified available in pgmpy 1.1.0.

### Change 14: Meta-Learned Scheduling

**Approach: Per-component Thompson Sampling bandit over discretized intervals.**

Each schedulable component (cpd_fit, structure_refine, gnn_train, rl_train) is treated as a separate bandit problem:

- **Arms** = candidate intervals. E.g., cpd_fit arms: [3d, 5d, 7d, 14d, 30d]. Structure arms: [30d, 60d, 90d, 180d].
- **Reward** = information gain from the refit:
  - CPD fit: ΔBIC (before vs after fit) normalized
  - Structure refine: number of confident edge changes (edges whose confidence crossed threshold)
  - GNN: validation loss improvement
  - RL: SAC Sharpe improvement over last window
- **Posterior**: Beta(α, β) per arm, updated online.
- **Decision**: Thompson sample per component; highest sample wins.

This reuses the exact `ToolRoutingBandit` pattern from Change 12.

**Why bandit over full RL:** The action space is small (5 discrete intervals per component), the reward signal is delayed (only observed at next refit), and the data is extremely limited (one observation per refit cycle). Full RL would need thousands of episodes we'll never have. A bandit with Beta posteriors converges with O(10) observations per arm.

**Trusted source:** Thompson Sampling for Bernoulli bandits is well-established (Thompson 1933, Chapelle & Li 2011 "An Empirical Evaluation of Thompson Sampling"). The application to scheduling intervals is repo-specific engineering.

**Alternative considered:** Gaussian process over continuous interval space (like BayesianParamOptimizer already in the codebase). Rejected because: (a) the interval space is naturally discrete and small, (b) we have very few observations per component, (c) Beta posterior is more robust with sparse data than GP.

## Implementation Intent

### Approved for implementation:
1. **Edge confidence tracking** with BIC-δ rolling windows + hysteresis-based add/remove decisions (Change 13)
2. **DAG versioning** via hash + stale-belief marking on structure change (Change 13)
3. **Per-component scheduling bandit** with Thompson Sampling over discrete interval arms (Change 14)
4. **Component performance history** storage in PipelineStore (Change 14)

### Deferred to Tier 8:
- Node creation/merge (full ontology expansion) — aligns with Change 16's self-extending ontology
- Continuous interval optimization via GP — may revisit if bandit arms prove too coarse

### Rejected:
- Full RL meta-scheduler (too data-hungry for this setting)
- Online MCMC structure learning (computationally expensive, pgmpy doesn't support it natively)

## Related

- [[learned_vs_handcoded_architecture_spec]] — Master spec
- [[tier7_self_modifying_structure_spec]] — Spec (to be created)
- [[learned_vs_handcoded_audit]] — Original audit
- [[chat_checkpoint_2026-04-15_tier6_complete]] — Previous checkpoint
