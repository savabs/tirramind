---
title: "Spec: Self-Improving Architecture — End-to-End Learned TirraMind"
tags:
  - doc/spec
  - phase/24
  - topic/learning-agent
  - topic/self-improving
  - topic/world-model
  - topic/meta-learning
  - layer/world-model
  - layer/fusion
  - layer/learning
---

# Spec: Self-Improving Architecture — End-to-End Learned TirraMind

**Date:** 2026-04-14
**Research:** [[learned_vs_handcoded_audit]]
**Goal:** Map the exact architectural changes needed to move TirraMind from ~25% learned / ~75% hand-coded to an end-to-end learning system where the agent improves its own structure, parameters, and strategy from experience.

---

## Design Principle

**The representation is hand-built; the intelligence is learned.**

We keep hand-coding schemas, invariants, safety constraints, and factual relationships that are explicitly stated by source data. We push everything else — causal structure, observation mappings, scoring weights, detector thresholds, reward shaping, state representation, and goal discovery — into learned components.

The target state: a system where running more data through it makes it measurably smarter, without human intervention in the math or parameter tuning.

---

## Priority-Ordered Architectural Changes

Organized by impact × feasibility. Each change describes: what's currently hand-coded, what replaces it, the mathematical basis, the implementation surface, and how to verify it works.

---

### Change 1: Close the Belief→Policy Loop (Critical, Low Effort)

**Currently:** In `inference.py:299-302`, world model beliefs are stubbed out (`beliefs = []`). The SAC policy gets zero-padded belief features. The learned world model exists (Kalman + DAG) but its output never reaches the learned policy.

**Change:** Wire `PipelineStore.query_latest_beliefs()` into the inference DAG so persisted beliefs flow into `InstrumentStateAssembler.assemble()`.

**Why it matters:** This is the single cheapest change that turns two disconnected learned components (world model + SAC) into one connected learned system. Without it, SAC is learning on partial state.

**Implementation:**
- In `_sac_inference()`, after loading entity alerts, query latest beliefs from store
- Filter for beliefs with `entity_id` matching instrument tickers
- Pass to assembler (the belief_block already exists in the state layout, it's just never populated with real data)

**Verification:** SAC inference metadata should show `n_beliefs > 0`. Walk-forward backtest Sharpe should not degrade (and may improve).

**Effort:** ~30 LOC. One afternoon.

---

### Change 2: Learn World Model Parameters from Data (Critical, Medium Effort)

**Currently:** The world model has two entirely hand-coded subsystems:

1. **Expert causal DAG** (`initial_graph.py`): 20 nodes, 19 edges, CPDs with hand-set probability tables. "Weakly informative priors" that never update from data.
2. **Kalman filter** (`world_model_update.py`): H (17×3 observation matrix), R (17×17 noise), F (3×3 per regime), Q (3×3 per regime) — all constants.

**Change:** Replace fixed parameters with data-fitted parameters using two well-established methods:

#### 2a: CPD Learning via MLE/Bayesian Estimation

The expert DAG structure (which nodes connect) stays fixed for now. But the CPD *values* should be learned from accumulated feature data.

**Math:** Maximum Likelihood Estimation for tabular CPDs is a counting operation:
$$\hat{P}(X_i = x | \text{Pa}(X_i) = \pi) = \frac{N(X_i = x, \text{Pa}(X_i) = \pi) + \alpha}{N(\text{Pa}(X_i) = \pi) + \alpha \cdot |X_i|}$$

where $\alpha$ is a Dirichlet smoothing parameter (keeps the Bayesian flavor of the current "weakly informative" priors but lets data dominate as $N$ grows).

**Implementation:**
- pgmpy already supports `BayesianEstimator` with `BDeu` or `K2` priors
- Add a `fit_cpds()` method to `WorldModel` that takes historical features, discretizes them, and calls `bn.fit(data, estimator=BayesianEstimator, prior_type='BDeu')`
- Run this as a periodic step in the world_model_update DAG (e.g., weekly re-fit from last 90 days of features)
- Keep the expert CPDs as warm-start priors; data updates them

**Verification:** On synthetic data with known CPDs, fitted CPDs should converge to ground truth. On real data, belief quality metrics (calibration, log-loss) should improve vs fixed priors after N>500 observations.

#### 2b: Kalman Parameter Estimation via EM

**Math:** The EM algorithm for linear Gaussian state-space models (Shumway & Stoffer, "Time Series Analysis and Its Applications", Ch. 6) alternates:
- **E-step:** Kalman smoother computes $E[x_t | y_{1:T}]$ and $\text{Cov}(x_t, x_{t-1} | y_{1:T})$
- **M-step:** Update $\hat{F}, \hat{Q}, \hat{H}, \hat{R}$ via closed-form MLE:

$$\hat{F} = \left(\sum_t E[x_t x_{t-1}^T]\right) \left(\sum_t E[x_{t-1} x_{t-1}^T]\right)^{-1}$$

$$\hat{Q} = \frac{1}{T} \sum_t \left(E[x_t x_t^T] - \hat{F} E[x_{t-1} x_t^T]\right)$$

Similarly for H and R. filterpy has partial EM support; statsmodels `UnobservedComponents` does full EM.

**Implementation:**
- Add `fit_filter_params()` to `ContinuousStateFilter` that runs EM on historical (features, regime_labels) pairs
- Run after CPD fitting (same periodic schedule)
- Keep current hand-set params as initialization for EM (warm start → faster convergence)
- Persist fitted params to PipelineStore alongside beliefs

**Verification:** On synthetic state-space data, EM-fitted F/Q/H/R should converge to ground truth within 20 iterations. On real data, Kalman innovation sequence should be closer to white noise (Ljung-Box test) after fitting.

**Effort:** ~200 LOC for 2a, ~300 LOC for 2b. One week total.

---

### Change 3: Learn Causal DAG Structure (Important, Medium Effort)

**Currently:** The 19 edges in `initial_graph.py` are expert-authored. The system cannot discover that, say, `obs.vessel_activity` directly influences `obs.company_anomaly` unless a human adds that edge.

**Change:** Use structure learning to discover or refine edges from data.

**Math:** Score-based structure learning with BIC/BDeu score:

$$\text{BIC}(G) = \sum_i \left[\log P(X_i | \text{Pa}_G(X_i)) - \frac{d_i}{2} \log N\right]$$

where $d_i$ is the number of free parameters in the CPD for node $i$. pgmpy implements `HillClimbSearch`, `K2Score`, `BDeuScore`, and `BicScore`.

**Approach:** Hybrid — start from the expert graph, allow hill-climbing to add/remove/reverse edges, but enforce constraints:
- Regime nodes must be roots (no parents)
- Observed nodes cannot be parents of latent nodes
- Acyclicity (enforced by pgmpy)
- Maximum in-degree of 4 (prevents overfitting)

**Implementation:**
- Add `refine_structure()` to `WorldModel` that runs constrained hill-climb on accumulated data
- Use current expert graph as initial structure (warm start)
- Accept new edges only if BIC improvement > threshold (conservative)
- Log all structural changes to PipelineStore for auditability
- Run quarterly (not daily — structure should be stable)

**Verification:** On synthetic data with known structure, recovery should exceed 80% F1 on edge presence. On real data, hold-out log-likelihood should improve vs fixed structure.

**Effort:** ~150 LOC. pgmpy does the heavy lifting.

---

### Change 4: Learn Surprise Fusion Weights (Important, Low Effort)

**Currently:** The 5 surprise channels are fused with hand-coded weights: `(obs_type=0.3, temporal=0.15, value=0.25, neighborhood=0.2, memory_drift=0.1)`.

**Change:** Learn the weights from downstream signal quality (e.g., which surprise channels best predict subsequent entity state changes or portfolio return attribution).

**Math:** Treat this as an online convex optimization problem. The composite surprise is:

$$s_{\text{composite}} = \sum_{k=1}^{5} w_k \cdot z_k, \quad w \in \Delta^4 \text{ (simplex)}$$

Use exponentiated gradient descent (EG) on the simplex:

$$w_k^{(t+1)} = \frac{w_k^{(t)} \exp(\eta \cdot \nabla_k \ell_t)}{\sum_j w_j^{(t)} \exp(\eta \cdot \nabla_j \ell_t)}$$

where $\ell_t$ is a loss measuring how well the composite surprise predicted actual anomalous outcomes (e.g., a subsequent large price move, a true entity alert confirmed by news).

**Implementation:**
- Add `AdaptiveSurpriseWeights` class in `agent/fusion/surprise.py`
- Initialize with current hand-coded weights (warm start)
- After each entity scoring run, compute loss from retrospective signal quality
- Update weights via EG on simplex
- Persist current weights to PipelineStore
- Log weight trajectory for monitoring

**Verification:** Weights should differ from uniform after 100+ updates. Composite surprise should have higher rank correlation with downstream outcomes than fixed weights.

**Effort:** ~80 LOC. One day.

---

### Change 5: Learn Reward Function Weights (Important, Medium Effort)

**Currently:** `reward.py` uses fixed weights: `eval_weight=0.4, sharpe_weight=0.3, facts_weight=0.2, novelty_bonus=0.1, dead_end_penalty=0.3`. These were set by human judgment and never adapt.

**Change:** Use meta-learning to tune reward weights based on downstream portfolio performance.

**Math:** This is a bi-level optimization (meta-RL):

- **Inner loop:** Bandit learns arm preferences from reward signal $r(w)$
- **Outer loop:** Adjust reward weights $w$ to maximize a ground-truth objective (e.g., rolling 90-day Sharpe ratio of the portfolio)

Population-Based Training (PBT, Jaderberg et al. 2017) is the practical approach: maintain a small population of reward weight vectors, run each for K iterations, copy-from-best + perturb, repeat.

For a simpler start: Bayesian optimization over the 5-dimensional reward weight space, using 90-day backtested Sharpe as the objective. Scikit-optimize `gp_minimize` with ~20 evaluations per quarter.

**Implementation:**
- Add `RewardWeightOptimizer` in `agent/learning/reward.py`
- Quarterly: sample 20 weight vectors from GP posterior → run shortened autonomous sessions → measure portfolio Sharpe → update GP
- Replace `DEFAULT_WEIGHTS` with `load_current_weights()` from PipelineStore
- Log all trials for analysis

**Verification:** Learned weights should yield statistically higher portfolio Sharpe than default weights over 4+ quarterly cycles. Use paired t-test on walk-forward periods.

**Effort:** ~200 LOC + integration with quarterly schedule.

---

### Change 6: Learn State Representation (Important, High Effort)

**Currently:** `InstrumentStateAssembler` constructs the SAC state vector via hand-designed layout: instrument surprises (N×5) + entity surprises (E×5) + beliefs (E×4) + market features (M) + entity count (1) + adversarial summary (4). The ordering, truncation rule (top-K by composite surprise), and feature grouping are all manual.

**Change:** Replace the hand-designed assembler with a learned state encoder.

**Math:** Two options:

**Option A: Attention-based state encoder** — A small Transformer that attends over variable-length entity sets and produces a fixed-dim state for SAC. Each entity's features (surprise + belief + alert) are tokens; the Transformer outputs a [CLS]-like summary token.

$$h_i = \text{EntityEmbed}(s_i, b_i, a_i), \quad z = \text{Transformer}([h_1, ..., h_K])_{\text{CLS}}$$

$$\text{state} = [z; \text{market\_features}; \text{adversarial}]$$

This is strictly better than top-K truncation because attention can learn which entities matter at each timestep.

**Option B: Set Transformer** (Lee et al. ICML 2019) — Designed specifically for set-to-vector encoding with $O(n)$ complexity via induced set attention blocks (ISAB). Perfect for variable-length entity sets.

**Implementation:**
- Replace `InstrumentStateAssembler` with `LearnedStateEncoder(nn.Module)`
- Train end-to-end with SAC (state encoder gradients flow through actor/critic loss)
- Keep hand-designed assembler as a fallback for cold start (before encoder has trained)

**Verification:** SAC + learned encoder should achieve ≥ equal Sharpe to SAC + hand-designed assembler on walk-forward backtest, with fewer human design decisions.

**Effort:** ~400 LOC. Needs careful integration with SAC training loop.

---

### Change 7: Learn Detector Thresholds Online (Moderate, Low Effort)

**Currently:** CUSUM (k=0.5, h=5.0), Hawkes (μ=0.1, α=0.5, β=1.0), convergence (z=2.0, p=0.05, fdr_q=0.05) — all hand-tuned.

**Change:** Bayesian optimization of detector parameters, evaluated by downstream signal quality.

**Math:** For each detector, define a small parameter space (2-5 dims) and an objective (e.g., F1 of entity alerts vs confirmed anomalous events). Run GP-BO:

$$\theta^* = \arg\max_\theta f(\theta), \quad f \sim \mathcal{GP}(\mu, k)$$

Use Expected Improvement acquisition:

$$\text{EI}(\theta) = E[\max(f(\theta) - f^+, 0)]$$

**Implementation:**
- Add `ThresholdOptimizer` that wraps scikit-optimize + stores param history
- Monthly: evaluate 10-20 parameter settings against retroactive ground truth
- Update running params in ScorerConfig / DetectorConfig

**Verification:** Optimized thresholds should yield tighter precision-recall tradeoff than hand-tuned defaults on holdout periods.

**Effort:** ~120 LOC. Reusable across CUSUM, Hawkes, convergence.

---

### Change 8: Dynamic Goal Arm Discovery (Moderate, Medium Effort)

**Currently:** The 45 GoalArms in `bandit.py` are a fixed list. The bandit learns *which* to pull, but cannot discover new goal categories. If a new tool is added and no arm covers it, the bandit can't explore it.

**Change:** Hierarchical bandit with a meta-arm for "novel exploration" that the LLM can fill with new tool combinations, and a promotion mechanism that turns successful novel goals into permanent arms.

**Math:** Hierarchical Thompson Sampling:
- Top level: $K$ known arms + 1 "novel" meta-arm, each with Beta(α, β)
- When "novel" is selected: LLM generates an unconstrained goal using any tool
- If novel goal succeeds with reward > threshold: create a new permanent arm from its description
- New arm starts with Beta(1+r, 1+(1-r)) where r is the reward (informative prior from first pull)

**Implementation:**
- Add `NovelExplorationArm` to bandit
- After N=3 successful novel pulls with similar tool signatures, auto-create a new GoalArm
- Persist the expanded arm set to bandit state file
- Existing arms are never removed (only their α/β change)

**Verification:** After 100+ autonomous iterations, the arm set should have grown by 5-10 new categories that humans didn't pre-define. Novel arms should have meaningful pull rates.

**Effort:** ~150 LOC.

---

### Change 9: GNN Loss Weight Auto-Tuning (Moderate, Low Effort)

**Currently:** `trainer.py` uses fixed loss weights: obs_type CE=1.0, time_delta MSE=0.1, contrastive=0.5, value Huber=0.3.

**Change:** Dynamic weight adjustment via uncertainty weighting (Kendall, Gal & Cipolla 2018, "Multi-Task Learning Using Uncertainty to Weigh Losses").

**Math:** Learn per-task log-variance $\log \sigma_k^2$ and minimize:

$$\mathcal{L} = \sum_k \frac{1}{2\sigma_k^2} \mathcal{L}_k + \log \sigma_k$$

This automatically up-weights well-calibrated tasks and down-weights noisy ones.

**Implementation:**
- Add 4 learnable `nn.Parameter` for log-variances
- Modify trainer loss computation to use uncertainty weighting
- $\sigma_k$ values are checkpointed alongside model weights

**Verification:** Learned weights should adapt: early training may emphasize obs_type (easier), later training may shift toward value prediction (harder but more useful). Total loss should decrease faster than fixed weights.

**Effort:** ~40 LOC.

---

### Change 10: End-to-End Gradient Flow (Aspirational, High Effort)

**Currently:** The pipeline is: GNN (gradients) → beliefs (no gradients, goes through pgmpy) → state assembler (no gradients, numpy) → SAC (gradients). The chain is broken at two points: pgmpy inference and numpy state assembly.

**Change:** Replace pgmpy Bayesian network with a differentiable probabilistic model so gradients can flow from portfolio loss all the way back through the world model to the GNN.

**Options:**

**Option A: Neural Bayesian Network** — Replace tabular CPDs with small neural networks: $P(X_i | \text{Pa}(X_i)) = \text{softmax}(\text{MLP}(\text{Pa}(X_i)))$. Structure stays the same (20 nodes, 19 edges), but CPDs become differentiable. Train end-to-end via ELBO or policy gradient.

**Option B: Variational world model** — Replace the DAG entirely with a latent variable model: $q(z_t | x_{1:t})$ encoder + $p(x_{t+1} | z_t)$ decoder. This is the "Dreamer V3" (Hafner et al. 2023) approach applied to financial time series.

**Option C: Differentiable Kalman** — Replace numpy Kalman with a PyTorch implementation (e.g., torchfilter). The F, Q, H, R matrices become `nn.Parameter` and gradients flow through predict/update.

**Recommendation:** Start with Option C (differentiable Kalman, ~200 LOC) as it's the smallest scope change. Option B is the long-term target but requires rearchitecting the world model entirely.

**Effort:** Option C: ~200 LOC. Option A: ~400 LOC. Option B: ~1000+ LOC (multi-week).

---

## Implementation Roadmap

### Tier 1: Wire What Exists (1-2 days)
- [ ] **Change 1** — Close belief→policy loop in inference DAG

### Tier 2: Learn Parameters, Keep Structure (1-2 weeks)
- [ ] **Change 2a** — CPD learning via MLE/Bayesian estimation
- [ ] **Change 2b** — Kalman EM parameter fitting
- [ ] **Change 4** — Adaptive surprise fusion weights (EG on simplex)
- [ ] **Change 9** — GNN multi-task loss auto-tuning

### Tier 3: Learn Meta-Parameters (2-4 weeks)
- [ ] **Change 5** — Reward weight optimization (Bayesian optimization)
- [ ] **Change 7** — Detector threshold optimization (GP-BO)
- [ ] **Change 8** — Dynamic goal arm discovery (hierarchical bandit)

### Tier 4: Learn Representations (1-2 months)
- [ ] **Change 6** — Learned state encoder (Set Transformer / attention)
- [ ] **Change 3** — Causal graph structure learning

### Tier 5: End-to-End Differentiable (2-3 months)
- [ ] **Change 10** — Differentiable Kalman or variational world model

### Tier 6: Learn What to Observe (3-6 months)
- [ ] **Change 11** — Learned feature selection: attention-based or gradient-based gating over the engineered feature set; the system decides which features matter per regime, not a hand-coded feature list
- [ ] **Change 12** — Learned tool routing: bandit or policy network decides which data sources to query and how often, replacing the fixed DAG schedule

### Tier 7: Self-Modifying Structure (6-12 months)
- [ ] **Change 13** — Self-modifying graph schema: structure learning on live data adds/removes/merges nodes in the causal DAG; the graph topology becomes a learned object, not a hand-drawn diagram
- [ ] **Change 14** — Meta-learned scheduling: the system learns when to re-fit parameters, how much history to use, and which components need retraining — replacing all fixed intervals and window sizes

### Tier 8: Autonomous Discovery (12+ months)
- [ ] **Change 15** — Autonomous data source discovery: the agent finds new free APIs, scrape targets, and data feeds on its own, evaluates their signal content, and wires them into the pipeline without human intervention
- [ ] **Change 16** — Self-extending entity ontology: new entity types and relationship types emerge from data rather than being predefined in code; the taxonomy grows as the system encounters novel structure

---

## Expected Impact on Learned/Hand-Coded Ratio

| After Tier | % Learned | % Hand-Coded | Key Unlocks |
|------------|-----------|-------------|-------------|
| Current | 25% | 75% | — |
| Tier 1 | 28% | 72% | World model beliefs flow into policy decisions |
| Tier 2 | 45% | 55% | World model params learned, surprise weights adaptive, loss weights auto-tuned |
| Tier 3 | 55% | 45% | Reward function adapts, detectors self-calibrate, goal space expands |
| Tier 4 | 65% | 35% | State representation learned, causal structure discovered |
| Tier 5 | 75% | 25% | Full gradient flow from portfolio loss to world model to GNN |
| Tier 6 | 82% | 18% | System chooses what to observe and when to query |
| Tier 7 | 90% | 10% | Graph topology and scheduling are learned objects |
| Tier 8 | 95% | 5% | System autonomously discovers new data and entity types |

At Tier 8, the system is **95% learned / 5% hand-coded**. The residual 5% is the irreducible core that *should* stay hand-coded — see below.

---

## What Stays Hand-Coded (The Irreducible 5%)

Even at the aspirational 95% learned end state, these remain manual by design:

1. **Safety constraints** — Leverage limits, position limits, max drawdown kills, legal/regulatory rules. These are non-negotiable human-defined guardrails.
2. **Schema invariants** — The *existence* of EngineeredFeature, BeliefState, EntityAlert as protocols (but their contents and relationships are learned)
3. **API plumbing** — HTTP calls, auth, serialization. Deterministic contracts that don't benefit from learning.
4. **Textbook equations** — Sharpe, VaR, Kalman predict/update steps. The equations are mathematical identities; only parameters are learned.
5. **Ethical/legal boundaries** — What the system is *not allowed* to do, regardless of what it learns is profitable.

Everything else — feature selection, graph topology, scheduling, tool routing, entity ontology, reward shaping, state representation — trends toward learned over time.

---

## Verification Strategy

Each change has a specific verification criterion (stated above). Additionally, the overall system health metric is:

**Walk-forward Sharpe ratio** on the multi-asset portfolio over 6-month rolling windows. Each tier should maintain or improve Sharpe relative to the previous tier. If a change degrades Sharpe, it's rolled back and investigated.

Secondary metrics:
- Belief calibration (Brier score) for world model changes
- Surprise precision-recall for fusion changes
- Arm diversity (entropy of pull distribution) for bandit changes
- Innovation whiteness (Ljung-Box p-value) for Kalman changes

---

## Related

- [[learned_vs_handcoded_audit]] — Research: full subsystem audit
- [[e2e_global_integration]] — Phase 24 research (current phase)
- [[rl_policy]] — Phase 21 research (SAC implementation)
- [[world_model_bridge]] — Phase 19 research (GNN→world model bridge)
- [[signal_fusion]] — Phase 20 research (surprise extraction)
- [[project_memory]] — Persistent architectural memory
