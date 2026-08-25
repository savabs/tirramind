---
title: "Feature: World Model (Phase 9)"
tags:
  - doc/research
  - layer/fusion
  - layer/learning
  - layer/world-model
  - phase/9
  - topic/world-model
---

# Feature: World Model (Phase 9)

## Goal

Build TirraMind's Layer 3 world model: a Bayesian network / causal graph that maintains a probabilistic belief state over hidden economic variables, updates those beliefs as new EngineeredFeature observations arrive, supports causal interventions (do-calculus) for counterfactual reasoning, and outputs posterior distributions — never point estimates — for downstream signal fusion (Phase 10) and RL policy (Phase 11).

The world model must:
1. Represent causal relationships between economic variables as a DAG
2. Propagate evidence (observed features) to update beliefs over latent states
3. Support temporal dynamics (DBN or state-space model)
4. Learn and validate structure from data (causal discovery)
5. Produce distributions with proper uncertainty quantification
6. Operate on numpy arrays as the native data format (per architecture decision)
7. Run deterministically on the Pipeline layer (no LLM involvement)

---

## Search Log

### GitHub keywords searched
- `bayesian network causal graph time series python`
- `causal discovery time series PCMCI tigramite`
- `sequential monte carlo particle filter python state space`
- `probabilistic programming JAX numpyro bayesian inference`
- `pgmpy bayesian network structure learning`
- `dowhy causal inference python`
- `causalnex bayesian network NOTEARS`
- `filterpy kalman filter unscented`

### Documentation keywords searched
- pgmpy DiscreteBayesianNetwork API, DynamicBayesianNetwork, LinearGaussianBN
- pgmpy structure learning PC HillClimb GES
- PyMC probabilistic programming distributions (AR, GARCH, GaussianRandomWalk)
- filterpy KalmanFilter UnscentedKalmanFilter ExtendedKalmanFilter
- Tigramite PCMCI PCMCIplus LPCMCI RPCMCI conditional independence tests
- DoWhy causal effects estimation intervention counterfactual GCM
- NumPyro MCMC NUTS SVI distributions inference algorithms
- particles SMC particle filtering state-space models

### Other search surfaces used
- readthedocs (filterpy, particles, causalnex, numpyro)
- PyPI for license verification

---

## External Repositories Reviewed

### pgmpy — Probabilistic Graphical Models
- **Why relevant**: Core DAG representation, belief propagation, CPDs, structure learning, do-calculus. Most mature Python BN library.
- **Key features**: DiscreteBayesianNetwork, DynamicBayesianNetwork (2-TBN), LinearGaussianBN, TabularCPD, VariableElimination, BeliefPropagation, do() operator, simulate(), structure learning (PC/HillClimb/GES/ExpertInLoop/ExhaustiveSearch), model validation (AIC/BIC/Fisher-C/RMSEA)
- **License**: MIT
- **Reuse conclusion**: **Reusable as dependency** — use directly for DAG structure, CPD management, belief propagation, and structure learning. MIT is fully compatible.

### Tigramite (jakobrunge/tigramite) — Causal Discovery for Time Series
- **Why relevant**: PCMCI family is the gold standard for time-series causal discovery. Handles autocorrelation, contemporaneous + lagged links, latent confounders, regime-dependent causation.
- **Key features**: PCMCI, PCMCIplus (contemporaneous + lagged, full CPDAG), LPCMCI (latent confounders, time-series PAG), RPCMCI (regime-dependent causal graphs), JPCMCIplus (multiple datasets/contexts). CI tests: ParCorr, RobustParCorr, GPDC (GP + distance correlation), CMIknn (k-NN conditional mutual information), CMIsymb (discrete), RegressionCI (mixed), ParCorrWLS (heteroskedastic). CausalEffects class for effect estimation.
- **License**: **GPL-3.0** ⚠️
- **Reuse conclusion**: **Use as dependency only** — GPL-3.0 means we can import and call tigramite, but cannot port/copy its code into our codebase. All TirraMind code remains our own. Tigramite is used as an external tool for causal structure discovery, not embedded.
- **Academic references**:
  - Runge et al. (2019) "Detecting and quantifying causal associations in large nonlinear time series datasets" — Sci. Adv. 5, eaau4996
  - Runge (2020) "Discovering contemporaneous and lagged causal relations" — UAI 2020
  - Gerhardus & Runge (2020) "High-recall causal discovery for autocorrelated time series with latent confounders" — NeurIPS 2020
  - Saggioro et al. (2020) "Reconstructing regime-dependent causal relationships" — Chaos 30(11)
  - Runge (2018) "Causal Network Reconstruction from Time Series" — Chaos 28(7):075310
  - Runge (2021) "Necessary and sufficient graphical conditions for optimal adjustment sets" — NeurIPS 2021

### DoWhy (PyWhy) — Causal Inference
- **Why relevant**: Provides causal effect estimation (backdoor/frontdoor/IV), causal influence quantification, root-cause analysis, intervention/counterfactual reasoning, causal prediction (OOD). Strong refutation API for testing causal assumptions.
- **Key features**: 5 causal task categories: (1) effect estimation, (2) causal influence quantification (mediation, arrow strength, intrinsic causal influence), (3) root-cause analysis (anomaly attribution, distributional changes), (4) what-if (interventions, counterfactuals), (5) causal prediction (OOD). Uses GCM (Graphical Causal Models) and Potential Outcomes frameworks. Graph-agnostic + task-agnostic refutations.
- **License**: MIT
- **Reuse conclusion**: **Reusable as dependency** — use for causal effect estimation and intervention queries once the DAG is learned. MIT compatible.

### filterpy — Kalman and Bayesian Filtering
- **Why relevant**: State estimation for the continuous-state components of the world model. KalmanFilter for linear Gaussian, UKF for nonlinear, EKF for mildly nonlinear.
- **Key features**: KalmanFilter (predict/update cycle, batch_filter, RTS smoother), UnscentedKalmanFilter (sigma points, MerweScaledSigmaPoints), ExtendedKalmanFilter (Jacobian-based linearization). Already used in our codebase planning for Phase 10 signal fusion.
- **License**: MIT
- **Reuse conclusion**: **Reusable as dependency** — use for linear state-space filtering. Better suited for Phase 10 (signal fusion) than Phase 9 (world model structure), but the world model must define the state-space equations that filterpy consumes.

### particles (nchopin/particles) — Sequential Monte Carlo
- **Why relevant**: For nonlinear, non-Gaussian state-space models that KalmanFilter cannot handle. Particle filters approximate arbitrary posterior distributions using weighted samples.
- **Key features**: Bootstrap filter, guided filter, APF, resampling (multinomial/residual/stratified/systematic/SSP), SQMC (quasi-MC), FFBS smoothing, Kalman (linear), forward-backward (HMM), waste-free SMC samplers, PMCMC (PMMH, Particle Gibbs), SMC², nested sampling. Probabilistic programming interface for defining state-space models (PX0, PX, PY).
- **License**: MIT
- **Reuse conclusion**: **Reusable as dependency** — MIT-licensed, well-designed API. Use for nonlinear state estimation when Kalman assumptions fail (e.g., regime-switching dynamics, fat-tailed observation noise). Companion book: Chopin & Papaspiliopoulos, "An Introduction to Sequential Monte Carlo" (Springer, 2020).

### NumPyro (pyro-ppl/numpyro) — Probabilistic Programming on JAX
- **Why relevant**: JAX-based probabilistic programming with JIT-compiled NUTS/HMC. Dramatically faster than PyMC for MCMC. Supports HMM enumeration, time-series models, stochastic volatility.
- **Key features**: NUTS (JIT-compiled tree building), HMC, MixedHMC (continuous + discrete), DiscreteHMCGibbs, SVI with ELBO variants (Trace, MeanField, TraceGraph, TraceEnum for discrete), rich distributions, effect handlers, HMM example, stochastic volatility example, time series forecasting.
- **License**: Apache 2.0
- **Reuse conclusion**: **Reusable as dependency** — Apache 2.0 compatible. Alternative to PyMC. JAX backend enables GPU acceleration and functional composition. Good for HMM inference and custom probabilistic models.

### PyMC — Probabilistic Programming
- **Why relevant**: Mature probabilistic programming framework. Rich distribution library including time series (AR, GARCH11, GaussianRandomWalk, EulerMaruyama). NUTS/HMC inference.
- **Key features**: NUTS sampler, ADVI, extensive distribution library, PyTensor backend, Stan-like model specification.
- **License**: Apache 2.0
- **Reuse conclusion**: **Reusable as dependency** — valid alternative to NumPyro. PyMC is more Pythonic; NumPyro is faster (JIT). Decision depends on whether we need JAX ecosystem integration.

### CausalNex (QuantumBlack/McKinsey) — Bayesian Networks for Causal Reasoning
- **Why relevant**: NOTEARS algorithm for continuous structure learning (formulates DAG constraint as smooth optimization), BayesianNetwork + InferenceEngine classes, do-calculus, sklearn interface (DAGRegressor/DAGClassifier).
- **Key features**: NOTEARS structure learning, BN fitting, marginal queries, do-calculus interventions, latent variable detection, distribution schema for mixed data.
- **License**: Apache 2.0 (per readthedocs)
- **Reuse conclusion**: **Use with caution** — the original GitHub repo (`quantumblacklabs/causalnex`) has been compromised via account takeover (verified: page says "THIS IS A VULNERABILITY BY SHAMIM_12 HACKERONE"). Real repo was at `mckinsey/causalnex`. Use only via `pip install causalnex` from PyPI (which predates the takeover) and verify integrity. NOTEARS algorithm concept is valuable regardless.
- **⚠️ SECURITY NOTE**: Do NOT clone from `github.com/quantumblacklabs/causalnex` — the account has been taken over. Install only from PyPI or reference the NOTEARS paper directly.

---

## Documentation Reviewed

### pgmpy DiscreteBayesianNetwork API
- Full inference via VariableElimination, BeliefPropagation
- do() operator for causal interventions
- simulate() for generating synthetic data from the fitted model
- fit() with MLE estimator from pandas DataFrames
- TabularCPD for conditional probability distributions
- check_model(), get_cpds(), get_parents()

### pgmpy DynamicBayesianNetwork (DBN) API
- 2-time-slice Bayesian network (2-TBN) with constant structure across time
- Nodes represented as (name, time_slice) tuples
- Inter-slice edges (temporal) and intra-slice edges (contemporaneous)
- fit(data, estimator='MLE') iterating over consecutive time slices
- simulate() with n_time_slices, do, evidence, virtual_evidence, virtual_intervention
- Multiple return_format options: 'wide', 'numpy3d', 'pd-multiindex', 'pd-list', 'sorted'
- Model validation: correlation, log-likelihood, AIC, BIC, Fisher-C, RMSEA
- get_inter_edges(), get_intra_edges(), get_interface_nodes()
- initialize_initial_state(), check_model(), moralize()

### pgmpy LinearGaussianBN
- Continuous variable support with LinearGaussianCPD
- edge_strength() using Pillai's Trace
- Naturally suited for continuous financial features

### pgmpy Structure Learning Algorithms
| Algorithm | Type | Strengths | F1 (alarm benchmark) |
|-----------|------|-----------|---------------------|
| PC | Constraint-based (CI tests) | Theory-grounded, handles large graphs | 0.825 |
| HillClimbSearch | Score-based (greedy) | Fast, intuitive, good defaults | 0.77 |
| GES | Greedy equivalence search | Consistent under faithfulness, explores CPDAG space | 0.84 |
| ExpertInLoop | LLM + CI hybrid | Can incorporate domain knowledge | - |
| ExhaustiveSearch | Exact (small N only) | Optimal for tiny graphs | - |

CI tests available: chi_square, g_sq, log_likelihood (discrete), pearsonr (continuous), pillai (mixed)
Scoring methods: bic-d/aic-d/k2/bdeu/bds (discrete), ll-g/aic-g/bic-g (continuous), aic-cg/bic-cg (mixed)

### Tigramite PCMCI Family
| Method | Handles Contemp. | Handles Latents | Output | Assumptions |
|--------|-------------------|-----------------|--------|-------------|
| PCMCI | No (lagged only) | No | Directed lagged links | Stationarity, no contemp., no hidden |
| PCMCIplus | Yes | No | Directed + undirected (CPDAG) | Stationarity, no hidden |
| LPCMCI | Yes | Yes | Time series PAG | Stationarity |
| RPCMCI | No | No | Regime-dependent causal graphs | No contemp., no hidden, regimes |
| JPCMCIplus | Yes | Latent contexts | Joint CPDAG across datasets | Multiple contexts |

CI tests: ParCorr (linear), RobustParCorr (robust linear), GPDC/GPDCtorch (GP + distance corr.), CMIknn (nonparametric, continuous), CMIsymb (discrete), Gsquared (categorical), RegressionCI (mixed), ParCorrWLS (heteroskedastic)

### DoWhy 5 Causal Task Categories
1. **Estimating Causal Effects**: backdoor/frontdoor/IV adjustment
2. **Quantify Causal Influence**: mediation, arrow strength, intrinsic causal influence
3. **Root-Cause Analysis**: anomaly attribution, distributional change attribution
4. **What-If**: interventional distributions, counterfactual reasoning
5. **Causal Prediction**: OOD prediction using causal structure

### particles library SSM interface
```python
class MySSM(ssm.StateSpaceModel):
    def PX0(self):        # Initial state distribution
    def PX(self, t, xp):  # Transition: P(X_t | X_{t-1})
    def PY(self, t, xp, x): # Observation: P(Y_t | X_t)
```
- Bootstrap filter: `particles.SMC(fk=ssm.Bootstrap(ssm=model, data=y), N=200)`
- Supports PMCMC for joint state/parameter estimation

---

## Current Architecture

### Relevant local modules
- `agent/features/protocol.py` — EngineeredFeature frozen dataclass (input contract for world model)
- `agent/features/builders.py` — FeatureBuilder ABC, ConvergenceFeatureBuilder, MacroStateFeatureBuilder
- `agent/pipeline/dags/feature_generation.py` — DAG scheduler for feature computation
- `agent/pipeline/store.py` — PipelineStore (SQLite WAL) for features table
- `agent/quant/` — Existing quant modules (BOCPD, HMM, FFT+CWT, scoring, backtest)
- `agent/models/` — **Target directory** for world model (Layer 3)

### Existing patterns to preserve
- EngineeredFeature as input: feature_name (dotted: `{domain}.{metric}.{horizon}`), value (float|None), quality (0-1), effective_at, computed_at, horizon
- PipelineStore for reading features and writing model outputs
- DAG scheduler for deterministic execution
- numpy arrays as native compute format (not JSONL)
- No LLM in the critical path

### Correct insertion points
- `agent/models/` — New module for Layer 3 world model
- `agent/pipeline/dags/` — New DAG for world model update cycle
- Integration: features table → world model → beliefs table → signal fusion (Phase 10)

---

## Observations

### What already exists
- 57 tools across Layer 1 (surveillance surface)
- Phase 8 feature engineering: ConvergenceFeatureBuilder (stress_breadth, stress_intensity, regime_persistence), MacroStateFeatureBuilder (rate_momentum, yield_curve_slope, liquidity_pressure)
- Phase 7c convergence detection with BOCPD + HMM + spectral analysis
- PipelineStore with SQLite WAL for all data persistence
- Clear EngineeredFeature protocol as input contract

### What is missing
- No causal graph structure (DAG) representing variable relationships
- No belief propagation engine
- No temporal model for state evolution
- No causal discovery pipeline to learn/validate structure from data
- No intervention/counterfactual reasoning capability
- No probabilistic output format for downstream consumers

### Important constraints
- **Cost discipline**: All libraries must be open source (pgmpy, tigramite, filterpy, particles — all are). No paid APIs.
- **Deterministic pipeline**: World model runs on Pipeline layer, no LLM calls during inference.
- **Data volume**: Currently 6 features from 2 builders. Structure learning needs more features (minimum ~10-20 nodes for meaningful discovery). Must plan for feature expansion.
- **Stationarity assumption**: PCMCI assumes stationary causal structure. Financial regimes violate this. RPCMCI handles regime-dependent causation but requires more data.
- **Sample size**: Tigramite recommends >1000 samples for reliable discovery. With daily features, this requires ~4+ years. Hourly features compress this.

---

## Risks

### Licensing risks
- **Tigramite GPL-3.0**: Must remain as external dependency only. No code copying. All TirraMind code stays MIT/proprietary.
- **CausalNex repo compromised**: GitHub account `quantumblacklabs` taken over. Only install from PyPI or reimplement NOTEARS independently.

### Technical risks
- **Sparse data**: With only 6 features currently, structure learning will overfit. Must defer automated discovery until feature count grows (Phase 9 should prepare the infrastructure, use expert-specified initial structure).
- **Non-stationarity**: Financial causal graphs change with regimes. Static DAGs are wrong. Must implement regime-aware or time-varying graph structure.
- **Computational cost**: Exact inference in large BNs is NP-hard. Approximate methods (belief propagation, variational inference) may accumulate errors.
- **Identifiability**: From observational data alone, causal direction is often unidentifiable (Markov equivalence class). Time ordering resolves some ambiguity, but contemporaneous links remain undirected.
- **Missing data**: Financial features have gaps (market closures, data delays). The model must handle missing evidence gracefully.

### Testing risks
- Validating causal structure is fundamentally hard — no ground truth in finance.
- Must rely on: (a) synthetic data with known structure, (b) held-out prediction performance, (c) intervention consistency checks, (d) expert review of learned edges.

---

## Data Requirements

### Required inputs
- EngineeredFeature instances from PipelineStore features table
- Feature metadata: feature_name, value, quality, effective_at, horizon
- Historical time series of features for structure learning
- Domain knowledge for initial graph structure (expert-specified edges)

### What already exists locally
- PipelineStore with features table (Phase 8)
- 6 features from 2 builders (convergence + macro)
- BOCPD changepoint probabilities (from Phase 7c)
- HMM regime labels (from Phase 7c)

### What still needs to be added
- More feature builders (expand from 6 to 20+ features for meaningful structure learning)
- Beliefs table in PipelineStore for world model outputs
- Graph persistence (store learned/expert DAG structure)
- Model checkpoint mechanism for graph + CPDs

---

## Math/Algorithm Survey

### Architecture: Hybrid Bayesian Network + State-Space Model

The world model combines two complementary mathematical frameworks:

**1. Causal DAG (pgmpy BayesianNetwork / DynamicBayesianNetwork)**
- Represents qualitative causal structure: "which variables cause which"
- Discrete or mixed CPDs for categorical states (regime labels, stress categories)
- Belief propagation for evidence → posterior inference
- do() calculus for causal interventions
- Structure learning from data (when sample size permits)

**2. Continuous State-Space Model (filterpy / particles)**
- Represents quantitative dynamics: "how do continuous hidden states evolve"
- Kalman filter for linear-Gaussian dynamics
- Particle filter for nonlinear / non-Gaussian dynamics (regime switching, fat tails)
- State vector: latent economic variables (true stress level, true growth momentum, etc.)
- Observation model: maps hidden state to observed EngineeredFeatures

### Why this hybrid?

Pure BN: Discrete CPDs lose the continuous dynamics of financial variables. Discretization introduces information loss and arbitrary bin boundaries.

Pure state-space: Linear/Gaussian assumptions fail for regime switches, causal interventions, and structural changes. No built-in causal semantics.

Hybrid: The DAG encodes causal structure and handles discrete regime variables. The state-space model tracks continuous latent dynamics conditioned on the active regime. This mirrors how the economy actually works: discrete regimes (expansion/recession/crisis) modulate continuous processes (yield curves, credit spreads, volatility).

### Candidate implementation approaches

#### Option A: pgmpy DBN + regime conditioning
- Use pgmpy DynamicBayesianNetwork for the full model
- Discretize continuous features into bins
- Pro: Single unified framework, built-in structure learning, do-calculus
- Con: Discretization loses information, doesn't scale to many continuous features, inference in large DBNs is slow

#### Option B: pgmpy DAG (discrete) + filterpy state-space (continuous)
- DAG handles regime labels, categorical states, causal structure
- Kalman/UKF handles continuous state evolution conditioned on regime
- Pro: Best of both worlds, preserves continuous dynamics, regime-aware
- Con: Two frameworks to maintain, handoff between discrete and continuous requires careful design

#### Option C: Full probabilistic program (NumPyro/PyMC)
- Single custom model specifying all relationships programmatically
- NUTS/HMC for posterior inference
- Pro: Maximum flexibility, proper uncertainty, HMM built-in
- Con: No built-in causal semantics (do-calculus), structure must be hand-coded, MCMC is slow for real-time updates

#### Option D: Tigramite discovery → pgmpy DAG → filterpy filtering
- Tigramite PCMCI discovers causal structure from historical data offline
- pgmpy implements the discovered DAG for online inference
- filterpy handles continuous state estimation
- Pro: Data-driven structure, proper statistical testing, separates discovery from inference
- Con: Tigramite GPL-3.0 (dependency only), requires sufficient data

### Recommended approach: Option B + D (phased)

**Phase 9a (immediate):** Expert-specified DAG in pgmpy + KalmanFilter for continuous states (Option B). This gives us a working world model immediately without requiring sufficient data for discovery.

**Phase 9b (after more features):** Add Tigramite PCMCI for automated structure discovery (Option D). Discovered structure augments/validates expert specification.

**Phase 9c (maturity):** Optional NumPyro for custom probabilistic models when pgmpy's discrete CPDs become limiting.

### Mathematical formulation

#### DAG component
Let $G = (V, E)$ be a DAG where $V$ are economic variables (both observed and latent) and $E$ are causal relationships.

For each variable $X_i \in V$ with parents $\text{pa}(X_i) \subseteq V$:
$$P(X_i | \text{pa}(X_i)) \text{ is specified by a CPD (conditional probability distribution)}$$

Joint distribution factorizes:
$$P(X_1, \ldots, X_n) = \prod_{i=1}^{n} P(X_i | \text{pa}(X_i))$$

Evidence injection: Given observed features $\mathbf{e}$, compute posterior via belief propagation:
$$P(X_i | \mathbf{e}) = \sum_{\mathbf{X} \setminus X_i} P(\mathbf{X} | \mathbf{e})$$

Intervention: do($X_i = x$) replaces $P(X_i | \text{pa}(X_i))$ with $\delta(X_i = x)$:
$$P(\mathbf{X} | \text{do}(X_i = x)) = \prod_{j \neq i} P(X_j | \text{pa}(X_j)) \cdot \delta(X_i = x)$$

#### State-space component
Hidden state vector $\mathbf{x}_t \in \mathbb{R}^d$ (latent economic variables):

**Transition model** (conditioned on regime $r_t$):
$$\mathbf{x}_t = f_{r_t}(\mathbf{x}_{t-1}) + \mathbf{w}_t, \quad \mathbf{w}_t \sim \mathcal{N}(0, \mathbf{Q}_{r_t})$$

**Observation model**:
$$\mathbf{y}_t = h(\mathbf{x}_t) + \mathbf{v}_t, \quad \mathbf{v}_t \sim \mathcal{N}(0, \mathbf{R})$$

where $\mathbf{y}_t$ are the observed EngineeredFeature values and $h(\cdot)$ maps hidden state to feature space.

For linear dynamics: standard Kalman filter.
For nonlinear / regime-switching: UKF or particle filter.

#### Causal discovery (offline, Tigramite)
PCMCI two-step procedure:
1. **Condition selection**: For each variable $X^j_t$, estimate superset of parents $\tilde{\mathcal{P}}(X^j_t)$ via PC algorithm
2. **MCI test**: $X^i_{t-\tau} \perp X^j_{t} | \tilde{\mathcal{P}}(X^j_t), \tilde{\mathcal{P}}(X^i_{t-\tau})$

Output: graph array of shape [N, N, tau_max+1] with directed/undirected edge markings and p-values.

### Numerical stability concerns
- Kalman filter: Use Joseph form for covariance update ($P = (I-KH)P(I-KH)^T + KRK^T$) to maintain positive-definiteness
- Belief propagation: Normalize messages to prevent underflow; use log-space for very small probabilities
- Structure learning: BIC/AIC scores can be numerically unstable with small sample sizes; regularize
- Particle filter resampling: Effective sample size (ESS) monitoring to trigger resampling; systematic resampling preferred for low variance

### Why this is the right level of complexity

Per copilot-instructions: "Prefer the smallest high-signal toolset that preserves edge."

We use 3 libraries for 3 distinct mathematical jobs:
1. **pgmpy**: Causal DAG structure + discrete inference + do-calculus
2. **filterpy/particles**: Continuous state estimation
3. **tigramite**: Offline causal discovery (data-driven structure learning)

Adding more (CausalNex, NumPyro, PyMC) is not justified yet. Each adds complexity without proportional signal gain at this stage. NumPyro becomes relevant if/when we need custom probabilistic models beyond what pgmpy can express.

---

## Trusted Sources for Mathematical Choices

| Method | Trusted Source | Why Trustworthy |
|--------|---------------|-----------------|
| Bayesian networks, belief propagation | Koller & Friedman, "Probabilistic Graphical Models" (MIT Press, 2009) | Standard reference. 1300 pages, covers exact & approximate inference, structure learning, temporal models |
| do-calculus, causal interventions | Pearl, "Causality" 2nd ed. (Cambridge, 2009) | Founding work. Defines do-calculus, backdoor/frontdoor criteria |
| PCMCI for time-series causal discovery | Runge et al. (2019), Sci. Adv. 5, eaau4996 | Peer-reviewed in top journal, handles autocorrelation and high-dimensional time series |
| PCMCIplus (contemporaneous links) | Runge (2020), UAI 2020 | Extends PCMCI to contemporaneous causal discovery |
| Kalman filtering | Sarkka, "Bayesian Filtering and Smoothing" (Cambridge, 2013) | Modern treatment covering KF, EKF, UKF, particle filters |
| Particle filtering / SMC | Chopin & Papaspiliopoulos, "An Introduction to Sequential Monte Carlo" (Springer, 2020) | Companion to the `particles` library we'll use |
| Dynamic Bayesian Networks | Murphy, "Machine Learning: A Probabilistic Perspective" Ch. 17 (MIT Press, 2012) | Standard ML textbook, DBN chapter |
| NOTEARS structure learning | Zheng et al. (2018), "DAGs with NO TEARS" NeurIPS 2018 | Reformulates DAG learning as continuous optimization |
| Regime-switching models | Hamilton (1989), "A New Approach to the Economic Analysis of Nonstationary Time Series" Econometrica 57(2) | Founding paper for Markov-switching models in economics |

---

## Implementation Intent

### Concepts approved for implementation
1. **WorldModelGraph** — pgmpy-backed DAG with expert-specified initial structure for economic variables
2. **WorldModelState** — numpy-array belief state vector, updateable via evidence
3. **BeliefPropagator** — wraps pgmpy VariableElimination / BeliefPropagation for posterior queries
4. **StateSpaceFilter** — wraps filterpy KalmanFilter for continuous state tracking, regime-conditioned
5. **CausalStructureDiscovery** — wraps tigramite PCMCI for offline structure learning (gated on sufficient data)
6. **InterventionEngine** — wraps pgmpy do() for counterfactual queries
7. **WorldModelDAG** — pipeline DAG node that runs the update cycle: read features → update beliefs → write posteriors
8. **Beliefs table** — new PipelineStore table for world model outputs (distribution parameters, not point estimates)

### Concepts deferred
- **NumPyro/PyMC custom models**: Deferred to Phase 9c or later. pgmpy sufficient for initial architecture.
- **CausalNex**: Deferred due to GitHub security concern and lower marginal value over pgmpy + tigramite.
- **Full DBN**: Deferred; start with static DAG + Kalman temporal dynamics. DBN adds complexity without proven value at 6 features.
- **Particle filter**: Deferred to Phase 10. Kalman filter sufficient for initial continuous state tracking. particles library available when nonlinear dynamics are needed.

### Concepts rejected
- Discretizing all continuous features into categorical bins for pure BN inference (information loss is unacceptable for financial signals)
- Using LLM for any part of the belief update cycle (violates architecture: math decides, LLM explains)
- Fitting a single massive probabilistic program (PyMC/NumPyro) for the entire world model (too monolithic, hard to test atomically, MCMC too slow for online updates)

### Notes for the spec
- Start with expert-specified graph (6 feature nodes + 3-4 latent nodes for regimes/risk/growth)
- EngineeredFeature.quality field should weight observations in belief updates
- Missing features (EngineeredFeature.value is None) must be handled as missing evidence, not zero
- World model outputs must be distributions: (mean, variance) minimum, full posterior when feasible
- The update cycle frequency should match feature_generation DAG (weekdays 19:00 UTC initially)
- Structure validation: simulate from fitted model, compare statistics with observed data
- Testing: synthetic DAGs with known structure → verify belief propagation, interventions, temporal updates

---

## Related

- [[world_model_spec|Spec: World Model]]
- [[convergence_detection]]
- [[signal_protocol_feature_engineering]]
- [[rl_layer]]
