---
title: "Feature: RL Policy Layer"
tags:
  - doc/research
  - phase/21
  - topic/rl-policy
  - topic/portfolio-optimization
  - topic/surprise-weighting
  - layer/learning
---

# Feature: RL Policy Layer (Phase 21)

## Goal

Build a reinforcement learning policy that closes the loop from **entity surprise signals → learned actions → portfolio outcomes → reward → updated policy**. Currently, the system produces EntityAlerts with 5 surprise signals weighted by hand-coded constants `[0.30, 0.15, 0.25, 0.20, 0.10]` and ConvergenceClusters, but nothing downstream consumes them to make decisions or learn from outcomes. Phase 21 turns those dead-end signals into a live learning system.

Three concrete sub-goals:
1. **Learn composite surprise weights** from market outcomes (replace hand-coded weights)
2. **Map surprise signals + beliefs → position sizing** (the "what to bet" question)
3. **Evaluate via walk-forward backtest** (the "does it actually predict anything" question)

The overarching principle: **the RL policy is the first component that converts mathematical signals into economic value**. Everything upstream (tools, GNN, surprise, world model) is perception. This is action.

## Search Log

- arXiv keywords searched: "model-based reinforcement learning world model", "DreamerV3", "decision transformer offline RL", "soft actor-critic maximum entropy", "MBPO model-based policy optimization", "TD-MPC2 world models continuous control", "risk-sensitive RL CVaR distributional", "DSAC distributional soft actor-critic", "FinRL deep RL portfolio", "DDPG portfolio management", "curiosity-driven exploration self-supervised prediction", "curriculum learning RL", "intrinsic motivation model-based RL", "Kelly criterion portfolio sizing"
- GitHub keywords searched: "model-based RL finance", "portfolio optimization reinforcement learning", "entity-based RL graph", "surprise-based RL"
- Wikipedia: "Kelly criterion" (comprehensive coverage of multi-asset Kelly formula)

## External Repositories Reviewed

- **FinRL (AI4Finance-LLC/FinRL-Library)**
  - Why relevant: DRL library for automated stock trading; supports DQN, DDPG, PPO, SAC, A2C, TD3
  - Useful idea: Layered architecture (market env → agent → reward), backtesting integration
  - License: MIT
  - Reuse conclusion: **concept only** — FinRL is a generic framework; TirraMind's entity-graph architecture is fundamentally different. Port the *idea* of modular env/agent/backtest separation, not the code.

- **stable-baselines3 (DLR-RM/stable-baselines3)**
  - Why relevant: Production-quality SAC, PPO, TD3 implementations
  - License: MIT
  - Reuse conclusion: **rejected as dependency** — SB3 assumes Gym environments with fixed obs/action spaces. TirraMind's state space is a heterogeneous graph + belief vector, which doesn't fit Gym's Box/Discrete paradigm. However, the SAC algorithm design is well-documented and we should implement a minimal SAC ourselves, using SB3's architecture as conceptual reference.

- **tianshou (thu-ml/tianshou)**
  - Why relevant: Modular RL library supporting model-based methods
  - License: MIT
  - Reuse conclusion: **concept only** — same Gym-dependency issue as SB3

## Documentation Reviewed

### Core RL Papers (Foundational)

1. **DreamerV3** — Hafner et al. (2023). "Mastering Diverse Domains through World Models." arXiv:2301.04104. ICLR 2024.
   - Key insight: Learn world model in latent space, train actor-critic purely in imagination. Single config works across 150+ tasks.
   - **Why it matters for TirraMind**: We already have a world model (HetTGN + Bayesian DAG + Kalman). The Dreamer paradigm — "train policy by imagining trajectories through the world model" — maps directly. Our GNN memory states ARE the latent space.
   - Specific technique: **Symlog predictions** for reward/value normalization across scales. Critical for finance where regimes shift magnitude.
   - Specific technique: **Return normalization** via percentile-based scaling (not mean/std). Avoids catastrophic value function collapse in non-stationary environments.

2. **DreamerV2** — Hafner et al. (2021). "Mastering Atari with Discrete World Models." arXiv:2010.02193. ICLR 2021.
   - Key insight: Discrete latent representations + KL balancing for stable world model learning.
   - Relevance: KL balancing (split α=0.8 on representation, 0.2 on dynamics) prevents posterior collapse — useful since our GNN embeddings are high-dimensional and could collapse.

3. **TD-MPC2** — Hansen et al. (2024). "Scalable, Robust World Models for Continuous Control." arXiv:2310.16828. ICLR 2024.
   - Key insight: Implicit (decoder-free) world model + temporal difference learning in latent space + model-predictive control (MPC) for action selection.
   - **Why it matters**: TD-MPC2 scales to 317M params and 80 tasks with single config. The "decoder-free" approach is relevant because we don't need to reconstruct observations — we just need to predict rewards from entity states.
   - Specific technique: **Local trajectory optimization in latent space** (MPPI/CEM). Instead of learning a global policy network, plan forward in the world model.

4. **Soft Actor-Critic (SAC)** — Haarnoja et al. (2018). "Off-Policy Maximum Entropy Deep RL with a Stochastic Actor." arXiv:1801.01290. ICML 2018.
   - Key insight: Maximum entropy RL — maximize reward + entropy → better exploration, more robust to perturbations, multi-modal optimal policies.
   - **Why it matters**: Financial markets are non-stationary. MaxEnt keeps the policy exploring, prevents premature convergence to regime-specific optima.
   - Specific technique: Automatic temperature tuning (α auto-adjusts to target entropy).

5. **MBPO** — Janner et al. (2019). "When to Trust Your Model: Model-Based Policy Optimization." arXiv:1906.08253. NeurIPS 2019.
   - Key insight: Short model-generated rollouts branched from real data. Don't trust the model for long horizons — use it for 1-5 step imagination, then anchor to real data.
   - **Critical for finance**: Financial world models have short reliable horizons. MBPO's "trust the model for k steps" approach is exactly right — maybe k=1 to k=3 for daily trading signals.

### Risk-Sensitive RL

6. **DSAC** — Ma et al. (2020/2025). "Distributional Soft Actor-Critic for Risk-Sensitive RL." arXiv:2004.14547. JAIR.
   - Key insight: Model full return distribution, not just mean. Optimize CVaR or other risk measures while maintaining SAC's entropy regularization.
   - **Why it matters**: Trading must be risk-sensitive. Maximizing expected return without CVaR constraint = ruin. DSAC unifies distributional RL + MaxEnt + risk metrics.
   - Specific technique: Quantile-based return distribution + CVaR optimization within actor update.

### Sequence Modeling / Offline RL

7. **Decision Transformer** — Chen et al. (2021). "Reinforcement Learning via Sequence Modeling." arXiv:2106.01345. NeurIPS 2021.
   - Key insight: Frame RL as sequence prediction: (return-to-go, state, action) → next action. No value function, no policy gradient — just autoregressive prediction conditioned on desired return.
   - Relevance: Could be useful for *offline* pre-training from historical backtests. Given a desired Sharpe ratio, what sequence of entity-signal → allocation decisions achieves it?
   - **Concern**: Decision Transformer struggles with stochastic environments (Chen et al. note), and markets are maximally stochastic.

### Exploration & Curiosity

8. **ICM** — Pathak et al. (2017). "Curiosity-driven Exploration by Self-supervised Prediction." arXiv:1705.05363. ICML 2017.
   - Key insight: Curiosity = prediction error in learned feature space. Agent is rewarded for finding states it can't predict.
   - **Direct parallel to TirraMind**: Our GNN prediction surprise IS already an intrinsic curiosity signal! The entity scoring pipeline already computes exactly what ICM proposes — prediction error as a signal. Phase 21 should use composite surprise as *intrinsic reward* for the RL policy alongside extrinsic P&L reward.

9. **Curriculum Learning for RL** — Narvekar et al. (2020). arXiv:2003.04960. JMLR.
   - Relevance: Start training on easier entities/markets (high liquidity, clear signals) before graduating to harder ones.

### Position Sizing Theory

10. **Kelly Criterion** — Kelly (1956). Wikipedia/Thorp (1997).
    - Formula for single asset: $f^* = \frac{\mu - r}{\sigma^2}$
    - Multi-asset: $\vec{u}^* = (1+r) \hat{\Sigma}^{-1} (\hat{\vec{r}} - r)$
    - **Critical caveat**: Kelly assumes known probabilities. With estimation error, half-Kelly or quarter-Kelly is standard practice (Thorp 1997).
    - **For TirraMind**: Kelly sizing is the *optimal theoretical target* when belief state confidence is high. As confidence → 0, position → 0. The RL policy should learn to interpolate between Kelly-optimal and zero-position based on confidence calibration.

## Current Architecture

### Existing RL Infrastructure

| Component | Location | What It Does |
|-----------|----------|-------------|
| `StrategyBandit` | `agent/learning/bandit.py` | Thompson Sampling (Beta(α,β)) for 40+ arms — selects *what category of work to do* |
| `Evaluator` | `agent/learning/evaluator.py` | LLM-based outcome scoring (0-1) |
| `compute_reward()` | `agent/learning/reward.py` | Weighted blend: eval 40%, Sharpe 30%, knowledge 20%, novelty 10% |
| `Reflector` | `agent/learning/reflection.py` | LLM-based history analysis |
| `GoalGenerator` | `agent/learning/goal_generator.py` | LLM fills specific goals within bandit-chosen arm |
| `AutonomousRunner` | `agent/core/autonomous.py` | Full loop: reflect → choose → generate → execute → evaluate → reward → update |

### Key Observation

The existing bandit is an **exploration policy** — it decides *which data source to investigate*. Phase 21's RL policy is fundamentally different: it decides *what to bet on and how much*, given entity surprise signals. These are two separate RL problems operating at different timescales.

| Dimension | Existing Bandit | Phase 21 RL Policy |
|-----------|----------------|-------------------|
| **Decision** | Which arm (investigation type) to pursue | How much capital to allocate per entity signal |
| **State** | Reflection + history | Entity surprises + beliefs + market state |
| **Action** | Categorical: choose 1 of 40+ arms | Continuous: weight vector over entities/assets |
| **Reward** | LLM evaluation score + Sharpe | Walk-forward P&L (Sharpe, CVaR) |
| **Horizon** | Per-session (5 iterations) | Daily/weekly trading windows |
| **Timescale** | Hours | Weeks to months |

### Signal Pipeline to Action Gap

```
Entity surprise signals (5d per entity)     ← Phase 20 produces this
ConvergenceClusters (grouped alerts)         ← Phase 20 produces this
BeliefState (macro regime, per-variable)     ← Phase 19 produces this
                                 ↓
              ╔════════════════════════════╗
              ║  >>> PHASE 21 GOES HERE <<< ║
              ╚════════════════════════════╝
                                 ↓
Position sizes (weight per asset/entity)     → to backtest/paper trading
P&L, Sharpe, CVaR                            → reward signal back to policy
```

### Files Relevant to Phase 21

- `agent/fusion/entity_scorer.py` — ScorerConfig has hard-coded surprise weights
- `agent/fusion/surprise.py` — SurpriseExtractor computes 5 signals + composite
- `agent/models/belief.py` — BeliefState distributions
- `agent/quant/backtest.py` — WalkForward backtester + Strategy ABC
- `agent/quant/scoring.py` — Sharpe, Sortino, MaxDrawdown, CVaR, VaR
- `agent/learning/bandit.py` — Existing Thompson Sampling (orthogonal, do not replace)
- `agent/pipeline/dags/` — DAG scheduler for pipeline steps

## Observations

1. **The existing bandit should NOT be replaced.** It serves a different purpose (exploration of investigation arms). Phase 21 adds a *second* RL system for portfolio allocation.
2. **No external RL library in dependencies.** torch + torch_geometric are already present via GNN. Build the RL policy with raw PyTorch — no gymnasium, no SB3. Our state/action spaces don't fit their APIs.
3. **The world model already exists.** HetTGN memory states + BeliefState distributions = latent state for policy. We don't need to learn a new world model — we already have one.
4. **Walk-forward backtest exists.** `WalkForward` + `Strategy` ABC in `agent/quant/backtest.py` provides the evaluation infrastructure. Phase 21's policy just needs to implement the `Strategy` interface.
5. **Composite surprise weights are the simplest "policy" to learn.** Before building a full SAC actor-critic, learn the 5 weights by gradient descent on backtest Sharpe. This is the minimum viable RL step.
6. **GNN prediction surprise ≈ ICM intrinsic reward.** The architecture already implements curiosity-driven exploration at the entity level. Phase 21 should explicitly use composite surprise as intrinsic reward alongside P&L as extrinsic reward.

## Risks

### Technical Risks

- **Non-stationarity**: Financial markets are non-stationary. An RL policy trained on 2020 data may be useless in 2025. Mitigation: short training windows, regime conditioning, entropy regularization (SAC).
- **Sparse rewards**: P&L is realized daily/weekly but entity signals arrive irregularly. Reward attribution is hard. Mitigation: use the GNN surprise as dense intrinsic reward.
- **Overfitting to backtest**: Walk-forward validation mitigates but doesn't eliminate look-ahead bias. Mitigation: strict separation of train/validation/holdout; penalize turnover; use fractional Kelly.
- **Catastrophic drawdown**: Without risk constraints, an RL agent can learn to take huge positions. Mitigation: CVaR constraint (DSAC), position caps, half-Kelly maximum.
- **Estimation error in Kelly**: Kelly assumes known probabilities. Our beliefs are estimates with significant uncertainty. Mitigation: use the belief variance directly — scale Kelly fraction by confidence.

### Licensing Risks
- None. All referenced papers are academic (CC-BY or academic license). All implementations will be original.

### Testing Risks
- RL policies are hard to unit test (stochastic, slow convergence). Mitigation: test with synthetic environments first (known-edge scenarios), then validate on historical data.

## Data Requirements

### Required Inputs
- **EntityAlerts** with 5 surprise signals — from `entity_alerts` table (Phase 20)
- **ConvergenceClusters** — from `convergence_clusters` table (Phase 20)
- **BeliefState distributions** — from world model (Phase 19)
- **Market returns** — from `market_data` tool (yfinance) or cached OHLCV
- **Risk-free rate** — from FRED (`DGS3MO` or `DTB3`)

### What Already Exists Locally
- All Phase 20 signals are stored in PipelineStore
- Phase 19 beliefs are stored in PipelineStore
- Walk-forward backtester is fully operational
- Market data tool fetches from yfinance

### What Still Needs to Be Added
- **Entity-to-asset mapping**: EntityAlerts are for entities (companies, people). Need a mapping from entity_id → tradeable asset (ticker symbol). Some entities (like "AAPL" company) map directly; others (vessels, wallets) require inference.
- **Historical replay**: Need ability to replay entity scoring at historical timestamps for training. Currently entity_scoring DAG runs live only.
- **Paper trading environment**: Simulated execution with realistic slippage/spread.

## Math/Algorithm Survey

### The Core Decision Problem

Given:
- $\mathbf{s}_t = (\text{EntitySurprise}_{1..N}, \text{BeliefState}_{1..K}, \text{MarketState}_t)$ — state at time $t$
- $\mathbf{a}_t = (w_1, w_2, ..., w_M)$ where $w_i \in [-1, 1]$, $\sum |w_i| \leq L$ (leverage limit $L$, typically 1.0) — portfolio weights

Maximize:
$$J(\pi) = \mathbb{E}_{\pi}\left[\sum_{t=0}^{T} \gamma^t \left(r_t^{\text{extrinsic}} + \lambda \cdot r_t^{\text{intrinsic}} + \alpha \mathcal{H}[\pi(\cdot|s_t)]\right)\right]$$

where:
- $r_t^{\text{extrinsic}} = \text{risk-adjusted P\&L}$ (Sharpe-normalized returns, with CVaR penalty)
- $r_t^{\text{intrinsic}} = \text{composite surprise}$ (the GNN prediction error signal — drives exploration of entities with novel behavior)
- $\mathcal{H}[\pi]$ = entropy of the policy (SAC-style exploration)
- $\alpha$ = temperature parameter (auto-tuned per SAC)
- $\lambda$ = intrinsic reward weight (decay over training as policy improves)

### Implementation Options Compared

| Approach | Pros | Cons | Fit for TirraMind |
|----------|------|------|-------------------|
| **A: Learned Surprise Weights Only** | Simplest; ~5 params; no RL framework needed; gradient descent on backtest Sharpe | Not a policy — just feature weighting. Doesn't learn position sizing. | **Phase 21a: MVP** |
| **B: Model-Free SAC** (Soft Actor-Critic) | Proven, off-policy (sample efficient), entropy regularization, continuous actions | Needs many environment interactions; no model-based imagination | Phase 21b |
| **C: MBPO-style** (short rollouts from real data) | Uses world model for 1-3 step imagination, anchored to real data | Requires differentiable world model transitions; complexity | Phase 21c |
| **D: DreamerV3-style** (full imagination training) | Most data-efficient; trains entirely in latent space | Requires latent dynamics model; highest complexity | Phase 21d (future) |
| **E: Decision Transformer** (offline, sequence modeling) | No value function; condition on desired return | Needs large offline dataset; poor in stochastic envs | **Rejected for now** |

### Recommended Progression

**Phase 21a (MVP — Surprise Weight Learning)**
- Learn 5 surprise weights + intercept by optimizing walk-forward Sharpe ratio
- Method: differentiable backtest (PyTorch), gradient ascent on Sharpe
- Why: proves whether entity surprise signals predict anything at all. If this fails, nothing downstream matters.
- Objective: $\max_{\mathbf{w}} \text{Sharpe}\left(\sum_{t} \mathbf{w}^T \mathbf{s}_t \cdot r_{t+1}\right)$ subject to $\|\mathbf{w}\|_1 = 1, w_i \geq 0$

**Phase 21b (Actor-Critic Position Sizing)**
- State: [composite_surprise, belief_mean, belief_variance, regime_label, market_features]
- Action: weight vector over tradeable entities (continuous, simplex-constrained)
- Reward: $r_t = \frac{\text{portfolio\_return}_t}{\text{CVaR}_{0.05}(\text{returns}_{t-W:t})} + \lambda \cdot \text{composite\_surprise}_t$
- Algorithm: SAC with automatic temperature tuning
- Risk constraint: position caps at half-Kelly: $|w_i| \leq \frac{1}{2} \cdot \frac{\mu_i - r}{\sigma_i^2} \cdot c_i$ where $c_i$ is belief confidence

**Phase 21c (Model-Based Imagination)**
- Use HetTGN memory as world model: given current entity states + hypothetical action, predict next-step entity states and reward
- 1-3 step imagination rollouts (MBPO-style), branched from real historical states
- Train SAC policy on mixture of real and imagined transitions

### Kelly Criterion Integration

For entity $i$ with estimated edge $\hat{\mu}_i$, volatility $\hat{\sigma}_i$, and belief confidence $c_i \in [0, 1]$:

$$w_i^{\text{Kelly}} = \frac{\hat{\mu}_i - r}{\hat{\sigma}_i^2}$$

$$w_i^{\text{fractional}} = \frac{c_i}{2} \cdot w_i^{\text{Kelly}}$$

The factor $c_i / 2$ serves dual purpose:
- Half-Kelly ($1/2$) — standard risk reduction for estimation error (Thorp 1997)
- Confidence scaling ($c_i$) — when belief is uncertain, don't bet. When $c_i = 0$, position = 0 regardless of estimated edge.

The RL policy should *learn* to approximate this relationship (and potentially improve on it), rather than having it hard-coded. The Kelly formula provides the initialization and theoretical ceiling.

### Reward Function Design

**Extrinsic reward** (P&L-based):
$$r_t^{\text{ext}} = \frac{\sum_i w_{i,t} \cdot \text{ret}_{i,t+1}}{\max(\sigma_W, \epsilon)} - \kappa \cdot \text{max}(0, -\text{CVaR}_{0.05})$$

where $\sigma_W$ is rolling portfolio volatility and $\kappa$ is the CVaR penalty coefficient.

**Intrinsic reward** (surprise-based):
$$r_t^{\text{int}} = \frac{1}{N_t} \sum_{i \in \text{alerted}} \text{composite\_surprise}_{i,t}$$

This rewards the policy for attending to entities with novel behavior. Decays with $\lambda(t) = \lambda_0 \cdot (1 - t/T)$ to shift from exploration to exploitation.

**Combined**:
$$r_t = r_t^{\text{ext}} + \lambda(t) \cdot r_t^{\text{int}}$$

### Numerical Stability Concerns

1. **Return normalization**: Use symlog transforms (per DreamerV3) rather than z-scoring, which fails with regime shifts.
2. **Gradient clipping**: Essential for differentiable backtest — returns can be extreme.
3. **Entropy collapse**: SAC temperature auto-tuning prevents premature policy collapse. Target entropy = $-\dim(\mathcal{A})/2$.
4. **CVaR estimation**: Use historical samples, not parametric assumption. Requires sufficient history (>50 periods).
5. **Position cap enforcement**: Hard clamp after softmax, not in loss function. Ensures constraints are always satisfied.

## Depth Roadmap

- **L1 (Phase 21a)**: Learn surprise signal weights. Single aggregate score per entity → asset mapping → simple long/short.
- **L2 (Phase 21b)**: Entity-level position sizing. Per-entity allocation based on its full surprise vector + belief state + neighborhood.
- **L3 (Phase 21c)**: Cross-entity portfolio optimization. Correlated entities (from convergence clusters) sized jointly. Copula-based tail dependence in risk model.
- **L4 (future)**: Model-based imagination. GNN world model generates counterfactual entity trajectories; policy trains entirely in imagination.

## Related

- [[rl_policy_spec]] — Spec for Phase 21
- [[rl_policy|RL Policy task]] — Task file for Phase 21
- [[signal_fusion]] — Research note for Phase 20 (EntityAlert, surprise signals)
- [[signal_fusion_spec]] — Spec for Phase 20
- [[world_model_bridge]] — Research note for Phase 19 (GNN ↔ World Model)
- [[world_model_bridge_spec]] — Spec for Phase 19
- [[quant_training_ground]] — Master task tracker
- [[rl_policy_spec]] — Spec (to be written next)
