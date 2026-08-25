---
title: "Feature: Tier 6 — Learned Feature Selection & Tool Routing"
tags:
  - doc/research
  - phase/25
  - topic/self-improving
  - topic/feature-selection
  - topic/tool-routing
  - layer/learning
  - layer/surveillance
---

# Feature: Tier 6 — Learned Feature Selection & Tool Routing

## Goal

Move from 75% → 82% learned by replacing two hand-coded observation decisions:

1. **Change 11 — Learned Feature Selection**: The SAC policy currently receives all 463 state dimensions regardless of regime. A differentiable feature gate should learn which features matter per regime, replacing the static feature list.

2. **Change 12 — Learned Tool Routing**: The daily collection DAG currently runs all 7 tools on a fixed weekday schedule. A contextual bandit should learn which tools to invoke based on regime context and tool contribution to downstream signal quality.

## Search Log

- GitHub/arXiv keywords searched: "learned feature selection reinforcement learning", "attention gating gradient based feature importance deep RL", "learned tool routing policy network data source selection bandit contextual multi-armed bandits"
- Key papers found:
  - Blakeman & Mareschal 2020 — Selective Particle Attention: feature-based attention in Deep RL (arXiv:2008.11491)
  - Jang et al. 2017 — Categorical Reparameterization with Gumbel-Softmax (arXiv:1611.01144)
  - Kendall, Gal & Cipolla 2018 — Multi-Task Learning Using Uncertainty (already implemented as Change 9)
  - Feature-Gating MoE — dynamic feature gating with sparsity (emergentmind.com)
  - BARP (arXiv:2510.07429) — Bandit routing of LLMs, relevant pattern for tool routing
  - Contextual bandits for resource allocation — standard RL formulation

## External Repositories Reviewed

No external repositories needed. All required infrastructure already exists in the codebase:
- `StrategyBandit` (Thompson Sampling with hierarchical bandits) — `agent/learning/bandit.py`
- `LearnedStateEncoder` (attention over entities) — `agent/learning/policy/state_encoder.py`
- `BayesianParamOptimizer` (GP-BO) — `agent/learning/param_optimizer.py`

## Documentation Reviewed

- PyTorch `nn.functional.gumbel_softmax` — differentiable discrete sampling
- PyTorch `MultiheadAttention` — already used in LearnedStateEncoder
- Existing `StrategyBandit` API — `suggest() -> GoalArm`, `update(arm, reward)`, Thompson Sampling with Beta priors

## Current Architecture

### Feature Pipeline (Change 11 target)

State vector layout (463 dims for E=50, M=8):
```
[0    : 250]   → surprise vectors (5 per entity × 50 entities)     ← FEATURE GROUP 0
[250  : 450]   → belief features (4 per entity × 50 entities)      ← FEATURE GROUP 1
[450  : 458]   → global market features (8 dims)                   ← FEATURE GROUP 2
[458  : 459]   → normalized entity count (1 dim)                   ← FEATURE GROUP 3
[459  : 463]   → adversarial summary (4 dims)                      ← FEATURE GROUP 4
```

The `StateAssembler` builds this vector. The `LearnedStateEncoder` (Change 6) reshapes it into entity tokens + global features, applies MHA, and outputs a compact state. **The encoder attends over entities but treats all per-entity features equally.** Change 11 adds gating over the feature dimensions themselves.

### Tool Routing (Change 12 target)

DAG schedule (all fixed, all run every weekday):
```
18:00  daily_collection — 7 parallel nodes:
         fetch_cftc (CFTC CoT)
         fetch_finra_scan (short volume)
         fetch_power_demand (NYISO demand)
         fetch_power_fuel (NYISO fuel mix)
         fetch_gdelt (geopolitical events)
         fetch_polymarket (prediction markets)
         fetch_instruments (instrument universe + prices)
```

`fetch_instruments` is a **must-run** (provides instrument universe). The other 6 are candidates for adaptive routing. Current cost: all free APIs, so the cost is latency + compute, not money. But downstream signal quality varies by tool × regime.

## Observations

### Change 11 — Learned Feature Selection

**What exists:**
- `StateAssembler` builds the raw 463-dim state vector
- `LearnedStateEncoder` (Change 6) provides attention over *entities* but not feature dimensions
- `DifferentiableStateAssembler` (Phase B) preserves gradients for Kalman beliefs
- `InstrumentStateAssembler` extends with per-instrument surprise vectors

**What's missing:**
- No mechanism to gate or weight *feature dimensions* based on regime
- All 5 surprise dims and 4 belief dims receive equal treatment regardless of market regime
- No sparsity incentive — the policy must learn to ignore irrelevant features implicitly

**Design space:**
| Approach | Pros | Cons | Complexity |
|----------|------|------|-----------|
| **A: Per-dim sigmoid gates (static)** | Simplest, acts like learned L1 | No regime conditioning, limited adaptation | ~30 LOC |
| **B: Regime-conditioned soft gating** | Adapts per regime, interpretable diagnostics | Needs regime signal input, more params | ~120 LOC |
| **C: Gumbel-Softmax hard selection** | True feature selection, sparse | Gradient variance, temperature annealing | ~150 LOC |
| **D: Feature-group attention** | Group-level gating (surprise/belief/market/adv), efficient | Coarse granularity | ~80 LOC |

**Recommendation: Option B — Regime-conditioned soft gating** at the per-feature-group level initially, with per-dim refinement as a future step. This is the sweet spot: adapts to regime, interpretable (we can inspect gate values per regime), and integrates cleanly with the existing encoder pipeline.

### Change 12 — Learned Tool Routing

**What exists:**
- `StrategyBandit` with Thompson Sampling (Beta priors), arm discovery, persistence
- `BayesianParamOptimizer` for GP-BO
- Fixed DAG schedule in `daily_collection.py`
- `DAGExecutor` that runs all nodes, no concept of "skip this tool today"

**What's missing:**
- No mechanism to conditionally run/skip tools based on context
- No measurement of per-tool signal contribution
- No cost model for tools (all free, but latency varies)
- No freshness tracking (how stale is each tool's data?)

**Design space:**
| Approach | Pros | Cons | Complexity |
|----------|------|------|-----------|
| **A: Contextual Thompson Sampling** | Extends existing bandit, works with small data | Limited policy expressiveness | ~200 LOC |
| **B: Policy network (MLP)** | Expressive, end-to-end differentiable | Needs more data, overfitting risk | ~300 LOC |
| **C: UCB with context features** | Well-understood regret bounds | Less adaptive than TS | ~150 LOC |

**Recommendation: Option A — Contextual Thompson Sampling** extending the existing `StrategyBandit` pattern. Each tool is an arm. Context features = regime state + day of week + tool freshness. Reward = tool's contribution to downstream entity alert quality (measured via leave-one-out signal degradation). This reuses existing infrastructure and provides interpretable routing decisions.

## Risks

- **Feature gate collapse**: All gates could go to 0 or 1 during training. Mitigate with gate entropy regularization and minimum gate floor.
- **Tool routing cold start**: Need initial data showing which tools contribute signal before the bandit can learn. Mitigate with warm-start from uniform exploration period.
- **Regime signal dependency**: If the regime signal is poor, conditioned gating won't improve over static gating. The HMM regime detector is already trained (Change 2a).
- **Integration complexity**: Both changes touch the hot path (state assembly → policy). Must preserve backward compatibility and cold-start fallback.

## Math/Algorithm Survey

### Change 11 — Feature Gate

Regime-conditioned soft gating over D feature groups:

$$g_k = \sigma(W_k \cdot r + b_k), \quad k \in \{0, \ldots, K-1\}$$

where $r$ is the regime context vector (HMM posterior + Kalman state summary), $\sigma$ is sigmoid, and $g_k \in [0, 1]$ is the gate value for feature group $k$.

The gated state is:

$$\tilde{x} = [g_0 \cdot x_{\text{surprise}}; \; g_1 \cdot x_{\text{belief}}; \; g_2 \cdot x_{\text{market}}; \; g_3 \cdot x_{\text{entity\_count}}; \; g_4 \cdot x_{\text{adversarial}}]$$

**Entropy regularization** to prevent gate collapse:

$$\mathcal{L}_{\text{gate}} = -\lambda \sum_k [g_k \log g_k + (1-g_k) \log(1-g_k)]$$

Gates are learnable parameters (via $W_k, b_k$) trained end-to-end with the SAC actor loss.

### Change 12 — Tool Routing Bandit

Contextual Thompson Sampling for tool selection:

For each tool $i$, maintain a Bayesian linear regression model:

$$r_i | \mathbf{c} \sim \mathcal{N}(\mathbf{w}_i^T \mathbf{c}, \, \sigma_i^2)$$

where $\mathbf{c}$ is the context vector and $\mathbf{w}_i$ are per-tool weights. At each decision point, sample from the posterior and select tools with positive expected reward:

$$\tilde{r}_i = \tilde{\mathbf{w}}_i^T \mathbf{c}, \quad \text{run tool } i \text{ iff } \tilde{r}_i > \tau$$

where $\tilde{\mathbf{w}}_i \sim \text{posterior}$ and $\tau$ is a minimum signal threshold. Simplification: use the existing Beta-based Thompson Sampling with regime-conditioned arms instead of Bayesian linear regression.

**Tool signal contribution** measured as leave-one-out: run the convergence detector with and without tool $i$'s data → measure drop in downstream entity alert confidence.

## Implementation Intent

**Approved for implementation:**
1. `FeatureGate` nn.Module — regime-conditioned soft gating over feature groups with entropy regularization
2. Integration into `LearnedStateEncoder` or as a standalone gate before SAC input
3. `ToolRoutingBandit` — contextual Thompson Sampling over tool arms, extending `StrategyBandit` pattern
4. Integration with `DAGExecutor` — skip/include tools based on bandit decision
5. Tool signal contribution measurement via leave-one-out

**Concepts rejected:**
- Gumbel-Softmax (too much gradient variance for our small-data regime)
- Policy network for tool routing (overfitting risk with 7 arms)
- Per-individual-dimension gating (start with group-level, refine later)

## Related

- [[tier6_learned_observation_spec]]
- [[learned_vs_handcoded_architecture_spec]]
- [[learned_architecture_impl]]
- [[tier5_differentiable_kalman]]
- [[learned_state_encoder]]
- [[causal_dag_structure_learning]]
