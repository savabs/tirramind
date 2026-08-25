---
title: "Spec: RL Policy — Surprise-Driven Portfolio Allocation"
tags:
  - doc/spec
  - phase/21
  - topic/rl-policy
  - topic/portfolio-optimization
  - topic/surprise-weighting
  - layer/learning
---

# Spec: RL Policy — Surprise-Driven Portfolio Allocation

## Goal

Close the learning loop from **entity surprise signals → learned allocation decisions → market outcomes → reward → updated policy**. Currently, the system produces EntityAlerts with 5 surprise signals weighted by hand-coded constants `[0.30, 0.15, 0.25, 0.20, 0.10]`, but nothing downstream consumes them to make decisions or learn from outcomes.

Phase 21 builds two stacked RL systems:

1. **Phase 21a (Surprise Weight Learning)**: Learn the 5 composite surprise weights + entity-to-asset mapping via differentiable backtest. This is the minimum viable test — if learned weights on surprise signals can't beat random weights in walk-forward Sharpe, nothing downstream matters.

2. **Phase 21b (SAC Actor-Critic)**: A Soft Actor-Critic policy that maps the full state (surprise vectors + belief states + market features) to continuous portfolio weights. Uses dual reward (extrinsic P&L + intrinsic surprise), CVaR risk constraint, and half-Kelly position caps.

Phase 21c (MBPO imagination) is deferred — it depends on 21b validation results.

### Non-goals
- Do **NOT** replace the existing Thompson Sampling bandit (`StrategyBandit`). It operates at a different timescale (investigation arm selection) and is orthogonal.
- Do **NOT** add external RL libraries (gymnasium, stable-baselines3). Our state space is a heterogeneous graph + belief vector; their APIs don't fit.
- Do **NOT** build a paper trading infrastructure yet. Walk-forward backtest is the evaluation framework.

## Paradigm

| Before Phase 21 | After Phase 21 |
|------------------|---------------|
| Hard-coded surprise weights `[0.30, 0.15, 0.25, 0.20, 0.10]` | Learned weights optimized for walk-forward Sharpe |
| Entity alerts as dead-end output | Entity alerts as RL state inputs |
| No entity → asset mapping | Entity-to-asset resolution via alias table |
| No position sizing | SAC policy outputs continuous portfolio weights |
| No P&L feedback | Walk-forward P&L drives reward signal |
| Surprise is anomaly score only | Surprise serves dual role: anomaly score + intrinsic reward (ICM parallel) |

## Math Overview

### Core Decision Problem

Given:
- $\mathbf{s}_t = (\text{EntitySurprise}_{1..N}, \text{BeliefState}_{1..K}, \text{MarketState}_t)$ — state at time $t$
- $\mathbf{a}_t = (w_1, w_2, ..., w_M)$ where $w_i \in [-1, 1]$, $\sum |w_i| \leq L$ (leverage limit $L$, typically 1.0) — portfolio weights

Maximize:
$$J(\pi) = \mathbb{E}_{\pi}\left[\sum_{t=0}^{T} \gamma^t \left(r_t^{\text{ext}} + \lambda(t) \cdot r_t^{\text{int}} + \alpha \mathcal{H}[\pi(\cdot|s_t)]\right)\right]$$

where:
- $r_t^{\text{ext}}$ = risk-adjusted P&L (Sharpe-normalized, with CVaR penalty)
- $r_t^{\text{int}}$ = composite surprise (GNN prediction error — ICM-style intrinsic reward)
- $\mathcal{H}[\pi]$ = policy entropy (SAC exploration)
- $\alpha$ = temperature (auto-tuned per SAC)
- $\lambda(t) = \lambda_0 \cdot (1 - t/T)$ = decaying intrinsic weight

### Phase 21a Objective (Surprise Weight Learning)

$$\max_{\mathbf{w}} \text{Sharpe}\left(\sum_{t} \mathbf{w}^T \mathbf{s}_t \cdot r_{t+1}\right) \quad \text{s.t.} \quad \|\mathbf{w}\|_1 = 1, \ w_i \geq 0$$

### Phase 21b Kelly Integration

For entity $i$ with estimated edge $\hat{\mu}_i$, volatility $\hat{\sigma}_i$, and belief confidence $c_i \in [0, 1]$:

$$w_i^{\text{Kelly}} = \frac{\hat{\mu}_i - r}{\hat{\sigma}_i^2}, \qquad w_i^{\text{max}} = \frac{c_i}{2} \cdot w_i^{\text{Kelly}}$$

The SAC policy learns to output weights bounded by $w_i^{\text{max}}$.

### Reward Function

**Extrinsic** (P&L):
$$r_t^{\text{ext}} = \frac{\sum_i w_{i,t} \cdot \text{ret}_{i,t+1}}{\max(\sigma_W, \epsilon)} - \kappa \cdot \max(0, -\text{CVaR}_{0.05})$$

**Intrinsic** (surprise):
$$r_t^{\text{int}} = \frac{1}{N_t} \sum_{i \in \text{alerted}} \text{composite\_surprise}_{i,t}$$

### Numerical Stability

- **Return normalization**: symlog transforms per DreamerV3 (not z-scoring, which fails with regime shifts)
- **Gradient clipping**: norm-clip at 1.0 for differentiable backtest
- **Entropy collapse prevention**: SAC auto-temperature with target entropy $= -\dim(\mathcal{A})/2$
- **CVaR estimation**: historical samples (not parametric), minimum 50 periods
- **Position cap enforcement**: hard clamp after softmax (not in loss function)

### Trusted Sources

| Method | Source | Rationale |
|--------|--------|-----------|
| SAC | Haarnoja et al. (2018) arXiv:1801.01290 | Standard MaxEnt RL, proven off-policy efficiency |
| Symlog normalization | Hafner et al. (2023) DreamerV3 arXiv:2301.04104 | Handles non-stationary reward scales |
| MBPO short rollouts | Janner et al. (2019) arXiv:1906.08253 | Anchors imagination to real data |
| ICM intrinsic reward | Pathak et al. (2017) arXiv:1705.05363 | Our surprise IS prediction error in learned feature space |
| Kelly sizing | Kelly (1956) / Thorp (1997) | Optimal sizing under known probabilities; half-Kelly for estimation error |
| DSAC risk-sensitivity | Ma et al. (2020) arXiv:2004.14547 | CVaR optimization within SAC framework |
| Differentiable Sharpe | Moody & Saffell (2001) | Gradient-based portfolio optimization via backprop through returns |

## Files Affected

### New Files

| File | Purpose | Layer |
|------|---------|-------|
| `agent/learning/policy/__init__.py` | Module init + public exports | L5 |
| `agent/learning/policy/config.py` | PolicyConfig, SACConfig, RewardConfig dataclasses | L5 |
| `agent/learning/policy/asset_mapper.py` | Entity-to-asset resolution via entity_aliases table | L5 |
| `agent/learning/policy/state_assembler.py` | Combine surprise + belief + market → state tensor | L5 |
| `agent/learning/policy/reward_fn.py` | Extrinsic + intrinsic reward computation | L5 |
| `agent/learning/policy/weight_learner.py` | Phase 21a: differentiable backtest for surprise weights | L5 |
| `agent/learning/policy/replay_buffer.py` | Off-policy experience replay buffer | L5 |
| `agent/learning/policy/sac.py` | SAC actor, twin critics, temperature, training loop | L5 |
| `agent/learning/policy/portfolio_strategy.py` | Strategy ABC adapter for walk-forward evaluation | L5 |
| `agent/learning/policy/symlog.py` | Symlog/symexp transforms for reward normalization | L5 |
| `agent/pipeline/dags/rl_training.py` | DAG for periodic RL policy training | L1 |
| `tests/test_asset_mapper.py` | Entity-to-asset mapping tests | - |
| `tests/test_state_assembler.py` | State tensor assembly tests | - |
| `tests/test_reward_fn.py` | Reward function tests | - |
| `tests/test_weight_learner.py` | Differentiable backtest + weight learning tests | - |
| `tests/test_replay_buffer.py` | Replay buffer tests | - |
| `tests/test_sac.py` | SAC network + training tests | - |
| `tests/test_portfolio_strategy.py` | Strategy adapter + walk-forward tests | - |
| `tests/test_symlog.py` | Symlog transform tests | - |
| `tests/test_rl_training_dag.py` | DAG integration tests | - |

### Modified Files

| File | Change |
|------|--------|
| `agent/fusion/entity_scorer.py` | Accept optional learned weights from PolicyConfig instead of hard-coded weights |
| `agent/fusion/surprise.py` | Accept optional weight override in SurpriseExtractor constructor |
| `agent/learning/__init__.py` | Export policy sub-module |
| `agent/pipeline/store.py` | Add `rl_transitions` and `rl_policy_checkpoints` tables |

## Implementation Steps

---

### Phase 21a: Surprise Weight Learning (MVP)

#### Step 21a.1: Create policy module + config dataclasses

**Files:** `agent/learning/policy/__init__.py`, `agent/learning/policy/config.py`

Define configuration dataclasses. Three configs for layered control:

```python
@dataclass(frozen=True)
class RewardConfig:
    """Reward function parameters."""
    cvar_confidence: float = 0.95
    cvar_penalty: float = 1.0           # κ
    intrinsic_weight_initial: float = 0.1  # λ₀
    intrinsic_decay: bool = True
    rolling_vol_window: int = 20
    vol_floor: float = 1e-8             # ε

@dataclass(frozen=True)
class WeightLearnerConfig:
    """Phase 21a: differentiable backtest config."""
    learning_rate: float = 0.01
    max_epochs: int = 200
    patience: int = 20                   # early stopping
    grad_clip_norm: float = 1.0
    min_train_periods: int = 104         # 2 years weekly
    test_periods: int = 52              # 1 year weekly
    walk_forward_step: int = 26          # 6 months
    l2_reg: float = 1e-4

@dataclass(frozen=True)
class SACConfig:
    """Phase 21b: SAC hyperparameters."""
    hidden_dim: int = 128
    num_hidden: int = 2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4               # temperature learning rate
    gamma: float = 0.99
    tau: float = 0.005                   # soft target update
    batch_size: int = 256
    buffer_size: int = 100_000
    target_entropy_scale: float = -0.5   # target = scale * dim(action)
    max_position: float = 0.5           # half-Kelly cap as fraction
    leverage_limit: float = 1.0
    warmup_steps: int = 1000

@dataclass(frozen=True)
class PolicyConfig:
    """Top-level config aggregating all sub-configs."""
    reward: RewardConfig = field(default_factory=RewardConfig)
    weight_learner: WeightLearnerConfig = field(default_factory=WeightLearnerConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    # Learned weights (populated after 21a training)
    surprise_weights: tuple[float, ...] | None = None
```

**Test:** Construct all configs with defaults. Verify frozen immutability. Verify field types.

---

#### Step 21a.2: Implement symlog transforms

**Files:** `agent/learning/policy/symlog.py`, `tests/test_symlog.py`

Per DreamerV3 §3.2:
```python
def symlog(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))

def symexp(x: Tensor) -> Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
```

Also implement numpy versions for scoring functions.

**Tests:** Round-trip `symexp(symlog(x)) ≈ x` for positive, negative, zero, large, tiny values. Gradient exists through symlog. Numerical stability at float32 extremes.

---

#### Step 21a.3: Implement entity-to-asset mapper

**Files:** `agent/learning/policy/asset_mapper.py`, `tests/test_asset_mapper.py`

Resolve entity_id → tradeable ticker using the existing `entity_aliases` table.

```python
class AssetMapper:
    def __init__(self, store: PipelineStore) -> None: ...

    def resolve(self, entity_id: str) -> str | None:
        """Return ticker symbol for entity, or None if not tradeable."""

    def resolve_batch(self, entity_ids: list[str]) -> dict[str, str]:
        """Return {entity_id: ticker} for all resolvable entities."""

    def tradeable_entities(self) -> dict[str, str]:
        """Return all entities with known tickers."""
```

Rules:
- Query `entity_aliases` where `source = 'ticker'`
- Only `company` entity_type is directly tradeable (for now)
- Cache alias lookups (entity→ticker mapping is stable within a session)

**Tests:** Mock PipelineStore with known aliases. Resolve single entity. Batch resolve with mix of tradeable/non-tradeable. Empty store returns empty. Verify caching.

---

#### Step 21a.4: Implement reward function

**Files:** `agent/learning/policy/reward_fn.py`, `tests/test_reward_fn.py`

Stateless reward computation — both extrinsic and intrinsic components.

```python
class RewardFunction:
    def __init__(self, config: RewardConfig) -> None: ...

    def extrinsic(
        self,
        portfolio_return: float,
        rolling_returns: np.ndarray,  # last N returns for vol/CVaR
    ) -> float:
        """Sharpe-normalized return with CVaR penalty."""

    def intrinsic(
        self,
        surprise_scores: np.ndarray,  # composite surprise values for alerted entities
    ) -> float:
        """Mean composite surprise of active alerts."""

    def combined(
        self,
        portfolio_return: float,
        rolling_returns: np.ndarray,
        surprise_scores: np.ndarray,
        step: int,
        total_steps: int,
    ) -> tuple[float, dict[str, float]]:
        """Return (total_reward, breakdown_dict)."""
```

The breakdown dict includes `{extrinsic, intrinsic, lambda_t, raw_return, vol, cvar}` for diagnostics.

**Tests:** Known returns → known Sharpe-normalized output. CVaR penalty activates only on tail loss. Intrinsic decays correctly. Zero-division guarded by vol_floor. Empty surprise array → 0.0 intrinsic.

---

#### Step 21a.5: Implement differentiable backtest (weight learner)

**Files:** `agent/learning/policy/weight_learner.py`, `tests/test_weight_learner.py`

This is the Phase 21a core: learn 5 surprise weights by gradient ascent on walk-forward Sharpe.

```python
class SurpriseWeightLearner:
    def __init__(self, config: WeightLearnerConfig) -> None:
        # 5 raw params → softmax → normalized weights
        self._raw_weights = torch.nn.Parameter(torch.zeros(5))

    @property
    def weights(self) -> torch.Tensor:
        """Normalized weights via softmax (sums to 1, all ≥ 0)."""
        return torch.softmax(self._raw_weights, dim=0)

    def composite_score(
        self,
        surprise_matrix: torch.Tensor,  # (T, 5) surprise signals per timestep
    ) -> torch.Tensor:
        """Return (T,) composite scores using learned weights."""

    def differentiable_sharpe(
        self,
        scores: torch.Tensor,   # (T,) composite scores
        returns: torch.Tensor,  # (T,) asset returns aligned to scores
    ) -> torch.Tensor:
        """Differentiable Sharpe ratio of score-weighted returns."""

    def fit(
        self,
        surprise_matrix: np.ndarray,  # (T, 5)
        returns: np.ndarray,          # (T,)
    ) -> dict[str, Any]:
        """Walk-forward training. Returns metrics dict."""

    def get_learned_weights(self) -> tuple[float, ...]:
        """Return the 5 learned weights as a tuple."""
```

Walk-forward inside `fit()`:
1. Split data into folds per `WeightLearnerConfig`
2. For each fold: train weights on train split, evaluate Sharpe on test split
3. Average test Sharpe across folds = final metric
4. Return best weights + convergence diagnostics

Differentiable Sharpe (Moody & Saffell 2001):
$$\text{Sharpe} = \frac{\bar{r}}{\sqrt{\overline{r^2} - \bar{r}^2 + \epsilon}}$$

where $r_t = w^T s_t \cdot \text{ret}_{t+1}$ and the means are over the test window.

**Tests:**
- Synthetic scenario: 1 signal perfectly predicts returns, others are noise. Learned weight should concentrate on the predictive signal.
- All-noise signals: weights should remain approximately uniform (no signal = no learning).
- Gradient flows through softmax + Sharpe computation.
- Walk-forward folds are non-overlapping.
- Early stopping triggers when Sharpe plateaus.
- Grad clipping prevents explosion.

---

#### Step 21a.6: Modify SurpriseExtractor to accept learned weights

**Files:** `agent/fusion/surprise.py` (modify)

Change: allow `SurpriseExtractor.__init__()` to accept an optional `weights` tuple override from `PolicyConfig.surprise_weights`. If provided, use those instead of the defaults. The hard-coded defaults remain as fallback.

This is a minimal change — add one parameter to `__init__`:

```python
def __init__(
    self,
    *,
    weights: tuple[float, ...] | None = None,  # NEW: learned override
    obs_type_weight: float = 0.3,               # existing defaults
    temporal_weight: float = 0.15,
    value_weight: float = 0.25,
    neighborhood_weight: float = 0.2,
    memory_weight: float = 0.1,
) -> None:
    if weights is not None:
        assert len(weights) == 5
        obs_type_weight, temporal_weight, value_weight, neighborhood_weight, memory_weight = weights
    # ... rest unchanged
```

**Tests:** Existing tests must still pass. New test: pass custom weights tuple, verify composite_surprise uses them.

---

#### Step 21a.7: Add PipelineStore tables for RL

**Files:** `agent/pipeline/store.py` (modify)

Add two tables:

```sql
CREATE TABLE IF NOT EXISTS rl_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    state_json TEXT NOT NULL,
    action_json TEXT NOT NULL,
    reward REAL NOT NULL,
    next_state_json TEXT NOT NULL,
    done INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_rl_transitions_ts ON rl_transitions(timestamp);

CREATE TABLE IF NOT EXISTS rl_policy_checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    saved_at REAL NOT NULL,
    policy_type TEXT NOT NULL,       -- 'weight_learner' | 'sac'
    config_json TEXT NOT NULL,
    state_dict_blob BLOB NOT NULL,
    metrics_json TEXT,
    is_best INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_rl_checkpoints_type ON rl_policy_checkpoints(policy_type, saved_at);
```

Add corresponding store/query methods:
- `store_rl_transition(timestamp, state, action, reward, next_state, done, metadata)`
- `query_rl_transitions(start_time, end_time, limit)` → list of dicts
- `store_rl_checkpoint(policy_type, config, state_dict_bytes, metrics, is_best)`
- `load_best_rl_checkpoint(policy_type)` → dict or None
- `load_latest_rl_checkpoint(policy_type)` → dict or None

**Tests:** Store and retrieve transitions. Store checkpoint with blob data and retrieve. Best checkpoint query. Empty table returns None.

---

#### Step 21a.8: Edge case test suite for Phase 21a

**Files:** `tests/test_weight_learner_edge.py`

Extensive edge case coverage for the entire 21a stack:

- **Weight learner**: zero returns, constant returns, single-period data (should raise), all-NaN surprise (should raise), negative Sharpe (weights should still converge), extreme surprise values (1e6), surprise matrix with all-zero column
- **Reward function**: NaN returns, empty rolling window, zero vol (floor activated), negative CVaR (penalty applied), very large returns (overflow protection via symlog)
- **Asset mapper**: entity with multiple ticker aliases (pick first), entity type not company (return None), deleted entity alias (should not appear), SQL injection in entity_id (parameterized queries)
- **Symlog**: float32 max/min, negative zero, ±inf → should not crash
- **Config**: all defaults valid, custom config with extreme values

---

### Phase 21b: SAC Actor-Critic

#### Step 21b.1: Implement state assembler

**Files:** `agent/learning/policy/state_assembler.py`, `tests/test_state_assembler.py`

Combine heterogeneous inputs into a fixed-size state tensor for SAC.

```python
class StateAssembler:
    def __init__(
        self,
        max_entities: int = 50,    # cap for fixed tensor size
        surprise_dim: int = 5,
        belief_dim: int = 4,       # mean, var, confidence, stale
        market_dim: int = 8,       # rolling ret, vol, regime, etc.
    ) -> None: ...

    def assemble(
        self,
        alerts: list[EntityAlert],
        beliefs: list[BeliefState],
        market_features: dict[str, float],
        asset_map: dict[str, str],  # entity_id → ticker
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return (state_tensor, metadata).

        State tensor layout:
        - [0 : max_entities*surprise_dim] — surprise vectors, zero-padded
        - [... : ... + max_entities*belief_dim] — belief features, zero-padded
        - [... : ... + market_dim] — global market features
        - [... : ... + 1] — number of active entities (normalized)

        Only tradeable entities (those in asset_map) are included.
        Entities sorted by composite_surprise descending (top-K truncation).
        """

    @property
    def state_dim(self) -> int:
        """Total dimensionality of assembled state."""
```

**Tests:** Empty alerts → zero-padded state. More alerts than max_entities → top-K by surprise. Belief states matched to alerts by entity_id. Market features with missing keys → default 0.0. State dim is deterministic.

---

#### Step 21b.2: Implement replay buffer

**Files:** `agent/learning/policy/replay_buffer.py`, `tests/test_replay_buffer.py`

Standard circular buffer for off-policy SAC training.

```python
class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int) -> None: ...

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None: ...

    def sample(self, batch_size: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return (states, actions, rewards, next_states, dones) as tensors."""

    def __len__(self) -> int: ...

    @property
    def is_ready(self) -> bool:
        """True if buffer has enough samples for one batch."""
```

Uses numpy arrays internally rather than storing individual dicts — much faster for large buffers.

**Tests:** Push and sample basic. Circular overwrite when full. Sample size > buffer raises. Empty buffer sample raises. Deterministic results with fixed seed.

---

#### Step 21b.3: Implement SAC networks (actor + twin critics + temperature)

**Files:** `agent/learning/policy/sac.py`, `tests/test_sac.py`

Three neural networks + learnable temperature:

**Actor** (Gaussian policy):
```python
class GaussianActor(nn.Module):
    """Maps state → mean, log_std of action distribution.

    Action is sampled via reparameterization trick, then tanh-squashed
    and rescaled to respect leverage_limit and max_position.
    """
    def __init__(self, state_dim: int, action_dim: int, config: SACConfig) -> None: ...
    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:  # mean, log_std
    def sample(self, state: Tensor) -> tuple[Tensor, Tensor]:   # action, log_prob
```

**Twin Critics** (clipped double-Q):
```python
class TwinCritic(nn.Module):
    """Two independent Q-networks. Use min(Q1, Q2) for pessimistic update."""
    def __init__(self, state_dim: int, action_dim: int, config: SACConfig) -> None: ...
    def forward(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:  # Q1, Q2
```

**Temperature** (auto-tuned α):
```python
class AlphaScheduler:
    """Learnable log(α) with target entropy constraint."""
    def __init__(self, action_dim: int, config: SACConfig) -> None: ...
    def update(self, log_probs: Tensor) -> float:  # returns current alpha
    @property
    def alpha(self) -> float: ...
```

Key implementation notes:
- Tanh squashing with log-prob correction: $\log \pi(a|s) = \log \mu(u|s) - \sum_i \log(1 - \tanh^2(u_i))$
- Action rescaling: `action * max_position` then enforce `sum(|a|) <= leverage_limit`
- Target networks for critics: soft update with $\tau$

**Tests:**
- Forward pass shapes are correct for various state/action dims.
- Actor sample produces valid actions within bounds.
- Twin critic returns two different Q-values (different random init).
- Alpha starts at reasonable value and adjusts toward target entropy.
- Tanh squashing + log-prob correction is numerically stable (test near ±1).

---

#### Step 21b.4: Implement SAC training loop

**Files:** `agent/learning/policy/sac.py` (extend), `tests/test_sac.py` (extend)

Add `SACTrainer` class that orchestrates the update:

```python
class SACTrainer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: SACConfig,
    ) -> None: ...

    def update(self, buffer: ReplayBuffer) -> dict[str, float]:
        """One SAC update step. Returns loss metrics dict.

        Steps:
        1. Sample batch from buffer
        2. Compute target Q: r + γ * (min(Q1', Q2') - α * log_π)
        3. Update critics: minimize MSE(Q, target)
        4. Update actor: maximize Q - α * log_π
        5. Update temperature: minimize α * (log_π + target_entropy)
        6. Soft-update target networks
        """

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        """Return action as numpy array. Deterministic uses mean (no sampling)."""

    def save(self) -> bytes:
        """Serialize all network state dicts + optimizer states to bytes."""

    @classmethod
    def load(cls, data: bytes, state_dim: int, action_dim: int, config: SACConfig) -> SACTrainer:
        """Deserialize from bytes."""
```

**Tests:**
- Single update step doesn't crash with random buffer data.
- Loss values are finite.
- Soft target update: target params move toward online params by τ.
- Save/load roundtrip preserves network weights exactly.
- Deterministic action selection is reproducible with same state.

---

#### Step 21b.5: Implement portfolio strategy adapter

**Files:** `agent/learning/policy/portfolio_strategy.py`, `tests/test_portfolio_strategy.py`

Implements the existing `Strategy` ABC from `agent/quant/backtest.py` so the SAC policy can be evaluated via walk-forward backtest.

```python
class SACPortfolioStrategy(Strategy):
    """Wraps a trained SAC policy as a walk-forward Strategy."""

    def __init__(
        self,
        trainer: SACTrainer,
        state_assembler: StateAssembler,
        asset_mapper: AssetMapper,
        store: PipelineStore,
    ) -> None: ...

    @property
    def name(self) -> str:
        return "sac_rl_policy"

    def generate_weights(
        self,
        train_returns: np.ndarray,
        test_length: int,
        *,
        train_extra: dict[str, Any] | None = None,
        test_extra: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Run SAC policy on test_extra['alerts'] and test_extra['beliefs']
        to generate portfolio weights for each test period.

        train_extra/test_extra must contain:
        - 'alerts': list[list[EntityAlert]] — alerts per timestamp
        - 'beliefs': list[list[BeliefState]] — beliefs per timestamp
        - 'market_features': list[dict] — market features per timestamp
        """
```

Also implement a simpler adapter for 21a:

```python
class WeightedSurpriseStrategy(Strategy):
    """Strategy that uses learned surprise weights for signal generation."""

    def __init__(
        self,
        weights: tuple[float, ...],
        asset_mapper: AssetMapper,
        threshold: float = 2.0,
    ) -> None: ...

    @property
    def name(self) -> str:
        return "weighted_surprise"

    def generate_weights(self, ...) -> np.ndarray:
        """Generate binary long positions for entities above surprise threshold."""
```

**Tests:** Strategy interface compliance. Strategy produces valid weight arrays (correct length, value range). Walk-forward integration with mock data. Empty alerts → zero weights.

---

#### Step 21b.6: Implement historical data loader for RL training

**Files:** `agent/learning/policy/state_assembler.py` (extend)

Add a method to `StateAssembler` that loads historical entity alerts, beliefs, and market returns aligned by timestamp from PipelineStore — this is the training data for both Phase 21a and 21b.

```python
class StateAssembler:
    # ... existing methods ...

    def load_historical_episodes(
        self,
        store: PipelineStore,
        start_time: float,
        end_time: float,
        frequency: str = "weekly",  # 'daily' | 'weekly'
    ) -> dict[str, Any]:
        """Load and align historical data for RL training.

        Returns:
            {
                'timestamps': np.ndarray,        # (T,)
                'surprise_matrix': np.ndarray,    # (T, N_entities, 5)
                'belief_matrix': np.ndarray,      # (T, K_variables, 4)
                'market_returns': np.ndarray,     # (T, N_tickers)
                'ticker_list': list[str],         # aligned to market_returns cols
                'entity_to_col': dict[str, int],  # entity_id → market_returns col index
                'states': list[torch.Tensor],     # pre-assembled state tensors
            }
        """
```

This bridges the gap between stored pipeline data and RL training tensors.

**Tests:** Mock store with known alerts/beliefs/returns. Verify alignment is correct. Missing data periods → NaN or zero-fill. Frequency resampling works.

---

#### Step 21b.7: Implement RL training DAG

**Files:** `agent/pipeline/dags/rl_training.py`, `tests/test_rl_training_dag.py`

A pipeline DAG that:
1. Loads historical episodes from PipelineStore
2. Trains (or continues training) the surprise weight learner (21a)
3. If sufficient data, trains/updates SAC policy (21b)
4. Saves checkpoint to `rl_policy_checkpoints`
5. Runs walk-forward evaluation and logs metrics

```python
def build_rl_training_dag() -> dict:
    """Return DAG definition for the pipeline scheduler."""

async def run_rl_training(store: PipelineStore, config: PolicyConfig) -> dict[str, Any]:
    """Execute RL training. Returns metrics dict."""
```

Scheduled after the entity_scoring DAG (which produces EntityAlerts).

**Tests:** DAG definition is valid. Mock store → training completes. Checkpoint is saved. Metrics dict has expected keys.

---

#### Step 21b.8: Edge case test suite for Phase 21b

**Files:** `tests/test_sac_edge.py`

Extensive edge cases for Phase 21b:

- **State assembler**: alerts with no matching beliefs, beliefs with no matching alerts, all entities non-tradeable (state is all zeros), max_entities boundary (exactly max, one over, one under)
- **SAC**: single-asset action space (dim=1), very large state dim (1000+), NaN in state (should raise or handle), action exactly at boundary (±max_position), leverage limit enforcement after rounding
- **Replay buffer**: buffer size 1, buffer size 0 (should raise), sample with replacement when buffer small
- **Portfolio strategy**: train/test split with zero test_extra, alerts with mismatched timestamps, all alerts below threshold → zero weights
- **Training loop**: training with 1 transition (should work but poorly), training with all-same transitions (degenerate), gradient magnitudes stay finite through 100 updates
- **Reward function with SAC**: reward includes both extrinsic + intrinsic, verify combined matches sum, verify lambda decay over steps
- **Serialization**: save/load with different config (should warn/fail), corrupt bytes (should raise cleanly)

---

### Phase 21: Integration & Validation

#### Step 21.9: Wire learned weights into entity scorer

**Files:** `agent/fusion/entity_scorer.py` (modify)

After 21a training produces learned weights, wire them into the EntityAnomalyScorer's SurpriseExtractor. Add an optional `policy_config` parameter to `EntityAnomalyScorer.__init__()` that, if provided and containing learned weights, passes them to SurpriseExtractor.

**Tests:** EntityAnomalyScorer with default config produces same results as before. With learned weights, composite_surprise values change accordingly. Existing Phase 20 tests still pass.

---

#### Step 21.10: Walk-forward validation of full pipeline

**Files:** `tests/test_rl_validation.py`

Integration test that validates the full end-to-end flow:

1. Seed PipelineStore with synthetic but realistic entity alerts, beliefs, and market returns (create a synthetic data generator)
2. Train surprise weight learner on the data
3. Evaluate `WeightedSurpriseStrategy` via `WalkForward.run()`
4. Train SAC policy on the data
5. Evaluate `SACPortfolioStrategy` via `WalkForward.run()`
6. Assert: both strategies produce finite Sharpe ratios
7. Assert: SAC strategy outperforms random weights (with synthetic signal-predictive data)
8. Assert: on pure-noise data, neither strategy has significantly positive Sharpe

This is the "does it actually predict anything" validation gate.

---

#### Step 21.11: Update learning module exports + documentation

**Files:** `agent/learning/__init__.py` (modify)

Export all new public classes:
- `PolicyConfig`, `SACConfig`, `RewardConfig`, `WeightLearnerConfig`
- `AssetMapper`, `StateAssembler`, `RewardFunction`
- `SurpriseWeightLearner`, `SACTrainer`, `AlphaScheduler`
- `SACPortfolioStrategy`, `WeightedSurpriseStrategy`
- `ReplayBuffer`

---

## Edge Cases (Cross-Cutting)

| Category | Edge Case | Expected Behavior |
|----------|-----------|-------------------|
| Empty data | No entity alerts in historical window | Weight learner raises `InsufficientDataError`; SAC returns zero-weight portfolio |
| Single entity | Only 1 tradeable entity | Valid degenerate case — single-asset sizing |
| Non-stationary | Market regime shift mid-training | Symlog normalization + short walk-forward windows mitigate; entropy regularization prevents collapse |
| NaN/Inf | NaN surprise values | Raise on input validation; never propagate NaN through policy |
| Position limits | SAC outputs exceeding leverage limit | Hard clamp after softmax; log warning |
| Confidence zero | BeliefState with confidence=0 | Kelly cap = 0 → position = 0 regardless of signal |
| Negative Sharpe | Backtest produces negative Sharpe | Valid result; weight learner reports but doesn't fail |
| Database | PipelineStore locked during concurrent access | Use WAL mode (already configured); retry on SQLITE_BUSY |

## Testing Plan

### Unit Tests (per-step, mandatory)
Every step has its own test file. All must pass before proceeding.

### Integration Tests
- `test_rl_validation.py` — full pipeline on synthetic data
- `test_rl_training_dag.py` — DAG integration with mock store

### Validation Criteria (Before Phase 21 is "done")
1. Surprise weight learner converges on synthetic data with known signal
2. SAC policy learns to size positions on synthetic data with clear signal
3. Walk-forward backtest completes without errors on both strategies
4. Learned weights differ meaningfully from the default `[0.30, 0.15, 0.25, 0.20, 0.10]`
5. On pure-noise data, neither strategy shows significantly positive Sharpe (sanity check against overfitting)
6. All edge case test suites pass

### Performance Criteria (Aspirational, Not Blocking)
- Weight learning: < 60 seconds on 2 years of weekly data
- SAC training: < 5 minutes for 10,000 update steps
- Strategy evaluation via walk-forward: < 30 seconds per fold

## Dependencies

### Existing (no new packages)
- `torch` — neural networks, autograd for differentiable backtest
- `numpy` — array ops, scoring
- `sqlite3` — PipelineStore (via existing store.py)

### No New Dependencies Required
SAC, replay buffer, symlog, and reward functions are implemented from scratch with raw PyTorch. No gymnasium, no stable-baselines3.

---

## Related

- [[rl_policy]] — Research doc for Phase 21
- [[signal_fusion_spec]] — Phase 20 spec (produces EntityAlerts consumed here)
- [[signal_fusion]] — Phase 20 research (surprise signals, convergence)
- [[world_model_bridge_spec]] — Phase 19 spec (produces BeliefState consumed here)
- [[rl_layer_spec]] — Phase 4b spec (Thompson Sampling bandit — orthogonal, not replaced)
- [[learning_stack_spec]] — Prior learning architecture spec
- [[backtest_performance_spec]] — Walk-forward backtest infrastructure
