---
title: "Spec: Adversarial Intelligence Layer"
tags:
  - doc/spec
  - phase/22
  - topic/adversarial
  - topic/manipulation-detection
  - topic/edge-decay
  - layer/adversarial
---

# Spec: Adversarial Intelligence Layer

## Goal

Implement Layer 6 of the computation stack: adversarial intelligence that monitors signal health, detects information asymmetry, estimates crowding risk, and feeds protective flags into the RL policy. All components operate on existing data (no new external data sources required).

**Output contract:** `list[AdversarialFlag]` per evaluation cycle, consumed by the reward function, weight learner, and RL state assembler.

---

## Files Affected

### New files
| File | Purpose |
|------|---------|
| `agent/adversarial/__init__.py` | Module docstring + public exports |
| `agent/adversarial/config.py` | Frozen dataclass configs for all adversarial components |
| `agent/adversarial/flags.py` | `AdversarialFlag` dataclass (output contract) |
| `agent/adversarial/edge_decay.py` | BOCPD-based signal health monitoring |
| `agent/adversarial/vpin.py` | Volume-synchronized PIN estimator (daily resolution) |
| `agent/adversarial/crowding.py` | Crowding risk estimator from convergence clusters |
| `agent/adversarial/scanner.py` | Orchestrator: runs all detectors, produces unified flag list |
| `agent/pipeline/dags/adversarial_scan.py` | DAG task for scheduled adversarial scanning |
| `tests/test_adversarial_flags.py` | Flag dataclass tests |
| `tests/test_edge_decay.py` | Edge decay monitor tests |
| `tests/test_vpin.py` | VPIN estimator tests |
| `tests/test_crowding.py` | Crowding risk tests |
| `tests/test_adversarial_scanner.py` | End-to-end scanner tests |
| `tests/test_adversarial_edge_cases.py` | Cross-cutting edge case tests |
| `tests/test_adversarial_validation.py` | Walk-forward validation on synthetic + historical events |

### Modified files
| File | Change |
|------|--------|
| `agent/learning/policy/reward_fn.py` | Add adversarial penalty term to combined reward |
| `agent/learning/policy/state_assembler.py` | Add adversarial summary features to state tensor |
| `agent/pipeline/dags/__init__.py` | Register adversarial_scan DAG (10th DAG) |

---

## Implementation Steps

### Phase 22a: Core Adversarial Infrastructure

#### Step 22a.1: Config + Flag protocol

**Files:** `agent/adversarial/__init__.py`, `agent/adversarial/config.py`, `agent/adversarial/flags.py`

Create the module with frozen dataclass configs and the `AdversarialFlag` output contract.

```python
@dataclass(frozen=True)
class EdgeDecayConfig:
    rolling_window: int = 52          # 1 year weekly
    bocpd_hazard: float = 1/100       # prior changepoint hazard
    decay_threshold: float = 0.5      # P(changepoint) > threshold → flagged
    min_history: int = 52             # minimum observations before monitoring

@dataclass(frozen=True)
class VPINConfig:
    n_buckets: int = 50               # volume bucket lookback
    sigma_window: int = 20            # rolling vol window for BVC
    spike_threshold: float = 0.7      # VPIN > threshold → flagged

@dataclass(frozen=True)
class CrowdingConfig:
    cluster_size_threshold: int = 5   # minimum cluster size to flag
    correlation_threshold: float = 0.7 # intra-cluster correlation threshold
    volume_lookback: int = 20         # days for liquidity proxy

@dataclass(frozen=True)
class AdversarialConfig:
    edge_decay: EdgeDecayConfig = EdgeDecayConfig()
    vpin: VPINConfig = VPINConfig()
    crowding: CrowdingConfig = CrowdingConfig()

@dataclass(frozen=True)
class AdversarialFlag:
    entity_id: str | None             # None for market-wide
    flag_type: str                    # "edge_decay" | "vpin_spike" | "crowding_risk"
    severity: float                   # [0.0, 1.0]
    confidence: float                 # [0.0, 1.0]
    signal_name: str | None           # which signal is affected
    evidence: dict                    # supporting metrics
    timestamp: float                  # unix epoch
```

**Tests:** `tests/test_adversarial_flags.py`
- Flag creation, immutability, field validation
- Config defaults, overrides, frozen enforcement
- Severity/confidence bounds

#### Step 22a.2: Edge Decay Monitor

**Files:** `agent/adversarial/edge_decay.py`, `tests/test_edge_decay.py`

**Math:**

Given signal $i$'s per-period return contribution $r_{i,t}$, compute the rolling annualized Sharpe:

$$S_{i,t} = \frac{\bar{r}_{i,[t-w,t]}}{\sigma_{r_{i,[t-w,t]}}} \cdot \sqrt{P}$$

where $P$ = periods per year (52 for weekly). Apply BOCPD to $\{S_{i,t}\}_{t=w}^{T}$.

**Class: `EdgeDecayMonitor`**

```python
class EdgeDecayMonitor:
    def __init__(self, config: EdgeDecayConfig | None = None): ...

    def update(self, signal_name: str, returns: np.ndarray) -> list[AdversarialFlag]:
        """Process a signal's return series, return any decay flags."""

    def get_decay_scores(self) -> dict[str, float]:
        """Return {signal_name: P(changepoint)} for all monitored signals."""

    def get_rolling_sharpes(self) -> dict[str, np.ndarray]:
        """Return rolling Sharpe series for diagnostics."""
```

**Mathematical proof obligations:**
1. Rolling Sharpe is finite when σ > 0 (guaranteed by adding ε floor).
2. BOCPD posterior is a valid probability (sums to 1) — guaranteed by the existing BOCPD implementation.
3. Decay score is monotonically related to true edge loss — BOCPD posterior on mean shift is a principled Bayesian measure.

**Tests:**
- Constant-Sharpe signal → decay score ≈ 0
- Half-life decay signal (Sharpe halves at known point) → decay score > threshold after changepoint
- Regime switch (not decay) → temporary spike, not permanent flag
- Minimum history enforcement
- NaN/Inf handling in return series

#### Step 22a.3: VPIN Estimator

**Files:** `agent/adversarial/vpin.py`, `tests/test_vpin.py`

**Math:** Bulk Volume Classification at daily resolution.

Given daily data $(r_t, V_t, \sigma_t)$:

$$V_{buy,t} = V_t \cdot \Phi\left(\frac{r_t}{\sigma_t}\right), \quad V_{sell,t} = V_t - V_{buy,t}$$

$$\text{OI}_t = |V_{sell,t} - V_{buy,t}|$$

$$\text{VPIN}_t = \frac{1}{n} \sum_{\tau=t-n+1}^{t} \frac{\text{OI}_\tau}{\bar{V}}$$

where $\bar{V}$ is the mean daily volume over the window, and $\Phi$ is the standard normal CDF.

**Class: `VPINEstimator`**

```python
class VPINEstimator:
    def __init__(self, config: VPINConfig | None = None): ...

    def compute(
        self,
        returns: np.ndarray,
        volumes: np.ndarray,
    ) -> np.ndarray:
        """Compute VPIN time series from daily returns and volumes."""

    def flag_spikes(
        self,
        vpin_series: np.ndarray,
        entity_id: str | None = None,
        timestamp: float | None = None,
    ) -> list[AdversarialFlag]:
        """Return flags for VPIN values above threshold."""
```

**Mathematical proof obligations:**
1. VPIN ∈ [0, 1]: OI ≤ V per bucket, so OI/V̄ ≤ max(V)/V̄. With normalized volumes, VPIN is bounded.
2. When buy=sell (symmetric flow), VPIN → 0 (no informed trading).
3. When all flow is one-sided, VPIN → 1 (maximum informed trading).
4. BVC classification is consistent: Φ(r/σ) → 1 as r → +∞ (all buys), → 0 as r → -∞ (all sells).

**Tests:**
- Symmetric returns → VPIN ≈ 0.5 (random walk baseline)
- One-sided positive returns → VPIN approaches 1
- Zero volume handling (division by zero guard)
- Known values: manual computation verification
- Boundary: single-period data, minimum window enforcement

#### Step 22a.4: Crowding Risk Estimator

**Files:** `agent/adversarial/crowding.py`, `tests/test_crowding.py`

**Math:**

For convergence cluster $C$ with entities $\{e_1, \ldots, e_k\}$:

$$\text{crowd}(C) = \frac{|C|}{\bar{|C|}} \cdot \rho_{\text{intra}}(C)$$

where $\rho_{\text{intra}}$ is the mean pairwise correlation of surprise vectors within the cluster, and $\bar{|C|}$ is the historical mean cluster size.

Unwind risk per entity:

$$\text{unwind}(e, C) = \text{crowd}(C) \cdot w_e \cdot \frac{1}{\text{liq}_e + \epsilon}$$

where $w_e$ is the current position weight and $\text{liq}_e$ is the rolling average volume proxy.

**Class: `CrowdingEstimator`**

```python
class CrowdingEstimator:
    def __init__(self, config: CrowdingConfig | None = None): ...

    def assess(
        self,
        clusters: list[ConvergenceCluster],
        position_weights: dict[str, float],
        volume_history: dict[str, np.ndarray],
    ) -> list[AdversarialFlag]:
        """Assess crowding risk from convergence clusters."""

    def cluster_crowding_score(
        self,
        cluster: ConvergenceCluster,
    ) -> float:
        """Compute crowding score for a single cluster."""
```

**Mathematical proof obligations:**
1. Crowding score ≥ 0 (cluster size and correlation are non-negative).
2. Isolated entities get crowd score = 0 (cluster size = 1 < threshold).
3. Unwind risk is monotone in position size and inverse in liquidity.

**Tests:**
- Isolated entities → no crowding flags
- Large dense cluster → high crowding score
- High position in low liquidity → high unwind risk
- Empty cluster list → no flags
- All zero positions → no unwind risk regardless of cluster

### Phase 22b: Integration

#### Step 22b.1: Adversarial Scanner (Orchestrator)

**Files:** `agent/adversarial/scanner.py`, `tests/test_adversarial_scanner.py`

Orchestrates all three detectors. Consumable by the pipeline DAG.

```python
class AdversarialScanner:
    def __init__(self, config: AdversarialConfig | None = None): ...

    def scan(
        self,
        signal_returns: dict[str, np.ndarray],      # per-signal return series
        market_returns: np.ndarray,                   # aggregate return series
        market_volumes: np.ndarray,                   # aggregate volume series
        clusters: list[ConvergenceCluster],           # current convergence clusters
        position_weights: dict[str, float],           # current portfolio weights
        volume_history: dict[str, np.ndarray],        # per-entity volume history
        timestamp: float | None = None,               # evaluation time
    ) -> list[AdversarialFlag]:
        """Run all adversarial detectors and return unified flag list."""
```

**Tests:**
- Empty inputs → empty flag list
- Each detector's flags included in combined output
- Flag deduplication (if same entity flagged by multiple detectors)
- Timestamp propagation

#### Step 22b.2: Reward Function Integration

**Files:** `agent/learning/policy/reward_fn.py` (modify)

Add adversarial penalty to `RewardFunction.combined()`:

$$R_{\text{total}} = R_{\text{extrinsic}} + \lambda(t) \cdot R_{\text{intrinsic}} - \beta \cdot \sum_f \text{sev}_f \cdot \text{conf}_f$$

where the sum runs over active adversarial flags $f$ affecting current positions, and $\beta$ is a configurable penalty scale.

**Changes:**
- Add `adversarial_penalty(flags: list[AdversarialFlag]) -> float` method
- Modify `combined()` to accept optional `adversarial_flags` parameter
- Keep backward compatibility: if no flags provided, penalty = 0

**Tests:**
- No flags → zero penalty (backward compatible)
- High-severity flags → proportional penalty
- Existing reward function tests still pass

#### Step 22b.3: State Assembler Integration

**Files:** `agent/learning/policy/state_assembler.py` (modify)

Add adversarial summary features to the state tensor:
- Mean edge decay score across active signals
- Current VPIN level
- Max crowding risk score
- Number of active adversarial flags

This adds 4 features to the state vector.

**Changes:**
- Add optional `adversarial_flags: list[AdversarialFlag] | None = None` parameter to `assemble()`
- Append 4-dim adversarial summary block after entity count
- Update `state_dim` property accordingly
- Backward compatible: if no flags, the 4 features are zero

**Tests:**
- Without flags → same state_dim + 4, adversarial block is zeros
- With flags → adversarial features populated
- Existing state assembler tests still pass

#### Step 22b.4: DAG Integration

**Files:** `agent/pipeline/dags/adversarial_scan.py`, `agent/pipeline/dags/__init__.py` (modify)

Create the adversarial scan DAG task and wire it as the 10th DAG.

Schedule: runs after the signal fusion DAG completes, before the RL training DAG.

**Tests:**
- DAG builds without error
- DAG registered in __init__.py
- Mock execution completes

### Phase 22c: Validation

#### Step 22c.1: Edge Case Test Suite

**Files:** `tests/test_adversarial_edge_cases.py`

Cross-cutting edge cases:
- All signals have constant returns (no variance) → Sharpe undefined, handled gracefully
- Zero volume data → VPIN computation doesn't divide by zero
- Single-entity portfolio → crowding score = 0
- All signals decaying simultaneously → flags produced for each
- Adversarial flags with entity_id=None (market-wide) → handled in reward and state assembler
- Empty convergence cluster list
- NaN/Inf in input data → ValueError raised at boundary

#### Step 22c.2: Walk-Forward Validation

**Files:** `tests/test_adversarial_validation.py`

**Synthetic data tests:**
1. Generate signal with planted edge decay (Sharpe drops from 2.0 to 0.0 at known changepoint). Assert EdgeDecayMonitor detects it within k periods.
2. Generate volume data with planted informed-trading event (one-sided flow burst). Assert VPIN spikes above threshold.
3. Generate large dense cluster with high correlation. Assert CrowdingEstimator flags it.
4. Generate normal (non-manipulated) data. Assert no adversarial flags above threshold (false positive control).
5. Full scanner integration: run scanner on synthetic pipeline data end-to-end.

---

## Edge Cases

| Category | Edge Case | Expected Behavior |
|----------|-----------|-------------------|
| Short history | Signal has < min_history observations | EdgeDecayMonitor skips signal, returns no flag |
| Zero variance | Signal returns are constant | Rolling Sharpe = 0 (ε floor), no decay flag |
| Zero volume | Volume data is all zeros | VPINEstimator raises ValueError or returns NaN-free series |
| Single entity | Portfolio has 1 entity | Crowding score = 0 (cluster size 1 < threshold) |
| All decaying | Every signal shows decay simultaneously | All flagged independently — cascade scenario |
| No clusters | Empty convergence cluster list | CrowdingEstimator returns empty flag list |
| NaN inputs | NaN in returns or volumes | Raise ValueError at input boundary |
| No positions | Empty position_weights dict | Unwind risk = 0 for all entities |
| Backward compat | No adversarial flags passed to reward/state | Zero penalty, zero adversarial state features |

---

## Testing Plan

### Unit Tests (per-step, mandatory)
- `test_adversarial_flags.py`: Flag + config dataclass tests
- `test_edge_decay.py`: EdgeDecayMonitor unit tests
- `test_vpin.py`: VPINEstimator unit tests
- `test_crowding.py`: CrowdingEstimator unit tests
- `test_adversarial_scanner.py`: Scanner orchestration tests
- `test_adversarial_edge_cases.py`: Cross-cutting edge cases

### Integration Tests
- `test_adversarial_validation.py`: Synthetic data + detection accuracy
- Existing Phase 21 tests still pass (backward compatibility of reward_fn and state_assembler)

### Validation Criteria (Before Phase 22 is "done")
1. EdgeDecayMonitor detects planted changepoint within k periods on synthetic data
2. VPINEstimator produces correct VPIN on hand-computed examples
3. CrowdingEstimator correctly flags dense clusters and ignores sparse ones
4. Adversarial penalty integrates into reward function without changing existing test outcomes
5. State assembler adds adversarial features without breaking existing tests
6. Scanner produces appropriate flags on synthetic manipulation scenarios and no flags on clean data
7. All edge case tests pass

### Performance Criteria (Aspirational, Not Blocking)
- Edge decay scan: < 1 second per signal for 5 years of weekly data
- VPIN computation: < 100ms for 2 years of daily data
- Full adversarial scan: < 5 seconds for typical portfolio

---

## Dependencies

### Existing (no new packages)
- `numpy` — array operations
- `scipy.stats` — `norm.cdf` for BVC classification in VPIN
- BOCPD from `agent/quant/changepoint.py` — already implemented
- ConvergenceCluster from `agent/fusion/convergence.py` — already implemented
- Scoring utilities from `agent/quant/scoring.py` — already implemented

### No New Dependencies Required
All adversarial components are implemented with existing libraries.

---

## Related

- [[adversarial]] — Research doc
- [[rl_policy_spec]] — Phase 21 spec (reward function + state assembler integration points)
- [[signal_fusion_spec]] — Phase 20 spec (convergence clusters consumed here)
- [[backtest_performance_spec]] — Walk-forward infrastructure
- [[quant_training_ground]] — Master phase tracker
