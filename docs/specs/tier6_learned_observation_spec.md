---
title: "Spec: Tier 6 — Learned Feature Selection & Tool Routing"
tags:
  - doc/spec
  - phase/25
  - topic/self-improving
  - topic/feature-selection
  - topic/tool-routing
  - layer/learning
  - layer/surveillance
---

# Spec: Tier 6 — Learned Feature Selection & Tool Routing

## Goal

Implement Changes 11 and 12 from [[learned_vs_handcoded_architecture_spec]]:
- **Change 11**: Replace static feature pass-through with a differentiable, regime-conditioned feature gate that learns which feature groups (surprise, belief, market, entity count, adversarial) matter per HMM regime.
- **Change 12**: Replace the fixed daily collection schedule with a contextual Thompson Sampling bandit that learns which tools to invoke based on regime state and tool contribution to downstream signal quality.

Together these move the system from 75% → 82% learned.

## Files Affected

### Change 11 — Feature Gate
| Action | File |
|--------|------|
| **Create** | `agent/learning/policy/feature_gate.py` — `FeatureGateConfig`, `FeatureGate(nn.Module)` |
| **Modify** | `agent/learning/policy/config.py` — add `FeatureGateConfig` to `PolicyConfig` |
| **Modify** | `agent/learning/policy/state_encoder.py` — apply feature gate before entity embedding |
| **Modify** | `agent/pipeline/dags/rl_training.py` — create + train feature gate alongside encoder |
| **Create** | `tests/test_feature_gate.py` — edge case test suite |

### Change 12 — Tool Routing
| Action | File |
|--------|------|
| **Create** | `agent/learning/tool_router.py` — `ToolRoutingBandit` |
| **Modify** | `agent/pipeline/dags/daily_collection.py` — conditional tool execution |
| **Modify** | `agent/pipeline/executor.py` — support `skip` flag on nodes |
| **Create** | `tests/test_tool_router.py` — edge case test suite |

## Implementation Steps

### Change 11: Learned Feature Selection

#### 11.1: FeatureGate nn.Module

Create `agent/learning/policy/feature_gate.py`:

```python
@dataclass(frozen=True)
class FeatureGateConfig:
    n_feature_groups: int = 5          # surprise, belief, market, entity_count, adversarial
    regime_dim: int = 4                # HMM posterior dim (num regimes)
    gate_hidden_dim: int = 16          # MLP hidden layer
    gate_floor: float = 0.05           # minimum gate value (prevents total suppression)
    entropy_weight: float = 0.01       # λ for entropy regularization
    group_dims: tuple[int, ...] = (250, 200, 8, 1, 4)  # dims per group (E=50)

class FeatureGate(nn.Module):
    """Regime-conditioned soft gating over feature groups.

    forward(state_flat, regime_context) → gated_state
    gate_values(regime_context) → (K,) gate vector for diagnostics
    entropy_loss() → scalar penalty to prevent gate collapse
    """
```

**Math**: For each feature group $k$:
$$g_k = (1 - f) \cdot \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot r + b_1) + b_2)_k + f$$

where $r$ is regime context, $f$ = gate_floor, $\sigma$ = sigmoid. Gate values are in $[f, 1]$.

The gated state multiplies each feature group by its gate:
$$\tilde{x} = [\,g_0 \cdot x_0 \;;\; g_1 \cdot x_1 \;;\; \ldots \;;\; g_4 \cdot x_4\,]$$

Entropy regularization:
$$\mathcal{L}_{\text{gate}} = -\lambda \sum_k [\hat{g}_k \log \hat{g}_k + (1-\hat{g}_k) \log(1-\hat{g}_k)]$$
where $\hat{g}_k = (g_k - f)/(1 - f)$ is the normalized gate.

#### 11.2: Config integration

Add `FeatureGateConfig` to `config.py`:
- Add to `PolicyConfig` as `feature_gate: FeatureGateConfig | None = None`
- Default `None` → gate disabled (backward compatible)

#### 11.3: Wire into LearnedStateEncoder

Modify `LearnedStateEncoder.forward()`:
- Accept optional `regime_context` tensor
- If `self._feature_gate` exists: apply gate to state_flat before parsing into entity/global blocks
- The gate operates on the flat 463-dim state, multiplying each feature group by its gate value
- Gate params are added to the encoder's optimizer (trained end-to-end with SAC)

#### 11.4: Wire into rl_training.py

In `_train_sac()`:
- If `PolicyConfig.feature_gate` is set, create `FeatureGate` and attach to encoder
- Pass regime context (from world model beliefs or fixed zero vector during cold start)
- Add `gate.entropy_loss()` to the total actor loss with configured weight
- Checkpoint gate state alongside encoder

#### 11.5: Diagnostic output

`FeatureGate.gate_diagnostics(regime_context) -> dict`:
- Returns `{"group_names": [...], "gate_values": [...], "entropy": float}`
- Logged during training for interpretability

### Change 12: Learned Tool Routing

#### 12.1: ToolRoutingBandit

Create `agent/learning/tool_router.py`:

```python
class ToolRoutingBandit:
    """Contextual Thompson Sampling for tool selection.

    Each tool is an arm with Beta(α, β). Context features condition
    the selection by adjusting the threshold.

    Arms:
      - fetch_cftc, fetch_finra_scan, fetch_power_demand,
        fetch_power_fuel, fetch_gdelt, fetch_polymarket
    Always-on:
      - fetch_instruments (required for instrument universe)

    Context features:
      - regime_id: int (current HMM regime)
      - day_of_week: int (0=Mon .. 4=Fri)
      - tool_staleness: float (hours since last successful fetch per tool)
    """
```

**Core API:**
- `decide(context: ToolContext) -> dict[str, bool]`: For each optional tool, sample from Beta posterior and decide whether to run.
- `record_outcome(tool_name: str, signal_contribution: float)`: Update Beta params with observed reward.
- `save(path)` / `load(path)`: Persist bandit state.

**Tool signal contribution**: Measured as normalized count of downstream entity alerts that used this tool's data divided by total alerts in the period. Simple and cheap to compute.

#### 12.2: Integrate with daily_collection DAG

Modify `build_daily_collection_dag()`:
- Accept optional `tool_router: ToolRoutingBandit` parameter
- If router is provided, call `router.decide(context)` before DAG execution
- Mark skipped nodes in the DAG (add `enabled: bool` flag to `Node`)
- `fetch_instruments` is always enabled (hardcoded must-run)

#### 12.3: Integrate with DAGExecutor

Modify `DAGExecutor._execute_layer()`:
- Check `node.enabled` flag — skip disabled nodes with status "skipped_by_router"
- This is a 2-line change in the existing executor

#### 12.4: Record tool outcomes after DAG completion

After `daily_collection` executes:
- For each tool that ran: count entity alerts generated from that tool's data
- Call `router.record_outcome(tool_name, contribution)`
- For skipped tools: no update (the bandit learns only from observed outcomes)

#### 12.5: Persistence and cold start

- Save bandit state to `.tirra_pipeline/tool_router.json`
- Cold start: uniform prior Beta(1, 1) → all tools equally likely → effectively runs everything initially
- After 10+ observations per tool, the bandit starts making informed decisions

## Edge Cases

### Change 11
1. **All gates collapse to floor**: Entropy regularization prevents this; monitored via diagnostics.
2. **No regime context available (cold start)**: Use zero vector → gates default to ~sigmoid(bias) ≈ 0.5.
3. **State dim mismatch**: Validate `sum(group_dims) == state_dim` at init.
4. **Backward compat**: Gate disabled when `feature_gate=None` in config.
5. **NaN/Inf in regime context**: Clamp regime input to [-10, 10].

### Change 12
1. **All tools skipped**: Impossible — `fetch_instruments` is always-on; also minimum exploration rate ensures at least 1 optional tool runs.
2. **Tool consistently fails**: Failed tools get reward=0 → Beta shifts left → tool is skipped more often → natural adaptive behavior.
3. **No prior data (cold start)**: Beta(1, 1) uniform → all tools run → equivalent to current behavior.
4. **Signal contribution = 0 for all tools**: All Betas stay near uniform → still runs most tools.
5. **Tool added/removed**: New tools start with Beta(1, 1); removed tools are dropped from state.

## Testing Plan

### Change 11 Tests (`tests/test_feature_gate.py`)
1. Forward shape: single / batch / custom group dims
2. Gate values in [floor, 1.0] for random regime contexts
3. Gradient flow through gate to regime input
4. Entropy loss computation: min (all gates at extremes), max (all gates at 0.5)
5. Zero regime context → stable default gates
6. Gate floor enforcement: floor=0.0 vs floor=0.1
7. Diagnostic output format and values
8. Config validation: mismatched group_dims raises
9. Integration with LearnedStateEncoder: gated output matches manual gating
10. Save/load round-trip of gate parameters
11. Batch consistency: same regime → same gates
12. Large/small/negative regime values → no NaN

### Change 12 Tests (`tests/test_tool_router.py`)
1. Cold start: all tools enabled (uniform prior)
2. After many reward=1 updates: tool consistently selected
3. After many reward=0 updates: tool consistently skipped
4. Always-on tools never skipped
5. Context features affect decisions (different regimes → different selections)
6. Persistence: save/load preserves Beta params
7. Minimum exploration rate: even low-reward tools occasionally selected
8. Unknown tool name in record_outcome → error
9. Empty context → fallback behavior
10. Concurrent calls (thread safety of state)
11. New tool added after initialization → starts with uniform prior

## Related

- [[tier6_learned_observation]]
- [[learned_vs_handcoded_architecture_spec]]
- [[learned_architecture_impl]]
- [[tier5_differentiable_kalman]]
- [[learned_state_encoder]]
