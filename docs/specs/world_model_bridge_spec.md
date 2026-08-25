---
title: "Spec: GNN ↔ World Model Bridge"
tags:
  - doc/spec
  - phase/19
  - topic/world-model
  - topic/surveillance
  - layer/world-model
  - layer/feature-engineering
---

# Spec: GNN ↔ World Model Bridge

## Goal

Connect the GNN entity graph to the Bayesian world model and produce the system's first real-data predictions. The end state: a scheduled pipeline that trains the GNN, extracts entity-level features, updates an expanded world model, produces posterior beliefs, and validates them against market outcomes.

## Files Affected

### New files
| File | Purpose |
|------|---------|
| `agent/features/gnn_builder.py` | GNNFeatureBuilder — embeddings → 11 EngineeredFeatures |
| `agent/pipeline/dags/gnn_inference.py` | GNN training + inference pipeline DAG step |
| `agent/quant/regime_strategy.py` | Strategy subclass using world model beliefs |
| `tests/test_gnn_feature_builder.py` | GNNFeatureBuilder edge case tests |
| `tests/test_expanded_dag.py` | Expanded DAG validation tests |
| `tests/test_gnn_inference_dag.py` | Pipeline DAG step tests |
| `tests/test_regime_strategy.py` | Strategy + backtest integration tests |
| `tests/test_e2e_bridge.py` | Full end-to-end bridge integration tests |

### Modified files
| File | Change |
|------|--------|
| `agent/models/initial_graph.py` | Add 11 observed nodes, ~8 causal edges, CPDs |
| `agent/pipeline/dags/feature_generation.py` | Add GNNFeatureBuilder to DEFAULT_BUILDERS |
| `agent/pipeline/dags/world_model_update.py` | Expand _FEATURE_NAMES, _FEATURE_TO_OBS_INDEX, Kalman dims |
| `agent/features/builders.py` | Add GNNFeatureBuilder import to __all__ if needed |

## Implementation Steps

### Phase 19a: GNN Training on Real Data (3 steps)

**19a.1: Add GNN inference entry point to trainer**
Create `Trainer.infer()` method that returns embeddings without training:
```python
def infer(self) -> tuple[dict[str, torch.Tensor], IDMap]:
    """Run forward pass on full graph, return (embeddings, id_map)."""
```
This reuses `build_model()` + `GraphBuilder.build()` + `forward()` but skips the training loop. If model has not been trained yet, it returns random embeddings (useful for testing the downstream pipeline).

**19a.2: Add real-data training smoke test**
Test that `Trainer(store).train()` completes without error on a store populated by ≥3 L2 tools with entity observations and links (use `SyntheticGraphGenerator` as fallback if the real store is empty, but document the difference). Verify loss dict has expected keys and losses decrease.

**19a.3: Add model persistence (save/load)**
```python
def save_model(self, path: str) -> None
@classmethod
def load_model(cls, path: str, store: PipelineStore) -> "Trainer"
```
Uses `torch.save` / `torch.load` for model state_dict. Tests: round-trip save/load produces identical embeddings.

### Phase 19b: GNN → Feature Bridge (4 steps)

**19b.1: Implement GNNFeatureBuilder**
New file: `agent/features/gnn_builder.py`

```python
class GNNFeatureBuilder(FeatureBuilder):
    """Converts GNN entity embeddings into EngineeredFeature records.
    
    Produces 11 features:
    - gnn.{type}_anomaly.spot  (5 types × 1) — centroid deviation z-score
    - gnn.{type}_activity.spot (5 types × 1) — mean embedding norm
    - gnn.cross_entity.spot    (1)            — mean pairwise type cosine similarity
    """
    
    def __init__(
        self,
        model_path: str | None = None,
        min_entities_per_type: int = 2,
        train_if_missing: bool = True,
        trainer_config: TrainerConfig | None = None,
    ) -> None
    
    def build(self, store: PipelineStore, as_of: float) -> list[EngineeredFeature]:
        """
        1. Load or train GNN model
        2. Build graph from store (until=as_of for point-in-time safety)
        3. Forward pass → embeddings
        4. For each entity type with ≥ min_entities:
           a. Compute centroid μ_τ
           b. Compute anomaly = mean(||h_i - μ_τ||) z-scored
           c. Compute activity = mean(||h_i||) z-scored
        5. Compute cross_entity = mean pairwise cosine of type centroids
        6. Return list of EngineeredFeature with quality based on entity count
        """
```

Quality scoring rule:
- `quality = min(1.0, entity_count / 10)` — full quality at ≥10 entities
- Types with <2 entities: `value=None, quality=0.0, missing_reason="insufficient_entities"`

**19b.2: Implement z-score normalization with rolling history**
The z-score needs a baseline. Options:
- (A) Rolling window over stored features (query previous N feature values from store)
- (B) In-memory running mean/std (resets on restart)

Decision: **(A) Query-based rolling z-score.** The builder queries the last 30 `gnn.{type}_anomaly.spot` values from the store, computes mean/std from those, and z-scores the current value. If <3 historical values exist, use raw value with `quality *= 0.5`.

**19b.3: Write GNNFeatureBuilder edge case tests**
Test file: `tests/test_gnn_feature_builder.py`
Coverage:
- Empty store → 11 features all with value=None, quality=0.0
- Single entity type → that type's features have value=None
- Normal case → 11 features with values, quality > 0
- Entity type with 1 entity → quality=0, missing_reason
- All types populated → quality=1.0 for all
- Point-in-time: as_of in the past excludes future entities
- Model path doesn't exist + train_if_missing=True → trains and succeeds
- Model path doesn't exist + train_if_missing=False → raises or returns NaN
- Cross-entity with only 1 type → value=0.0 (degenerate)
- Feature validation: all returned features pass `validate_feature()`
- Feature names follow naming convention

**19b.4: Register GNNFeatureBuilder in DEFAULT_BUILDERS**
Add to `agent/pipeline/dags/feature_generation.py`:
```python
DEFAULT_BUILDERS: list[FeatureBuilder] = [
    ConvergenceFeatureBuilder(),
    MacroStateFeatureBuilder(),
    GNNFeatureBuilder(),  # NEW
]
```

### Phase 19c: Expanded World Model DAG (3 steps)

**19c.1: Add 11 new observed nodes to initial_graph.py**

New NodeSpecs (all cardinality=3, states=("low", "normal", "high")):
```
obs.person_anomaly   — feature: gnn.person_anomaly.spot,    domain: entity
obs.company_anomaly  — feature: gnn.company_anomaly.spot,   domain: entity
obs.wallet_anomaly   — feature: gnn.wallet_anomaly.spot,    domain: entity
obs.country_anomaly  — feature: gnn.country_anomaly.spot,   domain: entity
obs.vessel_anomaly   — feature: gnn.vessel_anomaly.spot,    domain: entity
obs.person_activity  — feature: gnn.person_activity.spot,   domain: entity
obs.company_activity — feature: gnn.company_activity.spot,  domain: entity
obs.wallet_activity  — feature: gnn.wallet_activity.spot,   domain: entity
obs.country_activity — feature: gnn.country_activity.spot,  domain: entity
obs.vessel_activity  — feature: gnn.vessel_activity.spot,   domain: entity
obs.cross_entity     — feature: gnn.cross_entity.spot,      domain: entity
```

Bin edges for all: `(-inf, -1.0, 1.0, inf)` — mapping to "low / normal / high" via z-score.

New causal edges (8):
```
regime.stress → obs.{person,company,wallet,country,vessel}_anomaly  (5)
latent.risk_appetite → obs.{person,company,wallet}_activity         (3)
```
- **Not** connecting activity to `regime.macro` directly — entity activity is a second-order effect of risk appetite, which is itself driven by regime.
- Cross-entity not connected to regime nodes initially — it's observational evidence that informs regime but is not caused by a single regime variable. It improves inference by providing "everything is correlated" or "normal dispersion" evidence without strong prior causal direction.

CPDs: weakly informative (near-uniform with slight center bias), matching the pattern in existing initial_graph.py.

**19c.2: Expand Kalman filter in world_model_update.py**

Update constants:
```python
_OBS_DIM = 17  # was 6
_FEATURE_TO_OBS_INDEX = {
    # existing 6
    "macro.rate_momentum.30d": 0,
    "macro.yield_curve_slope.spot": 1,
    "macro.liquidity_pressure.30d": 2,
    "convergence.stress_breadth.7d": 3,
    "convergence.stress_intensity.7d": 4,
    "convergence.regime_persistence.7d": 5,
    # new 11
    "gnn.person_anomaly.spot": 6,
    "gnn.company_anomaly.spot": 7,
    "gnn.wallet_anomaly.spot": 8,
    "gnn.country_anomaly.spot": 9,
    "gnn.vessel_anomaly.spot": 10,
    "gnn.person_activity.spot": 11,
    "gnn.company_activity.spot": 12,
    "gnn.wallet_activity.spot": 13,
    "gnn.country_activity.spot": 14,
    "gnn.vessel_activity.spot": 15,
    "gnn.cross_entity.spot": 16,
}
```

H matrix expansion: 17×3 (was 6×3). New rows map to:
- anomaly features → stress_level (column 2): H[6:11, 2] = 0.5
- activity features → macro_momentum (column 1): H[11:16, 1] = 0.3
- cross_entity → liquidity_state (column 0): H[16, 0] = 0.4

R matrix: `np.diag([0.1]*6 + [0.3]*11)` — higher noise for GNN features initially (0.3 vs 0.1 for established features)

**19c.3: Write expanded DAG tests**
Test file: `tests/test_expanded_dag.py`
Coverage:
- `build_initial_graph()` returns 20 nodes (was 9)
- All 11 new nodes have valid CPDs
- DAG validates with pgmpy's `check_model()`
- All new nodes are observed type with feature_name set
- Bin edges have correct shape (cardinality + 1)
- New edges exist in the graph
- Belief propagation produces posteriors for all 20 nodes
- Kalman filter accepts 17-dim observation vector
- Missing GNN features (None values) handled gracefully

### Phase 19d: End-to-End Pipeline (3 steps)

**19d.1: Create GNN inference DAG step**
New file: `agent/pipeline/dags/gnn_inference.py`

```python
def run_gnn_inference(params: dict, upstream: dict) -> dict:
    """
    1. Open PipelineStore
    2. Check entity count — skip if < threshold
    3. Load or train GNN model (from model_path or train fresh)
    4. Save model to model_path
    5. Return {"model_path": ..., "entity_count": ..., "trained": bool}
    """

def build_gnn_inference_dag(db_path: str = ...) -> DAG:
    """Schedule: 18:30 UTC weekdays (before feature_generation at 19:00)"""
```

This step runs BEFORE feature_generation so that:
```
18:00 daily_collection → 18:30 gnn_inference → 19:00 feature_generation → 19:30 world_model_update
```

**19d.2: Wire GNNFeatureBuilder to use persisted model**
The GNNFeatureBuilder checks for a model file at a well-known path (e.g., `.tirra_pipeline/gnn_model.pt`). If the gnn_inference step ran and saved a model, the builder uses it. If not, it trains on the fly (with a warning).

**19d.3: Write pipeline DAG tests**
Test file: `tests/test_gnn_inference_dag.py`
Coverage:
- DAG builds without error
- FunctionOperator runs with synthetic store
- Empty store → skips training, returns skip status
- Model save/load round-trip
- Integration: gnn_inference → feature_generation → world_model_update sequence

### Phase 19e: Walk-Forward Backtest (3 steps)

**19e.1: Implement RegimeStrategy**
New file: `agent/quant/regime_strategy.py`

```python
class RegimeStrategy(Strategy):
    """Position sizing based on world model regime beliefs.
    
    Logic:
    - expansion / risk_on → long (weight = P(expansion))
    - contraction → neutral (weight = 0)
    - crisis / risk_off → short (weight = -P(crisis))
    
    Uses regime.macro posterior as the primary signal.
    """
    
    def generate_weights(
        self,
        train_returns: np.ndarray,
        test_length: int,
        *,
        train_extra: dict[str, Any] | None = None,
        test_extra: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """
        test_extra must contain:
            "beliefs": list[list[BeliefState]] — one belief set per test period
        """
```

**19e.2: Build backtest harness**
Add a runnable script or test that:
1. Loads SPY weekly returns from Yahoo Finance (via existing market_data tool or stored data)
2. For each week in test period:
   a. Runs the full pipeline: build graph → GNN infer → features → world model → beliefs
   b. Uses RegimeStrategy to generate position weights
3. Runs WalkForward.run() with the strategy
4. Reports Sharpe, Sortino, max drawdown, equity curve

**19e.3: Write backtest integration tests**
Test file: `tests/test_regime_strategy.py`
Coverage:
- Strategy with all-expansion beliefs → all-long weights
- Strategy with all-crisis beliefs → all-short weights
- Strategy with mixed beliefs → mixed weights
- Strategy with missing beliefs → neutral (0 weight)
- Weights bounded in [-1, 1]
- WalkForward integration with synthetic returns + beliefs

## Edge Cases

| Scenario | Expected behavior |
|----------|------------------|
| Empty entity store | GNNFeatureBuilder returns 11 features all None, quality=0 |
| Single entity type with 1 entity | That type's anomaly/activity = None |
| No entity links | GNN trains as per-type autoencoders — still produces embeddings |
| Model file corrupted | GNNFeatureBuilder.build() catches, retrains |
| All features None → world model | DAG marginalizes missing evidence; Kalman skips update |
| Very old model (stale) | Quality reduction: quality *= max(0.5, 1 - staleness/7days) |
| NaN in embeddings | Detect, set feature value=None, quality=0 |

## Testing Plan

Total expected test count: ~120 new tests across 5 files.

| Test file | Count | Coverage |
|-----------|-------|----------|
| test_gnn_feature_builder.py | ~30 | Builder logic, z-scoring, quality, edge cases |
| test_expanded_dag.py | ~25 | Node/edge/CPD validation, propagation, Kalman |
| test_gnn_inference_dag.py | ~15 | DAG step, model persistence, pipeline ordering |
| test_regime_strategy.py | ~20 | Strategy logic, weight bounds, belief handling |
| test_e2e_bridge.py | ~30 | Full pipeline integration, round-trip, regression |

## Related

- [[world_model_bridge]]
- [[entity_linking_layer]]
- [[tier1_tool_expansion]]
- [[signal_protocol_feature_engineering]]
- [[convergence_detection]]
