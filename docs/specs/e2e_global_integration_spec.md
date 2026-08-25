---
title: "Spec: End-to-End Global Multi-Asset Integration"
tags:
  - doc/spec
  - phase/24
  - topic/e2e-integration
  - topic/multi-asset
  - topic/instruments
  - layer/surveillance
  - layer/learning
  - layer/feature-engineering
---

# Spec: End-to-End Global Multi-Asset Integration

## Goal

Wire all 23 completed phases into a single end-to-end pipeline that:
1. Ingests daily prices for ~90 global instruments.
2. Registers instruments as first-class GNN entity nodes.
3. Trains the HetTGN with instruments included; extracts per-instrument surprise.
4. Allocates a multi-asset portfolio via SAC.
5. Backtests the allocation via walk-forward with attribution.
6. Launches paper trading with daily inference.

## Files Affected

### New Files

| File | Purpose |
|------|---------|
| `agent/tools/instrument_universe.py` | InstrumentDef dataclass + instrument list + daily ingest function |
| `agent/pipeline/dags/inference.py` | Inference DAG (load→GNN→SAC→emit) |
| `tests/test_instrument_universe.py` | Tests for 24a |
| `tests/test_multi_asset_backtest.py` | Tests for 24c |
| `tests/test_inference_dag.py` | Tests for 24d |
| `tests/test_e2e_integration.py` | Smoke test for full pipeline |
| `tests/test_walkforward_multi.py` | Tests for 24e |

### Modified Files

| File | Change |
|------|--------|
| `agent/models/gnn/graph_builder.py` | +1 entity type, +3 obs types, ENRICHMENT_DIM 30→33 |
| `agent/learning/policy/portfolio_strategy.py` | New `MultiAssetSACStrategy` class |
| `agent/learning/policy/state_assembler.py` | New `InstrumentStateAssembler` class |
| `agent/quant/backtest.py` | New `MultiAssetStrategy` ABC + `MultiAssetWalkForward` class |
| `agent/pipeline/dags/__init__.py` | Register inference DAG |
| `agent/pipeline/dags/daily_collection.py` | Add `fetch_instruments` node |
| `agent/pipeline/store.py` | New tables: `portfolio_weights`, `paper_trade_pnl` |

### Files That Need NO Changes (already asset-agnostic)

| File | Why |
|------|-----|
| `agent/models/gnn/het_tgn.py` | Cross-type attention is generic over entity types |
| `agent/fusion/surprise.py` | Per-entity surprise works for any entity type |
| `agent/fusion/entity_scorer.py` | Type-agnostic scoring pipeline |
| `agent/learning/policy/sac.py` | action_dim is a constructor arg — already multi-dimensional |
| `agent/learning/policy/reward_fn.py` | Takes scalar portfolio_return — asset-agnostic |
| `agent/models/world_model.py` | Generic state beliefs |
| `agent/adversarial/` | All detectors are entity-generic |

---

## Implementation Steps

### Sub-phase 24a: Instrument Universe + Daily Price Ingest

**24a.1: Create `agent/tools/instrument_universe.py`**

Create the module with:
- `InstrumentDef` frozen dataclass: `ticker: str`, `name: str`, `asset_class: str`, `region: str`, `is_tradeable: bool`.
- `asset_class` is one of: "commodity_future", "fx", "equity_index", "equity_etf", "sector_etf", "fixed_income", "vol", "crypto".
- `region` is one of: "US", "Europe", "Asia", "LatAm", "Pacific", "Global", "EM".
- `INSTRUMENTS: tuple[InstrumentDef, ...]` — all 90 instruments. `^VIX` has `is_tradeable=False`.
- Helper functions:
  - `tradeable_instruments() -> list[InstrumentDef]` — filter `is_tradeable=True`.
  - `instruments_by_class(cls: str) -> list[InstrumentDef]`.
  - `ticker_to_instrument() -> dict[str, InstrumentDef]` — lookup by ticker.

No imports of pipeline or GNN modules. This is a pure data module.

**24a.2: Add `ingest_daily_prices()` function to `instrument_universe.py`**

Signature:
```python
def ingest_daily_prices(
    store: PipelineStore,
    as_of: date | None = None,
    lookback_days: int = 5,
) -> dict[str, Any]:
```

Logic:
1. For each tradeable instrument, call `yf.download(ticker, period=f'{lookback_days}d', interval='1d')`.
2. Batch the download: `yf.download(all_tickers, period=..., group_by='ticker')` for efficiency.
3. For each instrument with data:
   a. Register entity: `store.register_entity(entity_id, entity_type="instrument", metadata={ticker, name, asset_class, region})`.
   b. Entity ID: `entity_id_from_key("instrument", ticker)`.
   c. Compute log return: `ln(close_today / close_yesterday)`.
   d. Compute 20d realized volatility: `std(log_returns[-20:]) * sqrt(252)` if enough history, else `NaN`.
   e. Store 3 observations via `store_entity_observation()`:
      - `obs_type="instrument_return"`, `value={"log_return": float, "close": float}`.
      - `obs_type="instrument_volume"`, `value={"volume": float, "avg_volume_20d": float}`.
      - `obs_type="instrument_volatility"`, `value={"realized_vol_20d": float, "intraday_range": float}`.
4. Return summary: `{"instruments_fetched": int, "instruments_failed": list[str], "observations_stored": int}`.

Error handling: per-ticker try/except. Log failures but continue. If >50% fail, raise an exception (likely API issue).

**24a.3: Extend graph builder**

In `agent/models/gnn/graph_builder.py`:
- Append `"instrument"` to `ENTITY_TYPES` list (index 9, after "wallet").
- Append `"instrument_return"`, `"instrument_volume"`, `"instrument_volatility"` to `OBSERVATION_TYPES` list. New indices: 21, 22, 23.
- `ENRICHMENT_DIM` changes from 30 to 33 (the obs_type_dist component grows from 21 to 24 features).
- `BASE_FEAT_DIM` changes from `len(ENTITY_TYPES) + 3 = 12` to `13` (one-hot grows by 1).

These are constant changes — no logic changes in the builder itself.

**24a.4: Add instrument ingest to daily collection DAG**

In `agent/pipeline/dags/daily_collection.py`:
- Add a new node `fetch_instruments` using a `FunctionOperator` that calls `ingest_daily_prices(store)`.
- Node has no dependencies (parallel with other fetch nodes).
- Timeout: 300s (90 tickers via yfinance may be slow).
- Retries: 1 (yfinance is flaky, but double-retry wastes time).

**24a.5: Write `tests/test_instrument_universe.py`**

Test cases:
- `InstrumentDef` creation and field access.
- `INSTRUMENTS` has exactly 90 entries; 89 are tradeable.
- `tradeable_instruments()` excludes ^VIX.
- `instruments_by_class()` returns correct subsets (spot-check commodity_future=20, fx=15).
- `ticker_to_instrument()` lookup works for all tickers.
- `ingest_daily_prices()` with mocked yfinance + mocked PipelineStore:
  - Correct entity registrations (entity_type="instrument").
  - Correct observation types stored.
  - Log return computation: given close_yesterday=100, close_today=105, return ≈ 0.04879.
  - Volume and volatility observations stored.
  - Failure handling: one ticker fails, others succeed, summary reflects failure.
  - >50% failure raises exception.
- Graph builder integration: after ingest, `GraphBuilder.build()` produces HeteroData with "instrument" nodes having correct feature dimensions.

Edge cases:
- Ticker returns NaN close → skip, log warning.
- Ticker returns 0 volume → store 0, don't skip.
- Only 1 day of history → log_return = NaN → skip return obs, still store volume.
- Empty download (weekend/holiday) → no observations stored, no error.

**24a.6: Update graph builder tests**

In `tests/test_graph_builder_expanded.py`:
- Update assertion on obs type count: 21 → 24.
- Update assertion on entity type count: 9 → 10 (if hardcoded).
- Add test: instrument entities produce node features with correct dimensionality.

---

### Sub-phase 24b: GNN Training with Instruments

**24b.1: Historical backfill script**

Create a one-time script (not a tool) `scripts/backfill_instruments.py`:
- Downloads 2 years daily data for all 90 instruments.
- Registers entities + stores observations day-by-day into PipelineStore.
- Idempotent: skips dates already stored.
- Run manually: `python scripts/backfill_instruments.py --db pipeline.db`.

**24b.2: Train HetTGN with instruments**

Run the existing GNN training pipeline (gnn_inference DAG or manual trainer call) on the backfilled data. Verify:
- Training loss converges (decreases over epochs).
- Instrument nodes have non-zero embeddings after training.
- Cross-type attention weights: at least some (entity_type, instrument) pairs show non-trivial attention.
- Per-instrument surprise extraction: `SurpriseExtractor.extract()` returns `EntitySurprise` objects for instrument entities.

**24b.3: Write test for training convergence**

Test (may be slow, mark with `@pytest.mark.slow`):
- Build a small synthetic graph with 5 instruments + 10 entities + observations.
- Train for 50 epochs.
- Assert loss at epoch 50 < loss at epoch 1.
- Assert instrument nodes have non-zero surprise scores.

---

### Sub-phase 24b.5: GNN Attention Diagnostic

**24b.5.1: Run `run_diagnostics` with instrument entities**

After 24b training, run the existing diagnostic function from Phase 16.
Report:
- Per-instrument-class (commodity, fx, equity, etc.) attention density: mean attention weight from entity nodes to instrument nodes of that class.
- Data-starved instruments: instruments with < 5 entity-type neighbors above attention threshold.
- Ranked list of which L1 tools would most benefit from L2 upgrade to serve starved instrument classes.

**24b.5.2: Record output as Phase 25 specification input**

Write the diagnostic output to `[[phase25_gnn_diagnostic]]`. This becomes the input for Phase 25 (L2 tool expansion for global instruments).

---

### Sub-phase 24c: Multi-Asset Strategy Refactor

**24c.1: Add `MultiAssetStrategy` ABC to `agent/quant/backtest.py`**

```python
class MultiAssetStrategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def generate_weights(
        self,
        train_returns: np.ndarray,   # (train_len, N)
        test_length: int,
        instrument_names: list[str],  # length N
        *,
        train_extra: dict[str, Any] | None = None,
        test_extra: dict[str, Any] | None = None,
    ) -> np.ndarray:  # (test_length, N)
        ...
```

The existing `Strategy` ABC stays untouched (backward compat).

**24c.2: Add `MultiAssetWalkForward` to `agent/quant/backtest.py`**

New class alongside existing `WalkForward`. Differences:
- Takes `returns: np.ndarray` of shape `(T, N)`.
- Takes `instrument_names: list[str]` of length N.
- Fold P&L: `portfolio_return_t = sum(weights_t * returns_t)` (dot product across instruments).
- Per-fold metrics computed on the 1-D portfolio return series.
- Aggregate metrics include: per-instrument average weight, per-asset-class P&L attribution, concentration (max single-instrument weight over time).

New result container:
```python
@dataclass
class MultiAssetBacktestResult:
    strategy_name: str
    folds: list[MultiAssetFoldResult]
    aggregate_metrics: dict[str, Any]
    equity_curve: np.ndarray
    all_portfolio_returns: np.ndarray
    all_weights: np.ndarray  # (T_total, N)
    instrument_names: list[str]
    attribution: dict[str, float]  # asset_class → cumulative return contribution
```

**24c.3: New `InstrumentStateAssembler` in `agent/learning/policy/state_assembler.py`**

New class (existing `StateAssembler` stays untouched). Difference:
- State layout: `[instrument_surprise_block | entity_surprise_block | belief_block | market_features | adversarial]`.
- `instrument_surprise_block`: (N_instruments × 5) — fixed ordering matching `INSTRUMENTS`.
- `entity_surprise_block`: top-K entity surprises (same as current).
- Result: state_dim = N_instruments*5 + max_entities*5 + max_entities*4 + market_dim + 1 + 4.

**24c.4: New `MultiAssetSACStrategy` in `agent/learning/policy/portfolio_strategy.py`**

New class implementing `MultiAssetStrategy`:
- Constructor: `SACTrainer`, `InstrumentStateAssembler`, `instrument_tickers: list[str]`.
- `generate_weights()`: for each test timestep, assemble instrument state → SAC deterministic action → N-dimensional weight vector.
- No `AssetMapper` dependency — instruments are already the action-space dimensions.

**24c.5: New `EqualWeightStrategy` implementing `MultiAssetStrategy`**

Baseline: returns `1/N` weight for all instruments at all timesteps.

**24c.6: New `BuyAndHoldBenchmarkStrategy`**

For benchmarks (SPY, 60/40). Constructor takes target weights (e.g., `{"SPY": 0.6, "AGG": 0.4}`). Returns fixed weights at all timesteps.

**24c.7: Write `tests/test_multi_asset_backtest.py`**

Test:
- `MultiAssetWalkForward` with synthetic (T=500, N=5) returns.
- Fold splitting: correct train/test sizes.
- Portfolio return computation: dot product is correct.
- Attribution: per-instrument cumulative contribution sums to total P&L.
- `EqualWeightStrategy`: weights are 1/N everywhere.
- `BuyAndHoldBenchmarkStrategy`: weights match target.
- `MultiAssetSACStrategy` with mocked SAC: correct state assembly, correct weight output shape.
- Edge cases: single instrument (reduces to 1-D). zero-variance returns. all-NaN instrument (dropped or zero-weighted).

---

### Sub-phase 24d: Inference DAG

**24d.1: Add `portfolio_weights` and `paper_trade_pnl` tables to PipelineStore**

In `agent/pipeline/store.py`:
- `portfolio_weights` table: `(id INTEGER PRIMARY KEY, date TEXT, ticker TEXT, weight REAL, metadata TEXT, created_at TEXT)`.
- `paper_trade_pnl` table: `(id INTEGER PRIMARY KEY, date TEXT, portfolio_return REAL, benchmark_return REAL, cumulative_return REAL, metadata TEXT, created_at TEXT)`.
- CRUD methods: `store_portfolio_weights(date, weights_dict)`, `query_portfolio_weights(date)`, `store_paper_pnl(date, portfolio_return, benchmark_return)`, `query_paper_pnl(start_date, end_date)`.

**24d.2: Create `agent/pipeline/dags/inference.py`**

`build_inference_dag()` with 4 sequential nodes:
1. `load_models`: load HetTGN + SAC checkpoints from PipelineStore model_artifacts table (or file).
2. `gnn_inference`: build today's graph → forward pass → extract surprises.
3. `sac_inference`: assemble instrument state → SAC deterministic action → weight vector.
4. `emit_portfolio`: write weights to `portfolio_weights` table. Compute yesterday's P&L (from yesterday's weights × today's returns), write to `paper_trade_pnl`.

Dependencies: each node depends on the previous (strictly sequential).
Schedule: `45 19 * * 1-5` (19:45 UTC, after all upstream DAGs complete).

**24d.3: Register inference DAG in `__init__.py`**

Import `build_inference_dag` and append to the list in `get_default_dags()`.

**24d.4: Write `tests/test_inference_dag.py`**

Test:
- DAG structure: 4 nodes, correct dependency chain.
- Mocked execution: each node receives correct inputs from upstream.
- `emit_portfolio` writes correct weights to store.
- P&L computation: yesterday's weights × today's returns = correct scalar.
- No model files → graceful failure at `load_models` node.

---

### Sub-phase 24e: Walk-Forward Backtest

**24e.1: Backfill check**

Verify PipelineStore has ≥2 years of instrument price observations from 24b.1.

**24e.2: Run multi-asset walk-forward**

Configuration:
- `min_train = 252` (1 year daily).
- `test_size = 21` (1 month daily).
- `step_size = 21` (monthly folds).
- Strategies: SAC, WeightedSurprise (adapted), EqualWeight, BuyAndHold(SPY), BuyAndHold(60/40).
- Metrics per fold: Sharpe, max drawdown, turnover, concentration.
- Aggregate: annualized Sharpe, total return, max drawdown, per-asset-class attribution.

**24e.3: Attribution report**

For each strategy:
- Per-asset-class P&L contribution: commodity_future, fx, equity, fixed_income, crypto.
- Per-region P&L contribution: US, Europe, Asia, LatAm, EM.
- Top 5 instruments by P&L contribution (+/-).
- Concentration analysis: histogram of weight magnitudes.

**24e.4: Write `tests/test_walkforward_multi.py`**

Test with synthetic data that the full walk-forward executes, produces correct fold count, metrics are plausible (Sharpe is finite, drawdown ≤ 0, attribution sums to total).

---

### Sub-phase 24f: Paper Trade Launch

**24f.1: Configure inference DAG schedule**

Verify cron `45 19 * * 1-5` fires after upstream DAGs.

**24f.2: Implement alert conditions**

In `emit_portfolio` node (or a separate monitoring node):
- Drawdown > 5% from peak → log WARNING.
- Single instrument weight > 30% → log WARNING.
- Edge decay flag from adversarial layer on any held instrument → log WARNING.
- Cumulative Sharpe < -0.5 at 30 calendar days → log CRITICAL.

**24f.3: Write `tests/test_e2e_integration.py`**

Smoke test: with mocked yfinance + synthetic entity data, run the full chain:
daily_collection → gnn_inference → entity_scoring → sac_inference → emit_portfolio.
Assert: portfolio_weights table has entries, P&L is computed.

---

## Edge Cases

| Case | Handling |
|------|----------|
| yfinance returns no data for a ticker on a given day | Skip that ticker's observations, log warning |
| All tickers fail (API down) | Raise exception, skip inference DAG that day |
| Instrument has < 20 days history | realized_vol = NaN → store NaN, GNN handles via obs_stats mean_value=0 |
| Market holiday (no new prices) | ingest function detects no new data (latest date = yesterday), stores nothing, no error |
| SAC outputs NaN weights | Clip to 0, log error, emit zero-allocation portfolio |
| Walk-forward has insufficient data for a fold | Skip that fold (existing WalkForward behavior) |
| GNN has not been trained yet | load_models node fails, inference DAG aborts, no portfolio emitted |

## Testing Plan

| Sub-phase | Test file | Expected tests |
|-----------|-----------|----------------|
| 24a | `tests/test_instrument_universe.py` | ~25 |
| 24a | `tests/test_graph_builder_expanded.py` (update) | ~3 updated |
| 24b | Manual + 1 slow test | ~3 |
| 24c | `tests/test_multi_asset_backtest.py` | ~30 |
| 24d | `tests/test_inference_dag.py` | ~15 |
| 24e | `tests/test_walkforward_multi.py` | ~10 |
| 24f | `tests/test_e2e_integration.py` | ~5 |
| **Total** | | **~91** |

## Related

- [[e2e_global_integration]] — Research doc
- [[e2e_global_integration]] — Task file (to be created with same name in tasks/active/)
- [[quant_training_ground]] — Master phase tracker
- [[temporal_het_gnn]] — Phase 12 GNN architecture
- [[signal_fusion]] — Phase 20 surprise extraction
- [[rl_policy]] — Phase 21 SAC policy
- [[adversarial]] — Phase 22 adversarial layer
