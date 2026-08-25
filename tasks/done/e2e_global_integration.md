---
title: "Task: End-to-End Global Multi-Asset Integration"
tags:
  - doc/task
  - status/done
  - phase/24
  - topic/e2e-integration
  - topic/multi-asset
  - topic/instruments
  - layer/surveillance
  - layer/learning
  - layer/feature-engineering
---

# Task: End-to-End Global Multi-Asset Integration

Status: completed
Research: [[e2e_global_integration]]
Spec: [[e2e_global_integration_spec]]

---

## Phase 24a: Instrument Universe + Daily Price Ingest

- [x] **24a.1**: Create `agent/tools/instrument_universe.py` — `InstrumentDef` dataclass, `INSTRUMENTS` tuple (90 entries), helper functions (`tradeable_instruments`, `instruments_by_class`, `ticker_to_instrument`)
- [x] **24a.2**: Add `ingest_daily_prices(store, as_of, lookback_days)` function — batch yfinance download, register instrument entities, store 3 obs types (return, volume, volatility), per-ticker error handling, >50% failure raises
- [x] **24a.3**: Extend `graph_builder.py` — append "instrument" to `ENTITY_TYPES`, append 3 obs types to `OBSERVATION_TYPES`, ENRICHMENT_DIM 30→33
- [x] **24a.4**: Add `fetch_instruments` node to `daily_collection.py` DAG — FunctionOperator, no deps, timeout 300s
- [x] **24a.5**: Write `tests/test_instrument_universe.py` — 46 tests: InstrumentDef, INSTRUMENTS count, helpers, ingest with mocked yfinance (entity registration, obs types, log return math, failure handling, >50% exception), graph builder integration, DAG integration
- [x] **24a.6**: Update `tests/test_graph_builder_expanded.py` — obs type count 21→24, entity type count 9→10, feature dim 12→13; also fixed obs type sort order

## Phase 24b: GNN Training with Instruments

- [x] **24b.1**: Create `scripts/backfill_instruments.py` — download 2yr daily data for all 90 instruments, register entities day-by-day, idempotent
- [x] **24b.2**: Train HetTGN on backfilled data — verified via slow test: loss convergence, non-zero instrument embeddings, cross-type attention activation, per-instrument surprise extraction
- [x] **24b.3**: Write slow test — synthetic 5-instrument + 10-entity graph, 50 epochs, assert loss decreases, instrument surprises non-zero

## Phase 24b.5: GNN Attention Diagnostic

- [x] **24b.5.1**: Run `compute_diagnostics` with instrument entities — verified entity_type_density, observation_density, edge_type_attention, neighborhood_sparsity all include instrument data
- [x] **24b.5.2**: Write diagnostic output to `[[phase25_gnn_diagnostic]]` — completed in [[phase25_gnn_diagnostic]] as Phase 25 specification input

## Phase 24c: Multi-Asset Strategy Refactor

- [x] **24c.1**: Add `MultiAssetStrategy` ABC to `agent/quant/backtest.py` — `generate_weights()` returns (test_length, N) array, takes `instrument_names: list[str]`
- [x] **24c.2**: Add `MultiAssetWalkForward` + `MultiAssetBacktestResult` + `MultiAssetFoldResult` to `agent/quant/backtest.py` — (T, N) returns, dot-product portfolio P&L, per-fold + aggregate metrics, per-asset-class attribution
- [x] **24c.3**: Add `InstrumentStateAssembler` to `agent/learning/policy/state_assembler.py` — state layout: instrument_surprise_block (N×5) + entity_surprise_block + belief_block + market + adversarial
- [x] **24c.4**: Add `MultiAssetSACStrategy` to `agent/learning/policy/portfolio_strategy.py` — wraps SACTrainer + InstrumentStateAssembler, action_dim = N_instruments, implements MultiAssetStrategy
- [x] **24c.5**: Add `EqualWeightStrategy` (MultiAssetStrategy) — returns 1/N weight for all instruments
- [x] **24c.6**: Add `BuyAndHoldBenchmarkStrategy` (MultiAssetStrategy) — fixed target weights from constructor
- [x] **24c.7**: Write `tests/test_multi_asset_backtest.py` — 46 tests: WalkForward with synthetic (T=500, N=5), fold splits, dot-product P&L, attribution sums, EqualWeight, BuyAndHold, mocked SAC, InstrumentStateAssembler, single-instrument edge case, zero-variance, NaN instrument, wrong shapes, auto-generated names

## Phase 24d: Inference DAG

- [x] **24d.1**: Add `portfolio_weights` and `paper_trade_pnl` tables + CRUD methods to `agent/pipeline/store.py`
- [x] **24d.2**: Create `agent/pipeline/dags/inference.py` — 4 sequential nodes: load_models → gnn_inference → sac_inference → emit_portfolio
- [x] **24d.3**: Register inference DAG in `agent/pipeline/dags/__init__.py`
- [x] **24d.4**: Write `tests/test_inference_dag.py` — 54 tests: DAG structure (10), load_models (4), gnn_inference (4), sac_inference (4), emit_portfolio (7), portfolio_weights CRUD (11), paper_trade_pnl CRUD (9), skip propagation (2), DAG registration (2), no-model graceful failure

## Phase 24e: Walk-Forward Backtest

- [x] **24e.1**: Verify ≥2yr instrument data in PipelineStore from 24b.1 backfill
- [x] **24e.2**: Run multi-asset walk-forward — `agent/quant/walkforward_runner.py`: `load_instrument_returns()`, `run_walkforward()` with configurable strategies, `build_default_strategies()` (EqualWeight, SPY, 60/40)
- [x] **24e.3**: Generate attribution report — `per_group_attribution()` (asset-class & region), `per_instrument_attribution()`, `top_instruments()`, `concentration_stats()`, `generate_attribution_report()` → `StrategyReport` dataclass
- [x] **24e.4**: Write `tests/test_walkforward_multi.py` — 48 tests: walk-forward execution (9), attribution sums (5), top instruments (3), concentration (4), data loading (6), strategy builder (3), MultiAssetWeightedSurpriseStrategy (9), report generation (3), edge cases (6)

## Phase 24f: Paper Trade Launch

- [x] **24f.1**: Verified inference DAG cron schedule `45 19 * * 1-5` fires after all upstream DAGs (latest upstream: world_model + rl_training at `30 19`)
- [x] **24f.2**: Implemented alert conditions in `_emit_portfolio` + 4 helper functions (`_check_concentration`, `_check_drawdown`, `_check_sharpe`, `_check_edge_decay`) — drawdown >5% WARNING, concentration >30% WARNING, edge decay on held instruments WARNING, Sharpe < -0.5 at 30d CRITICAL
- [x] **24f.3**: Write `tests/test_e2e_integration.py` — 29 tests: schedule verification (2), concentration alerts (5), drawdown alerts (3), Sharpe alerts (4), edge decay alerts (5), E2E chain (6), alert integration (4)

---

## Related

- [[e2e_global_integration]] — Research doc
- [[e2e_global_integration_spec]] — Spec
- [[quant_training_ground]] — Master phase tracker
- [[temporal_het_gnn]] — Phase 12 GNN architecture
- [[signal_fusion]] — Phase 20 surprise extraction
- [[rl_policy]] — Phase 21 SAC policy
- [[adversarial]] — Phase 22 adversarial layer
- [[gnn_guided_expansion_r2]] — Phase 23 (last completed)
- [[phase25_gnn_diagnostic]] — Phase 25 input derived from 24b.5 diagnostics
