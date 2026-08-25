---
title: "Spec: Phase 37 — First Live Pipeline Run"
tags:
  - doc/spec
  - phase/37
  - topic/pipeline
  - topic/backtest
  - layer/surveillance
  - layer/feature-engineering
---

# Spec: Phase 37 — First Live Pipeline Run

## Goal

Run the TirraMind pipeline end-to-end with real data for the first time.
Verify that every layer of the 7-layer stack functions correctly on real
API outputs. Backfill historical instrument prices and run the first
walk-forward backtest on real data.

## Sub-Phases

### 37a: First Real Collection Run

Get the daily_collection DAG running with real API calls and verify data
is persisted correctly to the pipeline DB.

### 37b: Historical Instrument Backfill

Create a backfill script that fetches 2+ years of historical daily prices
for all 90 instruments and stores them as entity observations. This provides
the data depth needed for walk-forward backtesting.

### 37c: Downstream DAG Validation

Run the downstream DAGs (convergence, features, GNN, world model) on real
data and verify they produce reasonable output. Fix any integration bugs.

### 37d: First Real Backtest

Train GNN on the real entity graph, run walk-forward backtest, measure
actual performance vs baselines.

## Files Affected

### New Files

| File | Purpose |
|------|---------|
| `scripts/backfill_prices.py` | Historical price backfill script |
| `scripts/run_collection.py` | Convenience script for manual daily_collection run |
| `tests/test_phase37_live_pipeline.py` | Integration tests (mocked APIs verify wiring) |

### Modified Files

| File | Change |
|------|--------|
| `agent/tools/instrument_universe.py` | Add `backfill_historical_prices()` function |
| `agent/pipeline/dags/daily_collection.py` | (if needed) Fix operator resolution issues |

## Implementation Steps

### Phase 37a: First Real Collection (Steps 37.1–37.5)

**37.1: Wire PipelineStore to tools in build_tool_registry()**
- In `agent/cli.py`, pass `pipeline_store=pipeline_store` to all tools that
  accept it. The PipelineStore is already created at line ~161. Move its
  creation earlier (before tool registration) and pass it to each tool.
- This is the critical integration fix: without it, no tool persists entities.
- Test: unit test that tools receive non-None `_store` after construction.

**37.2: Create `scripts/run_collection.py` convenience script**
- Standalone script that builds a fresh PipelineStore, constructs the
  daily_collection DAG, wires the ToolRegistry (with PipelineStore), and executes.
- Prints per-node status and timing.
- Test: script runs without error, prints node results.

**37.3: Run daily_collection → verify instrument ingest**
- Execute the script. Focus on `fetch_instruments` node first (most critical).
- Inspect pipeline DB: check `entities` table has instrument rows, 
  `entity_observations` has price/vol/return observations.
- Test: `SELECT COUNT(*) FROM entities WHERE entity_type='instrument'` > 0.

**37.4: Verify entity persistence from other tools**
- After DAG completes, verify each entity-persisting tool stored data:
  - CFTC: cftc_contract entities with positioning observations
  - FINRA: company entities with short_volume observations
  - GDELT: country entities with geopolitical observations  
  - Polymarket: topic entities with market_probability observations
- Test: entity_observations count > 0 for each source_tool.

**37.5: Verify entity links created**
- Check `entity_links` table: instruments should be linked to companies
  (tracks_issuer), countries (located_in), etc.
- These links come from the tools' `_persist_entities` methods.
- Test: `SELECT COUNT(*) FROM entity_links` > 0.

### Phase 37b: Historical Backfill (Steps 37.5–37.6)

**37.5: Implement backfill_historical_prices()**
- Add function to `instrument_universe.py` that fetches 2+ years of daily
  prices via yfinance and stores each day as a separate observation.
- Parameters: `store`, `lookback_years=3`, `batch_size=20`.
- Uses `yf.download(period="3y")` for full history.
- Stores: close, volume, realized_vol_20d, log_return per day per instrument.
- Rate limiting: batch downloads of 20 tickers at a time, 2s delay between.
- Test: function populates entity_observations with >500 rows per instrument.

**37.6: Create backfill_prices.py script**
- Standalone script that calls `backfill_historical_prices()`.
- Progress bar, error handling, resume capability (skip instruments with
  existing observations).
- Test: script runs end-to-end, DB has historical data.

### Phase 37c: Downstream Validation (Steps 37.7–37.9)

**37.7: Run convergence_detection on real data**
- Execute convergence_detection DAG after collection.
- Verify BOCPD/HMM produce non-degenerate output on real price series.
- Test: convergence scores are stored and non-zero.

**37.8: Run feature_generation on real data**
- Execute feature_generation DAG.
- Verify all 3 FeatureBuilders (convergence, macro, GNN) produce features.
- Test: engineered_features table has rows.

**37.9: Run gnn_inference on real entity graph**
- Execute gnn_inference DAG. 
- Train GNN on the real entity graph (accumulated from collection + backfill).
- Verify: model checkpoint saved, entity embeddings produced.
- Test: gnn_model.pt updated, entity_count > 0 in inference result.

### Phase 37d: First Real Backtest (Step 37.10)

**37.10: Run walk-forward backtest on real data**
- Load historical instrument prices from PipelineStore.
- Run WalkForward with WeightedSurpriseStrategy and equal-weight baseline.
- Report: Sharpe, max drawdown, win rate, vs buy-and-hold.
- Test: backtest completes without error, produces ≥1 fold.

## Edge Cases

1. **yfinance download failure for some tickers** — handle gracefully, log
   failed tickers, continue with available data.
2. **API rate limiting** — implement exponential backoff and batch chunking.
3. **Empty entity graph after first collection** — some tools may return no
   data on weekends/holidays. Check day-of-week.
4. **GNN training failure on sparse real graph** — may need minimum entity
   count threshold before attempting training.
5. **Duplicate observations on re-run** — backfill must be idempotent.
   Use `observed_at` timestamp as dedup key.
6. **NaN/Inf in real price data** — yfinance sometimes returns NaN for
   delisted or suspended tickers. Filter before storing.

## Testing Plan

### Unit Tests (mocked)
- `test_phase37_live_pipeline.py`: Verify backfill function stores correct
  schema, handles partial failures, deduplicates on re-run.

### Integration Tests (real APIs)
- Run `scripts/run_collection.py` manually and inspect DB.
- Run `scripts/backfill_prices.py` manually and verify row counts.
- These are manual because they hit real APIs.

### Validation Criteria
- [ ] Pipeline DB has >0 entities per entity type
- [ ] Pipeline DB has >0 observations per source_tool
- [ ] Entity links exist between instruments and companies/countries
- [ ] Historical backfill provides ≥2 years of daily data for ≥80% of instruments
- [ ] Walk-forward backtest completes with ≥1 fold
- [ ] GNN training loss converges on real data

## Related

- [[phase37_first_live_pipeline]]
- [[phase37_first_live_pipeline_task]]
- [[quant_training_ground]]
- [[e2e_global_integration]]
