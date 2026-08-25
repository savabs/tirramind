---
title: "Task: Phase 37 — First Live Pipeline Run"
tags:
  - doc/task
  - status/done
  - phase/37
  - topic/pipeline
  - topic/backtest
  - layer/surveillance
  - layer/feature-engineering
---

# Task: Phase 37 — First Live Pipeline Run

Status: completed
Research: [[phase37_first_live_pipeline]]
Spec: [[phase37_first_live_pipeline_spec]]

## Phase 37a: First Real Collection Run

- [x] 37.1: Wire PipelineStore to tools in `build_tool_registry()` (cli.py)
- [x] 37.2: Create `scripts/run_collection.py` convenience script
- [x] 37.3: Run daily_collection → verify instrument ingest to DB (89 instruments, 267 obs)
- [x] 37.4: Verify entity persistence (CFTC:20, GDELT:375, Poly:737; FINRA=weekend)
- [x] 37.5: Verify entity links (70 total: 36 topic→instrument, 29 event, 5 cftc)

## Phase 37b: Historical Instrument Backfill

- [x] 37.5: Implement `backfill_historical_prices()` in instrument_universe.py
- [x] 37.6: Create `scripts/backfill_prices.py` script (89/89 filled, 68k obs)

## Phase 37c: Downstream DAG Validation

- [x] 37.7: Run convergence_detection — 0 evidence (pipeline_data table empty; extractors need entity_observations bridge)
- [x] 37.8: Run feature_generation — 17 features produced but all missing (no upstream convergence/signals/GNN yet)
- [x] 37.9: Run gnn_inference on real entity graph → verify training converges
  - GNN trained on 918 entities, 69749 observations, 259 links (5 epochs, 18 min CPU)
  - Model checkpoint: `.tirra_pipeline/gnn_model.pt` (1.6MB, 6 entity types, 121 param groups)
  - Fixed: added `instrument_daily` to OBSERVATION_TYPES (was missing 98% of obs)
  - Fixed: 4 stale test assertions (obs count 39/45→46, ENRICHMENT_DIM 48/54→55)

## Phase 37d: First Real Backtest

- [x] 37.10: Run walk-forward backtest on real historical data
  - 89 instruments × 1097 dates (2023-04-18 to 2026-04-18), 40 folds
  - Fixed: `load_instrument_returns` to accept `instrument_daily` obs type
  - Created: `scripts/run_backtest.py`
  - Results: EqualWeight Sharpe=0.995 (+23%), SPY BuyHold Sharpe=0.900 (+49%), 60/40 Sharpe=0.991 (+32%)
  - Also fixed: `test_outcome_finetuning` (0.0 < 0.0*10 edge case), `test_pattern_extractor` (2-hop patterns)

---

## Related

- [[phase37_first_live_pipeline|Research: Phase37 First Live Pipeline]]
- [[phase37_first_live_pipeline_spec|Spec: Phase37 First Live Pipeline]]
- [[convergence_detection]]
- [[world_model]]
