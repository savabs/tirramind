---
title: "Research: Phase 37 — First Live Pipeline Run"
tags:
  - doc/research
  - phase/37
  - topic/pipeline
  - topic/backtest
  - layer/surveillance
  - layer/world-model
  - layer/feature-engineering
---

# Research: Phase 37 — First Live Pipeline Run

## Context

Phases 0–36 built the full 7-layer computation stack:

| Layer | Status | Key Components |
|-------|--------|----------------|
| 1. Surveillance | ✅ 60+ tools, 11 entity types, 45 obs types | daily_collection DAG (7 nodes, all free APIs) |
| 2. Feature Eng | ✅ 3 FeatureBuilders (convergence, macro, GNN) | feature_generation DAG |
| 3. World Model | ✅ Bayesian network (pgmpy/pymc) | world_model_update DAG |
| 4. Signal Fusion | ✅ SurpriseExtractor, entity alerts | entity_scoring DAG |
| 5. RL Policy | ✅ SAC actor-critic, multi-asset | rl_training DAG |
| 6. Adversarial | ✅ Edge decay, manipulation detection | adversarial_scan DAG |
| 7. LLM Support | ✅ Narrative synthesis | (not in pipeline) |

**Critical finding:** The pipeline DB (`pipeline.db`) has the schema but **zero real data rows**. Every entity table, observation table, and feature table is empty. The GNN has only been trained on synthetic data. No backtest has ever run on real data. The system has never made a real inference.

This is the most important gap. All further improvements (more tools, better math, more link types) are speculative until the system has run end-to-end on real data and we can measure whether the architecture actually produces useful signal.

## Current Architecture

### Daily Collection DAG (7 nodes, all free)

| Node | Tool | API | Entities Persisted? |
|------|------|-----|---------------------|
| `fetch_cftc` | `cftc` | CFTC.gov (free) | ✅ cftc_contract entities |
| `fetch_finra_scan` | `finra_short_volume` | FINRA API (free) | ✅ company entities |
| `fetch_power_demand` | `power_grid` | NYISO (free) | ❌ pipeline_data only |
| `fetch_power_fuel` | `power_grid` | NYISO (free) | ❌ pipeline_data only |
| `fetch_gdelt` | `gdelt` | GDELT (free) | ✅ country entities |
| `fetch_polymarket` | `polymarket` | Gamma API (free) | ✅ topic entities |
| `fetch_instruments` | `instrument_universe` | yfinance (free) | ✅ instrument entities |

### Downstream DAG Chain

```
daily_collection → convergence_detection → gnn_inference → feature_generation
                 → whale_tracking → whale_scoring → entity_scoring
                 → world_model_update → rl_training → adversarial_scan
                 → inference (emits portfolio weights)
```

### Pipeline Execution

- CLI: `python -m agent.cli --pipeline run daily_collection`
- Executor: `DAGExecutor.execute(dag, trigger="manual")` → topo sort → layer-by-layer
- Storage: `PipelineStore` (SQLite) — entities, observations, links, features, beliefs

## Observations

### What Works (Unit-Tested)

1. All 11 DAGs build successfully and topo-sort correctly
2. All tool entity persistence methods work in isolation (mocked APIs)
3. GNN trains and infers on synthetic graphs (32.9% top-1 accuracy)
4. Walk-forward backtest framework produces valid fold results
5. SAC policy trains and generates deterministic actions
6. Inference DAG gracefully skips when models are missing

### What Has Never Been Tested

1. **Real API calls** from daily_collection nodes
2. **Entity graph construction** from real observations
3. **GNN training** on real entity graph
4. **Feature generation** from real observations
5. **World model beliefs** from real data
6. **RL training** from real entity alerts
7. **Full DAG chain** execution (daily_collection → ... → inference)
8. **Walk-forward backtest** with real historical prices

### Known Gaps

1. **power_grid** tool does NOT persist entities — stores to `pipeline_data` only.
   This means NYISO demand/fuel data exists but doesn't feed the entity graph.
   Not critical for Phase 37 (power_grid is a macro conditioning signal).

2. **CRITICAL: CLI does not wire PipelineStore to tools.** In `build_tool_registry()`,
   all tools are constructed with `cache=cache` but **without** `pipeline_store`.
   This means entity persistence silently skips for every tool when run through the
   CLI. The `fetch_instruments` node works because it uses a FunctionOperator that
   creates its own PipelineStore, but all ToolOperator-based nodes (CFTC, FINRA,
   GDELT, Polymarket) would fetch data successfully but never write entities.
   21 tools have this issue. Fix: pass `pipeline_store=store` during construction.

3. **No historical backfill mechanism.** `ingest_daily_prices` fetches 30 days of
   lookback for vol calculation but stores only the latest observation. For a
   walk-forward backtest (min_train=104 weeks ≈ 2 years), we need historical data.
   Options:
   - a) Use yfinance's `period="max"` to backfill 2+ years of daily prices
   - b) Store daily observations over time (accumulate naturally)
   - c) Run a one-shot historical backfill script

3. **Operator resolution.** Some DAG nodes use string operator names (e.g., "cftc")
   that require a `ToolRegistry` at execution time. The `FunctionOperator` nodes
   (like `fetch_instruments`) work without registry. Need to ensure the CLI
   correctly wires the tool registry to the executor.

4. **No monitoring or alerting** for failed DAG runs. If a node fails silently,
   we won't know data collection stopped.

5. **Rate limiting.** Running all 7 nodes simultaneously might trigger rate limits
   on some APIs (yfinance in particular is known for aggressive rate limiting).

## Risks

1. **yfinance reliability.** Yahoo Finance is a scraping-based library, not an
   official API. It frequently breaks due to Yahoo HTML changes. Need a fallback
   or retry strategy.

2. **API format changes.** CFTC/FINRA/GDELT endpoints are stable but government
   APIs occasionally change format without warning. The parsers are tested against
   specific formats.

3. **Entity deduplication.** Different tools may create the same entity with
   different IDs (e.g., "AAPL" from instrument_universe vs "Apple Inc" from
   FINRA). The entity linking layer should handle this but has never been tested
   on real data.

4. **Graph sparsity.** With only one day of data, the entity graph will be very
   sparse. GNN training may not converge meaningfully. Need multiple days of
   accumulation.

5. **Disk space.** SQLite DB with 90 instruments × 365 days of observations =
   ~32K rows. Manageable.

## Data Requirements

### Minimum Viable Pipeline Run (Phase 37a)

- Run `daily_collection` DAG once → verify entities + observations stored
- Verify entity links are created between tool outputs
- Run downstream DAGs (feature_generation, gnn_inference) → verify they handle
  sparse real data gracefully

### Historical Backfill (Phase 37b)

- Backfill 2+ years of daily instrument prices via yfinance
- Store as daily observations in PipelineStore
- Enough data for walk-forward backtest (min_train=104 weeks)

### End-to-End Validation (Phase 37c)

- Run full DAG chain with accumulated data
- Train GNN on real entity graph
- Run walk-forward backtest
- Compare real-data performance vs synthetic-data baseline

## Math/Algorithm Survey

No new algorithms needed. Phase 37 is integration, not new math. The
mathematical components (BOCPD, HMM, FFT, SurpriseExtractor, SAC, Kalman)
are all built and unit-tested. The question is whether they produce useful
output on real data.

**Key metrics to monitor:**
- Entity graph density (nodes, edges per entity type)
- GNN training loss convergence on real data
- Surprise distribution shape (should not be degenerate)
- Walk-forward Sharpe ratio vs buy-and-hold baseline
- Feature importance ranking (which builders contribute signal?)

## Depth Roadmap

Phase 37 is L1 execution (get the pipeline running). L2 improvements
(entity resolution, cross-domain linking on real data) come after the
system demonstrates basic functionality.

## Approach

**Atomic decomposition:** Don't try to run the entire pipeline end-to-end
in one step. Instead:

1. Run `daily_collection` in isolation. Verify each node succeeds.
2. Inspect the pipeline DB after collection — entity counts, observation counts.
3. Run `convergence_detection` → verify it handles real observations.
4. Run `feature_generation` → verify features are computed.
5. Run `gnn_inference` → train GNN on real data.
6. Build historical backfill for instrument prices.
7. Run walk-forward backtest with real data.

Each step is independently verifiable. Fail fast if any step reveals
integration bugs.

## Related

- [[phase36_connect_disconnected_entities]]
- [[phase37_first_live_pipeline_spec]]
- [[quant_training_ground]]
- [[e2e_global_integration]]
