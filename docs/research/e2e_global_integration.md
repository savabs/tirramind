---
title: "Research: End-to-End Global Multi-Asset Integration"
tags:
  - doc/research
  - phase/24
  - topic/e2e-integration
  - topic/multi-asset
  - topic/instruments
  - layer/surveillance
  - layer/learning
  - layer/feature-engineering
---

# Research: End-to-End Global Multi-Asset Integration

## Problem Statement

All 23 phases of the TirraMind computation stack (Layers 1–6) are built and tested in isolation. 57 surveillance tools, 18 at L2, HetTGN, world model, signal fusion, RL policy, adversarial layer — all exist. But:

1. **No end-to-end pipeline.** The layers have never run together from raw data fetch through to portfolio weights.
2. **No inference DAG.** Training DAGs exist (gnn_inference, rl_training, entity_scoring, etc.) but nothing produces daily actionable portfolio allocations.
3. **Execution is locked to US equities.** `AssetMapper` resolves `entity_id → single US ticker` via entity_aliases. `Strategy.generate_weights()` returns a 1-D array (scalar weight per timestep). The upstream layers (surveillance, GNN, fusion) are asset-agnostic, but the downstream layers (strategy, backtest, reward) are single-asset.
4. **Reward signal uses synthetic returns.** The RL training DAG computes reward from surprise means, not real P&L.
5. **No instrument price ingest.** `MarketDataTool` exists (yfinance wrapper) but is an agent-facing tool, not a pipeline-integrated daily ingest.

## Current Architecture (Relevant Interfaces)

### Graph Builder (`agent/models/gnn/graph_builder.py`)

- `ENTITY_TYPES`: 9 types — company, country, domain, organization, person, protocol, topic, vessel, wallet.
- `OBSERVATION_TYPES`: 21 types — various per-tool observations.
- `ENRICHMENT_DIM = 30` — appended features (cusum, hawkes, distribution stats, obs_type histogram).
- `BASE_FEAT_DIM = len(ENTITY_TYPES) + 3 = 12` — one-hot type + count + recency + mean_value.
- The `build()` method queries all entities/links/observations from `PipelineStore` and emits a `HeteroData` object. **Fully generic over entity types** — any new type added to `ENTITY_TYPES` just works.

### Strategy ABC (`agent/quant/backtest.py`)

```python
class Strategy(ABC):
    def generate_weights(self, train_returns: np.ndarray, test_length: int, ...) -> np.ndarray:
        """Returns 1-D array of length test_length."""
```

- `train_returns` is 1-D. `generate_weights()` returns 1-D (one scalar weight per timestep).
- `WalkForward.run()` takes a 1-D returns array and one Strategy. Fold logic: `weighted_ret = test_ret * weights`.
- **This is inherently single-asset.** Multi-asset requires: returns = (T, N), weights = (test_len, N), fold P&L = sum across instruments.

### SAC (`agent/learning/policy/sac.py`)

- `SACTrainer(state_dim, action_dim, config)` — `action_dim` is already an integer.
- `GaussianActor` outputs N continuous actions in `[-max_pos, max_pos]`, leverage-constrained: `Σ|w_i| ≤ L`.
- **Already multi-dimensional.** Currently action_dim=1 (single aggregate weight). Changing to action_dim=N_instruments requires no architecture change, only a different constructor argument.

### State Assembler (`agent/learning/policy/state_assembler.py`)

- Assembles: top-K entity surprises + beliefs + market features + adversarial summary → fixed-dim tensor.
- Filters to "tradeable" entities via `asset_map: dict[str, str]`.
- **Needs extension** to include per-instrument features (return, vol, surprise) instead of only entity-level signals.

### Asset Mapper (`agent/learning/policy/asset_mapper.py`)

- Maps `entity_id → ticker` via `entity_aliases` table (source='ticker'), restricted to company type.
- **Will be deprecated/replaced.** In the new design, instruments ARE entities — they don't need to be "mapped from" other entities.

### Reward Function (`agent/learning/policy/reward_fn.py`)

- `extrinsic(portfolio_return, rolling_returns)` — already takes a scalar portfolio return.
- **Asset-agnostic.** No change needed if the caller computes portfolio_return = weights @ returns.

### DAG Infrastructure (`agent/pipeline/dags/__init__.py`)

- 10 DAGs registered. `get_default_dags()` returns all.
- Adding a new DAG: create a builder function, import in `__init__.py`, append to the list.
- Daily collection runs at 18:00 UTC weekdays. Other DAGs at 18:30–19:30.

### Surprise Extractor (`agent/fusion/surprise.py`)

- `SurpriseExtractor.extract()` → list of `EntitySurprise` per entity.
- Per-entity: 5 surprise signals + composite.
- **Fully generic over entity types.** Instrument entities will get surprises exactly like any other entity type.

## Architectural Decision: Instruments as GNN Entity Nodes

**Decision:** Add "instrument" as a 10th entity type in the HetTGN graph. Each instrument (~90 total, verified) is a node with daily observations (return, volume, realized_vol). The GNN's cross-type attention mechanism discovers which entity patterns (vessel diversions, insider filings, CFTC positioning) predict instrument behavior.

**Why this is correct:**

1. The HetTGN (`het_tgn.py`) dispatches attention computation per edge type. Any new (src_type, rel, dst_type) triple just works.
2. The graph builder dynamically builds nodes for all types in the `id_map`. Adding "instrument" to `ENTITY_TYPES` + registering instrument entities in `PipelineStore` is sufficient.
3. Per-instrument prediction surprise IS the trading signal. If the GNN predicts instrument X will have observation type A at time T, and the actual observation is type B, the surprise is our alpha.
4. No hand-coded entity→instrument mapping rules ("drought → coffee futures"). The GNN learns cross-type attention weights that discover these patterns.
5. Adding a new instrument = registering a new entity node. Zero code change beyond the initial framework.

**What changes:**

| Component | Change |
|-----------|--------|
| `ENTITY_TYPES` | Append "instrument" (index 9) |
| `OBSERVATION_TYPES` | Append 3: "instrument_return", "instrument_volume", "instrument_volatility" |
| `ENRICHMENT_DIM` | 30 → 33 (obs_type_dist grows by 3) |
| `BASE_FEAT_DIM` | 12 → 13 (one-hot grows by 1) |
| `PipelineStore` | Instrument entities registered via normal `register_entity()` + daily `store_entity_observation()` |

## Instrument Universe (Verified 2026-04-13)

All tickers verified via `yfinance` terminal test. 89/90 returned data for `period='5d'`. `LBS=F` (lumber) was delisted — replaced with `OJ=F` (orange juice).

### Commodity Futures (20)

| Ticker | Name | Sector |
|--------|------|--------|
| CL=F | WTI Crude Oil | Energy |
| BZ=F | Brent Crude Oil | Energy |
| NG=F | Natural Gas | Energy |
| RB=F | RBOB Gasoline | Energy |
| GC=F | Gold | Precious Metals |
| SI=F | Silver | Precious Metals |
| PL=F | Platinum | Precious Metals |
| PA=F | Palladium | Precious Metals |
| HG=F | Copper | Industrial Metals |
| ZW=F | Wheat | Agriculture |
| ZC=F | Corn | Agriculture |
| ZS=F | Soybeans | Agriculture |
| KC=F | Coffee | Agriculture |
| CC=F | Cocoa | Agriculture |
| CT=F | Cotton | Agriculture |
| SB=F | Sugar | Agriculture |
| ZO=F | Oats | Agriculture |
| OJ=F | Orange Juice | Agriculture |
| LE=F | Live Cattle | Livestock |
| HE=F | Lean Hogs | Livestock |

### FX Pairs (15)

| Ticker | Pair | Category |
|--------|------|----------|
| EURUSD=X | EUR/USD | G10 |
| USDJPY=X | USD/JPY | G10 |
| GBPUSD=X | GBP/USD | G10 |
| USDCHF=X | USD/CHF | G10 |
| AUDUSD=X | AUD/USD | G10 |
| USDCAD=X | USD/CAD | G10 |
| NZDUSD=X | NZD/USD | G10 |
| EURGBP=X | EUR/GBP | G10 Cross |
| EURJPY=X | EUR/JPY | G10 Cross |
| GBPJPY=X | GBP/JPY | G10 Cross |
| USDMXN=X | USD/MXN | EM |
| USDBRL=X | USD/BRL | EM |
| USDINR=X | USD/INR | EM |
| USDCNY=X | USD/CNY | EM |
| USDZAR=X | USD/ZAR | EM |

### Equity Index/ETF (25)

| Ticker | Exposure | Region |
|--------|----------|--------|
| ES=F, NQ=F, YM=F, RTY=F | US index futures | US |
| SPY, QQQ, IWM, DIA | US index ETFs | US |
| EWZ | Brazil | LatAm |
| EWG | Germany | Europe |
| FXI | China | Asia |
| EWJ | Japan | Asia |
| EWY | South Korea | Asia |
| EWA | Australia | Pacific |
| EWC | Canada | Americas |
| EWU | United Kingdom | Europe |
| EWQ | France | Europe |
| EWP | Spain | Europe |
| EWI | Italy | Europe |
| INDA | India | Asia |
| EWT | Taiwan | Asia |
| EWH | Hong Kong | Asia |
| THD | Thailand | Asia |
| EWW | Mexico | LatAm |
| VGK | FTSE Europe | Europe |

### Sector ETFs (15)

| Ticker | Sector |
|--------|--------|
| XLE | Energy |
| XLF | Financials |
| XLK | Technology |
| XLV | Healthcare |
| XLI | Industrials |
| XLP | Consumer Staples |
| XLY | Consumer Discretionary |
| XLB | Materials |
| XLU | Utilities |
| XLRE | Real Estate |
| XLC | Communication Services |
| GDX | Gold Miners |
| SLV | Silver ETF |
| USO | US Oil Fund |
| UNG | US Natural Gas Fund |

### Fixed Income (10)

| Ticker | Name |
|--------|------|
| ZN=F | 10-Year T-Note Futures |
| ZB=F | 30-Year T-Bond Futures |
| ZF=F | 5-Year T-Note Futures |
| TLT | 20+ Year Treasury ETF |
| IEF | 7-10 Year Treasury ETF |
| SHY | 1-3 Year Treasury ETF |
| HYG | High Yield Corporate ETF |
| LQD | Investment Grade Corporate ETF |
| EMB | EM Bond ETF |
| AGG | US Aggregate Bond ETF |

### Volatility + Crypto (5)

| Ticker | Name |
|--------|------|
| ^VIX | VIX Index (reference only — not directly tradeable) |
| VIXY | VIX Short-Term Futures ETF |
| UVXY | Ultra VIX Short-Term Futures ETF |
| BTC-USD | Bitcoin |
| ETH-USD | Ethereum |

**Total: 90 instruments (89 tradeable + ^VIX reference).**

## Sub-Phase Architecture

### 24a: Instrument Universe + Daily Price Ingest

**New file:** `agent/tools/instrument_universe.py`
- `InstrumentDef` dataclass: ticker, name, asset_class, region, is_tradeable.
- `INSTRUMENTS: list[InstrumentDef]` — all 90 instruments defined.
- `ingest_daily_prices(store: PipelineStore, as_of: date)` function:
  - Fetches 1d close via yfinance for all instruments.
  - Registers each instrument as entity type="instrument" if not exists.
  - Stores 3 observations per instrument: instrument_return (log return), instrument_volume, instrument_volatility (20d realized vol).
  - Non-fatal per-ticker: if one ticker fails, others continue.

**Modify:** `agent/models/gnn/graph_builder.py`
- `ENTITY_TYPES`: append "instrument" (new index 9).
- `OBSERVATION_TYPES`: append "instrument_return", "instrument_volume", "instrument_volatility".
- `ENRICHMENT_DIM`: 30 → 33 (obs_type_dist length grows by 3).

**Modify:** `agent/pipeline/dags/daily_collection.py`
- Add `fetch_instruments` node calling instrument_ingest.

**Test:** Verify instrument entities appear in graph with correct features, obs types distribute correctly.

### 24b: GNN Training with Instruments

- Backfill 2 years of historical prices for all 90 instruments.
- Train HetTGN with instrument nodes included.
- Verify: training converges (loss decreasing), cross-type attention activates between entity types and instruments, per-instrument surprise extraction works.

### 24b.5: GNN Attention Diagnostic

- Run `run_diagnostics` with instrument entities present.
- Report: per-instrument-class attention density, data-starved instruments, ranked L2 upgrade recommendations.
- **This output IS the Phase 25 spec.** Do not expand L2 tools without this data.

### 24c: Multi-Asset Strategy Refactor

**Current state (single-asset):**
- `Strategy.generate_weights()` → 1-D array (scalar per timestep).
- `WalkForward.run()` multiplies 1-D returns × 1-D weights.

**Target state (multi-asset):**
- `Strategy.generate_weights()` → 2-D array (N instruments × test_length), or maintain 1-D backward compat.
- New `MultiAssetWalkForward` class: takes (T, N) returns matrix, produces per-fold + aggregate metrics with portfolio-level P&L.
- `SACPortfolioStrategy` refactored: action_dim = N_instruments, state includes per-instrument features.
- `StateAssembler` extended: instrument surprise block (per-instrument, not per-entity-filtered).
- `AssetMapper` deprecated — instruments are entities, no mapping needed.

**Backward compatibility:** The existing 1-D `Strategy` / `WalkForward` remain untouched. New multi-asset classes are separate, not replacements.

### 24d: Inference DAG

New DAG with 4 nodes:
1. `load_models` — load trained HetTGN + SAC from PipelineStore.
2. `gnn_inference` — build graph from today's entities, run forward pass, extract surprises.
3. `sac_inference` — assemble state from instrument surprises + beliefs + adversarial flags, run SAC deterministic action.
4. `emit_portfolio` — write portfolio weights to `PipelineStore` (new table: `portfolio_weights`).

New `PipelineStore` tables:
- `portfolio_weights`: date, instrument_ticker, weight, metadata_json.
- `paper_trade_pnl`: date, portfolio_return, benchmark_return, metadata_json.

### 24e: Walk-Forward Backtest

- 2 years of daily prices for all 90 instruments.
- 90 days of surveillance data for entity observations.
- 252 trading days warm-up for GNN.
- 6 monthly folds (21 trading days each).
- Metrics: Sharpe, max drawdown, attribution by asset class/region/signal source.
- Compare: SAC vs WeightedSurprise vs equal-weight vs SPY vs 60/40.

### 24f: Paper Trade Launch

- Daily inference DAG at 19:45 (after all upstream DAGs complete).
- P&L tracking: daily mark-to-market, cumulative Sharpe, per-asset-class attribution.
- Alerts: drawdown > 5%, concentration > 30% in a single instrument, edge decay.
- Kill criteria: Sharpe < -0.5 at 30d → diagnose; Sharpe < 0 at 60d → redesign.

## Risks

1. **yfinance reliability.** Free tier, no SLA. Mitigations: retry logic, cache recent prices, detect stale data (compare last date to today).
2. **GNN training cost with 90 more nodes.** Current entity count is ~thousands from L2 tools. Adding 90 instrument nodes is <1% increase. No performance concern.
3. **Spurious cross-type attention.** With limited surveillance history, the GNN might find noise correlations between entities and instruments. Mitigation: min_samples threshold before trusting attention weights; 24b.5 diagnostic flags low-data instruments.
4. **Multi-asset backtest overfitting.** 90 instruments × 6 folds × multiple strategies = many comparisons. Mitigation: use Bonferroni or BH FDR correction on Sharpe significance tests across instruments.
5. **FX return computation.** Some pairs are quoted as USD/X (USDJPY) while others are X/USD (EURUSD). Must normalise to consistent convention. Log returns should be from the perspective of a USD-based portfolio.

## Data Requirements

- **Historical prices:** 2+ years daily OHLCV for all 90 instruments (yfinance `period='2y'`, `interval='1d'`).
- **Surveillance data:** Already being collected by daily DAGs for 57 tools. No new data collection needed for 24a–24b.
- **For 24e backtest:** Need to replay historical surveillance data alongside historical prices. If no stored historical surveillance exists, the first backtest will use only instrument-level features (return, vol) + whatever entity observations exist in PipelineStore.

## Related

- [[e2e_global_integration_spec]] — Spec
- [[e2e_global_integration]] — Task (to be created)
- [[quant_training_ground]] — Master phase tracker
- [[temporal_het_gnn]] — Phase 12 GNN architecture
- [[signal_fusion]] — Phase 20 surprise extraction
- [[rl_policy]] — Phase 21 SAC policy
- [[adversarial]] — Phase 22 adversarial layer
- [[gnn_guided_expansion_r2]] — Phase 23 (last completed)
- [[chat_checkpoint_2026-04-13_session2]] — Planning session
