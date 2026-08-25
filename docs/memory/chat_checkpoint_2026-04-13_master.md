---
title: "Master Checkpoint: TirraMind — Full Project State"
tags:
  - doc/checkpoint
  - phase/24
  - topic/e2e-integration
  - topic/multi-asset
  - topic/master-checkpoint
  - layer/surveillance
  - layer/feature-engineering
  - layer/world-model
  - layer/fusion
  - layer/learning
  - layer/adversarial
---

# Master Checkpoint — TirraMind — 2026-04-13

This is the comprehensive project-state document. Everything from the original creative idea, through every phase of development, to the current state and next steps. Written at the end of Phase 24c completion.

---

## Part 1: The Creative Idea — What Is TirraMind?

### The One-Sentence Pitch

**TirraMind is a machine intelligence system that autonomously discovers predictive structure in the world by observing reality at every layer — physical, behavioral, informational, and financial — and applying SOTA mathematics to extract asymmetric edge that nobody else can see.**

### The Core Thesis

Markets are outputs. Reality is the input. Most quant systems model Layer 3 (prices). Smart ones reach Layer 2 (filings, news). TirraMind operates at **Layers 0 and 1** — observing the physical world (ships, satellites, energy grids, weather) and human behavioral traces (prediction market whales, insider filings, lobbying spend, hiring patterns) — and derives price movements as consequences of deeper reality.

```
Layer 0: Physical reality (atoms, energy, weather, ships, factories)
    ↓ generates
Layer 1: Human decisions (policy, trades, consumption, conflict)
    ↓ generates
Layer 2: Information flows (news, filings, data releases)
    ↓ generates
Layer 3: Market prices (the scoreboard — what everyone stares at)
```

**Price is a symptom. It's the last thing to move, not the first.**

### The Firm Identity

TirraMind is a **technologically advanced information-arbitrage firm**. Not a quant fund that buys data. The moat is:

1. **Finding unique, cheap/free data sources** that leak predictive signal (prediction markets, on-chain flows, insider filings, behavioral traces, physical world observables)
2. **Applying SOTA mathematics** to extract orders-of-magnitude more edge than anyone else could
3. **Learning autonomously** which sources and methods produce the most alpha

**The formula:** Unique observation × Advanced math = Asymmetric edge.

**Cost discipline is strategic.** The most information-rich sources are often the cheapest — nobody else looks at them:
- Polymarket whale tracking — **free** — leaks insider knowledge
- SEC EDGAR insider filings — **free** — executives reveal private info
- On-chain wallet flows — **free** — every blockchain transaction is public
- FRED/ECB/BOJ central bank data — **free** — global liquidity plumbing
- AIS shipping data — **free** — physical world before news reports
- Government filings (patents, lobbying, FDA) — **free** — strategic intent in regulatory paperwork

**The purpose is money.** Every component exists to produce real financial returns. Edge → returns → capital → more data/compute → better intelligence → more edge. This is the flywheel.

### The Bar

**Renaissance Technologies + Anthropic Claude level ambition.** Every component should pass the test: *"Would a senior quant researcher at RenTech/DeepMind consider this serious?"* The system covers:

| Domain | Standard |
|--------|----------|
| Mathematics | Measure theory, stochastic calculus, information geometry, optimal transport, spectral methods |
| Machine Learning | Foundation architectures, neural ODEs, meta-learning, causal inference, world models, RL from scratch |
| Physics | Statistical mechanics (Ising), dynamical systems, phase transitions, renormalization |
| Finance | Microstructure, factor models, regime detection, volatility surfaces — every global market |
| Information Theory | Mutual information, rate-distortion, entropy-based feature selection |

### Architecture Priority: Math Before LLM

The LLM is scaffolding. The math is the product. Priority ordering:

1. **More data tools** — expand surveillance surface. Free APIs first.
2. **Standardized signals** — every source feeds normalized features (OFI, VPIN, Hurst, transfer entropy, MI)
3. **World model** — Bayesian network. Evidence → belief propagation → posteriors.
4. **Signal fusion** — Kalman/particle filter. Fuse noisy multi-source observations.
5. **Probabilistic output** — Monte Carlo, copulas for tails, Kelly for sizing.
6. **RL policy** — model-based RL for action selection.
7. **Adversarial layer** — manipulation detection, edge decay monitoring.
8. **LLM last** — text parsing, hypothesis generation, narrative synthesis. Explains, doesn't decide.

---

## Part 2: The 7-Layer Computation Stack

Every piece of code maps to exactly one layer. Layers don't mix.

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1: SURVEILLANCE SURFACE                                  │
│  60 tools: Polymarket, EDGAR, GDELT, CFTC, AIS, Grid,          │
│  Whale Alert, DarkPool, ClinicalTrials, Patent, 50+ more       │
│  Directory: agent/tools/ (60 files)                             │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: FEATURE ENGINEERING                                   │
│  OFI, VPIN, Hurst, Transfer entropy, Hawkes, spectral, scoring │
│  Directory: agent/quant/ (7 files)                              │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: WORLD MODEL                                           │
│  Temporal Heterogeneous GNN (HetTGN), Bayesian network,         │
│  causal graph, belief propagation, world model                  │
│  Directory: agent/models/ (14 files)                            │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4: SIGNAL FUSION                                         │
│  Surprise extraction, entity scoring, convergence detection     │
│  Directory: agent/fusion/ + agent/pipeline/dags/                │
├─────────────────────────────────────────────────────────────────┤
│  Layer 5: RL POLICY + PORTFOLIO OPTIMIZER                       │
│  SAC (Soft Actor-Critic), state assembler, reward function,     │
│  replay buffer, portfolio strategy, walk-forward backtest       │
│  Directory: agent/learning/ (21 files)                          │
├─────────────────────────────────────────────────────────────────┤
│  Layer 6: ADVERSARIAL INTELLIGENCE                              │
│  VPIN, edge decay, crowding detection, adversarial scanner      │
│  Directory: agent/adversarial/ (6 files)                        │
├─────────────────────────────────────────────────────────────────┤
│  Layer 7: LLM (SUPPORT ROLE ONLY)                               │
│  Text parsing, hypothesis generation, narrative synthesis       │
│  Directory: agent/reasoning/                                    │
└─────────────────────────────────────────────────────────────────┘
```

### The Signal Depth Doctrine

Every data source has three depth layers. The edge lives in L2 and L3:

| Depth | Name | Who Else Does It | Our Coverage |
|-------|------|-------------------|-------------|
| **L1** | Aggregate (top-line numbers) | Everyone | 60 tools |
| **L2** | Entity-Level (track individual actors over time) | A few specialized desks | 11 deep tools at L2 |
| **L3** | Cross-Entity (link entities across data domains) | **Nobody** — this is the moat | Architecture ready, GNN-guided |

### Data Source Tiering (By Real-Time-ness)

| Tier | Latency | Examples | Role |
|------|---------|---------|------|
| **T0** | Seconds-minutes | ADS-B jets, AIS vessels, blockchain, power grid | PRIMARY — unforgeable, pre-event |
| **T1** | Hours-days | GDELT, dark pool, Polymarket, clinical trials | ACTIVE — behavioral traces as they happen |
| **T2** | 1 week | CFTC COT, FINRA ATS, fund flows | CONTEXT — big player positioning |
| **T3** | 2-45 days | Form 4, Form 144, congressional trades, 13F | CONFIRMATION — validates hypotheses |
| **T4** | 45+ days | Patent filings, lobbying disclosure, job postings | STRATEGIC — long-term behavioral shifts |

---

## Part 3: Complete Phase History (Phases 0–24)

### Summary Table

| Phase | Name | Layer | Status | Tests Added |
|-------|------|-------|--------|-------------|
| 0 | Agent End-to-End | Infrastructure | ✅ | — |
| 1 | Data Foundation | L1 (Surveillance) | ✅ | — |
| 2 | Global Liquidity Regime Detection | L2 (Quant) | ✅ | — |
| 3 | Scoring & Validation | L2 (Quant) | ✅ | — |
| 4 | Agent Autonomy + RL Layer | L5 (Learning) | ✅ | — |
| 5 | Full Observational Surface | L1 (Surveillance) | ✅ | — |
| 6 | Extended Observational Surface | L1 (Surveillance) | ✅ | — |
| 7 | Pipeline Layer | Infrastructure | ✅ | 356 |
| 7b | Global Deep Surveillance | L1 (Surveillance) | ✅ | — |
| 7c | Convergence Detection Layer | L4 (Fusion) | ✅ | 883 |
| 8 | Signal Protocol + Feature Engineering | L2 (Quant) | ✅ | 294 |
| 10a | Deep Surveillance Framework | L1 (Surveillance) | ✅ | — |
| 10b | L2 Tool Upgrades | L1 (Surveillance) | ✅ | — |
| 12 | Temporal Heterogeneous GNN | L3 (World Model) | ✅ | 242 |
| 13 | L2 Tool Expansion | L1 (Surveillance) | ✅ | 147 |
| 14–15 | Pattern Recovery + Fine-Tuning | L3 (World Model) | ✅ | — |
| 16 | GNN-Guided Tool Expansion | L1+L3 | ✅ | 34 |
| 17 | Entity Linking Layer | L3 (World Model) | ✅ | 95 |
| 18 | Tier 1 Tool Expansion | L1 (Surveillance) | ✅ | — |
| 19 | GNN ↔ World Model Bridge | L3+L4 | ✅ | 249 |
| 20 | Signal Fusion | L4 (Fusion) | ✅ | 249 |
| 21 | RL Policy (SAC) | L5 (Learning) | ✅ | 276 |
| 22 | Adversarial Layer | L6 (Adversarial) | ✅ | 148 |
| 23 | GNN-Guided Expansion R2 | L1+L3 | ✅ | 174 |
| **24** | **E2E Global Multi-Asset** | **All Layers** | **ACTIVE** | **112 so far** |

### What Each Major Phase Built

**Phase 0 — Agent End-to-End:** CLI entry point, orchestrator loop (research → plan → execute → synthesize), LLM client, task planner, memory store (episodic + semantic + working), tool abstraction, web search, code executor, file manager.

**Phases 1–6 — Surveillance Surface Buildout:** 57 data tools covering global equities, macro, central banks, commodities, FX, prediction markets, insider filings, dark pool, shipping, disease, patents, geopolitics, crypto, energy grid, weather, earthquakes, food security, job postings, lobbying, sanctions, DNS monitoring, internet infrastructure, and more. All free-tier APIs.

**Phase 7 — Pipeline Layer:** DAG execution engine (`FunctionOperator`, `DAG`, executor). PipelineStore (SQLite) for entity registration, observation storage, model artifacts, diagnostics. 9 DAGs: daily_collection, entity_scoring, convergence_detection, feature_generation, gnn_inference, world_model_update, rl_training, adversarial_scan, whale_tracking.

**Phase 7c — Convergence Detection:** Cross-source convergence scoring, temporal alignment, multi-signal fusion for detecting when multiple data sources agree.

**Phase 8 — Signal Protocol + Feature Engineering:** Normalized signal extraction (OFI, VPIN, Hurst exponent, transfer entropy, mutual information, Hawkes intensity). Walk-forward backtester with Sharpe, Sortino, Calmar, max drawdown, VaR/CVaR.

**Phase 12 — Temporal Heterogeneous GNN (HetTGN):** The core world model — a heterogeneous temporal graph neural network. 10 entity types (company, person, country, commodity, currency, organization, event, vessel, wallet, instrument). 24 observation types. Cross-type attention learns which entity-type pairs are informative. Temporal encoding captures time-varying dynamics.

**Phases 13–16 — GNN Expansion + Pattern Recovery:** L2 tool upgrades, pattern extraction from GNN embeddings, fine-tuning for specific signal detection, GNN-guided tool expansion (the GNN's attention weights tell us which data domains are starved and need deeper tools).

**Phase 17 — Entity Linking:** Unified entity graph — maps names, IDs, tickers, wallets, CIK numbers, vessel IMOs to a single identity layer. Cross-domain linking: company CIK in EDGAR = ticker in market data = vessels in AIS.

**Phase 19 — GNN ↔ World Model Bridge:** Connected the GNN embeddings to the Bayesian world model for belief propagation and causal inference.

**Phase 20 — Signal Fusion:** Surprise extraction (`EntitySurprise` with 5-tuple: obs_type, temporal, value, neighborhood, memory_drift). Entity scoring pipeline. Convergence-surprise fusion.

**Phase 21 — RL Policy (SAC):** Soft Actor-Critic for portfolio optimization. `SACTrainer` (state_dim, action_dim, config). `StateAssembler` builds fixed-size state tensor from entity alerts + beliefs + market features + adversarial flags. Replay buffer, reward function (portfolio return + risk adjustment + surprise bonus). Portfolio strategies bridge SAC to walk-forward backtester.

**Phase 22 — Adversarial Layer:** VPIN (Volume-Synchronized Probability of Informed Trading), edge decay monitoring, crowding risk detection, adversarial scanner. `AdversarialFlag` with severity and flag_type.

**Phase 23 — GNN-Guided Expansion R2:** Second round of GNN-guided tool expansion. Used attention weights to identify starved entity neighborhoods and prioritize new data sources.

**Phase 24 — E2E Global Multi-Asset Integration (ACTIVE):** Wire all 23 phases into a single pipeline that ingests 90 global instruments, trains the GNN with instruments, allocates multi-asset portfolios via SAC, backtests with attribution, and launches paper trading.

---

## Part 4: Phase 24 — Detailed State

### The Problem Phase 24 Solves

All 23 prior phases built the layers independently. Phase 24 wires them together:
- No end-to-end pipeline existed — layers never ran together
- No inference DAG — training DAGs existed but nothing produced daily portfolio weights
- Execution was locked to US equities — `AssetMapper` did `entity_id → single US ticker`
- Strategy ABC took 1-D returns (one asset at a time)
- Reward signal used synthetic returns, not real P&L

### The 90-Instrument Universe

Defined in `agent/tools/instrument_universe.py`:

| Asset Class | Count | Examples |
|-------------|-------|---------|
| Commodity futures | 20 | CL=F, GC=F, SI=F, HG=F, NG=F, ZW=F, KC=F, CT=F, LBS=F |
| FX pairs | 15 | EURUSD=X, GBPUSD=X, USDJPY=X, AUDUSD=X, USDMXN=X |
| Equity indices/ETFs | 25 | SPY, QQQ, IWM, EFA, EEM, FXI, EWJ, EWZ, INDA |
| Sector ETFs | 15 | XLF, XLE, XLK, XLV, XLI, XLP, XLU, XLB, XLRE |
| Fixed income | 10 | AGG, TLT, IEF, SHY, LQD, HYG, EMB, BNDX, TIPS |
| Vol + Crypto | 5 | ^VIX (not tradeable), BTC-USD, ETH-USD, UVXY, SVXY |
| **Total** | **90** | (89 tradeable, ^VIX observation-only) |

### Sub-Phase Status

| Sub-phase | Description | Status | Steps |
|-----------|-------------|--------|-------|
| **24a** | Instrument Universe + Daily Price Ingest | ✅ Complete | 6/6 |
| **24b** | GNN Training with Instruments | ✅ Complete | 3/3 |
| **24b.5** | GNN Attention Diagnostic | ⬜ 1/2 (write-up pending) | 1/2 |
| **24c** | Multi-Asset Strategy Refactor | ✅ Complete | 7/7 |
| **24d** | Inference DAG | ⬜ Not started | 0/4 |
| **24e** | Walk-Forward Backtest | ⬜ Not started | 0/4 |
| **24f** | Paper Trade Launch | ⬜ Not started | 0/3 |

### 24a: What Was Built (Complete)

- **`agent/tools/instrument_universe.py`**: `InstrumentDef` dataclass, 90-entry `INSTRUMENTS` tuple, `tradeable_instruments()`, `instruments_by_class()`, `ticker_to_instrument()`, `ingest_daily_prices()` (batch yfinance → entity registration → 3 obs types: return, volume, volatility)
- **`agent/models/gnn/graph_builder.py`**: Extended — 10th entity type "instrument", 3 new obs types, ENRICHMENT_DIM 30→33
- **`agent/pipeline/dags/daily_collection.py`**: Added `fetch_instruments` node (FunctionOperator, no deps, 300s timeout)
- **`tests/test_instrument_universe.py`**: 46 tests covering InstrumentDef, helpers, ingest with mocked yfinance, graph builder integration, DAG integration
- **`tests/test_graph_builder_expanded.py`**: Updated for 24 obs types, 10 entity types, feature dim 13

### 24b: What Was Built (Complete)

- **`scripts/backfill_instruments.py`**: 2yr daily data backfill for all 90 instruments, day-by-day entity registration, idempotent
- **HetTGN training verified**: Loss convergence, non-zero instrument embeddings, cross-type attention activation, per-instrument surprise extraction
- **`tests/test_backfill_instruments.py`**: 20 tests (15 fast + 5 slow), synthetic 5-instrument + 10-entity graph, 50-epoch training

### 24c: What Was Built (Complete)

**3 files modified, 1 file created:**

1. **`agent/quant/backtest.py`** — Appended ~250 lines:
   - `MultiAssetStrategy(ABC)`: `generate_weights(train_returns, test_length, instrument_names) → np.ndarray (test_length, N)`
   - `MultiAssetFoldResult`: fold metrics, weights (T,N), portfolio_returns (T,), per_instrument_returns (T,N)
   - `MultiAssetBacktestResult`: aggregate metrics, equity curve, attribution dict (asset_class → cumulative return)
   - `MultiAssetWalkForward`: expanding-window, min_train=252, test_size=21, dot-product P&L, `score_returns()` per fold, concentration metrics, `_compute_attribution()` by asset class
   - `EqualWeightStrategy`: returns `1/N` for all instruments
   - `BuyAndHoldBenchmarkStrategy`: fixed target weights, auto-generates name

2. **`agent/learning/policy/state_assembler.py`** — Appended ~120 lines:
   - `InstrumentStateAssembler`: state_dim = N×5 + E×5 + E×4 + M + 1 + 4
   - State layout: `[inst_surprise(N,5) | entity_surprise(E,5) | belief(E,4) | market(M) | count(1) | adversarial(4)]`
   - Fixed instrument ordering via `_ticker_index` (SAC action dim i = instrument i)

3. **`agent/learning/policy/portfolio_strategy.py`** — Appended ~60 lines:
   - `MultiAssetSACStrategy(MultiAssetStrategy)`: wraps `SACTrainer` + `InstrumentStateAssembler`
   - Per-timestep: assemble state → SAC `select_action(deterministic=True)` → N-dim weight vector

4. **`tests/test_multi_asset_backtest.py`** — 46 tests:
   - ABC contract, EqualWeight (5 tests), BuyAndHold (5 tests), WalkForward (10 tests), Edge cases (9 tests), InstrumentStateAssembler (7 tests), MultiAssetSACStrategy with mocked SAC (8 tests)
   - Key verified properties: dot-product P&L correctness, attribution sums to total, fixed instrument ordering, NaN propagation, wrong-shape rejection

### Key Design Decisions in 24c

1. **Parallel hierarchy**: `MultiAssetStrategy` alongside `Strategy`, not replacing it. Backward compat preserved.
2. **Dot-product P&L**: `portfolio_return_t = Σ(weights_t × returns_t)` — standard log-return portfolio formula.
3. **Attribution by construction**: per-instrument weighted returns grouped by asset class. Sums to total by algebra.
4. **`score_returns()` reuse**: Both 1-D and multi-asset walk-forward use the same scoring function.

---

## Part 5: The Codebase Today

### Aggregate Metrics

| Metric | Count |
|--------|-------|
| Total Python files in `agent/` | 176 |
| Total lines of production Python | ~68,800 |
| Total test files | 165 |
| Total test functions | ~7,843 |
| Total lines of test Python | ~97,500 |
| **Total Python LOC** | **~166,200** |

### Module Inventory

**Surveillance Surface (`agent/tools/`):** 60 files, 57 data tools + 3 infrastructure (base, code_executor, shell_runner).

**Math Layer (`agent/quant/`):** 7 files — backtest.py (1-D + multi-asset walk-forward), changepoint.py (BOCPD), regime.py (HMM), spectral.py (FFT + CWT), scoring.py (Sharpe, Sortino, Calmar, drawdown, VaR/CVaR), liquidity.py (composite), regime_strategy.py.

**World Model (`agent/models/`):** 14 files — world_model.py, belief.py, graph.py, propagator.py, state_filter.py, discovery.py, intervention.py, initial_graph.py + GNN subdirectory: het_tgn.py (HetTGN), graph_builder.py, trainer.py, temporal.py, integration.py, pattern_extractor.py.

**Signal Fusion (`agent/fusion/`):** Surprise extraction (`EntitySurprise` 5-tuple), entity scoring, convergence detection.

**RL/Learning (`agent/learning/`):** 21 files — SAC policy (sac.py, config.py), state_assembler.py (original + InstrumentStateAssembler), portfolio_strategy.py (1-D + multi-asset), reward_fn.py, replay_buffer.py, asset_mapper.py, weight_learner.py, symlog.py + bandit.py, evaluator.py, goal_generator.py, reflection.py, reward.py.

**Adversarial (`agent/adversarial/`):** 6 files — vpin.py, edge_decay.py, crowding.py, scanner.py, flags.py, config.py.

**Pipeline (`agent/pipeline/`):** PipelineStore (SQLite), DAG executor, 9 DAGs: daily_collection, entity_scoring, convergence_detection, feature_generation, gnn_inference, world_model_update, rl_training, adversarial_scan, whale_tracking.

### Key Interfaces for Next Work

| Interface | Location | Signature |
|-----------|----------|-----------|
| `MultiAssetStrategy.generate_weights()` | `agent/quant/backtest.py` | `(train_returns, test_length, instrument_names) → (T, N)` |
| `MultiAssetWalkForward.run()` | `agent/quant/backtest.py` | `(strategy, returns_2D, extra) → MultiAssetBacktestResult` |
| `InstrumentStateAssembler.assemble()` | `agent/learning/policy/state_assembler.py` | `(inst_surprises, alerts, beliefs, market, ...) → (tensor, meta)` |
| `MultiAssetSACStrategy.generate_weights()` | `agent/learning/policy/portfolio_strategy.py` | Uses SAC + InstrumentStateAssembler per timestep |
| `SACTrainer.select_action()` | `agent/learning/policy/sac.py` | `(state_tensor, deterministic) → np.ndarray` |
| `SACTrainer.save() / load()` | `agent/learning/policy/sac.py` | `save() → bytes`, `load(data, state_dim, action_dim) → SACTrainer` |
| `PipelineStore` | `agent/pipeline/store.py` | Entity CRUD, observation CRUD, model artifacts, diagnostics |
| `FunctionOperator` / `DAG` | `agent/pipeline/` | DAG node definition and execution |
| `score_returns()` | `agent/quant/scoring.py` | `(returns, risk_free, periods_per_year, weights, benchmark) → dict` |

---

## Part 6: What's Next — Phases 24d Through 24f

### Phase 24d: Inference DAG (Next)

Build the daily inference pipeline that produces portfolio weights from trained models.

**24d.1: Database Tables** — Add to `agent/pipeline/store.py`:
- `portfolio_weights` table: date, ticker, weight, metadata, created_at
- `paper_trade_pnl` table: date, portfolio_return, benchmark_return, cumulative_return, metadata, created_at
- CRUD: `store_portfolio_weights()`, `query_portfolio_weights()`, `store_paper_pnl()`, `query_paper_pnl()`

**24d.2: Inference DAG** — Create `agent/pipeline/dags/inference.py`:
1. `load_models` — load HetTGN + SAC from PipelineStore
2. `gnn_inference` — build today's graph → forward pass → extract surprises
3. `sac_inference` — InstrumentStateAssembler → SAC deterministic action → weights
4. `emit_portfolio` — write weights to DB, compute yesterday's P&L

**24d.3: Register** in `agent/pipeline/dags/__init__.py`

**24d.4: Tests** — `tests/test_inference_dag.py`: DAG structure, mocked execution, weight writes, P&L computation, no-model graceful failure

### Phase 24e: Walk-Forward Backtest

Run multi-asset walk-forward on 2yr backfilled data: 252d train, 21d test, monthly folds, 5 strategies (SAC, WeightedSurprise, EqualWeight, SPY-only, 60/40). Generate attribution report per-asset-class, per-region, top instruments, concentration analysis.

### Phase 24f: Paper Trade Launch

Configure inference DAG cron (19:45 UTC weekdays). Implement alert conditions (drawdown >5%, concentration >30%, edge decay on held instruments, Sharpe < -0.5 at 30d). Write end-to-end smoke test.

### After Phase 24

Phase 24f completion means TirraMind has its first live paper-trading pipeline. From there:

- **Phase 25**: L2 tool expansion guided by GNN diagnostic output (24b.5.2)
- **Beyond**: Live trading with real capital (after demonstrated edge), expanding to more exotic data sources, L3 cross-entity intelligence, autonomous data source discovery

---

## Part 7: Cold-Start Instructions

To resume work on this project:

1. **Read this checkpoint** — you now have complete context
2. **Read the active task file**: `[[e2e_global_integration]]` — source of truth for what's done and what's next
3. **Read the spec**: `[[e2e_global_integration_spec]]` — detailed interface specs for 24d-24f
4. **For 24d specifically**: read `agent/pipeline/store.py` (DB patterns), `agent/pipeline/dags/daily_collection.py` (DAG patterns), `agent/learning/policy/sac.py` (model save/load)
5. **Follow the workflow**: research → spec → implement one atomic step → test → mark done → next

### Key Documents in the Knowledge Graph

| Document | Path | Purpose |
|----------|------|---------|
| Project Memory | `[[project_memory]]` | Persistent architectural knowledge |
| Active Task | `[[e2e_global_integration]]` | Phase 24 task checklist |
| Spec | `[[e2e_global_integration_spec]]` | Detailed implementation spec |
| Research | `[[e2e_global_integration]]` | Phase 24 research note |
| Master Tracker | `[[quant_training_ground]]` | All-phases status |
| This Checkpoint | `[[chat_checkpoint_2026-04-13_master]]` | You are here |

### Obsidian Navigation

The project root is an Obsidian vault. Use `[[wiki links]]` for all cross-references. Navigate by:
- **Cold-start**: latest checkpoint → active task → linked research/spec
- **Find backlinks**: `grep_search` for `[[filename]]` across docs/tasks/wiki
- **Find topic clusters**: `grep_search` for tag string in frontmatter (e.g., `topic/convergence`)
- **Follow the triad**: `[[research_note]]` → `[[spec]]` → `[[task]]`

---

## Related

- [[e2e_global_integration]] — Active task + research doc
- [[e2e_global_integration_spec]] — Full spec for 24a–24f
- [[quant_training_ground]] — Master phase tracker
- [[project_memory]] — Persistent architectural knowledge
- [[chat_checkpoint_2026-04-13_session3]] — Prior checkpoint (24c completion)
- [[chat_checkpoint_2026-04-13_session2]] — Prior checkpoint (Phase 24 planning)
- [[temporal_het_gnn]] — Phase 12 GNN architecture
- [[signal_fusion]] — Phase 20 surprise extraction
- [[rl_policy]] — Phase 21 SAC policy
- [[adversarial]] — Phase 22 adversarial layer
- [[gnn_guided_expansion_r2]] — Phase 23 (last completed before 24)
