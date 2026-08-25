---
title: "Research: Data Strategy Doctrine (Phase 47 Foundation)"
tags:
  - doc/research
  - phase/47
  - topic/data-strategy
  - topic/backfill
  - topic/training-data
  - layer/surveillance
  - status/active
---

# Research: Data Strategy Doctrine (Phase 47 Foundation)

> **Guiding question.** What data must TirraMind's GNN + world model + RL policy have *seen* — in depth, density, diversity, and temporal coverage — before it can generalise reliably, without catastrophic collapse into one modality, one regime, or one accidental correlation?
>
> **Guiding answer (preview).** Not more data. *Structurally balanced* data: enough temporal depth per entity type to cover multiple regimes, enough modal diversity that no single source dominates, enough density per edge type that the GNN's attention heads don't starve, and enough orthogonality between sources that shocks in one modality carry information about hidden states rather than just repeating a correlated signal.

This note is the governing research artifact for Phase 47 (Historical Backfill). It overrides any implicit assumption that "more years = better data". It is grounded in verified external sources — not memory-based guesses — and defines the depth, density, and diversity targets the backfill runner must meet.

---

## 1. Objective: why data quality dominates model quality in this system

The TirraMind stack follows a POMDP structure: hidden states evolve in a non-stationary environment, actors have latent intentions, and rewards are sparse and delayed. Every architectural layer — HetTGN, pgmpy DAG, Kalman fusion, SAC, EWC — is an inference method that *assumes the training distribution is representative of the state space it will encounter in deployment*.

If the training data:

- covers only calm regimes → the HMM never learns a turbulent transition matrix
- is dominated by one observation type → the GNN attention heads collapse onto that modality
- lacks temporal depth → the EWC Fisher diagonal has no meaningful "old task" to protect
- has temporal leakage → the world model encodes tomorrow's information as if it were yesterday's

…then no amount of parameter tuning, no architecture upgrade, and no ensembling fixes it. Every downstream signal is contaminated. This is the classic "garbage in, garbage out" failure mode, but compounded by the fact that a POMDP with a biased belief state does not *know* it is biased — it is overconfident in a wrong state.

**Therefore: the data strategy is a first-class architectural artifact, not a collection script.**

---

## 2. The prediction target (determines what data we need)

The TirraMind system emits probability distributions over Layer 2–3 events: information shifts, price moves, geopolitical changes, supply disruptions. To do that, it must *infer* hidden Layer 0 states (physical reality: ships, factories, grid, disease) and Layer 1 behavioral states (policy, positioning, filings, social) from partial observations.

This dictates two invariants for the data:

1. **Raw observations are primary; prices are derivative.** A price is the *result* of hidden-state evolution that we are trying to infer. If we train primarily on prices, we are asking the GNN to invert a function whose domain (hidden states) it has never seen directly. We must collect the raw drivers — vessel movement, CFTC positioning, insider filings, permit issuance, grid load, earthquake activity, disease surveillance — and let the system learn their relationship to prices. Prices can extend back further than some raw sources, but the raw sources are non-negotiable.
2. **Observations must resolve to entities, not aggregates.** A country-level PMI number is an aggregate; a single insider's Form 4 filing is an entity-level event. The GNN needs entity-resolved observations to learn how individual actors behave across time and across modalities. Aggregate signals remain useful as *conditioning variables* on country or market nodes, but they cannot be the backbone of a graph model.

---

## 3. The four backfill dimensions (the core framework)

Raw volume is not the metric. The backfill must be evaluated on four orthogonal dimensions, each of which can fail independently:

| Dimension | Definition | Failure mode if deficient |
|---|---|---|
| **Depth** | Temporal reach per entity (how many years of history we have for a given entity × observation type) | Model never sees old regimes; HMM transition matrix is calibrated to one epoch |
| **Density** | Observations per entity per unit time (how frequently each entity emits signal) | Attention heads for that entity type starve; representations are undertrained |
| **Coverage** | Number of distinct entities of each type in each country/domain | Graph is sparse in whole neighborhoods; GNN cannot learn local structure |
| **Diversity** | Number of *independent* observation channels per entity | Modal collapse: GNN attends to dominant modality and ignores others |

A backfill that adds 10 years of one modality but neglects the other three is worse than a backfill that adds 5 years evenly across all four. **More data in one direction can actively hurt the model by reinforcing modal imbalance.** This is the single most important lesson.

---

## 4. Sample complexity requirements (what "enough" means per layer)

Each layer of the stack has a different sample-complexity profile. The backfill depth target for each observation type must be calibrated against the layer that consumes it most heavily.

### 4.1 Temporal GNN (Layer 2 — the heaviest consumer)

Foundational reference: Rossi et al. 2020, "Temporal Graph Networks for Deep Learning on Dynamic Graphs" (arXiv:2006.10637). TirraMind's HetTGN is a direct descendant.

Critical finding: Hayes, Schumacher, Strohmaier 2025, "What Do Temporal Graph Learning Models Learn?" (arXiv:2510.09416) — temporal GNN models show a *mixed picture* in learning fundamental graph properties (density, recency, edge persistence, periodicity, homophily, preferential attachment). Models capture some well, fail on others. Implication: to reliably learn all eight properties, the training distribution must *exhibit variation* in each. A graph trained on one regime where density is constant will never learn a density-dependent predictor.

**Consequence for backfill:** depth must span at least 2 full cycles of each learnable property. For regime variables, that means at least 2 full business cycles (typically 8–12 years of macro data). For edge persistence, it means enough history to see long-lived relationships (e.g., a persistent insider–company link across 5+ years of filings).

### 4.2 World model (Layer 3 — Bayesian DAG via pgmpy)

The world model estimates conditional probability tables across ~100 nodes. The standard frequentist rule of thumb for reliable CPT estimation is ≥10 observations per parameter; for a node with 3 parents of 3 states each, this is 3³ × 10 = 270 observations minimum. For 100 nodes with average 2 parents each, total minimum training events ≈ 10,000. Below that, posterior uncertainty dominates.

### 4.3 Continual learning (Layer 5 — EWC + replay)

EWC (Kirkpatrick et al. 2017, PNAS) protects old-task parameters via the Fisher information diagonal. Known failure: Jones & Sprague 2018 ("Expandable EWC") show that standard EWC *diverges within ~18 tasks* without λ decay or model expansion. Implication for a non-stationary system: if we treat each regime transition as a "task", we cannot carry more than ~18 regime segments in memory without adaptation.

Proto-based replay (arXiv 2602.09720, "Continual Learning for non-stationary regression via Memory-Efficient Replay") outperforms raw replay buffers on non-stationary data because prototypes compress the distribution rather than sampling it. This is relevant for our Phase 46 online GNN: the replay buffer strategy should evolve toward prototype clustering as depth grows.

**Consequence for backfill:** the backfill must produce *distinct, labeled regime segments* (calm / turbulent / crisis / recovery), not just a continuous stream. This is what allows EWC to consolidate meaningfully.

### 4.4 POMDP belief state (the integrating layer)

Standard POMDP (Cassandra et al.; Barto & Sutton notes) assumes stationary transition probabilities. The financial/geopolitical environment is not stationary — it is a Time-Varying POMDP (TV-POMDP). Reference: Mornik et al. 2024 ("Learning and Planning in a Time-Varying Partially Observable Environment") introduces Memory Prioritized State Estimation (MPSE), weighting recent observations more heavily in belief updates while preserving access to older state estimates.

**Consequence for backfill:** observation timestamps must be exact (not `now()`), and the storage must preserve recency ordering. MPSE is useless if the data has been chronologically flattened.

### 4.5 Regime detection (HMM on Layer 2 features)

Reference: Song 2010, "Detecting Structural Breaks using HMM"; infinite-HMM variants (Song & Eraker via Journal of Applied Econometrics) avoid pre-specifying the number of regimes. Standard finding: reliable HMM estimation needs *at least 30 observations per state*. For a 3-regime model, minimum ~90 observations; for financial daily data, this is ~4 months minimum per regime. To see 2 full calm→turbulent→crisis cycles, we need ~10 years of daily data minimum.

### 4.6 Feature selection (Layer 2 → Layer 3 compression)

Reference: Peng et al. 2005 (mRMR); Brown 2012; arXiv:2207.08476 ("High-Order Conditional Mutual Information Maximization"). Conditional mutual information (CMI) feature selection requires:

$$I(X; Y \mid Z) = H(X \mid Z) - H(X \mid Y, Z)$$

To estimate this reliably at high dimension, we need N ≥ k^d samples where k is the average cardinality of discretized features and d is the conditioning dimension. For d = 3 and k = 5, N ≥ 125 per feature combination, and we scan hundreds of combinations, so N ≥ 10,000 samples per entity type. This matches the world-model bound above.

---

## 5. Modal diversity and information balance

A heterogeneous GNN with imbalanced modalities collapses onto the dominant one. References: GraphSMOTE (Wang et al., WSDM 2021), Tail-GNN (Fang, KDD 2021), CM-GCL (Qian et al., NeurIPS 2022) all document this failure mode.

**Current TirraMind symptom (2026-04-23 DB state):** 93.8% of 74,030 observations are `instrument_daily` from one tool (`instrument_universe`). The GNN *cannot* learn surveillance-domain structure from this — every attention head will be dominated by daily-return noise. The remaining 6.2% is spread across 11 other tools, each with insufficient density.

**Fix: balance targets.**

| Observation category | Current share | Target share after Phase 47 |
|---|---|---|
| Price/market (Layer 0 — exogenous price proxy) | 94% | 20–30% |
| Physical/activity (Layer 0 — ships, grid, permits, earthquakes) | <1% | 15–20% |
| Disclosure (Layer 1 — insider, form144, sanctions, lobbying) | <1% | 15–20% |
| Positioning (Layer 1 — CFTC, FINRA short, polymarket) | <1% | 10–15% |
| Macroeconomic (Layer 1 — FRED, comtrade, capital flows) | <1% | 10–15% |
| Information/narrative (Layer 1 — GDELT, wikipedia, academic) | <1% | 10–15% |

These are *observation-type shares*, not entity shares. The goal: no single category exceeds 30% of total observations.

### Exogenous vs endogenous balance

Reference: arXiv:2509.05779 ("Select, then Balance: Exogenous-Aware Spatio-Temporal Forecasting"). Exogenous variables (weather, policy, disease) carry information orthogonal to the target; endogenous variables (price, volume, momentum) are downstream of the state we want to predict. A model trained mostly on endogenous signals learns to predict price-from-price — which is autoregressive, not causal. We want exogenous : endogenous ≥ 2:1 in the final training distribution.

---

## 6. Temporal principles

### 6.1 Timestamp fidelity

Every observation must store `observed_at` as the *actual event time*, not the ingestion time. Violations kill the entire pipeline:
- SAC's on-policy updates see misordered transitions
- Kalman fusion estimates the wrong belief state
- GDELT article timestamps must use `SQLDATE`, not download time
- SEC filings must use `filedAt` from the header, not the scrape time

**Phase 47 gate:** Before the backfill runs, every tool must be verified to accept and emit correct historical timestamps. No "assume it works" — probe each one.

### 6.2 No look-ahead bias

If a CFTC report for 2020-03-15 was *published* on 2020-03-20, the model must see it at 2020-03-20, not 2020-03-15. Point-in-time discipline is non-negotiable. The `observed_at` field should be the *event* time; the `available_at` field (to be added) should be the publication/knowability time. The model trains on `available_at` ordering.

### 6.3 Resolution matching

A daily price observation and a quarterly 10-Q filing should not be joined as if they update at the same cadence. The Kalman filter must carry different process-noise assumptions per source. The backfill must preserve native resolution and let the feature layer decide how to align.

---

## 7. Depth targets per tool (the concrete output)

Targets derived from: source availability (verified), sample-complexity bounds above, and modal-balance targets in §5.

### 7.1 Full-depth sources (verified deep archives)

| Tool | Source | Earliest available | Backfill target | Verification |
|---|---|---|---|---|
| `market_data` | yfinance | 1990s for most tickers | `period="max"` | github.com/ranaroussi/yfinance |
| `macro_data` | FRED | varies; many series 1947+ | 30 years (1996+) | fred.stlouisfed.org — 840K+ series |
| `cftc` | CFTC.gov | Futures-only: 1986 | 33 years (1993+, when EDGAR starts — aligned epoch) | cftc.gov/MarketReports/CommitmentsofTraders |
| `gdelt` | GDELT Project | v1 from Jan 1979 | 15 years (2010+) — v2 starts 2015, but v1 goes back 46y | gdeltproject.org |
| `insider_filings` | SEC EDGAR | 1993 | 25 years via 100 sliding 90-day windows | efts.sec.gov |
| `form144` | SEC EDGAR | 1994+ | 25 years via 150 sliding 60-day windows | efts.sec.gov |
| `sanctions_monitor` | OFAC | 1995+ | 25 years via 25 sliding 365-day windows | treasury.gov/ofac |
| `finra_short_volume` | FINRA | 2015+ | 11 years via 200 sliding 20-day windows | finra.org |

### 7.2 Medium-depth sources (policy/filing data)

| Tool | Earliest available | Backfill target | Notes |
|---|---|---|---|
| `gov_contracts` | USAspending 2001 (FOIA) | 22 years (2004+, reliable post-FFATA) | usaspending.gov |
| `lobbying` | senate.gov 2000 | 26 years (2000+) | soprweb.senate.gov |
| `patent_filings` | USPTO 1976 | 26 years (2000+) | uspto.gov |
| `sovereign_debt` | IMF | 25 years (2001+) | imf.org |
| `comtrade` | UN Comtrade | 15 years (2011+; earlier data patchy) | comtradeplus.un.org |
| `consumer_sentiment` | BLS/Eurostat | 20 years (2006+) | bls.gov / eurostat |
| `capital_flows` | IMF BOP | 20 years (2006+) | data.imf.org |
| `central_bank_balance` | Fed/ECB/BOJ | 20 years (2006+) | fred.stlouisfed.org |
| `labor_disruptions` | BLS | 20 years (2006+) | bls.gov |
| `food_security` | FAO | 20 years (2006+) | fao.org |
| `energy_supply` | EIA | 20 years (2006+) | eia.gov |
| `earthquake_proximity` | USGS | 20 years (2006+) | earthquake.usgs.gov |

### 7.3 Short-depth sources (accept limitations)

| Tool | Earliest available | Backfill target | Notes |
|---|---|---|---|
| `polymarket` | 2022 (Gamma API) | Full history (~3 years) | gamma.polymarket.com |
| `polymarket_whales` | 2022 | Full history | gamma.polymarket.com |
| `ais_vessel` | Free tier: 1–2 years | Full available | Known limitation |
| `satellite_activity` | 2021+ (Sentinel-2 free) | 3 years | copernicus.eu |
| `defi_flows` | on-chain — years available but API-gated | Verify per endpoint | thegraph.com |

### 7.4 Live-only (do not backfill)

`cert_transparency`, `dns_monitor`, `internet_outages` — these are real-time observation streams with no historical archive from their free sources. Do not waste effort trying to backfill. Accept them as live-only signals starting from the Phase 47 completion date.

---

## 8. Core risks and mitigations

| Risk | Mechanism | Mitigation |
|---|---|---|
| **Modal collapse** | One observation type dominates; attention heads ignore others | Enforce §5 balance targets; reject backfill run if any category >40% |
| **Regime amnesia** | All training data from one regime (e.g., 2023 low-vol); HMM learns wrong transition matrix | Require ≥2 distinct market regimes in the training window (easiest: include 2020 COVID shock + 2022 rate cycle + 2008 GFC if possible) |
| **Survivor bias** | Only currently-existing entities are in the instrument universe | Augment with historical delistings where free data permits (market_data handles this via yfinance; entity graph must not prune delisted tickers during backfill) |
| **Timestamp corruption** | Tool records ingestion time instead of event time | Phase 47 gate: pre-flight test each tool with known-date samples; verify `observed_at` matches source timestamp |
| **Look-ahead bias** | Publication delay ignored | Add `available_at` column; train on `available_at` ordering (Phase 47b work) |
| **API rate-limit failure mid-backfill** | Tool halts; partial data; inconsistent state | Checkpoint per tool; resumable runner; retry with exponential backoff |
| **Disk / memory exhaustion** | 25 years of 51 tools ≈ terabytes of raw | Stream-write to SQLite; compress old partitions; run tools serially, not in parallel |
| **Duplicate writes** | Backfill re-runs same dates; double-counts | Upsert on `(entity_id, observation_type, observed_at)` unique index |
| **Free-API quota hit** | UN Comtrade 100 req/hour, etc. | Respect quota; schedule long-running tools over multiple days |
| **Silent source drift** | FRED series discontinued mid-backfill | Validate series IDs before run; record source version in each observation |

---

## 9. The invariant: preserve structure > maximize volume

If the choice is between:

- (A) 10 million more daily price observations
- (B) 100,000 new physical/disclosure observations across under-covered entity types

**Always choose B.** Adding (A) reinforces modal collapse; adding (B) fills structural gaps that the GNN currently cannot learn from any amount of (A). This is the cheapest-data-is-most-valuable doctrine operationalised.

**Practical rule:** after each backfill batch, run `scripts/density_audit.py` (to be built). Exit FAIL if:
- any observation category exceeds 40% of total
- any entity type has <10 observations/entity on average
- any country has fewer than 5 entities of the top 3 types

A failed audit triggers a *re-prioritisation* of the next batch (not an extension of the previous dimension).

---

## 10. GNN-guided expansion (Signal Depth Doctrine link)

Per `[[signal_protocol]]` and `[[depth_model]]`, each data tool has a depth roadmap (L1 aggregate → L2 entity-resolved → L3 cross-entity combinations). Phase 47 backfill collects at the *existing* depth of each tool — it does not upgrade tools to L2. That upgrade is a separate phase, guided by GNN attention-weight diagnostics after Phase 40 (real GNN retrain).

The logic: we cannot decide which tools *need* L2 upgrades until we can measure which entity neighborhoods are starved under the GNN. The backfill is the *prerequisite* for that measurement, not a replacement for it.

---

## 11. Exit conditions for Phase 47

The backfill runner succeeds only when all of the following are true:

1. All tools in Groups A and B of `[[historical_backfill]]` have completed without checkpoint failures
2. Density audit passes (§9 criteria met)
3. Modal balance check passes (§5 targets met within ±5pp)
4. Regime coverage check passes: the training window includes ≥2 distinct volatility regimes detected by BOCPD (at least one period with VIX >30 and one with VIX <15)
5. Timestamp fidelity spot-check passes: sample 100 random observations per tool, verify `observed_at` matches source record
6. Total training observations ≥ 2M (minimum for world-model CPT estimation per §4.2)
7. Per-entity-type depth meets §7 targets within ±10%

Failure on any of these triggers targeted re-runs, not a project-wide restart.

---

## 12. Redundancy, overlap, and the CMI plateau

The mRMR (minimum Redundancy Maximum Relevance) criterion formalises a well-known empirical pattern: adding a feature that is conditionally independent of the target given already-selected features contributes zero mutual information. In practice, CMI gain flattens to <1% per additional feature after a source cluster is saturated. This is the *CMI plateau*.

For TirraMind it manifests at two levels:

**Within-source redundancy.** After ingesting ~10 years of FRED macro series for a country node, additional FRED series from the same theme (e.g., more regional employment variants when one national employment series already exists) add almost nothing — the marginal information is shared. The plateau within a single source family is reached quickly.

**Cross-source orthogonality.** A model trained heavily on price data and CFTC positioning learns the correlation between the two — but this is one information cluster. Adding GDELT conflict events (an exogenous, structurally independent source) opens a new CMI dimension not captured by the price/positioning cluster. Cross-source additions are always higher value than within-source extensions once the first source is adequately sampled.

**Practical rule.** Track per-source CMI contribution at each density audit (§9). When a source's marginal CMI gain falls below 5% of the highest-contributing source, redirect backfill budget to the next source cluster instead of deepening the saturated one.

**Formal bound.** The CMI estimator

$$I(X; Y \mid Z) = H(X \mid Z) - H(X \mid Y, Z)$$

requires N ≥ k^d samples to be reliable at conditioning depth d with cardinality k (arXiv:2207.08476). At d=3, k=5 this is ≥125 samples per feature combination. The takeaway: with O(10,000) observations per entity type, we can reliably evaluate at most depth-3 CMI. Adding more observations of the same type beyond that does not improve feature-selection quality — adding new types does.

**Key implication for Phase 47.** The backfill is not a race for volume; it is a race for CMI coverage across source clusters. The moment adding more data from source A produces less CMI gain than adding the first data from source B, stop A and start B.

Reference: Brown et al. 2012. "Conditional likelihood maximisation: a unifying framework for information-theoretic feature selection." JMLR 13. arXiv:2207.08476.

---

## 13. Graph structural requirements

The HetTGN's attention mechanism degenerates when the graph does not maintain minimum structural properties. These are hard requirements — not soft targets — that the density audit must enforce:

| Requirement | Minimum threshold | Why it matters |
|---|---|---|
| Distinct entity types per snapshot | ≥ 5 | HGTConv type-specific linear projections collapse to a homogeneous GNN with fewer types |
| Average node degree (any type) | ≥ 3 | Sub-threshold density → isolated nodes → zero-information message-passing |
| Cross-type edge ratio | ≥ 50% of entities have ≥ 1 cross-type edge | Without cross-type edges, multi-head attention is equivalent to a homogeneous GNN — heterogeneity is wasted |
| Temporal edge density | ≥ 1 new edge per entity per 7-day window | Time2Vec and HeteroMemory GRU require temporal events; a static snapshot kills both components |
| Minimum observations per entity type | ≥ 30 | Below this, CPT parameter estimation (§4.2) is dominated by Dirichlet prior rather than data |
| Entity survival ratio | ≥ 60% of entities appear in ≥ 3 non-consecutive windows | Ephemeral entities cannot contribute to temporal pattern learning |
| Regime span per entity type | ≥ 2 distinct volatility regimes in training window | Single-regime training → degenerate HMM transition matrix (§4.5) |

**Phase 47 gate.** `scripts/density_audit.py` must validate all seven requirements after each backfill batch. Any failing row is a blocking deficiency — not a warning.

**Consequence for tool prioritisation.** Tools that create *new entity types* with cross-type edges (e.g., `gov_contracts` linking company→government, `insider_filings` linking person→company) are higher priority than tools that add observations to already-dense entity types. Structure before volume.

Reference: Hu et al. 2020. "Heterogeneous Graph Transformer." WWW. arXiv:2003.01332.

---

## 14. System boundary (what this doctrine does NOT govern)

This data strategy governs only **free public sources ingested via TirraMind's existing tool layer** and its planned extensions. The following are explicitly out of scope:

1. **Sub-daily (high-frequency) data.** Tick data, Level 2 order book, millisecond trade prints. These require time-series databases, different model families (sequential order-flow models), and different regulatory treatment. Out of scope until Phase 50+.

2. **Proprietary or paid data.** Bloomberg Terminal, Refinitiv, S&P Global, FactSet, Orbital Insight, SpaceKnow, etc. These change the cost model and eliminate the free-data moat. Introduce only if post-Phase 40 backtests demonstrate the free stack has verifiably hit a ceiling.

3. **Synthetic data.** GAN-generated time series, bootstrapped scenarios, agent-based market simulations. Synthetic data can amplify existing biases and introduces unverifiable distributional assumptions. Not forbidden in principle but requires a dedicated research note and ADR before introduction.

4. **Social media fire-hose.** Twitter/X raw stream, Reddit raw dump. GDELT already processes social text at scale and provides structured events. Adding raw social on top produces within-source CMI plateau gains (§12) while introducing moderation, storage, and legal complexity.

5. **Real-time streaming beyond the daily DAG cadence.** Sub-hourly streaming requires different pipeline architecture (Kafka, Flink). Not governed here.

6. **Satellite imagery (beyond Sentinel-2 free tier).** Commercial satellite tasking (Planet Labs, Maxar) is paid. Free Copernicus Sentinel-2 bands are in scope; commercial providers are not.

**Why the boundary matters.** Every data type crossing this boundary changes the system's cost model, infrastructure requirements, or legal exposure. Expansions beyond the boundary require a new research note and an ADR, not just a new tool file.

---

## 15. Statistical integrity requirements

These requirements apply to every observation written by every tool, regardless of source. They are the preconditions for valid downstream inference.

**15.1 Completeness over imputation.** If a value is not observed, write NULL or mark the observation absent. Do not impute zeros, carry-forward values, or interpolated estimates into raw observations. Imputation is a feature-engineering decision and belongs in Layer 2 (`agent/quant/`), not in the `entity_observations` table.

**15.2 Unit consistency.** All monetary values in USD (or normalised to USD with exchange rate logged). All rates as decimals, not percentages. All timestamps in UTC. Violations propagate silently into the Kalman fusion layer and produce miscalibrated belief states.

**15.3 Source version and vintage tracking.** Each observation must record which data revision it came from. FRED regularly revises historical series (ALFRED vintages). CFTC issues corrected COT reports. If a series is revised, write the revision as a new observation (updating `value` and `updated_at`) — do not delete the old record. This preserves the point-in-time view needed for look-ahead-clean backtesting.

**15.4 Deduplication semantics.** The `(entity_id, observation_type, observed_at)` triple must be unique. Use UPSERT (not INSERT) on conflict. Re-runs of the backfill must not inflate the observation count — only `updated_at` should change on a duplicate write.

**15.5 Survivorship and selection bias disclosure.** When a data source returns only active/surviving entities (e.g., yfinance silently drops delisted tickers; EDGAR search returns only existing filers), document this limitation in the tool's research note. The backfill runner must explicitly attempt to retrieve historical delistings where the source supports it, or annotate which entity types are survivorship-biased.

**15.6 Timestamp fidelity (restatement of §6.1 as a hard gate).** `observed_at` is the *event time*, never the ingestion time. Before the backfill of any tool begins, probe it with a known-date historical sample and verify the `observed_at` it writes matches the source record. A tool that fails this check must be fixed before its backfill data enters the database.

---

## 16. Final principle: the self-updating doctrine

This document defines the data strategy for Phase 47. It was calibrated against:
- current sample-complexity bounds derived from the TirraMind stack as of 2026-04-23
- source availability verified via official documentation
- modal imbalance measured on 74,030 live observations (93.8% instrument_daily)
- external architecture review findings (five priority gaps, see `[[tirramind_structure]]`)

**It will be wrong after Phase 40.** After Phase 40 produces the first real GNN retrain on backfilled data, the following outputs replace this document as the authoritative guide for further data acquisition:
- GNN attention-weight diagnostics (which entity types and source clusters are still starved?)
- World-model calibration metrics (which CPT parameters have posterior variance > prior variance?)
- EWC Fisher diagonal magnitudes (which parameters are being most aggressively constrained, indicating the model is fighting to preserve knowledge against catastrophic forgetting?)
- Walk-forward backtest Sharpe attribution (which signals contribute positive out-of-sample alpha?)

At that point, this doctrine becomes a historical record. `scripts/density_audit.py` output and GNN diagnostics become the operational guide.

**The one invariant that survives Phase 40:** No single observation category exceeds 30% of total observations. This is not a heuristic. It is a structural constraint derived from attention-collapse theory in heterogeneous GNNs (see §5 and §13). It must hold regardless of what Phase 40 diagnostics show. The imbalance discovered today (93.8% instrument_daily) must be corrected permanently — not temporarily for Phase 47 and then allowed to drift back.

**The test of this doctrine is empirical, not theoretical.** After Phase 40 walk-forward results are in hand, revisit each section and mark which predictions held (regime coverage → better HMM calibration; cross-source diversity → lower attention collapse; timestamp fidelity → tighter Kalman estimates) and which did not. Update this file accordingly. A doctrine that does not update on evidence is not a doctrine — it is a belief.

---

## 17. References

### Temporal graph learning
- Rossi, Chamberlain, Frasca, Eynard, Monti, Bronstein. 2020. "Temporal Graph Networks for Deep Learning on Dynamic Graphs." arXiv:2006.10637.
- Hayes, Schumacher, Strohmaier. 2025. "What Do Temporal Graph Learning Models Learn?" arXiv:2510.09416.
- Xu, Ruan, Korpeoglu, Kumar, Achan. 2020. "Inductive Representation Learning on Temporal Graphs." arXiv:2002.07962 (TGAT).

### Continual learning
- Kirkpatrick et al. 2017. "Overcoming catastrophic forgetting in neural networks." PNAS 114(13):3521–3526.
- Jones, Sprague. 2018. "Continual Learning Through Expandable Elastic Weight Consolidation." JMU CS.
- Anonymous. 2026. "Continual Learning for non-stationary regression via Memory-Efficient Replay." arXiv:2602.09720.

### POMDP / non-stationary RL
- Cassandra, Kaelbling, Littman. 1994. "Acting Optimally in Partially Observable Stochastic Domains."
- Mornik et al. 2024. "Learning and Planning in a Time-Varying Partially Observable Environment." PLMO24, U. Illinois.

### Information theory / feature selection
- Peng, Long, Ding. 2005. "Feature selection based on mutual information: criteria of max-dependency, max-relevance, and min-redundancy." IEEE PAMI 27(8).
- Brown et al. 2012. "Conditional likelihood maximisation: a unifying framework for information-theoretic feature selection." JMLR 13.
- Anonymous. 2022. "High-Order Conditional Mutual Information Maximization for Dealing with High-Order Dependencies." arXiv:2207.08476.

### Regime detection
- Song, Eraker. 2014. "Infinite Hidden Markov Model for Regime Switching and Structural Breaks." J. Applied Econometrics (Wiley).
- Quantstart / BSIC 2025. "Regime Detection using HMMs in QSTrader."

### Imbalanced heterogeneous GNN
- Zhao, Zhang, Wang. 2021. "GraphSMOTE: Imbalanced Node Classification on Graphs with Graph Neural Networks." WSDM.
- Liu, Fang. 2021. "Tail-GNN: Tail-Node Graph Neural Networks." KDD.
- Qian, Zhang, Zhang, Wen, Ye, Zhang. 2022. "Co-Modality Graph Contrastive Learning for Imbalanced Node Classification." NeurIPS.

### Exogenous modelling
- Anonymous. 2025. "Select, then Balance: A Plug-and-Play Framework for Exogenous-Aware Spatio-Temporal Forecasting." arXiv:2509.05779.

### Data sources (verified availability)
- SEC EDGAR: full-text search from 1993. https://efts.sec.gov/LATEST/search-index
- GDELT v1: Jan 1979. v2: Feb 2015. http://data.gdeltproject.org/events/
- FRED: 840K+ series. https://fred.stlouisfed.org/
- CFTC COT: Futures-only 1986, F+O 1995, Disaggregated 2006. https://www.cftc.gov/MarketReports/CommitmentsofTraders/
- USPTO PatentsView: 1976+. https://patentsview.org/
- UN Comtrade: reliable 2011+. https://comtradeplus.un.org/
- USASpending: 2001+, reliable post-FFATA 2008. https://www.usaspending.gov/

---

## 18. Operational summary (what this changes)

1. **Phase 47 is not "run every tool for 5 years".** It is "run every tool for the depth that closes its sample-complexity gap without worsening modal imbalance".
2. **A density audit script is a blocking deliverable** (Phase 47.4). Without it, we cannot evaluate the backfill and cannot gate Phase 40.
3. **Four capped tools (`insider_filings`, `form144`, `finra_short_volume`, `sanctions_monitor`) need an `as_of_date` parameter** to support sliding-window backfill. Spec'd in `[[historical_backfill_spec]]`.
4. **CFTC backfill uses year-looping** (1993..2026), not `days_back` — CFTC historical endpoint is year-indexed.
5. **GDELT backfill is deferred to Phase 47b** — native format is daily CSV files, not REST; needs separate runner logic. Not a blocker for Phase 40.
6. **The GNN retrain (Phase 40) is gated on density audit pass, not on calendar**. Do not run Phase 40 on a failed backfill.
7. **Phase 49b (convergence as control) can run in parallel with Phase 47** since it operates on the existing pipeline state and does not depend on deeper history.

---

## Related

- [[historical_backfill]]
- [[historical_backfill_spec]]
- [[quant_training_ground]]
- [[living_system_online_gnn]]
- [[signal_protocol]]
- [[depth_model]]
- [[convergence_as_control]]
- [[gnn_downstream_alignment]]
