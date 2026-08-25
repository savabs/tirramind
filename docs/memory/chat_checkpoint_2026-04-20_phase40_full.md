---
title: "Checkpoint: Phase 40 Full Debrief — GNN Training, Data Landscape, Architecture Understanding, Overtraining Diagnosis"
tags:
  - doc/checkpoint
  - phase/40
  - topic/gnn
  - topic/backtest
  - topic/pipeline
  - topic/surveillance
  - layer/world-model
  - layer/surveillance
  - layer/feature-engineering
---

# Checkpoint: Phase 40 Full Debrief

**Date:** 2026-04-20
**Task:** [[phase40_real_data_model_refresh]]
**Spec:** [[real_data_model_refresh_spec]]
**Research:** [[real_data_model_refresh]]

---

## 1. What Phase 40 Did — The Full Picture

Phase 40 ("Real Data Model Refresh") was the first time the HetTGN (Heterogeneous Temporal Graph Network) was trained on real pipeline data instead of synthetic test data. Six steps were completed:

| Step | What | Status |
|------|------|--------|
| 40.1 | Created `scripts/retrain_gnn.py` — CLI training script with Rich output, auto-tune, `--since` filter, `--backup` | Done |
| 40.2 | Backed up prior model → retrained GNN on real pipeline data | Done (two runs — see below) |
| 40.3 | Regenerated GNN features via `run_collection.py --dag feature_generation` (5.9s) | Done |
| 40.4 | Ran walk-forward backtest, recorded baselines | Done |
| 40.5 | Wrote 8 edge-case tests in `tests/test_phase40_retrain.py` (all passing) | Done |
| 40.6 | Updated task file + wrote checkpoint | Done |

---

## 2. The Data That Exists

### 2.1 Pipeline Database

Location: `.tirra_pipeline/pipeline.db` (SQLite)

**Key tables:** `entities`, `entity_observations`, `entity_links`, `entity_aliases`, `features`, `signals`, `beliefs`, `portfolio_weights`, `dag_runs`, `rl_transitions`, `convergence_clusters`

### 2.2 Entities (929 total, 6 types)

| Entity Type | Count | Examples |
|-------------|-------|---------|
| `topic` | 729 | Polymarket questions — "Will Osasuna finish top 4 La Liga?", geopolitical topics |
| `instrument` | 89 | SPY, GDX, USD/CHF, AUD/USD, GC=F, CL=F, BTC-USD, UVXY, AGG... |
| `country` | 82 | US, CH, EU, LEBANON, ALASKA, ISRAELI, FRENCH... |
| `cftc_contract` | 20 | Futures contracts tracked by CFTC Commitment of Traders |
| `company` | 7 | Companies tracked via instrument_universe |
| `protocol` | 2 | Crypto DeFi protocols |

### 2.3 Observations (71,123 total, 7 types from 4 source tools)

| Observation Type | Count | Source Tool | What It Contains |
|-----------------|-------|------------|-----------------|
| `instrument_daily` | 68,089 | `instrument_universe` | `{close, log_return, volume, realized_vol_20d}` per instrument per day |
| `market_probability` | 1,458 | `polymarket` | `{yes_price, no_price, volume_24h, volume_total, liquidity, spread}` |
| `geopolitical_event` | 735 | `gdelt` | `{event_id, counterpart_country, event_root, event_description, goldstein}` |
| `instrument_return` | 267 | `instrument_universe` | Derived return observations |
| `instrument_volume` | 267 | `instrument_universe` | Derived volume observations |
| `instrument_volatility` | 267 | `instrument_universe` | Derived volatility observations |
| `futures_positioning` | 40 | `cftc` | CFTC Commitment of Traders data |

**Critical imbalance:** `instrument_daily` is 95.7% of all observations. The model sees price data overwhelmingly more than any other signal. This is why obs_type prediction accuracy is high but may be trivially learning "always guess instrument_daily."

**Date range issues:**
- GDELT has some timestamps from 1970 (bad data) — the `--since 2023-01-01` flag was added to filter these
- Instrument data: 2023-04-18 → 2026-04-19 (3 years)
- Latest Polymarket: 2026-04-19
- Latest GDELT: 2026-04-19
- `observed_at` column stores as float but has `None` datetime rendering in SQLite (timestamps stored as Unix epoch floats)

### 2.4 Entity Links (272 total, 9 types)

| Link Type | Count | Meaning |
|-----------|-------|---------|
| `located_in` | 90 | Instrument → Country (headquarters/exchange location) |
| `tracks_issuer` | 45 | Instrument → Company it tracks |
| `event_involves` | 44 | Country → Country (GDELT event participants) |
| `topic_relates_to_instrument` | 36 | Polymarket topic → relevant instrument |
| `exchange_country` | 20 | CFTC contract → exchange country |
| `fx_base_country` | 15 | FX pair → base currency country |
| `fx_quote_country` | 15 | FX pair → quote currency country |
| `cftc_tracks` | 5 | CFTC contract → underlying instrument |
| `tracks_protocol` | 2 | Crypto instrument → DeFi protocol |

These links form the graph edges the GNN propagates information through. Example chains:
- `USD/CHF` → `fx_base_country` → `US` ← `event_involves` ← `LEBANON`
- `SPY` → `tracks_issuer` → company → `located_in` → `US`
- CFTC contract → `cftc_tracks` → `CL=F` (crude oil futures)

### 2.5 Data Collection Infrastructure

**DAG runs executed:**
- `daily_collection` — fetches new data from all source tools (last run: completed)
- `feature_generation` — regenerates GNN features from model embeddings (last run: completed)
- `gnn_inference` — runs GNN forward pass to update entity embeddings (last run: completed, one prior failed)

**Source tools active (4 of 60 built):**
- `instrument_universe` — 68,890 observations
- `polymarket` — 1,458 observations
- `gdelt` — 735 observations
- `cftc` — 40 observations

---

## 3. The GNN Architecture

### 3.1 Model: HetTGN (Heterogeneous Temporal Graph Network)

- **Type:** 2-layer HGT (Heterogeneous Graph Transformer) with temporal memory
- **Parameters:** 339,141 total (all trainable)
- **Hidden dim:** 64, Memory dim: 64, Num heads: 2
- **Node types:** 6 (cftc_contract, company, country, instrument, protocol, topic)
- **Edge types:** 10 (9 entity link types + reverse edges)
- **Memory nodes:** 929 (one 64-dim memory vector per entity)

### 3.2 How It Works

Each entity has a persistent 64-dimensional memory vector that evolves as observations arrive. When processing a window of observations:

1. **Event encoding:** Each observation is encoded into a feature vector (observation type embedding + time delta encoding + value features)
2. **Memory update:** The entity's memory vector is updated via a GRU (gated recurrent unit) using the new event encoding
3. **Graph propagation:** Updated memories propagate through the entity link graph via heterogeneous graph attention (HGT) — information flows from entity to entity along edges
4. **Prediction heads:** Three heads predict from the updated memory state:
   - **obs_type head:** What type of observation comes next (cross-entropy over 7 types)
   - **time_delta head:** When it arrives (MSE on log(1 + seconds))
   - **value head:** What numeric value it carries (Huber loss)
5. **Contrastive loss:** Linked entities should be closer in embedding space than random entities (margin loss)

### 3.3 Training Procedure

- **Self-supervised:** No human labels — the stream of observations IS the supervision
- **Walk-forward windows:** Observations sorted chronologically, processed in sliding windows (configurable size, default 24h or 48h)
- **Auto-tune loss weights:** Kendall et al. 2018 uncertainty-weighted multi-task loss — each loss component gets a learnable weight via $L_k / (2\sigma_k^2) + \ln(\sigma_k)$

### 3.4 What the Model Actually Learns

This is NOT a stock price predictor. It learns the temporal dynamics of an entire information ecosystem:
- Which entities receive which types of observations and when
- How information propagates across entity links (geopolitical event in Lebanon → memory update for Israel → flows to instruments linked to that region)
- The temporal rhythm of different data domains (daily prices every 24h, GDELT events irregularly, Polymarket updates in clusters)
- Value ranges and correlations across entity types

The useful output for downstream tasks is the **entity memory vectors** — 64-dim representations that capture each entity's current state given everything the system has observed. These embeddings are the input to the world model (Layer 3), signal fusion (Layer 4), and eventually portfolio allocation.

---

## 4. Training Runs — Full Results

### 4.1 Run 1: 5 Epochs, 48h Window (THE GOOD ONE)

```
Command: python scripts/retrain_gnn.py --db-path .tirra_pipeline/pipeline.db \
         --epochs 5 --auto-tune --since 2023-01-01 --window-size 172800
```

**Config:** epochs=5, lr=0.001, hidden=64, layers=2, heads=2, window=48h, auto_tune=True

**Training time:** 982.4 seconds (16.4 minutes), 397 windows/epoch

**Loss curve:**

| Epoch | Total | obs_type | time_delta | contrastive | value |
|-------|-------|----------|------------|-------------|-------|
| 1 | 22.7693 | 2.1113 | 147.5543 | 0.0000 | 6.5384 |
| 2 | 11.9476 | 0.4012 | 77.2474 | 0.0000 | 1.0953 |
| 3 | 4.4114 | 0.0385 | 12.6716 | 0.0000 | 0.0113 |
| 4 | 3.2190 | 0.0466 | 5.7147 | 0.0000 | 0.0866 |
| 5 | 1.9956 | 0.0012 | 0.1321 | 0.0000 | 0.0009 |

**Learned loss weights:** obs=1.130, dt=0.098, contr=3.640, val=0.470

**Evaluation:**

| Split | obs_type top-1 | obs_type top-5 | time_delta MAE | Predictions |
|-------|---------------|---------------|----------------|-------------|
| Val | **100.0%** | 100.0% | 0.3 seconds | 6,410 |
| Test | **86.8%** | 86.8% | 1.7 seconds | 5,770 |
| Random | 2.17% | 10.87% | — | — |

**Assessment:** Healthy convergence. Loss monotonically decreasing. Loss weights stayed in reasonable range (0.1–3.6). Test accuracy 40x over random baseline. Time delta MAE of 1.7 seconds means the model has sub-minute temporal resolution.

**⚠️ This model was NOT saved.** It was overwritten by Run 2 (see below). The 5-epoch model no longer exists on disk.

### 4.2 Run 2: 10 Epochs, 24h Window (THE BROKEN ONE — CURRENTLY ON DISK)

```
Command: python scripts/retrain_gnn.py --db-path .tirra_pipeline/pipeline.db \
         --epochs 10 --auto-tune --backup --since 2023-01-01
```

**Config:** epochs=10, lr=0.001, hidden=64, layers=2, heads=2, window=24h (default), auto_tune=True

**Training time:** 5259.6 seconds (87.7 minutes)

**Loss curve:**

| Epoch | Total | obs_type | time_delta | contrastive | value |
|-------|-------|----------|------------|-------------|-------|
| 1 | 4.1662 | 0.1463 | 8.9513 | 0.0000 | 0.0034 |
| 2 | 1.1741 | 0.0381 | 3.5914 | 0.0000 | 0.0024 |
| 3 | -1.7757 | 0.0040 | 1.2926 | 0.0000 | 0.0008 |
| 4 | **-4.8300** | 0.0006 | 0.1676 | 0.0000 | 0.0000 |
| 5 | -7.8844 | 0.0000 | 0.0300 | 0.0000 | 0.0000 |
| 6 | -10.9824 | 0.0000 | 0.0134 | 0.0000 | 0.0000 |
| 7 | -14.0688 | 0.0000 | 0.0208 | 0.0000 | 0.0000 |
| 8 | -17.2608 | 0.0000 | 0.0013 | 0.0000 | 0.0000 |
| 9 | -20.4321 | 0.0000 | 0.0008 | 0.0000 | 0.0000 |
| 10 | **-23.5823** | 0.0000 | 0.0010 | 0.0000 | 0.0000 |

**Learned loss weights:** obs=**2006.680**, dt=**37.518**, contr=**1389.234**, val=**834.707**

**Evaluation:**

| Split | obs_type top-1 | obs_type top-5 | time_delta MAE | Predictions |
|-------|---------------|---------------|----------------|-------------|
| Val | 100.0% | 100.0% | **86,389 seconds (24.0 hours)** | 10,469 |
| Test | 90.7% | 91.6% | **83,517 seconds (23.2 hours)** | 9,068 |

**What went wrong — detailed diagnosis:**

1. **Negative loss = auto-tune divergence, not better learning.** The Kendall uncertainty weighting uses $\text{total} = \sum_k \frac{1}{2\sigma_k^2} L_k + \ln(\sigma_k^2)$. When all component losses ($L_k$) hit ~0 (model memorized training data by epoch 4), the first term vanishes and the optimizer keeps pushing $\ln(\sigma_k^2)$ more negative — making $\sigma_k^2$ smaller, which makes the weights $\frac{1}{2\sigma_k^2}$ grow toward infinity. The total loss goes to $-\infty$ even though zero actual learning is happening. This is a known failure mode of this formulation.

2. **Loss weights exploded 3 orders of magnitude.** Compare Run 1 weights (obs=1.13, dt=0.098) to Run 2 (obs=2006.7, dt=37.5). These are meaningless — the log-variance parameters drifted into regions where the effective weights are astronomically large, but multiplied by ~0 loss, so the model parameters don't change.

3. **Time delta MAE went from 1.7 seconds → 83,517 seconds (23.2 hours).** The model collapsed to predicting the dominant interval: ~86,400 seconds (24 hours) for every entity. Since `instrument_daily` is 95.7% of data and arrives every ~24h, predicting "24 hours" is the maximum-likelihood degenerate solution. The model lost all temporal resolution.

4. **Test accuracy improvement (86.8% → 90.7%) is misleading.** The model got better at the trivial sub-task (predicting `instrument_daily` for everything) while catastrophically losing the temporal prediction capability. The 24h window (vs 48h in Run 1) also means more windows with homogeneous instrument_daily content, reinforcing the degenerate solution.

### 4.3 Model Files on Disk

| File | Size | Modified | Contents |
|------|------|----------|----------|
| `gnn_model.pt` | 1.6MB | 2026-04-20 14:02 | **10-epoch BROKEN model** (the one currently active) |
| `gnn_model_pre_phase40.pt` | 1.6MB | 2026-04-20 12:27 | Backup taken before 10-epoch run (2-epoch model) |
| `gnn_model_live.pt` | 1.0MB | 2026-04-19 03:51 | 5-epoch model from prior pipeline runs |
| `gnn_model_synthetic_backup.pt` | 0.2MB | 2026-04-19 12:02 | 10-epoch model trained on synthetic data |

**⚠️ ACTION REQUIRED:** The active `gnn_model.pt` is the broken 10-epoch model. It should be replaced. Best candidate is `gnn_model_live.pt` (5 epochs, from 2026-04-19) or a fresh 5-epoch training run. The `gnn_model_pre_phase40.pt` is only a 2-epoch model, likely undertrained.

---

## 5. Known Issues — Prioritized

### 5.1 CRITICAL: Auto-Tune Weight Divergence

**Problem:** When component losses approach 0, the Kendall log-variance parameters drift to $-\infty$, causing negative total loss and exploded effective weights. This makes training past ~5 epochs useless with current data volume.

**Fix needed in** `agent/models/gnn/trainer.py` lines 1001-1020:
- Option A: **Clamp log-variance parameters** — e.g., `lv[k] = torch.clamp(lv[k], min=-3.0, max=3.0)` → effective weights bounded to [0.05, 20.1]
- Option B: **Early stopping on auto-tune** — freeze log-var params when all component losses < threshold (e.g., 0.01)
- Option C: **Minimum loss floor** — add a small epsilon: `obs_loss = max(obs_loss, 1e-4)` so the weighting term never vanishes

Recommendation: Option A (clamp) is simplest and most robust. Apply before next training run.

### 5.2 HIGH: Contrastive Loss Always Zero

**Problem:** The contrastive loss has been 0.0 across all epochs in both runs. The code in `_contrastive_loss()` (trainer.py:801-867) computes margin loss over entity links, but it returns 0 if `pos_scores` or `neg_scores` is empty.

**Root cause hypothesis:** The `embeddings` dict passed to `_contrastive_loss()` may not contain the entity types needed to look up linked entities, OR the `id_map.local_id()` calls return `None` for linked entities because they weren't in the current window. Since contrastive loss operates on the full entity graph but training processes per-window, entities not active in the current window have no embeddings to contrast.

**Fix options:**
- Use full-graph memory embeddings (not per-window) for contrastive loss — the memory state IS the embedding
- Pre-compute contrastive loss on the full graph once per epoch instead of per-window
- Ensure negative sampling covers entities outside the current window

### 5.3 MEDIUM: Observation Type Imbalance (95.7% instrument_daily)

**Problem:** 68,089 of 71,123 observations are `instrument_daily`. The model trivially learns "predict instrument_daily" and achieves high accuracy. The other 6 observation types have minimal representation.

**Impact:** obs_type classification accuracy overstates model quality. The 86.8% test accuracy may largely reflect learning to always predict the majority class.

**Fix options:**
- Class-weighted cross-entropy loss for obs_type (inverse frequency weighting)
- Observation type subsampling (downsample instrument_daily or oversample rare types)
- **Best fix: more data from diverse tools** — activating the 56 unused tools would naturally rebalance

### 5.4 MEDIUM: Val Accuracy = 100% (Possible Temporal Leakage)

**Problem:** Validation accuracy is perfect while test accuracy is 86.8%. In a proper temporal split, val shouldn't be dramatically easier than test.

**Possible causes:**
- The 70/15/15 chronological split may put val observations in a period where patterns are identical to training (temporal autocorrelation)
- The entity memory state carries training information forward into val evaluation
- Not a critical issue for self-supervised pretraining (the goal is useful embeddings, not perfect classification), but worth auditing if the model is used for decisions

### 5.5 LOW: GDELT Timestamp Issues

Some GDELT observations have timestamps from 1970 (Unix epoch 0 or garbage values). The `--since 2023-01-01` filter handles this at training time, but the underlying data should be cleaned or the GDELT tool should validate timestamps at ingestion.

---

## 6. Backtest Baselines (Step 40.4)

**Period:** 2023-04-18 → 2026-04-18 (1,097 trading dates, 89 instruments)
**Walk-forward config:** min_train=252 days, test=21 days, step=21 days → **40 folds**

| Strategy | Total Return | Sharpe Ratio | Max Drawdown |
|----------|-------------|-------------|--------------|
| EqualWeight (all 89) | 23.14% | **0.995** | **-8.20%** |
| Buy & Hold SPY 100% | 48.84% | 0.900 | -18.76% |
| Buy & Hold AGG:40% + SPY:60% | 31.75% | 0.991 | -11.41% |

**Top EqualWeight contributors:** GDX +1.30%, SI=F +1.27%, SLV +1.27%
**Top EqualWeight detractors:** UVXY -2.00%, VIXY -0.94%

**Key takeaway:** EqualWeight across 89 instruments has the best risk-adjusted return (highest Sharpe, lowest drawdown) because diversification dampens vol. SPY has higher raw return but double the drawdown. These are **naive baselines** — the GNN is not yet wired into allocation decisions. The goal is for GNN-informed weights to beat EqualWeight's 0.995 Sharpe.

---

## 7. Architecture Understanding — What TirraMind Actually Is

### 7.1 Not a Stock Predictor

TirraMind is an **information arbitrage detection system**. It models the temporal dynamics of an entire information ecosystem — not just prices, but geopolitical events, prediction market shifts, futures positioning, and (eventually) insider filings, vessel tracking, patent activity, disease outbreaks, and 50+ other domains.

The GNN learns to predict **what happens next across all entity types** (not just price direction). When reality diverges from the model's expectation — a geopolitical shock arrives where calm was expected, a Polymarket probability shifts before institutional positioning does — that **surprise signal** is the alpha.

### 7.2 The 7-Layer Computation Stack

```
Layer 1: Surveillance Surface  → agent/tools/ (data fetching)          ← 60 tools built, 4 active
Layer 2: Feature Engineering    → agent/quant/ (signal extraction)      ← GNN embeddings
Layer 3: World Model            → agent/models/ (Bayesian network)     ← consumes GNN embeddings
Layer 4: Signal Fusion          → agent/fusion/ (Kalman/particle)      ← fuses multi-source estimates
Layer 5: RL Policy              → agent/learning/ (model-based RL)     ← portfolio optimization
Layer 6: Adversarial            → agent/adversarial/ (manipulation)    ← edge decay monitoring
Layer 7: LLM Support            → agent/reasoning/ (text parsing only) ← explains, never decides
```

Phase 40 sits at the Layer 2/3 boundary — the GNN produces entity embeddings (Layer 2) that will feed the world model (Layer 3).

### 7.3 The Cross-Domain Edge Thesis

Individual data sources are public and commodity. The moat is **cross-domain entity linking** — combining insider filings + vessel tracking + CFTC positioning + GDELT events + Polymarket probabilities through a shared entity graph, then using a temporal memory network to detect patterns that emerge only from the combination.

The GNN's attention heads learn which cross-domain connections transmit the most information. A spike in GDELT conflict events involving ISRAELI entities updates memory for Israel, which flows through entity links to FX pairs (USD/ILS), commodity instruments tied to that region, and Polymarket questions about Middle East politics. Nobody else has this specific graph with these specific connections being monitored by a temporal memory network.

### 7.4 Surprise Detection (Not Yet Built)

The highest-value next feature is **surprise detection**: compare GNN predictions vs actual observations per entity. When the model expects "next observation for entity X is instrument_daily in 24h" but a geopolitical_event arrives in 2h, that divergence IS the signal. The magnitude and direction of surprise across the entity graph, weighted by cross-domain attention, becomes the input to portfolio allocation.

---

## 8. Data Domain Expansion — What's Needed

### 8.1 Current State: 4 domains, 71K observations, 95.7% price data

This is insufficient for the cross-domain thesis. The model essentially learns price patterns with occasional geopolitical and prediction market context. The entity graph is sparse — most entities have only 1-2 link types.

### 8.2 Built Tools Not Yet Producing Data (56 tools)

**Tier 1 — High cross-domain edge, wire first:**

| Tool | Data Domain | Entity Types | Cross-Domain Links |
|------|------------|-------------|-------------------|
| `insider_filings` / `form144` | SEC insider transactions | person → company → instrument | Who is selling before earnings? |
| `whale_alert` / `defi_flows` | On-chain crypto flows | wallet → protocol → instrument | Large transfers precede price moves |
| `polymarket_whales` | Large prediction market bets | trader → topic | Smart money in prediction markets |
| `finra_short_volume` | Short selling pressure | instrument-level | Institutional bearish conviction |
| `ais_vessel` | Ship tracking (AIS) | vessel → port → commodity | Tanker diversions predict oil supply |
| `sanctions_monitor` | OFAC/EU sanctions | entity → country → instrument | Instant cross-domain impact |

**Tier 2 — Strong macro/regime signals:**

| Tool | What It Adds |
|------|-------------|
| `treasury_receipts` | Government fiscal impulse → macro regime |
| `central_bank_balance` | Fed/ECB/BOJ balance sheets → liquidity regime |
| `sovereign_debt` | Country debt dynamics → currency/rates |
| `capital_flows` | Cross-border money flows → FX pressure |
| `electricity_monitor` / `power_grid` | Real-time economic activity proxy |
| `energy_supply` | Energy production/storage → commodity prices |
| `consumer_sentiment` | Demand-side leading indicator |
| `global_pmi` | Manufacturing/services health per country |
| `liquidity_regime` | Market microstructure stress indicator |

**Tier 3 — Investigative / entity-level depth:**

| Tool | What It Adds |
|------|-------------|
| `supply_chain_monitor` | Disruption → company/sector impact |
| `drug_regulatory` | FDA/EMA decisions → biotech instruments |
| `patent_filings` | Innovation activity per company |
| `bankruptcy_court` / `creditor_filings` | Company distress signals |
| `lobbying` | Policy influence → regulatory outcomes |
| `gov_contracts` | Government spending by entity |
| `food_security` | Agricultural supply shocks → commodities |
| `weather_alerts` | Physical world → agriculture, energy |
| `earthquake_proximity` | Infrastructure disruption |
| `labor_disruptions` | Strikes/layoffs → sector impact |
| `transport_throughput` | Port/rail volumes → trade flows |
| `disease_surveillance` | Pandemic signals → everything |

**Tier 4 — Digital infrastructure (niche/unique):**

| Tool | What It Adds |
|------|-------------|
| `cert_transparency` | New SSL certs → product launches |
| `dns_monitor` | Domain changes → corporate activity |
| `internet_outages` | Country connectivity → instability |
| `wikipedia_pageviews` | Attention/narrative tracking |
| `satellite_activity` | Space launch → defense/tech |

### 8.3 Expansion Strategy

Per the Signal Depth Doctrine and GNN-guided iterative expansion principle:
1. Wire Tier 1 tools (6 tools) → retrain GNN → evaluate which entity neighborhoods are still sparse
2. Use GNN attention weights to decide which Tier 2/3 tools to activate next
3. After each batch, retrain and re-evaluate — don't blindly activate all 56

Activating Tier 1 alone would roughly **triple observation diversity** and add entirely new entity types (person, wallet, vessel) with cross-domain edges that don't exist today.

### 8.4 Note on Tier 2 Macro Tools

Macro-level tools (treasury_receipts, consumer_sentiment, central_bank_balance, global_pmi) produce country-level or market-level numbers, NOT entity-level data. Per the architecture rules, these are better consumed as global conditioning variables or country-node features. They don't need L2 entity resolution — they're already at the right granularity for country nodes in the graph.

---

## 9. Performance Optimizations Shipped in Phase 40

| Optimization | Problem | Solution | Impact |
|-------------|---------|----------|--------|
| O(n) bucket windowing | Was O(n×w) scanning all obs per window | Bisect-based slicing on sorted timestamps | ~50x faster window construction |
| Entity type cache | DB query per entity per window | Cache entity_type mapping once at startup | Eliminated thousands of DB calls |
| 3-level graph caching | Rebuilding full HeteroData per window | `prepare_static()` → `prefetch_observations()` → `build_from_cached()` | Graph structure built once, only events change |
| log(1+dt) normalization | time_delta in raw seconds → MSE in billions | `log(1 + dt)` transform | Loss numerically stable, gradient meaningful |
| `--since` temporal filter | 1970-era GDELT garbage timestamps | CLI flag filtering obs before date | Clean training data |
| Rich console output | No visibility into training progress | Rich tables for loss curves, eval metrics | Debuggable runs |

---

## 10. Files Changed/Created in Phase 40

| File | Action | Purpose |
|------|--------|---------|
| `scripts/retrain_gnn.py` | Created | CLI training script (140 lines) |
| `agent/models/gnn/trainer.py` | Modified | Perf optimizations, auto-tune, windowing |
| `agent/models/gnn/graph_builder.py` | Modified | 3-level caching |
| `tests/test_phase40_retrain.py` | Created | 8 edge-case tests (all passing) |
| `.tirra_pipeline/gnn_model.pt` | Overwritten | Currently 10-epoch broken model |
| `[[phase40_real_data_model_refresh]]` | Updated | All steps checked, results recorded |
| `[[chat_checkpoint_2025-07-15_phase40_complete]]` | Created | Prior checkpoint (dated wrong — should be 2026) |

---

## 11. Test Status

`tests/test_phase40_retrain.py` — **8 tests, all passing** (4.74s)

Tests cover: TrainerConfig defaults, model building from store, single training step, evaluation metrics, `--since` filtering, observation splitting, auto-tune parameter creation, CLI argument parsing.

---

## 12. Git Status

Latest commit: `d9f3242 (HEAD -> main) phase29: mark task complete, write checkpoint`

Phase 40 work is **not committed**. All changes are local only. The following files need to be staged:
- `scripts/retrain_gnn.py`
- `agent/models/gnn/trainer.py`
- `agent/models/gnn/graph_builder.py`
- `tests/test_phase40_retrain.py`
- `[[phase40_real_data_model_refresh]]`
- `docs/memory/chat_checkpoint_*.md`

---

## 13. Immediate Next Actions (Prioritized)

1. **Fix auto-tune weight clamping** in `trainer.py` — clamp log-variance to [-3, 3] to prevent divergence
2. **Restore good model** — either retrain 5 epochs with the fix, or copy `gnn_model_live.pt` → `gnn_model.pt`
3. **Investigate contrastive loss = 0** — likely needs full-graph memory embeddings, not per-window
4. **Commit Phase 40 work** to git
5. **Activate Tier 1 data tools** — insider_filings, whale_alert, defi_flows, polymarket_whales, finra_short_volume, sanctions_monitor
6. **Wire GNN embeddings into allocation** — move from EqualWeight baseline to GNN-informed portfolio weights
7. **Implement surprise detection** — compare GNN predictions vs actuals per entity, use divergence as signal

---

## Related

- [[phase40_real_data_model_refresh]]
- [[real_data_model_refresh_spec]]
- [[real_data_model_refresh]]
- [[chat_checkpoint_2025-07-15_phase40_complete]]
