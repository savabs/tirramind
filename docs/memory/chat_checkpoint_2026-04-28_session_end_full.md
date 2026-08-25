---
title: "Checkpoint: Full Session End — Phase 40 Complete + World-State Methodology — 2026-04-28"
tags:
  - doc/checkpoint
  - phase/40
  - phase/48
  - topic/gnn
  - topic/backtest
  - topic/world-model
  - topic/evaluation
  - status/done
---

# Checkpoint: Full Session End — Phase 40 Complete + World-State Methodology — 2026-04-28

> **This is the full log-off checkpoint. Everything that happened this session is recorded here.**
> Next session: read this document first, then [[quant_training_ground]] for the roadmap.

---

## 1. What Was Done This Session (Chronological)

### 1.1 Ran Phase 40 GNN Backtest (lookback fix applied)

The previous session had applied a 90-day lookback window to both GNN strategies but killed the process before it completed. This session ran the fixed version.

**Command used (always use this exact python path):**
```bash
cd /home/becmachlean/2024/projects/tirramind_v1
/home/becmachlean/anaconda3/bin/python -u scripts/phase40_gnn_backtest.py > /tmp/phase40_out.txt 2>&1 &
tail -f /tmp/phase40_out.txt
```

**Runtime:** ~190 seconds total
- 1s: price load (fast SQL path)
- 11s: model load + EWC state restore
- 31s: `prefetch_observations()` — 977,870 obs loaded into memory once
- ~56s: GNN-EmbNorm (40 folds × ~1.4s each)
- ~90s: GNN-ValueHead (40 folds × ~2.2s each)

### 1.2 Hit KeyError on Comparison Table

Script crashed at the comparison table block:
```
KeyError: 'EqualWeight'
```

**Root cause:** `EqualWeightStrategy.name` returns `"equal_weight"` (snake_case), not `"EqualWeight"`. The comparison block had `results["EqualWeight"]`.

**Fix applied in `scripts/phase40_gnn_backtest.py`:**
```python
# OLD:
baseline = results["EqualWeight"].aggregate_metrics
# NEW:
baseline = results["equal_weight"].aggregate_metrics
```

**File:** `scripts/phase40_gnn_backtest.py` line ~484.

### 1.3 Re-ran to get full results including comparison table

Full results captured. See §2 below.

### 1.4 Updated task file with Phase 40 results

Added section `## 40.7 — GNN-Signal Backtest Results (Phase 40 Final)` to:
`[[phase40_real_data_model_refresh]]`

Includes: model stats, infrastructure built, walk-forward results table, full conclusion, Phase 41 paths.

### 1.5 Wrote intermediate checkpoint

`[[chat_checkpoint_2026-04-28_phase40_gnn_backtest_results]]`

### 1.6 Deep conceptual discussion — Why Sharpe is the wrong metric

Discussed in depth why portfolio Sharpe is the wrong primary metric for evaluating whether the GNN learned anything. See §3 for full technical content. The outcome: IC (Information Coefficient) is the right first diagnostic, but even IC is a proxy — the real question is whether the GNN correctly infers world-state transitions.

### 1.7 Deep conceptual discussion — TirraMind is a world-state inference machine

Extended discussion on the correct framing of what TirraMind is building. See §4. Key conclusion: the system's job is to produce probability distributions over future entity-state transitions, not price forecasts. Financial signal is a downstream readout of correct world-state inference.

### 1.8 Deep conceptual discussion — Sensor cross-product as the moat

Discussed why the 62-tool sensor surface is the irreplicable competitive moat, not the math. See §5. The math is learnable by anyone. Cross-domain synergistic relationships that only emerge when observing 3+ independent sensor domains simultaneously — including at least one L0 physical sensor — are not.

### 1.9 Created new research document

`[[world_state_prediction_methodology]]`

Covers: the core problem (Sharpe conflates three failure modes), correct architecture framing, sensor inventory, four-tier evaluation framework (IC → event prediction → transfer entropy → Polymarket calibration), density prerequisites, math references, known causal chains to test.

### 1.10 Deep conceptual discussion — Synergistic causation and PID

Introduced the concept of **Partial Information Decomposition (PID)** and **synergistic causation** as the theoretical framework for "chains of things happening → outcome that can't be explained by any single cause." See §6 for full technical content. This is the Phase 48+ research frontier.

---

## 2. Phase 40 Final Results (Ground Truth Numbers)

### Model State
- **File:** `.tirra_pipeline/gnn_model.pt` (23.7 MB, created 2026-04-28 17:07)
- **Architecture:** HetTGN, 1,858,459 parameters
- **Config:** hidden_dim=128, num_layers=2, heads=2
- **Node types (9):** topic, person, company, country, instrument, wallet, protocol, cftc_contract, organization
- **Edge types:** 14
- **Graph size:** 2,451 nodes, 11,915 links
- **Training:** 10 epochs total (5 local + 5 Kaggle), final loss=14.14, contrastive=0.769, value=3.49
- **EWC:** 163 Fisher params, lambda=1000.0 (continuous learning from Phase 46)

### DB State (at time of backtest)
| observation_type | count | % |
|---|---|---|
| geopolitical_event (GDELT) | 901,704 | 92.2% |
| instrument_daily | 68,089 | 7.0% |
| market_probability | 5,356 | 0.5% |
| sell_intent | 493 | 0.1% |
| instrument_volume/volatility/return | 445 each | ~0% |
| futures_positioning | 300 | ~0% |
| insider_trade | 237 | ~0% |
| btc_transfer | 122 | ~0% |
| **TOTAL** | **977,870** | — |

Entity breakdown:
- topic: 1,235 | person: 459 | company: 339 | country: 245
- instrument: 89 | wallet: 33 | protocol: 23 | cftc_contract: 20 | organization: 8

### Walk-Forward Config
- Period: 2023-04-18 → 2026-04-18 (1,097 dates × 89 instruments)
- `min_train=252`, `test_size=21`, `step_size=21` → 40 folds

### Results Table (Phase 40 Final)

| Strategy | Total Return | Sharpe | Max Drawdown | Max Weight |
|---|---|---|---|---|
| **EqualWeight (baseline)** | 23.14% | **0.995** | -8.20% | 1.1% |
| GNN-EmbNorm | 25.10% | 0.582 | -11.59% | **44.65%** |
| GNN-ValueHead | 20.32% | 0.870 | -8.08% | 6.7% |

### Comparison (vs EqualWeight)

| Strategy | ΔSharpe | ΔTotal Return | ΔMax Drawdown |
|---|---|---|---|
| GNN-EmbNorm | **-0.413** | +1.95% | -3.39% (worse) |
| GNN-ValueHead | **-0.124** | -2.82% | +0.12% (marginally better) |

### Conclusion

**Neither GNN strategy beats EqualWeight on risk-adjusted returns.**

- **GNN-EmbNorm failure mode:** L2 embedding norm = connectivity/activity proxy, not a return signal. Softmax concentrates 44.65% in one high-norm asset. More raw return (+1.95%) but far worse risk-adjusted performance. The `z-score → softmax` signal design was the wrong choice for this feature.
- **GNN-ValueHead failure mode:** Value head trained with in-sample MSE on full history. No walk-forward splits during GNN training. In-sample bias baked in — the model "saw" the entire 2023-2026 period during training, so its predictions at any test fold already have knowledge of future states.
- **Win Rate = 0.000 and Volatility = 0.0000 for all strategies:** Cosmetic bug. `score_returns()` in `agent/quant/scoring.py` has no `win_rate` or `volatility` keys. The `_print_result()` function in `scripts/phase40_gnn_backtest.py` does `.get('win_rate', 0)` which defaults to 0. The important numbers (Sharpe, Return, MaxDD) are correct.

---

## 3. Why Sharpe Is the Wrong Primary Metric

(From the session discussion — this is the key methodological insight.)

Portfolio Sharpe conflates three independent failure modes into one number:
1. **Signal quality** — does the embedding encode anything predictively useful?
2. **Signal-to-weight translation** — does the weight rule (softmax of norms, value head scores) correctly extract that signal?
3. **Allocation quality** — does the resulting weight distribution reduce portfolio variance?

When Sharpe is bad, you cannot tell which layer failed.

**The right first diagnostic is IC (Information Coefficient):**

$$IC_t = \text{Spearman}\left(\hat{s}_{i,t},\ r_{i,t+21d}\right)$$

Where $\hat{s}_{i,t}$ is the GNN's score for instrument $i$ at fold cutoff $t$ and $r_{i,t+21d}$ is the actual 21-day forward return.

- Mean IC > 0.03 with t-stat > 2.0: weak but real signal present
- Mean IC > 0.07: meaningful signal
- IC ≈ 0: the embedding carries no return information

IC separates signal quality from allocation quality. A good IC with bad Sharpe = allocation rule is broken. A bad IC = the embedding is noise.

But even IC is just the first tier. See §4 for the full four-tier framework.

---

## 4. The Correct Architecture Framing — World-State Inference Machine

(The deepest insight from this session — informs everything downstream.)

**TirraMind is not a portfolio optimizer. It is a world-state inference machine.**

The correct question is not "does the GNN produce a signal that beats EqualWeight?" The correct question is:

> **Given sensor readings from physical and behavioral reality, does the GNN correctly infer the hidden state of the world — and does that inferred state propagate correctly to produce probability distributions over future observable events?**

The financial signal — if it exists — is a **readout from correct world-state inference**, not the primary target.

### The Prediction Hierarchy

| Input Layer | Output Layer | What is being inferred |
|---|---|---|
| L0 physical observations | L1 behavioral events | "Given vessel rerouting + grid load anomaly, which behavioral events become likely?" |
| L1 behavioral events | L2 information shifts | "Given insider selling + FINRA short buildup, which information state transitions are forming?" |
| L2 information shifts | L3 prices / geopolitics | "Given credit spread widening + capital flight signals, which market/geopolitical transitions are coming?" |

The GNN embedding `h_i` encodes the **latent state** of that entity at time T. Not a price forecast. A state representation: is this country under fiscal stress? Is this shipping company routing around something? Is this commodity supply chain degrading?

The GNN edges encode **how states propagate** — geopolitical stress in a wheat country → CFTC positioning shift → vessel rerouting → food security deterioration → political instability events.

None of that chain is price prediction. The financial signal, if it exists, emerges from correctly modeling the chain.

### Four-Tier Evaluation Framework (from `world_state_prediction_methodology.md`)

**Tier 1 — IC:** Does the embedding contain return signal at all? Min bar.

**Tier 2 — Event Prediction Accuracy:**
- Given GNN state of a country node at time T, does it assign higher probability to a significant GDELT event in [T, T+7d]?
- Given GNN state of a commodity instrument, does it predict a >2σ CFTC positioning change in [T, T+21d]?
- Given GNN state, does it predict a price breakout (>1.5× realized vol move) in [T, T+7d]?
- Metric: Precision@K, Recall@K, AUC-ROC in walk-forward fashion.

**Tier 3 — Transfer Entropy / Causal Chain Signal:**
$$TE_{A \to B} = H(B_t | B_{t-1}) - H(B_t | B_{t-1}, A_{t-\tau})$$
Does knowing entity A's embedding history reduce uncertainty about entity B's future state? For known causal chains: weather→food→political, vessel→CFTC→price, insider→FINRA→price.

**Tier 4 — Polymarket Calibration:**
Does the GNN's event probability estimate, computed from physical/behavioral signals alone, have prediction errors uncorrelated with Polymarket consensus errors? If yes = GNN is adding independent information the prediction market doesn't have.

---

## 5. The Sensor Moat

The math (HetTGN, Kalman, Bayesian propagation) is the nervous system. But a nervous system with no sensory organs is blind. The math can only detect relationships that exist within its current sensor space.

**The irreplicable moat is:** cross-domain synergistic relationships that only become visible when observing 3+ independent sensor domains simultaneously, at least one of which is L0 physical.

An economist with Bloomberg cannot simultaneously watch:
- 50,000 vessel position pings (AIS)
- Hourly grid load across 11 NYISO zones (power_grid)
- Real-time GDELT event streams across 245 countries
- CFTC COT positioning for 20 commodity contracts
- Satellite activity proxies
- Food security deterioration signals from FAO
- Whale on-chain flows
- Insider transaction patterns

And compute that all of them are shifting simultaneously in a configuration that implies a specific supply chain stress that won't appear in PMI for 6 weeks.

**The relationship doesn't exist in any one sensor. It only exists in the cross-tensor product of sensor space.**

### Current Sensor Inventory

62 tools total. L0 physical sensors already built:
- `ais_vessel.py` — vessel positions (OpenSky, ~3-5% global coverage)
- `power_grid.py` — NYISO grid load, fuel mix, pricing
- `weather_alerts.py` — NOAA weather anomalies
- `earthquake_proximity.py` — seismic events near infrastructure
- `satellite_activity.py` — satellite-based activity proxies
- `food_security.py` — FAO food security indicators
- `energy_supply.py` — energy supply signals
- `transport_throughput.py` — transport volume
- `supply_chain_monitor.py` — supply chain stress

The sensor surface **exists**. The problems are:
1. **Observation density is pathological** — 92.2% GDELT, 7.0% instrument_daily. Physical sensors are built but have thin backfill in the DB.
2. **Cross-domain entity links are sparse** — the GNN can't learn propagation without dense cross-domain observation history.
3. **Evaluation was measuring the wrong thing** — Sharpe doesn't test world-state inference.

### Known Causal Chains to Test After Backfill

1. Weather anomaly (NOAA) → crop yield signal (FAO) → grain vessel routing (AIS) → CFTC soy/wheat positioning → food security deterioration → GDELT political events in import countries
2. Corporate insider selling (Form 4) + short buildup (FINRA) → company GDELT mentions → price regime change
3. Port congestion (AIS) + supply chain stress → manufacturer GDELT events → industrial commodity price shift
4. Power grid anomaly → industrial sector GDELT events → equity sector return

---

## 6. Synergistic Causation — The Research Frontier (Phase 48+)

(The deepest architectural discussion of this session.)

The specific capability being described: **"Certain outcomes can't be pointed to one thing, but if you have an advanced way to link certain internal outcomes — that then shows WHY it happened."**

This has a formal name: **synergistic causation**, formalized through **Partial Information Decomposition (PID)** — Williams & Beer 2010.

### The Math

For two sensor sources A, B and an outcome C, mutual information decomposes as:

$$I(A,B;C) = \underbrace{\text{Unique}(A)}_{\text{A alone explains}} + \underbrace{\text{Unique}(B)}_{\text{B alone explains}} + \underbrace{\text{Redundant}(A,B)}_{\text{both say the same}} + \underbrace{\text{Synergy}(A,B)}_{\text{only visible in combination}}$$

**Synergy** is the portion of what you can know about outcome C that exists **only** in the combination of A and B — not in either alone, not in their sum. This is the "relationship that can't be seen by any individual analyst" term.

### Why It's Hard

- **Exponential complexity** — with 62 sensors, the number of possible synergistic combinations is $2^{62}$ if done naively.
- **PID is analytically solved only for 2 sources** — Williams & Beer 2010 solved it for 2 sensors, 1 target. For 3+ sources, multiple competing definitions exist and no consensus as of 2025.
- **The outcome is often unobservable** — you need clean labeled outcomes to compute $I(A,B;C)$.

### Why It's Tractable Here

The GNN changes the problem. Instead of computing PID on raw 62-dimensional sensor space (intractable), the HetTGN compresses each entity's multi-sensor history into a 128-dimensional embedding $h_i$. The synergy detection problem becomes:

> **Which combinations of entity embedding trajectories are synergistically predictive of a downstream entity's state transition?**

This is tractable. And it's exactly what the attention mechanism in HetTGN is already doing implicitly — learning which entity combinations matter.

### Recent Work That Makes It Feasible

- **DECI** (Geffner et al. 2022, Microsoft Research) — differentiable causal discovery on learned representations, not raw features. GPU-compatible.
- **AVICI** (Lorch et al. 2022) — amortized causal structure learning using transformers. Learns the causal DAG from data without exhaustive search.
- **Neural PID estimators** (Pakman et al. 2021) — approximate synergy/redundancy using variational bounds, tractable for continuous high-dimensional data.

None of these existed 5 years ago at this scale. They do now.

### The Full Proposed Architecture

```
Sensor Layer (62 tools)
      ↓ [GNN: compress multi-sensor history into entity states]
Entity Embeddings h_i(t) per entity
      ↓ [Causal structure learner: DECI/AVICI on embedding trajectories]
Learned causal DAG: which entity-state combinations synergistically precede which outcomes
      ↓ [PID / synergy detector: identify irreducibly joint combinations]
Synergistic precursor patterns: "vessel rerouting + CFTC positioning + weather anomaly in THIS configuration..."
      ↓ [World-state probability distribution over future entity state transitions]
Output: "78% probability — traceable to this 3-entity synergistic pattern, none of which was sufficient alone"
```

The **causal attribution** ("why it happened") comes from inverting the learned DAG — given an outcome, trace back through the graph to find the minimal set of precursor combinations with synergistic explanatory power. This is Pearl's do-calculus applied to the learned causal structure.

### Is It Possible?

Yes — with caveats:
- GNN foundation is right. Causal discovery on raw sensor data = intractable. On compressed entity embeddings = tractable.
- Sensor density prerequisite is real. Synergistic relationships only detectable if all participating sensors have sufficient history.
- The learned causal DAG will be approximate, not ground truth. That's still far beyond any existing system.
- The synergy component at >3 sources requires approximation. Viable with neural estimators since ~2022.

**This is the actual research frontier this system is pointing toward.** The components exist. Nobody has combined them at this scale on heterogeneous physical/behavioral sensor data.

---

## 7. Current State of Key Files

### Code Changes This Session

| File | Change |
|---|---|
| `scripts/phase40_gnn_backtest.py` | Fixed `KeyError: 'EqualWeight'` → `'equal_weight'` in comparison table (line ~484) |
| `[[phase40_real_data_model_refresh]]` | Added §40.7 with all GNN backtest results, infrastructure docs, conclusion, Phase 41 paths |
| `[[world_state_prediction_methodology]]` | **NEW** — Full methodology research doc (18KB) |
| `[[chat_checkpoint_2026-04-28_phase40_gnn_backtest_results]]` | **NEW** — Intermediate checkpoint (Phase 40 results only) |
| `[[chat_checkpoint_2026-04-28_session_end_full]]` | **THIS FILE** — Full session checkpoint |

### Model Files (`.tirra_pipeline/`)

| File | Size | Date | Notes |
|---|---|---|---|
| `gnn_model.pt` | 23.7 MB | 2026-04-28 17:07 | **Current live model** — 1,858,459 params, 10 epochs, EWC-enabled |
| `gnn_model_pre_phase40.pt` | 2.0 MB | 2026-04-21 15:33 | Backup before Phase 40 retrain |
| `gnn_model_broken_10ep.pt` | 1.6 MB | 2026-04-21 12:56 | The overtrained catastrophic 10-epoch model (from earlier in Phase 40 history) |
| `gnn_model_live.pt` | 1.0 MB | 2026-04-19 03:51 | 5-epoch healthy model from Phase 40.2 (pipeline live runs) |
| `gnn_model_synthetic_backup.pt` | 218 KB | 2026-04-19 12:02 | Pre-real-data synthetic backup |

### Key Code Architecture (unchanged this session)

**`scripts/phase40_gnn_backtest.py` — what it does:**
- `_load_instrument_returns_fast(db_path, entity_ids)`: Direct SQL `WHERE observation_type='instrument_daily' AND entity_id IN (...)` → 1s vs 60s via store. Returns `(dates: list[str], returns: np.ndarray[T,N])`.
- `GNN_LOOKBACK_DAYS = 90`: Constant limiting obs window per fold to 90 days → 26x obs reduction (29,993 vs 778,832).
- `GNNEmbeddingNormStrategy._compute_weights(fold_date, names)`: Builds graph with 90-day obs window, calls `model.forward()`, extracts `||h_i||_2` per instrument, z-scores, applies `softmax(z × temperature)`.
- `GNNValueHeadStrategy._compute_weights(fold_date, names)`: Same graph build, calls `model.predict_value(embeddings)["instrument"][:, 0]`, applies `softmax(v × temperature)`.
- Both strategies: fold-level cache `self._cache[fold_date]` to avoid redundant GNN forward passes.
- Pre-fetch design: `prepare_static()` + `prefetch_observations(since, until)` once at startup (31s), then bisect-slice per fold.
- `model.forward()` only **reads** TGN memory (does not update). Multiple fold calls on same loaded model are safe.

**Confirmed GNN API:**
```python
trainer._graph_builder.prepare_static()                                 # → (id_map, entities, links)
trainer._graph_builder.prefetch_observations(since, until)              # → sorted obs list
trainer._graph_builder.build_from_cached(id_map, links, obs=window)    # → (HeteroData, IDMap, events)
model.forward(data, id_map)                                             # → dict[str, tensor[N_type, 128]]
model.predict_value(embeddings)                                         # → dict[str, tensor[N_type, 1]]
id_map.local_id("instrument", eid)                                      # → int or None
Trainer.load_model(MODEL_PATH, store)                                   # classmethod
```

---

## 8. What Is NOT Done (Open Work)

### Phase 40 — Still Open
- [ ] Phase 40 `status/active` tag in task file not changed to `status/done`, file not moved to `tasks/done/` — intentionally deferred until Phase 41 path is confirmed
- [ ] `Win Rate = 0.000` and `Volatility = 0.0000` cosmetic bug in `_print_result()` not fixed — deprioritized (real metrics are correct)

### Phase 41 — Not Started
The four paths identified for Phase 41 (in order of priority):
1. **Downstream linear ranker on train-fold embeddings** — cleanest fix for in-sample bias. Train a per-fold linear model on train-fold GNN embeddings to predict next-period returns. No GNN retraining. Walk-forward aware. Eliminates the in-sample bias that crippled GNN-ValueHead.
2. **Temperature softmax on EmbNorm** — add `temperature` hyperparameter to EmbNorm to reduce concentration from 44.65% to something reasonable (~5-10%). Quick sanity check.
3. **Combine GNN + momentum/vol signals** — ensemble the GNN score with momentum (already computed in feature set) to get a blended signal.
4. **Walk-forward GNN retraining** — retrain GNN on each train fold. Expensive but eliminates ALL in-sample bias.

### Phase 47 — Historical Backfill (Blocking Phase 48)
- Historical backfill of all 51 tools for 2-5 years not done
- Without this, 92.2% of observations will remain geopolitical_event
- Physical sensors are built (ais_vessel, power_grid, weather, etc.) but have minimal backfill
- **This is the hard prerequisite before Phase 48** (transformer world model + Dreamer RL)
- Density target: ≥500 observations per entity per major entity type

### Phase 48 — Gated
- Transformer world model + Dreamer model-based RL
- **GATED:** Do not start until Phase 47 density audit passes (≥500 obs per entity type average, no entity type below 100) AND Phase 40 retrain on full history confirms GNN architecture is healthy
- AVICI/DECI causal structure learning (synergistic detection) — Phase 48+ research frontier

---

## 9. Operational Rules (Never Forget)

1. **Python path:** Always `/home/becmachlean/anaconda3/bin/python` — NOT `conda run`, NOT `python3`. A search engine venv is activated in the base terminal and breaks imports from the project.
2. **Run scripts in background:** `> /tmp/out.txt 2>&1 &` then `tail -f /tmp/out.txt`. Piping to `grep` buffers output and makes it look empty.
3. **GNN forward does not mutate TGN memory.** Multiple fold calls on the same loaded model are safe. The model's TGN memory was trained on all data (in-sample bias) — this is the Phase 41 issue, not a code bug.
4. **`model.predict_value()` requires the embeddings dict** — must call `model.forward()` first to get the embeddings, then pass to `predict_value()`.
5. **DB path is `.tirra_pipeline/pipeline.db`** (relative to project root). Always confirm with `from pathlib import Path; Path('.tirra_pipeline/pipeline.db').exists()`.
6. **`EqualWeightStrategy.name` returns `"equal_weight"`** (snake_case) — not `"EqualWeight"`. This is the bug we fixed this session.

---

## 10. Git State

```
HEAD → main (a2daf7b) fix: pop memory buffers before load_state_dict — strict=False doesn't skip shape mismatches
```

All Phase 40 work (task file update, research doc, backtest script fix) is uncommitted as of log-off. The changes are:
- `scripts/phase40_gnn_backtest.py` (1-line fix)
- `[[phase40_real_data_model_refresh]]` (added §40.7)
- `[[world_state_prediction_methodology]]` (new file)
- `docs/memory/` (new checkpoint files)

---

## 11. Roadmap from Here (Source of Truth: [[quant_training_ground]])

```
DONE:   Phase 46 — EWC online learning (2026-04-23)
DONE:   Phase 40 — GNN retrain + backtest (this session confirms Phase 40 complete)
NEXT:   Phase 47 — Historical backfill all 51 tools (2-5 years per tool)
THEN:   Phase 40 RETRAIN on dense history (the real test — current backtest used sparse data)
THEN:   Evaluation pass using four-tier framework (IC → event pred → transfer entropy → Polymarket)
GATE:   Phase 48 — Transformer world model + Dreamer RL (gated on density audit)
FUTURE: Synergistic causation / PID framework — Phase 48+ research frontier
```

---

## 12. Conceptual Summary (For Cold-Start Context)

**What TirraMind is:** A world-state inference machine, not a price predictor. It observes physical and behavioral reality through 62 sensor tools, compresses the multi-entity state into heterogeneous graph embeddings, and produces probability distributions over future entity-state transitions. Financial signal is a downstream readout of correct world-state inference.

**The moat:** Cross-domain synergistic relationships detectable only when 3+ independent sensor domains are observed simultaneously, at least one of which is L0 physical (vessel positions, grid load, weather). These relationships are invisible to any analyst with a single data terminal.

**The evaluation mistake we corrected:** Portfolio Sharpe conflates signal quality, signal-to-weight translation, and allocation quality into one number. For an unproven signal, the right first diagnostic is IC (rank correlation between predicted scores and actual returns). But even IC is just a proxy — the real test is whether the GNN correctly predicts entity-state transitions (GDELT events, CFTC regime shifts, price breakouts) before they are observable in prices.

**The research frontier:** Synergistic causation via Partial Information Decomposition (PID). The ability to say "this outcome required the co-occurrence of these three entity-state changes — none was sufficient alone — and that combination is only detectable if you were watching all three sensors simultaneously." This is the hardest problem in causal inference, made tractable by the GNN's ability to compress the 62-sensor space into entity embeddings, and by recent neural PID estimators and amortized causal structure learning (DECI, AVICI).

---

## Related
- [[phase40_real_data_model_refresh]]
- [[world_state_prediction_methodology]]
- [[phase41_model_refresh_hardening]]
- [[quant_training_ground]]
- [[real_data_model_refresh]]
- [[temporal_het_gnn]]
- [[living_system_online_gnn]]
- [[cross_entity_l3]]
