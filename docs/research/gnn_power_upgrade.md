---
title: "Research: GNN Power Upgrade — Architecture, Data, and Multi-Hypothesis Strategy"
tags:
  - doc/research
  - phase/42
  - topic/gnn
  - layer/world-model
  - status/active
---

# Research: GNN Power Upgrade

**Goal:** Identify every axis on which the HetTGN can be made more powerful — architecturally,
data-wise, entity/link topology, and via parallel multi-hypothesis model experiments on Kaggle.

**Current baseline (post-Phase 42 data fix):**
- ICIR = 0.115 (EmbNorm), 0.134 (ValueHead), 0.099 (ReturnHead)
- 1.87M params, hidden_dim=128, 2 HGT layers, 2 heads
- 982,650 obs | 2,502 entities | 12,048 links
- Dominant obs type: GDELT (92.2%)

---

## 1. Architecture Axes

### 1A. Current architecture (what we have)
```
Per-type Linear projection → HGT (2 layers, 2 heads) → HeteroMemory GRU → 3 prediction heads
```
References: HGT (Hu et al. arXiv:2003.01332), TGN (Rossi et al. arXiv:2006.10637),
Time2Vec (Kazemi et al. arXiv:1907.05321)

### 1B. SE-HTGNN (NeurIPS 2025 — strongest recent baseline)
**arXiv:2510.18467** — "Simple and Efficient Heterogeneous Temporal GNN"
- Key idea: **integrates temporal modeling directly INTO spatial learning** via a novel
  dynamic attention mechanism that retains attention from historical graph snapshots
  to guide subsequent attention computation
- Also: uses LLM to inject semantic node-type priors as prior knowledge
- Result: 10× speed-up over SOTA with best forecasting accuracy
- **What this means for us:** our architecture decouples temporal (Time2Vec/GRU) and
  spatial (HGT) — SE-HTGNN fuses them. This is the state-of-the-art paradigm shift.
- Source: CC-BY licensed paper. Concept reuse is safe; code reuse requires checking their repo.
- **Priority:** High — implement the coupled spatio-temporal attention as Hypothesis B

### 1C. THGNN (financial-specific, deployed in live trading system)
**arXiv:2305.08740** — "Temporal and Heterogeneous GNN for Financial Time Series"
- Key idea: **transformer encoder** for per-asset time series → **heterogeneous graph
  attention** aggregates across asset types. Jointly optimizes temporal + structural embeddings.
- Deployed in a real quantitative trading system (S&P 500, CSI 300). 
- Key lesson: using a transformer encoder per instrument node (not just GRU) captures
  price series autocorrelation that GRU misses.
- **What this means for us:** add a lightweight Transformer encoder block on the instrument
  node features (price returns + CFTC positioning) before they enter HGT message passing.
  Cost: ~200K extra params. Impact: potentially high for the 19 instrument nodes.
- Source: ACM CIKM 2022. Conceptual inspiration only — implement independently.
- **Priority:** Medium-High — implement as Hypothesis C

### 1D. Hidden dimension scaling
- Current: hidden_dim=128. Model capacity = 1.87M params.
- At 982K observations, the model is slightly underfit (ratio = ~525 obs/param = fine).
- Going to hidden_dim=256 = ~7M params. Would require more epochs to converge.
- **Rule:** Only useful if loss is still rapidly declining at epoch 40. Check loss curve first.
- **Priority:** Low until epoch 40 results are known.

### 1E. Depth: 2 → 3 HGT layers
- Currently: 2-hop message passing (entity → neighbour → entity)
- 3 hops: country → commodity → CFTC contract → commodity (triangular)
- Risk: oversmoothing if graph is not dense enough
- **Priority:** Medium. Try 3 layers as Hypothesis D.

### 1F. Multi-head attention: 2 → 4 heads
- More heads = more diverse attention patterns learned per edge type
- Cost: minimal (~5% parameter increase for HGT)
- **Priority:** Medium. Bundle with Hypothesis D (3 layers + 4 heads).

### 1G. GRU → LSTM in HeteroMemory
- LSTM has explicit forget gate — better for long-horizon temporal patterns
- Useful when commodity signals have seasonality (e.g. agricultural cycles 3-12 months)
- Cost: ~20% more params in memory module only (~300K extra)
- **Priority:** Low-Medium. Try as Hypothesis E.

### 1H. Graph Contrastive Learning (self-supervised pre-training boost)
Paper: "Dynamic Graph Representation with Contrastive Learning for Financial Market Prediction" (ICAART 2025)
- Idea: augment the graph by randomly dropping edges (pe probability), compute
  two views of the same entity, minimize contrastive loss between the views
- This forces embeddings to be ROBUST to missing edges — critical when our graph
  has sparse connectivity per instrument node
- Implementation: add CL loss term = InfoNCE(z_i, z_i_aug) to training loop
- Cost: ~50 lines of code, ~15% compute overhead per epoch
- **Priority:** High. Complements return_pred_head. Implement as Hypothesis F.

---

## 2. Data Axes (Highest Leverage)

### 2A. Fix the observation type imbalance (CRITICAL)
Current distribution: GDELT = 92.2% of all obs. This means:
- The GNN embedding is dominated by "geopolitical activity proxy"
- CFTC, AIS, insider filings, supply chain = collectively ~8% → barely affects embeddings

**Fix options:**
1. **GDELT fraction cap (already implemented):** `--gdelt-frac 0.05` during training = caps GDELT
   sampling at 5% of batch. Already in notebook. This is the most important single fix.
   VERIFY this is being used correctly in current Kaggle run.
2. **Observation-type resampling:** oversample rare obs types (CFTC, insider, AIS) in batches.
   Implementation: weighted sampler in trainer.py where weight[obs] = 1/count[obs_type]
3. **Source-specific embedding heads:** separate projection per obs-type group before HGT.
   E.g. financial_proj for CFTC/positioning, event_proj for GDELT, supply_proj for AIS.

**Priority:** CRITICAL. Must resolve before architectural changes matter.

### 2B. CFTC signal enrichment (partially done)
Done: 19 cftc_tracks (was 5), 5,080 observations.
Still missing:
- **Net positioning direction** as a signed feature: large_spec_net_long / open_interest
  = the key momentum/contrarian signal cited by every hedge fund using COT data
- **Positioning extremes** (95th percentile long, 5th percentile = historically bearish)
  = level signal, not just delta
- **Change in positioning week-over-week** = flow signal (speed)
Current CFTC obs stores the raw numbers. Compute these 3 derived features and add as
separate observation types: cftc_net_position (signed), cftc_positioning_extreme (0/1),
cftc_position_delta (week-on-week change).
**Priority:** High. Pure data transformation, no new API calls needed.

### 2C. AIS vessel tracking → supply route signals
Currently: AIS vessels exist as entities but weakly connected to commodity instruments.
Missing link: vessel cargo type → commodity.
Fix: add `carries_commodity` link type between vessel entities and instrument entities.
Data: MMSI → vessel type → cargo type → commodity (e.g. VLCC tanker → crude oil).
Effect: GNN can now route signals from "vessel congestion at Strait of Hormuz" → oil instrument.
**Priority:** High. ~50 lines in seed script.

### 2D. Weather/climate signals as country-node features
Missing entirely from current graph.
Free sources: NOAA ENSO index (La Niña/El Niño), Palmer Drought Severity Index, seasonal
temperature anomalies.
Why it matters: Agricultural commodity futures (corn, soybeans, wheat) are heavily driven
by weather in key production regions. If we add weather obs to country nodes + producer links,
the GNN has a direct path: weather(Brazil) → produced_in → Soybeans instrument.
**Priority:** High for agricultural commodities. Research task for Phase 43.

### 2E. Supply/demand balance data
Currently missing: physical commodity balance sheets.
Free source: EIA Short-Term Energy Outlook (oil, gas), USDA WASDE (agricultural).
These give actual supply-demand surplus/deficit numbers — the most direct predictor
of commodity prices beyond positioning.
**Priority:** Medium-High. Would require a new surveillance tool.

### 2F. Satellite/alternative data signals
Lower priority but asymmetric edge (nobody else uses these publicly):
- Satellite crop monitoring: MODIS NDVI (vegetation index per country) → yield estimate
- Shipping congestion: port call frequency from AIS data we already have
- Night-light intensity: proxy for economic activity in key production regions
**Priority:** Medium-Long term.

---

## 3. Entity and Link Topology Axes

### 3A. Current entity types and their connectivity
| Entity type | Count | Avg links | Assessment |
|---|---|---|---|
| country | ~170 | High (GDELT, macro) | Well-connected |
| instrument | 19 | Low (CFTC + produced_in) | Still thin |
| cftc_contract | 19 | 1 each (cftc_tracks) | Fine |
| vessel | ~100+ | Low | Isolated |
| company | unknown | Low | Weakly connected |
| wallet/protocol | few | Low | Ignore for now |

### 3B. Missing link types that would have high impact
1. **`carries_commodity`**: vessel → instrument (AIS cargo type mapping)
2. **`weather_affects`**: country → instrument (weather → agricultural production)
3. **`benchmark_of`**: WTI → Brent, Henry Hub → TTF (pricing relationship)
4. **`substitute_for`**: Coal → Natural Gas, Corn → Soybeans (cross-commodity competition)
5. **`denominated_in`**: commodity → currency (USD, EUR, CNY pricing)
   All commodity instruments denominated in USD → `denominated_in` → USD FX entity
   Effect: GNN can propagate dollar-index movements into commodity embeddings.
6. **`supply_route_through`**: country → country (commodity trade route)
   E.g. Saudi Arabia → Strait of Hormuz → Japan (oil supply route)

### 3C. Instrument sub-type edges
Currently all 19 instruments are flat. Missing:
- Energy cluster: WTI, Brent, Nat Gas, Heating Oil → all connect to same geopolitical events
- Metals cluster: Gold, Silver, Copper, Platinum → all sensitive to USD/real rates
- Agricultural cluster: Corn, Wheat, Soybeans, Coffee, Sugar → weather + USDA driven

Adding **`same_sector`** edges within these clusters lets the GNN do cross-instrument
message passing within an asset class, which is how professional traders think.

---

## 4. Multi-Hypothesis Parallel Model Strategy

### 4A. The core idea
Instead of running one model configuration and waiting to see if it improves,
run MULTIPLE different model variants in PARALLEL on Kaggle (or sequentially in
separate notebooks). Each has a distinct hypothesis.
Measure IC for each. The "truth" is which hypothesis produces the highest ICIR.
This is the scientific way to do ML: concurrent A/B testing.

### 4B. Proposed hypothesis matrix (to run concurrently)

| Hypothesis | Arch change | Data change | Expected IC delta | Cost |
|---|---|---|---|---|
| **H-A (baseline)** | Current HetTGN h=128 | GDELT frac 0.05 | +0.019 (current) | 0 |
| **H-B (SE-HTGNN style)** | Fused spatio-temporal attention | Same data | +0.03? | Medium |
| **H-C (Transformer instrument head)** | Transformer encoder on instrument nodes | Same | +0.04? | Medium |
| **H-D (deeper + wider)** | 3 layers, 4 heads, h=128 | Same | +0.01? | Low |
| **H-E (LSTM memory)** | GRU → LSTM in HeteroMemory | Same | +0.005? | Low |
| **H-F (CL augmentation)** | Add contrastive loss | Same | +0.02? | Low |
| **H-G (CFTC features)** | Same arch | + 3 derived CFTC features | +0.03? | Low (data only) |
| **H-H (vessel links)** | Same arch | + carries_commodity links | +0.01? | Low (data only) |

### 4C. Execution plan for parallel hypotheses

**Phase 1 (now, Kaggle):** Run H-A (baseline) to epoch 40. Measure IC.
**Phase 2 (next sprint):**
- Implement H-G (CFTC derived features) and H-H (vessel links) — data-only changes.
  These can reuse the existing model weights (no retrain). Run ablation to measure ΔIC.
- Implement H-F (contrastive learning) — add ~50 lines to trainer.py. Retrain from epoch 30.
**Phase 3:** Implement H-C (transformer instrument head) — most complex. Full retrain.
**Phase 4:** SE-HTGNN style refactor (H-B) — significant architecture change. Do last.

### 4D. Evaluation protocol for each hypothesis
Run `scripts/phase40_gnn_backtest.py` after each variant.
Primary metric: **ICIR** (mean IC / std IC). Secondary: Sharpe ratio.
Log each result to an experiment manifest via `experiment_tracker.py`.
Use `compare_experiments.py --latest 2` to diff any two experiments.

**ICIR thresholds:**
- > 0.40 = real signal, activate Kalman layer
- 0.25–0.40 = directional signal, continue improving
- 0.10–0.25 = weak signal, diagnose further
- < 0.10 = noise, do not activate downstream layers

---

## 5. Training Process Axes

### 5A. Loss function improvements

**Current loss terms:**
1. Self-supervised: next obs_type prediction (CE)
2. Self-supervised: time delta prediction (MSE)
3. Contrastive: link prediction (margin loss)
4. Supervised: return prediction (MSE on 21d log returns)
5. ListNet ranking loss (from Phase 41b)

**Missing:**
- **Cross-entropy on return direction** (up/down classification) in addition to MSE.
  MSE penalizes magnitude errors. CE on direction penalizes sign errors.
  Direction accuracy maps directly to IC. Use both.
- **Sharpe-ratio-maximizing loss**: $L = -\frac{\mu(r \cdot \hat{r})}{\sigma(r \cdot \hat{r})}$
  where $r$ = actual returns, $\hat{r}$ = predicted scores. Directly optimizes the metric we care about.
  Reference: "Portfolio-based ML" literature (Sharpe loss / Kelly loss).

### 5B. Curriculum learning
Currently all observations are sampled uniformly regardless of age.
**Improvement:** Weight more recent observations higher during training.
Recency weight: $w_t = e^{-\lambda (T - t)}$ where $\lambda$ = decay rate (e.g. 0.01/day).
This makes the model focus on recent patterns while still learning from history.

### 5C. Observation-type resampling
As noted in 2A: weight rare obs types (CFTC, insider) more heavily in batch sampling.
Concrete: in trainer.py's batch sampler, compute per-type count, assign
$w_{obs} = 1 / \sqrt{count_{obs\_type}}$ (inverse square root smooths extreme imbalances).

---

## 6. Architecture Decision Summary

### Recommended implementation order (by effort/expected impact ratio)

**Quick wins (< 1 day each, implement before next Kaggle run):**
1. Verify `--gdelt-frac 0.05` is working correctly (check loss curve obs_type distribution)
2. Add 3 derived CFTC features (net position, extreme flag, week-on-week delta) → H-G
3. Add `carries_commodity` vessel→instrument links → H-H
4. Add direction CE loss alongside return MSE loss

**Medium effort (1–3 days each, next sprint):**
5. Contrastive learning augmentation (InfoNCE, edge dropout) → H-F
6. Observation-type resampling in batch sampler
7. Add `same_sector` intra-cluster edges for instrument grouping

**Larger architectural changes (schedule separately, full retrain needed):**
8. Transformer encoder on instrument nodes → H-C (3-5 days)
9. SE-HTGNN style fused spatio-temporal attention → H-B (1 week)
10. hidden_dim 128→256 — only if ICIR plateaus at 0.25 and loss still declining

---

## 7. Sources and References

| Source | Type | Key finding | URL |
|---|---|---|---|
| SE-HTGNN (Wang et al., NeurIPS 2025) | Paper | Fused spatio-temporal attention, 10× speed-up, best accuracy | arXiv:2510.18467 |
| THGNN (Xiang et al., CIKM 2022) | Paper | Transformer+HetGAT deployed in live quant system | arXiv:2305.08740 |
| Multi-Hypothesis Portfolio (Rodriguez et al., 2025) | Paper | Structured ensemble with diversity control outperforms on S&P 500 | arXiv:2501.03919 |
| DGRCL (ICAART 2025) | Paper | Dynamic graph + contrastive learning improves stock trend prediction | scitepress.org |
| GCN-LSTM Futures (Michael et al., 2024) | Paper | GCN-LSTM on futures term structure (ES-mini, VIX) | arXiv:2408.05659 |
| CFTC COT reports | Data | Net positioning = key sentiment signal for commodities | cftc.gov |
| HGT (Hu et al., WWW 2020) | Paper | Current architecture base | arXiv:2003.01332 |
| TGN (Rossi et al., 2020) | Paper | Memory-based temporal graph learning | arXiv:2006.10637 |

---

## Related

- [[phase42_ghost_pattern_activation]] — active task file
- [[ghost_pattern_graph_audit]] — graph topology audit
- [[gnn_power_upgrade_spec]] — spec to be written after this research
- [[kaggle_runbook]] — training infrastructure
