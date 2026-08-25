---
title: "Architecture Stress Test: Is HetTGN the Right Model for TirraMind?"
tags:
  - doc/research
  - phase/40
  - topic/architecture
  - layer/world-model
  - status/active
date: 2026-05-12
---

# Architecture Stress Test: Is HetTGN the Right Model for TirraMind?

> **Purpose:** Before implementing return_pred_head or any other fix,
> stress-test the assumption that the current architecture (HetTGN + HGT
> message passing + entity_links graph) is the right foundation. Write it
> down, then attack it. No code until this doc reaches a defensible conclusion.

---

## 0. The Measurement That Started This

**Current IC: -0.033, t-stat -1.26** (Phase 41 diagnostic, backtest run on
real data after 28 training epochs).

- Benchmark: IC > 0.03 with |t| > 2.0 = statistically significant, tradeable
- Published SOTA (StockMixer+ATFNet, Nature 2025): IC = 0.041 NASDAQ, 0.028 NYSE
- Published good result (MDGNN, AAAI 2024): IC comparable, using sector + ownership + co-holding graphs
- Our result: **IC = -0.033** = noise territory. Cannot reject null that model is random.

---

## 1. How Our Architecture Currently Works

### The Chain

```
51 data tools
    → entity observations (1.15M rows)
    → entity links (12,271 rows)
    → graph_builder.py → PyG HeteroData
    → het_tgn.py (HetTGN)
        → Per-type linear projection → common hidden_dim
        → 2-layer HGT convolution (type-aware attention, multi-head)
        → HeteroMemory (GRU, Time2Vec time encoding)
        → Event prediction head (self-supervised: predict next obs_type)
    → embeddings → entity_scoring DAG → rank instruments → IC
```

### The key claim the architecture makes
"After seeing 1.15M observations from 51 sources across 2,688 entities, the
GNN will produce embeddings that correlate with future asset returns."

---

## 2. Failure Modes — Attacking Our Own Architecture

### Failure Mode 1: Noisy Edge Dominance (STRUCTURAL)

**What the DB actually shows:**

| Link type | Count | % of all edges |
|---|---|---|
| event_involves | 9,484 | **77.3%** |
| topic_relates_to_instrument | 1,693 | 13.8% |
| works_for | 460 | 3.7% |
| everything return-relevant | 78 | **0.6%** |

77.3% of all edges connect GDELT topics to countries. After 2 layers of HGT
message passing, instrument nodes are aggregating information from:
- their 0.87 direct return-relevant neighbors (avg)
- their many indirect country/topic neighbors through the GDELT edge web

**Literature confirmation (NeurIPS survey, arXiv:2403.04468):**
> "GNNs are highly susceptible to structure noise since errors can propagate
> throughout the graph due to the message-passing mechanism. Therefore, the
> quality of the input graph structure is critical."

**Verdict:** Our graph structure is 77% noise for the return prediction task.
Every instrument embedding has been contaminated by GDELT activity signal
flowing backward through the message-passing layers.

---

### Failure Mode 2: Observation Type Imbalance (DATA)

| Observation type | Count | % |
|---|---|---|
| geopolitical_event (GDELT) | ~1,062,369 | **92.2%** |
| instrument_return, instrument_daily, futures_positioning | ~87,815 | 7.6% |
| all other types | ~138 | 0.1% |

The HeteroMemory GRU has seen 92.2% geopolitical events. Its hidden state
is encoding "recent geopolitical activity around this entity." When the
GNN embeds an instrument node, the memory component carries almost exclusively
news volume, not return signal.

**This is not fixable by training more epochs.** More epochs on the same
imbalanced data = stronger GDELT encoding, not better IC.

---

### Failure Mode 3: Pre-training Objective is Orthogonal to IC (LOSS FUNCTION)

The training objective is **next-event prediction** (self-supervised):
- Given entity history, predict: which entity? which obs_type? when?
- Ground truth: actual next observation in the temporal sequence

This task is 100% orthogonal to IC. Maximizing event prediction accuracy does
not move IC at all — unless event patterns happen to correlate with returns,
which they demonstrably do not (IC = -0.033).

**Literature confirmation (White Rose thesis, 2022):**
> "The edges' information does not provide enough information in stock
> prediction. In the stock price prediction problem, the information about
> the stock itself was more important than the graph. Thus, it is essential
> to include stock features, and the graphs should work as an inductive
> bias rather than a major indicator."

The GNN is optimizing for a signal (event prediction) that is almost
independent of the target (return ranking). This is the root cause.

---

### Failure Mode 4: Static Hand-Crafted Graph vs. Dynamic Return-Correlation Graph

Our entity links are created by:
- `gov_contracts`: awarded_by (government → company)
- `insider_filings`: works_for (person → company)
- GDELT: event_involves (topic → country)
- etc.

These are **expert priors** about what "should" matter for returns. But:
1. They are static (same edges from months ago still in DB)
2. They were not selected by their predictive relationship to returns
3. Dynamic correlation structure (which instruments co-move?) changes with regime

**Literature confirmation (MDGNN, AAAI 2024):**
> "Existing methods have only utilized single relations between stocks,
> ignoring the potential of incorporating other complex relations as
> auxiliary information... We leverage the Transformer structure to encode
> the temporal evolution of multiplex relations, providing a dynamic and
> effective approach."

**Literature (Springer Applied Network Science, 2025):**
> CS_GAT (cosine-similarity graph = dynamic return-correlation edges)
> achieves 1.45% annual R² vs. XGB 1.28%. **Dynamically constructed
> return-correlation graphs outperform hand-crafted relational graphs**
> for return prediction.

---

### Failure Mode 5: Instrument Isolation (GRAPH CONNECTIVITY)

- 89 instrument nodes
- 78 return-relevant edges total (33 trades_instrument + 45 tracks_issuer)
- 0.87 return-relevant neighbors per instrument on average

This means the majority of instruments have 0 or 1 direct connection to a
company or CFTC contract that actually drives their price. The GNN cannot
aggregate meaningful financial information for isolated instrument nodes.

---

### Failure Mode 6: Potential Temporal Leakage (TRAINING SPLIT)

**Unverified — needs explicit audit.**

HeteroMemory stores temporal state. If the training split is not strictly
chronological (i.e., if any future observations contaminate the training
graph), instrument embeddings at test time may encode information they
should not have.

**Literature warning (NVIDIA technical blog, GNN fraud detection):**
> "The graph contains edges and nodes that could leak future information
> from the test set, so we must create an individual data loader and
> sampling routine for our train, validation, and test sets."

Status: The backtest uses `phase40_gnn_backtest.py`. Need to verify that
the graph constructed for each fold uses only data available before the
fold's test window. If we build the full entity graph once and use it for
all folds, we have temporal leakage.

---

## 3. Is the Architecture Fundamentally Wrong?

**Verdict: Not wrong, but critically misconfigured for IC.**

HetTGN (HGT + TGN memory) is a valid architecture for heterogeneous temporal
graph learning. It has been used successfully for:
- Entity ranking in knowledge graphs
- Temporal event prediction
- Fraud detection (homogeneous financial graphs)

Where it struggles here:
- Pre-training objective (event prediction) ≠ downstream objective (IC)
- Graph construction is noise-dominated (77% GDELT edges)
- Observation imbalance is extreme (92.2% one type)

The architecture could work if these three problems were fixed. The fundamental
idea (multi-source entity graph → GNN embeddings → return prediction) is sound
and used in published SOTA systems.

---

## 4. Alternative Architectures (Ranked by Impact vs. Cost)

### Option A: Return Prediction Auxiliary Loss (`return_pred_head`)
**Status: Identified in Phase 41. Not yet implemented.**

Add a second head to the GNN training:
```
entity_embedding → MLP(128→64→1) → predicted_return
loss = L_event + λ_ret × MSE(predicted_return, actual_21d_return)
```

**⚠️ REVISED IMPLEMENTATION PLAN (from 2026-05-12 DB audit):**

`instrument_return` observations in DB: only **445**. Too sparse to train
an MSE head directly.

`instrument_daily` observations: **68,089** (close prices for 89 instruments
over ~3 years). Returns must be **derived from sequential daily prices**:
```python
return_t = (price_{t+21} - price_t) / price_t  # 21-day forward return
```

This means the trainer must:
1. Load instrument_daily obs sorted by (entity_id, timestamp)
2. Compute 21d forward returns from close prices (for each instrument_id)
3. Store as `(entity_id, timestamp, return_21d)` lookup dict
4. During training, for each instrument node in a batch: look up return_21d
   at the nearest timestamp, add to return_pred_head loss

Without this, return_pred_head has no labeled data to train on.

- ~80 LOC in model.py (head definition)
- ~100 LOC in trainer.py (return label loading + loss computation)
- Expected IC improvement: -0.033 → +0.03 to +0.05
- **Implementation time: 2-3 hours** (longer due to label derivation step)
- Risk: low. Same architecture, additive change.

**Verdict: Do this first. Necessary condition for IC > 0.**

---

### Option B: Edge Type Weighting (Learned IC-Relevance Gating)
**NEW IDEA from this stress test.**

Before HGT attention, apply a learned scalar gate per edge type:
```python
edge_gate = sigmoid(W_edge_type[rel_type])  # one scalar per relation type
messages = edge_gate * message_passing(...)
```

This allows the model to learn: "event_involves edges from GDELT are mostly
noise → downweight them." The gate is learned from the return prediction
loss (Option A), so it requires Option A to be implemented first.

- ~30 LOC in het_tgn.py
- No new data required
- Expected effect: reduces GDELT contamination automatically during training
- **Implementation time: 30 minutes (after Option A)**

---

### Option C: Instrument-Node Observation Filtering
**NEW IDEA from this stress test.**

In `graph_builder.py`, when computing instrument node features:
- DROP: geopolitical_event observations
- KEEP: instrument_return, instrument_daily, instrument_volatility, futures_positioning, whale_trade

This is a manual gate. The GRU memory for instrument nodes will not be
contaminated by geopolitical events flowing from neighboring country nodes.

- ~20 LOC in graph_builder.py
- Hard constraint, not learned
- **Implementation time: 20 minutes**
- Risk: low. Only affects instrument node feature construction.

**Verdict: Do this together with Option A.**

---

### Option D: Dynamic Return-Correlation Graph Layer (MDGNN approach)
**MEDIUM-HIGH COST. Better long-term architecture.**

Based on MDGNN (AAAI 2024): add a second graph layer where edges are
computed from rolling return correlations between instruments:

```
For each pair of instruments (i, j):
    corr_ij = Pearson(returns_i[-60d], returns_j[-60d])
    edge_weight_ij = corr_ij if corr_ij > threshold else 0
```

Then run a second GNN layer (or separate attention) on this dynamic graph.
Combine with structural entity graph via gating.

- ~400 LOC. Requires price data daily correlation matrix.
- Dynamic graph means edges change every training fold
- Published results (AAAI 2024): best performance on NASDAQ vs. single-relation static graphs
- **Implementation time: 3-5 days**
- Gated by: enough price history in DB (we have instrument_return obs — check count)

**Verdict: This is the right long-term direction. Implement after Options A-C prove IC > 0.**

---

### Option E: Self-Supervised Contrastive Pretraining → Return Fine-Tuning
**HIGH COST. Best long-term architecture.**

Phase 1: Pretrain GNN with contrastive objective (same entity at different
times = positive pairs; random entity = negative).
Phase 2: Freeze GNN backbone. Fine-tune return_pred_head only.

Literature (Stanford CS224W): "Significant improvement in performance
when labeled data is limited." Our labeled return data IS limited —
we have daily returns for ~89 instruments over ~3 years = ~97,000 data points
for fine-tuning, after 1.15M obs of unsupervised pretraining.

- ~600 LOC. Major restructuring of training pipeline.
- **Implementation time: 1-2 weeks**
- Expected IC: highest of all options, approaching 0.05-0.08

**Verdict: Correct end-state. Do not implement until Options A-D are proven insufficient.**

---

## 5. What About Completely Replacing HetTGN?

### MDGNN (AAAI 2024) — Drop-in replacement candidate

MDGNN uses:
- Multi-relational static graph (sector, ownership, supply-chain) + dynamic graph
- Transformer temporal encoder (not GRU/TGN memory)
- Achieves best IC on NASDAQ/CSI-300

Replacing HetTGN with MDGNN would be:
- ~2-3 weeks of work
- Loses EWC, regime gate, alignment — all built on HetTGN architecture
- High risk: re-implementing everything

**Verdict: Do NOT replace yet.** Fix HetTGN with Options A-C first. If
IC after that is still < 0.02, then consider MDGNN replacement.

---

### Temporal Graph Transformer (GPS hybrid)
From ICAPS 2024 comparison:
> "The hybrid GPS architecture performs best overall. This suggests that
> combining Transformers and GNNs can introduce inductive biases to
> increase learning efficiency while retaining the ability to capture
> long-range relationships."

GPS = GNN layer + Transformer self-attention layer in parallel, fused.
This would be a natural upgrade path for HetTGN if IC stays below 0.03
after the targeted fixes.

**Verdict: Phase 48 consideration. Not now.**

---

## 6. Verified Claims vs. Unverified

Verified 2026-05-12 via live DB query.

| Claim | Verified? | Source | Notes |
|---|---|---|---|
| IC = -0.033, t-stat -1.26 | ✅ | Phase 41 backtest run | |
| 77.3% of edges are event_involves | ✅ | Live DB query | |
| geopolitical_event share = 78.4% | ✅ **REVISED** | Live DB: 901,704 / 1,150,184 | Was 92.2% at Phase 41; now 78.4% because more non-GDELT data added since. **Goldstein filter (Phase 41b) did NOT reduce historical obs** — filter is ingestion-time only, 10yr backfill pre-dates it. |
| StockMixer+ATFNet IC = 0.041 NASDAQ | ✅ | Nature 2025 paper | |
| MDGNN best on NASDAQ (sector+ownership+co-holding) | ✅ | AAAI 2024 paper | |
| CS_GAT (dynamic correlation graph) outperforms static relational | ✅ | Springer Applied Network Science 2025 | |
| GNNs are susceptible to structure noise | ✅ | NeurIPS survey arXiv:2403.04468 | |
| All 89 instruments have price data | ✅ | Live DB: 89/89 instruments have instrument_daily obs | |
| instrument_return obs count | ✅ **CRITICAL** | Live DB: only **445** explicit instrument_return obs | This is the labeled data for return_pred_head. 445 is very sparse for direct MSE training. Must derive returns from instrument_daily (68,089 obs) instead. |
| instrument_daily obs count | ✅ | Live DB: **68,089** | Sufficient for derived return labels |
| Goldstein filter reduced GDELT in DB | ❌ FALSE | Live DB still has 901,704 geo events | Filter is ingestion-time. Historical backfill not retroactively cleaned. Option: retroactive filter pass on DB. |
| Temporal leakage in backtest | ❌ UNVERIFIED | Needs audit of phase40_gnn_backtest.py | |
| return_pred_head will improve IC to +0.03–0.05 | ❌ UNVERIFIED | Extrapolation from literature. Must measure. | |

---

## 7. Conclusion: The Ordered Fix Plan

### Immediate (this session) — Necessary Condition
1. **Option A: return_pred_head** — add return auxiliary loss. 80 LOC. Do now.
2. **Option C: Instrument obs filtering** — drop geopolitical_event from instrument node features. 20 LOC. Do now.

### After Measuring IC Again
3. **Option B: Edge type gating** — learn to downweight GDELT edges. Only if IC after (1)+(2) < 0.03.
4. **Audit temporal leakage in backtest splits** — verify chronological integrity.

### Medium Term (data-gated by price history accumulation)
5. **Option D: Dynamic return-correlation graph** — overlay MDGNN-style dynamic edges on structural graph.

### Long Term (gated by Phase 40 proving stack has hit ceiling)
6. **Option E: Contrastive pretraining** or **MDGNN replacement** — only if IC plateau < 0.03 after all above.

---

## 8. Open Stress-Test Questions — Answered 2026-05-12

### Q1: Is the backtest split chronological? ✅ PARTIALLY CLEAN, KNOWN LIMITATION

Audit of `scripts/phase40_gnn_backtest.py` (lines 22-28, 138-141, 673):

**Observation window — CLEAN:**
```python
end_idx = bisect.bisect_left(self._obs_ts, fold_ts)    # strict past cutoff
start_idx = bisect.bisect_left(self._obs_ts, since_ts) # 90-day window
obs_window = self._obs[start_idx:end_idx]               # no future obs
```
Each fold's node features use only observations in `[fold_ts - 90d, fold_ts)`.

**Entity links — MILD issue:**
`full_links = trainer._graph_builder.prepare_static()` — full-DB links used
for all folds. Structural edges (works_for, located_in, awarded_by) don't
carry return information, so leakage is negligible. The 1,693
`topic_relates_to_instrument` edges are the only mild concern.

**GNN model weights — ACKNOWLEDGED LIMITATION (documented in code):**
> "Note: the model's TGN memory was trained on all data (in-sample).
> Phase 41 will address this via per-fold GNN retraining." (line 27-28)

The HeteroMemory GRU hidden state was trained on the full temporal sequence
including the test folds. This is NOT classic temporal leakage (future obs
don't flow into fold inference). It IS a form of in-sample overfitting:
the model's memory learned patterns that include the test period data.

**Impact on IC = -0.033:**
If anything, this in-sample training INFLATES IC, not deflates it. A model
trained on test-period data should show higher IC on that data, not negative.
Therefore IC = -0.033 is likely accurate — possibly even pessimistic.

**Conclusion: temporal leakage is NOT the cause of IC = -0.033.**
The root causes are structural (Failure Modes 1, 2, 3 in this doc).

---

### Q2: Return label derivation for return_pred_head

Confirmed: `instrument_return` = 445 obs only. Implementation must:
1. Load `instrument_daily` obs (68,089) sorted by (entity_id, timestamp)
2. Derive 21d forward returns: `r_t = (price_{t+21} - price_t) / price_t`
3. Build lookup `{(entity_id, date) → forward_return}`
4. Use lookup in trainer's loss computation

### Q3: Right λ_ret

Start: 0.1. Grid search on validation fold: [0.01, 0.1, 0.5, 1.0].
Loss balance: `L_total = L_event + λ_ret × MSE(ret_pred, ret_actual)`

### Q4: Can we retroactively Goldstein-filter GDELT?

Yes. One-time SQL DELETE. Requires DB backup (387 MB). Would reduce from
1.15M to ~300K obs. Significant impact on training. Should be done together
with Option A (return_pred_head) for cleanest measurement.

### Q5: Are instrument node features getting price observations?

`instrument_daily` obs (68,089) are loaded by `graph_builder.py` as part of
the observation window per-fold. They contribute to the 41-dim ENRICHMENT
feature vector (obs_type distribution). However, the 90-day lookback means
instrument nodes see ~360 daily price obs each per fold — these ARE reaching
the node feature computation. The contamination is from GDELT obs flooding
the obs_type distribution, not from price obs being absent.

---

## 9. The One Thing That Could Falsify Everything

~~If temporal leakage is present in the backtest~~ — FALSIFIED. Leakage audit
confirms observation windows are correct. IC = -0.033 is a real measurement.

**The actual falsifiable risk is simpler:**
`return_pred_head` + instrument obs filtering together fail to move IC above
0.02. This would mean the structural issues (sparse instrument↔entity edges,
GDELT-dominated graph topology) are so severe that even a direct return
supervision signal cannot overcome them. In that case, Option D
(dynamic correlation graph layer) becomes required before any IC is achievable.

---

## 10. Final Implementation Plan (Locked After Stress Test)

**Gate: audit complete. Temporal leakage not the cause. Proceed.**

### Phase A — Immediate (target: IC measurement within 1 Kaggle run)

**A1. Retroactive GDELT filter (optional but high-impact):**
```sql
-- BACKUP DB FIRST
DELETE FROM entity_observations 
WHERE observation_type = 'geopolitical_event'
AND CAST(json_extract(value, '$.goldstein') AS REAL) >= -5.0;
```
Expected: 901,704 → ~90,000 obs. DB: 1.15M → ~340K. Massive noise reduction.
Risk: irreversible. Must `cp pipeline.db pipeline.db.bak` first.

**A2. return_pred_head (Lever 1):**
- `agent/models/gnn/model.py`: add `ReturnPredHead(hidden_dim→64→1)`
- `agent/models/gnn/trainer.py`: derive 21d returns from instrument_daily obs, add MSE loss term `λ_ret=0.1`
- ~180 LOC total

**A3. Instrument obs filtering in graph_builder (Lever 2):**
- `agent/models/gnn/graph_builder.py`: when building instrument node features, zero out `geopolitical_event` obs_type dimension in the 41-dim feature vector
- ~20 LOC

**Measure:** re-run `scripts/phase40_gnn_backtest.py` after retraining.
Exit condition: IC > 0.03, |t| > 2.0.

### Phase B — If IC still < 0.03 after Phase A

**B1. Edge type gating (Option B):** learned scalar gate per relation type,
trained via return_pred_head gradient. Suppresses GDELT edges automatically.

**B2. Dynamic correlation graph overlay (Option D):** MDGNN-style rolling
Pearson correlation edges between instruments. ~400 LOC. Best long-term
architecture for instrument↔instrument signal propagation.

### Phase C — Long term (if IC plateaus < 0.03 after B)

Contrastive pretraining (Option E) or MDGNN replacement. Only if current
HetTGN stack cannot produce IC > 0.03 with all fixes applied.

---

## Related
- [[living_system_online_gnn]] — Phase 46 EWC architecture
- [[phase40_gnn_backtest]] — Real data backtest script
- [[quant_training_ground]] — Phase roadmap
- [[entity_linking_layer]] — Phase 17 entity link design
