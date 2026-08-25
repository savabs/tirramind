---
title: "Research: Phase 41b — GNN Signal Extraction (Making It Work)"
tags:
  - doc/research
  - phase/41b
  - topic/world-model
  - layer/feature-engineering
  - layer/world-model
  - status/active
---

# Research: Phase 41b — GNN Signal Extraction (Making It Work)

**Problem being solved:** IC diagnostic (Phase 41) showed IC=-0.033, t=-1.26 — no statistically significant 
return signal from GNN embeddings. This document explains *why* and *how to fix it*.

---

## Root Cause Analysis (Confirmed)

Three independent root causes explain the zero-IC result:

### RC1: Return loss receives ~0 gradient (loss scale problem)
- `return_weight=1.0` against a dt loss that reaches ~5–10 during training
- The dt loss dominates by 5–10x at epoch 20, but was up to 1,000,000x during earlier gradient explosions
- **Even with the Huber fix, the return head trains on ~10% the signal of the dt head**
- Evidence: `_compute_return_loss()` logs show return_loss=0.0001 while dt_loss=5.26

### RC2: Wrong loss function for cross-sectional ranking
- Huber loss on absolute returns measures "how close is prediction to realized return?"
- IC is a *ranking* metric: Spearman correlation of predicted rank vs realized rank
- Training with MSE/Huber does NOT optimize IC
- A model can minimize Huber loss perfectly while producing IC=0 (predicts the mean)
- **Need a ranking loss: ListNet, LambdaRank, or direct Spearman correlation loss**

### RC3: 92.2% GDELT observations → embedding space dominated by geopolitical activity
- The GNN learns representations that explain the training data distribution
- With 92.2% GDELT event edges, the optimal representation is one that encodes geopolitical activity
- Return signals from insider filings, CFTC positioning, etc. (8% of edges) receive near-zero attention
- This is the "quantity imbalance" problem identified in QTIAH-GNN (KDD 2023)
- **The embedding is a geopolitical activity proxy, not a multi-domain causal encoder**

---

## Literature Survey

### 1. THGNN — Temporal Heterogeneous GNN for Financial Time Series (CIKM 2022)
- **Authors:** Xiang, Cheng et al. | **arXiv:** 2305.08740
- **Pipeline:** Transformer encoder (temporal) → HetGAT (relational) → classification head
- **Key insight:** They construct the relation graph *dynamically* from historical prices, not hand-coded
- **Loss:** Cross-entropy on price movement direction (up/down), NOT absolute return prediction
- **Deployed in production** quantitative trading system at Tongji Finance Lab
- **Repo:** https://github.com/TongjiFinLab/THGNN
- **License:** Check before porting — treat as conceptual inspiration only

### 2. MDGNN — Multi-Relational Dynamic GNN (AAAI 2024, Ant Group)
- **Authors:** Qian, Zhou et al. | AAAI 2024
- **Pipeline:** Discrete dynamic graph → Transformer encoder (temporal evolution of relations) → investment prediction
- **Key insight:** Captures *multifaceted* relations (economic, financial, news, sentiment) — exactly what TirraMind does
- **Multi-task approach:** Economic/financial/news each have separate relation types; the GNN jointly learns
- **Lesson for TirraMind:** Each relation type should be weighted to prevent one domain dominating

### 3. Multi-Task Loss Balancing via Homoscedastic Uncertainty (Kendall et al., CVPR 2018)
- **Paper:** "Multi-Task Learning Using Uncertainty to Weigh Losses" | arXiv:1705.07115
- **Method:** Learn a task-specific log-variance $\log \sigma_k^2$ as a trainable parameter
- **Loss:** $\mathcal{L}_{total} = \sum_k \left( \frac{1}{2\sigma_k^2} \mathcal{L}_k + \log \sigma_k \right)$
- **Why it works:** If task $k$ loss scale is large, the network learns large $\sigma_k$, automatically down-weighting it
- **Implementation:** Two `nn.Parameter(torch.zeros(1))` scalars — `log_var_return`, `log_var_dt`
- **Trusted source:** https://arxiv.org/abs/1705.07115 — 4,000+ citations, production-validated

### 4. LambdaRank / ListNet — Ranking Losses for Cross-Sectional IC
- **LambdaRank:** Burges et al., 2006 (Microsoft Research) | "From RankNet to LambdaRank to LambdaMART"
  - Pairwise loss that weights each pair by the ΔNDCG the swap would cause
  - PyTorch implementation: `torch.nn.functional` compatible — use allRank library
  - **Best for** optimizing NDCG@K (top-K portfolio IC)
- **ListNet:** Cao et al., 2007 | Top-1 approximation — simplest listwise loss
  - `L = -sum_i(p_true_i * log(p_pred_i))` where `p = softmax(scores / tau)`
  - 5 LOC in PyTorch, differentiable, directly optimizes cross-sectional ranking
  - **Best starting point** — low complexity, strong IC signal
- **Spearman correlation loss:** Direct but non-differentiable; use soft Spearman (DeepRank, 2019)
  - `L = 1 - soft_spearman(pred_scores, forward_returns)` where soft_spearman ≈ Spearman via Gaussian CDFs
  - Medium complexity, directly optimizes the IC metric

### 5. Lead-Lag Detection as Temporal Link Prediction (ICLR 2026 submission, withdrawn)
- **Krstev, Rigoni et al.** | OpenReview: KsWRLyIAKP
- **Framing:** Lead-lag relationships = temporal link prediction task on dynamic graphs
- **Key insight:** Model assets as nodes; directed edges = lead-lag interactions; train TGNN to predict edge formation
- **Why withdrawn:** Reviewers wanted Granger causality baselines + larger dataset + backtesting
- **Lesson for TirraMind:** The North Star diagnostic (does entity A embedding at T-N predict instrument B price at T+1?) is the correct framing
- **Metrics they use:** AAUC, MRR, Recall@K — better than IC for the propagation diagnostic

### 6. QTIAH-GNN — Quantity and Topology Imbalance-aware HetGNN (KDD 2023)
- **Addresses:** Exactly the problem TirraMind has — dominant node/edge types in HetGNN
- **Solution:** Class-semantic representation + multi-hierarchy label-aware neighbor selection + class-balance loss
- **Simpler adaptation for TirraMind:** Per-observation-type loss re-weighting in the edge reconstruction task
  - Weight each edge in GNN training loss by $w_{\text{type}} = 1 / \sqrt{N_{\text{type}}}$
  - GDELT edges get weight $\approx 0.032$ (N≈977K), insider filing edges get $\approx 1.0$
  - This forces the GNN to explain ALL edge types equally rather than specializing in GDELT

---

## The Correct Architecture: Perceptual Layer → Downstream Ranker

### What the GNN should be (and IS)
```
Layer 0/1 events (ships, filings, GDELT, AIS, CFTC...) 
    → HetTGN edges
    → GNN embedding h_i ∈ R^128 per entity per time window
    → Encodes: who is connected to whom, how strongly, across all domains
```

The GNN embedding `h_instrument` at time T contains:
- Which companies/insiders recently filed material events (via entity links)
- Which commodity contracts show unusual positioning (via CFTC observations)
- What geopolitical events involving related countries occurred (via GDELT)
- How the company's supply chain is positioned (via AIS vessel data)

This IS a valid pre-emergence causal world-state representation. The problem is that no downstream head has been properly trained to extract return signal from it.

### What needs to be added
```
GNN embedding h_instrument ∈ R^128 (frozen for initial downstream train)
    → Downstream ranker (LightGBM or MLP with ListNet loss)
    → Cross-sectional score s_i (higher = more likely to outperform next month)
    → IC = Spearman(s_i, realized_21d_return_i)
```

Two variants to build and test:

**Variant A: Frozen embeddings → LightGBM LambdaRank**
- Extract GNN embeddings per instrument per 21-day window
- Train LightGBM with `objective='lambdarank'` on train fold
- Walk-forward: re-train ranker each fold, embeddings frozen
- Avoids TGN memory leakage (decoder sees future on training embeddings)
- Similar pattern: `FTabilo-ml/market-ia-trading-bot` (GitHub)

**Variant B: End-to-end GNN + ListNet head (correct loss)**
- Replace Huber return loss with ListNet (softmax cross-entropy on ranks)
- Add homoscedastic uncertainty weighting to balance dt and return tasks
- Add GDELT observation-type down-weighting in edge reconstruction loss
- Retrain on Kaggle, re-run backtest

**Start with Variant B** — it's cleaner and fixes the architectural problem at root.

---

## The North Star Diagnostic: Information Propagation Test

**Question:** Does the embedding of an upstream entity (company insider, GDELT country event, CFTC contract) at time T-N predict the embedding of a downstream instrument at time T?

**Why this matters:** If YES → the GNN is encoding real pre-emergence causal information.
If NO → the GNN is just encoding local structure, not information propagation.

**How to measure it:**
```python
# For each instrument i and each upstream entity type e:
# 1. Extract embedding norm time series: ||h_i(t)||_2 and ||h_e(t)||_2
# 2. Granger causality test: does h_e(t-N) → h_i(t)?
# 3. Multiple N lags: N = 1d, 7d, 14d, 21d
# 4. Report: which entity types Granger-cause which instrument types, at which lags?
```

**Expected finding (based on architecture):** CFTC + insider filings should Granger-cause instrument embeddings at 7-14 day lags. GDELT should show up at 1-3 day lags (fast-moving geopolitical events). This would confirm the perceptual layer is working.

**Implementation:** `scripts/phase41b_propagation_diagnostic.py`

---

## Concrete Fix Plan (Priority Ordered)

### Fix 1: Homoscedastic uncertainty weighting (addresses RC1)
**Math:** 
$$\mathcal{L}_{total} = \frac{1}{2e^{s_{ret}}} \mathcal{L}_{ret} + s_{ret} + \frac{1}{2e^{s_{dt}}} \mathcal{L}_{dt} + s_{dt}$$

where $s_k = \log \sigma_k^2$ is a learnable parameter.

**Code change:** In `agent/models/gnn/trainer.py`:
- Add `self.log_var_ret = nn.Parameter(torch.tensor(0.0))` to model
- Add `self.log_var_dt = nn.Parameter(torch.tensor(0.0))` to model  
- Replace: `total_loss = return_loss + dt_loss` with the uncertainty-weighted form

**Expected effect:** Model will auto-learn σ_dt ≈ large (since dt loss is 5–10x larger) and σ_ret ≈ small, giving return head a fair share of gradient.

### Fix 2: ListNet ranking loss (addresses RC2)
**Math:**
$$\mathcal{L}_{ListNet} = -\sum_{i} p^*_i \log \hat{p}_i \quad \text{where} \quad p^*_i = \frac{e^{r_i/\tau}}{\sum_j e^{r_j/\tau}}, \quad \hat{p}_i = \frac{e^{\hat{r}_i/\tau}}{\sum_j e^{\hat{r}_j/\tau}}$$

$r_i$ = normalized 21d forward return; $\hat{r}_i$ = model predicted score; $\tau$ = temperature (start at 1.0).

**Code change:** In `agent/models/gnn/trainer.py`:
- Replace `F.huber_loss(return_pred, return_target)` with `listnet_loss(return_pred, return_target)`
- Implement: 5 lines

**Expected effect:** Model directly optimizes cross-sectional rank ordering → IC should respond.

### Fix 3: GDELT down-weighting in edge reconstruction loss (addresses RC3)
**Math:**
$$\mathcal{L}_{recon} = \sum_{\text{type}} w_{\text{type}} \cdot \mathcal{L}_{\text{type}} \quad \text{where} \quad w_{\text{type}} = \frac{1}{\sqrt{N_{\text{type}}}}$$

**Code change:** In `agent/models/gnn/trainer.py`:
- Count obs_type distribution from batch
- Compute per-type weights
- Apply as edge-level loss multiplier

**Expected effect:** Model equally attends to CFTC signals (high per-edge weight) and GDELT (low per-edge weight), building a more balanced embedding space.

---

## File Impact
- `agent/models/gnn/model.py`: Add `log_var_ret`, `log_var_dt` learnable params
- `agent/models/gnn/trainer.py`: Uncertainty weighting, ListNet loss, GDELT down-weighting
- `scripts/phase40_gnn_backtest.py`: No change needed — test will run on retrained model
- `scripts/phase41b_propagation_diagnostic.py`: **New** — North Star diagnostic script
- `tirramind_kaggle_train.ipynb`: Bump TARGET_EPOCHS to 30 for retraining with fixes

---

## Testing Plan
1. Unit test: `test_listnet_loss.py` — verify ListNet loss is differentiable, produces non-zero gradients on return head
2. Unit test: `test_uncertainty_weighting.py` — verify log_var params are learned, loss scales converge
3. Integration test: train 3 epochs on synthetic data, verify return_head gradient norm > 0.01
4. System test: re-run IC diagnostic backtest, verify Mean IC > 0.03, t-stat > 2.0

---

## References
1. Kendall, Gal, Cipolla (2018). "Multi-Task Learning Using Uncertainty to Weigh Losses" arXiv:1705.07115
2. Cao, Liu, et al. (2007). "Learning to rank: from pairwise approach to listwise approach" ICML 2007
3. Burges et al. (2010). "From RankNet to LambdaRank to LambdaMART" MSR-TR-2010-82
4. Xiang, Cheng et al. (2022). "THGNN: Temporal and Heterogeneous GNN for Financial Time Series Prediction" CIKM 2022, arXiv:2305.08740
5. Qian et al. (2024). "MDGNN: Multi-Relational Dynamic GNN for Stock Investment Prediction" AAAI 2024
6. Li et al. (2021). "QTIAH-GNN: Quantity and Topology Imbalance-aware HetGNN" KDD 2023
7. Krstev et al. (2025). "A Temporal Graph Learning Framework for Lead-Lag Detection" OpenReview KsWRLyIAKP (withdrawn)

---

## Related
- [[phase40_real_data_model_refresh]] — parent task containing Phase 41 IC diagnostic results
- [[living_system_online_gnn]] — EWC training infrastructure these fixes build on
- [[het_tgn]] — model architecture being modified
- [[gnn_alignment]] — Phase 49 alignment work this extends

---

*Research date: 2026-05-01*
