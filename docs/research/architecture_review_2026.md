---
title: "Architecture Review 2026: Is the TirraMind Stack the Right Choice?"
tags:
  - doc/research
  - phase/41
  - topic/architecture
  - topic/gnn
  - topic/ewc
  - topic/world-model
  - topic/convergence
  - layer/world-model
  - layer/feature-engineering
  - layer/surveillance
  - layer/learning
---

# Architecture Review 2026: Is the TirraMind Stack the Right Choice?

**Date:** 2026-05-26  
**Trigger:** Pre-V16 Kaggle architectural pause — before committing another 11h training run, verify that the architecture is justified and identify the highest-value changes.  
**Scope:** HetTGN GNN, EWC continual learning, multi-task loss (obs_type + return + time_delta), IC objective, cross-domain entity linking, Kalman fusion, Bayesian world model, SAC RL, POMDP framing.

---

## Current Architecture Summary

```
Layer 1: Surveillance surface → 29-node DAG (51 tools, 32 obs_types)
Layer 2: HetTGN GNN (hidden_dim=128, 2L, 2H, ~1.96M params)
         Multi-task loss: obs_type CE + return Huber + time_delta MSE
         Auxiliary head: return_pred_head MLP 128→64→1
         Continual learning: EWC (λ=1000, Fisher diagonal, 173 param groups)
Layer 3: Bayesian world model (pgmpy DAG)
Layer 4: Signal fusion (Kalman filter, filterpy)
Layer 5: SAC RL (model-free, hidden_dim=128)
Layer 6: Adversarial (regime_gate, convergence detection)
Layer 7: LLM (support only)
```

Current training state: epoch_040, IC = −0.033 (WEAK, NOT SIGNIFICANT).

---

## 1. HetTGN Architecture

### Finding: VALIDATED — Right architecture class

**Evidence:**
- MDGNN (AAAI-24, Fan et al.) directly compares heterogeneous dynamic GNN vs static/homogeneous baselines on CSI 100/CSI 300 using IC, IR, CR, and Prec@K. MDGNN outperforms all homogeneous and static variants.
  Source: *MDGNN: Multi-Relational Dynamic Graph Neural Network for Comprehensive Events Driven Stock Price Movement Prediction*, AAAI-24, https://ojs.aaai.org/index.php/AAAI/article/view/29381/30608
- THGNN (Fan et al. 2021) specifically validates the temporal heterogeneous graph approach for financial prediction, cited directly in MDGNN comparisons.
- Cross-domain heterogeneous GNNs for international trade prediction: STGCN confirmed for multi-domain spatial+temporal signals (UPM thesis, ICAART 2025).
- MSub-GNN (PMC, 2022): Multi-source heterogeneous GNN outperforms single-source on trend prediction.

**Architecture alignment:**
- 32 observation types × multiple entity types (company, country, vessel, wallet, etc.) = heterogeneous structure is exactly what HetTGN is designed for.
- The graph evolves over time as new observations arrive = temporal structure required.
- Cross-type edges (entity linking layer) = multi-relational structure validated by MDGNN.
- 2-layer architecture: avoids deep oversmoothing (the primary GNN depth failure mode for node classification).

### Identified Risk: Over-squashing

**Evidence:** *Over-squashing in Spatiotemporal Graph Neural Networks* (NeurIPS 2025, Polimi) proves that spatiotemporal GNNs have an additional axis of information compression. Specifically:
$$\left\|\nabla_{u_i} h^{(L)}_v\right\| \leq (c\xi\theta_m)^{L \cdot L_S}(c\sigma_w)^{L \cdot L_T} \cdot S^{L \cdot L_S}_{uv} \cdot R^{L \cdot L_T}_{i0}$$

The bound decomposes into model × topology and time × space. Temporal over-squashing is counterintuitively **more sensitive to information far apart in time** than standard spatial over-squashing.

**Risk level for TirraMind:** LOW-MEDIUM. With L=2 spatial layers and 2 attention heads, we are not in the deep oversmoothing regime. But the temporal dimension may be squashing long-range signals from events months before a prediction (e.g., a vessel movement in January affecting a commodity price in March).

**Mitigation (deferred):** Graph rewiring for temporal over-squashing (NeurIPS 2025 paper outlines two approaches). Not a blocker for Phase 40 — monitor IC × temporal lag in backtest diagnostics.

### Gap: Attention head count

With 2 heads for 32 observation types × ~10 entity types, each head covers ~160 (obs_type, entity_type) pairs. This may be insufficient for the cross-domain signals (vessel × commodity, insider × equity, country × FX). Phase 49 GNN alignment diagnostic will surface this.

**Recommendation:** After Phase 40 walkforward confirms IC target (+0.03 to +0.07), run `scripts/gnn_attention_diagnostic.py` to identify starved attention heads before increasing head count.

---

## 2. EWC Continual Learning

### Finding: VALID FOR FORGETTING, INSUFFICIENT FOR REGIME CHANGES

**Evidence for validity:**
- Kirkpatrick et al. 2017 (NeurIPS): EWC reduces catastrophic forgetting via Fisher diagonal regularization. λF_i(θ_i − θ_i*)² anchors important weights.
- NeurIPS 2025 workshop: EWC reduces KG entity forgetting by 45.7% vs no regularization.

**Key distinction — two separate problems:**
1. **Catastrophic forgetting** (new task data overwrites weights for old tasks): EWC addresses this. Validated.
2. **Distributional shift / regime change** (the underlying data distribution changes, but it is NOT a new task — the same entities now behave differently): EWC does NOT address this.

Source: *Wandb: A Brief Introduction to Continual Learning*, 2024:
> "When the distribution changes, a Continual Learning model has to understand it and adapt to overcome forgetting. Such changes in the distribution are called distributional shifts or concept drift."

Source: *RETRACTED MDPI: Continual Learning for Dynamic Environments*, 2023:
> EWC "selectively protects important weights during training" — but does not adapt to new distributions, only preserves old ones.

**Financial application:**
- Task-incremental forgetting: adding a new instrument type, a new country, a new observation class → EWC handles this.
- Financial regime change (e.g., 2020 COVID crash, 2022 rate-hiking cycle): the relationships between entities change fundamentally (ship congestion now predicts inflation, not just supply chain delays) → EWC is NOT sufficient. The Fisher diagonal anchors weights from the pre-regime world, which are now WRONG.

**Supporting evidence:** Springer 2025 (Policy Weighting via Discounted Thompson Sampling for Non-Stationary Market-Making) shows that for non-stationary RL environments, hybrid approaches (EWC + replay + policy weighting) outperform EWC alone.

EVCL (arXiv:2406.15972, ICML 2024): extends EWC with variational inference — better calibrated uncertainty over weight importance, better plasticity-stability tradeoff than EWC alone.

### Current status in tirramind_structure.md (already documented as Gap #3):
> "Regime-stratified GNN replay buffer (EWC alone insufficient for non-stationary regime shifts)"

**Recommendation:**
- Phase V16 (now): EWC sidecar fix is correct and necessary. Keep λ=1000.
- Phase 50 (after Phase 40 backtest): Add regime-stratified replay buffer. Sample past training windows proportional to their regime label (normal vs. high-changepoint). This costs ~50MB RAM at 200 windows. Implement alongside Phase 40 data analysis.
- Phase 51 (optional, post-Phase 48): Consider EVCL upgrade (variational Fisher), which provides better weight importance estimation when regime uncertainty is high.

---

## 3. Multi-Task Loss: obs_type + return + time_delta

### 3a. Loss Weighting Imbalance

**Observed in v15 logs:** ep29 — obs_type loss ≈ 40,584; return loss ≈ 3,240. Ratio ≈ 12.5:1 in raw magnitude.

**Why this happens:**
- `obs_type_weight=1.0`, `return_weight=1.0` — appear equal.
- But `obs_type` CE loss sums over ~2,145 entity events per window.
- `return` loss covers only ~53 instrument observations per window.
- The ratio is structural: obs_type has ~40× more gradient contribution before any weighting.

**Current mitigation:** `_return_upscale` multiplier is applied in trainer.py (line ~1748). This was added in Phase 41 to address exactly this.

**Status:** Check `_return_upscale` value in trainer to confirm it is scaling to equalize gradient contribution. If this is working, the structural imbalance is corrected. If return loss is still 12.5× smaller than obs_type in final epoch, the upscale is insufficient.

### 3b. Return Loss: Huber vs ListNet vs LambdaRankIC

**Current default:** `use_listnet_return_loss=False` → Huber loss on 21d forward returns.

**Problem with Huber loss for IC optimization:**
- Huber minimizes |ŷ - y| — absolute prediction error.
- IC (Spearman rank correlation) measures *relative ranking*, not absolute error.
- Minimizing Huber does NOT directly maximize IC.
- A model can minimize Huber and still have IC = 0 (if it predicts all returns as 0 but with correct mean-squared error).

**Evidence for ranking loss superiority:**

1. **QuantBench (2024):** "For cross-sectional models like RGCN, IC loss yields the best IC and return metrics." (Confirmed in prior session research.)

2. **LambdaRankIC (arXiv:2605.00501, Lin et al., May 2026):**
   > "We propose LambdaRankIC, a novel learning-to-rank approach that directly optimizes Rank IC. We circumvent the non-differentiability of the ranking operator by deriving the closed-form expression for the lambda gradients induced by the pairwise rank swaps, which enables efficient gradient-based optimization within the LambdaRank framework."
   > "In empirical experiments using real market data, LambdaRankIC achieves the best out-of-sample performance on evaluation metrics commonly used in finance, including Rank IC, ICIR, monthly return, and **Sharpe ratio**."
   
   Key result: LambdaRankIC outperforms regression (Huber/MSE) AND NDCG-based ranking under low signal-to-noise and heavy-tailed noise. **This is the financial prediction regime TirraMind operates in.**

3. **ListNet (Cao et al. 2007, ICML):** Already implemented in trainer.py as `_listnet_loss()`. This is the top-1 approximation of LambdaRankIC's objective — weaker, but available now with `use_listnet_return_loss=True`.

**Recommendation: Enable ListNet NOW for V16.**
```
use_listnet_return_loss = True
listnet_temperature = 1.0
```
This is a zero-code change — the implementation exists. ListNet directly optimizes cross-sectional IC vs Huber which optimizes point error. Expected IC improvement: +0.01 to +0.03 based on QuantBench evidence.

**Longer term:** Consider implementing LambdaRankIC for Phase 50+. The key contribution is the closed-form lambda gradient for pairwise rank swaps — a 50-line addition to `_listnet_loss()`.

### 3c. time_delta Loss: NaN Every Epoch

**Observed:** All 12 metric rows show NaN for time_delta loss.

**Root cause hypothesis (not yet verified):**
- `time_delta` targets computed from `observed_at` timestamps — if two consecutive observations have the same `observed_at` float (or the bucket is computed with a log transform), delta = 0 → log(0) = -inf → NaN propagates through MSE loss.
- `time_delta_weight=0.1` — even at low weight, a NaN loss propagates to total loss via: `total = obs_type_loss + 0.1 * NaN + 1.0 * ret_loss = NaN`.
- HOWEVER: the training appears to converge (return loss 91.4 is finite), which suggests the time_delta component may be skipped or masked before entering total loss.

**Action needed:** Inspect the time_delta loss computation at line 812–1050 range in trainer.py. Add a `torch.isnan(time_delta_loss).any()` guard that zeros it out before adding to total. Do NOT propagate NaN.

**Architecture implication:** If time_delta is always NaN, the model is effectively NOT learning temporal dynamics — only entity types and return magnitudes. This limits the GNN's ability to model event timing (Hawkes-process-like spike detection). Medium-priority fix.

### 3d. auto_tune_loss_weights

**Current:** `auto_tune_loss_weights=False`.

**Option:** Kendall et al. 2018 uncertainty-weighted multi-task loss:
$$\mathcal{L} = \sum_k \frac{1}{2\sigma_k^2} \mathcal{L}_k + \log\sigma_k$$

This is already implemented (lines 879–895 in trainer.py). The Phase 40 hardening added `log_var_min/max` clamps to prevent divergence.

**Recommendation:** Enable `auto_tune_loss_weights=True` for V17+, but NOT V16. V16 should first validate that the EWC sidecar fix eliminated spikes AND that ListNet improves IC. Enabling auto-tune simultaneously makes debugging harder.

---

## 4. IC Optimization: Is IC the Right Objective?

### Finding: IC is NECESSARY but NOT SUFFICIENT for financial edge

**Key result from MDGNN (AAAI-24):**
IC measures ranking quality, IR = IC / σ(IC) measures risk-adjusted ranking quality, CR = cumulative return from a long-short portfolio. The benchmark uses all four simultaneously. IC alone does not imply positive CR.

**Key result from LambdaRankIC (arXiv:2605.00501):**
> "LambdaRankIC achieves the best out-of-sample performance on... Rank IC, ICIR, monthly return, AND Sharpe ratio."

The direct optimization of Rank IC via LambdaRankIC (or its approximation, ListNet) leads to Sharpe improvement. This is the key empirical result that closes the "IC optimization ≠ Sharpe optimization" gap.

**Current IC = -0.033:** This is NOT just suboptimal — it is actively NEGATIVE. The model is predicting inversely. Three possible causes:
1. **GDELT dominance** (92.2% of observations are geopolitical events) — embedding encodes geopolitical activity, which is inversely correlated with short-term returns (tension → uncertainty → drawdown → negative return prediction bias).
2. **Huber loss on daily returns** (known issue, documented: Phase 41) — daily AR(1) ≈ 0, so loss gradient is nearly zero. Already fixed by switching to 21d forward returns.
3. **obs_type CE loss overwhelming return gradient** — the model is optimized to classify entity observation sequences, not to rank instrument returns.

**Phase 41b Goldstein filter** (already applied): removes 92.2% → ~50% GDELT dominance. This is the right fix for (1).

**The return head with ListNet** addresses (3) directly.

**Recommendation:** After V16 (EWC fix), evaluate IC improvement. If IC does not cross +0.01 after 50 epochs with ListNet enabled and Goldstein filter active, investigate GDELT subsample further (`gdelt_subsample_frac=0.10` for v17 instead of 1.0).

---

## 5. Cross-Domain Entity Linking

### Finding: VALIDATED — critical for L3 signal depth

**Evidence:**
- MSub-GNN (PMC, 2022): Multi-source heterogeneous graph with cross-domain edges outperforms single-source on stock prediction. Each subgraph (trading data, index, news) connected via weighted inter-subgraph edges.
- MDGNN (AAAI-24): Multi-relational edges (industry, supply-chain, geographic) improve IC and IR significantly over homogeneous baselines.

**Current state:** Entity linking layer partially implemented (Phase 17 task active). 7 link types defined: `works_for`, `transacts_with`, `event_involves`, `port_call_to`, `lobbies_for`, `cert_for`, `located_in`, `patents_in`.

**Gap:** Without entity links, the GNN processes isolated node neighborhoods. The heterogeneous temporal signal exists within entity types but cannot propagate across domains (vessel → commodity company → commodity contract → futures price). This is the moat: cross-domain propagation.

**Recommendation:** Complete Phase 17 (entity linking) before Phase 40 final training. Without cross-domain edges, the GNN is underperforming relative to its architectural capability. 6 of the 7 link types map directly to L2 tools already built.

---

## 6. Kalman Fusion Layer

### Finding: VALID CHOICE, LINEAR ASSUMPTION IS A KNOWN LIMITATION

**Evidence:**
- Kalman-Enhanced DRL (IJACSA, 2025): Kalman filter + DRL for algorithmic trading. Standard KF models price dynamics as: $x_{t+1} = x_t + w_t$, $y_t = x_t + v_t$. Optimal state estimate via recursive Bayesian update.
- LLM-integrated Bayesian State Space Models (arXiv:2510.20952): Confirms the direction — Bayesian state space with multimodal fusion is state-of-art for financial time series with uncertainty.

**Key limitation:** Standard KF assumes linear state transitions and Gaussian noise. Financial time series exhibit:
- Fat tails (kurtosis > 3) → Gaussian assumption violated
- Nonlinear dynamics (momentum, mean reversion are nonlinear) → linear transition violated
- Regime switches (the transition matrix itself changes) → fixed Q, R violated

**Current phase:** Kalman is Phase 20 (not yet built beyond skeleton). This is acceptable.

**Recommendation for Phase 20 implementation:**
- Use **Unscented Kalman Filter (UKF)** or **Extended Kalman Filter (EKF)** for nonlinear dynamics, not standard KF.
- Use **Student-t noise model** instead of Gaussian for fat tails.
- Consider **Switching Kalman Filter** (aka Jump Markov Linear System) to handle regime switches. This naturally integrates with the regime_gate already in place (Phase 49b).

---

## 7. Bayesian World Model

### Finding: VALID AS BELIEF-STATE TRACKER, SCALABILITY CONSTRAINT AT >500 NODES

**Current:** pgmpy DAG. Provides conditional probability tables (CPD), belief propagation, posterior updates.

**Evidence for validity:**
- Informed POMDP / Informed Dreamer (RLC/ICLR 2024): proves that incorporating additional information at training time into the POMDP belief state model significantly improves policy convergence. The Bayesian network is the natural implementation of this informed belief state.
- Structured World Belief (SWB, ICML 2021): object-centric world model with belief state = probability distribution over hidden states. Directly maps to TirraMind's entity-centric architecture.

**pgmpy limitation:** Exact belief propagation in a DAG with 200+ nodes is O(n^2) in the worst case. At Phase 48 scale (500+ entity types), this becomes a computational bottleneck.

**Architecture decision (already documented in tirramind_structure.md, model agnosticism doctrine):**
> "World model DAG → variational inference (PyMC) at >500 nodes."

This is the correct upgrade path: PyMC + ADVI (Automatic Differentiation Variational Inference) scales to large DAGs via mini-batch VI.

**Phase 48 upgrade:** Transformer world model + Dreamer model-based RL. PO-Dreamer (ICLR 2026, Li et al.) validates this direction for POMDP scenarios with non-stationary dynamics. The memory-guided world model directly addresses the partially observable nature of TirraMind's problem.

---

## 8. SAC RL Policy

### Finding: APPROPRIATE FOR CURRENT PHASE, KNOWN MODEL-FREE LIMITATION

**Current:** SAC (Soft Actor-Critic), model-free, hidden_dim=128. Thompson Sampling bandit for goal selection.

**SAC appropriateness:** Model-free RL is correct when the world model is not yet validated. Running Dreamer (model-based RL) on a world model that has not demonstrated calibration would amplify world-model errors through imagined rollouts.

**Phase gate (already documented):** "Phase 48 target: Dreamer, gated behind Phase 40 proving the current stack has hit its ceiling AND Phase 47 density audit confirming sufficient observation history."

**Limitation of SAC for sparse financial rewards:** SAC with entropy regularization performs well on dense reward signals. Financial rewards (Sharpe attribution over 21d periods) are sparse and delayed. The reward shaping via world-model likelihood improvement (Gap #5 in tirramind_structure.md) addresses this.

---

## 9. POMDP Framing

### Finding: CORRECT AND WELL-SUPPORTED

**Evidence:**
- PO-Dreamer (ICLR 2026): directly addresses POMDP with non-stationary dynamics and memory-guided world models. Achieves SOTA on Atari 100K and SMAC multi-agent tasks.
- Belief States in POMDPs (Medium/Bowyer, 2024): confirms that maintaining a probability distribution over hidden states (belief state) is the correct framework when states are partially observable.
- Structured World Belief (ICML 2021): object-centric belief representation + RL in POMDP achieves strong performance when multiple stochastic objects interact under complex partial observability. This maps directly to TirraMind: companies, vessels, countries, contracts are interacting stochastic entities.

**Architecture alignment:**
- GNN perceives the partially observable state graph ✓
- Bayesian world model represents uncertainty over hidden states (belief state) ✓
- Kalman filter integrates noisy multi-source observations into belief-state estimate ✓
- SAC/Dreamer RL acts optimally given the belief state ✓

**Gap:** The belief state from the world model is not yet feeding into SAC as an explicit state representation. SAC currently operates on GNN embeddings, not belief-state posteriors. This is Gap #5 (tirramind_structure.md): "Reward shaping via world-model likelihood improvement."

---

## 10. Key Identified Risks (Priority-Ranked)

| Rank | Risk | Severity | Current Mitigation | Next Action |
|------|------|----------|-------------------|-------------|
| 1 | Return loss (Huber) doesn't optimize IC | HIGH | 21d forward returns (Phase 41) | Enable `use_listnet_return_loss=True` in V16 |
| 2 | EWC alone insufficient for regime changes | HIGH | regime_gate (Phase 49b) | Replay buffer Phase 50 |
| 3 | time_delta loss NaN every epoch | MEDIUM | (may be zeroed before total — unverified) | Add explicit NaN guard in trainer.py |
| 4 | obs_type CE dominates gradient | MEDIUM | `_return_upscale` multiplier | Verify upscale ratio is correct in V16 logs |
| 5 | Entity linking not complete (isolated nodes) | MEDIUM | 7 link types designed | Complete Phase 17 before Phase 40 final run |
| 6 | Over-squashing on long-range temporal signals | LOW | 2-layer architecture limits squashing | Monitor IC × temporal lag in Phase 40 backtest |
| 7 | Kalman assumes linearity + Gaussian noise | LOW | Phase 20 not yet built | Use UKF + Student-t when Phase 20 is built |
| 8 | pgmpy scales to ~500 nodes | LOW | Phase 48 is far out | Document upgrade path to PyMC |

---

## 11. Recommended Changes for V16

**Zero-code changes (training flags only):**
1. `use_listnet_return_loss = True` — enables ranking loss in trainer, directly optimizes IC instead of Huber point error
2. Confirm `gdelt_subsample_frac` is set correctly (0.05 or similar) — reduces GDELT dominance

**Code changes (pre-V16, low risk):**
3. Add NaN guard for time_delta_loss before it enters total loss (trainer.py ~line 812 region)

**Code changes (V17+):**
4. Enable `auto_tune_loss_weights=True` — Kendall et al. uncertainty weighting (already implemented)
5. Implement LambdaRankIC gradient (50 lines, upgrade from ListNet top-1 to full pairwise rank swap optimization)

**Future phases:**
6. Phase 17 completion: entity linking layer (vessel→company→contract edges)
7. Phase 50: regime-stratified replay buffer alongside EWC
8. Phase 20: UKF with Student-t noise for Kalman fusion
9. Phase 48: Dreamer + transformer world model (after Phase 40 ceiling confirmed)

---

## 12. What We Are Not Changing (and Why)

| Component | Status | Reason to Keep |
|---|---|---|
| HetTGN architecture | KEEP | Right architecture class. AAAI-24, ICAART 2025 validate. |
| EWC λ=1000 | KEEP | Appropriate for task-level forgetting. Replay buffer is additive. |
| hidden_dim=128 | KEEP | Sufficient for current entity count (~2,450 entities). Upgrade to 256 when replay buffer saturates (tirramind_structure.md doctrine). |
| 2-layer, 2-head | KEEP | Avoids deep oversmoothing. Over-squashing is bounded. |
| SAC model-free RL | KEEP | Correct for pre-Phase 40 (world model not validated). |
| pgmpy Bayesian DAG | KEEP | Scales adequately for <500 nodes. |
| POMDP framing | KEEP | Correct problem formulation. PO-Dreamer validates long-term direction. |
| 21d forward return supervision | KEEP | Daily AR(1) ≈ 0. Monthly persistence measurable. |

---

## Sources

| Finding | Source | URL |
|---|---|---|
| Heterogeneous dynamic GNN for financial prediction | MDGNN, AAAI-24 (Fan et al.) | https://ojs.aaai.org/index.php/AAAI/article/view/29381/30608 |
| IC/IR/CR/Prec@K as right metrics | MDGNN, AAAI-24 | ibid |
| STGCN for international trade prediction | UPM thesis, 2023 | https://oa.upm.es/76627/1/TFM_MARCOS_CASAS_CUADRADO.pdf |
| Oversmoothing and over-squashing survey | Shen et al., IEEE TNNLS 2024 | http://www.shendazhong.com/papers/DazhongShen-TNNLS-2024.pdf |
| Spatiotemporal over-squashing | NeurIPS 2025 (Polimi) | https://re.public.polimi.it/retrieve/0632f3c0-70e9-45a3-97e8-7f8e770401fa/2506.15507v2.pdf |
| EWC for KG forgetting | NeurIPS 2025 workshop | (from prior session research) |
| EVCL: variational EWC | ICML 2024 | https://arxiv.org/html/2406.15972v1 |
| EWC for distributional shift | W&B blog + MDPI 2023 | https://wandb.ai/shambhavicodes/ewc/reports/A-Brief-Introduction-to-Continual-Learning--VmlldzoxMTE4MTQ5 |
| Non-stationary market-making | Springer AI Review, 2025 | https://link.springer.com/article/10.1007/s10462-025-11312-9 |
| ListNet ranking loss | Cao et al. ICML 2007 | (in trainer.py comments) |
| LambdaRankIC: directly optimizing Rank IC | Lin et al. arXiv:2605.00501, May 2026 | https://arxiv.org/abs/2605.00501 |
| QuantBench IC loss validation | QuantBench 2024 | (from prior session research) |
| PO-Dreamer: POMDP world model | ICLR 2026 (withdrawn) | https://openreview.net/forum?id=QklhZ70C49 |
| Informed Dreamer | RLC/ICLR 2024 | https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_105.pdf |
| Structured World Belief | ICML 2021 (Singh et al.) | https://proceedings.mlr.press/v139/singh21a/singh21a.pdf |
| Kalman-Enhanced DRL trading | IJACSA 2025 | https://thesai.org/Downloads/Volume16No11/Paper_81-Kalman_Enhanced_Deep_Reinforcement_Learning_for_Noise_Resilient_Algorithmic_Trading.pdf |
| LLM-integrated Bayesian State Space | arXiv:2510.20952 | https://arxiv.org/html/2510.20952v1 |

---

## Related

- [[quant_training_ground]] — active task file (roadmap owner)
- [[tirramind_structure]] — canonical metrics owner
- [[living_system_online_gnn]] — EWC research doc
- [[phase40_architecture_review]] — external architecture review (2026-04-23)
- [[chat_checkpoint_2026-05-26]] — session checkpoint
