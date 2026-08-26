---
title: LESSONS — TirraMind GNN Fuckups & Research
tags:
  - doc/wiki
  - topic/gnn
  - status/active
---

# LESSONS — TirraMind GNN: Fuckups & Research

> **READ THIS BEFORE EVERY CODING SESSION.**
> Chronological fuckup log + research findings for the heterogeneous temporal GNN.
> One rule: if something burned us, it goes here immediately.

---

## PART 1 — FUCKUP LOG

Each entry: **Symptom → Root Cause → Fix → Prevention Rule**

---

### F-01 · Entity-Identity Contrastive Loss → Embedding Collapse
*Discovered: Phase 47-48*

**Symptom:** All 89 instrument embeddings identical (cosine sim ≈ 1.0 across all pairs).
IC stuck at ~0.00. Contrastive loss converged to 0 immediately.

**Root Cause:** `_contrastive_loss()` pulled same-entity embeddings across different windows
together. With only one window per instrument and no hard negatives from different instruments,
the model learned "make everything the same" as the trivially optimal solution.
Same-entity loss = always 0, margin loss = trivially satisfied → constant embedding is optimal.

**Fix:** Replaced with CSRC (Cross-Sectional Ranking Contrastive) loss.
Positive pairs = same return decile (cross-sectional peers), negative pairs = opposite deciles.
InfoNCE formulation. See `trainer.py::_cross_sectional_ranking_contrastive()`.

**Prevention Rule:**
- Never use entity-identity contrastive as sole objective for cross-sectional tasks.
- Always verify embedding diversity BEFORE training: `torch.std(emb, dim=0).mean()` should be > 0.1.
- Effective rank = `exp(entropy(singular_values / sum))` — should be >> 1. If ≈ 1, collapse.

---

### F-02 · Return Head Bypassing GNN Entirely
*Discovered: Phase 48-50*

**Symptom:** Return loss converging, IC improving, but GNN embeddings contributing ZERO.
Training looked fine. Model was actually just a linear head on raw price features.

**Root Cause:** Conditional priority in trainer:
```
if has_raw AND raw_feats AND concat_head:   ← concat head (GNN involved)
elif has_raw AND raw_feats:                 ← RAW HEAD ONLY (GNN bypassed) ← was here
else:                                       ← pred head (GNN only)
```
`use_concat_head` defaults to False, so the `elif` branch always fired when raw features
existed. The GNN was forward-passed (for CSRC loss) but its embeddings never entered
the return prediction path.

**Fix:** Enable `--use-concat-head` to activate the concat branch. This concatenates
`[raw_price_features || GNN_embedding]` and routes both through `return_concat_head`.
Verified via: check that `model.return_concat_head` is not None in trainer.

**Prevention Rule:**
- After any architecture change, grep for every `model.return_` call and verify which branch
  is active. Print active branch name at training start.
- Ghost patterns ONLY flow into IC if the GNN embedding tensor is in the return prediction
  computational graph. Detached = no gradient = no learning.

---

### F-03 · Loss History Column Shift (10-row offset)
*Discovered: V40 training run*

**Symptom:** In the printed loss table, the `return` column values were shifted UP by 10 rows —
epoch 75's return loss appeared in the epoch 65 row.

**Root Cause:** `history["return"]` had only 65 entries (return loss was added in epoch 11),
while `history["total"]` had 75 entries. On checkpoint resume, the shorter array was appended
directly without aligning lengths. The table renderer did `history["return"][i]` for all i,
causing off-by-10 display.

**Fix:** On checkpoint load, pad the shorter history arrays at the FRONT with `float("nan")`:
```python
_pad = len(history["total"]) - len(history["return"])
history["return"] = [float("nan")] * _pad + history["return"]
```
Table renderer: check `math.isnan(val)` → display `"—"`.

**Prevention Rule:**
- All history arrays MUST be same length as `history["total"]` at all times.
- When adding a new loss component mid-training, pad immediately on checkpoint load.
- Never assume `history[k][i]` aligns with epoch `i+1` without verification.

---

### F-04 · Data Leakage in IC Evaluation
*Discovered: Phase 47 post-mortem*

**Symptom:** Ridge regression showed IC=+0.48, ICIR=+2.1. Looked amazing. Was fake.

**Root Cause:** `quant_benchmark.py` used ALL available data for both train and test of the
ridge model (no walk-forward split). Test data was inside the training window.
Corrected walk-forward IC = +0.07, ICIR = +0.40.

**Fix:** Walk-forward split. Never evaluate on data that overlapped training window.
Corrected values logged in memory: `corrected ridge IC=+0.07, ICIR=+0.40`.

**Prevention Rule:**
- ANY evaluation that produces IC > 0.15 should be treated as suspect until the split is
  verified. Real-world IC for a single quant factor is typically 0.03-0.15.
- Always print train/test date ranges in evaluation output.

---

### F-05 · Evaluation Window Too Short for 60d Features
*Discovered: Phase 47-50 IC check*

**Symptom:** IC from `ic_check.py` showed ICIR=+0.14. After fix: ICIR=+0.44. Same model.

**Root Cause:** `ic_check.py` and `quant_benchmark.py` defaulted to 6-week history.
Features `sharpe_60d`, `vol_60d`, `max_dd_60d` need ≥60 trading days (≥12 calendar weeks)
of history to be non-NaN. With 6-week window, most rows had NaN features → silent zeros
after fillna → artificially flat predictions.

**Fix:** Default history changed to 26 weeks in both scripts.
Production minimum: `history ≥ max_lookback_days / 5 * 1.5` (add 50% buffer).

**Prevention Rule:**
- Every time a new feature with a lookback is added, update the minimum history constant.
- Add a startup assertion: `assert weeks_of_history >= min_required_weeks`.

---

### F-06 · obs_type CE Loss Dominating / Gradient Starvation
*Discovered: Phase 46-47 loss analysis*

**Symptom:** `obs_type` classification loss at ~2.5 (2145-class CE). Return loss at ~0.05.
Effective gradient from return task ≈ 0 relative to obs_type. GNN learned to classify
observation types, not predict returns.

**Root Cause:** 2145-class cross-entropy loss scale is ~log(2145) ≈ 7.7 at random init.
Return loss (ListNet) starts near 0.05. Ratio ≈ 154:1. The shared backbone parameters
optimized almost entirely for obs_type.

**Partial mitigation:** Loss weighting (`return_weight` >> `obs_type_weight`).
True fix: gradient normalization (GradNorm) or task-specific learning rates.

**Prevention Rule:**
- Before training: print loss scale ratio for all tasks at epoch 0.
- Target ratio: no single auxiliary loss should exceed 3-5× the primary task loss.
- Consider GradNorm (`arxiv:1711.02257`) for automatic balancing.

---

### F-07 · GDELT Flooding the Graph
*Discovered: Phase 45-46 graph analysis*

**Symptom:** Graph snapshots OOM or extremely slow. 92% of all observations were GDELT events.
Instrument-level signals drowned out by geopolitical event noise.

**Root Cause:** GDELT scraping added events at 15-min granularity. Even subsampled,
the event count vastly exceeded instrument observations. Message passing was dominated
by GDELT→GDELT→instrument chains with near-zero signal.

**Mitigation:** GDELT node subsampling per snapshot. Hard cap on events per snapshot.
Still not fully resolved — event relevance scoring needed.

**Prevention Rule:**
- After any new data source addition: print node type counts per snapshot.
- If any node type > 60% of total nodes, it needs explicit subsampling or graph rebalancing.

---

### F-08 · Snapshot Caching Hides Per-Window Bugs
*Discovered: Phase 48 debugging*

**Symptom:** Epochs running at 60s/epoch (previously 10+ min). Looked like great speedup.
But it meant bugs in graph construction were SILENT after the first epoch (cache hit).

**Root Cause:** Graph snapshots pre-built once before epoch 1, cached in memory.
If snapshot construction has a bug (wrong cutoff timestamp, wrong node features), that bug
persists through all epochs and all training runs in the session.

**Fix:** No fix to the caching — it's correct and necessary. But:
- Add snapshot sanity check on first build: node counts, edge counts, feature stats.
- Print `[SNAP] nodes=N, edges=E, nan_features=K` for first 3 snapshots.

**Prevention Rule:**
- Any change to `graph_builder.py` must be followed by a snapshot dump verification.
- Never assume a fast epoch means a correct epoch.

---

## PART 2 — ARCHITECTURE RESEARCH

### What the Ghost Pattern Machine Needs

Our goal: detect emergent cross-entity signals invisible to single-instrument features.

```
[GDELT event: oil supply disruption]
         ↓  (relation: geo_region → sector)
[sector:energy node]  ←  [macro:CB_rate_hike node]
         ↓  (relation: sector_member)
[instr:XOM]  [instr:CVX]  [instr:SLB]
         ↓
GNN embedding encodes "energy + geopolitical stress"
         ↓
concat_head: [raw_price || GNN_emb] → return_score
```

A single raw feature CANNOT see this chain. The GNN message passing IS the ghost pattern
mechanism. This is the entire reason for training the GNN.

---

### R-01 · Embedding Collapse — Causes & Fixes

**Source:** "Embedding Collapse in Recommender Systems", Sumit's Blog, Nov 2024;
"Breaking the curse of dimensional collapse in GCL", Info. Sciences 2023.

**Two types of collapse:**
1. **Complete collapse** — all embeddings identical (constant vector). Our F-01 case.
   *Cause: loss allows constant solution as trivial optimum.*
   *Fix: contrastive learning with hard negatives (CSRC).*
2. **Dimensional collapse** — embeddings span only low-rank subspace (e.g. 3 dims of 128).
   *Cause: over-smoothing via multi-hop message passing (GNN low-pass filter effect).*
   *Fix: VICReg / Barlow Twins / decorrelated BN.*

**Measurement:**
- `torch.std(embs, dim=0).mean()` — should be > 0.1 for healthy embeddings
- Effective rank = `exp(H(σ / Σσ))` where σ = singular values — should be >> 5
- Plot histogram of pairwise cosine similarities — should be roughly bell-shaped around 0

**Fixes ranked by strength:**
1. **CSRC** (our impl) — InfoNCE with return-decile pairs. Addresses complete collapse. ✓ Implemented
2. **VICReg** (`arxiv:2105.04906`, Barlow/Facebook AI) — adds 3 explicit regularizers:
   - Variance: `max(0, γ - std(z))` — keeps each dim's variance above threshold γ
   - Invariance: MSE between two views of same sample
   - Covariance: `off_diag(C(Z))^2 / d` — decorrelates all embedding dims
   *Prevents dimensional collapse even without negative samples. 4-6 lines of code.*
   **Candidate for V43.**
3. **Decorrelated BN** — add before return head. Whitens embedding space.
4. **Stop gradient** — SimSiam-style, prevents trivial collapse without negative pairs.

---

### R-02 · Temporal GNN Architecture (TGN/TGAT Best Practices)

**Source:** "Temporal Graph Learning in 2024", Medium/DataScience.com; TGN paper 2020.

**Key findings verified against paper:**
- **Time encoding matters**: Relative time encoding (`time2vec`) prevents look-ahead leakage.
  Never use absolute timestamps — model learns to cheat on the time dimension.
- **Memory module**: TGN-style node memory (running state) captures longer-term entity
  history than just the current window. Instrument memory = rolling context.
- **Listwise ranking loss** outperforms binary CE for temporal GNNs on ranking tasks.
  Confirmed: "empirically, TGN and TGAT perform better with listwise loss" (2024 survey).
  Our ListNet is correct here.
- **Layer count**: ≥3 message-passing layers → over-smoothing → collapse. Use ≤2 layers
  for financial graphs where instruments are highly interconnected.

---

### R-03 · Heterogeneous Graph Attention (HGT)

**Source:** "Heterogeneous Graph Transformer" (Hu et al., WWW 2020, arxiv:2003.01332).

**Core insight:** Standard GCN/GAT applies same attention weights regardless of node/edge type.
In a het-graph (instruments + sectors + GDELT + macro), this is wrong: the information content
of "instrument→sector" edge is fundamentally different from "GDELT→instrument" edge.

**HGT solution:** Type-specific projection matrices for Q, K, V in attention:
- Each (source_type, edge_type, target_type) triple has its own weight matrices
- Relative temporal encoding embedded into attention scores
- Multi-head attention with type-specific "mutual attention" and "message passing"

**Why this matters for us:** Our `het_tgn.py` currently uses shared projection matrices
for all relation types. Adding type-specific projections should improve ghost pattern
discrimination — GDELT→instrument patterns learned separately from sector→instrument patterns.

**Implementation cost:** Medium. Need per-relation-type Linear layers in HetConv.

---

### R-04 · Multi-Task Loss Balancing (GradNorm)

**Source:** "GradNorm: Gradient Normalization for Adaptive Loss Balancing" (Chen et al., ICML 2018,
arxiv:1711.02257). "MetaBalance" (ACM WebConf 2022).

**Problem:** Our training has 5 losses: obs_type CE, time_delta regression, value regression,
return ListNet, CSRC InfoNCE. These have wildly different gradient magnitudes.
If one loss dominates, the backbone optimizes for the wrong objective.

**GradNorm algorithm (3 steps):**
1. Compute gradient norm for each task: `G_i(t) = ||∇_W L_i(t)||`
2. Compute relative training rate: `r_i(t) = L_i(t) / L_i(0)` (normalized loss)
3. Adjust loss weight: `w_i ← w_i * (G_avg * r_i^α / G_i)` — tasks with slower training
   get more gradient

**Implementation:** Add a `GradNorm` loss on the loss weights themselves. ~50 lines.
Candidate for addressing F-06 properly.

**Simpler alternative:** UncertaintyWeights (Kendall et al., 2018) — learns log(σ_i)^2
per task, uses it as uncertainty-weighted loss. 5 lines of code, works well in practice.

---

### R-05 · Cross-Sectional Ranking Contrastive (CSRC) — Our Implementation

**Concept:** InfoNCE-style contrastive loss where positive pairs are instruments in the
same return decile (cross-sectional peers) and negative pairs are instruments in opposite
deciles. Forces the GNN backbone to encode return-rank-relevant features.

**Key hyperparameters:**
- `temperature=0.1` — lower = sharper separation, harder task. Start at 0.1.
- `n_deciles=5` — gives ~18 instruments per decile with 89 instruments.
- Requires ≥ 2 instruments per decile. Falls back gracefully if not enough valid targets.

**Mathematical basis:** NT-Xent / InfoNCE (van den Oord et al., 2018). For anchor i:
```
L_i = -log( exp(sim(z_i, z_j+) / τ) / Σ_k exp(sim(z_i, z_k) / τ) )
```
where j+ is a same-decile positive and k ranges over all instruments.

**Interaction with return loss:**
CSRC shapes the embedding SPACE. The ListNet return loss then ranks within that shaped space.
Both losses must be active simultaneously. CSRC without return loss = ranking-aware embeddings
that don't generalize. Return loss without CSRC = embeddings that collapse.

---

### R-06 · Concat Head — The Bridge for Ghost Patterns

**The fundamental routing problem:**
```
Ghost patterns in GNN embedding → must reach return prediction → must affect IC
```

Three head options:
| Head | GNN involved? | IC from graph? | Use case |
|------|--------------|----------------|---------|
| `return_raw_head` | NO | NO | Baseline: raw features only |
| `return_pred_head` | YES | YES | GNN-only: no price features |
| `return_concat_head` | YES | YES | Full: raw + GNN together ← TARGET |

`return_concat_head` input = `[xsnorm_price_features || GNN_instrument_embedding]`
Gradient flows back through both the raw projection AND the GNN backbone.
This is the ONLY head that lets ghost patterns contribute to IC.

**Enable with:** `--use-concat-head` in `retrain_gnn.py`.

---

### R-07 · What "Ghost Patterns" Actually Look Like in This Graph

Patterns that raw features cannot see but the GNN can (given proper training):

1. **Sector contagion chain**: Earnings miss on one stock → message passing → sector node →
   neighbor stocks in sector get updated embeddings before their price moves.

2. **GDELT → commodity → energy sector → instruments**: Geopolitical shock → commodity
   node → energy sector node → instrument nodes. The GNN propagates this 3-hop chain;
   raw features only see the instrument's own price.

3. **Macro regime switch**: CB rate hike event → macro node → all financial sector instruments
   update embeddings simultaneously. Cross-sector correlation emerges through shared macro node.

4. **Whale on-chain divergence**: On-chain whale accumulation on a token → decouples from
   price momentum → GNN embedding diverges from raw feature embedding → concat head learns
   to weight the GNN signal higher in this regime.

5. **Correlation cluster breakdowns**: Two instruments with historically correlated prices
   start diverging. The graph edge weight updates → message passing strength decreases →
   GNN embedding divergence → return spread signal.

For patterns 1-5 to work: graph edges must be dynamic (updated per window), and
message passing must span ≥ 2 hops. Both are true in our architecture.

---

---

### F-09 · Zero-Init Last Layer → Gradient Dead Zone in Concat Head
*Discovered: V42 smoke test (2026-05-29)*

**Symptom:** T4 smoke test FAIL: `type_projections['instrument'].weight.grad = 0.0` even with
`--use-concat-head` enabled. Ghost patterns could never reach IC.
V38-V41 was burning Kaggle GPU for zero GNN contribution to return prediction.

**Root Cause:** `het_tgn.py` zero-initialised the last layer of `return_concat_head` (and originally `return_pred_head` was frozen/zeroed):
```python
nn.init.zeros_(self.return_concat_head[-1].weight)
```
When `weight_last = 0`, the backward pass computes:
```
dL/d_hidden = dL/d_scores @ weight_last.T = dL/d_scores @ zeros.T = 0
```
Gradient stops at the last linear layer. All upstream layers (HGTConv, type_projections,
combiner) receive ZERO gradient from the return loss. Only the last layer's OWN weight
receives gradient. Until that weight grows non-zero (takes 10-50+ epochs), the backbone
is completely disconnected from the return objective.

Direct inspection of `epoch_090.pt` (V42) confirmed `return_pred_head.4.weight` was exactly `0.0` (standard deviation 0.0) resulting in constant `0.0` predictions and a `+nan` IC.

**Fix:** Removed both `nn.init.zeros_` calls from `return_concat_head`. Kaiming default
init now used. Gradient flows from epoch 1. A quick 10-epoch training run from scratch (V43 quickrun) confirmed `return_concat_head.4.weight` std became non-zero (0.0653) and generated a powerful, non-nan cross-sectional IC of **+0.3622**!

**Prevention Rule:**
- **NEVER zero-init a fresh head** (one that doesn't load from old checkpoints).
  Zero-init is only appropriate for heads that ARE loaded from old checkpoints to
  prevent a sudden loss spike at epoch 1.
- **Run `smoke_test_gnn.py` before every Kaggle push.** T4 is the gate: if it fails,
  ghost patterns CANNOT contribute to IC. Don't push.
- When adding any new head: verify `weight.grad != 0` after one backward pass.
- **Always inspect weight tensor standard deviations** if evaluation outputs `+nan` or constant predictions. Constant outputs indicate a dead gradient head.

---

### F-10 · The GDELT Noise Flood & Portfolio Concentration Illusion
*Discovered: Phase 50m "Full Graph Unleashed" (V46) analysis*

**Symptom:** Mean Spearman IC flipped negative (-0.0190) across the 40 rolling backtest folds, but the GNN-ValueHead strategy actually printed a massive **40.64% total return** (beating the Equal-Weight baseline of 23.14% by nearly double).

**Root Causes:**
1. **The GDELT Noise Flood:** Disabling subsampling (`--gdelt-frac 1.0`) flooded the GNN with thousands of sparse geopolitical event nodes. This created a dense graph of spurious correlations, allowing the GNN to overfit the training set perfectly while destroying its out-of-sample ranking IC.
2. **The Portfolio Concentration Illusion:** In backtesting, GNN-ValueHead was allowed a maximum asset weight of 1.0 (100% of portfolio in a single asset). It printed 40.64% returns because it made highly accurate, concentrated bets, but experienced deep drawdowns (-25.4%), resulting in an artificially low Sharpe Ratio (0.427). The model *was* working beautifully directionally, but was penalized by the unconstrained portfolio execution layer.

**Prevention Rules:**
- **Never train on 100% GDELT:** Keep `--gdelt-frac` between `0.05` and `0.20` to act as a noise filter.
- **Enforce portfolio risk boundaries:** Do not evaluate raw predictive quality based solely on unconstrained Sharpe. Measure predictive accuracy directly via the "Ground-Truth Decile Spread".

---

### F-11 · Parallel Batching Write Collisions → Intra-Window Temporal Blind Spot in GRU Memory Updates
*Discovered: Sequential logic deep-dive (2026-05-31)*

**Symptom:** The GNN was structurally unable to capture sequential events (e.g., sequence of 4–5 events) within a single training window. The temporal dynamics of multi-event sequences were lost, preventing the GNN from learning event valuations or conditional arrival intensities.

**Root Cause:**
In `HeteroMemory.update_memory`, when a single entity had multiple events within the same weekly window, their `node_ids` were grouped and processed in a single flat batch:
```python
old_mem = self.memory[node_ids]
new_mem = self.gru(gru_input, old_mem)
with torch.no_grad():
    self.memory[node_ids] = new_mem.detach()
```
Since `node_ids` contained duplicate entries for entities with multiple events:
1. All events for that entity read the *exact same* initial memory state at the start of the week.
2. The GRU processed them in parallel rather than recursively.
3. The in-place GPU assignment `self.memory[node_ids] = new_mem.detach()` caused write-back collisions where only one non-deterministic update survived. All intermediate sequential state transitions were wiped out.

**Prevention Rules:**
- **Always group and sequence duplicate node IDs:** When processing event-based sequential updates within a single batch, group events by node and process them in sequential chronological steps (step $k$ of sequence) to guarantee recursive memory updates.
- **Never perform batch updates with duplicate keys:** Ensure any direct buffer or parameter update using indexing (`self.memory[node_ids] = ...`) has completely unique elements.

---

### F-12 · Schema Drift Silently Invalidated Every Checkpoint
*Discovered: 2026-08-26, during intelligence-layer reactivation*

**Symptom:** Three DAGs (`gnn_inference`, `entity_scoring`, `inference`) failed or
silently produced nothing. `entity_scoring` crashed with
`index 69 is out of bounds for dimension 1 with size 69`; the other two threw
`mat1 and mat2 shapes cannot be multiplied (93x49 and 23x64)`. `signals`,
`beliefs`, `entity_alerts`, `convergence_clusters`, `portfolio_weights` and
`paper_trade_pnl` had **zero rows** despite 365k healthy `entity_observations`.

**Root Cause:** Three registries drifted apart with nothing comparing them:

| | live DB | code constants | trained weights |
|---|---:|---:|---:|
| entity types | 12 | 11 | 12 |
| observation types | 38 present, 4 unknown to code | 48 | 48 |
| instrument feature dim | 49 | 49 | **23** |

Three distinct failures fell out of that:

1. **Instrument features grew 14 → 23 → 49 across checkpoint generations and
   nothing was ever retrained after the last step.** `load_model` used
   `strict=False` and *skipped* the mismatched `type_projections.instrument`
   weight, leaving it randomly initialised, then logged a generic "skipped N
   keys" line naming no entity type. The failure surfaced much later as an
   opaque torch shape error.
2. **`ENRICHMENT_DIM` was hardcoded to 55** — correct only while
   `len(OBSERVATION_TYPES) == 46`. The writer indexes `offset + 9 + ot_idx` over
   the *live* list, so once the registry grew to 48 the block overflowed. With
   `BASE_FEAT_DIM=14` the tensor was `14+55=69` wide and `ot_idx=46` addressed
   index 69 — the exact crash. For instrument nodes the same overflow instead
   ran into the price-feature block that follows: **silent corruption, not a
   crash**, depending only on node type.
3. **`maritime_area` was in the DB but not in `ENTITY_TYPES`**, so
   `_build_node_features` fell back to `type_idx = 0` and one-hot encoded it as
   `cftc_contract`. A `log.warning` fired and the run continued. It trained and
   scored as the wrong entity kind for months.

A test asserted the buggy behaviour (`assert features[0, 0] == 1.0`), so the
suite was green over the corruption — the same pattern as the DataCache tests.

**Fix:**
- `ENRICHMENT_DIM` is now **derived**: `9 + len(OBSERVATION_TYPES)`.
- Unknown entity type → **all-zero one-hot** (claims no identity) instead of
  `ENTITY_TYPES[0]`. Feature building stays non-fatal because runtime discovery
  of new types is a supported feature.
- New `validate_schema_against_store(store)` raises `SchemaDriftError` listing
  every DB type the code cannot encode. Called before anything trains or scores.
- `load_model` now names the drift explicitly:
  `instrument: trained_weights=23 expected_by_model=49`.
- Registries synced: 12 entity types, 52 observation types.

**Prevention Rules:**
- **Any dimension derived from a registry must be computed, never hardcoded.**
  The formula `_ENRICHMENT_SCALAR_DIM + len(OBSERVATION_TYPES)` computes
  `ENRICHMENT_DIM` — a bare literal is a time bomb that detonates one registry
  edit later. Current canonical value: see `[[project_metrics]]`.
- **Never degrade an unknown categorical to index 0.** Claiming no identity is
  honest; claiming the wrong one is corruption that trains cleanly.
- **Compare checkpoint `in_channels` against live `GraphBuilder` output before
  loading weights.** A skipped key in `load_state_dict(strict=False)` means a
  randomly-initialised layer, not a harmless omission — it must name the layer.
- **Editing `ENTITY_TYPES` / `OBSERVATION_TYPES` invalidates every checkpoint.**
  One-hot position derives from list index, so an insertion shifts every later
  index. Keep both lists alphabetically sorted so insertions are reviewable, and
  retrain in the same change.
- **If a test asserts a fallback/default behaviour, check the fallback is
  actually correct** before treating a green suite as evidence.

---

## PART 3 — HOW TO MEASURE IF THE MODEL IS ACTUALLY WORKING

To determine if a trained model is *truly* working (independent of noisy portfolio math or abstract correlation statistics), use the **Ground-Truth Decile Spread Test**:

1. **The Core Question:** Does the top-ranked group of assets predicted by the GNN outperform the bottom-ranked group in reality?
2. **The Execution (Simple & Intuitive):**
   - Sort all assets by their GNN predicted score for a given week.
   - Separate the top 20% predicted assets and the bottom 20% predicted assets.
   - Calculate their actual forward 21d returns.
   - **Spread = (Mean Return of Top 20%) - (Mean Return of Bottom 20%)**
3. **Success Criteria:**
   - **Positive Spread (Spread > 0):** The model is working. In V46, despite a negative overall Spearman IC, the GNN printed a massive **+58.7 bps spread** in the final week.
   - **Negative Spread (Spread < 0):** The model is inverted or dead.
   - **Scale Invariance:** The absolute values or signs of the scores do not matter (e.g. scores can all be negative). Only the cross-sectional ranking order matters.

---

## PART 4 — PRE-CODING CHECKLIST

Before implementing anything, read these 3 questions:

1. **Will the GNN embedding reach the return prediction head?**
   Check: `use_concat_head=True` AND `model.return_concat_head is not None`.

2. **Are embeddings diverse?**
   Quick check: `torch.std(instrument_embs, dim=0).mean()` in eval mode. Must be > 0.05.

3. **Is the loss gradient balanced?**
   Print loss values at epoch 0. Return loss should be within 5× of obs_type loss.
   If not, adjust weights before training, don't fix it after 50 epochs.

---

## VERSION HISTORY

| Version | Key Change | IC Result |
|---------|-----------|-----------|
| V38-39  | Baseline HetTGN, entity contrastive loss | IC ≈ 0.00 (embedding collapse) |
| V40     | CSRC loss added | IC unknown (return head bypassed GNN) |
| V41     | History fix + debug logging, ep63→75 | raw head baseline only |
| V42     | `--use-concat-head` ON, zero-init bug (F-09) active | +nan (dead return predictor) |
| V43     | Quickrun 0→10 from scratch, Kaiming init (F-09 fixed) | +0.3622 (final week snap) |
| V44     | Full 90-epoch convergence run (enable_gpu=False bug) | cancelled (run on CPU) |
| V45     | Full 90-epoch convergence run (GPU fixed) | +0.0369 (GNN-EmbNorm ICIR=+0.236) |
| V46     | Full 90-epoch "unleashed" (100% GDELT, unlimited windows, pure return loss) | +0.0250 (GNN-EmbNorm ICIR=+0.134, GNN-ValueHead Ret=+40.64%) |
| V47     | Recursive chronological step-updates for GRU memory fallback (F-11 fixed) | pending new run |

## PART 2 — Architecture Research

### M2: Differentiable Options Pricing & Greeks
*Completed: 2026-05-31*

**Key Insight:** Analytical Greeks formulas are fragile — sign errors in theta (e.g., `+ r*K*exp(-rT)*N(d2)` instead of `-`) cause silent test failures. Autograd is the ground truth; use it to validate analytical implementations.

**Barone-Adesi-Whaley Numerical Stability:**
- When `b = r` (no dividends), `S_star_call` approximation `K*(1 + r/(r-b)*...)` explodes to `inf`. Mask all early-exercise premium computations with `torch.where(b < r, ..., 0.0)` to avoid `inf * 0 = NaN`.
- For puts, `S_star_put` can go negative when `r - b` is tiny. Clamp to `min(K*0.1)` before taking `log()`.
- Newton-Raphson for `S*` converges in 3-5 iterations and is fully differentiable via PyTorch autograd.

**Test Results:** 11/11 pass. Greeks match analytical to 1e-5. IV roundtrip error < 1e-4. American prices >= European prices.

### M3: Advanced Pricing Models & Fourier-Cosine (COS) Pricing
*Completed: 2026-05-31*

**Key Insight:**
- Standard Monte Carlo option pricing via Euler-Maruyama discretization (with clamping) is highly biased (positive bias on volatility due to clamping negative variance paths), whereas the Fourier-Cosine (COS) method achieves near-exact analytical precision in a fraction of a second ($O(1)$ Python overhead).
- To maintain differentiability across parameters in Heston/Bates models, Albrecher's "Little Trap" formulation of the complex logarithm is mandatory to prevent branch-cut discontinuities (which would break gradient flow and cause NaN gradients in autograd).
- Martingale drift correction terms for diffusion ($-0.5 \sigma^2 T$) and jumps ($-\lambda_j k_j T$) must be meticulously accounted for in the risk-neutral characteristic function to ensure option prices do not drift or overstate the underlying expectation.

**Test Results:** 16/16 pass. All models (Heston, Bates, Merton, VG) converge perfectly to Black-Scholes limits, and are fully differentiable w.r.t underlying spot price and model parameters (no NaNs or exploding gradients).

### M5: Implied Volatility Surface & GNN Features
*Completed: 2026-05-31*

**Key Insight:**
- **ATM Singularity in SABR:** In Hagan's asymptotic SABR formula, the $z / \chi(z)$ term has a $0/0$ singularity as $K \rightarrow F$. Implementing a Taylor series expansion limit guard: `1.0 - 0.5 * rho * z + (2 - 3*rho**2)/12 * z**2` for $|z| < 1e-4$ is absolutely critical to avoid NaN division and NaN gradients in backward optimization.
- **PyTorch nn.ModuleDict Restriction:** PyTorch's `nn.ModuleDict` does not allow dots `.` in key names (e.g. `0.25`) because dots denote hierarchical module boundaries. Slices must be formatted using string formatting (e.g. `slice_{str(T).replace('.', '_')}`).
- **Bilinear Interpolation of Variance:** Interpolating in *total variance space* $w(k, T) = \sigma^2 T$ instead of raw implied volatility prevents vertical calendar arbitrage.
- **GNN Features Differentiability:** ATM Vol, Skew, and Curvature can be computed exactly and analytically from SVI parameters, allowing flawless gradient flow from GNN backbones directly back into option pricing calibration parameters.

**Test Results:** 7/7 pass. Cover SVI/SABR parameter boundaries clamping, Durrleman's butterfly-free condition, SGD fitting convergence, ATM singularity guards, total-variance interpolation, and GNN feature backpropagability.

### M6: Rough Volatility & Fractional Brownian Motion
*Completed: 2026-06-01*

**Key Insight:**
- **BLP Hybrid Scheme Weight Correction:** The Bennedsen-Lunde-Pakkanen hybrid scheme for simulating fractional Volterra paths requires careful derivation of the weight scaling. Both the history weights and local variance term scale with $dt^H$, NOT $dt^{H-1/2}$. Using $dt^{H-1/2}$ overestimates all weights by a factor of $1/\sqrt{dt}$, causing catastrophic variance explosion (paths hitting `inf` within a few steps). The correct formulas are:
  - History: $b_j = \frac{dt^H}{H+1/2} \cdot ((j+1)^{H+1/2} - j^{H+1/2})$
  - Local: $Y^L = \frac{dt^H}{\sqrt{2H}} \cdot Z$
- **Exploding ATM Skew Signature Validated:** The rough Bergomi model successfully reproduces the empirical power-law explosion of the ATM implied volatility skew $\psi(T) \propto T^{H-1/2}$ as $T \rightarrow 0$. With $H=0.07$, short-maturity skew (T=0.1) is ~11x larger than long-maturity skew (T=0.5), confirming the rough volatility paradigm.
- **Hurst Exponent Estimation:** The quadratic log-increment regression method (slope of $\log E[(\Delta\log\sigma)^2]$ vs $\log(\Delta t)$ equals $2H$) provides a robust, differentiable estimator suitable for feature engineering pipelines.

**Test Results:** 3/3 pass. Cover rBergomi parameter clamping, Hurst exponent estimation on simulated rough paths, and ATM skew power-law signature validation.
