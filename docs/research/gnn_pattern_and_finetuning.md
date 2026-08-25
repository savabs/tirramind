---
title: "Research: GNN Pattern Recovery & Outcome Fine-Tuning"
tags:
  - doc/research
  - phase/14
  - phase/15
  - topic/world-model
  - topic/convergence
  - layer/world-model
  - layer/feature-engineering
---

# Research: GNN Pattern Recovery & Outcome Fine-Tuning

## Current Architecture

The Phase 12 HetTGN pipeline:
1. `GraphBuilder` converts PipelineStore → PyG HeteroData (9 entity types, 15 obs types)
2. `HetTGN` model: per-type projection → HGT conv → HeteroMemory (GRU + Time2Vec) → prediction heads
3. `Trainer` runs walk-forward self-supervised training (obs_type CE + time_delta MSE + contrastive link loss)
4. `PatternExtractor` scores meta-paths via attention hooks or embedding cosine similarity
5. `crystallize()` converts top patterns → `CrystallizedPattern` production rules
6. `AutoPatternDetector` runs crystallized rules to find co-occurrences

Phase 13 expanded the entity graph: 12 L2 tools now feed entities (company, person, domain, protocol, topic, vessel, wallet, country, organization) with 15 observation types. 352 tests passing.

## Observations — What's Broken or Weak

### 1. Attention extraction never works

`PatternExtractor._extract_attention_hooks` hooks HGTConv looking for `module._alpha_dict`. **PyG's HGTConv does not expose attention weights.** The `alpha` in `message()` is a local variable, never stored. Result: extraction always falls back to `_extract_embedding_scores` (cosine similarity), which is a much weaker signal.

Verified empirically: `inspect.getsource(HGTConv.message)` shows `alpha = softmax((q_i * k_j).sum(-1) * edge_attr / sqrt(d))` is local.

Fix: Subclass HGTConv to store attention per-edge in `message()`.

### 2. Only 1-hop meta-path scoring

Current scoring: `score = mean_attention × log1p(frequency)` for each `(src_type, edge_type, dst_type)` triplet. This misses multi-hop patterns like company → country → vessel (insider trades predict vessel movements in the headquartered country).

### 3. Naive crystallization

`crystallize()` picks the globally most common obs_type for the source and target entity types. This ignores which specific obs_types actually co-occur temporally along the edge. A pattern might connect companies to countries but the relevant signal is insider_trade → geopolitical_event, not form144_filing → geopolitical_event.

### 4. No pattern validation

Crystallized patterns have no empirical validation. A pattern might score high on attention but have zero predictive power. Need: hit rate, lift over baseline, statistical significance test.

### 5. No outcome supervision

Training is purely self-supervised (predict next event type). This is fine for pre-training but limits pattern quality. Adding even simple binary outcome labels (did target event happen after source event within window?) would let the model learn which cross-entity links carry real temporal signal.

## Risks

- Subclassing HGTConv couples us to PyG internals; version upgrades may break `message()` signature
- Multi-hop scoring grows combinatorially: limit to 2-hop, top-K pruning
- Outcome labels on synthetic data may not transfer to real data; walk-forward split is critical
- Fisher's exact test with many patterns → multiple comparison correction needed (Bonferroni or FDR)

## Data Requirements

No new data tools needed. Both phases operate on the existing entity graph in PipelineStore. Synthetic data from `SyntheticGraphGenerator` is sufficient for development and validation.

## Math/Algorithm Survey

### Attention Extraction

HGT per-edge attention (Hu et al. 2020, §3.2):

$$\alpha_{ij}^{(l)} = \text{softmax}_j\left(\frac{Q_i^{(l)} \cdot K_j^{(l)}}{\sqrt{d}} \cdot W_{\text{rel}}\right)$$

Where $Q_i = W_Q^{\tau(i)} h_i$, $K_j = W_K^{\tau(j)} h_j$, scaled by a per-relation prior $W_{\text{rel}}$. We subclass `HGTConv`, override `message()`, and store the per-edge $\alpha$ in a buffer accessible after forward pass.

### Multi-Hop Meta-Path Scoring

For a 2-hop path $A \xrightarrow{r_1} B \xrightarrow{r_2} C$:

$$\text{score}_{2\text{-hop}} = \bar{\alpha}_{r_1} \cdot \bar{\alpha}_{r_2} \cdot \log_2(1 + \text{freq}_{r_1} \cdot \text{freq}_{r_2})$$

Where $\bar{\alpha}_{r}$ is the mean attention on edge type $r$. The $\log$ damping prevents high-frequency but low-attention paths from dominating. Limit to 2-hop (3-hop is combinatorial explosion for our type count).

### Obs-Type Conditioned Crystallization

For each edge type $(s, r, t)$, build a contingency table of temporal co-occurrences:

Count all $(o_s, o_t)$ pairs where entity $e_s$ (type $s$) has observation type $o_s$ at time $\tau$ and linked entity $e_t$ (type $t$) has observation type $o_t$ at time $\tau' \in (\tau, \tau + w]$.

Pick the $(o_s, o_t)$ pair with highest count as the crystallized pattern's `obs_type_a` and `obs_type_b`.

### Pattern Validation — Hit Rate and Lift

For a crystallized pattern with source obs $o_a$ on type $s$ and target obs $o_b$ on type $t$ within window $w$:

$$\text{hit\_rate} = \frac{|\{(e_s, e_t, \tau): o_a(e_s, \tau) \wedge o_b(e_t, \tau') \text{ for some } \tau' \in (\tau, \tau+w]\}|}{|\{(e_s, \tau): o_a(e_s, \tau)\}|}$$

$$\text{baseline\_rate} = P(o_b \text{ on } e_t \text{ in random window of size } w)$$

$$\text{lift} = \frac{\text{hit\_rate}}{\text{baseline\_rate}}$$

Statistical significance via Fisher's exact test on the 2×2 contingency table (source happened/not × target happened/not). Apply Benjamini–Hochberg FDR correction for testing multiple patterns.

**Trusted source:** Fisher's exact test — R.A. Fisher, 1935; standard formulation in Agresti, *Categorical Data Analysis*, ch. 3. BH FDR correction — Benjamini & Hochberg, 1995, JRSS-B.

### Outcome-Labeled Fine-Tuning

**Binary label generation:** For each (entity_a, entity_b) pair linked by edge type $r$, create positive samples where $o_a$ on $e_a$ at time $\tau$ is followed by $o_b$ on $e_b$ within window $w$, and negative samples where no such $o_b$ occurs.

**Supervised head:** Bilinear scorer:

$$P(\text{hit}) = \sigma\left(\mathbf{h}_a^T \mathbf{W}_{\text{sup}} \mathbf{h}_b + b\right)$$

Where $\mathbf{h}_a, \mathbf{h}_b$ are node embeddings from the HGT forward pass.

**Loss:** Binary cross-entropy with class-weight balancing (positive samples are rare).

**Training protocol:**
1. Self-supervised pre-training (existing Trainer, all data up to time $T_1$)
2. Supervised fine-tuning on outcome labels ($T_1$ to $T_2$), freezing HGT conv layers, training supervised head + combiner
3. Evaluation on held-out future ($T_2$ to $T_3$)

**Metrics:** AUROC, precision@K, recall@K, F1, calibration (Brier score).

**Trusted source:** The two-phase "pre-train then fine-tune" protocol is standard in self-supervised graph learning. Reference: Hu et al. "Strategies for Pre-training Graph Neural Networks" (ICLR 2020, arXiv:1905.12265).

### GNN Diagnostic Output

After training, extract:
- **Entity-type observation density:** obs_count per entity type → identifies starved types
- **Edge-type attention distribution:** mean + variance of attention per edge type → identifies ignored edges
- **Neighborhood sparsity:** for each entity, count distinct edge types and neighbor types → identifies disconnected clusters
- **Supervised head confidence:** mean P(hit) by entity type → identifies types where the model is uncertain

This feeds directly into the GNN-guided tool expansion workflow (copilot-instructions.md Rule 6).

## Implementation Options Considered

| Approach | Pros | Cons | Decision |
|---|---|---|---|
| Subclass HGTConv | Clean, stored per-edge α | Fragile to PyG upgrades | **Selected** — only way to get real attention |
| Gradient-based importance | Version-agnostic | Expensive (backward per edge type) | Reject for now |
| Learn a separate edge-importance MLP | Decoupled from HGT | Adds parameters, might not reflect HGT behavior | Reject |
| 3-hop meta-paths | More expressive | Combinatorial (9³ = 729 paths) | Reject — 2-hop sufficient (9² = 81) |
| Mutual information validation | Strong theoretical basis | Requires binning, sample size concerns | Selected — alongside Fisher's test |
| Supervised head: MLP vs bilinear | MLP more expressive | Bilinear is simpler, fewer params | **Bilinear selected** — small graph, overfitting risk |

## Related

- [[temporal_het_gnn]] — Phase 12 research (GNN architecture)
- [[temporal_het_gnn_spec]] — Phase 12 spec
- [[l2_tool_expansion]] — Phase 13 research (entity graph expansion)
- [[gnn_pattern_and_finetuning_spec]] — Phase 14/15 spec
