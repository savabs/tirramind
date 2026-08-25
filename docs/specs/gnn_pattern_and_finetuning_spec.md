---
title: "Spec: GNN Pattern Recovery & Outcome Fine-Tuning"
tags:
  - doc/spec
  - phase/14
  - phase/15
  - topic/world-model
  - topic/convergence
  - layer/world-model
  - layer/feature-engineering
---

# Spec: GNN Pattern Recovery & Outcome Fine-Tuning

## Goal

Phase 14: Fix pattern extraction (broken attention hooks), add multi-hop scoring, improve crystallization with obs-type conditioning, and validate patterns with statistical tests.

Phase 15: Add outcome-labeled fine-tuning — generate binary labels from temporal co-occurrences, add supervised head, implement two-phase training, and output GNN diagnostics for guided tool expansion.

## Files Affected

### Modified
- `agent/models/gnn/het_tgn.py` — replace HGTConv with AttentionCapturingHGTConv subclass, add supervised head
- `agent/models/gnn/pattern_extractor.py` — fix attention extraction, add multi-hop scoring, add obs-type conditioned crystallization, add validation
- `agent/models/gnn/trainer.py` — add fine-tuning mode, outcome label generation, diagnostic output, expand SyntheticGraphGenerator
- `agent/models/gnn/integration.py` — emit diagnostic report, update retrain_and_discover

### Created
- `tests/test_attention_capture.py` — attention-capturing HGTConv tests
- `tests/test_pattern_recovery.py` — multi-hop, validation, conditioned crystallization
- `tests/test_outcome_finetuning.py` — supervised head, fine-tuning loop, diagnostics

## Implementation Steps

### Phase 14a: Attention-Capturing HGTConv

1. **14a.1** — Create `AttentionCapturingHGTConv` in `het_tgn.py`: subclass `HGTConv`, override `message()` to store per-edge attention weights in `self._edge_attention` dict keyed by edge type.
2. **14a.2** — Update `HetTGN.__init__` to use `AttentionCapturingHGTConv` instead of `HGTConv`.
3. **14a.3** — Add `HetTGN.get_attention_weights()` method that collects `_edge_attention` from all HGT layers.
4. **14a.4** — Update `PatternExtractor._extract_attention_hooks()` to use `model.get_attention_weights()` instead of forward hooks.
5. **14a.5** — Write tests: verify attention weights are captured, correct shapes, correct edge types.

### Phase 14b: Multi-Hop Meta-Path Scoring

1. **14b.1** — Add `_score_2hop_metapaths()` to `PatternExtractor`: enumerate all valid 2-hop paths (src→mid→dst), score = mean_attn(hop1) × mean_attn(hop2) × log2(1 + freq_hop1 × freq_hop2).
2. **14b.2** — Add `MetaPathPattern.hops: int = 1` field; 2-hop patterns get `hops=2`, `edge_type` becomes "hop1_via_hop2".
3. **14b.3** — Update `extract_metapath_importance()` to include 2-hop patterns alongside 1-hop, merged and sorted.
4. **14b.4** — Write tests: verify 2-hop patterns discovered, scoring math, empty-graph handling.

### Phase 14c: Obs-Type Conditioned Crystallization

1. **14c.1** — Add `_build_cooccurrence_table()` to `pattern_extractor.py`: for each edge type, count (src_obs_type, dst_obs_type) co-occurrences within window.
2. **14c.2** — Update `crystallize()` to use co-occurrence table instead of global most-common obs_type.
3. **14c.3** — Write tests: verify co-occurrence picks correct pair, handles missing data.

### Phase 14d: Pattern Validation

1. **14d.1** — Add `validate_patterns()` function to `pattern_extractor.py`: computes hit_rate, baseline_rate, lift, and Fisher's exact p-value for each CrystallizedPattern.
2. **14d.2** — Add `ValidationResult` dataclass: hit_rate, baseline_rate, lift, p_value, significant (bool after BH correction).
3. **14d.3** — Update `crystallize()` to call `validate_patterns()` and filter by significance.
4. **14d.4** — Write tests: verify hit rate calculation, lift > 1 for injected patterns, insignificant patterns filtered.

### Phase 15a: Outcome Label Generation

1. **15a.1** — Add `OutcomeLabel` dataclass and `generate_outcome_labels()` to `trainer.py`: for each CrystallizedPattern, for each linked (src, dst) pair, create positive/negative binary labels based on temporal co-occurrence.
2. **15a.2** — Add sampling: balanced subsampling (max 1:3 pos:neg ratio) to prevent class imbalance.
3. **15a.3** — Write tests: verify label counts, walk-forward temporal correctness.

### Phase 15b: Supervised Head

1. **15b.1** — Add `SupervisedHead` module to `het_tgn.py`: bilinear scorer σ(h_a^T W h_b + b).
2. **15b.2** — Add `HetTGN.predict_outcome()` method that takes (src_emb, dst_emb) → P(hit).
3. **15b.3** — Write tests: verify output shape, gradient flow, probability range.

### Phase 15c: Fine-Tuning Loop

1. **15c.1** — Add `FineTuner` class to `trainer.py`: takes pre-trained HetTGN + outcome labels, freezes HGT conv layers, trains supervised head + combiner.
2. **15c.2** — Walk-forward split: pre-train on [0, T1], fine-tune on [T1, T2], evaluate on [T2, T3].
3. **15c.3** — Add `evaluate_supervised()`: AUROC, precision@K, recall@K, F1 on held-out data.
4. **15c.4** — Write tests: verify fine-tuning loss decreases, metrics computed, frozen layers not updated.

### Phase 15d: GNN Diagnostics

1. **15d.1** — Add `compute_diagnostics()` to `integration.py`: entity-type density, edge-type attention distribution, neighborhood sparsity, supervised-head confidence by entity type.
2. **15d.2** — Update `retrain_and_discover()` to return diagnostics alongside crystallized patterns.
3. **15d.3** — Write tests: verify diagnostic structure, non-empty results.

## Edge Cases

- HGTConv `message()` signature changes in future PyG versions: add version check, graceful fallback
- No edges in graph: 2-hop scoring correctly returns empty
- All patterns fail validation: return empty list, not error
- Outcome labels: zero positive samples: skip fine-tuning, log warning
- Class imbalance: weight BCE loss inversely proportional to class frequency
- Very small graphs (< 5 entities): diagnostics still produce valid output

## Testing Plan

- Per-phase edge case tests (14a-d tests, 15a-d tests)
- Integration: SyntheticGraphGenerator with injected patterns → full pipeline through fine-tuning → validate injected pattern recovered with lift > 1
- Regression: all 352 existing tests must pass
- Diagnostic output: verify format compatible with GNN-guided expansion workflow

## Related

- [[gnn_pattern_and_finetuning]] — Phase 14/15 research
- [[quant_training_ground]] — master roadmap task covering Phase 14/15
- [[temporal_het_gnn_spec]] — Phase 12 spec
- [[l2_tool_expansion_spec]] — Phase 13 spec
