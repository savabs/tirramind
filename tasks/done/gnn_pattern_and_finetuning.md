---
title: "Task: GNN Pattern Recovery & Outcome Fine-Tuning"
tags:
  - doc/task
  - status/done
  - phase/14
  - phase/15
  - topic/world-model
  - topic/convergence
  - layer/world-model
---

# Task: GNN Pattern Recovery & Outcome Fine-Tuning

Status: completed
Research: [[gnn_pattern_and_finetuning]]
Spec: [[gnn_pattern_and_finetuning_spec]]

---

## Steps

### Phase 14a: Attention-Capturing HGTConv
- [x] 14a.1: Subclass HGTConv → AttentionCapturingHGTConv (store per-edge α)
- [x] 14a.2: Update HetTGN to use AttentionCapturingHGTConv
- [x] 14a.3: Add HetTGN.get_attention_weights()
- [x] 14a.4: Update PatternExtractor to use real attention
- [x] 14a.5: Write attention capture tests

### Phase 14b: Multi-Hop Meta-Path Scoring
- [x] 14b.1: Add _score_2hop_metapaths() to PatternExtractor
- [x] 14b.2: Extend MetaPathPattern with hops field
- [x] 14b.3: Merge 1-hop + 2-hop in extract_metapath_importance()
- [x] 14b.4: Write multi-hop tests

### Phase 14c: Obs-Type Conditioned Crystallization
- [x] 14c.1: Add _build_cooccurrence_table()
- [x] 14c.2: Update crystallize() to use co-occurrence
- [x] 14c.3: Write crystallization tests

### Phase 14d: Pattern Validation
- [x] 14d.1: Add validate_patterns() with hit_rate, lift, Fisher's test
- [x] 14d.2: Add ValidationResult dataclass
- [x] 14d.3: Update crystallize() to filter by significance
- [x] 14d.4: Write validation tests

### Phase 15a: Outcome Label Generation
- [x] 15a.1: Add OutcomeLabel dataclass + generate_outcome_labels()
- [x] 15a.2: Add balanced subsampling
- [x] 15a.3: Write label generation tests

### Phase 15b: Supervised Head
- [x] 15b.1: Add SupervisedHead bilinear module
- [x] 15b.2: Add HetTGN.predict_outcome()
- [x] 15b.3: Write supervised head tests

### Phase 15c: Fine-Tuning Loop
- [x] 15c.1: Add FineTuner class
- [x] 15c.2: Walk-forward split (pre-train / fine-tune / evaluate)
- [x] 15c.3: Add evaluate_supervised() with AUROC, precision, recall, F1
- [x] 15c.4: Write fine-tuning tests

### Phase 15d: GNN Diagnostics
- [x] 15d.1: Add compute_diagnostics()
- [x] 15d.2: Update retrain_and_discover() to return diagnostics
- [x] 15d.3: Write diagnostic tests

## Related

- [[gnn_pattern_and_finetuning]] — Phase 14/15 research
- [[gnn_pattern_and_finetuning_spec]] — Phase 14/15 spec
- [[temporal_het_gnn]] — Phase 12 (GNN architecture)
- [[l2_tool_expansion]] — Phase 13 (entity graph expansion)
