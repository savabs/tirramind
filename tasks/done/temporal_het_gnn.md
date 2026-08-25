---
title: "Task: Temporal Heterogeneous GNN"
tags:
  - doc/task
  - status/done
  - phase/12
  - topic/world-model
  - topic/surveillance
  - layer/world-model
  - layer/feature-engineering
---

# Task: Temporal Heterogeneous GNN

Status: completed
Research: [[temporal_het_gnn]]
Spec: [[temporal_het_gnn_spec]]

---

## Steps

### Phase 12a: Graph Construction Layer
- [x] 12a.1: Add query_all_entities/observations/links to PipelineStore
- [x] 12a.2: Install PyTorch + PyG CPU dependencies
- [x] 12a.3: Implement graph_builder.py (PipelineStore → HeteroData)
- [x] 12a.4: Edge case tests for graph builder (48/48 passing)

### Phase 12b: Temporal Feature Encoding
- [x] 12b.1: Implement Time2Vec nn.Module
- [x] 12b.2: Implement TemporalEncoder (per-entity history → features)
- [x] 12b.3: Edge case tests for temporal encoding (29/29 passing)

### Phase 12c: Model Architecture (HetTGN)
- [x] 12c.1: Implement HeteroMemory (type-aware TGN memory replacement)
- [x] 12c.2: Implement HetTGN model (HGT + HeteroMemory + Time2Vec)
- [x] 12c.3: Edge case tests for model (35/35 passing)

### Phase 12d: Self-Supervised Training
- [x] 12d.1: Implement SyntheticGraphGenerator
- [x] 12d.2: Implement training loop (next-event prediction + contrastive)
- [x] 12d.3: Implement walk-forward evaluation
- [x] 12d.4: Edge case tests for training (27/27 passing)

### Phase 12e: Pattern Extraction
- [x] 12e.1: Implement attention extraction (meta-path importance)
- [x] 12e.2: Implement temporal lag extraction
- [x] 12e.3: Implement crystallization (attention → production rules)
- [x] 12e.4: Edge case tests for pattern extraction (19/19 passing)

### Phase 12f: Integration + Evaluation
- [x] 12f.1: Integrate auto patterns alongside hand-crafted (AutoPatternDetector)
- [x] 12f.2: Comparative evaluation (compare_patterns)
- [x] 12f.3: Periodic retraining hook (retrain_and_discover) — 14/14 tests passing

---

## Notes

- **Critical constraint:** PyG's TGNMemory is homogeneous. Must build custom HeteroMemory. See research doc.
- **Dependencies:** torch (CPU-only), torch-geometric. Both BSD-3/MIT licensed.
- **New files go in:** `agent/models/gnn/` sub-package
- **Self-supervised only** — no market outcome labels yet. Fine-tuning deferred to future phase.
- **Synthetic data first** — real entity count is small during development.

## Related

- [[temporal_het_gnn]]
- [[temporal_het_gnn_spec]]
- [[cross_entity_l3]]
- [[world_model]]
