"""
TirraMind — GNN Sub-package (Layer 3: World Model)

Temporal Heterogeneous Graph Network for automated cross-entity
pattern discovery.  Reads from PipelineStore, builds a PyG
HeteroData graph, trains via self-supervised next-event prediction,
and extracts discovered patterns.

Modules:
    graph_builder      — PipelineStore → PyG HeteroData conversion
    temporal           — Time2Vec + TemporalEncoder
    het_tgn            — HetTGN model (HGT + HeteroMemory + Time2Vec)
    trainer            — Self-supervised training loop
    pattern_extractor  — Attention analysis → crystallized production rules
    integration        — AutoPatternDetector + retrain_and_discover

References:
    - Research: docs/research/temporal_het_gnn.md
    - Spec: docs/specs/temporal_het_gnn_spec.md
"""
