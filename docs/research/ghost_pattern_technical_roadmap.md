---
title: Ghost Pattern Detection Technical Roadmap
tags:
  - doc/research
  - phase/phase1
  - topic/ghost-patterns
  - topic/anomaly-detection
  - layer/all-layers
  - status/active
---

> Primary document: [[ghost_pattern_technical_roadmap.html]]

## Summary

Deep technical research plan for implementing cross-domain anomaly detection ("ghost patterns") for MP-1 Atlantic energy supply chain. Based on 2024-2026 arXiv papers, production GitHub repos, and pure mathematics foundations.

## Key Technical Foundations

- **OWLEYE** (ICLR 2026) — Zero-shot cross-domain graph anomaly detection
- **CGSTA** (Feb 2026) — Multi-scale hierarchical graph contrast
- **Tigramite** — PCMCI+ causal discovery for time series
- **CrossHGL** (Mar 2026) — Heterogeneous GNN with meta-path fusion
- **TemporalRI** — Temporal subgraph isomorphism for chain pattern matching

## Implementation Phases

1. **Phase 0 (Weeks 1-2):** Data pipeline foundations (COT, AIS, prices)
2. **Phase 1 (Weeks 3-4):** Single-domain anomaly detection baselines
3. **Phase 2 (Weeks 5-6):** Cross-domain feature fusion (SVD alignment)
4. **Phase 3 (Weeks 7-10):** Temporal knowledge graph + heterogeneous GNN
5. **Phase 4 (Weeks 11-12):** Chain template engine (pattern matching)
6. **Phase 5 (Weeks 13-14):** LLM narrative generation + MVP launch

## Target Performance (MVP)

- **Precision:** >60% (6 of 10 alerts correct)
- **Lead Time:** 3-7 days median before event
- **False Positive Rate:** <10% (≤1 false alert per 10 weeks)

## Related

- [[ghost_pattern_income_plan]] — Business model for monetizing ghost pattern alerts
- [[ghost_pattern_income_task]] — Active task for MP-1 launch
- [[ghost_pattern_graph_audit]] — Data requirements for detection
- [[long_term_vision]] — How ghost patterns fit into TirraMind's 5-10 year vision
- [[phase41b_gnn_signal_extraction]] — Current GNN training (different objective)
