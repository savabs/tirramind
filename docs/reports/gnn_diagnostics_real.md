---
title: "GNN Diagnostics: Real PipelineStore"
tags:
  - doc/research
  - phase/16
  - topic/surveillance
  - topic/world-model
  - layer/world-model
---

# GNN Diagnostics: Real PipelineStore

**Generated**: Phase 16b.2 — 2026-04-10
**DB Path**: `.tirra_pipeline/pipeline.db`
**DB Size**: 56 KB (schema only)
**Entry point**: `run_diagnostics()` from `agent/models/gnn/integration.py`

---

## Result

```
status: empty_graph
entity_count: 0
obs_count: 0
```

The real PipelineStore exists on disk but contains **zero entities and zero observations**. The 56 KB file size is schema and WAL metadata only — no L2 tools have been executed against a live API yet.

`run_diagnostics()` returned `status="empty_graph"` and exited cleanly without attempting training, as designed.

## Implications for Phase 16c

Because the real store is empty, Phase 16c (Gap Analysis & Ranking) cannot use real diagnostic evidence for prioritization. The ranking will be grounded in:

1. **Synthetic diagnostic baseline** — from [[gnn_diagnostics_synthetic]], which established:
   - `country`, `vessel`, `wallet` entity types were sparse (< 5 entities)
   - `port_call_to` and `exchange_based_in` edge types received zero attention
   - Supervised confidence stream was empty (no outcome labels)
   - Observation density was healthy across all types (synthetic generator produces uniformly)

2. **Architectural analysis** — from the candidate tool catalog in [[l2_tool_expansion]] and the ranking rubric in [[gnn_guided_tool_expansion]]:
   - Which entity types have the fewest observation channels (1 tool vs. 3+ tools)?
   - Which candidate tools add cross-domain links vs. same-domain density?
   - Which tools have the lowest implementation effort (L2 upgrade of existing tool)?

3. **Signal Depth Doctrine** — prefer tools that resolve to entity-level actors and enable cross-domain linking.

## When To Re-Run

This diagnostic should be re-run after any of:
- A batch of L2 tools has been executed against live APIs (populating entities + observations)
- A development cycle that adds new entity types to the graph
- Any time the tool priority ranking needs to be data-driven rather than architecture-driven

The command:
```python
from agent.models.gnn.integration import run_diagnostics
result = run_diagnostics(".tirra_pipeline/pipeline.db")
```

## Related

- [[gnn_diagnostics_synthetic]] — synthetic baseline (Phase 16a)
- [[gnn_guided_tool_expansion]] — Phase 16 research
- [[gnn_guided_tool_expansion_spec]] — Phase 16 spec
- [[l2_tool_expansion]] — candidate tool catalog
