---
title: "Task: GNN-Guided Tool Expansion"
tags:
  - doc/task
  - status/done
  - phase/16
  - topic/surveillance
  - topic/world-model
  - layer/surveillance
  - layer/world-model
---

# Task: GNN-Guided Tool Expansion

Status: completed
Research: [[gnn_guided_tool_expansion]]
Spec: [[gnn_guided_tool_expansion_spec]]

---

## Steps

### Phase 16a: Synthetic Diagnostic Validation
- [x] 16a.1: Add format_diagnostic_report() to integration.py
- [x] 16a.2: Write end-to-end synthetic diagnostic test
- [x] 16a.3: Capture synthetic diagnostic snapshot to docs/reports/

### Phase 16b: Real-Data Diagnostic Extraction
- [x] 16b.1: Add run_diagnostics() CLI-callable entry point
- [x] 16b.2: Run diagnostics on real PipelineStore + capture report

### Phase 16c: Gap Analysis & Ranking
- [x] 16c.1: Map flagged gaps to candidate tools
- [x] 16c.2: Score candidates on 5-dimension rubric
- [x] 16c.3: Produce Tier 1/2/3 ranking artifact
- [x] 16c.4: Update l2_tool_expansion task with Phase 16 results

## Related

- [[gnn_guided_tool_expansion]] — Phase 16 research
- [[gnn_guided_tool_expansion_spec]] — Phase 16 spec
- [[gnn_pattern_and_finetuning]] — Phase 14/15 (diagnostic outputs)
- [[l2_tool_expansion]] — Phase 13 (candidate tool catalog)
- [[temporal_het_gnn]] — Phase 12 (GNN architecture)
- [[tool_priority_ranking]] — Phase 16 ranking artifact
- [[gnn_diagnostics_synthetic]] — synthetic diagnostic baseline
- [[gnn_diagnostics_real]] — real store diagnostic
