---
title: "Spec: GNN-Guided Tool Expansion"
tags:
  - doc/spec
  - phase/16
  - topic/surveillance
  - topic/world-model
  - layer/surveillance
  - layer/world-model
---

# Spec: GNN-Guided Tool Expansion

## Goal

Run the Phase 15d diagnostic pipeline on both synthetic and real entity graphs, interpret the outputs, and produce a ranked priority list of tool upgrades/additions. No new tool code is written in this phase — only the diagnostic analysis and prioritization artifact.

## Files Affected

### Modified
- `agent/models/gnn/integration.py` — add `format_diagnostic_report()` helper to render diagnostics as structured markdown/dict
- `tests/test_gnn_integration.py` — add diagnostic report formatting tests

### Created
- `docs/reports/gnn_diagnostics_synthetic.md` — synthetic run diagnostic snapshot
- `docs/reports/gnn_diagnostics_real.md` — real PipelineStore diagnostic snapshot
- `docs/reports/tool_priority_ranking.md` — final Tier 1/2/3 ranking artifact

## Implementation Steps

### Phase 16a: Synthetic Diagnostic Validation

1. **16a.1** — Add `format_diagnostic_report(diagnostics: dict) -> dict` to `integration.py`: takes raw `compute_diagnostics()` output and returns a structured summary with flags for each threshold violation (entity density < 5, obs density < 10, mean attention < 0.05, mean degree < 1.0, supervised confidence near 0.5 ± 0.1). Pure function, no side effects.

2. **16a.2** — Write a test script (can be a pytest parametrize or standalone) that runs the full pipeline on SyntheticGraphGenerator data: `Trainer.train()` → `PatternExtractor` → `crystallize()` → `compute_diagnostics()` → `format_diagnostic_report()`. Assert the report has all five sections, flags are booleans, values are finite floats. This validates the workflow end-to-end on synthetic data.

3. **16a.3** — Capture the synthetic diagnostic output to `docs/reports/gnn_diagnostics_synthetic.md`. Include the raw numbers, which thresholds are violated, and a brief interpretation noting that synthetic data has uniform entity distribution so real data will differ.

### Phase 16b: Real-Data Diagnostic Extraction

4. **16b.1** — Write a CLI-callable script or function `run_diagnostics(db_path: str) -> dict` that loads a PipelineStore from disk, trains a model, extracts patterns, crystallizes, and runs `compute_diagnostics()` + `format_diagnostic_report()`. If no DB exists or graph is empty, exit with a clear message. This is the reusable entry point for periodic diagnostic runs.

5. **16b.2** — Run the diagnostic pipeline on the real PipelineStore (if it has data). Capture output to `docs/reports/gnn_diagnostics_real.md`. If the real store is empty or insufficient, document that fact and note that ranking is based on synthetic + architectural analysis only.

### Phase 16c: Gap Analysis & Ranking

6. **16c.1** — For each entity type flagged as sparse or low-confidence, list all candidate tools (from [[l2_tool_expansion]] Future Phases section) that could add entities, observations, or links involving that type. Map gaps to candidates in a structured table.

7. **16c.2** — Score each candidate tool on the 5-dimension rubric from the research doc (connectivity gain, signal uniqueness, implementation effort, data quality risk, overlap penalty). Produce a weighted score.

8. **16c.3** — Rank candidates into Tier 1 (top 3), Tier 2 (next 5), Tier 3 (rest). Write the ranking artifact to `docs/reports/tool_priority_ranking.md` with the score breakdown and rationale for each tier placement.

9. **16c.4** — Update [[l2_tool_expansion]] task file with a "Phase 16 Results" section that records the diagnostic-driven priority order and links to the ranking artifact.

## Edge Cases

- **Empty or tiny PipelineStore**: `run_diagnostics` exits cleanly. Ranking falls back to architectural analysis + synthetic results.
- **All entity types dense**: No tool upgrades needed — output a "graph is healthy" report.
- **All entity types sparse**: Prioritize by implementation effort (cheapest first to get broad coverage quickly).
- **Tied scores**: Break ties by implementation effort (lower effort wins).
- **No crystallized patterns**: Skip supervised_confidence stream; rank using the other four streams only.

## Testing Plan

- `format_diagnostic_report()` unit tests: correct flags, empty input handling, extreme values
- End-to-end synthetic pipeline test: validates the full workflow without manual intervention
- No tests for ranking itself (it's a one-time analytical artifact, not runtime code)

## Related

- [[gnn_guided_tool_expansion]] — Phase 16 research
- [[gnn_guided_tool_expansion]] — Phase 16 task
- [[gnn_pattern_and_finetuning_spec]] — Phase 14/15 spec (diagnostic API defined here)
- [[l2_tool_expansion]] — Phase 13 (candidate tool catalog)
- [[l2_tool_expansion_spec]] — Phase 13 spec
