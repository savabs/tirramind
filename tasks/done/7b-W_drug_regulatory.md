---
title: "Task: 7b-W Drug Regulatory Tool"
tags:
  - doc/task
  - layer/surveillance
  - phase/7b
  - status/done
  - topic/drug-regulatory
---

# Task: 7b-W Drug Regulatory Tool

Status: completed
Research: [[7b-W_drug_regulatory]]
Spec: [[7b-W_drug_regulatory_spec]]

## Steps

- [x] 3.1: Create `agent/tools/drug_regulatory.py` with DrugRegulatoryTool class (3 modes: approvals, adverse_events, labels)
- [x] 3.2: Register tool in `agent/cli.py` (import + registry.register)
- [x] 3.3: Add GoalArm in `agent/learning/bandit.py`
- [x] 3.4: Write `tests/test_drug_regulatory_edge.py` — comprehensive edge-case suite
- [x] 3.5: Run tests, verify pass, fix any issues

---

## Related

- [[7b-W_drug_regulatory|Research: 7B-W Drug Regulatory]]
- [[7b-W_drug_regulatory_spec|Spec: 7B-W Drug Regulatory]]
- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]
