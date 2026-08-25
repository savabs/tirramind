---
title: "Task: 7b-AA Global PMI Tool"
tags:
  - doc/task
  - layer/surveillance
  - phase/7b
  - status/done
  - topic/global-pmi
---

# Task: 7b-AA Global PMI Tool

Status: completed
Research: [[7b-AA_global_pmi]]
Spec: [[7b-AA_global_pmi_spec]]

## Steps

- [x] 3.1: Create `agent/tools/global_pmi.py` with GlobalPmiTool class (3 modes: cli, bci, cci)
- [x] 3.2: Register tool in `agent/cli.py` (import + registry.register)
- [x] 3.3: Add GoalArm in `agent/learning/bandit.py`
- [x] 3.4: Write `tests/test_global_pmi_edge.py` — comprehensive edge-case suite
- [x] 3.5: Run tests, verify pass, fix any issues

---

## Related

- [[7b-AA_global_pmi|Research: 7B-Aa Global Pmi]]
- [[7b-AA_global_pmi_spec|Spec: 7B-Aa Global Pmi]]
- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]
