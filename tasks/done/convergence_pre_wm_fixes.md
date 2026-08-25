---
title: "Task: Convergence Pre-World-Model Fixes"
tags:
  - doc/task
  - status/done
  - topic/convergence
---

# Task: Convergence Pre-World-Model Fixes

Status: completed
Research: [[convergence_audit_pre_worldmodel]]
Spec: [[convergence_audit_pre_worldmodel]] (Section 3)

## Steps

- [x] 1.1: Write extractor for `consumer_sentiment` tool
- [x] 1.2: Write extractor for `food_security` tool
- [x] 1.3: Write extractor for `political_risk` tool
- [x] 2.1: Propagate coincidence direction through DetectionResult to ConvergenceSignal
- [x] 3.1: Pass persistence_count from detector.persistence_history into from_detection_result in DAG callback
- [x] 4.1: Write edge-case tests for all 3 fixes (73 tests)
- [x] 4.2: Run full convergence test suite (956/956 pass)

---

## Related

- [[convergence_detection]]
- [[convergence_backtest]]
- [[convergence_template_expansion]]
- [[convergence_signal_expansion]]
- [[convergence_napm_refresh]]
- [[convergence_backtest_fast_mode]]
- [[convergence_backtest_score_cache]]
- [[convergence_template_batch2]]
- [[convergence_audit_pre_worldmodel]]
