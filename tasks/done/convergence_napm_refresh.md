---
title: "Task: convergence_napm_refresh"
tags:
  - doc/task
  - phase/7c
  - status/active
  - topic/convergence
---

# Task: convergence_napm_refresh

Status: completed
Research: [[convergence_napm_refresh]]
Spec: [[convergence_napm_refresh_spec]]

## Goal
Refresh the convergence macro backtest so it no longer depends on NAPM as a canonical free historical series.

## Steps

- [x] 1.1: Replace NAPM FRED mapping with a maintained monthly leading-indicator proxy
- [x] 1.2: Add targeted regression coverage for replacement semantics
- [x] 1.3: Run focused backtest validation and record result

## Verification

- `python -m pytest tests/test_convergence_backtest.py -q` — 57 passed
- `python -m pytest tests/test_convergence_templates.py tests/test_convergence_template_batch2.py tests/test_convergence_detector.py -q` — 336 passed

## Notes

- The backtest now uses `USSLIND` as the maintained free monthly proxy for the legacy `pmi.us.manufacturing` slot.
- The signal id was kept stable to avoid unnecessary template churn.

---

## Related

- [[convergence_napm_refresh|Research: Convergence Napm Refresh]]
- [[convergence_napm_refresh_spec|Spec: Convergence Napm Refresh]]
- [[convergence_detection]]
- [[convergence_backtest]]
- [[convergence_template_expansion]]
- [[convergence_signal_expansion]]
- [[convergence_backtest_fast_mode]]
- [[convergence_backtest_score_cache]]
- [[convergence_template_batch2]]
- [[convergence_audit_pre_worldmodel]]
