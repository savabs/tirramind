---
title: "Task: 7b-T Sovereign Debt / Government Bond Markets"
tags:
  - doc/task
  - layer/surveillance
  - phase/7b
  - status/done
  - topic/sovereign-debt
---

# Task: 7b-T Sovereign Debt / Government Bond Markets

Status: completed
Research: [[7b-T_sovereign_debt]]
Spec: [[7b-T_sovereign_debt_spec]]

## Summary

Tool #35 (sovereign_debt), Bandit arm #23 (sovereign_stress).
5 modes: us_yields, eu_yields, jp_yields, uk_gilts, spreads.
4 free API sources: US Treasury XML, ECB IRS CSV, Japan MOF CSV, UK DMO XML.

## Steps

- [x] Research — probe APIs, document endpoints and formats
- [x] Spec — 10 implementation steps, edge cases, testing plan
- [x] Implementation — 726 lines, SovereignDebtTool class
- [x] Cache API fix — removed invalid `ttl=` kwarg from 4 `put()` calls
- [x] Registration — cli.py import + bandit.py sovereign_stress arm
- [x] Edge case tests — 94 tests in test_sovereign_debt_edge.py, all passing
- [x] Count assertion updates — 19 stale assertions (34→35, 22→23) across 11 test files
- [x] Full suite green — 2194 passed, 0 failed, 6 skipped

---

## Related

- [[7b-T_sovereign_debt|Research: 7B-T Sovereign Debt]]
- [[7b-T_sovereign_debt_spec|Spec: 7B-T Sovereign Debt]]
- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]
