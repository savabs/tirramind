---
title: "Task: 7b-Z — Central Bank Balance Sheets"
tags:
  - doc/task
  - layer/surveillance
  - phase/7b
  - status/done
  - topic/central-bank
---

# Task: 7b-Z — Central Bank Balance Sheets

Status: completed
Research: [[7b-Z_central_bank_balance_sheets]]
Spec: [[7b-Z_central_bank_balance_spec]]

## Implementation
- `agent/tools/central_bank_balance.py` (~650 lines)
- 4 modes: balance_sheets, liquidity_index, policy_divergence, rate_monitor
- 7 CBs: Fed, ECB, BOJ, BOE, SNB, BOC, RBA
- Data sources: FRED API (with key), ECB SDW API (free, no auth)
- FX normalization: all CBs converted to USD
- Net liquidity: Gross CB assets - Fed RRP - Fed TGA

## Registration
- Tool #36: `central_bank_balance`
- Bandit arm #24: `global_liquidity`

## Tests
- `tests/test_central_bank_balance_edge.py` — 84 edge case tests
- Count assertions updated in 12 test files (35→36 tools, 23→24 arms)
- Full suite: 2278 passed, 0 failed, 6 skipped

---

## Related

- [[7b-Z_central_bank_balance_spec|Spec: 7B-Z Central Bank Balance]]
- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]
