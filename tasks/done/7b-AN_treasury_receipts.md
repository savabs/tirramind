---
title: "Task: 7b-AN Treasury Receipts Tool"
tags:
  - doc/task
  - layer/surveillance
  - phase/7b
  - status/done
  - topic/treasury
---

# Task: 7b-AN Treasury Receipts Tool

Status: completed
Research: [[7b-AN_treasury_receipts]]
Spec: [[7b-AN_treasury_receipts_spec]]

## Steps

- [x] 3.1: Create `agent/tools/treasury_receipts.py` with TreasuryReceiptsTool class (3 modes: cash_balance, deposits_withdrawals, public_debt)
- [x] 3.2: Register tool in `agent/cli.py` (import + registry.register)
- [x] 3.3: Add GoalArm in `agent/learning/bandit.py`
- [x] 3.4: Write `tests/test_treasury_receipts_edge.py` — comprehensive edge-case suite
- [x] 3.5: Run tests, verify pass, fix any issues

---

## Related

- [[7b-AN_treasury_receipts|Research: 7B-An Treasury Receipts]]
- [[7b-AN_treasury_receipts_spec|Spec: 7B-An Treasury Receipts]]
- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]
