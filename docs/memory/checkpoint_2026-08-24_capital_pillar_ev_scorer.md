---
title: "Checkpoint 2026-08-24 — Capital pillar complete: EV scoring + learned P(win)"
tags:
  - doc/checkpoint
  - phase/1
  - topic/capital
  - topic/compounding
  - status/active
---

# Checkpoint: Capital pillar complete — deterministic EV + learned P(win)

**Date:** 2026-08-24

## Summary

We closed the loop on the manifest's four pillars by giving **Capital** a real code address
inside TirraMind, and connected it to the learning loop we proved earlier.

The last pillar is no longer just framing: it now has a deterministic expected-value scorer and a
learned `P(win)` that improves as contract outcomes are recorded.

## What was built

### `agent/quant/contract_opportunity.py`
- `Opportunity` — a contract-award mapped into scoring inputs
  (`EV = P(win) * (Bid − Cost) − Risk`)
- `WinProbabilityLearner` — per (agency, amount-bucket) Beta(1,1) posterior learner.
  `p(win)` = posterior mean. Append-only persistence.
- `apply_learned_probabilities()` — re-ranks scored opportunities with learned P(win)
- `score_opportunities()` / `opportunity_to_json()` — EV ranking (Ray)

### Runnable artifact
`scripts/score_contract_opportunities.py`
- Live pipeline: `gov_contracts` (USASpending awards) → EV rank → learned P(win) → ranked JSON report
- Verified live: outputs 6 ranked awards, top = `36C25624C0002`; shows learner evidence count.

### Tests
- `tests/test_contract_opportunity.py` — **10 passed**
- Capital+learning regression slice — **48 passed**
- `ruff` — clean

## The four pillars now all have code addresses
| Pillar | Address |
|---|---|
| Mathematics | `EV = Σ P(i)·Outcome(i)` — in `expected_value` |
| Systems | `LEARNING` loop + pipeline (deterministic DAG / 353K observations) |
| Intelligence | the perceptron loop; the task is "what information should the model observe" (learn from every outcome) |
| Capital | `contract_opportunity.py` + `scripts/score_contract_opportunities.py` — option scoring + learned P/E |

## Honest status

- 🔬 **Proven**: EV scoring on live data (Valhalla Engineering / VA $191k → EV $54k etc.)
- ✅ **Learned P(win)**: 8W/2L → B(1+8, 1+2) posterior mean = 0.75, persists, re-ranks.
- ✅ **Compounding**: Bandit converged to the best arm (91.7%); the agent learning loop improves later outcomes
- (the 48-test regression slice covers the learning + capital).

This is the mileage and the actual loop now demonstrably compounds end-to-end. The remaining step to "capital" real dollars is to point this pipeline at a revenue surface and go live — which is the product decision we ended up building for, not the scaffold.

## Next
- Point the learned EV/P(win) scorer at a real output surface and wrap a bid/submission loop so the loop compounds against real money outcomes.

## Related
- [[checkpoint_2026-08-24_learning_signal_diagnosis]]
- [[signals_primer]]
- [[cross_domain_signal_proof]]