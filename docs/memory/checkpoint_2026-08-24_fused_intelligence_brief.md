---
title: "Checkpoint 2026-08-24 — Fused Intelligence Brief (the output surface)"
tags:
  - doc/checkpoint
  - phase/1
  - topic/live-path
  - topic/output-surface
  - status/active
---

# Checkpoint: Fused Intelligence Brief — the missing output surface

**Date:** 2026-08-24

## Why this step

The live path was proven piece-by-piece (real anomalies, honest learning loop,
contract EV + learned P(win)) — but there was **no single fused output** a human
or API could consume and act on. That gap was the real bridge to the
product/capital step. This checkpoint closes it.

## What was built

### `scripts/intelligence_brief.py`
One command that fuses the proven pieces into a decision-actionable brief:

1. **Contract opportunities** — live USASpending awards, EV-scored
   (`EV = P(win)·(Bid−Cost)−Risk`) with the **learned P(win)** (WinProbabilityLearner),
   **long-tail (small, overlooked) contracts first** — the underserved wedge the
   big procurement-intel tools ignore.
2. **Live anomalies** — real z-score / BOCPD-changepoint signals from stored
   observations (the digest), as decision context.

Emits clean JSON + optional human-readable markdown. Deterministic math, no LLM.

### Test + regression
- `tests/test_intelligence_brief.py` — 4 tests (long-tail sort, learned P(win) applied, brief structure, markdown render)
- Made the e2e learning test **deterministic** (seeded Thompson sampling) — was flaky across test order.

## Live proof (real data)

```
# AWOS Intelligence Brief
## Contract Opportunities (long-tail first, learned P(win))
- 🟢 15DDHQ26F00000723 TAVA PRODUCTS LLC — Dept of Justice
    amount=$90,870 · EV=$40,435 · P(win)=50%
- 🟢 15B20926P00000124 NAPHCARE LLC — Dept of Justice  EV=$32,500
- 🟢 191BWC26P0064 HSI WORKPLACE... — Dept of State    EV=$31,750
- 🟢 1240BK25P0035 ROGUE VALLEY H2O — Dept of Ag       EV=$13,766
- 🟢 15DDHQ26F00000742 SAFEWARE INC — Dept of Justice  EV=$13,570
## Live Anomalies
- cftc futures_positioning mm_net z=-3.09 [changepoint] ... (+6)
```

## Verification
- `pytest` (brief + signal-store + digest + contract + reward + learning + e2e) — **61 passed**
- `ruff` — clean

## Status

- ✅ Live data fetching
- ✅ Anomaly digest (real z / changepoints)
- ✅ Honest surface→realize learning loop (compounds)
- ✅ Contract EV + learned P(win)
- ✅ **Fused output surface (`intelligence_brief.py`)** — THIS STEP

**Still open (the true next frontier):** a distribution/consumption surface — an
API endpoint, scheduled job, or subscription that delivers this brief
regularly. That is the last bridge to real usage/capital.

## Related
- [[checkpoint_2026-08-24_live_path_intelligence]]
- [[checkpoint_2026-08-24_capital_pillar_ev_scorer]]
- [[checkpoint_2026-08-24_learning_signal_diagnosis]]
- [[signals_primer]]