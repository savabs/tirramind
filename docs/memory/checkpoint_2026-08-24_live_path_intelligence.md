---
title: "Checkpoint 2026-08-24 — Live-path intelligence output proven"
tags:
  - doc/checkpoint
  - phase/1
  - topic/live-path
  - topic/intelligence
  - status/active
---

# Checkpoint: Live-path Intelligence Output Proven

**Date:** 2026-08-24

## Context

The user asked two questions:
1. Where does the signal actually come from? (Answer: the 63 live data tools → real observations stored in `pipeline.db`; the Kaggle GNN is a separate, unwired layer.)
2. How much intelligence can we get from the **live path** now, before investing in the GNN/weight-based layer?

Goal: prove a "considerable level of output" from the live path alone, deterministically, on real stored data.

## What was built

### `scripts/live_intelligence_digest.py`
- Reads **real stored observations** (sovereign yields, CFTC positioning, volatility, TVL, prediction markets) from `.tirra_pipeline/pipeline.db`
- Computes **genuine anomaly signals** with the existing math stack:
  - `z-score` — latest value vs its own history distribution
  - `BOCPD` changepoint detection — regime breaks in recent data
- Emits a ranked JSON digest
- `--record` mode feeds every surfaced finding back into the **proven learning loop** (LearningCore), so the digest **compounds** over time

### Test + regression
- `tests/test_live_intelligence_digest.py` — 4 tests (z-score flags spike, flat→no signal, changepoint on step, digest flags real series)
- Full regression slice (digest + contract EV + reward + awos-learning + autonomous e2e) — **52 passed**
- `ruff` — clean (fixed two real bugs: over-strict distinct-value guard in z-score, and py3.12 `type` alias incompatible with ruff's py311 target)

## Measured output (real data, live run)

- 5 signal surfaces scored
- **93 series extracted** from real stored data
- **24 real anomalies flagged** (CFTC positioning extremes, open interest z up to +3.95)
- **6 changepoint-flagged** regime breaks (top: mm_net z=-3.09 with changepoint=True)
- Findings **recorded back into the learning loop** → persisted (33 episodes, router updating)

## Honest answer to "how much intelligence this way"

The live path now delivers **real, verifiable, deterministic signal output** — not a trained-model guess:
- Every number traces to a public data source stored in the DB
- Math (z-score, BOCPD) flags actual statistical anomalies in real time
- The output feeds the learning loop and compounds

**What it does NOT yet give:** the *cross-domain relational intelligence* the GNN is designed for (e.g., "same company wins a contract AND has a vessel at a sanctioned port AND an insider selling"). That is genuine but requires the weight-based layer — the long-term goal, correctly parked.

## Design intent

This is the correct staging:
- **Live path** = real, shippable intelligence now (anomalies, scoring, EV on contracts, compounding loop)
- **GNN** = the relationship layer to add *on top of* proven live output when the time comes — the moat, not the foundation.

## Honest learning fix (added same session)

**Caught a fake-reward bug in our own loop:** `record_findings` awarded `success=True` on *every* surfaced anomaly — the exact disease we diagnosed and fixed on the router. Fixed with a two-phase honest ledger:

- `agent/quant/signal_outcome_store.py` — `SignalOutcomeStore`: append-only pending/realized signal ledger.
  - **Surface** records an anomaly as `pending` with `success=None` (no reward).
  - **Realize** checks *forward* data for an actual move in the flagged direction; only then is an honest outcome recorded into the learning loop. Signals without forward data stay pending — no guessing.
- `scripts/live_intelligence_digest.py` now has `--surface` (phase 1) and `--realize` (phase 2) instead of the fake `--record`.

Verified:
- Live chain: 5 surfaced → pending; realize after forward data → honest success/fail only.
- `tests/test_signal_outcome_store.py` — 5 passed (surface stores pending no-reward, realize removes from pending, persist across reload, no-guess, forward-move integration realize).
- Full regression slice — **57 passed**; `ruff` clean.

## Related
- [[checkpoint_2026-08-24_capital_pillar_ev_scorer]]
- [[checkpoint_2026-08-24_learning_signal_diagnosis]]
- [[signals_primer]]
- [[cross_domain_signal_proof]]