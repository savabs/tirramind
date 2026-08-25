---
title: "Checkpoint 2026-08-24 — Systemic Learning-Signal Diagnosis + Fix"
tags:
  - doc/checkpoint
  - phase/1
  - topic/signals
  - topic/self-learning
  - status/active
---

# Checkpoint: Systemic Learning-Signal Diagnosis + Fix

**Date:** 2026-08-24

## Why this checkpoint exists

The user raised a critical concern: "learning signal is very much not working
across the whole stuff — not just this. That's the reason I dropped the
TirraMind project." This triggered an honest, evidence-based diagnosis of
whether the self-learning actually compounds, plus a fix.

## The honest diagnosis (evidence, not opinion)

We built an empirical compounding proof (`scripts/demo_awos_learning_compounding.py`):
a hidden world where each signal operation has a true-best method tier. The
system must discover it from outcomes. We measured method-accuracy over early /
mid / late run windows.

**Result BEFORE the fix (the systemic disease):**

| | Early | Late | Verdict |
|---|---|---|---|
| Method accuracy | 0.143 | 0.112 | ~random (baseline 0.17), not learning |

**Root causes found (these are the "not working across the whole thing" you suspected):**

1. **Cold-start collapse.** The LinUCB router had no exploration. With
   `A=identity`, `b=0`, every action scored identically and `argmax` locked onto
   action 0 forever. No epsilon-greedy, no forced exploration → it never tried
   the correct-but-costlier method tiers.
2. **ReplayGate starved the learner.** `LearningCore.record_outcome` only
   updated the router when `gate.admit()` returned True — and the gate scored
   only ~37% of episodes. Worse, its scoring correlated with reward magnitude,
   which correlated with *cost* → it systematically rejected the very episodes
   (correct-but-expensive) that teach the system which method wins.
3. **Same pattern exists in the original TirraMind bandit** (`agent/learning/`):
   the `compute_reward` in `reward.py` uses an LLM-graded `eval_score` as part
   of the signal — a subjective proxy — plus cost/novelty weights that dilute
   correctness.

## The fix (principled, minimal, tested)

1. **Forced exploration warm-up** in `ml_router.py`: until each action tier has
   been tried `min_trials_forced` (default 3) times, select the least-tried
   available action. Guarantees the bandit sees every tier before trusting
   empty Bayesian priors.
2. **Router learns from every outcome.** Removed the ReplayGate as a hard
   filter on router updates. The gate stays as an observability metric
   (`admit_rate`) but never starves the learning signal.

## Result AFTER the fix (reproducible across 6 seeds)

| | Early | Mid | Late |
|---|---|---|---|
| Method accuracy | 0.72 | 0.96 | **0.99** |
| Success rate | 0.76 | — | **0.89** |

- Mean late accuracy across 6 seeds: **0.996**, always ≫ random (0.17).
- Compounding confirmed: later runs pick the right method far more often.

## What this means

- The machinery (reward store, skill library, error memory, router) is sound.
  The previous "not learning" was a **starved + underexplored signal**, not an
  unlearnable problem.
- The same two root causes (no exploration, gated/LLM-proxy rewards) explain
  why the *original* TirraMind learning layer didn't compound. That is now a
  known, fixable pattern — not a fundamental wall.

## Files changed

- `agent/awos/learning/ml_router.py` — forced-exploration warm-up; trial-count
  tracking persisted with weights.
- `agent/awos/learning/learning_core.py` — router learns from every outcome;
  gate demoted to a metric.
- `agent/learning/reward.py` — **success-anchored reward**: objective `success`
  flag now dominates (weight 0.6); LLM score demoted to a secondary fine-tune
  (0.15). Original bandit now indexes on correctness, not model opinion.
- `scripts/demo_awos_learning_compounding.py` — the empirical proof (new).

## Verification

- `.venv/bin/python scripts/demo_awos_learning_compounding.py` — COMPOUNDING CONFIRMED
- `.venv/bin/python -m pytest tests/test_awos_learning.py -q` — 17 passed
- `.venv/bin/python -m pytest tests/test_reward_fn.py -q` — 20 passed
- `StrategyBandit` (original) empirical proof: with the success-anchored reward,
  converges to the true-best arm (arm_c = 91.7% of pulls) vs. the other three — **verified learning**.
- `.venv/bin/ruff check agent/awos/learning/ agent/learning/reward.py` — clean

## Next

- ~~Wire the success-anchored reward through the full `agent/core/autonomous.py`
  run loop and re-verify end-to-end~~ ✅ **DONE** — see below.
- Apply the same exploration/filter lessons to any other gated learning path.

## End-to-end loop proof (added same session)

Added `tests/test_autonomous_learning_e2e.py`: drives the *full*
`AutonomousRunner.run()` loop with the **real** `StrategyBandit` + **real**
success-anchored `compute_reward`, replacing only the LLM components
(reflector / goal generator / evaluator / orchestrator) with deterministic
fakes over a mock world where 3 arms succeed (backtest / research / insider)
and the rest fail.

Assertion: the bandit's arm selection shifts toward the successful arms between
the first and last thirds of the run, and >50% of late selections are successful
arms. **Passes, and is stable across multiple seeds (1.4–1.9s each).**

Verification after this addition:
- `tests/test_autonomous_learning_e2e.py` — 1 passed (re-run 3×, stable)
- `tests/test_reward_fn.py` + `tests/test_awos_learning.py` + e2e — **38 passed**
- `ruff check` on all learning + test files — clean

## Related

- [[checkpoint_2026-08-24_awos_learning_embedded]]
- [[signals_primer]]
- [[cross_domain_signal_proof]]

## Related

- [[checkpoint_2026-08-24_awos_learning_embedded]]
- [[signals_primer]]
- [[cross_domain_signal_proof]]