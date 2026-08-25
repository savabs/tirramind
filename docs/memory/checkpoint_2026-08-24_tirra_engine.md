---
title: "Checkpoint 2026-08-24 — TirraEngine: self-running product + full delivery"
tags:
  - doc/checkpoint
  - phase/1
  - topic/delivery
  - topic/product
  - status/active
---

# Checkpoint: TirraEngine — self-running intelligence product

**Date:** 2026-08-24

## Summary

Built all four "next possibilities" into one self-running engine. The pipeline is
now a real, consumable, deployable product — not a set of manual scripts.

## What was built

### A. Self-running entrypoint
`scripts/tirra_engine.py` — one command for the whole pipeline:
- `--once --collect` — fast refresh + build + deliver
- `--serve` / `--all` — deliver + serve over HTTP
- `--record-bid AGENCY AMOUNT WON` — feedback loop (see C)
- `--email` — email the brief (env-configured)
- Registered console scripts in `pyproject.toml`: `tirra-engine`, `tirra-serve`, `tirra-brief` (via `agent/app/__init__.py` wrappers)

### B. Live signals (freshness)
`refresh_fast_data()` — refreshes the fast, reliable digest-relevant tools (CFTC + gov_contracts) in **~2s in parallel**, instead of the slow/hanging full daily DAG. Verified: cftc ok + gov_contracts ok each cycle. The brief reads the latest available report data (CFTC is weekly-cadence by nature).

### C. P(win) feedback (personalization)
`--record-bid` records realized bid outcomes into the `WinProbabilityLearner`. Verified:
- VA win → P(win) 0.33 → 0.60 (learned)
- DoJ 2 losses → P(win) 0.14 (learned)
- Untouched agencies stay at the prior (0.33, basis=prior)
The brief now shows learned vs prior per agency — real personalization from evidence.

### D. Packaging
- `pyproject.toml` console scripts (3 new, work from any directory)
- `Dockerfile` — slim python:3.12, serves the brief on :8787
- `scripts/run_scheduled.sh` — cron/systemd-friendly scheduled runner
- Email delivery via stdlib `smtplib` (graceful no-op when SMTP unconfigured)

## Verification

- Console scripts registered + work from `/tmp` ✅
- `tirra-engine --once --collect` → delivers in ~seconds ✅
- `record_bid` personalizes P(win) ✅
- Scheduled runner runs one cycle ✅
- Server serves latest brief + status ✅
- **75 tests passed consistently ×3 runs** (fixed a real flakiness: the reward-weight BO in `autonomous.py` was non-hermetic — seeded it in the e2e test)
- ruff clean (Python)

## The full product pipeline (complete)

```
Live data (63 tools)
  → fast refresh (CFTC + gov, ~2s)
  → real anomalies (z/BOCPD) + contract EV + learned P(win)
  → fused Intelligence Brief
  → deliver (.tirra_delivery/)  → serve (HTTP)  → email (optional)
  → record-bid feedback → P(win) personalizes → repeat
```

One command: `tirra-engine --all` (or `scripts/run_scheduled.sh serve` on a schedule).

## Related
- [[checkpoint_2026-08-24_delivery_layer]]
- [[checkpoint_2026-08-24_fused_intelligence_brief]]
- [[checkpoint_2026-08-24_live_path_intelligence]]
- [[checkpoint_2026-08-24_capital_pillar_ev_scorer]]
- [[checkpoint_2026-08-24_learning_signal_diagnosis]]