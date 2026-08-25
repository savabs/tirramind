---
title: "Checkpoint 2026-08-24 — Delivery layer: scheduled, persisted brief output"
tags:
  - doc/checkpoint
  - phase/1
  - topic/delivery
  - topic/output-surface
  - status/active
---

# Checkpoint: Intelligence Brief Delivery Layer

**Date:** 2026-08-24

## Why this step

The fused brief existed but had **no delivery**: nothing produced it repeatedly,
persisted it, or served it to a consumer. This is the bridge from "pipeline
produces intelligence" to "a human/API can receive Intelligence repeatedly" —
the last frontier before real usage/capital.

## What was built

### `agent/delivery/brief_deliverer.py` (pure I/O layer, no layer inversion)
- `BriefDeliverer(out_dir, stem, render_md)` — persists a brief as
  `intelligence_brief.json` + `intelligence_brief.md` into a delivery directory
- `deliver()` builds a `DeliveryRecord` (delivered_at, paths, n_contracts,
  n_anomalies, duration_ms, checksum) and appends to `delivery_log.jsonl`
- `latest()` / `records()` / `status()` — persistent, crash-readable query layer
- Render function injected (no scripts import inside agent/) — clean layering

### `scripts/deliver_brief.py` (CLI + scheduler)
- `--once` — build + deliver + exit
- `--status` — show last delivery + total count
- `--interval-min N` — APScheduler background loop (deliver on schedule)
- `--contracts/--anomalies/--max-contract-rows/--learner/--db/--out` — knobs

### Tests
- `tests/test_brief_delivery.py` — 5 tests (writes files, log append + latest,
  record roundtrip across instances, status shape, no-renderer still writes JSON)

## Verification
- CLI `--once` → wrote JSON + MD + delivery_log entry in `.tirra_delivery/` ✅
- `--status` → returns latest record + total ✅
- APScheduler interval → job fired and delivered (record persisted) ✅
- Full regression (delivery + brief + signal-store + contract + digest + reward +
  learning + e2e) — **66 passed** ✅
- `ruff` — clean

## The full pipeline (now complete)

```
Live data (63 tools)
  → real anomalies (z-score / BOCPD)
  → honest surface→realize learning loop (compounds)
  → contract EV + learned P(win)
  → fused Intelligence Brief (contracts + anomalies)
  → DELIVERY: scheduled persist (JSON + Markdown + log)  ⬅ THIS STEP
```

## Next (natural, not started)
- ~A consumer surface: expose `.tirra_delivery/intelligence_brief.json` via a tiny endpoint — so an external system can fetch "today's brief" programmatically.~ ✅ **DONE (this session)**

## Consumer surface (added same session)

### `agent/brief_server.py` — minimal HTTP server (stdlib, no new deps)
- `GET /brief` & `/brief.json` → latest brief JSON (CORS-enabled)
- `GET /brief.md` → latest brief as Markdown
- `GET /status` → delivery status (total, latest, out dir)
- unknown paths → 404
- CLI: `python agent/brief_server.py --port 8787 --out .tirra_delivery`

### Bug caught + fixed (real)
`BriefDeliverer.latest()` returned `records()[-1]` but `records()` is newest-first —
so "latest" was actually the **oldest** delivery, meaning the server could serve a
stale brief. Fixed to `records()[0]`, added a regression test
(`test_latest_returns_newest_not_oldest`). Verified server now serves the newest.

### Tests
- `tests/test_brief_server.py` — 4 tests (JSON, MD, status, 404) against a live
  ephemeral-port server thread
- `tests/test_brief_delivery.py` — 6 tests (incl. the newest-not-oldest regression)

## Verification (final, this session)
- Live: `deliver_brief.py --once` → `brief_server.py --port 8788` → `curl /brief.json`, `/brief.md`, `/status` all correct; 404 for unknown ✅
- Full regression — **71 passed**; `ruff` clean

## Honest P(win) fix (added same session)

**Found the user's correct complaint:** P(win) was a flat 0.5 for every contract —
that was the naive Beta(1,1) coin-flip *prior*, not a real estimate. Zero realized
bid outcomes had been recorded, so nothing ever moved it.

### What was done — `agent/quant/contract_opportunity.py`
1. **Real prior instead of coin-flip:** `WinProbabilityLearner(prior_wins=0.5, prior_bids=1.0)` → prior mean = 1/3 (a random small-business bid is not an even coin flip). Cold-start P(win) is now 0.33, honest.
2. **Evidence basis exposed:** each Opportunity now carries `p_win_basis` = `prior` | `learned`, so consumers can see whether a probability is a guess or backed by realized outcomes.
3. **Confirmed the API reality:** USASpending's `number_of_offers_received` / `Set Aside Type` fields exist in the API schema but are **populated as None** on these records, and SAM.gov opportunities require an API key — so we cannot fabricate competition data. The honest path is documented-prior → learned-from-outcomes.

Result: prior contracts at 0.33 (basis=prior); once realized outcomes are recorded for an agency+bucket, their P(win) moves to the posterior (e.g. 8W/2L → 0.74, basis=learned). Untouched agencies stay at the prior.

## Related
- [[checkpoint_2026-08-24_fused_intelligence_brief]]
- [[checkpoint_2026-08-24_live_path_intelligence]]
- [[checkpoint_2026-08-24_capital_pillar_ev_scorer]]
- [[checkpoint_2026-08-24_learning_signal_diagnosis]]