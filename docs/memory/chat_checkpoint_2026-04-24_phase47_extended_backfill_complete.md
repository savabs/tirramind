---
title: "Checkpoint: Phase 47 Extended Backfill Complete — 2026-04-24"
tags:
  - doc/checkpoint
  - phase/47
  - topic/pipeline
  - topic/world-model
  - layer/surveillance
  - status/done
---

# Checkpoint: Phase 47 Extended Backfill Complete
**Date:** 2026-04-24  
**Session duration:** Full day  
**Status at logout:** Phase 47 COMPLETE. Phase 40 is next.

---

## What We Did

### Starting Point
Session picked up mid-stream. Phase 47 Group B DataCache bugs had already been fixed in the previous session (energy_supply, internet_infrastructure, political_risk, foia_requests). The group B backfill had run but country entities only had ~15 obs/entity — way too thin for Phase 40 GNN training.

The core problem identified: the backfill had only run with `--days-back 1825` (5 years) and GDELT was not even in the backfill plan yet (it was in the "deferred" group because nobody had added historical support to the GDELT tool). CFTC was capped at `max(2020, ...)` so only 2020–2026 data was fetched.

### Decision: 10-Year Extended Backfill
We extended the backfill to `--days-back 3650` (10 years) and fixed two root causes before running it:

**Fix 1: GDELT had zero historical support**
The GDELT tool only knew how to fetch the last 1–24 hours of 15-minute news batches (live mode). It had no concept of historical date ranges. We built a full `_backfill` mode from scratch inside `agent/tools/gdelt.py`:
- `execute()` accepts `_backfill=True`, `days_back`, `sample_every_days=7`
- `_execute_backfill()`: computes evenly-spaced historical batch timestamps going back N days, downloads each 15-min batch ZIP from GDELT's immutable historical archive, caches each batch (idempotent), parses events, persists entity observations
- `_compute_historical_sample_timestamps()`: generates timestamps at noon UTC, one per 7 days going back N days — format `YYYYMMDDHHMMSS`
- Sampling at weekly cadence (not daily) to keep runtime sane: 521 batches over 10 years

**Fix 2: GDELT `observed_at` was storing date strings, not timestamps**
A silent type bug in `_persist_entities_inner()`: the `observed_at` column in `entity_observations` is `REAL NOT NULL` (Unix timestamp float), but the code was storing `ev.get("date")` which returns a string like `"20160701"`. SQLite accepts it silently but the density_audit span calculations break because you can't compute time differences on strings. Fixed by converting: parse `date[:4]`, `date[4:6]`, `date[6:8]` → `datetime(..., 12, 0, 0, tzinfo=timezone.utc).timestamp()`.

**Fix 3: CFTC year floor was capping at 2020**
`scripts/backfill.py` had `max(2020, datetime.now().year - (days_back // 365) - 1)`. With days_back=3650, this resolves to start=2016, but the `max(2020, ...)` clamp was binding — cutting off 2015–2019. CFTC ZIPs are confirmed available from 2011+. Changed to `max(2011, ...)`.

**Fix 4: GDELT wired into backfill plan**
GDELT was in the "deferred: needs start_datetime param added" group. Added it to Group A:
```python
{"label": "gdelt_backfill", "tool": "gdelt", "kwargs": {"_backfill": True, "days_back": days_back, "sample_every_days": 7}}
```

### The Run
```
set -a && source .env && set +a && python scripts/backfill.py --days-back 3650
```
Ran for 17.2 minutes. Final result:
```
Done: 7 completed, 0 failed, 11 skipped
Total new observations: 900,442  (before=77,421, after=977,863)
Wall time: 17.2 min
```

Breakdown:
- `gdelt_backfill`: +900,306 obs — 521 weekly batches 2016–2026
- `cftc_2015` through `cftc_2019`: +20 obs each = +100 obs total
- `insider_filings`: +36 obs at 3,650d range

### Density Audit After
```
country        233 entities   901,766 obs   3870 obs/ent   38830d span   OK
instrument      89 entities    69,424 obs    780 obs/ent    1099d span   OK
topic        1,235 entities     5,394 obs      4.4 obs/ent   639d span   OK
person         459 entities       731 obs      1.6 obs/ent  1053d span   OK
cftc_contract   20 entities       300 obs     15.0 obs/ent  4116d span   OK
wallet          33 entities       122 obs      3.7 obs/ent     3d span   SPARSE
protocol        21 entities        60 obs      2.9 obs/ent     3d span   SPARSE
company         25 entities        50 obs      2.0 obs/ent 19754d span   SPARSE
organization     8 entities        16 obs      2.0 obs/ent     0d span   SPARSE
```

5 core GNN entity types: **all PASS**. 4 supplementary sparse types: already overridden in task file. Phase 40 is unblocked.

---

## What We Thought About

### Data Imbalance: GDELT Dominates
After seeing the per-tool breakdown (GDELT=901K, instrument_universe=69K, everything else=<7K combined), we discussed why this imbalance exists.

**The insight: it's not a data problem, it's a loop problem.**

Most of the 51 tools were designed as daily snapshot fetchers — they answer "what's happening now." They call an API with `mrv=1` (most recent value) or similar, write one row per entity, and exit. Run them 100 times → still 1 obs per entity.

GDELT and instrument_universe were the only two tools with real historical loops baked in:
- GDELT: after we added `_backfill` mode, it loops over 521 weekly batches
- instrument_universe: fetches daily OHLCV bars with a full date range → 89 instruments × ~780 bars = 69K

Every other tool has the data sitting in its upstream API (World Bank, FRED, SEC, CFTC). They just never loop over it. The fix is ~5–20 LOC per tool — add a `days_back` parameter and loop over years/months instead of fetching only "now."

### What Phase 40 Will Actually Train On
Given the data distribution, Phase 40 is effectively training a **geopolitical → instrument prediction model**:
- Country entities are rich (3,870 obs/ent) — GNN will learn country patterns well
- Instrument entities are OK (780 obs/ent) — price history is there
- Person/CFTC/topic: thin (1–15 obs/ent) — won't contribute much

Cross-domain signal chains that WILL work:
```
country:US ──geopolitical_event──→ instrument:SPY   (via GDELT + instrument_universe)
country:CN ──geopolitical_event──→ instrument:EWZ   (trade war signal)
```

Cross-domain signal chains that WON'T work yet (data too thin):
```
person:Musk ──insider_buy──→ instrument:TSLA       (insider_filings: 237 obs total)
cftc_contract:CL ──positioning──→ instrument:USO   (CFTC: 300 obs total)
```

This is honest: Phase 40 gives the first real geopolitical signal read. Not the full system. That comes as loops are added to other tools.

### Architecture Insight: New Data Auto-Integrates
The key architectural property discussed: the schema is universal. Every tool writes to the same `entity_observations` table. The GNN training pipeline reads from that table — it doesn't know or care which tool wrote a row. So:

1. Add historical loop to `sovereign_debt` → run backfill → 5,850 new rows appear
2. Next GNN retrain automatically sees those edges — no wiring change
3. New cross-domain chains emerge: `sovereign_yield → country → geopolitical_event → instrument`

The foundation is solid. From here it's a data ingestion problem, not an architecture problem.

### The "Missing Loop" Pattern
Articulated a reusable pattern for what's broken in thin tools and how to fix it:

**Broken (snapshot-only):**
```python
data = requests.get("https://api.worldbank.org/...?mrv=1")  # most recent only
```

**Fixed (with historical loop):**
```python
for year in range(start_year, today.year + 1):
    data = requests.get(f"https://api.worldbank.org/...?date={year}")
    for country in data:
        store.persist(..., observed_at=timestamp(year), ...)
```

This pattern applies to: sovereign_debt, global_pmi, central_bank_balance, capital_flows, political_risk — all confirmed to have APIs that support date range queries.

---

## Unique Ideas This Session

### GDELT as the Backbone Layer
GDELT is the only free source that covers every country simultaneously at 15-minute resolution going back to 2015. It's not the most precise signal — but it creates the temporal backbone that all other signals can attach to. When you add sovereign debt obs for country:US on 2019-08-01, the GNN can correlate it with the 47 GDELT events involving the US in the week before that date. The backbone makes the cross-source correlations learnable.

This is what we mean by "unconventional observation × SOTA math = asymmetric edge." Everyone has GDELT. Nobody builds a temporal heterogeneous graph that joins GDELT events with sovereign debt moves, insider filings, and CFTC positioning on the same country/person/instrument node timeline.

### The "Loop Audit" as a Distinct Phase
Every tool should be audited not just for "does it have L2 persistence" but "does it have a historical loop." These are separate capabilities. A tool can be L2 (entity-resolved, writes per-entity obs) but still snapshot-only (only writes today's value). Both need to be true for the tool to contribute to GNN training depth.

Proposed: add `"has_historical_loop": true/false` to each tool's audit entry in the task file. Tools without a loop should be upgraded before Phase 40 retraining so the next training run benefits immediately.

### Density Audit as Training Gate
The density_audit script's exit code design (exit 0 = Phase 40 ready, exit 1 = FAIL) was the right call. It creates a mechanical, non-negotiable gate: you cannot run Phase 40 until the audit passes. This prevents premature GNN training on junk data — a mistake that would produce misleading embeddings and a false "no signal" conclusion. The 4 SPARSE types being overridden (wallet, protocol, company, organization) was a deliberate engineering decision: these are supplementary cross-domain types, not primary GNN training nodes. Overriding them explicitly in the task file documents the decision.

---

## Current State at Logout

### DB State
```
Total observations:  977,863
Total entities:        2,450
DB path:             .tirra_pipeline/pipeline.db
```

### File Changes This Session
- `agent/tools/gdelt.py` — added `_backfill=True` mode, fixed `observed_at` string→float bug
- `scripts/backfill.py` — GDELT added to Group A, CFTC floor 2020→2011
- `[[quant_training_ground]]` — updated DB stats, Phase 47 final state block
- `/memories/repo/tirramind_structure.md` — Phase 47 entry updated to COMPLETE with new stats

### Test State
9,676+ tests passing (last confirmed 2026-04-23 post-Phase 46). No tests were touched this session.

---

## What's Next: Phase 40

**Phase 40: Real Data Model Refresh**

The DAG has been live since 2026-04-22. The 10-year GDELT backfill gives 10 years of historical training data. The density mandate is satisfied for all 5 core GNN entity types.

**Phase 40 steps (from task file):**
1. Connect real surveillance observations from `entity_observations` into GNN training pipeline
2. Pair with real price series (instrument_universe observations)  
3. Run walk-forward backtest against real entity observations (not synthetic)
4. First real signal-vs-noise read — does geopolitical signal precede price moves?

**Note from task file:** "⚠️ DO NOT START PHASE 40 BEFORE THE DAG HAS RUN FOR 3–4 WEEKS MINIMUM." — This was written before the 10-year backfill was possible. With 10 years of historical training data now in the DB, this constraint is arguably satisfied by the backfill. Discuss at next session start whether to proceed immediately or wait for more live accumulation.

**After Phase 40:**
- GNN attention diagnostic: which entity types and cross-domain paths are attention-starved?
- Loop audit: which tools need historical loops added to feed starved paths?
- Targeted tool upgrades: add `days_back` loops to sovereign_debt, global_pmi, central_bank_balance, capital_flows, political_risk
- Paper trade launch — first real capital-at-risk test

---

## Known Blockers (for reference)

| Tool | Blocker |
|---|---|
| electricity_monitor, interconnection_queue | Need `TIRRA_EIA_API_KEY` — not in `.env` |
| finra_short_volume | FINRA API returns 204 No Content |
| polymarket_whales, power_grid, treasury_receipts, patent_filings | No L2 persistence logic |
| whale_alert (wallet) | No historical loop — API is real-time only |
| defi_flows (protocol) | Snapshot-only, no historical date range |
| person density (1.6 obs/ent) | SEC EDGAR limited to ~500 filings per call window |

---

## Related
- [[quant_training_ground]] — master task file, canonical roadmap
- [[historical_backfill]] — research doc for backfill architecture
- [[chat_checkpoint_2026-04-23_phase46_ewc_complete]] — previous checkpoint
