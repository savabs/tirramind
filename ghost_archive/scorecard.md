---
title: Ghost Pattern Scorecard — MP-1
tags:
  - doc/memory
  - topic/ghost-patterns
  - topic/commercial
  - status/active
  - layer/surveillance
---

# Ghost Pattern Scorecard — MP-1 Atlantic Energy

**GTM paused** — honest track record for demo/ML labels only ([[0002-ghost-pattern-gtm-pause]]).

| Metric | Value |
|--------|-------|
| Micro-playground | MP-1 |
| Alerts issued | 3 |
| Resolved | 3 |
| Pending | 0 |
| Directional hit rate | **33%** (1/3) |
| Evaluation window | **2–5 trading sessions** |
| Auto-scanner | **live** (`eia_regime_cftc` fires at score ≥ 1.5) |

## Alerts

| Alert ID | Template | Issued | Outcome | Return |
|----------|----------|--------|---------|--------|
| 2026-06-09_MP-1_EIA_REGIME_CFTC_001 | eia_regime_cftc | 2026-06-09 | **up** | +4.19% (2 sessions) |
| 2026-06-09_MP-1_AIS_GDELT_CFTC_VELOCITY_002 | ais_gdelt_cftc_velocity | 2026-06-09 | down | -4.15% (5 sessions) |
| 2026-06-09_MP-1_AIS_EIA_CFTC_003 | ais_eia_cftc | 2026-06-09 | down | -4.15% (5 sessions) |

## Notes

- Alerts #2–#3 were **scenario flags** (not directional buy calls); CL=F fell 4.15% over 5 sessions after the Jun 2 CFTC anchor.
- Readout prices backfilled via `make ghost-readout-backfill` (per-trading-day `instrument_daily` bars).

## Public archive

**Live:** https://github.com/dry-clean/tirramind-ghost-archive

Republish: `make ghost-archive-publish` (uses `DRY_CLEAN_GITHUB_TOKEN` + `GH_USER=dry-clean` in `.env`)

## Publish workflow

| Folder | Upload? |
|--------|---------|
| `briefs/` | **Yes** — human-edited, post these |
| `briefs/draft/` | **No** — auto skeletons for editing |
| `alerts/` | **Yes** — machine record + transparency |

## Daily loop

```bash
make ghost-pattern-daily   # → alerts/ + briefs/draft/
# then edit draft → save to briefs/ before posting
```

## Related

- [[ghost_pattern_income_plan]]
- [[ghost_pattern_income_task]]
