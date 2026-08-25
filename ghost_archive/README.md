---
title: Ghost Pattern Archive
tags:
  - doc/memory
  - topic/ghost-patterns
  - topic/commercial
  - status/active
  - layer/surveillance
---

# TirraMind Ghost Pattern Archive

**GTM paused (2026-06-09).** Demo / reproducibility archive only — not actively marketed. Decision: [[0002-ghost-pattern-gtm-pause]].

Cross-domain commodity chain detector output (alert JSON + scorecard). Briefs are historical demo artifacts.

Each alert satisfies **G1–G4** (see [[ghost_pattern_income_plan]]):

| Gate | Requirement |
|------|-------------|
| G1 | Chain — ≥2 domains in one alert |
| G2 | Graph path — `chain_template` ID links entities |
| G3 | Source — every node has public URL + timestamp |
| G4 | Outcome log — archived before resolution |

## Layout

```
ghost_archive/
  schema/ghost_alert.schema.json   # JSON Schema for alerts
  alerts/                          # One JSON file per alert (machine record)
  briefs/                          # PUBLISH — human-edited briefs (upload these)
  briefs/draft/                    # AUTO — scanner drafts (edit before publishing)
  scorecard.md                     # Hit rate by template vs baseline
```

## Workflow

1. `make ghost-pattern-daily` → alert JSON + **draft** brief
2. Edit draft → save publish copy in `briefs/` (fix units, add context, falsifiers)
3. Post `briefs/*.md` to X / public GitHub repo
4. Resolver updates outcome footers in both draft and publish briefs

## Commands

```bash
make ghost-pattern-daily
python scripts/generate_brief.py --all    # refresh drafts only
```

## Publish

Public repo: https://github.com/dry-clean/tirramind-ghost-archive

Publish: `make ghost-archive-publish` (uses `DRY_CLEAN_GITHUB_TOKEN` + `GH_USER=dry-clean` in `.env`).

Sync **`alerts/`**, **`briefs/`** (not `briefs/draft/`), and **`scorecard.md`** only.

## Related

- [[0002-ghost-pattern-gtm-pause]]
- [[scorecard]]
- [[ghost_pattern_income_plan]]
- [[ghost_pattern_income_task]]
- [[ghost_pattern_technical_roadmap]]
