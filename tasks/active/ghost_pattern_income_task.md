---
title: "Task: Ghost Pattern Income — MP-1 Launch"
tags:
  - doc/task
  - topic/commercial
  - topic/world-model
  - status/paused
  - layer/surveillance
date: 2026-06-08
---

> Legacy reference. Primary doc: [[ghost_pattern_income_task.html]]

**Status:** **PAUSED** per [[0002-ghost-pattern-gtm-pause]] — founder role = ML/backend only; no sales track.

**Spec:** [[ghost_pattern_income_plan]] (GTM sections historical)  
**Vision:** [[long_term_vision]]

## Founder role (canonical)

| In scope | Out of scope |
|----------|--------------|
| Scanner, data pipeline, scorecard, archive as demo/labels | X, paid tier, audience, brief-as-product |
| Optional `make ghost-pattern-daily` | Convincing traders / domain voice |
| Phase B embeddings when tied to GNN | MP-2+ commercial expansion |

**Output (revised):** alert JSON + scorecard — not sold intelligence briefs.

## Steps

### Phase 0 — Foundation
- [x] 0.1 — Ghost archive scaffold + JSON schema
- [x] 0.2 — 3 MP-1 chain templates
- [x] 0.3 — MP-1 data health script

### Phase 0b — Scanner
- [x] 0b.1 — `agent/quant/ghost_chains.py` chain matcher
- [x] 0b.2 — `scripts/ghost_pattern_scan.py` CLI

### Phase 1 — Manual proof
- [x] 1.1–1.6 — 3 alerts, briefs, resolutions, daily loop, AIS backfill

### Phase 2 — Automation
- [x] 2.1 — Scanner + tuning
- [x] 2.2 — Draft/publish brief workflow (demo artifacts only)
- [x] ~~2.3 — Paid tier~~ **CANCELLED** (GTM pause)

### Phase 3 — Expand
- [x] ~~3.1 — MP-2 commercial~~ **CANCELLED**
- [x] ~~3.2 — Telegram/email paid pings~~ **CANCELLED**
- [ ] 3.3 — `embedding_snapshots` for MP-1 (optional; when GNN track needs it)

### Backend-only (optional, no deadline)
- [ ] B.1 — Run `make ghost-pattern-daily` on cron for archive/label growth
- [ ] B.2 — Revisit GTM only with domain partner or n≥20 + explicit opt-in

## Related

- [[0002-ghost-pattern-gtm-pause]]
- [[checkpoint_2026-06-09_ghost_gtm_pause]]
- [[ghost_pattern_income_plan]]
- [[ghost_pattern_technical_roadmap]]
- [[scorecard]]
- [[long_term_vision]]
