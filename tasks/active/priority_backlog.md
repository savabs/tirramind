---
title: Task — Priority Backlog (post agent-team audit)
tags:
  - doc/task
  - topic/product
  - topic/security
  - status/active
---

# Task — Priority Backlog

Research: `docs/research/evidence_ingest_path_traversal.md`
Spec: `docs/specs/evidence_ingest_hardening_spec.md`

Compiled 2026-08-27 from a four-specialist audit (payments, customer-lifecycle,
security, frontend) plus two multi-agent workflow rounds. Every number below was
measured, not estimated.

## The one-line summary

The plumbing was rebuilt today and now mostly works. Nothing above it is
validated. **Predictive edge remains unproven — that is the only item that
decides whether any of this is worth selling.**

---

## P0 — blocks correctness or revenue

| # | Item | Evidence | Owner |
|---|---|---|---|
| 1 | 4 test failures new this round (`test_gnn_integration` ×2, `test_awos_watchers` ×2) | Absent from the original 19; `executor.py` gained ~135 lines | me |
| 2 | `test_outcome_finetuning` ×6 untouched | Largest cluster of the original 19 | me |

Clean full-suite baseline 2026-08-27: **10 failed, 10,819 passed, 14 skipped**
(down from 19 failed). Two independent runs agreed exactly, so the count is not
an artifact of racing against concurrent edits.
| 3 | Source allowlist gap | `COUNT(DISTINCT source)` = 66, allowlist = 51 → 15 real sources 400 | me |
| 4 | **Timers do not exist on the server** | only `tirra-disk-check.timer` present | supervised prod work |
| 5 | ~1631 insertions uncommitted | `git status` | me |

Item 4 is the one that makes the others moot: the pipeline fix is real and
verified (a live run moved `pipeline_data` 330→375 and `entity_observations`
+2846), but nothing schedules it, so it only helps when run by hand.

## P1 — security not finished

| # | Item | Note |
|---|---|---|
| 6 | `TIRRA_REJECT_QUERY_KEYS` defaults OFF | prod still writes live API keys into Caddy access logs in cleartext |
| 7 | Key rotation route missing | store support exists, no HTTP route. `welcome.html` shows the key ONCE and there is no mailbox — a lost key is currently unrecoverable |
| 8 | Caddy log redaction | prod-side |

Closed already: `9fa68ca` (arbitrary file read via `/evidence/ingest` +
fail-open ingest token).

## P2 — product honesty

| # | Item | Evidence |
|---|---|---|
| 9 | Opportunity Brief has produced **zero** output, ever | `/status total_deliveries = 0` |
| 10 | Data Platform thin | 194→375 rows; sold as "daily-refreshed, 47 sources" |
| 11 | `entry["email"]` captured, never used | no backup key delivery |
| 12 | Contact messages unreadable | land server-side with no retrieval path |

Tiers 9 and 10 are gated to "coming soon" in `pricing.html` for exactly this
reason. Entity Graph (5,628 entities / 16,870 links) and Scheduler (65
`dag_runs`) are real and sellable on data quality.

## P3 — owner only

- Cloudflare Email Routing (fixes bouncing `support@`, unblocks email delivery)
- Paddle: domain approval → Currencies (INR) → UPI toggle
- UPI note: RBI/NPCI cap AutoPay subscriptions at ₹15,000/renewal, so Entity
  Graph (~₹25k) and Data Platform (~₹42k) can **never** use recurring UPI
- R2 bucket, free API keys (FRED / NASA FIRMS / EIA), HetrixTools
- Rotate the Cloudflare and Paddle tokens pasted into chat

## P4 — the actual question

| # | Item |
|---|---|
| 17 | **Predictive edge is UNVALIDATED.** Everything above is plumbing |
| 18 | GNN retrain (Kaggle, owner) · `_redirects` external-proxy decision · delete 2 unused Vercel projects |

## Outcome

_(update as items land)_
