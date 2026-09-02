---
title: "Task: Public Site — Open Prediction Ledger"
tags:
  - doc/task
  - phase/publish
  - topic/site
  - topic/interconnection
  - status/active
---

# Task: Public Site — Open Prediction Ledger

Status: active
Research: docs/research/interconnection_queue_attrition.md
Spec: docs/specs/prediction_ledger_spec.md
Source repo: `savabs/queue_attrition` (the model, the ledger, the snapshot archive)
Surface: tirramind.com (Cloudflare Pages, `products/site/`)

## Why

The signal product is withdrawn and the site says so. What replaced it is
interconnection-queue attrition: base rates on whether requested generation
ever gets built. `/queue` published those base rates. This task publishes the
thing the base rates are *for* — a forward record of calls made before their
outcomes exist.

A backtest is a claim about a past that has already been seen. The only
forecasting record that cannot be edited into looking good is one where the
predictions are timestamped in public and the outcomes have not happened yet.
That is the asset; the site is where it lives.

## Exit Condition

`https://tirramind.com/predictions` serves the ledger page, and
`https://tirramind.com/predictions.csv` serves all 1,636 rows, with the page
stating plainly that nothing has resolved. Reachable from `/` and `/queue`.

## Steps

- [x] PL.1 — Audit `resolves_by` before publishing anything (found 62 elapsed
      deadlines and 63 mis-formatted dates; fixed upstream in `queue_attrition`
      commit `abb4e26`)
- [x] PL.2 — Regenerate the ledger; keep the superseded file as evidence
- [x] PL.3 — Pull the walk-forward scorecard, including the calibration failure
- [x] PL.4 — Write `products/site/predictions.html`
- [x] PL.5 — Cross-link from `index.html` and `queue.html`
- [x] PL.6 — Deploy to Cloudflare Pages and verify all four URLs return 200
- [x] PL.7 — Publish the portfolio-risk result at `/portfolio`: outcomes
      co-move (permutation null, every level p < 0.00025), independence
      understates a concentrated book's spread by 4.3x
- [x] PL.11 — Rewrite all four pages in plain language with new diagrams. 76
      agents across three workflow passes; 53 invented claims caught and removed
      before anything shipped
- [ ] PL.9 — Fix the calibration over-spreading (model says 65%, means 49%);
      live hypothesis is thin segment-concentrated high bins
- [ ] PL.10 — Capacity calibration still runs +6.1% after MW-weighting, and the
      live book is 68% MISO against 19% in training. The `/portfolio` page says
      so; closing the gap is the work
- [ ] PL.8 — First resolutions: score the ledger as ISOs report terminal
      statuses into the snapshot archive, well before the 2028 deadlines

## Constraints

- The ledger is append-only and frozen from 2026-09-02. A modelling mistake
  gets a new model ID, not a rewritten file.
- The page publishes what the model does badly next to what it does well. A
  record that keeps only the flattering half is not a record.
