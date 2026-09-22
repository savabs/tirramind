---
title: "Checkpoint 2026-09-06 — tirramind-universe (Open Security Master probe)"
tags:
  - doc/checkpoint
  - topic/architecture
  - status/current
---

# Checkpoint 2026-09-06 — tirramind-universe (Open Security Master probe)

Spec: `docs/specs/open_security_master_spec.md` · Task: `tasks/active/open_security_master.md`
Repo: github.com/savabs/tirramind-universe (public) · local `~/Projects/tirramind-universe`
Pages: https://savabs.github.io/tirramind-universe/ · PyPI name `tirramind` reserved-by-absence, not yet published.

## What was built today (one session)
Phases A–D code complete, 41 tests. First data release pushed. Day 0 of our
own daily capture = 2026-09-06 09:11 UTC.

## What was learned (do not re-derive)
- `company_tickers.json` churn: 50% of removals re-appear; two partial
  captures (2020-06-05, 2021-08-09); 2019-09 file doubled + re-cased names.
  Handled by `annotate()` (relisted_at / suspect / cosmetic / superseded_*).
- Sibling ticker on same CIK within ±60d = symbol change or exchange
  transfer; **filings beat the sibling heuristic** except issuer-filed-25 +
  new exchange = transfer. Exchange for LISTED rows must be forward-filled.
- Item 2.01 fires for acquirers; merger needs 2.01 + (Form 15 | exchange-
  filed 25 | 3.01). Item 1.03 beside 2.01+5.01 is a mis-tick (Vista Outdoor).
- Co-registrants: match filings on any CIK in `all_ciks`.
- Accession prefix = submitter (filing agent), never the issuer.
- Form 4 history: SEC bulk quarterly zips (2026q2 not yet published →
  EFTS+XML gap fill from 2026-04-01). Form 144 has no bulk set: EFTS+XML.
- EFTS 500s are transient; client retries 5xx. Week-sized windows.
- Wayback also holds NASDAQ Trader `nasdaqlisted.txt` (79 captures to 2008):
  independent source; 7 Nasdaq-listed banks are absent from the SEC map
  (bank-regulator filers); SEC map lags Nasdaq by weeks after delistings.
- Results: 12,637 true delistings; NYSE/Nasdaq known-cause 93–94% (2022+);
  OTC 21% (dormant registrants — needs last-filing-date check); hand-check
  29/30 (`data/samples/handcheck_30.csv`); foreign-issuer 6-K blind spot.

## Running / pending when this was written
- Background: Form 144 (Sep 2025→) week by week, then Form 4 EFTS gap
  (2026-04-01→). Idempotent ledgers; safe to re-run
  (`collect_form144`, `collect_form4_recent`).
- Then: `python -m tirramind.build` → commit `data/ledgers/form144.jsonl`,
  `form4.jsonl`, `form144_links.parquet`, `accuracy.parquet`, `docs/index.html`;
  fill WRITEUP.md insider section; `gh workflow run daily.yml`; observe two
  scheduled commits; D3 post; D4 paid tier gated on signal.
- Owner: PyPI token for `tirramind`; Paddle products for D4 (later).

## Day-30 checkpoint due 2026-10-06 — success bar in the task file.

## Addendum — options surface lab (same day)
Owner redirected: engine-first, nothing accumulation-bound. Chose the options
surface lab (scan #1). Built in `~/Projects/voltorch`: eSSVI backbone + arb-checked
SVI refinement, executable-arbitrage check, keyless Deribit loader, live page on
GitHub Pages refit every 30 min. Spec/task: `docs/specs/options_surface_lab_spec.md`,
`tasks/active/options_surface_lab.md`. Three numerical bugs found and pinned by
tests: Newton IV stall on ordinary quotes (→ bisection), spot-vs-forward USD
conversion (3.7% at 292d), T-dependent warm start in the slice fitter.
Pending: PyPI 0.2.0 (after the Actions run proves the Linux install), write-up.

## 2026-09-07 — decisions
- **Options surface lab**: shipped (voltorch 0.2.0, live page every 30 min). Probe running; owner's verdict: too much doubt as a business — leave running, no further build.
- **Security master**: shipped (insider ledger released, daily job proven from GitHub). Same: leave running, no further build.
- **Prediction-market quant tooling**: elaborated; Polymarket unreachable from India, Kalshi rate-resets; superseded.
- **Autonomous trading operator (agent thinks / gateway decides)**: 6-agent verification → KILLED before code. Hummingbot Condor (Apr 2026) and Coinbase for Agents (Jun 2026) already ship the architecture; strategies professional-only; execution tools price at $29–149 with no subscription success (Hummingbot $309k/yr fee-share); **India: PROGA 2025 makes facilitation a non-bailable offence, Polymarket blocked, Kalshi withdrew — operator personally exposed.** Report: `docs/research/autonomous_trading_operator_verification_2026-09-07.md`.
- Owner's filter now: engineering moat · no selling · serious card-paying users · big companies avoid it · legal from India · value on day one · named ≤5-person comparable. Scan of non-finance fields launched (`scan_other_fields_2026-09-07.md`).

## 2026-09-10 — handoff out of this repo
Work moved to `~/Projects/tirramind-options` (TirraMind Options: BYO-broker options terminal on voltorch, card-paid, plus a day-one public track record). Read its `docs/memory/checkpoint_2026-09-10_start.md`. The studio storefront (savabs.github.io/work/) and the 157-firm lists in `docs/studio/` stay as passive assets — the owner will not do outreach. Staged-but-uncommitted docs in this repo sit on the Paddle PR branch; commit them to a docs branch or main when convenient.
