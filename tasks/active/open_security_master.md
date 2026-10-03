---
title: "Task: Open Security Master — 30-day demand probe"
tags:
  - doc/task
  - topic/api
  - status/active
---

# Task: Open Security Master — 30-day demand probe

Spec: `docs/specs/open_security_master_spec.md`
Research: `docs/research/open_security_master.md`
Started: 2026-09-06 · Day-30 checkpoint due: 2026-10-06

## Owner-only
- [x] Name: tirramind → repo `savabs/tirramind-universe`, PyPI `tirramind` (2026-09-06)
- [x] Contact email provided → `TIRRAMIND_CONTACT` (2026-09-06)
- [ ] (D4, later) Paddle products for the $9–19 tier

## Phase A — repo and store
- [x] A1 repo skeleton, pyproject, Apache-2.0 — `~/Projects/tirramind-universe`
- [x] A2 store ported, 4 tests
- [x] A3 SEC client; live: 50/50 ok at 8.5 req/s

## Phase B — listing ledger
- [x] B1 day-0 snapshot 2026-09-06T09:11Z: 10,415 rows (Nasdaq 4374, NYSE 3301, OTC 2495, CBOE 36, blank 209); rerun `unchanged`
- [x] B2 diff → LISTED / DELISTED_FROM_MAP / SYMBOL_CHANGED / NAME_CHANGED / EXCHANGE_CHANGED
- [x] B3 Wayback: 267 ticker captures 2017-08→, 26 exchange-file captures 2021-07→
- [x] B4 77,878 filings 2015→ (Form 25/15, 8-K 1.03/2.01/3.01/5.03); 2 months hit EFTS 500s (2015-01, 2018-02 form 25) — rerun pending
- [x] B5 live: 12,716 true delistings; UNKNOWN 48% overall, 6-7% on NYSE/Nasdaq (2022+), 79% on OTC. Hand-check 28/30 (`data/samples/handcheck_30.csv`); misses: foreign-issuer 6-K blind spot (OCFT). Artefacts handled: 50% flap rate, 2 suspect capture steps, 1,529 deferred symbol changes, 39 exchange transfers
- [x] B6 parquet + README published (commit "first data release")

## Phase C — insider ledger
- [x] C1 forms.py — Form 4 returns all non-derivative codes (old parser dropped S); 5 tests
- [ ] C2 Form 144 (2023-04→) and Form 4 (2023-01→) collected, row counts printed
- [x] C3 link.py + window_complete; live: CIK 83.1%, name 0.02%, none 16.9% (37,651 notices)
- [x] C4 live (full Form 4 coverage to 2026-09-06): 28,064 complete windows; P(sale ≤90d) 68.2% [67.2, 69.2], n_eff 20,607; officers 76.1%, directors 75.6%; 63.7% same-day
- [x] C5 page.py; GitHub Pages enabled → https://savabs.github.io/tirramind-universe/ (page renders after insider ledgers land)

## Phase D — distribution
- [x] D1 dispatch run green; first scheduled run 2026-09-07 22:41 UTC green (committed). Verified.
- [x] D2 published https://pypi.org/project/tirramind/0.1.0/ (twine via existing ~/.pypirc); fresh-venv install + call verified
- [~] D3 WRITEUP.md complete (both halves); posting pending — owner's call
- [ ] D4 paid tier (gated; fix handler.py atomic save + tier fallback first)
- [ ] D5 day-30 checkpoint with the decision

## Golden tests
- [ ] diff fixture · [ ] Form 25 subject CIK · [ ] Form 4 code S · [ ] Form 144 filer CIK · [ ] effective n < n · [ ] zero-row exit

## Success bar (day 30, any one)
≥300 stars · ≥500 installs/wk · ≥500 parquet downloads · ≥10 inbound history requests · ≥10 paying
None → STOP. Job keeps running; nothing further built.

## Log
- 2026-09-06 — spec written from the two venture scans.
- 2026-09-06 — A1–A3, B1, B2 done with 13 tests; B3 Wayback backfill running.
- 2026-09-06 — B4–B6, C1, C3–C5, D1 code complete, 31 tests, 6 commits in ~/Projects/tirramind-universe. Backfills running. Decision: Form 4 history via SEC bulk quarterly datasets, not per-filing XML (1.4M fetches = 39h).
- 2026-09-06 — repo public at github.com/savabs/tirramind-universe; first data release pushed. Insider collection (Form 4 bulk + 12mo Form 144) running.
- 2026-09-06 (late) — causes hardened (transfers, deferred symbol changes, co-registrants, mis-ticked 1.03); NASDAQ directory source + disagreement ledger; hand-check 29/30; checkpoint `docs/memory/checkpoint_2026-09-06_tirramind_universe.md`.
- 2026-09-07 — insider ledger released; daily job proven from GitHub; write-up complete.
- 2026-09-08 — Form 4 gap fill complete; scheduled daily run verified; write-up numbers final. Probe now fully autonomous. Owner decision pending on posting.
