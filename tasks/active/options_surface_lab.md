---
title: "Task: Options Surface Lab — 30-day engine-first probe"
tags:
  - doc/task
  - topic/quant
  - status/active
---

# Task: Options Surface Lab — 30-day engine-first probe

Spec: `docs/specs/options_surface_lab_spec.md` · Research: `docs/research/options_surface_lab.md`
Started 2026-09-06 · Day-30 checkpoint 2026-10-06 · Repo `~/Projects/voltorch`

## A — kernel
- [x] A1 SSVI **and eSSVI** (Hendriks–Martini; per-expiry skew, arb-free by construction) + SVISlice refinement (arb-checked, backbone fallback); synthetic recovery; Durrleman ≥ 0; mps+cpu
- [x] A2 durrleman / calendar / butterfly / vertical + **executable_violations** on bid/ask; bump flagged, flat clean, sub-tol ignored
- [x] A3 keyless loader; live 900 BTC / 784 ETH rows in <1s; one-sided 5%; **coin price × forward = Black-76 USD price (verified 1e-4 vs marks), pinned by test**; bisection IV solver replaced Newton (stalled on ordinary quotes)
- [x] A4 fit_chain: BTC refined 0.64 vol pts / 86% inside (backbone 1.8/62%), ETH 0.66 / 96%; venue marks arb-free (model-generated); **executable arbitrage in the book: 0** on both; ours 0 (g_min 0.25); greeks err ≤ 1e-9; ~5s total on CPU

## B — page
- [x] B1 live page rendered (22 SVG smiles, per-expiry table, executable-arb block, 7-day history)
- [~] B2 surface.yml; Pages live at https://savabs.github.io/voltorch/; dispatch run 34037700471 green (BTC 0.69/89%, ETH 0.56/97%, committed); scheduled runs every 30 min all green through 2026-09-07 04:37 UTC (B2 verified)

## C — release + write-up
- [x] C1 voltorch 0.2.0 on PyPI (https://pypi.org/project/voltorch/0.2.0/), fresh-venv verified
- [ ] C2 write-up posted — URLs/dates: ____
- [ ] C3 day-30 checkpoint

## Success bar (any one, day 30)
≥300 stars · ≥500 installs/wk · one stranger runs fit_chain on own data · ≥10 inbound for equity version

## Log
- 2026-09-06 — chosen over backtester / bitemporal store (owner: engine-first, no accumulation). Spec written.
- 2026-09-06 — A1–A4, B1 done in one session; Pages live. Findings: Deribit marks are model-generated and arb-free (the scan's "marks contain arbitrage" claim was false); the book showed 0 executable arbitrage; the page states both honestly.
