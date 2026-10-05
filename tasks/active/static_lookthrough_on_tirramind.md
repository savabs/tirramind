---
title: "Task: Ship the look-through on tirramind.com, computed in the browser"
tags:
  - doc/task
  - phase/publish
  - topic/site
  - topic/portfolio
  - status/active
---

# Task: Ship the look-through on tirramind.com, computed in the browser

Status: active
Research: docs/research/static_lookthrough.md
Spec: docs/specs/product_spec.md
Spec section: §5 step 1 — "what you're exposed to"
Surface: tirramind.com (Cloudflare Pages, `products/site/`), project `tirramind`
Engine: `agent/lookthrough/{indices,funds,combine}.py`

## Why

The look-through is the only finding in this project that survived honest
measurement. Six candidate headlines were tested and five died — "12 positions,
3 bets" occurred **0 times in 200** real books, picking cost was a coin flip
(223 of 493 rolling windows, median −0.2%), and ghost chains returned **0
corroborated of 371,368** twice. The survivor is simple addition about
something a broker genuinely does not show you, and it is checkable by hand
against a fund's own factsheet.

It was reachable only by running `agent/portfolio/server.py` on a laptop.

## Why static rather than on the VM

The look-through needs index **membership** and index **weights**. Both are
precomputable; neither needs price history. Only the picking-cost and factor
blocks need yfinance, and those stay on the server.

That makes the product a static page, which is not a cost saving dressed up as
architecture:

- `api.tirramind.com` resolves to `188.245.203.137`, which answers ping with
  **every port filtered** (22, 80, 443). Its state is unknown and it cannot be
  deployed to.
- `docs/runbooks/GET_THE_SERVER_UP.md` records **26 days** of silent collection
  downtime on the existing host. A page that computes in the browser cannot
  have that failure mode at all.
- Cloudflare Pages' free tier permits commercial use, so this adds **£0**.

Holdings never leave the tab, which is also the honest version of the privacy
claim the server could only assert.

## The risk this task is built around

A second implementation of one calculation in a second language is how one
question acquires two answers, and most of `LESSONS.md` is variations on that.
The mitigation is mechanical, not diligence:

1. `products/site/lookthrough.js` holds **no data**. Every table, phrase list
   and regex is serialised out of the Python modules by
   `scripts/export_lookthrough_data.py`. One source of truth, and it is Python's.
2. `tests/test_lookthrough_static_parity.py` runs the JS under node over
   **3,446 real AMFI scheme names** and **17 whole books**, requiring exact
   agreement — phrase for phrase, weights at `1e-12`.

The permitted divergence has one direction: the JS has no AMFI master, so it may
**refuse** a fund Python would unpack. It must never unpack one Python refused,
nor unpack it into a different index. That asymmetry is itself a test.

## Exit condition

`https://tirramind.com/holdings` serves the tool; it computes with no network
call beyond one 105 KB data file; the homepage leads with it; and `/queue`,
`/predictions`, `/portfolio` and `/predictions.csv` are unchanged at their URLs.

## Steps

- [x] LT.1 — `scripts/export_lookthrough_data.py`: 7 indices, 871 constituents,
      resolver tables, golden books and the parity corpus
- [x] LT.2 — Split the output: 105 KB of page data to `products/site/data/`,
      693 KB of test evidence to `tests/fixtures/` where no deploy can carry it
- [x] LT.3 — `products/site/lookthrough.js` — arithmetic and table lookups only
- [x] LT.4 — `tests/test_lookthrough_static_parity.py`; verified it catches a
      1e-9 weight drift and the null-vs-zero distinction by mutation
- [x] LT.5 — **F-19**: the parity harness found five defects in the SHIPPED
      Python engine, including a government-bond fund expanded into fifty
      equities and 20 of 26 ETF tickers read as companies. Fixed, with
      `LESSONS.md` F-19 and tests that iterate the tables rather than copies
- [x] LT.6 — `products/site/holdings.html`; verified in a browser including
      every refusal path and mobile at 390px with no horizontal overflow
- [x] LT.7 — Homepage leads with the tool; grid work moves to "Also published"
      with all content and URLs intact
- [x] LT.8 — `.gitignore`: a blanket `*.json` was excluding the page's own data
      file. Harmless for the first deploy (wrangler uploads the working
      directory) and fatal for the first deploy from a fresh clone
- [x] LT.9 — Deployed. Preview `472e4ce7` verified first, then promoted to
      `main`. `/holdings` serves the page (previously 200 only because the
      Pages 404 fallback was returning the old homepage). Rollback point:
      deployment `9d91dc2b` from commit `1521e3c`. A copy fix followed —
      "this module" was developer language, caught by reading the live page
- [ ] LT.10 — **The falsifier, deferred four times now:** show the output to
      five people and watch whether `You own 14.2% of HDFCBANK` actually
      surprises them. Six headlines died when measured; this one has only been
      reasoned about. `docs/specs/product_spec.md` §7 calls this the test that
      decides everything, and it costs an afternoon

## Known limits, stated rather than hidden

- **Seven indices.** Nifty 50, Next 50, 100, 500, Bank, IT, Midcap 150. A fund
  tracking anything else is named and left alone.
- **The weights are ours.** NSE publishes membership, not free weights, so they
  are computed from free-float market cap. Measured against INDY: mean error
  **0.37pp**, worst 0.62pp. The note renders beside the numbers, not in a footer.
- **NIFTY 500 ships 497 names.** Four have no Yahoo market cap; they are
  excluded with a reason and the rest renormalised.
- **Percentages or one-currency amounts only.** Share counts need a price per
  line and a fund has no price here, so they are refused with instructions.
- **Free-float factors are Yahoo's estimate**, not NSE's quarterly IWF, and
  index capping rules are not applied.

## Related

- [[product_spec]] — why this is step 1 and why picking cost is demoted
- [[GET_THE_SERVER_UP]] — the 26 dark days that argue for computing client-side
- [[public_site_prediction_ledger]] — the other thing on this domain; untouched
