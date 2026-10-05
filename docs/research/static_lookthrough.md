---
title: "Research: can the look-through run without a server?"
tags:
  - doc/research
  - topic/portfolio
  - topic/infrastructure
  - status/current
date: 2026-10-05
---

# Research: can the look-through run without a server?

Facts and measurements only. The decision they support is in
`docs/specs/product_spec.md`; the work is
`tasks/active/static_lookthrough_on_tirramind.md`.

## 1. The question

The portfolio audit computes several blocks. Only one of them survived honest
measurement. Does that one need a Python process, or can it be precomputed and
finished in a browser?

## 2. What each block actually requires

| block | needs | source |
|---|---|---|
| **look-through** | index membership, index weights | NSE constituent CSV + Yahoo market caps |
| picking cost | daily price history per holding | yfinance |
| factor exposure | daily price history + benchmark | yfinance |
| concentration / overlap | daily returns covariance | yfinance |

The look-through is the only block with **no time series in its inputs**. Its
whole calculation is

```
actual[company] = direct[company]
                + Σ over funds of  fund_weight × constituent_weight
```

Both right-hand terms are static for a given as-of date. Nothing is forecast,
nothing is fitted, and no price path enters.

## 3. Size of the precomputed input

Measured by running the exporter against all seven supported indices:

```
indices                    7
constituents             871
page data            105,259 bytes
```

For comparison, `products/site/predictions.csv` already served from the same
origin is **486,983 bytes** — 4.6× larger. The data file is not the constraint.

The parity corpus (3,446 AMFI scheme names) and golden books add 693 KB, but
those are test evidence and are written to `tests/fixtures/`, outside anything
a deploy carries.

## 4. State of the existing host

```
api.tirramind.com     →  188.245.203.137
ping                  →  2/2 packets, 160 ms avg
tcp/22, tcp/80, tcp/443  →  all filtered
GET /api/health       →  timeout after 15s
```

The host answers ICMP and nothing else. Its state is unknown from here and it
cannot be deployed to. Provisioning a replacement is ~$5/mo and needs the
owner's approval under CLAUDE.md §7.

## 5. Reliability, measured rather than argued

`docs/runbooks/GET_THE_SERVER_UP.md` records collection dark from **2026-08-27
to 2026-09-23** — 26 days, unrecoverable for the four time-gated sources. That
outage was not detected by monitoring; it was found later by counting rows.

A page whose computation happens in the visitor's tab has no uptime to lose.
The failure mode it replaces is not hypothetical and not small.

## 6. Hosting cost and licence

Cloudflare Pages' free tier **explicitly permits commercial use** (recorded in
`docs/research/production_deployment.md`). The static path adds £0 and no new
dependency. Vercel's Hobby tier does not permit commercial use, which is why
the site already moved to Pages.

## 7. What the browser cannot do, and must therefore refuse

- **Share counts and mixed currencies.** Converting them to weights needs a
  price per line, and a mutual fund has no price in any free feed. Refused with
  instructions rather than guessed.
- **AMFI's scheme master.** 14,366 schemes; far too large to send to a visitor.
  The server uses it to recover a fund's SEBI category by exact name or ISIN.
  Without it, a fund whose name does not itself name its index cannot be
  resolved — so the browser **refuses where the server would unpack**.

That asymmetry is acceptable in exactly one direction. A refusal is visible to
the reader and recoverable. Unpacking into the wrong index produces a plausible
table of real companies with real weights summing correctly, about a fund that
holds none of them — see `LESSONS.md` F-19, where that happened in the Python
engine and nothing on the page looked wrong.

## 8. The risk this creates, and the measurement that contains it

A second implementation of one calculation in a second language is the most
reliable way to acquire two answers to one question. The containment has to be
mechanical:

- the JS holds **no data** — every table, phrase list and regex is serialised
  out of the Python modules by `scripts/export_lookthrough_data.py`;
- parity is asserted over **3,446 real scheme names** and **17 whole books**,
  compared at `1e-12`, on every run.

Verified by mutation: a `1 + 1e-9` factor on one multiplication fails the
golden test, and so does rendering an absent `listed` as `0` instead of null.

**This paid for itself before shipping.** The harness found five defects in the
already-shipped Python engine, including a government-bond fund expanded into
fifty equities and 20 of 26 ETF tickers classified as individual companies.
None was visible on the page, because every total still summed to 100%.

## 9. Accuracy of the weights, measured

NSE publishes index membership but not free-float weights, so the weights are
computed from free-float market cap — the same basis NSE uses.

Measured against INDY, an ETF tracking NIFTY 50, on 2026-10-04:

```
mean absolute error     0.37 pp
worst                   0.62 pp   (ICICIBANK)
constituents compared      9      (INDY publishes only its top 9)
```

Caveats that travel with the number, rendered beside it on the page:
free-float share counts are Yahoo's estimate rather than NSE's quarterly
Investible Weight Factor; index capping rules are not applied; and the
comparison covers the top of the index, not the tail.

## 10. Coverage gap, stated

NIFTY 500 exports **497** of 501 names. Four (`BAGMANE`, `BHARATCOAL` among
them) have no Yahoo market cap, so their index weight cannot be computed. They
are listed in `excluded` with a reason and the remaining weights renormalised,
which is why the page's figure for a Nifty 500 fund is a share of 497 rather
than 501.

## Related

- [[product_spec]] — why the look-through is step 1 and picking cost is demoted
- [[production_deployment]] — the Pages vs VM comparison and the licence facts
- [[GET_THE_SERVER_UP]] — the 26 dark days
- [[what_tirramind_is_for]] — why the prediction premise stayed unproven
