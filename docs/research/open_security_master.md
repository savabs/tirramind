---
title: "Research: Open Security Master (tirramind)"
tags:
  - doc/research
  - topic/api
  - status/current
---

# Research: Open Security Master (tirramind)

Date: 2026-09-06. Feeds `docs/specs/open_security_master_spec.md`.

## Where this came from

Two agent scans (68 agents, ~5.6M tokens) — `ultraresearch_venture_scan_2026-09-06.md`
and `narrow_scan_relaxed_2026-09-06.md` — converged on one structural
answer: an EDGAR-derived, survivorship-bias-free US security master is the
one pipeline that feeds three card-paid products (master → backtester input →
options-lab input), has a licence that is clean end to end, and a channel
that is proven (PyPI/GitHub → card, per sec-api.io, edgartools, Norgate,
Sharadar). Its verified ceiling alone is ~$300k/yr; its value is as the spine.

This is a **30-day public demand probe**, not a product build. The kill
criterion is fixed in the spec before a line is written.

## Facts (verified by the scans; re-check anything marked ~)

**Licence**
- SEC EDGAR content is public domain: "may be copied or further distributed
  ... without the SEC's permission" (sec.gov/privacy#dissemination).
  "EDGAR" is a trademark — must not appear in the product name.
- SEC fair-access: ≤10 requests/second, declared `User-Agent` with a
  contact email. Undeclared agents are blocked.
- **FINRA Terms of Use prohibit building databases from site content and
  restrict to non-commercial use.** The existing `finra_short_volume`
  collector and FINRA Daily List are excluded from this product.
- OCC option-adjustment memos: terms unverified (403). Excluded until read.
- CourtListener/RECAP: fine for delisting-cause labels (already collected).

**Sources (all sec.gov)**
| Source | What it gives | History |
|---|---|---|
| `files/company_tickers.json`, `company_tickers_exchange.json` | current ticker↔CIK↔exchange map | current only; Wayback holds ~96 monthly snapshots since 2017 |
| Form 25 / 25-NSE | removal from listing (exchange- or issuer-filed) | full-text search 2001+; daily-index files |
| Form 15 / 15-12G / 15-15D | deregistration | same |
| 8-K Item 3.01 | notice of delisting / failure to satisfy listing rule | same |
| 8-K Item 1.03 | bankruptcy | same |
| 8-K Item 5.03 | charter amendment (reverse splits, name changes) | same |
| `submissions/CIK##########.json` | per-issuer filing history, former names, SIC, state | full |
| Forms 3/4/5 XML | insider transactions, incl. code S sales | XML since ~2003 |
| Form 144 XML | planned insider sales | **structured XML only since April 2023** |
| Free SEC "Insider Transactions Data Sets" | Forms 3/4/5 quarterly tables | Form 144 is **not** included — that gap is the product |

**Competitor prices (ceiling)**
- EODHD $19.99/mo bundles delisted tickers + symbol-rename API *with prices*.
- Massive (ex-Polygon) returns `delisted_utc` on paid plans.
- Norgate Platinum $630/yr, delisted securities to 1990, with prices.
- sec-api.io $49–239/mo: Form 144 + Form 4 + ticker map with `isDelisted`
  flag — **no dates, no point-in-time, no 144→4 join.**
- No GitHub repo versions `company_tickers.json` daily (0 results).
- Nobody publishes Form 144 → Form 4 execution linkage or hit-rates.

**What is defensible (post-skeptic)**
1. Daily-resolution diff of the ticker map from day 0 (Wayback is monthly).
2. Delisting *causes* reconciled from Form 25 + 8-K 3.01/1.03 + Form 15 +
   bankruptcy dockets, with a published disagreement ledger vs vendor lists.
3. Form 144 → Form 4 linkage with calibrated execution probabilities and
   block-bootstrap CIs (effective n clustered by issuer-week).

**What is not defensible** — the archive itself. The primary records are
permanent filings with acceptance timestamps; anyone can reconstruct them.
Correctness and the linkage are the product; the daily diff is a convenience
that becomes a mild moat in 2–3 years.

## Existing code to extract (Layer 1, tirramind `agent/tools/`)
- `form144.py` — EFTS search + Form 144 XML parser (`_parse_form144_xml`).
  Persists into PipelineStore; strip that.
- `insider_filings.py` — EFTS search + Form 4 XML parser. **Only extracts
  transaction code `P`.** Needs code `S` (and `F`, `M` for context) for
  the 144→4 join.
- `bankruptcy_court.py`, `creditor_filings.py` — CourtListener/RECAP.
- `dividend_data.py` — 150 lines, likely a paid-source shim; check.
- `queue_attrition/src/snapshot.py` — content-addressed append-only store
  and the self-committing GitHub Actions workflow. Reuse verbatim.
- Calibration/coverage engine (tirramind) — effective n, block bootstrap.

## Open questions (answer during build, not before)
- Does the Form 144 EFTS hit expose the filer CIK reliably enough to join to
  `rptOwnerCik` on Form 4, or is name-matching required? Measure the match
  rate on one month of filings before choosing.
- Form 25 is often filed by the exchange (NYSE/Nasdaq CIKs), not the
  issuer — the subject CIK is in the filing header, not the filer field.
- Reverse splits from 8-K 5.03 are text, not structured. v1 records the
  event and the ratio only when an XBRL fact or 8-K exhibit states it.
