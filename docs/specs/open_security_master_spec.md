---
title: "Spec: Open Security Master — 30-day public demand probe"
tags:
  - doc/spec
  - topic/architecture
  - status/current
---

# Spec: Open Security Master — 30-day public demand probe

Research: `docs/research/open_security_master.md`
Task: `tasks/active/open_security_master.md`
Name: **tirramind** (repo `savabs/tirramind-universe`, PyPI `tirramind`). Contact for SEC User-Agent: owner email, set as `TIRRAMIND_CONTACT`.
Local checkout: `~/Projects/tirramind-universe`.

## Goal

Ship, in public, in 30 days, at $0 cost:
1. A daily-committed, content-addressed diff of the US listed universe with
   delisting causes, reconciled from SEC filings only.
2. A weekly Form 144 → Form 4 execution page with a published accuracy table.
3. One write-up that benchmarks vendor delisting lists against the ledger.

Then measure whether strangers arrive. Nothing else gets built until they do.

## Success / kill (fixed now)

Day 30, any ONE of: ≥300 GitHub stars · ≥500 PyPI installs/week (excluding
CI) · ≥500 parquet downloads · ≥10 inbound requests for full history ·
≥10 paying at $9–19/mo.
**None of the above by day 30 → stop.** Write the checkpoint, leave the
daily job running (it costs nothing and compounds), and do not build the
bitemporal store on top of it.

## Non-goals (v1)
- Prices, adjusted history, splits/dividends as a complete corporate-action
  feed. Splits are recorded only when structurally stated.
- FINRA-sourced fields. OCC memos. Anything non-SEC.
- Accounts, teams, SSO, SLA. Enterprise anything.
- Backfilling `company_tickers.json` daily before day 0 (Wayback is monthly;
  use it as-is and label it).

## Constraints
- SEC fair access: ≤10 req/s, `User-Agent: tirramind/<ver> (<contact email>)`.
  **The declared email must have a mailbox** — `support@tirramind.com` has
  no MX record; use one that exists.
- Every job that writes rows prints row counts; a run that captures zero
  filings on a business day fails loudly (queue_attrition `ci_verify.py`
  pattern). Green-with-zero-rows is a known failure class here.
- Append-only, content-addressed. Nothing is ever edited in place; a
  correction is a new row with `supersedes`.
- Apache-2.0. Data outputs CC-BY-4.0 with attribution "derived from SEC
  public filings".

---

## Steps

Each step is independently verifiable. Do them in order.

### Phase A — repo and store (days 1–3)

**A1. Create repo `survivor`.** `src/survivor/`, `data/`, `tests/`,
`pyproject.toml` (hatchling, deps: `requests`, `pandas`, `pyarrow`,
`lxml`), Apache-2.0, README with the one-paragraph promise and nothing
else.
*Verify:* `pip install -e . && python -c "import survivor"`.

**A2. Port the snapshot store.** Copy `queue_attrition/src/snapshot.py`
semantics into `survivor/store.py`: `snapshot(name, df) -> (path|None,
digest, status)` with `index.jsonl`, content hash over canonical CSV,
`new|unchanged|fail` records.
*Verify:* test writes the same frame twice → one file, two index rows.

**A3. SEC client.** `survivor/sec.py`: rate-limited session (token bucket
10/s), declared User-Agent from `TIRRAMIND_CONTACT` env var (refuse to start
if unset), retry on 429/503 with backoff, on-disk response cache keyed by
URL for the backfill.
*Verify:* 50 sequential requests complete without a 403; log shows ≤10/s.

### Phase B — the listing ledger (days 3–10)

**B1. Ticker-map snapshot.** Daily fetch of `company_tickers.json` and
`company_tickers_exchange.json` → one canonical frame
`(cik, ticker, name, exchange)` → `store.snapshot("tickers", df)`.
*Verify:* first run writes; second run same day is `unchanged`.

**B2. Ticker-map diff.** `survivor/diff.py`: between any two snapshots
emit events `LISTED`, `DELISTED_FROM_MAP`, `SYMBOL_CHANGED`,
`NAME_CHANGED`, `EXCHANGE_CHANGED` with both sides and both snapshot
digests. Diffs are derived, not stored; they are recomputed from snapshots.
*Verify:* synthetic two-snapshot fixture produces exactly the expected
events; a real diff of two Wayback captures runs clean.

**B3. Wayback backfill of the ticker map.** Pull every distinct capture of
`company_tickers.json` from the Wayback CDX API (2017→day 0), snapshot
each with `captured_at` = Wayback timestamp and `source="wayback"`.
*Verify:* index shows ≥80 captures, monotone timestamps, and the diff
across the full sequence yields plausible counts (order of hundreds of
delistings/year).

**B4. Filing collectors.** EFTS full-text search by form type and date
range, then daily-index files for backfill 2015→now, for: Form 25 (all
variants), Form 15 (all variants), 8-K with Items 1.03, 3.01, 5.03.
Store one row per filing: `(accession, form, items, filer_cik,
subject_cik, filed_at, accepted_at, primary_doc_url)`. For Form 25 filed
by an exchange, `subject_cik` comes from the header, not the filer.
*Verify:* one known delisting (pick any 2024 Form 25) round-trips with the
correct subject CIK; a business day returns >0 filings or the job fails.

**B5. Delisting-cause reconciliation.** `survivor/reconcile.py`: for each
`DELISTED_FROM_MAP` event, look back 120 days for Form 25 / 8-K 3.01 /
8-K 1.03 / Form 15 on the same CIK, plus a RECAP bankruptcy docket, and
assign a cause from a fixed taxonomy:
`EXCHANGE_DELISTING | VOLUNTARY_DELISTING | DEREGISTRATION | BANKRUPTCY |
MERGER_ACQUISITION | UNKNOWN`. Record every contributing accession.
`UNKNOWN` is a valid, reported outcome — never guessed.
*Verify:* on 2024 events, `UNKNOWN` share is printed; a hand-checked sample
of 30 has ≥25 correct causes. The sample and its outcomes are committed.

**B6. Publish.** `data/events.parquet`, `data/delistings.parquet`,
`data/tickers_latest.parquet`, regenerated daily by the job, plus a
`data/README.md` stating source, licence, and the `UNKNOWN` rate.
*Verify:* files load with `pd.read_parquet`; schema documented; sizes
under GitHub's limits.

### Phase C — the insider ledger (days 8–16)

**C1. Extract parsers.** Copy `_parse_form144_xml` and `_parse_form4_xml`
from tirramind into `survivor/forms.py`, remove PipelineStore coupling,
and **extend Form 4 parsing to transaction codes `S`, `F`, `M`, `P`** with
`(rptOwnerCik, issuerCik, code, shares, price, transaction_date,
accession)`. Add the filer CIK from the EFTS hit to each Form 144 row.
*Verify:* unit tests on three committed XML fixtures per form; a sale
(`S`) that the old parser dropped is now returned.

**C2. Collect.** Daily: all Form 144 and Form 4 filed that day. Backfill:
Form 144 from 2023-04-01 (first structured XML), Form 4 from 2023-01-01.
Snapshot both as append-only tables keyed by accession.
*Verify:* row counts per month printed; zero-row business day fails.

**C3. Link.** `survivor/link.py`: for each Form 144 `(filer_cik,
issuer_cik, filed_at, shares_planned, approx_sale_date)`, candidate Form 4
rows = same `issuer_cik`, same `rptOwnerCik` (fallback: normalised name
match, flagged), code `S`, `transaction_date ∈ [filed_at − 3d, filed_at +
90d]`. Aggregate shares sold; emit `executed_fraction`, `days_to_first_sale`,
`match_method ∈ {cik, name, none}`.
*Verify:* on one month, print the share of 144s matched by CIK vs name vs
none. If CIK match < 70%, the research doc's open question is answered
"name matching required" and the flag is surfaced in the data.

**C4. Accuracy table.** Using the tirramind coverage engine: P(any
execution within 30/60/90 days), executed-fraction distribution, by
relationship (officer/director/10%), with block-bootstrap CIs clustered
by issuer-week and **effective n printed beside nominal n**. Also the
false-positive check: 144s with no Form 4 by day 90 vs late filings.
*Verify:* table regenerates from parquet in one command; effective n <
nominal n; CIs widen when the cluster size is increased.

**C5. Weekly page.** Static HTML generated by the job: "Form 144s filed
this week" (issuer, insider, relationship, shares, $) and "prior plans
that executed this week" with the accuracy table and the link method.
Hosted on GitHub Pages from the repo. No JS frameworks.
*Verify:* page builds in CI, links resolve, table matches parquet.

### Phase D — distribution (days 14–30)

**D1. Daily job.** GitHub Actions cron (reuse `queue_attrition`
`snapshot.yml`: `concurrency`, `contents: write`, rebase-retry, `ci_verify`
gate). Runs B1, B4, C2, B5, C3, C4, C5, commits `data/` and `docs/`.
*Verify:* two consecutive scheduled runs commit; a forced zero-row run fails
red.

**D2. PyPI client.** `survivor-master`: `survivor.load("delistings")`,
`survivor.events(since=)`, `survivor.form144(since=)` reading the
published parquet by raw GitHub URL with local cache.
*Verify:* fresh venv, `pip install survivor-master`, one call returns rows.

**D3. The write-up.** "Every US delisting since 2017, reconciled from SEC
filings — and where the vendor lists disagree." Method, the `UNKNOWN`
rate, the hand-checked sample, the disagreement ledger against at least
one free list (Wikipedia's or a broker's public delisted list), the 144→4
table with effective n. Posted to the repo README, HN, r/algotrading,
QuantStack/Nuclear Phynance-type forums. No marketing words.
*Verify:* it is posted. Record the URLs and the date in the task file.

**D4. Paid tier — gated.** Only after D3 has been live 7 days AND any
single success signal is trending (≥100 stars or ≥5 inbound). Full history
+ API key + email alerts at $9–19/mo via the existing Paddle account.
Before wiring it: fix the non-atomic `SubscriberStore._save` in
`brief_subscription/handler.py` (`.tmp` + `os.replace`) and the default
tier fallback. Refunds: manual, by email, within 24h — stated on the page.
*Verify:* sandbox checkout → key → authenticated parquet download → refund
path documented.

**D5. Day-30 checkpoint.** `docs/memory/checkpoint_2026-10-06_survivor.md`
with every metric in the success bar, actual vs threshold, and the
decision. If go: next spec is the bitemporal store. If stop: the job keeps
running, nothing else is built.

---

## Golden tests (must pass before D3)
- Two-snapshot diff fixture → exact event set.
- Known 2024 Form 25 → correct subject CIK and cause.
- Three Form 4 fixtures incl. code `S` → parsed rows.
- Three Form 144 fixtures → parsed rows with filer CIK.
- Accuracy table: effective n < nominal n; re-running is deterministic.
- Zero-row business day → non-zero exit.

## Cost
$0. SEC, Wayback, GitHub Actions, GitHub Pages, PyPI are free. Paddle only
charges on sales. The only owner-side input is a real contact email.
