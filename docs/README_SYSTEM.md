---
title: "TirraMind — System README (front door)"
tags:
  - doc/wiki
  - topic/architecture
  - topic/product
  - topic/ground-truth
  - status/active
date: 2026-08-27
---

# TirraMind — System README

The front door. Read this first if you did not build this system.

This is a lab notebook, not a brochure. Every number below was measured, and the
date and the machine it was measured on are stated. Where the result is
disappointing, it is written down as a disappointing result.

**Companion document:** `docs/API.md` — every HTTP route, gate, parameter and
error code.

---

## 1. What TirraMind is

TirraMind collects data from ~60 free public sources, turns it into features,
builds a graph of entities and their relationships, and tries to predict which
instruments will outperform.

Plain version: it reads government contract awards, shipping positions, futures
positioning reports, court filings, satellite fire data, prediction markets and
similar public feeds. It looks for signals in them before those signals show up
in prices.

The bet is stated in `README.md`: *math on common data is commoditized; math on
unique data is the moat.* The system does not try to have a better model than a
hedge fund. It tries to have data a hedge fund is not bothering to collect.

There is a second, separate thing in the repo: a **paid API business** with four
subscription tiers that sells access to the collected data, the entity graph,
the scheduler history, and a daily brief. Section 4 covers what those tiers
actually deliver today.

**The honest summary:** the collection machinery works. The API works. The
predictive edge has not been validated. See section 9.

---

## 2. The 7-layer architecture

Code lives in exactly one layer. This is the single strongest rule in the repo.

| Layer | Directory | Responsibility |
|---|---|---|
| 1 · Surveillance | `agent/tools/` | Fetch data. HTTP clients, free APIs. No feature logic. |
| 2 · Feature Engineering | `agent/quant/` | BOCPD, HMM, spectral, scoring. Stateless math. |
| 3 · World Model | `agent/models/` | Bayesian graphs, causal inference, the GNN. |
| 4 · Signal Fusion | `agent/fusion/` | Kalman filters, particle filters, multi-source combination. |
| 5 · RL Policy | `agent/learning/` | Model-based RL, bandit arms, portfolio optimization. |
| 6 · Adversarial | `agent/adversarial/` | Edge decay, manipulation resistance, robustness. |
| 7 · LLM Support | `agent/reasoning/` | Text parsing and narration. **The LLM does not decide.** |

Two supporting pieces sit outside the stack:

- `agent/pipeline/` — the deterministic DAG scheduler. Fetch, feature, model,
  signal. No LLM anywhere in it.
- `agent/payments/`, `agent/brief_server.py`, `agent/delivery/` — the commercial
  surface: Paddle webhooks, subscriber keys, the HTTP API.

### Why placement is load-bearing

Layer placement is not filing. It is a set of enforceable invariants:

1. **Layer 1 must be stateless and side-effect-light.** When it is not, other
   layers pay. A real, current example: importing
   `agent.pipeline.dags.whale_tracking` runs a live DNS-over-HTTPS probe at
   module import time, and importing `agent.pipeline.dags.adversarial_scan`
   transitively pulls in PyTorch. Both were caught on 2026-08-27 as flaky test
   timeouts in `agent/brief_server.py`, a file whose own docstring promises it
   is a "minimal, dependency-light consumer surface". The server now imports
   `daily_collection` alone to avoid both.
2. **Layer 7 must not decide.** If the LLM's output can change a trade, the
   system's behaviour is no longer reproducible and no backtest means anything.
3. **The API can only expose what a layer declares as public.** The
   `/api/v1/data` source catalog is derived from the DAG's own `table_name`
   declarations rather than a hand-written list, because a hand-written list is
   how internal model telemetry leaked to customers once already (section 7).

Practical rule before writing code: name the layer first. If a change touches
more than one layer, you are probably collapsing a boundary — refactor instead.
`.claude/CLAUDE.md` section 1 and `RulesForAI.md` have the full discipline.

---

## 3. Where the data actually lives

One SQLite file, `.tirra_pipeline/pipeline.db`. The tables that matter:

| Table | What it holds |
|---|---|
| `pipeline_data` | One JSON blob row per source fetch. The Data Platform tier reads this. |
| `entities` | Canonical entities: companies, agencies, vessels, instruments. |
| `entity_links` | Typed, confidence-scored relationships between entities. |
| `entity_observations` | Raw per-entity signal values. **Never exposed by the API.** |
| `dag_runs` | Pipeline execution history. The Scheduler tier reads this. |

`entity_observations` is deliberately unexposed at every tier. It is the raw
model input, and whether any of it should ever be customer-facing is a product
decision nobody has made.

---

## 4. What the four tiers actually deliver today

Prices from `products/brief_subscription/pricing.html`. Availability flags from
the same file, `TIER_AVAILABILITY`.

Two sets of numbers appear below, and they are not interchangeable:

- **Production** — measured on the production database, 2026-08-27.
- **Local dev** — measured by this document's author on
  `.tirra_pipeline/pipeline.db` on this machine, 2026-08-27, after three
  `run_chain.py` executions earlier the same day. It is larger than production
  because the pipeline was actually run here.

### $500/mo — Data Platform · **GATED, "coming soon"**

Sells API query access to collected source data.

- Production `pipeline_data`: **194 rows across 60 sources.** Many were last
  written in April.
- Local dev after three chain runs: 375 rows across 66 sources.
- The API's source allowlist exposes **51 source names**.

The tier is marketed as "47 daily-refreshed sources". 194 rows is not that.
`TIER_AVAILABILITY.data = false` for exactly this reason — the storefront shows
"Coming soon" and checkout is blocked. That is the correct call.

Open problem: 66 distinct sources exist locally but only 51 are allowlisted. Up
to 15 legitimate sources would return `400 unknown source`. This must be
reconciled before the tier is ungated.

### $300/mo — Entity Graph · **LIVE**

Sells read access to the entity/relationship graph.

- Production: **5,628 entities, 16,870 links** (~36h old at time of check).
- Local dev, 2026-08-27: 6,172 entities, 17,581 links.

This is real data, and it is the same graph
`agent/models/gnn/graph_builder.py` trains on. The tier is genuinely deliverable
and is the only one currently enabled.

One caveat a reader will otherwise trip over: the `/evidence/*` routes are a
**different, much smaller dataset** — a standalone document store using
regex-based entity extraction over manually POSTed documents. Measured locally
on 2026-08-27: **13 documents, 1,565 mentions, 5,346 links.** (The comment in
`agent/brief_server.py` still says "5 documents, ~155 entity strings"; that
comment is stale.) Both datasets live under the same tier gate. Each carries a
`dataset_scope` block in its response saying which one you are looking at. Do
not remove those blocks. Background:
`docs/research/entity_graph_tier_mismatch.md`.

### $50/mo — Scheduler · **LIVE**

Sells read-only visibility into DAG run history.

- Production `dag_runs`: **65 rows.** Local dev: 69.
- Production `daily_collection`: **1 completed run ever** (2026-04-19). 5
  failed, last on 2026-08-25. 2 stuck in `running` that never finished.
- The systemd timers that were supposed to drive this **do not exist on the
  server.** The unit files are in `deploy/systemd/`; they were never installed.

So the tier returns real rows, but those rows mostly describe a pipeline that
was not running. The two stuck runs were auto-reaped on 2026-08-27 by the new
heartbeat logic in `agent/pipeline/store.py`.

### $19/mo — Opportunity Brief · **GATED, "coming soon"**

Sells a recurring intelligence brief.

- Production `total_deliveries`: **0. No brief has ever been generated.**
- Local dev `.tirra_delivery`: 8 deliveries, latest 5 contracts and 8 anomalies.
- Production `subscribers.json` does not exist. **Zero subscribers, ever.**

`TIER_AVAILABILITY.brief = false`. Selling a recurring deliverable that has been
produced zero times in production would be fraud, so it is gated.

### Payment state

Paddle is live with four price IDs. **Live domain approval is still pending**,
so checkout cannot load on the domain. `tirramind.com` currently serves a
placeholder. Nobody can be charged right now.

`tirramind.com` has no MX records. `support@tirramind.com` bounces. The contact
form therefore writes to a file instead of sending mail (section 6).

---

## 5. The customer path

Three steps, end to end.

### Step 1 — Checkout

The customer picks a tier on `products/brief_subscription/pricing.html`. Paddle
opens an overlay checkout. On success, the page's `eventCallback` captures
`transaction_id` and redirects to `https://tirramind.com/welcome?txn=<id>`.
`successUrl` is set as a fallback in case the callback does not fire.

In parallel, Paddle POSTs a signed webhook to `/webhook`. The handler verifies
the signature, checks the event id against a disk-backed replay ledger, mints an
API key of the form `tirra_<random>`, resolves the tier from the price id, and
writes the subscriber record.

### Step 2 — Claim the key

The key is never emailed — there is no mail service. The customer collects it
themselves.

`welcome.html` polls `GET /api/v1/claim?txn=<id>`. That route is the only
unauthenticated route in the API, because at this point in the flow no key
exists yet. It calls Paddle's own `GET /transactions/{id}` — it never takes the
caller's word that a payment completed — then returns the key once the webhook
has landed.

Because the webhook and the redirect race each other, the route returns
`202 pending` until the subscriber record exists, and the page polls.

Once delivered, the key is shown once with a copy button. Re-claiming returns
`already_claimed` without the key, inside a 15-minute idempotency window and a
5-claim cap so a leaked claim URL cannot be replayed to re-extract the key.

**Two verified defects in this path, both unfixed as of 2026-08-27:**

1. **The poll loop trips its own rate limit.** `welcome.html` issues 15 requests
   for one `txn` over ~118 seconds. The server allows 8 per `txn` per 600
   seconds. Request 9, at roughly t=58s, gets `429 rate_limited`. `welcome.html`
   has no branch for that status, so it falls through to the generic terminal
   error and tells a paying customer setup failed. This fires in exactly the
   case `pending` was built to tolerate: a webhook arriving more than ~58s after
   checkout.
2. **CORS preflight is not handled.** `agent/brief_server.py` implements
   `do_GET` and `do_POST` only. There is no `do_OPTIONS`. Verified live on
   2026-08-27: `OPTIONS /api/v1/contact` returns **501**. The contact form posts
   cross-origin with `Content-Type: application/json`, which requires a
   preflight, so the contact form cannot work from a browser as written.

### Step 3 — First API call

Pass the key in the `X-Brief-Key` header:

```bash
curl -H "X-Brief-Key: tirra_..." https://api.tirramind.com/brief.json
```

`?key=` in the query string still works for backward compatibility, but it logs
a deprecation warning on every use, because query strings land in access logs
and `Referer` headers in cleartext. Setting `TIRRA_REJECT_QUERY_KEYS=1` turns it
into a hard 400. See `docs/API.md` for every route.

---

## 6. Running it

Use the venv. Bare `python3` has no pytest, and on this machine bare `python`
does not resolve at all.

### Tests

```bash
.venv/bin/python -m pytest tests/ -q          # full suite, ~15 min
.venv/bin/python -m pytest tests/test_brief_server.py -q
make test-fast                                 # skips slow/live markers
```

Full-suite result, measured 2026-08-27: **10 failed, 10,819 passed, 14 skipped
in 886s.** All 10 failures are two real production bugs, both listed in section
8. They are not flaky and they are not new.

### The pipeline

Do not rely on cron. The DAGs declare cron schedules, but those only fire if a
long-running `PipelineScheduler.start()` process is alive, and nothing in
production ever started one. Eight of eleven DAGs had never executed once.

Run the chain explicitly, in dependency order:

```bash
.venv/bin/python scripts/run_chain.py                        # full chain
.venv/bin/python scripts/run_chain.py --only daily_collection
.venv/bin/python scripts/run_chain.py --skip-collection      # downstream only
.venv/bin/python scripts/run_chain.py --dry-run
```

**Verify with row counts, not exit status.** "Completed" is not the same claim
as "wrote rows". Measured on 2026-08-27 across three consecutive
`--only daily_collection` runs: `pipeline_data` 195 → 240 → 285 → 330,
`entity_observations` 367,150 → 369,965 → 372,811. A run that reports success
with a flat row count has not done anything.

```bash
sqlite3 .tirra_pipeline/pipeline.db "SELECT COUNT(*) FROM pipeline_data;"
```

Current honest state of `daily_collection`: 46 of 56 nodes complete and store,
8 skip with `Missing credential:` (FRED, EIA and NASA FIRMS keys are empty in
production), and 2 genuinely fail — `lobbying` (LDA API returns HTTP 403) and
`patent_filings` (USPTO). The run correctly reports `failed` because of those
two, not because of the credential gaps.

### The API server

```bash
.venv/bin/python -m agent.brief_server --port 8777
```

With no `TIRRA_SUB_KEYS` and no `TIRRA_PADDLE_WEBHOOK_SECRET` configured, the
server runs open in dev mode. **Set `TIRRA_REQUIRE_AUTH=1` in production.**
Without it, an env file whose auth variables are present but empty — which is
exactly how `deploy/env.production.example` ships — authorizes anonymous callers
for every paid tier, silently, with nothing in the logs.

### Deploy

Full runbook: `docs/runbooks/production_deploy.md`. Target is a €~30/month
Hetzner VM behind Cloudflare, with Caddy for TLS and R2 for backups.
`deploy/systemd/` holds the unit files.

Two things to know before trusting a deploy:

- The systemd timers (`tirra-chain.timer`, `tirra-backup.timer`,
  `tirra-collect.timer`, `tirra-brief.timer`) exist as files in the repo but
  **are not installed on the current server.** That is why the pipeline stopped.
- The runbook's own verification checklist insists on per-DAG row deltas rather
  than `status=completed`. Follow it literally.

---

## 7. The documented failure modes

`LESSONS.md` is the fuckup log. Read it before every session. Four entries
matter most.

### F-01 · Embedding collapse

Entity-identity contrastive loss made all 89 instrument embeddings identical
(cosine similarity ≈ 1.0). IC stuck at 0.00. The model discovered that "make
everything the same" was the trivially optimal solution, because with one window
per instrument there were no true negatives.

Fixed by replacing it with cross-sectional ranking contrastive loss (InfoNCE;
positives are same-return-decile peers, negatives are opposite deciles).

**Check before training:** `torch.std(emb, dim=0).mean() > 0.1`.

### F-02 · GNN bypass

Return loss converged, IC improved, and the GNN contributed nothing. A
conditional chain in the trainer routed to a raw-price-only head whenever raw
features existed, because `use_concat_head` defaulted to False. The GNN was
forward-passed for the contrastive loss but its embeddings never entered the
return prediction path.

**Check:** print the active branch name at training start. An embedding that is
detached from the loss produces no gradient and learns nothing, however healthy
the loss curve looks.

### F-03 · History column shift

`history["return"]` had 65 entries while `history["total"]` had 75, because the
return loss was added at epoch 11. On checkpoint resume the shorter array was
appended without alignment, so the printed table showed epoch 75's value in the
epoch 65 row.

**Check:** every history array must be the same length as `history["total"]` at
all times. Front-pad with NaN on load when adding a loss component mid-training.

### F-04 · Data leakage in evaluation

Ridge regression showed IC = +0.48, ICIR = +2.1. It was fake. The benchmark used
all available data for both train and test, with no walk-forward split. Corrected
walk-forward: **IC = +0.07, ICIR = +0.40.**

**Check:** any evaluation producing IC > 0.15 is suspect until the split is
verified. Real single-factor IC is typically 0.03–0.15. Always print train and
test date ranges.

### Why they recur

These four are not independent mistakes. They are four instances of one failure
mode: **code that reports success while doing nothing.**

- F-01: loss converged to zero, which looked like success and was degeneracy.
- F-02: metrics improved while the component under test was disconnected.
- F-03: a display path read misaligned arrays and printed plausible numbers.
- F-04: a metric was computed correctly on the wrong split.

None of them threw an exception. None of them turned a dashboard red. That is
what makes them recur: the default signal a developer trusts — green tests, a
falling loss, a completed run — is precisely the signal these bugs leave intact.

The counter-practice throughout this repo is to **prove a claim with a row count
or a command output, never a status.** Recent hardening applies exactly that:
`agent/pipeline/executor.py` now marks a node `failed` when its storage call
raises (it previously logged a warning and left the node `completed`), and a run
where every node completed but nothing was persisted is downgraded to `failed`.

---

## 8. Known open defects

Confirmed on 2026-08-27, reproduced directly, unfixed.

| Defect | Location | Effect |
|---|---|---|
| `FineTuner.__init__` reads `self.config.zero_price_feats`, but `__init__` never sets `self.config` | `agent/models/gnn/trainer.py:4240` | Every `FineTuner()` construction raises `AttributeError`. Outcome fine-tuning is completely broken. 8 of the 10 failing tests. Introduced in commit 53d7543. |
| AWOS watchers hardcode `["python", ...]` instead of `[sys.executable, ...]` | `agent/awos/watchers/drift.py:28`, `obsidian.py:29` | `python` does not resolve on this machine. `FileNotFoundError` → return code 127, which the watchers treat identically to "ran clean, no findings". Both watchers are silently inert. 2 of the 10 failing tests. |
| Claim poll loop exceeds its own rate limit | `agent/brief_server.py` ↔ `welcome.html` | Paying customer sees a false "setup failed" (section 5). |
| No `do_OPTIONS` handler | `agent/brief_server.py` | CORS preflight returns 501; the contact form cannot work from a browser. |
| Source allowlist has 51 entries, DB has 66 distinct sources | `agent/brief_server.py`, `pipeline_data` | Up to 15 legitimate sources would 400 if the Data Platform tier were ungated. |
| `POST /api/v1/rotate-key` designed but not routed | `agent/brief_server.py` | `SubscriberStore.rotate_key_for_api_key()` exists; no HTTP route reaches it. A customer who loses a key has no self-service path. |
| `_persist_entities` swallows exceptions | `agent/tools/bankruptcy_court.py`, `agent/tools/defi_flows.py` | Prints "(non-fatal)" and continues; the node still reports `completed` while entity data may be dropped. Same class as F-01..F-04. |

`trainer.py` is a critical file under `.claude/CLAUDE.md` section 9 — it requires
research, spec and a targeted test before any change. That is why the
`FineTuner` bug is documented here rather than patched in passing.

---

## 9. The real open questions

### Is there any predictive edge? Unvalidated.

This is the question the entire system exists to answer, and the honest answer
today is that it has not been answered.

Measured IC results, from the JSON artifacts in `.tirra_pipeline/`:

**`honest_baseline_audit_full.json`** — walk-forward, purged:

| Strategy | Mean IC | ICIR | t-stat | Folds |
|---|---|---|---|---|
| Momentum-Rank (trivial baseline) | **+0.0741** | +0.321 | 2.00 | 39 |
| GNN-PurgedRanker | +0.0309 | +0.158 | 0.96 | 37 |
| RawPrice-PurgedRanker | −0.0325 | −0.194 | −1.15 | 35 |

**In the honest audit, a trivial momentum ranker beat the GNN.** The GNN's
t-stat of 0.96 is not distinguishable from zero. Fold counts differ between
strategies, so this is not a perfectly matched comparison — but it is not a
result that supports the GNN either.

**`ic_results_phase50.json`** — a specific checkpoint, more favourable:

| Head | Mean IC | ICIR | t-stat | Folds |
|---|---|---|---|---|
| GNN-EmbNorm | +0.0468 | +0.354 | 2.21 | 39 |
| GNN-PurgedRanker | +0.0418 | +0.359 | 2.12 | 35 |
| GNN-ValueHead | −0.0398 | −0.301 | −1.88 | 39 |
| GNN-ReturnHead | −0.0339 | −0.252 | −1.58 | 39 |

Two heads of the same model point in opposite directions with similar
magnitudes. That is what selecting the best of several heads on the same data
looks like, and F-04 is the standing reminder of what happens when a good-looking
number is not challenged.

What is missing before any of this counts:

- **No live trading P&L.** No forward test. Every number above is historical.
- **No out-of-sample confirmation** of the phase50 result on data collected after
  that checkpoint was trained.
- **The best result is model selection.** Picking the best-performing head after
  the fact is not evidence.
- **A mean IC near +0.04 is inside the noise band** for the fold counts involved.

### Is there a business? Not yet demonstrated.

Zero subscribers. Zero briefs generated. Zero revenue. Two of four tiers are
gated because they cannot deliver what they advertise. The site is offline and
Paddle's domain approval is pending. Nothing has been tested against a customer
who paid.

### Does the pipeline stay alive unattended?

Unknown. Production `daily_collection` has one successful run, ever, in April.
The timers are not installed. The 2026-08-27 heartbeat and zero-rows work makes
failure *visible* when it happens; it does not make the pipeline run.

### Open questions that need an owner

1. Reconcile the 51-source allowlist against the 66 sources in `pipeline_data`
   before ungating the Data Platform tier.
2. Reconcile the claim rate limiter against `welcome.html`'s poll cadence, or
   add a `rate_limited` branch that honours `retry_after_s`.
3. Add `do_OPTIONS`, or the contact form stays dead.
4. Fix `FineTuner.__init__` under a spec, per critical-file discipline.
5. Fix the AWOS watchers to use `sys.executable`.
6. Install the systemd timers, then verify with row deltas over several days.
7. Decide whether the Data Platform tier is a product at all at 194 rows, or
   whether it needs months of collection first.

---

## Related

- `README.md` — project overview and quick start
- `docs/API.md` — full HTTP route reference
- `LESSONS.md` — the fuckup log, F-01 onward
- `RulesForAI.md` — architecture and workflow discipline
- `.claude/CLAUDE.md` — operating rules for agents on this codebase
- `docs/runbooks/production_deploy.md` — deployment runbook
- `docs/research/deep_intelligence_roadmap.md` — pricing and cost strategy
- `docs/research/entity_graph_tier_mismatch.md` — the two-datasets problem
