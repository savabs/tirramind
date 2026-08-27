---
title: "TirraMind HTTP API Reference"
tags:
  - doc/wiki
  - topic/api
  - topic/product
  - topic/ground-truth
  - status/active
date: 2026-08-27
---

# TirraMind HTTP API Reference

Complete reference for every route in `agent/brief_server.py`, accurate to the
file as it stands on **2026-08-27**.

This document was written by reading the source and by calling a live server on
this machine, not by updating an older document. Where behaviour is surprising
or broken, it is written down as such rather than described the way it ought to
work.

Companion document: `docs/README_SYSTEM.md`.

---

## Running the server

```bash
.venv/bin/python -m agent.brief_server --port 8777 --host 127.0.0.1
```

`--out` selects the delivery directory, default `.tirra_delivery`.

The server is `http.server.ThreadingHTTPServer` — one thread per connection, no
framework. Concurrent request handling is bounded by a semaphore, default 20,
set via `TIRRA_MAX_CONCURRENT_REQUESTS`. Requests beyond the cap queue; they are
not rejected.

Only `GET` and `POST` are implemented. There is **no `do_OPTIONS`** — see
"Known gaps" at the end.

---

## Authentication

### How to send a key

Preferred:

```
X-Brief-Key: tirra_...
```

Deprecated but still accepted:

```
GET /brief.json?key=tirra_...
```

Every use of `?key=` logs a deprecation warning server-side, because a query
string lands in access logs and `Referer` headers in cleartext. Setting
`TIRRA_REJECT_QUERY_KEYS=1` makes it a hard `400` instead.

When a query key is present and rejected, the `X-Brief-Key` header is **not**
substituted in its place. Rejection is an unconditional protocol rule, not a
fallback.

### How a key is authorized

`_authorized_for()` resolves in this order:

1. `TIRRA_REQUIRE_AUTH` is truthy **and** neither `TIRRA_SUB_KEYS` nor
   `TIRRA_PADDLE_WEBHOOK_SECRET` is configured → **deny everything** and log an
   error. This is the fail-closed production setting.
2. Neither is configured and `TIRRA_REQUIRE_AUTH` is unset → **dev mode, fully
   open.** Every gate passes with no key.
3. No key presented → deny.
4. Key matches an entry in the comma-separated `TIRRA_SUB_KEYS` → **allow, all
   tiers.** This is the admin/dev bypass.
5. `TIRRA_PADDLE_WEBHOOK_SECRET` is set → look the key up in `SubscriberStore`.
   The key must be active; if the route names tiers, the subscriber's tier must
   be in that set.

Truthy values for flags: `1`, `true`, `yes`, `on`.

> **Deploy warning.** `deploy/env.production.example` ships auth variables
> present but empty. Without `TIRRA_REQUIRE_AUTH=1`, rule 2 fires and the entire
> paid API serves anonymously, with nothing in the logs. Always set
> `TIRRA_REQUIRE_AUTH=1` in production.

### Tier gates

| Constant | Tiers accepted | Monthly price of those tiers |
|---|---|---|
| `_ENTITY_GRAPH_TIERS` | `entity`, `data`, `scheduler` | $300, $500, $50 |
| `_DATA_PLATFORM_TIERS` | `data`, `scheduler` | $500, $50 |
| `_SCHEDULER_TIERS` | `scheduler` | $50 |
| (any valid key) | any active subscriber | — |

> **Gating inversion, unresolved.** The `scheduler` tier costs $50/month and
> appears in all three sets. A $50 Scheduler subscriber therefore passes the
> gate for the $500 Data Platform and the $300 Entity Graph. The source comment
> justifies this as "they paid for more surface", which is true of `data` but
> not of `scheduler`. Verify this is intended before ungating the Data Platform
> tier.

### Usage metering

`_log_usage()` records the caller's key, endpoint and tier for:
`/evidence/*` (read routes), `/api/v1/entity-graph/*`, `/api/v1/sources`,
`/api/v1/data`, `/api/v1/dag/runs`, `/brief`, `/brief.json`, `/brief.md`.

It is **not** recorded for `/api/v1/usage`, `/status`, `/buy`, `/`,
`/api/v1/claim`, `/api/v1/contact`, `/webhook`, or `/evidence/ingest`.

Metering is best-effort and never raises. A metering failure logs a warning and
the request proceeds.

---

## Response conventions

- Success bodies are JSON with `Content-Type: application/json` and, on most
  routes, an `"ok": true` field.
- **Error format is not uniform.** Tier-gate rejections (`403`) and brief-missing
  cases (`404`) return `text/plain`. Validation errors return JSON. A client must
  branch on the status code, not on the body shape.
- Every response carries `Access-Control-Allow-Origin`, value from
  `TIRRA_CORS_ORIGIN`, default `*`. `/api/v1/claim` overrides it with
  `TIRRA_WEB_ORIGIN`, default `https://tirramind.com`.
- `Server: AWOSBrief/0.1`. The Python patch version is no longer leaked here —
  verified live on 2026-08-27.
- Timestamps in payloads are Unix epoch seconds as floats.

---

# GET routes

Routes are matched in the order below. `/api/v1/claim` is dispatched **before**
key extraction, so it is unaffected by `TIRRA_REJECT_QUERY_KEYS`.

---

## `GET /api/v1/claim`

**Gate:** none. This is the only unauthenticated read route — no key exists yet
at this point in the purchase flow.

Exchanges a Paddle `transaction_id` for the subscriber's API key. It calls
Paddle's `GET /transactions/{id}` itself; it never trusts the caller's claim that
a payment completed. All claim state and idempotency logic lives in
`agent/payments/claim.py`; this route only maps results to HTTP.

**Parameters**

| Name | Required | Notes |
|---|---|---|
| `txn` | yes | Must match `^txn_[A-Za-z0-9_-]{6,64}$` |

**Rate limits** (in-process, sliding window, per single server process)

| Scope | Limit |
|---|---|
| per `txn` | 8 requests / 600s |
| per source IP | 20 requests / 3600s |

**Responses**

| Status | `status` | Body |
|---|---|---|
| 200 | `claimed` | `{ok, status, api_key, tier, subscription_id}` |
| 200 | `already_claimed` | `{ok, status, subscription_id, message}` — no key |
| 202 | `pending` | `{ok, status, retry_after_s: 3, message}` + `Retry-After: 3` |
| 400 | `bad_request` | missing or malformed `txn` |
| 404 | `unknown_transaction` | Paddle returned 404 for this id |
| 409 | `subscriber_inactive` | subscription exists but is not active |
| 422 | `not_completed` | `{ok, status, transaction_status, message}` |
| 429 | `rate_limited` | `{ok, status, retry_after_s}` + `Retry-After` |
| 502 | `upstream_error` | Paddle unreachable, config error, or any unrecognized status |

`202 pending` is the retry signal. Everything except `202` and `429` is terminal.
The route never returns a bare `500`.

**Verified live, 2026-08-27:**

```
$ curl -D- 'http://127.0.0.1:8913/api/v1/claim'
Access-Control-Allow-Origin: https://tirramind.com
{"ok": false, "status": "bad_request", "message": "missing or malformed txn parameter"}
```

> **Known defect.** `welcome.html` polls this route 15 times over ~118 seconds
> for a single `txn`, which exceeds the 8-per-600s cap at roughly t=58s. The page
> has no `rate_limited` branch and shows a terminal failure instead of retrying.
> See `docs/README_SYSTEM.md` section 5.

---

## `GET /brief` · `GET /brief.json`

**Gate:** any valid key (`_valid_key`). **Metered:** yes.

Returns the latest delivered intelligence brief as JSON, read straight from
`.tirra_delivery/intelligence_brief.json`.

| Status | Content-Type | Body |
|---|---|---|
| 200 | `application/json` | the brief document |
| 403 | `text/plain` | `subscribe required — see /buy` |
| 404 | `text/plain` | `no brief delivered yet` — nothing has ever been delivered |
| 404 | `text/plain` | `brief file missing` — a record exists but the file is gone |

The brief document's top-level keys, from the live payload:
`brief_type`, `contract_opportunities[]`, `live_anomalies[]`.

`contract_opportunities[]` entries carry `award_id`, `recipient`, `agency`,
`description`, `amount_usd`, `start_date`, `p_win`, `expected_value_usd`,
`estimated_bid_cost_usd`, `bucket`, `is_long_tail`.

`live_anomalies[]` entries carry `source`, `observation_type`, `entity_id`,
`field`, `zscore`, `changepoint`, `flagged_ts`, `n_points`, `latest_value`.

> **Production reality:** `total_deliveries` in production is **0**. This route
> returns `404 no brief delivered yet` there. The Brief tier is gated to
> "coming soon" on the storefront for that reason.

---

## `GET /brief.md`

**Gate:** any valid key. **Metered:** yes.

Same content as `/brief.json`, rendered as Markdown.

| Status | Content-Type | Body |
|---|---|---|
| 200 | `text/markdown; charset=utf-8` | the brief |
| 403 | `text/plain` | subscribe required |
| 404 | `text/plain` | `no brief delivered yet` / `brief markdown missing` |

---

## `GET /status`

**Gate:** none. **Metered:** no.

Delivery status. Unauthenticated — safe as a health check.

```json
{
  "out_dir": ".tirra_delivery",
  "total_deliveries": 8,
  "latest": {
    "delivered_at": 1787639422.314678,
    "json_path": ".tirra_delivery/intelligence_brief.json",
    "md_path": ".tirra_delivery/intelligence_brief.md",
    "n_contracts": 5,
    "n_anomalies": 8,
    "duration_ms": 0.69,
    "checksum": "5f0d435ed7c9ebc0"
  }
}
```

`latest` is `null` when nothing has been delivered. Always `200`.

The example above is from this machine. Production returns
`"total_deliveries": 0` and `"latest": null`.

---

## `GET /api/v1/sources`

**Gate:** Data Platform tiers. **Metered:** yes.

Catalog of queryable sources. Use this to discover valid `source` values for
`/api/v1/data` rather than guessing.

```json
{"ok": true, "sources": [
  {"source": "academic_preprints", "rows": 4, "last_fetched_at": 1787834185.710824},
  {"source": "ais_vessel_tracking", "rows": 8, "last_fetched_at": 1787834185.7180102}
]}
```

Sorted by `source`. `403 text/plain` if the tier gate fails.

### The source allowlist

The catalog is filtered through `_external_source_allowlist()`, which is
**derived, not hand-written**: it is every `table_name` declared by a node in
`agent/pipeline/dags/daily_collection.py`, plus three manually-stored Polymarket
sources (`pm_trades`, `pm_resolutions`, `pm_wallet_scores`).

It contains **51 names**, verified 2026-08-27:

```
academic_preprints, ais_vessel_tracking, bankruptcy_court, building_permits,
capital_flows, central_bank_balance, cftc, comtrade, consumer_sentiment,
creditor_filings, defi_flows, disease_surveillance, dns_monitor,
drug_regulatory, earthquake_proximity, electricity_monitor, energy_supply,
finra_short_volume, foia_requests, food_security, form144, gdelt, global_pmi,
gov_contracts, insider_filings, interconnection_queue, internet_infrastructure,
internet_outages, job_postings, labor_disruptions, lobbying, macro_data,
migration_flows, patent_filings, pm_resolutions, pm_trades, pm_wallet_scores,
political_risk, polymarket, polymarket_whales, power_grid, regulatory_gazette,
sanctions_monitor, satellite_activity, sovereign_debt, supply_chain_prices,
transport_throughput, treasury_receipts, weather_alerts, whale_alert,
wikipedia_pageviews
```

**Why an allowlist and not a denylist.** `pipeline_data` is shared by two kinds
of writer: Layer 1 tools storing real external data, and internal DAG stages
storing their own execution telemetry under the node name. Under the previous
denylist, `GET /api/v1/data?source=train_gnn` returned
`{"trained": false, "loss_ewc": 579753920.0, ...}` — a paying customer reading
the model's own untrained-state defect through the API they paid for. Any new
internal stage was customer-queryable the instant someone added it. A node only
gets a `table_name` when its author intends its output to be a named dataset, so
internal stages are now excluded by construction, with no list to maintain.

> **Known gap.** The local `pipeline_data` table holds **66 distinct sources**
> while the allowlist has **51**. Up to 15 legitimate sources return
> `400 unknown source`. Reconcile before ungating the Data Platform tier.

---

## `GET /api/v1/data`

**Gate:** Data Platform tiers. **Metered:** yes.

Query stored rows for one source.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `source` | string | — | **Required.** Must be in the allowlist. |
| `since` | float | none | Epoch seconds. Matches `fetched_at >= since`. |
| `until` | float | none | Epoch seconds. Matches `fetched_at <= until`. |
| `limit` | int | 100 | Clamped to 1–1000. |

`limit`, `since` and `until` are parsed leniently: a malformed value silently
falls back to its default rather than erroring. `?limit=abc` returns `200` with
`limit=100`, verified live 2026-08-27. Only `source` produces a validation error.

**Response**

```json
{"ok": true, "source": "cftc", "rows": [
  {"id": 1, "source": "cftc", "fetched_at": 1787834185.7,
   "params": {...}, "data": {...}}
]}
```

`params` and `data` are decoded from the stored JSON columns. `data` is the raw
tool payload and its shape is source-specific. Rows are ordered by `fetched_at`
descending.

**Errors**

| Status | Body |
|---|---|
| 400 | `{"ok": false, "error": "source required"}` |
| 400 | `{"ok": false, "error": "unknown source 'x' — see /api/v1/sources for valid values"}` |
| 403 | `text/plain`, tier gate |

Verified live:

```
$ curl -H 'X-Brief-Key: ...' '.../api/v1/data?source=train_gnn'
{"ok": false, "error": "unknown source 'train_gnn' — see /api/v1/sources for valid values"}
```

---

## `GET /api/v1/dag/runs`

**Gate:** Scheduler tier only. **Metered:** yes.

Pipeline execution history, newest first.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `dag_name` | string | none | Exact match filter |
| `limit` | int | 20 | Clamped to 1–1000 |

**Response**

```json
{"ok": true, "runs": [
  {"run_id": "69f521c3b194", "dag_name": "daily_collection",
   "started_at": 1787834100.0, "finished_at": 1787834185.7,
   "status": "failed", "trigger": "manual",
   "heartbeat_at": 1787834180.0,
   "node_results": {"fetch_cftc": {"status": "completed"}, ...}}
]}
```

`status` is one of `running`, `completed`, `failed`. `node_results` is the
decoded `node_results_json` column, or `null`.

Per-node `status` values include `completed`, `failed`, and `skipped`. A node
skipped for a missing API key carries `error` beginning `Missing credential:`.
A run can report `failed` while most nodes succeeded — that is intended.

`403 text/plain` if the caller is not on the Scheduler tier.

---

## `GET /api/v1/usage`

**Gate:** any valid key. **Metered:** no — it does not log itself.

The caller's own usage summary, for their own key only.

| Parameter | Type | Default |
|---|---|---|
| `since` | float | none |

```json
{"ok": true, "total": 42, "by_endpoint": {"/brief.json": 30, "/api/v1/data": 12}}
```

`403 text/plain` if the key is invalid.

---

## `GET /api/v1/entity-graph/entities`

**Gate:** Entity Graph tiers. **Metered:** yes.

The **real production entity graph** — the same `entities` table
`agent/models/gnn/graph_builder.py` trains on.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `type` | string | none | Filter on `entity_type` |
| `limit` | int | 100 | Clamped to 1–1000 |
| `offset` | int | 0 | Minimum 0 |

```json
{"ok": true,
 "entities": [{"entity_id": "01146d...", "entity_type": "company",
               "canonical_name": "...", "created_at": 1787000000.0}],
 "count": 100, "total": 6172, "limit": 100, "offset": 0,
 "dataset_scope": {...}}
```

Only four fields are returned per entity. `metadata_json` is **stripped
unconditionally** — it is populated ad hoc by 20+ independent `agent/tools/*`
call sites (transaction hashes, CIKs, exchange names) and has never been audited
as a set for tier-safety.

`entity_observations` is not reachable from any route.

---

## `GET /api/v1/entity-graph/entity`

**Gate:** Entity Graph tiers. **Metered:** yes.

One entity plus its links in both directions.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `id` | string | — | **Required** |
| `limit` | int | 100 | Clamped to 1–1000. Applies to links. |

```json
{"ok": true,
 "entity": {"entity_id": "...", "entity_type": "...",
            "canonical_name": "...", "created_at": 0.0},
 "links": [{"link_id": 1, "entity_id_a": "...", "entity_id_b": "...",
            "link_type": "...", "confidence": 0.9, "source": "gdelt",
            "created_at": 0.0}],
 "dataset_scope": {...}}
```

| Status | Body |
|---|---|
| 400 | `{"ok": false, "error": "id required"}` |
| 404 | `{"ok": false, "error": "no entity 'x' — see /api/v1/entity-graph/entities"}` |

---

## `GET /api/v1/entity-graph/links`

**Gate:** Entity Graph tiers. **Metered:** yes.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `link_type` | string | none | Exact match |
| `min_confidence` | float | 0.0 | Malformed value falls back to 0.0 |
| `limit` | int | 100 | Clamped to 1–1000 |
| `offset` | int | 0 | Minimum 0 |

```json
{"ok": true, "links": [...], "count": 100, "total": 17581,
 "limit": 100, "offset": 0, "dataset_scope": {...}}
```

Seven fields per link: `link_id`, `entity_id_a`, `entity_id_b`, `link_type`,
`confidence`, `source`, `created_at`. `metadata_json` is stripped.

### `dataset_scope` on the entity-graph routes

Every response above embeds:

```json
{"dataset": "production_entity_graph",
 "scope": "entities + entity_links only",
 "excludes": ["entity_observations (raw pipeline signal/feature values)",
              "per-row metadata_json (unreviewed free-form tool fields)"],
 "note": "This is the real graph agent/models/gnn/graph_builder.py trains on ..."}
```

Do not remove this block without updating the pricing copy to match what is
actually served.

---

## `GET /evidence/graph`

**Gate:** Entity Graph tiers. **Metered:** yes.

> **This is a different, much smaller dataset than `/api/v1/entity-graph/*`.**
> `/evidence/*` serves `agent/evidence/` — a standalone document store built in
> an unrelated session, using regex-based entity extraction over manually POSTed
> documents. Measured locally 2026-08-27: 13 documents, 1,565 mentions, 5,346
> links. (The in-source comment saying "5 documents, ~155 entity strings" is
> stale.) Both datasets sit behind the same tier gate. Background:
> `docs/research/entity_graph_tier_mismatch.md`.

| Parameter | Required | Notes |
|---|---|---|
| `q` | yes | Entity string. URL-decoded, trimmed, lowercased. |

```json
{"ok": true, "<search_entity fields>": ..., "related": [...],
 "dataset_scope": {...}}
```

`400 {"ok": false, "error": "q required"}` when `q` is absent.

---

## `GET /evidence/stats`

**Gate:** Entity Graph tiers. **Metered:** yes. No parameters.

```json
{"ok": true, "stats": {"documents": 13, "mentions": 1565, "links": 5346},
 "dataset_scope": {...}}
```

Values above measured on this machine, 2026-08-27.

---

## `GET /evidence/analytics`

**Gate:** Entity Graph tiers. **Metered:** yes.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `q` | string | none | If absent, `co_occurrences` is `[]` |
| `min_docs` | int | 2 | Malformed value falls back to 2 |

```json
{"ok": true, "co_occurrences": [...], "cross_doc_pairs": [...],
 "dataset_scope": {...}}
```

---

## `GET /evidence/graph/export`

**Gate:** Entity Graph tiers. **Metered:** yes. No parameters.

Full document-evidence graph export.

```json
{"ok": true, "graph": {...}, "dataset_scope": {...}}
```

---

## `GET /evidence/graph/centrality`

**Gate:** Entity Graph tiers. **Metered:** yes.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `q` | string | none | If absent, `neighbors` is `null` |
| `top` | int | 10 | Malformed value falls back to 10. **Not clamped.** |

```json
{"ok": true, "top_by_degree": [...], "neighbors": null,
 "dataset_scope": {...}}
```

Degree centrality over the document-evidence graph, not the production graph.

---

## `GET /` · `GET /landing` · `GET /index.html`

**Gate:** none. **Metered:** no.

Serves the file at `TIRRA_LANDING_HTML`, default
`products/brief_subscription/index.html`. Always `200`. On a read error it falls
back to `200 text/plain` with `Opportunity Brief — see /brief or /buy`.

---

## `GET /buy`

**Gate:** none — a key is extracted but never checked. **Metered:** no.

| Parameter | Notes |
|---|---|
| `tier` | Selects `TIRRA_BUY_URL_<TIER>` (uppercased) |

Resolution order: `TIRRA_BUY_URL_<TIER>`, then `TIRRA_BUY_URL`.

Always `200`. With a URL configured, returns an HTML meta-refresh redirect. With
nothing configured, returns `text/plain` saying the buy link is not configured.

---

## Unmatched GET paths

`404 text/plain` with body `not found`.

---

# POST routes

---

## `POST /api/v1/contact`

**Gate:** none. **Metered:** no.

The contact form's destination. Before this route existed the form posted to
nothing.

**Request** — `application/json`:

```json
{"name": "...", "email": "...", "subject": "...", "message": "..."}
```

All four fields are required and must be non-empty after trimming.

**Limits**

| Limit | Value |
|---|---|
| Body size | 8 KB, checked from `Content-Length` **before** the body is read |
| `name`, `email`, `subject` | 300 characters each |
| `message` | 8000 characters |
| Rate | 5 requests / 3600s per source IP |

**Responses**

| Status | Body |
|---|---|
| 200 | `{"ok": true, "message": "received"}` |
| 400 | `{"ok": false, "error": "bad json"}` |
| 400 | `{"ok": false, "error": "expected a JSON object"}` |
| 400 | `{"ok": false, "error": "name, email, subject and message are all required"}` |
| 400 | `{"ok": false, "error": "one or more fields exceed the maximum allowed length"}` |
| 413 | `{"ok": false, "error": "request body too large"}` |
| 429 | `{"ok": false, "error": "rate limited", "retry_after_s": N}` + `Retry-After` |
| 500 | `{"ok": false, "error": "could not store your message — please try again shortly"}` |

On `413` and `429` the connection is closed rather than left desynced by an
unread body.

**Where messages go.** A JSONL file, one object per line, at
`TIRRA_CONTACT_LOG`, default `.tirra_opportunities/contact_messages.jsonl`. Each
record holds `received_at`, `name`, `email`, `subject`, `message`, `source_ip`.

**No email is sent.** `tirramind.com` has no MX records and
`support@tirramind.com` bounces, so there is nothing to send through. Someone
must read the file manually. This is a deliberate choice, not an oversight.

> **Known defect.** `contact.html` posts cross-origin with
> `Content-Type: application/json`, which requires a CORS preflight. The server
> has no `do_OPTIONS`, so `OPTIONS /api/v1/contact` returns **501** — verified
> live 2026-08-27. The form cannot work from a browser until `do_OPTIONS` is
> added.

---

## `POST /webhook`

**Gate:** Paddle HMAC signature. **Metered:** no.

Paddle subscription lifecycle events. The raw request bytes are used for
signature verification — do not reformat the body in any proxy.

**Headers**

| Header | Notes |
|---|---|
| `Paddle-Signature` | Required. Verified against `TIRRA_PADDLE_WEBHOOK_SECRET`. |

**Behaviour**

1. Verify the signature. Timestamp tolerance is **300 seconds**. (Paddle's own
   SDKs default to 5 seconds; 300 is a deliberate, documented compromise that
   avoids rejecting legitimate webhooks under normal processing latency on a
   single-process server.)
2. Check `event_id` against a disk-backed replay ledger. A replayed event is a
   no-op **before** `SubscriberStore` is touched. The ledger is bounded by age
   (3600s) and by a hard 10,000-entry cap.
3. Apply the event: activate, cancel, or update the subscriber; mint a
   `tirra_...` key on activation; resolve the tier from the price id via
   `TIRRA_TIER_PRICE_MAP`.

**Responses**

| Status | Body |
|---|---|
| 200 | `{..., "ok": true}` — handler result, **with `api_key` removed** |
| 400 | `{"ok": false, "error": "<verification or config error>"}` |

The minted key is deliberately not echoed back to Paddle. Retrieve it via
`SubscriberStore` for support tooling, or let the customer claim it through
`/api/v1/claim`.

A duplicate event returns `200` with `{"handled": false, "reason": "duplicate event (already processed)"}`.

---

## `POST /evidence/ingest`

**Gate:** `X-Ingest-Token` (see below). **Metered:** no.

Ingest a document into the document-evidence store.

**Request** — `application/json`:

| Field | Notes |
|---|---|
| `doc_id` | Optional. Defaults to `doc_<epoch>`. |
| `text` | Document body. Either `text` or `path` is required. |
| `path` | Server-side file path. Requires `TIRRA_INGEST_DIR`. |
| `source` | Optional, default `""` |
| `title` | Optional, default `""` |
| `doc_type` | Optional, default `"text"` |

**Authorization** — `_ingest_authorized()`, which **fails closed**:

1. `TIRRA_INGEST_TOKEN` set → constant-time compare against `X-Ingest-Token`.
2. Token empty and `TIRRA_REQUIRE_AUTH` truthy → deny all, log an error.
3. Token empty and `TIRRA_REQUIRE_AUTH` unset → dev mode, open.

**Path safety** — `_resolve_ingest_path()`:

- With `TIRRA_INGEST_DIR` unset, path ingest is refused outright. That is the
  secure default.
- When set, `realpath` is taken on both the base directory and the target, and
  the target must sit inside the base. Taking `realpath` on both sides is what
  defeats `..` traversal *and* symlinks pointing out of the base directory,
  which a string-prefix check would not.

**Responses**

| Status | Body |
|---|---|
| 200 | `{"ok": true, "doc_id": "...", "new": true, "stats": {...}}` |
| 400 | `{"ok": false, "error": "bad json: ..."}` |
| 400 | `{"ok": false, "error": "provide 'text' or 'path'"}` |
| 400 | `{"ok": false, "error": "path ingest not permitted"}` |
| 403 | `{"ok": false, "error": "invalid ingest token"}` |

The `400 path ingest not permitted` response deliberately does not echo the
requested path back — doing so would turn the error into a filesystem-existence
oracle.

> This route and `_resolve_ingest_path` were added by commit **9fa68ca**, fixing
> an arbitrary-file-read chain: the original gate was
> `if admin_token and token != admin_token`, so an empty `TIRRA_INGEST_TOKEN`
> short-circuited the check and left ingest world-open. Do not reintroduce that
> shape. Background: `docs/research/evidence_ingest_path_traversal.md`.

---

## Unmatched POST paths

`404 text/plain` with body `not found`.

---

# Environment variables

| Variable | Default | Effect |
|---|---|---|
| `TIRRA_REQUIRE_AUTH` | unset | Fail closed when no credentials are configured. **Set to `1` in production.** |
| `TIRRA_SUB_KEYS` | unset | Comma-separated static keys. Grant every tier. |
| `TIRRA_PADDLE_WEBHOOK_SECRET` | unset | Enables signature verification and `SubscriberStore` lookups. |
| `TIRRA_REJECT_QUERY_KEYS` | unset | Hard-400 any `?key=` query-string key. |
| `TIRRA_MAX_CONCURRENT_REQUESTS` | `20` | Concurrent request handling cap. |
| `TIRRA_CORS_ORIGIN` | `*` | Default `Access-Control-Allow-Origin` on all responses. |
| `TIRRA_WEB_ORIGIN` | `https://tirramind.com` | CORS origin for `/api/v1/claim` only. |
| `TIRRA_CONTACT_LOG` | `.tirra_opportunities/contact_messages.jsonl` | Contact submission destination. |
| `TIRRA_INGEST_TOKEN` | unset | Token for `POST /evidence/ingest`. |
| `TIRRA_INGEST_DIR` | unset | Enables path-mode ingest, restricted to this directory. |
| `TIRRA_LANDING_HTML` | `products/brief_subscription/index.html` | Landing page file. |
| `TIRRA_BUY_URL` | unset | Fallback checkout URL for `/buy`. |
| `TIRRA_BUY_URL_<TIER>` | unset | Per-tier checkout URL. |
| `TIRRA_TIER_PRICE_MAP` | unset | Maps Paddle price ids to tier names. |

---

# Known gaps

Verified 2026-08-27. None of these are fixed.

1. **No `do_OPTIONS`.** CORS preflight returns `501`. Any browser request using a
   non-simple content type fails, which currently includes the contact form.
2. **Claim rate limit vs. poll cadence.** 8 requests / 600s per `txn` against a
   15-request client loop. Request 9 gets `429`, and `welcome.html` has no branch
   for it.
3. **Source allowlist is narrower than the data.** 51 allowlisted names, 66
   distinct sources in `pipeline_data`.
4. **`POST /api/v1/rotate-key` does not exist.**
   `SubscriberStore.rotate_key_for_api_key()` is implemented and tested, but no
   HTTP route reaches it. A customer who loses a key has no self-service path.
5. **Rate limiters are per-process and in-memory.** Correct only because the
   deployment is a single process. A horizontal fleet makes the caps bypassable
   and needs a shared store.
6. **Error format is inconsistent.** `403` and brief-missing `404` are
   `text/plain`; everything else is JSON. Branch on status codes.
7. **Numeric parameters fail silently.** `?limit=abc` returns `200` with the
   default, not `400`. This is deliberate — an unparseable value must never
   produce a `500` — but it means a client typo is invisible.
8. **Tier gate inversion.** The $50 `scheduler` tier passes the $500 Data
   Platform and $300 Entity Graph gates.

---

## Related

- `docs/README_SYSTEM.md` — system front door, tiers, architecture, open questions
- `agent/brief_server.py` — the implementation this document describes
- `docs/research/entity_graph_tier_mismatch.md` — the two entity datasets
- `docs/research/evidence_ingest_path_traversal.md` — the 9fa68ca fix
- `docs/runbooks/production_deploy.md` — deployment
- `LESSONS.md` — the fuckup log
