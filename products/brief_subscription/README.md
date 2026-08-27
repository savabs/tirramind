# TirraMind infrastructure products

Self-serve infrastructure + intelligence products. Sold via hosted checkout
buy-links (one per tier); delivered and served automatically by the engine.
**No selling** — buyers find a tier, pay, and get a subscriber key scoped to
that tier.

## What it is

Four independently-billed tiers, gated by subscriber key + tier on the same
backend (`agent/brief_server.py`):

| Tier | Price | Surface | Gate |
|---|---|---|---|
| **Data Platform** | $500–5,000/mo | `GET /api/v1/data`, `GET /api/v1/sources` — pre-computed deterministic signals from 47 sources | `tier ∈ {data, scheduler}` |
| **Entity Graph** | $300–2,000/mo | `GET /evidence/*` — learned entity graph, cross-doc co-occurrence | `tier ∈ {entity, data, scheduler}` |
| **Scheduler** | $50–2,000/mo | `GET /api/v1/dag/runs` — pipeline run visibility (custom DAG submission: beta) | `tier == scheduler` |
| **Brief** | $19/mo | `GET /brief` / `/brief.json` / `/brief.md` — weekly positioning and flow anomaly digest (BOCPD-detected, multi-year baselines) | any active subscriber |
| *(all tiers)* | — | `GET /api/v1/usage` — caller's own metered call summary | any active subscriber |

A subscriber's tier is set by the Paddle price they bought, mapped via
`TIRRA_TIER_PRICE_MAP` (see below) and stored alongside their access record.

**Auth model:** each subscriber authenticates with an opaque `tirra_...` API
key minted on first activation (`SubscriberStore.set_active`) — never with
the raw Paddle `subscription_id`. The key is stable across cancel/reactivate
cycles and is returned once, in the webhook handler's result (`api_key`),
for support/ops tooling to hand to the customer — there is no email-delivery
flow yet, so retrieving a customer's key today means reading it from
`.tirra_opportunities/subscribers.json` or the webhook handler result.

**Usage metering:** every authorized call to a metered endpoint (Data
Platform, Entity Graph, Scheduler, Brief) is logged to
`.tirra_opportunities/usage.db` (`agent/payments/usage.py`) keyed by the
caller's API key. Nothing is enforced yet (no hard quotas) — it's
visibility only, via `GET /api/v1/usage`.

Deterministic math on public data. No LLM, no black box.

## The "ghost" architecture

```
tirra-engine (scheduled, cron/systemd)
  → fast refresh (CFTC + gov)
  → build brief → deliver (.tirra_delivery/)
  → serve (agent/brief_server.py)
       GET /            → landing page
       GET /buy         → hosted checkout (Paddle/LemonSqueezy)
       GET /brief.json  → subscriber-only (key required)
       GET /brief.md    → subscriber-only (key required)
       POST /webhook    → Paddle subscription events (signed, Ed25519)
```

## Paddle integration (built)

`agent/payments/` provides a layered, migration-ready integration:
- `config.py` — `PaddleConfig` reads everything from env (`TIRRA_PADDLE_MODE`
  sandbox|live). Switching to live = change env, not code.
- `client.py` — server-side Paddle Billing API client (prices, products,
  subscriptions, customers, webhook IP source).
- `webhook.py` — Ed25519 signature verification (mandatory in live; skipped in
  sandbox/dev). Includes stale-timestamp replay protection.
- `handler.py` — maps subscription lifecycle events → subscriber access store.

`brief_server.py` serves `POST /webhook` (verifies + applies) and grants/revokes
brief access based on the subscriber store.

### Verified end-to-end
A signed `subscription.activated` webhook → grants `sub_<id>` access → that key
opens `/brief.json` (200); unknown keys get 403.

## Deploy — the public front door

The **static site** (this folder) is the public homepage on `tirramind.com`.
The **backend** (`agent/brief_server.py`) runs separately and serves the
protected brief + webhook.

```
                    tirramind.com  (static: Vercel / Netlify / Cloudflare Pages)
                          │  index.html + terms/privacy/refunds
                          ▼
                        /buy  ─────────────►  Paddle checkout (when live)
                          │
                          ▼
              api.tirramind.com  (Tirra backend)
                          │  /webhook  (Paddle events, signed)
                          ├  /brief.json  (subscriber-only)
                          └  /brief.md    (subscriber-only)
```

### 1. Deploy the static site (pick ONE host)

The publish root is this folder. It already has `vercel.json` and `netlify.toml`.

- **Vercel**: push this folder to a Vercel project, set root directory to
  `products/brief_subscription`. Domain → `tirramind.com`.
- **Netlify**: drag the folder into Netlify or link the repo, publish dir = this
  folder. Domain → `tirramind.com`.

### 2. Point the backend at `api.tirramind.com`

Run the Tirra backend (`agent/brief_server.py` / `scripts/run_scheduled.sh serve`)
on your server, add a DNS A/AAAA record: `api.tirramind.com` → your server IP.
Update `vercel.json`/`netlify.toml` so `/brief.json` proxies to
`https://api.tirramind.com/brief.json`.

### 3. Configure Paddle (after domain is live + approved)

1. Create one Paddle product/price per tier (Data Platform, Entity Graph,
   Scheduler, Opportunity Brief).
2. Get each live payment link → set `TIRRA_BUY_URL_<TIER>` per tier (e.g.
   `TIRRA_BUY_URL_DATA`, `TIRRA_BUY_URL_ENTITY`, `TIRRA_BUY_URL_SCHEDULER`,
   `TIRRA_BUY_URL_BRIEF`). `/buy?tier=data` reads `TIRRA_BUY_URL_DATA`,
   falling back to `TIRRA_BUY_URL` if unset — the pricing page links to
   `/buy?tier=<name>` for each plan.
3. Map each Paddle price ID to its tier name via `TIRRA_TIER_PRICE_MAP`
   (`pri_xxx:data,pri_yyy:entity,pri_zzz:scheduler`) so the webhook grants the
   right tier automatically. Unmapped/legacy prices default to the `brief`
   tier.
4. Create a live notification destination → `https://api.tirramind.com/webhook`,
   capture `endpoint_secret_key` → set `TIRRA_PADDLE_WEBHOOK_SECRET`.
5. Request domain approval in Paddle (Checkout → Request domain approval) —
   **required for live checkouts**; sandbox auto-approves, live does not.

### 4. Set backend env (live)

```bash
TIRRA_PADDLE_MODE=live
TIRRA_PADDLE_API_KEY=<live api key>
TIRRA_PADDLE_CLIENT_TOKEN=live_...
TIRRA_PADDLE_WEBHOOK_SECRET=<endpoint_secret_key>
TIRRA_TIER_PRICE_MAP=pri_data:data,pri_entity:entity,pri_scheduler:scheduler,pri_brief:brief
TIRRA_BUY_URL_DATA=https://checkout.paddle.com/...
TIRRA_BUY_URL_ENTITY=https://checkout.paddle.com/...
TIRRA_BUY_URL_SCHEDULER=https://checkout.paddle.com/...
TIRRA_BUY_URL_BRIEF=https://checkout.paddle.com/...
TIRRA_BUY_URL=https://checkout.paddle.com/...   # fallback if a tier-specific link is unset
TIRRA_INGEST_TOKEN=<random admin token>          # required to call POST /evidence/ingest in live
```

No code changes needed for sandbox → live; it's all env/config.

## Test locally (open mode, no key needed)

```bash
tirra-engine --once          # build + deliver
tirra-serve                  # serve; /brief.json open (no TIRRA_SUB_KEYS)
```

To test the paywall: set `TIRRA_SUB_KEYS` then `/brief.json` returns 403 without
a valid key, 200 with it.

## Files

| Path | Purpose |
|---|---|
| `products/brief_subscription/index.html` | Public homepage (`tirramind.com`) |
| `products/brief_subscription/terms.html` | Terms (verification) |
| `products/brief_subscription/privacy.html` | Privacy (verification) |
| `products/brief_subscription/refunds.html` | Refund policy (verification) |
| `products/brief_subscription/vercel.json` | Vercel routes (`/buy`, `/brief` proxy) |
| `products/brief_subscription/netlify.toml` | Netlify routes (alt) |
| `agent/brief_server.py` | Backend: /webhook, /brief (gated), /status, /api/v1/sources + /api/v1/data (Data tier), /api/v1/dag/runs (Scheduler tier), /evidence/* (Entity Graph tier), /api/v1/usage (any tier) |
| `agent/payments/usage.py` | `UsageStore` — SQLite log of metered API calls per subscriber key |
| `agent/payments/` | Paddle integration (config, client, webhook verify) |
| `scripts/tirra_engine.py` | Build + deliver + serve engine |
| `scripts/run_scheduled.sh` | Scheduled runner |
