# Opportunity Brief — subscription product

Self-serve weekly government-contract intelligence. Sold via a hosted checkout
buy-link; delivered and served automatically by the engine. **No selling** —
buyers find it, pay, and get a subscriber key.

## What it is

A weekly brief with:
- **Contract opportunities** — small, overlooked federal contracts ranked by
  expected value (`EV = P(win)·(Bid−Cost)−Risk`) and learned win-probability.
- **Market signals** — real anomalies (positioning extremes, regime changes).
- Deterministic math on public data. No LLM, no black box.

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

1. Create the Paddle product/price "Opportunity Brief — weekly" at $19/month.
2. Get the live payment link → paste into `vercel.json`/`netlify.toml` redirect
   for `/buy` (and set `TIRRA_BUY_URL` on the backend).
3. Create a live notification destination → `https://api.tirramind.com/webhook`,
   capture `endpoint_secret_key` → set `TIRRA_PADDLE_WEBHOOK_SECRET`.
4. Request domain approval in Paddle (Checkout → Request domain approval) —
   **required for live checkouts**; sandbox auto-approves, live does not.

### 4. Set backend env (live)

```bash
TIRRA_PADDLE_MODE=live
TIRRA_PADDLE_API_KEY=<live api key>
TIRRA_PADDLE_CLIENT_TOKEN=live_...
TIRRA_PADDLE_WEBHOOK_SECRET=<endpoint_secret_key>
TIRRA_PADDLE_PRICE_ID=pri_...
TIRRA_BUY_URL=https://checkout.paddle.com/...
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
| `agent/brief_server.py` | Backend: /webhook, /brief (gated), /status |
| `agent/payments/` | Paddle integration (config, client, webhook verify) |
| `scripts/tirra_engine.py` | Build + deliver + serve engine |
| `scripts/run_scheduled.sh` | Scheduled runner |
