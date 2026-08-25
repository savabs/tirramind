---
title: "Checkpoint 2026-08-24 — Paddle integration built (migration-ready)"
tags:
  - doc/checkpoint
  - phase/1
  - topic/payments
  - topic/product
  - status/active
---

# Checkpoint: Paddle Integration Built (sandbox→live ready)

**Date:** 2026-08-24

## Why this step

The prior session left only env *placeholders* for Paddle (`TIRRA_BUY_URL`,
`TIRRA_SUB_KEYS`) — no real integration. The audit confirmed no sandbox
integration existed to migrate. Per the user's decision ("C"), I built the
integration **from scratch**, designed so the sandbox→live migration is a
config change, not a code change.

## What was built — `agent/payments/`

| Module | Purpose |
|---|---|
| `config.py` | `PaddleConfig` — env-only (`TIRRA_PADDLE_MODE` sandbox\|live). Live requires a webhook secret; derives `api.paddle.com` vs `sandbox-api.paddle.com`. `to_public_dict()` never leaks the API key. |
| `client.py` | `PaddleClient` — server-side Billing API (prices, products, subscriptions, customers) + `fetch_webhook_ips()` (live allowlist source of truth). |
| `webhook.py` | Ed25519 signature verification (Paddle v2 `Paddle-Signature` `ts;h1`). Mandatory in live (fail-closed), skipped in sandbox/dev. Includes stale-timestamp replay guard. |
| `handler.py` | `PaddleWebhookHandler` + `SubscriberStore` — maps subscription lifecycle events → subscriber access store. Additive/idempotent; never touches live Paddle entities. |

## Wiring

`brief_server.py`:
- `POST /webhook` — reads raw body + `Paddle-Signature`, verifies (when secret set), applies event, grants/revokes access.
- `_valid_key` — now also accepts an **active Paddle subscriber** as a key; dev-open mode only when neither static keys nor Paddle secret is configured.
- `GET /brief.json|md` — subscriber-gated.

`.env.example` — added full Paddle section (no secrets). Product README updated.

## Verification

- **90 tests passed** (13 new payments tests: config, signature valid/tampered/stale, handler grant/revoke, no-secret dev mode).
- **ruff clean**.
- **End-to-end live proof**: generated an Ed25519 keypair, signed a `subscription.activated` webhook, POSTed to `/webhook` → returned `active:true`, then `/brief.json?key=sub_live1` → **200**, unknown key → **403**.

## Migration readiness

To go live (later steps, not this one):
1. Set `TIRRA_PADDLE_MODE=live` + live API key + `live_` client token + live webhook secret + live `pri_` price ID.
2. Create the live notification destination → capture `endpoint_secret_key` once → set `TIRRA_PADDLE_WEBHOOK_SECRET`.
3. Swap sandbox `pri_`→live `pri_` in env.
4. Add `pwCustomer` to `Paddle.Initialize()` using `TIRRA_PADDLE_RETAIN_ID` (live `ctm_` id).
5. Allowlist live webhook IPs from `https://api.paddle.com/ips`.

All are config/env changes — no code edits needed for the migration.

## Related
- [[checkpoint_2026-08-24_product_subscription]]
- [[checkpoint_2026-08-24_tirra_engine]]
- [[checkpoint_2026-08-24_delivery_layer]]
