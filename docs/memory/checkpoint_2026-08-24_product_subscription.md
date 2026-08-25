---
title: "Checkpoint 2026-08-24 — Product: self-serve brief subscription (buy-link)"
tags:
  - doc/checkpoint
  - phase/1
  - topic/product
  - topic/subscription
  - status/active
---

# Checkpoint: Product — self-serve Opportunity Brief subscription

**Date:** 2026-08-24

## What it is

A self-serve, buy-link subscription for the weekly government-contract + market
intelligence brief. **"Ghost" product**: runs automatically, sold via hosted
checkout, delivered without selling. Buyers find it, pay, get a key.

## How it works (the whole spine)

```
tirra-engine (scheduled)
  → fast refresh (CFTC + gov) → build brief → deliver (.tirra_delivery/)
  → serve (agent/brief_server.py)
       GET /            → landing page (products/brief_subscription/index.html)
       GET /buy         → hosted checkout (Paddle/LemonSqueezy buy URL)
       GET /brief.json  → subscriber-only (key required)
       GET /brief.md    → subscriber-only (key required)
       GET /status      → delivery status
```

## What was added

1. **Subscription gate** in `agent/brief_server.py`:
   - `TIRRA_SUB_KEYS` = valid subscriber keys; when set, `/brief*` returns 403
     without a key, 200 with it. Unset → open (dev) mode.
   - `TIRRA_BUY_URL` → `/buy` redirects to hosted checkout.
   - `TIRRA_LANDING_HTML` → `/` serves the landing page.
2. **Landing page**: `products/brief_subscription/index.html` ($19/mo, value prop).
3. **Setup README**: `products/brief_subscription/README.md` — deploy steps
   (Paddle/LemonSqueezy checkout, env config, key issuance).
4. **Make targets**: `make run` (build+deliver), `make serve`.

## Verification

- `pytest` — **77 passed** (6 server tests incl. key gate + buy redirect)
- `ruff` — clean
- **Live paywall demo**: with `TIRRA_SUB_KEYS` set —
  - `/` → landing page title ✅
  - `/brief.json` no key → **403** ✅
  - `/brief.json?key=...` → **200** ✅
  - `/buy` → redirects to checkout URL ✅

## What it's NOT yet

- No real checkout provider hooked up (need a Paddle/LemonSqueezy account + the
  buy URL + a key-delivery method). That's a one-time config, documented in the
  README — the code side is done.
- No real customers / revenue yet.

## Honest status vs north star (predictive intelligence)

The product is the live-path predictive intelligence (contract EV + learned
P(win) + market anomalies) delivered as a brief. The full GNN/weight-based
intelligence layer remains the longer-term north-star, parked until the live
product proves demand.

## Related
- [[checkpoint_2026-08-24_tirra_engine]]
- [[checkpoint_2026-08-24_delivery_layer]]
- [[checkpoint_2026-08-24_fused_intelligence_brief]]
- [[checkpoint_2026-08-24_live_path_intelligence]]