---
title: "Checkpoint 2026-08-24 — Front door: tirramind.com static site + deploy config"
tags:
  - doc/checkpoint
  - phase/1
  - topic/product
  - topic/deployment
  - status/active
---

# Checkpoint: Front door — public site + deploy config for tirramind.com

**Date:** 2026-08-24

## Context

The user secured the domain **tirramind.com**. The product backend (brief engine,
delivery, subscriber gating, Paddle integration) was already built. The missing
piece was the **public front door**: a real homepage on the domain, plus the
static-host + routing config to connect it to the backend and Paddle.

## What was built (in `products/brief_subscription/`)

- **`index.html`** — rewritten into a proper TirraMind homepage: brand, tagline,
  value-prop cards (contract opportunities / market anomalies / EV scoring /
  honest math), `$19/month` pricing, `Subscribe → /buy`, and a "How it works"
  section.
- **`terms.html`** — real Terms & Conditions (verification-required).
- **`privacy.html`** — real Privacy Policy (verification-required).
- **`refunds.html`** — real Refund & Cancellation Policy (verification-required).
- **`vercel.json`** — routes: `/buy` → Paddle checkout (fill-in), `/brief.json` +
  `/brief.md` → `api.tirramind.com` backend proxy.
- **`netlify.toml`** — equivalent Netlify config (alternative host).
- **`README.md`** — full deploy plan: static site on tirramind.com, backend on
  api.tirramind.com, Paddle domain-approval + env steps.

## Architecture (final)

```
tirramind.com (static: Vercel/Netlify/CF)
   │  index.html + terms/privacy/refunds
   ▼
 /buy ─────────────► Paddle checkout (when live)
   ▼
api.tirramind.com (Tirra backend)
   │  /webhook   (Paddle events, signed Ed25519)
   ├  /brief.json  (subscriber-only)
   └  /brief.md    (subscriber-only)
```

## Verification
- `vercel.json` parses as valid JSON.
- All 4 HTML pages present and coherent.
- `index.html` Subscribe → `/buy` (wired to redirect config).
- Backend endpoints (`/brief`, `/webhook`, `/status`) already implemented and tested.

## Honest status / next
- The **code is deploy-ready**. Deploying to a live host + pointing the domain is
  a manual (non-code) step: push the folder to Vercel/Netlify, add DNS for
  `tirramind.com` and `api.tirramind.com`.
- Paddle live checkout is **not yet configured** (per plan: only after domain is
  live + approved). `/buy` redirect is a fill-in placeholder.
- No real customers / revenue yet.

## Related
- [[checkpoint_2026-08-24_paddle_integration]]
- [[checkpoint_2026-08-24_product_subscription]]
- [[checkpoint_2026-08-24_tirra_engine]]
