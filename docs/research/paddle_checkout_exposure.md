---
title: "Research: The Live Paddle Checkout Exposure"
tags:
  - doc/research
  - topic/payments
  - status/current
---

# Research: The Live Paddle Checkout Exposure

Date: 2026-09-05

## The finding

`products/brief_subscription/pricing.html` shipped `PADDLE_ENV = "live"`, a live
Paddle client token, and four live price IDs. All four tiers were withdrawn on
2026-08-29 (commit `c594dd9`), but the withdrawal was implemented as
`TIER_AVAILABILITY` — four JavaScript booleans — plus an early return in
`openCheckout()`.

Both run on the visitor's machine. Neither is a control.

## Why the client token is not the problem

Paddle client-side tokens are designed to be published. A client token cannot
issue refunds, read customer records, mutate the catalogue, or call the Paddle
API. `TIRRA_PADDLE_API_KEY` would be a real credential; it is not in this repo,
and a full history scan of both repos found no committed secret.

## What the problem actually is

The composition, not any single component:

1. A public token, plus
2. four live price IDs in the same file, plus
3. a gate that exists only in client-side JavaScript, on
4. a merchant account with **zero legitimate transactions, ever** —
   `.tirra_opportunities/subscribers.json` has never existed.

Anyone can reconstruct a working checkout in three lines:

```js
Paddle.Initialize({ token: "<the token>" });
Paddle.Checkout.open({ items: [{ priceId: "<the price id>", quantity: 1 }] });
```

## Consequence, in order of likelihood

- A technically curious visitor — this repo is public and the live site links
  straight to it — completes a $500/mo subscription and receives nothing, because
  `api.tirramind.com` resolves to `188.245.203.137` but nothing listens on it.
- Refund obligation, then a dispute. **A single chargeback against zero
  transaction history is 100% of that account's record.** That is a materially
  worse signal to Paddle's risk team than one chargeback among a hundred good
  charges, and Paddle can freeze payouts or terminate the seller account.
- Card testing. Four live price IDs on a merchant with no fraud history is a
  usable target; the resulting burst of declines is the fastest route to
  account termination.

## What actually closes it

Only archiving the four prices in the Paddle dashboard. That is the sole
server-side control.

Removing the values from the repo does **not** close it — they remain in this
repo's public git history, and Cloudflare Pages retains every prior deployment
at a permanent `<hash>.tirramind.pages.dev` URL. Note also that
`products/site/` and `products/brief_subscription/` carry the same
`project_name: "tirramind"` in their `.wrangler` caches: the storefront was
superseded by deploy order, not by any deliberate control, and one
`wrangler pages deploy` of the wrong directory would republish it.

## The generalisable lesson

Every finding in the wider audit that produced this note lived in a seam
between two correct components: a JavaScript constant guarding a server-side
payment; a Dockerfile that never heard of the env template; a body-size cap
added in one route handler instead of at the read; a `mode` default that
disarms a secret check in a different module.

A gate belongs on the same side of the trust boundary as the thing it protects.
