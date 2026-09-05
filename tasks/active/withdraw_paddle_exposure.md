---
title: "Task: Withdraw the Live Paddle Checkout Exposure"
tags:
  - doc/task
  - status/active
  - phase/publish
  - topic/payments
  - layer/commerce
---

# Task: Withdraw the Live Paddle Checkout Exposure

Status: active
Research: docs/research/paddle_checkout_exposure.md
Spec: docs/specs/paddle_checkout_exposure_spec.md

## Goal

Make it impossible for a stranger to open a live Paddle checkout for a product
that was withdrawn on 2026-08-29 and has no server to fulfil it.

## Why

`products/brief_subscription/pricing.html` carried `PADDLE_ENV = "live"`, a live
client token, and four live price IDs. The only thing standing between a visitor
and a real $500/mo charge was `TIER_AVAILABILITY` — four JavaScript booleans and
an early return in `openCheckout()`, both of which run on the visitor's machine.

The client token is not the problem; Paddle designs those to be public. The
problem is the composition: public token + four live price IDs + a client-side
gate, on a merchant account with **zero legitimate transactions ever**
(`.tirra_opportunities/subscribers.json` has never existed). A first transaction
that becomes a refund or chargeback is a materially worse signal to Paddle's risk
team than one among many good charges, and live price IDs on a no-history
merchant are a usable card-testing target.

Meanwhile `api.tirramind.com` resolves to 188.245.203.137 but nothing listens, so
a buyer would receive nothing at all.

## Scope Notes

- Layer: commerce surface (static storefront), not one of the 7 pipeline layers.
- Main files changed: `products/brief_subscription/pricing.html`, `terms.html`.
- Non-goals: standing the API back up; the new portfolio-risk product; email.

## Steps

- [x] 1.1: Remove the live client token and the four live price IDs from
      `pricing.html`; set the token to a `REPLACE_`-prefixed placeholder so the
      file's own existing guard suppresses `Paddle.Initialize`.
      Verification: `grep -rn "live_f2e811852c6b0d1430b835c1a8c\|pri_01m115x"
      --exclude-dir=.git .` returns nothing. Confirmed clean 2026-09-05.
- [x] 1.2: Correct `terms.html`, which described all four withdrawn products as
      active recurring subscriptions, and withdraw the two Entity Graph
      capability claims that `agent/brief_server.py`'s own comments contradict.
      Verification: the page states the withdrawal above section 1 and no longer
      asserts an active subscription.
- [ ] 1.3: **Archive the four prices in the Paddle dashboard.** This is the only
      server-side control and the only step that actually closes the hole —
      steps 1.1 and 1.2 are defence in depth, because the old values remain in
      this repo's public git history and removing them from HEAD removes nothing
      from the internet. Requires dashboard access; cannot be done from here.
      Verification: `Paddle.Checkout.open` with each archived price ID is
      rejected by Paddle's API rather than rendering an overlay.
- [ ] 1.4: In the Cloudflare Pages dashboard, delete every retained deployment
      of the `tirramind` project that contains `pricing.html`. Pages keeps each
      prior deployment at a permanent `<hash>.tirramind.pages.dev` URL. Note
      that `products/site/` and `products/brief_subscription/` both deploy to
      the same project name, so the storefront was superseded by deploy order,
      not by any control.
      Verification: no retained deployment serves a page containing a `pri_`
      identifier.
- [ ] 1.5: Give `support@tirramind.com` a real mailbox. It is referenced 5 times
      across `terms.html`, `refunds.html` and `privacy.html` — including as the
      GDPR rights contact — and `dig MX tirramind.com` is empty, so all of it
      bounces. Cloudflare Email Routing is free and inbound-only, which is
      sufficient for receiving; sending needs a separate provider.
      Verification: a message sent to the address arrives.

## Completion Checklist

- [ ] Research note exists and is current — N/A, remediation
- [ ] Spec matches the actual implementation plan — N/A, subtractive fix
- [x] Each completed step has a verification result
- [ ] Edge-case testing was added and run for code changes — N/A, static HTML
- [ ] Checkpoint written at the end of the session or sub-phase
- [x] Frontmatter tags and `## Related` section are current

## Related

- `docs/research/production_deployment.md` — records that the storefront is
  half-deployed and the product behind it entirely undeployed.
- `products/site/index.html` — the live site, which already states correctly
  that the earlier signal product is withdrawn and not for sale.

## Notes

Steps 1.3 through 1.5 all require dashboard or DNS access and are the owner's to
perform. Until 1.3 is done, this task is not complete regardless of what the
repo looks like.
