---
title: "Spec: Withdraw the Live Paddle Checkout Exposure"
tags:
  - doc/spec
  - topic/payments
  - status/current
---

# Spec: Withdraw the Live Paddle Checkout Exposure

Research: `docs/research/paddle_checkout_exposure.md`

## Objective

No stranger can open a Paddle checkout for a withdrawn product.

## Ordered steps

1. **Strip the credentials from `pricing.html`.** Set `PADDLE_ENV` to
   `"sandbox"`, replace the client token with a `REPLACE_`-prefixed placeholder,
   and empty all four `TIER_PRICE_IDS`.
   *Why the prefix matters:* the file already guards on
   `!PADDLE_CLIENT_TOKEN.startsWith("REPLACE_")` in two places, and
   `openCheckout()` bails on a falsy `priceId`. Reusing the existing guard is
   preferable to adding a new one.
   **Verify:** `grep -rn "live_f2e811852c6b0d1430b835c1a8c\|pri_01m115x"
   --exclude-dir=.git .` returns nothing.

2. **Correct `terms.html`.** State the withdrawal above section 1; stop
   describing the four products as active recurring subscriptions; withdraw the
   two Entity Graph capability claims that `agent/brief_server.py:1641-1647`
   contradicts.
   **Verify:** the page no longer asserts a purchasable subscription.

3. **Archive the four prices in the Paddle dashboard.** The only step that
   closes the hole. Requires dashboard access.
   **Verify:** `Paddle.Checkout.open` with each archived price ID is rejected by
   Paddle rather than rendering an overlay.

4. **Delete retained Cloudflare Pages deployments containing `pricing.html`.**
   **Verify:** no retained deployment serves a page containing a `pri_`
   identifier.

5. **Give `support@tirramind.com` a mailbox.** Five references across `terms`,
   `refunds` and `privacy` — including the GDPR rights contact — and
   `dig MX tirramind.com` is empty.
   **Verify:** a message sent to the address arrives.

## Non-goals

Standing the API back up; the new portfolio-risk product; outbound email
sending. Steps 3-5 are the owner's; 1-2 are subtractive edits to static HTML.

## Explicitly not done

Rotating the client token. It is pointless while the prices are live and
unnecessary once they are archived.
