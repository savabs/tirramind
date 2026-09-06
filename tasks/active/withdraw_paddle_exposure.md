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

## Addendum — CI unblocked while landing this (2026-09-06)

Opening PR #1 was the first time this repo's CI had actually executed since
2026-09-02: every run in between failed in 3-7 seconds with "the job was not
started because your account is locked due to a billing issue", so no job ever
started and the suite's real state was invisible.

Once billing was fixed, `test (3.11)` and `test (3.12)` failed on:

    ERROR collecting tests/test_sde.py
    ModuleNotFoundError: No module named 'torchsde'
    Interrupted: 1 error during collection

`agent/quant/sde.py` imports `torchsde` at module scope. `torchsde` is in the
`[ml]` extra, and `.github/workflows/ci.yml` deliberately installs only
`.[dev,quant]` before hand-installing torch and torch-geometric to avoid the
2 GB CUDA wheel. So `tests/test_sde.py` could never be collected in CI -- and
because pytest aborts the whole run on a collection error, **one absent optional
dependency was taking all 10,943 tests down with it.**

Fixed with `pytest.importorskip("torchsde")` in `tests/test_sde.py`.

Verified by simulating CI locally -- blocking every module in the `[ml]` extra
that CI does not install (`torchsde`, `torchdiffeq`, `torchcde`, `mambapy`,
`ts2vec`) via a `sitecustomize` meta-path finder raising `ModuleNotFoundError`:

- `tests/test_sde.py` alone: 1 skipped, no error
- full suite: **10,943/10,952 collected, 9 deselected, zero collection errors**

`torchsde` was the only landmine of the five.

Not fixed here: the `lint` job fails on 211 pre-existing ruff errors and 111
files needing reformatting, none of them touched by this PR. Reformatting 111
files onto a security change would bury a five-file diff, so that is tracked
separately.

### Second CI break, found once the tests could finally run

With collection fixed, the suite executed for the first time and returned
**208 failed, 10,716 passed** on both 3.11 and 3.12. Every failure was the same
line:

    RuntimeError: Numpy is not available

Cause: `.github/workflows/ci.yml` pinned `torch==2.2.2+cpu`, which is built
against the **numpy 1.x ABI**, while the preceding `pip install -e ".[dev,quant]"`
resolves numpy to **2.5.2**. Every tensor<->numpy conversion therefore failed.

Pinning numpy backwards is not available: `scipy 1.18.1` requires
`numpy>=2.0.0,<2.8`, and scipy is in the `quant` extra. So torch had to move.

Chose **torch 2.13.0**, because it is what this project's own `.venv` runs, and
the suite passes there against the identical numpy 2.5.2. Verified before
changing anything:

- local `torch 2.13.0` + `numpy 2.5.2`: tensor->numpy conversion works
- local full suite under simulated CI conditions: **3 failed, 10,922 passed**
  (the 3 being 2 live-network tests hitting a real API, and 1 artifact of the
  simulation blocking `mambapy`) — against 208 on torch 2.2.2
- `data.pyg.org` publishes cpu wheels for torch 2.13.0
- `download.pytorch.org` publishes cp311 and cp312 cpu wheels, which the matrix
  requires

Matching CI to the environment the tests are developed in removes the drift
rather than papering over it.

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
