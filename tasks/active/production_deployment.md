---
title: Task — Production Deployment
tags:
  - doc/task
  - topic/deployment
  - topic/payments
  - status/active
---

# Task — Production Deployment

Research: `docs/research/production_deployment.md`
Spec: `docs/specs/production_deployment_spec.md`

Started 2026-08-26.

**Context in one line:** Paddle merchant verification passed, but there is no
production to sell into — `tirramind.com` fails TLS, `api.tirramind.com` is
NXDOMAIN, the webhook verified against the wrong crypto scheme, and a paying
customer would never receive their API key. This task covers everything
between "verified in Paddle" and "a customer can pay and use the product."

---

## Checklist

### Payments correctness (blocks going live at all)
- [x] Webhook signature verification rewritten Ed25519 → HMAC-SHA256 over
      `"<ts>:<raw body>"`, secret used directly as the HMAC key (Paddle
      Billing's actual scheme) — the previous implementation would have
      rejected every real webhook
- [x] `SubscriberStore` mints a customer-facing `tirra_<token>` API key on
      first activation, distinct from the internal Paddle `subscription_id`
- [x] Tier resolution from `TIRRA_TIER_PRICE_MAP` (price ID → tier), falling
      back to the base tier for subscriptions created before tiers existed
- [x] `scripts/setup_paddle_products.py` — creates the real products/prices in
      Paddle via `PaddleClient.create_product`/`create_price`
- [x] `brief_server.py` auth fails **closed**, not open, when unconfigured
      under `TIRRA_REQUIRE_AUTH=1` (previously: no secret configured → anyone
      authorized)

### Storefront (matches what's actually being sold)
- [x] `products/brief_subscription/pricing.html` reflects the real tiers/prices
      created in Paddle, not placeholder copy
- [x] `_redirects` added for Cloudflare Pages (Netlify `netlify.toml` syntax
      doesn't carry over) — mirrors existing rewrites to `api.tirramind.com`

### Infra (deploy/)
- [x] `deploy/env.production.example` — only vars the API server and chain
      read; the VM never gets WandB/HF/GitHub/Kaggle/Anthropic keys
- [x] `deploy/Caddyfile` — TLS termination for `api.tirramind.com`
- [x] `deploy/provision_vm.sh` — idempotent bootstrap, does not touch DNS/secrets
- [x] `deploy/backup_to_r2.sh` — refuses to prune retention on an empty
      snapshot or failed upload (previously could delete backups on a bad
      run); also fixed a missing-DB case that bypassed that guard, and a
      `grep -q`/`pipefail` race that could falsely report a good upload as
      unverifiable — both caught by `tests/test_backup_to_r2.py`, added this
      pass
- [x] `deploy/provision_vm.sh` — also fixed `/var/log/caddy` being created
      owned by the app user instead of `caddy:caddy`, which left the Caddy
      process unable to write its own log file (the official Debian package
      runs it as `User=caddy`)
- [x] `deploy/systemd/*` — `tirra-api`, `tirra-collect`, `tirra-chain`,
      `tirra-backup` services + timers; `deploy/systemd/README.md` now also
      documents `tirra-backup` (previously undocumented there)
- [x] Budget re-checked at $30/mo (was $10/mo) — `docs/research/production_deployment.md`
      §0 superseded by `docs/runbooks/production_deploy.md` §0.2: Hetzner
      CAX21 (4 vCPU/8 GB, ~$11.50/mo, current Aug-2026 pricing) + HetrixTools
      free uptime monitoring (commercial-use-safe, unlike UptimeRobot's free
      tier) — total ~$12/mo, headroom left deliberately unspent
- [x] `docs/runbooks/production_deploy.md` — the single ordered, copy-paste
      runbook: owner-only account/DNS/VM steps, then scripted provisioning,
      then a from-outside-the-box verification checklist
- [ ] Actually provision the VM, DNS record, R2 bucket — needs the operator's
      own cloud accounts/credentials (explicitly out of scope for Claude;
      `docs/runbooks/production_deploy.md` is the handoff)

### Free data API keys (unblocks 4 of 54 daily_collection sources)
- [x] `.env.example` documents `TIRRA_NASA_FIRMS_KEY` / `FIRMS_API_KEY` /
      `TIRRA_EIA_API_KEY` and which tools they unblock
- [ ] Operator registers the actual keys and sets them in `.env` / `.env.production`

### Supporting tooling (agent team)
- [x] `.claude/agents/` — 23-specialist team split across Direction / Product-ML
      / Revenue, each with an exclusive domain and documented boundaries, so
      future sessions triage instead of fanning out to every agent
- [x] `.claude/hooks/` — guards against overwriting protected files (`.env`,
      trained checkpoints, `pipeline.db`) and warns on schema drift
- [x] `.claude/CLAUDE.md` §12 — dispatch must name DISPATCHED/EXCLUDED/SEQUENCE;
      default 1–3 agents, 7+ needs explicit approval

## Outcome so far

Payments now verify against Paddle's real scheme and a customer who pays
receives a usable key. Auth fails closed by default. Deploy artifacts exist
but are not yet applied to a running VM — DNS still resolves to Vercel and
`api.tirramind.com` is still NXDOMAIN as of this commit. Not yet actioned:
provisioning the VM/DNS/R2, and registering the two free data API keys.
