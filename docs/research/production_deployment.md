---
title: Research — Production Deployment
tags:
  - doc/research
  - topic/deployment
  - status/active
---

# Research — Production Deployment

*Investigated 2026-08-26. DNS/TLS results are live checks, not assumptions.*

---

## 1. Current state: there is no production

| check | result |
|---|---|
| `tirramind.com` DNS | resolves → `64.29.17.65`, `216.198.79.65` (Vercel) |
| `tirramind.com` HTTPS | **TLS handshake fails** — certificate not provisioned |
| `www.tirramind.com` | **NXDOMAIN** |
| `api.tirramind.com` | **NXDOMAIN** |

Both `vercel.json` and `netlify.toml` rewrite the entire product surface to a
host that does not exist:

```
/brief.json      → https://api.tirramind.com/brief.json
/brief.md        → https://api.tirramind.com/brief.md
/api/v1/:path*   → https://api.tirramind.com/api/v1/:path*
/evidence/:path* → https://api.tirramind.com/evidence/:path*
```

The storefront is half-deployed; the product behind it is entirely undeployed.

---

## 2. Vercel cannot host this backend

Not a plan-tier limitation — an architectural mismatch:

| requirement | Vercel serverless |
|---|---|
| `brief_server.py` is a long-lived `ThreadingHTTPServer` | ✗ functions are request-scoped |
| Pipeline state is a **138 MB mutable SQLite file** | ✗ no persistent writable disk |
| `daily_collection` runs 40+ network sources for minutes | ✗ function timeout |
| Deploy story is systemd units on `/opt/tirramind` | ✗ no VM |

Upgrading to Pro changes none of these. The backend needs a **VM**.

---

## 3. Licensing constraint on the static site

Vercel's **Hobby plan is non-commercial use only**. Once Paddle goes live and
the domain sells subscriptions, staying on Hobby is a ToS violation — so the
real choice is "pay for Vercel Pro" or "move off Vercel", not "stay free".

**Cloudflare Pages' free tier explicitly permits commercial use.** Since
Cloudflare is already in the stack for DNS, moving the static site there
removes the licensing problem *and* the $20/mo, and consolidates vendors.

Verify current plan terms before acting — pricing and ToS change.

---

## 4. Recommended topology

```
                    Cloudflare (DNS + TLS, free)
                             │
        ┌────────────────────┴────────────────────┐
        │                                         │
  tirramind.com                            api.tirramind.com
  Cloudflare Pages (free, commercial OK)   VM (~$5/mo)
  static: index / pricing / terms          brief_server.py :8787
  Paddle.js overlay checkout               systemd: tirra-api.service
                                           systemd: tirra-chain.timer
                                           SQLite pipeline.db
```

Estimated cost: **~$5/month**, against a stated $10/month budget.

Rejected:
- **Vercel Pro ($20/mo)** — doesn't solve the backend gap; superseded by moving
  static hosting to Pages.
- **Cloudflare Pro ($20/mo)** — Free tier already covers DNS, TLS, DDoS and
  caching. Pro adds WAF rules and image optimization, neither of which binds at
  zero traffic. (A "$5 Cloudflare" tier likely refers to Workers Paid, which a
  static site does not need.)

Combined those two would be $25/mo — over budget — for capacity that isn't the
constraint.

### VM sizing

The pipeline DB is 138 MB and the GNN is not trained on this box (Kaggle does
that). So the VM only serves HTTP + runs the nightly chain:

- 2 vCPU / 4 GB / 40 GB disk is comfortable
- Hetzner CAX11 (ARM) or CX22, Fly.io, or Railway all fit the budget
- ARM is fine — no CUDA needed here

---

## 5. Open questions before deploying

1. **Where does the DB live?** Currently a local SQLite file. On one VM that's
   fine. It becomes the single point of failure — needs a backup cron to object
   storage (Cloudflare R2 free tier fits).
2. **Does the chain run on the VM or locally?** Running it on the VM keeps the
   served data fresh, but 40+ network sources on a small box is slow. Acceptable
   at 18:00 UTC once daily.
3. **Secrets on the VM.** `.env` currently holds live-ish Paddle credentials plus
   several unrelated tokens (WandB, HF, GitHub, Kaggle, Anthropic). The VM
   should get a **minimal** env — Paddle + FRED + FIRMS + EIA only. Do not copy
   the whole file.
4. **TLS for `api.`** — Caddy gets certs automatically and is simpler than nginx
   here; or terminate at Cloudflare with an origin cert.

---

## 6. Not addressed here

- Retrain (see `intelligence_layer_reactivation.md`) — the prediction tier can't
  ship until it's done.
- Whether the data carries edge. Nothing has been evaluated. This gates the
  prediction tier entirely, and is a separate research question.
