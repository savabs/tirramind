---
title: Spec — Production Deployment
tags:
  - doc/spec
  - topic/deployment
  - status/active
---

# Spec — Production Deployment

Research: `docs/research/production_deployment.md`
Task: `tasks/active/production_deployment.md`

Topology: Cloudflare (DNS, free) + Cloudflare Pages (static site, free,
commercial-safe) + a paid VM (backend, ~$20-40/mo) + Cloudflare R2 (DB backups).
No Vercel. No Cloudflare Pro.

---

## Step 1 — Minimal production secrets

**Problem:** `tirra-api.service` currently points `EnvironmentFile` at
`/opt/tirramind/.env` — the same file that holds `WANDB_API_KEY`, `HF_TOKEN`,
`GITHUB_TOKEN`, `KAGGLE_API_TOKEN`, `TIRRA_LLM_API_KEY` (Anthropic). None of
those belong on the internet-facing API server. A compromised web process
should not be able to leak training/CI credentials.

**Fix:** `deploy/env.production.example` — only the vars the API server and
chain actually read: Paddle (mode/api key/client token/webhook secret/tier
map), `TIRRA_SUB_KEYS`, `TIRRA_INGEST_TOKEN`, `TIRRA_FRED_API_KEY`,
`TIRRA_NASA_FIRMS_KEY`/`FIRMS_API_KEY`, `TIRRA_EIA_API_KEY`, `TIRRA_BUY_URL`.
systemd units point at `/opt/tirramind/.env.production`, not `.env`.

**Verify:** grep the template against `brief_server.py`/`agent/payments/*` env
reads — every var the server touches is present; nothing else is.

---

## Step 2 — Reverse proxy + TLS (Caddy)

`brief_server.py` binds `127.0.0.1:8787` only (already correct — see
`tirra-api.service`). Caddy terminates TLS for `api.tirramind.com` and proxies
to it. Caddy auto-provisions Let's Encrypt certs; no manual cert management.

**File:** `deploy/Caddyfile`

**Verify:** `caddy validate --config deploy/Caddyfile`

---

## Step 3 — VM bootstrap script

**File:** `deploy/provision_vm.sh` — idempotent, safe to re-run:
1. Create `tirra` system user + `/opt/tirramind`
2. Install Python 3.12, git, Caddy (official apt repo)
3. Clone/pull the repo, create venv, install deps
4. Install all `deploy/systemd/*` units, `daemon-reload`
5. Install `deploy/Caddyfile`, reload Caddy
6. Print a checklist of what the operator still must do by hand (DNS record,
   place `.env.production`, place backup credentials)

Explicitly does NOT: create the VM itself, touch DNS, or handle secrets — those
need human judgment/credentials I don't have.

**Verify:** running twice produces no errors and no duplicate units.

---

## Step 4 — DB backups to Cloudflare R2

The pipeline DB is a single 138 MB SQLite file on one disk — the one true
single point of failure in this topology.

**File:** `deploy/backup_to_r2.sh` — `aws s3 cp` (R2 is S3-API-compatible) of
`pipeline.db`, `subscribers.json`, `usage.db` to an R2 bucket, timestamped,
with retention pruning. New systemd `tirra-backup.service` + `.timer` (daily,
offset from the collection chain).

**Verify:** dry-run against a local MinIO or `--dry-run` flag; confirm pruning
logic keeps N most recent and deletes older ones correctly (unit test the
pruning function in isolation, not via a real R2 call).

---

## Step 5 — Static site on Cloudflare Pages

Cloudflare Pages uses a `_redirects` file (Netlify-compatible syntax), not
`vercel.json`/`netlify.toml`. Status `200` on an absolute destination proxies
(rewrite); other codes redirect.

**File:** `products/brief_subscription/_redirects`

Mirrors the existing `netlify.toml` rewrites: `/brief.json`, `/brief.md`,
`/api/v1/*`, `/evidence/*` → `api.tirramind.com`, plus `/buy` → `/pricing`.

`vercel.json`/`netlify.toml` are left in place (harmless, unused) rather than
deleted — no reason to lose them if either vendor is reconsidered later.

**Verify:** Cloudflare Pages dashboard build preview shows the redirects
active; manual check of `/buy` and one proxied path once live.

---

## Step 6 — Operator runbook

**File:** `tasks/active/production_deployment.md` — the actual checklist of
human actions (create the VM, add the Cloudflare Pages project, add the DNS
record, provision R2 + its API token, run the bootstrap script, place secrets,
verify).

---

## Out of scope

- Actually provisioning the VM, DNS records, or R2 bucket — requires the
  operator's own cloud accounts and credentials.
- Migrating SQLite to a managed database — not warranted at this scale/traffic.
- Monitoring/alerting (Sentry, uptime checks) — flagged as worth doing, not
  blocking first deploy.
