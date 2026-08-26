---
title: Runbook — Production Deploy
tags:
  - doc/runbook
  - topic/deployment
  - status/active
---

# Runbook — Production Deploy

Follow this top to bottom. Research: `docs/research/production_deployment.md`.
Spec: `docs/specs/production_deployment_spec.md`. This runbook supersedes both
on cost — the budget is **$30/month**, not the $10 originally assumed (see
§0.2).

Every code/config artifact this runbook uses already exists in the repo and
was re-verified while writing this runbook (2026-08-26): `deploy/Caddyfile`,
`deploy/provision_vm.sh`, `deploy/env.production.example`,
`deploy/backup_to_r2.sh`, `deploy/systemd/*`, `products/brief_subscription/_redirects`.
You are not writing anything new — only creating accounts, running scripts,
and pasting values into files.

**Legend:** 🧑 = only you can do this (account/payment/click-through). 💻 = a
command to run, copy-pasted exactly. ✅ = a verification step.

---

## 0. Before you start

### 0.1 What you're deploying

```
                    Cloudflare (DNS + TLS, free)
                             │
        ┌────────────────────┴────────────────────┐
        │                                          │
  tirramind.com                             api.tirramind.com
  Cloudflare Pages (free)                   Hetzner VM
  static: index / pricing / terms           Caddy (TLS) → brief_server.py :8787
  Paddle.js overlay checkout                 systemd: tirra-api, tirra-chain.timer,
                                                       tirra-backup.timer
                                             SQLite pipeline.db
                                                   │
                                                   ▼
                                          Cloudflare R2 (free) — nightly backup
                                                   │
                                          HetrixTools (free) — uptime alert
```

No Vercel. No Vercel Pro. Vercel's Hobby tier is non-commercial-use-only and
Paddle is now live, so staying on Vercel at all is a ToS problem, not just a
cost one — hence moving the static site to Cloudflare Pages, whose free tier
explicitly permits commercial use.

### 0.2 Budget — $30/month, current prices (checked 2026-08-26)

| Item | Choice | Cost/mo | Why |
|---|---|---|---|
| DNS | Cloudflare (free plan) | $0 | Already the plan |
| Static site | Cloudflare Pages (free plan) | $0 | Commercial-use-safe, unlike Vercel Hobby |
| **VM** | **Hetzner CAX21** — 4 vCPU (Ampere ARM) / 8 GB RAM / 80 GB NVMe | **~€10.49 (~$11.50)** | See §0.3 |
| VM public IPv4 | Hetzner add-on (Hetzner now charges for it) | ~€0.50 (~$0.55) | Needed for DNS `A` record |
| Backups | Cloudflare R2 (free tier: 10 GB storage, unlimited egress) | $0 | 138 MB DB, gzipped, × 14 daily snapshots stays well under 10 GB |
| Uptime monitoring | **HetrixTools** free plan | $0 | See §0.4 |
| **Total** | | **~$12/month** | ~$18/mo of the $30 budget deliberately unused — see below |

**Why CAX21 and not something bigger:** the original research (written for a
$10/mo budget) picked CAX11 (2 vCPU/4 GB, €5.99/mo post price-rise) — the
smallest tier that runs the API and nightly chain at all. At $30/mo there's
real room to remove risk cheaply: CAX21 doubles CPU and RAM for about $5.50/mo
more, which matters specifically because `tirra-chain.service` does GNN
inference on CPU (no CUDA on this box — training happens on Kaggle) back to
back with 40+ concurrent network fetches; doubling headroom reduces the
chance a slow night runs long enough to overlap the next day's chain, or gets
OOM-killed mid-DAG. It is **not** worth going further to a CPX-line box
(€19.49+/mo as of Aug 2026) — that line's price roughly tripled in the June
2026 Hetzner price hike and buys dedicated-vCPU throughput this workload
(bursty, not sustained, no concurrent user traffic beyond a handful of API
subscribers) doesn't need. The unused ~$18/mo of headroom is left unspent
deliberately (CLAUDE.md §7) — nothing else here is reliability-justified.

Hetzner raised Cloud prices twice in 2026 (April and again 15 June) — some
tiers 30–200%+. If you're reading this much later, re-check
<https://www.hetzner.com/cloud/> before committing; the CAX line is still the
cheapest per-vCPU option as of this writing but that has shifted before.

### 0.3 Why not the CX (x86) line instead

Hetzner's June 2026 renaming makes this confusing: CX23 (2 vCPU x86, 4 GB) is
now priced *below* CAX11 (€5.49 vs €5.99) — ARM is no longer automatically
cheaper. There is no published x86 tier at CAX21's spec for less. Two reasons
ARM (CAX) is still the pick here:
- It's what the existing research/spec already assumed, and nothing about the
  workload needs x86.
- The one real risk of ARM is a niche Python wheel (`mambapy`, `torchcde`,
  `torchsde`, `torch-geometric`) lacking an `aarch64` prebuilt wheel and
  falling back to a slow source build on first `pip install`. If
  `provision_vm.sh` hangs or fails during dependency install on this VM
  specifically, that's the likely cause — the fix is `apt-get install -y
  rustc cargo` (some builds need a Rust toolchain) or, if it's truly stuck,
  switching the server type to a same-priced/cheaper CX box in the Hetzner
  console (no reinstall needed, just a resize) and re-running
  `provision_vm.sh`.

### 0.4 Why HetrixTools and not UptimeRobot

UptimeRobot's free plan changed its ToS in Dec 2024: free is now
**personal/non-commercial use only** — the same trap as Vercel Hobby. Its paid
tier starts at ~$13/mo. HetrixTools' free plan (15 monitors, 1-minute
checks, email/Slack/Discord/Telegram alerts) has **no commercial-use
restriction** and is a strict upgrade over UptimeRobot's paid tier at $0/mo —
so it's the pick regardless of budget headroom. Re-verify its terms at
<https://hetrixtools.com/pricing/> before signing up; free-tier ToS changes
without much notice (see UptimeRobot).

---

## 1. 🧑 Accounts, DNS, and the VM — only you can do these

Do these in order; later steps depend on earlier ones.

### 1.1 Cloudflare — DNS + Pages

**Steps 1-3 below are already done** — verified live 2026-08-27:
`dig +short NS tirramind.com` returns `gabriel.ns.cloudflare.com` /
`autumn.ns.cloudflare.com`, so the zone is already on Cloudflare. No account
creation or nameserver change needed. Skip straight to step 4.

1. ~~Create a free Cloudflare account~~ — already done, the zone exists.
2. ~~Add `tirramind.com` as a site~~ — already done.
3. ~~Switch nameservers at the registrar~~ — already done and propagated.
4. **Pages project:** Cloudflare dashboard → Workers & Pages → Create →
   Pages → Connect to Git → pick this repo → set:
   - Build output directory: `products/brief_subscription`
   - Build command: (leave blank — it's a static folder, no build step)
   - Root directory: `/` (repo root, so `_redirects` — which lives inside
     `products/brief_subscription/` — is picked up from the output directory
     above)
   Deploy. Pages gives you a `*.pages.dev` URL immediately — confirm the site
   loads there, and that `/pricing` shows the four real tiers, before wiring
   the real domain.
5. Pages project → Custom domains → Add `tirramind.com` and `www.tirramind.com`.
   **Note:** the domain currently points at a Vercel project (the live-fix
   stopgap from 2026-08-26 — see docs/research/production_deployment.md). Adding
   it here will have Cloudflare either prompt to update the existing DNS
   records or add conflicting ones — let Cloudflare replace the record
   pointing at Vercel with the one pointing at Pages; don't leave both. This
   is the actual cutover moment, so re-verify `/pricing` immediately after.

**Verify (✅ 1.1):** `https://tirramind.com` loads the storefront over valid
TLS, and `/pricing` shows the four real tiers.

### 1.2 Hetzner — the VM

1. Create an account at <https://accounts.hetzner.com> if you don't have
   one, and add a payment method under Cloud → your project → Billing.
2. Cloud Console → your project → **Add Server**:
   - **Location:** whichever region is closest to you/your users (Falkenstein
     or Helsinki, EU, are cheapest; Ashburn/Hillsboro, US, cost slightly more)
   - **Image:** Debian 12 (bookworm) — `provision_vm.sh` targets Debian/Ubuntu apt
   - **Type:** Shared vCPU → **Arm64** tab → **CAX21** (4 vCPU / 8 GB / 80 GB) — **~€10.49/mo**
   - **Networking:** leave the public IPv4 checkbox on (it now costs
     ~€0.50/mo extra — this is the only way DNS can point at the box)
   - **SSH keys:** add your public key here (Add SSH Key → paste
     `~/.ssh/id_ed25519.pub` or generate one first with `ssh-keygen -t
     ed25519`) — do **not** rely on the emailed root password
   - **Name:** anything, e.g. `tirramind-api`
   - Create & Buy Now
3. Note the server's public IPv4 address, shown in the console after it
   boots (~30 seconds).

**Verify (✅ 1.2):** `ssh root@<the-ip>` connects without a password prompt.

### 1.3 DNS — point the API subdomain at the VM

Cloudflare dashboard → `tirramind.com` → DNS → Add record:
- Type: `A`, Name: `api`, IPv4 address: `<hetzner-vm-ip>`, **Proxy status:
  DNS only (grey cloud)**.

Use grey-cloud (not orange/proxied) for `api.tirramind.com`. Caddy handles its
own TLS via Let's Encrypt on this box; orange-clouding would additionally
require setting Cloudflare's SSL/TLS mode to **Full (strict)** (Flexible
breaks the ACME challenge and leaves the Cloudflare→origin hop unencrypted) —
grey-cloud sidesteps that whole failure mode and is simpler for an API host
with no need for Cloudflare's CDN/WAF layer.

**Verify (✅ 1.3):** `dig +short api.tirramind.com` returns the VM's IP
(may take a few minutes to propagate).

### 1.4 Cloudflare R2 — backup bucket + API token

1. Cloudflare dashboard → R2 → **Create bucket** → name it
   `tirramind-backups` (or anything; you'll put the real name in
   `.env.backup`). Any location hint is fine — R2 is not region-pinned the
   way S3 is.
2. R2 → **Manage API Tokens** → **Create API Token**:
   - Permissions: **Object Read & Write**
   - Scope it to the one bucket you just created (not "Apply to all buckets")
   - TTL: no expiry (or set a reminder to rotate it)
3. Copy the three values Cloudflare shows you **once**: Access Key ID, Secret
   Access Key, and the Account ID (shown in the R2 overview page's sidebar,
   or in the token creation confirmation). You'll paste these into
   `.env.backup` in §2.

**Verify (✅ 1.4):** the bucket exists and shows 0 objects; you have all
three credential values saved somewhere (password manager, not this repo).

### 1.5 HetrixTools — free uptime monitor

1. Sign up free at <https://hetrixtools.com/>.
2. Add Monitor → Uptime Monitor → URL: `https://api.tirramind.com/status` →
   check interval: 1 minute (free tier default) → save.
3. Notifications → add your email (and Slack/Discord/Telegram if you use
   one) as an alert contact, attach it to the monitor.

This is the **only** thing standing between an outage and "a paying customer
tells you" — there is currently no alerting of any kind. Don't skip it.

**Verify (✅ 1.5):** the monitor shows red/down right now (nothing is live
yet) — that's expected. It should flip green once §3 is done.

### 1.6 Paddle — already done, nothing to do here

Merchant verification and live webhook signature handling were finished this
session. The only Paddle-related step left is pasting the **live** keys into
`.env.production` in §2 — don't recreate products; `pri_01m0xg0...` price IDs
already exist and are already in `pricing.html`.

---

## 2. 💻 Provisioning — commands to run, in order

SSH into the VM for all of this (`ssh root@<hetzner-vm-ip>`).

### 2.1 Clone the repo and run the bootstrap script

```bash
git clone https://github.com/<you>/tirramind /opt/tirramind
bash /opt/tirramind/deploy/provision_vm.sh
```

This installs Python, Caddy, creates the `tirra` system user, sets up the
venv, installs every systemd unit, and installs the Caddyfile — then prints a
checklist of exactly the remaining manual steps (§2.2–2.5 below, restated).
It is idempotent: re-run it any time after a `git pull` with no ill effects.

**If it fails partway through a `pip install`** on an ARM-specific package,
see §0.3 — it's very likely a missing prebuilt wheel, not a real error.

### 2.2 Production secrets

```bash
cp /opt/tirramind/deploy/env.production.example /opt/tirramind/.env.production
nano /opt/tirramind/.env.production
```

Fill in, using your Paddle **live** dashboard and the free API keys you may
already have registered:

- `TIRRA_PADDLE_API_KEY`, `TIRRA_PADDLE_CLIENT_TOKEN`,
  `TIRRA_PADDLE_WEBHOOK_SECRET`, `TIRRA_PADDLE_PRICE_ID` — from Paddle's live
  dashboard (Developer Tools → Authentication / Notifications)
- `TIRRA_TIER_PRICE_MAP` — the four price IDs already in `pricing.html`:
  `pri_01m0xg0kmxgwka7tafgs28qkwp:data,pri_01m0xg0n0q1aw68qnzk97et1ff:entity,pri_01m0xg0pv444hfvy45fadzf74n:scheduler,pri_01m0xg0rn6w31kh6bhf9k39sk8:brief`
- `TIRRA_INGEST_TOKEN` — invent a long random string (`openssl rand -hex 32`)
- `TIRRA_FRED_API_KEY`, `TIRRA_NASA_FIRMS_KEY`, `FIRMS_API_KEY` (same value as
  the FIRMS key), `TIRRA_EIA_API_KEY` — free registrations, see
  `.env.example` for the signup URLs, if not already obtained
- Leave `TIRRA_REQUIRE_AUTH=1` — do not change this

Then lock it down:

```bash
chown tirra:tirra /opt/tirramind/.env.production
chmod 600 /opt/tirramind/.env.production
```

**Do not** copy your development `.env` here — it holds WandB/HuggingFace/
GitHub/Kaggle/Anthropic keys that must never reach an internet-facing process.

### 2.3 Backup credentials

```bash
nano /opt/tirramind/.env.backup
```

```
R2_ACCOUNT_ID=<from §1.4>
R2_ACCESS_KEY_ID=<from §1.4>
R2_SECRET_ACCESS_KEY=<from §1.4>
R2_BUCKET=tirramind-backups
BACKUP_KEEP=14
```

```bash
chown root:root /opt/tirramind/.env.backup
chmod 600 /opt/tirramind/.env.backup
```

Then dry-run it before trusting it:

```bash
sudo -u tirra /opt/tirramind/deploy/backup_to_r2.sh --dry-run
```

It should print `DRY RUN — would upload to s3://tirramind-backups/...` and
exit 0, touching nothing.

### 2.4 Seed the pipeline database

The graph took months to accumulate; a fresh VM cannot rebuild it (most
upstream APIs only serve a recent window). From your **working machine**
(not the VM):

```bash
rsync -avz .tirra_pipeline/pipeline.db root@<hetzner-vm-ip>:/opt/tirramind/.tirra_pipeline/
ssh root@<hetzner-vm-ip> "chown tirra:tirra /opt/tirramind/.tirra_pipeline/pipeline.db"
```

### 2.5 Start everything

Back on the VM:

```bash
systemctl enable --now tirra-api.service
systemctl enable --now tirra-chain.timer
systemctl enable --now tirra-backup.timer
systemctl enable --now tirra-disk-check.timer
```

Skip `tirra-collect.timer` — `tirra-chain` already runs collection as its
first step; enabling both means running the 40+-source collection twice a
day for no reason.

`tirra-disk-check.timer` needs no secrets — `provision_vm.sh` already enables
it for you, this line is only here in case you're wiring units up by hand.

---

## 3. ✅ Final verification checklist

Run every one of these from **outside** the box (your own machine), not on
the VM itself — "the process started" is not the same as "the internet can
reach it."

```bash
# TLS + the API is actually reachable through the real domain
curl -sI https://api.tirramind.com/status
# expect: HTTP/2 200 (or your server's real status code — anything but a
# connection error / cert error)

# The storefront proxies through to the API correctly
curl -sI https://tirramind.com/brief.json
# expect: same status as calling api.tirramind.com/brief.json directly —
# confirms the Cloudflare Pages _redirects rule is live

# Webhook path resolves (Paddle will actually be able to reach it)
curl -sI https://api.tirramind.com/webhook
```

```bash
# On the VM: services are actually running, not crash-looping
systemctl status tirra-api.service
systemctl list-timers 'tirra-*'
# expect: tirra-chain.timer and tirra-backup.timer both listed with a real
# NEXT time, not "n/a"
```

```bash
# After the first scheduled 18:00 UTC chain run:
journalctl -u tirra-chain.service -n 100
# expect per-DAG row deltas like "+23 beliefs" — NOT just "status=completed"
# with zero rows landing (see LESSONS: "completed" is not the same claim as
# "wrote rows")
```

```bash
# After the first scheduled 22:00 UTC backup run:
journalctl -u tirra-backup.service -n 50
# expect: "[backup] uploaded s3://tirramind-backups/<timestamp>/ (N files)"
```

- [ ] `https://tirramind.com` loads over valid TLS
- [ ] `https://tirramind.com/pricing` shows the real Paddle checkout (test a
      purchase in Paddle sandbox mode first if you haven't already end-to-end
      tested the live webhook this session's payments work covered)
- [ ] `https://api.tirramind.com/status` returns 200 over valid TLS
- [ ] `tirra-api.service` is `active (running)`, not restarting
- [ ] `tirra-chain.timer` has fired at least once and its journal shows real
      row deltas, not zeros
- [ ] `tirra-backup.timer` has fired at least once and R2 shows one
      timestamped folder in the bucket (Cloudflare dashboard → R2 → the
      bucket)
- [ ] HetrixTools monitor shows **up** (green), not down
- [ ] `deploy/backup_to_r2.sh --dry-run` still exits 0 (re-check any time you
      touch backup config)

Once every box above is checked, this is genuinely live — not "deployed" in
the sense of "the process exists," but in the sense that a customer who pays
right now gets a working key against a running, backed-up, monitored system.
