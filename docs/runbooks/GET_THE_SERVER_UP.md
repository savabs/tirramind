---
title: "Get the server up — 15 minutes, ~$5/mo"
tags:
  - doc/runbook
  - topic/infrastructure
  - status/current
date: 2026-09-26
---

# Get the server up

**Why this is the priority.** Sources split into *backfillable* (the API serves
history, so a gap can be refilled later) and *time-gated* (the API only serves
recent data, so every day the collector is down is data that can never be
recovered at any price). Four sources are genuinely time-gated:

| source | why it cannot be re-fetched |
|---|---|
| `whale_alert` | reads the Bitcoin **mempool** — pending transactions that vanish once confirmed |
| `dns_monitor` | current DNS state, diffed against its own prior scan |
| `options_chain` | open-interest / IV snapshots; historical chains are not freely available |
| `finra_short_volume` | API window caps at **20 trading days** |

Collection was already dark from **2026-08-27 to 2026-09-23** — 26 days, gone
permanently. A laptop that sleeps cannot hold this. That is the whole argument.

---

## Step 1 — create the VM (only you can do this)

Cheapest tier works; collection is network-bound, not CPU-bound.

| | |
|---|---|
| spec | **1 vCPU, 1 GB RAM, 25 GB disk** |
| OS | **Debian 12** or Ubuntu 22.04+ |
| cost | ~$5/mo — Hetzner CX22 (~€4), DigitalOcean, Vultr, Linode |
| region | EU or US. Latency is irrelevant for a daily batch |

The DB is ~170 MB today, so 25 GB is years of headroom.
`provision_vm.sh` adds a swapfile precisely so 1 GB cannot OOM.

## Step 2 — provision (one command)

```bash
ssh root@<your-ip>
git clone https://github.com/savabs/tirramind /opt/tirramind
bash /opt/tirramind/deploy/provision_vm.sh
```

Idempotent — safe to re-run after any repo update. It installs python/sqlite,
creates the `tirra` user, hardens the box (unattended security upgrades,
deny-by-default ufw with only 22/80/443, a journald cap, a disk-space monitor),
and installs the systemd units.

It deliberately does **not** create DNS, place secrets, or restrict SSH to your
IP — it prints a checklist for those at the end. **Do step 1a on that checklist
(narrow SSH to your IP); port 22 is open to the world until you do.**

## Step 3 — two free keys, and nothing else

```bash
cp /opt/tirramind/deploy/env.production.example /opt/tirramind/.env.production
nano /opt/tirramind/.env.production
```

Only these two matter for collection, both free with instant signup:

```
TIRRA_FRED_API_KEY=...     # https://fred.stlouisfed.org/docs/api/api_key.html
TIRRA_EIA_API_KEY=...      # https://www.eia.gov/opendata/register.php
```

**Skip every `TIRRA_PADDLE_*` variable.** That is the retired checkout. The only
Paddle credentials on the laptop are sandbox keys and they must not go near a
public host — the `security/withdraw-paddle-exposure` branch exists to keep it
that way.

## Step 4 — start the timers

```bash
systemctl enable --now tirra-collect.timer             # weekday market sources
systemctl enable --now tirra-collect-continuous.timer  # DAILY, time-gated sources
systemctl enable --now tirra-freshness.timer           # the watchdog
systemctl list-timers 'tirra-*'
```

The split matters. `tirra-collect.timer` runs `Mon..Fri 18:00 UTC`, which is
right for CFTC and FINRA (they do not publish at weekends) and **wrong** for the
continuous time-gated sources — weekday-only silently discards ~28% of the only
data that constitutes a moat.

## Step 5 — prove it actually wrote rows

```bash
systemctl start tirra-collect.service     # run once by hand
journalctl -u tirra-collect.service -n 50 --no-pager
/opt/tirramind/.venv/bin/python /opt/tirramind/scripts/check_freshness.py
```

**Green is not evidence — row counts are.** A collector that runs and writes
nothing looks identical to one that runs and finds nothing; that is exactly how
26 days went unnoticed. `check_freshness.py` exits non-zero on any stale source
and shouts louder for the time-gated ones.

---

## Two things to decide once it is up

**Where the DB lives.** The laptop currently holds the only copy: 417,593
observations, ~170 MB, including the 34,921-row form144 backfill. Either the
server becomes the primary and the laptop syncs from it, or the laptop stays
primary and the server writes a second stream that gets merged. **Pick one
before both are collecting** — two independent writers producing rows that must
later be reconciled is a migration nobody wants to run. `deploy/backup_to_r2.sh`
exists for offsite copies and needs R2 credentials.

**Whether training moves.** It should not. Measured on the laptop: the graph
workload is *3× slower* on MPS than CPU (small tensors, launch overhead
dominates), it uses 2 of 10 cores and 0.3 GB of 16 GB, and a 1 GB VM would be
worse at it. The server is for **collection uptime**, which is the moat. Training
stays local.

## Related

- [[what_tirramind_is_for]] — why time-gated data is the only uncopyable asset
- [[verification_business_plan]] — what the data is ultimately for
- [[production_deploy]] — the fuller deployment notes
