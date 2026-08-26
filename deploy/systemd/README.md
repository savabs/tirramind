# systemd units — production scheduling

Independent, always-additive pieces. All assume the repo is deployed to
`/opt/tirramind` as user `tirra`, matching the `WorkingDirectory`/`User`
fields — edit those if your deployment differs.

| Unit | What it does | Cadence |
|---|---|---|
| `tirra-api.service` | Long-running API server: `/webhook`, `/brief*`, `/api/v1/*`, `/evidence/*` | Always on (`Restart=on-failure`) |
| `tirra-brief.service` + `.timer` | Fast refresh (CFTC + gov contracts) → build → deliver the brief | Every 30 min |
| `tirra-chain.service` + `.timer` | **Collect + every downstream DAG in dependency order** | Weekdays 18:00 UTC |
| `tirra-collect.service` + `.timer` | `daily_collection` only — 40+ public data sources | Weekdays 18:00 UTC |
| `tirra-backup.service` + `.timer` | Snapshot the pipeline DB + subscriber/usage state to Cloudflare R2 (`deploy/backup_to_r2.sh`) | Daily 22:00 UTC |

## Pick `tirra-chain` OR `tirra-collect`, not both

They overlap: `chain` runs `daily_collection` as its first step. Enable
**`tirra-chain`** unless you specifically want collection without any downstream
processing.

`tirra-collect` alone leaves layers 2–6 completely empty. Each of the other 10
DAGs declares its own cron schedule, but those only fire under a long-running
`PipelineScheduler.start()` process — and nothing in production ever started
one. The result, verified against `dag_runs`: **8 of 11 DAGs had never executed
even once**, so `signals`, `beliefs`, `entity_alerts`, `convergence_clusters`,
`rl_transitions`, `portfolio_weights` and `paper_trade_pnl` sat at zero rows
while collection filled `entity_observations` with 365k.

`tirra-chain` runs them in **dependency order** rather than by wall-clock,
because cron cannot express "after upstream actually succeeded" — and the chain
has real cold-start dependencies (`rl_training` needs alerts+beliefs;
`inference` needs a SAC checkpoint only `rl_training` produces; `rl_transitions`
only materialises on the *second* consecutive `inference` run).

It exits non-zero if any DAG failed, so `systemctl status` shows the failure.

`tirra-collect`/`tirra-chain` are intentionally separate from `tirra-brief`: the full DAG is
slow (network calls to 40+ APIs, can take several minutes) and only needs to
run once a day, on the cadence documented in
`agent/pipeline/dags/daily_collection.py`. The brief only needs the two fast
tools (CFTC + gov contracts) refreshed frequently — bundling them would mean
either running the slow DAG every 30 minutes (wasteful, rate-limit risk) or
serving a stale brief for a full day (defeats the point).

## Install

```bash
sudo cp deploy/systemd/*.service deploy/systemd/*.timer /etc/systemd/system/
sudo systemctl daemon-reload

sudo systemctl enable --now tirra-api.service
sudo systemctl enable --now tirra-brief.timer

# Full pipeline (recommended) — collect AND all downstream intelligence layers:
sudo systemctl enable --now tirra-chain.timer

# ...OR collection only, if you deliberately want no downstream processing:
# sudo systemctl enable --now tirra-collect.timer

# DB/subscriber-state backups to R2 — needs /opt/tirramind/.env.backup in
# place first (see deploy/backup_to_r2.sh); without it every run fails closed
# (missing R2 credentials), once per day, until the file is there:
sudo systemctl enable --now tirra-backup.timer
```

## Verify

```bash
systemctl status tirra-api.service
systemctl list-timers 'tirra-*'
journalctl -u tirra-chain.service -n 100    # after its first scheduled run
```

The chain prints per-DAG row deltas, so a healthy run shows rows actually
landing rather than just "completed":

```
[chain] ── world_model_update ──
[chain]    status=completed  nodes_ok=1/1
      +23 beliefs
```

To run it by hand (or to backfill), skipping the slow collection step:

```bash
./scripts/run_scheduled.sh chain --skip-collection
.venv/bin/python scripts/run_chain.py --dry-run          # show the plan
.venv/bin/python scripts/run_chain.py --only inference   # one DAG
```

## Reverse proxy

`tirra-api.service` binds `127.0.0.1:8787` only. Put nginx/Caddy in front for
TLS + the public `api.tirramind.com` hostname — this repo doesn't manage that
layer.
