# systemd units — production scheduling

Independent, always-additive pieces. All assume the repo is deployed to
`/opt/tirramind` as user `tirra`, matching the `WorkingDirectory`/`User`
fields — edit those if your deployment differs.

| Unit | What it does | Cadence |
|---|---|---|
| `tirra-api.service` | Long-running API server: `/webhook`, `/brief*`, `/api/v1/*`, `/evidence/*` | Always on (`Restart=on-failure`) |
| `tirra-brief.service` + `.timer` | Fast refresh (CFTC) → build → deliver + archive the brief | Weekly, Mon 20:00 UTC |
| `tirra-chain.service` + `.timer` | **Collect + every downstream DAG in dependency order** | Weekdays 18:00 UTC |
| `tirra-collect.service` + `.timer` | `daily_collection` only — 40+ public data sources | Weekdays 18:00 UTC |
| `tirra-collect-continuous.timer` | Starts **`tirra-collect.service`** at the weekend — the other half of 7-day collection for the time-gated sources | Sat/Sun 18:00 UTC |
| `tirra-freshness.service` + `.timer` | Freshness watchdog: fails loudly when a source stops landing rows — data age, collector silence, **and day-over-day row growth** (`scripts/check_freshness.py`) | Daily 21:00 UTC |
| `tirra-backup.service` + `.timer` | Snapshot the pipeline DB + subscriber/usage state to Cloudflare R2 (`deploy/backup_to_r2.sh`) | Daily 22:00 UTC |
| `tirra-disk-check.service` + `.timer` | Warn (journal + optional email) when the disk holding the pipeline DB crosses 85% (`deploy/disk_space_check.sh`) | Daily 06:00 UTC |

## Collection runs 7 days a week, in two halves

`tirra-collect.timer` / `tirra-chain.timer` fire **Mon..Fri 18:00 UTC**, which is
right for the market-hours sources that dominate `daily_collection` — CFTC's
Friday COT release, FINRA short volume, `instrument_universe` closes,
`options_chain`. None of them publish at the weekend.

It is wrong for the *continuous, time-gated* sources in the same DAG:
`ais_vessel`, `whale_alert`, `dns_monitor`, `cert_transparency`,
`finra_short_volume`, `form144`, `political_risk`, `academic_preprints`,
`regulatory_gazette`, `bankruptcy_court`, `options_chain`, `global_pmi` — the
`TIME_GATED` set in `scripts/check_freshness.py`. Their APIs serve a rolling
recent window, so a day not collected is a day that cannot be bought back at any
price. Weekday-only collection was silently discarding two days in seven of
exactly the data this project's moat is made of.

`tirra-collect-continuous.timer` fills the weekend. It carries no `ExecStart` of
its own — `Unit=tirra-collect.service` starts the existing service — so it works
whether you enabled `tirra-chain.timer` or `tirra-collect.timer` for weekdays,
and "how collection runs" stays defined in one file.

**It runs the whole DAG, not just the time-gated nodes, because a node subset
cannot be invoked.** `DAGExecutor.execute()` takes only `(dag, trigger)`;
`scripts/run_collection.py` exposes `--db-path/--workers/--dag`;
`scripts/run_chain.py --only` selects whole DAGs. The one node-level lever,
`DAGNode.enabled`, is set programmatically by the `ToolRoutingBandit` inside
`build_daily_collection_dag()` and no CLI reaches it.

**What the weekend run costs — measured, not assumed.** An earlier version of
this section said the weekday sources dedup away and cost "network time, not
rows". That is false for two of them. `instrument_universe` stamps `observed_at`
with the *run* date (`as_of=date.today()`, `agent/tools/instrument_universe.py`
:879/:907) and `options_chain` falls back to `time.time()` with a
`fetched_at`-of-today inside `value_json` (`agent/tools/options_chain.py`
:105/:129/:210). The dedup key includes `observed_at` and `value_json`
(`agent/pipeline/store.py:1469-1479`), so neither can collapse. Measured
read-only on the live DB (2026-09-14..09-21): `instrument_universe` writes **89
rows per weekday run and 2 per weekend day** — the 2 are crypto, which really
does trade at weekends and goes through the bar-dated path. So each weekend run
adds ~87 equity rows carrying **Friday's close stamped Saturday or Sunday**,
plus ~8–32 `options_chain` rows with the same defect.

Enable it anyway, on an asymmetry: those rows are identifiable and deletable
(`instrument_universe` observations whose `observed_at` lands on a weekend),
while a missed AIS / CT-log / DNS window is gone at any price. The clean fix is
at the collector — pass the quote date into `persist_options_snapshot`, use the
bar's own date instead of `as_of=date.today()` — and it belongs to those Layer 1
files, not to a timer. `cftc` is already correct (`agent/tools/cftc.py:640`
derives `observed_at` from `report_date`).

**Sat/Sun, not a daily `*-*-*` timer.** The weekday slot is already collecting at
18:00 UTC; a daily timer would fire a second collection into the middle of the
chain's run — two concurrent writers on `.tirra_pipeline/pipeline.db`. The split
gives one collection run per calendar day **in steady state**. Do not widen
`tirra-collect.timer` to `Mon..Sun` while this timer is enabled, for the same
reason.

**"In steady state" is the honest limit, and nothing enforces it.** Two timers
on the *same* unit are safe — systemd coalesces the jobs — which is why
`tirra-collect.timer` and `tirra-collect-continuous.timer` can share
`Unit=tirra-collect.service`. Two timers on *different* units are not:
`tirra-chain.timer` has `Persistent=true`, so a boot after downtime queues its
missed chain run alongside whatever else is due. There is no lock anywhere
(`grep flock\|LOCK_EX\|Conflicts=` over `scripts/run_*.{py,sh}` and
`deploy/systemd/`: no matches), `PipelineStore`'s write guard is a
`threading.RLock` and says so itself ("does not make it atomic across
processes; only a UNIQUE index will", `store.py:1481-1486`), and the live table
has no UNIQUE index. Worse, a concurrent run defeats the executor's own
zero-rows guard: `DagRun` records `domain_rows_before` (`executor.py:225`) and
diffs it after (`:133-140`), so another process's inserts let a run that wrote
nothing report rows landed.

Two things close this, both outside the timer files:

- wrap the `collect` and `chain` branches of `scripts/run_scheduled.sh` in
  `flock -n /opt/tirramind/.tirra_pipeline/.collect.lock` so the second run
  refuses instead of interleaving, and/or
- add `Conflicts=tirra-chain.service` to `tirra-collect.service`.

Until then `tirra-collect-continuous.timer` deliberately carries **no
`Persistent=`**: a Monday boot after a weekend outage would otherwise start
`tirra-chain.service`'s catch-up and `tirra-collect.service`'s catch-up in the
same transaction, two `daily_collection` processes into one WAL database. The
catch-up buys little in exchange — the time-gated sources are point-in-time
probes, so a Monday catch-up samples *Monday*, exactly what Monday's scheduled
run already does. Restore `Persistent=true` once the lock exists.

**`tirra-collect.service` has no memory ceiling.** `tirra-chain.service:37-41`
caps itself at `MemoryMax=1200M` / `MemorySwapMax=1500M` on the strength of a
measured run (2026-08-27: 84% RSS, 1.24 GB swap, 1 vCPU / 1.9 GB box) and
records why — "a process OOM-killed mid-DAG leaves a run stuck in `running` with
no error recorded". `tirra-collect.service` carries neither directive, and in
the recommended chain deployment it never ran at all until this weekend timer
started it. Copy the same pair onto `tirra-collect.service`;
`daily_collection` is a strict subset of the chain's workload, so that ceiling
is headroom rather than a constraint.

## `tirra-freshness` — silence is the bug, so silence has to fail

Collection stopped on 2026-08-27 and nobody noticed for 26 days: nodes green,
collectors returning HTTP 200, and "no new rows today" is byte-for-byte
indistinguishable from "ran fine, nothing new upstream".

`tirra-freshness.timer` runs `scripts/check_freshness.py` daily at 21:00 UTC —
after the 18:00 collection run (which can take ~90 min) so it grades *today's*
run, and before the 22:00 backup.

**It runs in row-growth mode, not age-only mode.** The script answers three
questions, and only the third catches a partial write on the day it happens:
data age (`MAX(observed_at)`), collector silence (`MAX(ingested_at)`), and
"did a run that reported success actually write anything" — per-source
`COUNT(*)` then vs now, gated behind `--since-snapshot`. An age threshold is
satisfied by a *single* row, so a unit that passes only `--db` reports green
against a collector writing 1 row a day instead of thousands. The other two
clocks need 3–4 days of silence before they fire; this one fires the next
morning.

The baseline is a copy of the DB that the unit writes itself, via `ExecStopPost`
into `StateDirectory=tirramind` (`/var/lib/tirramind/freshness-snapshot.db`).
`ExecStopPost`, not `ExecStartPost`, because systemd skips `ExecStartPost` after
a non-zero `ExecStart` — and this unit is designed to exit non-zero, so the
baseline would otherwise freeze on the first red day. It is taken through
`mode=ro` + `PRAGMA query_only` using sqlite's backup API, staged at `.tmp` and
`os.replace`d into position, so the unit never opens the live DB writable and an
interrupted copy cannot read back as "every source lost rows". Budget ~165 MB in
`/var/lib` alongside `tirra-backup` and the 85% disk check.

`ExecStart` passes `--since-snapshot` only when the baseline is over 3 hours old
(`find -mmin +180`). Comparing the database against a copy of itself flags
`NO_NEW_ROWS` for every source that has ever written a row, so without that guard
`systemctl start tirra-freshness` twice in one collection cycle would produce a
red the operator caused by testing the unit. Below the threshold the run
degrades to the age-only check; it never reports green on evidence it did not
have.

What `NO_NEW_ROWS` does *not* flag: a collector that re-fetched an unchanged
record. That adds no row, but `store_entity_observation` refreshes `ingested_at`
on a duplicate, so it is not silent either — the script reports a source only
when **both** clocks are flat. Measured read-only on 2026-09-23, 31 of the 34
sources wrote on the most recent run; the 3 that did not (`cert_transparency`,
`academic_preprints`, `cftc_derived`) are dead collectors already 27–143 days
silent, so this flag has no false-alarm population to draw on.

Its exit code is the alarm and is deliberately not masked:

| exit | meaning |
|---|---|
| 0 | every source inside its threshold |
| 1 | a **backfillable** source is stale, silent, or wrote nothing — refill it |
| 2 | a **time-gated** source is stale, silent, or wrote nothing — unrecoverable |
| 3 | the watchdog itself could not run (missing/unreadable DB) |

1, 2 and 3 all leave the unit `failed`, so it appears in
`systemctl list-units --failed` with the per-source table in
`journalctl -u tirra-freshness`. **Do not add `SuccessExitStatus=` to tidy that
up** — that converts the alarm back into the silence it exists to break. On an
unattended box, add a notifier via `OnFailure=` instead.

Not every finding pages, and that is not a softening. `FUTURE_DATED` is printed
in the table and kept out of the exit code (`NON_PAGING_REASONS` in
`check_freshness.py`), because the live DB holds **912 `gov_contracts` rows
dated up to 2030-01-30** — verified read-only, 2026-09-23 — and a future
timestamp never ages out. Grading it would pin this unit to `failed` forever
regardless of collection health, and a unit that is red every day is
indistinguishable from one that is red for a reason. Those rows are still a
date-parsing bug worth repairing at the collector; the watchdog reports them
rather than paging on them.

The watchdog opens the pipeline database read-only (`mode=ro` URI plus
`PRAGMA query_only`) and needs no API keys, which is why its unit carries no
`EnvironmentFile`. The only thing it writes is its own baseline under
`/var/lib/tirramind`.

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
`agent/pipeline/dags/daily_collection.py`. The brief only needs the one fast
tool (CFTC positioning) refreshed before it builds — bundling them would mean
either running the slow DAG on the brief's own cadence (wasteful, rate-limit
risk) or serving a stale brief (defeats the point).

`tirra-brief` runs **weekly** (Mon 20:00 UTC, matching the "weekly" cadence
promised in the customer-facing copy — index.html, pricing.html, terms.html,
`products/brief_subscription/README.md`), not every 30 minutes: see
`deploy/systemd/tirra-brief.timer` for why that specific day/time. Each
delivery is also archived to `.tirra_delivery/archive/intelligence_brief_<UTC-date>.{json,md}`
(plus an append-only `archive/index.jsonl`) by `scripts/tirra_engine.py`'s
`_archive_delivery()`, run right after `BriefDeliverer.deliver()` writes the
mutable "latest" `intelligence_brief.{json,md}` that `agent/brief_server.py`
serves — so a subscriber can retrieve a past edition instead of only ever
seeing whatever the mutable file currently holds.

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

# Weekend half of collection — enable this alongside EITHER of the two above.
# Without it the time-gated sources lose two days in seven, permanently.
#
# It starts tirra-collect.service, which (unlike tirra-chain.service) has NO
# MemoryMax/MemorySwapMax. On the 1 vCPU / 1.9 GB box, add the same ceiling to
# tirra-collect.service before enabling this, or the weekend runs are the only
# uncapped DAG runs on the machine — see "tirra-collect.service has no memory
# ceiling" above. It also adds ~87 run-dated instrument_universe rows per
# weekend day until the collector provenance is fixed:
sudo systemctl enable --now tirra-collect-continuous.timer

# Freshness watchdog — no secrets, safe to enable immediately. Reads the
# pipeline DB read-only; writes only its own row-count baseline into
# /var/lib/tirramind (StateDirectory=, created by systemd, ~165 MB).
# The FIRST run has no baseline, so it is an age-only check; row-growth
# comparison starts on the second run, 24h later.
# Expect it to fail on its first run if collection has been down; that is the
# unit working, not the unit broken:
sudo systemctl enable --now tirra-freshness.timer

# DB/subscriber-state backups to R2 — needs /opt/tirramind/.env.backup in
# place first (see deploy/backup_to_r2.sh); without it every run fails closed
# (missing R2 credentials), once per day, until the file is there:
sudo systemctl enable --now tirra-backup.timer

# Disk-space monitor — no secrets required, safe to enable immediately.
# deploy/provision_vm.sh already does this for you.
sudo systemctl enable --now tirra-disk-check.timer
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
