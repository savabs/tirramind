#!/usr/bin/env bash
# TirraEngine — scheduled runner (cron / systemd friendly).
#
# Independent cadences, meant to be scheduled separately:
#   ./scripts/run_scheduled.sh             one brief cycle (fast refresh + build + deliver)
#   ./scripts/run_scheduled.sh serve       same, then keep serving the brief over HTTP
#   ./scripts/run_scheduled.sh collect     full daily_collection DAG — 40+ sources, SLOW
#                                          (minutes, not seconds). Run once/day, not per-minute.
#   ./scripts/run_scheduled.sh chain       collect AND every downstream DAG, in
#                                          dependency order — this is what turns
#                                          collected data into intelligence.
#
# `chain` exists because `collect` alone leaves layers 2-6 empty. The other 10
# DAGs declare cron schedules, but those only fire under a long-running
# PipelineScheduler.start() process that nothing in production ever started —
# so 8 of 11 DAGs had never run even once. See scripts/run_chain.py.
#
# `collect` runs synchronously and exits only once every source has been
# fetched — it deliberately does NOT use `tirra_engine.py --full-collect`'s
# background-thread path, which is only safe when the process stays alive
# afterward (e.g. --serve). A cron job that exits right after invoking it
# would silently kill that thread before anything gets persisted.
#
# Environment:
#   TIRRA_DELIVERY_DIR    output dir (default .tirra_delivery)
#   TIRRA_PIPELINE_DB     pipeline SQLite path (default .tirra_pipeline/pipeline.db)
#   TIRRA_PIPELINE_WORKERS  parallel workers for `collect` (default 4)
#   TIRRA_PORT            server port (default 8787)
set -euo pipefail

cd "$(dirname "$0")/.."

OUT="${TIRRA_DELIVERY_DIR:-.tirra_delivery}"
DB="${TIRRA_PIPELINE_DB:-.tirra_pipeline/pipeline.db}"
PORT="${TIRRA_PORT:-8787}"

if [[ "${1:-}" == "collect" ]]; then
  echo "[scheduled] running full daily_collection DAG (40+ sources) — this can take several minutes"
  .venv/bin/python scripts/run_collection.py --db-path "$DB" --workers "${TIRRA_PIPELINE_WORKERS:-4}"
  echo "[scheduled] collection complete"
  exit 0
fi

if [[ "${1:-}" == "chain" ]]; then
  echo "[scheduled] running the full DAG chain in dependency order — this can take a while"
  # Synchronous, like `collect`: exits non-zero if any DAG failed, so the
  # timer/cron surfaces it rather than reporting success.
  .venv/bin/python scripts/run_chain.py --db-path "$DB" --workers "${TIRRA_PIPELINE_WORKERS:-4}" "${@:2}"
  exit $?
fi

echo "[scheduled] refreshing data + delivering brief"
.venv/bin/python scripts/tirra_engine.py --once --collect \
  --contracts "${CONTRACTS:-10}" --anomalies "${ANOMALIES:-8}" \
  --max-contract-rows "${MAX_ROWS:-5}" --out "$OUT" --db "$DB"

if [[ "${1:-}" == "serve" ]]; then
  echo "[scheduled] serving brief at :$PORT"
  .venv/bin/python agent/brief_server.py --port "$PORT" --out "$OUT"
fi

echo "[scheduled] cycle complete"
