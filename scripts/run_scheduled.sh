#!/usr/bin/env bash
# TirraEngine — scheduled runner (cron / systemd friendly).
#
# Runs a fast refresh + build + deliver of the Intelligence Brief on a schedule,
# and keeps the brief HTTP server running. Intended to be invoked by cron or a
# process supervisor, not run interactively for long periods.
#
# Usage:
#   ./scripts/run_scheduled.sh            # one cycle (for cron)
#   ./scripts/run_scheduled.sh serve      # keep serving after delivering
#
# Environment:
#   TIRRA_DELIVERY_DIR   output dir (default .tirra_delivery)
#   TIRRA_PORT           server port (default 8787)
set -euo pipefail

cd "$(dirname "$0")/.."

OUT="${TIRRA_DELIVERY_DIR:-.tirra_delivery}"
PORT="${TIRRA_PORT:-8787}"

echo "[scheduled] refreshing data + delivering brief"
.venv/bin/python scripts/tirra_engine.py --once --collect \
  --contracts "${CONTRACTS:-10}" --anomalies "${ANOMALIES:-8}" \
  --max-contract-rows "${MAX_ROWS:-5}" --out "$OUT"

if [[ "${1:-}" == "serve" ]]; then
  echo "[scheduled] serving brief at :$PORT"
  .venv/bin/python agent/brief_server.py --port "$PORT" --out "$OUT"
fi

echo "[scheduled] cycle complete"
