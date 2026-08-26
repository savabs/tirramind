#!/usr/bin/env bash
# TirraMind — cheap disk-space guard for the VM running tirra-api + the chain.
#
# The pipeline DB (deploy/backup_to_r2.sh) is the one true SPOF in this
# topology and it cannot be regenerated — running the disk out of space is
# exactly the kind of failure that would otherwise only surface when SQLite
# writes start failing or backups start silently coming up empty. There is no
# alerting of any kind on this box (see .claude/agents/infra-operator.md), so
# this is deliberately the cheapest thing that could possibly work: no paid
# monitoring service, no new dependency.
#
# Always logs to the systemd journal at WARNING when over threshold — visible
# via `journalctl -p warning -u tirra-disk-check`. Additionally emails if
# TIRRA_SMTP_HOST/TIRRA_BRIEF_TO are already set in .env.production (the same
# vars scripts/tirra_engine.py uses to deliver the brief) — opt-in, zero extra
# cost, no new secret to provision.
#
# Environment:
#   TIRRA_DISK_ALERT_PCT   warn at or above this usage percentage (default 85)
#
# Usage:
#   ./deploy/disk_space_check.sh
set -euo pipefail

cd "$(dirname "$0")/.."

APP_DIR="${TIRRA_DIR:-/opt/tirramind}"
THRESHOLD="${TIRRA_DISK_ALERT_PCT:-85}"

usage_pct="$(df --output=pcent "$APP_DIR" 2>/dev/null | tail -1 | tr -dc '0-9')"

if [[ -z "$usage_pct" ]]; then
  echo "[tirra-disk-check] could not read disk usage for $APP_DIR" >&2
  exit 1
fi

if (( usage_pct < THRESHOLD )); then
  echo "[tirra-disk-check] ${APP_DIR} at ${usage_pct}% — below ${THRESHOLD}% threshold, ok"
  exit 0
fi

msg="[tirra-disk-check] ${APP_DIR} filesystem at ${usage_pct}% (threshold ${THRESHOLD}%) — pipeline DB and R2 backups live here; free space or grow the disk before the next chain/backup run"

if command -v systemd-cat >/dev/null 2>&1; then
  echo "$msg" | systemd-cat -t tirra-disk-check -p warning
else
  echo "$msg" >&2
fi

# Optional email — only fires if SMTP is already configured for brief delivery.
if [[ -f "$APP_DIR/.env.production" ]]; then
  set -a; source "$APP_DIR/.env.production"; set +a
fi

if [[ -n "${TIRRA_SMTP_HOST:-}" && -n "${TIRRA_BRIEF_TO:-}" ]]; then
  "$APP_DIR/.venv/bin/python" - "$msg" <<'PYEOF'
import os
import smtplib
import sys
from email.mime.text import MIMEText

body = sys.argv[1]
msg = MIMEText(body)
msg["Subject"] = "TirraMind: disk space warning"
msg["From"] = os.getenv("TIRRA_SMTP_FROM", os.getenv("TIRRA_SMTP_USER", "awos@localhost"))
to_addrs = [a.strip() for a in os.getenv("TIRRA_BRIEF_TO", "").split(",") if a.strip()]
msg["To"] = ", ".join(to_addrs)

host = os.getenv("TIRRA_SMTP_HOST")
port = int(os.getenv("TIRRA_SMTP_PORT", "587"))
try:
    with smtplib.SMTP(host, port, timeout=10) as s:
        s.starttls()
        user = os.getenv("TIRRA_SMTP_USER")
        if user:
            s.login(user, os.getenv("TIRRA_SMTP_PASS", ""))
        s.sendmail(msg["From"], to_addrs, msg.as_string())
except Exception as exc:  # pragma: no cover — best-effort, journal entry above already fired
    print(f"[tirra-disk-check] email alert failed (journal entry already logged): {exc}", file=sys.stderr)
PYEOF
fi

exit 0
