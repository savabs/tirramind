#!/usr/bin/env bash
# TirraMind — provision a fresh Debian/Ubuntu VM to run the API + nightly chain.
#
# Idempotent: safe to re-run after a config change or a repo update.
#
# What it does NOT do, deliberately — these need your credentials or judgement:
#   • create the VM, DNS records, or the R2 bucket
#   • place any secret (see the checklist it prints at the end)
#   • start tirra-api before .env.production exists (it would crash-loop)
#
# Usage, as root on the VM:
#   git clone https://github.com/<you>/tirramind /opt/tirramind
#   bash /opt/tirramind/deploy/provision_vm.sh
#
# Re-run after pulling changes:
#   bash /opt/tirramind/deploy/provision_vm.sh
set -euo pipefail

APP_USER="${TIRRA_USER:-tirra}"
APP_DIR="${TIRRA_DIR:-/opt/tirramind}"
PY="${TIRRA_PYTHON:-python3}"

log() { printf '\n[provision] %s\n' "$*"; }

[[ $EUID -eq 0 ]] || { echo "run as root (sudo bash $0)" >&2; exit 1; }

# ── System packages ───────────────────────────────────────────────────────
log "installing system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
  python3 python3-venv python3-dev \
  build-essential git curl ca-certificates \
  sqlite3 \
  debian-keyring debian-archive-keyring apt-transport-https

# awscli — used by backup_to_r2.sh against R2's S3-compatible API.
if ! command -v aws >/dev/null 2>&1; then
  log "installing awscli"
  apt-get install -y -qq awscli || pip3 install --quiet awscli
fi

# ── Caddy (official repo) ─────────────────────────────────────────────────
if ! command -v caddy >/dev/null 2>&1; then
  log "installing Caddy"
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
    | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
    | tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
  apt-get update -qq
  apt-get install -y -qq caddy
else
  log "Caddy already installed"
fi

# ── Service user ──────────────────────────────────────────────────────────
if ! id -u "$APP_USER" >/dev/null 2>&1; then
  log "creating system user $APP_USER"
  useradd --system --create-home --home-dir "/home/$APP_USER" --shell /usr/sbin/nologin "$APP_USER"
else
  log "user $APP_USER exists"
fi

[[ -d "$APP_DIR" ]] || { echo "$APP_DIR does not exist — clone the repo there first" >&2; exit 1; }

# ── Python environment ────────────────────────────────────────────────────
log "creating/updating virtualenv"
if [[ ! -x "$APP_DIR/.venv/bin/python" ]]; then
  "$PY" -m venv "$APP_DIR/.venv"
fi
"$APP_DIR/.venv/bin/pip" install --quiet --upgrade pip setuptools wheel

# CPU-only torch. The default wheels pull ~2 GB of CUDA libraries this VM will
# never use — GPU training happens on Kaggle, not here. Install torch FIRST from
# the CPU index so the dependency resolver doesn't pull the CUDA build.
log "installing CPU-only torch (GPU training runs on Kaggle, not here)"
"$APP_DIR/.venv/bin/pip" install --quiet \
  --index-url https://download.pytorch.org/whl/cpu torch

log "installing application dependencies"
"$APP_DIR/.venv/bin/pip" install --quiet -e "$APP_DIR[quant,ml]"

# ── Runtime directories ───────────────────────────────────────────────────
log "preparing runtime directories"
install -d -o "$APP_USER" -g "$APP_USER" \
  "$APP_DIR/.tirra_pipeline" \
  "$APP_DIR/.tirra_delivery" \
  "$APP_DIR/.tirra_opportunities"
chown -R "$APP_USER:$APP_USER" "$APP_DIR"
chmod +x "$APP_DIR/scripts/run_scheduled.sh" "$APP_DIR/deploy/backup_to_r2.sh" 2>/dev/null || true

# The Caddyfile's `log { output file /var/log/caddy/... }` needs this dir
# writable by whichever user the caddy.service unit actually runs as — the
# official Debian package runs Caddy as `User=caddy Group=caddy`, NOT as
# $APP_USER. Owning it as $APP_USER (as an earlier version of this script did)
# leaves Caddy unable to create its own log file: it fails closed, the
# Caddyfile never loads, and api.tirramind.com never gets TLS. `caddy` is
# installed above, so the user already exists by this point.
install -d -o caddy -g caddy /var/log/caddy

# ── systemd units ─────────────────────────────────────────────────────────
log "installing systemd units"
install -m 644 "$APP_DIR"/deploy/systemd/*.service /etc/systemd/system/
install -m 644 "$APP_DIR"/deploy/systemd/*.timer   /etc/systemd/system/
systemctl daemon-reload

# ── Caddy config ──────────────────────────────────────────────────────────
log "installing Caddyfile"
install -m 644 "$APP_DIR/deploy/Caddyfile" /etc/caddy/Caddyfile
if caddy validate --config /etc/caddy/Caddyfile >/dev/null 2>&1; then
  systemctl reload caddy 2>/dev/null || systemctl restart caddy
  log "Caddy config valid, reloaded"
else
  log "WARNING: Caddyfile failed validation — not reloading. Run:"
  log "  caddy validate --config /etc/caddy/Caddyfile"
fi

# ── Remaining manual steps ────────────────────────────────────────────────
cat <<CHECKLIST

======================================================================
 Provisioning complete. These steps need YOUR credentials — do them now.
======================================================================

1. DNS — point api.tirramind.com at this VM:
       A   api   $(curl -s --max-time 5 ifconfig.me 2>/dev/null || echo '<this-vm-ip>')
   If proxied through Cloudflare (orange cloud), set SSL/TLS mode to
   "Full (strict)". "Flexible" breaks the ACME challenge AND leaves the
   Cloudflare->origin hop unencrypted. Grey-cloud is simplest for an API host.

2. Production secrets — minimal set ONLY:
       cp $APP_DIR/deploy/env.production.example $APP_DIR/.env.production
       \$EDITOR $APP_DIR/.env.production
       chown $APP_USER:$APP_USER $APP_DIR/.env.production
       chmod 600 $APP_DIR/.env.production
   Do NOT copy your development .env here — it holds WandB / HuggingFace /
   GitHub / Kaggle / Anthropic keys that must never sit on a public-facing box.

3. Backup credentials — separate file, root-owned:
       \$EDITOR $APP_DIR/.env.backup     # R2_ACCOUNT_ID, R2_ACCESS_KEY_ID,
                                        # R2_SECRET_ACCESS_KEY, R2_BUCKET
       chmod 600 $APP_DIR/.env.backup
       chown root:root $APP_DIR/.env.backup
   Verify before trusting it:
       sudo -u $APP_USER $APP_DIR/deploy/backup_to_r2.sh --dry-run

4. Seed the pipeline database. The graph took months to accumulate and most
   upstream APIs only serve a recent window — a fresh VM cannot rebuild it.
   Copy it from your working machine:
       rsync -avz .tirra_pipeline/pipeline.db root@<vm>:$APP_DIR/.tirra_pipeline/
       chown $APP_USER:$APP_USER $APP_DIR/.tirra_pipeline/pipeline.db

5. Start services (only after steps 2-4):
       systemctl enable --now tirra-api.service
       systemctl enable --now tirra-chain.timer
       systemctl enable --now tirra-backup.timer
   Skip tirra-collect.timer — tirra-chain already runs collection as step 1.

6. Verify from OUTSIDE the box:
       curl -sI https://api.tirramind.com/status
       systemctl status tirra-api.service
       systemctl list-timers 'tirra-*'

======================================================================
CHECKLIST
