#!/usr/bin/env bash
# TirraMind — provision a fresh Debian/Ubuntu VM to run the API + nightly chain.
#
# Idempotent: safe to re-run after a config change or a repo update.
#
# Also hardens the box: unattended security upgrades, a deny-by-default
# firewall (ufw: only 22/80/443 open), a swapfile so CPU-only torch inference
# can't OOM-kill the chain, a journald disk cap, and a free disk-space monitor
# (deploy/disk_space_check.sh) — there is no alerting of any kind otherwise.
#
# What it does NOT do, deliberately — these need your credentials or judgement:
#   • create the VM, DNS records, or the R2 bucket
#   • place any secret (see the checklist it prints at the end)
#   • start tirra-api before .env.production exists (it would crash-loop)
#   • restrict SSH to your IP (22/tcp is open to the world until you narrow it
#     — see the printed checklist, step 1a)
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

# ── Automatic security updates ─────────────────────────────────────────────
# Free, no new service — this box gets discovered-by-a-customer as its only
# alerting today, so unpatched CVEs sitting for weeks is a real gap.
log "enabling unattended security upgrades"
apt-get install -y -qq unattended-upgrades
dpkg-reconfigure -f noninteractive unattended-upgrades >/dev/null 2>&1 || true

# ── Firewall ───────────────────────────────────────────────────────────────
# Deny-by-default, allow only SSH + HTTP (ACME challenge/redirect) + HTTPS
# (Caddy). Order matters: SSH is allowed BEFORE `ufw enable` so a re-run never
# has a window where the running session could get locked out. Idempotent —
# `ufw allow` on an existing rule is a no-op, `ufw enable` on an already-active
# firewall is a no-op.
log "configuring firewall (ufw): allow 22/80/443, deny everything else"
if ! command -v ufw >/dev/null 2>&1; then
  apt-get install -y -qq ufw
fi
ufw default deny incoming >/dev/null
ufw default allow outgoing >/dev/null
ufw allow 22/tcp comment 'SSH' >/dev/null
ufw allow 80/tcp comment 'HTTP - ACME + redirect' >/dev/null
ufw allow 443/tcp comment 'HTTPS - Caddy' >/dev/null
ufw --force enable >/dev/null
log "ufw active — status:"
ufw status verbose
log "NOTE: port 22 is open to the world. Once you know your own IP, tighten it:"
log "  ufw delete allow 22/tcp && ufw allow from <your-ip> to any port 22 proto tcp"

# ── Swap ────────────────────────────────────────────────────────────────────
# CPU-only torch inference in the nightly chain can be memory-hungry on a
# small VM; a modest swapfile is cheap insurance against an OOM kill mid-chain
# without eating meaningfully into the disk budget the growing pipeline DB and
# R2-bound backups need. Skipped entirely if swap is already configured.
if ! swapon --show 2>/dev/null | grep -q .; then
  log "no swap active — creating a 2G swapfile"
  if [[ ! -f /swapfile ]]; then
    fallocate -l 2G /swapfile 2>/dev/null || dd if=/dev/zero of=/swapfile bs=1M count=2048 status=none
    chmod 600 /swapfile
    mkswap /swapfile >/dev/null
  fi
  swapon /swapfile
  grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
  # Prefer RAM over swap (avoid thrashing the SQLite working set); swap stays
  # purely a crash-prevention backstop.
  sysctl -w vm.swappiness=10 >/dev/null
  grep -q '^vm.swappiness' /etc/sysctl.conf 2>/dev/null \
    || echo 'vm.swappiness=10' >> /etc/sysctl.conf
else
  log "swap already active, skipping"
fi

# ── journald disk cap ───────────────────────────────────────────────────────
# journald's own default cap (10% of the filesystem) is generous relative to
# the disk budget this VM actually has once the pipeline DB and local backup
# staging are accounted for. Pin it explicitly instead of relying on the
# default. Caddy's own log already rolls (see deploy/Caddyfile); every other
# unit's stdout/stderr goes through journald, so this is the one knob that
# matters for "logs can't fill the disk".
log "capping journald disk usage at 500M"
install -d /etc/systemd/journald.conf.d
cat > /etc/systemd/journald.conf.d/tirra.conf <<'EOF'
[Journal]
SystemMaxUse=500M
EOF
systemctl restart systemd-journald

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
chmod +x "$APP_DIR/scripts/run_scheduled.sh" "$APP_DIR/deploy/backup_to_r2.sh" "$APP_DIR/deploy/disk_space_check.sh" 2>/dev/null || true

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

# Safe to enable unconditionally — no secrets required (it degrades to a
# journal-only warning if SMTP isn't configured yet), unlike tirra-api/
# tirra-chain/tirra-backup which need .env.production/.env.backup in place
# first.
log "enabling disk-space monitor (tirra-disk-check.timer)"
systemctl enable --now tirra-disk-check.timer

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

1a. Firewall is already active (ufw: 22/80/443 open, everything else denied).
    Port 22 is currently open to the world — once you know your own IP (or a
    small static set of them), tighten it:
        ufw delete allow 22/tcp
        ufw allow from <your-ip> to any port 22 proto tcp

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
       systemctl enable --now tirra-brief.timer
   Skip tirra-collect.timer — tirra-chain already runs collection as step 1.
   (tirra-disk-check.timer is already enabled — no secrets needed for it.)

   tirra-brief.timer fires the \$19/mo Brief product (weekly, Mon 20:00 UTC —
   see deploy/systemd/tirra-brief.timer for why). It needs
   .env.production in place (step 2) same as tirra-api/tirra-chain. This is
   the ONE step that has silently regressed before: unit files get installed
   by the wildcard globs above regardless, but nothing enables this timer
   unless you run this line. If it's not in \`systemctl list-timers\` below,
   the paying Brief subscribers get nothing and nothing will tell you.

6. Verify from OUTSIDE the box:
       curl -sI https://api.tirramind.com/status
       systemctl status tirra-api.service
       systemctl list-timers 'tirra-*'
       ufw status verbose
       swapon --show
   Confirm tirra-brief.timer appears in that list-timers output with a real
   NEXT time (Mon 20:00 UTC) — not just tirra-api/tirra-chain/tirra-backup.

======================================================================
CHECKLIST
