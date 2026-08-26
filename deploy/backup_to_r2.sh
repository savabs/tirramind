#!/usr/bin/env bash
# TirraMind — back up pipeline state to Cloudflare R2.
#
# The pipeline DB is a single ~138 MB SQLite file on one VM disk holding 5,628
# entities / 365k observations / 16,870 typed links accumulated over months.
# It is the one true single point of failure in this topology, and it cannot be
# regenerated — most upstream APIs only serve a recent window.
#
# R2 is S3-API-compatible, so the standard aws CLI works against it.
#
# Environment (put these in /opt/tirramind/.env.backup, root-owned, mode 600 —
# NOT in .env.production; the API server has no business holding storage
# write credentials):
#   R2_ACCOUNT_ID          Cloudflare account id
#   R2_ACCESS_KEY_ID       R2 API token access key
#   R2_SECRET_ACCESS_KEY   R2 API token secret
#   R2_BUCKET              bucket name, e.g. tirramind-backups
#   BACKUP_KEEP            how many timestamped backups to retain (default 14)
#
# Usage:
#   ./deploy/backup_to_r2.sh            # back up + prune
#   ./deploy/backup_to_r2.sh --dry-run  # show what would happen, touch nothing
set -euo pipefail

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

cd "$(dirname "$0")/.."

ENV_FILE="${TIRRA_BACKUP_ENV:-/opt/tirramind/.env.backup}"
if [[ -f "$ENV_FILE" ]]; then
  set -a; source "$ENV_FILE"; set +a
fi

: "${R2_ACCOUNT_ID:?set R2_ACCOUNT_ID}"
: "${R2_ACCESS_KEY_ID:?set R2_ACCESS_KEY_ID}"
: "${R2_SECRET_ACCESS_KEY:?set R2_SECRET_ACCESS_KEY}"
: "${R2_BUCKET:?set R2_BUCKET}"
BACKUP_KEEP="${BACKUP_KEEP:-14}"

ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export AWS_DEFAULT_REGION="auto"

DB="${TIRRA_PIPELINE_DB:-.tirra_pipeline/pipeline.db}"

echo "[backup] stamp=$STAMP bucket=$R2_BUCKET keep=$BACKUP_KEEP dry_run=$DRY_RUN"

# ── Snapshot SQLite safely ────────────────────────────────────────────────
# Never `cp` a live SQLite file — a concurrent write produces a torn copy that
# looks fine until you try to restore it. `.backup` takes a consistent snapshot
# even while the chain is mid-write.
if [[ -f "$DB" ]]; then
  echo "[backup] snapshotting $DB ($(du -h "$DB" | cut -f1))"
  if [[ $DRY_RUN -eq 0 ]]; then
    sqlite3 "$DB" ".backup '$STAGING/pipeline.db'"
    gzip -6 "$STAGING/pipeline.db"
  fi
elif [[ $DRY_RUN -eq 1 ]]; then
  echo "[backup] WARNING: $DB not found — a real run would refuse to back up without it"
else
  # The DB is the one artifact this entire script exists to protect. If we
  # only warned and continued here, the subscribers.json/usage.db staged
  # below would make the STAGED_COUNT check further down pass (count > 0)
  # even though the pipeline DB itself was never captured — retention
  # pruning would then run normally and quietly rotate out a real historical
  # backup for a "backup" that is missing the one thing that cannot be
  # regenerated. Fail loudly instead of silently uploading a partial backup.
  echo "[backup] ERROR: $DB not found — refusing to take a partial backup." >&2
  echo "[backup] The pipeline DB cannot be regenerated; a backup missing it is worse than no backup." >&2
  echo "[backup] Existing backups are untouched." >&2
  exit 1
fi

# Small but genuinely unrecoverable: subscriber API keys and usage metering.
for extra in ".tirra_opportunities/subscribers.json" ".tirra_opportunities/usage.db"; do
  if [[ -f "$extra" ]]; then
    echo "[backup] including $extra"
    [[ $DRY_RUN -eq 0 ]] && cp "$extra" "$STAGING/"
  fi
done

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[backup] DRY RUN — would upload to s3://$R2_BUCKET/$STAMP/ and prune to $BACKUP_KEEP"
  exit 0
fi

# ── Refuse to proceed on an empty snapshot ────────────────────────────────
# Without this, a night where the DB was missing or sqlite3 failed produced an
# EMPTY staging dir, uploaded nothing, then pruned normally — quietly deleting
# one real backup per night while exiting 0 the whole time. Fourteen nights of
# that and the entire history is gone, with every run reported successful.
#
# An empty snapshot is a failure, not a no-op. Exit non-zero so the systemd
# timer surfaces it, and prune NOTHING.
STAGED_COUNT=$(find "$STAGING" -type f | wc -l | tr -d ' ')
if (( STAGED_COUNT == 0 )); then
  echo "[backup] ERROR: snapshot is empty — nothing was captured." >&2
  echo "[backup] Refusing to upload or prune. Existing backups are untouched." >&2
  echo "[backup] Check that $DB exists and sqlite3 is installed." >&2
  exit 1
fi

# ── Upload ────────────────────────────────────────────────────────────────
if ! aws s3 cp "$STAGING/" "s3://$R2_BUCKET/$STAMP/" \
     --endpoint-url "$ENDPOINT" --recursive --only-show-errors; then
  # Pruning after a failed upload deletes old backups to make room for a new
  # one that does not exist. Never prune unless the new backup actually landed.
  echo "[backup] ERROR: upload failed — not pruning. Existing backups are untouched." >&2
  exit 1
fi
echo "[backup] uploaded s3://$R2_BUCKET/$STAMP/ ($STAGED_COUNT files)"

# Confirm the upload is really there before treating it as a retained backup.
#
# Deliberately NOT `aws ... | grep -q .`: grep -q exits the instant it matches
# the first byte and closes its end of the pipe, which can SIGPIPE the aws
# process before it finishes writing/exiting. With `pipefail` (set above)
# that SIGPIPE surfaces as a non-zero pipeline status even when the listing
# genuinely succeeded — a spurious "not listable" failure race. Capturing the
# output in a variable forces a full read to EOF instead, so there is nothing
# left to close early.
LISTING="$(aws s3 ls "s3://$R2_BUCKET/$STAMP/" --endpoint-url "$ENDPOINT" || true)"
if [[ -z "$LISTING" ]]; then
  echo "[backup] ERROR: upload reported success but $STAMP is not listable — not pruning." >&2
  exit 1
fi

# ── Prune ─────────────────────────────────────────────────────────────────
# List top-level timestamp prefixes, keep the newest $BACKUP_KEEP, delete the
# rest. Timestamps are lexicographically sortable by construction (UTC,
# zero-padded), so `sort` is correct here.
mapfile -t PREFIXES < <(
  aws s3 ls "s3://$R2_BUCKET/" --endpoint-url "$ENDPOINT" \
    | awk '/PRE/ {print $2}' | tr -d '/' | sort
)

TOTAL=${#PREFIXES[@]}
if (( TOTAL > BACKUP_KEEP )); then
  DELETE_COUNT=$(( TOTAL - BACKUP_KEEP ))
  echo "[backup] pruning $DELETE_COUNT of $TOTAL backups (keeping $BACKUP_KEEP newest)"
  for (( i=0; i<DELETE_COUNT; i++ )); do
    old="${PREFIXES[$i]}"
    echo "[backup]   removing $old"
    aws s3 rm "s3://$R2_BUCKET/$old/" \
      --endpoint-url "$ENDPOINT" --recursive --only-show-errors
  done
else
  echo "[backup] $TOTAL backups retained, nothing to prune"
fi

echo "[backup] done"
