"""Tests for deploy/backup_to_r2.sh's retention/pruning logic.

These exercise the REAL script end-to-end against a fake `aws` CLI (a bash
stand-in that simulates an S3-compatible bucket as a local directory) rather
than reimplementing the pruning logic in Python — the point is to catch
regressions in the actual bytes that run in production, not in a parallel
Python model of them.

Requires a bash new enough for `mapfile` (bash >= 4). macOS ships bash 3.2 at
/bin/bash; if no newer bash is found (e.g. `brew install bash`), these tests
are skipped rather than failing on an environment gap unrelated to the script.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "deploy" / "backup_to_r2.sh"


def _find_bash4() -> str | None:
    """Locate a bash binary with mapfile support (bash >= 4)."""
    candidates = [
        os.environ.get("TIRRA_TEST_BASH"),
        shutil.which("bash"),
        "/opt/homebrew/bin/bash",
        "/usr/local/bin/bash",
        "/bin/bash",
    ]
    for c in candidates:
        if not c or not os.path.exists(c):
            continue
        try:
            out = subprocess.run([c, "-c", "echo $BASH_VERSINFO"], capture_output=True, text=True, timeout=5)
            major = int(out.stdout.strip().split()[0]) if out.stdout.strip() else 0
        except Exception:
            continue
        if major >= 4:
            return c
    return None


BASH4 = _find_bash4()
pytestmark = pytest.mark.skipif(
    BASH4 is None, reason="no bash>=4 with mapfile found (script targets Debian/Ubuntu, not macOS's system bash)"
)

# ---------------------------------------------------------------------------
# Fake `aws` CLI — simulates `s3 cp/ls/rm` against a local directory.
# ---------------------------------------------------------------------------

_FAKE_AWS = r"""#!/usr/bin/env bash
# Fake aws CLI for backup_to_r2.sh tests. Only implements the `s3 cp/ls/rm`
# calls the script actually makes, against $FAKE_S3_ROOT as the "bucket".
set -euo pipefail
: "${FAKE_S3_ROOT:?}"

[[ "${1:-}" == "s3" ]] || { echo "fake aws: only s3 supported" >&2; exit 2; }
verb="$2"; shift 2

positional=()
skip_next=0
for a in "$@"; do
  if [[ $skip_next -eq 1 ]]; then skip_next=0; continue; fi
  case "$a" in
    --endpoint-url) skip_next=1 ;;
    --recursive|--only-show-errors) ;;
    *) positional+=("$a") ;;
  esac
done

strip_prefix() {  # s3://bucket/rest -> rest (keeps trailing slash if present)
  local p="${1#s3://}"
  p="${p#*/}"
  printf '%s' "$p"
}

case "$verb" in
  cp)
    src="${positional[0]}"; dst="${positional[1]}"
    if [[ -f "$FAKE_S3_ROOT/.fail_cp" ]]; then
      echo "fake aws: simulated cp failure" >&2
      exit 1
    fi
    rel="$(strip_prefix "$dst")"; rel="${rel%/}"
    mkdir -p "$FAKE_S3_ROOT/$rel"
    shopt -s dotglob nullglob
    for f in "$src"*; do
      cp -r "$f" "$FAKE_S3_ROOT/$rel/"
    done
    ;;
  ls)
    target="${positional[0]}"
    rel="$(strip_prefix "$target")"
    if [[ -z "$rel" ]]; then
      for d in "$FAKE_S3_ROOT"/*/; do
        [[ -d "$d" ]] || continue
        printf '                           PRE %s/\n' "$(basename "$d")"
      done
    else
      rel="${rel%/}"
      if [[ -f "$FAKE_S3_ROOT/.fail_ls_stamp" ]]; then
        exit 0
      fi
      if [[ -d "$FAKE_S3_ROOT/$rel" ]]; then
        for f in "$FAKE_S3_ROOT/$rel"/*; do
          [[ -e "$f" ]] && printf '%s\n' "$(basename "$f")"
        done
      fi
    fi
    ;;
  rm)
    rel="$(strip_prefix "${positional[0]}")"; rel="${rel%/}"
    [[ -n "$rel" ]] && rm -rf "${FAKE_S3_ROOT:?}/${rel:?}"
    ;;
  *)
    echo "fake aws: unsupported verb $verb" >&2; exit 2 ;;
esac
"""


@pytest.fixture
def fake_bin(tmp_path: Path) -> Path:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    aws = bindir / "aws"
    aws.write_text(_FAKE_AWS)
    aws.chmod(aws.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return bindir


@pytest.fixture
def bucket(tmp_path: Path) -> Path:
    b = tmp_path / "bucket"
    b.mkdir()
    return b


@pytest.fixture
def real_db(tmp_path: Path) -> Path:
    """A real (tiny) SQLite file — exercises the actual `sqlite3 .backup` path."""
    db = tmp_path / "pipeline.db"
    subprocess.run(["sqlite3", str(db), "CREATE TABLE t (x INTEGER); INSERT INTO t VALUES (1);"], check=True)
    return db


def run_backup(
    fake_bin: Path,
    bucket: Path,
    *,
    db: Path | None,
    keep: int = 14,
    extra_env: dict | None = None,
    args: list[str] | None = None,
):
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["FAKE_S3_ROOT"] = str(bucket)
    env["R2_ACCOUNT_ID"] = "acct"
    env["R2_ACCESS_KEY_ID"] = "key"
    env["R2_SECRET_ACCESS_KEY"] = "secret"  # noqa: S105 — test fixture, not a real credential
    env["R2_BUCKET"] = "tirramind-backups"
    env["BACKUP_KEEP"] = str(keep)
    env["TIRRA_BACKUP_ENV"] = "/nonexistent"  # don't source a real .env.backup
    if db is not None:
        env["TIRRA_PIPELINE_DB"] = str(db)
    else:
        env["TIRRA_PIPELINE_DB"] = str(bucket.parent / "no_such_pipeline.db")
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [BASH4, str(SCRIPT), *(args or [])],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def bucket_stamps(bucket: Path) -> list[str]:
    return sorted(p.name for p in bucket.iterdir() if p.is_dir())


def seed_stamp(bucket: Path, stamp: str) -> None:
    d = bucket / stamp
    d.mkdir(parents=True)
    (d / "pipeline.db.gz").write_text("fake old backup")


# ---------------------------------------------------------------------------


def test_successful_backup_creates_one_new_stamp(fake_bin, bucket, real_db):
    result = run_backup(fake_bin, bucket, db=real_db, keep=14)
    assert result.returncode == 0, result.stderr
    assert "uploaded" in result.stdout
    assert len(bucket_stamps(bucket)) == 1


def test_dry_run_creates_no_stamp_and_touches_nothing(fake_bin, bucket, real_db):
    seed_stamp(bucket, "20200101T000000Z")
    result = run_backup(fake_bin, bucket, db=real_db, keep=14, args=["--dry-run"])
    assert result.returncode == 0, result.stderr
    assert "DRY RUN" in result.stdout
    # Only the pre-seeded stamp exists — dry-run uploaded nothing.
    assert bucket_stamps(bucket) == ["20200101T000000Z"]


def test_prune_keeps_only_the_newest_N(fake_bin, bucket, real_db):
    old_stamps = [f"2020010{i}T000000Z" for i in range(1, 6)]  # 5 old backups
    for s in old_stamps:
        seed_stamp(bucket, s)

    result = run_backup(fake_bin, bucket, db=real_db, keep=3)
    assert result.returncode == 0, result.stderr

    stamps = bucket_stamps(bucket)
    assert len(stamps) == 3, stamps
    # The newest 2 of the pre-seeded 5, plus the just-uploaded one, survive —
    # in lexicographic (== chronological, given the UTC/zero-padded format) order.
    assert stamps[:2] == old_stamps[-2:]
    assert stamps[2] not in old_stamps  # the freshly uploaded stamp


def test_prune_noop_when_under_the_limit(fake_bin, bucket, real_db):
    seed_stamp(bucket, "20200101T000000Z")
    result = run_backup(fake_bin, bucket, db=real_db, keep=14)
    assert result.returncode == 0, result.stderr
    assert "nothing to prune" in result.stdout
    assert "20200101T000000Z" in bucket_stamps(bucket)


def test_missing_db_refuses_and_prunes_nothing(fake_bin, bucket):
    """F: a missing pipeline DB must hard-fail, not silently back up only the
    small extras (subscribers.json/usage.db) and let pruning run as if the
    backup were complete — see backup_to_r2.sh's DB-missing branch."""
    seed_stamp(bucket, "20200101T000000Z")
    result = run_backup(fake_bin, bucket, db=None, keep=1)
    assert result.returncode != 0
    assert "refusing" in result.stderr.lower()
    # Existing backups must be untouched, and nothing new uploaded.
    assert bucket_stamps(bucket) == ["20200101T000000Z"]


def test_failed_upload_does_not_prune(fake_bin, bucket, real_db):
    (bucket / ".fail_cp").touch()
    seed_stamp(bucket, "20200101T000000Z")

    result = run_backup(fake_bin, bucket, db=real_db, keep=1)
    assert result.returncode != 0
    assert "not pruning" in result.stderr.lower()
    # The one pre-existing backup must survive a failed upload attempt.
    assert bucket_stamps(bucket) == ["20200101T000000Z"]


def test_upload_not_listable_does_not_prune(fake_bin, bucket, real_db):
    """The upload command can report success while the object isn't actually
    listable yet (eventual consistency / silent partial failure). The script
    must verify the upload landed before trusting it enough to prune."""
    (bucket / ".fail_ls_stamp").touch()
    seed_stamp(bucket, "20200101T000000Z")

    result = run_backup(fake_bin, bucket, db=real_db, keep=1)
    assert result.returncode != 0
    assert "not listable" in result.stderr.lower()
    assert "20200101T000000Z" in bucket_stamps(bucket)
