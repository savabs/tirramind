#!/usr/bin/env python3
"""
kaggle_preflight.py — Mandatory gate before every Kaggle training push.

Run this and fix ALL FAIL items before running `kaggle kernels push`.
A single FAIL = do not push. Zero shortcuts.

Checks
------
1.  EWC tests pass               (prevents the spike-every-block catastrophe)
2.  Trainer smoke import          (catches broken trainer.py before Kaggle sees it)
3.  Git working tree clean        (no uncommitted changes to tracked files)
4.  Git HEAD pushed to origin     (Kaggle clones from GitHub — stale = trains on old code)
5.  Notebook has secrets cell     (WANDB_API_KEY loaded from Kaggle Secrets)
6.  Notebook has resume guard     (won't accidentally restart from epoch 0)
7.  Staging zip structure valid   (flat checkpoints/, no h_g/ subfolder, no -j artifact)
8.  Pipeline DB has observations  (don't train on an empty DB)
9.  Checkpoint file exists        (the resume epoch .pt is actually staged)

Usage
-----
    python3 scripts/kaggle_preflight.py
    python3 scripts/kaggle_preflight.py --zip /tmp/tirramind_data_v16.zip
    python3 scripts/kaggle_preflight.py --checkpoint /tmp/staging_v16/.tirra_pipeline/checkpoints/epoch_040.pt
    python3 scripts/kaggle_preflight.py --db .tirra_pipeline/pipeline.db
    python3 scripts/kaggle_preflight.py --skip-tests   # CI mode: skip slow pytest
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── colours ──────────────────────────────────────────────────────────────────
_GREEN = "\033[32m"
_RED = "\033[31m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _pass(msg: str) -> str:
    return f"  {_GREEN}[PASS]{_RESET} {msg}"


def _fail(msg: str) -> str:
    return f"  {_RED}[FAIL]{_RESET} {msg}"


def _warn(msg: str) -> str:
    return f"  {_BOLD}[WARN]{_RESET} {msg}"


# ── helpers ───────────────────────────────────────────────────────────────────
def _run(
    cmd: list[str], timeout: int = 360, env_extra: dict | None = None
) -> tuple[int, str]:
    import os

    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=timeout, env=env
        )
        return r.returncode, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return 1, "TIMEOUT"
    except FileNotFoundError:
        return 1, f"Command not found: {cmd[0]}"


def _notebook() -> dict:
    nb_path = ROOT / "notebooks" / "tirramind-h-g" / "tirramind-h-g.ipynb"
    return json.loads(nb_path.read_text()) if nb_path.exists() else {}


def _notebook_source() -> str:
    nb = _notebook()
    return "\n".join("".join(c["source"]) for c in nb.get("cells", []))


# ── individual checks ─────────────────────────────────────────────────────────


def check_ewc_tests() -> tuple[bool, str]:
    """EWC tests must pass — these guard against catastrophic forgetting."""
    pytest = ROOT / ".venv" / "bin" / "pytest"
    if not pytest.exists():
        pytest = Path.home() / ".local" / "bin" / "pytest"
    code, out = _run(
        [str(pytest), "tests/test_ewc.py", "-q", "--no-header", "--tb=short"],
        env_extra={"PYTHONPATH": str(ROOT)},
    )
    lines = [l for l in out.splitlines() if l.strip()]
    summary = lines[-1] if lines else "no output"
    if code == 0:
        return True, f"EWC tests: {summary}"
    # Surface first failure
    failures = [l for l in lines if "FAILED" in l or "ERROR" in l]
    detail = failures[0] if failures else summary
    return False, f"EWC tests FAILED — {detail}"


def check_trainer_import() -> tuple[bool, str]:
    """Trainer must import without error — catches syntax/import bugs."""
    code, out = _run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, '.'); "
            "from agent.models.gnn.trainer import Trainer; print('OK')",
        ],
    )
    if code == 0 and "OK" in out:
        return True, "agent.models.gnn.trainer imports cleanly"
    first_err = next((l for l in out.splitlines() if l.strip()), out[:120])
    return False, f"Trainer import failed: {first_err}"


def check_git_clean() -> tuple[bool, str]:
    """No uncommitted changes to tracked files."""
    code, out = _run(["git", "status", "--porcelain"])
    dirty = [l for l in out.splitlines() if l and not l.startswith("??")]
    if not dirty:
        return True, "Git working tree clean"
    return (
        False,
        f"{len(dirty)} tracked file(s) with uncommitted changes:\n"
        + "\n".join(f"    {l}" for l in dirty[:5]),
    )


def check_git_pushed() -> tuple[bool, str]:
    """Local HEAD must match origin/main — Kaggle clones from GitHub."""
    code, local = _run(["git", "rev-parse", "HEAD"])
    _, remote = _run(["git", "rev-parse", "origin/main"])
    if code != 0:
        return False, "Could not read git HEAD"
    if local.strip() == remote.strip():
        return True, f"HEAD pushed to origin/main ({local.strip()[:10]})"
    _, log = _run(["git", "log", "--oneline", "origin/main..HEAD"])
    commits = [l for l in log.splitlines() if l.strip()]
    return False, f"{len(commits)} commit(s) not pushed to origin/main:\n" + "\n".join(
        f"    {l}" for l in commits[:5]
    )


def check_notebook_secrets_cell() -> tuple[bool, str]:
    """Notebook must have the UserSecretsClient cell that sets WANDB_API_KEY."""
    src = _notebook_source()
    if not src:
        return False, "Notebook not found"
    has_secrets = "UserSecretsClient" in src and "WANDB_API_KEY" in src
    if has_secrets:
        return True, "Secrets cell present (WANDB_API_KEY loaded from Kaggle Secrets)"
    return False, (
        "Missing secrets cell — WANDB_API_KEY will not be set. "
        "Add the UserSecretsClient cell before the training cell."
    )


def check_notebook_resume_guard() -> tuple[bool, str]:
    """Notebook must not restart from epoch 0 by accident."""
    src = _notebook_source()
    if not src:
        return False, "Notebook not found"
    has_resume = "next_config.json" in src or "--resume" in src or "resume_epoch" in src
    if has_resume:
        return True, "Notebook has resume guard (next_config.json / --resume logic)"
    return False, (
        "Notebook has no resume guard — will restart from epoch 0! "
        "Add next_config.json check or explicit --resume flag."
    )


def check_zip_structure(zip_path: Path | None) -> tuple[bool, str]:
    """Zip must have flat checkpoints/ — no h_g/ subfolder, no -j artifact."""
    if zip_path is None:
        # Try known staging locations
        candidates = [
            Path("/tmp/tirramind_data_v16.zip"),
            Path("/tmp/tirramind_data_upload.zip"),
        ]
        zip_path = next((p for p in candidates if p.exists()), None)

    if zip_path is None or not zip_path.exists():
        return True, "No staging zip found (skipped) — verify manually before upload"

    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
    except Exception as e:
        return False, f"Cannot read zip: {e}"

    checkpoints = [n for n in names if "epoch_" in n and n.endswith(".pt")]
    if not checkpoints:
        return False, f"Zip has no epoch_*.pt files: {zip_path.name}"

    # Must be flat under checkpoints/ — NOT under h_g/
    bad = [n for n in checkpoints if "/h_g/" in n]
    if bad:
        return False, (
            f"Checkpoint in h_g/ subfolder (Cell 5 will fail): {bad[0]}\n"
            "    Fix: stage with flat path .tirra_pipeline/checkpoints/epoch_N.pt"
        )

    # Must not be at root (symptom of -j flag)
    bad_flat = [n for n in checkpoints if "/" not in n.rstrip("/")]
    if bad_flat:
        return False, (
            f"Checkpoint has no directory (was -j used?): {bad_flat[0]}\n"
            "    Fix: cd /tmp/staging && zip -r out.zip .tirra_pipeline/"
        )

    db_present = any("pipeline.db" in n for n in names)
    if not db_present:
        return False, f"pipeline.db missing from zip: {zip_path.name}"

    return True, (
        f"Zip OK — {len(checkpoints)} checkpoint(s), "
        f"pipeline.db present ({zip_path.name})"
    )


def check_db_has_data(db_path: Path | None) -> tuple[bool, str]:
    """Pipeline DB must have entity_observations."""
    import sqlite3

    if db_path is None:
        candidates = [
            ROOT / ".tirra_pipeline" / "pipeline.db",
            Path("/tmp/hg_v15_out/tirramind_v1/.tirra_pipeline/pipeline.db"),
        ]
        db_path = next((p for p in candidates if p.exists()), None)

    if db_path is None or not db_path.exists():
        return False, "pipeline.db not found — cannot verify observation count"

    try:
        conn = sqlite3.connect(str(db_path))
        n = conn.execute("SELECT COUNT(*) FROM entity_observations").fetchone()[0]
        ents = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        conn.close()
    except Exception as e:
        return False, f"DB read error: {e}"

    if n == 0:
        return (
            False,
            f"DB is EMPTY (0 entity_observations) — training on empty DB produces junk",
        )
    return True, f"DB has {n:,} observations, {ents:,} entities ({db_path.name})"


def check_checkpoint_exists(ckpt_path: Path | None) -> tuple[bool, str]:
    """The checkpoint we intend to resume from must actually be staged."""
    if ckpt_path is None:
        candidates = [
            Path("/tmp/staging_v16/.tirra_pipeline/checkpoints/epoch_040.pt"),
            Path("/tmp/staging_v17/.tirra_pipeline/checkpoints/epoch_050.pt"),
        ]
        ckpt_path = next((p for p in candidates if p.exists()), None)

    if ckpt_path is None:
        return True, "No staged checkpoint path specified (skipped)"

    if ckpt_path.exists():
        size_mb = ckpt_path.stat().st_size / 1e6
        return True, f"Checkpoint exists: {ckpt_path.name} ({size_mb:.1f} MB)"
    return (
        False,
        f"Checkpoint NOT FOUND: {ckpt_path} — zip will be empty of checkpoints",
    )


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-flight checks before kaggle kernels push."
    )
    parser.add_argument("--zip", type=Path, help="Path to staging zip to validate")
    parser.add_argument("--checkpoint", type=Path, help="Path to staged checkpoint .pt")
    parser.add_argument("--db", type=Path, help="Path to pipeline.db to validate")
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip pytest (faster, use only if tests ran recently)",
    )
    args = parser.parse_args()

    print(f"\n{_BOLD}{'='*60}{_RESET}")
    print(f"{_BOLD}  Kaggle Training Preflight{_RESET}")
    print(f"{_BOLD}{'='*60}{_RESET}\n")

    checks = []

    if not args.skip_tests:
        checks.append(("EWC Tests", check_ewc_tests))
    else:
        print(_warn("EWC tests skipped (--skip-tests)"))

    checks += [
        ("Trainer Import", check_trainer_import),
        ("Git Clean", check_git_clean),
        ("Git Pushed", check_git_pushed),
        ("Notebook Secrets Cell", check_notebook_secrets_cell),
        ("Notebook Resume Guard", check_notebook_resume_guard),
    ]

    # Parametric checks
    def _zip_check():
        return check_zip_structure(args.zip)

    def _db_check():
        return check_db_has_data(args.db)

    def _ckpt_check():
        return check_checkpoint_exists(args.checkpoint)

    checks += [
        ("Zip Structure", _zip_check),
        ("DB Has Observations", _db_check),
        ("Checkpoint Staged", _ckpt_check),
    ]

    results: list[tuple[str, bool, str]] = []
    for name, fn in checks:
        passed, msg = fn()
        results.append((name, passed, msg))

    # Print results
    for name, passed, msg in results:
        line = _pass(msg) if passed else _fail(msg)
        print(line)

    print()
    failures = [(n, m) for n, p, m in results if not p]
    if not failures:
        print(f"{_GREEN}{_BOLD}  ✓ All checks passed. Safe to push.{_RESET}\n")
        sys.exit(0)
    else:
        print(
            f"{_RED}{_BOLD}  ✗ {len(failures)} check(s) failed — DO NOT PUSH.{_RESET}"
        )
        print(
            f"{_RED}    Fix every FAIL above before running: kaggle kernels push{_RESET}\n"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
