#!/usr/bin/env python3
"""Push CPU-only Kaggle eval kernel (no training).

Runs aligned phase40 IC + honest baseline + data label audit on Kaggle CPU.

Usage:
    python scripts/kaggle_eval_launch.py --push-only
    python scripts/kaggle_eval_launch.py --push-only --upload-artifacts
    python scripts/kaggle_eval_launch.py --status
    python scripts/kaggle_eval_launch.py --download-only

Kernel: deeperisbetter/tirramind-phase50-eval  (enable_gpu=false)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / ".kaggle_eval_kernel"
KERNEL_SLUG = "deeperisbetter/tirramind-phase50-eval"
CODE_DATASET = "deeperisbetter/tirramind-code"
ARTIFACTS_DATASET = "deeperisbetter/tirramind-phase50-eval-artifacts"
STATE_FILE = ROOT / ".tirra_pipeline" / "kaggle_eval_state.json"
DOWNLOAD_DIR = ROOT / ".tirra_pipeline" / "kaggle_eval_downloads"

MODEL_PATH = ROOT / ".tirra_pipeline" / "gnn_model_phase50.pt"
EPOCH_PATH = ROOT / ".tirra_pipeline" / "checkpoints" / "phase50" / "epoch_090.pt"


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(">>>", " ".join(cmd))
    return subprocess.run(cmd, check=True, **kwargs)


def write_state(extra: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state = {}
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    state.update(extra)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def upload_code_dataset() -> None:
    print("\n[1/3] Upload tirramind-code …")
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "tirramind-code"
        stage.mkdir()
        for folder in ("agent", "scripts"):
            shutil.copytree(
                ROOT / folder,
                stage / folder,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
            )
        (stage / "dataset-metadata.json").write_text(
            json.dumps(
                {
                    "title": "tirramind-code",
                    "id": CODE_DATASET,
                    "licenses": [{"name": "proprietary"}],
                },
                indent=2,
            )
        )
        _run(
            [
                "kaggle",
                "datasets",
                "version",
                "-p",
                str(stage),
                "-m",
                f"eval-aligned {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC",
                "--dir-mode",
                "zip",
            ]
        )
    print("  ✓ code uploaded")


def upload_artifacts() -> None:
    print("\n[2/3] Upload eval artifacts (gnn_model + epoch_090) …")
    if not MODEL_PATH.exists():
        raise SystemExit(f"Missing {MODEL_PATH}")
    if not EPOCH_PATH.exists():
        raise SystemExit(f"Missing {EPOCH_PATH}")
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "artifacts"
        stage.mkdir()
        shutil.copy2(MODEL_PATH, stage / "gnn_model_phase50.pt")
        shutil.copy2(EPOCH_PATH, stage / "epoch_090.pt")
        (stage / "dataset-metadata.json").write_text(
            json.dumps(
                {
                    "title": "tirramind-phase50-eval-artifacts",
                    "id": ARTIFACTS_DATASET,
                    "licenses": [{"name": "proprietary"}],
                },
                indent=2,
            )
        )
        _run(
            [
                "kaggle",
                "datasets",
                "version",
                "-p",
                str(stage),
                "-m",
                f"epoch_090 aligned-eval {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
                "--dir-mode",
                "zip",
            ]
        )
    print("  ✓ artifacts uploaded")


def push_eval_kernel() -> None:
    print("\n[3/3] Push eval kernel (CPU) …")
    meta_src = EVAL_DIR / "kernel-metadata.json"
    if not meta_src.exists():
        raise SystemExit(f"Missing {meta_src}")
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        shutil.copy2(meta_src, stage / "kernel-metadata.json")
        shutil.copy2(
            EVAL_DIR / "tirramind_kaggle_phase50_eval.ipynb",
            stage / "tirramind_kaggle_phase50_eval.ipynb",
        )
        _run(["kaggle", "kernels", "push", "-p", str(stage)])
    pushed_at = datetime.now(timezone.utc).isoformat()
    write_state(
        {
            "kernel_slug": KERNEL_SLUG,
            "pushed_at": pushed_at,
            "enable_gpu": False,
            "batch": "aligned_eval_cpu",
            "url": f"https://www.kaggle.com/code/{KERNEL_SLUG}",
        }
    )
    print(f"  ✓ pushed → https://www.kaggle.com/code/{KERNEL_SLUG}")
    print("  → Open Kaggle UI and click Run All (CPU, ~30–60 min for full 40-fold IC).")


def show_status() -> None:
    state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    print(json.dumps(state, indent=2) if state else "No eval state recorded.")
    r = subprocess.run(
        ["kaggle", "kernels", "status", KERNEL_SLUG],
        capture_output=True,
        text=True,
    )
    print(r.stdout.strip() or r.stderr.strip())


def download_outputs() -> None:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _run(["kaggle", "kernels", "output", KERNEL_SLUG, "-p", str(DOWNLOAD_DIR)])
    for name in (
        "ic_results_eval_phase50.json",
        "honest_baseline_audit_full.json",
        "data_label_audit_full.json",
    ):
        p = DOWNLOAD_DIR / name
        if p.exists():
            dest = ROOT / ".tirra_pipeline" / name
            shutil.copy2(p, dest)
            print(f"  ✓ {name} → {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Kaggle CPU eval kernel launcher")
    ap.add_argument("--push-only", action="store_true")
    ap.add_argument("--upload-artifacts", action="store_true", help="Refresh artifact dataset")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.status:
        show_status()
        return
    if args.download_only:
        download_outputs()
        return
    if not args.push_only:
        ap.print_help()
        return

    upload_code_dataset()
    if args.upload_artifacts:
        upload_artifacts()
    else:
        print("\n[2/3] Skipping artifact upload (use --upload-artifacts to refresh epoch_090)")
    push_eval_kernel()


if __name__ == "__main__":
    main()
