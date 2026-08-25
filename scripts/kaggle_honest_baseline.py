#!/usr/bin/env python3
"""Run honest baseline + data label audits on Kaggle CPU.

Invoke from tirramind_kaggle_phase50.ipynb after datasets mount:

    !python scripts/honest_baseline_audit.py \\
        --out /kaggle/working/honest_baseline_audit_full.json
    !python scripts/data_label_audit.py \\
        --out /kaggle/working/data_label_audit.json

Push aligned eval to Kaggle CPU (recommended for full 40-fold runs):

    python scripts/kaggle_eval_launch.py --push-only --upload-artifacts

Or locally before push:

    python scripts/kaggle_honest_baseline.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Print commands only")
    ap.add_argument(
        "--push-cpu",
        action="store_true",
        help="Push tirramind-phase50-eval CPU kernel (via kaggle_eval_launch.py)",
    )
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    baseline_cmd = [
        sys.executable,
        str(ROOT / "scripts/honest_baseline_audit.py"),
        "--out",
        str(ROOT / ".tirra_pipeline/honest_baseline_audit_full.json"),
    ]
    if args.smoke:
        baseline_cmd.append("--smoke")

    audit_cmd = [
        sys.executable,
        str(ROOT / "scripts/data_label_audit.py"),
        "--out",
        str(ROOT / ".tirra_pipeline/data_label_audit.json"),
    ]
    if args.smoke:
        audit_cmd.append("--smoke")

    if args.dry_run or not args.push_cpu:
        print("Baseline:", " ".join(baseline_cmd))
        print("Data audit:", " ".join(audit_cmd))
        if args.dry_run:
            return

    for cmd in (baseline_cmd, audit_cmd):
        print(f"\n>>> {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(ROOT), check=True)

    if args.push_cpu:
        push = [
            sys.executable,
            str(ROOT / "scripts/kaggle_eval_launch.py"),
            "--push-only",
            "--upload-artifacts",
        ]
        print(f"\n>>> {' '.join(push)}")
        subprocess.run(push, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
