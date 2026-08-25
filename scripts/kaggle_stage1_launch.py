#!/usr/bin/env python3
"""Push Phase B Stage-1 SSL training to Kaggle (GPU).

Syncs V73+ kernel version, uploads code, pushes tirramind-phase50.

Usage:
    python scripts/kaggle_stage1_launch.py --verify
    python scripts/kaggle_stage1_launch.py --push-only --epochs 90
    python scripts/kaggle_stage1_launch.py --push-only --epochs 10 --smoke

Smoke first (10ep + eval_smoke): recommended before 90ep full run.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "tirramind_kaggle_phase50.ipynb"
LAUNCH = ROOT / "scripts/kaggle_launch.py"
VERSIONS = ROOT / "VERSIONS.md"
STATE = ROOT / ".tirra_pipeline/kaggle_state.json"
DEFAULT_KERNEL_VERSION = 73


def _sync_versions_md(kernel_version: int) -> None:
    text = VERSIONS.read_text()
    text = re.sub(
        r"\*\*Current notebook target: V\d+\*\*",
        f"**Current notebook target: V{kernel_version}**",
        text,
        count=1,
    )
    text = re.sub(
        r"`kernel_version: \d+` in `tirramind_kaggle_phase50.ipynb`, "
        r"`CANONICAL_KERNEL_VERSION = \d+`",
        f"`kernel_version: {kernel_version}` in `tirramind_kaggle_phase50.ipynb`, "
        f"`CANONICAL_KERNEL_VERSION = {kernel_version}`",
        text,
        count=1,
    )
    VERSIONS.write_text(text)


def _sync_launcher(kernel_version: int) -> None:
    text = LAUNCH.read_text()
    text = re.sub(
        r"CANONICAL_KERNEL_VERSION = \d+",
        f"CANONICAL_KERNEL_VERSION = {kernel_version}",
        text,
        count=1,
    )
    LAUNCH.write_text(text)


def _patch_notebook_epochs(*, epochs: int, smoke: bool) -> None:
    text = NOTEBOOK.read_text()
    text = re.sub(
        r'("epochs":\s*)\d+',
        rf"\g<1>{epochs}",
        text,
        count=1,
    )
    text = re.sub(
        r'("eval_smoke":\s*)(True|False)',
        rf"\g<1>{smoke}",
        text,
        count=1,
    )
    NOTEBOOK.write_text(text)
    print(f"Patched epochs={epochs}, eval_smoke={smoke}")


def _compute_fingerprint() -> str:
    sys.path.insert(0, str(ROOT / "scripts"))
    from kaggle_launch import read_notebook_fingerprint  # noqa: WPS433

    fp, cfg = read_notebook_fingerprint()
    print(f"Config fingerprint: {fp}  fix={cfg.get('fix', '?')}")
    return fp


def main() -> None:
    ap = argparse.ArgumentParser(description="V73 Stage-1 SSL Kaggle push")
    ap.add_argument("--verify", action="store_true", help="Version sync check only")
    ap.add_argument("--push-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--kernel-version", type=int, default=DEFAULT_KERNEL_VERSION)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Post-train eval smoke (auto-on if epochs<=10)",
    )
    args = ap.parse_args()

    _sync_versions_md(args.kernel_version)
    _sync_launcher(args.kernel_version)

    smoke_eval = args.smoke or args.epochs <= 10
    _patch_notebook_epochs(epochs=args.epochs, smoke=smoke_eval)

    if args.verify:
        subprocess.run(
            [sys.executable, str(LAUNCH), "--verify-version"],
            cwd=str(ROOT),
            check=True,
        )
        _compute_fingerprint()
        return

    if not args.push_only:
        ap.print_help()
        print(
            "\nPrep complete. Push with:\n"
            f"  python scripts/kaggle_stage1_launch.py --push-only --epochs {args.epochs}"
            + (" --smoke" if smoke_eval else "")
        )
        return

    subprocess.run(
        [sys.executable, str(LAUNCH), "--verify-version"],
        cwd=str(ROOT),
        check=True,
    )
    fp = _compute_fingerprint()
    subprocess.run(
        [
            sys.executable,
            str(LAUNCH),
            "--push-only",
            "--epochs",
            str(args.epochs),
        ],
        cwd=str(ROOT),
        check=True,
    )
    if STATE.exists():
        state = json.loads(STATE.read_text())
        state.update(
            {
                "batch": "v73_stage1_ssl",
                "loop_halt": True,
                "fix": "v73_stage1_ssl",
                "fingerprint": fp,
                "epochs": args.epochs,
                "config_note": "phase50_stage1_ssl preset",
            }
        )
        STATE.write_text(json.dumps(state, indent=2))
    print("\n✓ V73 pushed. Open Kaggle → Run All (GPU T4).")
    print("  Smoke first: --epochs 10 --smoke")
    print("  Full train:  --epochs 90")


if __name__ == "__main__":
    main()
