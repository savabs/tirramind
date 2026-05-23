#!/usr/bin/env python3
"""kaggle_launch.py — one command to run training on Kaggle and retrieve results.

VERIFIED KAGGLE CLI COMMANDS (from `kaggle --help`):
    kaggle kernels push -p <folder>          # push kernel, folder must have kernel-metadata.json
    kaggle kernels status <owner>/<slug>     # get current run status
    kaggle kernels logs -f <owner>/<slug>    # tail logs live (like tail -f)
    kaggle kernels output -p <path> <slug>   # download all output files
    kaggle kernels files <owner>/<slug>      # list output files
    kaggle datasets version -p <folder> -m "msg" --dir-mode zip  # update dataset

Usage:
    python scripts/kaggle_launch.py                  # full flow: upload → push → tail logs → download → backtest
    python scripts/kaggle_launch.py --push-only      # upload code + push kernel, then exit
    python scripts/kaggle_launch.py --logs-only      # tail logs of currently running kernel
    python scripts/kaggle_launch.py --download-only  # download outputs + run backtest
    python scripts/kaggle_launch.py --backtest-only  # run local backtest on existing model
    python scripts/kaggle_launch.py --status         # show what's currently running

STATE FILE: .tirra_pipeline/kaggle_state.json
    Always records the active kernel slug, version, epochs, and pushed_at.
    Check it any time to know what's running.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# ── constants ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
KERNEL_SLUG = "deeperisbetter/tirramind-phase50"  # single canonical kernel
NOTEBOOK_FILE = ROOT / "tirramind_kaggle_phase50.ipynb"
CODE_DATASET = "deeperisbetter/tirramind-code"
STATE_FILE = ROOT / ".tirra_pipeline" / "kaggle_state.json"
DOWNLOAD_DIR = ROOT / ".tirra_pipeline" / "kaggle_downloads"
MODEL_OUT = ROOT / ".tirra_pipeline" / "gnn_model_phase50.pt"
CKPT_DIR = ROOT / ".tirra_pipeline" / "checkpoints" / "phase50"

KAGGLE_URL = f"https://www.kaggle.com/code/{KERNEL_SLUG}"

# ── state file ────────────────────────────────────────────────────────────────


def write_state(extra: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state = read_state()
    state.update(extra)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def read_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def show_status() -> None:
    state = read_state()
    if not state:
        print("No active Kaggle run recorded.")
        return
    print("\n── Active Kaggle Run ─────────────────────────────")
    print(f"  Kernel  : {state.get('kernel_slug')}")
    print(f"  Epochs  : {state.get('epochs')}")
    print(f"  Pushed  : {state.get('pushed_at')}")
    print(f"  URL     : {KAGGLE_URL}")
    result = subprocess.run(
        ["kaggle", "kernels", "status", state.get("kernel_slug", KERNEL_SLUG)],
        capture_output=True,
        text=True,
    )
    print(f"  Status  : {result.stdout.strip()}")
    print("──────────────────────────────────────────────────\n")


# ── step 1: upload code dataset ───────────────────────────────────────────────


def upload_code_dataset() -> None:
    print("\n[1/4] Packaging code → tirramind-code dataset...")
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "tirramind-code"
        stage.mkdir()
        for folder in ("agent", "scripts"):
            shutil.copytree(
                ROOT / folder,
                stage / folder,
                ignore=shutil.ignore_patterns(
                    "__pycache__", "*.pyc", "*.pyo", ".pytest_cache"
                ),
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

        # kaggle datasets version -p <folder> -m "message" --dir-mode zip
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "version",
                "-p",
                str(stage),
                "-m",
                f"Phase50 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC",
                "--dir-mode",
                "zip",
            ],
            check=True,
        )
    print("  ✓ tirramind-code uploaded")


# ── step 2: push kernel ───────────────────────────────────────────────────────


def push_kernel(epochs: int) -> None:
    print(f"\n[2/4] Pushing kernel (epochs={epochs})...")

    # Patch epochs in notebook
    nb = json.loads(NOTEBOOK_FILE.read_text())
    for cell in nb.get("cells", []):
        if cell.get("id") == "train-phase50":
            cell["source"] = [
                (
                    line
                    if '"--epochs"' not in line
                    else f'    "--epochs",              "{epochs}",\n'
                )
                for line in cell["source"]
            ]
    NOTEBOOK_FILE.write_text(json.dumps(nb, indent=1))

    # Write kernel-metadata.json (gitignored — generated each run)
    meta = {
        "id": KERNEL_SLUG,
        "title": "tirramind-phase50",
        "code_file": NOTEBOOK_FILE.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": [],
        "dataset_sources": ["deeperisbetter/tirramind-data", CODE_DATASET],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    (ROOT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

    # kaggle kernels push -p <folder>
    result = subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(ROOT)],
        capture_output=True,
        text=True,
    )
    print(f"  {result.stdout.strip()}")
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}")
        sys.exit(1)

    write_state(
        {
            "kernel_slug": KERNEL_SLUG,
            "epochs": epochs,
            "pushed_at": datetime.now(timezone.utc).isoformat(),
            "url": KAGGLE_URL,
        }
    )
    print(f"  ✓ Kernel pushed  →  {KAGGLE_URL}")
    print(f"  ✓ State saved to {STATE_FILE}")


# ── step 3: tail logs ─────────────────────────────────────────────────────────


def tail_logs() -> str:
    """Stream logs with `kaggle kernels logs -f`. Returns final status."""
    slug = read_state().get("kernel_slug", KERNEL_SLUG)
    print(f"\n[3/4] Tailing logs (Ctrl+C to stop and keep kernel running)...")
    print(f"      {KAGGLE_URL}\n")

    # kaggle kernels logs -f <slug>  — streams until kernel completes
    proc = subprocess.run(
        ["kaggle", "kernels", "logs", "-f", "--interval", "10", slug],
    )
    _ = proc  # logs -f blocks until kernel done or user Ctrl+C

    # Check final status
    result = subprocess.run(
        ["kaggle", "kernels", "status", slug],
        capture_output=True,
        text=True,
    )
    status_line = result.stdout.strip()
    print(f"\n  Final status: {status_line}")

    if "COMPLETE" in status_line.upper():
        return "complete"
    if "ERROR" in status_line.upper() or "CANCEL" in status_line.upper():
        return "failed"
    return "unknown"


# ── step 4: download outputs ──────────────────────────────────────────────────


def download_outputs() -> None:
    slug = read_state().get("kernel_slug", KERNEL_SLUG)
    print(f"\n[4/4] Downloading outputs from {slug}...")

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # List files first
    files_result = subprocess.run(
        ["kaggle", "kernels", "files", slug],
        capture_output=True,
        text=True,
    )
    print(f"  Output files:\n{files_result.stdout.strip()}")

    # kaggle kernels output -p <path> <slug>
    subprocess.run(
        ["kaggle", "kernels", "output", "-p", str(DOWNLOAD_DIR), slug],
        check=True,
    )

    # Promote model
    src_model = DOWNLOAD_DIR / "gnn_model_phase50.pt"
    if src_model.exists():
        shutil.copy2(src_model, MODEL_OUT)
        print(f"  ✓ Model → {MODEL_OUT}  ({MODEL_OUT.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"  ✗ gnn_model_phase50.pt not in output (check {DOWNLOAD_DIR})")

    # Promote checkpoints
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpts = sorted(DOWNLOAD_DIR.glob("epoch_*.pt"))
    for ckpt in ckpts:
        shutil.copy2(ckpt, CKPT_DIR / ckpt.name)
        print(f"  ✓ {ckpt.name}")


# ── step 5: local backtest ────────────────────────────────────────────────────


def run_backtest() -> None:
    print("\n[Backtest] Running local IC backtest...")
    if not MODEL_OUT.exists():
        print(f"  ✗ {MODEL_OUT} not found — download first.")
        return
    result = subprocess.run(
        [
            sys.executable,
            "scripts/phase40_gnn_backtest.py",
            "--model-path",
            str(MODEL_OUT),
            "--out",
            str(ROOT / ".tirra_pipeline" / "ic_results_phase50.json"),
        ],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print("  ✗ Backtest failed.")
    else:
        print("  ✓ Backtest complete — check ic_results_phase50.json")


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description="One-command Kaggle training: upload code → push kernel → tail logs → download → backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--epochs", type=int, default=30, help="Training epochs (default 30)"
    )
    p.add_argument("--push-only", action="store_true", help="Upload + push, then exit")
    p.add_argument("--logs-only", action="store_true", help="Tail logs of current run")
    p.add_argument(
        "--download-only", action="store_true", help="Download outputs + backtest"
    )
    p.add_argument(
        "--backtest-only", action="store_true", help="Local backtest on existing model"
    )
    p.add_argument("--status", action="store_true", help="Show active run state")
    args = p.parse_args()

    if args.status:
        show_status()
        return

    if args.backtest_only:
        run_backtest()
        return

    if args.logs_only:
        tail_logs()
        return

    if args.download_only:
        download_outputs()
        run_backtest()
        return

    # Full flow
    upload_code_dataset()
    push_kernel(args.epochs)

    if args.push_only:
        print(f"\nDone. To tail logs:\n  python scripts/kaggle_launch.py --logs-only")
        return

    status = tail_logs()
    if status == "failed":
        print("\nTraining failed. To check logs:\n  kaggle kernels logs", KERNEL_SLUG)
        sys.exit(1)

    download_outputs()
    run_backtest()


if __name__ == "__main__":
    main()
