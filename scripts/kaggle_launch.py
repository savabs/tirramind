#!/usr/bin/env python3
"""kaggle_launch.py — one command to push code, run training on Kaggle, and download results.

Usage:
    python scripts/kaggle_launch.py [--epochs N] [--no-monitor] [--download-only]

What it does:
    1. Zips agent/ + scripts/ into tirramind-code dataset and uploads to Kaggle
    2. Pushes the Phase 50 notebook as a new kernel version
    3. Polls Kaggle every 60s until done / failed
    4. Downloads gnn_model_phase50.pt and epoch checkpoints
    5. Runs phase40_gnn_backtest.py locally and prints IC results
    6. Tails W&B for live loss while waiting (if wandb is available)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNEL_SLUG = "deeperisbetter/tirramind-phase50"
CODE_DATASET_SLUG = "deeperisbetter/tirramind-code"
MODEL_OUT = ROOT / ".tirra_pipeline" / "gnn_model_phase50.pt"
CKPT_DIR = ROOT / ".tirra_pipeline" / "checkpoints" / "phase50"

# ── helpers ──────────────────────────────────────────────────────────────────

def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, check=True, **kw)


def kaggle(*args) -> subprocess.CompletedProcess:
    return run(["kaggle", *[str(a) for a in args]], capture_output=False)


# ── step 1: upload code dataset ───────────────────────────────────────────────

def upload_code_dataset() -> None:
    print("\n[1/5] Packaging and uploading tirramind-code dataset...")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        code_dir = tmp_path / "tirramind-code"
        code_dir.mkdir()

        for folder in ("agent", "scripts"):
            src = ROOT / folder
            dst = code_dir / folder
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", "*.pyo", ".pytest_cache"
            ))

        meta = {
            "title": "tirramind-code",
            "id": CODE_DATASET_SLUG,
            "licenses": [{"name": "proprietary"}],
        }
        (code_dir / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

        subprocess.run(
            ["kaggle", "datasets", "version", "-p", str(code_dir),
             "-m", "Phase 50: price features + residual returns + deeper return head",
             "--dir-mode", "zip"],
            check=True,
        )

    print("  ✓ tirramind-code uploaded")


# ── step 2: push notebook ─────────────────────────────────────────────────────

def push_notebook(epochs: int) -> None:
    print("\n[2/5] Pushing Phase 50 notebook to Kaggle...")

    # Patch the epoch count in the notebook
    nb_src = ROOT / "tirramind_kaggle_phase50.ipynb"
    nb_data = json.loads(nb_src.read_text())
    for cell in nb_data["cells"]:
        if cell.get("id") == "train-phase50":
            src = cell["source"]
            cell["source"] = [
                line.replace('"30"', f'"{epochs}"') if '"--epochs"' in prev or '"30"' in line else line
                for prev, line in zip([""] + src, src)
            ]
    nb_src.write_text(json.dumps(nb_data, indent=1))

    meta_path = ROOT / "kernel-metadata.json"
    meta_path.write_text(json.dumps({
        "id": KERNEL_SLUG,
        "title": "tirramind-phase50",
        "code_file": "tirramind_kaggle_phase50.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": [],
        "dataset_sources": [
            "deeperisbetter/tirramind-data",
            CODE_DATASET_SLUG,
        ],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }, indent=2))

    subprocess.run(["kaggle", "kernels", "push", "-p", str(ROOT)], check=True)
    print("  ✓ Kernel pushed")


# ── step 3: poll until done ───────────────────────────────────────────────────

def poll_until_done(interval: int = 60) -> str:
    print(f"\n[3/5] Waiting for kernel to finish (polling every {interval}s)...")
    print(f"      Kaggle URL: https://www.kaggle.com/code/{KERNEL_SLUG}")

    _wandb_run_name = "phase50-ep1-30"
    _shown_wandb = False
    start = time.time()

    while True:
        result = subprocess.run(
            ["kaggle", "kernels", "status", KERNEL_SLUG],
            capture_output=True, text=True,
        )
        line = result.stdout.strip()
        elapsed = int(time.time() - start)
        print(f"  [{elapsed//60:02d}m{elapsed%60:02d}s] {line}")

        if "COMPLETE" in line.upper():
            return "complete"
        if "ERROR" in line.upper() or "CANCEL" in line.upper():
            print("  ✗ Kernel failed or was cancelled.")
            return "failed"

        # Print W&B latest metrics if available
        if not _shown_wandb:
            try:
                import wandb
                api = wandb.Api(timeout=10)
                runs = list(api.runs("999-sbpatel/tirramind", per_page=5))
                active = [r for r in runs if r.state == "running"]
                if active:
                    r = active[0]
                    hist = list(r.scan_history(keys=["loss/total", "loss/return"], min_step=0, max_step=9999))
                    if hist:
                        last = hist[-1]
                        print(f"      W&B [{r.name}] step={len(hist)} | total={last.get('loss/total', '?'):.4f} | ret={last.get('loss/return', '?'):.4f}")
                    _shown_wandb = False  # keep refreshing
            except Exception:
                pass

        time.sleep(interval)


# ── step 4: download outputs ──────────────────────────────────────────────────

def download_outputs() -> None:
    print("\n[4/5] Downloading outputs from Kaggle...")
    out_dir = ROOT / ".tirra_pipeline" / "kaggle_downloads"
    out_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["kaggle", "kernels", "output", KERNEL_SLUG, "-p", str(out_dir)],
        check=True,
    )

    # Copy model
    src_model = out_dir / "gnn_model_phase50.pt"
    if src_model.exists():
        shutil.copy2(src_model, MODEL_OUT)
        print(f"  ✓ Model → {MODEL_OUT}  ({MODEL_OUT.stat().st_size / 1e6:.1f} MB)")
    else:
        print("  ✗ gnn_model_phase50.pt not found in output")

    # Copy checkpoints
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpts = sorted(out_dir.glob("epoch_*.pt"))
    for ckpt in ckpts:
        dst = CKPT_DIR / ckpt.name
        shutil.copy2(ckpt, dst)
        print(f"  ✓ {ckpt.name}")
    print(f"  {len(ckpts)} checkpoint(s) saved to {CKPT_DIR}")


# ── step 5: local backtest ────────────────────────────────────────────────────

def run_backtest() -> None:
    print("\n[5/5] Running local IC backtest...")
    if not MODEL_OUT.exists():
        print(f"  ✗ Model not found at {MODEL_OUT}, skipping backtest.")
        return

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "phase40_gnn_backtest.py"),
         "--model-path", str(MODEL_OUT),
         "--out", str(ROOT / ".tirra_pipeline" / "ic_results_phase50.json")],
        cwd=str(ROOT),
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        print("  ✗ Backtest failed.")
    else:
        print("  ✓ Backtest complete.")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Push Phase 50 to Kaggle, monitor, download, backtest.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default 30)")
    parser.add_argument("--no-monitor", action="store_true", help="Push and exit without polling")
    parser.add_argument("--download-only", action="store_true", help="Skip push, just download + backtest")
    parser.add_argument("--backtest-only", action="store_true", help="Only run local backtest on existing model")
    args = parser.parse_args()

    if args.backtest_only:
        run_backtest()
        return

    if not args.download_only:
        upload_code_dataset()
        push_notebook(args.epochs)
        if args.no_monitor:
            print("\nDone. Monitor at: https://www.kaggle.com/code/" + KERNEL_SLUG)
            return

    if not args.download_only:
        status = poll_until_done()
        if status == "failed":
            print("\nTraining failed — check Kaggle for details.")
            sys.exit(1)

    download_outputs()
    run_backtest()


if __name__ == "__main__":
    main()
