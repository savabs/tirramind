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
import hashlib
import json
import re
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
VERSIONS_FILE = ROOT / "VERSIONS.md"

# Single source of truth for the *next* push. Must match:
#   tirramind_kaggle_phase50.ipynb → _NOTEBOOK_CONFIG["kernel_version"]
#   VERSIONS.md → "Current notebook target" line
#   Kaggle UI "Version N" on deeperisbetter/tirramind-phase50 (increment before each push)
CANONICAL_KERNEL_VERSION = 73
CODE_DATASET = "deeperisbetter/tirramind-code"
STATE_FILE = ROOT / ".tirra_pipeline" / "kaggle_state.json"
DOWNLOAD_DIR = ROOT / ".tirra_pipeline" / "kaggle_downloads"
MODEL_OUT = ROOT / ".tirra_pipeline" / "gnn_model_phase50.pt"
CKPT_DIR = ROOT / ".tirra_pipeline" / "checkpoints" / "phase50"

KAGGLE_URL = f"https://www.kaggle.com/code/{KERNEL_SLUG}"

# ── kernel version sync (notebook ↔ launcher ↔ VERSIONS.md) ─────────────────


def read_notebook_kernel_version() -> int:
    """Parse kernel_version from _NOTEBOOK_CONFIG in the Phase 50 notebook."""
    text = NOTEBOOK_FILE.read_text()
    match = re.search(r'kernel_version\\":\s*(\d+)', text)
    if not match:
        raise SystemExit(
            f"kernel_version not found in {NOTEBOOK_FILE.name} — add to _NOTEBOOK_CONFIG"
        )
    return int(match.group(1))


def read_versions_md_target() -> int | None:
    """Read 'Current notebook target: V47' from VERSIONS.md if present."""
    if not VERSIONS_FILE.exists():
        return None
    match = re.search(
        r"Current notebook target:\s*\*\*V(\d+)\*\*",
        VERSIONS_FILE.read_text(),
    )
    return int(match.group(1)) if match else None


def require_kernel_version_sync() -> int:
    """Abort if notebook, CANONICAL_KERNEL_VERSION, and VERSIONS.md disagree."""
    nb_ver = read_notebook_kernel_version()
    versions_ver = read_versions_md_target()

    errors: list[str] = []
    if nb_ver != CANONICAL_KERNEL_VERSION:
        errors.append(
            f"notebook kernel_version={nb_ver} != "
            f"CANONICAL_KERNEL_VERSION={CANONICAL_KERNEL_VERSION} in kaggle_launch.py"
        )
    if versions_ver is not None and versions_ver != nb_ver:
        errors.append(
            f"VERSIONS.md target V{versions_ver} != notebook kernel_version={nb_ver}"
        )
    if versions_ver is not None and versions_ver != CANONICAL_KERNEL_VERSION:
        errors.append(
            f"VERSIONS.md target V{versions_ver} != "
            f"CANONICAL_KERNEL_VERSION={CANONICAL_KERNEL_VERSION}"
        )

    if errors:
        print("KERNEL VERSION MISMATCH — fix before push:\n")
        for err in errors:
            print(f"  • {err}")
        print(
            "\nSync checklist:\n"
            f"  1. tirramind_kaggle_phase50.ipynb → kernel_version: {CANONICAL_KERNEL_VERSION}\n"
            f"  2. scripts/kaggle_launch.py → CANONICAL_KERNEL_VERSION = {CANONICAL_KERNEL_VERSION}\n"
            f"  3. VERSIONS.md → Current notebook target: **V{CANONICAL_KERNEL_VERSION}**\n"
        )
        sys.exit(1)

    print(f"✓ Kernel version sync OK: V{nb_ver} (notebook = launcher = VERSIONS.md)")
    return nb_ver


# ── notebook config fingerprint (sha256 of sorted JSON, same as notebook cell) ─


def read_notebook_config() -> dict:
    """Parse _NOTEBOOK_CONFIG from the Phase 50 notebook banner cell."""
    nb = json.loads(NOTEBOOK_FILE.read_text())
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        if "_NOTEBOOK_CONFIG" not in src:
            continue
        ns: dict = {}
        config_block = src.split("_cfg_str")[0]
        exec(config_block, ns)  # noqa: S102
        cfg = ns.get("_NOTEBOOK_CONFIG")
        if isinstance(cfg, dict):
            return cfg
    raise SystemExit(
        f"_NOTEBOOK_CONFIG not found in {NOTEBOOK_FILE.name} — add banner cell"
    )


def compute_config_fingerprint(config: dict) -> str:
    """12-char hex fingerprint; identical logic to tirramind_kaggle_phase50.ipynb."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def read_notebook_fingerprint() -> tuple[str, dict]:
    """Return (fingerprint, config) for the current notebook."""
    cfg = read_notebook_config()
    return compute_config_fingerprint(cfg), cfg


def update_versions_md_fingerprint(
    kernel_version: int, fingerprint: str, fix_label: str
) -> None:
    """Append fingerprint to VERSIONS.md known table if not already present."""
    if not VERSIONS_FILE.exists():
        return
    text = VERSIONS_FILE.read_text()
    if fingerprint in text:
        return
    row = f"| `{fingerprint}` | V{kernel_version} | {fix_label} |"
    marker = "**Known fingerprints:**"
    if marker not in text:
        return
    insert_at = text.index(marker) + len(marker)
    # Skip header row + separator if present
    rest = text[insert_at:]
    text = text[:insert_at] + "\n" + row + rest
    VERSIONS_FILE.write_text(text)


def stamp_versions_md_pushed(kernel_version: int, fingerprint: str) -> None:
    """Set pushed_at on the V{N} row in VERSIONS.md if still blank."""
    if not VERSIONS_FILE.exists():
        return
    text = VERSIONS_FILE.read_text()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    pattern = rf"(\| \*\*V{kernel_version}\*\* \| )—( \| —)"
    new_text, n = re.subn(pattern, rf"\g<1>{now}\g<2>", text, count=1)
    if n:
        VERSIONS_FILE.write_text(new_text)


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
    state_ver = state.get("kernel_version", "?")
    print(f"  Version : V{state_ver}  (repo canonical V{CANONICAL_KERNEL_VERSION})")
    fp = state.get("fingerprint")
    if fp:
        print(f"  Fingerprint : {fp}  (fix: {state.get('fix', '?')})")
    if state_ver != "?" and state_ver != CANONICAL_KERNEL_VERSION:
        print(
            f"  ⚠ state kernel_version={state_ver} != CANONICAL_KERNEL_VERSION={CANONICAL_KERNEL_VERSION}"
        )
    print(f"  Epochs  : {state.get('epochs')}")
    gpu = state.get("enable_gpu")
    if gpu is not None:
        print(f"  Accel   : {'GPU' if gpu else 'CPU'}")
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

        retrain = stage / "scripts" / "retrain_gnn.py"
        if not retrain.exists():
            raise FileNotFoundError(f"Packaged dataset missing {retrain}")
        retrain_text = retrain.read_text()
        for marker in ("--preset", "_apply_training_preset"):
            if marker not in retrain_text:
                raise RuntimeError(
                    f"tirramind-code package stale: scripts/retrain_gnn.py missing {marker!r}"
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
                f"V{CANONICAL_KERNEL_VERSION} Phase50 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC",
                "--dir-mode",
                "zip",
            ],
            check=True,
        )
    print("  ✓ tirramind-code uploaded")


# ── step 2: push kernel ───────────────────────────────────────────────────────

_GPU_QUOTA_MARKERS = (
    "maximum batch gpu",
    "gpu session count",
    "gpu quota",
    "no gpu available",
    "insufficient gpu",
    "gpu limit",
)


def _is_gpu_quota_error(text: str) -> bool:
    """True when Kaggle rejected the push due to GPU capacity / session limits."""
    lower = text.lower()
    return any(m in lower for m in _GPU_QUOTA_MARKERS)


def _prepare_and_push(epochs: int, *, enable_gpu: bool) -> tuple[int, str]:
    """Write kernel-metadata and run `kaggle kernels push`. Returns (rc, output)."""
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

    meta = {
        "id": KERNEL_SLUG,
        "title": "tirramind-phase50",
        "code_file": NOTEBOOK_FILE.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": enable_gpu,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": [],
        "dataset_sources": [
            "deeperisbetter/tirramind-data",
            CODE_DATASET,
        ],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    (ROOT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

    result = subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(ROOT)],
        capture_output=True,
        text=True,
    )
    combined = f"{result.stdout or ''}\n{result.stderr or ''}".strip()
    return result.returncode, combined


def push_kernel(
    epochs: int,
    kernel_version: int,
    *,
    enable_gpu: bool = True,
    fallback_cpu: bool = True,
) -> bool:
    """Push kernel. On GPU quota errors, automatically retries with CPU."""
    fingerprint, cfg = read_notebook_fingerprint()
    fix_label = str(cfg.get("fix", ""))

    accelerators: list[tuple[bool, str]] = []
    if enable_gpu:
        accelerators.append((True, "GPU"))
        if fallback_cpu:
            accelerators.append((False, "CPU"))
    else:
        accelerators.append((False, "CPU"))

    last_output = ""
    for use_gpu, label in accelerators:
        print(f"\n[2/4] Pushing kernel V{kernel_version} (epochs={epochs}, {label})...")
        print(f"  Config fingerprint: {fingerprint}  fix={fix_label}")

        rc, last_output = _prepare_and_push(epochs, enable_gpu=use_gpu)
        print(f"  {last_output}")

        gpu_blocked = _is_gpu_quota_error(last_output)
        if rc != 0 or gpu_blocked:
            if use_gpu and fallback_cpu and len(accelerators) > 1:
                print("  ⚠ GPU push blocked (quota/limit) — falling back to CPU...")
                continue
            print(f"  ERROR: push failed (rc={rc})")
            if last_output:
                print(f"  {last_output}")
            sys.exit(1)

        write_state(
            {
                "kernel_slug": KERNEL_SLUG,
                "kernel_version": kernel_version,
                "kaggle_ui_version": kernel_version,
                "epochs": epochs,
                "enable_gpu": use_gpu,
                "pushed_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": None,
                "fingerprint": fingerprint,
                "fix": fix_label,
                "config": cfg,
                "url": KAGGLE_URL,
                "kernel_status": "RUNNING",
                "kernel_status_note": (
                    "cpu_fallback_after_gpu_quota" if not use_gpu and enable_gpu else None
                ),
                "log_path": None,
            }
        )
        update_versions_md_fingerprint(kernel_version, fingerprint, fix_label)
        stamp_versions_md_pushed(kernel_version, fingerprint)
        if not use_gpu and enable_gpu:
            print("  ✓ Kernel pushed on CPU (GPU quota fallback)")
        else:
            print(f"  ✓ Kernel pushed  →  {KAGGLE_URL}")
        print(f"  ✓ Fingerprint {fingerprint} → {STATE_FILE}")
        return use_gpu

    print(f"  ERROR: push failed after all accelerators tried.\n  {last_output}")
    sys.exit(1)


# ── step 3: tail logs ─────────────────────────────────────────────────────────


def get_logs_text() -> str:
    """Fetch full log text from the last kernel run, decoded from JSON entries."""
    result = subprocess.run(
        ["kaggle", "kernels", "logs", KERNEL_SLUG],
        capture_output=True,
        text=True,
    )
    raw = result.stdout.strip()
    if not raw:
        return ""
    lines = []
    for l in raw.split("\n"):
        try:
            d = json.loads(l.strip())
            t = d.get("data", "").rstrip()
            if t:
                lines.append(t)
        except Exception:
            pass
    return "\n".join(lines)


def classify_failure(log_text: str) -> str:
    """Return a short failure category string for retry decisions."""
    if "INCOMPATIBLE GPU" in log_text or "sm_60" in log_text or "P100" in log_text:
        return "bad_gpu"
    if "CUDA error" in log_text or "no kernel image" in log_text:
        return "cuda_error"
    if "Out of memory" in log_text or "CUDA out of memory" in log_text:
        return "oom"
    return "unknown"


def tail_logs() -> tuple[str, str]:
    """Stream logs with `kaggle kernels logs -f`.
    Returns (status, captured_log_text) so failure diagnosis uses the
    same text we already streamed, avoiding the empty-log bug from a
    secondary `kaggle kernels logs` call after `-f` exits.
    """
    slug = read_state().get("kernel_slug", KERNEL_SLUG)
    print(f"\n[3/4] Tailing logs (Ctrl+C to stop and keep kernel running)...")
    print(f"      {KAGGLE_URL}\n")

    proc = subprocess.Popen(
        ["kaggle", "kernels", "logs", "-f", "--interval", "10", slug],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None

    captured_lines: list[str] = []
    for line in proc.stdout:
        print(line, end="")
        line = line.rstrip("\n")
        if not line:
            continue
        try:
            d = json.loads(line)
            txt = d.get("data", "").rstrip()
            if txt:
                captured_lines.append(txt)
        except Exception:
            captured_lines.append(line)

    proc.wait()
    log_text = "\n".join(captured_lines)

    # Check final status
    result = subprocess.run(
        ["kaggle", "kernels", "status", slug],
        capture_output=True,
        text=True,
    )
    status_line = result.stdout.strip()
    print(f"\n  Final status: {status_line}")

    if "COMPLETE" in status_line.upper():
        return "complete", log_text
    if "ERROR" in status_line.upper() or "CANCEL" in status_line.upper():
        return "failed", log_text
    return "unknown", log_text


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

    # Promote checkpoints (written to phase50_ckpts/ inside the output)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_subdir = DOWNLOAD_DIR / "phase50_ckpts"
    search_dirs = [ckpt_subdir, DOWNLOAD_DIR]
    ckpts: list[Path] = []
    for d in search_dirs:
        ckpts = sorted(d.glob("epoch_*.pt"))
        if ckpts:
            break
    for ckpt in ckpts:
        shutil.copy2(ckpt, CKPT_DIR / ckpt.name)
        print(f"  ✓ {ckpt.name}")
    if not ckpts:
        print(f"  ✗ No epoch_*.pt found in output (check {DOWNLOAD_DIR})")


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
    p.add_argument(
        "--verify-version",
        action="store_true",
        help="Check notebook ↔ launcher ↔ VERSIONS.md version sync and exit",
    )
    p.add_argument(
        "--no-gpu",
        action="store_true",
        help="Push kernel with enable_gpu=false (CPU-only; notebook auto-selects device)",
    )
    args = p.parse_args()

    if args.verify_version:
        require_kernel_version_sync()
        fp, cfg = read_notebook_fingerprint()
        print(f"✓ Config fingerprint: {fp}  fix={cfg.get('fix', '?')}")
        return

    if args.status:
        show_status()
        return

    if args.backtest_only:
        run_backtest()
        return

    if args.logs_only:
        status, _ = tail_logs()
        print(f"\n  Final status: {status}")
        return

    if args.download_only:
        download_outputs()
        run_backtest()
        return

    kernel_version = require_kernel_version_sync()

    # Full flow  (auto-retry up to 6 times on bad GPU assignment)
    enable_gpu = not args.no_gpu
    MAX_RETRIES = 1 if not enable_gpu else 6
    upload_code_dataset()

    for attempt in range(1, MAX_RETRIES + 1):
        if attempt > 1:
            print(f"\n  Retry {attempt}/{MAX_RETRIES} — repushing kernel...")
        push_kernel(args.epochs, kernel_version, enable_gpu=enable_gpu)

        if args.push_only:
            print(
                f"\nDone. To tail logs:\n  python scripts/kaggle_launch.py --logs-only"
            )
            return

        status, log_text = tail_logs()

        if status == "complete":
            break

        # Diagnose the failure using the captured log text
        failure = classify_failure(log_text)
        print(f"\n  Failure category: {failure}")

        if failure == "bad_gpu" and attempt < MAX_RETRIES:
            print("  Kaggle assigned an incompatible GPU (P100 sm_60).")
            print("  Repushing to get a different GPU assignment...")
            time.sleep(10)
            continue

        # Non-retryable or out of retries
        print(f"\nTraining failed ({failure}). Last 20 log lines:")
        for line in log_text.splitlines()[-20:]:
            print(" ", line)
        print("\nFull logs:  kaggle kernels logs", KERNEL_SLUG)
        sys.exit(1)

    download_outputs()
    run_backtest()


if __name__ == "__main__":
    main()
