#!/usr/bin/env python3
"""kaggle_loop.py — Autonomous Kaggle training loop for Phase50.

Watches the current kernel run, analyzes logs when complete, patches the
notebook config, bumps the version, and pushes the next run — without
manual prompting.

Usage:
    python scripts/kaggle_loop.py --loop              # watch → analyze → patch → push (repeat)
    python scripts/kaggle_loop.py --once              # analyze last completed run only
    python scripts/kaggle_loop.py --analyze-v63       # analyze V63 logs, apply V64, push
    python scripts/kaggle_loop.py --loop --max-runs 5 --interval 90

State: .tirra_pipeline/kaggle_loop_state.json
History: .tirra_pipeline/kaggle_loop_history.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from kaggle_analyze import (  # noqa: E402
    LOOP_HISTORY,
    append_loop_history,
    decide_next_config,
    diagnose_from_file,
    diagnose_log_text,
    fetch_logs,
    print_report,
)
from kaggle_launch import (  # noqa: E402
    CANONICAL_KERNEL_VERSION,
    KERNEL_SLUG,
    NOTEBOOK_FILE,
    ROOT as LAUNCH_ROOT,
    STATE_FILE,
    VERSIONS_FILE,
    push_kernel,
    read_notebook_config,
    read_state,
    require_kernel_version_sync,
    upload_code_dataset,
    write_state,
)

LOOP_STATE = ROOT / ".tirra_pipeline" / "kaggle_loop_state.json"
LAUNCH_SCRIPT = ROOT / "scripts" / "kaggle_launch.py"


def _load_loop_state() -> dict:
    if LOOP_STATE.exists():
        try:
            return json.loads(LOOP_STATE.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def _save_loop_state(state: dict) -> None:
    LOOP_STATE.parent.mkdir(parents=True, exist_ok=True)
    LOOP_STATE.write_text(json.dumps(state, indent=2))


def _fetch_raw_logs() -> str:
    result = subprocess.run(
        ["kaggle", "kernels", "logs", KERNEL_SLUG],
        capture_output=True,
        text=True,
        timeout=90,
    )
    return result.stdout or result.stderr or ""


def _run_complete_in_logs(raw: str, *, fingerprint: str | None) -> bool:
    """True only when logs show THIS run's fingerprint AND post-train eval done."""
    if not raw:
        return False
    done_markers = ("Post-train eval complete", "ARTIFACT AUDIT")
    if not any(m in raw for m in done_markers):
        return False
    if fingerprint and fingerprint not in raw:
        return False
    return True


def _kernel_running() -> bool:
    result = subprocess.run(
        ["kaggle", "kernels", "status", KERNEL_SLUG],
        capture_output=True,
        text=True,
        timeout=30,
    )
    raw = (result.stdout + result.stderr).lower()
    if "complete" in raw:
        return False
    return "running" in raw or "queued" in raw


def _format_config_value(v) -> str:
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    return json.dumps(v)


def patch_notebook_config(cfg: dict) -> str:
    """Rewrite _NOTEBOOK_CONFIG in the notebook banner cell. Returns fingerprint."""
    import hashlib

    nb = json.loads(NOTEBOOK_FILE.read_text())
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        if "_NOTEBOOK_CONFIG" not in src:
            continue
        lines = [
            "import hashlib, json",
            "",
            "_NOTEBOOK_CONFIG = {",
        ]
        for k, v in cfg.items():
            lines.append(f'    "{k}": {_format_config_value(v)},')
        lines.append("}")
        config_block = "\n".join(lines)

        # Keep fingerprint banner code after config dict
        tail = src.split("_cfg_str", 1)
        if len(tail) == 2:
            new_src = config_block + "\n\n_cfg_str" + tail[1]
        else:
            new_src = config_block
        cell["source"] = [line + "\n" for line in new_src.split("\n")]
        break
    else:
        raise SystemExit("_NOTEBOOK_CONFIG cell not found in notebook")

    NOTEBOOK_FILE.write_text(json.dumps(nb, indent=1))
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def bump_canonical_version(new_version: int) -> None:
    text = (ROOT / "scripts" / "kaggle_launch.py").read_text()
    text = re.sub(
        r"CANONICAL_KERNEL_VERSION = \d+",
        f"CANONICAL_KERNEL_VERSION = {new_version}",
        text,
        count=1,
    )
    (ROOT / "scripts" / "kaggle_launch.py").write_text(text)

    if VERSIONS_FILE.exists():
        vtext = VERSIONS_FILE.read_text()
        vtext = re.sub(
            r"\*\*Current notebook target: V\d+\*\*",
            f"**Current notebook target: V{new_version}**",
            vtext,
            count=1,
        )
        vtext = re.sub(
            r"CANONICAL_KERNEL_VERSION = \d+",
            f"CANONICAL_KERNEL_VERSION = {new_version}",
            vtext,
            count=1,
        )
        row = (
            f"| **V{new_version}** | — | — | pending | pending | 0→10 | — | "
            f"**auto-loop** — see kaggle_loop_history.jsonl |"
        )
        if f"**V{new_version}**" not in vtext:
            marker = "| **V63**"
            if marker in vtext:
                vtext = vtext.replace(marker, row + "\n" + marker, 1)
            else:
                vtext += "\n" + row + "\n"
        VERSIONS_FILE.write_text(vtext)

    # Reload module constant for push
    import importlib

    import kaggle_launch as kl  # noqa: E402

    importlib.reload(kl)


def apply_config(next_cfg: dict) -> tuple[int, str]:
    new_ver = int(next_cfg["kernel_version"])
    bump_canonical_version(new_ver)
    fp = patch_notebook_config(next_cfg)
    require_kernel_version_sync()
    return new_ver, fp


def push_run(epochs: int) -> None:
    upload_code_dataset()
    # Re-import after version bump
    import importlib

    import kaggle_launch as kl

    importlib.reload(kl)
    kl.push_kernel(epochs, kl.CANONICAL_KERNEL_VERSION, enable_gpu=True)


def analyze_and_decide(
    *,
    kernel_version: int,
    fingerprint: str | None,
    log_path: Path | None = None,
) -> tuple[dict | None, object]:
    save = log_path or ROOT / ".tirra_pipeline" / f"kaggle_logs_v{kernel_version}.txt"
    if log_path and log_path.exists():
        diag = diagnose_from_file(
            log_path, kernel_version=kernel_version, fingerprint=fingerprint
        )
    else:
        text = fetch_logs(save_path=save)
        diag = diagnose_log_text(
            text, kernel_version=kernel_version, fingerprint=fingerprint
        )

    print_report(diag)
    current_cfg = read_notebook_config()
    new_ver = kernel_version + 1
    next_cfg = decide_next_config(current_cfg, diag, new_version=new_ver)
    return next_cfg, diag


def run_cycle(
    *,
    max_runs: int,
    interval: int,
    start_run: int = 0,
) -> int:
    """Returns number of pushes performed."""
    loop_state = _load_loop_state()
    runs_done = start_run
    last_analyzed_fp = loop_state.get("last_analyzed_fingerprint")

    while runs_done < max_runs:
        state = read_state()
        fp = state.get("fingerprint")
        kv = state.get("kernel_version", CANONICAL_KERNEL_VERSION)
        status = state.get("kernel_status", "unknown")

        print(
            f"\n[loop] Watching V{kv} fingerprint={fp} "
            f"(run {runs_done + 1}/{max_runs}, poll every {interval}s)"
        )

        # Wait until logs show this fingerprint finished (not a prior run)
        while True:
            raw = _fetch_raw_logs()
            if _run_complete_in_logs(raw, fingerprint=fp):
                break
            running = _kernel_running()
            print(
                f"  … {'running' if running else 'waiting for logs'} "
                f"({datetime.now().strftime('%H:%M:%S')})"
            )
            time.sleep(interval)

        time.sleep(10)  # let logs flush

        if fp and fp == last_analyzed_fp:
            print(f"  Already analyzed fingerprint {fp}. Waiting for new push…")
            time.sleep(interval)
            continue

        log_path = ROOT / ".tirra_pipeline" / f"kaggle_logs_v{kv}.txt"
        next_cfg, diag = analyze_and_decide(
            kernel_version=kv,
            fingerprint=fp,
            log_path=log_path if log_path.exists() else None,
        )

        append_loop_history(
            {
                "from_version": kv,
                "fingerprint": fp,
                "diagnosis": diag.to_dict(),
                "pattern": diag.pattern,
                "structural_halt": diag.structural_halt,
            }
        )

        last_analyzed_fp = fp
        loop_state["last_analyzed_fingerprint"] = fp
        loop_state["last_diagnosis"] = diag.to_dict()
        _save_loop_state(loop_state)

        if next_cfg is None or diag.structural_halt:
            print("\n[loop] HALT — no further automated pushes.")
            write_state(
                {
                    "kernel_status": "COMPLETE",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "loop_halt": True,
                    "loop_pattern": diag.pattern,
                }
            )
            return runs_done

        new_ver, new_fp = apply_config(next_cfg)
        epochs = int(next_cfg.get("epochs", 10))
        print(f"\n[loop] Pushing V{new_ver} fix={next_cfg.get('fix')} epochs={epochs}")
        push_run(epochs)

        write_state(
            {
                "kernel_slug": KERNEL_SLUG,
                "kernel_version": new_ver,
                "epochs": epochs,
                "pushed_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": None,
                "kernel_status": "RUNNING",
                "fingerprint": new_fp,
                "fix": next_cfg.get("fix"),
                "config": next_cfg,
                "enable_gpu": True,
                "loop_run": runs_done + 1,
            }
        )

        runs_done += 1
        loop_state["runs_pushed"] = runs_done
        loop_state["current_version"] = new_ver
        _save_loop_state(loop_state)

    print(f"\n[loop] Reached max_runs={max_runs}. Stopping.")
    return runs_done


def setup_v64_from_v63() -> None:
    """Bootstrap: analyze V63, apply V64 diagnostic config, push."""
    v63_log = ROOT / ".tirra_pipeline" / "kaggle_logs_v63.txt"
    if not v63_log.exists():
        print("Fetching V63 logs…")
        fetch_logs(save_path=v63_log)

    _, diag = analyze_and_decide(
        kernel_version=63,
        fingerprint="7c689eabb6d4",
        log_path=v63_log,
    )

    # V64: V52 base + grad-flow diagnostics (trainer code change)
    next_cfg = read_notebook_config()
    next_cfg.update(
        {
            "kernel_version": 64,
            "fix": "grad_flow_diag_v64",
            "resume_epoch": 0,
            "epochs": 10,
            "return_weight": 3.0,
            "obs_type_weight": 1.0,
            "time_delta_weight": 1.0,
            "value_weight": 1.0,
            "contrastive_weight": 1.0,
            "max_windows": 200,
            "gdelt_frac": 0.05,
            "defi_frac": 1.0,
            "vicreg_weight": 0.0,
            "use_contranorm": False,
            "use_log_loss": False,
            "auto_tune": True,
            "use_concat_head": True,
            "n1_doctrine": False,
            "run_full_backtest": False,
            "eval_smoke": True,
        }
    )

    append_loop_history(
        {
            "from_version": 63,
            "pattern": "bootstrap_v64",
            "diagnosis": diag.to_dict(),
            "note": "Manual bootstrap after V63 gate fail",
        }
    )

    new_ver, new_fp = apply_config(next_cfg)
    print(f"\n[loop] Bootstrap push V{new_ver} fingerprint={new_fp}")
    push_run(10)

    write_state(
        {
            "kernel_slug": KERNEL_SLUG,
            "kernel_version": new_ver,
            "epochs": 10,
            "pushed_at": datetime.now(timezone.utc).isoformat(),
            "kernel_status": "RUNNING",
            "fingerprint": new_fp,
            "fix": next_cfg["fix"],
            "config": next_cfg,
            "enable_gpu": True,
        }
    )
    _save_loop_state(
        {
            "bootstrapped_from": 63,
            "current_version": new_ver,
            "last_analyzed_fingerprint": "7c689eabb6d4",
        }
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Autonomous Kaggle Phase50 training loop")
    p.add_argument(
        "--loop",
        action="store_true",
        help="Watch current run, analyze, patch, push, repeat",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="Analyze the latest completed run and print decision (no push)",
    )
    p.add_argument(
        "--bootstrap-v64",
        action="store_true",
        help="Analyze V63, set up V64 diagnostic run, push",
    )
    p.add_argument("--max-runs", type=int, default=4, help="Max auto-pushes per loop")
    p.add_argument("--interval", type=int, default=90, help="Poll interval seconds")
    args = p.parse_args()

    if args.bootstrap_v64:
        setup_v64_from_v63()
        if args.loop:
            run_cycle(max_runs=args.max_runs, interval=args.interval, start_run=1)
        return

    if args.once:
        state = read_state()
        kv = state.get("kernel_version", 63)
        fp = state.get("fingerprint")
        log_path = ROOT / ".tirra_pipeline" / f"kaggle_logs_v{kv}.txt"
        next_cfg, diag = analyze_and_decide(
            kernel_version=kv, fingerprint=fp, log_path=log_path
        )
        if next_cfg:
            print("Proposed next config:")
            print(json.dumps(next_cfg, indent=2))
        return

    if args.loop:
        run_cycle(max_runs=args.max_runs, interval=args.interval)
        return

    p.print_help()


if __name__ == "__main__":
    main()
