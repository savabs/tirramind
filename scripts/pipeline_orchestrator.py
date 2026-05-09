"""pipeline_orchestrator.py — Fully automated GNN training loop for Kaggle.

Replaces notebook cell 11. Runs 5-epoch training blocks in a loop until the
time budget is exhausted, a structural halt is detected, or the model collapses.

After each block it calls auto_improve.py to classify the training pattern
and optionally write next_config.json. At session end it pushes lightweight
state files (metrics.jsonl, next_config.json, improvement_history.jsonl,
session_summary) to the `training-state` branch on GitHub so GitHub Actions
can trigger the next Kaggle run automatically.

NEVER imports torch — this is the outer shell only.

Exit codes:
    0  — session ended normally (budget exhausted or improving)
    1  — config changed; state synced; expects cross-session continuation
    2  — structural halt; human review required (GitHub Issue opened by Actions)
    3  — model collapse detected; human review required
    4  — training crash (retrain_gnn.py exited non-zero)
"""

from __future__ import annotations

import argparse
import enum
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class PipelineState(enum.Enum):
    TRAINING_BLOCK = "training_block"
    IMPROVING = "improving"
    CONFIG_CHANGED = "config_changed"
    STRUCTURAL_HALT = "structural_halt"
    COLLAPSE_HALT = "collapse_halt"
    SESSION_END = "session_end"
    CRASH = "crash"


# ---------------------------------------------------------------------------
# Helper: find latest checkpoint epoch
# ---------------------------------------------------------------------------


def find_latest_checkpoint(checkpoint_dir: Path) -> int:
    """Return the highest epoch number from epoch_*.pt files, or 0 if none."""
    pts = list(checkpoint_dir.glob("epoch_*.pt"))
    if not pts:
        return 0
    epochs = []
    for p in pts:
        try:
            epochs.append(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return max(epochs) if epochs else 0


# ---------------------------------------------------------------------------
# Helper: collapse detection (reads metrics.jsonl, no torch)
# ---------------------------------------------------------------------------


def load_metrics_tail(metrics_path: Path, n: int = 3) -> list[dict]:
    """Load the last n epoch records from metrics.jsonl."""
    if not metrics_path.exists():
        return []
    records = []
    with metrics_path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records[-n:] if len(records) >= n else records


def is_collapsed(checkpoint_dir: Path) -> bool:
    """Return True if training has catastrophically diverged.

    Criteria (OR):
    - Total loss is >10x the first-epoch total loss for 3 consecutive epochs.
    - Return loss mean < -0.05 for the last 3 epochs (negative IC signal).
    """
    metrics_path = checkpoint_dir / "metrics.jsonl"
    all_records = load_metrics_tail(metrics_path, n=100)
    if len(all_records) < 3:
        return False

    tail = all_records[-3:]

    # Criterion 1: total loss explosion
    first_total = all_records[0].get("loss", {}).get("total", float("nan"))
    if not math.isnan(first_total) and first_total > 1e-8:
        explosion_threshold = first_total * 10.0
        tail_totals = [r.get("loss", {}).get("total", float("nan")) for r in tail]
        if all(not math.isnan(t) and t > explosion_threshold for t in tail_totals):
            return True

    # Criterion 2: negative return loss proxy (IC < -0.05 for 3 epochs)
    tail_returns = [r.get("loss", {}).get("return", float("nan")) for r in tail]
    if all(not math.isnan(r) and r < -0.05 for r in tail_returns):
        return True

    return False


# ---------------------------------------------------------------------------
# Helper: build retrain command (matches cell 11 flags exactly)
# ---------------------------------------------------------------------------


def build_retrain_cmd(
    *,
    work_dir: Path,
    checkpoint_dir: Path,
    db_path: Path,
    target_epoch: int,
    resume_epoch: int,
    device: str,
    config_file: Path | None,
    model_out: str = ".tirra_pipeline/gnn_model_h_g.pt",
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
) -> list[str]:
    """Build the retrain_gnn.py subprocess command.

    Mirrors the flags used in Kaggle notebook cell 11 exactly.
    If config_file is provided and exists, appends --config-file.
    """
    cmd = [
        sys.executable,
        str(work_dir / "scripts" / "retrain_gnn.py"),
        "--epochs",
        str(target_epoch),
        "--hidden-dim",
        "128",
        "--num-layers",
        "2",
        "--num-heads",
        "2",
        "--lr",
        "1e-3",
        "--backup",
        "--window-size",
        "604800",
        "--gdelt-frac",
        "0.05",
        "--max-windows",
        "200",
        "--auto-tune",
        "--listnet",
        "--return-log-var-max",
        "0.0",
        "--device",
        device,
        "--skip-eval",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--model-out",
        model_out,
        "--resume",
        str(resume_epoch),
    ]
    if config_file is not None and config_file.exists():
        cmd += ["--config-file", str(config_file)]
    if wandb_project:
        cmd += ["--wandb-project", wandb_project]
    if wandb_run_name:
        cmd += ["--wandb-run-name", wandb_run_name]
    return cmd


# ---------------------------------------------------------------------------
# Helper: write session summary (frontmatter machine-readable by GH Actions)
# ---------------------------------------------------------------------------


def write_session_summary(
    *,
    knowledge_dir: Path,
    state: PipelineState,
    start_epoch: int,
    end_epoch: int,
    blocks_completed: int,
    session_duration_hours: float,
    last_pattern: str,
    flag_overrides: dict | None,
) -> Path:
    """Write a session_summary_YYYYMMDD_HHMM.md to knowledge_dir.

    The YAML frontmatter is parsed by GitHub Actions via grep — keep it clean.
    """
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    summary_path = knowledge_dir / f"session_summary_{ts}.md"

    overrides_str = json.dumps(flag_overrides or {})
    content = f"""---
state: {state.value}
epoch: {end_epoch}
pattern: {last_pattern}
flag_overrides: {overrides_str}
blocks_completed: {blocks_completed}
session_duration_hours: {session_duration_hours:.2f}
---

# Training Session Summary

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Epochs this session:** {start_epoch} → {end_epoch}
**Blocks completed:** {blocks_completed}
**Duration:** {session_duration_hours:.1f} hours
**Final state:** `{state.value}`
**Pattern detected:** `{last_pattern}`
**Flag overrides applied:** `{overrides_str}`

## Notes

- Session ended at epoch {end_epoch}
- Next run should resume from epoch {end_epoch}
- State synced to training-state branch on GitHub
"""
    summary_path.write_text(content, encoding="utf-8")
    return summary_path


# ---------------------------------------------------------------------------
# Helper: sync state to GitHub (training-state branch)
# ---------------------------------------------------------------------------


def sync_state_to_github(
    *,
    work_dir: Path,
    checkpoint_dir: Path,
    knowledge_dir: Path,
    summary_path: Path,
    state: PipelineState,
    end_epoch: int,
    github_token: str | None,
) -> bool:
    """Push lightweight state files to the training-state branch on GitHub.

    Files pushed:
        - checkpoint_dir/metrics.jsonl
        - checkpoint_dir/next_config.json  (if exists)
        - knowledge_dir/improvement_history.jsonl  (if exists)
        - summary_path  (session summary just written)

    Returns True on success, False on failure (non-fatal — caller continues).
    """
    files_to_stage: list[Path] = []
    for candidate in [
        checkpoint_dir / "metrics.jsonl",
        checkpoint_dir / "next_config.json",
        knowledge_dir / "improvement_history.jsonl",
        summary_path,
    ]:
        if candidate.exists():
            files_to_stage.append(candidate)

    if not files_to_stage:
        print("[sync] No state files to sync — skipping.", flush=True)
        return True

    # Set git remote URL with write token (if provided)
    if github_token:
        _set_remote = subprocess.run(
            [
                "git",
                "remote",
                "set-url",
                "origin",
                f"https://{github_token}@github.com/savabs/tirramind.git",
            ],
            cwd=str(work_dir),
            capture_output=True,
        )
        if _set_remote.returncode != 0:
            print(
                f"[sync] WARNING: failed to set remote URL: "
                f"{_set_remote.stderr.decode(errors='replace')}",
                flush=True,
            )

    # Stage state files
    add_result = subprocess.run(
        ["git", "add"] + [str(f) for f in files_to_stage],
        cwd=str(work_dir),
        capture_output=True,
        text=True,
    )
    if add_result.returncode != 0:
        print(f"[sync] WARNING: git add failed: {add_result.stderr}", flush=True)
        return False

    # Commit
    commit_msg = f"training: epoch {end_epoch} | state={state.value}"
    commit_result = subprocess.run(
        ["git", "commit", "-m", commit_msg],
        cwd=str(work_dir),
        capture_output=True,
        text=True,
    )
    if commit_result.returncode != 0:
        # Likely "nothing to commit" — treat as success
        if "nothing to commit" in commit_result.stdout + commit_result.stderr:
            print("[sync] Nothing new to commit — state already current.", flush=True)
            return True
        print(f"[sync] WARNING: git commit failed: {commit_result.stderr}", flush=True)
        return False

    # Force-push to training-state branch (single-commit branch, not main)
    push_result = subprocess.run(
        ["git", "push", "origin", "HEAD:training-state", "--force"],
        cwd=str(work_dir),
        capture_output=True,
        text=True,
    )
    if push_result.returncode != 0:
        print(
            f"[sync] WARNING: git push failed (non-fatal): {push_result.stderr}",
            flush=True,
        )
        return False

    print(
        f"[sync] State pushed to training-state branch (epoch={end_epoch}, "
        f"state={state.value})",
        flush=True,
    )
    return True


# ---------------------------------------------------------------------------
# Helper: sync latest checkpoint + state files to Hugging Face Hub
# ---------------------------------------------------------------------------


def sync_checkpoint_to_hf(
    *,
    checkpoint_dir: Path,
    knowledge_dir: Path,
    end_epoch: int,
    hf_repo: str,
    hf_token: str | None,
) -> bool:
    """Push latest checkpoint + state files to a HF Hub dataset repo.

    Uploads:
        - checkpoint_dir/epoch_NNN.pt  (latest only)
        - checkpoint_dir/metrics.jsonl
        - checkpoint_dir/next_config.json  (if exists)
        - knowledge_dir/improvement_history.jsonl  (if exists)

    Returns True on success, False on failure (non-fatal).

    The HF repo is expected to be a dataset repo, e.g.
    ``savabs/tirramind-hg-data``.  Files are placed in the repo root.
    Requires ``huggingface_hub`` (pre-installed on Kaggle).
    """
    if not hf_token:
        print("[hf-sync] HF_TOKEN not set — skipping HF Hub upload.", flush=True)
        return False

    try:
        from huggingface_hub import HfApi  # noqa: PLC0415
    except ImportError:
        print(
            "[hf-sync] huggingface_hub not installed — skipping.",
            flush=True,
        )
        return False

    api = HfApi(token=hf_token)

    # Ensure the repo exists (no-op if already present)
    try:
        api.create_repo(repo_id=hf_repo, repo_type="dataset", exist_ok=True, private=True)
    except Exception as exc:
        print(f"[hf-sync] WARNING: could not create/verify repo: {exc}", flush=True)
        return False

    # Collect files to upload
    to_upload: list[tuple[Path, str]] = []  # (local_path, path_in_repo)

    # Latest checkpoint only (keeps the repo small)
    ckpt_name = f"epoch_{end_epoch:03d}.pt"
    ckpt_path = checkpoint_dir / ckpt_name
    if ckpt_path.exists():
        to_upload.append((ckpt_path, ckpt_name))
    else:
        # Fallback: find the latest available checkpoint
        existing = sorted(checkpoint_dir.glob("epoch_*.pt"))
        if existing:
            to_upload.append((existing[-1], existing[-1].name))

    for fname in ["metrics.jsonl", "next_config.json"]:
        p = checkpoint_dir / fname
        if p.exists():
            to_upload.append((p, fname))

    hist = knowledge_dir / "improvement_history.jsonl"
    if hist.exists():
        to_upload.append((hist, "improvement_history.jsonl"))

    if not to_upload:
        print("[hf-sync] No files to upload.", flush=True)
        return True

    failed = []
    for local, repo_path in to_upload:
        try:
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=repo_path,
                repo_id=hf_repo,
                repo_type="dataset",
                commit_message=f"training: epoch {end_epoch} — {repo_path}",
            )
            size_mb = local.stat().st_size / 1_000_000
            print(f"[hf-sync] ✓ {repo_path} ({size_mb:.1f} MB) → {hf_repo}", flush=True)
        except Exception as exc:
            print(f"[hf-sync] WARNING: failed to upload {repo_path}: {exc}", flush=True)
            failed.append(repo_path)

    return len(failed) == 0


# ---------------------------------------------------------------------------
# Helper: load next_config.json and check if it's current
# ---------------------------------------------------------------------------


def load_config_if_current(
    config_path: Path, current_epoch: int, block_size: int
) -> Path | None:
    """Return config_path if it's valid for the current epoch, else None.

    A config file is considered stale if its based_on_epoch is more than
    2 blocks behind the current epoch (was written for a very old state).
    """
    if not config_path.exists():
        return None
    try:
        data = json.loads(config_path.read_text())
        based_on = data.get("based_on_epoch", current_epoch)
        if current_epoch - based_on > block_size * 2:
            print(
                f"[config] next_config.json is stale "
                f"(based_on={based_on}, current={current_epoch}) — discarding.",
                flush=True,
            )
            return None
    except (json.JSONDecodeError, OSError):
        pass  # If unreadable, use it anyway — retrain_gnn will validate
    return config_path


# ---------------------------------------------------------------------------
# Main orchestration loop
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:  # noqa: PLR0912, PLR0915
    """Execute the full training loop. Returns the exit code."""
    work_dir = Path(args.work_dir).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    knowledge_dir = Path(args.knowledge_dir).resolve()
    db_path = Path(args.db_path).resolve()
    block_size: int = args.block_size
    budget_seconds: float = args.total_budget_hours * 3600.0
    device: str = args.device

    session_start = time.monotonic()
    start_epoch = find_latest_checkpoint(checkpoint_dir)
    latest_epoch = start_epoch
    blocks_completed = 0
    last_pattern = "none"
    state = PipelineState.SESSION_END
    last_flag_overrides: dict | None = None

    # Resolve optional github token (Kaggle secret name: tirramind_token)
    github_token: str | None = None
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]

        github_token = UserSecretsClient().get_secret("tirramind_token")
        print(
            "[sync] GitHub token loaded from Kaggle secret 'tirramind_token'.",
            flush=True,
        )
    except Exception:
        github_token = os.environ.get("GITHUB_WRITE_TOKEN") or os.environ.get(
            "GITHUB_TOKEN"
        )
        if github_token:
            print("[sync] GitHub token loaded from environment.", flush=True)
        else:
            print(
                "[sync] WARNING: no GitHub token found — state sync will be skipped.",
                flush=True,
            )

    # Initial config file (from previous session or explicit arg)
    current_config_file: Path | None = None
    if args.config_file:
        current_config_file = load_config_if_current(
            Path(args.config_file), latest_epoch, block_size
        )
        if current_config_file:
            print(
                f"[config] Using provided config file: {current_config_file}",
                flush=True,
            )
    else:
        # Check if auto_improve left one from a previous block/session
        candidate = checkpoint_dir / "next_config.json"
        current_config_file = load_config_if_current(
            candidate, latest_epoch, block_size
        )
        if current_config_file:
            print(
                f"[config] Found previous session next_config.json — applying.",
                flush=True,
            )

    # ── One-time pre-flight: CFTC enrichment ─────────────────────────────────
    print("\n[pre-flight] Running CFTC derived feature enrichment...", flush=True)
    enrich_script = work_dir / "scripts" / "add_cftc_derived_features.py"
    if enrich_script.exists():
        enrich_result = subprocess.run(
            [sys.executable, str(enrich_script), "--db-path", str(db_path)],
            cwd=str(work_dir),
        )
        if enrich_result.returncode != 0:
            print(
                "[pre-flight] WARNING: CFTC enrichment failed (non-fatal).", flush=True
            )
        else:
            print("[pre-flight] CFTC derived features ready.", flush=True)
    else:
        print(f"[pre-flight] {enrich_script} not found — skipping.", flush=True)

    print(
        f"\n[orchestrator] Starting training loop from epoch {start_epoch}. "
        f"Budget: {args.total_budget_hours}h. Block size: {block_size} epochs.\n",
        flush=True,
    )

    # ── Main training loop ────────────────────────────────────────────────────
    headroom_seconds = 30 * 60  # stop 30 min before hard deadline

    while True:
        elapsed = time.monotonic() - session_start
        remaining = budget_seconds - elapsed
        if remaining < headroom_seconds:
            print(
                f"[orchestrator] Budget exhausted ({elapsed / 3600:.1f}h elapsed). "
                f"Ending session.",
                flush=True,
            )
            state = PipelineState.SESSION_END
            break

        target_epoch = latest_epoch + block_size
        print(
            f"\n[orchestrator] Block {blocks_completed + 1}: "
            f"epochs {latest_epoch} → {target_epoch}  "
            f"({remaining / 3600:.1f}h remaining)",
            flush=True,
        )

        # ── Train one block ───────────────────────────────────────────────────
        run_name = f"h-g-ep{latest_epoch}-{target_epoch}"
        cmd = build_retrain_cmd(
            work_dir=work_dir,
            checkpoint_dir=checkpoint_dir,
            db_path=db_path,
            target_epoch=target_epoch,
            resume_epoch=latest_epoch,
            device=device,
            config_file=current_config_file,
            wandb_project=args.wandb_project if args.wandb_project else None,
            wandb_run_name=run_name if args.wandb_project else None,
        )
        print("[orchestrator] Running:", " ".join(cmd), flush=True)

        retcode = subprocess.call(cmd, cwd=str(work_dir))

        if retcode != 0:
            print(
                f"[orchestrator] retrain_gnn.py exited {retcode} — crash halt.",
                flush=True,
            )
            state = PipelineState.CRASH
            break

        # Update latest epoch from checkpoint files (not from target_epoch,
        # in case training was interrupted mid-block)
        new_epoch = find_latest_checkpoint(checkpoint_dir)
        if new_epoch <= latest_epoch:
            print(
                "[orchestrator] WARNING: no new checkpoint after training block — "
                "something went wrong.",
                flush=True,
            )
            state = PipelineState.CRASH
            break
        latest_epoch = new_epoch
        blocks_completed += 1

        # ── Collapse detection ────────────────────────────────────────────────
        if is_collapsed(checkpoint_dir):
            print(
                "[orchestrator] Model collapse detected — stopping. "
                "Human review required.",
                flush=True,
            )
            state = PipelineState.COLLAPSE_HALT
            break

        # ── Auto-improve: pattern detection + next_config ─────────────────────
        auto_improve_script = work_dir / "scripts" / "auto_improve.py"
        ai_cmd = [
            sys.executable,
            str(auto_improve_script),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--knowledge-dir",
            str(knowledge_dir),
            "--no-watch",
        ]
        print("\n[orchestrator] Running auto_improve...", flush=True)
        ai_retcode = subprocess.call(ai_cmd, cwd=str(work_dir))

        if ai_retcode == 2:
            print(
                "[orchestrator] Structural halt signalled by auto_improve. "
                "Human review required.",
                flush=True,
            )
            state = PipelineState.STRUCTURAL_HALT
            last_pattern = "structural"
            break
        elif ai_retcode == 1:
            # Config changed — load the new next_config.json for next block
            candidate = checkpoint_dir / "next_config.json"
            if candidate.exists():
                try:
                    cfg_data = json.loads(candidate.read_text())
                    last_pattern = cfg_data.get("pattern", "config_changed")
                    last_flag_overrides = cfg_data.get("flag_overrides", {})
                except (json.JSONDecodeError, OSError):
                    last_pattern = "config_changed"
            current_config_file = candidate
            state = PipelineState.CONFIG_CHANGED
            print(
                f"[orchestrator] Config updated ({last_pattern}). "
                f"Will apply next_config.json in next block.",
                flush=True,
            )
        else:
            # Improving — clear any stale config
            current_config_file = None
            last_pattern = "improving"
            state = PipelineState.IMPROVING

    # ── Session end: write summary and sync to GitHub ─────────────────────────
    session_duration_hours = (time.monotonic() - session_start) / 3600.0

    print(
        f"\n[orchestrator] Session complete. "
        f"State={state.value}, epochs={start_epoch}→{latest_epoch}, "
        f"duration={session_duration_hours:.1f}h",
        flush=True,
    )

    summary_path = write_session_summary(
        knowledge_dir=knowledge_dir,
        state=state,
        start_epoch=start_epoch,
        end_epoch=latest_epoch,
        blocks_completed=blocks_completed,
        session_duration_hours=session_duration_hours,
        last_pattern=last_pattern,
        flag_overrides=last_flag_overrides,
    )
    print(f"[orchestrator] Session summary written → {summary_path}", flush=True)

    sync_ok = sync_state_to_github(
        work_dir=work_dir,
        checkpoint_dir=checkpoint_dir,
        knowledge_dir=knowledge_dir,
        summary_path=summary_path,
        state=state,
        end_epoch=latest_epoch,
        github_token=github_token,
    )
    if not sync_ok:
        print(
            "[orchestrator] WARNING: state sync failed — "
            "run sync manually or check GitHub token.",
            flush=True,
        )

    hf_repo = getattr(args, "hf_repo", None)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if hf_repo:
        hf_ok = sync_checkpoint_to_hf(
            checkpoint_dir=checkpoint_dir,
            knowledge_dir=knowledge_dir,
            end_epoch=latest_epoch,
            hf_repo=hf_repo,
            hf_token=hf_token,
        )
        if not hf_ok:
            print(
                "[orchestrator] WARNING: HF Hub sync failed — "
                "checkpoint may not persist. Check HF_TOKEN secret.",
                flush=True,
            )
    else:
        print(
            "[orchestrator] --hf-repo not set — checkpoint will not persist across sessions. "
            "Set --hf-repo savabs/tirramind-hg-data to enable.",
            flush=True,
        )

    # Map internal state to exit code
    exit_codes = {
        PipelineState.SESSION_END: 0,
        PipelineState.IMPROVING: 0,
        PipelineState.CONFIG_CHANGED: 1,
        PipelineState.STRUCTURAL_HALT: 2,
        PipelineState.COLLAPSE_HALT: 3,
        PipelineState.CRASH: 4,
        PipelineState.TRAINING_BLOCK: 0,
    }
    return exit_codes.get(state, 0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Automated GNN training orchestrator for Kaggle sessions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--work-dir",
        default=str(Path("/kaggle/working/tirramind_v1")),
        help="Root of the cloned repo (cwd for subprocesses).",
    )
    p.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Directory containing epoch_*.pt checkpoints and metrics.jsonl. "
        "Typically .tirra_pipeline/checkpoints/h_g/",
    )
    p.add_argument(
        "--db-path",
        required=True,
        help="Path to pipeline.db.",
    )
    p.add_argument(
        "--knowledge-dir",
        default="knowledge",
        help="Directory for improvement_history.jsonl and session summaries.",
    )
    p.add_argument(
        "--block-size",
        type=int,
        default=5,
        help="Number of epochs per training block.",
    )
    p.add_argument(
        "--total-budget-hours",
        type=float,
        default=11.0,
        help="Stop training this many hours after start (Kaggle limit is 12h).",
    )
    p.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Compute device.",
    )
    p.add_argument(
        "--config-file",
        default=None,
        help="Path to next_config.json from a previous session. "
        "If not given, checks checkpoint-dir/next_config.json automatically.",
    )
    p.add_argument(
        "--wandb-project",
        default=None,
        help="wandb project name to log metrics to (e.g. 'tirramind'). "
        "Passed through to retrain_gnn.py for each block. "
        "Requires WANDB_API_KEY env var or ~/.netrc on Kaggle.",
    )
    p.add_argument(
        "--hf-repo",
        default=None,
        help="Hugging Face Hub dataset repo to push checkpoints to after each session "
        "(e.g. 'aurabear/tirramind-hg-data'). "
        "Requires HF_TOKEN env var (add as Kaggle secret 'HF_TOKEN').",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    sys.exit(run(args))


if __name__ == "__main__":
    main()
