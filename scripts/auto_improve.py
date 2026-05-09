#!/usr/bin/env python3
"""
auto_improve.py — Karpathy-style self-improvement loop for TirraMind GNN training.

Watches a checkpoint directory for new epoch_*.pt files. After each new epoch
it evaluates whether the return loss (ListNet IC-proxy) has stagnated. If so,
it writes a trigger file and calls auto_research.py to start the research loop.

Usage (run alongside training, in a separate terminal):
    python scripts/auto_improve.py --checkpoint-dir .tirra_pipeline/checkpoints

Or one-shot check on an already-completed run:
    python scripts/auto_improve.py --checkpoint-dir .tirra_pipeline/checkpoints --no-watch

The trigger file written to knowledge/trigger_{timestamp}.md is then read by:
    python scripts/auto_research.py --from-trigger knowledge/trigger_<ts>.md --github-search

Architecture:
    retrain_gnn.py  →  epoch_*.pt files
    auto_improve.py →  watches *.pt → detects stagnation → writes trigger
                    →  calls auto_research.py → triage_*.md
    Claude skill    →  reads triage + papers → diag_*.md → code patch
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Stagnation detection
# ---------------------------------------------------------------------------

STAGNATION_WINDOW = 5      # look at the last N epochs
STAGNATION_THRESHOLD = 0.005  # return loss must improve by at least this fraction
MIN_EPOCHS_BEFORE_CHECK = 3   # don't trigger before we have enough history


def _load_history_from_checkpoint(ckpt_path: Path) -> dict[str, list[float]] | None:
    """Load training history from a PyTorch checkpoint file.

    Uses torch only if available; falls back to a raw pickle read so this script
    can run without a GPU/CUDA environment.
    """
    try:
        import torch  # noqa: PLC0415
        payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        return payload.get("history")
    except ImportError:
        pass

    # Fallback: raw pickle (checkpoint is a dict saved with torch.save = pickle)
    import pickle  # noqa: PLC0415
    try:
        with open(ckpt_path, "rb") as fh:
            payload = pickle.load(fh)  # noqa: S301 — local file only
        return payload.get("history")
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARN] Could not load {ckpt_path.name}: {exc}", file=sys.stderr)
        return None


def _is_stagnant(return_losses: list[float], window: int, threshold: float) -> bool:
    """Return True if the return loss has not improved by `threshold` fraction
    over the last `window` epochs.

    Improvement = (max_recent - min_recent) / (max_recent + 1e-8).
    If improvement < threshold → stagnant.
    """
    if len(return_losses) < window:
        return False
    recent = return_losses[-window:]
    best = min(recent)   # lower return loss = better
    worst = max(recent)
    improvement = (worst - best) / (worst + 1e-8)
    return improvement < threshold


def _diagnose_stagnation(return_losses: list[float], window: int) -> str:
    """Return a human-readable diagnosis of the stagnation pattern."""
    recent = return_losses[-window:]
    mean = sum(recent) / len(recent)
    slope_num = recent[-1] - recent[0]
    if slope_num > 0:
        pattern = "increasing (diverging)"
    elif abs(slope_num) < 1e-6:
        pattern = "completely flat"
    else:
        improvement_pct = abs(slope_num) / (recent[0] + 1e-8) * 100
        pattern = f"very slowly improving ({improvement_pct:.1f}% over {window} epochs)"
    return f"Return loss {pattern} over last {window} epochs (mean={mean:.4f}, " \
           f"first={recent[0]:.4f}, last={recent[-1]:.4f})"


# ---------------------------------------------------------------------------
# Trigger file writer
# ---------------------------------------------------------------------------

KNOWN_CAUSES = [
    "92%+ GDELT node type imbalance dominates GNN embedding space",
    "Return head receiving <2% of gradient budget (dt/ret loss ratio too high)",
    "ListNet temperature tau may need tuning for this data distribution",
    "Node embedding collapse — all instruments mapped to similar representation",
    "Learning rate too high — return loss oscillating rather than converging",
    "Cross-domain message passing depth insufficient for GDELT→instrument signal",
]


def write_trigger_file(
    return_losses: list[float],
    epoch: int,
    checkpoint_dir: Path,
    knowledge_dir: Path,
) -> Path:
    """Write a trigger file describing the stagnation for auto_research.py."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trigger_path = knowledge_dir / f"trigger_{ts}.md"
    knowledge_dir.mkdir(parents=True, exist_ok=True)

    diagnosis = _diagnose_stagnation(return_losses, STAGNATION_WINDOW)
    recent = return_losses[-STAGNATION_WINDOW:]
    losses_str = ", ".join(f"{x:.4f}" for x in recent)

    content = f"""---
title: "Auto-Improve Trigger: IC stagnation at epoch {epoch}"
tags:
  - doc/research
  - topic/auto-research
  - topic/training
  - status/active
---

# Auto-Improve Trigger — Epoch {epoch}

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Checkpoint dir:** `{checkpoint_dir}`
**Trigger condition:** Return loss stagnation detected

---

## Problem

Return loss (ListNet IC-proxy) has stagnated at epoch {epoch}.

{diagnosis}

**Last {STAGNATION_WINDOW} epochs of return loss:** {losses_str}

This suggests the GNN is NOT improving its cross-sectional return ranking ability
despite continued training. IC (Spearman rank correlation) is likely near zero.

---

## Known Likely Causes (TirraMind-Specific)

{chr(10).join(f"- {c}" for c in KNOWN_CAUSES)}

---

## Next Step

Run the research triage tool:

```bash
python scripts/auto_research.py \\
  --from-trigger {trigger_path} \\
  --github-search \\
  --max-papers 5
```

Then invoke the Claude skill in Copilot chat:

```
research this training issue: "GNN return loss stagnation at epoch {epoch}, {diagnosis}"
```

---

## Related

- [[auto_ml_researcher_task]]
- [[auto_ml_researcher]]
"""
    trigger_path.write_text(content)
    return trigger_path


# ---------------------------------------------------------------------------
# Core monitoring loop
# ---------------------------------------------------------------------------

def get_sorted_checkpoints(checkpoint_dir: Path) -> list[Path]:
    """Return all epoch_*.pt files sorted by epoch number."""
    return sorted(checkpoint_dir.glob("epoch_*.pt"))


def check_once(
    checkpoint_dir: Path,
    knowledge_dir: Path,
    auto_research: bool,
    stagnation_window: int,
    stagnation_threshold: float,
    min_epochs: int,
) -> bool:
    """Check the latest checkpoint for stagnation. Returns True if triggered."""
    checkpoints = get_sorted_checkpoints(checkpoint_dir)
    if not checkpoints:
        print(f"  No epoch_*.pt files found in {checkpoint_dir}", file=sys.stderr)
        return False

    latest = checkpoints[-1]
    epoch = int(latest.stem.split("_")[1])  # epoch_021.pt → 21

    if epoch < min_epochs:
        print(f"  Epoch {epoch} < min_epochs ({min_epochs}), skipping.", file=sys.stderr)
        return False

    history = _load_history_from_checkpoint(latest)
    if history is None:
        print(f"  Could not load history from {latest.name}", file=sys.stderr)
        return False

    return_losses = history.get("return", [])
    if len(return_losses) < stagnation_window:
        print(
            f"  Only {len(return_losses)} return loss entries — need {stagnation_window}",
            file=sys.stderr,
        )
        return False

    diagnosis = _diagnose_stagnation(return_losses, stagnation_window)
    print(f"  Epoch {epoch}: {diagnosis}")

    if _is_stagnant(return_losses, stagnation_window, stagnation_threshold):
        print(f"\n[AUTO-IMPROVE] ⚠ Stagnation detected at epoch {epoch}!")
        trigger_path = write_trigger_file(return_losses, epoch, checkpoint_dir, knowledge_dir)
        print(f"[AUTO-IMPROVE] Trigger written → {trigger_path}")

        if auto_research:
            print("[AUTO-IMPROVE] Running auto_research.py...")
            script = Path(__file__).parent / "auto_research.py"
            result = subprocess.run(
                [
                    sys.executable, str(script),
                    "--from-trigger", str(trigger_path),
                    "--github-search",
                    "--max-papers", "5",
                ],
                capture_output=False,  # let it print to stdout/stderr live
            )
            if result.returncode == 0:
                print("[AUTO-IMPROVE] ✓ Triage report written to knowledge/")
                print("[AUTO-IMPROVE] → Now invoke Claude skill in Copilot chat:")
                print(f'    research this training issue: "{diagnosis}"')
            else:
                print("[AUTO-IMPROVE] ⚠ auto_research.py returned non-zero exit code")
        return True
    else:
        print(f"  ✓ Return loss improving — no trigger needed.")
        return False


def watch_loop(
    checkpoint_dir: Path,
    knowledge_dir: Path,
    auto_research: bool,
    poll_interval: int,
    stagnation_window: int,
    stagnation_threshold: float,
    min_epochs: int,
) -> None:
    """Poll the checkpoint directory every poll_interval seconds."""
    seen_checkpoints: set[str] = set()
    print(f"[AUTO-IMPROVE] Watching {checkpoint_dir} (poll every {poll_interval}s)")
    print(f"  Stagnation window={stagnation_window} epochs, threshold={stagnation_threshold:.3f}")
    print(f"  Press Ctrl+C to stop.\n")

    while True:
        checkpoints = get_sorted_checkpoints(checkpoint_dir)
        new = [c for c in checkpoints if c.name not in seen_checkpoints]

        if new:
            for ckpt in new:
                seen_checkpoints.add(ckpt.name)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] New checkpoint(s): {[c.name for c in new]}")
            check_once(
                checkpoint_dir, knowledge_dir, auto_research,
                stagnation_window, stagnation_threshold, min_epochs,
            )
            print()

        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TirraMind auto-improve: watches GNN checkpoints for IC stagnation "
                    "and triggers the research loop automatically."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(".tirra_pipeline/checkpoints"),
        help="Directory containing epoch_*.pt checkpoints (default: .tirra_pipeline/checkpoints)",
    )
    parser.add_argument(
        "--knowledge-dir",
        type=Path,
        default=Path("knowledge"),
        help="Directory to write trigger files (default: knowledge/)",
    )
    parser.add_argument(
        "--no-watch",
        action="store_true",
        help="Check once and exit (don't poll). Useful for CI / post-training analysis.",
    )
    parser.add_argument(
        "--no-auto-research",
        action="store_true",
        help="Write trigger file only — don't call auto_research.py automatically.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Seconds between checkpoint dir polls in watch mode (default: 60).",
    )
    parser.add_argument(
        "--stagnation-window",
        type=int,
        default=STAGNATION_WINDOW,
        help=f"Number of recent epochs to evaluate for stagnation (default: {STAGNATION_WINDOW}).",
    )
    parser.add_argument(
        "--stagnation-threshold",
        type=float,
        default=STAGNATION_THRESHOLD,
        help=f"Min fractional improvement required to not trigger (default: {STAGNATION_THRESHOLD}).",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=MIN_EPOCHS_BEFORE_CHECK,
        help=f"Skip check if fewer than this many epochs completed (default: {MIN_EPOCHS_BEFORE_CHECK}).",
    )

    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    if not checkpoint_dir.exists():
        print(f"ERROR: checkpoint dir not found: {checkpoint_dir}", file=sys.stderr)
        sys.exit(1)

    auto_research = not args.no_auto_research

    if args.no_watch:
        triggered = check_once(
            checkpoint_dir,
            args.knowledge_dir,
            auto_research,
            args.stagnation_window,
            args.stagnation_threshold,
            args.min_epochs,
        )
        sys.exit(0 if not triggered else 2)  # exit 2 = stagnation detected
    else:
        try:
            watch_loop(
                checkpoint_dir,
                args.knowledge_dir,
                auto_research,
                args.poll_interval,
                args.stagnation_window,
                args.stagnation_threshold,
                args.min_epochs,
            )
        except KeyboardInterrupt:
            print("\n[AUTO-IMPROVE] Stopped.")


if __name__ == "__main__":
    main()
