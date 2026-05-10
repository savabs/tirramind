#!/usr/bin/env python3
"""
auto_improve.py — Self-improving training loop for TirraMind GNN.

Reads metrics.jsonl (one JSON line per epoch, written by trainer.py after every
checkpoint save). Analyses training patterns and produces concrete recommendations:

  1. FAST PATH (config change): writes {checkpoint_dir}/next_config.json with
     flag overrides the next run should apply. Also prints the full resumable
     command so the user can paste it into Kaggle immediately.

  2. SLOW PATH (structural): writes knowledge/trigger_{ts}.md and calls
     auto_research.py to start the paper research + Claude diagnostic loop.

Usage (one-shot, run after training ends or downloads):
    python scripts/auto_improve.py --checkpoint-dir .tirra_pipeline/checkpoints

Usage (watch mode, runs alongside training):
    python scripts/auto_improve.py --checkpoint-dir .tirra_pipeline/checkpoints --watch

The self-improvement loop (full cycle):
    retrain_gnn.py [--auto-improve]
       → metrics.jsonl per epoch
       → auto_improve.py
            → Fast path:  next_config.json   → retrain_gnn.py --config-file next_config.json
            → Slow path:  trigger_{ts}.md    → auto_research.py --from-trigger
                          triage_{slug}.md   → Claude: "research this training issue"
                          diag_{slug}.md     → Claude: "apply training fix"
                          patch_{slug}.md    → code changed → retrain_gnn.py --resume N
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAGNATION_WINDOW = 5  # epochs to evaluate
STAGNATION_THRESHOLD = 0.005  # <0.5% improvement = stagnant
MIN_EPOCHS_BEFORE_CHECK = 4  # don't trigger before enough data
OSCILLATION_CV_THRESHOLD = 0.12  # CV > 12% = oscillating

# History file: tracks what patterns have been tried
IMPROVEMENT_HISTORY_FILE = "improvement_history.jsonl"

# ---------------------------------------------------------------------------
# Metrics loading
# ---------------------------------------------------------------------------


def load_metrics(checkpoint_dir: Path) -> list[dict]:
    """Read all epoch records from metrics.jsonl. Returns list sorted by epoch."""
    metrics_path = checkpoint_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return []
    records = []
    with open(metrics_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    records.sort(key=lambda r: r.get("epoch", 0))
    return records


def load_run_config(checkpoint_dir: Path) -> dict:
    """Load run_config.json written at training start."""
    rc_path = checkpoint_dir / "run_config.json"
    if not rc_path.exists():
        return {}
    try:
        return json.loads(rc_path.read_text())
    except Exception:
        return {}


def load_improvement_history(knowledge_dir: Path) -> list[dict]:
    """Load improvement history (what patterns have been tried)."""
    hist_path = knowledge_dir / IMPROVEMENT_HISTORY_FILE
    if not hist_path.exists():
        return []
    records = []
    with open(hist_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def append_improvement_history(knowledge_dir: Path, entry: dict) -> None:
    """Append one entry to improvement_history.jsonl."""
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    hist_path = knowledge_dir / IMPROVEMENT_HISTORY_FILE
    with open(hist_path, "a") as fh:
        fh.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def _finite(values: list[float]) -> list[float]:
    return [v for v in values if not math.isnan(v) and not math.isinf(v)]


def is_stagnant(ret_losses: list[float], window: int, threshold: float) -> bool:
    """True if return loss hasn't improved by threshold fraction over window."""
    vals = _finite(ret_losses[-window:])
    if len(vals) < 2:
        return False
    best = min(vals)
    worst = max(vals)
    improvement = (worst - best) / (worst + 1e-8)
    return improvement < threshold


def trend(vals: list[float]) -> float:
    """Slope of a linear fit to vals. Positive = increasing (bad for loss)."""
    n = len(vals)
    if n < 2:
        return 0.0
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(vals) / n
    num = sum((xs[i] - x_mean) * (vals[i] - y_mean) for i in range(n))
    den = sum((xs[i] - x_mean) ** 2 for i in range(n))
    return num / (den + 1e-10)


def coefficient_of_variation(vals: list[float]) -> float:
    """Relative std (CV). High CV = oscillating."""
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    if abs(mean) < 1e-10:
        return 0.0
    var = sum((x - mean) ** 2 for x in vals) / len(vals)
    return var**0.5 / abs(mean)


# ---------------------------------------------------------------------------
# LLM-powered classification
# ---------------------------------------------------------------------------

_LLM_SYSTEM = """You are a GNN training optimizer for TirraMind, a financial prediction system.
Your job: analyse epoch-by-epoch training metrics and decide what single hyperparameter change will best improve the return prediction head.

Context:
- The model trains on heterogeneous temporal graph data: ~92% GDELT news nodes, ~8% financial instrument nodes.
- Key metric to minimize: `return_loss` (lower = better IC/alpha).
- `total_loss` = return_loss + temporal_loss. `dt_ret_ratio` = dt_loss / return_loss.
- If dt_ret_ratio > 20, the temporal objective is starving the return head.
- If return_loss variance < 0.1 over 5+ epochs despite other losses moving, the return head is frozen — gradient budget is being stolen by obs_type or dt losses.

Tunable parameters:
- lr (float, 5e-6 to 5e-3): learning rate
- return_weight (float, 0.5–10.0): gradient weight on return head — raise aggressively (8–10) when return head is frozen
- obs_type_weight (float, 0.1–1.0): weight on obs_type classification loss — lower (0.2–0.3) when obs_type loss spikes are stealing gradient budget
- gdelt_frac (float, 0.01–0.10): GDELT subsample fraction (lower = less noise)
- listnet_temperature (float, 0.1–2.0): lower = sharper IC gradient
- auto_tune (bool): uncertainty weighting — can suppress return head when ratio is high

Output ONLY a JSON object — no markdown, no text outside the JSON:
{
  "pattern": "<name>",
  "rationale": "<1-2 sentences citing specific metric values>",
  "flag_overrides": {},
  "remove_flags": [],
  "resume_epoch": <int>,
  "escalate": false
}

Valid pattern names: improving, divergence, oscillation, dt_dominance, auto_tune_suppressing, gdelt_noise, listnet_temperature, return_head_frozen, structural
Use "structural" with escalate=true only when all other options have already been tried and failed.
Use "return_head_frozen" when return_loss variance < 0.1 over the last 5+ epochs — set return_weight=10 and obs_type_weight=0.3."""


def _call_anthropic(messages: list[dict], api_key: str, model: str) -> str:
    """POST to Anthropic /v1/messages. Returns assistant content string."""
    import urllib.request

    payload = json.dumps(
        {"model": model, "max_tokens": 512, "system": _LLM_SYSTEM, "messages": messages}
    ).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=45) as resp:
        body = json.loads(resp.read())
    return body["content"][0]["text"]


def _llm_classify(
    records: list[dict],
    improvement_history: list[dict],
) -> "tuple[str, dict] | None":
    """Ask Claude to classify the training pattern.

    Returns (pattern, action) on success, or None if the LLM is unavailable
    or the response is unparseable — triggers heuristic fallback.
    """
    api_key = os.getenv("TIRRA_LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None  # no key → fall back silently

    model = os.getenv("TIRRA_LLM_MODEL", "claude-haiku-4-5-20251001")
    last = records[-1]
    last_epoch = last.get("epoch", 0)

    metrics_summary = [
        {
            "epoch": r.get("epoch"),
            "total_loss": round(r.get("loss", {}).get("total", float("nan")), 5),
            "return_loss": round(r.get("loss", {}).get("return", float("nan")), 5),
            "obs_type_loss": round(r.get("loss", {}).get("obs_type", float("nan")), 5),
            "dt_ret_ratio": round(r.get("dt_ret_ratio", float("nan")), 2),
            "trainer_warnings": r.get("warnings", []),
        }
        for r in records[-15:]
    ]
    hist_summary = [
        {
            "epoch": h.get("epoch"),
            "pattern": h.get("pattern"),
            "change": h.get("flag_overrides"),
        }
        for h in improvement_history[-8:]
    ]
    user_msg = json.dumps(
        {
            "current_epoch": last_epoch,
            "current_config": last.get("config", {}),
            "epoch_metrics": metrics_summary,
            "recent_decisions": hist_summary,
        },
        indent=2,
    )

    try:
        raw = _call_anthropic([{"role": "user", "content": user_msg}], api_key, model)
        raw = raw.strip()
        # Strip markdown fences if the model adds them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        decision = json.loads(raw.strip())
        action = {
            "rationale": decision.get("rationale", "LLM decision"),
            "flag_overrides": decision.get("flag_overrides", {}),
            "remove_flags": decision.get("remove_flags", []),
            "resume_epoch": decision.get("resume_epoch", last_epoch),
            "escalate": bool(decision.get("escalate", False)),
        }
        pattern = decision.get("pattern", "structural")
        print(f"  [LLM] Pattern classified by Claude ({model}): {pattern}")
        return pattern, action
    except Exception as exc:
        print(
            f"  [LLM] Failed ({type(exc).__name__}: {exc}). Falling back to heuristic.",
            file=sys.stderr,
        )
        return None


# ---------------------------------------------------------------------------
# Heuristic decision tree (fallback when LLM unavailable)
# ---------------------------------------------------------------------------


def _heuristic_classify(
    records: list[dict],
    improvement_history: list[dict],
    window: int = STAGNATION_WINDOW,
    stagnation_threshold: float = STAGNATION_THRESHOLD,
) -> tuple[str, dict]:
    """
    Analyse the last `window` epoch records and return (pattern_name, action).

    action keys:
      flag_overrides  dict[str, value]  — flags to add/change for next run
      remove_flags    list[str]         — flags to disable (store_true → False)
      resume_epoch    int               — resume from this epoch
      rationale       str               — human-readable explanation
      escalate        bool              — True = use research loop (slow path)
    """
    if len(records) < 2:
        return "insufficient_data", {"rationale": "Not enough epochs to analyse."}

    recent = records[-window:]
    last = records[-1]
    config = last.get("config", {})

    lr = config.get("lr", 1e-3)
    return_weight = config.get("return_weight", 1.0)
    gdelt_frac = config.get("gdelt_frac", 0.05)
    listnet_temp = config.get("listnet_temp", 1.0)
    auto_tune = config.get("auto_tune", False)
    last_epoch = last.get("epoch", 0)

    ret_losses = _finite(
        [r.get("loss", {}).get("return", float("nan")) for r in recent]
    )

    if len(ret_losses) < 2:
        return "insufficient_data", {
            "rationale": "Not enough finite return loss values."
        }

    # Patterns tried in recent history (last 6 recommendations)
    recent_patterns = [h["pattern"] for h in improvement_history[-6:]]

    # Compute trend first — needed both for divergence check and later patterns.
    # Divergence must be checked BEFORE the stagnation gate because is_stagnant()
    # measures the value range and returns False for a monotonically rising loss
    # (high range ≠ improvement). Without this pre-check, diverging runs are
    # misclassified as "improving".
    last_ratio = last.get("dt_ret_ratio", float("nan"))
    ret_trend = trend(ret_losses)

    # ── PATTERN 1: Divergence (checked before stagnation gate) ───────────────
    if ret_trend > 0.001 and recent_patterns.count("divergence") < 2:
        new_lr = max(lr * 0.3, 5e-6)
        return "divergence", {
            "rationale": (
                f"Return loss INCREASING (slope={ret_trend:.5f}/epoch). "
                f"Aggressive LR reduction: {lr:.1e} → {new_lr:.1e}."
            ),
            "flag_overrides": {"lr": new_lr},
            "remove_flags": [],
            "resume_epoch": last_epoch,
            "escalate": False,
        }

    # ── PATTERN 1b: return_head_frozen ───────────────────────────────────────
    # Return loss variance near zero for 5+ epochs while other losses are active.
    # This is the "gradient budget stolen" failure mode: obs_type or dt losses
    # are so large the return head gets <2% of gradients and stops moving.
    # Fix: hammer return_weight up to 10 AND damp obs_type_weight to 0.3.
    if len(ret_losses) >= 5:
        ret_var = coefficient_of_variation(ret_losses[-5:])
        obs_losses = _finite(
            [r.get("loss", {}).get("obs_type", float("nan")) for r in recent]
        )
        obs_mean = sum(obs_losses) / len(obs_losses) if obs_losses else 0.0
        if (
            ret_var < 0.01  # variance < 1% of mean = frozen
            and return_weight < 8.0
            and recent_patterns.count("return_head_frozen") < 3
        ):
            new_rw = min(return_weight * 5.0, 10.0)
            new_obs = 0.3
            return "return_head_frozen", {
                "rationale": (
                    f"Return loss completely frozen (CV={ret_var:.4f} over last 5 epochs, "
                    f"mean={ret_losses[-1]:.4f}). obs_type mean={obs_mean:.2f} stealing gradient budget. "
                    f"Boosting return_weight {return_weight:.1f} → {new_rw:.1f} and "
                    f"damping obs_type_weight 1.0 → {new_obs}."
                ),
                "flag_overrides": {
                    "return_weight": new_rw,
                    "obs_type_weight": new_obs,
                },
                "remove_flags": [],
                "resume_epoch": last_epoch,
                "escalate": False,
            }

    # ── Not stagnant (and not diverging)? Genuinely improving. ───────────────
    if not is_stagnant(ret_losses, window, stagnation_threshold):
        improvement_pct = (
            (ret_losses[0] - ret_losses[-1]) / (ret_losses[0] + 1e-8) * 100
        )
        return "improving", {
            "rationale": (
                f"Return loss improving {improvement_pct:.1f}% over last {len(ret_losses)} epochs."
            ),
            "resume_epoch": last_epoch,
            "flag_overrides": {},
            "escalate": False,
        }

    # ── PATTERN 2: auto_tune silencing the return head ────────────────────────
    if (
        auto_tune
        and not math.isnan(last_ratio)
        and last_ratio > 20
        and recent_patterns.count("auto_tune_suppressing") < 2
    ):
        return "auto_tune_suppressing", {
            "rationale": (
                f"auto_tune active but dt_ret_ratio={last_ratio:.1f} (>20). "
                "Uncertainty weighting may be suppressing the return head. "
                "Switching to explicit return_weight=2.0 with auto_tune disabled."
            ),
            "flag_overrides": {"return_weight": 2.0},
            "remove_flags": ["--auto-tune"],
            "resume_epoch": last_epoch,
            "escalate": False,
        }

    # ── PATTERN 3: dt_dominance (temporal loss drowning return signal) ─────────
    if (
        not math.isnan(last_ratio)
        and last_ratio > 20
        and return_weight < 4.0
        and recent_patterns.count("dt_dominance") < 3
    ):
        new_rw = min(return_weight * 2.0, 4.0)
        return "dt_dominance", {
            "rationale": (
                f"dt_loss/return_loss ratio = {last_ratio:.1f} (>20). "
                f"Temporal loss drowning return signal. "
                f"Doubling return_weight: {return_weight:.2f} → {new_rw:.2f}."
            ),
            "flag_overrides": {"return_weight": new_rw},
            "remove_flags": [],
            "resume_epoch": last_epoch,
            "escalate": False,
        }

    # ── PATTERN 4: Oscillation ────────────────────────────────────────────────
    cv = coefficient_of_variation(ret_losses)
    if cv > OSCILLATION_CV_THRESHOLD and recent_patterns.count("oscillation") < 2:
        new_lr = max(lr * 0.5, 5e-6)
        return "oscillation", {
            "rationale": (
                f"Return loss oscillating (CV={cv:.1%} > {OSCILLATION_CV_THRESHOLD:.0%}). "
                f"Halving LR: {lr:.1e} → {new_lr:.1e}."
            ),
            "flag_overrides": {"lr": new_lr},
            "remove_flags": [],
            "resume_epoch": last_epoch,
            "escalate": False,
        }

    # ── PATTERN 5: gdelt_noise ────────────────────────────────────────────────
    if gdelt_frac > 0.02 and recent_patterns.count("gdelt_noise") < 2:
        new_gdelt = max(round(gdelt_frac * 0.5, 3), 0.01)
        return "gdelt_noise", {
            "rationale": (
                f"Return loss flat despite balanced losses. "
                f"Reducing GDELT noise: {gdelt_frac:.3f} → {new_gdelt:.3f}. "
                f"GDELT is 92% of DB; subsampling isolates financial signal."
            ),
            "flag_overrides": {"gdelt_frac": new_gdelt},
            "remove_flags": [],
            "resume_epoch": last_epoch,
            "escalate": False,
        }

    # ── PATTERN 6: listnet_temperature ───────────────────────────────────────
    if listnet_temp > 0.35 and recent_patterns.count("listnet_temperature") < 2:
        new_temp = max(round(listnet_temp * 0.5, 2), 0.3)
        return "listnet_temperature", {
            "rationale": (
                f"Trying sharper ListNet target: tau {listnet_temp:.2f} → {new_temp:.2f}. "
                f"Sharper distribution strengthens IC gradient."
            ),
            "flag_overrides": {"listnet_temperature": new_temp},
            "remove_flags": [],
            "resume_epoch": last_epoch,
            "escalate": False,
        }

    # ── PATTERN 7: Structural (all fast-path options exhausted) ──────────────
    return "structural", {
        "rationale": (
            "No fast-path config change matched (or all tried). "
            "Issue may be architectural: embedding collapse, insufficient cross-domain "
            "message passing, or data sparsity. Escalating to paper research loop."
        ),
        "flag_overrides": {},
        "remove_flags": [],
        "resume_epoch": last_epoch,
        "escalate": True,
    }


# ---------------------------------------------------------------------------
# Public classifier: LLM first, heuristic fallback
# ---------------------------------------------------------------------------


def classify_pattern(
    records: list[dict],
    improvement_history: list[dict],
    window: int = STAGNATION_WINDOW,
    stagnation_threshold: float = STAGNATION_THRESHOLD,
) -> tuple[str, dict]:
    """Classify the training pattern. Tries LLM first; falls back to heuristic."""
    llm_result = _llm_classify(records, improvement_history)
    if llm_result is not None:
        return llm_result
    print("  [heuristic] No LLM available — using rule-based classification.")
    return _heuristic_classify(
        records, improvement_history, window, stagnation_threshold
    )


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_next_config(
    checkpoint_dir: Path,
    pattern: str,
    action: dict,
    run_config: dict,
) -> Path:
    """Write next_config.json — read by retrain_gnn.py --config-file."""
    out_path = checkpoint_dir / "next_config.json"
    payload = {
        "generated": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "based_on_epoch": action.get("resume_epoch", 0),
        "pattern": pattern,
        "rationale": action.get("rationale", ""),
        "resume_epoch": action.get("resume_epoch", 0),
        "flag_overrides": action.get("flag_overrides", {}),
        "remove_flags": action.get("remove_flags", []),
        "previous_config": run_config.get("config", {}),
        "example_command": (
            f"python3 scripts/retrain_gnn.py "
            f"--resume {action.get('resume_epoch', 0)} "
            f"--checkpoint-dir {checkpoint_dir} "
            f"--config-file {out_path} "
            f"--listnet --skip-eval --gdelt-frac 0.05 --max-windows 200 --epochs 10"
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


KNOWN_CAUSES = [
    "92%+ GDELT node type imbalance dominates GNN embedding space",
    "Return head receiving <2% of gradient budget (dt/ret loss ratio too high)",
    "ListNet temperature tau may need tuning for this data distribution",
    "Node embedding collapse — all instruments mapped to similar representation",
    "Learning rate too high — return loss oscillating rather than converging",
    "Cross-domain message passing depth insufficient for GDELT→instrument signal",
]


def write_trigger_file(
    records: list[dict],
    pattern: str,
    action: dict,
    knowledge_dir: Path,
) -> Path:
    """Write trigger file for auto_research.py (slow path only)."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trigger_path = knowledge_dir / f"trigger_{ts}.md"
    knowledge_dir.mkdir(parents=True, exist_ok=True)

    last = records[-1]
    epoch = last.get("epoch", 0)
    recent_ret = [r.get("loss", {}).get("return", float("nan")) for r in records[-5:]]
    losses_str = ", ".join(
        f"{x:.4f}" if not math.isnan(x) else "nan" for x in recent_ret
    )

    content = f"""---
title: "Auto-Improve Trigger: {pattern} at epoch {epoch}"
tags:
  - doc/research
  - topic/auto-research
  - topic/training
  - status/active
---

# Auto-Improve Trigger — Epoch {epoch} ({pattern})

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}

---

## Problem

{action.get("rationale", "")}

**Last 5 epochs of return loss:** {losses_str}

---

## Known Likely Causes (TirraMind-Specific)

{chr(10).join(f"- {c}" for c in KNOWN_CAUSES)}

---

## Next Step

```bash
python scripts/auto_research.py \\
  --from-trigger {trigger_path} \\
  --github-search \\
  --max-papers 5
```

Then in Copilot chat:
```
research this training issue: "GNN {pattern} at epoch {epoch}: {action.get('rationale', '')[:100]}"
```

## Related

- [[auto_ml_researcher_task]]
"""
    trigger_path.write_text(content)
    return trigger_path


# ---------------------------------------------------------------------------
# Core check
# ---------------------------------------------------------------------------


def check_once(
    checkpoint_dir: Path,
    knowledge_dir: Path,
    auto_research: bool,
    stagnation_window: int,
    stagnation_threshold: float,
    min_epochs: int,
    verbose: bool = True,
) -> tuple[str, dict]:
    """Load metrics, run decision tree, write output. Returns (pattern, action)."""
    records = load_metrics(checkpoint_dir)
    if not records:
        print(
            f"  [WARN] No metrics.jsonl in {checkpoint_dir}. "
            "Run at least 1 epoch with the updated trainer.",
            file=sys.stderr,
        )
        return "no_metrics", {}

    last_epoch = records[-1].get("epoch", 0)
    if verbose:
        print(f"  Loaded {len(records)} epoch records (latest: epoch {last_epoch})")

    if last_epoch < min_epochs:
        print(
            f"  Epoch {last_epoch} < min_epochs ({min_epochs}), skipping.",
            file=sys.stderr,
        )
        return "too_early", {}

    run_config = load_run_config(checkpoint_dir)
    improvement_history = load_improvement_history(knowledge_dir)

    pattern, action = classify_pattern(
        records, improvement_history, stagnation_window, stagnation_threshold
    )

    if verbose:
        print(f"  Pattern detected: {pattern}")
        print(f"  {action.get('rationale', '')}")

    if pattern in ("improving", "insufficient_data", "too_early", "no_metrics"):
        return pattern, action

    if action.get("escalate"):
        # ── Slow path ─────────────────────────────────────────────────────────
        print(
            f"\n[AUTO-IMPROVE] ⚠ Structural issue at epoch {last_epoch}. Escalating to research."
        )
        trigger_path = write_trigger_file(records, pattern, action, knowledge_dir)
        print(f"[AUTO-IMPROVE] Trigger written → {trigger_path}")
        append_improvement_history(
            knowledge_dir,
            {
                "ts": datetime.now().isoformat(),
                "epoch": last_epoch,
                "pattern": pattern,
                "action": "escalated_to_research",
            },
        )
        if auto_research:
            print("[AUTO-IMPROVE] Running auto_research.py...")
            script = Path(__file__).parent / "auto_research.py"
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--from-trigger",
                    str(trigger_path),
                    "--github-search",
                    "--max-papers",
                    "5",
                ],
                capture_output=False,
            )
            print("[AUTO-IMPROVE] → Then in Copilot chat: 'apply training fix'")
        return pattern, action

    # ── Fast path ─────────────────────────────────────────────────────────────
    print(f"\n[AUTO-IMPROVE] ✓ Config change: {pattern}")
    next_cfg_path = write_next_config(checkpoint_dir, pattern, action, run_config)
    print(f"[AUTO-IMPROVE] next_config.json → {next_cfg_path}")
    append_improvement_history(
        knowledge_dir,
        {
            "ts": datetime.now().isoformat(),
            "epoch": last_epoch,
            "pattern": pattern,
            "flag_overrides": action.get("flag_overrides", {}),
            "rationale": action.get("rationale", ""),
        },
    )

    resume_epoch = action.get("resume_epoch", last_epoch)
    flags = " ".join(
        f"--{k.replace('_', '-')} {v}"
        for k, v in action.get("flag_overrides", {}).items()
    )
    remove = " ".join(action.get("remove_flags", []))

    print("\n[AUTO-IMPROVE] ─── NEXT TRAINING RUN ──────────────────────────────")
    print(f"  Resume from:    epoch {resume_epoch}  (checkpoint already saved)")
    print(f"  Change:         {flags or '(see next_config.json)'}")
    if remove:
        print(f"  Remove flags:   {remove}")
    print(f"\n  Paste into Kaggle cell 10 (replace retrain command):")
    print(f"    python3 scripts/retrain_gnn.py \\")
    print(f"      --resume {resume_epoch} \\")
    print(f"      --checkpoint-dir {{CKPT_DIR}} \\")
    print(f"      --config-file {{CKPT_DIR}}/next_config.json \\")
    print(
        f"      --listnet --skip-eval --gdelt-frac 0.05 --max-windows 200 --epochs 10"
    )
    print("─────────────────────────────────────────────────────────────────────\n")
    return pattern, action


# ---------------------------------------------------------------------------
# Watch loop
# ---------------------------------------------------------------------------


def watch_loop(
    checkpoint_dir: Path,
    knowledge_dir: Path,
    auto_research: bool,
    poll_interval: int,
    stagnation_window: int,
    stagnation_threshold: float,
    min_epochs: int,
) -> None:
    """Poll metrics.jsonl for new epochs and run check_once on each new epoch."""
    last_epoch_seen = 0
    print(f"[AUTO-IMPROVE] Watching {checkpoint_dir} (poll every {poll_interval}s)")
    print(
        f"  window={stagnation_window}, threshold={stagnation_threshold:.3f}  |  Ctrl-C to stop\n"
    )
    while True:
        records = load_metrics(checkpoint_dir)
        if records:
            latest_epoch = records[-1].get("epoch", 0)
            if latest_epoch > last_epoch_seen:
                last_epoch_seen = latest_epoch
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] New epoch: {latest_epoch}"
                )
                if latest_epoch >= min_epochs:
                    check_once(
                        checkpoint_dir,
                        knowledge_dir,
                        auto_research,
                        stagnation_window,
                        stagnation_threshold,
                        min_epochs,
                    )
                    print()
        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "TirraMind auto-improve: reads metrics.jsonl, detects IC stagnation "
            "patterns, and writes next_config.json with concrete flag changes for "
            "the next training run (fast path) or escalates to the research loop (slow path)."
        )
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(".tirra_pipeline/checkpoints"),
        help="Directory with metrics.jsonl and epoch_*.pt (default: .tirra_pipeline/checkpoints)",
    )
    parser.add_argument(
        "--knowledge-dir",
        type=Path,
        default=Path("knowledge"),
        help="Directory for trigger files and history (default: knowledge/)",
    )
    parser.add_argument(
        "--no-watch",
        action="store_true",
        help="(Legacy) One-shot check then exit. This is now the default.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Poll continuously for new epochs (background mode alongside training).",
    )
    parser.add_argument(
        "--no-auto-research",
        action="store_true",
        help="On slow path: write trigger only, don't call auto_research.py.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Seconds between polls in --watch mode (default: 60).",
    )
    parser.add_argument(
        "--stagnation-window",
        type=int,
        default=STAGNATION_WINDOW,
        help=f"Epochs to evaluate for stagnation (default: {STAGNATION_WINDOW}).",
    )
    parser.add_argument(
        "--stagnation-threshold",
        type=float,
        default=STAGNATION_THRESHOLD,
        help=f"Min fractional improvement to avoid triggering (default: {STAGNATION_THRESHOLD}).",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=MIN_EPOCHS_BEFORE_CHECK,
        help=f"Skip check until this many epochs complete (default: {MIN_EPOCHS_BEFORE_CHECK}).",
    )

    args = parser.parse_args()

    if not args.checkpoint_dir.exists():
        print(
            f"ERROR: checkpoint dir not found: {args.checkpoint_dir}", file=sys.stderr
        )
        sys.exit(1)

    auto_research = not args.no_auto_research

    if args.watch:
        try:
            watch_loop(
                args.checkpoint_dir,
                args.knowledge_dir,
                auto_research,
                args.poll_interval,
                args.stagnation_window,
                args.stagnation_threshold,
                args.min_epochs,
            )
        except KeyboardInterrupt:
            print("\n[AUTO-IMPROVE] Stopped.")
        return

    # One-shot (default / --no-watch)
    pattern, action = check_once(
        args.checkpoint_dir,
        args.knowledge_dir,
        auto_research,
        args.stagnation_window,
        args.stagnation_threshold,
        args.min_epochs,
    )

    ok_patterns = {"improving", "no_metrics", "insufficient_data", "too_early"}
    if action.get("escalate"):
        sys.exit(2)
    sys.exit(0 if pattern in ok_patterns else 1)


if __name__ == "__main__":
    main()
