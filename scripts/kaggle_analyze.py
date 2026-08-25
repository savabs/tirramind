#!/usr/bin/env python3
"""kaggle_analyze.py — Parse Kaggle Phase50 logs and decide the next training config.

Used by kaggle_loop.py for autonomous watch → analyze → patch → push cycles.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
KERNEL_SLUG = "deeperisbetter/tirramind-phase50"
LOG_DIR = ROOT / ".tirra_pipeline"
LOOP_HISTORY = LOG_DIR / "kaggle_loop_history.jsonl"

_EPOCH_RE = re.compile(
    r"Epoch (\d+)/(\d+) — loss: [\d.]+ "
    r"\([^)]*return: ([\d.]+)\)"
)
_IC_RE = re.compile(r"GNN-PurgedRanker\s+-?([\d.]+)")
_GRAD_FLOW_RE = re.compile(
    r"\[GRAD_FLOW\] ep(\d+) head=(\S+) pred_std=([\d.]+) tgt_std=([\d.]+)"
)
_COLLAPSE_RE = re.compile(r"Collapse frac:\s+([\d.]+)%")
_LISTNET_FLOOR = 200.0
_PLATEAU_EPS = 0.05


@dataclass
class RunDiagnosis:
    kernel_version: int
    fingerprint: str | None
    epochs: list[dict] = field(default_factory=list)
    grad_flow: list[dict] = field(default_factory=list)
    ic_purged_ranker: float | None = None
    ic_emb_norm: float | None = None
    collapse_pct: float | None = None
    return_ep3: float | None = None
    return_ep10: float | None = None
    return_plateau_from_epoch: int | None = None
    gate_verdict: str = "unknown"  # pass | fail | marginal
    pred_std_ep10: float | None = None
    active_head: str | None = None
    listnet_floor: bool = False
    pattern: str = "unknown"
    rationale: str = ""
    structural_halt: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_json_log_lines(raw: str) -> str:
    lines: list[str] = []
    for part in raw.split("\n"):
        part = part.strip()
        if not part or part in ("[", "]"):
            continue
        if part.startswith(","):
            part = part[1:]
        try:
            obj = json.loads(part)
            if isinstance(obj, dict) and "data" in obj:
                lines.append(str(obj["data"]).rstrip("\n"))
        except json.JSONDecodeError:
            lines.append(part)
    return "\n".join(lines) if lines else raw


def fetch_logs(slug: str = KERNEL_SLUG, *, save_path: Path | None = None) -> str:
    result = subprocess.run(
        ["kaggle", "kernels", "logs", slug],
        capture_output=True,
        text=True,
        timeout=120,
    )
    raw = result.stdout or result.stderr or ""
    text = _parse_json_log_lines(raw)
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(raw)
    return text


def parse_epoch_summaries(log_text: str) -> list[dict]:
    epochs: dict[int, dict] = {}
    for m in _EPOCH_RE.finditer(log_text):
        ep, total, ret = int(m.group(1)), int(m.group(2)), float(m.group(3))
        epochs[ep] = {"epoch": ep, "total_epochs": total, "return_loss": ret}
    return [epochs[k] for k in sorted(epochs)]


def parse_grad_flow(log_text: str) -> list[dict]:
    out = []
    for m in _GRAD_FLOW_RE.finditer(log_text):
        out.append(
            {
                "epoch": int(m.group(1)),
                "active_head": m.group(2),
                "return_pred_std": float(m.group(3)),
                "return_tgt_std": float(m.group(4)),
            }
        )
    return out


def parse_ic(log_text: str) -> dict[str, float | None]:
    ic: dict[str, float | None] = {
        "purged_ranker": None,
        "concat_return_head": None,
        "emb_norm": None,
    }
    for line in log_text.splitlines():
        if "GNN-PurgedRanker" in line and "Mean IC" not in line:
            m = re.search(r"GNN-PurgedRanker\s+(-?[\d.]+)", line)
            if m:
                ic["purged_ranker"] = float(m.group(1))
        if "GNN-ConcatReturnHead" in line and "Mean IC" not in line:
            m = re.search(r"GNN-ConcatReturnHead\s+(-?[\d.]+)", line)
            if m:
                ic["concat_return_head"] = float(m.group(1))
        if "GNN-EmbNorm" in line and "Mean IC" not in line:
            m = re.search(r"GNN-EmbNorm\s+(-?[\d.]+)", line)
            if m:
                ic["emb_norm"] = float(m.group(1))
    return ic


def parse_collapse(log_text: str) -> float | None:
    matches = _COLLAPSE_RE.findall(log_text)
    return float(matches[-1]) if matches else None


def _detect_plateau(by_ep: dict[int, float]) -> int | None:
    """First epoch (1-based) after which return loss never moves again."""
    sorted_eps = sorted(by_ep)
    if len(sorted_eps) < 3:
        return None
    for ep in sorted_eps:
        if ep < 3:
            continue
        tail = [by_ep[e] for e in sorted_eps if e >= ep]
        if len(tail) >= 3 and max(tail) - min(tail) < _PLATEAU_EPS:
            return ep
    return None


def evaluate_gate(epochs: list[dict]) -> tuple[str, float | None, float | None, int | None]:
    if not epochs:
        return "unknown", None, None, None
    by_ep = {e["epoch"]: e["return_loss"] for e in epochs}
    ep3 = by_ep.get(3)
    max_ep = max(by_ep)
    ep10 = by_ep.get(10) or by_ep.get(max_ep)
    plateau_from = _detect_plateau(by_ep)

    if ep3 is None or ep10 is None:
        return "unknown", ep3, ep10, plateau_from

    # ListNet entropy floor: all scores identical → loss frozen ~209+
    if ep3 > _LISTNET_FLOOR and abs(ep3 - ep10) < _PLATEAU_EPS:
        return "fail", ep3, ep10, plateau_from or 3

    # V52 PASS signature: return still moving at final epoch (105→116)
    v52_pass = ep10 > ep3 + 0.5 and not (plateau_from and plateau_from <= max_ep - 1)

    if plateau_from and plateau_from <= 7:
        return "fail", ep3, ep10, plateau_from

    if abs(ep3 - ep10) < _PLATEAU_EPS:
        return "fail", ep3, ep10, plateau_from

    if plateau_from and plateau_from <= max_ep - 1:
        return "marginal", ep3, ep10, plateau_from

    if v52_pass:
        return "pass", ep3, ep10, plateau_from

    return "marginal", ep3, ep10, plateau_from


def load_loop_history() -> list[dict]:
    if not LOOP_HISTORY.exists():
        return []
    rows = []
    for line in LOOP_HISTORY.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def append_loop_history(entry: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry.setdefault("ts", datetime.now(timezone.utc).isoformat())
    with LOOP_HISTORY.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")


def classify_pattern(
    diagnosis: RunDiagnosis,
    history: list[dict],
    current_cfg: dict,
) -> tuple[str, str, bool]:
    """Return (pattern, rationale, structural_halt)."""
    recent_patterns = [h.get("pattern") for h in history[-6:]]
    ic = diagnosis.ic_purged_ranker
    gate = diagnosis.gate_verdict

    if gate == "pass" and ic is not None and ic > 0.03:
        return (
            "gate_pass_ready_90ep",
            f"ep3/10 gate PASS (return {diagnosis.return_ep3:.2f}→{diagnosis.return_ep10:.2f}), "
            f"IC={ic:+.4f} > 0.03. Ready for 90ep full run.",
            False,
        )

    if diagnosis.listnet_floor or (
        diagnosis.return_ep3 and diagnosis.return_ep3 > _LISTNET_FLOOR
    ):
        if recent_patterns.count("listnet_floor") >= 2:
            return (
                "listnet_floor",
                "ListNet entropy floor persists after 2 fix attempts. Structural halt.",
                True,
            )
        return (
            "listnet_floor",
            f"Return loss stuck at ListNet floor (~{diagnosis.return_ep3:.1f}). "
            "Scores likely uniform — boost return_weight, damp obs_type.",
            False,
        )

    pred_std = diagnosis.pred_std_ep10
    if pred_std is not None and pred_std < 0.01 and ic is not None and ic < 0:
        if (
            current_cfg.get("use_concat_head")
            and recent_patterns.count("concat_ablation") < 1
        ):
            return (
                "concat_ablation",
                f"pred_std={pred_std:.4f} at ep10 with IC={ic:+.4f}. "
                "Ablation: embedding-only return head (disable concat).",
                False,
            )

    if diagnosis.collapse_pct and diagnosis.collapse_pct > 25:
        if recent_patterns.count("vicreg_antcollapse") < 2 and (
            current_cfg.get("vicreg_weight") or 0
        ) == 0:
            return (
                "vicreg_antcollapse",
                f"Collapse {diagnosis.collapse_pct:.1f}% with IC={ic or 0:+.4f}. "
                "Enable VICReg 0.25.",
                False,
            )

    if gate in ("fail", "marginal") and ic is not None and ic < -0.05:
        if recent_patterns.count("return_head_frozen") < 3:
            return (
                "return_head_frozen",
                f"Gate {gate}, plateau@ep{diagnosis.return_plateau_from_epoch}, "
                f"IC={ic:+.4f}. Boost return_weight, damp obs_type.",
                False,
            )

    if ic is not None and ic < -0.01 and recent_patterns.count("ic_degrading") < 2:
        return (
            "ic_degrading",
            f"IC={ic:+.4f} negative. Halve gdelt_frac to reduce GDELT noise.",
            False,
        )

    if recent_patterns.count(diagnosis.pattern) >= 3:
        return (
            "exhausted",
            f"Pattern {diagnosis.pattern} tried 3+ times without improvement.",
            True,
        )

    return (
        "no_action",
        "No automated fix matched. Review logs manually.",
        True,
    )


def decide_next_config(
    current_cfg: dict,
    diagnosis: RunDiagnosis,
    *,
    new_version: int,
) -> dict | None:
    """Return updated _NOTEBOOK_CONFIG dict, or None if loop should halt."""
    history = load_loop_history()
    pattern, rationale, halt = classify_pattern(diagnosis, history, current_cfg)
    diagnosis.pattern = pattern
    diagnosis.rationale = rationale
    diagnosis.structural_halt = halt

    if diagnosis.gate_verdict == "unknown" and not diagnosis.epochs:
        diagnosis.pattern = "incomplete_logs"
        diagnosis.rationale = "Run not finished or logs missing epoch summaries."
        diagnosis.structural_halt = True
        return None

    if halt and pattern in ("exhausted", "no_action", "listnet_floor"):
        return None

    next_cfg = dict(current_cfg)
    next_cfg["kernel_version"] = new_version
    next_cfg["resume_epoch"] = 0
    next_cfg["epochs"] = 10
    next_cfg["run_full_backtest"] = False
    next_cfg["eval_smoke"] = True

    if pattern == "gate_pass_ready_90ep":
        next_cfg["epochs"] = 90
        next_cfg["fix"] = f"v52_full_gate_pass_v{new_version}"
        next_cfg["run_full_backtest"] = True
        return next_cfg

    if pattern == "listnet_floor":
        next_cfg["fix"] = f"listnet_floor_fix_v{new_version}"
        next_cfg["return_weight"] = min(
            float(current_cfg.get("return_weight", 3.0)) * 2.0, 10.0
        )
        next_cfg["obs_type_weight"] = 0.3
        next_cfg["auto_tune"] = False
        return next_cfg

    if pattern == "concat_ablation":
        next_cfg["fix"] = f"emb_only_ablation_v{new_version}"
        next_cfg["use_concat_head"] = False
        next_cfg["embedding_only_return"] = True
        return next_cfg

    if pattern == "vicreg_antcollapse":
        next_cfg["fix"] = f"vicreg_antcollapse_v{new_version}"
        next_cfg["vicreg_weight"] = 0.25
        return next_cfg

    if pattern == "return_head_frozen":
        next_cfg["fix"] = f"return_boost_v{new_version}"
        next_cfg["return_weight"] = min(
            float(current_cfg.get("return_weight", 3.0)) * 2.0, 10.0
        )
        next_cfg["obs_type_weight"] = 0.3
        return next_cfg

    if pattern == "ic_degrading":
        next_cfg["fix"] = f"gdelt_halve_v{new_version}"
        next_cfg["gdelt_frac"] = max(
            round(float(current_cfg.get("gdelt_frac", 0.05)) * 0.5, 3), 0.01
        )
        return next_cfg

    return None


def diagnose_log_text(
    log_text: str,
    *,
    kernel_version: int = 0,
    fingerprint: str | None = None,
) -> RunDiagnosis:
    epochs = parse_epoch_summaries(log_text)
    grad_flow = parse_grad_flow(log_text)
    ic = parse_ic(log_text)
    collapse = parse_collapse(log_text)
    gate, ep3, ep10, plateau = evaluate_gate(epochs)

    gf10 = next((g for g in grad_flow if g["epoch"] == 10), None)
    if not gf10 and grad_flow:
        gf10 = grad_flow[-1]

    listnet_floor = bool(
        ep3 and ep3 > _LISTNET_FLOOR and ep10 and abs(ep3 - ep10) < _PLATEAU_EPS
    )

    diag = RunDiagnosis(
        kernel_version=kernel_version,
        fingerprint=fingerprint,
        epochs=epochs,
        grad_flow=grad_flow,
        ic_purged_ranker=ic.get("purged_ranker"),
        ic_emb_norm=ic.get("emb_norm"),
        collapse_pct=collapse,
        return_ep3=ep3,
        return_ep10=ep10,
        return_plateau_from_epoch=plateau,
        gate_verdict=gate,
        pred_std_ep10=gf10.get("return_pred_std") if gf10 else None,
        active_head=gf10.get("active_head") if gf10 else None,
        listnet_floor=listnet_floor,
    )
    return diag


def diagnose_from_file(log_path: Path, **kwargs: Any) -> RunDiagnosis:
    text = _parse_json_log_lines(log_path.read_text())
    return diagnose_log_text(text, **kwargs)


def print_report(diag: RunDiagnosis) -> None:
    print(f"\n── Kaggle Run Diagnosis (V{diag.kernel_version}) ──")
    if diag.fingerprint:
        print(f"  Fingerprint : {diag.fingerprint}")
    print(f"  Gate        : {diag.gate_verdict.upper()}")
    if diag.return_ep3 is not None:
        print(
            f"  Return loss : ep3={diag.return_ep3:.4f}  ep10={diag.return_ep10:.4f}"
            + (
                f"  plateau@ep{diag.return_plateau_from_epoch}"
                if diag.return_plateau_from_epoch
                else ""
            )
        )
    if diag.ic_purged_ranker is not None:
        print(f"  IC Purged   : {diag.ic_purged_ranker:+.4f}")
    if diag.collapse_pct is not None:
        print(f"  Collapse    : {diag.collapse_pct:.1f}%")
    if diag.pred_std_ep10 is not None:
        print(
            f"  pred_std    : {diag.pred_std_ep10:.4f}  (head={diag.active_head})"
        )
    print(f"  Pattern     : {diag.pattern}")
    print(f"  Rationale   : {diag.rationale}")
    if diag.gate_verdict == "unknown" and not diag.epochs:
        print("  ⚠ Incomplete logs — run may still be in progress")
    if diag.structural_halt:
        print("  ⚠ STRUCTURAL HALT — manual review required")
    print()


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Analyze Kaggle Phase50 training logs")
    p.add_argument("--log-file", type=Path, help="Local log file (default: fetch live)")
    p.add_argument("--kernel-version", type=int, default=0)
    p.add_argument("--fingerprint", default=None)
    p.add_argument("--save", type=Path, help="Save raw logs to this path")
    p.add_argument("--json", action="store_true", help="Print diagnosis as JSON")
    args = p.parse_args()

    if args.log_file:
        diag = diagnose_from_file(
            args.log_file,
            kernel_version=args.kernel_version,
            fingerprint=args.fingerprint,
        )
    else:
        save = args.save or LOG_DIR / f"kaggle_logs_v{args.kernel_version or 'latest'}.txt"
        text = fetch_logs(save_path=save)
        diag = diagnose_log_text(
            text,
            kernel_version=args.kernel_version,
            fingerprint=args.fingerprint,
        )

    if args.json:
        print(json.dumps(diag.to_dict(), indent=2))
    else:
        print_report(diag)


if __name__ == "__main__":
    main()
