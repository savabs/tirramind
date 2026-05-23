"""
wandb_monitor.py — pull live training state from wandb and print a structured
diagnostic table that can be read by the agent during any conversation.

Usage:
    python scripts/wandb_monitor.py                        # latest run
    python scripts/wandb_monitor.py --run h-a-ep31-40      # specific run
    python scripts/wandb_monitor.py --last 5               # last 5 epochs only
    python scripts/wandb_monitor.py --project tirramind    # override project

Requirements:
    export WANDB_API_KEY=<key>      # or set in .env at project root
    export WANDB_ENTITY=<username>  # your wandb username / org
"""

import argparse
import os
import sys
from pathlib import Path

# ── Load .env if present ─────────────────────────────────────────────────────
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

# ── Validate env ─────────────────────────────────────────────────────────────
_api_key = os.environ.get("WANDB_API_KEY")
if not _api_key:
    print(
        "ERROR: WANDB_API_KEY not set. Export it or add to .env at project root.",
        file=sys.stderr,
    )
    sys.exit(1)

import wandb  # noqa: E402  (import after key check so error is clear)

DEAD_THRESHOLD = 1e-4  # return loss change smaller than this = dead head
EXPLODE_RATIO = 10.0  # loss increases by this ratio = explosion


def _flag_return_head(history: list[dict]) -> str:
    """Return a human-readable status for the return head."""
    returns = [
        r.get("loss/return") for r in history if r.get("loss/return") is not None
    ]
    if not returns:
        return "ABSENT (never logged)"
    if len(returns) < 2:
        return f"only {len(returns)} epoch — too early"
    delta = abs(returns[-1] - returns[-2])
    if returns[-1] != returns[-1]:  # NaN
        return "NaN — likely exploded"
    if delta < DEAD_THRESHOLD:
        return f"DEAD (delta={delta:.2e}, last={returns[-1]:.4f})"
    # Check for explosion: compare last to first
    if len(returns) >= 5 and returns[-1] > returns[0] * EXPLODE_RATIO:
        return f"EXPLODING ({returns[0]:.4f} → {returns[-1]:.4f})"
    return f"ok  (last={returns[-1]:.4f}, Δ={delta:.2e})"


def _flag_time_delta(history: list[dict]) -> str:
    """Warn if time_delta loss looks unnormalized (very large absolute value)."""
    tds = [
        r.get("loss/time_delta")
        for r in history
        if r.get("loss/time_delta") is not None
    ]
    if not tds:
        return "absent"
    mean_td = sum(tds[-5:]) / len(tds[-5:])
    if mean_td > 100:
        return f"LARGE ({mean_td:.1f}) — likely raw Unix timestamps, not normalized"
    return f"ok  (mean last-5={mean_td:.4f})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Query live wandb training state")
    parser.add_argument("--project", default="tirramind", help="wandb project name")
    parser.add_argument(
        "--entity",
        default=os.environ.get("WANDB_ENTITY", ""),
        help="wandb entity (username/org)",
    )
    parser.add_argument("--run", default=None, help="specific run name or ID")
    parser.add_argument(
        "--last", type=int, default=10, help="show last N epochs per run"
    )
    parser.add_argument(
        "--all-runs", action="store_true", help="summarise all runs in project"
    )
    args = parser.parse_args()

    api = wandb.Api(api_key=_api_key)

    project_path = f"{args.entity}/{args.project}" if args.entity else args.project

    # ── Fetch runs ────────────────────────────────────────────────────────────
    try:
        if args.run:
            # Try exact name match first
            runs = [
                r
                for r in api.runs(project_path)
                if r.name == args.run or r.id == args.run
            ]
            if not runs:
                print(
                    f"No run found with name/id '{args.run}' in {project_path}",
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            all_runs = list(api.runs(project_path, order="-created_at"))
            if not all_runs:
                print(f"No runs found in project '{project_path}'", file=sys.stderr)
                sys.exit(1)
            runs = all_runs if args.all_runs else [all_runs[0]]
    except Exception as exc:
        print(f"ERROR fetching runs from '{project_path}': {exc}", file=sys.stderr)
        sys.exit(1)

    # ── Print header ──────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  wandb project : {project_path}")
    print(f"  runs shown    : {len(runs)}")
    print(f"{'='*80}\n")

    for run in runs:
        print(f"┌─ Run: {run.name}  [{run.id}]  state={run.state}")
        print(f"│  Tags : {', '.join(run.tags) if run.tags else '—'}")
        print(f"│  URL  : {run.url}")

        # Fetch metric history
        try:
            hist_df = run.history(samples=500, pandas=True)
        except Exception as exc:
            print(f"│  ERROR reading history: {exc}\n")
            continue

        if hist_df.empty:
            print("│  No metrics logged yet.\n")
            continue

        # Convert to list of dicts for easy slicing
        history = hist_df.to_dict("records")
        last_n = history[-args.last :]

        # ── Epoch loss table ──────────────────────────────────────────────────
        col_w = 10
        headers = [
            "epoch",
            "total",
            "obs_type",
            "time_delta",
            "contrastive",
            "value",
            "return",
        ]
        print("│")
        print("│  " + "  ".join(h.rjust(col_w) for h in headers))
        print("│  " + "  ".join("-" * col_w for _ in headers))
        for row in last_n:
            epoch = row.get("epoch", row.get("_step", "?"))
            vals = [epoch] + [row.get(f"loss/{h}", float("nan")) for h in headers[1:]]

            def _fmt(v):
                if isinstance(v, float) and v != v:  # NaN
                    return "—".rjust(col_w)
                return (
                    f"{v:.4f}".rjust(col_w)
                    if isinstance(v, float)
                    else str(v).rjust(col_w)
                )

            print("│  " + "  ".join(_fmt(v) for v in vals))

        # ── Effective weights (auto-tune) ─────────────────────────────────────
        weight_cols = [c for c in hist_df.columns if c.startswith("weight/")]
        if weight_cols:
            last_weights = history[-1]
            print("│")
            print("│  Effective uncertainty weights (last epoch):")
            for wc in sorted(weight_cols):
                print(f"│    {wc:<35} {last_weights.get(wc, float('nan')):.4f}")

        # ── Diagnostics ───────────────────────────────────────────────────────
        print("│")
        print("│  DIAGNOSTICS:")
        print(f"│    return head  : {_flag_return_head(history)}")
        print(f"│    time_delta   : {_flag_time_delta(history)}")

        # Total loss trend (last 5 vs first 5 of window)
        totals = [
            r.get("loss/total") for r in history if r.get("loss/total") is not None
        ]
        if len(totals) >= 10:
            first5 = sum(totals[:5]) / 5
            last5 = sum(totals[-5:]) / 5
            pct = (last5 - first5) / (first5 + 1e-9) * 100
            trend = (
                f"{'↓' if pct < 0 else '↑'} {abs(pct):.1f}% over {len(totals)} epochs"
            )
            print(f"│    total loss   : {trend}")

        print(f"└{'─'*78}\n")


if __name__ == "__main__":
    main()
