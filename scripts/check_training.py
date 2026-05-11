#!/usr/bin/env python3
"""
check_training.py — One-command training status from your laptop.

Usage:
    python scripts/check_training.py              # latest run, last 10 epochs
    python scripts/check_training.py --epochs 20  # last 20 epochs
    python scripts/check_training.py --local       # read local metrics.jsonl

Shows:
  - Current epoch, loss/total, loss/return, loss/dt, loss/val
  - Stagnation verdict (improving / stagnant / diverging)
  - wandb run URL
  - Kaggle session status

Requires WANDB_API_KEY in .env or environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"


def load_env() -> None:
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


# ---------------------------------------------------------------------------
# wandb GraphQL query
# ---------------------------------------------------------------------------
WANDB_GQL = "https://api.wandb.ai/graphql"

QUERY = """
query GetRuns($entityName: String!, $project: String!, $n: Int!) {
  project(entityName: $entityName, name: $project) {
    runs(first: $n, order: "-createdAt") {
      edges {
        node {
          name
          displayName
          state
          createdAt
          heartbeatAt
          summaryMetrics
        }
      }
    }
  }
}
"""


def wandb_query(api_key: str, entity: str, project: str, n: int = 5) -> list[dict]:
    payload = json.dumps(
        {
            "query": QUERY,
            "variables": {"entityName": entity, "project": project, "n": n},
        }
    ).encode()
    req = urllib.request.Request(
        WANDB_GQL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    edges = data["data"]["project"]["runs"]["edges"]
    return [e["node"] for e in edges]


# ---------------------------------------------------------------------------
# Metrics row query — get last N history rows for a run
# ---------------------------------------------------------------------------
HISTORY_QUERY = """
query GetHistory($entityName: String!, $project: String!, $runName: String!, $n: Int!) {
  project(entityName: $entityName, name: $project) {
    run(name: $runName) {
      history(samples: $n) {
        ... on RunHistoryRows { rows }
      }
    }
  }
}
"""


def wandb_history(
    api_key: str, entity: str, project: str, run_name: str, n: int = 30
) -> list[dict]:
    payload = json.dumps(
        {
            "query": HISTORY_QUERY,
            "variables": {
                "entityName": entity,
                "project": project,
                "runName": run_name,
                "n": n,
            },
        }
    ).encode()
    req = urllib.request.Request(
        WANDB_GQL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        rows = data["data"]["project"]["run"]["history"].get("rows", [])
        return [json.loads(r) if isinstance(r, str) else r for r in rows]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Local metrics.jsonl reader (fallback / --local mode)
# ---------------------------------------------------------------------------
def read_local_metrics(epochs: int) -> list[dict]:
    candidates = [
        REPO_ROOT / ".tirra_pipeline" / "checkpoints" / "h_g" / "metrics.jsonl",
        REPO_ROOT / ".tirra_pipeline" / "checkpoints" / "metrics.jsonl",
        Path(
            "/tmp/hg-output/tirramind_v1/.tirra_pipeline/checkpoints/h_g/metrics.jsonl"
        ),
        Path(
            "/tmp/hg-output2/tirramind_v1/.tirra_pipeline/checkpoints/h_g/metrics.jsonl"
        ),
    ]
    for path in candidates:
        if path.exists():
            rows = []
            for line in path.read_text().splitlines():
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            if rows:
                print(f"[local] Reading from {path}")
                return rows[-epochs:]
    return []


# ---------------------------------------------------------------------------
# Stagnation check (mirrors auto_improve.py logic)
# ---------------------------------------------------------------------------
STAGNATION_WINDOW = 5
STAGNATION_THRESHOLD = 0.005


def assess(records: list[dict]) -> str:
    losses = [_get_loss(r, "return") for r in records]
    losses = [x for x in losses if not (x != x)]  # drop NaN
    if len(losses) < 3:
        return "insufficient data"
    recent = losses[-STAGNATION_WINDOW:]
    if len(recent) < 2:
        return "insufficient data"
    improvement = (recent[0] - recent[-1]) / (abs(recent[0]) + 1e-9)
    if losses[-1] > losses[0] * 1.05:
        return "⚠️  DIVERGING — consider stopping"
    if improvement < STAGNATION_THRESHOLD:
        return "⚠️  STAGNANT  — improvement <0.5% over last 5 epochs"
    return f"✅  IMPROVING — {improvement * 100:.1f}% improvement over last {len(recent)} epochs"


# ---------------------------------------------------------------------------
# Kaggle status
# ---------------------------------------------------------------------------
def kaggle_status() -> str:
    try:
        import subprocess

        r = subprocess.run(
            ["kaggle", "kernels", "status", "deeperisbetter/tirramind-h-g"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return r.stdout.strip() or r.stderr.strip()
    except Exception as e:
        return f"(kaggle CLI unavailable: {e})"


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def _get_loss(r: dict, key: str) -> float:
    """Read loss value from either wandb flat format or local nested format."""
    # wandb: "loss/return" or "loss_return"
    v = r.get(f"loss/{key}", r.get(f"loss_{key}"))
    if v is not None:
        return float(v)
    # local metrics.jsonl: {"loss": {"return": ...}}
    loss_dict = r.get("loss", {})
    v2 = loss_dict.get(key)
    if v2 is not None:
        return float(v2)
    return float("nan")


def render_table(records: list[dict]) -> None:
    header = f"{'Epoch':>6}  {'total':>10}  {'return':>10}  {'dt':>8}  {'obs':>8}  {'val':>8}  {'warnings'}"
    print(header)
    print("-" * (len(header) + 30))
    for r in records:
        epoch = r.get("epoch", r.get("_step", "?"))
        lt = _get_loss(r, "total")
        lr_ = _get_loss(r, "return")
        ldt = _get_loss(r, "dt")
        lobs = _get_loss(r, "obs_type")
        lv = _get_loss(r, "value")
        warns = r.get("warnings", [])
        warn_str = " | ".join(w.split(":")[0] for w in warns) if warns else ""
        print(
            f"{epoch!s:>6}  {lt:>10.4f}  {lr_:>10.4f}  {ldt:>8.4f}  {lobs:>8.4f}  {lv:>8.4f}  {warn_str}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Check live training status from CLI.")
    p.add_argument(
        "--epochs", type=int, default=10, help="Number of recent epochs to show."
    )
    p.add_argument(
        "--local", action="store_true", help="Read local metrics.jsonl only (no wandb)."
    )
    p.add_argument("--entity", default="999-sbpatel", help="wandb entity.")
    p.add_argument("--project", default="tirramind", help="wandb project.")
    args = p.parse_args()

    load_env()

    print("=" * 60)
    print("  TirraMind Training Status")
    print("=" * 60)

    # ── Kaggle status ──────────────────────────────────────────────
    print(f"\nKaggle:  {kaggle_status()}")

    # ── Metrics ────────────────────────────────────────────────────
    records: list[dict] = []

    if not args.local:
        api_key = os.environ.get("WANDB_API_KEY") or os.environ.get(
            "TIRRA_WANDB_API_KEY"
        )
        if not api_key:
            print(
                "\n[WARN] WANDB_API_KEY not set — falling back to local metrics.jsonl"
            )
        else:
            try:
                runs = wandb_query(api_key, args.entity, args.project, n=5)
                if not runs:
                    print("\n[INFO] No wandb runs found for project 'tirramind'.")
                else:
                    latest = runs[0]
                    state = latest.get("state", "?")
                    name = latest.get("displayName") or latest.get("name", "?")
                    heartbeat = latest.get("heartbeatAt", "?")
                    steps = (latest.get("historySummary") or {}).get("totalSteps", "?")
                    run_url = f"https://wandb.ai/{args.entity}/{args.project}/runs/{latest['name']}"
                    print(f"\nwandb run:  {name}  [{state}]")
                    print(f"Last heartbeat: {heartbeat}   Total steps logged: {steps}")
                    print(f"URL: {run_url}")

                    # Fetch history rows
                    history = wandb_history(
                        api_key,
                        args.entity,
                        args.project,
                        latest["name"],
                        n=args.epochs,
                    )
                    if history:
                        records = history
                    else:
                        # Fall back to summary metrics if history endpoint fails
                        summary_raw = latest.get("summaryMetrics", "{}")
                        summary = (
                            json.loads(summary_raw)
                            if isinstance(summary_raw, str)
                            else summary_raw
                        )
                        if summary:
                            print(
                                "\n[INFO] History rows unavailable — showing summary metrics:"
                            )
                            for k, v in summary.items():
                                if not k.startswith("_") and isinstance(
                                    v, (int, float)
                                ):
                                    print(f"  {k}: {v:.4f}")
            except Exception as e:
                print(f"\n[WARN] wandb query failed: {e} — falling back to local")

    if not records:
        records = read_local_metrics(args.epochs)

    if records:
        print(f"\nLast {min(args.epochs, len(records))} epoch(s):\n")
        render_table(records[-args.epochs :])
        print(f"\nVerdict: {assess(records)}")
    else:
        print(
            "\n[INFO] No metrics available yet — session may still be in first block."
        )
        print(
            "       Check Kaggle UI for live cell output, or wait for block 1 to complete."
        )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
