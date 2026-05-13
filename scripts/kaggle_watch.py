#!/usr/bin/env python3
"""
kaggle_watch.py — Live training dashboard. Runs in a terminal and auto-refreshes.

Shows: Kaggle kernel status + current W&B epoch/loss table in one screen.
No copy-pasting, no manual commands.

Usage:
    python scripts/kaggle_watch.py                  # refresh every 60s
    python scripts/kaggle_watch.py --interval 30    # refresh every 30s
    python scripts/kaggle_watch.py --epochs 12      # show last 12 epochs
    python scripts/kaggle_watch.py --once           # print once and exit

Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_env() -> None:
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


# ---------------------------------------------------------------------------
# Kaggle kernel status
# ---------------------------------------------------------------------------
def _kaggle_status(kernel: str) -> dict:
    env = os.environ.copy()
    token = env.get("KAGGLE_API_TOKEN", "")
    if token:
        env["KAGGLE_API_TOKEN"] = token
    try:
        r = subprocess.run(
            ["kaggle", "kernels", "status", kernel],
            capture_output=True,
            text=True,
            timeout=20,
            env=env,
        )
        raw = (r.stdout + r.stderr).strip()
        raw_l = raw.lower()
        if "running" in raw_l:
            status = "running"
        elif "complete" in raw_l:
            status = "complete"
        elif "cancel" in raw_l:
            status = "cancelled"
        elif "error" in raw_l or "fail" in raw_l:
            status = "error"
        elif "queue" in raw_l:
            status = "queued"
        else:
            status = "unknown"
        return {
            "status": status,
            "raw": raw,
            "url": f"https://www.kaggle.com/code/{kernel}",
        }
    except FileNotFoundError:
        return {"status": "kaggle CLI not found", "raw": "", "url": ""}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "raw": "kaggle CLI timed out", "url": ""}
    except Exception as e:
        return {"status": "error", "raw": str(e), "url": ""}


# ---------------------------------------------------------------------------
# W&B GraphQL (no wandb package needed — uses urllib only)
# ---------------------------------------------------------------------------
_GQL_URL = "https://api.wandb.ai/graphql"

_RUNS_Q = """
query GetRuns($entity: String!, $project: String!, $n: Int!) {
  project(entityName: $entity, name: $project) {
    runs(first: $n, order: "-createdAt") {
      edges {
        node { name displayName state createdAt heartbeatAt summaryMetrics }
      }
    }
  }
}
"""

_HIST_Q = """
query GetHistory($entity: String!, $project: String!, $run: String!, $n: Int!) {
  project(entityName: $entity, name: $project) {
    run(name: $run) {
      history(samples: $n) {
        ... on RunHistoryRows { rows }
      }
    }
  }
}
"""


def _gql(api_key: str, query: str, variables: dict) -> dict:
    payload = json.dumps({"query": query, "variables": variables}).encode()
    req = urllib.request.Request(
        _GQL_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read())


def _latest_run(api_key: str, entity: str, project: str) -> dict | None:
    try:
        data = _gql(api_key, _RUNS_Q, {"entity": entity, "project": project, "n": 1})
        edges = data["data"]["project"]["runs"]["edges"]
        return edges[0]["node"] if edges else None
    except Exception as e:
        return {"_error": str(e)}


def _run_history(
    api_key: str, entity: str, project: str, run_name: str, n: int
) -> list[dict]:
    try:
        data = _gql(
            api_key,
            _HIST_Q,
            {"entity": entity, "project": project, "run": run_name, "n": n},
        )
        rows = data["data"]["project"]["run"]["history"].get("rows", [])
        return [json.loads(r) if isinstance(r, str) else r for r in rows]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------
def _v(row: dict, *keys: str):
    """Look up a value by multiple key variants (flat and loss/x nested)."""
    for k in keys:
        if k in row:
            return row[k]
        # wandb flat: "loss/return" stored as-is
        # local jsonl nested: {"loss": {"return": ...}}
        parts = k.split("/")
        if len(parts) == 2:
            sub = row.get(parts[0])
            if isinstance(sub, dict) and parts[1] in sub:
                return sub[parts[1]]
    return None


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "      —"
    return f"{float(v):>9.4f}"


# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------
_B = "\033[1m"
_DIM = "\033[2m"
_RST = "\033[0m"
_GRN = "\033[92m"
_YLW = "\033[93m"
_RED = "\033[91m"
_CYN = "\033[96m"
_MGN = "\033[95m"

_STATUS_CLR = {
    "running": _GRN,
    "complete": _CYN,
    "error": _RED,
    "queued": _YLW,
    "cancelled": _YLW,
    "unknown": _DIM,
    "timeout": _RED,
    "crashed": _RED,
}


def _clr(text: str, color: str) -> str:
    return f"{color}{text}{_RST}"


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def _render(
    *,
    kernel: str,
    kst: dict,
    run: dict | None,
    history: list[dict],
    entity: str,
    project: str,
    refresh_at: float,
) -> str:
    W = 64
    lines: list[str] = []
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    lines.append(f"{_B}{'─' * W}{_RST}")
    lines.append(f"{_B}  TirraMind Live Dashboard{_RST}   {_DIM}{ts}{_RST}")
    lines.append(f"{_B}{'─' * W}{_RST}")

    # ── Kaggle ──────────────────────────────────────────────────────────────
    st = kst.get("status", "unknown")
    sc = _STATUS_CLR.get(st.lower(), _DIM)
    lines.append(f"\n{_B}Kaggle Kernel:{_RST}")
    lines.append(f"  Slug:    {_DIM}{kernel}{_RST}")
    lines.append(f"  Status:  {_clr(st.upper(), sc)}")
    lines.append(f"  URL:     {_DIM}{kst.get('url', '')}{_RST}")
    raw = kst.get("raw", "")
    if raw:
        lines.append(f"  Info:    {_DIM}{raw[:100]}{_RST}")

    # ── W&B ─────────────────────────────────────────────────────────────────
    lines.append(f"\n{_B}W&B Run:{_RST}")
    if run is None:
        lines.append("  (no runs found in project)")
    elif "_error" in run:
        lines.append(f"  {_clr('ERROR: ' + run['_error'], _RED)}")
    else:
        run_state = run.get("state", "?")
        rsc = _STATUS_CLR.get(run_state.lower(), _DIM)
        name = run.get("displayName") or run.get("name", "?")
        heartbeat = run.get("heartbeatAt", "?")
        run_url = f"https://wandb.ai/{entity}/{project}/runs/{run.get('name', '')}"

        lines.append(f"  Name:    {_B}{name}{_RST}")
        lines.append(f"  State:   {_clr(run_state, rsc)}")
        lines.append(f"  Pulse:   {_DIM}{heartbeat}{_RST}")
        lines.append(f"  URL:     {_DIM}{run_url}{_RST}")

        # Latest summary epoch
        try:
            summary = json.loads(run.get("summaryMetrics") or "{}")
        except Exception:
            summary = {}
        if summary:
            ep = summary.get("epoch", summary.get("_step", "?"))
            lt = summary.get("loss/total", summary.get("loss_total"))
            lr_ = summary.get("loss/return", summary.get("loss_return"))
            lines.append(
                f"\n  {_B}Latest:{_RST}  epoch {_clr(str(ep), _MGN)}"
                f"  total {_fmt(lt)}  return {_fmt(lr_)}"
            )

        # Epoch table
        if history:
            lines.append(
                f"\n  {_B}{'Ep':>4}  {'total':>9}  {'return':>9}  {'dt':>8}  {'obs':>8}  {'val':>8}{_RST}"
            )
            lines.append(f"  {'─' * 54}")
            for row in history:
                ep = row.get("epoch", row.get("_step", "?"))
                lt_ = _v(row, "loss/total", "loss_total")
                lr_ = _v(row, "loss/return", "loss_return")
                ldt = _v(row, "loss/time_delta", "loss_dt", "loss/dt")
                lob = _v(row, "loss/obs_type", "loss_obs_type")
                lv_ = _v(row, "loss/value", "loss_value")
                lines.append(
                    f"  {str(ep):>4}  {_fmt(lt_)}  {_fmt(lr_)}  {_fmt(ldt)}  {_fmt(lob)}  {_fmt(lv_)}"
                )

            # Trend
            rets = [_v(r, "loss/return", "loss_return") for r in history]
            rets = [x for x in rets if x is not None and x == x]
            if len(rets) >= 3:
                delta = rets[-1] - rets[0]
                if delta < -0.001:
                    verdict = _clr(
                        f"▼ IMPROVING  ({delta:+.4f} over {len(rets)} epochs)", _GRN
                    )
                elif delta > 0.001:
                    verdict = _clr(f"▲ DIVERGING  ({delta:+.4f})", _RED)
                else:
                    verdict = _clr(f"≈ FLAT       ({delta:+.4f})", _YLW)
                lines.append(f"\n  Trend:   {verdict}")

    # ── Footer ──────────────────────────────────────────────────────────────
    secs_left = max(0, int(refresh_at - time.time()))
    lines.append(f"\n{_DIM}{'─' * W}{_RST}")
    lines.append(f"{_DIM}  Next refresh in {secs_left}s  |  Ctrl+C to stop{_RST}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    _load_env()

    p = argparse.ArgumentParser(description="Live Kaggle / W&B training dashboard")
    p.add_argument("--kernel", default="deeperisbetter/tirramind-h-g")
    p.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "999-sbpatel"))
    p.add_argument("--project", default="tirramind")
    p.add_argument("--epochs", type=int, default=8, help="Epoch rows to show in table")
    p.add_argument(
        "--interval", type=int, default=60, help="Refresh interval in seconds"
    )
    p.add_argument("--once", action="store_true", help="Print once and exit")
    p.add_argument(
        "--log-file",
        default=str(REPO_ROOT / "logs" / "training_live.jsonl"),
        help="Append JSON snapshots here each refresh (for offline analysis)",
    )
    args = p.parse_args()

    api_key = os.environ.get("WANDB_API_KEY") or os.environ.get("TIRRA_WANDB_API_KEY")
    if not api_key:
        print(
            "ERROR: WANDB_API_KEY not set. Add it to .env at project root.",
            file=sys.stderr,
        )
        sys.exit(1)

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _append_log(kst: dict, run: dict | None, history: list[dict]) -> None:
        """Write one timestamped snapshot to the persistent JSONL log."""
        try:
            summary: dict = {}
            if run and "_error" not in run:
                try:
                    summary = json.loads(run.get("summaryMetrics") or "{}")
                except Exception:
                    pass
            latest_row = history[-1] if history else {}
            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "kernel_status": kst.get("status", "unknown"),
                "run_name": (run or {}).get("displayName") or (run or {}).get("name"),
                "run_state": (run or {}).get("state"),
                "run_heartbeat": (run or {}).get("heartbeatAt"),
                "epoch": summary.get(
                    "epoch", latest_row.get("epoch", latest_row.get("_step"))
                ),
                "loss_total": _v(latest_row, "loss/total", "loss_total"),
                "loss_return": _v(latest_row, "loss/return", "loss_return"),
                "loss_obs_type": _v(latest_row, "loss/obs_type", "loss_obs_type"),
                "loss_time_delta": _v(latest_row, "loss/time_delta", "loss_dt"),
                "loss_value": _v(latest_row, "loss/value", "loss_value"),
                "ic_emb_norm": _v(summary, "ic/emb_norm", "ic_emb_norm"),
                "ic_value_head": _v(summary, "ic/value_head", "ic_value_head"),
                "history_epochs": len(history),
            }
            with open(log_path, "a") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception:
            pass  # never crash the dashboard on log write failure

    def _fetch():
        kst = _kaggle_status(args.kernel)
        run = _latest_run(api_key, args.entity, args.project)
        history: list[dict] = []
        if run and "_error" not in run:
            history = _run_history(
                api_key, args.entity, args.project, run["name"], args.epochs
            )
        return kst, run, history

    if args.once:
        kst, run, history = _fetch()
        _append_log(kst, run, history)
        print(
            _render(
                kernel=args.kernel,
                kst=kst,
                run=run,
                history=history,
                entity=args.entity,
                project=args.project,
                refresh_at=0,
            )
        )
        return

    print(f"Starting live dashboard (refresh every {args.interval}s) — Ctrl+C to stop")
    time.sleep(0.3)

    try:
        while True:
            kst, run, history = _fetch()
            _append_log(kst, run, history)
            refresh_at = time.time() + args.interval

            sys.stdout.write("\033[H\033[J")  # clear screen
            sys.stdout.write(
                _render(
                    kernel=args.kernel,
                    kst=kst,
                    run=run,
                    history=history,
                    entity=args.entity,
                    project=args.project,
                    refresh_at=refresh_at,
                )
            )
            sys.stdout.write("\n")
            sys.stdout.flush()

            # Countdown — update bottom line every second without full redraw
            while time.time() < refresh_at:
                secs = max(0, int(refresh_at - time.time()))
                sys.stdout.write(
                    f"\033[s\033[999;0H\033[2K"  # save pos, goto last line, clear it
                    f"{_DIM}  Next refresh in {secs}s  |  Ctrl+C to stop{_RST}"
                    f"\033[u"  # restore pos
                )
                sys.stdout.flush()
                time.sleep(1)

    except KeyboardInterrupt:
        sys.stdout.write("\033[H\033[J")
        print("Dashboard stopped.")


if __name__ == "__main__":
    main()
