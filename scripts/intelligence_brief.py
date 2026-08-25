"""Intelligence Brief — one fused, consumer-actionable output.

Combines the proven live-path pieces into a single brief a human or API can
act on:

  1. CONTRACT OPPORTUNITIES — live USASpending awards, EV-scored with the
     learned P(win) (WinProbabilityLearner), long-tail (small contracts) first.
  2. LIVE ANOMALIES — real z-score / changepoint signals from stored
     observations (the digest), as decision context.

Everything is deterministic math on real data. No LLM. Clean JSON output, with
an optional human-readable markdown render.

Usage:
    .venv/bin/python scripts/intelligence_brief.py --contracts 10 --anomalies 8
    .venv/bin/python scripts/intelligence_brief.py --md --out brief.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.quant.contract_opportunity import (  # noqa: E402
    WinProbabilityLearner,
    apply_learned_probabilities,
    opportunity_to_json,
    score_opportunities,
)
from agent.tools.gov_contracts import GovContractsTool  # noqa: E402
from scripts.live_intelligence_digest import build_digest  # noqa: E402

# Amount below which a contract is "long-tail" — the underserved wedge the big
# procurement-intel tools ignore (deliberately conservative).
LONG_TAIL_MAX_USD = 100_000.0


def fetch_opportunities(
    limit: int = 10,
    learner_path: str = ".tirra_opportunities/win_learner.jsonl",
    db_path: str = ".tirra_pipeline/pipeline.db",
    max_rows: int = 5,
) -> list[dict[str, Any]]:
    """Live contract opportunities, EV-ranked, long-tail first, learned P(win)."""
    tool = GovContractsTool()
    result = tool.execute(mode="recent", limit=limit)
    awards = (result.data or {}).get("awards", [])
    if not awards:
        return []

    opps = score_opportunities(awards)
    learner = WinProbabilityLearner(learner_path)
    learned = apply_learned_probabilities(opps, learner)

    # Long-tail (small, overlooked) contracts first, then by EV.
    def sort_key(o):
        is_tail = (o.amount_usd or 0.0) <= LONG_TAIL_MAX_USD
        return (not is_tail, -o.expected_value)

    learned.sort(key=sort_key)
    rows = opportunity_to_json(learned)
    for row in rows:
        row["is_long_tail"] = (row.get("amount_usd") or 0.0) <= LONG_TAIL_MAX_USD
    return rows[:max_rows]


def fetch_anomalies(
    db_path: str = ".tirra_pipeline/pipeline.db",
    max_rows: int = 8,
) -> list[dict[str, Any]]:
    """Top current anomalies from real stored observations (digest)."""
    from agent.pipeline.store import PipelineStore

    store = PipelineStore(db_path)
    report = build_digest(store, top_n=max_rows)
    return report["digest"]


def build_brief(
    *,
    contracts_limit: int = 10,
    anomalies_limit: int = 8,
    learner_path: str = ".tirra_opportunities/win_learner.jsonl",
    db_path: str = ".tirra_pipeline/pipeline.db",
    max_contract_rows: int = 5,
) -> dict[str, Any]:
    """Fuse contract opportunities + live anomalies into one brief."""
    return {
        "brief_type": "intelligence",
        "contract_opportunities": fetch_opportunities(
            limit=contracts_limit, learner_path=learner_path,
            db_path=db_path, max_rows=max_contract_rows,
        ),
        "live_anomalies": fetch_anomalies(db_path=db_path, max_rows=anomalies_limit),
    }


def render_markdown(brief: dict[str, Any]) -> str:
    lines: list[str] = ["# AWOS Intelligence Brief", ""]
    lines.append("## Contract Opportunities (long-tail first, learned P(win))")
    lines.append("")
    opps = brief["contract_opportunities"]
    if not opps:
        lines.append("_(none available — run again when awards post)_")
    for o in opps:
        tag = "🟢" if o.get("is_long_tail") else "🔵"
        lines.append(
            f"- {tag} `{o['award_id']}` **{o['recipient'][:40]}** — {o['agency']}"
        )
        lines.append(
            f"    amount=${o['amount_usd'] or 0:,.0f} · EV=${o['expected_value_usd']:,.0f} · "
            f"P(win)={o['p_win']:.0%}"
        )
        if o.get("description"):
            lines.append(f"    {o['description'][:100]}")
    lines.append("")
    lines.append("## Live Anomalies (real z-score / changepoint signals)")
    lines.append("")
    anom = brief["live_anomalies"]
    if not anom:
        lines.append("_(none currently flagged)_")
    for a in anom:
        cp = " [changepoint]" if a["changepoint"] else ""
        lines.append(
            f"- `{a['source']}` {a['observation_type']} · {a['field']} "
            f"→ z={a['zscore']:+.2f}{cp}"
        )
    lines.append("")
    lines.append(
        "_Deterministic math on live public data. No LLM involved. "
        "EV = P(win)·(Bid−Cost)−Risk._"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fused intelligence brief (contracts + anomalies)")
    parser.add_argument("--contracts", type=int, default=10, help="awards to fetch")
    parser.add_argument("--anomalies", type=int, default=8, help="anomalies to include")
    parser.add_argument("--max-contract-rows", type=int, default=5, help="rows to keep")
    parser.add_argument("--out", type=str, default=None, help="write JSON to path")
    parser.add_argument("--md", action="store_true", help="render markdown too")
    parser.add_argument("--learner", type=str, default=".tirra_opportunities/win_learner.jsonl")
    parser.add_argument("--db", type=str, default=".tirra_pipeline/pipeline.db")
    args = parser.parse_args()

    brief = build_brief(
        contracts_limit=args.contracts,
        anomalies_limit=args.anomalies,
        learner_path=args.learner,
        db_path=args.db,
        max_contract_rows=args.max_contract_rows,
    )

    if args.out:
        Path(args.out).write_text(json.dumps(brief, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote brief -> {args.out}")
    else:
        print(json.dumps(brief, indent=2, ensure_ascii=False))

    if args.md:
        md_path = args.out + ".md" if args.out else None
        text = render_markdown(brief)
        if md_path:
            Path(md_path).write_text(text, encoding="utf-8")
            print(f"Wrote markdown -> {md_path}")
        else:
            print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
