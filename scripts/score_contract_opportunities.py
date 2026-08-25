"""Score live government-contract opportunities into a ranked EV report.

Capital-pillar runnable artifact: pulls real awards from gov_contracts
(USASpending), applies learned P(win) from the beta learner, and emits a ranked
JSON of expected-value opportunities. Demonstrates the full loop:

    fetch (gov_contracts) → score (EV = P(win)·(Bid−Cost)−Risk)
    → learn (WinProbabilityLearner from realized bids)

Usage:
    .venv/bin/python scripts/score_contract_opportunities.py [--limit N] [--out PATH]

Environment:
    TIRRA_OPPORTUNITY_LEARNER   store path for realized bid outcomes
                               (default .tirra_opportunities/win_learner.jsonl)
    TIRRA_OPPORTUNITY_OUT       default output path
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.quant.contract_opportunity import (  # noqa: E402
    WinProbabilityLearner,
    apply_learned_probabilities,
    opportunity_to_json,
    score_opportunities,
)
from agent.tools.gov_contracts import GovContractsTool  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Score live contract opportunities by EV")
    parser.add_argument("--limit", type=int, default=10, help="number of awards to fetch")
    parser.add_argument("--out", type=str, default=None, help="output JSON path (default: stdout)")
    parser.add_argument("--learner", type=str, default=None, help="learner store path")
    args = parser.parse_args()

    tool = GovContractsTool()
    result = tool.execute(mode="recent", limit=args.limit)
    awards = (result.data or {}).get("awards", [])
    if not awards:
        print("No awards returned (network/API issue?).", file=sys.stderr)
        return 1

    opps = score_opportunities(awards)

    learner_path = args.learner or ".tirra_opportunities/win_learner.jsonl"
    learner = WinProbabilityLearner(learner_path)
    learned = apply_learned_probabilities(opps, learner)

    rows = opportunity_to_json(learned)
    report = {
        "generated_from": "gov_contracts (USASpending)",
        "n_awards": len(awards),
        "n_scored": len(rows),
        "learner_evidence": learner.state()["n_outcomes"],
        "top_opportunity": rows[0]["award_id"] if rows else None,
        "opportunities": rows,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)

    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Wrote {len(rows)} scored opportunities -> {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
