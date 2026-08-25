"""Empirical proof of whether the AWOS learning loop actually compounds.

Tests the "self-improvement / compounding" claim by measuring whether the
LearningCore's learned router improves its method-choice over time against a
hidden world.

World model:
  Each signal-operation has a hidden TRUE-best method tier (unknown to the
  system). Choosing the right tier succeeds often; choosing the wrong tier
  succeeds rarely. The LinUCB router must DISCOVER this from outcomes.

Measurement (the honest part):
  - We bin runs into early / mid / late windows.
  - We measure (a) router method-accuracy vs the hidden truth, and
    (b) realized success rate.
  - Compounding means: later windows are statistically better than earlier ones.

If the loop is NOT learning, accuracy stays ~1/6 (random) and success stays
flat — and we report that honestly.
"""

from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.awos.learning.learning_core import LearningCore


# ── Hidden world: which method tier is truly best per operation ─────────────
OPERATIONS = {
    "fetch gov_contracts": 0,   # heuristic fetch
    "score gov_contracts": 2,   # statistical scoring
    "fuse cross_domain": 3,     # ml fusion
    "alert digest": 1,          # cheap-llm alert
    "clean parse": 0,           # heuristic normalize
}
N_ACTIONS = 6
TRUE_SUCCESS = 0.9     # P(success) when choosing the correct tier
FALSE_SUCCESS = 0.25   # P(success) when choosing a wrong tier


def world_success(operation: str, chosen_action: int) -> bool:
    """Simulate the world: correct tier → likely success, wrong → likely fail."""
    truth = OPERATIONS[operation]
    p = TRUE_SUCCESS if chosen_action == truth else FALSE_SUCCESS
    return random.random() < p


def run_compounding_proof(episodes: int = 400, seed: int = 42) -> dict:
    random.seed(seed)
    with tempfile.TemporaryDirectory() as d:
        core = LearningCore(state_dir=d)

        records: list[tuple[int, bool]] = []  # (chose_correct_action, success)

        for _ in range(episodes):
            op = random.choice(list(OPERATIONS.keys()))
            chosen = core.route_method(op)
            success = world_success(op, chosen)
            core.record_outcome(
                task_id=f"ep{_ % 1000}",
                operation=op,
                action_id=chosen,
                success=success,
                cost_usd=0.003 if chosen < 2 else 0.015,
                signal_name=op.split()[1],
                source_tool=op.split()[0],
                attempts=1,
            )
            records.append((chosen == OPERATIONS[op], success))

        # ── Bin outcomes and measure improvement ─────────────────────────────
        third = len(records) // 3
        early_chosen = [r[0] for r in records[:third]]
        mid_chosen = [r[0] for r in records[third : 2 * third]]
        late_chosen = [r[0] for r in records[2 * third :]]
        early_acc = sum(early_chosen) / max(len(early_chosen), 1)
        mid_acc = sum(mid_chosen) / max(len(mid_chosen), 1)
        late_acc = sum(late_chosen) / max(len(late_chosen), 1)

        early_succ = sum(r[1] for r in records[:third]) / max(len(records[:third]), 1)
        late_succ = sum(r[1] for r in records[2 * third :]) / max(len(records[2 * third :]), 1)

        random_baseline = 1.0 / N_ACTIONS

        return {
            "random_baseline_accuracy": random_baseline,
            "early_accuracy": round(early_acc, 3),
            "mid_accuracy": round(mid_acc, 3),
            "late_accuracy": round(late_acc, 3),
            "early_success": round(early_succ, 3),
            "late_success": round(late_succ, 3),
            "improvement": round(late_acc - early_acc, 3),
            "router_updates": core.rewards.total_episodes(),
        }


def main() -> None:
    result = run_compounding_proof()
    print("=== AWOS LEARNING COMPOUNDING PROOF ===")
    print(f"Random baseline (method accuracy if no learning): {result['random_baseline_accuracy']:.2f}")
    print(f"Early-run method accuracy : {result['early_accuracy']}")
    print(f"Mid-run  method accuracy : {result['mid_accuracy']}")
    print(f"Late-run method accuracy : {result['late_accuracy']}")
    print(f"Early success rate       : {result['early_success']}")
    print(f"Late success rate        : {result['late_success']}")
    print(f"Improvement (late-early) : {result['improvement']}")
    print(f"Episodes logged          : {result['router_updates']}")

    improved = result["late_accuracy"] > result["early_accuracy"] + 0.1
    beats_random = result["late_accuracy"] > result["random_baseline_accuracy"] + 0.2
    print()
    if improved and beats_random:
        print("VERDICT: COMPOUNDING CONFIRMED — later runs pick the right method "
              "far more often than early runs, and beat random.")
    elif improved:
        print("VERDICT: IMPROVING but below a strong bar — late > early, but still marginal.")
    else:
        print("VERDICT: NOT COMPOUNDING — the loop is not learning (late ~ early). "
              "This is an honest negative result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())