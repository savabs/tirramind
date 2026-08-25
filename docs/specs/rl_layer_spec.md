---
title: "Spec: RL Layer (Phase 4b)"
tags:
  - doc/spec
  - topic/reinforcement-learning
---

# Spec: RL Layer (Phase 4b)

## Goal

Replace LLM-only decision-making in the autonomous loop with a Thompson Sampling bandit that learns from numeric rewards. The bandit picks WHAT TYPE of work to do; the LLM fills in specifics.

## Files Affected

### New files
- `agent/learning/bandit.py` — GoalArm, StrategyBandit (Thompson Sampling), persistence
- `agent/learning/reward.py` — RewardComputer: Evaluation → scalar reward ∈ [0, 1]

### Modified files
- `agent/learning/goal_generator.py` — Add `generate_for_arm()` constrained to arm's tools
- `agent/core/autonomous.py` — Wire bandit into loop: choose → generate_for_arm → execute → reward → update
- `agent/memory/store.py` — Add `arm` and `reward` fields to LearningEntry
- `agent/learning/__init__.py` — Export new components

## Implementation Steps

### Research & Spec
- [x] 4b.1: Write research doc
- [x] 4b.2: Write spec doc

### StrategyBandit (`agent/learning/bandit.py`)
- [ ] 4b.3: Define `GoalArm` dataclass — name, description, tools, examples
- [ ] 4b.4: Implement `StrategyBandit` — Thompson Sampling with Beta(α, β) per arm
- [ ] 4b.5: Implement `choose()` — sample from each Beta distribution, return highest
- [ ] 4b.6: Implement `update(arm, reward)` — α += reward, β += (1 - reward)
- [ ] 4b.7: Implement persistence — save/load to JSON file
- [ ] 4b.8: Define `DEFAULT_ARMS` — 5 categories covering agent capabilities
- [ ] 4b.9: Implement `stats()` — return per-arm summary (pulls, mean reward, α, β)

### RewardComputer (`agent/learning/reward.py`)
- [ ] 4b.10: Implement `compute_reward(evaluation, is_first_pull)` → float ∈ [0, 1]
- [ ] 4b.11: Configurable weights: eval_weight, sharpe_weight, facts_weight, novelty_bonus, dead_end_penalty

### GoalGenerator Extension
- [ ] 4b.12: Add `generate_for_arm(arm, reflection, attempted)` — constrained goal within arm category

### Autonomous Loop Integration
- [ ] 4b.13: Wire bandit.choose() → generate_for_arm() → execute → reward → bandit.update()
- [ ] 4b.14: Add bandit stats to AutonomousRunSummary report

### Memory Extension
- [ ] 4b.15: Add `arm` and `reward` fields to LearningEntry (backward compatible defaults)

### Wrap-up
- [ ] 4b.16: Update `agent/learning/__init__.py` with exports
- [ ] 4b.17: Test all components

## Edge Cases

- **First run (cold start):** All arms have Beta(1,1) = uniform. First iterations are random. This is correct behavior.
- **All arms exhausted:** If every arm has been pulled many times and all have low reward, the bandit still picks the least-bad option. No special handling needed.
- **New arm added:** Starts with Beta(1,1) — fresh prior with high uncertainty → will be explored quickly.
- **Reward of exactly 0 or 1:** Both are valid. β += 1 or α += 1 respectively.

## Testing Plan

1. **Bandit convergence:** Create 3 arms with known reward distributions (0.8, 0.5, 0.2). Pull 100 times. Verify best arm gets >50% of pulls.
2. **Persistence roundtrip:** Save bandit state, create new instance from same file, verify α/β match.
3. **Reward computation:** Feed known Evaluations, verify reward values match expectations.
4. **Arm-constrained goals:** Verify generate_for_arm produces goals using the arm's tools.
5. **Integration:** Mock orchestrator.run(), verify bandit.update() is called with correct reward.

---

## Related

- [[rl_layer|Research: Rl Layer]]
