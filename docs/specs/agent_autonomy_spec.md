---
title: "Spec: Agent Autonomy (Phase 4)"
tags:
  - doc/spec
---

# Spec: Agent Autonomy (Phase 4)

## Goal

Make the agent self-directed: it reflects on past runs, generates its own goals, executes them through the existing pipeline, evaluates outcomes, and learns — in a loop. Replace "human types goal → agent executes" with "agent decides what to do next."

## Files Affected

### New files
- `agent/learning/reflection.py` — Reflection engine: review episodes, produce structured assessment
- `agent/learning/goal_generator.py` — Goal generation: propose most valuable next action
- `agent/learning/evaluator.py` — Run evaluation: score outcomes, detect dead ends
- `agent/core/autonomous.py` — Autonomous loop: reflect → generate → execute → evaluate → repeat

### Modified files
- `agent/learning/__init__.py` — Export learning components
- `agent/memory/store.py` — Add structured learning entries to semantic memory
- `agent/cli.py` — Add `--autonomous` mode

## Implementation Steps

### Research & Spec (no code)
- [ ] 4.1: Write research doc (`[[agent_autonomy]]`)
- [ ] 4.2: Write spec doc (`[[agent_autonomy_spec]]`)

### Reflection Engine
- [ ] 4.3: Create `agent/learning/reflection.py` — `Reflector` class with `reflect(episodes, semantic_facts) -> ReflectionResult`
- [ ] 4.4: Define `ReflectionResult` dataclass — what_worked, what_failed, open_questions, suggested_next_actions
- [ ] 4.5: Test Reflector — feed synthetic episodes, verify structured output is parseable

### Goal Generator
- [ ] 4.6: Create `agent/learning/goal_generator.py` — `GoalGenerator` class with `generate(reflection, available_tools, attempted_goals) -> Goal`
- [ ] 4.7: Define `Goal` dataclass — description, rationale, expected_tool, priority, is_novel (not previously attempted)
- [ ] 4.8: Implement goal deduplication — check against semantic memory for previously attempted goals
- [ ] 4.9: Test GoalGenerator — verify it produces concrete, tool-mappable goals (not vague)

### Run Evaluator
- [ ] 4.10: Create `agent/learning/evaluator.py` — `Evaluator` class with `evaluate(agent_result, goal) -> Evaluation`
- [ ] 4.11: Define `Evaluation` dataclass — success (bool), score (float 0-1), new_facts_count, strategy_metrics (if backtest), dead_end (bool), lessons (list[str])
- [ ] 4.12: Implement quantitative evaluation — for backtest results, extract Sharpe/significance. For research results, count new semantic facts.
- [ ] 4.13: Test Evaluator — feed mock AgentResult, verify scoring logic

### Learning Memory Extensions
- [ ] 4.14: Add `LearningEntry` to `store.py` — structured record: goal, evaluation, timestamp, dead_end flag
- [ ] 4.15: Add `store_learning()` and `get_attempted_goals()` to SemanticMemory
- [ ] 4.16: Test learning memory — store and retrieve entries, verify dedup query works

### Autonomous Loop
- [ ] 4.17: Create `agent/core/autonomous.py` — `AutonomousRunner` class
- [ ] 4.18: Implement the loop: reflect → generate goal → orchestrator.run() → evaluate → store learning → repeat
- [ ] 4.19: Add guardrails: max_iterations, goal dedup, stuck detection (3+ failures → pause), budget tracking
- [ ] 4.20: Test autonomous loop with max_iterations=2 — verify it runs two goals and stops

### CLI Integration
- [ ] 4.21: Add `--autonomous` flag to CLI — starts AutonomousRunner instead of single Orchestrator.run()
- [ ] 4.22: Add `--max-goals` flag — caps iterations (default: 5)
- [ ] 4.23: Test end-to-end: `python -m agent.cli --autonomous --max-goals 1` completes without error

### Wrap-up
- [ ] 4.24: Update `agent/learning/__init__.py` with exports
- [ ] 4.25: Update task file, mark Phase 4 complete

## Edge Cases

- **Empty episode history:** First run has no episodes to reflect on. Reflector should produce a "cold start" reflection: "No prior experience. Suggest exploratory goals."
- **LLM returns unparseable reflection:** Fallback to a default reflection with generic exploration goals.
- **Goal generator produces duplicate:** Goal dedup rejects it, generator retries (max 3 retries, then escalate).
- **All goals fail:** After 3 consecutive failures, autonomous loop pauses and prints summary for human review.
- **Budget exhaustion:** Track cumulative LLM calls. Warn at 80% of configurable limit, stop at 100%.

## Testing Plan

1. Reflection: feed 5 synthetic episodes (3 success, 2 failure) → verify ReflectionResult has all fields populated.
2. Goal generation: provide reflection + tool list → verify goal is concrete and maps to a real tool.
3. Evaluation: feed mock AgentResult with backtest data → verify Sharpe is extracted and scored.
4. Learning memory: store 3 goals, query attempted → verify all 3 returned.
5. Autonomous loop (mock): replace orchestrator.run() with a mock → verify loop executes N iterations with reflect/generate/evaluate at each step.
6. CLI: `--autonomous --max-goals 1` runs and exits cleanly.

---

## Related

- [[agent_autonomy|Research: Agent Autonomy]]
