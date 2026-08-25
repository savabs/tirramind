---
title: "Feature: Agent Autonomy (Phase 4)"
tags:
  - doc/research
---

# Feature: Agent Autonomy (Phase 4)

## Current Architecture

### What exists
The agent is a **single-shot executor**: human gives a goal → orchestrator runs Research → Plan → Execute → Synthesize → returns result → stops.

| Component | State | What it does |
|-----------|-------|-------------|
| `orchestrator.py` | Working | 4-phase pipeline: research → plan → execute → synthesize. Replans up to 2× on failure. |
| `memory/store.py` | Working | 3-tier: episodic (action log, JSONL), semantic (key-value facts), working (rolling LLM context). |
| `task_planner.py` | Working | LLM generates JSON task tree from goal. 3-level max depth. |
| `learning/__init__.py` | Empty | The entire learning layer is a blank file. |
| `cli.py` | Working | 13 tools registered. Interactive REPL mode exists. |

### What's missing for autonomy

**The agent can't operate without a human typing a goal.** Specifically:

1. **No goal generation.** The agent doesn't decide what to work on. A human must provide every goal.
2. **No continuous loop.** After one goal completes, the agent stops. It can't run unattended.
3. **No learning from outcomes.** Episodic memory logs what happened but the agent never reviews it. It repeats the same mistakes.
4. **No reflection.** The agent doesn't ask "did that work? what should I do differently?"
5. **No adaptive planning.** All planning happens upfront. The agent can replan on failure, but doesn't adapt to intermediate discoveries.

## What "Autonomy" Actually Means (Scoped for Phase 4)

Full autonomy (the agent runs indefinitely, discovers data sources, generates strategies, and manages a portfolio) is a Phase 5+ vision. Phase 4 should build the **minimum infrastructure** that makes the agent genuinely self-directed on a bounded task.

### The autonomy loop

```
┌─────────────────────────────────────────────┐
│                AUTONOMY LOOP                │
│                                             │
│  1. REFLECT: Review recent episodes.        │
│     "What did I do? What worked? What       │
│      didn't? What should I try next?"       │
│                                             │
│  2. GENERATE GOAL: Based on reflection      │
│     + current quant state, pick the most    │
│     valuable next action.                   │
│                                             │
│  3. EXECUTE: Run the existing pipeline      │
│     (research → plan → execute → synth).    │
│                                             │
│  4. EVALUATE: Score the outcome. Was it     │
│     successful? Did it produce edge?        │
│     Store evaluation in semantic memory.    │
│                                             │
│  5. LEARN: Update beliefs, prune dead       │
│     ends, record what works. Feed into      │
│     next reflection.                        │
│                                             │
│  └──────────── loop back to 1 ────────────┘ │
└─────────────────────────────────────────────┘
```

### What to build NOW vs. LATER

**NOW (Phase 4):**

1. **Reflection engine** (`agent/learning/reflection.py`) — LLM reviews recent episodes and produces structured assessments: what worked, what failed, what to try next.
2. **Goal generator** (`agent/learning/goal_generator.py`) — Given reflection output + current quant state (tools, data, regimes), propose the most valuable next goal.
3. **Run evaluator** (`agent/learning/evaluator.py`) — After a run completes, score it: did it produce new knowledge? Did a strategy have edge? Record evaluation.
4. **Autonomous loop** (`agent/core/autonomous.py`) — A wrapper around Orchestrator.run() that loops: reflect → generate goal → execute → evaluate → repeat.
5. **Learning memory** — Extend semantic memory to store structured learning artifacts (strategy results, dead ends, open questions).
6. **CLI integration** — `python -m agent.cli --autonomous` mode that runs the loop.

**LATER (Phase 5+):**
- MCTS-augmented planning (simulate plan outcomes before committing)
- World model (predict what would happen if...)
- Multi-objective optimization (balance exploration vs. exploitation)
- Continuous background daemon with scheduling

## Observations

### The learning primitives are simpler than they sound

Reflection is an LLM call: "Here are the last N episodes. What worked? What didn't? What should I try next?" That's it. The output is structured text stored in semantic memory.

Goal generation is also an LLM call: "Given what I've learned, what's the most valuable thing to work on? Pick from: [list of available tools and capabilities]." The LLM already knows the domain from the system prompt.

Evaluation is mostly calling `score_returns()` or similar functions from the quant engine. The backtest tool already returns structured metrics.

### The hard part is preventing loops and wasted computation

Without guardrails, an autonomous agent will:
- Repeat the same failed approach
- Generate vague or untestable goals
- Never converge on actionable output
- Burn API credits on circular reasoning

**Guardrails to implement:**
- **Goal dedup:** Don't regenerate a goal that was already attempted (check semantic memory).
- **Max iterations per session:** Configurable cap (e.g., 10 goals per autonomous run).
- **Escalation:** If the agent is stuck (3+ failed goals in a row), pause and surface a summary for human review.
- **Budget awareness:** Track LLM API calls per run. Warn before exceeding a threshold.

## Risks

1. **Circular reasoning.** Agent reflects, generates a goal, fails, reflects on the failure, generates the same goal. Solved by goal dedup + failure tracking.
2. **Vague goals.** "Understand markets better" is not actionable. Goal generator must produce goals that map to specific tool calls. Validate before execution.
3. **LLM hallucination in reflection.** The agent might "learn" false lessons from noisy outcomes. Solved by requiring quantitative evaluation (Sharpe, significance tests) rather than narrative.
4. **API cost explosion.** Each loop iteration is ~5-10 LLM calls (reflect + generate + research + plan + synthesize + evaluate). Need budget tracking.

## Data Requirements

No new external data sources. The autonomous loop operates on the same tools and data already available. The new components are internal (reflection, evaluation, goal generation).

## Algorithm Survey

### Reflection
Simple prompt engineering. No new algorithms. The key design decision is the reflection prompt template — it must produce structured output (not freeform narrative).

### Goal generation
LLM-based with constraints. The generation prompt includes: (a) available tools, (b) recent learning, (c) open questions from reflection, (d) list of goals already attempted. Output: a single concrete goal string.

### Evaluation
Already built: `score_returns()`, `block_bootstrap_ci()`, `regime_conditional_analysis()`. For non-backtest goals (e.g., "research X"), evaluation is simpler: "did the agent produce new facts in semantic memory?"

### Dead-end detection
Track: goal → outcome → evaluation. If a goal leads to a failed or low-value outcome, record it as a dead end. Don't revisit dead ends unless new information (new data, new tools) makes them viable again.

---

## Related

- [[agent_autonomy_spec|Spec: Agent Autonomy]]
