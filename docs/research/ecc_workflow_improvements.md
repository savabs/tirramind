---
title: "Research: ECC-Inspired Workflow Improvements"
tags:
  - doc/research
  - topic/workflow
  - topic/agent-infrastructure
---

# Feature: ECC-Inspired Workflow Improvements

Derived from analysis of:
- **andrej-karpathy-skills** repo (17K+ stars) — 4 behavioral rules in a `CLAUDE.md` file
- **everything-claude-code** repo (156K+ stars, `affaan-m/everything-claude-code`) — 38 agents, 156 skills, hooks, memory persistence, continuous learning
- Google engineer's autonomous GitLab pipeline — 15-min polling loop, Claude reads issues → creates branches → writes code → pushes PRs

## Current Architecture

TirraMind already has:
- `copilot-instructions.md` — extensive behavioral constraints (far beyond Karpathy's 4 rules)
- `AGENTS.md` — 4 specialized agents with tool permissions
- `.github/prompts/` — 9 prompts (brainstorm-to-spec, debug, full-pipeline, next-step, post-mortem, research, review-quant, spec-to-task, sprint)
- `docs/memory/` — manual checkpoint system (~80 checkpoint files)
- `[[project_memory]]` — persistent architectural knowledge
- Mandatory research → spec → implement pipeline
- Obsidian knowledge graph with wiki links, tags, and backlinks

## Gap Analysis: What ECC Has That We Don't

### 1. Automated Session Context Capture
**Current:** Manual checkpoints written by the agent when asked or at natural breakpoints. Relies on agent discipline.
**ECC approach:** Hooks that auto-save session context on session end. SessionStart hooks that auto-load relevant context.
**Gap:** If a session ends abruptly or the agent forgets, context is lost. No automation layer.
**Fix:** A `scripts/session_checkpoint.py` script that can be called as a shell alias or git hook. Also a new prompt `session-start.prompt.md` that codifies the cold-start procedure.

### 2. Pattern Extraction from Completed Work
**Current:** Completed tasks move to `tasks/done/` but lessons learned stay buried in checkpoint files.
**ECC approach:** Auto-extract patterns from sessions into reusable skills.
**Gap:** No systematic mining of completed tasks for reusable patterns, common pitfalls, or workflow improvements.
**Fix:** A `scripts/extract_patterns.py` script that scans `tasks/done/` and `docs/memory/` for recurring themes and outputs a summary. A new prompt `extract-learnings.prompt.md` for agent-driven extraction.

### 3. Session-Start Cold-Start Prompt
**Current:** `copilot-instructions.md` says "read latest checkpoint → active task → follow links" but there's no dedicated prompt for it.
**ECC approach:** Dedicated session start hooks that restore context.
**Fix:** A `session-start.prompt.md` that automates the cold-start sequence.

### 4. Verification Loops / Quality Gates
**Current:** Edge case testing is mandatory per workflow rules, but there's no automated "quality gate" check.
**ECC approach:** Checkpoint vs continuous evals, grader types, pass@k metrics.
**Gap:** The edge case testing mandate relies on agent discipline. No script validates that tests exist and pass before marking a step done.
**Fix:** A `scripts/quality_gate.py` that checks: (1) test files exist for changed modules, (2) tests pass, (3) obsidian lint passes. Can be called from prompts.

### 5. Token Optimization / Context Hygiene
**Current:** `copilot-instructions.md` has good context efficiency rules but no tooling to enforce them.
**ECC approach:** Token optimization skills, model routing, system prompt slimming.
**Gap:** Checkpoint files accumulate (~80 files). Old ones waste context if the agent reads them. No rotation policy is automated.
**Fix:** A `scripts/rotate_checkpoints.py` that archives old checkpoints (>30 days) into a yearly summary file, keeping only the last N checkpoints active.

### 6. Loop Prevention
**Current:** Debug prompt exists but no systematic loop detection.
**ECC approach:** Observer loop prevention with 5-layer guard.
**Gap:** Agent can loop on the same fix attempt without triggering debug mode.
**Fix:** Add loop detection guidance to `copilot-instructions.md` — explicit "after 2 failed attempts at the same fix, switch to debug mode."

## Observations

- TirraMind's constraint file (`copilot-instructions.md`) is already more sophisticated than the Karpathy `CLAUDE.md` — we have layer separation, math-first priority, mandatory research, Obsidian integration, etc.
- The main gaps are in **automation** (hooks, scripts) and **knowledge recycling** (pattern extraction from completed work).
- ECC's agent count (38) is overkill for our use case. Our 4 agents + 9 prompts cover the workflow well. No need to add more agents.
- The autonomous issue→PR pipeline is premature for TirraMind (we need the prediction oracle first).

## Risks

- Over-engineering the workflow layer while the core product needs work. Keep changes minimal.
- Scripts that are never actually used. Each script must be wired into a prompt or alias to be useful.

## Related

- [[copilot-instructions]]
- [[project_memory]]
