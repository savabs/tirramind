---
title: "Autonomic Workflow System (AWOS Runtime)"
tags:
  - doc/research
  - phase/awos-1
  - topic/autonomic-workflow
  - topic/agent-operating-system
  - layer/meta
---

# Feature: Autonomic Workflow System

> Goal: make the project's own workflow self-maintaining. The agent should notice
> important architectural moments, roadmap shifts, lessons, and drift — and act
> on them (update AWOS, write proposals, nudge checkpoints, open ADR stubs)
> **without being asked**.

## Motivation

The user's exact words:
- "things happen automatically, i dont need to tell it always"
- "extreme level of intelligence in the making of the project itself"
- "if automated than that itself determine that our system engineering is very good"

The static AWOS memory file captures patterns but depends on the agent
remembering to update it. That is a fragile seam. Every missed update is a lost
compounding asset. The solution is not "a smarter agent" — it is a runtime that
watches conversation + repo state + time, classifies events, applies policies,
and produces proposals or direct updates.

## Current Architecture

- `.github/copilot-instructions.md` — workflow rules (strict research→spec→task pipeline).
- `AGENTS.md` — agent role definitions.
- `memories/agent_workflow_os.md` — the AWOS document (user-scope persistent).
- `scripts/` — already contains the right primitives:
  - `fact_lint.py` — detects copied numeric constants outside canonical owners.
  - `quality_gate.py` — pre-completion checks (tests + lint + task steps).
  - `obsidian_lint.py` — frontmatter + link integrity.
  - `rotate_checkpoints.py` — archive old checkpoints.
  - `session_checkpoint.py` — auto-generate checkpoint from git state.
  - `context_graph.py` — NetworkX + TF-IDF index over repo.
  - `extract_patterns.py` — mine completed tasks for lessons.
- `pyproject.toml` — Python 3.11, pydantic v2, apscheduler, httpx, pytest.
- `[project.scripts]` registers CLI entry points — new CLI fits natively.

## Prior Art Survey

### Agentic workflow frameworks
- **LangGraph** — stateful agent graphs. Too heavy for our need; we do not need
  multi-agent orchestration, we need a passive watcher + classifier + policy loop.
- **Temporal / Airflow / Prefect** — durable task orchestration. Overkill; we are
  event-driven with short, idempotent actions.
- **Event-sourced CQRS patterns** — the mental model is right: append-only event
  log, idempotent projections/actions. We adopt the spirit without the complexity.

### Observability / automation tooling
- **Watchdog** (Python inotify wrapper) — standard for file watchers; avoid when
  a simple `stat()+mtime` poll is enough to keep deps minimal.
- **APScheduler** — already a project dependency; good fit for periodic watchers.
- **Pre-commit framework** — already in repo (`.pre-commit-config.yaml`); our
  git-hook layer can coexist or reuse it rather than compete.

### Classification / structured LLM output
- **Anthropic Messages API** — explicit structured output via system prompt +
  JSON schema; confirmed working with `claude-haiku-3-5-20241022`.
  (Pricing: $0.80/M input, $4/M output — trivial for 1-2k token turns.)
- **Instructor / outlines** — libraries that enforce schema output. Not needed
  if we validate with pydantic on the response.
- **Guardrails** — heavy. Skip.

### Copilot chat log format
- VS Code stores Copilot Chat session logs under
  `~/.config/Code/User/workspaceStorage/<ws-hash>/GitHub.copilot-chat/debug-logs/<session-id>`.
- Format is **not a public API** and is considered unstable. We must parse
  defensively: treat unparseable entries as opaque blobs, tail-follow by offset,
  never depend on specific field names surviving.

## Observations

1. **The conversation channel is the missing input.** Every other surface
   (git, filesystem, tests) already has tools. What we lack is a classifier
   that reads turns and says "this one is AWOS-worthy."

2. **The AWOS file itself lives outside the workspace** at
   `~/.config/Code/User/globalStorage/github.copilot-chat/memory-tool/memories/agent_workflow_os.md`.
   This means actions need a configurable target path, not a workspace-relative one.

3. **Existing scripts are already the right atomic actions.** We do not need to
   reimplement drift detection, lint, or checkpoints — we need to *schedule* them
   and *route their outputs* into events.

4. **The user wants proposals, not unchecked autonomy.** Appending to AWOS is
   low-risk and can be direct. Writing ADRs, moving tasks, pushing code —
   these should become proposals the user approves.

5. **Latency tolerance is high.** Even minute-scale delays between "insight
   emerged in chat" and "AWOS updated" are fine. We can run async with polling.

## Design Principles

1. **Append-only event log.** Everything that happens (turn classified, watcher
   finding, hook fired, action executed) becomes an event row. This is auditable
   and restartable.

2. **Idempotent actions.** Re-processing the same event produces the same result.
   Each event has a UUID; actions record `processed_event_id` so they skip work.

3. **Fail-open, not fail-closed.** If the classifier crashes, the watchers still
   run and log events. If watchers crash, manual CLI still works. If the daemon
   crashes, nothing breaks — the next invocation replays from the event log.

4. **Proposals before mutations.** High-impact actions write a proposal file and
   wait for explicit acceptance. Low-impact ones (appending a dated note to AWOS
   changelog) can be direct — but are still logged as events for auditability.

5. **Policies in YAML, not code.** Rules like "if event.category == architectural
   then write_awos_update" should be editable without a code change.

6. **Heuristic fallback for classification.** Keyword/regex classifier runs
   locally for free and handles the common cases. LLM classifier is called only
   when the heuristic is uncertain. This keeps cost near zero while preserving
   quality on ambiguous turns.

7. **Copilot log parsing is best-effort.** We tail the newest session log file
   in the expected directory, try several JSON shapes, and gracefully degrade to
   "blob classification" when structure is unknown.

## The Four-Layer Model

### Layer 1 — Deterministic Hooks (reflexes)
Fire on concrete events. No intelligence required.
- Git hooks: post-commit → fact_lint + obsidian_lint → emit events.
- File watchers: tasks/active/*.md mtime change → emit `task_touched` event.
- Scheduled (APScheduler): nightly rotate_checkpoints, weekly extract_patterns.

### Layer 2 — Conversation Classifier (trigger detector)
- Input: a chunk of text (user message, assistant reply, or pasted block).
- Output: `TriggerCategory` ∈ {architectural, workflow_pattern, lesson,
  roadmap_shift, decision, routine, unknown} + confidence + rationale.
- Implementation: heuristic first, Anthropic fallback (or always-on
  depending on config.mode).

### Layer 3 — Repo State Watchers (health monitors)
- `drift_watcher` — wraps `scripts/fact_lint.py`.
- `staleness_watcher` — finds active tasks unmodified > N days.
- `coverage_watcher` — reads pytest json report for coverage gaps.
- `graph_watcher` — runs `scripts/obsidian_lint.py`, flags orphans/broken links.

### Layer 4 — Meta-Orchestrator (the autonomic loop)
- Event bus: SQLite (WAL mode) at `.awos/events.db`.
- Policy engine: loads YAML rules, matches events → dispatches actions.
- Action registry: `awos_update`, `proposal`, `checkpoint_nudge`, `adr_stub`.
- Daemon: asyncio event loop + APScheduler periodic tasks.

## Data Requirements

- **Conversation text**: obtained via chat log parser OR CLI stdin OR VS Code
  post-turn task. Degrades gracefully if log parsing fails.
- **Git state**: read-only via `subprocess(["git", "status", "--porcelain"])`
  and `git log --since`. No commits made without proposal acceptance.
- **Task files**: mtime + frontmatter parse. Cheap.
- **Existing lint output**: run existing scripts, capture stdout/exit-code,
  convert to events. Zero duplication.

## Math / Algorithm Survey

This is an engineering system, not a mathematical one. The only classifier
"math" is:

1. **Heuristic classifier scoring**: weighted keyword hit count, normalized by
   length, thresholded. Cheap, deterministic, explainable.

2. **LLM structured classification**: constrained JSON output with pydantic
   validation. Confidence comes from the model directly (self-reported probability).

3. **Policy matching**: boolean predicate DSL (simple `field == value`,
   `field in [...]`, `field > x`, `and/or`). No backtracking; rules are
   evaluated in declaration order and the first match wins unless marked
   `all: true`.

## Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Copilot log format changes | Medium | Best-effort parser + CLI fallback + unit tests with fixture logs |
| LLM classifier produces invalid JSON | Medium | Pydantic validation + heuristic fallback on parse failure |
| Anthropic billing runs out | Low | Heuristic-only mode; no hard dep on paid credits |
| Action writes corrupt AWOS file | High | Atomic writes (tmp + rename), backup of previous version, append-only by default |
| Daemon crashes leave partial state | Low | All state in SQLite; event log is source of truth; restart replays |
| Feedback loop: AWOS update triggers classifier triggers update | Medium | Mark self-generated writes with a marker → classifier filters them out |
| Noisy events flood the bus | Low | Rate limiting per category + dedup hash on recent events |
| Watcher runs slow tests and blocks loop | Medium | All watchers have a timeout; run in thread pool; log timeouts as events |
| Proposals pile up unreviewed | Low | `tirra-awos proposals` CLI + staleness warning on old proposals |
| Misclassification causes bad AWOS entries | Medium | Low-confidence classifications write proposals, not direct updates |

## Dependencies to Add

Minimal. All fit within the project's stack constraints.

| Package | Purpose | Already present? |
|---|---|---|
| `pydantic>=2.0` | Event + config schemas | Yes |
| `apscheduler>=3.10` | Periodic scheduler | Yes |
| `httpx>=0.25` | Anthropic HTTP client | Yes |
| `pyyaml` | Policy YAML loading | Implicit (Obsidian frontmatter already uses it via scripts) |
| `rich` | Pretty CLI output | Yes |

No new heavy dependencies. No watchdog, no LangGraph, no temporal. Standard
library `sqlite3`, `asyncio`, `threading`, `pathlib` handle the rest.

## Depth Roadmap (per signal-depth doctrine)

L1 (aggregate): "the AWOS file got updated today."
L2 (entity-level): "this specific principle was added because of turn X in
session Y, under rule `architectural_update_v1`, with confidence 0.82."
L3 (cross-entity): "this principle contradicts principle Y added 3 weeks ago —
contradiction flag + ADR proposal."

MVP targets L1+L2. L3 (contradiction detection across AWOS entries) is a
follow-up phase once there is enough AWOS content to exhibit contradictions.

## Related

- [[agent_workflow_os]]
- [[autonomic_workflow_system_spec]]
- [[autonomic_workflow_system]]
- [[fact_lint]]
- [[quality_gate]]
- [[obsidian_lint]]
- [[session_checkpoint]]
