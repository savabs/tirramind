---
title: "Spec: Autonomic Workflow System"
tags:
  - doc/spec
  - phase/awos-1
  - topic/autonomic-workflow
  - layer/meta
---

# Spec: Autonomic Workflow System

## Goal

Build a runtime (`agent/awos/`) that turns the AWOS document from a passive
markdown file into a self-maintaining, event-driven system. It must:

1. Classify conversation turns and propose/apply AWOS updates automatically.
2. Run periodic repo-state watchers and raise actionable events.
3. Execute a policy engine that routes events to actions (direct updates or
   proposals).
4. Expose a `tirra-awos` CLI for daemon control, one-shot classification,
   and proposal management.

## Files Affected

### New package — `agent/awos/`

```
agent/awos/
├── __init__.py                  # exports version, AWOSConfig
├── version.py
├── config.py                    # pydantic settings, env + yaml loader
├── events/
│   ├── __init__.py
│   ├── schema.py                # Event, TriggerCategory, EventStatus
│   └── bus.py                   # SQLite-backed event bus (WAL mode)
├── classifiers/
│   ├── __init__.py
│   ├── base.py                  # Classifier protocol + Classification result
│   ├── heuristic.py             # keyword/regex classifier (free, offline)
│   ├── anthropic.py             # Claude Haiku 3.5 classifier (structured JSON)
│   └── prompt.py                # system + user prompt templates
├── watchers/
│   ├── __init__.py
│   ├── base.py                  # Watcher protocol
│   ├── chat_log.py              # tails Copilot chat debug-logs
│   ├── drift.py                 # wraps scripts/fact_lint.py
│   ├── staleness.py             # finds stale active tasks
│   └── obsidian.py              # wraps scripts/obsidian_lint.py
├── policies/
│   ├── __init__.py
│   ├── engine.py                # rule loader + matcher + dispatcher
│   ├── predicates.py            # DSL parser for rule conditions
│   └── default_policies.yaml    # shipped rule set
├── actions/
│   ├── __init__.py
│   ├── base.py                  # Action protocol + ActionResult
│   ├── awos_update.py           # append to AWOS memory file atomically
│   ├── proposal.py              # write .awos/proposals/NNNN-<slug>.md
│   ├── checkpoint_nudge.py      # write checkpoint suggestion proposal
│   ├── adr_stub.py              # write docs/adr/ stub proposal
│   └── registry.py              # name→action lookup
├── orchestrator/
│   ├── __init__.py
│   ├── daemon.py                # asyncio loop — drain events, dispatch
│   └── scheduler.py             # APScheduler wrapper for periodic watchers
├── hooks/
│   ├── post-commit              # shell wrapper: tirra-awos scan --quick
│   └── install.sh               # idempotent installer
└── cli.py                       # `tirra-awos` typer/argparse CLI
```

### Tests — `tests/`

```
tests/
├── test_awos_events.py
├── test_awos_classifier_heuristic.py
├── test_awos_classifier_anthropic.py    # HTTP mocked
├── test_awos_policy_engine.py
├── test_awos_policy_predicates.py
├── test_awos_watcher_drift.py
├── test_awos_watcher_staleness.py
├── test_awos_watcher_chat_log.py
├── test_awos_action_awos_update.py
├── test_awos_action_proposal.py
├── test_awos_daemon.py
└── test_awos_cli.py
```

### Config + runtime state

- `.awos/events.db` — SQLite event log (gitignored).
- `.awos/proposals/` — pending proposals directory (gitignored).
- `.awos/state.json` — last-scan offsets per watcher (gitignored).
- `agent/awos/policies/default_policies.yaml` — shipped, editable.
- `.awos/policies.yaml` — optional user override (merged on top of default).

### Pyproject changes

- Add entry point: `tirra-awos = "agent.awos.cli:main"`.
- Add to `dependencies`: `pyyaml>=6.0` (if missing — check).

### Gitignore additions

- `.awos/`

## Data Schemas

### `events/schema.py`

```python
class TriggerCategory(str, Enum):
    ARCHITECTURAL = "architectural"      # design decisions, layer changes
    WORKFLOW_PATTERN = "workflow_pattern"  # process rules, how-we-work
    LESSON = "lesson"                    # post-mortem, mistake-avoidance
    ROADMAP_SHIFT = "roadmap_shift"      # phase reorder, priority change
    DECISION = "decision"                # any user-approved choice
    DRIFT = "drift"                      # fact drift, stale doc, broken link
    STALENESS = "staleness"              # task/doc hasn't moved
    ROUTINE = "routine"                  # implementation, test, typo
    UNKNOWN = "unknown"

class EventStatus(str, Enum):
    NEW = "new"
    PROCESSED = "processed"
    IGNORED = "ignored"
    ERRORED = "errored"

class Event(BaseModel):
    id: str                              # uuid4
    ts: datetime                         # UTC
    source: str                          # "chat_log" | "drift_watcher" | "cli" | ...
    category: TriggerCategory
    confidence: float = Field(ge=0.0, le=1.0)
    status: EventStatus = EventStatus.NEW
    payload: dict                        # source-specific
    rationale: str | None = None
    parent_event_id: str | None = None   # for chained events
```

### `classifiers/base.py`

```python
class Classification(BaseModel):
    category: TriggerCategory
    confidence: float
    rationale: str
    extracted_principle: str | None = None   # if classifier distilled a rule
    suggested_section: str | None = None     # AWOS section header hint

class Classifier(Protocol):
    name: str
    def classify(self, text: str, context: dict | None = None) -> Classification: ...
```

### `policies/engine.py` — YAML rule schema

```yaml
- id: architectural_to_awos
  when:
    category: architectural
    confidence_gte: 0.7
  then:
    action: awos_update
    params:
      section_hint: "{suggested_section | default('11. New Principles')}"
      require_approval: false

- id: low_confidence_to_proposal
  when:
    category: in: [architectural, workflow_pattern, lesson]
    confidence_lt: 0.7
  then:
    action: proposal
    params:
      target_file: memories/agent_workflow_os.md
```

## Implementation Steps

Each step is atomic, independently verifiable, and atomic at the file level.

1.1: **Scaffold package.** Create `agent/awos/` with empty `__init__.py` and
    `version.py` (returns "0.1.0"). Add `.awos/` to `.gitignore`.
    Verify: `python -c "from agent import awos; print(awos.__version__)"` prints 0.1.0.

1.2: **Write `config.py`.** Pydantic `AWOSConfig` with fields for db path,
    proposals dir, awos_file_path, anthropic_api_key, classifier_mode
    (heuristic | llm | hybrid), watcher intervals. Loads from env + optional
    `.awos/config.yaml`. Verify: unit test constructs config from env dict.

1.3: **Write `events/schema.py`.** Enums + Event model. Verify: pydantic validates
    round-trip JSON.

1.4: **Write `events/bus.py`.** SQLite WAL-mode backend. Methods:
    `publish(event)`, `fetch(limit, category=None, status=None)`,
    `mark_processed(id)`, `mark_ignored(id)`. Dedup via a
    content-hash column to prevent double-publish of identical payloads within
    a 10-minute window. Verify: test_awos_events.py — publish then fetch,
    dedup test, concurrent-publish test (2 threads).

1.5: **Write `classifiers/base.py`.** Protocol + Classification model. Verify:
    imports cleanly.

1.6: **Write `classifiers/heuristic.py`.** Keyword-weighted classifier with
    per-category regex lists. Returns confidence 0.3–0.7 based on hit density.
    Verify: table-driven tests on representative snippets covering each category.

1.7: **Write `classifiers/prompt.py`.** System + user prompt constants.
    Include schema in prompt so Haiku produces valid JSON. Verify: prompt
    length sanity check (< 2k tokens).

1.8: **Write `classifiers/anthropic.py`.** httpx-based Anthropic Messages API
    client. Retries with exponential backoff (max 3). Pydantic-validates the
    JSON response. Falls back to UNKNOWN + confidence 0.0 on API/parse error.
    Verify: tests mock httpx.AsyncClient.post; confirm retry, success, failure,
    bad-json paths.

1.9: **Write `watchers/base.py`.** Watcher protocol: `scan() -> list[Event]`.
    Timeout wrapper. Verify: imports cleanly.

1.10: **Write `watchers/drift.py`.** Shells out to `scripts/fact_lint.py`,
    parses its stdout (non-zero exit = findings), emits DRIFT events.
    Verify: mock subprocess — test success case (no events), failure with
    findings (events emitted).

1.11: **Write `watchers/staleness.py`.** Walks `tasks/active/*.md`, checks
    file mtime, emits STALENESS event if > configured threshold (default 7d).
    Verify: fixture tasks dir with old + new files; assert correct event count.

1.12: **Write `watchers/obsidian.py`.** Shells out to `scripts/obsidian_lint.py`,
    converts its findings into DRIFT events with payload tagged `subtype:
    obsidian`. Verify: mock subprocess.

1.13: **Write `watchers/chat_log.py`.** Tails the newest Copilot debug log
    file. State (last byte offset) persisted in `.awos/state.json`. Best-effort
    JSON line parser; unparseable lines treated as blob text and queued for
    classification. Emits events with `source="chat_log"` and
    `payload={text, role?, session_id?}`. Verify: fixture log files with
    mixed JSON and garbage; assert correct parsing.

1.14: **Write `policies/predicates.py`.** Small DSL: dict of conditions like
    `{"category": "architectural", "confidence_gte": 0.7}`. Support operators:
    `eq` (default), `in`, `_gte`, `_lt`, `_gt`, `_lte`. Verify: table-driven
    tests with 10+ combinations.

1.15: **Write `policies/engine.py`.** Load YAML, match events against rules
    in declaration order, dispatch matching rule's action. Verify: load
    default_policies.yaml, fire synthetic events, assert correct action
    dispatched.

1.16: **Write `policies/default_policies.yaml`.** Ship a sensible default set:
    architectural high-confidence → awos_update; anything low-confidence →
    proposal; drift → proposal; staleness → proposal.

1.17: **Write `actions/base.py`.** Protocol + ActionResult(success, artifact_path,
    message). Verify: imports cleanly.

1.18: **Write `actions/awos_update.py`.** Atomic append to AWOS file:
    1. Read current AWOS file.
    2. Backup to `<path>.bak`.
    3. Locate section by header (or append new section at end).
    4. Insert entry with date + rationale + event id marker.
    5. Atomic write (tmp + rename).
    6. Update changelog table at bottom.
    Mark all self-written content with a hidden HTML comment
    `<!-- awos:self -->` so classifier can skip it.
    Verify: golden-file test — start with known AWOS, run update, compare.

1.19: **Write `actions/proposal.py`.** Writes `.awos/proposals/NNNN-<slug>.md`
    with frontmatter. Includes source event, suggested change, accept/reject
    command hints. Verify: temp proposals dir, run action, assert file exists
    with correct content.

1.20: **Write `actions/checkpoint_nudge.py`.** Specialized proposal: suggests
    running `scripts/session_checkpoint.py`. Verify: asserts expected content.

1.21: **Write `actions/adr_stub.py`.** Writes `docs/adr/NNNN-<slug>.md` stub as
    a proposal (not directly — ADRs need review). Verify: generated ADR has
    valid frontmatter.

1.22: **Write `actions/registry.py`.** Dict-based name→callable lookup. Verify:
    all registered names resolve.

1.23: **Write `orchestrator/scheduler.py`.** APScheduler wrapper. Registers
    watchers as periodic jobs with configurable intervals. Verify: scheduler
    starts, fires a fake job, shuts down cleanly.

1.24: **Write `orchestrator/daemon.py`.** Main asyncio loop:
    - On start: initialize bus, scheduler, policy engine.
    - Periodic: pull new events from bus, run heuristic classifier on
      unclassified text events, escalate to Anthropic if mode allows, match
      policies, dispatch actions, mark events processed.
    - Signal handlers for graceful shutdown (SIGINT, SIGTERM).
    Verify: integration test — publish synthetic event, assert action ran.

1.25: **Write `cli.py`.** Commands (argparse subparsers — no new deps):
    - `daemon [--foreground]`
    - `classify <text> | classify --stdin | classify --file <path>`
    - `events [--limit N] [--status new]`
    - `proposals [--all]`
    - `accept <proposal_id>` / `reject <proposal_id>`
    - `scan [--watcher <name>] [--all]`
    - `install-hooks`
    - `status` — one-line summary
    Verify: tests hit each command with fixture env; assert exit codes
    and stdout contains expected substrings.

1.26: **Write `hooks/post-commit` + `hooks/install.sh`.** Post-commit runs
    `tirra-awos scan --watcher drift --quick`. Installer copies hooks into
    `.git/hooks/` (idempotent, backs up existing). Verify: installer run
    twice is a no-op second time.

1.27: **Wire pyproject entry point.** Add
    `tirra-awos = "agent.awos.cli:main"` to `[project.scripts]`.
    Verify: `pip install -e . && which tirra-awos`.

1.28: **Integration test.** Publish a synthetic architectural event via CLI,
    verify AWOS file gets a new entry and changelog row.

## Edge Cases

| # | Scenario | Expected behavior |
|---|---|---|
| E1 | Anthropic API down | Classifier returns UNKNOWN, heuristic fallback kicks in |
| E2 | Anthropic returns invalid JSON | Retry once; on second failure, UNKNOWN |
| E3 | AWOS file has no matching section for entry | Append a new section at end |
| E4 | AWOS file write fails mid-way (disk full) | Restore from `.bak`; log ERRORED event |
| E5 | Two watchers emit identical drift finding | Dedup on content hash within 10-min window |
| E6 | Chat log file rotates (new session) | State tracks per-file offset; old file's offset retained for reruns |
| E7 | Chat log contains assistant-written AWOS update text | Filter turns containing `<!-- awos:self -->` |
| E8 | Daemon restarted mid-processing | Unprocessed events in bus retried on next run |
| E9 | Proposal already exists for identical finding | `proposal.py` dedups on title slug |
| E10 | User accepts non-existent proposal | CLI returns exit 2 with clear error |
| E11 | Policy YAML has syntax error | Fail fast at engine init with line number |
| E12 | Policy rule references unknown action | Engine raises at load time, not at dispatch |
| E13 | Event payload too large (>1MB) | Truncate to 1MB, set payload_truncated flag |
| E14 | SQLite locked by another process | WAL mode + retry with backoff (max 3) |
| E15 | Watcher timeout exceeded | Cancel, log TIMEOUT event, continue loop |

## Testing Plan

- **Unit tests** per module (~80% of test count). Mock subprocess, mock httpx,
  use tmp_path fixtures for files.
- **Integration tests** for daemon loop with synthetic events end-to-end.
- **Golden file tests** for AWOS update action.
- **Fixture-based tests** for chat log parser with multiple formats.
- **Edge-case tests** for every scenario in the table above.

Minimum target: every module in `agent/awos/` has at least one dedicated
test file. Target total AWOS test count: ~60+ tests.

## Non-Goals (Explicit)

- Not building a multi-agent framework.
- Not parsing arbitrary IDE logs (Cursor, Zed, etc.) — Copilot only for MVP.
- Not implementing an LLM-powered policy engine (rules stay deterministic).
- Not replacing existing `scripts/` — wrapping them.
- Not committing or pushing code autonomously.

## Acceptance Criteria

1. `tirra-awos status` prints a one-line status when run in the repo.
2. `tirra-awos classify "we should always run fact_lint before merging"` returns
   a classification with category ∈ {WORKFLOW_PATTERN, ARCHITECTURAL}.
3. `tirra-awos daemon --foreground` starts, runs watchers on schedule, and
   responds to SIGINT gracefully.
4. A synthetic ARCHITECTURAL event with confidence 0.8 written directly to the
   bus causes AWOS to receive a new entry within one daemon cycle.
5. Test suite passes (target ≥ 60 new tests, 0 regressions in existing suite).

## Related

- [[autonomic_workflow_system]]
- [[agent_workflow_os]]
- [[autonomic_workflow_system_task]]
