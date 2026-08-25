---
title: "Task: Autonomic Workflow System"
tags:
  - doc/task
  - status/active
  - phase/awos-1
  - topic/autonomic-workflow
  - layer/meta
---

# Task: Autonomic Workflow System

Status: active
Research: [[autonomic_workflow_system]]
Spec: [[autonomic_workflow_system_spec]]

## Goal

Build `agent/awos/` — a runtime that turns the AWOS from a passive markdown file
into a self-maintaining, event-driven system.

## Scope Notes

- Layer: meta (cross-cutting, not part of the 7-layer computation stack).
- Main files expected to change: new `agent/awos/` package, new `tests/test_awos_*.py`,
  `pyproject.toml`, `.gitignore`.
- Non-goals: multi-agent orchestration, IDE-beyond-Copilot support,
  autonomous git commits.

## Steps

- [x] 1.1: Scaffold `agent/awos/` package skeleton
  Verification: `python -c "from agent import awos; print(awos.__version__)"` → `0.1.0`
- [x] 1.2: Write `config.py` (AWOSConfig pydantic settings)
  Verification: `test_awos_config.py::test_config_from_env_and_yaml`
- [x] 1.3: Write `events/schema.py` (Event model + enums)
  Verification: `test_awos_events.py::test_event_roundtrip`
- [x] 1.4: Write `events/bus.py` (SQLite WAL bus)
  Verification: `test_awos_events.py::test_publish_fetch_dedup`
- [x] 1.5: Write `classifiers/base.py` (Protocol + Classification model)
  Verification: module imports cleanly
- [x] 1.6: Write `classifiers/heuristic.py`
  Verification: `test_awos_classifier_heuristic.py` table-driven cases
- [x] 1.7: Write `classifiers/prompt.py`
  Verification: prompt length sanity check
- [x] 1.8: Write `classifiers/anthropic.py`
  Verification: `test_awos_classifier_anthropic.py` with httpx mock
- [x] 1.9: Write `watchers/base.py`
  Verification: imports cleanly
- [x] 1.10: Write `watchers/drift.py` (wraps fact_lint)
  Verification: `test_awos_watcher_drift.py`
- [x] 1.11: Write `watchers/staleness.py`
  Verification: `test_awos_watcher_staleness.py`
- [x] 1.12: Write `watchers/obsidian.py` (wraps obsidian_lint)
  Verification: module imports, subprocess mocked
- [x] 1.13: Write `watchers/chat_log.py`
  Verification: `test_awos_watcher_chat_log.py` with fixture logs
- [x] 1.14: Write `policies/predicates.py` (DSL parser)
  Verification: `test_awos_policy_predicates.py`
- [x] 1.15: Write `policies/engine.py`
  Verification: `test_awos_policy_engine.py`
- [x] 1.16: Write `policies/default_policies.yaml`
  Verification: engine loads it without error
- [x] 1.17: Write `actions/base.py`
  Verification: imports cleanly
- [x] 1.18: Write `actions/awos_update.py`
  Verification: `test_awos_action_awos_update.py` golden-file test
- [x] 1.19: Write `actions/proposal.py`
  Verification: `test_awos_action_proposal.py`
- [x] 1.20: Write `actions/checkpoint_nudge.py`
  Verification: produces proposal with expected content
- [x] 1.21: Write `actions/adr_stub.py`
  Verification: produces valid ADR frontmatter
- [x] 1.22: Write `actions/registry.py`
  Verification: all actions register and resolve
- [x] 1.23: Write `orchestrator/scheduler.py`
  Verification: scheduler fires fake job
- [x] 1.24: Write `orchestrator/daemon.py`
  Verification: `test_awos_daemon.py` integration test
- [x] 1.25: Write `cli.py`
  Verification: `test_awos_cli.py` per command
- [x] 1.26: Write git hooks + installer
  Verification: installer idempotent
- [x] 1.27: Wire `pyproject.toml` entry point + gitignore
  Verification: `pip install -e .` exposes `tirra-awos`
- [x] 1.28: Integration test end-to-end
  Verification: synthetic event → AWOS entry appended

## Completion Checklist

- [x] Research note exists and is current
- [x] Spec matches the actual implementation plan
- [x] Each completed step has a verification result (102 AWOS tests passing)
- [x] Edge-case testing added and run for code changes
- [ ] Checkpoint written at the end of the session
- [x] Frontmatter tags and `## Related` section are current

## Related

- [[autonomic_workflow_system]]
- [[autonomic_workflow_system_spec]]
- [[agent_workflow_os]]

## Notes

- Keep steps atomic: one change, one test, one proof.
- Anthropic tests must mock httpx — no real API calls in tests.
- Chat log parser must be best-effort — Copilot log format is unstable.
