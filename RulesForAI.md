---
title: Rules For AI
tags:
  - layer/feature-engineering
---

# Rules For AI

This file is a thin operating guide for day-to-day collaboration.

Primary architecture and workflow rules still live in `.github/copilot-instructions.md`.
If this file conflicts with that file, `.github/copilot-instructions.md` wins.

## 1. Start With Structure

- For any non-trivial change, begin with research, then spec, then task decomposition before code.
- Fail closed: if the work is not obviously trivial, do not start implementation until those three artifacts exist.
- Preflight checklist for non-trivial work:
	- `docs/research/<feature_name>.md`
	- `docs/specs/<feature_name>_spec.md`
	- `tasks/active/<task_name>.md`
- Before the preflight is complete, edit only research/spec/task/checkpoint files.
- Treat multi-file changes, behavior changes, config changes, prompt changes, and architecture changes as non-trivial by default.
- Do not merge multiple problems into one request when they can be separated into atomic steps.
- If a task step changes more than one thing, split it.

## 2. Use The Right Mode

- Use chat for brainstorming, codebase research, architecture decisions, and spec drafting.
- Use the coding agent for bounded implementation steps that are already clear and testable.
- When implementation starts, cite the task/spec being executed rather than improvising a new plan in chat.
- If the request is still ambiguous, stay in planning mode rather than generating code early.

## 3. One Problem Per Step

- Separate extraction from summarization.
- Separate data fetching from feature engineering.
- Separate implementation from registration.
- Separate code changes from tests when the test scope is not yet defined.

## 4. Prefer Explicit File Responsibility

- File names should reflect a single responsibility.
- Avoid vague catch-all files when a clearer module boundary exists.
- Before creating a new file, confirm that an existing file with the same responsibility does not already exist.

## 5. Docs First For Unfamiliar Tech

- If the integration involves unfamiliar or fast-moving technology, gather authoritative docs before planning implementation.
- For new features or external concepts, search GitHub for strong OSS repositories and search official documentation with multiple keyword variants before coding.
- Do not rely on model memory for APIs, payment gateways, vendor SDKs, or newly released libraries.
- Convert repo/doc findings into a research note and spec before writing code.
- If a repository is not clearly compatible with commercial use, take only the concept, store it in the research note, and implement it in repository style rather than copying code.
- Start new research notes from `[[RESEARCH_TEMPLATE]]` unless the feature needs a deliberately different structure.

## 6. Debug Before Guessing

- If the agent starts looping, stop regenerating guesses.
- Reproduce the issue.
- Add targeted logging or debug instrumentation.
- Capture the failing input, output, and error path.
- Write or tighten a failing test before attempting another fix when practical.

## 7. Tests Are Part Of Done

- Every implementation step ends with validation.
- For code changes, write and run edge-case tests, not just happy-path checks.
- For workflow or documentation changes, validate the files and leave a checkpoint so future sessions can resume cleanly.
- Research artifacts come before implementation artifacts for new ideas; do not reverse that order.

## 8. Keep Sessions Clean

- Start a fresh chat after a feature or major sub-phase completes.
- Use checkpoint files and task files as the handoff artifacts.
- Do not rely on long chat history as the source of truth.

## 9. Commit Regularly

- Commit after meaningful, validated increments.
- Do not batch unrelated changes into one commit.
- Preserve the ability to roll back to a known-good step.

## 10. Review Before Accepting

- Working code is not enough.
- Check scope, naming, failure handling, tests, and architectural fit before treating a step as complete.