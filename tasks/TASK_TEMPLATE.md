---
title: "Task: <name>"
tags:
  - doc/task
  - status/active
  - phase/<N>
  - topic/<slug>
  - layer/<slug>
---

# Task: <name>

Status: active
Research: [[<name>]]
Spec: [[<name>_spec]]

## Goal

One-sentence description of what this task should accomplish.

## Scope Notes

- Layer:
- Main files expected to change:
- Non-goals:

## Steps

- [ ] 1.1: Define the first atomic step
  Verification: one-line proof or test command
- [ ] 1.2: Define the second atomic step
  Verification: one-line proof or test command
- [ ] 1.3: Define the third atomic step
  Verification: one-line proof or test command

## Completion Checklist

- [ ] Research note exists and is current
- [ ] Spec matches the actual implementation plan
- [ ] Each completed step has a verification result
- [ ] Edge-case testing was added and run for code changes
- [ ] Checkpoint written at the end of the session or sub-phase
- [ ] Frontmatter tags and `## Related` section are current

## Related

- [[<name>]]
- [[<name>_spec]]

## Notes

- Keep steps atomic: one change, one test, one proof.
- If a step description contains the word `and`, split it.
- If the task becomes too large, break it into a new task file rather than growing this one indefinitely.