---
title: "Spec: workflow_gate_hardening"
tags:
  - doc/spec
  - topic/workflow
---

# Spec: workflow_gate_hardening

## Goal
Add a fail-closed workflow gate so non-trivial requests must produce research, spec, and task artifacts before implementation begins.

## Files Affected
- `.github/copilot-instructions.md`
- `AGENTS.md`
- `RulesForAI.md`
- `[[workflow_gate_hardening]]`
- `[[workflow_gate_hardening_spec]]`
- `[[workflow_gate_hardening]]`
- `[[checkpoint_archive_2026]]`

## Implementation Steps
1. Create `[[workflow_gate_hardening]]` documenting the existing workflow gap and the intended fail-closed gate.
2. Create `[[workflow_gate_hardening_spec]]` defining the exact gate behavior and the narrow trivial-task exception.
3. Create `[[workflow_gate_hardening]]` with atomic implementation and validation steps.
4. Update `.github/copilot-instructions.md` to add a mandatory workflow preflight section that:
   - defines what counts as non-trivial by default
   - requires research, spec, and task artifacts before implementation
   - restricts pre-gate edits to research/spec/task/checkpoint files only
   - requires implementation-mode work to reference the governing spec/task
5. Update `AGENTS.md` to mirror the hard-gate behavior in concise agent-default language.
6. Update `RulesForAI.md` to add a short preflight checklist and fail-closed wording.
7. Validate the changed markdown files for consistency and write a checkpoint summarizing the hardening change.

## Edge Cases
- Trivial edits should remain possible without full workflow overhead.
- The hard gate must not conflict with requests that are explicitly brainstorming-, planning-, or review-only.
- The wording should not accidentally permit implementation after only research or only a spec.
- The allowed pre-gate file scope must be narrow enough to prevent accidental production code edits.

## Testing Plan
- Validate the edited markdown files with diagnostics.
- Manually verify that `.github/copilot-instructions.md`, `AGENTS.md`, and `RulesForAI.md` all express the same gate without contradiction.
- Manually verify that the trivial-task carveout is explicit and narrow.
- Write a checkpoint in `docs/memory/` so future sessions can resume with the hardened workflow in view.

---

## Related

- [[workflow_gate_hardening|Research: Workflow Gate Hardening]]
