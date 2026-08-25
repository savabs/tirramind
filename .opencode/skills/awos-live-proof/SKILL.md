---
name: awos-live-proof
description: Protocol for creating runnable end-to-end demos as proof of kernel/CLI/safety features
license: MIT
---

# AWOS Live Proof Protocol

From AWOS.md §2.6 and `protocols/LIVE_PROOF_PROTOCOL.md`.

## When required
Live proof is mandatory for:
- Orchestrator changes
- Session management features
- CLI features
- Safety/consent features
- Any feature marked "requires live proof" in a task file

## What to produce
1. **Runnable demo** — a single command that exercises the feature end-to-end
2. **Observable markers** — stdout markers, log entries, state file changes the user can verify
3. **Proof doc** — a markdown file at `docs/live_proof/<feature-name>.md` containing:
   - What the feature does
   - The demo command to run
   - Expected output / observable markers
   - A screenshot or terminal transcript

## Verification checklist
- [ ] Demo command runs without errors
- [ ] Observable markers appear in output
- [ ] No side effects break the existing test suite
- [ ] Proof doc is committed

## Anti-patterns
- Do NOT mark a feature done with only unit tests if it falls under live proof requirements
- Do NOT skip the proof doc
- Do NOT use mocked internal state as the only proof
