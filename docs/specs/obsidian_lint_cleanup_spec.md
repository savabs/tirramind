---
title: "Spec: Obsidian Lint Cleanup"
tags:
  - doc/spec
  - phase/29
  - topic/documentation
  - topic/obsidian
  - layer/llm-support
---

# Spec: Obsidian Lint Cleanup

## Goal

Clear the remaining safe Obsidian lint failures by fixing missing frontmatter and stale wiki links in markdown files.

## Files Affected

- `[[chat_checkpoint_2026-04-13_session2]]`
- `efficient_operations_guide.md`
- Historical checkpoint, research, spec, and task markdown files referenced by current LK01 failures

## Implementation Steps

1. Add valid YAML frontmatter and a minimal placeholder body to `[[chat_checkpoint_2026-04-13_session2]]`.
2. Add valid YAML frontmatter to `efficient_operations_guide.md`.
3. Repair LK01 references where an existing target is obvious.
4. Replace broken shorthand wiki links with plain text or existing valid targets where no matching page exists.
5. Re-run `python scripts/obsidian_lint.py` and verify the resulting failure set.

## Edge Cases

- Preserve historical content; avoid rewriting narratives beyond link repair.
- Do not create speculative placeholder pages solely to satisfy one broken link.
- Keep all new or edited markdown files compliant with frontmatter and `## Related` conventions when applicable.

## Testing Plan

- Run `python scripts/obsidian_lint.py`.
- Confirm FM01 is cleared for the two targeted files.
- Confirm LK01 is reduced or cleared for the files edited in this pass.

## Related

- [[obsidian_lint_cleanup]]
- [[markdown_docs_cleanup]]
- [[project_memory]]