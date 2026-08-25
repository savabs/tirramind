---
title: "Spec: Markdown Docs Cleanup"
tags:
  - doc/spec
  - phase/29
  - topic/documentation
  - layer/llm-support
---

# Spec: Markdown Docs Cleanup

## Goal

Bring the most visible markdown files back into alignment with the current project state by removing stale phase references, outdated counts, and redundant wording that no longer adds value.

## Files Affected

- `README.md`
- `wiki/pages/roadmap/current_phases.md`
- `wiki/pages/architecture/system_overview.md`
- `wiki/pages/architecture/execution_engines.md`
- `[[quant_training_ground]]`

## Implementation Steps

1. Update `README.md` to replace obsolete tool/test counts with current, conservative wording and tighten any now-redundant phrasing.
2. Update `wiki/pages/roadmap/current_phases.md` metadata and narrative to reflect the Phase 29-complete state.
3. Update `wiki/pages/architecture/system_overview.md` and `wiki/pages/architecture/execution_engines.md` metadata so they cite the latest checkpoint instead of the 2026-04-10 snapshot.
4. Update `[[quant_training_ground]]` so its current-phase line and phase checklist no longer contradict completed Phase 28 and 29 work.
5. Validate by searching for the old stale markers and running the markdown lint/health check that is appropriate for the vault.

## Edge Cases

- Avoid introducing exact counts that are contradicted by active pre-existing failures or partial regressions.
- Preserve existing wiki links, frontmatter structure, and document roles.
- Do not rewrite historical sections that are descriptive rather than stale.

## Testing Plan

- Search for the removed stale strings (`47+ data tools`, `3000+ edge-case tests`, `chat_checkpoint_2026-04-10`, `updated_on: 2026-04-10`, `Phase 17 (Entity Linking Layer) is the active infrastructure step`).
- Run `python scripts/obsidian_lint.py` and review whether the edited files introduce frontmatter or link issues.

## Related

- [[markdown_docs_cleanup]]
- [[quant_training_ground]]
- [[README]]
- [[current_phases]]
- [[system_overview]]
- [[execution_engines]]