---
title: "Feature: Obsidian Lint Cleanup"
tags:
  - doc/research
  - phase/29
  - topic/documentation
  - topic/obsidian
  - layer/llm-support
---

# Feature: Obsidian Lint Cleanup

## Goal

Resolve the remaining low-risk Obsidian lint failures that are caused by missing frontmatter and stale markdown wiki links, without rewriting historical technical content.

## Search Log

- No external search required. This is a repository-internal documentation repair pass.
- Sources inspected: `[[chat_checkpoint_2026-04-13_session2]]`, `[[efficient_operations_guide]]`, `[[chat_checkpoint_2026-04-10_session2]]`, `[[entity_linking_layer]]`, `[[chat_checkpoint_2025-07-24_tier8_complete]]`, `[[chat_checkpoint_2026-04-15_tier6_complete]]`, `[[chat_checkpoint_2026-04-16_extensive_phase29_handoff]]`, `[[ais_vessel_l2]]`, `[[whale_alert_l2]]`, `[[gnn_guided_tool_expansion]]`.

## Current Architecture

- `python scripts/obsidian_lint.py` currently reports two FM01 failures and a group of LK01 failures concentrated in historical checkpoints, research docs, and spec docs.
- The remaining lint issues are primarily documentation graph integrity issues, not content-quality issues.
- Several broken links point to aliases or task filenames that were never created or were renamed later.

## Observations

- `[[chat_checkpoint_2026-04-13_session2]]` is empty and only needs valid frontmatter plus a minimal body to satisfy the vault rules.
- `efficient_operations_guide.md` has substantive content but no frontmatter.
- Some broken links have obvious replacements, such as outdated tier checkpoint names.
- Other broken links refer to non-existent shorthand targets like an `integration` alias or old `*_task` page names; these should be replaced with valid existing docs or downgraded to plain text when no Obsidian page exists.

## Risks

- Historical checkpoint files should be repaired conservatively; changing factual narratives is out of scope.
- Some targets do not have an exact replacement. In those cases, remove the broken wiki link rather than inventing a document.
- Orphan and stale-length warnings are advisory and should not drive unnecessary edits in this pass.

## Cleanup Scope

1. Add frontmatter to the two FM01 files.
2. Repair or remove the LK01 broken links that have clear, safe fixes.
3. Re-run Obsidian lint and verify that only pre-existing advisory warnings remain, or document any remaining blockers.

## Related

- [[obsidian_lint_cleanup_spec]]
- [[obsidian_lint_cleanup]]
- [[markdown_docs_cleanup]]
- [[project_memory]]
- [[chat_checkpoint_2026-04-16_phase29_complete]]