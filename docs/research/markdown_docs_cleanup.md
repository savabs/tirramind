---
title: "Feature: Markdown Docs Cleanup"
tags:
  - doc/research
  - phase/29
  - topic/documentation
  - layer/llm-support
---

# Feature: Markdown Docs Cleanup

## Goal

Remove redundant or outdated information from the highest-traffic markdown files so the vault's current state matches Phase 29 completion and the latest checkpoint trail.

## Search Log

- No external search required. This is an internal documentation consistency pass.
- Evidence gathered from `[[chat_checkpoint_2026-04-16_phase29_complete]]`, `[[project_memory]]`, `[[README]]`, `[[current_phases]]`, `[[system_overview]]`, `[[execution_engines]]`, and `[[quant_training_ground]]`.

## Current Architecture

- `[[README]]` is the public-facing entry point and currently still reports older counts for tools and tests.
- `[[current_phases]]` is the wiki roadmap summary and still points to the 2026-04-10 checkpoint and an active Phase 17 state.
- `[[system_overview]]` and `[[execution_engines]]` are short architecture pages that still cite the same outdated checkpoint metadata.
- `[[quant_training_ground]]` is the long-running roadmap task and still marks Phases 28 and 29 as incomplete even though the latest checkpoint shows both are complete.

## Observations

- `[[README]]` says "47+ data tools" and "3000+ edge-case tests", while `[[project_memory]]` and `[[chat_checkpoint_2026-04-16_phase29_complete]]` indicate materially newer counts.
- `[[current_phases]]` is stale in both metadata and narrative: `updated_on: 2026-04-10`, source checkpoint still anchored to the 2026-04-10 snapshot, and Phase 17 shown as active.
- `[[system_overview]]` and `[[execution_engines]]` are not deeply wrong in content, but their metadata still anchors them to the older checkpoint and update date.
- `[[quant_training_ground]]` has roadmap drift: its "Current phase" line and unchecked Phases 28-29 conflict with the phase-complete checkpoint trail.

## Risks

- Counts in `[[README]]` should avoid overclaiming when the latest checkpoint reflects a known pre-existing failing test elsewhere. Prefer conservative phrasing where exact global counts are unstable.
- `[[quant_training_ground]]` is a large historical file. The cleanup should be minimal and limited to clearly stale phase-state markers rather than broad rewriting.
- Wiki metadata should be updated consistently so related pages do not disagree about the same project snapshot.

## Cleanup Scope

1. Update `[[README]]` to remove obsolete tool/test counts and reflect the current completed-phase horizon more conservatively.
2. Refresh `[[current_phases]]` to Phase 29 state and point metadata at `[[chat_checkpoint_2026-04-16_phase29_complete]]`.
3. Refresh `[[system_overview]]` and `[[execution_engines]]` metadata to the same checkpoint.
4. Update `[[quant_training_ground]]` to mark Phases 28 and 29 complete and advance the current roadmap state.

## Related

- [[markdown_docs_cleanup_spec]]
- [[markdown_docs_cleanup]]
- [[README]]
- [[current_phases]]
- [[system_overview]]
- [[execution_engines]]
- [[quant_training_ground]]
- [[chat_checkpoint_2026-04-16_phase29_complete]]