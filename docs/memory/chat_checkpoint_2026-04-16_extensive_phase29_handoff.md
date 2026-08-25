---
title: "Checkpoint: Extensive Phase 29 Handoff"
tags:
  - doc/checkpoint
  - phase/29
  - topic/entity-linking
  - topic/bankruptcy
  - topic/foia
  - topic/academic-preprints
  - topic/gnn
  - topic/world-model
  - layer/surveillance
  - layer/feature-engineering
  - layer/world-model
---

# Checkpoint: Extensive Phase 29 Handoff

## Executive Summary

This checkpoint captures the full end-to-end state after completing **Phase 29: Company + Investigative L2** and then cleaning up the immediate post-phase repository state.

Phase 29 is functionally complete. The core implementation work landed in three commits and the phase-level checkpoint/task updates were also committed. The repo now contains the new company/topic investigative L2 persistence paths, expanded graph builder observation support, refreshed diagnostics, and an explicit handoff point for deciding the next phase.

The short version:

- Phase 29 implementation is complete.
- The main code changes are committed.
- The phase task was marked complete and a short checkpoint was written.
- Three stale completed task files were moved out of `tasks/active/` into `tasks/done/`.
- The current active tasks are now reduced to:
  - `phase26_mcp_agent_upgrade`
  - `quant_training_ground`
- The current strategic decision is whether to continue with more L2 upgrades, run a GNN-guided diagnostic selection pass, or begin the next major capability layer around world modeling / signal fusion refinement.

There is still one small piece of repository housekeeping left in the working tree at the moment this checkpoint is being written:

- `[[phase29_company_investigative_l2]]` shows as deleted in git status because it was committed in `tasks/active/` and then physically moved to `tasks/done/` afterward without a follow-up git commit to record the move.

That is not a functional problem, but it is the main leftover state to clean up before declaring the repo fully tidy again.

## Session Intent And What Actually Happened

The session effectively covered four logical objectives:

1. Finish and verify the remaining Phase 29 implementation work.
2. Commit the Phase 29 code in logical chunks.
3. Write a checkpoint and mark the task complete.
4. Decide what the next meaningful phase should be.

That work was completed in the following order:

1. The remaining Phase 29 work was finished.
2. L2 and diagnostic tests were run and passed.
3. Regression coverage was checked.
4. The phase output was committed in three commits.
5. The phase task was marked complete.
6. The active task list was reduced by moving already-completed tasks to `tasks/done/`.
7. A brief checkpoint was already written.
8. This long-form checkpoint is now being added as the fuller handoff artifact.

## Phase 29 Scope

Phase 29 extended the graph/entity pipeline so that three previously L1-style investigative/company tools now write **L2 entity observations** into the PipelineStore / entity graph path.

The three tool upgrades were:

1. `bankruptcy_court`
2. `foia_requests`
3. `academic_preprints`

These upgrades introduced three new observation types into the graph builder surface:

- `bankruptcy_status`
- `investigation_signal`
- `research_velocity`

This increased:

- `OBSERVATION_TYPES`: 32 -> 35
- `ENRICHMENT_DIM`: 41 -> 44

Those numbers matter because downstream graph feature construction depends directly on the observation type distribution width.

## Detailed Implementation Summary

### 1. bankruptcy_court L2 Upgrade

File touched:

- `agent/tools/bankruptcy_court.py`

What changed:

- Added the standard L2 persistence import pattern:
  - `time`
  - `TYPE_CHECKING`
  - `PipelineStore` type-only import
  - guarded import of `_entity_id_from_key`
- Updated the constructor to accept a `pipeline_store` and retain it on `self._store`
- Refactored `execute()` to capture the mode result in a `result` variable and then call `_persist_entities(result.data, mode)` after a successful tool response
- Added `_persist_entities()` outer wrapper:
  - no-op if store is absent
  - no-op if entity-id helper is absent
  - non-fatal try/except wrapper for persistence failures
- Added `_persist_entities_inner()` containing the actual L2 persistence logic

Entity mapping behavior:

- `us_bankruptcy` -> company entity from debtor name
- `sec_enforcement` -> company entity from title
- `sec_bankruptcy` -> company entity from `company_name` when not `Unknown`
- `uk_insolvency` -> company entity from title-equivalent record field

Observation written:

- observation type: `bankruptcy_status`
- depth level: `2`

Important normalization behavior:

- Common legal prefix stripping for names such as `In re:` so the canonical company node is cleaner and more stable.

Verification performed:

- 23 L2-specific tests added and passed.

### 2. foia_requests L2 Upgrade

File touched:

- `agent/tools/foia_requests.py`

What changed:

- Added the same standard L2 persistence infrastructure as above.
- Updated constructor for `pipeline_store`.
- Refactored `execute()` to route through a single result variable and persist after success.
- Added `_persist_entities()` and `_persist_entities_inner()`.
- Also fixed a structural problem in three tool returns by adding missing `data=` payloads to the `ToolResult` objects.

That `data=` fix was necessary because L2 persistence depends on the result payload being available after the tool call. Without that, the persistence hook would have nothing structured to read.

Entity mapping behavior:

- `search` -> company entity from FOIA request title
- `entity_cluster` -> company entity from cluster title
- `agency_activity` -> company-like node keyed by agency name

Observation written:

- observation type: `investigation_signal`
- depth level: `2`

Verification performed:

- 14 L2-specific tests added and passed.

### 3. academic_preprints L2 Upgrade

File touched:

- `agent/tools/academic_preprints.py`

What changed:

- Added the standard L2 persistence imports and `pipeline_store` constructor wiring.
- Refactored `execute()` to return through a single `result` object and persist after a successful response.
- Added `_persist_entities()` and `_persist_entities_inner()`.

Entity mapping behavior was more interesting here because this tool spans two entity classes depending on mode:

- `trials` mode -> company entity via clinical-trial sponsor
- `papers` mode -> topic entity via first arXiv category
- `trending` mode -> topic entity via first arXiv category

Observation written:

- observation type: `research_velocity`
- depth level: `2`

Why this matters:

- `trials` gives an entity-level company research / pipeline signal.
- `papers` and `trending` give topic-level momentum on scientific / technical areas.

This is a useful structural addition because it introduces topic-node enrichment in a way that is not just a link, but an explicit observation stream.

Verification performed:

- 21 L2-specific tests added and passed.

## Graph Builder Changes

Files touched:

- `agent/models/gnn/graph_builder.py`
- `tests/test_graph_builder_expanded.py`
- `tests/test_phase28_diagnostic.py`
- `tests/test_phase29_diagnostic.py`

### Observation Registry Changes

Three new observation types were inserted into `OBSERVATION_TYPES` in sorted order:

- `bankruptcy_status`
- `investigation_signal`
- `research_velocity`

An intermediate test failure caught an ordering mistake around `investigation_signal` vs the `instrument_*` entries. That was fixed immediately after the failing alphabetical-order test surfaced it.

### Enrichment Dimension Changes

The graph enrichment feature dimension was updated from 41 to 44.

Rationale:

- The enrichment block contains a distribution over observation types.
- That vector grew from length 32 to length 35.
- Therefore total enrichment dimension grew by 3.

### Test Updates

`tests/test_graph_builder_expanded.py` was updated to:

- assert the three Phase 29 observation types exist
- assert the observation types remain sorted
- assert the total count is now 35

`tests/test_phase28_diagnostic.py` was updated to fix a stale expectation:

- `ENRICHMENT_DIM == 41` -> `ENRICHMENT_DIM == 44`

## New Diagnostic Coverage

New file created:

- `tests/test_phase29_diagnostic.py`

This file added 18 integration-style diagnostics covering the full path from persisted observation to graph-builder visibility.

The diagnostics include:

- registration checks for the new observation types
- single-tool persistence for each new observation family
- multi-tool enrichment on the same company node
- cross-entity-type coexistence for company + topic research velocity
- coexistence of Phase 29 company/topic observations with prior country-level macro observations
- deterministic entity ID behavior for company and topic nodes

This matters because it verifies the new work at the graph-surface level, not only at the individual tool unit-test level.

## Phase 29 Test Totals

Specific test counts added or updated:

- `tests/test_bankruptcy_court_edge.py` -> 23 L2 tests
- `tests/test_foia_requests_edge.py` -> 14 L2 tests
- `tests/test_academic_preprints_edge.py` -> 21 L2 tests
- `tests/test_phase29_diagnostic.py` -> 18 integration tests

High-level grouped verification that was run:

- 21/21 `academic_preprints` L2 tests passed
- 25/25 graph builder expanded tests passed after sort-order fix
- 57/57 grouped graph/phase28/phase29 tests passed

Regression summary from full repo run:

- 4483 passed
- 1 failed
- 3 warnings

The one failure is pre-existing and unrelated to Phase 29.

## Known Test Failures / Pre-Existing Issues

### 1. `tests/test_form144_edge.py::TestXMLParser::test_ross_stores_xml`

Observed failure:

- expected `shares_to_sell == 4454`
- actual `shares_to_sell == 4154`

This was confirmed directly after the regression run and is not related to the Phase 29 implementation.

### 2. `tests/test_feature_generation_dag.py`

There is also a pre-existing stale-count issue referenced during regression planning:

- stale feature count mismatch: `17 vs 6`

That test was excluded from a targeted regression path earlier in the session because it was already known to be unrelated noise.

### 3. Obsidian Lint

Running `python scripts/obsidian_lint.py` still fails, but the reported issues are pre-existing vault hygiene issues rather than new Phase 29 problems.

The reported categories include:

- broken wiki links
- orphan pages
- stale / oversized markdown files

Examples from the output include broken links to older checkpoints and missing shorthand page aliases, along with some older checkpoint references.

No evidence surfaced that the new Phase 29 checkpoint or task files introduced fresh lint breakage beyond the repo's existing background state.

## Commits Created During This Work

The key recent commits at the top of `main` are:

1. `e245576` — `phase29: add L2 entity persistence to bankruptcy_court, foia_requests, academic_preprints`
2. `d31f9fd` — `phase29: OBSERVATION_TYPES 32→35, ENRICHMENT_DIM 41→44, diagnostic tests`
3. `d9f3242` — `phase29: mark task complete, write checkpoint`

Immediately before those are earlier commits from the prior broader working-tree cleanup and workflow work:

4. `cb1c447` — workflow automation scripts
5. `c204586` — pipeline DAG / executor / CLI updates

## Task / Workflow State

### Phase 29 Task

The Phase 29 task file was:

- marked complete
- updated from `status/active` to `status/done`
- committed in its active-path version
- then physically moved into `tasks/done/`

Important nuance:

The move happened after the commit. That means the file move itself is currently not yet recorded in git history. Git status shows:

- `D [[phase29_company_investigative_l2]]`

At the time of writing this checkpoint, that is the main leftover working-tree change.

### Active Tasks After Cleanup

The active task list was cleaned up so it now contains only:

- `phase26_mcp_agent_upgrade.md`
- `quant_training_ground.md`

Three completed task files that were still sitting in `tasks/active/` were physically moved to `tasks/done/`:

- `ecc_workflow_improvements.md`
- `phase27_fx_country_monetary_linking.md`
- `predictive_platform_positioning_task.md`

This was the right cleanup move because all three already had `Status: completed` while still polluting the active-task directory.

### quant_training_ground Is Stale

`[[quant_training_ground]]` is now stale relative to reality.

The file still says:

- current phase: Phase 27
- Phase 27 not completed
- Phase 28 not completed
- Phase 29 not completed

That is no longer accurate.

In reality:

- Phase 27 is done
- Phase 28 is done
- Phase 29 is done

So one of the next repository-maintenance steps should be to refresh `quant_training_ground.md` so it reflects the actual completed-phase frontier.

## Current Repository State At Time Of Checkpoint

`git status --short` at checkpoint time showed:

- `D [[phase29_company_investigative_l2]]`

That means the code and tests are committed, but the task move cleanup still needs either:

- a follow-up commit that records the rename/move properly
- or a decision to revert the physical move if the project wants task files to remain in `tasks/active/` until a larger housekeeping batch is committed

There were no uncommitted Phase 29 code-file modifications showing in status at the moment this long checkpoint was gathered. The main residue is task-file bookkeeping.

## Strategic Interpretation Of Phase 29

Phase 29 meaningfully improved the entity graph in two ways.

### 1. Company-Side Investigative Surface Improved

The graph now captures company-level pressure / activity from:

- bankruptcy and court-related distress
- FOIA/investigative request activity
- clinical-trial sponsor activity

That broadens the company observation surface beyond purely market / filing / contract style signals.

### 2. Topic Nodes Now Receive Better Native Observation Streams

`academic_preprints` contributing `research_velocity` to topic nodes is important because it gives topic entities a more direct observation history rather than relying only on links or derived co-occurrence logic.

This should make topic nodes more useful in later graph reasoning and downstream world-model or signal-fusion logic.

## L2 Coverage Position After Phase 29

By the end of Phase 29, the project now has a materially broader L2 surface. The rough project interpretation from the repo exploration was:

- approximately 16 tools now have L2 persistence paths
- many others remain at L1 / aggregate-only depth

This means the repo is no longer in an early proof-of-concept state for L2. It has enough entity-resolved surface area that the next step should probably be decided strategically rather than mechanically.

That strategic choice is important because there are two very different paths from here:

1. continue expanding more tools into L2
2. start exploiting the accumulated graph/entity surface more aggressively at higher layers

## Next-Phase Decision Surface

The next major decision is not just "what file to touch next," but what layer of the stack should receive the next serious investment.

### Option A: Continue L2 Expansion

This path means continuing Phase 30+ style upgrades across more tools.

Pros:

- expands entity coverage
- improves graph density
- likely still low-risk from an implementation standpoint
- follows the successful established pattern from Phases 10b, 13, 18, 23, 28, and 29

Cons:

- risks turning into checklist-driven breadth rather than signal-guided expansion
- can delay exploitation of the graph surface that already exists
- may keep pushing higher-value inference work further out

### Option B: GNN-Guided Diagnostic Pass Before More L2

This path means pausing blind expansion long enough to measure where the graph is actually sparse or starved.

Pros:

- matches the doctrine in the repo instructions: expand based on starved neighborhoods, not coverage vanity
- could identify the 1–3 most valuable next L2 upgrades rather than 5–10 mediocre ones
- gives evidence to prioritize company/country/topic/protocol/wallet/instrument neighborhoods

Cons:

- not as visible as shipping more tools
- requires diagnostic discipline and interpretation work

### Option C: Shift Toward Layer 3 / Layer 4 Capability

This path means moving toward the next major capability jump:

- world model
- signal fusion refinement
- probabilistic inference

Pros:

- aligns with the architecture priority that the math and probabilistic engine are the actual product
- starts converting the now-richer observation graph into inference rather than only storage
- likely creates more product-grade predictive outputs than adding another small batch of L2 tools

Cons:

- bigger design surface
- requires cleaner specs and likely updated research artifacts
- may expose stale assumptions in older world-model planning docs

## My Best Current Read On The Right Next Move

The strongest next move is not blind Phase 30 implementation by default.

The repo now has enough L2 coverage that a short diagnostic selection phase is probably the highest-signal step before more tool work. In other words:

1. clean up the remaining repo/task bookkeeping
2. refresh `quant_training_ground.md`
3. run a GNN-guided expansion diagnostic / starved-neighborhood pass
4. use that output to decide whether Phase 30 should be:
   - crypto linking expansion
   - country-signal completion
   - or a shift toward world-model exploitation

That avoids overfitting the roadmap to whatever phase number comes next in sequence.

## Quant Training Ground Drift

`quant_training_ground.md` currently encodes a roadmap that is partially stale relative to actual completed work.

This is not just clerical. It matters because this file acts like the top-level navigation artifact for the long-range roadmap. If it is stale, future sessions may cold-start into the wrong phase assumptions.

Specific drift to correct later:

- mark Phase 27 complete
- mark Phase 28 complete
- mark Phase 29 complete
- update current-phase text
- reflect the current frontier more honestly

Potentially also update the phrasing around whether the next phase is truly Phase 30 as written, or whether a diagnostic sub-phase should intervene first.

## Active Phase 26 Status

`phase26_mcp_agent_upgrade.md` remains active.

Its implemented steps are complete for the main MCP stack setup and verification. The remaining open items in that task are future-oriented custom TirraMind MCP server designs:

- entity graph query MCP
- pipeline state / cached data MCP
- backtest runner MCP

That means Phase 26 is not blocking current quant work, but it remains an active infrastructure task with future leverage.

## What A Fresh Session Should Do First

If someone resumes from this checkpoint in a new session, the best opening sequence is:

1. Read this checkpoint.
2. Read the short checkpoint `[[chat_checkpoint_2026-04-16_phase29_complete]]` for the compressed implementation summary.
3. Check `git status` and decide whether to commit the outstanding Phase 29 task-file move.
4. Refresh `quant_training_ground.md` to reflect Phases 27–29 as complete.
5. Decide whether the next step is:
   - repo cleanup only
   - diagnostic pass
   - next L2 phase
   - world-model shift

## Recommended Immediate Housekeeping

These are the most sensible next housekeeping items, in order:

1. Record the move of `phase29_company_investigative_l2.md` into `tasks/done/` with a small follow-up commit.
2. Update `quant_training_ground.md` so it matches actual project state.
3. Optionally re-run `obsidian_lint.py` after task-file and roadmap updates, but expect the older vault issues to remain unless a dedicated documentation cleanup pass is done.

## Recommended Immediate Technical Next Step

If the goal is to maximize strategic signal rather than just close paperwork, the next technical step should be:

- a focused GNN-guided starved-neighborhood diagnostic pass

Why:

- L2 coverage is now broad enough that expansion should be selected, not guessed.
- The repo doctrine explicitly favors GNN-guided prioritization rather than blanket upgrading everything.
- This would produce a defensible Phase 30 target instead of treating roadmap ordering as destiny.

## Concrete Facts To Preserve For Future Sessions

- Phase 29 is implemented and committed.
- Three new observation types were added: `bankruptcy_status`, `investigation_signal`, `research_velocity`.
- `OBSERVATION_TYPES` is now 35.
- `ENRICHMENT_DIM` is now 44.
- `academic_preprints` now persists to both company and topic entities depending on mode.
- `foia_requests` required `data=` payload fixes in `ToolResult`, not just persistence hooks.
- The only regression failure observed in the fresh full run was the pre-existing Form 144 XML shares mismatch.
- The active task directory has been cleaned down to two files.
- `quant_training_ground.md` is stale and should be updated soon.
- The Phase 29 task move is not yet reflected in git history.

## Final State Snapshot

At the moment this checkpoint is written, the codebase is in a good functional state for the completed phase, with the following caveat pattern:

- implementation: complete
- tests for phase work: green
- full regression: green except one known unrelated failure
- documentation/checkpoint: present
- roadmap artifact: stale
- task move cleanup: partially done in filesystem, not yet committed as git history

This is a strong handoff point. The only real risk is losing time in the next session to stale task/roadmap metadata rather than to technical blockers.

## Related

- [[chat_checkpoint_2026-04-16_phase29_complete]]
- [[phase29_company_investigative_l2]]
- [[phase29_company_investigative_l2_spec]]
- [[quant_training_ground]]
- [[phase26_mcp_agent_upgrade]]