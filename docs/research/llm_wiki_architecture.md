---
title: "Feature: llm_wiki_architecture"
tags:
  - doc/research
  - topic/wiki
---

# Feature: llm_wiki_architecture

Use this research note to apply a persistent LLM-maintained wiki architecture to TirraMind's project knowledge.

## Goal
- Create a compiled knowledge layer that sits between raw project artifacts and future LLM sessions.
- Reduce repeated rediscovery across research docs, specs, task files, checkpoints, and architecture notes.
- Preserve the repo's existing workflow while adding a persistent, cross-linked wiki the LLM can maintain.

## Search Log
- GitHub keywords searched:
  - none directly via repo search; the user-provided idea is the primary concept source for the wiki pattern
- Documentation keywords searched:
  - Obsidian Web Clipper
  - Obsidian Dataview metadata indexing
  - Marp markdown slides
- Other search surfaces used:
  - `README.md`
  - `autonomous_agent_idea.md`
  - `[[project_memory]]`
  - `[[workflow_templates]]`
  - `[[quant_training_ground]]`

## External Repositories Reviewed
- Repository:
  - user-provided LLM wiki concept note (concept source, not code)
  - Why it is relevant:
    - defines the target operating model: raw sources + persistent wiki + schema/instructions
  - Useful implementation idea:
    - keep `index.md` content-oriented and `log.md` chronological
    - file valuable query outputs back into the wiki as persistent pages
    - separate immutable sources from LLM-maintained synthesis
  - License:
    - not a code repository; concept only
  - Reuse conclusion: concept only

## Documentation Reviewed
- Source title / URL:
  - Obsidian Dataview documentation — https://blacksmithgu.github.io/obsidian-dataview/
  - What it clarified:
    - frontmatter and inline metadata are enough to support dynamic note indexes and health queries
  - API or concept details to carry forward:
    - standardize YAML frontmatter fields on wiki pages
    - treat Dataview as optional display/query tooling, not a dependency for correctness

- Source title / URL:
  - Obsidian Web Clipper help — https://obsidian.md/help/Extending+Obsidian/Obsidian+Web+Clipper
  - What it clarified:
    - web clipping is an ingestion convenience, not a core architectural requirement
  - API or concept details to carry forward:
    - keep source ingestion format markdown-friendly, but avoid making browser tooling mandatory

- Source title / URL:
  - Marp — https://marp.app/
  - What it clarified:
    - slide decks can be generated from markdown pages later without changing the core wiki structure
  - API or concept details to carry forward:
    - treat slide generation as an optional query/output mode, not part of the initial slice

## Current Architecture
- Relevant local modules:
  - `docs/research/`, `docs/specs/`, `tasks/active/`, `docs/memory/` already hold the raw and semi-structured knowledge the agent repeatedly consults
  - `README.md` and `[[project_memory]]` are high-value architectural summaries
  - `agent/workflow_guard.py` already enforces research/spec/task preflight for non-trivial work
- Existing patterns to preserve:
  - markdown-first workflow
  - append-only checkpoint history
  - deterministic tooling where possible
  - no dependence on external SaaS or paid infrastructure
- Correct insertion points:
  - top-level `wiki/` directory for the compiled knowledge layer
  - `agent/wiki/` for deterministic indexing/lint support
  - `tests/` for focused edge-case validation of wiki tooling

## Observations
- What already exists:
  - the repo already has high-signal source material, but it is fragmented across research notes, specs, tasks, checkpoints, and memory files
  - the current memory system is strong for process continuity but weak for topic-centric synthesis
  - there is no stable `index.md` or `log.md` for project knowledge outside of ad hoc docs browsing
- What is missing:
  - a persistent knowledge layer organized by topic/entity/concept rather than by workflow phase
  - a schema that tells the LLM how to maintain that layer consistently
  - deterministic health checks for missing frontmatter, orphan pages, broken links, and stale index entries
- Important constraints:
  - do not replace the existing research/spec/task workflow
  - avoid heavy dependencies for markdown parsing or search in the first slice
  - keep raw sources immutable; the new wiki should synthesize from them, not mutate them

## Risks
- Licensing or reuse risks:
  - the user note is a concept source only; no external code should be ported
- Technical risks:
  - duplicating the existing docs hierarchy instead of compiling it into a clearer synthesis layer
  - creating a wiki that drifts because index maintenance is manual
  - choosing metadata fields that are too vague to lint deterministically
- Testing risks:
  - wiki tooling can silently accept malformed frontmatter or broken links unless explicit edge-case tests are added

## Data Requirements
- Required inputs or sources:
  - existing project memory, README, active tasks, research docs, specs, and checkpoints
- What already exists locally:
  - all initial source material needed to seed the wiki
- What still needs to be added:
  - `wiki/index.md`
  - `wiki/log.md`
  - `wiki/SCHEMA.md`
  - initial wiki pages with standardized frontmatter
  - deterministic catalog/lint tool

## Math/Algorithm Survey
- Candidate approaches:
  - pure markdown + manual index maintenance
  - markdown + deterministic filesystem scan + frontmatter validation
  - embedding/BM25 search stack from day one
- Why one approach is preferred:
  - markdown + deterministic scan is the smallest high-signal toolset: cheap, local, testable, and enough for the repo's current scale
  - embedding search is useful later but is not required to prove the architecture
- Complexity or dependency notes:
  - frontmatter parsing can be implemented with a small stdlib parser for a constrained schema
  - broken link and orphan checks are linear in page count and trivial at current scale

## Implementation Intent
- Concepts approved for implementation:
  - a top-level compiled knowledge wiki
  - strict page frontmatter conventions
  - `index.md` as content catalog and `log.md` as append-only chronology
  - a deterministic tool that rebuilds the index and reports lint findings
  - initial seed pages that synthesize current architecture and roadmap
- Concepts rejected:
  - Obsidian-only assumptions
  - mandatory Dataview/Marp/qmd dependencies
  - moving or mutating existing research/spec/task docs
- Notes for the spec:
  - first slice should prove the architecture with minimal moving parts
  - implementation should make later ingestion/query workflows easier without committing to a full autonomous ingestion agent yet

---

## Related

- [[llm_wiki_architecture_spec|Spec: Llm Wiki Architecture]]
