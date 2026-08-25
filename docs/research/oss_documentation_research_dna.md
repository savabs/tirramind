---
title: "Feature: oss_documentation_research_dna"
tags:
  - doc/research
---

# Feature: oss_documentation_research_dna

## Goal
- Make OSS-and-documentation-first research part of the repository's permanent workflow DNA.
- Ensure new features and external concepts start from explicit GitHub/doc research, license review, and research-file capture before code is written.

## Search Log
- GitHub keywords searched: none required for this internal workflow update itself
- Documentation keywords searched: none required for this internal workflow update itself
- Other search surfaces used: repository instruction files, project memory, active tasks, and existing workflow notes

## External Repositories Reviewed
- Repository: none
	- Why it is relevant: not applicable for this repository-internal workflow change
	- Useful implementation idea: not applicable
	- License: not applicable
	- Reuse conclusion: rejected

## Documentation Reviewed
- Source title / URL: `.github/copilot-instructions.md`
	- What it clarified: the current workflow phases and the exact places where OSS/docs-first behavior should be enforced
	- API or concept details to carry forward: the research phase is the right place to require GitHub/doc search and reuse constraints
- Source title / URL: `AGENTS.md`
	- What it clarified: all agents need the same default research discipline, not just the default coding agent
	- API or concept details to carry forward: license-aware reuse rules belong in agent defaults too
- Source title / URL: `RulesForAI.md`
	- What it clarified: the tactical guide is the correct place for a short human-readable checklist version of the rule
	- API or concept details to carry forward: keep the tactical wording concise and subordinate to the main instruction file
- Source title / URL: `[[project_memory]]`
	- What it clarified: the rule should become part of persistent project identity, not just session-level behavior
	- API or concept details to carry forward: frame the rule as aligned with cost discipline and concept extraction from public knowledge

## Current Architecture
- Relevant local modules:
	- `.github/copilot-instructions.md` is the primary repository-wide instruction source for workflow, phased execution, architecture boundaries, and agent behavior.
	- `AGENTS.md` defines default agent obligations and the shared workflow expectations across agents.
	- `RulesForAI.md` is a concise human-readable operating checklist that complements, but does not override, the main instruction file.
	- `[[project_memory]]` stores persistent project identity and execution principles that should survive across sessions.
	- The existing workflow already enforces research -> spec -> implement, but it does not yet explicitly force OSS repository search, documentation search, or license-aware concept extraction before implementation.
- Existing patterns to preserve:
	- phased workflow
	- atomic task decomposition
	- project memory for durable operating principles
- Correct insertion points:
	- repo-wide instructions for enforceable behavior
	- folder-level instructions for local reinforcement
	- research template for repeatable note structure
	- checkpoint/memory files for persistence

## Observations
- What already exists:
	- The repository identity already favors cheap/free information sources, cost discipline, and learning from overlooked surfaces.
	- The workflow already enforces research -> spec -> implement.
- What is missing:
	- The missing behavior is procedural: explicitly search GitHub and official docs first, use multiple search keyword variants, and record what was found before coding.
	- A license-aware rule is needed so future sessions do not copy code from non-commercial or otherwise incompatible repositories into a commercial codebase.
	- Three folder-level instruction files exist in `agent/tools/`, `agent/quant/`, and `tests/`; they should reinforce the same rule so local module work does not drift from repo-level workflow.
	- No reusable research-note template exists in `docs/research/`, so each new feature currently has to reconstruct the same checklist manually.
- Important constraints:
	- The workflow should require that reusable concepts from external sources be converted into research notes first, then reimplemented in repository style rather than copied blindly.
	- This policy belongs in the instruction stack, not just in chat, otherwise it will decay across sessions.

## Risks
- Licensing or reuse risks:
	- If license handling is omitted, future sessions may accidentally pull code from repositories that are incompatible with commercial use.
- Technical risks:
	- If the new rule is stated vaguely, agents may claim they "researched" without actually searching GitHub or documentation.
	- If the policy is duplicated inconsistently across multiple files, the instruction stack may drift.
- Testing risks:
	- If the rule is too rigid, it could force pointless external searching for trivial local-only refactors; the wording should target new features, unfamiliar technology, and external concepts.

## Data Requirements
- Required inputs or sources:
	- existing instruction files, folder-level guidance, and project memory
- What already exists locally:
	- `.github/copilot-instructions.md`, `AGENTS.md`, `RulesForAI.md`, `[[project_memory]]`, and the folder-level `.instructions.md` files
- What still needs to be added:
	- explicit OSS/docs-first wording, local reinforcement, and a reusable research-note template

## Math/Algorithm Survey
- Candidate approaches:
	- update only the main instruction file
	- update the full instruction stack plus add a reusable research template
- Why one approach is preferred:
	- the full-stack update reduces drift and makes the rule visible both globally and locally
- Complexity or dependency notes:
	- low implementation complexity; documentation-only change

## Implementation Intent
- Concepts approved for implementation:
	- repo-wide OSS/docs-first workflow wording
	- license-aware concept extraction rule
	- folder-level reinforcement
	- reusable research template
	- checkpoint and project-memory persistence
- Concepts rejected:
	- relying on chat-only reminders instead of durable repository files
	- allowing direct code porting from unclear or incompatible licenses
- Notes for the spec:
	- the spec should cover both instruction-stack updates and example/template adoption so the rule is concrete in practice

---

## Related

- [[oss_documentation_research_dna_spec|Spec: Oss Documentation Research Dna]]
