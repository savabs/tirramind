---
title: "Spec: oss_documentation_research_dna"
tags:
  - doc/spec
---

# Spec: oss_documentation_research_dna

## Goal
Make OSS-and-documentation-first research a permanent repository rule so future work starts by searching GitHub and authoritative documentation, recording findings in research artifacts, checking reuse constraints, and only then implementing code in repository style.

## Files Affected
- `.github/copilot-instructions.md` — add explicit external research, license, and concept-extraction rules
- `AGENTS.md` — add default agent obligations for GitHub/docs-first research and license-aware reuse
- `RulesForAI.md` — add a concise tactical checklist for OSS/documentation-first work
- `agent/tools/.instructions.md` — reinforce OSS/docs-first behavior for tool work
- `agent/quant/.instructions.md` — reinforce OSS/docs-first behavior for quant work
- `tests/.instructions.md` — reinforce OSS/docs-first behavior for test design work
- `[[project_memory]]` — store the rule as persistent project identity
- `[[RESEARCH_TEMPLATE]]` — reusable checklist template for new research notes
- `[[ai_workflow_rules]]` — example research note updated to the template structure
- `[[oss_documentation_research_dna]]` — research record
- `[[oss_documentation_research_dna_spec]]` — implementation spec
- `[[oss_documentation_research_dna]]` — task tracker
- `[[checkpoint_archive_2026]]` — archive containing the early-April workflow checkpoint entry

## Implementation Steps
1. Create `[[oss_documentation_research_dna]]` documenting the current workflow gap and the desired OSS/documentation-first rule.
2. Create `[[oss_documentation_research_dna_spec]]` defining how the rule should propagate through the instruction stack.
3. Create `[[oss_documentation_research_dna]]` with atomic implementation and validation steps.
4. Update `.github/copilot-instructions.md` so the main agent workflow explicitly requires GitHub and documentation search, multiple keyword strategies, license checks, research-file capture, and concept-only reuse when repository licenses are incompatible.
5. Update `AGENTS.md` and `RulesForAI.md` so both agent defaults and human-readable tactical rules reflect the same policy.
6. Update the folder-level `.instructions.md` files so module-local work inherits the same research and license discipline.
7. Create `[[RESEARCH_TEMPLATE]]` with a lightweight checklist for repositories searched, documentation reviewed, keyword variants used, license conclusions, architecture observations, and intended concepts.
8. Update a couple of recent workflow research notes to the new template structure so the repo contains concrete examples alongside the blank template.
9. Update `[[project_memory]]` so the rule becomes part of the repository's persistent operating identity.
10. Validate the changed instruction files, the new template, and the example research notes with targeted content checks and write a checkpoint summarizing the new operating rule.

## Edge Cases
- The rule must permit simple local refactors without pretending an external code search is always necessary; it should apply to new features, unfamiliar technology, and externally sourced concepts.
- The wording must prohibit direct copying from incompatible licenses while still allowing concept extraction into research notes.
- The instruction files must remain aligned so future sessions do not receive conflicting guidance.
- The research template should be lightweight enough to use routinely; if it is too heavy, people will ignore it.

## Testing Plan
- Validate the presence of the new OSS/documentation-first language in each targeted instruction file.
- Manually verify that the license-handling rule is explicit: compatible repositories may inform implementation, incompatible repositories may only contribute concepts captured in research notes.
- Validate that the research template explicitly includes search keywords, repository/doc findings, and license/reuse conclusions.
- Validate that the example research notes now follow the template sections without inventing external research that did not happen.
- Write a checkpoint file in `docs/memory/` so the change is discoverable in future sessions.

---

## Related

- [[oss_documentation_research_dna|Research: Oss Documentation Research Dna]]
