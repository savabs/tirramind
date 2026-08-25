---
title: "Spec: llm_wiki_architecture"
tags:
  - doc/spec
  - topic/wiki
---

# Spec: llm_wiki_architecture

## Goal

Add a persistent TirraMind knowledge wiki that compiles high-value project knowledge into topic-centric markdown pages, and add deterministic tooling to rebuild its index and surface health issues.

## Files Affected

### New files
- `wiki/SCHEMA.md` — maintenance rules and page conventions for the wiki
- `wiki/index.md` — catalog of wiki pages generated from page metadata
- `wiki/log.md` — append-only chronological record of wiki operations
- `wiki/raw/README.md` — immutable-source policy for future ingests
- `wiki/pages/architecture/system_overview.md` — seed page for system architecture
- `wiki/pages/architecture/execution_engines.md` — seed page for agent vs. pipeline execution model
- `wiki/pages/roadmap/current_phases.md` — seed page for project phase state
- `agent/wiki/__init__.py` — wiki module exports
- `agent/wiki/catalog.py` — deterministic page scanner, index builder, and linter
- `tests/test_wiki_catalog.py` — focused unit and edge-case tests

### Modified files
- `pyproject.toml` — optional script entry point for the catalog tool
- `README.md` — short mention of the wiki layer and its purpose
- `[[llm_wiki_architecture]]` — task tracking and verification updates

## Implementation Steps

### Preflight artifacts
- [ ] 1.1: Write research doc `[[llm_wiki_architecture]]`
- [ ] 1.2: Write spec doc `[[llm_wiki_architecture_spec]]`
- [ ] 1.3: Create active task file `[[llm_wiki_architecture]]`

### Wiki scaffold
- [ ] 1.4: Create `wiki/` structure with `SCHEMA.md`, `index.md`, `log.md`, and `raw/README.md`
- [ ] 1.5: Define required page frontmatter fields: `title`, `type`, `summary`, `status`, `source_docs`, `updated_on`
- [ ] 1.6: Seed the wiki with 3 initial pages that synthesize current architecture and roadmap

### Deterministic catalog tool
- [ ] 1.7: Create `agent/wiki/catalog.py` with a constrained frontmatter parser for wiki pages
- [ ] 1.8: Implement page scanning over `wiki/pages/**/*.md`
- [ ] 1.9: Implement lint checks for missing frontmatter, missing required fields, duplicate page titles, broken wiki links, and orphan pages
- [ ] 1.10: Implement index generation so `wiki/index.md` is rebuilt from page metadata in a stable order
- [ ] 1.11: Add a CLI entry point returning non-zero exit code on lint errors

### Documentation integration
- [ ] 1.12: Update `README.md` with the role of the wiki layer and how to run the catalog tool
- [ ] 1.13: Add a project script entry in `pyproject.toml`

### Verification
- [ ] 1.14: Add focused tests for normal cases and edge cases
- [ ] 1.15: Run targeted tests and a direct CLI smoke check
- [ ] 1.16: Update the active task file with results

## Edge Cases

- Page missing frontmatter entirely
- Frontmatter missing one required field
- Page links to a non-existent wiki page
- Two pages reuse the same title
- Page with no inbound links except index listing should be marked orphaned
- Empty wiki pages directory should still produce a valid index and no crash
- Non-page markdown files in `wiki/` should be ignored by the scanner

## Testing Plan

1. Unit test frontmatter parsing with valid, malformed, and partial metadata blocks.
2. Unit test scanning with multiple pages, broken links, duplicate titles, and ignored files.
3. Unit test index generation ordering and rendered content.
4. Unit test lint output and exit-code behavior.
5. Edge-case suite covering empty page directory, orphan detection, malformed links, and missing frontmatter.
6. Smoke test the CLI entry point against the real repo wiki after scaffolding.

---

## Related

- [[llm_wiki_architecture|Research: Llm Wiki Architecture]]
