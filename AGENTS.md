# TirraMind Agent Definitions

## Default Agent Behavior

All agents operating in this repository must follow:
0. `[[agent_playground_doctrine]]` — canonical company identity (agent company, market playground #1, ML-first, ℰ, niche competition)
1. `.github/copilot-instructions.md` — project workflow and architecture rules
2. Folder-level `.instructions.md` — conventions for the specific module being edited
3. The 7-layer computation stack — know which layer you're working in, don't mix layers
4. GitHub/docs-first research discipline — for new features, unfamiliar tech, or external concepts, search OSS repositories and authoritative documentation before implementation, capture findings in research docs, and treat incompatible licenses as concept-only inputs
5. For math-heavy work, explanation is mandatory — state the statistic or objective, assumptions, implementation options, and why the chosen method and tool count are justified instead of merely coding the result
6. For substantive mathematical methods, name the trusted paper, standard reference, or authoritative documentation you are relying on and separate source-backed theory from repo-specific engineering choices
7. Before each meaningful implementation step, gather and write down the step-local references for the exact topic, its subtopics, and adjacent concepts that may affect the step
8. The Signal Depth Doctrine — by default, tools should progress through L1 (aggregate) → L2 (entity-level) → L3 (cross-entity combinations), but aggregate/global-conditioning tools may remain L1 when GNN-guided evaluation shows L2 is unnecessary. Design for L2 from day one when the source supports entity resolution.
9. Maximize learnable structure over hand-coded logic — hand-code only schemas, invariants, and explicit factual relationships present in source data; push ambiguous relations, scoring, predictive behavior, and latent structure into learned components whenever feasible.
10. **Responsible AI coding (Vibe Coding Protocol).** Target AI-owned implementation at *leaf nodes* — components that nothing else depends on (small blast radius, easy to revert). Before each execution: spend 15–20 min exploring the codebase with the model, build a joint plan, then compress everything into a single structured prompt. For leaf-node code, enforce minimal acceptance tests: 1 happy path + 2 failure cases. Reserve exhaustive edge-case test suites for non-leaf, architecturally significant code. Compress context within a session after large code generation to prevent drift.
11. **Zero-hallucination policy on factual claims.** Never assert facts about external systems (API endpoints, response schemas, ticker symbols, exchange specs, library interfaces, mathematical method properties, data availability, geographic coverage) without verifying against real sources. Always do an internet search, fetch official documentation, or probe the actual endpoint before writing code or specs that depend on external facts. If a fact cannot be verified, state that explicitly — never fill the gap with a plausible-sounding guess. See rule #15 in `copilot-instructions.md` for the full list of categories requiring verification.

### Obsidian Knowledge Base (Mandatory for All Agents)

The project root is an Obsidian vault. Every agent must treat it as the primary knowledge base.

**HTML-First output rule:** All new artifacts (research notes, specs, checkpoints, task files, ADRs, wiki pages) are `.html` files. Each HTML artifact gets a companion thin `.md` stub for Obsidian navigation. The `.html` file is canonical content; the `.md` stub is the indexing layer only. See the HTML-First Artifact System section in `.github/copilot-instructions.md` for stub template and file conventions.

1. **Every new or modified `.md` stub must have YAML frontmatter** with `title` and `tags` using the project tag taxonomy (`doc/*`, `phase/*`, `topic/*`, `layer/*`, `status/*`). The stub body must contain a link to the companion `.html` file and a `## Related` section with `[[wiki links]]`.
2. **All cross-references must use `[[wiki links]]`**, never bare file paths. Write `[[foo]]` or `[[foo|display text]]` — Obsidian resolves by filename.
3. **Every research, spec, and task `.md` stub must have a `## Related` section** with `[[wiki links]]` to the triad (research ↔ spec ↔ task) and topically related docs.
4. **Navigate context via the knowledge graph.** Before linearly scanning directories:
   - Grep for `[[filename]]` to find backlinks (what references this file).
   - Grep for a tag string (e.g., `topic/convergence`) to find all related docs across `docs/`, `tasks/`, `wiki/`.
   - Follow `[[wiki links]]` from the current file to discover related context.
   - Read the latest `doc/checkpoint` and the active task/spec/research triad to cold-start a session. Use a project memory file only when one is actually maintained.
5. **When creating a new file**, add it to the knowledge graph immediately: create both the `.html` content file and the `.md` stub, link the stub from related documents, and add a `## Related` section linking back.
6. **When deleting or renaming a file**, update all `[[wiki links]]` that reference it.
7. **When completing a task**, update the `status/active` tag to `status/done` in the stub's frontmatter before moving both `.html` and `.md` to `tasks/done/`.
8. **Run `python scripts/obsidian_linkify.py`** after batch-creating multiple docs. For single-file edits, add frontmatter and links manually.
9. **Run `python scripts/obsidian_lint.py`** periodically and after batch changes. Fix FM01/FM02 (frontmatter) and LK01 (broken links) before committing. LK02 (orphans) and ST03 (stale) are advisory.
10. **Page creation threshold:** Only create a new wiki page when a topic appears in 2+ research notes or is central to a single source. See `[[SCHEMA]]`.
11. **Contradiction handling:** When sources disagree, note both positions with dates and add `contradicted: true` to frontmatter. Never silently overwrite old information.
12. **Log rotation:** When `wiki/log.md` exceeds 500 entries, rotate to `wiki/log-YYYY.md` and start fresh.

## MCP Tools (Cursor — `make mcp-setup`)

Local config: `.cursor/mcp.json` (gitignored; copy from `.cursor/mcp.json.example`).

| Server | Purpose |
|--------|---------|
| `tavily` | Web search for external fact verification |
| `wolfram-alpha` | Symbolic math / computational queries (`.env`: `WOLFRAM_APP_ID`) |
| `context7` | Up-to-date library/API documentation |
| `sequential-thinking` | Structured multi-step math reasoning |
| `git` | Repo log/diff/blame via MCP |
| `playwright` | Headless browser for scraping/testing |
| `memory` | Persistent knowledge graph (`.tirra_memory/`) |

After `make mcp-setup`: add `TAVILY_API_KEY`, run `pip install mcp-server-git`, restart Cursor.

## Tool Permissions

All agents may use:
- File read/write tools
- Terminal execution (for running tests)
- Search tools (grep, semantic search)
- MCP tools above when configured locally

No agent should:
- Make HTTP requests to external services during implementation (use mocks in tests)
- Modify files outside the scope defined in the active spec
- Skip the research → spec → implement pipeline for non-trivial changes
- Start non-trivial implementation without first creating or confirming the research note, spec, and active task file
- Edit non-workflow files before that preflight is complete; preflight-only edits are limited to `docs/research/`, `docs/specs/`, `tasks/active/`, and checkpoint files in `docs/memory/`
- Port code from repositories with unclear or incompatible commercial-use terms; extract concepts into research notes and reimplement independently instead
- Create or edit `.md` stub files without YAML frontmatter and proper `[[wiki links]]`
- Create content artifacts as `.md` files — all new output artifacts must be `.html` with a companion `.md` stub
- Convert or edit old inactive `.md` files — they are read-only historical records
- Write bare file paths (like `[[foo]]`) when a `[[wiki link]]` would work
- Leave a new file unlinked from the knowledge graph — always add it to Related sections of connected docs

**Active `.md` companion rule:** When resuming work on an existing `tasks/active/*.md` or any other live `.md` artifact, create a `<same-name>.html` companion alongside it. The `.html` becomes the primary working document. Add a single line at the top of the old `.md`: `> Legacy reference. Primary doc: [[<name>.html]]`. Stop updating the `.md` from that point forward.

## Key File Locations

| Purpose | Path |
|---------|------|
| Active tasks | `tasks/active/` |
| Completed tasks | `tasks/done/` |
| Research docs | `docs/research/` |
| Specs | `docs/specs/` |
| Architecture Decision Records | `docs/adr/` |
| Checkpoints | `docs/memory/` |
| Project memory | `docs/memory/` (checkpoint-first; dedicated project memory file if maintained) |
| Build automation | `Makefile` |

## Available Agents

| Agent | Purpose | Modifies Code? |
|-------|---------|----------------|
| `code-reviewer` | Reviews changes for correctness, style, security, test coverage | No |
| `quant-researcher` | Analyzes architecture, writes research/spec docs | No |
| `test-writer` | Generates comprehensive edge case test suites | Tests only |
| `architect` | Analyzes design tradeoffs, writes ADR documents | No |

## Available Skills (Cursor — invoke with `/skill-name`)

Canonical location: `.cursor/skills/<name>/SKILL.md` (migrated from `.github/prompts/`).

| Skill | Purpose |
|-------|---------|
| `/brainstorm-to-spec` | Convert rough idea → research/spec-ready plan (no code) |
| `/spec-to-task` | Convert completed spec → active task file with atomic steps |
| `/research` | Research-only phase — read code, search OSS, write research doc |
| `/full-pipeline` | Full 5-phase pipeline: research → spec → implement → review → checkpoint |
| `/next-step` | Execute one atomic step from the active task file |
| `/sprint` | Execute ALL remaining task steps without stopping |
| `/review-quant` | Review quant module for numerical stability/correctness |
| `/debug` | Structured diagnosis: reproduce → instrument → hypothesis → fix → regress |
| `/post-mortem` | Retrospective after a hard bug or failed approach |
| `/session-start` | Manual cold-start (basic context also injected by `sessionStart` hook) |
| `/extract-learnings` | Mine completed tasks for reusable patterns, update project memory |

Regenerate skills after editing prompts: `make cursor-skills`

## Context Efficiency Rules

1. Read the task file first — it's the source of truth for what to do next.
2. Reference specs by step number ("per spec step 2.3") — don't re-derive.
3. After completing a feature, suggest starting a new session.
4. Write checkpoints at natural breakpoints.
5. When external inspiration is needed, search with multiple keyword variants and write the relevant repo/doc findings into the research file instead of leaving them implicit in chat.
6. If a request is not obviously trivial, treat it as non-trivial and complete workflow preflight before implementation.
7. For math-heavy work, identify the trusted source for the chosen method before coding instead of relying on memory alone.
8. For each meaningful step, record the step-local references before coding so later implementation can cite them directly.

### Write-Gate Protocol (prevents lost decisions)

**Rule: write before confirming.** A decision is not complete until it exists in a file. The agent must
write to the canonical file *in the same turn* as the approval, then confirm. Deferring the write is
not allowed — if the session ends before the write happens, the decision is lost.

Correct flow:
```
user approves → agent writes canonical file → agent confirms "written to [file]"
```

Broken flow (forbidden):
```
user approves → agent says "great, I'll do that" → [session ends] → LOST
```

If the agent responds to an approval with prose instead of a file write, the user should say: **"write it first"**.

### Single-Owner Rule (prevents fact drift)

Each fact lives in exactly one file. All other files reference it; they never copy it.

| Fact type | Canonical owner | All others |
|---|---|---|
| Current metrics (test counts, node counts, ENRICHMENT_DIM, DAG size) | `memories/repo/tirramind_structure.md` | `[[tirramind_structure]]` link only |
| Roadmap / next phases / phase ordering | `[[quant_training_ground]]` | `[[quant_training_ground]]` link only |
| What happened in a session | `docs/memory/checkpoint_<date>.html` | Append-only — never edited after session |
| Architecture decisions | `docs/adr/NNNN-<slug>.html` | `[[adr-slug]]` link only |

**Checkpoints are historical records.** They record what was believed at the time the checkpoint was
written. They are never corrected inline after the session ends. If a checkpoint stated something
wrong, record the correction in the canonical owner file and the *next* checkpoint — never by editing
the old one. An edited checkpoint has a timestamp that lies about when the information was current.

**Fact drift lint:** Run `python scripts/fact_lint.py` to detect numeric constants that have been
copied outside their canonical owner. Fix FL01/FL03 before committing. FL02 is advisory (run with
`--strict` to enforce). This catches drift before it silently diverges across files.

### Obsidian-First Context Navigation

9. **Cold-start via the knowledge graph:** Read the latest `doc/checkpoint` → active task → linked research/spec docs. Use a project memory file only if it exists and is maintained. Don't scan directories.
10. **Find related context by backlinks:** `grep_search` for `[[filename]]` across `docs/`, `tasks/`, `wiki/` to see everything that references a given file.
11. **Find topic clusters by tags:** `grep_search` for a tag (e.g., `topic/convergence`) in frontmatter to find all research, specs, tasks, and wiki pages for that topic.
12. **Follow the triad:** Every feature has three linked HTML files: `[[research_note]]` → `[[spec]]` → `[[task]]` (each with a companion `.md` stub). Follow these links instead of guessing file paths.
13. **Update the graph as you work:** When you create, modify, or complete a file, ensure its frontmatter tags and `## Related` links are current. The graph is only useful if it's maintained.
