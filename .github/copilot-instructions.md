# TirraMind — Copilot Agent Workflow Instructions

**TirraMind is an advanced self-improving agent company.** Canonical identity: [[agent_playground_doctrine]]. **Markets are playground #1** — the first scoring environment for the agent, not the permanent definition of the firm.

**The core asset is raw intelligence:** a learned **cross-domain entity embedding space** (ℰ) from heterogeneous sensors + HetTGN/memory. **ML is the primary R&D path;** quant finance is implemented as constraints and readouts (`agent/quant/`), not as hand-crafted factor research. Commercial **niche is not decided** until [[niche_playground_competition]] completes.

**Outward product:** probability distributions and decision advantage over Layer 0–3 reality (physical → behavioral → information → prices). Not a trading algorithm alone. Not a generic copilot. A living system that compounds from experience on the playground, then generalizes to other domains later.

**The edge is: unconventional observation × SOTA math × living system architecture.** The system combines methods nobody else combines: temporal heterogeneous graph neural networks, Bayesian belief propagation, Kalman signal fusion, RL policy, EWC continuous learning, causal chain detection, and investigative entity linking across 51 data sources. Conventional AI companies use one or two of these. TirraMind uses all of them on data most AI companies never look at.

**Cost discipline is strategic.** The cheapest data is often the most valuable because nobody else is looking at it. Math on common data is commoditized. Math on unique data is the moat. Every design decision should reflect this: unique observation × advanced science = asymmetric edge that cannot be replicated by throwing money at Bloomberg.

This project follows a strict phased workflow. Never skip phases.

---

## Obsidian Knowledge Base (Mandatory)

**The project root is an Obsidian vault.** All markdown files in `docs/`, `tasks/`, `wiki/`, and root-level `.md` files are part of one interconnected knowledge graph. Obsidian's Graph View, backlinks, and tag search are the primary navigation tools for understanding the project.

### Rules for Every Markdown File

1. **YAML frontmatter is mandatory.** Every `.md` file (except templates) must start with `---` frontmatter containing at minimum `title` and `tags`.
2. **Use `[[wiki links]]` for all cross-references.** Never write bare paths like `docs/research/foo.md`. Write `[[foo]]` or `[[foo|display text]]` instead. Obsidian resolves these by filename — no path prefix needed.
3. **Use the tag taxonomy.** Tags are hierarchical with `/` separators:
   - Document type: `doc/research`, `doc/spec`, `doc/task`, `doc/adr`, `doc/checkpoint`, `doc/wiki`, `doc/memory`
   - Status: `status/active`, `status/done`
   - Phase: `phase/7b`, `phase/7c`, `phase/8`, `phase/9`, etc.
   - Topic: `topic/convergence`, `topic/world-model`, `topic/pipeline`, `topic/polymarket`, etc.
   - Layer: `layer/surveillance`, `layer/feature-engineering`, `layer/world-model`, `layer/fusion`, `layer/learning`, `layer/adversarial`, `layer/llm-support`
4. **Add a `## Related` section** at the bottom of every research, spec, and task file with `[[wiki links]]` to related documents (the research↔spec↔task triad, plus topically related files).
5. **Keep links current.** When you create, rename, or delete a file, update all `[[wiki links]]` that reference it. When a task moves from `tasks/active/` to `tasks/done/`, update its `status` tag from `status/active` to `status/done`.
6. **Run `python scripts/obsidian_linkify.py`** after batch-creating new docs to auto-generate frontmatter, convert paths to wiki links, and add Related sections. For single-file edits, add frontmatter and links manually.

### Using Obsidian for Context Navigation

Before reading a file, use Obsidian's structure to find the right file faster:
- **To understand a topic:** Search by tag (e.g., `tag:topic/convergence`) to find all research, specs, tasks, and wiki pages for that topic.
- **To find what depends on a file:** Check the file's backlinks — every file that `[[links]]` to it.
- **To understand the current state:** Read the latest `doc/checkpoint` file, then the active task files (`tag:status/active`).
- **To trace a feature end-to-end:** Follow the triad: `[[research_note]]` → `[[spec]]` → `[[task]]`.
- **To see the big picture:** Use Graph View filtered by tag to see clusters of related work.

The agent should use `grep_search` for `[[filename]]` to discover backlinks programmatically when Obsidian UI is not available. For tag-based discovery, grep for the tag string in frontmatter across `docs/`, `tasks/`, and `wiki/`.

### Vault Health

- **Run `python scripts/obsidian_lint.py`** periodically and after batch changes. Fix any FM01/FM02 (frontmatter) and LK01 (broken links) findings before committing. LK02 (orphans) and ST03 (stale) are advisory.
- **Page creation threshold:** Create a new wiki page only when a topic appears in 2+ research notes or is central to a single source. See `[[SCHEMA]]` for full rules.
- **Contradiction handling:** When sources disagree, note both positions with dates in the page and add `contradicted: true` to frontmatter. Never silently overwrite old information.
- **Log rotation:** When `wiki/log.md` exceeds 500 entries, rotate to `wiki/log-YYYY.md` and start fresh.

### Stub Frontmatter Template

Every `.md` stub (the Obsidian navigation file that accompanies each `.html` artifact) uses this template:

```yaml
---
title: <title>
tags:
  - doc/<type>
  - phase/<N>
  - topic/<slug>
  - layer/<slug>
  - status/<state>   # only for task files
---

> **Content:** [<filename>.html](<filename>.html) — open in browser.

## Related
- [[linked_research_or_spec]]
- [[linked_task]]
```

---

## HTML-First Artifact System

**All new output artifacts are HTML files.** Research notes, specs, checkpoints, task files, ADRs, wiki pages — everything goes in HTML. Markdown is reserved for thin Obsidian navigation stubs only.

### Why HTML over Markdown

- **Information density:** Tables, SVG diagrams, CSS layouts, annotated code, color. Everything Markdown fakes with ASCII, HTML does natively.
- **Readability:** Nobody reads a 100-line markdown file. HTML with tabs, color, and visual hierarchy gets read — and shared.
- **Two-way interaction:** Sliders, knobs, copy-to-prompt buttons. The artifact talks back.
- **Custom editing interfaces:** Throwaway HTML tools (ticket triagers, DAG reviewers, parameter tuners) always ending with a "Copy as prompt" or "Copy as JSON" export button.

### The Hybrid Model: HTML Content + MD Stub

Every artifact consists of two files:

| File | Purpose |
|---|---|
| `docs/research/feature.html` | **Canonical content** — the rich HTML document you read and share |
| `docs/research/feature.md` | **Navigation stub** — Obsidian frontmatter + wiki links only |

The `.html` file is what you open, read, and share. The `.md` stub is what Obsidian indexes for graph view, backlinks, and tag search.

### HTML File Conventions

- **Open locally:** `open feature.html` or ask the agent to open in browser. Upload to S3 for shareable links.
- **Structure:** Every HTML artifact must include a header (title, phase, date), a navigation sidebar or tab structure for long documents, and SVG/CSS diagrams instead of ASCII art.
- **Interactive artifacts** (triagers, tuners, editors) must end with a **"Copy as prompt"** or **"Copy as JSON"** button that exports current state to clipboard.
- **Code snippets** inside HTML artifacts use `<pre><code>` with syntax highlighting — never raw markdown code fences.

### When Plain Markdown Is Allowed (no HTML companion needed)

- The `.md` navigation stubs themselves
- `README.md` at the project root
- `wiki/log.md` (append-only log entries)
- Code comments and docstrings (not documents)

### Existing `.md` Files — Migration Policy

**Do not convert old `.md` files to HTML.** The migration is forward-only and touch-based:

| File state | Action |
|---|---|
| `tasks/done/*.md` or inactive research/specs | Leave untouched — historical record, never modify |
| `tasks/active/*.md` currently being worked on | Create an `.html` companion alongside it. The `.html` is the primary working document from that point. The `.md` becomes a legacy reference — mention it in the HTML's Related section but stop updating it. |
| Any new artifact created today or later | `.html` + thin `.md` stub from the start — no legacy `.md` involved |

**How to companion an active `.md`:** create `<same-name>.html` next to it, open the `.md` for context, build the rich HTML version, then add a note at the top of the old `.md`: `> Legacy reference. Primary doc: [[<name>.html]]`. The `.md` is now read-only.

---

## Core Principle: Atomic Decomposition

**Break everything down to the smallest possible unit of work.**

Every feature, every fix, every idea — decompose it until each piece is so small it's trivially clear what to do. No step should require you to "figure out" multiple things at once.

- A task that takes more than ~2 hours of focused work is too big. Split it.
- If a step description has the word "and" in it, it's probably two steps.
- Each step should change one thing, test one thing, prove one thing.
- Prefer 10 tiny PRs over 1 medium PR. Prefer 5 tiny functions over 1 clever one.
- When in doubt, break it down further.

This applies to everything: planning, specs, implementation, debugging, research.

### How to Break Down a Phase

When a new phase starts, follow this exact sequence:

1. **Research** — Write `docs/research/<phase_name>.html` (+ thin `.md` stub). Read only relevant files. No code.
2. **Spec** — Write `docs/specs/<phase_name>_spec.html` (+ thin `.md` stub). Transform research into ordered atomic steps.
3. **Decompose into task file** — Write `tasks/active/<task_name>.html` (+ thin `.md` stub) with numbered steps (e.g., 2.1, 2.2, ...).
4. **Implement one step at a time** — each step: make the change → test it → mark it done → move to next.

**Each step must be independently verifiable.** If you can't describe a one-line test for it, it's too vague. Break it further.

**Step naming convention:** `<phase>.<step>: <verb> <specific thing>` — e.g., `2.3: Implement BOCPD class with synthetic test`.

---

## Context Efficiency (Critical)

LLM context windows fill up. Every wasted token is a lost thought. Protect context ruthlessly:

1. **Move reasoning into files, not chat.** Analysis goes in research docs. Plans go in specs. Don't repeat them in conversation.
2. **Reference documents instead of re-explaining.** Say "per the spec, step 2.3" — don't re-describe what step 2.3 is.
3. **Read only necessary files.** Don't explore the whole codebase when you need one function.
4. **Use Obsidian backlinks and tags to navigate.** Instead of scanning directories, grep for `[[filename]]` to find what references a file, or grep for a tag to find all related docs. This is faster than reading file after file.
5. **Start new chat sessions after completing a feature.** Old context becomes stale and wastes tokens.
6. **Write checkpoint files** (`docs/memory/chat_checkpoint_<date>.html` + `.md` stub) at natural breakpoints so the next session can pick up without re-reading everything. The `.md` stub must have `doc/checkpoint` tag.
7. **Keep task files as the source of truth** for what's done and what's next. The task file should be enough to resume work cold.
8. **Don't re-analyze architecture during implementation.** That's what the research phase was for. If something is wrong, update the spec, don't re-derive in chat.
9. **Use the knowledge graph to cold-start.** When beginning a new session, read the latest checkpoint and active task first, then follow `[[wiki links]]` to reach the relevant research/spec context. If a maintained project memory file exists, use it as an accelerator, not a hard dependency. Don't re-read the entire codebase.

## Context Efficiency (Critical)

LLM context windows fill up. Every wasted token is a lost thought. Protect context ruthlessly:

1. **Move reasoning into files, not chat.** Analysis goes in research docs. Plans go in specs. Don't repeat them in conversation.
2. **Reference documents instead of re-explaining.** Say "per the spec, step 2.3" — don't re-describe what step 2.3 is.
3. **Read only necessary files.** Don't explore the whole codebase when you need one function.
4. **Use Obsidian backlinks and tags to navigate.** Instead of scanning directories, grep for `[[filename]]` to find what references a file, or grep for a tag to find all related docs. This is faster than reading file after file.
5. **Start new chat sessions after completing a feature.** Old context becomes stale and wastes tokens.
6. **Write checkpoint files** (`docs/memory/chat_checkpoint_<date>.html` + `.md` stub) at natural breakpoints so the next session can pick up without re-reading everything. The `.md` stub must have `doc/checkpoint` tag.
7. **Keep task files as the source of truth** for what's done and what's next. The task file should be enough to resume work cold.
8. **Don't re-analyze architecture during implementation.** That's what the research phase was for. If something is wrong, update the spec, don't re-derive in chat.
9. **Use the knowledge graph to cold-start.** When beginning a new session, read the latest checkpoint and active task first, then follow `[[wiki links]]` to reach the relevant research/spec context. If a maintained project memory file exists, use it as an accelerator, not a hard dependency. Don't re-read the entire codebase.

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
| Roadmap / next phases / phase ordering | `tasks/active/quant_training_ground.md` | `[[quant_training_ground]]` link only |
| What happened in a session | `docs/memory/checkpoint_<date>.html` | Append-only — never edited after session |
| Architecture decisions | `docs/adr/NNNN-<slug>.html` | `[[adr-slug]]` link only |

**Checkpoints are historical records.** They record what was believed at the time the checkpoint was
written. They are never corrected inline after the session ends. If a checkpoint stated something
wrong, record the correction in the canonical owner file and the *next* checkpoint — never by editing
the old one. An edited checkpoint has a timestamp that lies about when the information was current.

**Fact drift lint:** Run `python scripts/fact_lint.py` to detect numeric constants that have been
copied outside their canonical owner. Fix FL01/FL03 before committing. FL02 is advisory (run with
`--strict` to enforce). This catches drift before it silently diverges across files.

## Operational Collaboration Rules

These rules are tactical. They shape how the agent collaborates during a session without replacing the repository's core workflow.

1. **Use chat for thinking, agent for execution.** Brainstorming, research, and spec work stay in planning mode. Bounded, testable steps move to implementation mode.
2. **Keep one problem per step.** If a request combines multiple distinct problems, decompose it before editing code.
3. **Research OSS and docs before coding new concepts.** For unfamiliar technology, new features, or externally inspired implementations, first search GitHub for strong open-source repositories and search authoritative documentation using multiple keyword variants. Record the useful repositories, docs, and search terms in the research file before implementation. **This is not optional.** When in doubt about whether research is needed, do the research.
4. **Before each meaningful implementation step, run a step-local resource pass.** Identify the exact topic of the step, the immediate subtopics, and adjacent concepts that may affect the design or implementation. Gather the relevant trusted references for that step before writing code.
5. **Write the step-local references down before coding.** Record the references for the current step in the research or spec artifacts so implementation can cite and follow them directly instead of relying on memory.
6. **Switch to debugging mode when stuck — with a hard limit.** If the model starts looping, stop guessing. Reproduce the issue, add targeted instrumentation, inspect outputs, and tighten tests before another fix attempt. **After 2 failed attempts at the same fix, mandatory switch to debug mode.** Do not attempt a 3rd fix without first completing Steps 1-4 of the debug protocol (`/debug`). This is a hard rule, not a suggestion.
7. **Prefer explicit file responsibility.** Choose file names and module boundaries that make ownership obvious and reduce duplicate implementations.
8. **Treat checkpoints as part of clean session boundaries.** After a feature or natural breakpoint, write a checkpoint and recommend a fresh chat.
9. **Respect licenses and extract concepts safely.** If an external repository has a license that is incompatible with commercial use or unclear for our use case, do not port its code. Capture the design, algorithm, or workflow insight in the research file, then implement it independently in TirraMind style.
10. **When the work becomes mathematical, explanation becomes mandatory.** Once the task moves beyond data collection into scoring, estimation, inference, filtering, optimization, or statistical control, explicitly explain the math: objective or test statistic, null and alternative if applicable, assumptions, numerical stability concerns, and why the chosen formulation matches the problem.
11. **Present viable mathematical implementation options before locking in.** For substantive math code, compare the main implementation choices (for example exact vs approximate, parametric vs empirical, batch vs online, sparse vs full-pairwise), state the tradeoffs, and explain why one choice is preferred in this codebase.
12. **Prefer the smallest high-signal toolset that preserves edge.** If adding more tools, signals, or methods improves theoretical coverage but materially harms interpretability, computational cost, or maintenance burden, prefer a smaller set of the best tools and justify that choice explicitly.
13. **Anchor substantive mathematical choices to trusted sources.** Before applying a statistical test, estimator, filter, optimization method, graphical model, or other mathematical concept, identify the trusted source being relied on (primary paper, standard reference, or authoritative library documentation), explain why that source is trustworthy here, and distinguish source-backed theory from repo-specific engineering choices.
14. **Maximize learnable structure; minimize hand-coded intelligence.** Hand-code only schemas, invariants, safety constraints, and explicit factual relationships directly stated by source data. Push ambiguous relations, weighting, scoring, predictive behavior, and latent structure into learned components whenever feasible.
15. **Never hallucinate factual claims — verify via internet search.** Any factual assertion about external systems must be verified against a real source before it enters code, specs, or research docs. This includes but is not limited to:
    - **API endpoints, parameters, response schemas, authentication requirements, rate limits, data formats** — always fetch and read the official documentation or test the endpoint. Never guess a URL, field name, or response structure from memory.
    - **Library interfaces, function signatures, class hierarchies, default values** — read the actual library docs or source code. Do not assume an API is the same as a similar library.
    - **Financial instrument identifiers** (ticker symbols, ISIN, exchange codes, contract specs, margin requirements, lot sizes) — verify against the exchange or data provider documentation.
    - **Regulatory structures, filing formats, data availability windows** — verify against the regulator's website (SEC, CFTC, FINRA, ECB, etc.).
    - **Mathematical method properties** (convergence guarantees, complexity, numerical stability, assumptions) — cite the paper or reference; do not state properties from memory alone.
    - **Geographic coverage, data freshness, historical availability** of any external data source — verify by probing the source or reading its documentation.
    When extensive internet search is required (new data source, new API, new instrument class, new mathematical method), perform it thoroughly using multiple keyword variants, synonyms, and search surfaces before writing any code. Record the verified sources (URLs, doc titles, version numbers) in the research file. If a source cannot be verified, say so explicitly — never fill the gap with a plausible-sounding guess.

---

## Internet Research Protocol

The agent has access to internet search tools (Tavily MCP) and a free page reader (`fetch_webpage`). Use them correctly to minimize cost while maximizing correctness.

### Tool Selection Decision Tree

```
User provides a specific URL?
  YES → fetch_webpage (FREE — 0 credits)
  NO  → Do I know the exact URL for the official docs?
          YES → fetch_webpage (FREE — 0 credits)
          NO  → tavily_search (1 credit, basic depth)
                  → Read results → follow best URL with fetch_webpage (FREE)
```

### Tool Capabilities and Costs

| Tool | Cost | When to Use |
|------|------|-------------|
| `fetch_webpage` | **FREE** | Always the first choice when URL is known. User gave a link, official docs URL, GitHub README, etc. |
| `tavily_search` (basic) | **1 credit** | Discovery — find the right sources when you don't have a URL. Use `max_results=5`, basic depth. |
| `tavily_search` (advanced) | **2 credits** | Only when basic search returned insufficient results on a complex/niche topic. |
| `tavily_extract` | **1 credit/5 URLs** | Batch extraction of multiple known URLs. Use only when fetching 3+ URLs at once (cheaper than individual fetch_webpage calls only in batch). |
| `tavily_crawl` | **1 credit/5 pages** | Map and read an entire documentation site. Use when you need comprehensive coverage of a library/API (e.g., "read all yfinance docs"). |
| `tavily_map` | **1 credit/10 pages** | Discover the URL structure of a site before crawling. Use before `tavily_crawl` to target specific sections. |
| `tavily_research` | **5-20 credits** | Deep multi-query research. ONLY for complex topics requiring multiple search angles (e.g., "compare all free commodity futures data APIs"). Ask before using — this is expensive. |

### Mandatory Rules

1. **When the user provides a URL, use `fetch_webpage` — never route through Tavily.** This is free and gives full page content.
2. **When you know the official docs URL from context, use `fetch_webpage` directly.** Don't search for something you already know how to find.
3. **Use `tavily_search` with basic depth and `max_results=5` as the default discovery tool.** Only escalate to advanced depth if basic returns nothing useful.
4. **After `tavily_search` finds relevant URLs, read them with `fetch_webpage` (free) — don't use `tavily_extract` for single URLs.**
5. **Use `tavily_extract` only for batch operations** (3+ URLs at once) where it's more efficient than multiple `fetch_webpage` calls.
6. **Use `tavily_crawl` sparingly** — only when you genuinely need to read the majority of a documentation site. For 1-3 specific pages, use `fetch_webpage` on each.
7. **Use `tavily_research` only with explicit user approval** — it consumes many credits.
8. **Always record verified sources** (URL, title, date accessed) in the research file. Every factual claim must be traceable to a source.
9. **If search returns no results or contradictory information, say so explicitly.** Never fill the gap with a plausible guess. Mark the claim as "UNVERIFIED" and suggest manual verification (e.g., test in terminal).
10. **For API/library verification, prefer terminal testing over search when possible.** Running `python -c "import yfinance; print(yfinance.Ticker('KC=F').history(period='5d'))"` is free and definitive — no Tavily credits needed.

### Token-Saving Strategies

- **Cache knowledge within a session.** If you searched for "yfinance futures ticker format" once, don't search again — reference the earlier result.
- **Batch related searches.** Instead of 5 separate searches for 5 tickers, do one search: "yfinance supported futures symbols commodity energy metals agricultural".
- **Use terminal for verification.** Testing an API call in the terminal is free and provides ground truth. Save Tavily credits for discovery, not verification.
- **Use `fetch_webpage` for known documentation sites.** PyPI, GitHub READMEs, official library docs — you often know the URL pattern without needing to search.
- **Write findings to files immediately.** Prevents re-searching in future sessions. The research doc is the persistent search cache.

---

## Pipeline

```
User Request → Research Phase → Specification Phase → Implementation (one atomic step at a time)
```

## Mandatory Workflow Preflight

For any non-trivial request, the agent must fail closed and complete the workflow setup before implementation begins.

Treat a request as non-trivial by default if it does any of the following:
- changes behavior
- changes architecture or file/module boundaries
- touches dependencies, configuration, prompts, instructions, or schemas
- touches more than one file
- requires external concepts, unfamiliar technology, or design judgment

Before implementing a non-trivial request, the agent must first ensure all three artifacts exist and are current:
1. `docs/research/<feature_name>.html` (+ `.md` stub)
2. `docs/specs/<feature_name>_spec.html` (+ `.md` stub)
3. `tasks/active/<task_name>.html` (+ `.md` stub)

Before that preflight passes, the agent may edit only workflow artifacts needed to satisfy the preflight:
- `docs/research/`
- `docs/specs/`
- `tasks/active/`
- `docs/memory/` checkpoint files

Until those artifacts exist, do not edit implementation files, tests, configs, prompts, or package manifests.

The only exception is a truly trivial request: a single-file, low-risk change with no behavior or interface change and no design ambiguity, such as typo fixes, comment wording, or narrowly scoped markdown cleanup. If there is doubt, treat the work as non-trivial.

When implementation starts, explicitly reference the governing task file and spec step instead of re-deriving the plan in chat.

---

## Phase 1: Research (before any code changes)

1. Read only the files relevant to the requested feature.
2. For new features, unfamiliar technology, or external concepts, search GitHub and authoritative documentation first. Use multiple keyword variants, synonyms, and search surfaces until the landscape is clear enough to cite concrete repos/docs in the research note.
3. For each meaningful implementation step, identify the step-local topic, subtopics, and adjacent concepts, and add the relevant references to the research note or spec before implementation begins.
4. Analyze project structure and dependencies.
5. Identify the correct insertion points for new code.
6. Record findings in `docs/research/<feature_name>.html`, including relevant repositories, documentation links/titles, reuse constraints, and the concepts to implement. Create a companion `.md` stub with frontmatter for Obsidian navigation.
7. Use `docs/research/RESEARCH_TEMPLATE.html` as the default structure when creating a new research note unless a task clearly needs a more specialized structure.
8. **Add Obsidian metadata to the `.md` stub** with appropriate `tags` (doc/research, phase/N, topic/slug, layer/slug) and a `## Related` section linking to the spec, task, and topically related docs using `[[wiki links]]`.

**No code is edited during this phase.**

Research HTML document structure — use tabs or collapsible sections for:
- **Current Architecture** — relevant modules, patterns, dependencies (include module relationship SVG)
- **Observations** — what exists, what's missing, what connects to what
- **Risks** — edge cases, breaking changes, security concerns (color-coded by severity)
- **Data Requirements** — data series/sources needed, available, missing
- **Math/Algorithm Survey** — algorithms, libraries, complexity (include decision table for implementation options)
- **References** — verified sources (URLs, doc titles, version numbers) for every external claim
- **Related** — navigation links to companion spec and task HTML files

---

## Phase 2: Specification (before any code changes)

Transform research into a precise implementation plan.
Write to `docs/specs/<feature_name>_spec.html` with a companion `.md` stub for Obsidian navigation.

If external codebases informed the design, the spec must state whether the source is only conceptual or whether the license permits implementation patterns to be reused. When in doubt, treat the source as conceptual only.

**Add Obsidian metadata to the `.md` stub** with `doc/spec` tag and link back to the research note and task file in a `## Related` section.

Specification HTML document structure — use tabs or sections for:
- **Goal** — what the feature must accomplish
- **Files Affected** — list of files to create or modify
- **Implementation Steps** — ordered numbered steps with color-coded status (not started / in progress / done)
- **Edge Cases** — possible failure scenarios
- **Testing Plan** — how the feature should be validated

Include: flowchart SVG of the implementation pipeline, code mockups for key interfaces, and links back to the research HTML file.

---

## Phase 3: Implementation

1. Follow the spec strictly.
2. Modify only files listed in the spec.
3. Do not re-analyze architecture during coding.
4. Before starting each meaningful implementation step, confirm the step-local references for that step and its nearby concepts have been written down in the research/spec artifacts.
5. If issues arise, update the spec first, then continue.
6. **After each sub-phase completes, write and run an extensive edge case test suite.** This is mandatory, not optional. Cover: invalid inputs, boundary values, error paths, security leakage, timeout behavior, type mismatches, missing required fields, exception handling, and any domain-specific edge cases. A sub-phase is not complete until its edge case tests pass.
7. After implementation and tests pass, update the task file.

Implementation is not allowed to start for non-trivial work until the mandatory workflow preflight has been completed.

Implementation must be traceable back to the research note. Code first, research later is not acceptable for new concepts in this repository.

---

## Responsible AI Coding (Vibe Coding Protocol)

AI models can write large, correct production changes. Use this power deliberately.

### Leaf Node Rule
Target AI-owned implementation at **leaf nodes** — components that nothing else depends on. These have a small blast radius, are easy to revert if wrong, and require the least architectural judgment. Core architecture, cross-cutting concerns, and foundational modules still require human oversight and line-level review. Before delegating a task fully to AI, ask: "Is this a leaf?" If not, be more prescriptive in the spec.

### Pre-Execution PM Ritual (15–20 min before each meaningful step)
Do not hand a vague task to the model and expect correct output. Before execution:
1. Let the model explore the codebase — find the relevant files, understand existing patterns.
2. Build a joint execution plan. Confirm edge cases and constraints explicitly.
3. Compress all context and specifications into a single well-structured prompt before triggering the implementation run.

Skipping this ritual multiplies failure rate and produces drift. The upfront investment collapses total time.

### Minimal Acceptance Tests (not exhaustive review)
For AI-generated code in leaf nodes, enforce minimal but definitive end-to-end tests:
- **1 happy path** — the expected behaviour works end-to-end.
- **2 failure cases** — the two most likely error paths are handled correctly.

This is cheaper than reading every line and more reliable than complex unit test suites for catching behavioural regressions. Reserve the full edge-case test suite (rule 6 above) for non-leaf, architecturally significant code.

### Context Compression
Periodically compress context within a session — especially after generating large blocks of code. Summarise what was accomplished into a checkpoint or task file note, then continue. This prevents drift where the model starts contradicting earlier decisions as context fills.

---

## Task Management

Each active feature has a task file: `tasks/active/<task_name>.html` with a companion `tasks/active/<task_name>.md` stub for Obsidian navigation.

The HTML task file tracks phase, status, numbered steps with completion state, and links to the research and spec HTML files.

The `.md` stub uses this frontmatter:

```yaml
---
title: "Task: <name>"
tags:
  - doc/task
  - status/active
  - phase/<N>
  - topic/<slug>
  - layer/<slug>
---

> **Content:** [<task_name>.html](<task_name>.html) — open in browser.

## Related
- [[<name>]]        # research
- [[<name>_spec]]   # spec
```

Mark completed tasks by:
1. Updating the task HTML file status to `completed`
2. Updating the stub's `status/active` tag to `status/done`
3. Moving both the `.html` and `.md` files to `tasks/done/`

---

## Architecture Decision Records

When a design decision affects multiple modules, layers, or has non-obvious tradeoffs, capture it in `docs/adr/NNNN-<slug>.html` (+ thin `.md` stub) using the template at `docs/adr/TEMPLATE.html`. ADRs are numbered sequentially and never deleted — only superseded.

---

## Memory System

Store important architectural knowledge in a maintained project memory file under `docs/memory/` when one exists.
Until then, treat the latest checkpoint plus the active task/spec/research triad as the persistent context for future tasks.

Write checkpoint files (`docs/memory/chat_checkpoint_<date>.html` + `.md` stub) at natural breakpoints so the next session can pick up without re-reading everything.

**Always save snapshots and summaries to files.** When the user asks for a snapshot, status update, summary, or says they're logging off — write it to `docs/memory/chat_checkpoint_<date>.html` (and a companion `.md` stub) automatically. Never just print a summary to chat without also persisting it. The checkpoint file is the handoff artifact; treat it as mandatory, not optional.

**All memory HTML files must have a companion `.md` stub.** Checkpoint stubs get `doc/checkpoint` tag. Project memory stubs get `doc/memory` tag. Include `[[wiki links]]` to the task files and specs that were worked on during the session.

### Automation Tools

- **`python scripts/session_checkpoint.py [-m "summary"]`** — Auto-generate a checkpoint from git state and active tasks. Use when ending a session quickly or when a manual checkpoint would be too verbose.
- **`python scripts/rotate_checkpoints.py [--keep 15]`** — Archive old checkpoints (keep last 15 by default). Run when checkpoint count exceeds ~30 or when cold-start is slow due to too many checkpoint files.
- **`python scripts/quality_gate.py [--task file]`** — Run pre-completion quality checks (tests pass, obsidian lint clean, all task steps checked). Use before marking a task as complete.
- **`python scripts/extract_patterns.py`** — Mine completed tasks and checkpoints for recurring lessons and patterns. Use with the `/extract-learnings` prompt.

---

## Context Efficiency

1. Move reasoning into files, not chat.
2. Reference spec documents using `[[wiki links]]` instead of repeating analysis.
3. Read only necessary files — use Obsidian tags and backlinks to find them.
4. Start new chat sessions after completing a feature.
5. Write checkpoints at natural breakpoints with proper frontmatter.
6. Task file = source of truth for progress. Must be resumable cold.
7. Never re-derive architecture during implementation — that's what research was for.
8. Use the knowledge graph: follow `[[links]]` to navigate, grep for `[[filename]]` to find backlinks, grep for tags to find topic clusters.

---

## Project Architecture (quick reference)

- **Entry point:** `agent/cli.py` → `agent/core/orchestrator.py`
- **LLM:** `agent/reasoning/llm_client.py` (OpenAI-compatible — SUPPORT ROLE ONLY, never makes trading decisions)
- **Planning:** `agent/planner/task_planner.py` (hierarchical task decomposition)
- **Memory:** `agent/memory/store.py` (episodic + semantic + working)
- **Tools:** `agent/tools/` (web_search, web_browse, execute_python, run_shell, file ops, market_data, macro_data, polymarket, insider_filings)
- **Data:** `agent/data/cache.py` (local file cache, SHA256 keys, 6hr TTL)
- **Config:** `agent/config/settings.py` (env-var based, `TIRRA_` prefix)
- **Quant:** `agent/quant/` (changepoint/BOCPD, regime/HMM, spectral/FFT+CWT, scoring, backtest/WalkForward)

The orchestrator pipeline: Research → Plan → Execute → Synthesize.

---

## Architecture Priority: Math Before LLM

**The LLM is scaffolding. The math is the product.**

**The representation is hand-built; the intelligence is learned.** TirraMind may hand-code the graph/data schema and explicit factual edges from source records, but it should not hard-code predictive rules or market-direction logic where a learnable component can absorb the task. If a relationship is explicit in the source, encode it. If it is ambiguous, predictive, or latent, prefer probabilistic inference or learned structure.

The system produces probability distributions, not recommendations. Every output is a distribution with uncertainty bounds, never a point estimate or text opinion.

**The world is a POMDP.** The global system is a Partially Observable Markov Decision Process: states are partially hidden (entities act on information we cannot see), the environment is non-stationary (regimes shift), there are millions of actors with latent intentions, and rewards are sparse and delayed (a rate hike plays out over 12–18 months). The full stack is designed around this reality — GNN perceives the partially observable state, the world model represents uncertainty over hidden states, Kalman fusion integrates noisy evidence, and the RL policy plans under that uncertainty. Every layer serves this POMDP structure. Use RL and the world model where this sequential, uncertain, partially observable structure demands it — not everywhere, but precisely where hand-coded rules or point estimates would fail.

The priority ordering for all implementation work:

1. **More data tools** — expand the surveillance surface. Free APIs first. Physical/real-time signals (T0) before disclosure-lag signals (T3).
2. **Standardized signals** — every data source feeds into a normalized feature (OFI, VPIN, Hurst, transfer entropy, mutual information, Hawkes intensity).
3. **World model** — Bayesian network (pgmpy/pymc). Nodes = hidden states. Evidence injection → belief propagation → posterior updates. This is the transition model of the POMDP — how the world evolves given actions and observations.
4. **Signal fusion** — Kalman/particle filter (filterpy). Fuse noisy multi-source observations into optimal state estimates. This is the belief-state estimator of the POMDP.
5. **Probabilistic output** — Monte Carlo simulation, copulas for tail dependence, Kelly criterion for sizing.
6. **RL policy** — the policy layer of the POMDP. Current: SAC (model-free — learns from real experience). Target (Phase 48): Dreamer-style model-based RL — plans by imagining thousands of future trajectories through the world model before acting. Thompson Sampling bandit is preserved as the exploration heuristic for goal selection (what to investigate), while the RL policy governs action under uncertainty (what to do with the information).
7. **Adversarial layer** — manipulation detection, edge decay monitoring, game-theoretic counterparty modeling.
8. **LLM last** — text parsing, hypothesis generation, narrative synthesis. The LLM explains what the math decided. It doesn't decide.

If a proposed change improves LLM capabilities but doesn't touch layers 1-7, question whether it's needed now. Build the mathematical intelligence first.

When implementing layers 2-6, communicate like a mathematician writing production code: define the quantity being estimated, the decision rule, the assumptions under which it is valid, the failure modes, and the strongest competing implementation options that were rejected.
When selecting a mathematical method for layers 2-6, state which documentation or primary source you trust for that method before implementation if the concept is substantive and not already established locally.

**Model agnosticism doctrine (standing rule):** No model, method, or architecture is sacred. The current stack (pgmpy Bayesian DAG, SAC RL, HetTGN GNN, Kalman fusion) is the best-justified choice *right now* given data volume and scale. If real backtests after Phase 40 show a component is not producing genuine predictive edge — measured by out-of-sample calibration, Sharpe attribution, or convergence signal quality — that component must be replaced or upgraded, regardless of how much code was written for it. The test is always: does this produce real edge in the real world? If no, change it. Sunk cost is not a reason to keep a weak model. The candidates for upgrade when triggered: world model DAG → variational inference (PyMC) at >500 nodes; SAC hidden_dim 128→256 when replay buffer saturates; GNN sparse attention when entity count >500K; copula tail model when correlated crisis risk becomes the dominant risk factor. **Phase 48 target:** transformer world model + Dreamer model-based RL. The transformer learns causal structure from data (replacing hand-coded DAG edges); Dreamer plans by imagining trajectories through it. This is the natural POMDP solver at scale — but it is gated behind Phase 40 proving the current stack has hit its ceiling and Phase 47 density audit confirming sufficient observation history per entity type.

---

## Signal Depth Doctrine: Investigative Intelligence

**"We are secret journalists. Our job is to find data that has patterns — things that lead to things that not many people can see."**

Every data tool should be evaluated against three depth layers. Surface-level aggregates (L1) are commodity. Entity-level resolution (L2) is where alpha starts. Cross-domain entity combinations (L3) are the moat nobody else has.

**Default depth progression for tools:**
- **L1 (Aggregate):** Top-line numbers, summaries, indices — what everyone sees. This is infrastructure, not edge.
- **L2 (Entity-Level):** Resolve to individual actors, transactions, vessels, wallets, filers. Track entities over time. Build per-entity time-series and anomaly baselines.
- **L3 (Cross-Entity Combinations):** Link entities across data domains. Patterns that only emerge from combining insider filings + vessel tracking + CFTC positioning + GDELT events. This is where nobody else operates.

**The investigative journalist test for every tool decision:** Would an investigative reporter find this output useful for uncovering something hidden? If the answer is no, the tool is not deep enough.

**Rules:**
1. When building or upgrading a data tool, design for L2 from day one — entity resolution, time-series per entity, not just latest snapshots.
2. When designing features (Layer 2 of the computation stack), consider cross-domain entity links, not just single-source statistics.
3. Every tool's research note must include a "Depth Roadmap" section: what L1/L2/L3 look like for that specific data source.
4. Prefer tools that reveal hidden entity behavior over tools that aggregate public numbers.
5. The entity resolution layer (unified entity graph across data sources) is a first-class architectural component, not an afterthought.
6. **GNN-guided iterative expansion.** Do not blindly upgrade all tools to L2 or add new data tools based on coverage checklists. After each batch of L2 upgrades, train the GNN on the current entity graph and evaluate: which entity neighborhoods are sparse, which attention heads are starved for signal, which cross-domain edges are missing. Use that evaluation to prioritize which tools to upgrade or build next. The GNN's attention weights and pattern recovery metrics are the signal for expansion decisions — not a spreadsheet of untouched tools.
7. **Aggregate tools may not need L2.** Macro-level tools (treasury_receipts, consumer_sentiment, central_bank_balance, global_pmi, etc.) produce country-level or market-level numbers, not entity-level data. These are better consumed as global conditioning variables or country-node features, not forced into the entity graph. Only upgrade to L2 if the GNN evaluation shows that entity type is starved.
8. **New tool creation follows the same principle.** Before building a new data tool, check whether the GNN evaluation indicates a gap that tool would fill. A new tool that adds a 4th observation channel to an already-dense entity type is lower priority than one that creates the first link between two disconnected entity clusters.

See the latest checkpoint, active surveillance tasks, and maintained doctrine notes under `docs/memory/` for the current Deep Surveillance Doctrine, entity-linking patterns, and tool depth audit.

---

## Cost Discipline

**$0 until proven edge.**

- All data sources: free public APIs (SEC EDGAR, GDELT, CFTC, Polymarket Gamma, ClinicalTrials.gov, ADS-B via OpenSky, ISO/RTO grid data)
- All math libraries: open source (pgmpy, pymc, filterpy, cvxpy, scipy, hmmlearn, numpyro, statsmodels)
- LLM: Groq free tier or Ollama local. No OpenAI spend during data integration.
- Compute: local-first, Ray for parallelism when needed.
- Only spend money when the system has demonstrated alpha on backtests with statistical significance.

---

## The 7-Layer Computation Stack

Every implementation decision maps to one of these layers:

```
Layer 1: Surveillance Surface → agent/tools/ (data fetching)
Layer 2: Feature Engineering  → agent/quant/ (signal extraction)
Layer 3: World Model          → agent/models/ (Bayesian network, causal graph)
Layer 4: Signal Fusion        → agent/fusion/ (Kalman, particle filter)
Layer 5: RL Policy            → agent/learning/ (model-based RL, portfolio optimization)
Layer 6: Adversarial          → agent/adversarial/ (manipulation detection, edge decay)
Layer 7: LLM Support          → agent/reasoning/ (text parsing, narration only)
```

When creating new code, know which layer it belongs to. Don't mix layers. A tool should not contain math. A model should not fetch data. Clean separation enables independent testing and swappability.
