---
title: How to Operate Efficiently
tags:
  - doc/wiki
  - topic/workflow
  - topic/engineering
  - layer/llm-support
---

# How to Operate Efficiently: A Complete Guide to AI-Assisted Software Development

*Distilled from 12+ months of building a complex autonomous system with AI pair-programming. Every principle here was learned the hard way — by violating it and paying the price in wasted context, broken code, or lost momentum.*

---

## Table of Contents

1. [The Core Insight](#the-core-insight)
2. [Atomic Decomposition](#atomic-decomposition)
3. [The Research → Spec → Implement Pipeline](#the-research--spec--implement-pipeline)
4. [Context Window Management](#context-window-management)
5. [Knowledge Base as Infrastructure](#knowledge-base-as-infrastructure)
6. [Working with AI Code Assistants](#working-with-ai-code-assistants)
7. [Task Management That Actually Works](#task-management-that-actually-works)
8. [The Debugging Protocol](#the-debugging-protocol)
9. [Testing Discipline](#testing-discipline)
10. [Factual Verification — The Zero-Hallucination Policy](#factual-verification--the-zero-hallucination-policy)
11. [Internet Research Protocol](#internet-research-protocol)
12. [Session Management and Checkpoints](#session-management-and-checkpoints)
13. [Code Architecture Principles](#code-architecture-principles)
14. [Cost Discipline](#cost-discipline)
15. [Handling Complex Logic and Math](#handling-complex-logic-and-math)
16. [Anti-Patterns — What Will Burn You](#anti-patterns--what-will-burn-you)
17. [Operational Checklists](#operational-checklists)

---

## The Core Insight

Most software fails not from bad code, but from unclear thinking executed too soon.

The single most valuable operational change you can make is to **separate thinking from doing**. Research is not implementation. Planning is not execution. Understanding is not building. When these blur together, you get code that "sort of works" but is expensive to maintain, extend, or trust.

This sounds obvious. Nobody does it. The pressure to "just start coding" is overwhelming, especially with AI assistants that make code appear instantly. Resist it. The 30 minutes you spend understanding before touching a file saves you 3 hours of debugging, rewriting, and re-understanding later.

---

## Atomic Decomposition

**Break everything down to the smallest possible unit of work.**

This is the single highest-leverage habit in all of software engineering. It applies to everything: planning, specs, implementation, debugging, research.

### The Rules

- A task that takes more than ~2 hours of focused work is too big. Split it.
- If a step description contains the word "and," it's probably two steps.
- Each step should change one thing, test one thing, prove one thing.
- Prefer 10 tiny PRs over 1 medium PR. Prefer 5 tiny functions over 1 clever one.
- When in doubt, break it down further.

### How to Tell If Your Step Is Small Enough

Ask: **"Can I describe a one-line test that proves this step worked?"**

- ✅ "After this step, `test_parse_timestamp_utc()` passes" — good, atomic.
- ❌ "After this step, the data pipeline handles all edge cases" — too vague, too big.

### How to Break Down a Feature

1. **Research** — Read relevant code, search docs, write findings. No code changes.
2. **Spec** — Transform research into ordered atomic steps with clear acceptance criteria.
3. **Decompose into task file** — Numbered steps, each independently verifiable.
4. **Implement one step at a time** — Make the change → test it → mark done → next.

### The Naming Convention That Works

`<phase>.<step>: <verb> <specific thing>`

Examples:
- `2.1: Add timestamp normalization to parser module`
- `2.2: Write edge case tests for timezone handling`
- `2.3: Integrate parser into ingestion pipeline`

This format eliminates ambiguity about scope and makes progress visible in any tracking system.

---

## The Research → Spec → Implement Pipeline

**Never skip phases. The ordering exists to prevent expensive mistakes.**

```
User Request → Research Phase → Specification Phase → Implementation (one atomic step at a time)
```

### Why Three Phases?

Each phase produces a different artifact with a different purpose:

| Phase | Output | Purpose |
|-------|--------|---------|
| Research | Research doc | Understand what exists, what's missing, what connects to what |
| Spec | Specification | Transform understanding into precise, ordered steps |
| Implementation | Code + Tests | Execute steps one at a time against the spec |

**The key insight: each phase catches different categories of errors.**

- Research catches "we're solving the wrong problem" and "this already exists."
- Spec catches "the steps are out of order" and "we forgot an edge case."
- Implementation catches "this doesn't actually work" and "the API is different than expected."

When you skip phases, those error categories don't go away — they just get caught later, when they're 10x more expensive to fix.

### When Can You Skip Phases?

Only for **truly trivial** changes:
- Typo fixes
- Comment wording changes
- Narrowly scoped single-file changes with no behavior or interface change

**If there's any doubt, treat the work as non-trivial.** The cost of doing unnecessary research is 15 minutes. The cost of skipping necessary research is hours of rework.

### Research Phase (Before Any Code Changes)

1. Read only the files relevant to the requested feature.
2. For new features or unfamiliar technology: **search GitHub and authoritative documentation first using multiple keyword variants.** Record repos, docs, and search terms.
3. Analyze project structure and dependencies.
4. Identify the correct insertion points for new code.
5. Record findings in a research document.

**Research document structure:**
```markdown
# Feature: <name>

## Current Architecture
- (relevant modules, patterns, dependencies)

## Observations
- (what exists, what's missing, what connects to what)

## Risks
- (edge cases, breaking changes, security concerns)

## External References
- (repos, docs, papers consulted — with URLs)

## Open Questions
- (things that need resolution before implementation)
```

**No code is edited during this phase.** This is a hard rule. The temptation to "just fix this one thing while I'm here" is how research phases become unfinished implementation phases.

### Specification Phase (Before Any Code Changes)

Transform research into a precise implementation plan.

```markdown
# Spec: <feature_name>

## Goal
What the feature must accomplish. One paragraph.

## Files Affected
List of files to create or modify. Nothing else gets touched.

## Implementation Steps
Ordered, atomic steps. Each has a one-line acceptance test.

## Edge Cases
Possible failure scenarios and how they're handled.

## Testing Plan
How the feature is validated. Both happy path and edge cases.
```

**Spec rule: if you can't explain the full implementation plan before writing code, you don't understand the problem well enough yet.** Go back to research.

### Implementation Phase

1. Follow the spec strictly.
2. Modify only files listed in the spec.
3. Do not re-analyze architecture during coding. That's what research was for.
4. If issues arise, **update the spec first**, then continue implementation.
5. After each sub-phase completes, write and run an edge case test suite.

---

## Context Window Management

**LLM context windows fill up. Every wasted token is a lost thought. Protect context ruthlessly.**

This section applies whether you're using Copilot, Cursor, Claude, ChatGPT, or any other AI assistant for development. Context is your most scarce resource.

### The Rules

1. **Move reasoning into files, not chat.** Analysis goes in research docs. Plans go in specs. Don't repeat them in conversation. A 500-line analysis in chat is wasted — it'll scroll off. The same 500 lines in a file persists forever and can be referenced by name.

2. **Reference documents instead of re-explaining.** Say "per the spec, step 2.3" — don't re-describe what step 2.3 is. This saves hundreds of tokens per reference.

3. **Read only necessary files.** Don't let the AI explore the whole codebase when you need one function. Be directive: "Read lines 40-80 of parser.py" not "look at the parser module."

4. **Start new sessions after completing a feature.** Old context becomes stale and wastes tokens. A fresh session with a checkpoint file is 10x more efficient than a bloated session with 50 messages of irrelevant history.

5. **Write checkpoint files at natural breakpoints** so the next session can cold-start without re-reading everything.

6. **Keep task files as the single source of truth** for what's done and what's next. The task file should be enough to resume work cold — no session history required.

7. **Don't re-analyze architecture during implementation.** That's what the research phase was for. If something is wrong, update the spec first.

### Batching Strategy

The AI assistant's capability determines optimal batch size:

- **Too small:** Feeding one trivially simple step per message wastes overhead tokens on boilerplate, confirmation, and context re-loading.
- **Too large:** Feeding 20 steps in one message causes blurred scope, missed details, and harder debugging when something fails.
- **Sweet spot:** Group 3-5 adjacent, testable changes that naturally relate to each other. Enough to maintain momentum, small enough that failures are isolatable.

Adjust this based on the model's capability. More advanced models handle larger batches; simpler models need smaller ones.

### Token-Saving Patterns

| Instead of... | Do this... | Savings |
|--------------|-----------|---------|
| Pasting full error logs in chat | "The error is [type] on line [N] of [file]" | ~500 tokens |
| Asking the AI to read 10 files to "understand the project" | Point it at the 2 files relevant to the current task | ~2000 tokens |
| Re-explaining what you're building each message | "Continue from step 3.2 in [[feature]]" | ~300 tokens |
| Debugging in chat | Write a failing test and point the AI at it | ~400 tokens |

---

## Knowledge Base as Infrastructure

**Your documentation isn't an afterthought — it's a navigation system.**

### The Cross-Linked Knowledge Graph

Every project benefits from three types of interconnected documents:

1. **Research notes** — What you learned about a topic, with sources.
2. **Specifications** — Precise implementation plans derived from research.
3. **Task files** — The current state of work, resumable cold.

These three form a **triad** for every feature: `research → spec → task`. They link to each other. You can trace any piece of code back to the research that justified it.

### Why Bidirectional Links Matter

When you use `[[wiki links]]` (or any cross-referencing system), you create bidirectional navigation:
- "What depends on module X?" → check what links *to* X.
- "What was the research behind this decision?" → follow the link *from* the task.
- "What other features touch this area?" → check the backlinks.

This replaces "grep through the entire codebase hoping to find something relevant" with precise, O(1) navigation.

### Tools like Obsidian, Notion, or GitHub Wiki

Any tool that supports cross-linking works. The choice of tool matters less than the discipline:

1. **Every document gets metadata.** Tags, categories, status. Something machine-searchable.
2. **All cross-references are explicit links**, not "see the parser docs" in plain text.
3. **Every research, spec, and task file links to its counterparts** in a Related section.
4. **When you create, rename, or delete a file, update all references.** Broken links are worse than no links — they waste time.

### Navigation Shortcuts

Before reading files linearly:
- **Search by tag** to find all docs on a topic.
- **Check backlinks** to see what depends on a file.
- **Follow the triad chain** (research → spec → task) to understand a feature end-to-end.
- **Read the latest checkpoint** to understand current project state.

This is dramatically faster than scanning directories.

---

## Working with AI Code Assistants

**The AI is a power tool. Power tools are dangerous without a framework.**

### The Collaboration Model

Think of AI-assisted development as having two roles:

| Role | Responsibility |
|------|---------------|
| **You (human)** | Intent, judgment, verification, architecture decisions |
| **AI assistant** | Execution, search, boilerplate, pattern application |

The failure mode is letting the AI make judgment calls it isn't equipped for. The AI doesn't know your business context, your team's conventions, or what trade-offs matter. It's incredibly fast at doing what you tell it — and incredibly fast at doing the wrong thing if you don't tell it clearly.

### Rules for Effective AI Collaboration

1. **Use chat for thinking, agent mode for execution.** Brainstorming, research, and spec work stay in planning mode. Bounded, testable steps move to implementation mode.

2. **Keep one problem per step.** If a request combines multiple distinct problems, decompose before asking the AI to implement.

3. **Be prescriptive about scope.** "Edit parser.py lines 40-60 to add timezone handling" is 10x better than "fix the date parsing issue." The more precise your instruction, the less likely the AI wastes context exploring wrong paths.

4. **Verify, don't trust.** Run the tests. Read the diff. The AI can produce plausible-looking code that has subtle bugs. Your job is verification.

5. **Ask the AI to explain changes.** Don't accept output on faith. "Why did you choose this approach?" and "What edge cases does this handle?" are cheap questions that catch expensive mistakes.

6. **Front-load context.** Give the AI the relevant file contents, error messages, and constraints *before* asking it to act. Re-prompting because you forgot to mention a constraint wastes an entire round-trip.

7. **Don't let the AI over-engineer.** AI assistants love to add extra error handling, docstrings, type annotations, and "improvements" to code you didn't ask them to change. Be explicit: "Only change the lines I specified. Don't modify anything else."

### What AI Is Good At

- Boilerplate generation (CRUD, test scaffolding, repetitive patterns)
- Finding code in large codebases (search + read)
- Applying known patterns to new contexts
- Explaining unfamiliar code
- Generating edge case test scenarios
- Refactoring when given clear instructions
- Running commands and interpreting their output

### What AI Is Bad At

- Making architectural decisions (it doesn't know your constraints)
- Judging "is this good enough" (it has no business context)
- Remembering decisions from 50 messages ago (context window)
- Knowing when to stop (it will keep "improving" forever)
- Verifying its own output (it generates confidently regardless of correctness)
- Understanding subtle business logic (it pattern-matches from training data)

### The Instruction File System

For projects with repeated AI interaction, create instruction files that describe conventions:

- **Project-level instructions** (`.github/copilot-instructions.md` or equivalent): Architecture rules, workflow requirements, coding standards that apply everywhere.
- **Folder-level instructions** (`.instructions.md` in module directories): Module-specific conventions, patterns, import rules.
- **Agent definitions** (`AGENTS.md`): Different roles the AI should assume for different tasks (researcher, implementer, reviewer, test-writer).

This is an investment that pays off within days. Instead of re-explaining conventions every session, the AI reads them automatically.

---

## Task Management That Actually Works

### The Task File as Source of Truth

A task file should answer one question: **"If I'm starting cold, what do I do next?"**

```markdown
# Task: <name>

Status: active | completed
Research: <link to research doc>
Spec: <link to spec doc>

## Steps

- [x] 1.1: Add base parser class with timestamp support
- [x] 1.2: Write unit tests for parser edge cases
- [ ] 1.3: Integrate parser into ingestion pipeline
- [ ] 1.4: Write integration test for full pipeline
- [ ] 1.5: Update configuration to expose parser options
```

**The rules:**
- Steps are numbered and named with the `<verb> <specific thing>` convention.
- Completed steps are checked off immediately — not in batches.
- The task file is updated *during* implementation, not after.
- Anyone should be able to open this file and know exactly where work stands.

### Why Not Use Jira/Linear/GitHub Issues?

You can — and should for team coordination. But the task `.md` file serves a different purpose: it's the **AI's context anchor**. When you tell the AI "continue from step 1.3," it reads the file and knows exactly what's been done and what's next. Jira can't do that.

Use both: the issue tracker for team visibility, the task file for session-level execution context.

### Completion Checklist

Before marking a task complete:
1. All steps are checked off.
2. All tests pass (including edge case suite).
3. The spec's acceptance criteria are met.
4. Documentation is updated if interfaces changed.
5. The task file's status is updated.

---

## The Debugging Protocol

**When stuck, stop guessing. Switch to structured diagnosis.**

This is the single most common failure mode in AI-assisted development: the AI suggests a fix, it doesn't work, so it suggests another fix, that doesn't work either, and you're 8 attempts deep with a codebase that's now modified in 4 different places and you don't know which changes helped and which hurt.

### The Hard Rule

**After 2 failed attempts at the same fix, mandatory switch to debug mode.**

Do not attempt a 3rd fix without first completing all four diagnostic steps. This is a hard rule, not a suggestion.

### The Four Steps

**Step 1: Reproduce.** Write a minimal test or script that triggers the exact failure. If you can't reproduce it, you can't fix it. Don't guess — prove the bug exists with a specific, repeatable input → output pair.

**Step 2: Instrument.** Add targeted logging, print statements, or assertions to narrow down where the failure occurs. Binary search through the code path: is the input correct? Is the middle correct? Where exactly does reality diverge from expectation?

**Step 3: Hypothesize.** Based on the instrumentation output, form a single, specific hypothesis: "The failure occurs because X is None when it should be a list, because function Y returns None when the database query returns zero rows."

**Step 4: Fix and regress.** Fix the specific hypothesis. Run the original reproduction test. Run the full test suite. If the fix creates new failures, you're in a different problem — don't stack fixes.

### Why This Works Better Than Guessing

Guessing is O(n) in the number of possible causes. Instrumentation is O(log n) because you binary-search the code path. For any bug with more than 3 possible causes, structured diagnosis is faster.

More importantly: each diagnostic step produces *information*. Even if the hypothesis is wrong, the instrumentation tells you something. Guessing produces nothing when wrong — just a new broken state.

---

## Testing Discipline

**After every sub-phase implementation, write and run an extensive edge case test suite. This is mandatory, not optional.**

### What to Cover

Every test suite should address:
- **Invalid inputs** — None, empty strings, wrong types, negative numbers
- **Boundary values** — zero, one, max int, empty collections, single-element collections
- **Error paths** — network failures, missing files, permission denied, timeout
- **Security scenarios** — injection attempts, path traversal, unauthorized access
- **Type mismatches** — string where int expected, list where dict expected
- **Missing required fields** — partial payloads, null required fields
- **Exception handling** — verify exceptions are caught, logged, and don't leak internal state
- **Concurrency** — if applicable, race conditions and deadlocks
- **Domain-specific edge cases** — whatever is weird about your specific data/logic

### The Testing Workflow

1. Implement the feature.
2. Write happy-path tests (basic functionality works).
3. Write edge case tests (everything above).
4. Run all tests. Fix failures.
5. **Only then** mark the step as complete.

**Don't batch test-writing.** Write tests immediately after each implementation step. Tests written 3 days after implementation are less effective because you've forgotten the edge cases you considered during implementation.

### Test Naming Convention

`test_<what>_<scenario>_<expected>`

Examples:
- `test_parse_timestamp_empty_string_raises_valueerror`
- `test_fetch_data_network_timeout_returns_cached`
- `test_score_negative_input_clamps_to_zero`

Good test names are documentation. You should be able to read the test name and know what it verifies without reading the test body.

---

## Factual Verification — The Zero-Hallucination Policy

**Never assert facts about external systems without verification.**

This is the rule that prevents the most expensive bugs. AI assistants hallucinate plausible-sounding but incorrect facts about APIs, libraries, and services. A single wrong assumption about an API response format can waste an entire day.

### What Must Be Verified

Before writing code that depends on any of these, verify against a real source:

- **API endpoints, parameters, response schemas, auth requirements, rate limits**
- **Library interfaces, function signatures, class hierarchies, defaults**
- **Financial instrument identifiers** (tickers, ISIN, contract specs)
- **Data formats and availability windows**
- **Mathematical method properties** (convergence, complexity, assumptions)
- **Regulatory structures and filing formats**

### How to Verify

In order of preference:

1. **Test it in the terminal.** Run the actual call. This is free and definitive.
   ```bash
   python -c "import requests; print(requests.get('https://api.example.com/v1/status').json())"
   ```

2. **Read the official documentation.** Go to the source. Not Stack Overflow, not a blog post — the official docs.

3. **Search for corroborating sources.** If the docs are ambiguous, find 2-3 sources that agree.

4. **If unverifiable, say so explicitly.** Mark the claim as "UNVERIFIED" and suggest manual verification. Never fill the gap with a plausible guess.

### Record What You Verified

In your research doc:
```markdown
## Verified Facts
- API endpoint: `GET /v2/market/tickers` — verified against official docs (URL, date)
- Response includes `last_price` field as float — tested in terminal 2024-03-15
- Rate limit: 100 req/min — per docs, UNTESTED at scale
```

This prevents re-verification in future sessions and makes assumptions auditable.

---

## Internet Research Protocol

**Use the cheapest tool that gets you the answer. Never pay for what you can get for free.**

### The Decision Tree

```
Do you have a specific URL?
  YES → Fetch the page directly (free)
  NO  → Do you know the official docs URL?
          YES → Fetch it directly (free)
          NO  → Search to find the right URL
                  → Then fetch the page (free)
```

### Rules

1. **When you have a URL, fetch it directly.** Don't route through search for something you can access directly.
2. **Prefer terminal testing over search.** Running `python -c "import lib; print(lib.some_func())"` is free and gives ground truth.
3. **Cache knowledge within a session.** If you searched for something once, reference the earlier result — don't search again.
4. **Batch related searches.** Instead of 5 separate searches for related topics, combine: "topic1 topic2 topic3 comparison."
5. **Write findings to files immediately.** Your research doc is the persistent search cache. Prevents re-searching in future sessions.
6. **Official docs > blog posts > Stack Overflow.** The further from the source, the more likely the information is outdated or wrong.

---

## Session Management and Checkpoints

**Every session should be bootable from cold storage.**

### Why Sessions Should Be Short

Long sessions accumulate stale context. After 20-30 messages, the AI is carrying:
- Decisions that were revised but still occupy context
- File contents that were since changed
- Debugging tangents that are irrelevant to the current step

**Start a new session after completing a feature** or at any natural breakpoint. The 2 minutes to write a checkpoint and start fresh saves 20 minutes of fighting stale context.

### Checkpoint Format

```markdown
# Checkpoint: <date> — <what was accomplished>

## Completed This Session
- Step 2.1: Added parser class (tests pass)
- Step 2.2: Wrote edge case suite (14 tests, all pass)

## Current State
- Working on step 2.3 (integration)
- Parser module is complete and tested
- Ingestion pipeline has a placeholder at line 47 of pipeline.py

## Next Steps
- Step 2.3: Wire parser into pipeline at the placeholder point
- Step 2.4: Integration test

## Open Issues
- Timezone handling for Australian Eastern is untested (low priority)

## Key Decisions Made
- Chose dateutil over arrow for parsing (lighter dependency, sufficient API)
- Used UTC internally, convert only at display boundaries
```

### The Cold-Start Procedure

When beginning a new session:

1. Read the latest checkpoint.
2. Read the active task file.
3. Follow links to the relevant spec and research docs.
4. Start from the first unchecked step.

This should take under 2 minutes. If it takes longer, your checkpoint was incomplete.

---

## Code Architecture Principles

### Explicit File Responsibility

Choose file names and module boundaries that make ownership obvious. Every module should have a single clear purpose. If you can't describe what a file does in one sentence, it probably does too much.

**The test:** If a new developer reads the filename, do they know what's inside?

- ✅ `timestamp_parser.py`, `user_auth.py`, `cache_store.py`
- ❌ `utils.py`, `helpers.py`, `core.py`, `manager.py`

### Layer Separation

Whatever your architecture has as layers (data access, business logic, API, etc.), don't mix them:

- A data-fetching module should not contain business logic.
- A model should not fetch its own data.
- A controller should not contain database queries.

Clean separation enables independent testing, swappability, and maintainability. When you mix layers, a change in one area breaks another, and tests become integration-level nightmares.

### Don't Over-Engineer

- Don't add features beyond what was requested.
- Don't refactor code you didn't change.
- Don't add docstrings or type annotations to code you didn't modify.
- Don't add error handling for scenarios that can't happen.
- Don't create abstractions for one-time operations.
- Don't build a generic framework when you need a specific function.

**The rule:** If you can't point to a requirement that justifies this code, delete it.

### Prefer Learned Structure Over Hand-Coded Logic

When building systems with any kind of intelligence (scoring, ranking, prediction, classification):

- **Hard-code:** Schemas, invariants, safety constraints, explicit factual relationships directly stated by source data.
- **Learn:** Ambiguous relations, weighting, scoring, predictive behavior, anything where the "right answer" depends on patterns in data.

Hand-coded intelligence is brittle — it breaks when conditions change. Learned structure adapts. The investment in setting up a learning loop pays off faster than the investment in hard-coding edge cases.

---

## Cost Discipline

**$0 until proven value.**

Before spending money on any tool, service, or API:

1. **Is there a free alternative?** Open-source versions, free tiers, public APIs?
2. **Is it proven necessary?** Have you validated the approach with free data first?
3. **Is the ROI measurable?** Can you quantify what you gain?

### In Practice

- Start with free data sources and public APIs.
- Use open-source libraries before commercial ones.
- Run models locally before paying for cloud compute.
- Only spend money when the system has demonstrated value with free resources.

**The cheapest data is often the most valuable** because nobody else looks at it. Premium data sources are already priced into competitors' models.

---

## Handling Complex Logic and Math

When the work moves beyond simple CRUD into scoring, estimation, inference, optimization, or statistical control:

### Explanation Is Mandatory

For every non-trivial mathematical choice, document:

1. **What quantity is being estimated** — the objective or statistic.
2. **The assumptions** — under what conditions this is valid.
3. **Why this formulation** — what alternatives exist and why this one was chosen.
4. **Numerical stability concerns** — overflow, underflow, precision loss.
5. **Failure modes** — when this method gives wrong answers.

### Present Options Before Committing

For substantive math code, compare the main implementation choices:

| Aspect | Option A | Option B | Option C |
|--------|----------|----------|----------|
| Approach | Exact | Approximate | Empirical |
| Complexity | O(n³) | O(n log n) | O(n) |
| Accuracy | Exact | 95th percentile | Data-dependent |
| When it fails | n > 10K | Non-stationary data | Small sample |

State the trade-offs and explain why one choice is preferred in **this** codebase, with **these** constraints.

### Anchor to Trusted Sources

Before applying a statistical test, estimator, filter, or optimization method:

1. Identify the trusted source (paper, textbook, library docs).
2. Explain why that source is trustworthy for this use case.
3. Distinguish source-backed theory from your engineering choices.

"I'm using BOCPD because Adams & MacKay (2007) established it for exact online changepoint detection, which matches our streaming requirement. The hazard function is our engineering choice — we use constant hazard because our regime changes aren't periodic."

---

## Anti-Patterns — What Will Burn You

### 1. "Let Me Just Start Coding"
**Symptom:** You skip research and spec, start implementing, realize the approach is wrong at step 5, and rewrite everything.
**Cost:** 3-10x the time of doing research first.
**Fix:** Follow the pipeline. Always.

### 2. "The AI Will Figure It Out"
**Symptom:** You give a vague prompt, the AI generates a large block of code, you paste it in, and spend 2 hours debugging it.
**Cost:** More time than writing it yourself would have taken.
**Fix:** Give precise, scoped instructions. Verify output. Use the AI as a power tool, not an oracle.

### 3. "I'll Write Tests Later"
**Symptom:** You implement 5 features, then discover the second one has a subtle bug that affects the other three.
**Cost:** Debugging cascading failures across untested code.
**Fix:** Test after every step. Not after every feature. After every *step*.

### 4. "Let Me Just Try One More Fix"
**Symptom:** The bug fix didn't work, so you try another, and another, and now you have 6 uncommitted changes and no idea which ones are correct.
**Cost:** Starting over from a clean state, losing all work.
**Fix:** After 2 failed fixes, switch to the debugging protocol. Reproduce → Instrument → Hypothesize → Fix.

### 5. "I Know What That API Returns"
**Symptom:** You write code based on assumed API behavior, deploy it, and it fails because the response format changed, or you remembered wrong.
**Cost:** Runtime failures, often in production.
**Fix:** Verify against real sources. Test in terminal. Read the docs.

### 6. "While I'm Here, Let Me Improve This"
**Symptom:** You open a file to fix a bug and notice the code could be cleaner, so you refactor it, which breaks something else, which leads to more changes...
**Cost:** Scope creep that delays the original task and introduces new bugs.
**Fix:** Only change what the current step requires. Note improvements for future tasks.

### 7. "The Long Session"
**Symptom:** You've been in the same AI chat for 40 messages. The AI contradicts earlier decisions. Responses get less relevant. You're spending more time re-explaining context than making progress.
**Cost:** Fighting the context window instead of building features.
**Fix:** Checkpoint and start a fresh session. 2 minutes of checkpoint writing saves 30 minutes of degraded performance.

### 8. "I'll Remember the Decision"
**Symptom:** You make a design decision during coding, don't write it down, start a new session, and can't remember why you chose approach A over approach B.
**Cost:** Re-deriving decisions or, worse, unknowingly reversing them.
**Fix:** Write an Architecture Decision Record (ADR) for any non-obvious choice. Even a 5-line note is better than trusting memory.

### 9. "The Over-Engineered Abstraction"
**Symptom:** You build a generic, extensible framework for something that will only have one implementation. The framework takes longer to build than the feature.
**Cost:** Complexity that serves no purpose. Future developers (including you) will spend time understanding the abstraction before understanding the actual logic.
**Fix:** Build the specific thing. Generalize only when you have 3+ concrete cases that would benefit.

### 10. "Research Without Boundaries"
**Symptom:** You keep reading more docs, more repos, more papers, never quite feeling "ready" to start coding.
**Cost:** Analysis paralysis. The research phase never ends.
**Fix:** Set a time-box. Research answers specific questions — when those questions have answers, research is done. If you've read 5 sources and they agree, you have enough.

---

## Operational Checklists

### Before Starting Any Non-Trivial Feature

- [ ] Research doc exists with current architecture and observations
- [ ] External references (APIs, libraries, methods) are verified, not assumed
- [ ] Spec exists with ordered atomic steps
- [ ] Task file exists with numbered, checkable steps
- [ ] Each step has a one-line acceptance test described
- [ ] Files affected are listed (nothing outside this list gets modified)

### Before Starting Each Implementation Step

- [ ] Step-local references gathered (docs, examples, source code for this specific step)
- [ ] Previous step's tests are passing
- [ ] Step scope is clear: what changes, what's tested, what's NOT touched

### After Each Implementation Step

- [ ] The specific change is made
- [ ] Happy-path test passes
- [ ] Edge case tests written and pass
- [ ] Task file is updated (step checked off)
- [ ] No unrelated changes were made

### Before Marking a Feature Complete

- [ ] All task steps are checked off
- [ ] Full test suite passes (not just the new tests)
- [ ] Spec acceptance criteria met
- [ ] Documentation updated if interfaces changed
- [ ] Task file status updated to complete
- [ ] Checkpoint written for session handoff

### When Starting a New Session

- [ ] Read latest checkpoint
- [ ] Read active task file
- [ ] Follow links to relevant spec and research
- [ ] Identify the next unchecked step
- [ ] Verify test suite still passes before making changes

### When Debugging

- [ ] Confirm the bug with a reproducible test case (Step 1)
- [ ] Add instrumentation to narrow the location (Step 2)
- [ ] Form a specific, testable hypothesis (Step 3)
- [ ] Fix only the hypothesized cause and run full regression (Step 4)
- [ ] If still failing after 2 attempts, stop and re-instrument

---

## Summary: The 10 Commandments

1. **Break everything into atoms.** If a step has "and," it's two steps.
2. **Research before spec. Spec before code. Always.**
3. **Protect context like it's money**, because it is.
4. **Test after every step**, not after every feature.
5. **Verify external facts**, never assume them.
6. **Debug with structure**, not with guessing.
7. **Write things down.** If it's not in a file, it doesn't exist.
8. **Start fresh when context is stale.** Checkpoints are cheap; stale sessions are expensive.
9. **Don't over-engineer.** Build what's needed; nothing more.
10. **The AI is a power tool, not an oracle.** Direct it precisely, verify its output.

---

*This guide is a living document. Update it when you learn something new the hard way.*
