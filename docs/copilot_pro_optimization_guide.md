---
title: Copilot Pro+ Advanced Optimization Guide
tags:
  - doc/research
  - topic/workflow
---

# Copilot Pro+ Advanced Optimization Guide

This guide covers the **non-trivial techniques** for getting maximum value from GitHub Copilot Pro+. The quick-win setup files are already in this repo. This document explains the deeper strategies.

---

## 1. Model Selection Strategy

Copilot Pro+ gives you access to multiple models. Using the wrong model for a task wastes premium requests.

### When to Use What

| Task Type | Model | Why |
|-----------|-------|-----|
| Multi-file architecture, complex refactors | **Claude Opus 4** or **o1-pro** | Deep reasoning, long context |
| Math-heavy quant code (BOCPD, HMM, filters) | **Claude Opus 4** | Best at formal/mathematical reasoning |
| Quick edits, boilerplate, test generation | **GPT-4.1** or **Claude Sonnet** | Fast, lower quota cost |
| Code review, explanation, docs | **Any fast model** | Doesn't need deep generation |
| Debugging with stack traces | **Claude Opus 4** | Better at tracing through code |

### How to Switch Models
- In VS Code: click the model name in the chat panel header to switch
- Keyboard: use the model picker dropdown before sending a message
- Default to a fast model (Sonnet/GPT-4.1), switch to Opus/o1-pro only for complex tasks

### Quota Management
- Premium requests (Opus, o1-pro) are limited. Fast model requests are much cheaper.
- Rule of thumb: **80% of your requests should use fast models.** Reserve premium for architecture decisions, complex algorithms, and multi-file refactors.
- Check your usage at https://github.com/settings/copilot

---

## 2. Session Management (The 2-Hour Rule)

LLM quality degrades in long sessions because:
1. Context window fills with old, potentially contradictory information
2. The model starts "forgetting" earlier instructions
3. Accumulated errors compound

### The Session Protocol

**Before starting a session:**
```
Read `[[project_memory]]` and the latest `docs/memory/chat_checkpoint_*.md`.
Read `tasks/active/<current_task>.md`.
Continue from the last completed step.
```

**During a session (every 1-2 hours or at natural breakpoints):**
- Ask the agent: "Write a checkpoint to `[[chat_checkpoint_YYYY-MM-DD]]`"
- The checkpoint should contain: what was accomplished, current state, next steps, any open issues

**When to start a fresh session:**
- After completing a feature or major sub-phase
- When the agent starts repeating itself or making mistakes it didn't make earlier
- After ~2 hours of continuous work
- When switching between different features

**Starting the new session:**
- Open a new chat (don't continue the old one)
- First message: "Read checkpoint at `docs/memory/chat_checkpoint_<date>.md` and task file at `tasks/active/<task>.md`. Resume from step X."
- This gives the agent fresh context without the accumulated noise

### Task Files as Cold-Start Artifacts

Your task files in `tasks/active/` should be written so that **any session can pick up the work cold**. This means:
- Each step has a clear `[ ]` / `[x]` status
- The spec reference is linked
- Any interim decisions or gotchas are noted inline
- The task file is the single source of truth, not the chat history

---

## 3. Context Engineering (Deep Dive)

### The `#` Reference System

These are the most powerful tools for reducing wasted agent exploration:

| Reference | What It Does | When to Use |
|-----------|-------------|-------------|
| `#file:path/to/file.py` | Injects entire file contents | When the agent needs to see a specific file |
| `#folder:agent/quant` | Injects all files in folder | When the agent needs module-wide context |
| `#selection` | Current editor selection | When discussing specific code |
| `#terminal` | Recent terminal output | When debugging test failures |
| `#problems` | VS Code error diagnostics | When fixing compile/lint errors |
| `#codebase` | Semantic search over workspace | When you don't know where something is |
| `#changes` | Git diff of unstaged changes | When reviewing what you just did |
| `#<prompt-file>` | Invokes a .prompt.md template | For repeatable workflows |

**Key insight:** Every `#` reference you provide is context the agent doesn't have to discover via search tools. Search tools cost tokens and time. Direct references are instant.

### Prompt Quality Patterns

**Bad prompt (wastes 3-5 requests on exploration):**
> "Add VPIN calculation"

**Good prompt (agent starts implementing immediately):**
> "Add `compute_vpin()` to `#file:agent/quant/liquidity.py`. It should take a DataFrame with columns `price`, `volume`, `timestamp` and return a Series of VPIN values using bulk volume classification with `n_buckets=50`. Follow the pattern in `#file:agent/quant/scoring.py`. Write tests in `tests/test_liquidity_edge.py`. Per spec step 3.2 in `#file:liquidity_spec`."

### The "Spec Then Execute" Deep Pattern

This is the highest-value workflow for complex features:

1. **Research session** (use fast model):
   - "Use `@quant-researcher` — research <topic>, write to `docs/research/<topic>.md`"
   - Agent reads code, writes analysis. Cheap requests.

2. **Spec session** (use fast model):
   - "Read `#file:docs/research/<topic>.md`. Write implementation spec to `docs/specs/<topic>_spec.md` with atomic numbered steps."
   - Agent converts research into ordered steps. Still cheap.

3. **Implementation session** (switch to premium model):
   - "Use `#next-step` prompt. Task file: `#file:tasks/active/<task>.md`."
   - Premium model now has a precise, constrained task. No wasted exploration = fewer premium requests burned.

**Result:** You use maybe 5-10 fast model requests for research/spec, then each implementation step takes 1-2 premium requests because the agent knows exactly what to do. Without this pipeline, you'd burn 10-20 premium requests on a single feature because the agent keeps exploring and re-deriving.

---

## 4. Custom Agents (How They Work)

Custom agents (`.agent.md` files) are already created in `.github/agents/`. Here's how to use them effectively:

### Invoking Custom Agents
In the chat panel, type: `@quant-researcher`, `@test-writer`, or `@code-reviewer` followed by your request.

### Why They Matter
- **Tool restrictions**: The quant-researcher agent can't modify code, so it can't accidentally break things during research.
- **Focused expertise**: Each agent has domain-specific instructions, so you don't need to repeat them every time.
- **Composability**: You can chain agents: research → review → implement → test-write

### When to Create New Agents
Create a new agent when you find yourself:
- Repeating the same instructions across multiple prompts
- Needing to restrict an agent's capabilities for safety
- Wanting a specific persona for a recurring task type

### Agent File Structure
```yaml
---
description: "One-line description shown in agent picker"
tools:        # Restrict which tools this agent can use
  - read_file
  - grep_search
  # ... only the tools this agent needs
---

# Agent Name

System prompt with specific expertise and rules.
```

---

## 5. TaskSync Integration (Optional but High-Value)

### What It Gives You
TaskSync solves the "idle waiting" problem. Without it:
- You send a prompt → wait 30-120 seconds → read output → type next prompt → wait...
- Dead time between requests adds up to hours per day

With TaskSync:
- Queue 5 tasks → agent works through them → you review when ready
- Autopilot: agent auto-approves file edits and terminal commands → no click-waiting
- Remote access: monitor/steer from your phone while the agent works on your desktop

### Installation
```bash
# From VS Code Marketplace
# Search: "TaskSync" by 4regab
# Or: ext install 4regab.tasksync-chat
```

### Configuration for TirraMind
After installing, configure these settings:
1. **Auto-approve**: Enable for file edits and terminal commands (since our workflow uses tests as safeguards, not manual approval)
2. **Session timeout warning**: Keep enabled (warns at 2 hours)
3. **Smart Queue Mode**: Enable — this is the core feature

### Workflow With TaskSync
```
1. Open TaskSync sidebar
2. Queue your tasks:
   - "Execute step 3.1 from [[quant_training_ground]]"
   - "Execute step 3.2 from [[quant_training_ground]]"
   - "Execute step 3.3 from [[quant_training_ground]]"
   - "Run @code-reviewer on agent/quant/liquidity.py"
3. Enable autopilot
4. Walk away (or monitor from phone via remote access)
5. Come back, review the diffs and test results
```

### Without TaskSync (Manual Equivalent)
If you don't want to install the extension, you can approximate the pattern:
- Write all steps in the task file with `[ ]` checkboxes
- Tell the agent: "Work through steps 3.1-3.5 in `#file:tasks/active/<task>.md` sequentially. Don't stop between steps. Mark each step done as you complete it."
- Enable auto-approve in VS Code settings
- Set `"chat.agent.maxRequests": 999`

---

## 6. MCP Servers (Advanced)

MCP (Model Context Protocol) servers give Copilot agent new tools beyond the built-in ones.

### Relevant MCP Servers for TirraMind

| Server | What It Does | Value |
|--------|-------------|-------|
| **Filesystem MCP** | Read/write files outside workspace | Manage data cache, config files in home dir |
| **SQLite MCP** | Query SQLite databases directly | Query pipeline store without Python wrapper |
| **Fetch MCP** | HTTP requests from the agent | Test API endpoints, fetch data source docs |
| **Memory MCP** | Persistent key-value memory | Alternative to your checkpoint system |

### Setup (Example: SQLite MCP)
1. Add to `.vscode/mcp.json`:
```json
{
  "servers": {
    "sqlite": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-sqlite", "--db-path", "./data/pipeline.db"]
    }
  }
}
```

2. Once configured, the agent can directly query your pipeline database in agent mode.

### When to Add MCP Servers
- When you find yourself repeatedly copying data from external tools into chat
- When the agent needs to interact with a system the built-in tools can't reach
- When you want to automate a workflow that currently requires manual steps

---

## 7. The "Diff Review" Pattern

After any non-trivial implementation, always do:

> "Show me a summary of every file you changed, what you changed, and why. Flag anything that might break existing functionality."

This costs one request but catches:
- Unnecessary changes (agent "improving" code you didn't ask about)
- Missing test updates
- Accidental import additions
- Style inconsistencies

Combine with `@code-reviewer` for the most thorough review.

---

## 8. Measuring and Improving

### Track What Works
After each session, briefly note in your checkpoint:
- How many requests were used
- Which model was used for what
- What worked well / what wasted time

### Common Waste Patterns
| Pattern | Fix |
|---------|-----|
| Agent explores codebase for 5 requests | Use `#file:` and `#folder:` references |
| Agent re-derives architecture every session | Write checkpoints, use task files |
| Agent makes unnecessary "improvements" | Add to instructions: "only change what's requested" |
| Premium model used for simple edits | Switch to fast model for boilerplate |
| Long sessions degrade | Start fresh every 1-2 hours |
| Agent asks clarifying questions | Front-load context in the prompt |

---

## Summary: The Optimal Loop

```
1. Start session → read checkpoint + task file (fast model)
2. Research phase → @quant-researcher (fast model)
3. Spec phase → write spec (fast model)
4. Switch to premium model
5. Implementation → #next-step prompt (premium model, 1 step at a time)
6. Test → @test-writer (fast model)
7. Review → @code-reviewer (fast model)
8. Write checkpoint → start new session
```

This workflow means you spend **~70% of requests on fast models** and only use premium for the actual hard implementation steps. Maximum output per premium request.
