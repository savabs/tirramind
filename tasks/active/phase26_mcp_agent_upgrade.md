---
title: "Task: Phase 26 — MCP Agent Capability Upgrade"
tags:
  - doc/task
  - status/active
  - phase/26
  - topic/agent-tooling
  - layer/surveillance
---

# Task: Phase 26 — MCP Agent Capability Upgrade

Status: active
Research: [[mcp_agent_upgrade]]
Spec: [[mcp_agent_upgrade_spec]]

## Goal

Upgrade the Copilot agent from the original minimal MCP setup to a stable research-first tool stack, using 7 configured MCP servers plus VS Code's built-in fetch capability for web reads.

## Servers Added

| # | Server | Purpose | Cost | Status |
|---|--------|---------|------|--------|
| 1 | **Tavily** | Internet search | FREE | ✅ Configured |
| 2 | **Built-in Fetch** | Webpage reading via VS Code built-in tool | FREE | ✅ Kept |
| 3 | **GitHub** | Code search across all public repos, issues, PRs | FREE (Copilot auth) | ✅ Configured |
| 4 | **Context7** | Up-to-date library docs (pgmpy, scipy, filterpy, etc.) | FREE | ✅ Configured |
| 5 | **Sequential Thinking** | Structured multi-step math reasoning | FREE | ✅ Configured |
| 6 | **Git** | Version control intelligence (log, diff, blame) | FREE | ✅ Configured |
| 7 | **Playwright** | Browser automation for web scraping & testing | FREE | ✅ Configured |
| 8 | **Memory** | Persistent knowledge graph across sessions | FREE | ✅ Configured |

## Steps

### 26.1: Configure all MCP servers in `.vscode/mcp.json`
- [x] Add GitHub MCP (remote, Copilot OAuth)
- [x] Add Context7 (library docs)
- [x] Add Sequential Thinking (math reasoning)
- [x] Add Git MCP (repo intelligence)
- [x] Add Playwright (browser automation)
- [x] Add Knowledge Graph Memory (persistent entity memory)
- [x] Create `.tirra_memory/` directory for Memory server storage
- [x] Remove redundant Fetch MCP server and rely on VS Code built-in fetch tool
- [x] Pin stdio MCP server package versions for startup stability
- [x] Use a deterministic Python path for Git MCP
- [x] Add workspace MCP autostart setting (`chat.mcp.autostart = newAndOutdated`)

### 26.2: Verify each server starts and responds
- [x] Verify GitHub MCP connects via Copilot OAuth (remote server — needs first-use OAuth popup; built-in github_repo covers basics)
- [x] Verify Context7 resolves a library and returns docs (tested: pgmpy resolved to /pgmpy/pgmpy)
- [x] Verify Sequential Thinking tool is available (tested: returned valid thought object)
- [x] Verify Git MCP can run `git_status` on this repo (tested: returned full working tree status)
- [x] Verify Playwright can open a headless page (tested live)
- [x] Verify Memory server can create/read an entity (tested: read_graph returned empty graph — ready for use)
- [x] Re-verify MCP stack after hardening changes (Tavily, GitHub, Context7, Sequential Thinking, Git, Playwright, Memory all responded)

### 26.3: Update agent instructions to leverage new tools
- [x] Update `copilot-instructions.md` Internet Research Protocol with GitHub MCP usage patterns
- [x] Add Context7 usage rules (always use for library API lookups before coding)
- [x] Add Sequential Thinking rules (use for multi-step math derivations)
- [x] Document when to use Git MCP vs terminal git commands
- [x] Document Playwright usage for data tool development/debugging
- [x] Add Memory MCP usage rules (persistent knowledge graph)
- [x] Update AGENTS.md tool permissions with all new MCP servers

### 26.4: Future — Custom TirraMind MCP Server (strategic)
- [ ] Design MCP server exposing entity graph queries
- [ ] Design MCP server exposing pipeline state & cached data
- [ ] Design MCP server exposing backtest runner
- [ ] Implement using `fastapi_mcp` or Python MCP SDK

## Why This Matters

The agent's effectiveness is bottlenecked by its ability to access external knowledge. Each new MCP server removes a friction point:

- **GitHub** → No more Tavily→fetch two-step dance to read OSS code
- **Context7** → Zero-hallucination library APIs (critical for pgmpy/pymc/filterpy)
- **Sequential Thinking** → Better mathematical reasoning on complex derivations
- **Git** → Structured version control instead of raw terminal output
- **Playwright** → Debug and build data scrapers with a real browser
- **Memory** → Remember entity relationships, API quirks, and decisions across sessions
- **Stability hardening** → Fewer overlapping tools, less startup drift, and more deterministic MCP launches

## Related

- [[copilot-instructions]]
- [[AGENTS]]
