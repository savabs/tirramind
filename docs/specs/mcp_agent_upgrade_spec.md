---
title: MCP Agent Upgrade Spec
tags:
  - doc/spec
  - phase/26
  - topic/agent-tooling
  - layer/surveillance
---

# Spec: MCP Agent Upgrade

## Goal
Stabilize MCP usage in this workspace by reducing tool overlap, removing startup nondeterminism from `@latest` resolution, and configuring VS Code MCP settings for explicit workspace-only server management.

## Files Affected
- `.vscode/mcp.json`
- `.vscode/settings.json`
- `[[mcp_agent_upgrade]]`
- `[[phase26_mcp_agent_upgrade]]`

## Implementation Steps
1. Preserve workflow preflight by creating the missing phase 26 research and spec artifacts referenced by the active task.
2. Update `.vscode/mcp.json`:
   - remove the redundant Fetch MCP server
   - pin package versions for Tavily, Context7, Sequential Thinking, Playwright, and Memory
   - replace absolute workspace paths with `${workspaceFolder}` where supported
   - use a deterministic Python path for the Git MCP server
3. Update `.vscode/settings.json` with the supported MCP stability setting:
  - set `chat.mcp.autostart` to `"newAndOutdated"`
  - rely on the current VS Code default for MCP discovery rather than forcing a workspace override
4. Re-test the remaining configured servers with live tool calls.
5. Update the active task file to record the hardening work and verification status.

## Edge Cases
- If a pinned package version is unavailable in the future, VS Code will fail to start that server until the version is updated.
- If the fixed Python interpreter path changes, Git MCP will fail to start and should be updated to the new interpreter.
- If the user intentionally relies on MCP discovery from another app, disabling discovery will hide those extra servers.
- GitHub remote MCP may still intermittently fail due to upstream VS Code or policy/auth issues outside workspace control.

## Testing Plan
- Confirm `.vscode/mcp.json` remains valid and references supported fields from the official schema.
- Live-test each remaining configured MCP server:
  - Tavily search
  - GitHub remote issue search
  - Context7 library resolution
  - Sequential Thinking minimal thought
  - Git local branch listing
  - Playwright page load
  - Memory filesystem access
- Validate the edited JSON files are error-free in the workspace.

## Related

- [[mcp_agent_upgrade]]
- [[phase26_mcp_agent_upgrade]]
- [[copilot-instructions]]
- [[AGENTS]]