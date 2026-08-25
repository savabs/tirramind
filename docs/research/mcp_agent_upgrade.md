---
title: MCP Agent Upgrade
tags:
  - doc/research
  - phase/26
  - topic/agent-tooling
  - layer/surveillance
---

# Feature: MCP Agent Upgrade

## Current Architecture
- MCP servers are configured in `.vscode/mcp.json` and consumed by VS Code Copilot agent mode.
- The current workspace configuration mixes remote HTTP MCP (`github`) with local stdio servers launched via `npx` and `python`.
- VS Code already provides built-in web fetching in chat (`#fetch` / built-in fetch tool), separate from any third-party Fetch MCP server.
- The active task file for phase 26 existed, but its linked research/spec artifacts were missing.

## Observations
- All eight configured servers responded successfully when tested live in this session: Tavily, Fetch MCP, GitHub, Context7, Sequential Thinking, Git, Playwright, and Memory.
- The user-reported failure mode is intermittent chat-side behavior such as `Sorry, no response was returned` and `Try Again`, not persistent server startup failure.
- The workspace was using `@latest` for multiple stdio MCP servers. This adds version drift and forces `npx` to resolve package metadata on startup, which increases variance and can amplify transient startup failures.
- The workspace included both the built-in VS Code fetch capability and a third-party `fetch` MCP server. VS Code has an open issue about tool conflicts between MCP and built-in tools when names overlap. Even when the conflict does not fail deterministically, the overlap increases routing ambiguity.
- VS Code MCP docs recommend using `MCP: List Servers`, `Show Output`, `MCP: Reset Cached Tools`, and `MCP: Reset Trust` for troubleshooting, and expose `chat.mcp.autostart` plus discovery controls. In this local VS Code build, `chat.mcp.autostart` is validated as an enum setting and MCP discovery is already disabled by default.
- Current Git MCP launch depends on whatever `python` resolves to in the VS Code host environment. A fixed interpreter path is more deterministic.

## Risks
- Removing the Fetch MCP server slightly reduces explicit MCP surface area, but it does not remove fetch capability because VS Code's built-in fetch tool remains available.
- Pinning versions improves determinism but requires manual updates later when you intentionally want newer server versions.
- Relying on the default disabled discovery behavior keeps the workspace toolset explicit without adding a schema-sensitive override.
- Remote GitHub MCP issues that stem from VS Code client bugs or GitHub policy enforcement are not fully fixable from workspace config alone.

## Data Requirements
- No project data changes required.
- Runtime prerequisites verified:
  - Node.js and `npx` are available.
  - Python 3.11 is available at `/home/becmachlean/anaconda3/bin/python`.
  - `mcp_server_git` is installed in that interpreter.

## Math/Algorithm Survey
- Not applicable. This is tooling and client reliability work, not a mathematical implementation.

## Verified References
- VS Code docs: `Add and manage MCP servers in VS Code` (`https://code.visualstudio.com/docs/copilot/customization/mcp-servers`)
- VS Code docs: `MCP configuration reference` (`https://code.visualstudio.com/docs/copilot/reference/mcp-configuration`)
- VS Code docs: `Chat overview` / chat troubleshooting (`https://code.visualstudio.com/docs/copilot/chat/copilot-chat`)
- GitHub issue surfaced via live MCP call: `Tool conflict between MCP and built-in tools`
- GitHub issue surfaced via live MCP call: `VSCode MCP Task Compliance - Terrible UX`

## Recommended Direction
- Remove the redundant Fetch MCP server from workspace config.
- Pin stdio MCP package versions instead of relying on `@latest`.
- Use `${workspaceFolder}` for repo-relative paths and a fixed Python executable for Git MCP.
- Add the workspace MCP autostart setting using the documented enum value: `chat.mcp.autostart: "newAndOutdated"`.
- Rely on the current VS Code default of MCP discovery being disabled rather than forcing an override in workspace settings.
- Re-verify the remaining servers after the config hardening.

## Related

- [[mcp_agent_upgrade_spec]]
- [[phase26_mcp_agent_upgrade]]
- [[copilot-instructions]]
- [[AGENTS]]