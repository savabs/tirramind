---
title: "Spec: prompt_injection_hardening"
tags:
  - doc/spec
  - topic/security
---

# Spec: prompt_injection_hardening

## Goal
Prevent untrusted content (web pages, API responses, memory) from escalating into
dangerous tool calls (shell, code execution, file writes) or exfiltrating data
(SSRF to private IPs). Make the agent safe to run autonomously against arbitrary web content.

## Files Affected

### New files
- `agent/security/__init__.py` — package init
- `agent/security/tool_policy.py` — risk tiers, policy guard, URL validation
- `tests/test_tool_policy.py` — unit tests for policy guard
- `tests/test_prompt_injection.py` — integration tests for injection resistance

### Modified files
- `agent/tools/base.py` — add `trust_level` to `ToolResult`, integrate policy guard into `ToolRegistry.execute()`
- `agent/tools/web_browse.py` — add SSRF protection before HTTP requests
- `agent/tools/web_search.py` — add SSRF protection (result URLs)
- `agent/tools/file_manager.py` — add path restriction (sandbox to allowed dirs)
- `agent/tools/shell_runner.py` — strengthen blocklist, add policy gate integration
- `agent/tools/code_executor.py` — add policy gate integration
- `agent/core/orchestrator.py` — use trust labels when storing to memory; phase-based tool filtering
- `agent/memory/store.py` — add `tainted` field to `Fact`

## Implementation Steps

### Step 1: Tool Risk Tiers + Trust Labels (foundation)
1.1 Create `agent/security/__init__.py`
1.2 Create `agent/security/tool_policy.py` with `ToolRisk` enum, `TrustLevel` enum
1.3 Add `trust_level` field to `ToolResult` in `agent/tools/base.py`
1.4 Add `TOOL_RISK_REGISTRY` mapping tool names → risk levels

### Step 2: SSRF Protection
2.1 Add `is_safe_url()` to `agent/security/tool_policy.py`
2.2 Integrate into `WebBrowseTool.execute()`
2.3 Add SSRF tests

### Step 3: Policy Guard
3.1 Add `ToolPolicyGuard` class to `agent/security/tool_policy.py`
3.2 Integrate into `ToolRegistry.execute()` — check before every call
3.3 Add policy guard tests (block dangerous tools in autonomous mode, allow with approval)

### Step 4: File Path Sandboxing
4.1 Add `is_safe_path()` to `agent/security/tool_policy.py`
4.2 Integrate into `FileWriteTool`, `FileReadTool`
4.3 Add path traversal tests

### Step 5: Memory Taint Tracking
5.1 Add `tainted` field to `Fact` dataclass
5.2 Mark facts from web tools as tainted in orchestrator
5.3 Add `trust_level` metadata when storing tool results in working memory

### Step 6: Phase-based Tool Filtering
6.1 Add `get_tools_for_phase()` to policy module
6.2 Use in orchestrator research/planning phases to limit available tools

### Step 7: Prompt Injection Test Suite
7.1 Test: web content with "ignore previous instructions" cannot trigger shell
7.2 Test: SSRF to localhost/metadata/private IPs blocked
7.3 Test: path traversal in file tools blocked
7.4 Test: tainted memory facts are tagged correctly
7.5 Test: phase filtering restricts tool availability

## Edge Cases
- Tools with no risk classification default to STATE_CHANGING (safe default)
- Policy guard must not break existing tests — backward compatible default
- URL validation must handle edge cases: IPv6, encoded IPs, DNS rebinding (partial)
- File sandbox must allow writing to designated workspace dirs

## Testing Plan
- Unit tests for every new function in tool_policy.py
- Integration tests that simulate prompt injection scenarios
- All existing tests must continue to pass

---

## Related

- [[prompt_injection_hardening|Research: Prompt Injection Hardening]]
