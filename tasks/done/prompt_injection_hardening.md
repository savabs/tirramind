---
title: "Task: prompt_injection_hardening"
tags:
  - doc/task
  - status/done
  - topic/security
---

# Task: prompt_injection_hardening

Status: completed
Research: [[prompt_injection_hardening]]
Spec: [[prompt_injection_hardening_spec]]

## Steps

- [x] 1.1 Create agent/security/__init__.py
- [x] 1.2 Create agent/security/tool_policy.py (ToolRisk, TrustLevel enums, TOOL_RISK_REGISTRY)
- [x] 1.3 Add trust_level to ToolResult in agent/tools/base.py
- [x] 1.4 Add is_safe_url() SSRF protection
- [x] 1.5 Integrate SSRF protection into WebBrowseTool
- [x] 2.1 Add ToolPolicyGuard class
- [x] 2.2 Integrate policy guard into ToolRegistry.execute()
- [x] 2.3 Add is_safe_path() and integrate into file tools
- [x] 3.1 Add tainted field to Fact, mark web-sourced facts
- [x] 3.2 Phase-based tool filtering in orchestrator
- [x] 4.1 Write unit tests for tool_policy.py
- [x] 4.2 Write prompt injection integration tests
- [x] 4.3 Run full test suite, fix regressions — 86 new tests pass, 0 regressions

---

## Related

- [[prompt_injection_hardening|Research: Prompt Injection Hardening]]
- [[prompt_injection_hardening_spec|Spec: Prompt Injection Hardening]]
