---
title: MCP External Help Tool Guide
tags:
  - doc/wiki
  - topic/mcp
  - topic/agent-orchestration
  - layer/llm-support
---

# MCP External Help Tool Guide

This guide explains how to integrate external help tools through MCP so an agent can handle complex tasks without turning into an uncontrolled tool-calling loop. The core idea is simple: MCPs are capability pipes, not the intelligence. The intelligence lives in the orchestration layer that decides what kind of help is needed, which source is trustworthy, what must be read before action, and what must be verified before a task is considered complete.

## Core Model

Treat the MCP stack as a set of distinct primitives with clear responsibilities:

1. Tools: executable actions.
2. Resources: read-only contextual data.
3. Prompts: reusable reasoning or workflow templates.
4. Roots: filesystem boundaries for safe access.
5. Sampling: server-side requests for model help from the client.
6. Elicitation: precise requests for missing user input.
7. Tool annotations: hints about risk and behavior, not guarantees.

The clean rule is:

- Context belongs in resources.
- Actions belong in tools.
- Reusable workflows belong in prompts.
- Safety boundaries belong in roots and client policy.
- Human clarification belongs in elicitation.
- Model recursion belongs in sampling only when necessary.

## What An External Help Tool Really Is

In practice, external help tools usually fall into these buckets:

1. Research help: docs, web search, repo search, page fetch.
2. Workspace help: file search, symbol search, test running, linting.
3. Execution help: shell, Python, database, browser automation.
4. System help: cloud APIs, CI/CD, ticketing, deployment, logging.
5. Memory help: checkpoint stores, vector stores, knowledge graphs.
6. Human help: approvals, missing scope decisions, secret-handling, business-rule clarifications.

Do not expose these as one flat tool pile. Organize them by policy, trust, cost, and side effects.

## The Right Architecture

For complex work, use a star topology:

1. One orchestrator client or central agent sits in the middle.
2. Multiple MCP servers expose narrow, specialized capabilities.
3. The orchestrator chooses what to call and in what order.
4. Verification stays separate from execution.

Do not start with server-to-server sprawl. Keep orchestration in one place until reuse justifies a controller server.

### Recommended Orchestrator Layers

1. Intent classifier: determines whether the task is research, coding, debugging, deployment, extraction, monitoring, or mixed.
2. Planner: breaks the task into stages with evidence requirements.
3. Tool selector: chooses candidate MCPs based on capability, safety, cost, and latency.
4. Executor: performs the calls.
5. Verifier: checks correctness through tests, assertions, read-backs, diffs, or secondary evidence.
6. Memory layer: stores prior attempts, successful paths, failures, and checkpoints.

That structure is what makes a multi-MCP system intelligent. More tools alone do not.

## Tool Selection Policy

The agent should not pick tools loosely. Give it deterministic selection rules:

1. Prefer read-only tools before write tools.
2. Prefer local evidence before remote evidence.
3. Prefer low-cost tools before expensive tools.
4. Prefer deterministic tools before open-ended tools.
5. Prefer one sufficient tool over a long chain.
6. Only call external search when local context is insufficient.
7. Only call write or destructive tools after a plan exists.
8. Always verify a write with an independent read path.

### A Good Default Escalation Order

1. Local resources and workspace tools.
2. Official or authoritative documentation.
3. Structured APIs.
4. Browser or web search.
5. Execution tools.
6. Write tools.
7. Destructive or deployment tools.

## Capability Registry

Do not maintain only a tool list. Maintain a capability registry with routing metadata.

Each MCP tool or server should track:

- Name
- Purpose
- Inputs and outputs
- Read-only vs mutating
- Destructive risk
- Idempotence
- Trust tier
- Latency class
- Cost class
- Auth scope
- Approval requirement
- Best-for task types
- Never-use-for task types

Example structure:

```json
{
  "name": "docs_search",
  "kind": "research",
  "read_only": true,
  "destructive": false,
  "idempotent": true,
  "latency_class": "medium",
  "cost_class": "low",
  "trust_tier": "high",
  "needs_approval": false,
  "best_for": [
    "api verification",
    "library syntax",
    "version-specific docs"
  ],
  "avoid_for": [
    "workspace-local code facts",
    "state-changing operations"
  ]
}
```

This lets the selector use policy and scoring instead of raw prompting.

## Trust Tiers

Not all MCPs deserve equal trust. Define trust explicitly.

1. Tier 1: local workspace tools, tests, structured internal APIs, official docs.
2. Tier 2: known external APIs, controlled browser flows, cloud interfaces.
3. Tier 3: broad web search, scraped pages, unknown third-party systems.

Use the tiers operationally:

- High-risk actions should require Tier 1 confirmation.
- External factual claims should be confirmed by Tier 1 or official Tier 2 evidence.
- Tier 3 results may suggest, but should not decide.

## How To Combine Multiple MCPs

Use staged specialization. A practical stack for complex engineering tasks is:

1. Docs/search MCP for external facts and library or API verification.
2. Workspace MCP for file reads, symbol lookups, tests, and diagnostics.
3. Execution MCP for shell, scripts, database, or browser work.
4. Memory MCP for checkpoints and prior-attempt recall.
5. Approval or elicitation path for missing requirements and risky actions.

### Routing Rules

1. If the task is unclear, use elicitation first.
2. If the task is factual or library-specific, use docs first.
3. If the task is a code change, use workspace tools first and docs only when uncertainty appears.
4. If the task is a bug, use logs, tests, and reproduction before edits.
5. If the task touches external systems, use dry-run or read APIs before write APIs.
6. If the task is long-running, checkpoint after each major stage.

## Recommended Execution Loop

For complex tasks, avoid freeform tool use. Use a phased loop:

1. Discover: gather relevant resources, local context, logs, schemas, or docs.
2. Plan: summarize what was learned and choose the smallest next step.
3. Execute: perform the narrowest action that advances the task.
4. Verify: run tests, assertions, read-backs, comparisons, or secondary checks.
5. Reflect: if the step failed, revise the hypothesis before another attempt.

This loop is better than an unstructured autonomous agent because it forces evidence before action and verification after action.

## Roots

Roots define filesystem boundaries. Use them as an actual safety control.

Best practice:

1. Give each server only the roots it needs.
2. Do not expose the entire machine if one repo path is enough.
3. In multi-repo workflows, expose separate named roots.
4. Treat root changes as a boundary change that may require re-approval.

If a filesystem-oriented server has broad root access by default, the safety posture is weak.

## Elicitation

Use elicitation only when the task cannot safely continue without user input.

Good uses:

1. Missing credentials or secrets.
2. Approval for destructive actions.
3. Ambiguous business rules.
4. Competing valid targets where user intent matters.
5. Underspecified scope after local discovery is exhausted.

Bad uses:

1. Asking for facts already available in the workspace.
2. Asking for information available through a local file or API.
3. Asking too early before basic discovery.

## Sampling

Sampling allows a server to request model output from the client. It is useful, but it should be used carefully.

Use it when:

1. A server needs a localized reasoning pass inside a tool workflow.
2. A workflow needs model help without handing control back to the main planner.
3. Structured classification or transformation belongs naturally inside the server.

Avoid using it when the client planner already owns the global reasoning. Otherwise the system becomes difficult to debug because reasoning is duplicated across layers.

The clean rule is:

- Global reasoning belongs in the client.
- Small local reasoning belongs in the server only when it improves modularity.

## Tool Annotations

Annotations improve selection and UX, but they are not a hard safety mechanism.

The important ones are:

1. `readOnlyHint`
2. `destructiveHint`
3. `idempotentHint`
4. `openWorldHint`

Use them as hints for routing and approval policies, but keep real controls elsewhere:

1. Allowlists
2. Approval gates
3. Roots
4. Auth scoping
5. Retry budgets
6. Rate limits
7. Audit logging

## Subagent Pattern For Hard Tasks

For genuinely hard workflows, use role-separated subagents with restricted MCP sets.

1. Research agent
   Allowed: docs, search, browser, repo search, resources.
   Goal: gather evidence and produce a compact fact pack.

2. Planner agent
   Allowed: fact pack and local context, not heavy execution tools.
   Goal: break work into steps and define the execution policy.

3. Executor agent
   Allowed: workspace, shell, database, deployment, or targeted external APIs.
   Goal: perform the smallest necessary action.

4. Verifier agent
   Allowed: tests, diffs, logs, assertions, reads.
   Goal: independently confirm success.

This usually works better than one agent with every tool because context stays cleaner and policy becomes enforceable.

## Parallelism Rules

Parallelism is useful for independent read operations.

Parallelize:

1. Reading multiple resources.
2. Searching code and docs at the same time.
3. Collecting logs from independent systems.
4. Comparing multiple data sources.

Do not parallelize:

1. Writes to the same system.
2. Sequential workflows with dependencies.
3. Expensive calls before the first hypothesis is stable.
4. Destructive operations.

Rule of thumb:

- Parallelize discovery.
- Serialize mutation.

## Budgeting And Circuit Breakers

Complex agents need operating budgets and stop conditions.

Recommended budgets:

1. Maximum external calls per task.
2. Maximum retries per tool.
3. Maximum cost per task.
4. Maximum runtime per stage.
5. Maximum destructive actions before re-verification.
6. Maximum unresolved ambiguity before elicitation is mandatory.

Recommended circuit breakers:

1. Stop if two hypotheses fail in the same way.
2. Stop if no new evidence is being collected.
3. Stop if the planner repeats the same tool path.
4. Stop if a write is requested without fresh reads.
5. Stop if the task crosses trust boundaries unexpectedly.

## Provenance And Logging

Every MCP call should be recorded with enough detail to audit and improve the system:

1. Which server was called.
2. Why it was chosen.
3. What inputs were sent.
4. What outputs were returned.
5. Whether state changed.
6. What verification followed.
7. Whether the result was accepted or rejected.

A compact trace is usually enough:

```text
Task: add retry handling for payment webhooks
Step 1: searched local code for handler
Step 2: read official docs for retry semantics
Step 3: inspected current retry policy
Step 4: patched handler
Step 5: ran unit tests
Step 6: replayed duplicate-delivery scenario
Step 7: accepted result
```

## Default Workflow For Intelligent MCP Combination

This is the strongest default loop for a new project:

1. Classify the task.
2. Gather local context first.
3. Pull authoritative external docs only if local context is insufficient.
4. Summarize findings into a compact working state.
5. Choose the smallest next action.
6. Execute with the narrowest tool that can do the job.
7. Verify with an independent read or test path.
8. If the step fails, switch to debug mode instead of blind retry.
9. Checkpoint the attempt.
10. Continue only if evidence improved.

## Anti-Patterns

Avoid these:

1. Giving one agent every MCP and no routing policy.
2. Using MCP where instructions or prompts would be simpler.
3. Using external search before checking the workspace.
4. Letting write tools run before a clear plan exists.
5. Treating tool annotations as hard safety controls.
6. Using sampling everywhere and duplicating reasoning across layers.
7. Skipping a separate verifier path.
8. Adding more MCPs when the real problem is poor orchestration.
9. Having no checkpoint or memory layer.
10. Measuring quality by number of tool calls instead of verified outcomes.

## Minimal Blueprint

For most advanced projects, this minimal architecture is enough:

1. One central orchestrator.
2. Three MCP groups: research, execution/workspace, memory/approval.
3. One policy layer for trust, cost, approvals, retries, and verification.
4. One verifier path.
5. One checkpoint store.

This is usually stronger than an overbuilt swarm of agents.

## Recommended Implementation Order

If you are building this from scratch:

1. Define task types.
2. Define a capability metadata schema.
3. Add narrow MCP servers.
4. Add the selector policy.
5. Add verification rules.
6. Add elicitation and approval gates.
7. Add logging and checkpoints.
8. Only then consider subagents or sampling-heavy workflows.

## New-MCP Admission Checklist

Before adding a new MCP, answer these questions:

1. What exact capability gap does it fill?
2. Is this really a tool, resource, or prompt problem?
3. Can an existing MCP already cover it?
4. What is its trust tier?
5. What are the side effects?
6. What verification will prove success?
7. What task type should trigger it?
8. What task types should never trigger it?

If those answers are not clear, do not add the MCP yet.

## Practical Recommendation

For a new project that needs an intelligent combination of MCPs, build around these layers:

1. Policy
2. Planner
3. Selector
4. Executor
5. Verifier
6. Memory

That gives you a system that is controllable, auditable, and able to improve without becoming chaotic.

## Source Notes

This guide is grounded in the MCP specification and architecture concepts, including tools, resources, prompts, roots, sampling, elicitation, and tool annotations, plus practical orchestration guidance for multi-tool agent systems.

## Related

- [[efficient_operations_guide]]