---
title: "ADR-0001: Pipeline Layer Uses No LLM"
tags:
  - doc/adr
  - layer/feature-engineering
  - layer/surveillance
  - topic/pipeline
---

# ADR-0001: Pipeline Layer Uses No LLM

**Status:** accepted
**Date:** 2026-03-24
**Deciders:** Core team

## Context

TirraMind runs two execution engines: the Agent Layer (LLM-driven exploration) and the Pipeline Layer (scheduled data collection and feature computation). The question was whether the Pipeline Layer should use the LLM for any part of its execution — error recovery, data interpretation, scheduling decisions, or output formatting.

Layers affected: Pipeline Layer, Layer 1 (Surveillance Surface), Layer 2 (Feature Engineering).

## Decision

The Pipeline Layer is **fully deterministic**. No LLM calls. No randomness beyond retry jitter. The pipeline uses:
- SQLite for structured state persistence (WAL mode)
- APScheduler for cron-based triggers
- ThreadPoolExecutor for parallel node execution
- Static DAG definitions (no dynamic graph construction)

The LLM may *read* pipeline outputs (via the `pipeline_query` tool), but it cannot *write* to or *control* the pipeline.

## Alternatives Considered

### Alternative A: LLM-assisted error recovery
- Pros: Could interpret ambiguous API errors and retry with adjusted parameters
- Cons: Non-deterministic behavior in a critical data path. Hard to test. Hard to debug. Adds latency and cost. Failure modes are unpredictable.

### Alternative B: LLM-driven scheduling
- Pros: Could dynamically prioritize data sources based on market conditions
- Cons: Scheduling decisions should be based on data freshness and signal value, not LLM judgment. The RL layer (Phase 11) will handle prioritization with proper reward signals.

### Alternative C: Hybrid (LLM for narration, deterministic for execution)
- Pros: Best of both worlds conceptually
- Cons: Adds LLM dependency to pipeline startup. Complicates testing. The Agent Layer already provides narration — duplicating it in the pipeline violates single-responsibility.

## Consequences

### Positive
- Pipeline is fully testable with deterministic assertions
- No LLM cost for data collection (runs hundreds of times per day)
- Failures are reproducible — same input always produces same output
- Pipeline can run on machines without LLM access

### Negative
- Error messages from failed API calls are raw, not human-friendly
- No adaptive retry logic (fixed exponential backoff only)
- New data sources require code changes, not natural language instructions

### Risks
- If API formats change, pipeline breaks until code is updated (no LLM to adapt)
- Mitigation: robust parsing with graceful degradation, alerts on failure

## References
- `[[pipeline_layer]]` — original research
- `[[pipeline_layer_spec]]` — implementation spec
- `agent/pipeline/` — implementation

---

## Related

- [[convergence_detection]]
- [[world_model]]
