---
title: Execution Engines
tags:
  - doc/wiki
  - topic/pipeline
type: architecture
summary: The repo runs an LLM-driven agent layer and a deterministic pipeline layer in parallel, with shared state between them.
status: active
source_docs:
  - '[[README]]'
  - [[quant_training_ground]]
  - [[chat_checkpoint_2026-04-16_phase29_complete]]
updated_on: 2026-04-16
---

# Execution Engines

TirraMind runs two execution engines with different responsibilities.

## Agent Layer

The agent layer is LLM-driven. It handles research, planning, hypothesis generation, and high-level orchestration.

## Pipeline Layer

The pipeline layer is deterministic. It runs scheduled DAGs, persists structured state, and executes fetch-to-signal flows without LLM calls.

## Design Consequence

The split keeps expensive, ambiguous reasoning out of the production signal path while still using the LLM where it helps most.

## Related Pages

- [[pages/architecture/system_overview]]
- [[pages/roadmap/current_phases]]