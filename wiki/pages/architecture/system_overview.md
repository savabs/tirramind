---
title: System Overview
tags:
  - doc/wiki
  - topic/engineering
type: architecture
summary: High-level picture of TirraMind's layered architecture and why the LLM is a support layer rather than the decision-maker.
status: active
source_docs:
  - '[[README]]'
  - [[quant_training_ground]]
  - [[chat_checkpoint_2026-04-16_phase29_complete]]
updated_on: 2026-04-16
---

# System Overview

TirraMind is a machine intelligence system built to discover predictive structure across heterogeneous data. The system prioritizes unique observations and advanced mathematics over generic chat workflows.

## Core Shape

- Layer 1 gathers surveillance data from free or cheap sources.
- Layers 2-6 transform those observations into features, world models, fused estimates, policies, and adversarial defenses.
- Layer 7 uses the LLM as support for planning, explanation, and text synthesis, not as the final decision-maker.

The decisive architectural rule is that the math and evidence layers produce the signal, while the LLM explains and organizes the result.

## Related Pages

- [[pages/architecture/execution_engines]]
- [[pages/roadmap/current_phases]]