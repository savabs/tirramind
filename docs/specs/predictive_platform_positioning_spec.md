---
title: "Spec: Predictive Platform Positioning"
tags:
  - doc/spec
  - phase/25
  - topic/productization
  - topic/predictive-intelligence
  - topic/quant
  - layer/surveillance
  - layer/world-model
---

# Spec: Predictive Platform Positioning

## Goal

Update the repo's main source-of-truth documents so TirraMind is explicitly framed as an advanced predictive intelligence platform whose customers buy outcomes and decision advantage through a tool-shaped product, rather than raw tool or model access. Trading should remain an important downstream application, but not the sole product definition. The docs should also make the product hierarchy explicit: the base asset is the intelligence engine, and productized delivery layers are built on top of it. The business doctrine should be explicit: models are replaceable internal infrastructure, while the software surface packages the workflow and outcome in a way that is harder to swap out than a bare model endpoint. Better models should widen gross margin because the SKU is the delivered outcome. Within that broader framing, the initial commercial wedge should remain clear: quant firms and prop firms first.

## Files Affected

- `[[predictive_platform_positioning]]`
- `[[predictive_platform_positioning_spec]]`
- `[[predictive_platform_positioning_task]]`
- `[[project_memory]]`
- `[[quant_training_ground]]`
- `[[chat_checkpoint_2026-04-16_outcome_over_tool_positioning]]`

## Implementation Steps

1. Create the research note describing the mismatch between current trading-first wording and the broader predictive-platform goal.
2. Create the spec and active task file for the documentation realignment.
3. Update the research/spec/task artifacts so the commercial doctrine is outcome-first and tool-shaped rather than engine licensing or generic model-seat software.
4. Update `project_memory.md` so monetization is described as productized predictive intelligence first, with trading as one route rather than the default route.
5. Update `project_memory.md` and `quant_training_ground.md` to describe the platform hierarchy: base intelligence engine first, productized delivery layers second, with quant firms and prop firms as the initial commercial wedge.
6. Add explicit wording that models are replaceable internal infrastructure, while the software layer should package workflow, telemetry, and customer-specific operating context so the product still feels like a tool but is not easily replaceable by another tool.
7. Write a fresh checkpoint so today's durable context reflects the revised positioning without rewriting historical checkpoints.

## Edge Cases

- Do not weaken the economic goal; the point is broader monetization, not a softer mission.
- Do not rewrite historical checkpoints from older sessions unless they are current source-of-truth for active planning.
- Do not turn the project into a vague enterprise platform narrative. Keep the language mathematically serious and prediction-first.
- Do not over-index on bespoke services at the expense of the core model. The core intelligence engine remains the moat.
- Do not drift into a body-shop framing. The correct target is productized service delivery with telemetry, repeatability, and compounding data capture.
- Do not imply that UI seats, model access, or a chat surface are the primary SKU. Those can exist, but only as subordinate delivery surfaces around the actual outcome.
- Do not overcorrect into an anti-software posture. The product can and should look like software when that improves adoption and retention; the constraint is that the value proposition must remain outcome-anchored rather than model-anchored.

## Testing Plan

- Verify all new markdown files have valid frontmatter and `## Related` sections.
- Verify cross-references use `[[wiki links]]`.
- Manually confirm the updated docs consistently describe TirraMind as a predictive intelligence platform with multiple customer verticals.
- Manually confirm the updated docs consistently state that customers buy outcomes and decision advantage, while the model/tool layer remains internal and replaceable.
- Manually confirm the updated docs distinguish between a tool-shaped surface and a commodity tool business.

## Related

- [[predictive_platform_positioning]]
- [[predictive_platform_positioning_task]]
- [[project_memory]]
- [[quant_training_ground]]