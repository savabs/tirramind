"""Prompt templates for the LLM classifier."""

from __future__ import annotations

import json

from agent.awos.events.schema import TriggerCategory

SYSTEM_PROMPT = """You are a classification head for a software-agent
governance system called AWOS. Your only job is to read a snippet of
conversation between a user and a coding agent, and decide whether it
contains something worth recording in the project's living workflow
memory.

Return STRICT JSON with the following fields:
  category: one of [architectural, workflow_pattern, lesson,
            roadmap_shift, decision, drift, staleness, routine, unknown]
  confidence: number in [0.0, 1.0]
  rationale: one short sentence explaining your choice
  extracted_principle: if category is workflow_pattern / architectural /
                       lesson / decision, a single imperative sentence
                       stating the rule or insight. Otherwise null.
  suggested_section: best-fit AWOS section title. Use exactly one of
                     the section titles provided below. If unsure, use
                     "Changelog".

AWOS sections:
  - "1. Purpose"
  - "2. High-Level Goals"
  - "3. Agent Operating Principles"
  - "4. Workflow Pipeline"
  - "5. Codebase Structure"
  - "6. Data Flows"
  - "7. Anti-Patterns"
  - "8. Testing & Validation"
  - "9. Memory & Context Protocol"
  - "10. Open Questions"
  - "11. Changelog"
  - "Lessons"
  - "Roadmap Notes"
  - "Decisions"

Definitions of categories (decide based on the dominant signal):
  architectural   - a rule about module layout, layer separation, naming,
                    or dependency direction. Example: "quant code never
                    fetches data".
  workflow_pattern - a standing instruction about how to work. Example:
                    "agent should write checkpoints without being asked".
  lesson          - a post-mortem observation about a prior failure and
                    what to do differently.
  roadmap_shift   - a change to the ordering or scope of upcoming phases.
  decision        - a concrete design/tech choice (library, threshold,
                    schema) that should be ADR-worthy.
  drift           - evidence that docs/code/state are inconsistent.
  staleness       - old tasks, old checkpoints, old branches still open.
  routine         - ordinary implementation chatter (running tests,
                    fixing a typo, small refactor). Do NOT update AWOS
                    for routine work.
  unknown         - unclear or not enough information.

Be conservative. If the snippet is routine, return routine with high
confidence. Do not return workflow_pattern or architectural unless the
user is clearly stating a durable rule.
"""


def render_user_message(text: str, extra: dict | None = None) -> str:
    ctx = {}
    if extra:
        for k in ("source", "task", "recent_git"):
            if k in extra:
                ctx[k] = extra[k]
    ctx_json = json.dumps(ctx) if ctx else "{}"
    return (
        "Context (JSON): "
        f"{ctx_json}\n\n"
        "Conversation snippet:\n"
        "---\n"
        f"{text.strip()}\n"
        "---\n"
        "Return JSON only."
    )


def allowed_categories() -> list[str]:
    return [c.value for c in TriggerCategory]


__all__ = [
    "SYSTEM_PROMPT",
    "render_user_message",
    "allowed_categories",
]
