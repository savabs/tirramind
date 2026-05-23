"""Proposal-only actions — never touch code, only write files under
``cfg.proposals_dir`` for the human to accept or reject.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.awos.actions._text import atomic_write, now_iso
from agent.awos.actions.base import Action, ActionResult, register
from agent.awos.policies.engine import PlannedAction


def _write(proposals_dir: Path, kind: str, event, body: str) -> Path:
    proposals_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{now_iso().replace(':', '')}__{kind}__{event.id[:8]}.md"
    path = proposals_dir / fname
    atomic_write(path, body)
    return path


@register("adr_propose")
class AdrProposeAction(Action):
    def run(self, planned: PlannedAction) -> ActionResult:
        e = planned.event
        principle = e.payload.get("extracted_principle") or e.rationale or "(fill in)"
        body = (
            "# ADR proposal\n\n"
            f"Event: {e.id}\n"
            f"Confidence: {e.confidence:.3f}\n\n"
            "## Context\n\n"
            f"{principle}\n\n"
            "## Decision\n\n"
            "TODO — fill in once accepted.\n\n"
            "## Consequences\n\n"
            "TODO.\n\n"
            "## Event payload\n\n"
            f"```json\n{json.dumps(e.payload, indent=2, default=str)[:3000]}\n```\n"
        )
        path = _write(self.cfg.proposals_dir, "adr", e, body)
        return ActionResult.success(f"ADR proposal: {path.name}", [str(path)])


@register("roadmap_propose")
class RoadmapProposeAction(Action):
    def run(self, planned: PlannedAction) -> ActionResult:
        e = planned.event
        body = (
            "# Roadmap-shift proposal\n\n"
            f"Event: {e.id}\n"
            f"Confidence: {e.confidence:.3f}\n\n"
            "## Proposed change\n\n"
            f"{e.rationale or '(see payload)'}\n\n"
            "## Impact\n\n"
            "Review `tasks/active/quant_training_ground.md` phase ordering.\n\n"
            "## Event payload\n\n"
            f"```json\n{json.dumps(e.payload, indent=2, default=str)[:3000]}\n```\n"
        )
        path = _write(self.cfg.proposals_dir, "roadmap", e, body)
        return ActionResult.success(f"roadmap proposal: {path.name}", [str(path)])


@register("drift_triage")
class DriftTriageAction(Action):
    def run(self, planned: PlannedAction) -> ActionResult:
        e = planned.event
        body = (
            "# Drift triage\n\n"
            f"Event: {e.id}\n"
            f"Source: {e.source}\n"
            f"Confidence: {e.confidence:.3f}\n\n"
            "## Summary\n\n"
            f"{e.rationale or '(none)'}\n\n"
            "## Findings\n\n"
            f"```json\n{json.dumps(e.payload.get('findings', []), indent=2, default=str)[:4000]}\n```\n"
        )
        path = _write(self.cfg.proposals_dir, "drift", e, body)
        return ActionResult.success(f"drift triage: {path.name}", [str(path)])


__all__ = [
    "AdrProposeAction",
    "DriftTriageAction",
    "RoadmapProposeAction",
]
