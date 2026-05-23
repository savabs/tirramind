"""awos_update action.

Behavior:
- confidence >= cfg.awos_auto_update_confidence → direct write into the
  AWOS file (append to suggested section + add changelog line).
- else → proposal written to cfg.proposals_dir.

Idempotent: a signature hash of (section, principle) prevents duplicate
writes across repeated events.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from agent.awos.actions._text import (
    SELF_MARKER,
    append_changelog,
    append_to_section,
    atomic_write,
    bump_last_updated,
    now_iso,
)
from agent.awos.actions.base import Action, ActionResult, register
from agent.awos.policies.engine import PlannedAction


@register("awos_update")
class AwosUpdateAction(Action):
    def run(self, planned: PlannedAction) -> ActionResult:
        e = planned.event
        section = planned.params.get("section_hint") or e.payload.get("suggested_section") or "11. Changelog"
        changelog_only = bool(planned.params.get("changelog_only", False))
        principle = _principle_from_event(planned)

        if not principle:
            return ActionResult.failure("awos_update: could not derive principle text from event")

        threshold = self.cfg.awos_auto_update_confidence
        direct = e.confidence >= threshold and not changelog_only

        awos_path = self.cfg.awos_file
        if not awos_path.exists():
            return ActionResult.failure(f"awos file missing: {awos_path}")

        body = awos_path.read_text(encoding="utf-8")

        # check idempotency via a stable signature
        sig = _signature(section, principle)
        if f"awos-sig:{sig}" in body:
            return ActionResult.success("no-op: already present")

        bullet = _format_bullet(principle, e, sig)

        if changelog_only:
            new_body = append_changelog(body, principle)
            new_body = bump_last_updated(new_body)
            atomic_write(awos_path, new_body)
            return ActionResult.success(
                f"appended changelog entry to {awos_path}",
                artifacts=[str(awos_path)],
            )

        if direct:
            new_body = append_to_section(body, section, bullet)
            new_body = append_changelog(
                new_body,
                f"updated section '{section}' from event {e.id[:8]} "
                f"(confidence={e.confidence:.2f}, classifier="
                f"{e.payload.get('classifier', '?')})",
            )
            new_body = bump_last_updated(new_body)
            atomic_write(awos_path, new_body)
            return ActionResult.success(
                f"direct update to '{section}'",
                artifacts=[str(awos_path)],
            )

        # propose
        prop_path = _write_proposal(self.cfg.proposals_dir, section, bullet, e)
        return ActionResult.success(
            f"wrote proposal (confidence {e.confidence:.2f} < threshold {threshold:.2f})",
            artifacts=[str(prop_path)],
        )


# ======================================================================
def _principle_from_event(planned: PlannedAction) -> str:
    """Pull the best-available principle text from the event."""
    e = planned.event
    p = e.payload or {}
    explicit = p.get("extracted_principle")
    if explicit:
        return str(explicit).strip()
    prefix = p.get("chunk_prefix")
    if prefix:
        return str(prefix).strip().splitlines()[0][:200]
    if e.rationale:
        return e.rationale.strip()[:200]
    return ""


def _signature(section: str, principle: str) -> str:
    h = hashlib.sha256()
    h.update(section.encode())
    h.update(b"|")
    h.update(principle.encode())
    return h.hexdigest()[:12]


def _format_bullet(principle: str, event, sig: str) -> str:
    ts = now_iso()
    return (
        f"- {SELF_MARKER} [{ts}] (awos-sig:{sig}) "
        f"{principle.rstrip('.')}.  _source={event.source}, "
        f"confidence={event.confidence:.2f}_"
    )


def _write_proposal(proposals_dir: Path, section: str, bullet: str, event) -> Path:
    proposals_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{now_iso().replace(':', '')}__awos__{event.id[:8]}.md"
    path = proposals_dir / fname
    content = (
        f"# AWOS update proposal\n\n"
        f"Event: {event.id}\n"
        f"Category: {event.category.value}\n"
        f"Confidence: {event.confidence:.3f}\n"
        f"Suggested section: {section}\n\n"
        f"## Proposed addition\n\n"
        f"{bullet}\n\n"
        f"## Rationale\n\n"
        f"{event.rationale or '(none)'}\n\n"
        f"## Event payload\n\n"
        f"```json\n{_safe_json(event.payload)}\n```\n"
    )
    atomic_write(path, content)
    return path


def _safe_json(d: dict) -> str:
    import json

    try:
        return json.dumps(d, indent=2, default=str)[:4000]
    except Exception:
        return "{}"


__all__ = ["AwosUpdateAction"]
