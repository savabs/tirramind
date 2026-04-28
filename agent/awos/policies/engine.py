"""Policy engine — turns events into planned actions."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from agent.awos.events.schema import Event, TriggerCategory

log = logging.getLogger(__name__)

_DEFAULT_RULES = Path(__file__).parent / "rules.yaml"


@dataclass(frozen=True)
class PlannedAction:
    type: str
    params: dict[str, Any]
    rule_id: str
    event: Event


@dataclass(frozen=True)
class PolicyRule:
    id: str
    description: str
    match: dict[str, Any]
    actions: list[dict[str, Any]]

    def matches(self, event: Event) -> bool:
        cats = self.match.get("category", [])
        if cats and event.category.value not in cats:
            return False
        min_conf = float(self.match.get("min_confidence", 0.0))
        if event.confidence < min_conf:
            return False
        sources = self.match.get("source_in")
        if sources and event.source not in sources:
            return False
        substrs = self.match.get("payload_contains")
        if substrs:
            blob = json.dumps(event.payload, default=str)
            if not any(s in blob for s in substrs):
                return False
        excluded = self.match.get("category_not")
        if excluded and event.category.value in excluded:
            return False
        return True


class PolicyEngine:
    def __init__(self, rules: list[PolicyRule]) -> None:
        self.rules = rules

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, user_file: Path | None = None) -> "PolicyEngine":
        rules = _load_yaml_rules(_DEFAULT_RULES)
        if user_file is not None and user_file.exists():
            user_rules = _load_yaml_rules(user_file)
            # user rules win on ID collision — they are appended last and
            # evaluation order preserves first-match (so we prepend user
            # rules to let them take precedence)
            rules = user_rules + rules
        return cls(rules)

    # ------------------------------------------------------------------
    def plan(self, event: Event) -> list[PlannedAction]:
        planned: list[PlannedAction] = []
        for rule in self.rules:
            if not rule.matches(event):
                continue
            for action_spec in rule.actions:
                planned.append(
                    PlannedAction(
                        type=action_spec["type"],
                        params=dict(action_spec.get("params") or {}),
                        rule_id=rule.id,
                        event=event,
                    )
                )
        return planned


# ======================================================================
def _load_yaml_rules(path: Path) -> list[PolicyRule]:
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except (yaml.YAMLError, OSError) as e:
        log.error("failed to load policy rules from %s: %s", path, e)
        return []
    raw = data.get("rules") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return []
    out: list[PolicyRule] = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        try:
            out.append(
                PolicyRule(
                    id=str(r["id"]),
                    description=str(r.get("description", "")),
                    match=dict(r.get("match") or {}),
                    actions=list(r.get("actions") or []),
                )
            )
        except KeyError:
            log.warning("skipping malformed rule: %s", r)
    return out


__all__ = ["PlannedAction", "PolicyEngine", "PolicyRule"]


# ensure the category values reference the enum so imports stay live
_ = TriggerCategory  # noqa: F841
