"""Dispatcher — takes an event, runs policies, executes actions.

Separated from the daemon so it is callable from tests and CLI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

# importing actions with side effects registers them
from agent.awos.actions import awos_update as _awos_update  # noqa: F401
from agent.awos.actions import learning as _learning  # noqa: F401
from agent.awos.actions import proposals as _proposals  # noqa: F401
from agent.awos.actions.base import ActionResult, build_action
from agent.awos.config import AWOSConfig
from agent.awos.events.bus import EventBus
from agent.awos.events.schema import Event, EventStatus
from agent.awos.policies.engine import PolicyEngine

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DispatchReport:
    event_id: str
    planned: int
    executed: int
    failed: int
    results: list[tuple[str, ActionResult]]


class Dispatcher:
    def __init__(
        self,
        cfg: AWOSConfig,
        bus: EventBus,
        policies: PolicyEngine,
    ) -> None:
        self.cfg = cfg
        self.bus = bus
        self.policies = policies

    # ------------------------------------------------------------------
    def dispatch(self, event: Event) -> DispatchReport:
        planned = self.policies.plan(event)
        if not planned:
            self.bus.mark(event.id, EventStatus.IGNORED)
            return DispatchReport(event.id, 0, 0, 0, [])

        executed = 0
        failed = 0
        results: list[tuple[str, ActionResult]] = []
        for pa in planned:
            action = build_action(pa.type, self.cfg)
            if action is None:
                failed += 1
                results.append((pa.type, ActionResult.failure(f"unknown action type: {pa.type}")))
                continue
            try:
                res = action.run(pa)
            except Exception as e:
                log.exception("action %s failed: %s", pa.type, e)
                res = ActionResult.failure(str(e))
            results.append((pa.type, res))
            if res.ok:
                executed += 1
            else:
                failed += 1

        final_status = EventStatus.PROCESSED if failed == 0 else EventStatus.ERRORED
        self.bus.mark(event.id, final_status)
        return DispatchReport(event.id, len(planned), executed, failed, results)

    # ------------------------------------------------------------------
    def drain(self, limit: int = 100) -> list[DispatchReport]:
        events = self.bus.fetch(status=EventStatus.NEW, limit=limit)
        return [self.dispatch(e) for e in events]


__all__ = ["DispatchReport", "Dispatcher"]
