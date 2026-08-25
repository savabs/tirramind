"""Wiring the AWOS learning subpackage into the embedded runtime.

Provides:
  - ``default_learning(state_dir=None)`` — a ``LearningCore`` bound to the
    runtime config's state directory.
  - A registered ``record_learning`` Action that lets policy rules write a
    signal-operation outcome into the learning pipeline (reward, skills,
    error memory, router).

This is the integration seam between the event-driven ``agent/awos`` runtime
and the self-improving learning layers. It is intentionally small.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agent.awos.actions.base import Action, ActionResult, register
from agent.awos.policies.engine import PlannedAction

logger = logging.getLogger(__name__)


def default_learning(state_dir: str | None = None):
    """Return a LearningCore bound to ``state_dir`` (default cwd/.awos).

    Imported lazily so the runtime does not require the learning stack at
    import time; only when an outcome is actually recorded.
    """
    from agent.awos.learning.learning_core import LearningCore  # local import

    base = Path(state_dir) if state_dir else Path.cwd() / ".awos"
    return LearningCore(state_dir=str(base))


@register("record_learning")
class RecordLearningAction(Action):
    """Record a signal-operation outcome into the learning pipeline.

    Expected params (from a policy rule):
      task_id, operation, action_id, success, [signal_name, source_tool,
      cost_usd, error_type, error_msg, critique, attempts]
    """

    type = "record_learning"

    def run(self, planned: PlannedAction) -> ActionResult:
        params = dict(planned.params or {})
        payload = dict(planned.event.payload) if planned.event is not None else {}
        merged: dict[str, Any] = {**payload, **params}

        required = ("task_id", "operation", "action_id", "success")
        missing = [k for k in required if k not in merged]
        if missing:
            return ActionResult.failure(f"record_learning missing params: {missing}")

        core = default_learning(str(self.cfg.state_dir))
        try:
            core.record_outcome(
                task_id=str(merged["task_id"]),
                operation=str(merged["operation"]),
                action_id=int(merged["action_id"]),
                success=bool(merged["success"]),
                cost_usd=float(merged.get("cost_usd", 0.0)),
                signal_name=str(merged.get("signal_name", "")),
                source_tool=str(merged.get("source_tool", "")),
                error_type=str(merged.get("error_type", "")),
                error_msg=str(merged.get("error_msg", "")),
                critique=str(merged.get("critique", "")),
                attempts=int(merged.get("attempts", 1)),
            )
        except Exception as exc:  # learning must never break the pipeline
            logger.warning("[record_learning] failed: %s", exc)
            return ActionResult.failure(f"record_learning error: {exc}")

        return ActionResult.success(f"recorded outcome for task {merged['task_id']}")


__all__ = ["default_learning", "RecordLearningAction"]
