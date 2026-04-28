"""Action base class + registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from agent.awos.config import AWOSConfig
from agent.awos.policies.engine import PlannedAction


@dataclass(frozen=True)
class ActionResult:
    ok: bool
    message: str = ""
    artifacts: list[str] | None = None

    @classmethod
    def success(cls, message: str = "", artifacts: list[str] | None = None) -> "ActionResult":
        return cls(ok=True, message=message, artifacts=artifacts or [])

    @classmethod
    def failure(cls, message: str) -> "ActionResult":
        return cls(ok=False, message=message, artifacts=[])


class Action(ABC):
    type: str

    def __init__(self, cfg: AWOSConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def run(self, planned: PlannedAction) -> ActionResult:
        ...


# --- registry ------------------------------------------------------------
_REGISTRY: dict[str, Callable[[AWOSConfig], Action]] = {}


def register(type_name: str) -> Callable[[type[Action]], type[Action]]:
    def deco(cls: type[Action]) -> type[Action]:
        cls.type = type_name
        _REGISTRY[type_name] = lambda cfg: cls(cfg)
        return cls

    return deco


def build_action(type_name: str, cfg: AWOSConfig) -> Action | None:
    factory = _REGISTRY.get(type_name)
    if factory is None:
        return None
    return factory(cfg)


def registered_types() -> list[str]:
    return sorted(_REGISTRY)


__all__ = [
    "Action",
    "ActionResult",
    "build_action",
    "register",
    "registered_types",
]


def _get_registry() -> dict[str, Any]:  # pragma: no cover
    return dict(_REGISTRY)
