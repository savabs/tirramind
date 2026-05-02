"""
TirraMind Agent — Tool Base Class & Registry

Every tool the agent can use inherits from `Tool`.
The `ToolRegistry` discovers and manages them.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import jsonschema

log = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Standard return from any tool execution."""

    success: bool
    output: str
    data: Any = None  # structured data for downstream processing
    trust_level: str = "tool_trusted"  # provenance tag (see security.tool_policy.TrustLevel)


class Tool(ABC):
    """Base class for all agent tools.

    Subclasses must define:
      - name: unique identifier
      - description: what the tool does (shown to LLM)
      - parameters: JSON Schema dict describing arguments
      - execute(**kwargs) -> ToolResult
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]: ...

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult: ...

    def to_openai_tool(self) -> dict[str, Any]:
        """Convert to OpenAI function-calling tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Manages available tools. The orchestrator queries this to know what the agent can do."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._policy_guard: Any = None  # Optional ToolPolicyGuard

    def set_policy_guard(self, guard: Any) -> None:
        """Attach a ToolPolicyGuard for security enforcement."""
        self._policy_guard = guard

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            log.warning("Tool %r already registered — overwriting", tool.name)
        self._tools[tool.name] = tool
        log.debug("Registered tool: %s", tool.name)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """All tools in OpenAI function-calling format."""
        return [t.to_openai_tool() for t in self._tools.values()]

    def validate_args(self, name: str, args: dict[str, Any]) -> list[str]:
        """Validate tool arguments against the tool's JSON Schema.

        Returns a list of error messages (empty = valid).
        """
        tool = self._tools.get(name)
        if tool is None:
            return [f"Unknown tool: {name}"]
        try:
            jsonschema.validate(instance=args, schema=tool.parameters)
            return []
        except jsonschema.ValidationError as exc:
            return [exc.message]
        except jsonschema.SchemaError as exc:
            log.warning("Invalid schema for tool %s: %s", name, exc)
            return []  # don't block on bad schema, log it

    def execute(self, name: str, **kwargs: Any) -> ToolResult:
        """Look up tool by name, validate arguments, and execute."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(success=False, output=f"Unknown tool: {name}")

        # Security: policy guard check before execution
        if self._policy_guard is not None:
            allowed, reason = self._policy_guard.check_execution(name, kwargs)
            if not allowed:
                return ToolResult(success=False, output=reason)

        # Validate arguments against tool's JSON Schema
        errors = self.validate_args(name, kwargs)
        if errors:
            msg = f"Invalid arguments for tool '{name}': {'; '.join(errors)}"
            log.warning(msg)
            return ToolResult(success=False, output=msg)

        try:
            result = tool.execute(**kwargs)
            # Tag result with trust level based on tool classification
            try:
                from agent.security.tool_policy import get_trust_level_for_tool

                result.trust_level = get_trust_level_for_tool(name)
            except ImportError:
                pass
            return result
        except Exception as exc:
            log.exception("Tool %s failed", name)
            return ToolResult(success=False, output=f"Tool {name} error: {exc}")

    def __len__(self) -> int:
        return len(self._tools)
