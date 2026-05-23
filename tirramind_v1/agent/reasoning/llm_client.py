"""
TirraMind Agent — LLM Reasoning Interface

Thin wrapper around OpenAI-compatible chat completions.
Works with: OpenAI, Ollama, Azure, any compatible endpoint.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from agent.config.settings import LLMConfig

log = logging.getLogger(__name__)


class LLMClient:
    """Stateless LLM reasoning interface.

    All calls go through `chat()` which takes messages and optional
    tool definitions, returns the assistant response.
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client = OpenAI(
            api_key=config.api_key or "not-set",
            base_url=config.base_url,
        )

    # ------------------------------------------------------------------
    # Core call
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Send a chat completion request.

        Returns the full assistant message dict, including any tool_calls.
        """
        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self._config.temperature,
            "max_tokens": self._config.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        log.debug(
            "LLM request: model=%s messages=%d tools=%s", self._config.model, len(messages), len(tools) if tools else 0
        )

        response = self._client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        result: dict[str, Any] = {
            "role": "assistant",
            "content": msg.content or "",
        }
        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        return result

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def ask(self, prompt: str, system: str = "") -> str:
        """Simple question → answer. Returns content string."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages)["content"]

    def structured_output(self, prompt: str, system: str = "") -> Any:
        """Ask for JSON and parse it. Falls back to raw string on parse failure."""
        raw = self.ask(prompt, system=system or "Respond only with valid JSON.")
        # Strip markdown fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            log.warning("LLM did not return valid JSON, returning raw string")
            return raw

    def decide_tool(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
    ) -> tuple[str | None, dict[str, Any] | None, str]:
        """Ask the LLM which tool to call.

        Returns (tool_name, arguments_dict, reasoning_text).
        If no tool call, returns (None, None, content).
        """
        result = self.chat(messages, tools=tools)
        tool_calls = result.get("tool_calls")
        if tool_calls:
            tc = tool_calls[0]
            try:
                args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, KeyError):
                args = {}
            return tc["function"]["name"], args, result.get("content", "")
        return None, None, result.get("content", "")
