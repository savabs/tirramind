"""Anthropic Claude-backed classifier.

Uses the Messages API with JSON-only output. Handles:
- missing/invalid API key
- HTTP 400/401/429 / 5xx with exponential backoff (bounded)
- JSON parse failures → falls back to UNKNOWN / low confidence
- timeouts → falls back cleanly (caller composes with heuristic in hybrid mode)

No real network I/O happens at import time. Callers wire in an httpx
client factory for testability.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

import httpx

from agent.awos.classifiers.base import Classification
from agent.awos.classifiers.prompt import SYSTEM_PROMPT, render_user_message
from agent.awos.events.schema import TriggerCategory

log = logging.getLogger(__name__)

_API_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicClassifier:
    """Claude Haiku-backed classifier.

    Parameters
    ----------
    api_key:
        Anthropic API key. If ``None`` or empty, ``classify`` returns a
        low-confidence UNKNOWN result — never raises.
    model, max_tokens, timeout_s:
        Request parameters.
    client_factory:
        Optional callable returning an ``httpx.Client``. Used for tests.
    """

    name = "anthropic"

    def __init__(
        self,
        api_key: str | None,
        *,
        model: str = "claude-haiku-3-5-20241022",
        max_tokens: int = 512,
        timeout_s: float = 15.0,
        max_retries: int = 3,
        client_factory: Callable[[], httpx.Client] | None = None,
    ) -> None:
        self.api_key = api_key or ""
        self.model = model
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._client_factory = client_factory or (lambda: httpx.Client(timeout=timeout_s))

    # ------------------------------------------------------------------
    def classify(self, text: str, context: dict | None = None) -> Classification:
        if not self.api_key:
            return _fallback("no anthropic api key configured")
        if not text.strip():
            return _fallback("empty text")

        user_msg = render_user_message(text, context)
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_msg}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

        try:
            body = self._call_with_retry(payload, headers)
        except Exception as e:
            log.warning("anthropic classifier failed: %s", e)
            return _fallback(f"api error: {e}")

        return _parse_response(body)

    # ------------------------------------------------------------------
    def _call_with_retry(self, payload: dict, headers: dict) -> dict[str, Any]:
        delay = 0.5
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with self._client_factory() as client:
                    r = client.post(_API_URL, json=payload, headers=headers)
                if r.status_code == 200:
                    return r.json()
                if r.status_code in (401, 403):
                    raise RuntimeError(f"auth error: {r.status_code} {r.text[:200]}")
                if r.status_code == 400:
                    # bad request — don't retry
                    raise RuntimeError(f"bad request: {r.text[:200]}")
                if r.status_code in (429, 500, 502, 503, 504):
                    last_exc = RuntimeError(f"transient {r.status_code}: {r.text[:200]}")
                else:
                    raise RuntimeError(f"unexpected {r.status_code}: {r.text[:200]}")
            except httpx.HTTPError as e:
                last_exc = e

            time.sleep(delay)
            delay = min(delay * 2, 4.0)

        raise last_exc or RuntimeError("exhausted retries")


# ======================================================================
def _parse_response(body: dict[str, Any]) -> Classification:
    try:
        content = body.get("content", [])
        text = ""
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text", "")
        text = text.strip()
        if not text:
            return _fallback("empty response body")

        # some models wrap JSON in code fences
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()

        data = json.loads(text)
        raw_cat = str(data.get("category", "unknown")).lower()
        try:
            category = TriggerCategory(raw_cat)
        except ValueError:
            category = TriggerCategory.UNKNOWN

        confidence = float(data.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))

        return Classification(
            category=category,
            confidence=confidence,
            rationale=str(data.get("rationale", ""))[:500],
            extracted_principle=_nonempty(data.get("extracted_principle")),
            suggested_section=_nonempty(data.get("suggested_section")),
            classifier="anthropic",
        )
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        return _fallback(f"parse error: {e}")


def _nonempty(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _fallback(reason: str) -> Classification:
    return Classification(
        category=TriggerCategory.UNKNOWN,
        confidence=0.0,
        rationale=reason,
        classifier="anthropic",
    )


__all__ = ["AnthropicClassifier"]
