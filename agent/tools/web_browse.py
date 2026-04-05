"""
Tool: Web Browse

Fetch and extract readable text content from a URL.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from agent.tools.base import Tool, ToolResult

log = logging.getLogger(__name__)

_MAX_CONTENT_LEN = 12_000  # characters — keep context window manageable


class WebBrowseTool(Tool):
    name = "web_browse"
    description = (
        "Fetch a web page and extract its readable text content. "
        "Use this to read articles, documentation, reports, or any URL."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch.",
            },
        },
        "required": ["url"],
    }

    def execute(self, *, url: str, **_: Any) -> ToolResult:
        # Basic URL validation
        if not url.startswith(("http://", "https://")):
            return ToolResult(
                success=False, output="URL must start with http:// or https://"
            )

        # SSRF protection
        try:
            from agent.security.tool_policy import is_safe_url

            safe, reason = is_safe_url(url)
            if not safe:
                return ToolResult(success=False, output=f"SSRF blocked: {reason}")
        except ImportError:
            pass

        try:
            with httpx.Client(timeout=20, follow_redirects=True) as client:
                resp = client.get(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) TirraMind/0.1"
                    },
                )
                resp.raise_for_status()

            text = self._extract_text(resp.text)
            if len(text) > _MAX_CONTENT_LEN:
                text = text[:_MAX_CONTENT_LEN] + "\n\n[... truncated]"
            return ToolResult(success=True, output=text)
        except Exception as exc:
            log.exception("Web browse failed for %s", url)
            return ToolResult(success=False, output=f"Browse error: {exc}")

    @staticmethod
    def _extract_text(html: str) -> str:
        """Rough HTML → text extraction. Good enough for prototyping."""
        # Remove script and style blocks
        text = re.sub(
            r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE
        )
        # Remove tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text
