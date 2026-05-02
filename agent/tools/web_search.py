"""
Tool: Web Search

Searches the web using DuckDuckGo (no API key required).
Upgrade path: SerpAPI, Brave Search, or custom scraping.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from agent.tools.base import Tool, ToolResult

log = logging.getLogger(__name__)

_DDG_URL = "https://html.duckduckgo.com/html/"


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Search the web for information. Returns a list of result titles, URLs, and snippets. "
        "Use this to find current information, research topics, or discover data sources."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default 5).",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    def execute(self, *, query: str, max_results: int = 5, **_: Any) -> ToolResult:
        try:
            results = self._search_ddg(query, max_results)
            if not results:
                return ToolResult(success=True, output="No results found.", data=[])
            formatted = "\n\n".join(
                f"[{i + 1}] {r['title']}\n    {r['url']}\n    {r['snippet']}" for i, r in enumerate(results)
            )
            return ToolResult(success=True, output=formatted, data=results)
        except Exception as exc:
            log.exception("Web search failed")
            return ToolResult(success=False, output=f"Search error: {exc}")

    def _search_ddg(self, query: str, max_results: int) -> list[dict[str, str]]:
        """Scrape DuckDuckGo HTML search results."""
        with httpx.Client(timeout=15, follow_redirects=True) as client:
            resp = client.post(
                _DDG_URL,
                data={"q": query, "b": ""},
                headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) TirraMind/0.1"},
            )
            resp.raise_for_status()

        results: list[dict[str, str]] = []
        html = resp.text
        # Simple parsing — extract result blocks
        # DuckDuckGo HTML results are in <a class="result__a"> tags
        import re

        blocks = re.findall(
            r'<a[^>]+class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
            r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
            html,
            re.DOTALL,
        )
        for url, title, snippet in blocks[:max_results]:
            # Clean HTML tags from title and snippet
            clean = lambda s: re.sub(r"<[^>]+>", "", s).strip()
            results.append(
                {
                    "title": clean(title),
                    "url": url,
                    "snippet": clean(snippet),
                }
            )
        return results
